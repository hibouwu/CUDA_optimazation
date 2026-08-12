#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
  std::cerr << "CUDA error " << __FILE__ << ':' << __LINE__ << ' ' \
            << cudaGetErrorString(e_) << '\n'; std::exit(EXIT_FAILURE); } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t s_=(call); \
  if(s_!=CUBLAS_STATUS_SUCCESS){ std::cerr << "cuBLAS error " << __FILE__ \
  << ':' << __LINE__ << " status=" << int(s_) << '\n'; \
  std::exit(EXIT_FAILURE); } } while(0)

namespace {
namespace wmma = nvcuda::wmma;
constexpr int kWarmup = 3;
constexpr int kRepeat = 10;

template <class Launch>
float benchmark(Launch launch) {
  for (int i=0;i<kWarmup;++i) launch();
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t begin{}, end{}; CHECK_CUDA(cudaEventCreate(&begin));
  CHECK_CUDA(cudaEventCreate(&end)); CHECK_CUDA(cudaEventRecord(begin));
  for (int i=0;i<kRepeat;++i) launch();
  CHECK_CUDA(cudaEventRecord(end)); CHECK_CUDA(cudaEventSynchronize(end));
  float total=0; CHECK_CUDA(cudaEventElapsedTime(&total,begin,end));
  CHECK_CUDA(cudaEventDestroy(begin)); CHECK_CUDA(cudaEventDestroy(end));
  return total/kRepeat;
}

double rate(int n, float milliseconds) {
  return 2.0*double(n)*n*n*1000.0/double(milliseconds);
}

void fill_floats(std::vector<float>& a, std::vector<float>& b) {
  for (std::size_t i=0;i<a.size();++i)
    a[i]=float(int(i%17)-8)*0.073f;
  for (std::size_t i=0;i<b.size();++i)
    b[i]=float(int(i%13)-6)*0.041f;
}

float round_to_tf32_rn(float value) {
  std::uint32_t bits=0; std::memcpy(&bits,&value,sizeof(bits));
  if((bits&0x7f800000u)==0x7f800000u) return value;
  const std::uint32_t retained_lsb=(bits>>13)&1u;
  bits=(bits+0x00000fffu+retained_lsb)&0xffffe000u;
  std::memcpy(&value,&bits,sizeof(value)); return value;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float float_from_bits(std::uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool host_self_test() {
  // TF32 retains FP32 sign/exponent and ten fraction bits.  These two values
  // are exact halfway cases: the retained low bit is respectively even/odd.
  const bool tie_even_down =
      float_bits(round_to_tf32_rn(float_from_bits(0x3f801000u))) ==
      0x3f800000u;
  const bool tie_odd_up =
      float_bits(round_to_tf32_rn(float_from_bits(0x3f803000u))) ==
      0x3f804000u;
  const bool infinity_unchanged =
      float_bits(round_to_tf32_rn(float_from_bits(0x7f800000u))) ==
      0x7f800000u;
  const bool nan_payload_unchanged =
      float_bits(round_to_tf32_rn(float_from_bits(0x7fc12345u))) ==
      0x7fc12345u;
  const bool pass = tie_even_down && tie_odd_up && infinity_unchanged &&
                    nan_payload_unchanged;
  std::cout << "self_test=" << (pass ? "pass" : "fail")
            << " tf32_tie_even_down=" << tie_even_down
            << " tf32_tie_odd_up=" << tie_odd_up
            << " tf32_infinity_unchanged=" << infinity_unchanged
            << " tf32_nan_payload_unchanged=" << nan_payload_unchanged << '\n';
  return pass;
}

template <class T, class Convert>
std::vector<T> convert_vector(const std::vector<float>& in, Convert convert) {
  std::vector<T> out(in.size());
  for (std::size_t i=0;i<in.size();++i) out[i]=convert(in[i]);
  return out;
}

template <class T, class ToFloat>
bool validate_float_reference_samples(const char* contract, int n,
    const std::vector<T>& a, const std::vector<T>& b,
    const std::vector<float>& reference, ToFloat to_float,
    float atol, float rtol) {
  constexpr int samples=64; int errors=0; float max_abs=0, max_ratio=0;
  for(int sample=0;sample<samples;++sample) {
    int row=(sample*131)%n, col=(sample*197)%n; float expected=0;
    for(int k=0;k<n;++k)
      expected += to_float(a[std::size_t(row)*n+k]) *
                  to_float(b[std::size_t(k)*n+col]);
    float observed=reference[std::size_t(row)*n+col];
    float difference=std::fabs(expected-observed);
    float tolerance=atol+rtol*std::fabs(expected);
    bool finite=std::isfinite(expected)&&std::isfinite(observed)&&
                std::isfinite(difference)&&std::isfinite(tolerance);
    max_abs=std::max(max_abs,difference);
    if(tolerance>0) max_ratio=std::max(max_ratio,difference/tolerance);
    if(!finite||difference>tolerance) ++errors;
  }
  std::cout << "reference_contract=" << contract
            << " reference_sample_count=" << samples
            << " reference_mismatch_count=" << errors
            << " reference_max_abs_error=" << max_abs
            << " reference_max_tolerance_ratio=" << max_ratio
            << " reference_atol=" << atol << " reference_rtol=" << rtol << '\n';
  return errors==0;
}

bool compare_float_full(const char* contract, const std::vector<float>& ref,
                        const std::vector<float>& got, float atol, float rtol) {
  std::size_t errors=0; float max_abs=0, max_ratio=0;
  for(std::size_t i=0;i<ref.size();++i) {
    float d=std::fabs(ref[i]-got[i]), t=atol+rtol*std::fabs(ref[i]);
    bool finite=std::isfinite(ref[i])&&std::isfinite(got[i])&&
                std::isfinite(d)&&std::isfinite(t);
    max_abs=std::max(max_abs,d); if(t>0) max_ratio=std::max(max_ratio,d/t);
    if(!finite||d>t) ++errors;
  }
  std::cout << "numerical_contract=" << contract
            << " mismatch_count=" << errors << " max_abs_error=" << max_abs
            << " max_tolerance_ratio=" << max_ratio << " atol=" << atol
            << " rtol=" << rtol << '\n';
  return errors==0;
}

template <class T, class Acc, int KTile>
__global__ void wmma_m128n64_kernel(const T* a, const T* b,
                                     Acc* c, int n) {
  int warp=threadIdx.x/32, row=blockIdx.y*128+warp*32;
  int col=blockIdx.x*64;
  wmma::fragment<wmma::matrix_a,32,8,KTile,T,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,32,8,KTile,T,wmma::row_major> bf[8];
  wmma::fragment<wmma::accumulator,32,8,KTile,Acc> cf[8];
  #pragma unroll
  for(int i=0;i<8;++i) wmma::fill_fragment(cf[i],Acc(0));
  for(int k=0;k<n;k+=KTile) {
    wmma::load_matrix_sync(af,a+std::size_t(row)*n+k,n);
    #pragma unroll
    for(int i=0;i<8;++i) {
      wmma::load_matrix_sync(bf[i],b+std::size_t(k)*n+col+i*8,n);
      wmma::mma_sync(cf[i],af,bf[i],cf[i]);
    }
  }
  #pragma unroll
  for(int i=0;i<8;++i)
    wmma::store_matrix_sync(c+std::size_t(row)*n+col+i*8,cf[i],n,
                            wmma::mem_row_major);
}

__global__ void tf32_wmma_m64n64_kernel(const float* a,
                                        const float* b,
                                        float* c, int n) {
  int warp=threadIdx.x/32, wr=warp/4, wc=warp%4;
  int row=blockIdx.y*64+wr*16, col=blockIdx.x*64+wc*16;
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,
                 wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,
                 wmma::row_major> bf;
  wmma::fragment<wmma::accumulator,16,16,8,float> cf;
  wmma::fill_fragment(cf,0.0f);
  for(int k=0;k<n;k+=8) {
    wmma::load_matrix_sync(af,a+std::size_t(row)*n+k,n);
    wmma::load_matrix_sync(bf,b+std::size_t(k)*n+col,n);
    wmma::mma_sync(cf,af,bf,cf);
  }
  wmma::store_matrix_sync(c+std::size_t(row)*n+col,cf,n,wmma::mem_row_major);
}

__device__ __forceinline__ std::uint32_t fp8_byte(const __nv_fp8_e5m2* p,
                                                   std::size_t index) {
  return reinterpret_cast<const std::uint8_t*>(p)[index];
}

__global__ void e5m2_mma_m16n8k32_smem128x64_kernel(
    const __nv_fp8_e5m2* a, const __nv_fp8_e5m2* b, float* c, int n) {
  __shared__ alignas(16) std::uint8_t as[128*32], bs[32*64];
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,group=lane>>2,in_group=lane&3;
  int tile_m=blockIdx.y*128,tile_n=blockIdx.x*64,warp_m=warp*16;
  float d[8][4]{};
  for(int k0=0;k0<n;k0+=32) {
    for(int i=tid;i<128*32;i+=blockDim.x)
      as[i]=fp8_byte(a,std::size_t(tile_m+i/32)*n+k0+i%32);
    for(int i=tid;i<32*64;i+=blockDim.x)
      bs[i]=fp8_byte(b,std::size_t(k0+i/64)*n+tile_n+i%64);
    __syncthreads(); int sr=warp_m+group;
    std::uint32_t a0=0,a1=0,a2=0,a3=0;
    #pragma unroll
    for(int i=0;i<4;++i) { int k=in_group*4+i;
      a0|=std::uint32_t(as[sr*32+k])<<(8*i);
      a1|=std::uint32_t(as[(sr+8)*32+k])<<(8*i);
      a2|=std::uint32_t(as[sr*32+k+16])<<(8*i);
      a3|=std::uint32_t(as[(sr+8)*32+k+16])<<(8*i); }
    #pragma unroll
    for(int ct=0;ct<8;++ct) { std::uint32_t b0=0,b1=0;
      #pragma unroll
      for(int i=0;i<4;++i) { int k=in_group*4+i,col=ct*8+group;
        b0|=std::uint32_t(bs[k*64+col])<<(8*i);
        b1|=std::uint32_t(bs[(k+16)*64+col])<<(8*i); }
      asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4."
                   "f32.e5m2.e5m2.f32 {%0,%1,%2,%3},"
                   "{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[ct][0]),"+f"(d[ct][1]),"+f"(d[ct][2]),"+f"(d[ct][3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); }
    __syncthreads();
  }
  int r0=tile_m+warp_m+group,r1=r0+8;
  #pragma unroll
  for(int ct=0;ct<8;++ct) { int col=tile_n+ct*8+in_group*2;
    c[std::size_t(r0)*n+col]=d[ct][0]; c[std::size_t(r0)*n+col+1]=d[ct][1];
    c[std::size_t(r1)*n+col]=d[ct][2]; c[std::size_t(r1)*n+col+1]=d[ct][3]; }
}

template <class T>
float cublas_float_reference(int n, const T* da, const T* db, float* dc,
                             cudaDataType_t input_type,
                             cublasComputeType_t compute_type) {
  cublasHandle_t h{}; CHECK_CUBLAS(cublasCreate(&h)); float alpha=1,beta=0;
  auto launch=[&]{ CHECK_CUBLAS(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,
    &alpha,db,input_type,n,da,input_type,n,&beta,dc,CUDA_R_32F,n,compute_type,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP)); };
  float ms=benchmark(launch); CHECK_CUBLAS(cublasDestroy(h)); return ms;
}

void emit(const std::string& backend,const std::string& precision,
          const std::string& reference,int n,float custom_ms,float reference_ms,
          bool matched) {
  double custom_rate=rate(n,custom_ms), reference_rate=rate(n,reference_ms);
  const char* unit="flop";
  std::cout<<std::setprecision(17)<<"backend_id="<<backend<<" N="<<n
           <<" time_ms="<<custom_ms<<" reference_time_ms="<<reference_ms
           <<" work_unit="<<unit<<" rate_per_second="<<custom_rate
           <<" reference_rate_per_second="<<reference_rate
           <<" matched="<<(matched?1:0)<<'\n';
  std::ofstream csv("extended_sm110_benchmark.csv"); csv<<std::setprecision(17);
  csv<<"BackendId,N,Precision,Reference,TimeMs,ReferenceTimeMs,RatePerSecond,ReferenceRatePerSecond,Matched\n";
  csv<<backend<<','<<n<<','<<precision<<','<<reference<<','<<custom_ms<<','
     <<reference_ms<<','<<custom_rate<<','<<reference_rate<<','<<(matched?1:0)<<'\n';
}

template<class T,class Convert,class ToFloat,class Kernel>
void run_float_case(int n,const std::string& backend,const std::string& precision,
                    const std::string& reference,const char* ref_contract,
                    const char* num_contract,cudaDataType_t cuda_type,
                    cublasComputeType_t compute_type,Convert convert,
                    ToFloat to_float,Kernel kernel,float atol,float rtol) {
  std::vector<float> fa(std::size_t(n)*n),fb(fa.size()); fill_floats(fa,fb);
  auto a=convert_vector<T>(fa,convert),b=convert_vector<T>(fb,convert);
  std::vector<float> ref(fa.size()),got(fa.size()); T *da{},*db{}; float *dc{};
  CHECK_CUDA(cudaMalloc(&da,a.size()*sizeof(T))); CHECK_CUDA(cudaMalloc(&db,b.size()*sizeof(T)));
  CHECK_CUDA(cudaMalloc(&dc,ref.size()*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(da,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
  float ref_ms=cublas_float_reference(n,da,db,dc,cuda_type,compute_type);
  CHECK_CUDA(cudaMemcpy(ref.data(),dc,ref.size()*sizeof(float),cudaMemcpyDeviceToHost));
  if(!validate_float_reference_samples(ref_contract,n,a,b,ref,to_float,atol,rtol)) std::exit(EXIT_FAILURE);
  CHECK_CUDA(cudaMemset(dc,0,ref.size()*sizeof(float)));
  auto launch=[&]{ kernel(da,db,dc); CHECK_CUDA(cudaGetLastError()); };
  float custom_ms=benchmark(launch); CHECK_CUDA(cudaMemcpy(got.data(),dc,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
  bool matched=compare_float_full(num_contract,ref,got,atol,rtol);
  emit(backend,precision,reference,n,custom_ms,ref_ms,matched);
  CHECK_CUDA(cudaFree(da));CHECK_CUDA(cudaFree(db));CHECK_CUDA(cudaFree(dc));
  if(!matched) std::exit(EXIT_FAILURE);
}

} // namespace

int main(int argc,char**argv){
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    return host_self_test() ? 0 : 1;
  }
  if(argc!=3){std::cerr<<"usage: "<<argv[0]<<" N bf16|tf32\n";return 2;}
  int n=std::atoi(argv[1]);std::string mode=argv[2];
  if(n<=0||n%128){std::cerr<<"N must be a positive multiple of 128\n";return 2;}
  cudaDeviceProp p{};CHECK_CUDA(cudaGetDeviceProperties(&p,0));
  std::cout<<"GPU="<<p.name<<" compute_capability="<<p.major<<'.'<<p.minor<<" N="<<n<<" mode="<<mode<<'\n';
  if(mode=="bf16")run_float_case<__nv_bfloat16>(n,"bf16_q0_wmma_m128n64k16","bf16->fp32","cuBLAS BF16","bf16_f32_cpu_samples","bf16_f32",CUDA_R_16BF,CUBLAS_COMPUTE_32F,
    [] (float x){return __float2bfloat16(x);},[](__nv_bfloat16 x){return __bfloat162float(x);},
    [=](__nv_bfloat16*a,__nv_bfloat16*b,float*c){wmma_m128n64_kernel<__nv_bfloat16,float,16><<<dim3(n/64,n/128),128>>>(a,b,c,n);},0.05f,0.005f);
  else if(mode=="tf32")run_float_case<float>(n,"tf32_q0_wmma_m64n64k8","tf32->fp32","cuBLAS TF32","tf32_f32_cpu_samples","tf32_f32",CUDA_R_32F,CUBLAS_COMPUTE_32F_FAST_TF32,
    [](float x){return round_to_tf32_rn(x);},[](float x){return x;},
    [=](float*a,float*b,float*c){tf32_wmma_m64n64_kernel<<<dim3(n/64,n/64),512>>>(a,b,c,n);},0.05f,0.005f);
  else {std::cerr<<"unknown mode\n";return 2;} return 0;
}
