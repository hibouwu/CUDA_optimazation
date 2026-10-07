// R15: 16KiB output chunks, explicit source-read / release / full-write boundaries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gaps_common.hpp"
#include "r15_fragment_store.cuh"
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <utility>
namespace G = cute::SM90::GMMA;
#define R15_CHECK(x) do { auto e = (x); if(e != cudaSuccess) \
  throw std::runtime_error(cudaGetErrorString(e)); } while(0)
constexpr int kThreads = 384;
constexpr int kBackgroundBytes = 40960, kOutputBytes = 32768;
constexpr int kSharedBytes = kBackgroundBytes + kOutputBytes;
struct Stamp { uint64_t cycle, ns; unsigned sm; };
struct Trace { uint64_t prepare, issued, read_done, released, full_done, background_done; };
__device__ unsigned shared_address(const void* p) {
  return unsigned(__cvta_generic_to_shared(p));
}
__device__ uint64_t nanos() {
  uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ unsigned smid() {
  unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s;
}
// The shared load is consumed BEFORE both clocks; ptxas must retain its scoreboard wait.
__device__ void guarded_stamp(const volatile unsigned* p, Stamp& s) {
  unsigned value = *p; uint64_t c, n;
  asm volatile("{.reg .pred ok; setp.ne.u32 ok,%2,0xffffffff;\n"
               "@ok mov.u64 %0,%%clock64; @ok mov.u64 %1,%%globaltimer;\n"
               "@!ok mov.u64 %0,0; @!ok mov.u64 %1,0;}"
               : "=l"(c), "=l"(n) : "r"(value) : "memory");
  s = {c, n, smid()};
}
__device__ void synchronized_begin(volatile unsigned* publication, Stamp& begin) {
  publication[threadIdx.x] = 1; __syncthreads();
  if(threadIdx.x == 0) guarded_stamp(&publication[383],begin);
  __syncthreads();
}
__device__ void synchronized_end(volatile unsigned* publication, Stamp& end,
                                 float checksum) {
  publication[threadIdx.x] = __float_as_uint(checksum); __syncthreads();
  if(threadIdx.x == 0) guarded_stamp(&publication[383],end);
  __syncthreads();
}
__host__ __device__ float output_value(int row, int column) {
  return float(((row % 16) * 7 + (column % 128) * 11) % 31 - 15) / 32.f;
}
template<class Mma, size_t... I>
__device__ __forceinline__ void call_mma(uint64_t a, uint64_t b, float* d,
                                      std::index_sequence<I...>) {
  Mma::fma(a, b, d[I]..., G::ScaleOut::One);
}
__device__ void tensor_store(const CUtensorMap* map, unsigned source, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
               " [%0,{%1,%2}],[%3];"
               :: "l"(map), "r"(x), "r"(y), "r"(source) : "memory");
}
// 0 reg->SMEM; 1 TMA-only; 2 full-output; 3 background; 4 serial; 5 concurrent.
// All modes reserve identical 384-thread/72KiB SMEM footprint. Live registers are reported.
template<int N, int Buffers, int Mode, bool Traced>
__global__ __launch_bounds__(kThreads, 1)
void output_probe(const __grid_constant__ CUtensorMap map, int repeats,
                  float* background, float* smem_witness, Stamp* bounds, Trace* trace) {
  using namespace cute;
  constexpr int Registers = N / 2, Chunks = 128 * N / 4096;
  extern __shared__ __align__(1024) unsigned char bytes[];
  float* slots = reinterpret_cast<float*>(bytes);
  __half* ap = reinterpret_cast<__half*>(bytes + kOutputBytes);
  __half* bp = ap + 4096;
  auto la = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<_64,_64>{});
  auto lb = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<_128,_64>{});
  auto a = make_tensor(make_smem_ptr(ap), la);
  auto b = make_tensor(make_smem_ptr(bp), lb);
  for(int q = threadIdx.x; q < 4096; q += blockDim.x) a(q/64,q%64) = __half(0.0625f);
  for(int q = threadIdx.x; q < 8192; q += blockDim.x) b(q/64,q%64) = __half(0.0625f);
  for(int q = threadIdx.x; q < kOutputBytes/4; q += blockDim.x) {
    int local=q%4096;
    slots[q] = output_value(local/128,local%128);
  }
  __shared__ volatile unsigned publication[384];
  publication[threadIdx.x] = 1; __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  int role = cutlass::canonical_warp_group_idx(), lane = int(threadIdx.x) % 128;
  // One physical fragment lifetime, selected by role. Do not allocate independent
  // output128 and background64 arrays across barrier/control-flow reconvergence.
  float d[Registers];
#pragma unroll
  for(int j = 0; j < Registers; ++j) {
    int row = (lane/32)*16 + (lane%32)/4 + ((j/2)%2)*8 + role*64;
    int col = (lane%4)*2 + j%2 + (j/4)*8;
    d[j] = role < 2 ? output_value(row,col) : 0.f;
    cute::warpgroup_fence_operand(d[j]);
  }
  __shared__ Stamp begin, end;
  if(role < 2) {
    synchronized_begin(publication,begin);
#pragma unroll 1
    for(int repeat = 0; repeat < repeats; ++repeat) {
#pragma unroll 1
      for(int chunk = 0; chunk < Chunks; ++chunk) {
        int slot = chunk % Buffers, chunk_m = chunk % 4, chunk_n = chunk / 4;
        int owner = chunk_m / 2;
        bool observe = repeat == repeats-1 && chunk == Chunks-1;
        Trace* record = trace + repeat*Chunks + chunk;
        // Wait for the slot's old TMA source read BEFORE overwriting that slot.
        if constexpr(Mode == 1 || Mode == 2 || Mode == 5) {
          if(threadIdx.x == 0 && chunk >= Buffers) {
            if constexpr(Buffers == 1)
              asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            else asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
          }
        }
        publication[threadIdx.x] = 2; __syncthreads();
        if constexpr(Mode == 0 || Mode == 2 || Mode == 4 || Mode == 5) {
          // The selected two warps own this32x128 block. Keep a single address
          // live; individual fragment offsets are PTX immediates, not64 GPRs.
          if(role == owner && lane/64 == chunk_m%2) {
            int row_base = ((lane/32)%2)*16 + (lane%32)/4;
            int col_base = (lane%4)*2;
            unsigned dst = shared_address(slots+slot*4096+row_base*128+col_base);
            if constexpr(N == 128) {
              r15_store_fragment64(dst,d);
            } else {
              if(chunk_n == 0) r15_store_fragment64(dst,d);
              else r15_store_fragment64(dst,d+64);
            }
          }
        }
        // TMA-only uses the same immutable periodic source preloaded before the clock.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        publication[threadIdx.x] = 3;
        asm volatile("bar.sync 2, 256;" ::: "memory");
        if constexpr(Traced && Mode != 1) if(observe && threadIdx.x == 0) {
          Stamp s; guarded_stamp(&publication[255],s); record->prepare = s.cycle;
        }
        if constexpr(Mode == 1 || Mode == 2 || Mode == 4 || Mode == 5) {
          if(threadIdx.x == 0) {
            tensor_store(&map, shared_address(slots+slot*4096),
                         128*chunk_n, 32*chunk_m);
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            if constexpr(Traced && Mode != 1) if(observe) record->issued = clock64();
            if constexpr(Mode == 4) {
              asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
              if constexpr(Traced) if(observe) record->read_done = clock64();
              asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
              if constexpr(Traced) if(observe) record->full_done = clock64();
            }
          }
        }
        // Serial lets WG2 start only after current chunk's full write; other modes
        // meet here after both roles finish their independent per-chunk work.
        if constexpr(Mode == 4) {
          __syncthreads();
          if constexpr(Traced) if(observe && threadIdx.x==0) {
            Stamp s;guarded_stamp(&publication[383],s);record->released=s.cycle;
          }
        }
        publication[threadIdx.x] = 4; __syncthreads();
      }
      if constexpr(Mode == 1 || Mode == 2 || Mode == 5) {
        if(threadIdx.x == 0) {
          asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          if constexpr(Traced) if(repeat == repeats-1)
            trace[repeats*Chunks-1].read_done=clock64();
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
          if constexpr(Traced) if(repeat == repeats-1)
            trace[repeats*Chunks-1].full_done=clock64();
        }
      }
      publication[threadIdx.x] = 5; __syncthreads();
      if constexpr(Traced && (Mode==2||Mode==5))
        if(repeat == repeats-1 && threadIdx.x==0) {
          Stamp s; guarded_stamp(&publication[383],s);
          trace[repeats*Chunks-1].released=s.cycle;
        }
    }
    float checksum=0;
#pragma unroll
    for(int j=0;j<Registers;++j) checksum+=d[j];
    synchronized_end(publication,end,checksum);
  } else {
    using MMA = G::MMA_64x128x16_F32F16F16_SS<G::Major::K,G::Major::K>;
    auto av = local_tile(a, make_shape(_64{},_16{}), make_coord(_0{},_0{}));
    auto bv = local_tile(b, make_shape(_128{},_16{}), make_coord(_0{},_0{}));
    uint64_t ad = G::make_gmma_desc<G::Major::K>(av);
    uint64_t bd = G::make_gmma_desc<G::Major::K>(bv);
    synchronized_begin(publication,begin);
#pragma unroll 1
    for(int repeat = 0; repeat < repeats; ++repeat) {
#pragma unroll 1
      for(int chunk = 0; chunk < Chunks; ++chunk) {
        publication[threadIdx.x] = 2; __syncthreads();
        if constexpr(Mode == 4) __syncthreads();
        if constexpr(Mode == 3 || Mode == 4 || Mode == 5) {
          cute::warpgroup_arrive();
#pragma unroll
          for(int q = 0; q < 16; ++q)
            call_mma<MMA>(ad,bd,d,std::make_index_sequence<64>{});
          cute::warpgroup_commit_batch(); cute::warpgroup_wait<0>();
#pragma unroll
          for(int j = 0; j < 64; ++j) cute::warpgroup_fence_operand(d[j]);
          if constexpr(Traced)
            if(repeat == repeats-1 && chunk == Chunks-1 && threadIdx.x == 256)
              trace[repeats*Chunks-1].background_done = clock64();
        }
        publication[threadIdx.x] = 4; __syncthreads();
      }
      publication[threadIdx.x] = 5; __syncthreads();
    }
    float checksum=0;
#pragma unroll
    for(int j=0;j<64;++j) {cute::warpgroup_fence_operand(d[j]);checksum+=d[j];}
    synchronized_end(publication,end,checksum);
#pragma unroll
    for(int j = 0; j < 64; ++j) background[lane*64+j] = d[j];
  }
  // Complete background/witness dumps are outside the service clock window.
  if(threadIdx.x == 0) { bounds[0]=begin; bounds[1]=end; }
  for(int q = threadIdx.x; q < kOutputBytes/4; q += blockDim.x) smem_witness[q]=slots[q];
}
void driver_check(CUresult result) {
  if(result != CUDA_SUCCESS) {
    const char* name=nullptr; cuGetErrorString(result,&name);
    throw std::runtime_error(name ? name : "CUDA driver error");
  }
}
struct R15Options {
  int kib=64, buffers=1, mode=2, repeats=32;
  bool traced=false;
  R15Options(int argc,char**argv) {
    for(int i=1;i<argc;i+=2) {
      if(i+1==argc) throw std::runtime_error("missing option value");
      std::string key=argv[i]; int value=std::stoi(argv[i+1]);
      if(key=="--kib") kib=value;
      else if(key=="--buffers") buffers=value;
      else if(key=="--mode") mode=value;
      else if(key=="--repeats") repeats=value;
      else if(key=="--trace") traced=value!=0;
      else throw std::runtime_error("unknown option");
    }
    if((kib!=64&&kib!=128)||(buffers!=1&&buffers!=2)||mode<0||mode>5||
       repeats<1||repeats>1024||(buffers==2&&mode!=2&&mode!=4&&mode!=5))
      throw std::runtime_error("configuration outside R15 matrix");
  }
};
template<class T> struct Buffer {
  T* p=nullptr; size_t count;
  explicit Buffer(size_t n):count(n) { R15_CHECK(cudaMalloc(&p,n*sizeof(T))); }
  ~Buffer() { if(p) cudaFree(p); }
};
double cv(const std::vector<double>& v) {
  double mean=0,var=0; for(double x:v)mean+=x; mean/=v.size();
  for(double x:v)var+=(x-mean)*(x-mean);
  return std::sqrt(var/v.size())/mean;
}
void binary_output(const char*name,const void*p,size_t bytes) {
  std::ofstream stream(name,std::ios::binary); stream.write(static_cast<const char*>(p),bytes);
  if(!stream.good()) throw std::runtime_error("output file write failed");
}
template<int N,int Buffers,int Mode,bool Traced> int execute(const R15Options& options) {
  constexpr int Chunks=128*N/4096,Ldd=N+32;
  gaps::validate_layout(128,N,Ldd);
  Buffer<float> output(128*Ldd),bg(8192),witness(8192);
  Buffer<Stamp> bounds(2); Buffer<Trace> trace(options.repeats*Chunks);
  CUtensorMap map{};
  uint64_t dimensions[2]={N,128},stride[1]={Ldd*4};
  uint32_t box[2]={128,32},element_stride[2]={1,1};
  driver_check(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,output.p,
      dimensions,stride,box,element_stride,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  auto kernel=output_probe<N,Buffers,Mode,Traced>;
  R15_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes));
  cudaFuncAttributes attr{};R15_CHECK(cudaFuncGetAttributes(&attr,kernel));
  int occupancy=0;R15_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy,kernel,kThreads,kSharedBytes));
  if(occupancy<1||attr.localSizeBytes!=0) throw std::runtime_error("resource rejection or local spill");
  std::vector<float> values(output.count),bg_values(bg.count),smem(witness.count);
  Stamp times[2];std::vector<Trace> traces(trace.count);
  auto launch=[&]() {
    // Exact shared gaps sentinel0x7fc12345 is checked when this mode issues no output stores.
    gaps::fill_output_sentinel<<<64,128>>>(output.p,output.count);
    R15_CHECK(cudaGetLastError());
    R15_CHECK(cudaMemset(trace.p,0,trace.count*sizeof(Trace)));
    kernel<<<1,kThreads,kSharedBytes>>>(map,options.repeats,bg.p,witness.p,bounds.p,trace.p);
    R15_CHECK(cudaGetLastError());R15_CHECK(cudaDeviceSynchronize());
    R15_CHECK(cudaMemcpy(times,bounds.p,sizeof(times),cudaMemcpyDeviceToHost));
    if(times[1].cycle<=times[0].cycle||times[0].sm!=times[1].sm)
      throw std::runtime_error("invalid guarded interval");
    return double(times[1].cycle-times[0].cycle);
  };
  std::vector<double>warm;bool converged=false;
  for(int q=0;q<30;++q) {
    warm.push_back(launch());
    if(q>=7) { std::vector<double> last(warm.end()-5,warm.end());
      if(cv(last)<=.02) {converged=true;break;} }
  }
  uint64_t elapsed=uint64_t(launch());
  R15_CHECK(cudaMemcpy(values.data(),output.p,output.count*4,cudaMemcpyDeviceToHost));
  R15_CHECK(cudaMemcpy(bg_values.data(),bg.p,bg.count*4,cudaMemcpyDeviceToHost));
  R15_CHECK(cudaMemcpy(smem.data(),witness.p,witness.count*4,cudaMemcpyDeviceToHost));
  R15_CHECK(cudaMemcpy(traces.data(),trace.p,trace.count*sizeof(Trace),cudaMemcpyDeviceToHost));
  bool stores=Mode==1||Mode==2||Mode==4||Mode==5;
  double error=0;
  for(int row=0;row<128;++row)for(int col=0;col<N;++col) {
    float x=values[row*Ldd+col];
    if(stores) {if(!std::isfinite(x))throw std::runtime_error("nonfinite output");
      error=std::max(error,std::abs(double(x)-output_value(row,col)));}
    else if(reinterpret_cast<const uint32_t*>(values.data())[row*Ldd+col]!=0x7fc12345u)
      throw std::runtime_error("inactive output was modified");
  }
  if(gaps::padding_errors(output.p,128,N,Ldd,gaps::output_sentinel())!=0)
    throw std::runtime_error("output padding changed");
  float bg_reference=(Mode>=3)?float(options.repeats*Chunks):0.f;
  for(float x:bg_values) if(x!=bg_reference)throw std::runtime_error("background mismatch");
  for(int q=0;q<8192;++q) {
    int local=q%4096;
    if(smem[q]!=output_value(local/128,local%128))
      throw std::runtime_error("SMEM fragment witness mismatch");
  }
  if(error!=0)throw std::runtime_error("coordinate fragment mismatch");
  binary_output("output.f32",values.data(),values.size()*4);
  binary_output("background.f32",bg_values.data(),bg_values.size()*4);
  binary_output("smem.f32",smem.data(),smem.size()*4);
  binary_output("trace.u64",traces.data(),traces.size()*sizeof(Trace));
  std::cout<<std::setprecision(17)<<"{\"status\":\"measured\",\"kib\":"<<options.kib
    <<",\"ldd\":"<<Ldd<<",\"padding_errors\":0"
    <<",\"buffers\":"<<Buffers<<",\"mode\":"<<Mode<<",\"trace\":"<<(Traced?"true":"false")
    <<",\"trace_profile\":\""<<(Mode==1?"read_full_pair":"lifecycle")<<"\""
    <<",\"trace_selection\":\"last_repeat_last_chunk\""
    <<",\"trace_selected_record\":"<<(options.repeats*Chunks-1)
    <<",\"repeats\":"<<options.repeats<<",\"chunks\":"<<Chunks
    <<",\"elapsed_cycles\":"<<elapsed<<",\"elapsed_ns\":"<<times[1].ns-times[0].ns
    <<",\"sm_id\":"<<times[0].sm<<",\"threads\":384,\"dynamic_smem_bytes\":"<<kSharedBytes
    <<",\"static_smem_bytes\":"<<attr.sharedSizeBytes<<",\"registers_per_thread\":"<<attr.numRegs
    <<",\"local_bytes_per_thread\":"<<attr.localSizeBytes
    <<",\"occupancy_limit_ctas_per_sm\":"<<occupancy<<",\"warmup_converged\":"
    <<(converged?"true":"false")<<",\"max_output_error\":0,\"output_elements\":"<<values.size()
    <<",\"background_elements\":8192,\"smem_elements\":8192,\"trace_records\":"<<trace.count
    <<",\"warmup_cycles\":[";
  for(size_t q=0;q<warm.size();++q){if(q)std::cout<<',';std::cout<<warm[q];}
  std::cout<<"]}\n";return 0;
}
template<int N,int B,bool T>int select_mode(const R15Options& o) {
  switch(o.mode) {
    case 0:return execute<N,B,0,T>(o);case 1:return execute<N,B,1,T>(o);
    case 2:return execute<N,B,2,T>(o);case 3:return execute<N,B,3,T>(o);
    case 4:return execute<N,B,4,T>(o);case 5:return execute<N,B,5,T>(o);
  }throw std::runtime_error("mode");
}
template<int N>int select_buffer(const R15Options& o) {
  if(o.buffers==1)return o.traced?select_mode<N,1,true>(o):select_mode<N,1,false>(o);
  return o.traced?select_mode<N,2,true>(o):select_mode<N,2,false>(o);
}
int main(int argc,char**argv) {
  try { R15Options o(argc,argv);driver_check(cuInit(0));
    return o.kib==64?select_buffer<128>(o):select_buffer<256>(o);
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
