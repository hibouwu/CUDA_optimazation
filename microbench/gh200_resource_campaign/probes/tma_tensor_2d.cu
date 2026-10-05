#include <cuda.h>
#include "../common/probe_runtime.cuh"
#include "../common/tma_tensor_2d_reference.hpp"
#include "../common/tensor_artifacts.hpp"
static_assert(sizeof(CUtensorMap)==128,"S15 tensor map ABI");
constexpr unsigned tt_threads=128;
__device__ __forceinline__ unsigned tt_shared(const void* p){return static_cast<unsigned>(__cvta_generic_to_shared(p));}
__device__ __forceinline__ gh::u64 tt_ns(){gh::u64 x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ gh::u64 tt_cycle(){gh::u64 x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ gh::u64 tt_wait(unsigned bar,gh::u64 token,std::uint64_t* failure){
 const auto start=tt_ns();gh::u64 attempts=0;
 for(;;){unsigned done;asm volatile("{.reg .pred p; mbarrier.try_wait.acquire.cta.shared::cta.b64 p,[%1],%2,64; selp.b32 %0,1,0,p;}":"=r"(done):"r"(bar),"l"(token):"memory");++attempts;if(done)return attempts;
  if(tt_ns()-start>=1000000000ull){*reinterpret_cast<volatile std::uint64_t*>(failure)=1;__threadfence();asm volatile("trap;":::"memory");}
 }
}
template<unsigned W,bool SW> __device__ __forceinline__ unsigned tt_index(unsigned x,unsigned y){return SW?y*64+((x/8)^(y%8))*8+x%8:y*W+x;}

template<unsigned Q,bool G2S,bool SW> __device__ __forceinline__ void tt_transport(
 const CUtensorMap* map,std::uint16_t* global,unsigned pitch,int iterations,unsigned seed,bool capture,
 std::uint16_t* logical,std::uint16_t* physical,gh::Stamp* stamps,std::uint64_t* completion){
 constexpr unsigned W=Q==65536?128:64,H=Q/(2*W),elements=Q/2;
 extern __shared__ __align__(16) unsigned char storage[];
 const unsigned delta=(1024-(tt_shared(storage)&1023))&1023;
 auto* tile=reinterpret_cast<std::uint16_t*>(storage+delta);const unsigned shared=tt_shared(tile),bar=tt_shared(storage+delta+Q);
 for(unsigned n=threadIdx.x;n<elements;n+=tt_threads){const unsigned x=n%W,y=n/W;const std::uint16_t expected=std::uint16_t(17ull*x+31ull*y+151ull*blockIdx.x+seed);tile[tt_index<W,SW>(x,y)]=G2S?std::uint16_t(expected^0xffffu):expected;}
 if constexpr(G2S)if(threadIdx.x==0)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar):"memory");
 asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
 gh::Stamp stamp{};gh::u64 done_count=0,attempts=0;
 if(threadIdx.x==0){asm volatile("mov.u32 %0, %%smid;":"=r"(stamp.smid));stamp.begin_ns=tt_ns();stamp.begin_cycle=tt_cycle();}__syncthreads();
 #pragma unroll 1
 for(int i=0;i<iterations;++i){
  const unsigned slot=unsigned(i)%32;const int coordinate_y=int((gh::u64(blockIdx.x)*32+slot)*H);
  const gh::u64 base=gh::u64(coordinate_y)*(pitch/2);
  if(capture){
   for(unsigned n=threadIdx.x;n<elements;n+=tt_threads){const unsigned x=n%W,y=n/W;const std::uint16_t expected=std::uint16_t(17ull*x+31ull*y+(G2S?73ull*slot:0ull)+151ull*blockIdx.x+seed);
    if constexpr(G2S)tile[tt_index<W,SW>(x,y)]=std::uint16_t(expected^0xffffu);else global[base+gh::u64(y)*(pitch/2)+x]=std::uint16_t(expected^0xffffu);
   }
   if constexpr(G2S)asm volatile("fence.proxy.async.shared::cta;":::"memory");else asm volatile("fence.proxy.async.global;":::"memory");__syncthreads();
  }
  if(threadIdx.x==0){const int x=0;
   if constexpr(G2S){
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0],%1;"::"r"(bar),"r"(Q):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::"r"(shared),"l"(map),"r"(x),"r"(coordinate_y),"r"(bar):"memory");
    gh::u64 token;asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0,[%1];":"=l"(token):"r"(bar):"memory");attempts+=tt_wait(bar,token,completion+gh::u64(blockIdx.x)*5+2);
   }else{
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%1,%2}],[%3];"::"l"(map),"r"(x),"r"(coordinate_y),"r"(shared):"memory");
    asm volatile("cp.async.bulk.commit_group;":::"memory");asm volatile("cp.async.bulk.wait_group 0;":::"memory");
   }++done_count;
  }__syncthreads();
  if(capture){
   if constexpr(!G2S)asm volatile("fence.proxy.async.global;":::"memory");
   for(unsigned n=threadIdx.x;n<elements;n+=tt_threads){const unsigned x=n%W,y=n/W;const gh::u64 at=(gh::u64(blockIdx.x)*iterations+i)*elements+n;
    logical[at]=G2S?tile[tt_index<W,SW>(x,y)]:reinterpret_cast<volatile std::uint16_t*>(global)[base+gh::u64(y)*(pitch/2)+x];physical[at]=tile[n];
   }__syncthreads();
  }
 }
 if(threadIdx.x==0){stamp.end_cycle=tt_cycle();stamp.end_ns=tt_ns();stamps[blockIdx.x]=stamp;}
 if constexpr(G2S){if(!capture)for(unsigned n=threadIdx.x;n<elements;n+=tt_threads){logical[gh::u64(blockIdx.x)*elements+n]=tile[tt_index<W,SW>(n%W,n/W)];physical[gh::u64(blockIdx.x)*elements+n]=tile[n];}__syncthreads();if(threadIdx.x==0)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(bar):"memory");}
 if(threadIdx.x==0){const gh::u64 at=gh::u64(blockIdx.x)*5;completion[at]=done_count;completion[at+1]=attempts;completion[at+2]=0;completion[at+3]=0;completion[at+4]=G2S?0:done_count;}
}
#define TT_FORMS(Q,SW,NAME) \
extern "C" __global__ void tt_g2s_q##Q##_##NAME(const __grid_constant__ CUtensorMap map,std::uint16_t* global,unsigned pitch,int iterations,unsigned seed,bool capture,std::uint16_t* logical,std::uint16_t* physical,gh::Stamp* stamps,std::uint64_t* completion){tt_transport<Q,true,SW>(&map,global,pitch,iterations,seed,capture,logical,physical,stamps,completion);} \
extern "C" __global__ void tt_s2g_q##Q##_##NAME(const __grid_constant__ CUtensorMap map,std::uint16_t* global,unsigned pitch,int iterations,unsigned seed,bool capture,std::uint16_t* logical,std::uint16_t* physical,gh::Stamp* stamps,std::uint64_t* completion){tt_transport<Q,false,SW>(&map,global,pitch,iterations,seed,capture,logical,physical,stamps,completion);}
TT_FORMS(1024,false,none) TT_FORMS(4096,false,none) TT_FORMS(8192,false,none)
TT_FORMS(16384,false,none) TT_FORMS(32768,false,none) TT_FORMS(65536,false,none)
TT_FORMS(1024,true,sw128) TT_FORMS(4096,true,sw128) TT_FORMS(8192,true,sw128)
TT_FORMS(16384,true,sw128) TT_FORMS(32768,true,sw128)

struct TtCase {const char* id;unsigned q,p;bool g2s,sw128,all_gpu;const char* symbol;const void* function;};
static const TtCase tt_cases[]={
 {"gmem_to_smem_1kib_continuous_none_one_cta",1024,128,true,false,false,"tt_g2s_q1024_none",reinterpret_cast<const void*>(tt_g2s_q1024_none)},
 {"gmem_to_smem_1kib_padding_none_one_cta",1024,144,true,false,false,"tt_g2s_q1024_none",reinterpret_cast<const void*>(tt_g2s_q1024_none)},
 {"gmem_to_smem_1kib_continuous_sw128_one_cta",1024,128,true,true,false,"tt_g2s_q1024_sw128",reinterpret_cast<const void*>(tt_g2s_q1024_sw128)},
 {"gmem_to_smem_4kib_continuous_none_one_cta",4096,128,true,false,false,"tt_g2s_q4096_none",reinterpret_cast<const void*>(tt_g2s_q4096_none)},
 {"gmem_to_smem_4kib_padding_none_one_cta",4096,144,true,false,false,"tt_g2s_q4096_none",reinterpret_cast<const void*>(tt_g2s_q4096_none)},
 {"gmem_to_smem_4kib_continuous_sw128_one_cta",4096,128,true,true,false,"tt_g2s_q4096_sw128",reinterpret_cast<const void*>(tt_g2s_q4096_sw128)},
 {"gmem_to_smem_8kib_continuous_none_one_cta",8192,128,true,false,false,"tt_g2s_q8192_none",reinterpret_cast<const void*>(tt_g2s_q8192_none)},
 {"gmem_to_smem_8kib_padding_none_one_cta",8192,144,true,false,false,"tt_g2s_q8192_none",reinterpret_cast<const void*>(tt_g2s_q8192_none)},
 {"gmem_to_smem_8kib_continuous_sw128_one_cta",8192,128,true,true,false,"tt_g2s_q8192_sw128",reinterpret_cast<const void*>(tt_g2s_q8192_sw128)},
 {"gmem_to_smem_16kib_continuous_none_one_cta",16384,128,true,false,false,"tt_g2s_q16384_none",reinterpret_cast<const void*>(tt_g2s_q16384_none)},
 {"gmem_to_smem_16kib_padding_none_one_cta",16384,144,true,false,false,"tt_g2s_q16384_none",reinterpret_cast<const void*>(tt_g2s_q16384_none)},
 {"gmem_to_smem_16kib_continuous_sw128_one_cta",16384,128,true,true,false,"tt_g2s_q16384_sw128",reinterpret_cast<const void*>(tt_g2s_q16384_sw128)},
 {"gmem_to_smem_32kib_continuous_none_one_cta",32768,128,true,false,false,"tt_g2s_q32768_none",reinterpret_cast<const void*>(tt_g2s_q32768_none)},
 {"gmem_to_smem_32kib_padding_none_one_cta",32768,144,true,false,false,"tt_g2s_q32768_none",reinterpret_cast<const void*>(tt_g2s_q32768_none)},
 {"gmem_to_smem_32kib_continuous_sw128_one_cta",32768,128,true,true,false,"tt_g2s_q32768_sw128",reinterpret_cast<const void*>(tt_g2s_q32768_sw128)},
 {"gmem_to_smem_64kib_continuous_none_one_cta",65536,256,true,false,false,"tt_g2s_q65536_none",reinterpret_cast<const void*>(tt_g2s_q65536_none)},
 {"gmem_to_smem_64kib_padding_none_one_cta",65536,272,true,false,false,"tt_g2s_q65536_none",reinterpret_cast<const void*>(tt_g2s_q65536_none)},
 {"gmem_to_smem_1kib_continuous_none_all_gpu",1024,128,true,false,true,"tt_g2s_q1024_none",reinterpret_cast<const void*>(tt_g2s_q1024_none)},
 {"gmem_to_smem_1kib_padding_none_all_gpu",1024,144,true,false,true,"tt_g2s_q1024_none",reinterpret_cast<const void*>(tt_g2s_q1024_none)},
 {"gmem_to_smem_1kib_continuous_sw128_all_gpu",1024,128,true,true,true,"tt_g2s_q1024_sw128",reinterpret_cast<const void*>(tt_g2s_q1024_sw128)},
 {"gmem_to_smem_4kib_continuous_none_all_gpu",4096,128,true,false,true,"tt_g2s_q4096_none",reinterpret_cast<const void*>(tt_g2s_q4096_none)},
 {"gmem_to_smem_4kib_padding_none_all_gpu",4096,144,true,false,true,"tt_g2s_q4096_none",reinterpret_cast<const void*>(tt_g2s_q4096_none)},
 {"gmem_to_smem_4kib_continuous_sw128_all_gpu",4096,128,true,true,true,"tt_g2s_q4096_sw128",reinterpret_cast<const void*>(tt_g2s_q4096_sw128)},
 {"gmem_to_smem_8kib_continuous_none_all_gpu",8192,128,true,false,true,"tt_g2s_q8192_none",reinterpret_cast<const void*>(tt_g2s_q8192_none)},
 {"gmem_to_smem_8kib_padding_none_all_gpu",8192,144,true,false,true,"tt_g2s_q8192_none",reinterpret_cast<const void*>(tt_g2s_q8192_none)},
 {"gmem_to_smem_8kib_continuous_sw128_all_gpu",8192,128,true,true,true,"tt_g2s_q8192_sw128",reinterpret_cast<const void*>(tt_g2s_q8192_sw128)},
 {"gmem_to_smem_16kib_continuous_none_all_gpu",16384,128,true,false,true,"tt_g2s_q16384_none",reinterpret_cast<const void*>(tt_g2s_q16384_none)},
 {"gmem_to_smem_16kib_padding_none_all_gpu",16384,144,true,false,true,"tt_g2s_q16384_none",reinterpret_cast<const void*>(tt_g2s_q16384_none)},
 {"gmem_to_smem_16kib_continuous_sw128_all_gpu",16384,128,true,true,true,"tt_g2s_q16384_sw128",reinterpret_cast<const void*>(tt_g2s_q16384_sw128)},
 {"gmem_to_smem_32kib_continuous_none_all_gpu",32768,128,true,false,true,"tt_g2s_q32768_none",reinterpret_cast<const void*>(tt_g2s_q32768_none)},
 {"gmem_to_smem_32kib_padding_none_all_gpu",32768,144,true,false,true,"tt_g2s_q32768_none",reinterpret_cast<const void*>(tt_g2s_q32768_none)},
 {"gmem_to_smem_32kib_continuous_sw128_all_gpu",32768,128,true,true,true,"tt_g2s_q32768_sw128",reinterpret_cast<const void*>(tt_g2s_q32768_sw128)},
 {"gmem_to_smem_64kib_continuous_none_all_gpu",65536,256,true,false,true,"tt_g2s_q65536_none",reinterpret_cast<const void*>(tt_g2s_q65536_none)},
 {"gmem_to_smem_64kib_padding_none_all_gpu",65536,272,true,false,true,"tt_g2s_q65536_none",reinterpret_cast<const void*>(tt_g2s_q65536_none)},
 {"smem_to_gmem_1kib_continuous_none_one_cta",1024,128,false,false,false,"tt_s2g_q1024_none",reinterpret_cast<const void*>(tt_s2g_q1024_none)},
 {"smem_to_gmem_1kib_padding_none_one_cta",1024,144,false,false,false,"tt_s2g_q1024_none",reinterpret_cast<const void*>(tt_s2g_q1024_none)},
 {"smem_to_gmem_1kib_continuous_sw128_one_cta",1024,128,false,true,false,"tt_s2g_q1024_sw128",reinterpret_cast<const void*>(tt_s2g_q1024_sw128)},
 {"smem_to_gmem_4kib_continuous_none_one_cta",4096,128,false,false,false,"tt_s2g_q4096_none",reinterpret_cast<const void*>(tt_s2g_q4096_none)},
 {"smem_to_gmem_4kib_padding_none_one_cta",4096,144,false,false,false,"tt_s2g_q4096_none",reinterpret_cast<const void*>(tt_s2g_q4096_none)},
 {"smem_to_gmem_4kib_continuous_sw128_one_cta",4096,128,false,true,false,"tt_s2g_q4096_sw128",reinterpret_cast<const void*>(tt_s2g_q4096_sw128)},
 {"smem_to_gmem_8kib_continuous_none_one_cta",8192,128,false,false,false,"tt_s2g_q8192_none",reinterpret_cast<const void*>(tt_s2g_q8192_none)},
 {"smem_to_gmem_8kib_padding_none_one_cta",8192,144,false,false,false,"tt_s2g_q8192_none",reinterpret_cast<const void*>(tt_s2g_q8192_none)},
 {"smem_to_gmem_8kib_continuous_sw128_one_cta",8192,128,false,true,false,"tt_s2g_q8192_sw128",reinterpret_cast<const void*>(tt_s2g_q8192_sw128)},
 {"smem_to_gmem_16kib_continuous_none_one_cta",16384,128,false,false,false,"tt_s2g_q16384_none",reinterpret_cast<const void*>(tt_s2g_q16384_none)},
 {"smem_to_gmem_16kib_padding_none_one_cta",16384,144,false,false,false,"tt_s2g_q16384_none",reinterpret_cast<const void*>(tt_s2g_q16384_none)},
 {"smem_to_gmem_16kib_continuous_sw128_one_cta",16384,128,false,true,false,"tt_s2g_q16384_sw128",reinterpret_cast<const void*>(tt_s2g_q16384_sw128)},
 {"smem_to_gmem_32kib_continuous_none_one_cta",32768,128,false,false,false,"tt_s2g_q32768_none",reinterpret_cast<const void*>(tt_s2g_q32768_none)},
 {"smem_to_gmem_32kib_padding_none_one_cta",32768,144,false,false,false,"tt_s2g_q32768_none",reinterpret_cast<const void*>(tt_s2g_q32768_none)},
 {"smem_to_gmem_32kib_continuous_sw128_one_cta",32768,128,false,true,false,"tt_s2g_q32768_sw128",reinterpret_cast<const void*>(tt_s2g_q32768_sw128)},
 {"smem_to_gmem_64kib_continuous_none_one_cta",65536,256,false,false,false,"tt_s2g_q65536_none",reinterpret_cast<const void*>(tt_s2g_q65536_none)},
 {"smem_to_gmem_64kib_padding_none_one_cta",65536,272,false,false,false,"tt_s2g_q65536_none",reinterpret_cast<const void*>(tt_s2g_q65536_none)},
 {"smem_to_gmem_1kib_continuous_none_all_gpu",1024,128,false,false,true,"tt_s2g_q1024_none",reinterpret_cast<const void*>(tt_s2g_q1024_none)},
 {"smem_to_gmem_1kib_padding_none_all_gpu",1024,144,false,false,true,"tt_s2g_q1024_none",reinterpret_cast<const void*>(tt_s2g_q1024_none)},
 {"smem_to_gmem_1kib_continuous_sw128_all_gpu",1024,128,false,true,true,"tt_s2g_q1024_sw128",reinterpret_cast<const void*>(tt_s2g_q1024_sw128)},
 {"smem_to_gmem_4kib_continuous_none_all_gpu",4096,128,false,false,true,"tt_s2g_q4096_none",reinterpret_cast<const void*>(tt_s2g_q4096_none)},
 {"smem_to_gmem_4kib_padding_none_all_gpu",4096,144,false,false,true,"tt_s2g_q4096_none",reinterpret_cast<const void*>(tt_s2g_q4096_none)},
 {"smem_to_gmem_4kib_continuous_sw128_all_gpu",4096,128,false,true,true,"tt_s2g_q4096_sw128",reinterpret_cast<const void*>(tt_s2g_q4096_sw128)},
 {"smem_to_gmem_8kib_continuous_none_all_gpu",8192,128,false,false,true,"tt_s2g_q8192_none",reinterpret_cast<const void*>(tt_s2g_q8192_none)},
 {"smem_to_gmem_8kib_padding_none_all_gpu",8192,144,false,false,true,"tt_s2g_q8192_none",reinterpret_cast<const void*>(tt_s2g_q8192_none)},
 {"smem_to_gmem_8kib_continuous_sw128_all_gpu",8192,128,false,true,true,"tt_s2g_q8192_sw128",reinterpret_cast<const void*>(tt_s2g_q8192_sw128)},
 {"smem_to_gmem_16kib_continuous_none_all_gpu",16384,128,false,false,true,"tt_s2g_q16384_none",reinterpret_cast<const void*>(tt_s2g_q16384_none)},
 {"smem_to_gmem_16kib_padding_none_all_gpu",16384,144,false,false,true,"tt_s2g_q16384_none",reinterpret_cast<const void*>(tt_s2g_q16384_none)},
 {"smem_to_gmem_16kib_continuous_sw128_all_gpu",16384,128,false,true,true,"tt_s2g_q16384_sw128",reinterpret_cast<const void*>(tt_s2g_q16384_sw128)},
 {"smem_to_gmem_32kib_continuous_none_all_gpu",32768,128,false,false,true,"tt_s2g_q32768_none",reinterpret_cast<const void*>(tt_s2g_q32768_none)},
 {"smem_to_gmem_32kib_padding_none_all_gpu",32768,144,false,false,true,"tt_s2g_q32768_none",reinterpret_cast<const void*>(tt_s2g_q32768_none)},
 {"smem_to_gmem_32kib_continuous_sw128_all_gpu",32768,128,false,true,true,"tt_s2g_q32768_sw128",reinterpret_cast<const void*>(tt_s2g_q32768_sw128)},
 {"smem_to_gmem_64kib_continuous_none_all_gpu",65536,256,false,false,true,"tt_s2g_q65536_none",reinterpret_cast<const void*>(tt_s2g_q65536_none)},
 {"smem_to_gmem_64kib_padding_none_all_gpu",65536,272,false,false,true,"tt_s2g_q65536_none",reinterpret_cast<const void*>(tt_s2g_q65536_none)},
};
struct TtArtifact {std::string path,sha,dtype;std::vector<gh::u64> shape;};
static const char* tt_profiles[]={"tensor_short_1_seed0_v1","tensor_short_2_seed3_v1","tensor_short_33_seed4294967295_v1"};
int main(int argc,char** argv) try {
 const auto device=gh::device();gh::emit_device(device);
 if(argc==2&&std::string(argv[1])=="device")return 0;
 if(argc!=5||std::string(argv[1])!="validate-only")throw std::runtime_error("S15 short-only: validate-only CASE PROFILE SEED; formal gate not enabled");
 const TtCase* selected=nullptr;for(const auto& c:tt_cases)if(c.id==std::string(argv[2]))selected=&c;if(!selected)throw std::runtime_error("unknown S15 case");const auto& c=*selected;
 unsigned profile=3;for(unsigned i=0;i<3;++i)if(std::string(argv[3])==tt_profiles[i])profile=i;if(profile==3)throw std::runtime_error("unknown S15 profile");
 const unsigned lengths[]={1,2,33},seeds[]={0,3,4294967295u};int iterations=lengths[profile];unsigned seed=gh::integer(argv[4],0,4294967295ull);if(seed!=seeds[profile])throw std::runtime_error("S15 fixed profile/seed pair");
 namespace ref=tma_tensor_2d_reference;const auto l=ref::layout(c.q,c.p,c.sw128);const size_t shared=c.q+1056;
 GH_CUDA(cudaFuncSetAttribute(c.function,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));cudaFuncAttributes attr{};GH_CUDA(cudaFuncGetAttributes(&attr,c.function));int occupancy=0;
 GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,c.function,128,shared));
 if(occupancy<1||attr.localSizeBytes||attr.sharedSizeBytes+shared>device.prop.sharedMemPerBlockOptin)throw std::runtime_error("S15 unsupported resources/spill");
 const unsigned blocks=c.all_gpu?std::min(4,occupancy)*device.prop.multiProcessorCount:1;
 const gh::u64 allocation=ref::allocation_bytes(blocks,l),elements=c.q/2,rows=ref::global_rows(blocks,l),global_elements=(allocation-256)/2;
 const gh::u64 capture_elements=ref::multiply(ref::multiply(blocks,iterations),elements),capture_bytes=ref::multiply(capture_elements,2);
 const gh::u64 controls=ref::multiply(blocks,sizeof(gh::Stamp)+5*sizeof(std::uint64_t));size_t free_bytes=0,total_bytes=0;GH_CUDA(cudaMemGetInfo(&free_bytes,&total_bytes));
 if(allocation>free_bytes||capture_bytes>(free_bytes-allocation)/2||controls>free_bytes-allocation-2*capture_bytes)throw std::runtime_error("S15 insufficient memory budget");
 std::uint16_t *global=nullptr,*logical=nullptr,*physical=nullptr;gh::Stamp* stamps=nullptr;std::uint64_t* completion=nullptr;
 GH_CUDA(cudaMalloc(&global,allocation));GH_CUDA(cudaMalloc(&logical,capture_bytes));GH_CUDA(cudaMalloc(&physical,capture_bytes));GH_CUDA(cudaMalloc(&stamps,blocks*sizeof(gh::Stamp)));GH_CUDA(cudaMalloc(&completion,blocks*5*sizeof(std::uint64_t)));
 auto* payload=global+64;if(reinterpret_cast<std::uintptr_t>(payload)%128)throw std::runtime_error("S15 descriptor base alignment");
 alignas(64) CUtensorMap map{};const cuuint64_t dims[]={l.w,rows},strides[]={l.p};const cuuint32_t box[]={l.w,l.h},step[]={1,1};
 const auto encoded=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT16,2,payload,dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,c.sw128?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 if(encoded!=CUDA_SUCCESS)throw std::runtime_error("S15 cuTensorMapEncodeTiled failed code="+std::to_string(int(encoded)));
 std::vector<std::uint16_t> initial(allocation/2);
 for(unsigned i=0;i<64;++i){initial[i]=ref::guard(i);initial[global_elements+64+i]=ref::guard(64+i);}
 for(gh::u64 row=0;row<rows;++row)for(unsigned column=0;column<l.p/2;++column){
  const unsigned b=row/(32*l.h),slot=(row/l.h)%32,y=row%l.h;
  const auto expected=column<l.w?ref::input(b,slot,column,y,seed,c.g2s):ref::padding(row,column);
  initial[64+row*(l.p/2)+column]=column<l.w&&!c.g2s?ref::poison(expected):expected;
 }
 GH_CUDA(cudaMemcpy(global,initial.data(),allocation,cudaMemcpyHostToDevice));
 std::vector<std::uint16_t> logical_poison(capture_elements),physical_poison(capture_elements);
 for(gh::u64 index=0;index<capture_elements;++index){const unsigned b=index/(iterations*elements),i=(index/elements)%iterations,n=index%elements;
  logical_poison[index]=ref::poison(ref::input(b,i%32,n%l.w,n/l.w,seed,c.g2s));
  const unsigned logical_n=ref::physical_index(n%l.w,n/l.w,l);physical_poison[index]=ref::poison(ref::input(b,i%32,logical_n%l.w,logical_n/l.w,seed,c.g2s));
 }
 GH_CUDA(cudaMemcpy(logical,logical_poison.data(),capture_bytes,cudaMemcpyHostToDevice));GH_CUDA(cudaMemcpy(physical,physical_poison.data(),capture_bytes,cudaMemcpyHostToDevice));
 GH_CUDA(cudaMemset(stamps,0xff,blocks*sizeof(gh::Stamp)));GH_CUDA(cudaMemset(completion,0xff,blocks*5*sizeof(std::uint64_t)));
 bool capture=true;unsigned pitch=c.p;void* args[]={&map,&payload,&pitch,&iterations,&seed,&capture,&logical,&physical,&stamps,&completion};
 GH_CUDA(cudaLaunchKernel(c.function,dim3(blocks),dim3(128),args,shared));GH_CUDA(cudaDeviceSynchronize());
 std::vector<std::uint16_t> logical_values(capture_elements),physical_values(capture_elements),global_values(allocation/2);
 std::vector<gh::Stamp> host_stamps(blocks);std::vector<std::uint64_t> counts(blocks*5);
 GH_CUDA(cudaMemcpy(logical_values.data(),logical,capture_bytes,cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(physical_values.data(),physical,capture_bytes,cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(global_values.data(),global,allocation,cudaMemcpyDeviceToHost));
 GH_CUDA(cudaMemcpy(host_stamps.data(),stamps,blocks*sizeof(gh::Stamp),cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(counts.data(),completion,blocks*5*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
 gh::u64 checked=0,errors=0;auto check=[&](unsigned actual,unsigned expected){++checked;if(actual!=expected)++errors;};
 for(gh::u64 index=0;index<capture_elements;++index){const unsigned b=index/(iterations*elements),i=(index/elements)%iterations,n=index%elements;
  check(logical_values[index],ref::input(b,i%32,n%l.w,n/l.w,seed,c.g2s));const unsigned logical_n=ref::physical_index(n%l.w,n/l.w,l);check(physical_values[index],ref::input(b,i%32,logical_n%l.w,logical_n/l.w,seed,c.g2s));
 }
 for(gh::u64 index=0;index<global_values.size();++index){unsigned expected=initial[index];if(!c.g2s&&index>=64&&index<64+global_elements){const gh::u64 at=index-64,row=at/(l.p/2);const unsigned column=at%(l.p/2),slot=(row/l.h)%32;
   if(column<l.w&&(iterations>=32||slot<unsigned(iterations)))expected=ref::input(row/(32*l.h),slot,column,row%l.h,seed,false);
  }check(global_values[index],expected);
 }
 const auto poison=std::numeric_limits<gh::u64>::max();std::vector<std::uint64_t> stamp_words;
 for(unsigned b=0;b<blocks;++b){const auto& t=host_stamps[b];const auto at=gh::u64(b)*5;
  if(t.smid==~unsigned(0)||t.begin_ns==poison||t.end_ns==poison||t.begin_cycle==poison||t.end_cycle==poison||t.end_ns<t.begin_ns||t.end_cycle<t.begin_cycle||counts[at]!=unsigned(iterations)||(c.g2s?(counts[at+1]<unsigned(iterations)||counts[at+1]==poison):counts[at+1]!=0)||counts[at+2]!=0||counts[at+3]!=0||counts[at+4]!=(c.g2s?0:unsigned(iterations)))++errors;
  for(auto value:{t.begin_ns,t.end_ns,t.begin_cycle,t.end_cycle,gh::u64(t.smid)})stamp_words.push_back(value);
 }
 std::vector<TtArtifact> artifacts;
 auto data_artifact=[&](const char* name,const std::vector<std::uint16_t>& values,std::vector<gh::u64> shape){artifacts.push_back({name,tensor_artifacts::write(name,values),"uint16",std::move(shape)});};
 data_artifact("tensor_logical.u16le",logical_values,{blocks,gh::u64(iterations),l.h,l.w});data_artifact("tensor_physical.u16le",physical_values,{blocks,gh::u64(iterations),l.h,l.w});data_artifact("tensor_global.u16le",global_values,{allocation/2});
 const auto count_values=word_artifacts::split_u64(counts),stamp_values=word_artifacts::split_u64(stamp_words);
 artifacts.push_back({"tensor_completion.u32le",word_artifacts::write_words("tensor_completion.u32le",count_values),"uint32",{blocks,5,2}});artifacts.push_back({"tensor_stamps.u32le",word_artifacts::write_words("tensor_stamps.u32le",stamp_values),"uint32",{blocks,5,2}});
 const auto* map_bytes=reinterpret_cast<const std::uint8_t*>(&map);const std::vector<std::uint8_t> descriptor(map_bytes,map_bytes+128);const auto descriptor_hash=tensor_artifacts::write("tensor_descriptor.u8",descriptor);artifacts.push_back({"tensor_descriptor.u8",descriptor_hash,"uint8",{128}});
 std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"<<gh::quote(c.id)<<",\"profile_id\":"<<gh::quote(tt_profiles[profile])<<",\"seed\":"<<seed<<",\"scope\":"<<gh::quote(c.all_gpu?"all_gpu":"one_cta")<<",\"threads\":128,\"blocks\":"<<blocks<<",\"errors\":"<<errors<<",\"performance_eligible\":false,\"warmup_executed\":false,\"pilot_executed\":false,\"target_launches\":[{\"launch_index\":0,\"iterations\":"<<iterations<<",\"input_profile\":\"nonuniform_uint16_tensor_pattern\",\"threads\":128,\"blocks\":"<<blocks<<"}],\"checks\":[{\"launch_index\":0,\"reference_model\":\"tma_tensor_2d_word_reference_v1\",\"reference_sha256\":\"58791bc1c67434b450af2be8115f9ae47335a9f2c9d45519ef27171483ed3ec6\",\"comparison\":\"exact\",\"tolerance_id\":null,\"checked_elements\":"<<checked<<",\"expected_elements\":"<<2*capture_elements+allocation/2<<",\"errors\":"<<errors<<",\"completed\":"<<(errors?"false":"true")<<",\"verified_CTA_ids\":[";
 for(unsigned b=0;b<blocks;++b)std::cout<<(b?",":"")<<b;std::cout<<"],\"output_artifacts\":[";
 for(size_t i=0;i<artifacts.size();++i){const auto& a=artifacts[i];std::cout<<(i?",":"")<<"{\"path\":"<<gh::quote(a.path)<<",\"sha256\":"<<gh::quote(a.sha)<<",\"dtype\":"<<gh::quote(a.dtype)<<",\"evidence_kind\":\"full_values\",\"shape\":[";for(size_t j=0;j<a.shape.size();++j)std::cout<<(j?",":"")<<a.shape[j];std::cout<<"]}";}
 std::cout<<"]}],\"resource_identity\":{\"kernel_symbol\":"<<gh::quote(c.symbol)<<",\"registers_per_thread\":"<<attr.numRegs<<",\"static_smem_bytes\":"<<attr.sharedSizeBytes<<",\"dynamic_smem_bytes\":"<<shared<<",\"local_size_bytes\":"<<attr.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<occupancy<<",\"extensions\":{\"payload_bytes\":"<<c.q<<",\"row_stride_bytes\":"<<c.p<<",\"global_slots_per_cta\":32,\"global_allocation_bytes\":"<<allocation<<",\"capture_bytes_per_array\":"<<capture_bytes<<",\"data_type\":\"UINT16\",\"interleave\":\"none\",\"l2_promotion\":\"none\",\"oob_fill\":\"none\",\"global_base_address\":"<<gh::u64(reinterpret_cast<std::uintptr_t>(payload))<<",\"descriptor_host_address\":"<<gh::u64(reinterpret_cast<std::uintptr_t>(&map))<<",\"tensor_rank\":2,\"global_dimensions\":["<<l.w<<','<<rows<<"],\"box_dimensions\":["<<l.w<<','<<l.h<<"],\"element_strides\":[1,1],\"swizzle\":"<<gh::quote(c.sw128?"SW128":"none")<<",\"descriptor_encode_status\":"<<int(encoded)<<",\"descriptor_sha256\":"<<gh::quote(descriptor_hash)<<",\"descriptor_alignment_bytes\":64,\"global_base_alignment_bytes\":128,\"shared_tile_alignment_bytes\":1024}}}\n";
 GH_CUDA(cudaFree(completion));GH_CUDA(cudaFree(stamps));GH_CUDA(cudaFree(physical));GH_CUDA(cudaFree(logical));GH_CUDA(cudaFree(global));return errors?2:0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
