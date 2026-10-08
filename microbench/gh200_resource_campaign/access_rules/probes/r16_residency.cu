// R16 residency: two warpgroup CTA, stage ring, identical 1/2-CTA capacity controls.
#include "r01_support.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
namespace G=cute::SM90::GMMA;
using MMA16=G::MMA_64x64x16_F32F16F16_SS<G::Major::K,G::Major::K>;
constexpr int TileBytes16=16384;
struct Stamp16 { uint64_t start,end,start_ns,end_ns;unsigned first_sm,last_sm; };
__device__ unsigned address16(const void*p){return unsigned(__cvta_generic_to_shared(p));}
__device__ void wait16(unsigned bar,unsigned phase){unsigned done;do{
  asm volatile("{.reg .pred p;mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p,[%1],%2,64;selp.b32 %0,1,0,p;}"
  :"=r"(done):"r"(bar),"r"(phase):"memory");}while(!done);}
__device__ uint64_t clock16(uint64_t v){uint64_t out;asm volatile("{.reg .pred p;setp.ne.u64 p,%1,0;mov.u64 %0,0;@p mov.u64 %0,%%clock64;}":"=l"(out):"l"(v):"memory");return out;}
template<size_t... I>__device__ __forceinline__ void mma16(uint64_t a,uint64_t b,float*d,std::index_sequence<I...>){MMA16::fma(a,b,d[I]...,G::ScaleOut::One);}
__device__ unsigned physical16(unsigned q){return q^((q>>3)&0x38);}
__global__ void fill16(__half* dst,int blocks,int tiles,bool large){
  uint64_t count=large?uint64_t(blocks)*tiles*8192:8192;
  for(uint64_t q=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;q<count;q+=uint64_t(gridDim.x)*blockDim.x){
    unsigned within=q%8192;bool b=within>=4096;int x=(within%4096)/64,k=within%64;
    int packet=large?q/8192:0,block=packet/tiles,tile=packet%tiles;
    float value=b?(1+(x+3*k+2*block+tile)%11)/32.f:(1+(x+2*k+block+tile)%7)/16.f;
    uint64_t offset=uint64_t(packet)*8192+(b?4096:0)+physical16(within%4096);
    dst[offset]=__float2half_rn(value);
  }
}

template<int Stages>
__global__ __launch_bounds__(256,1) void resident16(const __half* source,int tiles,bool large,
                       Stamp16* stamps,float* output,unsigned short* final_tile){
  using namespace cute;
  extern __shared__ __align__(1024) unsigned char storage[];
  uint64_t* ready=reinterpret_cast<uint64_t*>(storage+65536);uint64_t* empty=ready+4;
  __shared__ volatile uint64_t gate[256];
  const int t=threadIdx.x,wg=cutlass::canonical_warp_group_idx();
  for(int q=t;q<65536/2;q+=256)reinterpret_cast<unsigned short*>(storage)[q]=0xdead;
  if(t==0)for(int j=0;j<4;++j){
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(address16(ready+j)):"memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(address16(empty+j)):"memory");
  }
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  gate[t]=(uint64_t(1)<<48)+t+1;__syncthreads();
  Stamp16 s{};s.first_sm=r_sm();s.start=clock16(gate[(t+128)%256]);s.start_ns=r_ns();
  if(wg==0){
    if(t==0)for(int tile=0;tile<tiles;++tile){
      int slot=tile%Stages;
      if(tile>=Stages)wait16(address16(empty+slot),(tile/Stages-1)%2);
      unsigned bar=address16(ready+slot),dst=address16(storage+slot*TileBytes16);
      uint64_t token;asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0,[%1],%2;"
        :"=l"(token):"r"(bar),"r"(TileBytes16):"memory");
      const __half* src=source+(large?(uint64_t(blockIdx.x)*tiles+tile)*8192:0);
      asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"
        ::"r"(dst),"l"(src),"r"(TileBytes16),"r"(bar):"memory");
    }
  }else{
    auto layout=tile_to_shape(G::Layout_K_SW128_Atom<__half>{},Shape<_64,_64>{});
    float d[32];
#pragma unroll
    for(int j=0;j<32;++j){d[j]=(t-128+1)/1024.f+j/32.f;warpgroup_fence_operand(d[j]);}
    for(int tile=0;tile<tiles;++tile){
      int slot=tile%Stages;wait16(address16(ready+slot),(tile/Stages)%2);
      auto a=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage+slot*TileBytes16)),layout);
      auto b=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage+slot*TileBytes16+8192)),layout);
      warpgroup_arrive();
#pragma unroll
      for(int k=0;k<4;++k){
        auto at=local_tile(a,make_shape(_64{},_16{}),make_coord(_0{},Int<0>{}));
        auto bt=local_tile(b,make_shape(_64{},_16{}),make_coord(_0{},Int<0>{}));
        // The swizzle-aware tensor is sliced at the four K offsets.
        auto ak=local_tile(a,make_shape(_64{},_16{}),make_coord(_0{},k));
        auto bk=local_tile(b,make_shape(_64{},_16{}),make_coord(_0{},k));
        uint64_t ad=G::make_gmma_desc<G::Major::K>(ak),bd=G::make_gmma_desc<G::Major::K>(bk);
        mma16(ad,bd,d,std::make_index_sequence<32>{});
      }
      warpgroup_commit_batch();warpgroup_wait<0>();
#pragma unroll
      for(int j=0;j<32;++j)warpgroup_fence_operand(d[j]);
      asm volatile("bar.sync 2,128;":::"memory");
      if(t==128){uint64_t token;asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0,[%1];":"=l"(token):"r"(address16(empty+slot)):"memory");}
    }
    float sum=0;
#pragma unroll
    for(int j=0;j<32;++j)asm volatile("add.rn.f32 %0,%0,%1;":"+f"(sum):"f"(d[j]):"memory");
    gate[t]=(uint64_t(1)<<48)+__float_as_uint(sum);
#pragma unroll
    for(int j=0;j<32;++j)output[(blockIdx.x*128+t-128)*32+j]=d[j];
  }
  if(wg==0)gate[t]=(uint64_t(1)<<48)+t+1;
  asm volatile("bar.sync 1,256;":::"memory");
  s.end=clock16(gate[(t+128)%256]);s.end_ns=r_ns();s.last_sm=r_sm();
  if(t==0)stamps[blockIdx.x]=s;
  int last=(tiles-1)%Stages;
  for(int q=t;q<8192;q+=256)final_tile[blockIdx.x*8192+q]=reinterpret_cast<unsigned short*>(storage+last*TileBytes16)[q];
}

struct Opt16{int stage=1,resident=1,tiles=32,large=0;
  Opt16(int argc,char**argv){for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("missing value");std::string k=argv[i];int v=std::stoi(argv[i+1]);
    if(k=="--stage")stage=v;else if(k=="--resident")resident=v;else if(k=="--tiles")tiles=v;else if(k=="--large")large=v;else throw std::runtime_error("unknown option "+k);}
    if((stage!=1&&stage!=2&&stage!=4)||(resident!=1&&resident!=2)||tiles<1||tiles>32||large<0||large>1)throw std::runtime_error("invalid R16 residency coordinate");}
};
template<int Stages>int run16(const Opt16&o){
  cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));int blocks=prop.multiProcessorCount*2;
  size_t elements=o.large?size_t(blocks)*o.tiles*8192:8192;
  DeviceBuffer<__half> source(elements);DeviceBuffer<float> result(size_t(blocks)*4096);
  DeviceBuffer<unsigned short> last(size_t(blocks)*8192);DeviceBuffer<Stamp16> stamps(blocks);
  fill16<<<512,256>>>(source.pointer,blocks,o.tiles,o.large);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
  auto kernel=resident16<Stages>;int smem=o.resident==1?131072:98304;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  cudaFuncAttributes attr{};CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));int occupancy=0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,256,smem));
  if(occupancy!=o.resident)throw std::runtime_error("queried occupancy differs from requested control");
  if(attr.localSizeBytes)throw std::runtime_error("spilled residency kernel");
  std::vector<Stamp16> times(blocks);
  auto launch=[&](){kernel<<<blocks,256,smem>>>(source.pointer,o.tiles,o.large,stamps.pointer,result.pointer,last.pointer);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(times.data(),stamps.pointer,sizeof(Stamp16)*blocks,cudaMemcpyDeviceToHost));
    uint64_t first=UINT64_MAX,end=0;for(auto s:times)first=std::min(first,s.start_ns),end=std::max(end,s.end_ns);return end-first;};
  Windows warm=warm_up(launch);uint64_t ns=launch();std::vector<float> out(result.count);std::vector<unsigned short> payload(last.count);
  CUDA_CHECK(cudaMemcpy(out.data(),result.pointer,out.size()*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(payload.data(),last.pointer,payload.size()*2,cudaMemcpyDeviceToHost));
  save_binary("accum.f32",out);save_binary("last.u16",payload);size_t errors=0;
  for(auto s:times)errors+=s.start==0||s.end<s.start||s.first_sm!=s.last_sm;
  std::cout<<std::setprecision(17)<<"{\"stage\":"<<Stages<<",\"resident\":"<<o.resident<<",\"tiles\":"<<o.tiles<<",\"large\":"<<o.large<<",\"blocks\":"<<blocks
    <<",\"threads\":256,\"source_bytes\":"<<elements*2<<",\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes<<",\"occupancy_limit\":"<<occupancy
    <<",\"reserved_smem\":"<<smem<<",\"elapsed_ns\":"<<ns<<",\"event_errors\":"<<errors<<",\"warmup_converged\":"<<(warm.converged?"true":"false")<<",\"warmup\":";
  vector_json(warm.warmup);std::cout<<",\"stamps\":[";for(int i=0;i<blocks;++i){auto s=times[i];std::cout<<(i?",":"")<<'['<<s.start<<','<<s.end<<','<<s.start_ns<<','<<s.end_ns<<','<<s.first_sm<<','<<s.last_sm<<']';}std::cout<<"]}\n";
  return errors?2:warm.converged?0:3;
}
int main(int argc,char**argv){try{Opt16 o(argc,argv);if(o.stage==1)return run16<1>(o);if(o.stage==2)return run16<2>(o);return run16<4>(o);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
