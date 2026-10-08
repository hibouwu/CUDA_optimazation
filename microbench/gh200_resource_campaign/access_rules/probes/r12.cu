// R12: independent TMA, WGMMA and STS regions; fixed 384-thread role assignment.
#include "r01_support.hpp"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cstring>
namespace G=cute::SM90::GMMA;
using MMA=G::MMA_64x128x16_F32F16F16_SS<G::Major::K,G::Major::MN>;
constexpr int MatrixBytes=24576,PayloadBytes=65536,TotalBytes=155648;
struct Stamp12 { uint64_t start,end;unsigned sm; };

__device__ unsigned smem_address(const void* p){return unsigned(__cvta_generic_to_shared(p));}
__device__ uint64_t ready_clock(uint64_t v){
  uint64_t t;asm volatile("{.reg .pred p;setp.ne.u64 p,%1,0;mov.u64 %0,0;@p mov.u64 %0,%%clock64;}"
                          :"=l"(t):"l"(v):"memory");return t;
}
__device__ void wait_tx(unsigned bar,unsigned phase){
  unsigned done;
  do{asm volatile("{.reg .pred p;mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p,[%1],%2,64;"
                  "selp.b32 %0,1,0,p;}":"=r"(done):"r"(bar),"r"(phase):"memory");}while(!done);
}
template<size_t... I>
__device__ __forceinline__ void mma12(uint64_t a,uint64_t b,float* d,std::index_sequence<I...>){
  MMA::fma(a,b,d[I]...,G::ScaleOut::One);
}
__host__ __device__ float a12(int r,int k){return (1+(r+2*k)%7)/16.f;}
__host__ __device__ float b12(int c,int k){return (1+(c+3*k)%11)/32.f;}
__host__ __device__ unsigned short sts12(int q,int unit){return static_cast<unsigned short>(((q*17+unit*31)^0x1234)&65535);}

template<bool Swizzle> __device__ auto matrix_a_layout(){
  using namespace cute;
  if constexpr(Swizzle)return tile_to_shape(G::Layout_K_SW128_Atom<__half>{},Shape<_64,_64>{});
  else return tile_to_shape(G::Layout_K_INTER_Atom<__half>{},Shape<_64,_64>{});
}
template<bool Swizzle> __device__ auto matrix_b_layout(){
  using namespace cute;
  if constexpr(Swizzle)return tile_to_shape(G::Layout_MN_SW128_Atom<__half>{},Shape<_128,_64>{},Step<_2,_1>{});
  else return tile_to_shape(G::Layout_MN_INTER_Atom<__half>{},Shape<_128,_64>{},Step<_2,_1>{});
}

// The transport/store tiles have 64 b16 elements per row. SW128 changes only
// the XOR permutation inside each aligned 1024 B region, not logical payload.
__device__ __forceinline__ unsigned physical_byte(int row,int col,bool swizzle){
  unsigned byte=(row*64+col)*2;
  return swizzle?byte^((byte>>3)&0x70):byte;
}

// Pair0 TMA+MMA, pair1 MMA+STS, pair2 TMA+STS. Order0 A,1 B,2 serial,3 concurrent.
template<int Pair,bool Swizzle,int Order>
__global__ __launch_bounds__(384,1) void r12_probe(const __grid_constant__ CUtensorMap map,
                      const float* initial,int units,Stamp12* stamps,float* accum,
                      unsigned short* snapshot){
  using namespace cute;
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ __align__(8) uint64_t barrier;
  __shared__ volatile uint64_t gate[384];
  const int t=threadIdx.x,wg=cutlass::canonical_warp_group_idx();
  for(int q=t;q<TotalBytes/2;q+=384)reinterpret_cast<unsigned short*>(storage)[q]=0xdead;
  __syncthreads();
  auto la=matrix_a_layout<Swizzle>();auto lb=matrix_b_layout<Swizzle>();
  auto a=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)),la);
  auto b=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage+8192)),lb);
  for(int q=t;q<4096;q+=384)a(q/64,q%64)=__float2half_rn(a12(q/64,q%64));
  for(int q=t;q<8192;q+=384)b(q/64,q%64)=__float2half_rn(b12(q/64,q%64));
  if(t==0)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(smem_address(&barrier)):"memory");
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  __syncthreads();
  auto at=local_tile(a,make_shape(_64{},_16{}),make_coord(_0{},_0{}));
  auto bt=local_tile(b,make_shape(_128{},_16{}),make_coord(_0{},_0{}));
  uint64_t ad=G::make_gmma_desc<G::Major::K>(at),bd=G::make_gmma_desc<G::Major::MN>(bt);
  const unsigned bar=smem_address(&barrier);
  auto rendezvous=[&](uint64_t v){
    gate[t]=v;asm volatile("bar.sync 1,384;":::"memory");
    const uint64_t peer=gate[(t+128)%384];if(peer==0)asm volatile("trap;":::"memory");return peer;
  };
  constexpr bool FirstA=Order==0||Order==2||Order==3;
  constexpr bool FirstB=Order==3||Order==4;
  constexpr bool SecondA=Order==5;
  constexpr bool SecondB=Order==1||Order==2;
  constexpr bool TmaFirst=(Pair==0||Pair==2)&&FirstA;
  constexpr bool TmaSecond=(Pair==0||Pair==2)&&SecondA;
  constexpr bool MmaFirst=(Pair==0&&FirstB)||(Pair==1&&FirstA);
  constexpr bool MmaSecond=(Pair==0&&SecondB)||(Pair==1&&SecondA);
  constexpr bool StsFirst=(Pair==1||Pair==2)&&FirstB;
  constexpr bool StsSecond=(Pair==1||Pair==2)&&SecondB;
  Stamp12 s{};s.sm=r_sm();
  if(wg==1){
    float d[64];
#pragma unroll
    for(int j=0;j<64;++j){d[j]=reinterpret_cast<const volatile float*>(initial)[(t-128)*64+j];warpgroup_fence_operand(d[j]);}
    auto consume=[&](int epoch){float sum=float(epoch)/1024.f;
#pragma unroll
      for(int j=0;j<64;++j){warpgroup_fence_operand(d[j]);asm volatile("add.rn.f32 %0,%0,%1;":"+f"(sum):"f"(d[j]):"memory");}
      return uint64_t(__float_as_uint(sum))+(uint64_t(1)<<48);
    };
    s.start=ready_clock(rendezvous(consume(0)));
    for(int unit=0;unit<units;++unit){
      auto compute=[&](){
#pragma unroll 1
        for(int batch=0;batch<8;++batch){
          warpgroup_arrive();
#pragma unroll
          for(int j=0;j<8;++j)mma12(ad,bd,d,std::make_index_sequence<64>{});
          warpgroup_commit_batch();warpgroup_wait<1>();
        }
        warpgroup_wait<0>();
#pragma unroll
        for(int j=0;j<64;++j)warpgroup_fence_operand(d[j]);
      };
      if constexpr(MmaFirst)compute();rendezvous(consume(unit*2+1));
      if constexpr(MmaSecond)compute();rendezvous(consume(unit*2+2));
    }
    s.end=ready_clock(gate[(t+128)%384]);
#pragma unroll
    for(int j=0;j<64;++j)accum[(t-128)*64+j]=d[j];
  }else{
    s.start=ready_clock(rendezvous((uint64_t(1)<<48)+t+1));
    for(int unit=0;unit<units;++unit){
      auto transport=[&](){if(t==0){
        uint64_t token;
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0,[%1],%2;"
          :"=l"(token):"r"(bar),"r"(PayloadBytes):"memory");
#pragma unroll
        for(int i=0;i<4;++i){
          unsigned dst=smem_address(storage+MatrixBytes+i*16384);int x=0,y=i*128;
          asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%0],[%1,{%2,%3}],[%4];"::"r"(dst),"l"(&map),"r"(x),"r"(y),"r"(bar):"memory");
        }
        wait_tx(bar,unit%2);
      }};
      auto stores=[&](){
        if(wg==2){
#pragma unroll 1
          for(int j=0;j<32;++j){
            int vector=(t-256)+j*128,row=vector/8,col=(vector%8)*8;
            unsigned dst=smem_address(storage+MatrixBytes+PayloadBytes)+physical_byte(row,col,Swizzle);
            unsigned words[4];
#pragma unroll
            for(int k=0;k<4;++k){int q=row*64+col+k*2;words[k]=unsigned(sts12(q,unit))|(unsigned(sts12(q+1,unit))<<16);}
            asm volatile("st.volatile.shared.v4.u32 [%0],{%1,%2,%3,%4};"::"r"(dst),"r"(words[0]),"r"(words[1]),"r"(words[2]),"r"(words[3]):"memory");
          }
        }
      };
      if constexpr(TmaFirst)transport();
      if constexpr(StsFirst)stores();
      rendezvous((uint64_t(1)<<48)+unit+1);
      if constexpr(TmaSecond)transport();
      if constexpr(StsSecond)stores();
      rendezvous((uint64_t(1)<<48)+unit+1);
    }
    s.end=ready_clock(gate[(t+128)%384]);
  }
  if(t%32==0)stamps[t/32]=s;
  // Untimed full physical snapshot: layout and untouched-region checks are CPU-side.
  for(int q=t;q<TotalBytes/2;q+=384)snapshot[q]=reinterpret_cast<volatile unsigned short*>(storage)[q];
}

struct Options12{
  int pair=0,swizzle=0,order=0,units=32;
  Options12(int argc,char**argv){
    for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("missing value");std::string k=argv[i];int v=std::stoi(argv[i+1]);
      if(k=="--pair")pair=v;else if(k=="--swizzle")swizzle=v;else if(k=="--order")order=v;else if(k=="--units")units=v;else throw std::runtime_error("unknown option "+k);
    }
    if(pair<0||pair>2||swizzle<0||swizzle>1||order<0||order>6||units<1||units>32)throw std::runtime_error("invalid R12 coordinate");
  }
};
CUtensorMap map12(void* source,bool swizzle){
  CUtensorMap map;uint64_t dims[2]={64,512},strides[1]={128};uint32_t box[2]={64,128},elements[2]={1,1};
  CUresult status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT16,2,source,dims,strides,box,elements,
    CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(status!=CUDA_SUCCESS)throw std::runtime_error("tensor map encode failed");return map;
}
template<int Pair,bool Swizzle,int Order>int run12(const Options12& o){
  std::vector<unsigned short> source(PayloadBytes/2);
  for(int q=0;q<PayloadBytes/2;++q)source[q]=static_cast<unsigned short>((q*13+17)^0x4321);
  std::vector<float> initial(128*64);
  for(int t=0;t<128;++t)for(int j=0;j<64;++j)initial[t*64+j]=(t+1)/1024.f+j/32.f;
  DeviceBuffer<unsigned short> src(source.size()),snapshot(TotalBytes/2);
  DeviceBuffer<float> init(initial.size()),accum(initial.size());DeviceBuffer<Stamp12> stamps(12);
  CUDA_CHECK(cudaMemcpy(src.pointer,source.data(),source.size()*2,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(init.pointer,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));
  CUtensorMap map=map12(src.pointer,Swizzle);auto kernel=r12_probe<Pair,Swizzle,Order>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,TotalBytes));
  cudaFuncAttributes attr{};CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
  int occupancy=0;CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,384,TotalBytes));
  std::vector<Stamp12> times(12);
  auto launch=[&](){kernel<<<1,384,TotalBytes>>>(map,init.pointer,o.units,stamps.pointer,accum.pointer,snapshot.pointer);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(times.data(),stamps.pointer,sizeof(Stamp12)*12,cudaMemcpyDeviceToHost));
    uint64_t start=UINT64_MAX,end=0;for(auto s:times)start=std::min(start,s.start),end=std::max(end,s.end);return end-start;};
  Windows w=warm_up(launch);uint64_t elapsed=launch();
  std::vector<float> d(accum.count);std::vector<unsigned short> shared(snapshot.count);
  CUDA_CHECK(cudaMemcpy(d.data(),accum.pointer,d.size()*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(shared.data(),snapshot.pointer,shared.size()*2,cudaMemcpyDeviceToHost));
  save_binary("accum.f32",d);save_binary("shared.u16",shared);save_binary("source.u16",source);save_binary("initial.f32",initial);
  size_t errors=0;for(auto s:times)errors+=s.start==0||s.end<s.start||s.sm!=times[0].sm;
  std::cout<<std::setprecision(17)<<"{\"pair\":"<<Pair<<",\"swizzle\":"<<Swizzle<<",\"order\":"<<Order<<",\"units\":"<<o.units
    <<",\"threads\":384,\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes<<",\"static_shared_bytes\":"<<attr.sharedSizeBytes
    <<",\"dynamic_shared_bytes\":"<<TotalBytes<<",\"occupancy_limit\":"<<occupancy<<",\"elapsed_cycles\":"<<elapsed
    <<",\"event_errors\":"<<errors<<",\"warmup_converged\":"<<(w.converged?"true":"false")<<",\"warmup\":";vector_json(w.warmup);
  std::cout<<",\"stamps\":[";for(size_t i=0;i<times.size();++i){auto s=times[i];std::cout<<(i?",":"")<<'['<<s.start<<','<<s.end<<','<<s.sm<<']';}std::cout<<"]}\n";
  return errors?2:w.converged?0:3;
}
template<int P,bool S>int choose12(const Options12&o){
  switch(o.order){case 0:return run12<P,S,0>(o);case 1:return run12<P,S,1>(o);case 2:return run12<P,S,2>(o);case 3:return run12<P,S,3>(o);case 4:return run12<P,S,4>(o);case 5:return run12<P,S,5>(o);case 6:return run12<P,S,6>(o);}return 1;
}
template<int P>int layout12(const Options12&o){return o.swizzle?choose12<P,true>(o):choose12<P,false>(o);}
int main(int argc,char**argv){try{Options12 o(argc,argv);if(o.pair==0)return layout12<0>(o);if(o.pair==1)return layout12<1>(o);return layout12<2>(o);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
