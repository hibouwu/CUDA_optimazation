// R13: TMA supply, consumer completion, slot retirement. Four slots always reserved.
#include "gaps_common.hpp"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <fstream>
#include <utility>
namespace G = cute::SM90::GMMA;
constexpr int Tiles = 32;
struct Stamp { uint64_t bc, ec, bn, en; unsigned first_sm, last_sm; };
// Trace pair1 word3 is retire_arrive_start before the WG empty.arrive.
// Trace pair2 producer record word3/4 is slot_reusable_observed/refill_issue_before_A.
// Both are observed boundaries; neither identifies the physical barrier completion instant.
struct Event { uint64_t issue, wait_return, consume, release, refill; };
__device__ unsigned smid() { unsigned x; asm volatile("mov.u32 %0, %%smid;":"=r"(x)); return x; }
__device__ uint64_t nanos() { uint64_t x; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)); return x; }
__device__ unsigned shared_address(const void* p) { return unsigned(__cvta_generic_to_shared(p)); }
__device__ void wait_ready(unsigned bar, unsigned phase) {
  unsigned done;
  do {
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p,[%1],%2,64;"
                 "selp.b32 %0,1,0,p;}" :"=r"(done):"r"(bar),"r"(phase):"memory");
  } while (!done);
}
#if defined(R13_TRACE) && R13_TRACE_PAIR == 2
__device__ __forceinline__ uint64_t reuse_wait_stamp(unsigned bar, unsigned phase) {
  uint64_t cycle=0;
  // Keep acquire wait, its successful predicate, and the clock in one PTX block.
  asm volatile("{.reg .pred ready; R13ReuseWait:\n"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready,[%1],%2,64;\n"
               "@!ready bra R13ReuseWait;\n"
               // Both immediate success and retry exit have ready=true. Do not predicate
               // the output again: CUDA12.9 lowered it to a stale retry-only P2 SEL in v10.
               "mov.u64 %0, %%clock64;\n}"
               :"+l"(cycle):"r"(bar),"r"(phase):"memory");
  return cycle;
}
#endif
// A real shared consumer blocks the end clock after all CTA participants have published.
__device__ void dependent_stamp(const volatile unsigned* word, uint64_t& cycle, uint64_t& ns) {
  unsigned value = *word;
  asm volatile("{.reg .pred p; setp.ne.u32 p,%2,0xffffffff;"
               "@p mov.u64 %0,%%clock64; @p mov.u64 %1,%%globaltimer;"
               "@!p mov.u64 %0,0; @!p mov.u64 %1,0;}"
               :"=l"(cycle),"=l"(ns):"r"(value):"memory");
}
template<class Mma, size_t... I>
__device__ void mma(uint64_t a, uint64_t b, float* d, std::index_sequence<I...>) {
  Mma::fma(a,b,d[I]...,G::ScaleOut::One);
}
__device__ __forceinline__ void retire_tile(int tile,int stages,int trace_tile,int tid,int wg,
    volatile unsigned* publication,uint64_t* empty,Event* events,unsigned published_value) {
    publication[tid]=published_value; // independent of asynchronous accumulator writes
    if(wg==1)asm volatile("bar.sync 1,128;":::"memory");
    else asm volatile("bar.sync 2,128;":::"memory");
    if(tid%128==0){
      #if defined(R13_TRACE) && R13_TRACE_PAIR == 1
      if(blockIdx.x==0 && tile==trace_tile){
        uint64_t cycle,ns;dependent_stamp(&publication[tid+127],cycle,ns);
        auto& event=events[(blockIdx.x*Tiles+tile)*2+wg-1];
        event.consume=cycle;
        event.release=clock64(); // retire_arrive_start, not completed slot release
      }
      #endif
      uint64_t token;
      asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0,[%1];"
        :"=l"(token):"r"(shared_address(empty+tile%stages)):"memory");
    }
}
template<int N, int Consumer, int Stages>
__global__ __launch_bounds__(384,1) void supply(
    const __grid_constant__ CUtensorMap amap, const __grid_constant__ CUtensorMap bmap,
    int trace_tile, Stamp* stamps, Event* events, float* accum, unsigned* final_input) {
  using namespace cute;
  constexpr int Bytes = (128+N)*64*2, Regs=N/2;
  extern __shared__ __align__(1024) unsigned char data[];
  uint64_t* bars=reinterpret_cast<uint64_t*>(data+4*Bytes);
  uint64_t* empty=bars+4;
  __shared__ volatile unsigned publication[384];
  auto la=tile_to_shape(G::Layout_K_SW128_Atom<__half>{},Shape<_128,_64>{});
  auto lb=tile_to_shape(G::Layout_MN_SW128_Atom<__half>{},Shape<Int<N>,_64>{},Step<_2,_1>{});
  int tid=threadIdx.x, wg=tid/128;
  if(tid==0)for(int s=0;s<4;++s){
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(shared_address(bars+s)):"memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0],2;"::"r"(shared_address(empty+s)):"memory");
  }
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  float d[Regs];
  #pragma unroll
  for(int j=0;j<Regs;++j){d[j]=0;asm volatile("":"+f"(d[j])::"memory");}
  float chain=(tid+1)/1024.f;
  publication[tid]=0;__syncthreads();Stamp stamp{};
  if(tid==0){stamp.first_sm=smid();dependent_stamp(&publication[383],stamp.bc,stamp.bn);}
  __syncthreads();
  if(wg==0){
    if(tid==0)for(int tile=0;tile<Tiles;++tile){
      int slot=tile%Stages;
      #if defined(R13_TRACE) && R13_TRACE_PAIR == 2
      uint64_t slot_reusable_cycle=0,refill_cycle=0;
      #endif
      if(tile>=Stages){
        #if defined(R13_TRACE) && R13_TRACE_PAIR == 2
        if(blockIdx.x==0 && tile==trace_tile+Stages){
          slot_reusable_cycle=reuse_wait_stamp(shared_address(empty+slot),(tile/Stages-1)%2);
        }else
        #endif
        wait_ready(shared_address(empty+slot),(tile/Stages-1)%2);
      }
      unsigned dst=shared_address(data+slot*Bytes),bar=shared_address(bars+slot);
      int ax=0,ay=tile*128,bx=0,by=tile*64,bz=0;
      asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0],%1;"
        ::"r"(bar),"r"(Bytes):"memory");
      #if defined(R13_TRACE) && R13_TRACE_PAIR == 0
      if(blockIdx.x==0 && tile==trace_tile){
        uint64_t now=clock64();
        for(int g=0;g<2;++g){
          events[(blockIdx.x*Tiles+tile)*2+g].issue=now;
        }
      }
      #endif
      #if defined(R13_TRACE) && R13_TRACE_PAIR == 2
      if(blockIdx.x==0 && tile==trace_tile+Stages){
        asm volatile("mov.u64 %0, %%clock64;":"=l"(refill_cycle)::"memory");
        // Boundary before first A request; both trace stores are deferred until after B.
      }
      #endif
      asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0],[%1,{%2,%3}],[%4];"::"r"(dst),"l"(&amap),"r"(ax),"r"(ay),"r"(bar):"memory");
      asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0],[%1,{%2,%3,%4}],[%5];"::"r"(dst+16384),"l"(&bmap),"r"(bx),"r"(by),"r"(bz),"r"(bar):"memory");
      #if defined(R13_TRACE) && R13_TRACE_PAIR == 2
      if(blockIdx.x==0 && tile==trace_tile+Stages){
        events[trace_tile*2].release=slot_reusable_cycle;
        events[trace_tile*2].refill=refill_cycle;
      }
      #endif
      uint64_t token;
      asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0,[%1];"
        :"=l"(token):"r"(bar):"memory");
    }
  }else{
    if constexpr(Consumer==2 && Stages>1){
      // Prologue: commit tile zero with no retirement and no wait1/0 branch.
      {
        constexpr int tile=0;
    wait_ready(shared_address(bars+tile%Stages),(tile/Stages)%2);
    #if defined(R13_TRACE) && R13_TRACE_PAIR == 0
    if(blockIdx.x==0 && tid%128==0 && tile==trace_tile)
      events[(blockIdx.x*Tiles+tile)*2+wg-1].wait_return=clock64();
    #endif
        auto a=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(data+(tile%Stages)*Bytes)),la);
        auto b=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(data+(tile%Stages)*Bytes+16384)),lb);
        // Match the batch-level WG arrive placement in the working async collective.
        // Fence does not drain pending groups; no accumulator values are read here.
        #pragma unroll
        for(int j=0;j<Regs;++j)cute::warpgroup_fence_operand(d[j]);
        asm volatile("wgmma.fence.sync.aligned;":::"memory");
        #pragma unroll
        for(int q=0;q<4;++q){
          auto av=local_tile(a,make_shape(_64{},_16{}),make_coord(wg-1,q));
          auto bv=local_tile(b,make_shape(Int<N>{},_16{}),make_coord(_0{},q));
          uint64_t ad=G::make_gmma_desc<G::Major::K>(av),bd=G::make_gmma_desc<G::Major::MN>(bv);
          using Mma=std::conditional_t<N==128,G::MMA_64x128x16_F32F16F16_SS<G::Major::K,G::Major::MN>,
            G::MMA_64x256x16_F32F16F16_SS<G::Major::K,G::Major::MN>>;
          mma<Mma>(ad,bd,d,std::make_index_sequence<Regs>{});
        }
        asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
      }
      // Uniform steady loop: group tile-1 is covered by wait1; tile remains in flight.
      #pragma unroll 1
      for(int tile=1;tile<Tiles;++tile){
    wait_ready(shared_address(bars+tile%Stages),(tile/Stages)%2);
    #if defined(R13_TRACE) && R13_TRACE_PAIR == 0
    if(blockIdx.x==0 && tid%128==0 && tile==trace_tile)
      events[(blockIdx.x*Tiles+tile)*2+wg-1].wait_return=clock64();
    #endif
        auto a=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(data+(tile%Stages)*Bytes)),la);
        auto b=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(data+(tile%Stages)*Bytes+16384)),lb);
        // Match the batch-level WG arrive placement in the working async collective.
        // Fence does not drain pending groups; no accumulator values are read here.
        #pragma unroll
        for(int j=0;j<Regs;++j)cute::warpgroup_fence_operand(d[j]);
        asm volatile("wgmma.fence.sync.aligned;":::"memory");
        #pragma unroll
        for(int q=0;q<4;++q){
          auto av=local_tile(a,make_shape(_64{},_16{}),make_coord(wg-1,q));
          auto bv=local_tile(b,make_shape(Int<N>{},_16{}),make_coord(_0{},q));
          uint64_t ad=G::make_gmma_desc<G::Major::K>(av),bd=G::make_gmma_desc<G::Major::MN>(bv);
          using Mma=std::conditional_t<N==128,G::MMA_64x128x16_F32F16F16_SS<G::Major::K,G::Major::MN>,
            G::MMA_64x256x16_F32F16F16_SS<G::Major::K,G::Major::MN>>;
          mma<Mma>(ad,bd,d,std::make_index_sequence<Regs>{});
        }
        asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");
        #pragma unroll
        for(int j=0;j<Regs;++j)cute::warpgroup_fence_operand(d[j]);
        retire_tile(tile-1,Stages,trace_tile,tid,wg,publication,empty,events,unsigned(tile));
      }
      // Drain inside the same consumer branch before reconverging with producer.
      asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
      #pragma unroll
      for(int j=0;j<Regs;++j)cute::warpgroup_fence_operand(d[j]);
      retire_tile(Tiles-1,Stages,trace_tile,tid,wg,publication,empty,events,unsigned(Tiles));
    }else{
      #pragma unroll 1
      for(int tile=0;tile<Tiles;++tile){
    wait_ready(shared_address(bars+tile%Stages),(tile/Stages)%2);
    #if defined(R13_TRACE) && R13_TRACE_PAIR == 0
    if(blockIdx.x==0 && tid%128==0 && tile==trace_tile)
      events[(blockIdx.x*Tiles+tile)*2+wg-1].wait_return=clock64();
    #endif
        if constexpr(Consumer==1){
          auto source=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(data+(tile%Stages)*Bytes)),la);
          chain=__half2float(source((tid-128)%128,0));
          #pragma unroll
          for(int i=0;i<64;++i)
            asm volatile("fma.rn.f32 %0,%0,0f3f800000,0f3a800000;":"+f"(chain));
        }
        if constexpr(Consumer==2){
        auto a=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(data+(tile%Stages)*Bytes)),la);
        auto b=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(data+(tile%Stages)*Bytes+16384)),lb);
        // Match the batch-level WG arrive placement in the working async collective.
        // Fence does not drain pending groups; no accumulator values are read here.
        #pragma unroll
        for(int j=0;j<Regs;++j)cute::warpgroup_fence_operand(d[j]);
        asm volatile("wgmma.fence.sync.aligned;":::"memory");
        #pragma unroll
        for(int q=0;q<4;++q){
          auto av=local_tile(a,make_shape(_64{},_16{}),make_coord(wg-1,q));
          auto bv=local_tile(b,make_shape(Int<N>{},_16{}),make_coord(_0{},q));
          uint64_t ad=G::make_gmma_desc<G::Major::K>(av),bd=G::make_gmma_desc<G::Major::MN>(bv);
          using Mma=std::conditional_t<N==128,G::MMA_64x128x16_F32F16F16_SS<G::Major::K,G::Major::MN>,
            G::MMA_64x256x16_F32F16F16_SS<G::Major::K,G::Major::MN>>;
          mma<Mma>(ad,bd,d,std::make_index_sequence<Regs>{});
        }
        asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
          asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
          #pragma unroll
          for(int j=0;j<Regs;++j)cute::warpgroup_fence_operand(d[j]);
        }
        retire_tile(tile,Stages,trace_tile,tid,wg,publication,empty,events,
          Consumer==1?__float_as_uint(chain):unsigned(tile+1));
      }
    }
  }
  __syncthreads(); // all producer transfers and both consumer chains have ended.
  if(tid==0){dependent_stamp(&publication[383],stamp.ec,stamp.en);stamp.last_sm=smid();stamps[blockIdx.x]=stamp;}
  // Freeze final slot witness before invalidation; output copies are outside plain window.
  int last=(Tiles-1)%Stages;
  for(int i=tid;i<Bytes/4;i+=384)
    final_input[blockIdx.x*(Bytes/4)+i]=reinterpret_cast<unsigned*>(data+last*Bytes)[i];
  if(wg>0){
    if constexpr(Consumer==2){
      #pragma unroll
      for(int j=0;j<Regs;++j)accum[(blockIdx.x*256+tid-128)*Regs+j]=d[j];
    }else accum[blockIdx.x*256+tid-128]=chain;
  }
  __syncthreads();
  if(tid==0)for(int s=0;s<8;++s)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(shared_address(bars+s)):"memory");
}
struct Options13 {
  int n=256,stages=4,consumer=2,trace_tile=16;std::string scope="one_cta";
  Options13(int argc,char**argv){for(int i=1;i<argc;i+=2){
    if(i+1==argc)throw std::runtime_error("missing option value");
    std::string k=argv[i],v=argv[i+1];
    if(k=="--n")n=std::stoi(v);else if(k=="--stages")stages=std::stoi(v);
    else if(k=="--trace-tile")trace_tile=std::stoi(v);
    else if(k=="--consumer")consumer=std::stoi(v);else if(k=="--scope")scope=v;
    else throw std::runtime_error("unknown option");}
    if((n!=128&&n!=256)||(stages!=1&&stages!=2&&stages!=4)||consumer<0||consumer>2||trace_tile<0||trace_tile>=Tiles||
      (scope!="one_cta"&&scope!="all_gpu"))throw std::runtime_error("invalid configuration");
  }
};
CUtensorMap tensor_map(void* ptr,int rows) {
  gaps::validate_layout(rows,64,64);
  CUtensorMap map;uint64_t dims[2]={64,uint64_t(rows)},stride[1]={128};
  uint32_t box[2]={64,unsigned(rows/Tiles)},element[2]={1,1};
  CUresult result=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,2,ptr,dims,stride,
    box,element,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(result!=CUDA_SUCCESS)throw std::runtime_error("tensor map encode failed");return map;
}
CUtensorMap b_tensor_map(void* ptr,int n) {
  gaps::validate_layout(64*Tiles,n,n);
  CUtensorMap map;
  // B[K,N] row-major split N into contiguous 64-column panels. This non-monotone
  // logical stride ordering gives a legal 128B SW128 box and MN SMEM panel order.
  uint64_t dims[3]={64,64*Tiles,uint64_t(n/64)},strides[2]={uint64_t(n)*2,128};
  uint32_t box[3]={64,64,unsigned(n/64)},elements[3]={1,1,1};
  CUresult result=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,3,ptr,dims,
    strides,box,elements,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(result!=CUDA_SUCCESS)throw std::runtime_error("B MN tensor map encode failed");
  return map;
}
__host__ __device__ float a_value(int row,int k){return ((row*7+k*13)%17-8)/32.f;}
__host__ __device__ float b_value(int col,int k){return ((k*5+col*11)%17-8)/32.f;}
// Tile-major inputs match logical K-major SMEM tensors, row stride 128B.
__global__ void fill13(__half* p,int outer,bool a){
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<outer*64*Tiles;i+=gridDim.x*blockDim.x){
    int tile=i/(outer*64),row=(i/64)%outer,k=tile*64+i%64;
    if(a)p[i]=__float2half_rn(a_value(row,k));
    else {int global_k=i/outer,col=i%outer;p[i]=__float2half_rn(b_value(col,global_k));}
  }
}
template<class T> void save13(const char* path,const std::vector<T>& values){
  std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),
    values.size()*sizeof(T));if(!out)throw std::runtime_error("output write failed");
}
template<int N,int Consumer,int Stages>int execute(const Options13&o){
  constexpr int Bytes=(128+N)*128,Smem=4*Bytes+8*8;
  cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  auto kernel=supply<N,Consumer,Stages>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Smem));
  cudaFuncAttributes attr{};CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));int occupancy;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,384,Smem));
  if(occupancy!=1)throw std::runtime_error("R13 requires resource limit one CTA/SM");
  int blocks=o.scope=="one_cta"?1:prop.multiProcessorCount;
  DeviceBuffer<__half>a(128*64*Tiles),b(N*64*Tiles);
  fill13<<<256,256>>>(a.pointer,128,true);fill13<<<256,256>>>(b.pointer,N,false);
  CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
  auto am=tensor_map(a.pointer,128*Tiles),bm=b_tensor_map(b.pointer,N);
  DeviceBuffer<Stamp>stamp(blocks);DeviceBuffer<Event>event(blocks*Tiles*2);
  size_t count=size_t(blocks)*256*(Consumer==2?N/2:1);
  DeviceBuffer<float>accum(count);DeviceBuffer<unsigned>last(size_t(blocks)*Bytes/4);
  std::vector<Stamp>times(blocks);std::vector<float>warmup;bool converged=false;
  auto launch=[&](){
    CUDA_CHECK(cudaMemset(event.pointer,0,event.count*sizeof(Event)));
    kernel<<<blocks,384,Smem>>>(am,bm,o.trace_tile,stamp.pointer,event.pointer,accum.pointer,last.pointer);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(times.data(),stamp.pointer,blocks*sizeof(Stamp),cudaMemcpyDeviceToHost));
    if(o.scope=="one_cta")return times[0].ec-times[0].bc;
    uint64_t first=UINT64_MAX,end=0;for(auto s:times){first=std::min(first,s.bn);end=std::max(end,s.en);}
    return end-first;
  };
  for(int i=0;i<30;++i){warmup.push_back(float(launch()));if(i>=7){
    std::vector<float>tail(warmup.end()-5,warmup.end());
    if(coefficient_of_variation(tail)<=.02){converged=true;break;}}}
  auto elapsed=launch();std::vector<float>values(count);
  CUDA_CHECK(cudaMemcpy(values.data(),accum.pointer,count*4,cudaMemcpyDeviceToHost));
  double error=0;
  for(size_t q=0;q<count;++q){double ref;int t;
    if constexpr(Consumer==2){
      int j=q%(N/2);t=(q/(N/2))%256;int local=t%128;
      int row=(t/128)*64+(local/32)*16+(local%32)/4+((j/2)%2)*8;
      int col=(local%4)*2+j%2+(j/4)*8;ref=0;
      for(int k=0;k<64*Tiles;++k)ref+=double(a_value(row,k))*b_value(col,k);
    }else{t=q%256;ref=Consumer==1?double(a_value(t%128,31*64))+64/1024.:(t+129)/1024.;}
    if(!std::isfinite(values[q]))throw std::runtime_error("nonfinite output");
    error=std::max(error,std::abs(values[q]-ref));
  }
  save13("accum.f32",values);
  std::vector<unsigned>input(last.count);CUDA_CHECK(cudaMemcpy(input.data(),last.pointer,
    input.size()*4,cudaMemcpyDeviceToHost));save13("last-input.u32",input);
  #ifdef R13_TRACE
  std::vector<Event>trace(event.count);CUDA_CHECK(cudaMemcpy(trace.data(),event.pointer,
    trace.size()*sizeof(Event),cudaMemcpyDeviceToHost));save13("trace.u64",trace);
  #endif
  #ifdef R13_TRACE
  constexpr int TracePair=R13_TRACE_PAIR;
  #else
  constexpr int TracePair=-1;
  #endif
  std::cout<<std::setprecision(17)<<"{\"status\":\""<<(error==0?"measured":"numeric_error")
    <<"\",\"n\":"<<N<<",\"stages\":"<<o.stages<<",\"consumer\":"<<Consumer<<",\"trace_tile\":"<<o.trace_tile<<",\"trace_pair\":"<<TracePair<<",\"wgmma_wait\":"<<(Consumer==2?(o.stages==1?0:1):-1)
    <<",\"scope\":\""<<o.scope<<"\",\"b_major\":\"MN\",\"b_global_layout\":\"row_major_K_N\",\"ldb\":"<<N<<",\"tiles\":32,\"reserved_slots\":4,\"threads\":384"
    <<",\"blocks\":"<<blocks<<",\"tile_bytes\":"<<Bytes<<",\"dynamic_smem_bytes\":"<<Smem
    <<",\"registers_per_thread\":"<<attr.numRegs<<",\"local_bytes_per_thread\":"<<attr.localSizeBytes
    <<",\"occupancy_limit\":"<<occupancy<<",\"input_bytes\":"<<uint64_t(blocks)*Tiles*Bytes
    <<",\"elapsed\":"<<elapsed<<",\"unit\":\""<<(o.scope=="one_cta"?"cycle/CTA":"ns/GPU")
    <<"\",\"max_error\":"<<error<<",\"warmup_converged\":"<<(converged?"true":"false")
    <<",\"warmup\":[";
  for(size_t i=0;i<warmup.size();++i){if(i)std::cout<<',';std::cout<<warmup[i];}
  std::cout<<"],\"stamps\":[";
  for(int i=0;i<blocks;++i){auto s=times[i];if(i)std::cout<<',';
    std::cout<<'['<<s.bc<<','<<s.ec<<','<<s.bn<<','<<s.en<<','<<s.first_sm<<','<<s.last_sm<<']';}
  std::cout<<"]}\n";return error==0?0:2;
}
template<int N,int Consumer>int select_stage(const Options13&o){
  if(o.stages==1)return execute<N,Consumer,1>(o);
  if(o.stages==2)return execute<N,Consumer,2>(o);
  return execute<N,Consumer,4>(o);
}
template<int N>int select13(const Options13&o){
  if(o.consumer==0)return select_stage<N,0>(o);
  if(o.consumer==1)return select_stage<N,1>(o);
  return select_stage<N,2>(o);
}
int main(int argc,char**argv){try{Options13 o(argc,argv);return o.n==128?select13<128>(o):select13<256>(o);}
 catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
