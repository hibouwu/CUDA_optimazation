// R11: identical action counts, state, consumers and two phase boundaries per unit.
// Order is a runtime input: all organizations of one pair use one kernel image.
#include "r01_support.hpp"
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>
#include <cstring>

namespace G = cute::SM90::GMMA;
using Mma = G::MMA_64x128x16_F32F16F16_SS<G::Major::K, G::Major::K>;
constexpr int kWords = 129;
constexpr int kBatch = 8;
constexpr int kBatches = 16;
struct R11Stamp { uint64_t start, end; unsigned sm; };

__device__ __forceinline__ uint64_t dependent_clock(uint64_t value) {
  uint64_t t;
  asm volatile("{ .reg .pred p; setp.ne.u64 p,%1,0; mov.u64 %0,0; "
               "@p mov.u64 %0,%%clock64; }" : "=l"(t) : "l"(value) : "memory");
  return t;
}

template <size_t... I>
__device__ __forceinline__ void mma_step(uint64_t a, uint64_t b, float* d,
                                        std::index_sequence<I...>) {
  Mma::fma(a, b, d[I]..., G::ScaleOut::One);
}

__host__ __device__ float matrix_a(int r, int k) { return (1+(r+2*k)%7)/16.f; }
__host__ __device__ float matrix_b(int c, int k) { return (1+(c+3*k)%11)/32.f; }

struct R11Options {
  int pair=0, order=0, units=256, b_role=0;
  R11Options(int argc,char** argv) {
    for(int i=1;i<argc;i+=2) {
      if(i+1==argc) throw std::runtime_error("missing value");
      std::string key=argv[i];int val=std::stoi(argv[i+1]);
      if(key=="--pair") pair=val;
      else if(key=="--order") order=val;
      else if(key=="--units") units=val;
      else if(key=="--b-role") b_role=val;
      else throw std::runtime_error("unknown option "+key);
    }
    if(pair<0||pair>2||order<0||order>6||units<1||units>256||b_role<0||b_role>1)
      throw std::runtime_error("invalid R11 coordinate");
  }
};

// Pair 0 FFMA+IMAD.WIDE; 1 LDS.128+CVT; 2 WGMMA+IMAD32.
template <int Pair>
__global__ __launch_bounds__(256,1) void r11_probe(const uint64_t* initial,int order,
                          int units,int b_role,R11Stamp* times,uint64_t* output) {
  using namespace cute;
  constexpr int Group=Pair==2?128:32;
  constexpr int Threads=Group*2;
  const int t=threadIdx.x,local=t%Group,role=t/Group;
  const bool active_a=role==0;
  const bool active_b=role==((order==5||b_role)?1:0);
  float d[64],f[8],x[8];uint64_t u[8];unsigned v[8],loaded[32];
  const volatile uint64_t* row=initial+t*kWords;
#pragma unroll
  for(int j=0;j<64;++j) d[j]=Pair==2?__uint_as_float(unsigned(row[j])):0.f;
#pragma unroll
  for(int j=0;j<8;++j) {
    f[j]=Pair==0?__uint_as_float(unsigned(row[64+j])):0.f;
    x[j]=Pair==1?__uint_as_float(unsigned(row[72+j])):0.f;
    u[j]=Pair==0?row[80+j]:0;v[j]=Pair!=0?unsigned(row[88+j]):0;
  }
#pragma unroll
  for(int j=0;j<32;++j) loaded[j]=Pair==1?unsigned(row[96+j]):0;
  extern __shared__ __align__(128) unsigned char storage[];
  uint64_t ad=0,nd=0,bd=0;
  if constexpr(Pair==2) {
    auto la=tile_to_shape(G::Layout_K_SW128_Atom<__half>{},Shape<_64,_64>{});
    auto lb=tile_to_shape(G::Layout_K_SW128_Atom<__half>{},Shape<_128,_64>{});
    auto a=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)),la);
    auto an=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)+4096),la);
    auto b=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)+8192),lb);
    for(int q=t;q<4096;q+=Threads) {
      a(q/64,q%64)=__float2half_rn(matrix_a(q/64,q%64));
      an(q/64,q%64)=__float2half_rn(-matrix_a(q/64,q%64));
    }
    for(int q=t;q<8192;q+=Threads) b(q/64,q%64)=__float2half_rn(matrix_b(q/64,q%64));
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    auto at=local_tile(a,make_shape(_64{},_16{}),make_coord(_0{},_0{}));
    auto nt=local_tile(an,make_shape(_64{},_16{}),make_coord(_0{},_0{}));
    auto bt=local_tile(b,make_shape(_128{},_16{}),make_coord(_0{},_0{}));
    ad=G::make_gmma_desc<G::Major::K>(at);
    nd=G::make_gmma_desc<G::Major::K>(nt);
    bd=G::make_gmma_desc<G::Major::K>(bt);
#pragma unroll
    for(int j=0;j<64;++j) warpgroup_fence_operand(d[j]);
  } else if constexpr(Pair==1) {
    unsigned* s=reinterpret_cast<unsigned*>(storage);
    for(int q=t;q<1024;q+=Threads)
      s[q]=unsigned((q/4)%32+1)*256+(q/128)*4+q%4;
  }
  __shared__ volatile uint64_t gate[Threads];
  const unsigned address=unsigned(__cvta_generic_to_shared(storage))+local*16;
  const unsigned multiplier=local+17;
  auto consume=[&](int epoch) {
    float sf=float(epoch)/1024.f;uint64_t su=(uint64_t(1)<<48)+unsigned(epoch);
#pragma unroll
    for(int j=0;j<8;++j) {
      asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(sf) : "f"(f[j]) : "memory");
      asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(sf) : "f"(x[j]) : "memory");
      asm volatile("{.reg .b64 a,b;shl.b64 a,%0,7;shr.u64 b,%0,5;xor.b64 a,a,b;"
                   "xor.b64 a,a,%1;add.u64 a,a,%2;add.u64 %0,a,0x9e3779b9;}"
                   : "+l"(su) : "l"(u[j]),"l"(uint64_t(v[j])) : "memory");
    }
    if constexpr(Pair==2) {
#pragma unroll
      for(int j=0;j<64;++j) {
        warpgroup_fence_operand(d[j]);
        asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(sf) : "f"(d[j]) : "memory");
      }
    }
    if constexpr(Pair==1) {
#pragma unroll
      for(int j=0;j<32;++j)
        asm volatile("{.reg .b64 a,b;shl.b64 a,%0,7;shr.u64 b,%0,5;xor.b64 a,a,b;"
                   "xor.b64 a,a,%1;add.u64 %0,a,0x9e3779b9;}"
                   : "+l"(su) : "l"(uint64_t(loaded[j])) : "memory");
    }
    return su^uint64_t(__float_as_uint(sf));
  };
  auto phase=[&](int epoch) {
    gate[t]=consume(epoch);
    asm volatile("bar.sync 1,%0;" :: "n"(Threads) : "memory");
    const uint64_t peer=gate[(t+Group)%Threads];
    // The comparison must issue before the next phase. A barrier followed only
    // by independent register instructions is not a sufficient timing boundary.
    if(peer==0) asm volatile("trap;" ::: "memory");
    return peer;
  };
  const uint64_t ready=phase(0);
  R11Stamp stamp{};stamp.sm=r_sm();stamp.start=dependent_clock(ready);
  for(int unit=0;unit<units;++unit) {
    const float sign=(unit&1)?-1.f:1.f;
    const float ffma_source=sign*(local+1)/1024.f;
    uint64_t aa=(unit&1)?nd:ad;
    asm volatile("" : "+l"(aa) :: "memory");
    auto A=[&](int j) {
      if constexpr(Pair==0)
        asm volatile("fma.rn.f32 %0,%1,0f3fa00000,%0;" : "+f"(f[j])
                     : "f"(ffma_source) : "memory");
      else if constexpr(Pair==1)
        asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];"
          : "=r"(loaded[j*4]),"=r"(loaded[j*4+1]),"=r"(loaded[j*4+2]),"=r"(loaded[j*4+3])
          : "r"(address+j*512) : "memory");
      else mma_step(aa,bd,d,std::make_index_sequence<64>{});
    };
    auto B=[&](int j) {
      if constexpr(Pair==0)
        asm volatile("mad.wide.u32 %0,%1,6,%0;" : "+l"(u[j]) : "r"(unsigned(u[j])) : "memory");
      else if constexpr(Pair==1)
        asm volatile("{ .reg .b16 h; .reg .b32 bits; add.rn.f32 %0,%0,%2; "
          "cvt.rn.f16.f32 h,%0; cvt.u32.u16 bits,h; add.u32 %1,%1,bits; }"
          : "+f"(x[j]),"+r"(v[j]) : "f"(sign/256.f) : "memory");
      else
        asm volatile("mad.lo.u32 %0,%0,%1,3;" : "+r"(v[j]) : "r"(multiplier) : "memory");
    };
    auto commit=[&]() {
      if constexpr(Pair==2) {
        warpgroup_commit_batch();warpgroup_wait<1>();
      }
    };
    auto finish=[&]() {
      if constexpr(Pair==2) if(active_a) {
        warpgroup_wait<0>();
#pragma unroll
        for(int j=0;j<64;++j) warpgroup_fence_operand(d[j]);
      }
    };
    auto runA=[&]() {
      if(active_a) {
        for(int batch=0;batch<kBatches;++batch) {
          if constexpr(Pair==2) warpgroup_arrive();
#pragma unroll
          for(int j=0;j<kBatch;++j) A(j);
          commit();
        }
      }
    };
    auto runB=[&]() {
      if(active_b) for(int batch=0;batch<kBatches;++batch) {
#pragma unroll
        for(int j=0;j<kBatch;++j) B(j);
      }
    };
    if(order==0||order==2) runA();
    else if(order==3) runB();
    else if(order==4||order==5) {
      for(int batch=0;batch<kBatches;++batch) {
        if constexpr(Pair==2) if(active_a) warpgroup_arrive();
#pragma unroll
        for(int j=0;j<kBatch;++j) {
          if(active_a) A(j);
          if(active_b) B(j);
        }
        if(active_a) commit();
      }
    }
    finish();phase(unit*2+1);
    if(order==1||order==2) runB();
    else if(order==3) runA();
    finish();phase(unit*2+2);
  }
  stamp.end=dependent_clock(gate[(t+Group)%Threads]);
  if(local%32==0) times[t/32]=stamp;
  uint64_t* out=output+t*kWords;
#pragma unroll
  for(int j=0;j<64;++j) out[j]=__float_as_uint(d[j]);
#pragma unroll
  for(int j=0;j<8;++j) {
    out[64+j]=__float_as_uint(f[j]);out[72+j]=__float_as_uint(x[j]);
    out[80+j]=u[j];out[88+j]=v[j];
  }
#pragma unroll
  for(int j=0;j<32;++j) out[96+j]=loaded[j];
  out[128]=gate[t];
}

// Isolate the warpgroup accumulators from the other role, as in the validated
// R04 cross kernel. Static organization avoids an automatic full wait per MMA.
template<int Order>
__global__ __launch_bounds__(256,1) void r11_tensor(const uint64_t* initial,int ignored,
                         int units,int b_role,R11Stamp* times,uint64_t* output) {
  using namespace cute;
  constexpr int Threads=256;
  int t=threadIdx.x,local=t%128;
  extern __shared__ __align__(128) unsigned char storage[];
  auto la=tile_to_shape(G::Layout_K_SW128_Atom<__half>{},Shape<_64,_64>{});
  auto lb=tile_to_shape(G::Layout_K_SW128_Atom<__half>{},Shape<_128,_64>{});
  auto a=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)),la);
  auto an=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)+4096),la);
  auto b=make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)+8192),lb);
  for(int q=t;q<4096;q+=256) {
    a(q/64,q%64)=__float2half_rn(matrix_a(q/64,q%64));
    an(q/64,q%64)=__float2half_rn(-matrix_a(q/64,q%64));
  }
  for(int q=t;q<8192;q+=256)b(q/64,q%64)=__float2half_rn(matrix_b(q/64,q%64));
  __syncthreads();asm volatile("fence.proxy.async.shared::cta;":::"memory");
  auto at=local_tile(a,make_shape(_64{},_16{}),make_coord(_0{},_0{}));
  auto nt=local_tile(an,make_shape(_64{},_16{}),make_coord(_0{},_0{}));
  auto bt=local_tile(b,make_shape(_128{},_16{}),make_coord(_0{},_0{}));
  uint64_t ad=G::make_gmma_desc<G::Major::K>(at);
  uint64_t nd=G::make_gmma_desc<G::Major::K>(nt);
  uint64_t bd=G::make_gmma_desc<G::Major::K>(bt);
  __shared__ volatile uint64_t gate[256];
  const volatile uint64_t* row=initial+t*kWords;
  unsigned v[8];
#pragma unroll
  for(int j=0;j<8;++j)v[j]=unsigned(row[88+j]);
  const unsigned multiplier=local+17;
  auto B=[&](int j){asm volatile("mad.lo.u32 %0,%0,%1,3;":"+r"(v[j]):"r"(multiplier):"memory");};
  auto allB=[&](){
#pragma unroll 1
    for(int batch=0;batch<16;++batch){
#pragma unroll
      for(int j=0;j<8;++j)B(j);
    }
  };
  uint64_t* out=output+t*kWords;
  if(cutlass::canonical_warp_group_idx()==0) {
    float d[64];
#pragma unroll
    for(int j=0;j<64;++j){d[j]=__uint_as_float(unsigned(row[j]));warpgroup_fence_operand(d[j]);}
    auto consume=[&](int epoch){
      float sf=float(epoch)/1024.f;uint64_t su=(uint64_t(1)<<48)+unsigned(epoch);
#pragma unroll
      for(int j=0;j<64;++j){warpgroup_fence_operand(d[j]);asm volatile("add.rn.f32 %0,%0,%1;":"+f"(sf):"f"(d[j]):"memory");}
#pragma unroll
      for(int j=0;j<8;++j)asm volatile("{.reg .b64 a,b;shl.b64 a,%0,7;shr.u64 b,%0,5;xor.b64 a,a,b;"
          "xor.b64 a,a,%1;add.u64 %0,a,0x9e3779b9;}":"+l"(su):"l"(uint64_t(v[j])):"memory");
      return su^uint64_t(__float_as_uint(sf));
    };
    auto phase=[&](int epoch){gate[t]=consume(epoch);asm volatile("bar.sync 1,256;":::"memory");uint64_t peer=gate[(t+128)%256];if(peer==0)asm volatile("trap;":::"memory");return peer;};
    R11Stamp stamp{};stamp.sm=r_sm();stamp.start=dependent_clock(phase(0));
    for(int unit=0;unit<units;++unit) {
      uint64_t aa=(unit&1)?nd:ad;asm volatile("":"+l"(aa)::"memory");
      auto allA=[&](){
#pragma unroll 1
        for(int batch=0;batch<16;++batch){
          warpgroup_arrive();
#pragma unroll
          for(int j=0;j<8;++j){
            mma_step(aa,bd,d,std::make_index_sequence<64>{});
            if constexpr(Order==4)B(j);
          }
          warpgroup_commit_batch();warpgroup_wait<1>();
        }
        warpgroup_wait<0>();
#pragma unroll
        for(int j=0;j<64;++j)warpgroup_fence_operand(d[j]);
      };
      if constexpr(Order==0||Order==2||Order==4||Order==5)allA();
      if constexpr(Order==3)allB();
      phase(unit*2+1);
      if constexpr(Order==1){if(!b_role)allB();}
      if constexpr(Order==2)allB();
      if constexpr(Order==3)allA();
      phase(unit*2+2);
    }
    stamp.end=dependent_clock(gate[t+128]);if(local%32==0)times[t/32]=stamp;
#pragma unroll
    for(int j=0;j<64;++j)out[j]=__float_as_uint(d[j]);
  } else {
    auto consume=[&](int epoch){uint64_t sum=(uint64_t(1)<<48)+unsigned(epoch);
#pragma unroll
      for(int j=0;j<8;++j)asm volatile("{.reg .b64 a,b;shl.b64 a,%0,7;shr.u64 b,%0,5;xor.b64 a,a,b;"
        "xor.b64 a,a,%1;add.u64 %0,a,0x9e3779b9;}":"+l"(sum):"l"(uint64_t(v[j])):"memory");
      return sum;
    };
    auto phase=[&](int epoch){gate[t]=consume(epoch);asm volatile("bar.sync 1,256;":::"memory");uint64_t peer=gate[t-128];if(peer==0)asm volatile("trap;":::"memory");return peer;};
    R11Stamp stamp{};stamp.sm=r_sm();stamp.start=dependent_clock(phase(0));
    for(int unit=0;unit<units;++unit){
      if constexpr(Order==5)allB();
      phase(unit*2+1);
      if constexpr(Order==1){if(b_role)allB();}
      phase(unit*2+2);
    }
    stamp.end=dependent_clock(gate[t-128]);if(local%32==0)times[t/32]=stamp;
#pragma unroll
    for(int j=0;j<64;++j)out[j]=0;
  }
#pragma unroll
  for(int j=64;j<128;++j)out[j]=(j>=88&&j<96)?v[j-88]:0;
  out[128]=gate[t];
}

static uint32_t bits(float f) { uint32_t x;std::memcpy(&x,&f,4);return x; }

// The host writes a nonuniform opaque initial state; kernels of one pair keep the
// same state/resources even when one producer is disabled.
template <int Pair,int Order=-1>
int run(const R11Options& o) {
  constexpr int Group=Pair==2?128:32,Threads=Group*2;
  std::vector<uint64_t> initial(Threads*kWords,0);
  for(int t=0;t<Threads;++t) {
    int local=t%Group;auto row=initial.data()+t*kWords;
    if constexpr(Pair==2) if(t<128) for(int j=0;j<64;++j) row[j]=bits((local+1)/1024.f+j/32.f);
    for(int j=0;j<8;++j) {
      row[64+j]=Pair==0?bits((local+1)/1024.f+j/16.f):0;
      row[72+j]=Pair==1?bits((local+1)/32.f+j/16.f):0;
      row[80+j]=Pair==0?(uint64_t(local+1)<<32)+j+1:0;
      row[88+j]=Pair!=0?(local+1)*257+j+1:0;
    }
  }
  DeviceBuffer<uint64_t> init(initial.size()),output(initial.size());
  DeviceBuffer<R11Stamp> device_times(Threads/32);
  CUDA_CHECK(cudaMemcpy(init.pointer,initial.data(),initial.size()*8,cudaMemcpyHostToDevice));
  auto kernel=[](){
    if constexpr(Pair==2)return r11_tensor<Order>;
    else return r11_probe<Pair>;
  }();
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
  std::vector<R11Stamp> times(Threads/32);
  constexpr int Smem=Pair==2?32768:Pair==1?4096:128;
  auto launch=[&]() {
    kernel<<<1,Threads,Smem>>>(init.pointer,o.order,o.units,o.b_role,device_times.pointer,output.pointer);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(times.data(),device_times.pointer,times.size()*sizeof(R11Stamp),cudaMemcpyDeviceToHost));
    uint64_t start=UINT64_MAX,end=0;
    for(auto s:times) start=std::min(start,s.start),end=std::max(end,s.end);
    return end-start;
  };
  Windows warm=warm_up(launch);uint64_t elapsed=launch();
  std::vector<uint64_t> values(output.count);
  CUDA_CHECK(cudaMemcpy(values.data(),output.pointer,values.size()*8,cudaMemcpyDeviceToHost));
  save_binary("initial.u64",initial);save_binary("output.u64",values);
  size_t errors=0;
  for(auto s:times) errors+=s.start==0||s.end<s.start||s.sm!=times[0].sm;
  std::cout<<std::setprecision(17)<<"{\"pair\":"<<Pair<<",\"order\":"<<o.order
    <<",\"units\":"<<o.units<<",\"b_role\":"<<o.b_role<<",\"threads\":"<<Threads
    <<",\"actions_per_unit\":128,\"batch_actions\":8,\"registers\":"<<attr.numRegs
    <<",\"local_bytes\":"<<attr.localSizeBytes<<",\"static_shared_bytes\":"<<attr.sharedSizeBytes
    <<",\"dynamic_shared_bytes\":"<<Smem<<",\"elapsed_cycles\":"<<elapsed
    <<",\"event_errors\":"<<errors<<",\"warmup_converged\":"<<(warm.converged?"true":"false")
    <<",\"warmup\":";vector_json(warm.warmup);
  std::cout<<",\"stamps\":[";
  for(size_t i=0;i<times.size();++i) {
    auto s=times[i];std::cout<<(i?",":"")<<'['<<s.start<<','<<s.end<<','<<s.sm<<']';
  }
  std::cout<<"]}\n";
  return errors?2:warm.converged?0:3;
}

int main(int argc,char** argv) {
  try {
    R11Options o(argc,argv);
    if(o.pair==0)return run<0>(o);
    if(o.pair==1)return run<1>(o);
    switch(o.order){
      case 0:return run<2,0>(o);case 1:return run<2,1>(o);case 2:return run<2,2>(o);
      case 3:return run<2,3>(o);case 4:return run<2,4>(o);case 5:return run<2,5>(o);
      case 6:return run<2,6>(o);
    }
    return 1;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
