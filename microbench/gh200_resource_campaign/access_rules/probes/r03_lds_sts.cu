#include "r00_common.hpp"
#include <fstream>
#include <map>
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/gemm/gemm.h"

struct SmemStamp {
  uint64_t begin, end, begin_ns, end_ns;
  unsigned smid, shared_base;
};
__device__ __forceinline__ unsigned smem_id() {
  unsigned value; asm volatile("mov.u32 %0, %%smid;" : "=r"(value)); return value;
}
__device__ __forceinline__ uint64_t smem_cycles() {
  uint64_t value; asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory"); return value;
}
__device__ __forceinline__ uint64_t smem_ns() {
  uint64_t value; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory"); return value;
}
template<int Width> __device__ __forceinline__ uint4 smem_read(unsigned address) {
  uint4 v = make_uint4(0, 0, 0, 0);
  if constexpr (Width == 4) asm volatile("ld.volatile.shared.u32 %0,[%1];" : "=r"(v.x) : "r"(address) : "memory");
  if constexpr (Width == 8) asm volatile("ld.volatile.shared.v2.u32 {%0,%1},[%2];" : "=r"(v.x), "=r"(v.y) : "r"(address) : "memory");
  if constexpr (Width == 16) asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(address) : "memory");
  return v;
}
template<int Width> __device__ __forceinline__ void smem_write(unsigned address, uint4 v) {
  if constexpr (Width == 4) asm volatile("st.volatile.shared.u32 [%0],%1;" :: "r"(address), "r"(v.x) : "memory");
  if constexpr (Width == 8) asm volatile("st.volatile.shared.v2.u32 [%0],{%1,%2};" :: "r"(address), "r"(v.x), "r"(v.y) : "memory");
  if constexpr (Width == 16) asm volatile("st.volatile.shared.v4.u32 [%0],{%1,%2,%3,%4};" :: "r"(address), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
using EpilogueIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
    cutlass::gemm::GemmShape<32,64,16>, cutlass::gemm::GemmShape<16,8,8>,
    unsigned, cutlass::layout::RowMajor>;
static_assert(EpilogueIterator::Fragment::kElements == 16, "eight 64-bit epilogue accesses");

// Op: 0 read, 1 store, 2 loop control. Native uses the actual CUTLASS iterator.
// A dependent read feeds the next address with returned.x - current_address;
// initialization makes this zero, but the compiler must preserve the RAW chain.
template<int Op, int Width, bool Dependent, bool Native = false>
__global__ void smem_probe(int iterations, unsigned seed, unsigned const* maps,
                          unsigned* output, SmemStamp* stamp) {
  extern __shared__ __align__(128) unsigned data[];
  __shared__ volatile unsigned sink[256];
  const unsigned tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const unsigned words = blockDim.x / 32 * 4096;
  unsigned addresses[8], indices[8], sums[8] = {};
#pragma unroll
  for (int q=0; q<8; ++q) {
    indices[q] = maps[q * blockDim.x + tid];
    addresses[q] = unsigned(__cvta_generic_to_shared(data + indices[q]));
  }
  for (unsigned word=tid; word<words; word+=blockDim.x)
    data[word] = Op == 0 ? unsigned(__cvta_generic_to_shared(data + word)) : 0xdeadbeefu;
  EpilogueIterator iterator(typename EpilogueIterator::TensorRef(
      data + warp * 4096, cutlass::layout::RowMajor(72)), lane);
  typename EpilogueIterator::Fragment fragment;
  __syncthreads();
  if (tid == 0) {
    stamp->smid = smem_id();
    stamp->shared_base = unsigned(__cvta_generic_to_shared(data));
    stamp->begin_ns = smem_ns(); stamp->begin = smem_cycles();
  }
  __syncthreads();
  unsigned dependency = 0, serial_sum = 0;
#pragma unroll 1
  for (int i=0; i<iterations; ++i) {
    // This is part of the declared protocol for every organization. It gives
    // the actual C++ CUTLASS shared-memory sequence an iteration boundary.
    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
    if constexpr (Native) {
      asm volatile("" ::: "memory");
      if constexpr (Op == 0) {
        iterator.load(fragment);
#pragma unroll
        for (int q=0; q<8; ++q) sums[q] += fragment[q*2] + fragment[q*2+1];
      } else {
#pragma unroll
        for (int q=0; q<8; ++q) {
          fragment[q*2] = 17u*indices[q] + 31u*(i+1) + seed + q;
          fragment[q*2+1] = fragment[q*2] + 17;
        }
        iterator.store(fragment);
      }
    } else {
#pragma unroll
      for (int q=0; q<8; ++q) {
        if constexpr (Op == 0) {
          unsigned address = addresses[q] + (Dependent ? dependency : 0);
          uint4 v = smem_read<Width>(address);
          if constexpr (Dependent) {
            dependency = v.x - addresses[q];
            serial_sum += v.x + v.y + v.z + v.w;
          } else sums[q] += v.x + v.y + v.z + v.w;
        } else if constexpr (Op == 1) {
          unsigned a = 17u*indices[q] + 31u*(i+1) + seed + q;
          smem_write<Width>(addresses[q], make_uint4(a, a+17, a+34, a+51));
        } else {
          asm volatile("add.u32 %0,%0,1;" : "+r"(serial_sum));
        }
      }
    }
  }
  unsigned sum = serial_sum;
#pragma unroll
  for (int q=0; q<8; ++q) sum += sums[q];
  sink[tid] = sum;
  __syncthreads();
  if (tid == 0) { stamp->end = smem_cycles(); stamp->end_ns = smem_ns(); }
  if constexpr (Op == 1) {
    for (unsigned word=tid; word<words; word+=blockDim.x) output[word] = data[word];
  } else output[tid] = sum;
}
__global__ void smem_timer_grain(uint64_t* cycles, uint64_t* ns, unsigned* smid) {
  if (threadIdx.x != 0) return;
  uint64_t c[65], t[65];
#pragma unroll
  for(int i=0;i<65;++i) c[i]=smem_cycles();
#pragma unroll
  for(int i=0;i<65;++i) t[i]=smem_ns();
  for(int i=0;i<64;++i) { cycles[i]=c[i+1]-c[i]; ns[i]=t[i+1]-t[i]; }
  *smid=smem_id();
}

template<int Op,int Width,bool Dependent,bool Native=false>
void measure(std::vector<unsigned> const& maps, int warps, int iterations, unsigned seed) {
  const unsigned threads=warps*32, words=warps*4096;
  const unsigned count=Op==1 ? words : threads;
  auto kernel=smem_probe<Op,Width,Dependent,Native>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,words*4));
  cudaFuncAttributes attr{}; CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
  DeviceBuffer<unsigned> dm(maps.size()), output(count); DeviceBuffer<SmemStamp> ds(1);
  CUDA_CHECK(cudaMemcpy(dm.pointer,maps.data(),maps.size()*4,cudaMemcpyHostToDevice));
  std::vector<unsigned> got(count); SmemStamp stamp{};
  cudaEvent_t start_event,end_event; float event_ms=0;
  CUDA_CHECK(cudaEventCreate(&start_event)); CUDA_CHECK(cudaEventCreate(&end_event));
  auto launch=[&](int n) {
    CUDA_CHECK(cudaEventRecord(start_event));
    kernel<<<1,threads,words*4>>>(n,seed,dm.pointer,output.pointer,ds.pointer);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(end_event));
    CUDA_CHECK(cudaEventSynchronize(end_event));
    CUDA_CHECK(cudaEventElapsedTime(&event_ms,start_event,end_event));
    CUDA_CHECK(cudaMemcpy(&stamp,ds.pointer,sizeof(stamp),cudaMemcpyDeviceToHost));
    if(stamp.end<=stamp.begin || stamp.end_ns<stamp.begin_ns ||
        double(stamp.end_ns-stamp.begin_ns)>event_ms*1e6*1.05)
      throw std::runtime_error("invalid cycle/ns/event timer cross-check");
  };
  auto check=[&](int n) {
    CUDA_CHECK(cudaMemcpy(got.data(),output.pointer,count*4,cudaMemcpyDeviceToHost));
    std::vector<unsigned> expected(count,Op==1 ? 0xdeadbeefu : 0);
    for (unsigned tid=0;tid<threads;++tid) for(int q=0;q<8;++q) {
      unsigned word=maps[q*threads+tid];
      if constexpr (Op==1) for(int r=0;r<Width/4;++r)
        expected[word+r]=17u*(word+r)+31u*n+seed+q;
      if constexpr (Op==0) for(int r=0;r<Width/4;++r)
        expected[tid]+=unsigned(n)*(stamp.shared_base+4*(word+r));
    }
    if constexpr (Op==2) for(auto& v:expected) v=unsigned(n)*8;
    if(got!=expected) throw std::runtime_error("exact full-output correctness failed");
  };
  for(int n:{1,2,9}) { launch(n); check(n); }
  std::vector<uint64_t> warm;
  bool converged=false;
  for(int i=0;i<30;++i) {
    launch(iterations); warm.push_back(stamp.end-stamp.begin);
    if(i>=7) {
      std::vector<float> tail(warm.end()-5,warm.end());
      if(coefficient_of_variation(tail)<=.02) { converged=true; break; }
    }
  }
  launch(iterations); check(iterations);
  std::ofstream file("output.u32",std::ios::binary);
  file.write(reinterpret_cast<char const*>(got.data()),got.size()*4);
  std::cout<<"{\"status\":\"measured\",\"elapsed_cycles\":"<<stamp.end-stamp.begin
    <<",\"elapsed_ns\":"<<stamp.end_ns-stamp.begin_ns<<",\"event_ms\":"<<event_ms<<",\"smid\":"<<stamp.smid
    <<",\"shared_base\":"<<stamp.shared_base<<",\"iterations\":"<<iterations
    <<",\"target_warp_instructions\":"<<(Op==2?0:uint64_t(iterations)*8*warps)
    <<",\"width\":"<<Width<<",\"warps\":"<<warps<<",\"dependent\":"<<(Dependent?"true":"false")
    <<",\"native_cutlass\":"<<(Native?"true":"false")<<",\"errors\":0,\"checked_elements\":"<<count
    <<",\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes
    <<",\"static_smem_bytes\":"<<attr.sharedSizeBytes<<",\"dynamic_smem_bytes\":"<<words*4
    <<",\"warmup_converged\":"<<(converged?"true":"false")<<",\"warmup_cycles\":[";
  for(size_t i=0;i<warm.size();++i) std::cout<<(i?",":"")<<warm[i];
  std::cout<<"]}\n";
  CUDA_CHECK(cudaEventDestroy(start_event)); CUDA_CHECK(cudaEventDestroy(end_event));
}
template<int Width> void dispatch(int op,int dependent,bool native,std::vector<unsigned> const& maps,int warps,int n,unsigned seed) {
  if(native) {
    if constexpr(Width==8) {
      if(op==0) measure<0,8,false,true>(maps,warps,n,seed);
      else if(op==1) measure<1,8,false,true>(maps,warps,n,seed);
      else throw std::runtime_error("native op unsupported");
    } else throw std::runtime_error("native CUTLASS iterator width is 8B");
  } else if(op==0 && dependent) measure<0,Width,true>(maps,warps,n,seed);
  else if(op==0) measure<0,Width,false>(maps,warps,n,seed);
  else if(op==1) measure<1,Width,false>(maps,warps,n,seed);
  else measure<2,Width,false>(maps,warps,n,seed);
}
int main(int argc,char** argv) try {
  int device_count; CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if(device_count!=1) throw std::runtime_error("exactly one allocated visible GPU required");
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=9 || prop.minor!=0 || std::string(prop.name).find("GH200")==std::string::npos)
    throw std::runtime_error("GH200 required");
  if(argc==2 && std::string(argv[1])=="device") {
    std::ostringstream uuid; uuid<<"GPU-";
    for(int i=0;i<16;++i) {
      if(i==4||i==6||i==8||i==10) uuid<<'-';
      uuid<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));
    }
    int driver,runtime;CUDA_CHECK(cudaDriverGetVersion(&driver));CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::cout<<"{\"gpu_uuid\":\""<<uuid.str()<<"\",\"name\":\""<<prop.name
      <<"\",\"driver\":"<<driver<<",\"runtime\":"<<runtime<<",\"sms\":"<<prop.multiProcessorCount<<"}\n";
    return 0;
  }
  if(argc==2 && std::string(argv[1])=="grain") {
    DeviceBuffer<uint64_t> dc(64),dt(64); DeviceBuffer<unsigned> sm(1);
    smem_timer_grain<<<1,32>>>(dc.pointer,dt.pointer,sm.pointer); CUDA_CHECK(cudaDeviceSynchronize());
    uint64_t c[64],t[64]; unsigned id;
    CUDA_CHECK(cudaMemcpy(c,dc.pointer,sizeof(c),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(t,dt.pointer,sizeof(t),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&id,sm.pointer,4,cudaMemcpyDeviceToHost));
    std::cout<<"{\"smid\":"<<id<<",\"cycle_deltas\":[";
    for(int i=0;i<64;++i) std::cout<<(i?",":"")<<c[i];
    std::cout<<"],\"ns_deltas\":["; for(int i=0;i<64;++i) std::cout<<(i?",":"")<<t[i];
    std::cout<<"]}\n"; return 0;
  }
  if(argc!=9) throw std::runtime_error("OP WIDTH WARPS DEPENDENT NATIVE ITERATIONS SEED MAPFILE");
  int op=std::stoi(argv[1]),width=std::stoi(argv[2]),warps=std::stoi(argv[3]);
  int dependent=std::stoi(argv[4]),native=std::stoi(argv[5]),n=std::stoi(argv[6]);
  unsigned seed=std::stoul(argv[7]);
  if(op<0 || op>2 || (width!=4 && width!=8 && width!=16) || (warps!=1 && warps!=4 && warps!=8) ||
      n<0 || n>65536 || (dependent && op!=0)) throw std::runtime_error("unsupported configuration");
  std::ifstream file(argv[8]); std::vector<unsigned> maps; unsigned word;
  while(file>>word) maps.push_back(word);
  if(maps.size()!=unsigned(warps*32*8)) throw std::runtime_error("map size mismatch");
  std::map<unsigned,unsigned> owners;
  for(unsigned i=0;i<maps.size();++i) {
    unsigned tid=i%(warps*32),warp=tid/32;
    if(maps[i]%(width/4) || maps[i]<warp*4096 || maps[i]+width/4>(warp+1)*4096)
      throw std::runtime_error("unaligned or out-of-region lane address");
    if(op==1) for(int r=0;r<width/4;++r) {
      auto at=owners.emplace(maps[i]+r,tid);
      if(!at.second && at.first->second!=tid) throw std::runtime_error("racing store ownership");
    }
    if(native && maps[i]!=warp*4096+(tid%32/4)*72+(tid%4)*2+(i/(warps*32))*8)
      throw std::runtime_error("map differs from bound CUTLASS iterator");
  }
  if(width==4) dispatch<4>(op,dependent,native,maps,warps,n,seed);
  if(width==8) dispatch<8>(op,dependent,native,maps,warps,n,seed);
  if(width==16) dispatch<16>(op,dependent,native,maps,warps,n,seed);
  return 0;
} catch(std::exception const& error) { std::cerr<<error.what()<<"\n"; return 2; }
