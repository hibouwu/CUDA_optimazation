// CUTLASS barrier return paths: already complete versus waiting before arrival.
#include "r00_common.hpp"
#include <cooperative_groups.h>
#include <cutlass/arch/barrier.h>

using Barrier = cutlass::arch::ClusterBarrier;
using Transaction = cutlass::arch::ClusterTransactionBarrier;
constexpr int SharedBytes = 215040;
struct Stamp { uint64_t before, after; };
struct CtaRecord { uint64_t begin_c, end_c, begin_ns, end_ns; unsigned sm; };

__device__ __forceinline__ uint64_t cycles() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
  return value;
}
__device__ __forceinline__ uint64_t nanoseconds() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

__global__ void barrier_return(int rounds, int arrivers, int waiters, int gap,
    int signal_kind, int wait_kind, int acquire_tx, Stamp* records, CtaRecord* ctas) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* barrier = reinterpret_cast<uint64_t*>(storage);
  __shared__ uint64_t epoch;
  if (threadIdx.x == 0) {
    Barrier::init(barrier, signal_kind ? 1 : arrivers);
    if (acquire_tx) Barrier::init(barrier+1, 1);
    cutlass::arch::fence_barrier_init();
    asm volatile("mov.u32 %0, %%smid;" : "=r"(ctas[blockIdx.x].sm));
  }
  __syncthreads();
  cooperative_groups::this_grid().sync();
  uint64_t begin_c = cycles(), begin_ns = nanoseconds();
  for (int round = 0; round < rounds; ++round) {
    if (threadIdx.x == 0) {
      if (signal_kind) Transaction::arrive_and_expect_tx(barrier, 32768);
      epoch = cycles() + 2048;
    }
    __syncthreads();
    Stamp stamp{};
    if (threadIdx.x < waiters) {
      while (cycles() < epoch + 1024 + gap) {}
      stamp.before = cycles();
      if (wait_kind == 0) {
        bool ready = Barrier::try_wait(barrier, round & 1);
        if (!ready) Barrier::wait(barrier, round & 1);
      } else if (wait_kind == 1) {
        while (!Barrier::test_wait(barrier, round & 1, 1)) {}
      } else {
        Barrier::wait(barrier, round & 1);
      }
      if (acquire_tx && threadIdx.x==0) Transaction::arrive_and_expect_tx(barrier+1, 32768);
      stamp.after = cycles();
    } else if (threadIdx.x >= 128 && threadIdx.x < 128 + arrivers) {
      while (cycles() < epoch + 1024) {}
      stamp.before = cycles();
      if (signal_kind) Transaction::complete_transaction(barrier, 0, 32768);
      else Barrier::arrive(barrier, 0, 1);
      stamp.after = cycles();
    }
    __syncthreads();
    records[(size_t(blockIdx.x)*rounds + round)*256 + threadIdx.x] = stamp;
    if (acquire_tx && threadIdx.x==0) {
      Transaction::complete_transaction(barrier+1, 0, 32768);
      Barrier::wait(barrier+1, round & 1);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    ctas[blockIdx.x].begin_c = begin_c;
    ctas[blockIdx.x].begin_ns = begin_ns;
    ctas[blockIdx.x].end_c = cycles();
    ctas[blockIdx.x].end_ns = nanoseconds();
    Barrier::invalidate(barrier);
    if (acquire_tx) Barrier::invalidate(barrier+1);
  }
}

int main(int argc, char** argv) try {
  int blocks=1, rounds=32, arrivers=128, waiters=128, gap=-512, signal=0, wait=0;
  int acquire_tx=0;
  for (int i=1; i<argc; i+=2) {
    if (i+1 == argc) throw std::runtime_error("missing option value");
    std::string name=argv[i]; int value=std::stoi(argv[i+1]);
    if (name=="--ctas") blocks=value;
    else if (name=="--rounds") rounds=value;
    else if (name=="--arrivers") arrivers=value;
    else if (name=="--waiters") waiters=value;
    else if (name=="--gap-cycles") gap=value;
    else if (name=="--signal-kind") signal=value;
    else if (name=="--wait-kind") wait=value;
    else if (name=="--acquire-tx") acquire_tx=value;
    else throw std::runtime_error("unknown option: "+name);
  }
  if (blocks<1 || rounds<1 || (arrivers!=1 && arrivers!=128) ||
      (waiters!=1 && waiters!=128) || gap < -1024 || gap>8192 ||
      (signal!=0 && signal!=1) || wait<0 || wait>2 || (signal && arrivers!=1) ||
      (acquire_tx!=0 && acquire_tx!=1) || (acquire_tx && signal))
    throw std::runtime_error("invalid condition");
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  if (prop.major!=9 || !prop.cooperativeLaunch || blocks>prop.multiProcessorCount)
    throw std::runtime_error("SM90 cooperative allocation required");
  CUDA_CHECK(cudaFuncSetAttribute(barrier_return,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  int occupancy=0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,barrier_return,256,SharedBytes));
  if (occupancy!=1) throw std::runtime_error("expected one CTA per SM");
  size_t count=size_t(blocks)*rounds*256;
  Stamp* device_records; CtaRecord* device_ctas;
  CUDA_CHECK(cudaMalloc(&device_records,count*sizeof(Stamp)));
  CUDA_CHECK(cudaMalloc(&device_ctas,blocks*sizeof(CtaRecord)));
  void* arguments[]={&rounds,&arrivers,&waiters,&gap,&signal,&wait,&acquire_tx,&device_records,&device_ctas};
  for (int launch=0; launch<3; ++launch) {
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(barrier_return),blocks,256,arguments,SharedBytes));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  std::vector<Stamp> records(count); std::vector<CtaRecord> ctas(blocks);
  CUDA_CHECK(cudaMemcpy(records.data(),device_records,count*sizeof(Stamp),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ctas.data(),device_ctas,blocks*sizeof(CtaRecord),cudaMemcpyDeviceToHost));
  uint64_t checked=0;
  for (int cta=0; cta<blocks; ++cta) for (int round=0; round<rounds; ++round) {
    auto* stamps=records.data()+(size_t(cta)*rounds+round)*256;
    uint64_t last_pre=0;
    for (int t=128; t<128+arrivers; ++t) {
      if (!stamps[t].before || stamps[t].after<stamps[t].before) throw std::runtime_error("invalid arrival stamp");
      last_pre=std::max(last_pre,stamps[t].before); ++checked;
    }
    for (int t=0; t<waiters; ++t) {
      if (!stamps[t].before || stamps[t].after<stamps[t].before || stamps[t].after<last_pre)
        throw std::runtime_error("invalid completion order");
      ++checked;
    }
  }
  std::cout<<"{\"schema\":\"barrier-return/1\",\"ctas\":"<<blocks<<",\"rounds\":"<<rounds
    <<",\"arrivers\":"<<arrivers<<",\"waiters\":"<<waiters<<",\"gap_cycles\":"<<gap
    <<",\"signal_kind\":"<<signal<<",\"wait_kind\":"<<wait<<",\"checked\":"<<checked
    <<",\"acquire_tx\":"<<acquire_tx
    <<",\"shared_bytes\":"<<SharedBytes<<",\"occupancy\":"<<occupancy<<",\"cta_records\":[";
  for (int cta=0; cta<blocks; ++cta) {
    auto c=ctas[cta]; if(cta)std::cout<<',';
    std::cout<<'['<<c.sm<<','<<c.begin_c<<','<<c.end_c<<','<<c.begin_ns<<','<<c.end_ns<<']';
  }
  std::cout<<"],\"stamps\":[";
  for(size_t i=0; i<count; ++i) {
    if(i)std::cout<<',';
    std::cout<<'['<<records[i].before<<','<<records[i].after<<']';
  }
  std::cout<<"]}\n";
  CUDA_CHECK(cudaFree(device_records)); CUDA_CHECK(cudaFree(device_ctas));
} catch(std::exception const& error) {
  std::cerr<<error.what()<<'\n'; return 1;
}
