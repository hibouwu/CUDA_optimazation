// S14 compile-only PTX 8.8 feasibility forms. No kernel is launched by main.
// These are not a runtime-qualified benchmark or the formal probe CLI.
// Hypothetical launch:128 threads, Q in{1024,4096,8192,16384,32768,65536},
// dynamic shared Q+32 bytes,16-byte-aligned disjoint per-CTA global ranges.
// The64KiB case requires dynamic-shared opt-in in a future controlled runner.
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ unsigned long long timer_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

__device__ __forceinline__ unsigned long long local_cycle() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
  return value;
}

extern "C" __global__ void tma_bulk_g2s_form(
    const uint32_t* input, uint32_t* output, unsigned q) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* tile = reinterpret_cast<uint32_t*>(storage);
  const unsigned destination = shared_address(tile);
  const unsigned barrier = shared_address(storage + q);
  const uint64_t base = uint64_t(blockIdx.x) * uint64_t(q / 4);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                 :: "r"(barrier) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // Split expect/issue/arrive keeps the copy before the release arrival.
    // Q counts transaction bytes; only one thread performs an arrival.
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                 :: "r"(barrier), "r"(q) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];"
                 :: "r"(destination), "l"(input + base), "r"(q), "r"(barrier)
                 : "memory");
    unsigned long long state;
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];"
                 : "=l"(state) : "r"(barrier) : "memory");
    const auto start = timer_ns();
    for (;;) {
      unsigned done;
      asm volatile("{ .reg .pred p; "
                   "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2, 64; "
                   "selp.b32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(barrier), "l"(state) : "memory");
      if (done) break;
      if (timer_ns() - start >= 1000000000ull) {
        // Failure cannot return normally and reclaim pending async storage.
        // A future runtime must preserve failure and confirm process cleanup.
        asm volatile("trap;" ::: "memory");
      }
    }
  }
  __syncthreads();
  // All words become observable; this is not a measured loop.
  for (unsigned word = threadIdx.x; word < q / 4; word += blockDim.x)
    output[base + word] = tile[word];
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                 :: "r"(barrier) : "memory");
}

extern "C" __global__ void tma_bulk_s2g_full_form(
    uint32_t* output, unsigned q, unsigned seed) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* tile = reinterpret_cast<uint32_t*>(storage);
  const uint64_t base = uint64_t(blockIdx.x) * uint64_t(q / 4);
  for (unsigned word = threadIdx.x; word < q / 4; word += blockDim.x)
    tile[word] = uint32_t(29ull * (base + word) + uint64_t(seed));
  // Every generic-proxy producer publishes its own writes before rendezvous.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(output + base), "r"(shared_address(tile)), "r"(q)
                 : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  __syncthreads();
}

extern "C" __global__ void tma_bulk_s2g_read_then_full_form(
    uint32_t* output, unsigned long long* clocks, unsigned q, unsigned seed) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* tile = reinterpret_cast<uint32_t*>(storage);
  const uint64_t base = uint64_t(blockIdx.x) * uint64_t(q / 4);
  for (unsigned word = threadIdx.x; word < q / 4; word += blockDim.x)
    tile[word] = uint32_t(29ull * (base + word) + uint64_t(seed));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  unsigned long long start = 0, released = 0;
  if (threadIdx.x == 0) {
    start = local_cycle();
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(output + base), "r"(shared_address(tile)), "r"(q)
                 : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    released = local_cycle();
  }
  __syncthreads();
  // Legal source reuse after read completion, even if writes remain pending.
  for (unsigned word = threadIdx.x; word < q / 4; word += blockDim.x) {
    const uint32_t original = uint32_t(29ull * (base + word) + uint64_t(seed));
    const unsigned address = shared_address(tile + word);
    // Volatile PTX store preserves this otherwise dead diagnostic overwrite.
    asm volatile("st.shared.u32 [%0], %1;"
                 :: "r"(address), "r"(~original) : "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    const auto completed = local_cycle();
    const uint64_t index = uint64_t(blockIdx.x) * 3;
    clocks[index] = start;
    clocks[index + 1] = released;
    clocks[index + 2] = completed;
  }
  __syncthreads();
  // A future diagnostic runner must validate every output word against the
  // original pattern, not its complement, after CUDA completion.
}

int main() { return 0; }
