// R06: fixed-work FFMA service under three register-allocation tiers, one CTA or a
// 4 x SM grid. 128 threads, 8 independent chains per lane; each timed loop iteration holds
// kUnroll FFMA per chain (8 x kUnroll FFMA) so loop control is paid once per iteration.
#include "r01_support.hpp"

#ifndef R06_UNROLL
#define R06_UNROLL 32
#endif
constexpr int kUnroll = R06_UNROLL;
constexpr int kChains = 8;
constexpr int kThreads = 128;
// Extra live values per thread for the ~64/96/128-register tiers (actual count from ptxas).
constexpr int kExtra64 = 40, kExtra96 = 72, kExtra128 = 104;

struct R6Options {
  int registers = 64, steps = 2048;  // steps = dependent FFMA per chain
  std::string scope = "one_cta";

  R6Options(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("missing value");
      std::string k = argv[i], v = argv[i + 1];
      if (k == "--registers") registers = std::stoi(v);
      else if (k == "--steps") steps = std::stoi(v);
      else if (k == "--scope") scope = v;
      else throw std::runtime_error("unknown option " + k);
    }
    if (steps < kUnroll || steps % kUnroll || (scope != "one_cta" && scope != "all_gpu"))
      throw std::runtime_error("invalid options (steps must be a multiple of unroll)");
  }
};

// Extra live values are set before the window and consumed after it; they only raise the
// register allocation and are not touched inside the window.
template <int Extra>
__global__ void resource_probe(int iterations, RStamp* stamps, float* output, float* extra_output) {
  float live[Extra], d[kChains];
#pragma unroll
  for (int i = 0; i < Extra; ++i) {
    live[i] = (threadIdx.x + 1 + i) / 1024.f;
    asm volatile("" : "+f"(live[i])::"memory");
  }
#pragma unroll
  for (int i = 0; i < kChains; ++i) {
    d[i] = (threadIdx.x + 1) / 1024.f + i / 16.f;
    asm volatile("" : "+f"(d[i])::"memory");
  }
  __shared__ volatile float drain[kThreads];
  RStamp s{};
  __syncthreads();
  if (threadIdx.x == 0) stamp_begin(s);
  __syncthreads();
#pragma unroll 1
  for (int q = 0; q < iterations; ++q) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
#pragma unroll
      for (int i = 0; i < kChains; ++i)
        asm volatile("fma.rn.f32 %0,%0,0f3f800000,0f3a800000;" : "+f"(d[i]));
    }
  }
  float sum = 0;
#pragma unroll
  for (int i = 0; i < kChains; ++i) {
    asm volatile("" : "+f"(d[i])::"memory");
    sum += d[i];
  }
  drain[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) stamp_end(s, stamps + blockIdx.x);
  float other = 0;
#pragma unroll
  for (int i = 0; i < Extra; ++i) {
    asm volatile("" : "+f"(live[i])::"memory");
    other += live[i];
  }
  extra_output[blockIdx.x * kThreads + threadIdx.x] = other;
#pragma unroll
  for (int i = 0; i < kChains; ++i) output[(blockIdx.x * kThreads + threadIdx.x) * kChains + i] = d[i];
}

template <int Extra>
int run(const R6Options& o) {
  auto kernel = resource_probe<Extra>;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  cudaFuncAttributes a{};
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  int occupancy = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, kThreads, 0));
  const bool full = o.scope == "all_gpu";
  const int blocks = full ? 4 * p.multiProcessorCount : 1;
  DeviceBuffer<RStamp> stamps(blocks);
  DeviceBuffer<float> output(blocks * kThreads * kChains), extra(blocks * kThreads);
  std::vector<RStamp> times(blocks);
  auto launch = [&]() {
    kernel<<<blocks, kThreads>>>(o.steps / kUnroll, stamps.pointer, output.pointer, extra.pointer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(times.data(), stamps.pointer, blocks * sizeof(RStamp),
                          cudaMemcpyDeviceToHost));
    return elapsed_window(times, full);
  };
  Windows w = warm_up(launch);
  launch();

  std::vector<float> values(output.count), others(extra.count);
  CUDA_CHECK(cudaMemcpy(values.data(), output.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(others.data(), extra.pointer, others.size() * 4, cudaMemcpyDeviceToHost));
  bool valid = true;
  for (size_t q = 0; q < values.size(); ++q) {
    const int t = (q / kChains) % kThreads, c = q % kChains;
    valid &= values[q] == (t + 1) / 1024.f + c / 16.f + o.steps / 1024.f;
  }
  for (size_t q = 0; q < others.size(); ++q) {
    const int t = q % kThreads;
    float ref = 0;
    for (int i = 0; i < Extra; ++i) ref += (t + 1 + i) / 1024.f;
    valid &= others[q] == ref;
  }
  save_binary("output.bin", values);
  save_binary("extra.bin", others);

  std::cout << std::setprecision(17) << "{\"status\":\"" << (valid ? "measured" : "numeric_error")
            << "\",\"scope\":\"" << o.scope << "\",\"register_target\":" << o.registers
            << ",\"live_extra\":" << Extra << ",\"steps\":" << o.steps << ",\"unroll\":" << kUnroll
            << ",\"iterations\":" << o.steps / kUnroll << ",\"chains\":" << kChains
            << ",\"ffma_per_lane\":" << o.steps * kChains << ",\"blocks\":" << blocks
            << ",\"threads\":" << kThreads << ",\"sm_count\":" << p.multiProcessorCount
            << ",\"clock_rate_khz\":" << p.clockRate
            << ",\"work_flop\":" << uint64_t(2) * blocks * kThreads * kChains * o.steps
            << ",\"registers_per_thread\":" << a.numRegs
            << ",\"local_bytes_per_thread\":" << a.localSizeBytes
            << ",\"static_smem_bytes\":" << a.sharedSizeBytes
            << ",\"occupancy_limit_ctas_per_sm\":" << occupancy << ",\"unit\":\""
            << (full ? "globaltimer_ns/GPU" : "clock64_cycle/CTA") << '"';
  windows_json(w, times, full);
  std::cout << "}\n";
  return valid ? 0 : 2;
}

int main(int argc, char** argv) {
  try {
    R6Options o(argc, argv);
    if (o.registers == 64) return run<kExtra64>(o);
    if (o.registers == 96) return run<kExtra96>(o);
    if (o.registers == 128) return run<kExtra128>(o);
    throw std::runtime_error("outside matrix");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
