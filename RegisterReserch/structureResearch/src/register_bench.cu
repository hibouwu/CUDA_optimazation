#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error_ = (call);                                           \
    if (error_ != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(error_));                      \
    }                                                                           \
  } while (false)

namespace {

constexpr int kWarpSize = 32;
constexpr int kSourceCount = 16;

enum class ProbeKind {
  kImadChain,
  kImadIndependent4,
  kReuseHot4,
  kBankDense4,
  kBankSparse4,
};

struct CaseSpec {
  const char* name;
  const char* category;
  ProbeKind kind;
  int operations_per_iteration;
  const char* purpose;
};

constexpr CaseSpec kCases[] = {
    {"R0_imad_chain", "latency", ProbeKind::kImadChain, 32,
     "dependent integer multiply-add latency"},
    {"R1_imad_independent_x4", "throughput", ProbeKind::kImadIndependent4, 128,
     "four independent integer multiply-adds"},
    {"R2_reuse_hot_x4", "reuse", ProbeKind::kReuseHot4, 128,
     "four IMADs sharing the same two source operands"},
    {"R3_bank_dense_x4", "bank_candidate", ProbeKind::kBankDense4, 128,
     "four IMADs using densely selected virtual sources"},
    {"R4_bank_sparse_x4", "bank_candidate", ProbeKind::kBankSparse4, 128,
     "four IMADs using sparsely selected virtual sources"},
};

struct Options {
  std::string case_selector = "all";
  int iterations = 100000;
  int warmups = 5;
  int repeats = 20;
  bool list_cases = false;
  bool quiet = false;
};

struct Measurement {
  const CaseSpec* spec = nullptr;
  double average_cycles = 0.0;
  double median_cycles = 0.0;
  std::uint64_t minimum_cycles = 0;
  int registers_per_thread = 0;
  std::size_t local_bytes = 0;
  bool passed = false;
};

__device__ __forceinline__ std::uint64_t read_clock64() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) : : "memory");
  return value;
}

template <ProbeKind Kind>
__global__ __launch_bounds__(kWarpSize, 1) void register_probe(
    const std::uint32_t* sources, std::uint64_t* elapsed_cycles,
    std::uint32_t* sinks, int iterations) {
  const int lane = threadIdx.x;
  const std::uint32_t s0 = sources[0 * kWarpSize + lane];
  const std::uint32_t s1 = sources[1 * kWarpSize + lane];
  const std::uint32_t s2 = sources[2 * kWarpSize + lane];
  const std::uint32_t s3 = sources[3 * kWarpSize + lane];
  const std::uint32_t s4 = sources[4 * kWarpSize + lane];
  const std::uint32_t s5 = sources[5 * kWarpSize + lane];
  const std::uint32_t s6 = sources[6 * kWarpSize + lane];
  const std::uint32_t s7 = sources[7 * kWarpSize + lane];
  const std::uint32_t s8 = sources[8 * kWarpSize + lane];
  const std::uint32_t s9 = sources[9 * kWarpSize + lane];
  const std::uint32_t s10 = sources[10 * kWarpSize + lane];
  const std::uint32_t s11 = sources[11 * kWarpSize + lane];
  const std::uint32_t s12 = sources[12 * kWarpSize + lane];
  const std::uint32_t s13 = sources[13 * kWarpSize + lane];
  const std::uint32_t s14 = sources[14 * kWarpSize + lane];
  const std::uint32_t s15 = sources[15 * kWarpSize + lane];

  std::uint32_t acc0 = 0x1234567u + static_cast<std::uint32_t>(lane);
  std::uint32_t acc1 = 0x2345678u + static_cast<std::uint32_t>(lane);
  std::uint32_t acc2 = 0x3456789u + static_cast<std::uint32_t>(lane);
  std::uint32_t acc3 = 0x456789au + static_cast<std::uint32_t>(lane);

  __syncwarp();
  const std::uint64_t start = read_clock64();

#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int unroll = 0; unroll < 32; ++unroll) {
      if constexpr (Kind == ProbeKind::kImadChain) {
        asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                     : "+r"(acc0)
                     : "r"(s0), "r"(s1));
      } else if constexpr (Kind == ProbeKind::kImadIndependent4) {
        asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                     : "+r"(acc0)
                     : "r"(s0), "r"(s4));
        asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                     : "+r"(acc1)
                     : "r"(s1), "r"(s5));
        asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                     : "+r"(acc2)
                     : "r"(s2), "r"(s6));
        asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                     : "+r"(acc3)
                     : "r"(s3), "r"(s7));
      } else if constexpr (Kind == ProbeKind::kReuseHot4) {
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc0)
                     : "r"(s0), "r"(s1));
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc1)
                     : "r"(s0), "r"(s1));
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc2)
                     : "r"(s0), "r"(s1));
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc3)
                     : "r"(s0), "r"(s1));
      } else if constexpr (Kind == ProbeKind::kBankDense4) {
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc0)
                     : "r"(s0), "r"(s1));
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc1)
                     : "r"(s2), "r"(s3));
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc2)
                     : "r"(s4), "r"(s5));
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc3)
                     : "r"(s6), "r"(s7));
      } else if constexpr (Kind == ProbeKind::kBankSparse4) {
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc0)
                     : "r"(s0), "r"(s4));
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc1)
                     : "r"(s8), "r"(s12));
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc2)
                     : "r"(s1), "r"(s5));
        asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                     : "+r"(acc3)
                     : "r"(s9), "r"(s13));
      }
    }
  }

  const std::uint64_t stop = read_clock64();
  __syncwarp();

  elapsed_cycles[lane] = stop - start;
  sinks[lane] = acc0 ^ acc1 ^ acc2 ^ acc3 ^ s10 ^ s11 ^ s14 ^ s15;
}

template <ProbeKind Kind>
void launch_typed(const std::uint32_t* sources, std::uint64_t* elapsed_cycles,
                  std::uint32_t* sinks, int iterations) {
  register_probe<Kind><<<1, kWarpSize>>>(sources, elapsed_cycles, sinks,
                                        iterations);
}

void launch_case(ProbeKind kind, const std::uint32_t* sources,
                 std::uint64_t* elapsed_cycles, std::uint32_t* sinks,
                 int iterations) {
  switch (kind) {
    case ProbeKind::kImadChain:
      launch_typed<ProbeKind::kImadChain>(sources, elapsed_cycles, sinks,
                                         iterations);
      break;
    case ProbeKind::kImadIndependent4:
      launch_typed<ProbeKind::kImadIndependent4>(
          sources, elapsed_cycles, sinks, iterations);
      break;
    case ProbeKind::kReuseHot4:
      launch_typed<ProbeKind::kReuseHot4>(sources, elapsed_cycles, sinks,
                                         iterations);
      break;
    case ProbeKind::kBankDense4:
      launch_typed<ProbeKind::kBankDense4>(sources, elapsed_cycles, sinks,
                                          iterations);
      break;
    case ProbeKind::kBankSparse4:
      launch_typed<ProbeKind::kBankSparse4>(sources, elapsed_cycles, sinks,
                                           iterations);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

template <ProbeKind Kind>
cudaFuncAttributes typed_attributes() {
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, register_probe<Kind>));
  return attributes;
}

cudaFuncAttributes case_attributes(ProbeKind kind) {
  switch (kind) {
    case ProbeKind::kImadChain:
      return typed_attributes<ProbeKind::kImadChain>();
    case ProbeKind::kImadIndependent4:
      return typed_attributes<ProbeKind::kImadIndependent4>();
    case ProbeKind::kReuseHot4:
      return typed_attributes<ProbeKind::kReuseHot4>();
    case ProbeKind::kBankDense4:
      return typed_attributes<ProbeKind::kBankDense4>();
    case ProbeKind::kBankSparse4:
      return typed_attributes<ProbeKind::kBankSparse4>();
  }
  throw std::runtime_error("unknown probe kind");
}

std::vector<const CaseSpec*> select_cases(const std::string& selector) {
  std::vector<const CaseSpec*> selected;
  for (const auto& spec : kCases) {
    if (selector == "all" || selector == spec.name ||
        selector == spec.category) {
      selected.push_back(&spec);
    }
  }
  return selected;
}

int parse_integer(const char* text, const char* option, bool allow_zero) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  const long minimum = allow_zero ? 0 : 1;
  if (end == text || *end != '\0' || value < minimum ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid value for ") + option +
                             ": " + text);
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&](const char* option) {
      if (++index >= argc) {
        throw std::runtime_error(std::string("missing value for ") + option);
      }
      return argv[index];
    };

    if (argument == "--case") {
      options.case_selector = require_value("--case");
    } else if (argument == "--iters") {
      options.iterations =
          parse_integer(require_value("--iters"), "--iters", false);
    } else if (argument == "--warmups") {
      options.warmups =
          parse_integer(require_value("--warmups"), "--warmups", true);
    } else if (argument == "--repeats") {
      options.repeats =
          parse_integer(require_value("--repeats"), "--repeats", false);
    } else if (argument == "--list-cases") {
      options.list_cases = true;
    } else if (argument == "--quiet") {
      options.quiet = true;
    } else if (argument == "--help") {
      std::cout
          << "Usage: register_bench [--case NAME|CATEGORY|all] [--iters N] "
             "[--warmups N] [--repeats N] [--list-cases] [--quiet]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  return options;
}

std::vector<Measurement> measure_cases(
    const std::vector<const CaseSpec*>& specs, const Options& options,
    const std::uint32_t* sources, std::uint64_t* elapsed_cycles,
    std::uint32_t* sinks) {
  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    for (const auto* spec : specs) {
      launch_case(spec->kind, sources, elapsed_cycles, sinks,
                  options.iterations);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::vector<std::uint64_t>> samples(specs.size());
  std::vector<bool> sinks_valid(specs.size(), true);
  for (auto& values : samples) {
    values.reserve(options.repeats * kWarpSize);
  }
  std::vector<std::uint64_t> host_cycles(kWarpSize);
  std::vector<std::uint32_t> host_sinks(kWarpSize);

  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    for (std::size_t slot = 0; slot < specs.size(); ++slot) {
      const std::size_t index = (slot + repeat) % specs.size();
      launch_case(specs[index]->kind, sources, elapsed_cycles, sinks,
                  options.iterations);
      CUDA_CHECK(cudaMemcpy(host_cycles.data(), elapsed_cycles,
                            host_cycles.size() * sizeof(std::uint64_t),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_sinks.data(), sinks,
                            host_sinks.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost));
      samples[index].insert(samples[index].end(), host_cycles.begin(),
                            host_cycles.end());
      sinks_valid[index] =
          sinks_valid[index] &&
          std::any_of(host_sinks.begin(), host_sinks.end(),
                      [](std::uint32_t value) { return value != 0; });
    }
  }

  std::vector<Measurement> measurements;
  measurements.reserve(specs.size());
  for (std::size_t index = 0; index < specs.size(); ++index) {
    auto sorted = samples[index];
    std::sort(sorted.begin(), sorted.end());
    const double average =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) /
        static_cast<double>(sorted.size());
    const std::size_t middle = sorted.size() / 2;
    const double median =
        sorted.size() % 2 == 0
            ? (static_cast<double>(sorted[middle - 1]) + sorted[middle]) / 2.0
            : static_cast<double>(sorted[middle]);
    const bool passed =
        !sorted.empty() && sorted.front() > 0 && sinks_valid[index];
    const cudaFuncAttributes attributes = case_attributes(specs[index]->kind);
    measurements.push_back({specs[index],
                            average,
                            median,
                            sorted.front(),
                            attributes.numRegs,
                            attributes.localSizeBytes,
                            passed});
  }
  return measurements;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto selected = select_cases(options.case_selector);
    if (selected.empty()) {
      throw std::runtime_error("unknown case selector: " +
                               options.case_selector);
    }
    if (options.list_cases) {
      for (const auto* spec : selected) std::cout << spec->name << '\n';
      return 0;
    }

    std::vector<std::uint32_t> host_sources(kSourceCount * kWarpSize);
    for (int source = 0; source < kSourceCount; ++source) {
      for (int lane = 0; lane < kWarpSize; ++lane) {
        host_sources[source * kWarpSize + lane] =
            static_cast<std::uint32_t>(0x9e3779b9u * (source + 1) +
                                       0x45d9f3bu * (lane + 1));
      }
    }

    std::uint32_t* sources = nullptr;
    std::uint64_t* elapsed_cycles = nullptr;
    std::uint32_t* sinks = nullptr;
    CUDA_CHECK(cudaMalloc(&sources, host_sources.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&elapsed_cycles, kWarpSize * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&sinks, kWarpSize * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemcpy(sources, host_sources.data(),
                          host_sources.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));

    const std::vector<Measurement> measurements =
        measure_cases(selected, options, sources, elapsed_cycles, sinks);

    if (!options.quiet) {
      std::cout
          << "case,category,ops_per_iter,iters,avg_cycles,median_cycles,"
             "min_cycles,median_cycles_per_op,min_cycles_per_op,"
             "registers_per_thread,local_bytes,correctness\n";
      for (const auto& measurement : measurements) {
        const double operations =
            static_cast<double>(options.iterations) *
            measurement.spec->operations_per_iteration;
        const double median_cycles_per_op =
            measurement.median_cycles / operations;
        const double minimum_cycles_per_op =
            measurement.minimum_cycles / operations;
        std::cout << measurement.spec->name << ','
                  << measurement.spec->category << ','
                  << measurement.spec->operations_per_iteration << ','
                  << options.iterations << ',' << std::fixed
                  << std::setprecision(3) << measurement.average_cycles << ','
                  << measurement.median_cycles << ','
                  << measurement.minimum_cycles << ',' << std::setprecision(6)
                  << median_cycles_per_op << ',' << minimum_cycles_per_op << ','
                  << measurement.registers_per_thread << ','
                  << measurement.local_bytes << ','
                  << (measurement.passed ? "PASS" : "FAIL") << '\n';
      }
    }

    CUDA_CHECK(cudaFree(sinks));
    CUDA_CHECK(cudaFree(elapsed_cycles));
    CUDA_CHECK(cudaFree(sources));

    return std::all_of(measurements.begin(), measurements.end(),
                       [](const Measurement& value) { return value.passed; })
               ? 0
               : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
