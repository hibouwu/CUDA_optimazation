#include <cuda.h>

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

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    const CUresult result_ = (call);                                            \
    if (result_ != CUDA_SUCCESS) {                                              \
      const char* name_ = nullptr;                                              \
      const char* message_ = nullptr;                                           \
      cuGetErrorName(result_, &name_);                                          \
      cuGetErrorString(result_, &message_);                                     \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               (name_ ? name_ : "CUDA_ERROR") + " (" +         \
                               (message_ ? message_ : "unknown") + ")");       \
    }                                                                          \
  } while (false)

namespace {

constexpr int kWarpSize = 32;
constexpr int kSourceCount = 64;
constexpr int kOperationsPerIteration = 128;

struct Options {
  int iterations = 100000;
  int warmups = 5;
  int repeats = 20;
  std::vector<std::string> cubins;
};

int parse_integer(const char* text, const char* option, bool allow_zero) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  const long minimum = allow_zero ? 0 : 1;
  if (end == text || *end != '\0' || value < minimum ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid value for ") + option);
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
    if (argument == "--iters") {
      options.iterations =
          parse_integer(require_value("--iters"), "--iters", false);
    } else if (argument == "--warmups") {
      options.warmups =
          parse_integer(require_value("--warmups"), "--warmups", true);
    } else if (argument == "--repeats") {
      options.repeats =
          parse_integer(require_value("--repeats"), "--repeats", false);
    } else if (argument == "--cubin") {
      options.cubins.emplace_back(require_value("--cubin"));
    } else if (argument == "--help") {
      std::cout << "Usage: sass_register_bench [--iters N] [--warmups N] "
                   "[--repeats N] --cubin FILE [--cubin FILE ...]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (options.cubins.empty()) {
    throw std::runtime_error("at least one --cubin is required");
  }
  return options;
}

std::string case_name(const std::string& path) {
  const std::size_t slash = path.find_last_of('/');
  std::string name =
      slash == std::string::npos ? path : path.substr(slash + 1);
  const std::string suffix = ".cubin";
  if (name.size() >= suffix.size() &&
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
    name.resize(name.size() - suffix.size());
  }
  return name;
}

struct LoadedCase {
  std::string name;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  int registers = 0;
  int local_bytes = 0;
  std::vector<std::uint64_t> samples;
  bool sink_valid = true;
};

void launch(LoadedCase& probe, CUdeviceptr sources, CUdeviceptr elapsed,
            CUdeviceptr sinks, int iterations) {
  void* arguments[] = {&sources, &elapsed, &sinks, &iterations};
  CU_CHECK(cuLaunchKernel(probe.function, 1, 1, 1, kWarpSize, 1, 1, 0,
                          nullptr, arguments, nullptr));
}

}  // namespace

int main(int argc, char** argv) {
  CUcontext context = nullptr;
  CUdeviceptr sources = 0;
  CUdeviceptr elapsed = 0;
  CUdeviceptr sinks = 0;
  std::vector<LoadedCase> probes;
  try {
    const Options options = parse_options(argc, argv);
    CU_CHECK(cuInit(0));
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuCtxCreate(&context, nullptr, 0, device));

    std::vector<std::uint32_t> host_sources(kSourceCount * kWarpSize);
    for (int source = 0; source < kSourceCount; ++source) {
      for (int lane = 0; lane < kWarpSize; ++lane) {
        host_sources[source * kWarpSize + lane] =
            static_cast<std::uint32_t>(0x9e3779b9u * (source + 1) +
                                       0x45d9f3bu * (lane + 1));
      }
    }
    CU_CHECK(cuMemAlloc(&sources, host_sources.size() * sizeof(std::uint32_t)));
    CU_CHECK(cuMemAlloc(&elapsed, kWarpSize * sizeof(std::uint64_t)));
    CU_CHECK(cuMemAlloc(&sinks, kWarpSize * sizeof(std::uint32_t)));
    CU_CHECK(cuMemcpyHtoD(sources, host_sources.data(),
                          host_sources.size() * sizeof(std::uint32_t)));

    probes.reserve(options.cubins.size());
    for (const auto& path : options.cubins) {
      LoadedCase probe;
      probe.name = case_name(path);
      probe.samples.reserve(options.repeats * kWarpSize);
      CU_CHECK(cuModuleLoad(&probe.module, path.c_str()));
      CU_CHECK(cuModuleGetFunction(&probe.function, probe.module,
                                   "sass_register_probe"));
      CU_CHECK(cuFuncGetAttribute(&probe.registers, CU_FUNC_ATTRIBUTE_NUM_REGS,
                                  probe.function));
      CU_CHECK(cuFuncGetAttribute(&probe.local_bytes,
                                  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                  probe.function));
      probes.push_back(std::move(probe));
    }

    for (int warmup = 0; warmup < options.warmups; ++warmup) {
      for (auto& probe : probes) {
        launch(probe, sources, elapsed, sinks, options.iterations);
      }
    }
    CU_CHECK(cuCtxSynchronize());

    std::vector<std::uint64_t> host_cycles(kWarpSize);
    std::vector<std::uint32_t> host_sinks(kWarpSize);
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      for (std::size_t slot = 0; slot < probes.size(); ++slot) {
        auto& probe = probes[(slot + repeat) % probes.size()];
        launch(probe, sources, elapsed, sinks, options.iterations);
        CU_CHECK(cuMemcpyDtoH(host_cycles.data(), elapsed,
                              host_cycles.size() * sizeof(std::uint64_t)));
        CU_CHECK(cuMemcpyDtoH(host_sinks.data(), sinks,
                              host_sinks.size() * sizeof(std::uint32_t)));
        probe.samples.insert(probe.samples.end(), host_cycles.begin(),
                             host_cycles.end());
        probe.sink_valid =
            probe.sink_valid &&
            std::any_of(host_sinks.begin(), host_sinks.end(),
                        [](std::uint32_t value) { return value != 0; });
      }
    }

    std::cout << "case,ops_per_iter,iters,avg_cycles,median_cycles,min_cycles,"
                 "median_cycles_per_op,min_cycles_per_op,"
                 "registers_per_thread,local_bytes,correctness\n";
    bool all_passed = true;
    for (auto& probe : probes) {
      std::sort(probe.samples.begin(), probe.samples.end());
      const double average =
          std::accumulate(probe.samples.begin(), probe.samples.end(), 0.0) /
          static_cast<double>(probe.samples.size());
      const std::size_t middle = probe.samples.size() / 2;
      const double median =
          probe.samples.size() % 2 == 0
              ? (static_cast<double>(probe.samples[middle - 1]) +
                 probe.samples[middle]) /
                    2.0
              : static_cast<double>(probe.samples[middle]);
      const std::uint64_t minimum = probe.samples.front();
      const double operation_count =
          static_cast<double>(options.iterations) * kOperationsPerIteration;
      const bool passed =
          minimum > 0 && probe.sink_valid && probe.local_bytes == 0;
      all_passed = all_passed && passed;
      std::cout << probe.name << ',' << kOperationsPerIteration << ','
                << options.iterations << ',' << std::fixed
                << std::setprecision(3) << average << ',' << median << ','
                << minimum << ',' << std::setprecision(6)
                << median / operation_count << ',' << minimum / operation_count
                << ',' << probe.registers << ',' << probe.local_bytes << ','
                << (passed ? "PASS" : "FAIL") << '\n';
    }

    for (auto& probe : probes) CU_CHECK(cuModuleUnload(probe.module));
    CU_CHECK(cuMemFree(sinks));
    CU_CHECK(cuMemFree(elapsed));
    CU_CHECK(cuMemFree(sources));
    CU_CHECK(cuCtxDestroy(context));
    return all_passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    for (auto& probe : probes) {
      if (probe.module) cuModuleUnload(probe.module);
    }
    if (sinks) cuMemFree(sinks);
    if (elapsed) cuMemFree(elapsed);
    if (sources) cuMemFree(sources);
    if (context) cuCtxDestroy(context);
    return 1;
  }
}
