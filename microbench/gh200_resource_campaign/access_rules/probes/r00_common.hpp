#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

inline void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr)

struct Options {
  int m = 2048, n = 2048, k = 2048, seed = 17, samples = 4096;
  std::string dtype = "fp16", cache = "repeat", backend = "cublaslt";
  Options(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc)
        throw std::runtime_error("missing option value");
      std::string name = argv[i], value = argv[i + 1];
      if (name == "--m")
        m = std::stoi(value);
      else if (name == "--n")
        n = std::stoi(value);
      else if (name == "--k")
        k = std::stoi(value);
      else if (name == "--seed")
        seed = std::stoi(value);
      else if (name == "--samples")
        samples = std::stoi(value);
      else if (name == "--dtype")
        dtype = value;
      else if (name == "--cache")
        cache = value;
      else if (name == "--backend")
        backend = value;
      else
        throw std::runtime_error("unknown option: " + name);
    }
    if (m <= 0 || n <= 0 || k <= 0 || samples <= 0)
      throw std::runtime_error("positive dimensions and sample count required");
    if (cache != "repeat" && cache != "evict_prepared")
      throw std::runtime_error("invalid cache preparation");
  }
};
template <class T>
struct DeviceBuffer {
  T* pointer = nullptr;
  size_t count;
  explicit DeviceBuffer(size_t count) : count(count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (pointer)
      cudaFree(pointer);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
};

// Dyadic, signed, coordinate-dependent inputs. All four storage types represent
// these values exactly. This finite correctness witness is not arbitrary data.
__host__ __device__ inline float input_value(int row, int column, int seed, bool a) {
  int value = (row * (a ? 7 : 5) + column * (a ? 13 : 11) + seed * (a ? 3 : 5)) % 17 - 8;
  return value / 32.0f;
}
template <class T>
__global__ void fill_input(T* data, int rows, int columns, int seed, bool a, bool column_major) {
  size_t total = size_t(rows) * columns;
  for (size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x; q < total;
       q += size_t(gridDim.x) * blockDim.x) {
    int row = column_major ? q % rows : q / columns, column = column_major ? q / rows : q % columns;
    data[q] = T(input_value(row, column, seed, a));
  }
}
__global__ void evict_buffer(unsigned* data, size_t words, unsigned seed) {
  for (size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x; q < words;
       q += size_t(gridDim.x) * blockDim.x)
    data[q] = unsigned(q) * 1664525u + seed;
}
template <class T>
__global__ void gather_output(const T* data, const uint64_t* indices, float* result, int count,
                              int m, int n, bool column_major) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q < count) {
    uint64_t index = indices[q], row = index / n, column = index % n;
    result[q] = float(data[column_major ? row + column * m : index]);
  }
}
inline std::vector<uint64_t> sample_indices(int m, int n, int requested, unsigned seed) {
  size_t count = std::min(size_t(requested), size_t(m) * n);
  std::set<uint64_t> points;
  if (count == size_t(m) * n) {
    for (uint64_t i = 0; i < count; ++i)
      points.insert(i);
  } else {
    for (int row : {0, std::min(127, m - 1), std::min(128, m - 1), m - 1})
      for (int column : {0, std::min(255, n - 1), std::min(256, n - 1), n - 1})
        if (points.size() < count)
          points.insert(uint64_t(row) * n + column);
    uint64_t state = seed;
    while (points.size() < count) {
      state = state * 6364136223846793005ull + 1442695040888963407ull;
      points.insert((state >> 16) % (uint64_t(m) * n));
    }
  }
  return {points.begin(), points.end()};
}
struct Errors {
  double max_absolute = 0, sample_relative_l2 = 0, max_relative_nonzero = 0, max_storage_error = 0;
  size_t nonfinite = 0;
  std::vector<uint64_t> indices;
  std::vector<float> values;
};
template <class Input, class Output>
Errors check_output(const Options& options, const Output* data, bool column_major) {
  auto indices = sample_indices(options.m, options.n, options.samples, options.seed);
  DeviceBuffer<uint64_t> device_indices(indices.size());
  DeviceBuffer<float> gathered(indices.size());
  CUDA_CHECK(cudaMemcpy(device_indices.pointer, indices.data(), indices.size() * 8,
                        cudaMemcpyHostToDevice));
  gather_output<<<(indices.size() + 255) / 256, 256>>>(data, device_indices.pointer,
                                                       gathered.pointer, indices.size(), options.m,
                                                       options.n, column_major);
  CUDA_CHECK(cudaGetLastError());
  std::vector<float> values(indices.size());
  CUDA_CHECK(
      cudaMemcpy(values.data(), gathered.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
  Errors result;
  double difference_squared = 0, reference_squared = 0;
  for (size_t q = 0; q < indices.size(); ++q) {
    int row = indices[q] / options.n, column = indices[q] % options.n;
    double reference = 0;
    for (int k = 0; k < options.k; ++k)
      reference += double(float(Input(input_value(row, k, options.seed, true)))) *
                   double(float(Input(input_value(k, column, options.seed, false))));
    if (!std::isfinite(values[q])) {
      ++result.nonfinite;
      continue;
    }
    double difference = std::abs(double(values[q]) - reference);
    result.max_absolute = std::max(result.max_absolute, difference);
    result.max_storage_error =
        std::max(result.max_storage_error,
                 std::abs(double(values[q]) - double(float(Output(float(reference))))));
    if (std::abs(reference) > 1e-8)
      result.max_relative_nonzero =
          std::max(result.max_relative_nonzero, difference / std::abs(reference));
    difference_squared += difference * difference;
    reference_squared += reference * reference;
  }
  result.sample_relative_l2 = reference_squared ? std::sqrt(difference_squared / reference_squared)
                                                : std::sqrt(difference_squared);
  result.indices = std::move(indices);
  result.values = std::move(values);
  return result;
}
inline double coefficient_of_variation(const std::vector<float>& values) {
  double mean = 0;
  for (float value : values)
    mean += value;
  mean /= values.size();
  double variance = 0;
  for (float value : values)
    variance += (value - mean) * (value - mean);
  return mean ? std::sqrt(variance / values.size()) / mean : 0;
}
template <class Input, class Output, class Launch>
int measure_gemm(const Options& options, Input* a, Input* b, Output* d, bool b_column,
                 bool d_column, Launch launch, const std::string& implementation) {
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  fill_input<<<256, 256>>>(a, options.m, options.k, options.seed, true, false);
  fill_input<<<256, 256>>>(b, options.k, options.n, options.seed, false, b_column);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemset(d, 0xff, size_t(options.m) * options.n * sizeof(Output)));
  DeviceBuffer<unsigned> eviction(std::max<size_t>(1, size_t(properties.l2CacheSize) * 2 / 4));
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  auto timed = [&]() {
    if (options.cache == "evict_prepared") {
      evict_buffer<<<256, 256>>>(eviction.pointer, eviction.count, options.seed);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaEventRecord(begin));
    launch();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    return milliseconds;
  };
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> warmup;
  bool converged = false;
  for (int i = 0; i < 30; ++i) {
    warmup.push_back(timed());
    if (i >= 7) {
      std::vector<float> last(warmup.end() - 5, warmup.end());
      if (coefficient_of_variation(last) <= 0.02) {
        converged = true;
        break;
      }
    }
  }
  float milliseconds = timed();
  Errors errors = check_output<Input, Output>(options, d, d_column);
  // This check applies only to the exact dyadic witness above; workload error
  // tolerance remains unspecified and is reported separately.
  bool witness_ok = errors.nonfinite == 0 && errors.max_storage_error <= 1e-5;
  std::cout << std::setprecision(17) << "{\"kind\":\"gemm\",\"status\":\""
            << (witness_ok ? "measured" : "numeric_error") << "\",\"backend\":\"" << options.backend
            << "\",\"dtype\":\"" << options.dtype << "\",\"m\":" << options.m
            << ",\"n\":" << options.n << ",\"k\":" << options.k << ",\"cache_preparation\":\""
            << options.cache << "\",\"seed\":" << options.seed << ",\"elapsed_ms\":" << milliseconds
            << ",\"work_flop\":" << uint64_t(2) * options.m * options.n * options.k
            << ",\"sample_count\":"
            << std::min(size_t(options.samples), size_t(options.m) * options.n)
            << ",\"full_output_checked\":"
            << (size_t(options.samples) >= size_t(options.m) * options.n ? "true" : "false")
            << ",\"max_absolute_error\":" << errors.max_absolute
            << ",\"sample_relative_l2_error\":" << errors.sample_relative_l2
            << ",\"max_relative_error_nonzero\":" << errors.max_relative_nonzero
            << ",\"max_storage_reference_error\":" << errors.max_storage_error
            << ",\"nonfinite\":" << errors.nonfinite
            << ",\"task_tolerance_status\":\"unspecified\",\"load_frequency_mhz\":null"
            << ",\"sm_count\":" << properties.multiProcessorCount
            << ",\"l2_bytes\":" << properties.l2CacheSize
            << ",\"global_memory_bytes\":" << properties.totalGlobalMem
            << ",\"compute_capability\":[" << properties.major << ',' << properties.minor << ']'
            << ",\"warmup_converged\":" << (converged ? "true" : "false") << ",\"warmup_ms\":[";
  for (size_t i = 0; i < warmup.size(); ++i) {
    if (i)
      std::cout << ',';
    std::cout << warmup[i];
  }
  std::cout << "],\"implementation\":" << implementation << ",\"checked_indices\":[";
  for (size_t i = 0; i < errors.indices.size(); ++i) {
    if (i)
      std::cout << ',';
    std::cout << errors.indices[i];
  }
  std::cout << "],\"checked_values\":[";
  for (size_t i = 0; i < errors.values.size(); ++i) {
    if (i)
      std::cout << ',';
    if (std::isfinite(errors.values[i]))
      std::cout << errors.values[i];
    else
      std::cout << "null";
  }
  std::cout << "]}\n";
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return witness_ok ? 0 : 2;
}
