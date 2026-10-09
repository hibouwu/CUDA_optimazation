#pragma once

// Helpers for the R10/R13/R14/R15 follow-up only. Leading dimensions are in elements.
// Include r00_common.hpp explicitly here so old probe semantics remain unchanged.
#include "r00_common.hpp"
#include <cstring>
#include <limits>

namespace gaps {

// Random input depends on logical coordinates, not pitch or launch geometry.
__host__ __device__ inline float random_input_value(int row, int column, int seed, bool is_a) {
  uint32_t x = uint32_t(seed) ^ (is_a ? 0xa511e9b3u : 0x63d83595u);
  x ^= uint32_t(row) * 0x9e3779b9u;
  x ^= uint32_t(column) * 0x85ebca6bu;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return float(x >> 8) * 0x1p-23f - 1.0f;
}

inline void validate_layout(int rows, int columns, int64_t ld) {
  if (rows <= 0 || columns <= 0 || ld < columns)
    throw std::runtime_error("positive dimensions and ld >= logical columns required");
  if (uint64_t(rows) > std::numeric_limits<size_t>::max() / uint64_t(ld))
    throw std::runtime_error("matrix element count overflows size_t");
}

template <class T>
__global__ void fill_input_strided(T* data, int rows, int columns, int64_t ld,
                                   int seed, bool is_a, int input_mode = 0, size_t capacity = 0) {
  size_t total = capacity ? capacity : size_t(rows) * size_t(ld);
  for (size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x; q < total;
       q += size_t(gridDim.x) * blockDim.x) {
    int row = int(q / ld), column = int(q % ld);
    float value = input_mode == 2 ? random_input_value(row, column, seed, is_a)
                                : input_mode == 1 ? 0.0f : input_value(row, column, seed, is_a);
    data[q] = T(row < rows && column < columns ? value : 65504.0f);
  }
}

__global__ void fill_output_sentinel(float* data, size_t count) {
  for (size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x; q < count;
       q += size_t(gridDim.x) * blockDim.x)
    data[q] = __uint_as_float(0x7fc12345u);
}

inline float output_sentinel() {
  uint32_t bits = 0x7fc12345u;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Check every padding element. Copy padding only, outside all performance windows.
template <class T>
size_t padding_errors(const T* data, int rows, int columns, int64_t ld, T expected) {
  validate_layout(rows, columns, ld);
  if (ld == columns)
    return 0;
  size_t width = size_t(ld - columns);
  std::vector<T> padding(size_t(rows) * width);
  CUDA_CHECK(cudaMemcpy2D(padding.data(), width * sizeof(T), data + columns,
                         size_t(ld) * sizeof(T), width * sizeof(T), rows,
                         cudaMemcpyDeviceToHost));
  size_t errors = 0;
  for (const T& value : padding)
    errors += std::memcmp(&value, &expected, sizeof(T)) != 0;
  return errors;
}

inline std::vector<uint64_t> checked_indices(int m, int n, int requested, unsigned seed) {
  if (m <= 0 || n <= 0 || requested <= 0)
    throw std::runtime_error("positive shape and sample count required");
  size_t total = size_t(m) * size_t(n);
  size_t count = std::min(size_t(requested), total);
  std::set<uint64_t> points;
  if (count == total) {
    for (size_t q = 0; q < total; ++q)
      points.insert(q);
  } else {
    // Both 64/128/256 tile boundaries, first/last rows and columns.
    for (int row : {0, 63, 64, 127, 128, 255, 256, m - 2, m - 1})
      for (int col : {0, 63, 64, 127, 128, 255, 256, n - 2, n - 1})
        if (row >= 0 && row < m && col >= 0 && col < n && points.size() < count)
          points.insert(uint64_t(row) * n + col);
    uint64_t state = seed;
    while (points.size() < count) {
      state = state * 6364136223846793005ull + 1442695040888963407ull;
      points.insert((state >> 16) % total);
    }
  }
  return {points.begin(), points.end()};
}

__global__ void gather_strided(const float* data, const uint64_t* indices, float* values,
                               int count, int columns, int64_t ld) {
  int q = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (q < count) {
    uint64_t index = indices[q];
    values[q] = data[(index / columns) * ld + index % columns];
  }
}

// Exact for the existing period-17 dyadic FP16 input witness, not arbitrary inputs.
inline double dyadic_reference(int row, int col, int k, int seed) {
  int64_t period = 0, remainder = 0;
  for (int t = 0; t < 17; ++t) {
    int64_t a = (int64_t(row) * 7 + t * 13 + int64_t(seed) * 3) % 17 - 8;
    int64_t b = (t * 5 + int64_t(col) * 11 + int64_t(seed) * 5) % 17 - 8;
    period += a * b;
    if (t < k % 17)
      remainder += a * b;
  }
  return double(period * (k / 17) + remainder) / 1024.0;
}

inline Errors check_gemm_output(const float* data, int m, int n, int k, int64_t ldd,
                                int seed = 17, int requested = 4096) {
  validate_layout(m, n, ldd);
  if (k <= 0 || seed < 0)
    throw std::runtime_error("positive K and nonnegative witness seed required");
  Errors result;
  result.indices = checked_indices(m, n, requested, unsigned(seed));
  DeviceBuffer<uint64_t> indices(result.indices.size());
  DeviceBuffer<float> values(result.indices.size());
  CUDA_CHECK(cudaMemcpy(indices.pointer, result.indices.data(), indices.count * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  gather_strided<<<(indices.count + 255) / 256, 256>>>(
      data, indices.pointer, values.pointer, int(indices.count), n, ldd);
  CUDA_CHECK(cudaGetLastError());
  result.values.resize(indices.count);
  CUDA_CHECK(cudaMemcpy(result.values.data(), values.pointer, values.count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double squared_error = 0, squared_reference = 0;
  for (size_t q = 0; q < result.indices.size(); ++q) {
    uint64_t index = result.indices[q];
    double reference = dyadic_reference(int(index / n), int(index % n), k, seed);
    float value = result.values[q];
    if (!std::isfinite(value)) {
      ++result.nonfinite;
      continue;
    }
    double error = std::abs(double(value) - reference);
    result.max_absolute = std::max(result.max_absolute, error);
    result.max_storage_error = std::max(result.max_storage_error,
                                       std::abs(double(value) - double(float(reference))));
    if (reference != 0)
      result.max_relative_nonzero = std::max(result.max_relative_nonzero,
                                            error / std::abs(reference));
    squared_error += error * error;
    squared_reference += reference * reference;
  }
  result.sample_relative_l2 = std::sqrt(squared_reference ? squared_error / squared_reference
                                                         : squared_error);
  return result;
}

// Emit an individual JSON event. Keep the exact observed values for independent CPU replay.
inline void print_check(const Errors& errors, size_t padding_bad, bool numerical_ok) {
  bool ok = errors.nonfinite == 0 && numerical_ok && padding_bad == 0;
  std::cout << std::setprecision(17) << "{\"event\":\"check\",\"status\":\""
            << (ok ? "ok" : "numeric_error") << "\",\"samples\":" << errors.values.size()
            << ",\"nonfinite\":" << errors.nonfinite
            << ",\"padding_errors\":" << padding_bad
            << ",\"max_storage_reference_error\":" << errors.max_storage_error
            << ",\"checked_indices\":[";
  for (size_t q = 0; q < errors.indices.size(); ++q)
    std::cout << (q ? "," : "") << errors.indices[q];
  std::cout << "],\"checked_values\":[";
  for (size_t q = 0; q < errors.values.size(); ++q) {
    if (q)
      std::cout << ',';
    if (std::isfinite(errors.values[q]))
      std::cout << errors.values[q];
    else
      std::cout << "null";
  }
  std::cout << "]}\n";
}

inline void print_check(const Errors& errors, size_t padding_bad = 0) {
  print_check(errors, padding_bad, errors.max_storage_error == 0);
}

}  // namespace gaps
