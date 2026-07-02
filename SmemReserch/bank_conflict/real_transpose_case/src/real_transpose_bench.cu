#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

struct Options {
  std::string case_selector = "all";
  int width = 4096;
  int height = 4096;
  int iters = 10;
  int warmups = 2;
  int repeats = 10;
  bool list_cases = false;
};

enum class Backend {
  kNaive,
  kCoalescedRead,
  kCoalescedWrite,
  kSmemPitch32,
  kSmemPitch33,
  kPackedPitch33,
  kXorSwizzle,
  kCopy,
};

struct CaseDefinition {
  const char* experiment;
  const char* name;
  Backend backend;
  int smem_pitch;
  int vector_width;
  const char* swizzle;
};

constexpr CaseDefinition kCases[] = {
    {"R0", "R0_transpose_naive", Backend::kNaive, 0, 1, "none"},
    {"R0", "R0_transpose_coalesced_read", Backend::kCoalescedRead, 0, 1,
     "none"},
    {"R0", "R0_transpose_coalesced_write", Backend::kCoalescedWrite, 0, 1,
     "none"},
    {"R1", "R1_transpose_smem_pitch32", Backend::kSmemPitch32, 32, 1,
     "none"},
    {"R2", "R2_transpose_smem_pitch33", Backend::kSmemPitch33, 33, 1,
     "none"},
    {"R3", "R3_transpose_smem_packed_pitch33", Backend::kPackedPitch33, 33,
     4, "none"},
    {"R4", "R4_transpose_smem_xor_swizzle", Backend::kXorSwizzle, 32, 1,
     "xor"},
    {"R5", "R5_transpose_copy_baseline", Backend::kCopy, 0, 1, "none"},
};

__global__ void transpose_naive(const float* input, float* output, int width,
                                int height) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t elements = static_cast<size_t>(width) * height;
  if (index >= elements) {
    return;
  }
  const int row = static_cast<int>(index / width);
  const int col = static_cast<int>(index - static_cast<size_t>(row) * width);
  output[static_cast<size_t>(col) * height + row] = input[index];
}

__global__ void transpose_coalesced_read(const float* input, float* output,
                                         int width, int height) {
  const int col = static_cast<int>(blockIdx.x) * kTileDim + threadIdx.x;
  const int row_base =
      static_cast<int>(blockIdx.y) * kTileDim + threadIdx.y;
  if (col >= width) {
    return;
  }
  #pragma unroll
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    const int row = row_base + offset;
    if (row < height) {
      output[static_cast<size_t>(col) * height + row] =
          input[static_cast<size_t>(row) * width + col];
    }
  }
}

__global__ void transpose_coalesced_write(const float* input, float* output,
                                          int width, int height) {
  const int output_col =
      static_cast<int>(blockIdx.x) * kTileDim + threadIdx.x;
  const int output_row_base =
      static_cast<int>(blockIdx.y) * kTileDim + threadIdx.y;
  if (output_col >= height) {
    return;
  }
  #pragma unroll
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    const int output_row = output_row_base + offset;
    if (output_row < width) {
      output[static_cast<size_t>(output_row) * height + output_col] =
          input[static_cast<size_t>(output_col) * width + output_row];
    }
  }
}

template <int Pitch>
__global__ void transpose_smem(const float* input, float* output, int width,
                               int height) {
  __shared__ float tile[kTileDim][Pitch];

  const int input_col =
      static_cast<int>(blockIdx.x) * kTileDim + threadIdx.x;
  const int input_row_base =
      static_cast<int>(blockIdx.y) * kTileDim + threadIdx.y;

  #pragma unroll
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    const int input_row = input_row_base + offset;
    if (input_col < width && input_row < height) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[static_cast<size_t>(input_row) * width + input_col];
    }
  }
  __syncthreads();

  const int output_col =
      static_cast<int>(blockIdx.y) * kTileDim + threadIdx.x;
  const int output_row_base =
      static_cast<int>(blockIdx.x) * kTileDim + threadIdx.y;

  #pragma unroll
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    const int output_row = output_row_base + offset;
    if (output_col < height && output_row < width) {
      output[static_cast<size_t>(output_row) * height + output_col] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

__global__ void transpose_smem_packed_pitch33(const float* input,
                                               float* output, int width,
                                               int height,
                                               bool use_float4) {
  __shared__ float tile[kTileDim][kTileDim + 1];

  const int linear_thread = threadIdx.y * kTileDim + threadIdx.x;
  const int local_row = linear_thread / (kTileDim / 4);
  const int local_col_base = (linear_thread % (kTileDim / 4)) * 4;
  const int input_row =
      static_cast<int>(blockIdx.y) * kTileDim + local_row;
  const int input_col_base =
      static_cast<int>(blockIdx.x) * kTileDim + local_col_base;

  if (input_row < height) {
    if (use_float4 && input_col_base + 3 < width) {
      const float4 value = *reinterpret_cast<const float4*>(
          input + static_cast<size_t>(input_row) * width + input_col_base);
      tile[local_row][local_col_base + 0] = value.x;
      tile[local_row][local_col_base + 1] = value.y;
      tile[local_row][local_col_base + 2] = value.z;
      tile[local_row][local_col_base + 3] = value.w;
    } else {
      #pragma unroll
      for (int component = 0; component < 4; ++component) {
        const int input_col = input_col_base + component;
        if (input_col < width) {
          tile[local_row][local_col_base + component] =
              input[static_cast<size_t>(input_row) * width + input_col];
        }
      }
    }
  }
  __syncthreads();

  const int output_col_base =
      static_cast<int>(blockIdx.y) * kTileDim + local_col_base;
  const int output_row =
      static_cast<int>(blockIdx.x) * kTileDim + local_row;

  if (output_row < width) {
    if (use_float4 && output_col_base + 3 < height) {
      const float4 value = {
          tile[local_col_base + 0][local_row],
          tile[local_col_base + 1][local_row],
          tile[local_col_base + 2][local_row],
          tile[local_col_base + 3][local_row],
      };
      *reinterpret_cast<float4*>(
          output + static_cast<size_t>(output_row) * height +
          output_col_base) = value;
    } else {
      #pragma unroll
      for (int component = 0; component < 4; ++component) {
        const int output_col = output_col_base + component;
        if (output_col < height) {
          output[static_cast<size_t>(output_row) * height + output_col] =
              tile[local_col_base + component][local_row];
        }
      }
    }
  }
}

__device__ __forceinline__ int swizzled_col(int logical_row,
                                            int logical_col) {
  return logical_col ^ logical_row;
}

__global__ void transpose_smem_xor_swizzle(const float* input, float* output,
                                            int width, int height) {
  __shared__ float tile[kTileDim][kTileDim];

  const int input_col =
      static_cast<int>(blockIdx.x) * kTileDim + threadIdx.x;
  const int input_row_base =
      static_cast<int>(blockIdx.y) * kTileDim + threadIdx.y;

  #pragma unroll
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    const int logical_row = threadIdx.y + offset;
    const int input_row = input_row_base + offset;
    if (input_col < width && input_row < height) {
      tile[logical_row][swizzled_col(logical_row, threadIdx.x)] =
          input[static_cast<size_t>(input_row) * width + input_col];
    }
  }
  __syncthreads();

  const int output_col =
      static_cast<int>(blockIdx.y) * kTileDim + threadIdx.x;
  const int output_row_base =
      static_cast<int>(blockIdx.x) * kTileDim + threadIdx.y;

  #pragma unroll
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    const int logical_col = threadIdx.y + offset;
    const int output_row = output_row_base + offset;
    if (output_col < height && output_row < width) {
      output[static_cast<size_t>(output_row) * height + output_col] =
          tile[threadIdx.x][swizzled_col(threadIdx.x, logical_col)];
    }
  }
}

__global__ void copy_kernel(const float* input, float* output,
                            size_t elements) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = input[index];
  }
}

int parse_positive_int(const std::string& option, const char* text) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(option + " requires a positive integer");
  }
  return static_cast<int>(value);
}

int parse_nonnegative_int(const std::string& option, const char* text) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(option + " requires a nonnegative integer");
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next_value = [&](const std::string& option) -> const char* {
      if (++index >= argc) {
        throw std::runtime_error(option + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--case") {
      options.case_selector = next_value(argument);
    } else if (argument == "--width") {
      options.width = parse_positive_int(argument, next_value(argument));
    } else if (argument == "--height") {
      options.height = parse_positive_int(argument, next_value(argument));
    } else if (argument == "--iters") {
      options.iters = parse_positive_int(argument, next_value(argument));
    } else if (argument == "--warmups") {
      options.warmups =
          parse_nonnegative_int(argument, next_value(argument));
    } else if (argument == "--repeats") {
      options.repeats = parse_positive_int(argument, next_value(argument));
    } else if (argument == "--list-cases") {
      options.list_cases = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "Usage: real_transpose_bench [--case all|R0|...|case-name]"
          << " [--width N] [--height N] [--iters N] [--warmups N]"
          << " [--repeats N] [--list-cases]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

std::vector<const CaseDefinition*> select_cases(const std::string& selector) {
  std::vector<const CaseDefinition*> selected;
  for (const auto& definition : kCases) {
    if (selector == "all" || selector == definition.experiment ||
        selector == definition.name) {
      selected.push_back(&definition);
    }
  }
  if (selected.empty()) {
    throw std::runtime_error("unknown case selector: " + selector);
  }
  return selected;
}

void launch_case(const CaseDefinition& definition, const float* input,
                 float* output, int width, int height, bool use_float4,
                 cudaStream_t stream = nullptr) {
  const dim3 block(kTileDim, kBlockRows);
  const dim3 input_grid((width + kTileDim - 1) / kTileDim,
                        (height + kTileDim - 1) / kTileDim);
  const dim3 output_grid((height + kTileDim - 1) / kTileDim,
                         (width + kTileDim - 1) / kTileDim);
  const size_t elements = static_cast<size_t>(width) * height;

  switch (definition.backend) {
    case Backend::kNaive: {
      constexpr int threads = 256;
      const int blocks = static_cast<int>((elements + threads - 1) / threads);
      transpose_naive<<<blocks, threads, 0, stream>>>(input, output, width,
                                                       height);
      break;
    }
    case Backend::kCoalescedRead:
      transpose_coalesced_read<<<input_grid, block, 0, stream>>>(
          input, output, width, height);
      break;
    case Backend::kCoalescedWrite:
      transpose_coalesced_write<<<output_grid, block, 0, stream>>>(
          input, output, width, height);
      break;
    case Backend::kSmemPitch32:
      transpose_smem<32><<<input_grid, block, 0, stream>>>(
          input, output, width, height);
      break;
    case Backend::kSmemPitch33:
      transpose_smem<33><<<input_grid, block, 0, stream>>>(
          input, output, width, height);
      break;
    case Backend::kPackedPitch33: {
      transpose_smem_packed_pitch33<<<input_grid, block, 0, stream>>>(
          input, output, width, height, use_float4);
      break;
    }
    case Backend::kXorSwizzle:
      transpose_smem_xor_swizzle<<<input_grid, block, 0, stream>>>(
          input, output, width, height);
      break;
    case Backend::kCopy: {
      constexpr int threads = 256;
      const int blocks = static_cast<int>((elements + threads - 1) / threads);
      copy_kernel<<<blocks, threads, 0, stream>>>(input, output, elements);
      break;
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

bool check_correctness(const CaseDefinition& definition,
                       const std::vector<float>& input,
                       const std::vector<float>& output, int width,
                       int height) {
  if (definition.backend == Backend::kCopy) {
    for (size_t index = 0; index < input.size(); ++index) {
      if (output[index] != input[index]) {
        std::cerr << definition.name << ": mismatch at linear index " << index
                  << ", expected " << input[index] << ", got " << output[index]
                  << '\n';
        return false;
      }
    }
    return true;
  }

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const float expected = input[static_cast<size_t>(row) * width + col];
      const float actual =
          output[static_cast<size_t>(col) * height + row];
      if (actual != expected) {
        std::cerr << definition.name << ": mismatch at output[" << col << ","
                  << row << "], expected " << expected << ", got " << actual
                  << '\n';
        return false;
      }
    }
  }
  return true;
}

struct Timing {
  double average_ms;
  double minimum_ms;
};

Timing benchmark_case(const CaseDefinition& definition, const float* input,
                      float* output, const Options& options,
                      bool use_float4) {
  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    launch_case(definition, input, output, options.width, options.height,
                use_float4);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<double> samples;
  samples.reserve(options.repeats);
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < options.iters; ++iteration) {
      launch_case(definition, input, output, options.width, options.height,
                  use_float4);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    samples.push_back(static_cast<double>(elapsed_ms) / options.iters);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  return {sum / samples.size(),
          *std::min_element(samples.begin(), samples.end())};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto selected = select_cases(options.case_selector);
    if (options.list_cases) {
      for (const auto* definition : selected) {
        std::cout << definition->name << '\n';
      }
      return 0;
    }

    const size_t elements =
        static_cast<size_t>(options.width) * options.height;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
      throw std::runtime_error("matrix size overflows allocation size");
    }
    const size_t bytes = elements * sizeof(float);
    std::vector<float> host_input(elements);
    std::vector<float> host_output(elements);
    for (size_t index = 0; index < elements; ++index) {
      host_input[index] =
          static_cast<float>((index * static_cast<size_t>(17) + 3) % 65521);
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));
    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), bytes,
                          cudaMemcpyHostToDevice));

    std::cout << "experiment,case,width,height,dtype,tile_dim,block_rows,"
                 "smem_pitch,vector_width,swizzle,avg_ms,min_ms,"
                 "effective_GBps,correctness\n";
    bool all_passed = true;
    for (const auto* definition : selected) {
      const bool use_float4 =
          definition->backend == Backend::kPackedPitch33 &&
          options.width % 4 == 0 && options.height % 4 == 0;
      if (definition->backend == Backend::kPackedPitch33 && !use_float4) {
        std::cerr << definition->name
                  << ": width and height must both be divisible by 4 for "
                     "float4 global accesses; using scalar fallback\n";
      }

      CUDA_CHECK(cudaMemset(device_output, 0, bytes));
      const Timing timing =
          benchmark_case(*definition, device_input, device_output, options,
                         use_float4);
      CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, bytes,
                            cudaMemcpyDeviceToHost));
      const bool correct = check_correctness(*definition, host_input,
                                             host_output, options.width,
                                             options.height);
      all_passed = all_passed && correct;

      const double transferred_bytes = static_cast<double>(bytes) * 2.0;
      const double effective_gbps =
          transferred_bytes / (timing.average_ms * 1.0e6);
      const bool is_copy = definition->backend == Backend::kCopy;
      const int tile_dim = is_copy || definition->backend == Backend::kNaive
                               ? 0
                               : kTileDim;
      const int block_rows =
          is_copy || definition->backend == Backend::kNaive ? 0 : kBlockRows;
      const int actual_vector_width =
          definition->backend == Backend::kPackedPitch33
              ? (use_float4 ? 4 : 1)
              : definition->vector_width;

      std::cout << definition->experiment << ',' << definition->name << ','
                << options.width << ',' << options.height
                << ",f32," << tile_dim << ',' << block_rows << ','
                << definition->smem_pitch << ',' << actual_vector_width << ','
                << definition->swizzle << ',' << std::fixed
                << std::setprecision(6) << timing.average_ms << ','
                << timing.minimum_ms << ',' << std::setprecision(3)
                << effective_gbps << ',' << (correct ? "PASS" : "FAIL")
                << '\n';
    }

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));
    return all_passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
