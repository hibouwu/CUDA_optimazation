#include "gemm_common.cuh"
#include "requant/requant_backend.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using gemm_sm110::requant::Sm110Nvfp4BlockScale;

constexpr int kWarpThreads = 32;
constexpr int kElementsPerThread = 2;
constexpr int kElementsPerCta = kWarpThreads * kElementsPerThread;
constexpr int kNvfp4BlockSize = 16;
constexpr int kScaleGroupsPerCta = kElementsPerCta / kNvfp4BlockSize;
constexpr unsigned int kTmemColumns = 128;
constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3Max = 448.0f;

struct Options {
  int rows = 256;
  int cols = 256;
  int warmup = 10;
  int iterations = 100;
  unsigned int seed = 1234;
  std::string distribution = "normal";
  std::string csv_path;
};

struct ReferenceResult {
  std::vector<std::uint8_t> quantized;
  std::vector<std::uint8_t> block_scales;
  float tensor_scale = 1.0f;
};

void print_usage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "  --rows N\n"
      << "  --cols N\n"
      << "  --distribution uniform|normal|laplace|outlier|lognormal|constant\n"
      << "  --seed N\n"
      << "  --warmup N\n"
      << "  --iterations N\n"
      << "  --csv PATH\n";
}

bool parse_positive_int(const char* value, int* output) {
  const int parsed = std::atoi(value);
  if (parsed <= 0) return false;
  *output = parsed;
  return true;
}

bool parse_options(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (i + 1 >= argc) return false;
    const char* value = argv[++i];
    if (arg == "--rows") {
      if (!parse_positive_int(value, &options->rows)) return false;
    } else if (arg == "--cols") {
      if (!parse_positive_int(value, &options->cols)) return false;
    } else if (arg == "--warmup") {
      if (!parse_positive_int(value, &options->warmup)) return false;
    } else if (arg == "--iterations") {
      if (!parse_positive_int(value, &options->iterations)) return false;
    } else if (arg == "--seed") {
      options->seed = static_cast<unsigned int>(std::strtoul(value, nullptr, 10));
    } else if (arg == "--distribution") {
      options->distribution = value;
    } else if (arg == "--csv") {
      options->csv_path = value;
    } else {
      return false;
    }
  }

  const std::string& d = options->distribution;
  return d == "uniform" || d == "normal" || d == "laplace" ||
         d == "outlier" || d == "lognormal" || d == "constant";
}

std::vector<float> make_input(std::size_t logical_elements,
                              std::size_t padded_elements,
                              const Options& options) {
  std::vector<float> input(padded_elements, 0.0f);
  std::mt19937 rng(options.seed);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::exponential_distribution<float> exponential(1.0f);
  std::lognormal_distribution<float> lognormal(0.0f, 1.25f);
  std::bernoulli_distribution sign(0.5);

  for (std::size_t i = 0; i < logical_elements; ++i) {
    float value = 0.0f;
    if (options.distribution == "uniform") {
      value = uniform(rng);
    } else if (options.distribution == "normal") {
      value = normal(rng);
    } else if (options.distribution == "laplace") {
      value = exponential(rng) * (sign(rng) ? 1.0f : -1.0f);
    } else if (options.distribution == "outlier") {
      value = normal(rng);
      if (i % 257 == 0) value *= 128.0f;
    } else if (options.distribution == "lognormal") {
      value = lognormal(rng) * (sign(rng) ? 1.0f : -1.0f);
    } else {
      value = 0.75f;
    }
    input[i] = value;
  }
  return input;
}

float decode_positive_e4m3(std::uint8_t bits) {
  const int exponent = (bits >> 3) & 0x0f;
  const int mantissa = bits & 0x07;
  if (exponent == 0) {
    return std::ldexp(static_cast<float>(mantissa), -9);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                    exponent - 7);
}

std::uint8_t encode_positive_e4m3_round_up(float value) {
  if (!(value > 0.0f)) return 0u;
  for (int bits = 1; bits <= 0x7e; ++bits) {
    if (decode_positive_e4m3(static_cast<std::uint8_t>(bits)) >= value) {
      return static_cast<std::uint8_t>(bits);
    }
  }
  return 0x7eu;
}

float decode_e2m1(std::uint8_t bits) {
  static constexpr float magnitudes[8] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = magnitudes[bits & 0x07u];
  return (bits & 0x08u) != 0u ? -magnitude : magnitude;
}

ReferenceResult make_reference(const std::vector<float>& input) {
  ReferenceResult result;
  result.quantized.resize(input.size() / 2);
  result.block_scales.resize(input.size() / kNvfp4BlockSize);

  float global_amax = 0.0f;
  for (float value : input) {
    global_amax = std::max(global_amax, std::fabs(value));
  }
  result.tensor_scale =
      global_amax > 0.0f ? global_amax / (kE4M3Max * kE2M1Max) : 1.0f;
  const float inverse_tensor_scale = 1.0f / result.tensor_scale;

  for (std::size_t block = 0; block < result.block_scales.size(); ++block) {
    const std::size_t begin = block * kNvfp4BlockSize;
    float normalized_amax = 0.0f;
    for (int i = 0; i < kNvfp4BlockSize; ++i) {
      normalized_amax =
          std::max(normalized_amax,
                   std::fabs(input[begin + i] * inverse_tensor_scale));
    }

    const std::uint8_t scale_bits =
        encode_positive_e4m3_round_up(normalized_amax / kE2M1Max);
    result.block_scales[block] = scale_bits;
    const float block_scale = decode_positive_e4m3(scale_bits);
    const float multiplier =
        block_scale > 0.0f ? inverse_tensor_scale / block_scale : 0.0f;

    for (int i = 0; i < kNvfp4BlockSize; i += 2) {
      const std::uint8_t value0 = gemm_sm110::requant::encode_e2m1_rn(
          input[begin + i] * multiplier);
      const std::uint8_t value1 = gemm_sm110::requant::encode_e2m1_rn(
          input[begin + i + 1] * multiplier);
      result.quantized[(begin + i) / 2] =
          gemm_sm110::requant::pack_e2m1x2(value0, value1);
    }
  }
  return result;
}

__device__ __forceinline__ unsigned int shared_u32_address(const void* ptr) {
  return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
}

__global__ void sm110_nvfp4_epilogue_benchmark_kernel(
    const float* input, std::uint8_t* quantized,
    std::uint8_t* block_scales, float inverse_tensor_scale,
    int total_chunks) {
  __shared__ unsigned int tmem_base;

#if GEMM_SM110_HAS_NATIVE_REQUANT
  if (threadIdx.x == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(shared_u32_address(&tmem_base)), "r"(kTmemColumns)
        : "memory");
  }
  __syncthreads();

  const int lane = static_cast<int>(threadIdx.x);
  constexpr unsigned int kFullWarpMask = 0xffffffffu;
  for (int chunk = static_cast<int>(blockIdx.x); chunk < total_chunks;
       chunk += static_cast<int>(gridDim.x)) {
    const int element_base = chunk * kElementsPerCta + lane * 2;
    const float source0 = input[element_base];
    const float source1 = input[element_base + 1];
    const unsigned int source0_bits = __float_as_uint(source0);
    const unsigned int source1_bits = __float_as_uint(source1);

    asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};"
                 :
                 : "r"(tmem_base), "r"(source0_bits), "r"(source1_bits)
                 : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

    const auto accumulator =
        gemm_sm110::requant::sm110_tcgen05_load_32x32b_x2(tmem_base);
    float normalized_amax =
        fmaxf(fabsf(accumulator.value0 * inverse_tensor_scale),
              fabsf(accumulator.value1 * inverse_tensor_scale));

    // Eight lanes own 16 consecutive values (two values per lane).
    for (int offset = 4; offset > 0; offset >>= 1) {
      normalized_amax =
          fmaxf(normalized_amax,
                __shfl_xor_sync(kFullWarpMask, normalized_amax, offset, 8));
    }

    const int group = lane / 8;
    const int leader = group * 8;
    Sm110Nvfp4BlockScale scale{0u, 0.0f};
    if ((lane & 7) == 0) {
      scale =
          gemm_sm110::requant::sm110_make_nvfp4_block_scale(normalized_amax);
    }
    scale.e4m3_bits = static_cast<std::uint8_t>(__shfl_sync(
        kFullWarpMask, static_cast<unsigned int>(scale.e4m3_bits), leader));
    scale.decoded =
        __shfl_sync(kFullWarpMask, scale.decoded, leader);

    quantized[chunk * kWarpThreads + lane] =
        gemm_sm110::requant::sm110_requant_nvfp4_e2m1x2(
            accumulator.value0, accumulator.value1, inverse_tensor_scale,
            scale);
    if ((lane & 7) == 0) {
      block_scales[chunk * kScaleGroupsPerCta + group] = scale.e4m3_bits;
    }
    __syncwarp();
  }

  if (threadIdx.x == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(tmem_base), "r"(kTmemColumns)
                 : "memory");
  }
#else
  if (threadIdx.x == 0) {
    quantized[blockIdx.x * kWarpThreads] = 0u;
    block_scales[blockIdx.x * kScaleGroupsPerCta] = 0u;
  }
  (void)total_chunks;
#endif
}

void compute_error(const std::vector<float>& input,
                   const std::vector<std::uint8_t>& quantized,
                   const std::vector<std::uint8_t>& block_scales,
                   float tensor_scale, std::size_t logical_elements,
                   double* rmse, float* max_abs_error) {
  double squared_error = 0.0;
  float maximum = 0.0f;
  for (std::size_t i = 0; i < logical_elements; ++i) {
    const std::uint8_t packed = quantized[i / 2];
    const std::uint8_t nibble =
        (i & 1u) == 0u ? static_cast<std::uint8_t>(packed >> 4)
                       : static_cast<std::uint8_t>(packed & 0x0fu);
    const float block_scale =
        decode_positive_e4m3(block_scales[i / kNvfp4BlockSize]);
    const float reconstructed =
        decode_e2m1(nibble) * block_scale * tensor_scale;
    const float error = reconstructed - input[i];
    squared_error += static_cast<double>(error) * error;
    maximum = std::max(maximum, std::fabs(error));
  }
  *rmse = std::sqrt(squared_error / logical_elements);
  *max_abs_error = maximum;
}

void append_csv(const Options& options, std::size_t logical_elements,
                std::size_t padded_elements, float average_ms,
                double gelements_per_second, double effective_gbps,
                std::size_t value_mismatches, std::size_t scale_mismatches,
                double rmse, float max_abs_error) {
  if (options.csv_path.empty()) return;

  std::ifstream existing(options.csv_path, std::ios::binary | std::ios::ate);
  const bool write_header = !existing || existing.tellg() == 0;
  existing.close();

  std::ofstream csv(options.csv_path, std::ios::app);
  if (!csv) {
    std::cerr << "Unable to open CSV: " << options.csv_path << '\n';
    std::exit(EXIT_FAILURE);
  }
  if (write_header) {
    csv << "rows,cols,logical_elements,padded_elements,distribution,seed,"
           "warmup,iterations,average_ms,gelements_per_second,effective_gbps,"
           "value_mismatches,scale_mismatches,rmse,max_abs_error\n";
  }
  csv << options.rows << ',' << options.cols << ',' << logical_elements << ','
      << padded_elements << ',' << options.distribution << ',' << options.seed
      << ',' << options.warmup << ',' << options.iterations << ','
      << average_ms << ',' << gelements_per_second << ',' << effective_gbps
      << ',' << value_mismatches << ',' << scale_mismatches << ',' << rmse
      << ',' << max_abs_error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaSetDevice(device));
  cudaDeviceProp property{};
  CHECK_CUDA(cudaGetDeviceProperties(&property, device));
  if (property.major != 11) {
    std::cout << "SKIP: benchmark requires an SM110-family GPU, found sm_"
              << property.major << property.minor << '\n';
    return EXIT_SUCCESS;
  }

  const std::size_t logical_elements =
      static_cast<std::size_t>(options.rows) * options.cols;
  const std::size_t padded_elements =
      ((logical_elements + kElementsPerCta - 1) / kElementsPerCta) *
      kElementsPerCta;
  const std::size_t quantized_bytes = padded_elements / 2;
  const std::size_t scale_bytes = padded_elements / kNvfp4BlockSize;

  const std::vector<float> input =
      make_input(logical_elements, padded_elements, options);
  const ReferenceResult reference = make_reference(input);
  const float inverse_tensor_scale = 1.0f / reference.tensor_scale;

  float* device_input = nullptr;
  std::uint8_t* device_quantized = nullptr;
  std::uint8_t* device_scales = nullptr;
  CHECK_CUDA(cudaMalloc(&device_input, padded_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_quantized, quantized_bytes));
  CHECK_CUDA(cudaMalloc(&device_scales, scale_bytes));
  CHECK_CUDA(cudaMemcpy(device_input, input.data(),
                        padded_elements * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int total_chunks =
      static_cast<int>(padded_elements / kElementsPerCta);
  const int worker_count =
      std::min(total_chunks, property.multiProcessorCount * 4);
  const dim3 grid(static_cast<unsigned int>(worker_count));
  const dim3 block(kWarpThreads);
  auto launch = [&]() {
    sm110_nvfp4_epilogue_benchmark_kernel<<<grid, block>>>(
        device_input, device_quantized, device_scales, inverse_tensor_scale,
        total_chunks);
    CHECK_CUDA(cudaGetLastError());
  };

  for (int i = 0; i < options.warmup; ++i) launch();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < options.iterations; ++i) launch();
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  std::vector<std::uint8_t> quantized(quantized_bytes);
  std::vector<std::uint8_t> scales(scale_bytes);
  CHECK_CUDA(cudaMemcpy(quantized.data(), device_quantized, quantized_bytes,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(scales.data(), device_scales, scale_bytes,
                        cudaMemcpyDeviceToHost));

  std::size_t value_mismatches = 0;
  for (std::size_t i = 0; i < quantized.size(); ++i) {
    if (quantized[i] != reference.quantized[i]) ++value_mismatches;
  }
  std::size_t scale_mismatches = 0;
  for (std::size_t i = 0; i < scales.size(); ++i) {
    if (scales[i] != reference.block_scales[i]) ++scale_mismatches;
  }

  double rmse = 0.0;
  float max_abs_error = 0.0f;
  compute_error(input, quantized, scales, reference.tensor_scale,
                logical_elements, &rmse, &max_abs_error);

  const float average_ms = total_ms / options.iterations;
  const double gelements_per_second =
      static_cast<double>(padded_elements) / (average_ms * 1.0e6);
  const double bytes_per_element =
      sizeof(float) + 0.5 + 1.0 / kNvfp4BlockSize;
  const double effective_gbps = gelements_per_second * bytes_per_element;

  std::cout << std::fixed << std::setprecision(6)
            << "distribution=" << options.distribution
            << " shape=" << options.rows << 'x' << options.cols
            << " padded_elements=" << padded_elements
            << " tensor_scale=" << reference.tensor_scale << '\n'
            << "average_ms=" << average_ms
            << " gelements_per_second=" << gelements_per_second
            << " effective_gbps=" << effective_gbps << '\n'
            << "value_mismatches=" << value_mismatches
            << " scale_mismatches=" << scale_mismatches
            << " rmse=" << rmse
            << " max_abs_error=" << max_abs_error << '\n';

  append_csv(options, logical_elements, padded_elements, average_ms,
             gelements_per_second, effective_gbps, value_mismatches,
             scale_mismatches, rmse, max_abs_error);

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_quantized));
  CHECK_CUDA(cudaFree(device_scales));
  return value_mismatches == 0 && scale_mismatches == 0 ? EXIT_SUCCESS
                                                        : EXIT_FAILURE;
}
