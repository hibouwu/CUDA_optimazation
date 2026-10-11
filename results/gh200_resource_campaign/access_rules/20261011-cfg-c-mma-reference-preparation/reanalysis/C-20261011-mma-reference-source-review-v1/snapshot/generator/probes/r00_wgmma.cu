#include "r00_common.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <fstream>
#include <type_traits>
#include <utility>
namespace G = cute::SM90::GMMA;
struct Stamp {
  uint64_t begin_ns, end_ns, begin_cycle, end_cycle;
  unsigned sm;
};
__device__ uint64_t ns_clock() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__device__ unsigned sm_id() {
  unsigned value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}
__host__ __device__ float short_value(int outer, int inner, bool a) {
  return (1 + (outer + (a ? 2 : 3) * inner) % (a ? 7 : 11)) / (a ? 16.f : 32.f);
}
template <class Mma, size_t... I>
__device__ void call_ss(uint64_t a, uint64_t b, float* d, std::index_sequence<I...>) {
  Mma::fma(a, b, d[I]..., G::ScaleOut::One);
}
template <class Mma, size_t... I>
__device__ void call_rs(const unsigned* a, uint64_t b, float* d, std::index_sequence<I...>) {
  Mma::fma(a[0], a[1], a[2], a[3], b, d[I]..., G::ScaleOut::One);
}

template <class T, class Mma, int N, int K, int Chains, bool RS, int Batch, bool BMajorMN = false,
          bool Serialized = false>
__global__ void probe(int iterations, bool coordinate, Stamp* stamps, float* output) {
  using namespace cute;
  constexpr int StorageK = 128 / sizeof(T), Registers = N / 2;
  auto la = tile_to_shape(G::Layout_K_SW128_Atom<T>{}, Shape<_64, Int<StorageK>>{});
  auto lb = [&]() {
    if constexpr (BMajorMN)
      return tile_to_shape(G::Layout_MN_SW128_Atom<T>{}, Shape<Int<N>, Int<StorageK>>{});
    else
      return tile_to_shape(G::Layout_K_SW128_Atom<T>{}, Shape<Int<N>, Int<StorageK>>{});
  }();
  extern __shared__ __align__(128) unsigned char bytes[];
  T* positive = reinterpret_cast<T*>(bytes);
  T* negative = positive + 64 * StorageK;
  T* bp = negative + 64 * StorageK;
  auto a = make_tensor(make_smem_ptr(positive), la);
  auto an = make_tensor(make_smem_ptr(negative), la);
  auto b = make_tensor(make_smem_ptr(bp), lb);
  for (int i = threadIdx.x; i < 64 * StorageK; i += blockDim.x) {
    int row = i / StorageK, k = i % StorageK;
    float value = coordinate ? short_value(row, k, true) : 0.0625f;
    a(row, k) = T(value);
    an(row, k) = T(-value);
  }
  for (int i = threadIdx.x; i < N * StorageK; i += blockDim.x) {
    int column = i / StorageK, k = i % StorageK;
    b(column, k) = T(coordinate ? short_value(column, k, false) : 0.0625f);
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  // Describe the instruction's K extent while retaining 128B padded row stride.
  auto av = local_tile(a, make_shape(_64{}, Int<K>{}), make_coord(_0{}, _0{}));
  auto nv = local_tile(an, make_shape(_64{}, Int<K>{}), make_coord(_0{}, _0{}));
  auto bv = local_tile(b, make_shape(Int<N>{}, Int<K>{}), make_coord(_0{}, _0{}));
  uint64_t ad = G::make_gmma_desc<G::Major::K>(av), nd = G::make_gmma_desc<G::Major::K>(nv),
           bd = G::make_gmma_desc<BMajorMN ? G::Major::MN : G::Major::K>(bv);
  unsigned ar[4] = {};
  unsigned local = threadIdx.x % 128, group = threadIdx.x / 128;
  if constexpr (RS) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      unsigned row = (local / 32) * 16 + (local % 32) / 4 + ((e / 2) % 2) * 8,
               k = (local % 4) * 2 + e % 2 + (e / 4) * 8;
      __half value = __float2half_rn(coordinate ? short_value(row, k, true) : 0.0625f);
      ar[e / 2] |= unsigned(__half_as_ushort(value)) << (16 * (e % 2));
    }
#pragma unroll
    for (int e = 0; e < 4; ++e)
      asm volatile("" : "+r"(ar[e])::"memory");
  }
  float d[Chains][Registers];
#pragma unroll
  for (int c = 0; c < Chains; ++c) {
#pragma unroll
    for (int j = 0; j < Registers; ++j) {
      d[c][j] = coordinate ? (1 + group + c) / 64.f : 0.f;
      asm volatile("" : "+f"(d[c][j])::"memory");
    }
  }
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ uint64_t begin_ns, begin_cycle;
  __shared__ volatile float drain[256];
  __syncthreads();
  if (threadIdx.x == 0) {
    begin_ns = ns_clock();
    begin_cycle = clock64();
  }
  __syncthreads();
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    if constexpr (Serialized)
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int q = 0; q < Batch; ++q) {
#pragma unroll
      for (int c = 0; c < Chains; ++c) {
        if constexpr (RS)
          call_rs<Mma>(ar, bd, d[c], std::make_index_sequence<Registers>{});
        else
          call_ss<Mma>((K == 32 && Batch == 16 && (q % 2)) ? nd : ad, bd, d[c],
                       std::make_index_sequence<Registers>{});
      }
    }
    if constexpr (Serialized)
      asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
    else
      asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 1;" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  float sum = 0;
#pragma unroll
  for (int c = 0; c < Chains; ++c) {
#pragma unroll
    for (int j = 0; j < Registers; ++j) {
      asm volatile("" : "+f"(d[c][j])::"memory");
      sum += d[c][j];
    }
  }
  drain[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t end_cycle = clock64(), end_ns = ns_clock();
    stamps[blockIdx.x] = {begin_ns, end_ns, begin_cycle, end_cycle, sm_id()};
  }
#pragma unroll
  for (int c = 0; c < Chains; ++c) {
#pragma unroll
    for (int j = 0; j < Registers; ++j)
      output[(size_t(blockIdx.x) * blockDim.x + threadIdx.x) * Chains * Registers + c * Registers +
             j] = d[c][j];
  }
}
struct WOptions {
  std::string dtype = "fp16", source = "SS", scope = "one_cta", b_major = "K";
  bool serialized = false;
  int n = 256, groups = 1, iterations = 4096;
  WOptions(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc)
        throw std::runtime_error("missing value");
      std::string key = argv[i], value = argv[i + 1];
      if (key == "--dtype")
        dtype = value;
      else if (key == "--n")
        n = std::stoi(value);
      else if (key == "--groups")
        groups = std::stoi(value);
      else if (key == "--source")
        source = value;
      else if (key == "--scope")
        scope = value;
      else if (key == "--iterations")
        iterations = std::stoi(value);
      else if (key == "--b-major")
        b_major = value;
      else if (key == "--mode") {
        if (value != "batch16_wait1" && value != "single_wait0")
          throw std::runtime_error("--mode must be batch16_wait1 or single_wait0");
        serialized = value == "single_wait0";
      }
      else
        throw std::runtime_error("unknown WGMMA option");
    }
    if ((groups != 1 && groups != 2) || iterations < 1 ||
        (scope != "one_cta" && scope != "all_gpu"))
      throw std::runtime_error("invalid WGMMA configuration");
    if (b_major != "K" && b_major != "MN")
      throw std::runtime_error("--b-major must be K or MN");
  }
};
template <class T, class Mma, int N, int K, int C, bool RS, bool BMajorMN = false,
          bool Serialized = false>
int run_wgmma(const WOptions& o) {
  constexpr int threads_per_group = 128, registers = N / 2;
  int threads = threads_per_group * o.groups;
  constexpr int bytes = (128 + N) * 128;  // Two padded A matrices and one B, 128B per row.
  constexpr int Batch = Serialized ? 1 : 16;
  auto kernel = probe<T, Mma, N, K, C, RS, Batch, BMajorMN, Serialized>,
       short_kernel = probe<T, Mma, N, K, C, RS, 1, BMajorMN, Serialized>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  CUDA_CHECK(
      cudaFuncSetAttribute(short_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int occupancy = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, threads, bytes));
  if (occupancy < 1)
    throw std::runtime_error("zero occupancy");
  int blocks = o.scope == "all_gpu" ? prop.multiProcessorCount * std::min(4, occupancy) : 1;
  DeviceBuffer<Stamp> stamps(blocks);
  DeviceBuffer<float> output(size_t(blocks) * threads * C * registers);
  auto check = [&](int count, int batch, bool coordinate) {
    std::vector<float> values(output.count);
    CUDA_CHECK(
        cudaMemcpy(values.data(), output.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
    double maximum = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      int thread = (i / (C * registers)) % threads, j = i % registers, c = (i / registers) % C,
          local = thread % 128;
      int row = (local / 32) * 16 + (local % 32) / 4 + ((j / 2) % 2) * 8,
          column = (local % 4) * 2 + j % 2 + (j / 4) * 8;
      double reference = 0;
      if (coordinate) {
        for (int k = 0; k < K; ++k)
          reference += double(float(T(short_value(row, k, true)))) *
                       double(float(T(short_value(column, k, false))));
        reference = reference * count * batch + (1 + thread / 128 + c) / 64.0;
      } else if (K != 32)
        reference = double(count) * batch * K / 256.0;
      if (!std::isfinite(values[i]))
        throw std::runtime_error("nonfinite accumulator");
      maximum = std::max(maximum, std::abs(double(values[i]) - reference));
    }
    return maximum;
  };
  for (int count : {1, 2}) {
    short_kernel<<<1, threads, bytes>>>(count, true, stamps.pointer, output.pointer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // Only one CTA was launched; temporarily limit the checked allocation.
    size_t original = output.count;
    output.count = size_t(threads) * C * registers;
    double error = check(count, 1, true);
    output.count = original;
    if (error > 1e-4)
      throw std::runtime_error("coordinate short-check mismatch: " + std::to_string(error));
  }
  auto launch = [&]() {
    kernel<<<blocks, threads, bytes>>>(o.iterations, false, stamps.pointer, output.pointer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  };
  std::vector<Stamp> times(blocks);
  std::vector<float> warmup;
  bool converged = false;
  auto window = [&]() {
    CUDA_CHECK(
        cudaMemcpy(times.data(), stamps.pointer, blocks * sizeof(Stamp), cudaMemcpyDeviceToHost));
    uint64_t begin = UINT64_MAX, end = 0;
    for (auto s : times) {
      begin = std::min(begin, s.begin_ns);
      end = std::max(end, s.end_ns);
    }
    return o.scope == "one_cta" ? times[0].end_cycle - times[0].begin_cycle : end - begin;
  };
  for (int i = 0; i < 30; ++i) {
    launch();
    warmup.push_back(float(window()));
    if (i >= 7) {
      std::vector<float> last(warmup.end() - 5, warmup.end());
      if (coefficient_of_variation(last) <= 0.02) {
        converged = true;
        break;
      }
    }
  }
  launch();
  uint64_t elapsed = window();
  double error = check(o.iterations, Batch, false);
  bool valid = error <= 1e-4;
  std::vector<float> values(output.count);
  CUDA_CHECK(cudaMemcpy(values.data(), output.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
  std::ofstream file("output.f32", std::ios::binary);
  file.write(reinterpret_cast<const char*>(values.data()), values.size() * 4);
  file.close();
  uint64_t work = uint64_t(2) * 64 * N * K * C * Batch * o.groups * o.iterations * blocks;
  std::cout << std::setprecision(17) << "{\"kind\":\"wgmma\",\"status\":\""
            << (valid ? "measured" : "numeric_error") << "\",\"dtype\":\"" << o.dtype
            << "\",\"source\":\"" << o.source << "\",\"scope\":\"" << o.scope << "\",\"n\":" << N
            << ",\"k\":" << K << ",\"chains\":" << C << ",\"groups\":" << o.groups
            << ",\"threads\":" << threads << ",\"blocks\":" << blocks
            << ",\"iterations\":" << o.iterations << ",\"per_chain_batch\":" << Batch
            << ",\"wait\":" << (Serialized ? 0 : 1)
            << ",\"b_major\":\"" << o.b_major << "\""
            << ",\"work_flop\":" << work << ",\"elapsed\":" << elapsed << ",\"unit\":\""
            << (o.scope == "one_cta" ? "clock64_cycle/CTA" : "globaltimer_ns/GPU")
            << "\",\"registers_per_thread\":" << attributes.numRegs
            << ",\"local_bytes_per_thread\":" << attributes.localSizeBytes
            << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes
            << ",\"dynamic_smem_bytes\":" << bytes
            << ",\"occupancy_limit_ctas_per_sm\":" << occupancy << ",\"max_output_error\":" << error
            << ",\"input_profile\":\""
            << (K == 32 ? "uniform_paired_sign_bounded" : "uniform_positive")
            << "\",\"coordinate_checks\":[1,2],\"warmup_converged\":"
            << (converged ? "true" : "false")
            << ",\"output_file\":\"output.f32\",\"output_elements\":" << output.count
            << ",\"stamps\":[";
  for (int i = 0; i < blocks; ++i) {
    if (i)
      std::cout << ',';
    auto s = times[i];
    std::cout << "[" << s.begin_ns << ',' << s.end_ns << ',' << s.begin_cycle << ',' << s.end_cycle
              << ',' << s.sm << ']';
  }
  std::cout << "]}\n";
  return valid ? 0 : 2;
}
int main(int argc, char** argv) {
  try {
    WOptions o(argc, argv);
    if (o.serialized) {
      if (o.dtype != "fp16" || o.source != "SS" || o.n != 256)
        throw std::runtime_error("single_wait0 comparison requires FP16 SS n256");
      if (o.b_major == "MN")
        return run_wgmma<__half, G::MMA_64x256x16_F32F16F16_SS<G::Major::K, G::Major::MN>,
                           256, 16, 1, false, true, true>(o);
      return run_wgmma<__half, G::MMA_64x256x16_F32F16F16_SS<G::Major::K, G::Major::K>,
                         256, 16, 1, false, false, true>(o);
    }
    if (o.b_major == "MN") {
      if (o.dtype != "fp16" || o.source != "SS" || o.n != 256)
        throw std::runtime_error("MN comparison requires FP16 SS n256");
      return run_wgmma<__half, G::MMA_64x256x16_F32F16F16_SS<G::Major::K, G::Major::MN>,
                         256, 16, 1, false, true>(o);
    }
    if (o.dtype == "fp16" && o.source == "SS") {
      if (o.n == 64)
        return run_wgmma<__half, G::MMA_64x64x16_F32F16F16_SS<G::Major::K, G::Major::K>, 64, 16, 4,
                         false>(o);
      if (o.n == 128)
        return run_wgmma<__half, G::MMA_64x128x16_F32F16F16_SS<G::Major::K, G::Major::K>, 128, 16,
                         2, false>(o);
      if (o.n == 256)
        return run_wgmma<__half, G::MMA_64x256x16_F32F16F16_SS<G::Major::K, G::Major::K>, 256, 16,
                         1, false>(o);
    }
    if (o.dtype == "fp16" && o.source == "RS" && o.n == 256)
      return run_wgmma<__half, G::MMA_64x256x16_F32F16F16_RS<G::Major::K, G::Major::K>, 256, 16, 1,
                       true>(o);
    if (o.dtype == "bf16" && o.source == "SS" && o.n == 256)
      return run_wgmma<__nv_bfloat16, G::MMA_64x256x16_F32BF16BF16_SS<G::Major::K, G::Major::K>,
                       256, 16, 1, false>(o);
    if (o.dtype == "fp8" && o.source == "SS" && o.n == 256)
      return run_wgmma<__nv_fp8_e4m3, G::MMA_64x256x32_F32E4M3E4M3_SS_TN<>, 256, 32, 1, false>(o);
    throw std::runtime_error("configuration outside R00-B matrix");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
