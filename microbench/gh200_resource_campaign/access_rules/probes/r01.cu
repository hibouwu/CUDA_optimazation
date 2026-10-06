// R01: dependency chains (FFMA, 32-bit add, SMEM/global address chains, ldmatrix, WGMMA).
// Every timed loop iteration holds kUnroll dependent steps per chain, so the loop counter,
// compare and backward branch are paid once per kUnroll steps instead of once per step.
#include "r01_support.hpp"

#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>

#ifndef R01_UNROLL
#define R01_UNROLL 64
#endif
constexpr int kUnroll = R01_UNROLL;

namespace G = cute::SM90::GMMA;

enum ScalarOp { kFfma = 0, kAdd = 1, kShared = 2, kGlobalSmall = 3, kGlobalLarge = 4 };
constexpr unsigned kRingSeed = 20261006;
constexpr int kSharedWords = 4096;  // 16 KiB SMEM ring

struct ROptions {
  std::string op = "ffma";
  int steps = 128, warps = 1, streams = 1, n = 256, control = 0;

  ROptions(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("missing value");
      std::string k = argv[i], v = argv[i + 1];
      if (k == "--op") op = v;
      else if (k == "--steps") steps = std::stoi(v);
      else if (k == "--warps") warps = std::stoi(v);
      else if (k == "--streams") streams = std::stoi(v);
      else if (k == "--n") n = std::stoi(v);
      else if (k == "--control") control = std::stoi(v);
      else throw std::runtime_error("unknown option " + k);
    }
    // control 0: chain + consumer; 1: empty window; 2: consumer of preloaded values only.
    if (steps < kUnroll || steps % kUnroll || (warps != 1 && warps != 4) ||
        (streams != 1 && streams != 4) || control < 0 || control > 2)
      throw std::runtime_error("invalid configuration (steps must be a multiple of unroll)");
  }
  int iterations() const { return steps / kUnroll; }
};

// ---------------------------------------------------------------------------------------
// R01-A/B: scalar chains. Memory chains follow a random ring of indices; the address
// conversion (index -> address) is part of every step.
template <int Op>
__device__ __forceinline__ void scalar_step(float& f, unsigned& x, unsigned& prev,
                                            const unsigned* ring, unsigned shared_base) {
  if constexpr (Op == kFfma) {
    asm volatile("fma.rn.f32 %0,%0,0f3f800000,0f3a800000;" : "+f"(f));
  } else if constexpr (Op == kAdd) {
    // x_{n+1} = x_n + x_{n-1} (mod 2^32). With a loop-invariant addend ptxas merges pairs
    // of adds into one IMAD/LEA; here every intermediate is read twice, so it cannot.
    unsigned next;
    asm volatile("add.u32 %0,%1,%2;" : "=r"(next) : "r"(x), "r"(prev));
    prev = x;
    x = next;
  } else if constexpr (Op == kShared) {
    asm volatile("ld.shared.u32 %0,[%1];" : "=r"(x) : "r"(shared_base + 4 * x) : "memory");
  } else {
    asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(x) : "l"(ring + x) : "memory");
  }
}

template <int Op, int Streams, bool Single>
__global__ void scalar_chain(const unsigned* ring, unsigned words, unsigned add_prev,
                             int iterations, int control, RStamp* stamps, float* result) {
  __shared__ unsigned shared_ring[kSharedWords];
  __shared__ volatile float drain[128];
  if constexpr (Op == kShared) {
    for (int i = threadIdx.x; i < kSharedWords; i += blockDim.x) shared_ring[i] = ring[i];
  }
  const unsigned shared_base = static_cast<unsigned>(__cvta_generic_to_shared(shared_ring));
  const bool active = !Single || threadIdx.x == 0;  // single memory chain: lane 0 only
  float f[Streams];
  unsigned x[Streams], prev[Streams];
#pragma unroll
  for (int c = 0; c < Streams; ++c) {
    f[c] = (threadIdx.x + 1) / 1024.f + c / 16.f;
    x[c] = (threadIdx.x * Streams + c) % words;
    prev[c] = add_prev;
    asm volatile("" : "+f"(f[c]), "+r"(x[c]), "+r"(prev[c])::"memory");
  }
  __syncthreads();
  RStamp s{};
  if (threadIdx.x == 0) stamp_begin(s);
  __syncthreads();
  if (control == 0 && active) {
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
#pragma unroll
        for (int c = 0; c < Streams; ++c)
          scalar_step<Op>(f[c], x[c], prev[c], ring, shared_base);
      }
    }
  }
  if (control != 1) {
    float sum = 0;
#pragma unroll
    for (int c = 0; c < Streams; ++c) {
      asm volatile("" : "+f"(f[c]), "+r"(x[c])::"memory");
      sum += Op == kFfma ? f[c] : float(x[c]);
    }
    drain[threadIdx.x] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) stamp_end(s, stamps);
#pragma unroll
  for (int c = 0; c < Streams; ++c) {
    if constexpr (Op == kFfma) result[threadIdx.x * Streams + c] = f[c];
    else reinterpret_cast<unsigned*>(result)[threadIdx.x * Streams + c] = x[c];
  }
}

// Linear .cg read of the whole global ring before each window (not timed).
__global__ void prepare_ring(const unsigned* ring, unsigned words, unsigned* sink) {
  unsigned value = 0;
  for (size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < words;
       i += size_t(gridDim.x) * blockDim.x) {
    unsigned v;
    asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(v) : "l"(ring + i) : "memory");
    value ^= v;
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

// ---------------------------------------------------------------------------------------
// R01-D: ldmatrix.x4 chain over three 512 B blocks. Every even b16 element of block k holds
// the byte offset of block (k+1)%3 and every odd element is 0, so d0 of each lane equals
// the next block offset and is added directly to the lane's row address.
constexpr int kMatrixBlockHalves = 256;  // 32 rows x 16 B

__global__ void matrix_chain(int iterations, int control, RStamp* stamps, float* result) {
  __shared__ __align__(128) unsigned short data[3 * kMatrixBlockHalves];
  __shared__ volatile unsigned drain[32];
  for (int i = threadIdx.x; i < 3 * kMatrixBlockHalves; i += 32) {
    const int block = i / kMatrixBlockHalves;
    data[i] = (i % 2) ? 0 : 512 * ((block + 1) % 3);
  }
  __syncthreads();
  const unsigned row = static_cast<unsigned>(__cvta_generic_to_shared(data)) + threadIdx.x * 16;
  unsigned tile = 0, d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  RStamp s{};
  if (threadIdx.x == 0) stamp_begin(s);
  __syncthreads();
  if (control == 0) {
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                     : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                     : "r"(row + tile)
                     : "memory");
        tile = d0;
      }
    }
  }
  if (control != 1) drain[threadIdx.x] = d0 + d1 + d2 + d3 + tile;
  __syncthreads();
  if (threadIdx.x == 0) stamp_end(s, stamps);
  unsigned* out = reinterpret_cast<unsigned*>(result) + threadIdx.x * 5;
  out[0] = d0;
  out[1] = d1;
  out[2] = d2;
  out[3] = d3;
  out[4] = tile;
}

// ---------------------------------------------------------------------------------------
// R01-C: one warpgroup, one accumulator chain; each step is one SS m64nNk16 WGMMA,
// commit and wait_group 0.
template <class Mma, size_t... I>
__device__ __forceinline__ void wgmma_ss(uint64_t a, uint64_t b, float* d,
                                         std::index_sequence<I...>) {
  Mma::fma(a, b, d[I]..., G::ScaleOut::One);
}

__host__ __device__ inline float wgmma_a(int row, int k) { return (1 + (row + 2 * k) % 7) / 16.f; }
__host__ __device__ inline float wgmma_b(int col, int k) { return (1 + (col + 3 * k) % 11) / 32.f; }

template <class Mma, int N>
__global__ void wgmma_chain(int iterations, int control, RStamp* stamps, float* result) {
  using namespace cute;
  constexpr int StorageK = 64, Registers = N / 2;
  auto la = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<_64, Int<StorageK>>{});
  auto lb = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<Int<N>, Int<StorageK>>{});
  extern __shared__ __align__(128) unsigned char bytes[];
  auto a = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(bytes)), la);
  auto b = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(bytes) + 64 * StorageK), lb);
  for (int i = threadIdx.x; i < 64 * StorageK; i += 128)
    a(i / StorageK, i % StorageK) = __float2half(wgmma_a(i / StorageK, i % StorageK));
  for (int i = threadIdx.x; i < N * StorageK; i += 128)
    b(i / StorageK, i % StorageK) = __float2half(wgmma_b(i / StorageK, i % StorageK));
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  auto av = local_tile(a, make_shape(_64{}, _16{}), make_coord(_0{}, _0{}));
  auto bv = local_tile(b, make_shape(Int<N>{}, _16{}), make_coord(_0{}, _0{}));
  const uint64_t ad = G::make_gmma_desc<G::Major::K>(av);
  const uint64_t bd = G::make_gmma_desc<G::Major::K>(bv);
  float d[Registers];
#pragma unroll
  for (int j = 0; j < Registers; ++j) {
    d[j] = 1 / 64.f;
    asm volatile("" : "+f"(d[j])::"memory");
  }
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ volatile float drain[128];
  RStamp s{};
  __syncthreads();
  if (threadIdx.x == 0) stamp_begin(s);
  __syncthreads();
  if (control == 0) {
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        wgmma_ss<Mma>(ad, bd, d, std::make_index_sequence<Registers>{});
        asm volatile("wgmma.commit_group.sync.aligned;\n\twgmma.wait_group.sync.aligned 0;" ::
                         : "memory");
      }
    }
  }
  if (control != 1) {
    float sum = 0;
#pragma unroll
    for (int j = 0; j < Registers; ++j) {
      asm volatile("" : "+f"(d[j])::"memory");
      sum += d[j];
    }
    drain[threadIdx.x] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) stamp_end(s, stamps);
#pragma unroll
  for (int j = 0; j < Registers; ++j) result[threadIdx.x * Registers + j] = d[j];
}

// ---------------------------------------------------------------------------------------
// Host side. Each process: initialize, 8-30 warmup windows, one formal window, CPU check,
// raw witness files, one JSON line on stdout.
static void common_json(const char* op, const ROptions& o, int streams, int warps) {
  std::cout << std::setprecision(17) << "\"op\":\"" << op << "\",\"steps\":" << o.steps
            << ",\"unroll\":" << kUnroll << ",\"iterations\":" << o.iterations()
            << ",\"warps\":" << warps << ",\"streams\":" << streams
            << ",\"control\":" << o.control;
}

template <int Op, int Streams, bool Single>
int scalar_run(const ROptions& o) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // SMEM ring 16 KiB; global small = L2/4 bytes; global large = 4 x L2 bytes.
  const unsigned words = Op <= kShared ? kSharedWords
                         : Op == kGlobalSmall ? unsigned(prop.l2CacheSize / 16)
                                              : unsigned(prop.l2CacheSize);
  std::vector<unsigned> ring(words);
  if constexpr (Op >= kShared) {
    std::vector<unsigned> order(words);
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937 rng(kRingSeed);
    std::shuffle(order.begin(), order.end(), rng);
    for (unsigned i = 0; i < words; ++i) ring[order[i]] = order[(i + 1) % words];
  }
  const unsigned add_prev = 17;  // add chain x_{-1}; x_0 is the lane start value
  const int threads = o.warps * 32;
  DeviceBuffer<unsigned> device(words), sink(256 * 256);
  CUDA_CHECK(cudaMemcpy(device.pointer, ring.data(), words * 4, cudaMemcpyHostToDevice));
  DeviceBuffer<RStamp> stamp(1);
  DeviceBuffer<float> out(threads * Streams);
  auto kernel = scalar_chain<Op, Streams, Single>;
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  std::vector<RStamp> times(1);
  auto launch = [&]() {
    if constexpr (Op >= kGlobalSmall) {
      prepare_ring<<<256, 256>>>(device.pointer, words, sink.pointer);
      CUDA_CHECK(cudaGetLastError());
    }
    kernel<<<1, threads>>>(device.pointer, words, add_prev, o.iterations(), o.control,
                           stamp.pointer, out.pointer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(times.data(), stamp.pointer, sizeof(RStamp), cudaMemcpyDeviceToHost));
    return elapsed_window(times);
  };
  Windows w = warm_up(launch);
  launch();

  // CPU replay of every lane and chain; memory chains also save the visited edges.
  std::vector<float> values(out.count);
  CUDA_CHECK(cudaMemcpy(values.data(), out.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
  std::map<unsigned, unsigned> edges;
  std::vector<unsigned> starts(values.size());
  bool valid = true;
  for (int t = 0; t < threads; ++t) {
    for (int c = 0; c < Streams; ++c) {
      const int q = t * Streams + c;
      unsigned x = q % words, prev = add_prev;
      starts[q] = x;
      float f = (t + 1) / 1024.f + c / 16.f;
      const bool active = !Single || t == 0;
      if (o.control == 0 && active) {
        for (int i = 0; i < o.steps; ++i) {
          if constexpr (Op == kFfma) {
            f = std::fma(f, 1.f, 1 / 1024.f);
          } else if constexpr (Op == kAdd) {
            const unsigned next = x + prev;
            prev = x;
            x = next;
          } else {
            edges[x] = ring[x];
            x = ring[x];
          }
        }
      }
      if constexpr (Op == kFfma) valid &= values[q] == f;
      else valid &= reinterpret_cast<unsigned*>(values.data())[q] == x;
    }
  }
  std::vector<unsigned> edge_data;
  for (auto e : edges) {
    edge_data.push_back(e.first);
    edge_data.push_back(e.second);
  }
  save_binary("edges.u32", edge_data);
  save_binary("output.bin", values);
  save_binary("starts.u32", starts);

  const uint64_t lanes = Single ? 1 : threads;
  const uint64_t ops = o.control == 0 ? lanes * Streams * o.steps : 0;
  const char* names[] = {"ffma", "add", "shared", "global_small", "global_large"};
  std::cout << "{\"status\":\"" << (valid ? "measured" : "numeric_error") << "\",";
  common_json(names[Op], o, Streams, o.warps);
  std::cout << ",\"active_lanes\":" << lanes << ",\"work_flop\":" << (Op == kFfma ? 2 * ops : 0)
            << ",\"work_op\":" << (Op == kAdd ? ops : 0)
            << ",\"requested_bytes\":" << (Op >= kShared ? 4 * ops : 0)
            << ",\"workset_bytes\":" << (Op >= kShared ? uint64_t(words) * 4 : 0)
            << ",\"l2_bytes\":" << prop.l2CacheSize << ",\"prepare\":\""
            << (Op >= kGlobalSmall ? "full_linear_cg_read_before_each_window"
                                   : "initialized_before_window")
            << "\",\"ring_seed\":" << kRingSeed << ",\"registers_per_thread\":" << attr.numRegs
            << ",\"local_bytes_per_thread\":" << attr.localSizeBytes
            << ",\"static_smem_bytes\":" << attr.sharedSizeBytes
            << ",\"unit\":\"clock64_cycle/CTA\"";
  windows_json(w, times);
  std::cout << "}\n";
  return valid ? 0 : 2;
}

template <class Mma, int N>
int tensor_run(const ROptions& o) {
  auto kernel = wgmma_chain<Mma, N>;
  constexpr int bytes = (64 + N) * 128;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  DeviceBuffer<RStamp> stamp(1);
  DeviceBuffer<float> out(128 * N / 2);
  std::vector<RStamp> times(1);
  auto launch = [&]() {
    kernel<<<1, 128, bytes>>>(o.iterations(), o.control, stamp.pointer, out.pointer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(times.data(), stamp.pointer, sizeof(RStamp), cudaMemcpyDeviceToHost));
    return elapsed_window(times);
  };
  Windows w = warm_up(launch);
  launch();
  std::vector<float> values(out.count);
  CUDA_CHECK(cudaMemcpy(values.data(), out.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
  // m64nNk16 FP32 accumulator fragment: thread t, register j -> (row, col).
  double error = 0;
  for (int t = 0; t < 128; ++t) {
    for (int j = 0; j < N / 2; ++j) {
      const int row = (t / 32) * 16 + (t % 32) / 4 + ((j / 2) % 2) * 8;
      const int col = (t % 4) * 2 + j % 2 + (j / 4) * 8;
      double dot = 0;
      for (int k = 0; k < 16; ++k) dot += double(wgmma_a(row, k)) * wgmma_b(col, k);
      const double ref = 1 / 64. + (o.control == 0 ? o.steps * dot : 0);
      error = std::max(error, std::abs(values[t * N / 2 + j] - ref));
    }
  }
  save_binary("output.bin", values);
  std::cout << "{\"status\":\"" << (error == 0 ? "measured" : "numeric_error") << "\",";
  common_json("wgmma", o, 1, 4);
  std::cout << ",\"n\":" << N
            << ",\"work_flop\":" << (o.control == 0 ? uint64_t(2) * 64 * N * 16 * o.steps : 0)
            << ",\"work_op\":0,\"requested_bytes\":0,\"registers_per_thread\":" << attr.numRegs
            << ",\"local_bytes_per_thread\":" << attr.localSizeBytes
            << ",\"dynamic_smem_bytes\":" << bytes << ",\"max_error\":" << error
            << ",\"unit\":\"clock64_cycle/CTA\"";
  windows_json(w, times);
  std::cout << "}\n";
  return error == 0 ? 0 : 2;
}

int matrix_run(const ROptions& o) {
  DeviceBuffer<RStamp> stamp(1);
  DeviceBuffer<float> out(32 * 5);
  std::vector<RStamp> times(1);
  auto launch = [&]() {
    matrix_chain<<<1, 32>>>(o.iterations(), o.control, stamp.pointer, out.pointer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(times.data(), stamp.pointer, sizeof(RStamp), cudaMemcpyDeviceToHost));
    return elapsed_window(times);
  };
  Windows w = warm_up(launch);
  launch();
  std::vector<unsigned> values(out.count);
  CUDA_CHECK(cudaMemcpy(values.data(), out.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
  // After N steps from offset 0: tile = 512*(N%3), and d0..d3 hold the same value.
  const unsigned steps = o.control == 0 ? o.steps : 0;
  const unsigned tile = 512 * (steps % 3);
  bool valid = true;
  for (int t = 0; t < 32; ++t)
    for (int j = 0; j < 5; ++j) valid &= values[t * 5 + j] == tile;
  save_binary("output.bin", values);
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, matrix_chain));
  std::cout << "{\"status\":\"" << (valid ? "measured" : "numeric_error") << "\",";
  common_json("ldmatrix", o, 1, 1);
  std::cout << ",\"work_flop\":0,\"work_op\":0,\"requested_bytes\":"
            << (o.control == 0 ? 512ull * o.steps : 0)
            << ",\"registers_per_thread\":" << attr.numRegs
            << ",\"local_bytes_per_thread\":" << attr.localSizeBytes
            << ",\"unit\":\"clock64_cycle/CTA\"";
  windows_json(w, times);
  std::cout << "}\n";
  return valid ? 0 : 2;
}

template <int Op>
int dispatch(const ROptions& o) {
  if (o.streams == 4) return scalar_run<Op, 4, false>(o);
  if (o.warps == 1 && Op >= kShared) return scalar_run<Op, 1, true>(o);
  return scalar_run<Op, 1, false>(o);
}

int main(int argc, char** argv) {
  try {
    ROptions o(argc, argv);
    if (o.op == "ffma") return dispatch<kFfma>(o);
    if (o.op == "add") return dispatch<kAdd>(o);
    if (o.op == "shared") return dispatch<kShared>(o);
    if (o.op == "global_small") return dispatch<kGlobalSmall>(o);
    if (o.op == "global_large") return dispatch<kGlobalLarge>(o);
    if (o.op == "ldmatrix") return matrix_run(o);
    if (o.op == "wgmma") {
      using G::Major;
      if (o.n == 64) return tensor_run<G::MMA_64x64x16_F32F16F16_SS<Major::K, Major::K>, 64>(o);
      if (o.n == 128) return tensor_run<G::MMA_64x128x16_F32F16F16_SS<Major::K, Major::K>, 128>(o);
      if (o.n == 256) return tensor_run<G::MMA_64x256x16_F32F16F16_SS<Major::K, Major::K>, 256>(o);
    }
    throw std::runtime_error("outside matrix");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
