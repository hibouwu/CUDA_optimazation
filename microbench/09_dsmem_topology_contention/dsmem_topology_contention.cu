#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err__));                             \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

namespace cg = cooperative_groups;

static constexpr int kUnroll = 8;
static constexpr int kBytesPerOp = 16;

enum class Mode {
  kRingReadD1,
  kRingReadD2,
  kFaninReadRoot0,
  kRingWriteD1,
  kRingWriteD2,
  kFaninWriteRoot0,
};

__device__ __forceinline__ uint4 make_pattern(uint32_t base, uint32_t i) {
  return make_uint4(0x9e3779b9u ^ base ^ i,
                    0x7f4a7c15u + base * 3u + i,
                    0x94d049bbu ^ (base << 1) ^ (i * 17u),
                    0x2545f491u + base * 17u + i * 13u);
}

__device__ __forceinline__ uint4 add_mix(uint4 a, uint4 b, uint32_t salt) {
  a.x ^= b.x + salt;
  a.y += b.y ^ (salt * 3u + 1u);
  a.z ^= b.z + a.x;
  a.w += b.w ^ a.y;
  return a;
}

__device__ __forceinline__ uint4 ld_u128(uint4* ptr) {
  uint4 v = *ptr;
  asm volatile("" : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) :: "memory");
  return v;
}

__device__ __forceinline__ void st_u128(uint4* ptr, uint4 v) {
  *ptr = v;
  asm volatile("" ::: "memory");
}

template <Mode kMode>
__global__ __launch_bounds__(256, 1)
void dsmem_topology_kernel(int iters,
                           size_t element_mask,
                           uint32_t* sink,
                           unsigned long long* cycles_out) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* smem = reinterpret_cast<uint4*>(smem_raw);
  auto cluster = cg::this_cluster();
  const unsigned tid = threadIdx.x;
  const unsigned rank = cluster.block_rank();
  const unsigned cluster_blocks = cluster.num_blocks();
  const unsigned block_linear = blockIdx.x;
  const size_t element_count = element_mask + 1u;

  for (size_t idx = tid; idx < element_count; idx += blockDim.x) {
    smem[idx] = make_pattern(block_linear * 65537u + uint32_t(idx), 0u);
  }
  __syncthreads();
  cluster.sync();

  constexpr bool is_read =
      kMode == Mode::kRingReadD1 || kMode == Mode::kRingReadD2 ||
      kMode == Mode::kFaninReadRoot0;
  constexpr bool is_fanin =
      kMode == Mode::kFaninReadRoot0 || kMode == Mode::kFaninWriteRoot0;
  constexpr int distance =
      (kMode == Mode::kRingReadD2 || kMode == Mode::kRingWriteD2) ? 2 : 1;

  const bool active = !is_fanin || rank != 0;
  const unsigned target_rank = is_fanin ? 0u : ((rank + distance) % cluster_blocks);
  auto* target = reinterpret_cast<uint4*>(cluster.map_shared_rank(smem, int(target_rank)));

  const size_t step = size_t(blockDim.x);
  const size_t source_partition = is_fanin ? (size_t(rank) * 257u) : 0u;
  const size_t round_stride = step * size_t(kUnroll) + 1u + source_partition;
  size_t idx0 = (size_t(tid) + step * 0u + source_partition) & element_mask;
  size_t idx1 = (size_t(tid) + step * 1u + source_partition) & element_mask;
  size_t idx2 = (size_t(tid) + step * 2u + source_partition) & element_mask;
  size_t idx3 = (size_t(tid) + step * 3u + source_partition) & element_mask;
  size_t idx4 = (size_t(tid) + step * 4u + source_partition) & element_mask;
  size_t idx5 = (size_t(tid) + step * 5u + source_partition) & element_mask;
  size_t idx6 = (size_t(tid) + step * 6u + source_partition) & element_mask;
  size_t idx7 = (size_t(tid) + step * 7u + source_partition) & element_mask;

  uint4 acc = make_pattern(block_linear * blockDim.x + tid, 1u);
  __shared__ unsigned long long block_start;

  __syncthreads();
  cluster.sync();
  if (tid == 0) block_start = clock64();
  __syncthreads();

  if (active) {
    if constexpr (is_read) {
      for (int i = 0; i < iters; ++i) {
        uint4 v0 = ld_u128(target + idx0);
        uint4 v1 = ld_u128(target + idx1);
        uint4 v2 = ld_u128(target + idx2);
        uint4 v3 = ld_u128(target + idx3);
        uint4 v4 = ld_u128(target + idx4);
        uint4 v5 = ld_u128(target + idx5);
        uint4 v6 = ld_u128(target + idx6);
        uint4 v7 = ld_u128(target + idx7);
        acc = add_mix(acc, v0, uint32_t(i));
        acc = add_mix(acc, v1, uint32_t(i) + 1u);
        acc = add_mix(acc, v2, uint32_t(i) + 2u);
        acc = add_mix(acc, v3, uint32_t(i) + 3u);
        acc = add_mix(acc, v4, uint32_t(i) + 4u);
        acc = add_mix(acc, v5, uint32_t(i) + 5u);
        acc = add_mix(acc, v6, uint32_t(i) + 6u);
        acc = add_mix(acc, v7, uint32_t(i) + 7u);
        idx0 = (idx0 + round_stride) & element_mask;
        idx1 = (idx1 + round_stride) & element_mask;
        idx2 = (idx2 + round_stride) & element_mask;
        idx3 = (idx3 + round_stride) & element_mask;
        idx4 = (idx4 + round_stride) & element_mask;
        idx5 = (idx5 + round_stride) & element_mask;
        idx6 = (idx6 + round_stride) & element_mask;
        idx7 = (idx7 + round_stride) & element_mask;
      }
    } else {
      for (int i = 0; i < iters; ++i) {
        st_u128(target + idx0, make_pattern(acc.x + uint32_t(idx0), uint32_t(i)));
        st_u128(target + idx1, make_pattern(acc.y + uint32_t(idx1), uint32_t(i) + 1u));
        st_u128(target + idx2, make_pattern(acc.z + uint32_t(idx2), uint32_t(i) + 2u));
        st_u128(target + idx3, make_pattern(acc.w + uint32_t(idx3), uint32_t(i) + 3u));
        st_u128(target + idx4, make_pattern(acc.x + uint32_t(idx4), uint32_t(i) + 4u));
        st_u128(target + idx5, make_pattern(acc.y + uint32_t(idx5), uint32_t(i) + 5u));
        st_u128(target + idx6, make_pattern(acc.z + uint32_t(idx6), uint32_t(i) + 6u));
        st_u128(target + idx7, make_pattern(acc.w + uint32_t(idx7), uint32_t(i) + 7u));
        acc.x += 0x9e3779b9u + uint32_t(i);
        acc.y ^= acc.x + uint32_t(tid);
        acc.z += acc.y ^ uint32_t(idx0 + idx7);
        acc.w ^= acc.z + uint32_t(i);
        idx0 = (idx0 + round_stride) & element_mask;
        idx1 = (idx1 + round_stride) & element_mask;
        idx2 = (idx2 + round_stride) & element_mask;
        idx3 = (idx3 + round_stride) & element_mask;
        idx4 = (idx4 + round_stride) & element_mask;
        idx5 = (idx5 + round_stride) & element_mask;
        idx6 = (idx6 + round_stride) & element_mask;
        idx7 = (idx7 + round_stride) & element_mask;
      }
    }
  }

  cluster.sync();
  if (tid == 0) cycles_out[blockIdx.x] = active ? (clock64() - block_start) : 0;

  if (active) {
    sink[block_linear * blockDim.x + tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
  }
}

static size_t floor_power_of_two(size_t x) {
  if (x == 0) return 0;
  size_t p = 1;
  while (p <= x / 2) p <<= 1;
  return p;
}

static const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::kRingReadD1: return "ring-read-d1";
    case Mode::kRingReadD2: return "ring-read-d2";
    case Mode::kFaninReadRoot0: return "fanin-read-root0";
    case Mode::kRingWriteD1: return "ring-write-d1";
    case Mode::kRingWriteD2: return "ring-write-d2";
    case Mode::kFaninWriteRoot0: return "fanin-write-root0";
  }
  return "unknown";
}

static Mode parse_mode(const char* text) {
  if (std::strcmp(text, "ring-read-d1") == 0) return Mode::kRingReadD1;
  if (std::strcmp(text, "ring-read-d2") == 0) return Mode::kRingReadD2;
  if (std::strcmp(text, "fanin-read-root0") == 0) return Mode::kFaninReadRoot0;
  if (std::strcmp(text, "ring-write-d1") == 0) return Mode::kRingWriteD1;
  if (std::strcmp(text, "ring-write-d2") == 0) return Mode::kRingWriteD2;
  if (std::strcmp(text, "fanin-write-root0") == 0) return Mode::kFaninWriteRoot0;
  std::fprintf(stderr, "unknown mode: %s\n", text);
  std::exit(2);
}

struct Options {
  Mode mode = Mode::kRingReadD1;
  int iters = 4096;
  int warmup_iters = 64;
  int threads = 256;
  int cluster_size = 4;
  int clusters = 0;
  size_t shared_bytes = 65536;
  bool allow_waves = false;
  bool csv = false;
  bool csv_header = false;
};

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      opt.mode = parse_mode(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      opt.iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup-iters") == 0 && i + 1 < argc) {
      opt.warmup_iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      opt.threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--cluster-size") == 0 && i + 1 < argc) {
      opt.cluster_size = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--clusters") == 0 && i + 1 < argc) {
      opt.clusters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--shared-bytes") == 0 && i + 1 < argc) {
      opt.shared_bytes = std::strtoull(argv[++i], nullptr, 0);
    } else if (std::strcmp(argv[i], "--allow-waves") == 0) {
      opt.allow_waves = true;
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      opt.csv = true;
    } else if (std::strcmp(argv[i], "--csv-header") == 0) {
      opt.csv_header = true;
    } else {
      std::fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
      std::exit(2);
    }
  }
  return opt;
}

using KernelFn = void (*)(int, size_t, uint32_t*, unsigned long long*);

static KernelFn kernel_for_mode(Mode mode) {
  switch (mode) {
    case Mode::kRingReadD1: return dsmem_topology_kernel<Mode::kRingReadD1>;
    case Mode::kRingReadD2: return dsmem_topology_kernel<Mode::kRingReadD2>;
    case Mode::kFaninReadRoot0: return dsmem_topology_kernel<Mode::kFaninReadRoot0>;
    case Mode::kRingWriteD1: return dsmem_topology_kernel<Mode::kRingWriteD1>;
    case Mode::kRingWriteD2: return dsmem_topology_kernel<Mode::kRingWriteD2>;
    case Mode::kFaninWriteRoot0: return dsmem_topology_kernel<Mode::kFaninWriteRoot0>;
  }
  return nullptr;
}

static bool is_fanin(Mode mode) {
  return mode == Mode::kFaninReadRoot0 || mode == Mode::kFaninWriteRoot0;
}

static void configure_kernel(KernelFn kernel, size_t shared_bytes) {
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  int(shared_bytes)));
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
}

static cudaLaunchConfig_t make_launch_config(dim3 grid,
                                             dim3 block,
                                             int cluster_size,
                                             size_t shared_bytes,
                                             cudaLaunchAttribute* attrs) {
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shared_bytes;
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = cluster_size;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  return config;
}

static void launch_kernel(KernelFn kernel,
                          int iters,
                          size_t element_mask,
                          uint32_t* sink,
                          unsigned long long* cycles,
                          int clusters,
                          int cluster_size,
                          int threads,
                          size_t shared_bytes) {
  cudaLaunchAttribute attrs[1]{};
  cudaLaunchConfig_t config = make_launch_config(dim3(clusters * cluster_size),
                                                 dim3(threads),
                                                 cluster_size,
                                                 shared_bytes,
                                                 attrs);
  CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, iters, element_mask, sink, cycles));
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  if (opt.csv_header) {
    std::puts("mode,requested_bytes,elapsed_cycles,bytes_per_cycle,per_active_block_bytes_per_cycle,sm_count,launched_blocks,active_remote_blocks,clusters,cluster_size,resident_cluster_limit,cuda_max_active_clusters,threads,iters,unroll,shared_bytes,working_set_elements,occupancy_limited");
    return 0;
  }
  if (opt.iters <= 0 || opt.warmup_iters < 0 || opt.threads <= 0 ||
      opt.threads > 256 || opt.cluster_size < 2 || opt.cluster_size > 8) {
    std::fprintf(stderr, "invalid iteration/thread/cluster configuration\n");
    return 2;
  }
  if ((opt.mode == Mode::kRingReadD2 || opt.mode == Mode::kRingWriteD2) &&
      opt.cluster_size < 3) {
    std::fprintf(stderr, "d2 modes require cluster-size >= 3\n");
    return 2;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  size_t shared_bytes = floor_power_of_two(opt.shared_bytes);
  shared_bytes = std::max(shared_bytes, size_t(opt.threads * kUnroll * kBytesPerOp));
  shared_bytes = floor_power_of_two(shared_bytes);
  if (shared_bytes > size_t(prop.sharedMemPerBlockOptin)) {
    std::fprintf(stderr, "requested shared bytes %zu exceeds opt-in limit %zu\n",
                 shared_bytes, size_t(prop.sharedMemPerBlockOptin));
    return 2;
  }
  const size_t element_count = shared_bytes / sizeof(uint4);
  if (element_count == 0 || (element_count & (element_count - 1)) != 0) {
    std::fprintf(stderr, "shared working set must contain power-of-two uint4 elements\n");
    return 2;
  }

  KernelFn kernel = kernel_for_mode(opt.mode);
  configure_kernel(kernel, shared_bytes);

  cudaLaunchAttribute attrs[1]{};
  cudaLaunchConfig_t occ_config = make_launch_config(dim3(opt.cluster_size),
                                                     dim3(opt.threads),
                                                     opt.cluster_size,
                                                     shared_bytes,
                                                     attrs);
  int cuda_max_active_clusters = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveClusters(&cuda_max_active_clusters, kernel, &occ_config));
  if (cuda_max_active_clusters <= 0) {
    std::fprintf(stderr, "cudaOccupancyMaxActiveClusters returned %d\n", cuda_max_active_clusters);
    return 1;
  }

  const int resident_cluster_limit = std::max(1, prop.multiProcessorCount / opt.cluster_size);
  const int clusters = opt.clusters > 0 ? opt.clusters : resident_cluster_limit;
  const bool occupancy_limited = clusters > resident_cluster_limit;
  if (occupancy_limited && !opt.allow_waves) {
    std::fprintf(stderr, "requested clusters=%d exceeds resident cluster limit=%d\n",
                 clusters, resident_cluster_limit);
    return 2;
  }

  const int launched_blocks = clusters * opt.cluster_size;
  const int active_remote_blocks = clusters * (is_fanin(opt.mode) ? (opt.cluster_size - 1) : opt.cluster_size);
  uint32_t* d_sink = nullptr;
  unsigned long long* d_cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, size_t(launched_blocks) * opt.threads * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_cycles, size_t(launched_blocks) * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_sink, 0, size_t(launched_blocks) * opt.threads * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_cycles, 0, size_t(launched_blocks) * sizeof(unsigned long long)));

  if (opt.warmup_iters > 0) {
    launch_kernel(kernel, opt.warmup_iters, element_count - 1u, d_sink, d_cycles,
                  clusters, opt.cluster_size, opt.threads, shared_bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  launch_kernel(kernel, opt.iters, element_count - 1u, d_sink, d_cycles,
                clusters, opt.cluster_size, opt.threads, shared_bytes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> cycles;
  cycles.resize(static_cast<size_t>(launched_blocks));
  CUDA_CHECK(cudaMemcpy(cycles.data(), d_cycles, cycles.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  unsigned long long elapsed_cycles = 0;
  for (auto c : cycles) elapsed_cycles = std::max(elapsed_cycles, c);

  const unsigned long long ops =
      static_cast<unsigned long long>(active_remote_blocks) *
      static_cast<unsigned long long>(opt.threads) *
      static_cast<unsigned long long>(opt.iters) *
      static_cast<unsigned long long>(kUnroll);
  const unsigned long long requested_bytes = ops * kBytesPerOp;
  const double bpc = elapsed_cycles ? double(requested_bytes) / double(elapsed_cycles) : 0.0;
  const double per_active_block = active_remote_blocks ? bpc / double(active_remote_blocks) : 0.0;

  if (opt.csv) {
    std::printf("%s,%llu,%llu,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%d\n",
                mode_name(opt.mode), requested_bytes, elapsed_cycles, bpc, per_active_block,
                prop.multiProcessorCount, launched_blocks, active_remote_blocks, clusters,
                opt.cluster_size, resident_cluster_limit, cuda_max_active_clusters,
                opt.threads, opt.iters, kUnroll, shared_bytes, element_count,
                occupancy_limited ? 1 : 0);
  } else {
    std::printf("mode=%s requested_bytes=%llu elapsed_cycles=%llu bytes_per_cycle=%.6f active_remote_blocks=%d clusters=%d cluster_size=%d\n",
                mode_name(opt.mode), requested_bytes, elapsed_cycles, bpc,
                active_remote_blocks, clusters, opt.cluster_size);
  }

  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cycles));
  return 0;
}
