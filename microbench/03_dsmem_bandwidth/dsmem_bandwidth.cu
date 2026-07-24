#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                             \
  cudaError_t err__ = (call);                                             \
  if (err__ != cudaSuccess) {                                             \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                 cudaGetErrorString(err__));                              \
    std::exit(1);                                                         \
  }                                                                       \
} while (0)

namespace cg = cooperative_groups;

static constexpr int kUnroll = 8;
static constexpr int kBytesPerOp = 16;

enum class Mode {
  kLocalRead,
  kLocalWrite,
  kRemoteRead,
  kRemoteWrite,
};

__device__ __forceinline__ uint4 add_mix(uint4 a, uint4 b, uint32_t salt) {
  a.x ^= b.x + salt;
  a.y += b.y ^ (salt * 3u + 1u);
  a.z ^= b.z + a.x;
  a.w += b.w ^ a.y;
  return a;
}

__device__ __forceinline__ uint4 make_pattern(uint32_t base, uint32_t i) {
  return make_uint4(0x9e3779b9u ^ base ^ i,
                    0x7f4a7c15u + base * 3u + i,
                    0x94d049bbu ^ (base << 1) ^ (i * 17u),
                    0x2545f491u + base * 17u + i * 13u);
}

__device__ __forceinline__ uint4 ld_shared_u128(uint4* ptr) {
  uint4 v = *ptr;
  asm volatile("" : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) :: "memory");
  return v;
}

__device__ __forceinline__ void st_shared_u128(uint4* ptr, uint4 v) {
  *ptr = v;
  asm volatile("" ::: "memory");
}

template <Mode kMode>
__global__ __launch_bounds__(256, 1)
void dsmem_bandwidth_kernel(int iters,
                            size_t element_mask,
                            uint32_t* sink,
                            unsigned long long* cycles_out) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* smem = reinterpret_cast<uint4*>(smem_raw);
  auto cluster = cg::this_cluster();
  const unsigned int tid = threadIdx.x;
  const unsigned int block_linear = blockIdx.x;
  const unsigned int cluster_blocks = cluster.num_blocks();
  const unsigned int rank = cluster.block_rank();
  const unsigned int peer_rank = (rank + 1u) % cluster_blocks;
  const size_t element_count = element_mask + 1u;

  for (size_t idx = tid; idx < element_count; idx += blockDim.x) {
    smem[idx] = make_pattern(block_linear * 65537u + uint32_t(idx), 0u);
  }

  __syncthreads();
  cluster.sync();

  uint4* target_base = smem;
  if constexpr (kMode == Mode::kRemoteRead || kMode == Mode::kRemoteWrite) {
    target_base = cluster.map_shared_rank(smem, int(peer_rank));
  }
  auto* target = reinterpret_cast<uint4*>(target_base);

  const size_t step = size_t(blockDim.x);
  const size_t round_stride = step * size_t(kUnroll) + 1u;
  size_t idx0 = (size_t(tid) + step * 0u) & element_mask;
  size_t idx1 = (size_t(tid) + step * 1u) & element_mask;
  size_t idx2 = (size_t(tid) + step * 2u) & element_mask;
  size_t idx3 = (size_t(tid) + step * 3u) & element_mask;
  size_t idx4 = (size_t(tid) + step * 4u) & element_mask;
  size_t idx5 = (size_t(tid) + step * 5u) & element_mask;
  size_t idx6 = (size_t(tid) + step * 6u) & element_mask;
  size_t idx7 = (size_t(tid) + step * 7u) & element_mask;

  uint4 acc = make_pattern(block_linear * blockDim.x + tid, 1u);
  __shared__ unsigned long long block_start;

  __syncthreads();
  cluster.sync();
  if (tid == 0) {
    block_start = clock64();
  }
  __syncthreads();

  if constexpr (kMode == Mode::kLocalRead || kMode == Mode::kRemoteRead) {
    for (int i = 0; i < iters; ++i) {
      uint4 v0 = ld_shared_u128(target + idx0);
      uint4 v1 = ld_shared_u128(target + idx1);
      uint4 v2 = ld_shared_u128(target + idx2);
      uint4 v3 = ld_shared_u128(target + idx3);
      uint4 v4 = ld_shared_u128(target + idx4);
      uint4 v5 = ld_shared_u128(target + idx5);
      uint4 v6 = ld_shared_u128(target + idx6);
      uint4 v7 = ld_shared_u128(target + idx7);
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
    __syncthreads();
  } else {
    for (int i = 0; i < iters; ++i) {
      st_shared_u128(target + idx0, make_pattern(acc.x + uint32_t(idx0), uint32_t(i)));
      st_shared_u128(target + idx1, make_pattern(acc.y + uint32_t(idx1), uint32_t(i) + 1u));
      st_shared_u128(target + idx2, make_pattern(acc.z + uint32_t(idx2), uint32_t(i) + 2u));
      st_shared_u128(target + idx3, make_pattern(acc.w + uint32_t(idx3), uint32_t(i) + 3u));
      st_shared_u128(target + idx4, make_pattern(acc.x + uint32_t(idx4), uint32_t(i) + 4u));
      st_shared_u128(target + idx5, make_pattern(acc.y + uint32_t(idx5), uint32_t(i) + 5u));
      st_shared_u128(target + idx6, make_pattern(acc.z + uint32_t(idx6), uint32_t(i) + 6u));
      st_shared_u128(target + idx7, make_pattern(acc.w + uint32_t(idx7), uint32_t(i) + 7u));
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
    if constexpr (kMode == Mode::kRemoteWrite) {
      cluster.sync();
    } else {
      __syncthreads();
    }
  }

  if (tid == 0) {
    unsigned long long stop = clock64();
    cycles_out[blockIdx.x] = stop - block_start;
  }

  if constexpr (kMode == Mode::kLocalRead || kMode == Mode::kRemoteRead) {
    sink[block_linear * blockDim.x + tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
  } else {
    uint4 v = smem[tid & element_mask];
    sink[block_linear * blockDim.x + tid] =
        v.x ^ v.y ^ v.z ^ v.w ^ acc.x ^ acc.y ^ acc.z ^ acc.w;
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
    case Mode::kLocalRead: return "local-read";
    case Mode::kLocalWrite: return "local-write";
    case Mode::kRemoteRead: return "remote-read";
    case Mode::kRemoteWrite: return "remote-write";
  }
  return "unknown";
}

static Mode parse_mode(const char* text) {
  if (std::strcmp(text, "local-read") == 0) return Mode::kLocalRead;
  if (std::strcmp(text, "local-write") == 0) return Mode::kLocalWrite;
  if (std::strcmp(text, "remote-read") == 0) return Mode::kRemoteRead;
  if (std::strcmp(text, "remote-write") == 0) return Mode::kRemoteWrite;
  std::fprintf(stderr, "unknown mode: %s\n", text);
  std::exit(2);
}

struct Options {
  Mode mode = Mode::kRemoteRead;
  int iters = 4096;
  int warmup_iters = 64;
  int threads = 256;
  int cluster_size = 2;
  int clusters = 0;
  size_t shared_bytes = 32768;
  bool allow_waves = false;
  bool csv = false;
  bool csv_header = false;
};

static void usage(const char* argv0) {
  std::printf(
      "Usage:\n"
      "  %s [--mode local-read|local-write|remote-read|remote-write] [options]\n"
      "\n"
      "Options:\n"
      "  --iters N             Timed outer-loop iterations (default: 4096)\n"
      "  --warmup-iters N      Untimed warmup iterations (default: 64)\n"
      "  --threads N           Threads per block, max 256 (default: 256)\n"
      "  --cluster-size N      Thread-block cluster size (default: 2)\n"
      "  --clusters N          Clusters to launch, 0 = max resident clusters\n"
      "  --shared-bytes N      Dynamic shared-memory bytes per CTA, power-of-two rounded\n"
      "  --allow-waves         Allow launching more than max resident clusters\n"
      "  --csv                 Print one CSV data row\n"
      "  --csv-header          Print the CSV header and exit\n",
      argv0);
}

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
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(2);
    }
  }
  return opt;
}

using KernelFn = void (*)(int, size_t, uint32_t*, unsigned long long*);

static KernelFn kernel_for_mode(Mode mode) {
  switch (mode) {
    case Mode::kLocalRead: return dsmem_bandwidth_kernel<Mode::kLocalRead>;
    case Mode::kLocalWrite: return dsmem_bandwidth_kernel<Mode::kLocalWrite>;
    case Mode::kRemoteRead: return dsmem_bandwidth_kernel<Mode::kRemoteRead>;
    case Mode::kRemoteWrite: return dsmem_bandwidth_kernel<Mode::kRemoteWrite>;
  }
  return nullptr;
}

static void configure_kernel(KernelFn kernel, size_t shared_bytes) {
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  int(shared_bytes)));
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeNonPortableClusterSizeAllowed,
                                  1));
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
    std::puts("mode,requested_bytes,elapsed_cycles,bytes_per_cycle,per_active_block_bytes_per_cycle,sm_count,active_blocks,clusters,cluster_size,resident_cluster_limit,cuda_max_active_clusters,threads,iters,unroll,shared_bytes,working_set_elements,index_stride_elements,stream_period_iters,occupancy_limited");
    return 0;
  }

  if (opt.iters <= 0 || opt.warmup_iters < 0 || opt.threads <= 0 ||
      opt.threads > 256 || opt.cluster_size <= 0 || opt.cluster_size > 8) {
    std::fprintf(stderr, "invalid iteration/thread/cluster configuration\n");
    return 2;
  }
  if ((opt.mode == Mode::kRemoteRead || opt.mode == Mode::kRemoteWrite) &&
      opt.cluster_size < 2) {
    std::fprintf(stderr, "remote modes require --cluster-size >= 2\n");
    return 2;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  size_t shared_bytes = floor_power_of_two(opt.shared_bytes);
  shared_bytes = std::max(shared_bytes, size_t(opt.threads * kUnroll * kBytesPerOp));
  shared_bytes = floor_power_of_two(shared_bytes);
  size_t element_count = shared_bytes / sizeof(uint4);
  if (element_count == 0 || (element_count & (element_count - 1)) != 0) {
    std::fprintf(stderr, "shared working set must contain a power-of-two number of uint4 elements\n");
    return 2;
  }
  if (shared_bytes > size_t(prop.sharedMemPerBlockOptin)) {
    std::fprintf(stderr, "requested shared bytes %zu exceeds opt-in limit %zu\n",
                 shared_bytes, size_t(prop.sharedMemPerBlockOptin));
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

  int resident_cluster_limit = std::max(1, prop.multiProcessorCount / opt.cluster_size);
  int clusters = opt.clusters > 0 ? opt.clusters : resident_cluster_limit;
  bool occupancy_limited = clusters > resident_cluster_limit;
  if (occupancy_limited && !opt.allow_waves) {
    std::fprintf(stderr,
                 "requested clusters=%d exceeds one-CTA-per-SM resident cluster limit=%d; use --allow-waves to override\n",
                 clusters, resident_cluster_limit);
    return 2;
  }

  const int active_blocks = clusters * opt.cluster_size;
  const size_t sink_count = size_t(active_blocks) * size_t(opt.threads);
  uint32_t* d_sink = nullptr;
  unsigned long long* d_cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, sink_count * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_cycles, size_t(active_blocks) * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_sink, 0, sink_count * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_cycles, 0, size_t(active_blocks) * sizeof(unsigned long long)));

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

  std::vector<unsigned long long> cycles(static_cast<size_t>(active_blocks));
  CUDA_CHECK(cudaMemcpy(cycles.data(), d_cycles,
                        cycles.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  unsigned long long elapsed_cycles = 0;
  for (unsigned long long c : cycles) {
    elapsed_cycles = std::max(elapsed_cycles, c);
  }

  const unsigned long long ops =
      static_cast<unsigned long long>(active_blocks) *
      static_cast<unsigned long long>(opt.threads) *
      static_cast<unsigned long long>(opt.iters) *
      static_cast<unsigned long long>(kUnroll);
  const unsigned long long requested_bytes = ops * kBytesPerOp;
  const double bytes_per_cycle = double(requested_bytes) / double(elapsed_cycles);
  const double per_active_block = bytes_per_cycle / double(active_blocks);
  const size_t index_stride_elements = size_t(opt.threads) * size_t(kUnroll) + 1u;

  size_t stream_period_iters = element_count;
  size_t stride_mod = index_stride_elements & (element_count - 1u);
  if (stride_mod != 0) {
    size_t a = element_count;
    size_t b = stride_mod;
    while (b != 0) {
      size_t t = a % b;
      a = b;
      b = t;
    }
    stream_period_iters = element_count / a;
  }

  if (opt.csv) {
    std::printf("%s,%llu,%llu,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%zu,%zu,%d\n",
                mode_name(opt.mode),
                requested_bytes,
                elapsed_cycles,
                bytes_per_cycle,
                per_active_block,
                prop.multiProcessorCount,
                active_blocks,
                clusters,
                opt.cluster_size,
                resident_cluster_limit,
                cuda_max_active_clusters,
                opt.threads,
                opt.iters,
                kUnroll,
                shared_bytes,
                element_count,
                index_stride_elements,
                stream_period_iters,
                occupancy_limited ? 1 : 0);
  } else {
    std::printf("mode=%s requested_bytes=%llu elapsed_cycles=%llu bytes_per_cycle=%.6f per_active_block_bytes_per_cycle=%.6f sm_count=%d active_blocks=%d clusters=%d cluster_size=%d resident_cluster_limit=%d cuda_max_active_clusters=%d threads=%d iters=%d shared_bytes=%zu working_set_elements=%zu occupancy_limited=%d\n",
                mode_name(opt.mode),
                requested_bytes,
                elapsed_cycles,
                bytes_per_cycle,
                per_active_block,
                prop.multiProcessorCount,
                active_blocks,
                clusters,
                opt.cluster_size,
                resident_cluster_limit,
                cuda_max_active_clusters,
                opt.threads,
                opt.iters,
                shared_bytes,
                element_count,
                occupancy_limited ? 1 : 0);
  }

  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cycles));
  return 0;
}
