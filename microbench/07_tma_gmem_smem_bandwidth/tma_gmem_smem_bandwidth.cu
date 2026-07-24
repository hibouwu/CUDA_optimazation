#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err__ = (call);                                             \
    if (err__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err__));                              \
      std::exit(3);                                                         \
    }                                                                       \
  } while (0)

#define CU_CHECK(call)                                                       \
  do {                                                                       \
    CUresult err__ = (call);                                                 \
    if (err__ != CUDA_SUCCESS) {                                             \
      const char* msg__ = "unknown";                                        \
      cuGetErrorString(err__, &msg__);                                       \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s\n", __FILE__,       \
                   __LINE__, msg__);                                         \
      std::exit(3);                                                          \
    }                                                                        \
  } while (0)

enum class Mode { kL2Hit, kDramStream };

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(addr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint32_t addr, uint32_t phase) {
  constexpr uint32_t kTicks = 0x989680;
  asm volatile(
      "{ .reg .pred p; wait_loop_%=: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
      "@!p bra wait_loop_%=; }"
      :
      : "r"(addr), "r"(phase), "r"(kTicks)
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t barrier,
                                                          uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
      :
      : "r"(barrier), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void tma_load_3d(uint32_t dst,
                                            const CUtensorMap* map,
                                            int x,
                                            int y,
                                            int z,
                                            uint32_t barrier) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global."
      "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
      :
      : "r"(dst), "l"(map), "r"(x), "r"(y), "r"(z), "r"(barrier)
      : "memory");
}

__global__ void init_kernel(uint32_t* data, size_t words) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < words; i += stride) {
    data[i] = static_cast<uint32_t>(i * 1664525u + 1013904223u);
  }
}

__global__ __launch_bounds__(128, 1)
void tma_kernel(const __grid_constant__ CUtensorMap map,
                int slots,
                int iters,
                int warmup_iters,
                int tile_words,
                int total_tiles,
                unsigned long long* cycles,
                uint32_t* sink) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ alignas(16) uint64_t barrier_storage;
  __shared__ unsigned long long start_clock;
  __shared__ unsigned long long stop_clock;

  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) {
    mbarrier_init(smem_u32(&barrier_storage), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    start_clock = 0;
    stop_clock = 0;
  }
  __syncthreads();

  const uint32_t barrier = smem_u32(&barrier_storage);
  uint32_t phase = 0;
  uint32_t checksum = 0;
  const int total_iters = warmup_iters + iters;
  const int tile_words_per_slot = tile_words;

  for (int i = 0; i < total_iters; ++i) {
    if (i == warmup_iters && tid == 0) {
      start_clock = clock64();
    }
    const int slot = i & (slots - 1);
    const int tile = (blockIdx.x * total_iters + i) % total_tiles;
    unsigned char* dst = smem + static_cast<size_t>(slot) * tile_words_per_slot * sizeof(uint32_t);
    if (tid == 0) {
      tma_load_3d(smem_u32(dst), &map, 0, 0, tile, barrier);
      mbarrier_arrive_expect_tx(barrier, static_cast<uint32_t>(tile_words_per_slot * sizeof(uint32_t)));
    }
    mbarrier_wait(barrier, phase);
    phase ^= 1;

    uint32_t* words = reinterpret_cast<uint32_t*>(dst);
    if ((i >= warmup_iters) && tid < 32) {
      checksum ^= words[(tid * 17 + i) & (tile_words_per_slot - 1)];
    }
  }

  if (tid == 0) {
    stop_clock = clock64();
  }
  __syncthreads();

  if (tid < 32) {
    atomicXor(sink, checksum + static_cast<uint32_t>(tid));
  }
  if (tid == 0) {
    cycles[blockIdx.x] = stop_clock - start_clock;
  }
}

const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::kL2Hit: return "l2-hit";
    case Mode::kDramStream: return "dram-stream";
  }
  return "unknown";
}

Mode parse_mode(const char* text) {
  if (std::strcmp(text, "l2-hit") == 0) return Mode::kL2Hit;
  if (std::strcmp(text, "dram-stream") == 0) return Mode::kDramStream;
  std::fprintf(stderr, "Unknown mode: %s\n", text);
  std::exit(2);
}

struct Options {
  Mode mode = Mode::kL2Hit;
  size_t bytes = 16ull << 20;
  int tile_bytes = 32768;
  int slots = 4;
  int threads = 128;
  int blocks_per_sm = 1;
  int iters = 4096;
  int warmup_iters = 32;
  bool csv = false;
  bool csv_header = false;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--mode l2-hit|dram-stream] [--bytes N] [--tile-bytes N]\n"
               "          [--slots N] [--threads N] [--blocks-per-sm N]\n"
               "          [--iters N] [--warmup-iters N] [--csv] [--csv-header]\n",
               argv0);
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      o.mode = parse_mode(argv[++i]);
    } else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      o.bytes = std::strtoull(argv[++i], nullptr, 0);
    } else if (std::strcmp(argv[i], "--tile-bytes") == 0 && i + 1 < argc) {
      o.tile_bytes = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--slots") == 0 && i + 1 < argc) {
      o.slots = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      o.threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--blocks-per-sm") == 0 && i + 1 < argc) {
      o.blocks_per_sm = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      o.iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup-iters") == 0 && i + 1 < argc) {
      o.warmup_iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      o.csv = true;
    } else if (std::strcmp(argv[i], "--csv-header") == 0) {
      o.csv_header = true;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      usage(argv[0]);
      std::exit(2);
    }
  }
  return o;
}

void encode_tma_3d(CUtensorMap* map, void* base, int tile_words, int total_tiles) {
  constexpr int kInnerWords = 256;
  const int tile_rows = tile_words / kInnerWords;
  const uint32_t rank = 3;
  uint64_t global_dim[rank] = {
      static_cast<uint64_t>(kInnerWords),
      static_cast<uint64_t>(tile_rows),
      static_cast<uint64_t>(total_tiles)};
  uint64_t global_stride[rank - 1] = {
      static_cast<uint64_t>(kInnerWords * sizeof(uint32_t)),
      static_cast<uint64_t>(tile_words * sizeof(uint32_t))};
  uint32_t box_dim[rank] = {
      static_cast<uint32_t>(kInnerWords),
      static_cast<uint32_t>(tile_rows),
      1u};
  uint32_t element_stride[rank] = {1u, 1u, 1u};
  CU_CHECK(cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_UINT32, rank, base, global_dim,
      global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main(int argc, char** argv) {
  Options o = parse_args(argc, argv);
  if (o.csv_header) {
    std::puts("mode,requested_bytes,elapsed_cycles,bytes_per_cycle,per_sm_bytes_per_cycle,sm_count,blocks,blocks_per_sm,threads,iters,warmup_iters,tile_bytes,slots,working_set_bytes,total_tiles,occupancy_blocks_per_sm,sink");
    if (argc == 2) return 0;
  }

  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  const int blocks = sm_count * o.blocks_per_sm;
  if (o.tile_bytes <= 0 || (o.tile_bytes % 1024) != 0) {
    std::fprintf(stderr, "tile-bytes must be positive and 1024B aligned\n");
    return 2;
  }
  if ((o.tile_bytes & (o.tile_bytes - 1)) != 0) {
    std::fprintf(stderr, "tile-bytes must be power-of-two for checksum indexing\n");
    return 2;
  }
  if (o.slots <= 0 || (o.slots & (o.slots - 1)) != 0) {
    std::fprintf(stderr, "slots must be a positive power of two\n");
    return 2;
  }
  if (o.threads != 128) {
    std::fprintf(stderr, "this kernel is launch-bounds tuned for 128 threads\n");
    return 2;
  }

  const int tile_words = o.tile_bytes / static_cast<int>(sizeof(uint32_t));
  int total_tiles = static_cast<int>(o.bytes / static_cast<size_t>(o.tile_bytes));
  total_tiles = std::max(total_tiles, blocks);
  if (o.mode == Mode::kDramStream) {
    total_tiles = std::max(total_tiles, blocks * (o.warmup_iters + o.iters));
  }
  size_t working_set_bytes = static_cast<size_t>(total_tiles) * o.tile_bytes;
  if (working_set_bytes != o.bytes && !o.csv) {
    std::fprintf(stderr, "rounded working set to %zu bytes\n", working_set_bytes);
  }

  CU_CHECK(cuInit(0));
  uint32_t* d_data = nullptr;
  unsigned long long* d_cycles = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, working_set_bytes));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long) * sm_count * o.blocks_per_sm));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(uint32_t)));

  int init_blocks = std::min(4096, std::max(1, static_cast<int>((working_set_bytes / sizeof(uint32_t) + 255) / 256)));
  init_kernel<<<init_blocks, 256>>>(d_data, working_set_bytes / sizeof(uint32_t));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUtensorMap map;
  encode_tma_3d(&map, d_data, tile_words, total_tiles);

  const size_t smem_bytes = static_cast<size_t>(o.tile_bytes) * o.slots;
  int max_optin_smem = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  if (smem_bytes > static_cast<size_t>(max_optin_smem)) {
    std::fprintf(stderr, "dynamic shared memory request %zu exceeds opt-in limit %d\n",
                 smem_bytes, max_optin_smem);
    return 2;
  }
  CUDA_CHECK(cudaFuncSetAttribute(tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem_bytes)));
  tma_kernel<<<blocks, o.threads, smem_bytes>>>(map, o.slots, o.iters, o.warmup_iters,
                                                tile_words, total_tiles, d_cycles, d_sink);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned long long> h_cycles(blocks);
  uint32_t h_sink = 0;
  CUDA_CHECK(cudaMemcpy(h_cycles.data(), d_cycles, h_cycles.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_sink, d_sink, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  unsigned long long elapsed = 0;
  for (auto c : h_cycles) elapsed = std::max(elapsed, c);
  const unsigned long long requested =
      static_cast<unsigned long long>(blocks) * static_cast<unsigned long long>(o.iters) *
      static_cast<unsigned long long>(o.tile_bytes);
  const double bpc = elapsed ? static_cast<double>(requested) / static_cast<double>(elapsed) : 0.0;

  int occupancy = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, tma_kernel, o.threads, smem_bytes));

  if (o.csv) {
    std::printf("%s,%llu,%llu,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%d,%d,%u\n",
                mode_name(o.mode), requested, elapsed, bpc, bpc / sm_count, sm_count,
                blocks, o.blocks_per_sm, o.threads, o.iters, o.warmup_iters,
                o.tile_bytes, o.slots, working_set_bytes, total_tiles, occupancy, h_sink);
  } else {
    std::printf("mode=%s\nrequested_bytes=%llu\nelapsed_cycles=%llu\nbytes_per_cycle=%.6f\nsink=%u\n",
                mode_name(o.mode), requested, elapsed, bpc, h_sink);
  }

  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_data));
  return 0;
}
