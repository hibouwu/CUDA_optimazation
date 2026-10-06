// V01 held-out combinations on GH200 (sm_90a).
//
//   kind 0  LDS -> FFMA, 128 threads, inputs preloaded in SMEM.
//   kind 1  small TC, 64x64x64, 2 stages, one warpgroup.  Legacy serialized lifecycle
//           (fence/MMA/commit/wait0 per instruction), kept unchanged from 2026-10-06.
//   kind 2  target TC, 128x256x64, 4 stages x 48 KiB, 384 threads, asynchronous WGMMA
//           pipeline in the style of CUTLASS's warp-specialized collective.
//
// Timing: one CTA uses its own clock64; all-GPU uses the globaltimer envelope of all CTAs.
#include "r00_common.hpp"

#include <cuda.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>

#include <fstream>
#include <utility>

#include "v01_wg_inline.cuh"

namespace G = cute::SM90::GMMA;

struct VStamp {
  uint64_t bc, ec, bn, en;  // begin/end clock64, begin/end globaltimer
  unsigned sm;
};

__device__ uint64_t vns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ unsigned vsm() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

__device__ unsigned addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// A tiles are stored tile-major: tile t holds rows [0, outer) and k in [64t, 64t+64).
__global__ void vfill(__half* data, int outer, int tiles, bool is_a) {
  for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < outer * tiles * 64;
       q += gridDim.x * blockDim.x) {
    int tile = q / (outer * 64);
    int row = (q / 64) % outer;
    int k = q % 64 + tile * 64;
    float value = is_a ? input_value(row, k, 17, true) : input_value(k, row, 17, false);
    data[q] = __float2half_rn(value);
  }
}

// ---------------------------------------------------------------------------------------------
// kind 0: LDS -> FFMA
// ---------------------------------------------------------------------------------------------
__global__ void lds_combo(int steps, VStamp* stamp, float* out) {
  extern __shared__ float input[];
  for (int q = threadIdx.x; q < 516 * steps; q += blockDim.x) {
    int row = q / steps;
    int k = q % steps;
    input[q] = input_value(row < 512 ? row : k, row < 512 ? k : row - 512, 17, row < 512);
  }
  __syncthreads();
  float d[16] = {};
  __shared__ uint64_t bc, bn;
  if (threadIdx.x == 0) {
    bn = vns();
    bc = clock64();
  }
  __syncthreads();
  for (int k = 0; k < steps; ++k) {
    float a[4], b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      a[i] = input[(threadIdx.x * 4 + i) * steps + k];
      b[i] = input[(512 + i) * steps + k];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(d[i * 4 + j]) : "f"(a[i]), "f"(b[j]));
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[blockIdx.x * 2048 + threadIdx.x * 16 + i] = d[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    stamp[blockIdx.x] = {bc, uint64_t(clock64()), bn, vns(), vsm()};
  }
}

// ---------------------------------------------------------------------------------------------
// kind 1: small TC (legacy serialized lifecycle; source kept equivalent to the 2026-10-06 run)
// ---------------------------------------------------------------------------------------------
__device__ void wait_bar(unsigned bar, uint64_t token) {
  uint64_t start = vns();
  for (;;) {
    unsigned done;
    asm volatile(
        "{.reg .pred p; mbarrier.try_wait.acquire.cta.shared::cta.b64 p,[%1],%2,64; "
        "selp.b32 %0,1,0,p;}"
        : "=r"(done)
        : "r"(bar), "l"(token)
        : "memory");
    if (done) {
      return;
    }
    if (vns() - start > 1000000000ull) {
      asm volatile("trap;" ::: "memory");
    }
  }
}

template <int M, int N, int Stages, int Threads>
__global__ __launch_bounds__(Threads, 1) void tensor_combo(int tiles,
                                                           const __grid_constant__ CUtensorMap amap,
                                                           const __grid_constant__ CUtensorMap bmap,
                                                           const __grid_constant__ CUtensorMap dmap,
                                                           VStamp* stamp, uint64_t* phases) {
  static_assert(M == 64 && N == 64 && Stages == 2 && Threads == 128, "small TC only");
  using namespace cute;
  constexpr int TileBytes = (M + N) * 64 * 2;
  constexpr int TotalBytes = Stages * TileBytes;
  constexpr int Regs = N / 2;
  extern __shared__ __align__(1024) unsigned char storage[];
  uint64_t* barrier = reinterpret_cast<uint64_t*>(storage + TotalBytes);
  uint64_t* token = barrier + Stages;
  auto la = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<Int<M>, _64>{});
  auto lb = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<Int<N>, _64>{});
  if (threadIdx.x == 0) {
    for (int i = 0; i < Stages; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(addr(barrier + i)) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  float d[Regs];
#pragma unroll
  for (int j = 0; j < Regs; ++j) {
    d[j] = 0;
    asm volatile("" : "+f"(d[j])::"memory");
  }
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __shared__ uint64_t bc, bn, prefill_done, compute_done;
  __syncthreads();
  if (threadIdx.x == 0) {
    bn = vns();
    bc = clock64();
  }
  __syncthreads();
  auto issue = [&](int tile, int slot) {
    if (threadIdx.x == 0) {
      unsigned bar = addr(barrier + slot);
      uint64_t tok;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0,[%1],%2;"
                   : "=l"(tok)
                   : "r"(bar), "r"(TileBytes)
                   : "memory");
      token[slot] = tok;
      int x = 0, ay = tile * M, by = tile * N;
      unsigned ap = addr(storage + slot * TileBytes);
      unsigned bp = ap + M * 128;
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
          "[%0],[%1,{%2,%3}],[%4];" ::"r"(ap),
          "l"(&amap), "r"(x), "r"(ay), "r"(bar)
          : "memory");
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
          "[%0],[%1,{%2,%3}],[%4];" ::"r"(bp),
          "l"(&bmap), "r"(x), "r"(by), "r"(bar)
          : "memory");
    }
  };
  for (int t = 0; t < min(tiles, Stages); ++t) {
    issue(t, t);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    prefill_done = clock64();
  }
  __syncthreads();
  for (int t = 0; t < tiles; ++t) {
    int slot = t % Stages;
    wait_bar(addr(barrier + slot), token[slot]);
    auto a = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage + slot * TileBytes)), la);
    auto b = make_tensor(
        make_smem_ptr(reinterpret_cast<__half*>(storage + slot * TileBytes) + M * 64), lb);
    uint64_t ad[4], bd[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      auto at = local_tile(a, make_shape(_64{}, _16{}), make_coord(0, q));
      auto bt = local_tile(b, make_shape(Int<N>{}, _16{}), make_coord(_0{}, q));
      ad[q] = G::make_gmma_desc<G::Major::K>(at);
      bd[q] = G::make_gmma_desc<G::Major::K>(bt);
    }
    v01_wg4_n64(ad, bd, d);
    __syncthreads();
    if (t >= 1 && t - 1 + Stages < tiles) {
      issue(t - 1 + Stages, (t - 1) % Stages);
    }
    __syncthreads();
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    compute_done = clock64();
  }
  __syncthreads();
  int lane = threadIdx.x % 128;
#pragma unroll
  for (int j = 0; j < Regs; ++j) {
    asm volatile("" : "+f"(d[j])::"memory");
    int row = (lane / 32) * 16 + (lane % 32) / 4 + ((j / 2) % 2) * 8;
    int col = (lane % 4) * 2 + j % 2 + (j / 4) * 8;
    reinterpret_cast<float*>(storage)[row * N + col] = d[j];
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    int x = 0, y = blockIdx.x * M;
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%1,%2}],[%3];" ::"l"(&dmap),
        "r"(x), "r"(y), "r"(addr(storage))
        : "memory");
    asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    stamp[blockIdx.x] = {bc, uint64_t(clock64()), bn, vns(), vsm()};
    phases[blockIdx.x * 2] = prefill_done;
    phases[blockIdx.x * 2 + 1] = compute_done;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < Stages; ++i) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(addr(barrier + i)) : "memory");
    }
  }
}

// ---------------------------------------------------------------------------------------------
// kind 2: target TC, asynchronous WGMMA pipeline
//
//   warpgroup 0  producer; one elected lane of warp 0 issues A/B TMA into 4 stages
//   warpgroup 1  consumer, rows 0..63   (m64n256k16, 4 per Ktile)
//   warpgroup 2  consumer, rows 64..127
//
// Per Ktile t a consumer waits full[t%4], issues 4 WGMMA as one group, commits, then
// wait_group 1: group t-1 is complete, so slot (t-1)%4 is released through empty[].
// At most one group stays in flight.  After the last tile both consumers wait_group 0.
// Output: input SMEM is reused as two 64 KiB halves; half h is written by consumer h,
// published (proxy fence + CTA barrier), stored by 8 SW128 TMA boxes of 32x64 FP32, and
// fully drained (bulk wait_group 0) before the next half.
//
// Segments (same SM clock64): t0 start -> t1 first stage observed by consumer 1 ->
// t2 later of the two consumers' wait_group 0 returns (all MMAs complete) -> t3 the second
// output half fully stored (thread 0 after bulk wait_group 0).  Every stamp follows a wait
// that stalls the reading warp itself (mbarrier try_wait loop, WARPGROUP.DEPBAR, bulk DEPBAR):
// a clock64 read right after BAR.SYNC can issue before the barrier releases a warp that
// arrived early, so no stamp is taken directly after a CTA barrier.
// ---------------------------------------------------------------------------------------------
namespace target {
using namespace cute;
constexpr int kTileM = 128, kTileN = 256, kTileK = 64, kStages = 4, kThreads = 384;
constexpr int kBytesA = kTileM * kTileK * 2;        // 16 KiB per stage
constexpr int kBytesB = kTileN * kTileK * 2;        // 32 KiB per stage
constexpr int kStageBytes = kBytesA + kBytesB;      // 48 KiB per stage
constexpr int kInputBytes = kStages * kStageBytes;  // 192 KiB
constexpr int kHalfRows = 64;
constexpr int kHalfBytes = kHalfRows * kTileN * 4;   // 64 KiB of FP32 per consumer
constexpr int kBoxCols = 32;                         // 128 B rows, required by SWIZZLE_128B
constexpr int kBoxBytes = kHalfRows * kBoxCols * 4;  // 8 KiB per TMA store box
constexpr int kBoxesPerHalf = kTileN / kBoxCols;     // 8
constexpr int kDynamicBytes = kInputBytes + 2 * kStages * 8;  // + full/empty mbarriers
constexpr uint32_t kCtaBarrier = 1;                           // named barrier, all 384 threads

using SmemLayoutA = decltype(tile_to_shape(G::Layout_K_SW128_Atom<half_t>{},
                                           Shape<Int<kTileM>, Int<kTileK>, Int<kStages>>{}));
using SmemLayoutB = decltype(tile_to_shape(G::Layout_K_SW128_Atom<half_t>{},
                                           Shape<Int<kTileN>, Int<kTileK>, Int<kStages>>{}));
// Two m64n256k16 atoms along M, one per consumer warpgroup (CUTLASS "cooperative" layout).
using TiledMma = decltype(make_tiled_mma(SM90_64x256x16_F32F16F16_SS<G::Major::K, G::Major::K>{},
                                         Layout<Shape<_2, _1, _1>>{}));

__device__ __forceinline__ void cta_sync() {
  asm volatile("bar.sync %0, %1;" ::"r"(kCtaBarrier), "r"(kThreads) : "memory");
}

__device__ __forceinline__ void tma_load(const CUtensorMap* map, uint64_t* bar, void* dst, int x,
                                         int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(addr(dst)),
      "l"(map), "r"(x), "r"(y), "r"(addr(bar))
      : "memory");
}

__device__ __forceinline__ void tma_store(const CUtensorMap* map, const void* src, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(map),
      "r"(x), "r"(y), "r"(addr(src))
      : "memory");
}

// Predicated arrive without a branch, so the consumer path stays non-divergent.
__device__ __forceinline__ void arrive_if(uint64_t* bar, bool pred) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}" ::
          "r"(addr(bar)),
      "r"(int(pred))
      : "memory");
}

// Register -> SMEM for one consumer's 64x256 FP32 block, in the SW128 layout of eight
// 32-column TMA boxes (box b at b*8 KiB, 128 B per row, 16 B chunk index XOR row%8).
// Register j of thread (warp, lane) holds row warp*16 + lane/4 + 8*((j/2)%2) and column
// 8*(j/4) + 2*(lane%4) + j%2, so the pair (j, j+1) is one float2.  Its chunk inside the box is
// (2*((j/4)%4) + (lane%4)/2) ^ (lane/4) = (2*((j/4)%4)) ^ w with a per-thread w, giving four
// per-thread base pointers plus compile-time offsets.  Each warp-wide float2 store covers
// all 32 banks twice (two wavefronts, no extra conflict).
template <class Accum>
__device__ __forceinline__ void store_half(const Accum& accum, unsigned char* base,
                                           unsigned thread) {
  unsigned warp = thread / 32, lane = thread % 32;
  unsigned row_part = (warp * 16 + lane / 4) * 128 + (lane % 2) * 8;
  unsigned w = ((lane % 4) / 2) ^ (lane / 4);
  unsigned char* chunk_base[4];
#pragma unroll
  for (unsigned q = 0; q < 4; ++q) {
    chunk_base[q] = base + row_part + (((q << 1) ^ w) * 16);
  }
#pragma unroll
  for (int j = 0; j < 128; j += 2) {
    constexpr int kRowBytes = 128;
    int offset = (j / 16) * kBoxBytes + ((j / 2) % 2) * 8 * kRowBytes;
    *reinterpret_cast<float2*>(chunk_base[(j / 4) % 4] + offset) =
        make_float2(accum(j), accum(j + 1));
  }
}
}  // namespace target

__global__ __launch_bounds__(target::kThreads, 1) void target_async(
    int tiles, const __grid_constant__ CUtensorMap amap, const __grid_constant__ CUtensorMap bmap,
    const __grid_constant__ CUtensorMap dmap, VStamp* stamp, uint64_t* phases) {
  using namespace target;
  using Barrier = cutlass::arch::ClusterBarrier;
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  extern __shared__ __align__(1024) unsigned char smem[];
  half_t* smem_a = reinterpret_cast<half_t*>(smem);
  half_t* smem_b = reinterpret_cast<half_t*>(smem + kStages * kBytesA);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + kInputBytes);
  uint64_t* empty = full + kStages;

  if (threadIdx.x == 0) {
    if (addr(smem) % 1024 != 0) {
      asm volatile("trap;");  // SW128 layouts need a 1024 B aligned base
    }
    for (int s = 0; s < kStages; ++s) {
      Barrier::init(&full[s], 1);   // producer arrive + TMA bytes
      Barrier::init(&empty[s], 2);  // one arrive per consumer warpgroup
    }
    cutlass::arch::fence_barrier_init();
    cute::prefetch_tma_descriptor(reinterpret_cast<const TmaDescriptor*>(&amap));
    cute::prefetch_tma_descriptor(reinterpret_cast<const TmaDescriptor*>(&bmap));
    cute::prefetch_tma_descriptor(reinterpret_cast<const TmaDescriptor*>(&dmap));
  }
  __syncthreads();
  uint64_t t0 = 0, t0_ns = 0;
  if (threadIdx.x == 0) {
    t0_ns = vns();
    t0 = clock64();
  }
  __syncthreads();

  // Warp-uniform role indices (broadcast from lane 0), as CUTLASS does.
  int warp_group = cutlass::canonical_warp_group_idx();
  int warp = cutlass::canonical_warp_idx_sync();

  if (warp_group == 0) {
    // ---------------- producer ----------------
    if (warp == 0 && cute::elect_one_sync()) {
      for (int t = 0; t < tiles; ++t) {
        int slot = t % kStages;
        if (t >= kStages) {
          Barrier::wait(&empty[slot], ((t / kStages) - 1) & 1);
        }
        TxBarrier::arrive_and_expect_tx(&full[slot], kStageBytes);
        tma_load(&amap, &full[slot], smem_a + slot * kTileM * kTileK, 0, t * kTileM);
        tma_load(&bmap, &full[slot], smem_b + slot * kTileN * kTileK, 0, t * kTileN);
      }
    }
    __syncwarp();
    cta_sync();  // all MMAs complete; input area is free
    uint64_t t3 = 0, t3_ns = 0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      cta_sync();  // half h is in SMEM and published to the async proxy
      if (threadIdx.x == 0) {
        unsigned char* base = smem + h * kHalfBytes;
        int y = blockIdx.x * kTileM + h * kHalfRows;
        for (int b = 0; b < kBoxesPerHalf; ++b) {
          tma_store(&dmap, base + b * kBoxBytes, b * kBoxCols, y);
        }
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");  // full completion
        if (h == 1) {
          t3 = clock64();
          t3_ns = vns();
        }
      }
      __syncwarp();
      cta_sync();  // half h complete
    }
    if (threadIdx.x == 0) {
      stamp[blockIdx.x] = {t0, t3, t0_ns, t3_ns, vsm()};
    }
  } else {
    // ---------------- consumers ----------------
    unsigned consumer_thread = threadIdx.x - 128;  // 0..255
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(consumer_thread);
    Tensor sA = make_tensor(make_smem_ptr(smem_a), SmemLayoutA{});  // (M, K, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem_b), SmemLayoutB{});  // (N, K, PIPE)
    Tensor tCsA = thr_mma.partition_A(sA);                          // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // GMMA descriptors
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor accum = partition_fragment_C(tiled_mma, Shape<Int<kTileM>, Int<kTileN>>{});
    static_assert(size(accum) == 128, "one m64n256 FP32 accumulator per thread");
    clear(accum);
    warpgroup_fence_operand(accum);

    // Tile 0: issue and commit; nothing to release yet.
    Barrier::wait(&full[0], 0);
    uint64_t t1 = clock64();
    warpgroup_arrive();
    gemm(tiled_mma, tCrA(_, _, _, 0), tCrB(_, _, _, 0), accum);
    warpgroup_commit_batch();
    for (int t = 1; t < tiles; ++t) {
      int slot = t % kStages;
      Barrier::wait(&full[slot], (t / kStages) & 1);
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      gemm(tiled_mma, tCrA(_, _, _, slot), tCrB(_, _, _, slot), accum);
      warpgroup_commit_batch();
      warpgroup_wait<1>();  // group t-1 complete -> its slot may be refilled
      warpgroup_fence_operand(accum);
      arrive_if(&empty[(t - 1) % kStages], consumer_thread % 128 == 0);
    }
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);
    uint64_t t2 = clock64();
    cta_sync();  // all MMAs complete
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (warp_group == 1 + h) {
        store_half(accum, smem + h * kHalfBytes, consumer_thread % 128);
        cutlass::arch::fence_view_async_shared();
      }
      cta_sync();  // publish half h
      cta_sync();  // half h stored
    }
    if (consumer_thread == 0) {
      phases[blockIdx.x * 3] = t1;
      phases[blockIdx.x * 3 + 1] = t2;
    }
    if (consumer_thread == 128) {
      phases[blockIdx.x * 3 + 2] = t2;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------------
void driver(CUresult r) {
  if (r != CUDA_SUCCESS) {
    const char* err;
    cuGetErrorString(r, &err);
    throw std::runtime_error(err);
  }
}

// Row-major 2D tensor map: `width` contiguous elements per row, `height` rows.
CUtensorMap make_map(void* p, int width, int height, int box_width, int box_rows, bool half,
                     bool swizzle) {
  CUtensorMap result;
  uint64_t dims[2] = {uint64_t(width), uint64_t(height)};
  uint64_t strides[1] = {uint64_t(width) * (half ? 2 : 4)};
  uint32_t box[2] = {unsigned(box_width), unsigned(box_rows)};
  uint32_t elements[2] = {1, 1};
  driver(cuTensorMapEncodeTiled(
      &result, half ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, p, dims,
      strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return result;
}

template <int Kind>
int execute(int length, bool whole) {
  constexpr int M = Kind == 2 ? 128 : 64;
  constexpr int N = Kind == 2 ? 256 : 64;
  constexpr int Threads = Kind == 2 ? 384 : 128;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int blocks = whole ? 4 * prop.multiProcessorCount : 1;
  int elements = Kind == 0 ? 2048 : M * N;
  int bytes =
      Kind == 0 ? 516 * length * 4 : (Kind == 1 ? 2 * (M + N) * 128 + 256 : target::kDynamicBytes);
  DeviceBuffer<VStamp> stamps(blocks);
  constexpr int PhaseCount = Kind == 2 ? 3 : 2;  // target: t1, t2 (consumer 1), t2 (consumer 2)
  DeviceBuffer<uint64_t> phases(Kind == 0 ? 1 : blocks * PhaseCount);
  DeviceBuffer<float> output(size_t(blocks) * elements);
  DeviceBuffer<__half> a(Kind == 0 ? 1 : M * length * 64);
  DeviceBuffer<__half> b(Kind == 0 ? 1 : N * length * 64);
  CUtensorMap am{}, bm{}, dm{};
  if constexpr (Kind != 0) {
    vfill<<<64, 256>>>(a.pointer, M, length, true);
    vfill<<<64, 256>>>(b.pointer, N, length, false);
    CUDA_CHECK(cudaGetLastError());
    am = make_map(a.pointer, 64, M * length, 64, M, true, true);
    bm = make_map(b.pointer, 64, N * length, 64, N, true, true);
    if constexpr (Kind == 1) {
      dm = make_map(output.pointer, N, M * blocks, N, 64, false, false);
    } else {
      dm =
          make_map(output.pointer, N, M * blocks, target::kBoxCols, target::kHalfRows, false, true);
    }
  }
  const void* kernel = Kind == 0   ? reinterpret_cast<const void*>(lds_combo)
                       : Kind == 1 ? reinterpret_cast<const void*>(tensor_combo<64, 64, 2, 128>)
                                   : reinterpret_cast<const void*>(target_async);
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  cudaFuncAttributes attrs{};
  CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
  int occupancy = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, Threads, bytes));
  if (!occupancy) {
    throw std::runtime_error("resource rejection: zero occupancy");
  }
  std::vector<VStamp> times(blocks);
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(output.pointer, 0xff, output.count * 4));
    if constexpr (Kind == 0) {
      lds_combo<<<blocks, Threads, bytes>>>(length, stamps.pointer, output.pointer);
    } else if constexpr (Kind == 1) {
      tensor_combo<64, 64, 2, 128>
          <<<blocks, Threads, bytes>>>(length, am, bm, dm, stamps.pointer, phases.pointer);
    } else {
      target_async<<<blocks, Threads, bytes>>>(length, am, bm, dm, stamps.pointer, phases.pointer);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(
        cudaMemcpy(times.data(), stamps.pointer, blocks * sizeof(VStamp), cudaMemcpyDeviceToHost));
  };
  auto window = [&]() {
    if (!whole) {
      return times[0].ec - times[0].bc;
    }
    uint64_t first = UINT64_MAX, last = 0;
    for (auto t : times) {
      first = std::min(first, t.bn);
      last = std::max(last, t.en);
    }
    return last - first;
  };
  // Warm-up: 8-30 launches until the last five windows have CV <= 2%.
  std::vector<float> warm;
  bool converged = false;
  for (int i = 0; i < 30; ++i) {
    launch();
    warm.push_back(window());
    if (i >= 7) {
      std::vector<float> last(warm.end() - 5, warm.end());
      if (coefficient_of_variation(last) <= .02) {
        converged = true;
        break;
      }
    }
  }
  launch();
  uint64_t elapsed = window();
  std::vector<uint64_t> phase_values(Kind == 0 ? 0 : blocks * PhaseCount);
  if constexpr (Kind != 0) {
    CUDA_CHECK(cudaMemcpy(phase_values.data(), phases.pointer, phase_values.size() * 8,
                          cudaMemcpyDeviceToHost));
  }
  std::vector<float> values(output.count);
  CUDA_CHECK(cudaMemcpy(values.data(), output.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
  // Inputs have period 17 in row, column and k, so a 17x17 table is the full reference.
  double reference[17][17] = {};
  int kmax = Kind == 0 ? length : length * 64;
  for (int row = 0; row < 17; ++row) {
    for (int col = 0; col < 17; ++col) {
      for (int k = 0; k < kmax; ++k) {
        reference[row][col] +=
            double(input_value(row, k, 17, true)) * input_value(k, col, 17, false);
      }
    }
  }
  double error = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    int local = i % elements;
    int row = Kind == 0 ? (local / 16) * 4 + (local % 16) / 4 : local / N;
    int col = Kind == 0 ? local % 4 : local % N;
    if (!std::isfinite(values[i])) {
      throw std::runtime_error("nonfinite output");
    }
    error = std::max(error, std::abs(double(values[i]) - reference[row % 17][col % 17]));
  }
  std::ofstream file("output.f32", std::ios::binary);
  file.write(reinterpret_cast<char*>(values.data()), values.size() * 4);
  const char* mode = Kind == 0   ? "none"
                     : Kind == 1 ? "explicit_fence_commit_wait0_per_instruction"
                                 : "async_wait1_mbarrier_release";
  const char* end_event =
      Kind == 0 ? "global output stores and CTA consumer synchronization"
                : "both consumers wait0; input area reused; all TMA output wait_group0 complete";
  std::cout << std::setprecision(17) << "{\"status\":\""
            << (error <= 1e-5 ? "measured" : "numeric_error") << "\",\"kind\":" << Kind
            << ",\"length\":" << length << ",\"blocks\":" << blocks << ",\"threads\":" << Threads
            << ",\"elapsed\":" << elapsed << ",\"unit\":\""
            << (whole ? "globaltimer_ns/GPU" : "clock64_cycle/CTA")
            << "\",\"registers_per_thread\":" << attrs.numRegs
            << ",\"local_bytes_per_thread\":" << attrs.localSizeBytes
            << ",\"static_smem_bytes\":" << attrs.sharedSizeBytes
            << ",\"dynamic_smem_bytes\":" << bytes
            << ",\"occupancy_limit_ctas_per_sm\":" << occupancy << ",\"max_output_error\":" << error
            << ",\"output_elements\":" << values.size()
            << ",\"warmup_converged\":" << (converged ? "true" : "false")
            << ",\"warmup_windows\":[";
  for (size_t i = 0; i < warm.size(); ++i) {
    std::cout << (i ? "," : "") << warm[i];
  }
  std::cout << "],\"input_profile\":\"dyadic_coordinate_seed17; all CTAs share inputs; "
               "repeat-cache\",\"end_event\":\""
            << end_event << "\",\"stamps\":[";
  for (int i = 0; i < blocks; ++i) {
    auto t = times[i];
    std::cout << (i ? "," : "") << '[' << t.bn << ',' << t.en << ',' << t.bc << ',' << t.ec << ','
              << t.sm << ']';
  }
  std::cout << "],\"wgmma_pipeline_mode\":\"" << mode << "\",\"phase_stamps\":[";
  if constexpr (Kind != 0) {
    for (int i = 0; i < blocks; ++i) {
      std::cout << (i ? "," : "") << '[';
      for (int j = 0; j < PhaseCount; ++j) {
        std::cout << (j ? "," : "") << phase_values[i * PhaseCount + j];
      }
      std::cout << ']';
    }
  }
  std::cout << "]}\n";
  return error <= 1e-5 ? 0 : 2;
}

int main(int argc, char** argv) {
  try {
    int kind = 0, length = 2;
    bool whole = false;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) {
        throw std::runtime_error("missing value");
      }
      std::string key = argv[i], value = argv[i + 1];
      if (key == "--kind") {
        kind = std::stoi(value);
      } else if (key == "--length") {
        length = std::stoi(value);
      } else if (key == "--scope") {
        if (value != "one_cta" && value != "all_gpu") {
          throw std::runtime_error("bad scope");
        }
        whole = value == "all_gpu";
      } else {
        throw std::runtime_error("bad option");
      }
    }
    if (length < 1 || length > 111) {
      throw std::runtime_error("length outside supported range");
    }
    driver(cuInit(0));
    switch (kind) {
      case 0:
        return execute<0>(length, whole);
      case 1:
        return execute<1>(length, whole);
      case 2:
        return execute<2>(length, whole);
    }
    throw std::runtime_error("bad kind");
  } catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
