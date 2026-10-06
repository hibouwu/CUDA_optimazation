// R04 joint service on GH200 (sm_90a), single CTA, clock64 window.
//
// Families: 0 WGMMA+FFMA in the same warpgroup, 1 WGMMA in warpgroup 0 + FFMA in warpgroup 1,
// 2 LDS+FFMA, 3 LDS+CVT.  Orders: 0 serial, 1 interleaved, 2 A-only, 3 B-only.
// Family 1 uses its own kernel (cross_joint): roles are dispatched on the warp-uniform
// warpgroup index and the WGMMA accumulators live only in warpgroup 0, so ptxas keeps the
// asynchronous wait_group 1 pipeline (no C7520 serialization).
#include "r00_common.hpp"

#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include <fstream>
#include <utility>
namespace G = cute::SM90::GMMA;
using MMA = G::MMA_64x256x16_F32F16F16_SS<G::Major::K, G::Major::K>;
struct Stamp {
  uint64_t start, end, start_ns, end_ns;
  unsigned sm;
};
__device__ uint64_t ns() {
  uint64_t v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
}
__device__ unsigned sm() {
  unsigned v;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
  return v;
}
__host__ __device__ float av(int row, int k) {
  return (1 + (row + 2 * k) % 7) / 16.f;
}
__host__ __device__ float bv(int col, int k) {
  return (1 + (col + 3 * k) % 11) / 32.f;
}
template <size_t... I>
__device__ __forceinline__ void mma(uint64_t a, uint64_t b, float* d, std::index_sequence<I...>) {
  MMA::fma(a, b, d[I]..., G::ScaleOut::One);
}
__device__ void ffma(float* f, int lane, float sign, int j) {
  float x = sign * (1 + lane % 7) / 1024.f;
  asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(f[j]) : "f"(x), "f"(float(j + 1)));
}
__device__ void cvt(float* input, unsigned* checksum, int j) {
  asm volatile(
      "{.reg .b16 converted; .reg .b32 bits; add.rn.f32 %0,%0,0f3b800000; cvt.rn.f16.f32 "
      "converted,%0; cvt.u32.u16 bits,converted; add.u32 %1,%1,bits; and.b32 %1,%1,65535;}"
      : "+f"(input[j]), "+r"(checksum[j])::"memory");
}
// Exact sum of the RNE half bit patterns for (base+1..base+count)/256.
uint64_t floor_prefix(int64_t value, uint64_t step) {
  if (value < 0)
    return 0;
  uint64_t q = value / step, r = value % step;
  return step * q * (q - 1) / 2 + q * (r + 1);
}
uint64_t rounded_prefix(int64_t value, uint64_t step) {
  if (value < 0)
    return 0;
  uint64_t half = step / 2;
  uint64_t ties = value >= int64_t(half) ? (value - half) / (2 * step) + 1 : 0;
  return floor_prefix(value + half, step) - floor_prefix(half - 1, step) - ties;
}
unsigned half_checksum(uint64_t base, uint64_t count) {
  uint64_t first = base + 1, last = base + count, total = 0;
  while (first <= last) {
    int n = 0;
    for (uint64_t t = first; t > 1; t >>= 1)
      ++n;
    uint64_t end = std::min(last, (uint64_t(1) << (n + 1)) - 1), items = end - first + 1;
    uint64_t constant = (n + 7) * 1024 - 1024;
    if (n <= 10)
      total += items * constant + ((first + end) * items / 2) * (uint64_t(1) << (10 - n));
    else {
      uint64_t step = uint64_t(1) << (n - 10);
      total += items * constant + rounded_prefix(end, step) - rounded_prefix(first - 1, step);
    }
    first = end + 1;
  }
  return total & 65535;
}
// Families 0, 2 and 3 (single warpgroup).  Source kept equivalent to the 2026-10-06 runs.
// Orders: serial=0, interleaved=1, A-only=2, B-only=3.  Family 0 issues 8 WGMMA per commit/wait1.
template <int Family, int Order>
__global__ void joint(int repeats, int na, int nb, bool coordinate, Stamp* stamp, float* out) {
  static_assert(Family != 1, "family 1 uses cross_joint");
  using namespace cute;
  extern __shared__ __align__(128) unsigned char storage[];
  auto la = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<_64, _64>{});
  auto lb = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<_256, _64>{});
  auto a = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)), la);
  auto an = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage) + 4096), la);
  auto b = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage) + 8192), lb);
  if constexpr (Family < 2) {
    for (int q = threadIdx.x; q < 4096; q += blockDim.x) {
      float value = coordinate ? av(q / 64, q % 64) : 0.0625f;
      a(q / 64, q % 64) = __float2half_rn(value);
      an(q / 64, q % 64) = __float2half_rn(-value);
    }
    for (int q = threadIdx.x; q < 16384; q += blockDim.x)
      b(q / 64, q % 64) = __float2half_rn(coordinate ? bv(q / 64, q % 64) : 0.0625f);
  } else {
    float* s = reinterpret_cast<float*>(storage);
    for (int q = threadIdx.x; q < 512; q += blockDim.x)
      s[q] = (q + 1) / 32.f;
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  uint64_t ad = 0, nd = 0, bd = 0;
  if constexpr (Family < 2) {
    auto at = local_tile(a, make_shape(_64{}, _16{}), make_coord(_0{}, _0{}));
    auto nt = local_tile(an, make_shape(_64{}, _16{}), make_coord(_0{}, _0{}));
    auto bt = local_tile(b, make_shape(_256{}, _16{}), make_coord(_0{}, _0{}));
    ad = G::make_gmma_desc<G::Major::K>(at);
    nd = G::make_gmma_desc<G::Major::K>(nt);
    bd = G::make_gmma_desc<G::Major::K>(bt);
  }
  float d[128], f[4] = {0, 0, 0, 0}, loaded[4] = {0, 0, 0, 0};
  unsigned h[4] = {0, 0, 0, 0};
  float cvt_input[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    cvt_input[j] = (threadIdx.x + 1 + j) / 32.f;
#pragma unroll
  for (int j = 0; j < 128; ++j) {
    d[j] = 0;
    if constexpr (Family < 2)
      asm volatile("" : "+f"(d[j])::"memory");
  }
  if constexpr (Family == 0)
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(storage)) + 16 * threadIdx.x;
  __shared__ uint64_t start, start_ns;
  __syncthreads();
  if (threadIdx.x == 0) {
    start_ns = ns();
    start = clock64();
  }
  __syncthreads();
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto B = [&](int j) {
      if constexpr (Family == 3)
        cvt(cvt_input, h, j);
      else
        ffma(f, threadIdx.x, (repeat & 1) ? -1.f : 1.f, j);
    };
    auto allB = [&]() {
#pragma unroll 1
      for (int i = 0; i < nb / 4; ++i) {
        B(0);
        B(1);
        B(2);
        B(3);
      }
    };
    if constexpr (Family < 2) {
      uint64_t active_ad = (repeat & 1) ? nd : ad;
      asm volatile("" : "+l"(active_ad)::"memory");
      auto A = [&]() { mma(active_ad, bd, d, std::make_index_sequence<128>{}); };
      auto commit = [&]() {
        asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 1;" ::
                         : "memory");
      };
      auto allA = [&]() {
#pragma unroll 1
        for (int i = 0; i < na / 8; ++i) {
#pragma unroll
          for (int q = 0; q < 8; ++q)
            A();
          commit();
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      };
      if constexpr (Order == 0) {
        allA();
        allB();
      }
      if constexpr (Order == 2)
        allA();
      if constexpr (Order == 3)
        allB();
      if constexpr (Order == 1) {
        if (na == 80) {
#pragma unroll 1
          for (int i = 0; i < 10; ++i) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              A();
              B(q % 4);
            }
            commit();
          }
        } else if (na == 32) {
#pragma unroll 1
          for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              A();
              B(0);
              B(1);
              B(2);
              B(3);
            }
            commit();
          }
        } else {
#pragma unroll 1
          for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              A();
              A();
              A();
              A();
              B(j);
              if (j % 2 == 1)
                commit();
            }
          }
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      }
    } else {
      auto A = [&]() {
        asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(loaded[0]), "=f"(loaded[1]), "=f"(loaded[2]), "=f"(loaded[3])
                     : "r"(address)
                     : "memory");
      };
      if constexpr (Order == 0) {
#pragma unroll 1
        for (int i = 0; i < na; ++i)
          A();
        __shared__ volatile float phase_sink[128];
        phase_sink[threadIdx.x] = loaded[0] + loaded[1] + loaded[2] + loaded[3];
        __syncthreads();
        allB();
      }
      if constexpr (Order == 2) {
#pragma unroll 1
        for (int i = 0; i < na; ++i)
          A();
        __shared__ volatile float phase_sink[128];
        phase_sink[threadIdx.x] = loaded[0] + loaded[1] + loaded[2] + loaded[3];
        __syncthreads();
      }
      if constexpr (Order == 3)
        allB();
      if constexpr (Order == 1) {
        if (na == 80) {
#pragma unroll 1
          for (int i = 0; i < 20; ++i) {
            A();
            B(0);
            A();
            B(1);
            A();
            B(2);
            A();
            B(3);
          }
        } else if (na == 32) {
#pragma unroll 1
          for (int i = 0; i < 32; ++i) {
            A();
            B(0);
            B(1);
            B(2);
            B(3);
          }
        } else {
#pragma unroll 1
          for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              A();
              A();
              A();
              A();
              B(j);
            }
          }
        }
      }
    }
  }
  // Volatile memory and dependencies force the independent results to complete.
  __shared__ volatile float drain[256];
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    sum += f[j] + loaded[j] + float(h[j]);
#pragma unroll
  for (int j = 0; j < 128; ++j) {
    if constexpr (Family < 2) {
      asm volatile("" : "+f"(d[j])::"memory");
      sum += d[j];
    }
  }
  drain[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    stamp[0] = {start, uint64_t(clock64()), start_ns, ns(), sm()};
  }
#pragma unroll
  for (int j = 0; j < 128; ++j)
    out[threadIdx.x * 140 + j] = d[j];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    out[threadIdx.x * 140 + 128 + j] = f[j];
    out[threadIdx.x * 140 + 132 + j] = loaded[j];
    out[threadIdx.x * 140 + 136 + j] = float(h[j]);
  }
}

// Family 1, cross-warpgroup.  Warpgroup 0 issues WGMMA: per 8 instructions one fence (arrive),
// 8 x m64n256k16, commit, wait_group 1, so at most one group stays in flight; wait_group 0
// ends each unit.  Warpgroup 1 runs the FFMA work.  Serial order separates the two phases with
// a CTA barrier; every unit ends with a CTA barrier.  Barrier 1 (256 threads) is used from
// both role branches in the same sequence.
__device__ __forceinline__ void cta_sync256() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}

// clock64/globaltimer read that cannot issue before the preceding barrier has released this
// warp.  A plain clock read after BAR.SYNC can issue early when the warp arrived first; here
// the read is predicated on a shared-memory load, and the load waits for the barrier.
__device__ __forceinline__ void stamp_after_sync(const volatile float* word, uint64_t& cycles,
                                                 uint64_t& nanoseconds) {
  unsigned bits = __float_as_uint(*word);
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.u32 p, %2, 0xffffffff;\n"
      "@p mov.u64 %0, %%clock64;\n@p mov.u64 %1, %%globaltimer;\n"
      "@!p mov.u64 %0, 0;\n@!p mov.u64 %1, 0;\n}"
      : "=l"(cycles), "=l"(nanoseconds)
      : "r"(bits)
      : "memory");
}

template <int Order>
__global__ __launch_bounds__(256, 1) void cross_joint(int repeats, int na, int nb, bool coordinate,
                                                      Stamp* stamp, float* out) {
  using namespace cute;
  extern __shared__ __align__(128) unsigned char storage[];
  auto la = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<_64, _64>{});
  auto lb = tile_to_shape(G::Layout_K_SW128_Atom<__half>{}, Shape<_256, _64>{});
  auto a = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage)), la);
  auto an = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage) + 4096), la);
  auto b = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(storage) + 8192), lb);
  for (int q = threadIdx.x; q < 4096; q += blockDim.x) {
    float value = coordinate ? av(q / 64, q % 64) : 0.0625f;
    a(q / 64, q % 64) = __float2half_rn(value);
    an(q / 64, q % 64) = __float2half_rn(-value);
  }
  for (int q = threadIdx.x; q < 16384; q += blockDim.x) {
    b(q / 64, q % 64) = __float2half_rn(coordinate ? bv(q / 64, q % 64) : 0.0625f);
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  auto at = local_tile(a, make_shape(_64{}, _16{}), make_coord(_0{}, _0{}));
  auto nt = local_tile(an, make_shape(_64{}, _16{}), make_coord(_0{}, _0{}));
  auto bt = local_tile(b, make_shape(_256{}, _16{}), make_coord(_0{}, _0{}));
  uint64_t ad = G::make_gmma_desc<G::Major::K>(at);
  uint64_t nd = G::make_gmma_desc<G::Major::K>(nt);
  uint64_t bd = G::make_gmma_desc<G::Major::K>(bt);
  __shared__ uint64_t start, start_ns;
  __shared__ volatile float drain[256];
  drain[threadIdx.x] = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    stamp_after_sync(&drain[255], start, start_ns);
  }
  __syncthreads();
  float* row = out + threadIdx.x * 140;
  if (cutlass::canonical_warp_group_idx() == 0) {
    // ---------------- warpgroup 0: WGMMA ----------------
    float d[128];
#pragma unroll
    for (int j = 0; j < 128; ++j) {
      d[j] = 0;
      warpgroup_fence_operand(d[j]);
    }
    for (int repeat = 0; repeat < repeats; ++repeat) {
      if constexpr (Order != 3) {
        uint64_t active_ad = (repeat & 1) ? nd : ad;  // alternate sign keeps sums bounded
        asm volatile("" : "+l"(active_ad)::"memory");
#pragma unroll 1
        for (int i = 0; i < na / 8; ++i) {
          warpgroup_arrive();
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            mma(active_ad, bd, d, std::make_index_sequence<128>{});
          }
          warpgroup_commit_batch();
          warpgroup_wait<1>();
        }
        warpgroup_wait<0>();
#pragma unroll
        for (int j = 0; j < 128; ++j) {
          warpgroup_fence_operand(d[j]);
        }
      }
      if constexpr (Order == 0) {
        cta_sync256();  // A phase complete, B phase may start
      }
      cta_sync256();  // unit complete
    }
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 128; ++j) {
      sum += d[j];
    }
    drain[threadIdx.x] = sum;
    cta_sync256();
    if (threadIdx.x == 0) {
      uint64_t end, end_ns;
      stamp_after_sync(&drain[255], end, end_ns);  // drain[255] is written by warpgroup 1
      stamp[0] = {start, end, start_ns, end_ns, sm()};
    }
#pragma unroll
    for (int j = 0; j < 128; ++j) {
      row[j] = d[j];
    }
#pragma unroll
    for (int j = 128; j < 140; ++j) {
      row[j] = 0;
    }
  } else {
    // ---------------- warpgroup 1: FFMA ----------------
    float f[4] = {0, 0, 0, 0};
    for (int repeat = 0; repeat < repeats; ++repeat) {
      if constexpr (Order == 0) {
        cta_sync256();
      }
      if constexpr (Order != 2) {
        float sign = (repeat & 1) ? -1.f : 1.f;
#pragma unroll 1
        for (int i = 0; i < nb / 4; ++i) {
          ffma(f, threadIdx.x, sign, 0);
          ffma(f, threadIdx.x, sign, 1);
          ffma(f, threadIdx.x, sign, 2);
          ffma(f, threadIdx.x, sign, 3);
        }
      }
      cta_sync256();
    }
    drain[threadIdx.x] = f[0] + f[1] + f[2] + f[3];
    cta_sync256();
#pragma unroll
    for (int j = 0; j < 128; ++j) {
      row[j] = 0;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      row[128 + j] = f[j];
      row[132 + j] = 0;
      row[136 + j] = 0;
    }
  }
}

template <int F, int O>
const void* kernel_for() {
  if constexpr (F == 1) {
    return reinterpret_cast<const void*>(cross_joint<O>);
  } else {
    return reinterpret_cast<const void*>(joint<F, O>);
  }
}

template <int F, int O>
int run(int repeats, int na, int nb) {
  int threads = F == 1 ? 256 : 128, bytes = F < 2 ? 49152 : 2048;
  const void* kernel = kernel_for<F, O>();
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  cudaFuncAttributes attrs{};
  CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
  int occ;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, threads, bytes));
  DeviceBuffer<Stamp> s(1);
  DeviceBuffer<float> output(threads * 140);
  std::vector<float> values(output.count);
  Stamp stamp{};
  auto launch = [&](int length, bool coord) {
    if constexpr (F == 1) {
      cross_joint<O><<<1, threads, bytes>>>(length, na, nb, coord, s.pointer, output.pointer);
    } else {
      joint<F, O><<<1, threads, bytes>>>(length, na, nb, coord, s.pointer, output.pointer);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&stamp, s.pointer, sizeof(stamp), cudaMemcpyDeviceToHost));
  };
  auto check = [&](int length, bool coord) {
    CUDA_CHECK(
        cudaMemcpy(values.data(), output.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
    double error = 0;
    for (int t = 0; t < threads; ++t)
      for (int j = 0; j < 140; ++j) {
        double ref = 0;
        bool do_a = O != 3, do_b = O != 2;
        if (j < 128 && F < 2 && (F == 0 || t < 128) && do_a) {
          int lane = t % 128, row = (lane / 32) * 16 + (lane % 32) / 4 + ((j / 2) % 2) * 8,
              col = (lane % 4) * 2 + j % 2 + (j / 4) * 8;
          for (int k = 0; k < 16; ++k)
            ref += coord ? double(av(row, k)) * bv(col, k) : 1. / 256;
          ref *= (length % 2) * na;
        }
        if (j >= 128 && j < 132 && F != 3 && (F != 1 || t >= 128) && do_b)
          ref = double(length % 2) * (nb / 4) * (1 + t % 7) * (j - 127) / 1024;
        if (j >= 132 && j < 136 && F >= 2 && do_a)
          ref = (t * 4 + j - 132 + 1) / 32.;
        if (j >= 136 && F == 3 && do_b)
          ref = half_checksum((t + 1 + j - 136) * 8, uint64_t(length) * (nb / 4));
        if (!std::isfinite(values[t * 140 + j]))
          throw std::runtime_error("nonfinite output");
        error = std::max(error, std::abs(double(values[t * 140 + j]) - ref));
      }
    return error;
  };
  launch(1, true);
  double short_error = check(1, true);
  if (short_error > 1e-4)
    throw std::runtime_error("short coordinate check: " + std::to_string(short_error));
  std::vector<float> warm;
  bool converged = false;
  for (int i = 0; i < 30; ++i) {
    launch(repeats, false);
    warm.push_back(stamp.end - stamp.start);
    if (i >= 7) {
      std::vector<float> last(warm.end() - 5, warm.end());
      if (coefficient_of_variation(last) <= 0.02) {
        converged = true;
        break;
      }
    }
  }
  launch(repeats, false);
  double error = check(repeats, false);
  std::ofstream raw("output.f32", std::ios::binary);
  raw.write(reinterpret_cast<char*>(values.data()), values.size() * 4);
  std::cout << std::setprecision(17) << "{\"status\":\""
            << (error <= 1e-4 ? "measured" : "numeric_error") << "\",\"family\":" << F
            << ",\"order\":" << O << ",\"a_count\":" << na << ",\"b_count\":" << nb
            << ",\"repeats\":" << repeats << ",\"elapsed\":" << stamp.end - stamp.start
            << ",\"elapsed_ns\":" << stamp.end_ns - stamp.start_ns << ",\"sm_id\":" << stamp.sm
            << ",\"threads\":" << threads << ",\"registers_per_thread\":" << attrs.numRegs
            << ",\"local_bytes_per_thread\":" << attrs.localSizeBytes
            << ",\"static_smem_bytes\":" << attrs.sharedSizeBytes
            << ",\"dynamic_smem_bytes\":" << bytes << ",\"occupancy_limit_ctas_per_sm\":" << occ
            << ",\"max_output_error\":" << error << ",\"short_coordinate_error\":" << short_error
            << ",\"warmup_converged\":" << (converged ? "true" : "false") << ",\"warmup_cycles\":[";
  for (size_t i = 0; i < warm.size(); ++i) {
    if (i)
      std::cout << ',';
    std::cout << warm[i];
  }
  const char* mode = F >= 2 ? "none" : (F == 1 ? "async_wait1_uniform_roles" : "batch8_wait1");
  std::cout << "],\"output_elements\":" << values.size()
            << ",\"end_event\":\"all paths drained, CTA result consumption complete\""
            << ",\"wgmma_pipeline_mode\":\"" << mode << "\"}\n";
  return error <= 1e-4 ? 0 : 2;
}
template <int F>
int dispatch(int o, int r, int a, int b) {
  switch (o) {
    case 0:
      return run<F, 0>(r, a, b);
    case 1:
      return run<F, 1>(r, a, b);
    case 2:
      return run<F, 2>(r, a, b);
    case 3:
      return run<F, 3>(r, a, b);
  }
  throw std::runtime_error("bad order");
}
int main(int argc, char** argv) {
  try {
    int f = 0, o = 0, r = 128, a = 80, b = 80;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc)
        throw std::runtime_error("missing value");
      std::string k = argv[i];
      int v = std::stoi(argv[i + 1]);
      if (k == "--family")
        f = v;
      else if (k == "--order")
        o = v;
      else if (k == "--repeats")
        r = v;
      else if (k == "--a")
        a = v;
      else if (k == "--b")
        b = v;
      else
        throw std::runtime_error("bad option");
    }
    if (r < 1 || r > 65536 || a < 0 || b < 0 || a + b != 160)
      throw std::runtime_error("invalid workload");
    if ((o == 2 && (a != 160 || b != 0)) || (o == 3 && (a != 0 || b != 160)) ||
        ((o == 0 || o == 1) && (a != 32 && a != 80 && a != 128)))
      throw std::runtime_error("configuration outside R04 matrix");
    switch (f) {
      case 0:
        return dispatch<0>(o, r, a, b);
      case 1:
        return dispatch<1>(o, r, a, b);
      case 2:
        return dispatch<2>(o, r, a, b);
      case 3:
        return dispatch<3>(o, r, a, b);
    }
    throw std::runtime_error("bad family");
  } catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
