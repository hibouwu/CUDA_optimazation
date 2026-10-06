#include "r00_common.hpp"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <fstream>
#include <functional>
#include <utility>
struct R05Stamp {
  uint64_t begin_cycle, end_cycle, begin_ns, end_ns, wait_position, wait_return, release, complete;
  unsigned sm;
};
__device__ unsigned r05_shared(const void* p) {
  return unsigned(__cvta_generic_to_shared(p));
}
__device__ uint64_t r05_ns() {
  uint64_t x;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(x));
  return x;
}
__device__ unsigned r05_smid() {
  unsigned x;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(x));
  return x;
}
__device__ void r05_wait(unsigned bar, uint64_t token) {
  uint64_t start = r05_ns();
  for (;;) {
    unsigned done;
    asm volatile(
        "{.reg .pred p; mbarrier.try_wait.acquire.cta.shared::cta.b64 p,[%1],%2,64;selp.b32 %0,1,0,p;}"
        : "=r"(done)
        : "r"(bar), "l"(token)
        : "memory");
    if (done)
      return;
    if (r05_ns() - start > 1000000000ull)
      asm volatile("trap;");
  }
}
__device__ void r05_filler(int count, unsigned& value) {
  for (int j = 0; j < count; ++j)
    asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(value) : "r"(1664525u), "r"(1013904223u));
}
__device__ void r05_cp_wait(int pending) {
  if (pending == 0)
    asm volatile("cp.async.wait_group 0;" ::: "memory");
  else if (pending == 1)
    asm volatile("cp.async.wait_group 1;" ::: "memory");
  else if (pending == 2)
    asm volatile("cp.async.wait_group 2;" ::: "memory");
  else
    asm volatile("cp.async.wait_group 3;" ::: "memory");
}
// Mode 0 cp.async input, 1 bulk TMA input, 2 bulk TMA output.
template <int Mode>
__global__ void r05_transport(unsigned* global, int iterations, int filler, int stages,
                              int consumer, int requests, bool release_read, bool pipeline,
                              bool control, bool capture, unsigned* trace, float* consumed,
                              R05Stamp* stamps) {
  constexpr int Q = Mode == 0 ? 2048 : 16384, Words = Q / 4;
  extern __shared__ __align__(128) unsigned char storage[];
  auto* tile = reinterpret_cast<unsigned*>(storage);
  unsigned bars = r05_shared(storage + 4 * Q);
  uint64_t tokens[4] = {};
  unsigned thread = threadIdx.x, fill = thread + 17;
  float sum = 0;
  R05Stamp stamp{};
  for (int n = thread; n < 4 * Words; n += 128)
    tile[n] = 29u * (n % Words) + 151u * blockIdx.x + 17;
  if constexpr (Mode == 1)
    if (thread == 0)
      for (int s = 0; s < 4; ++s)
        asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bars + s * 8) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (thread == 0) {
    stamp.sm = r05_smid();
    stamp.begin_ns = r05_ns();
    stamp.begin_cycle = clock64();
  }
  __syncthreads();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    int issued = 0;
    int count = pipeline ? 8 : 1;
    // cp groups are committed by each thread; TMA barriers are managed by producer lane0.
    for (int request = 0; request < count; ++request) {
      uint64_t issue_cycle = thread == 0 ? clock64() : 0;
      int desired = min(count, request + stages);
      while (issued < desired) {
        int slot = issued % stages;
        unsigned dst = r05_shared(tile + slot * Words);
        unsigned src = ((pipeline ? issued : iteration) % 8) * Words;
        if (!control) {
          if constexpr (Mode == 0) {
            asm volatile("cp.async.ca.shared.global [%0],[%1],16;" ::"r"(dst + thread * 16),
                         "l"(global + src + thread * 4)
                         : "memory");
            asm volatile("cp.async.commit_group;" ::: "memory");
          }
          if constexpr (Mode == 1)
            if (thread == 0) {
              asm volatile(
                  "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0],%1;" ::"r"(bars + slot * 8),
                  "r"(Q)
                  : "memory");
              asm volatile(
                  "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                      "r"(dst),
                  "l"(global + src), "r"(Q), "r"(bars + slot * 8)
                  : "memory");
              asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0,[%1];"
                           : "=l"(tokens[slot])
                           : "r"(bars + slot * 8)
                           : "memory");
            }
        }
        ++issued;
      }
      int slot = request % stages;
      if constexpr (Mode == 2) {
        // Requests in one bulk group read one source; each has a distinct global destination.
        for (int n = thread; n < Words; n += 128)
          tile[n] = 29u * n + 31u * iteration + 17;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (!control && thread == 0) {
          for (int r = 0; r < requests; ++r)
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0],[%1],%2;" ::"l"(
                             global + ((iteration % 8) * requests + r) * Words),
                         "r"(r05_shared(tile)), "r"(Q)
                         : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
      }
      // Only the issuing lane performs TMA output independent work.
      if constexpr (Mode == 2) {
        if (thread == 0)
          r05_filler(filler, fill);
      } else
        r05_filler(filler, fill);
      if (thread == 0)
        stamp.wait_position += clock64() - issue_cycle;
      if (!control) {
        if constexpr (Mode == 0)
          r05_cp_wait(issued - request - 1);
        if constexpr (Mode == 1)
          if (thread == 0)
            r05_wait(bars + slot * 8, tokens[slot]);
        if constexpr (Mode == 2)
          if (thread == 0) {
            if (release_read)
              asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            else
              asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
          }
      }
      if (thread == 0)
        stamp.wait_return += clock64() - issue_cycle;
      __syncthreads();
      if constexpr (Mode < 2) {
        unsigned value = tile[slot * Words + (Mode == 0 ? thread * 4 : thread)];
        float x = float(int(value & 15u) - 8) / 32.f;
        for (int c = 0; c < consumer; ++c)
          asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(x) : "f"(1.f), "f"(1.f / 1024));
        sum += x;
        if (capture)
          for (int n = thread; n < Words; n += 128)
            trace[(iteration * count + request) * Words + n] = tile[slot * Words + n];
        __syncthreads();  // Consumer completion releases the input slot.
      } else {
        if (thread == 0)
          stamp.release = clock64() - stamp.begin_cycle;
        for (int n = thread; n < Words; n += 128)
          tile[n] = ~(29u * n + 31u * iteration + 17u);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (!control && thread == 0)
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        __syncthreads();
        if (thread == 0)
          stamp.complete = clock64() - stamp.begin_cycle;
      }
    }
  }
  // Final drain precedes consumer materialization / full-output completion.
  if constexpr (Mode == 0)
    if (!control)
      asm volatile("cp.async.wait_group 0;" ::: "memory");
  __shared__ volatile float sink[128];
  sink[thread] = sum;
  __syncthreads();
  if (thread == 0) {
    stamp.end_cycle = clock64();
    stamp.end_ns = r05_ns();
    stamps[0] = stamp;
  }
  consumed[thread] = sum;
  if constexpr (Mode == 1) {
    __syncthreads();
    if (thread == 0)
      for (int s = 0; s < 4; ++s)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(bars + s * 8) : "memory");
  }
  if (capture && Mode == 2)
    for (int n = thread; n < Words; n += 128)
      trace[n] = tile[n];
  // Prevent removal of the independent integer chain, outside timing.
  if (thread == 0)
    trace[capture ? (Mode == 2 ? Words : iterations * (pipeline ? 8 : 1) * Words) : 0] = fill;
}
namespace R05G = cute::SM90::GMMA;
template <size_t... I>
__device__ void r05_mma(uint64_t a, uint64_t b, float* d, std::index_sequence<I...>) {
  R05G::MMA_64x256x16_F32F16F16_SS<R05G::Major::K, R05G::Major::K>::fma(a, b, d[I]...,
                                                                        R05G::ScaleOut::One);
}
#include "r05_wg_inline.cuh"
template <int Filler, bool Control>
__global__ void r05_wg(int iterations, float* out, unsigned* integer_out, R05Stamp* stamps) {
  using namespace cute;
  extern __shared__ __align__(128) unsigned char bytes[];
  auto layout = tile_to_shape(R05G::Layout_K_SW128_Atom<__half>{}, Shape<_64, _64>{});
  auto lb = tile_to_shape(R05G::Layout_K_SW128_Atom<__half>{}, Shape<_256, _64>{});
  auto a = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(bytes)), layout);
  auto b = make_tensor(make_smem_ptr(reinterpret_cast<__half*>(bytes) + 4096), lb);
  for (int i = threadIdx.x; i < 4096; i += 128)
    a(i / 64, i % 64) = __float2half((1 + (i / 64 + 2 * (i % 64)) % 7) / 16.f);
  for (int i = threadIdx.x; i < 16384; i += 128)
    b(i / 64, i % 64) = __float2half((1 + (i / 64 + 3 * (i % 64)) % 11) / 32.f);
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  auto av = local_tile(a, Shape<_64, _16>{}, make_coord(_0{}, _0{}));
  auto bv = local_tile(b, Shape<_256, _16>{}, make_coord(_0{}, _0{}));
  uint64_t ad = R05G::make_gmma_desc<R05G::Major::K>(av),
           bd = R05G::make_gmma_desc<R05G::Major::K>(bv);
  float d[128] = {};
  unsigned fill = threadIdx.x + 17, seed = 0;
  R05Stamp s{};
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    s.sm = r05_smid();
    s.begin_ns = r05_ns();
    s.begin_cycle = clock64();
  }
  __syncthreads();
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    uint64_t issue_cycle = clock64(), position = 0, returned = 0;
    r05_sequence<Filler, Control>(ad, bd, d, fill, position, returned, seed);
    if (threadIdx.x == 0) {
      s.wait_position += position - issue_cycle;
      s.wait_return += returned - issue_cycle;
    }
  }

  float sum = 0;
  for (int j = 0; j < 128; ++j) {
    asm volatile("" : "+f"(d[j]));
    sum += d[j];
  }
  __shared__ volatile float sink[128];
  sink[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    s.end_cycle = clock64();
    s.end_ns = r05_ns();
    stamps[0] = s;
  }
  for (int j = 0; j < 128; ++j)
    out[threadIdx.x * 128 + j] = d[j];
  out[16384 + threadIdx.x] = float(fill);
  integer_out[threadIdx.x * 2] = seed;
  integer_out[threadIdx.x * 2 + 1] = fill;
}
// D uses two SW128 tensor requests per 48KiB stage and records every CTA interval.
__global__ void r05_supply(const __grid_constant__ CUtensorMap amap,
                           const __grid_constant__ CUtensorMap bmap, int iterations, int stages,
                           int slots, bool independent, R05Stamp* stamps, unsigned* final) {
  extern __shared__ __align__(1024) unsigned char supply_storage[];
  unsigned char* storage = supply_storage;
  unsigned base = r05_shared(storage), bars = base + 196608;
  uint64_t tokens[4] = {};
  unsigned tid = threadIdx.x;
  R05Stamp stamp{};
  if (tid == 0)
    for (int s = 0; s < 4; ++s)
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bars + s * 8) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    stamp.sm = r05_smid();
    stamp.begin_ns = r05_ns();
    stamp.begin_cycle = clock64();
  }
  __syncthreads();
  int issued = 0;
  for (int i = 0; i < iterations; ++i) {
    if (tid == 0)
      while (issued < min(iterations, i + stages)) {
        int slot = issued % stages, panel = (independent ? blockIdx.x * slots : 0) + issued % slots;
        int ax = 0, ay = panel * 128, bx = 0, by = panel * 256;
        unsigned dst = base + slot * 49152;
        asm volatile(
            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0],%1;" ::"r"(bars + slot * 8),
            "r"(49152)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                "r"(dst),
            "l"(&amap), "r"(ax), "r"(ay), "r"(bars + slot * 8)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                "r"(dst + 16384),
            "l"(&bmap), "r"(bx), "r"(by), "r"(bars + slot * 8)
            : "memory");
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0,[%1];"
                     : "=l"(tokens[slot])
                     : "r"(bars + slot * 8)
                     : "memory");
        ++issued;
      }
    if (tid == 0)
      r05_wait(bars + (i % stages) * 8, tokens[i % stages]);
    __syncthreads();
    // One consumer warpgroup joins after acquire; its no-compute join releases the slot.
    __syncthreads();
  }
  if (tid == 0) {
    stamp.end_cycle = clock64();
    stamp.end_ns = r05_ns();
    stamps[blockIdx.x] = stamp;
  }
  for (int i = tid; i < 49152 / 4; i += blockDim.x)
    final[blockIdx.x * 12288 + i] =
        reinterpret_cast<unsigned*>(storage)[((iterations - 1) % stages) * 12288 + i];
  __syncthreads();
  if (tid == 0)
    for (int s = 0; s < 4; ++s)
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(bars + s * 8) : "memory");
}
__global__ void r05_rows(const __grid_constant__ CUtensorMap map, int iterations,
                         R05Stamp* stamps) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto* tile = reinterpret_cast<uint16_t*>(storage);
  for (int i = threadIdx.x; i < 8192; i += 128)
    tile[i] = uint16_t(17 * (i % 64) + 31 * (i / 64) + 151 * blockIdx.x + 17);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  R05Stamp s{};
  if (threadIdx.x == 0) {
    s.sm = r05_smid();
    s.begin_ns = r05_ns();
    s.begin_cycle = clock64();
  }
  __syncthreads();
  for (int i = 0; i < iterations; ++i) {
    if (threadIdx.x == 0) {
      int x = 0, y = (blockIdx.x * 8 + i % 8) * 128;
      asm volatile(
          "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%1,%2}],[%3];" ::"l"(&map),
          "r"(x), "r"(y), "r"(r05_shared(tile))
          : "memory");
      asm volatile("cp.async.bulk.commit_group;cp.async.bulk.wait_group 0;" ::: "memory");
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s.end_cycle = clock64();
    s.end_ns = r05_ns();
    stamps[blockIdx.x] = s;
  }
}
struct R05Options {
  std::string family = "A", protocol = "cp", source = "shared";
  int iterations = 128, filler = 0, stages = 1, consumer = 0, requests = 1, pitch = 128;
  bool control = false, read = false, capture = false;
  R05Options(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc)
        throw std::runtime_error("missing value");
      std::string k = argv[i], v = argv[i + 1];
      if (k == "--family")
        family = v;
      else if (k == "--protocol")
        protocol = v;
      else if (k == "--source")
        source = v;
      else if (k == "--iterations")
        iterations = std::stoi(v);
      else if (k == "--filler")
        filler = std::stoi(v);
      else if (k == "--stages")
        stages = std::stoi(v);
      else if (k == "--consumer")
        consumer = std::stoi(v);
      else if (k == "--requests")
        requests = std::stoi(v);
      else if (k == "--pitch")
        pitch = std::stoi(v);
      else if (k == "--control")
        control = std::stoi(v);
      else if (k == "--read")
        read = std::stoi(v);
      else if (k == "--capture")
        capture = std::stoi(v);
      else
        throw std::runtime_error("unknown option " + k);
    }
    if (iterations < 1 || iterations > 2048 || (stages != 1 && stages != 2 && stages != 4))
      throw std::runtime_error("invalid iterations/stages");
  }
};
CUtensorMap r05_map(void* ptr, int width, int rows, int pitch, int height, bool sw) {
  alignas(64) CUtensorMap map{};
  cuuint64_t dims[] = {unsigned(width), unsigned(rows)}, strides[] = {unsigned(pitch)};
  cuuint32_t box[] = {unsigned(width), unsigned(height)}, step[] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, ptr, dims, strides, box, step,
      CU_TENSOR_MAP_INTERLEAVE_NONE, sw ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS)
    throw std::runtime_error("tensor map encode code " + std::to_string(int(r)));
  return map;
}
template <class T>
void r05_save(const char* path, const std::vector<T>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}
int main(int argc, char** argv) try {
  R05Options o(argc, argv);
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  bool all = o.family == "D" || o.family == "E";
  int blocks = all ? prop.multiProcessorCount : 1, threads = o.family == "D" ? 160 : 128;
  int mode = o.protocol == "cp" ? 0 : o.protocol == "tin" ? 1 : o.protocol == "tout" ? 2 : 3;
  int q = mode == 0 ? 2048 : 16384, words = q / 4;
  int slots = o.family == "D"
                  ? std::max(1, int((o.source == "shared"
                                         ? prop.l2CacheSize / 4
                                         : (4 * prop.l2CacheSize + blocks * 49152 - 1) / blocks) /
                                    49152))
                  : 8;
  int panels = o.family == "D" ? slots * (o.source == "independent" ? blocks : 1) : 8;
  size_t gwords = o.family == "D"   ? size_t(panels) * 49152 / 4
                  : o.family == "E" ? size_t(blocks) * 8 * 128 * o.pitch / 4
                                    : size_t(8) * o.requests * words;
  DeviceBuffer<unsigned> global(gwords),
      trace(mode == 3   ? 256
            : o.capture ? (o.family == "B" ? o.iterations * 8 : o.iterations) * words + 1
                        : 1);
  DeviceBuffer<float> consumed(mode == 3 ? 16512 : 128);
  DeviceBuffer<R05Stamp> stamps(blocks);
  DeviceBuffer<unsigned> final(o.family == "D" ? size_t(blocks) * 12288 : 1);
  std::vector<unsigned> initial(gwords);
  for (size_t i = 0; i < gwords; ++i)
    initial[i] = 17u * unsigned(i) + 17u;
  if (o.family == "E")
    std::fill(initial.begin(), initial.end(), 0xdeadbeef);
  auto reset = [&]() {
    CUDA_CHECK(cudaMemcpy(global.pointer, initial.data(), gwords * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(consumed.pointer, 0xff, consumed.count * 4));
    CUDA_CHECK(cudaMemset(trace.pointer, 0xff, trace.count * 4));
  };
  CUtensorMap amap{}, bmap{}, emap{};
  if (o.family == "D") {
    amap = r05_map(global.pointer, 64, panels * 128, 128, 128, true);
    bmap = r05_map(global.pointer + size_t(panels) * 4096, 64, panels * 256, 128, 256, true);
  }
  if (o.family == "E")
    emap = r05_map(global.pointer, 64, blocks * 8 * 128, o.pitch, 128, false);
  const void* wg = nullptr;
#define R05_SELECT(F)                                               \
  if (o.filler == F)                                                \
    wg = o.control ? reinterpret_cast<const void*>(r05_wg<F, true>) \
                   : reinterpret_cast<const void*>(r05_wg<F, false>);
  R05_SELECT(0)
  R05_SELECT(32) R05_SELECT(128) R05_SELECT(512)
#undef R05_SELECT
      if (!wg) throw std::runtime_error("unsupported filler count");
  const void* kernel = o.family == "D"   ? reinterpret_cast<const void*>(r05_supply)
                       : o.family == "E" ? reinterpret_cast<const void*>(r05_rows)
                       : mode == 3       ? wg
                       : mode == 0       ? reinterpret_cast<const void*>(r05_transport<0>)
                       : mode == 1       ? reinterpret_cast<const void*>(r05_transport<1>)
                                         : reinterpret_cast<const void*>(r05_transport<2>);
  int smem = o.family == "D"   ? 196608 + 1024
             : o.family == "E" ? 16384
             : mode == 3       ? 40960
                               : 4 * q + 128;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  int occupancy;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, threads, smem));
  if (occupancy < 1 || (o.family == "D" && occupancy != 1))
    throw std::runtime_error("invalid occupancy limit");
  auto launch = [&]() {
    reset();
    if (o.family == "D")
      r05_supply<<<blocks, threads, smem>>>(amap, bmap, o.iterations, o.stages, slots,
                                            o.source == "independent", stamps.pointer,
                                            final.pointer);
    else if (o.family == "E")
      r05_rows<<<blocks, threads, smem>>>(emap, o.iterations, stamps.pointer);
    else if (mode == 3) {
      int it = o.iterations;
      float* data = consumed.pointer;
      R05Stamp* ts = stamps.pointer;
      unsigned* ints = trace.pointer;
      void* args[] = {&it, &data, &ints, &ts};
      CUDA_CHECK(cudaLaunchKernel(wg, dim3(1), dim3(128), args, smem));
    } else {
      bool pipeline = o.family == "B";
      if (mode == 0)
        r05_transport<0><<<1, 128, smem>>>(
            global.pointer, o.iterations, o.filler, o.stages, o.consumer, o.requests, o.read,
            pipeline, o.control, o.capture, trace.pointer, consumed.pointer, stamps.pointer);
      else if (mode == 1)
        r05_transport<1><<<1, 128, smem>>>(
            global.pointer, o.iterations, o.filler, o.stages, o.consumer, o.requests, o.read,
            pipeline, o.control, o.capture, trace.pointer, consumed.pointer, stamps.pointer);
      else
        r05_transport<2><<<1, 128, smem>>>(
            global.pointer, o.iterations, o.filler, o.stages, o.consumer, o.requests, o.read,
            pipeline, o.control, o.capture, trace.pointer, consumed.pointer, stamps.pointer);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  };
  std::vector<R05Stamp> times(blocks);
  auto window = [&]() {
    CUDA_CHECK(cudaMemcpy(times.data(), stamps.pointer, blocks * sizeof(R05Stamp),
                          cudaMemcpyDeviceToHost));
    uint64_t begin = UINT64_MAX, end = 0;
    for (auto s : times) {
      begin = std::min(begin, s.begin_ns);
      end = std::max(end, s.end_ns);
    }
    return all ? end - begin : times[0].end_cycle - times[0].begin_cycle;
  };
  std::vector<float> warm;
  bool converged = false;
  if (!o.capture)
    for (int i = 0; i < 30; ++i) {
      launch();
      warm.push_back(float(window()));
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
  std::vector<float> values(consumed.count);
  CUDA_CHECK(
      cudaMemcpy(values.data(), consumed.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
  if (!all)
    r05_save("consumed.f32", values);
  if (o.family == "D") {
    std::vector<unsigned> v(final.count);
    CUDA_CHECK(cudaMemcpy(v.data(), final.pointer, v.size() * 4, cudaMemcpyDeviceToHost));
    r05_save("final.u32", v);
  }
  if (o.family == "E" || mode == 2) {
    std::vector<unsigned> v(gwords);
    CUDA_CHECK(cudaMemcpy(v.data(), global.pointer, gwords * 4, cudaMemcpyDeviceToHost));
    r05_save("global.u32", v);
  }
  if (mode == 3) {
    std::vector<unsigned> v(trace.count);
    CUDA_CHECK(cudaMemcpy(v.data(), trace.pointer, v.size() * 4, cudaMemcpyDeviceToHost));
    r05_save("integers.u32", v);
  }
  if (o.capture && !all && mode < 3) {
    std::vector<unsigned> v(trace.count);
    CUDA_CHECK(cudaMemcpy(v.data(), trace.pointer, v.size() * 4, cudaMemcpyDeviceToHost));
    r05_save("trace.u32", v);
  }
  uint64_t work = o.control         ? 0
                  : o.family == "D" ? uint64_t(blocks) * o.iterations * 49152
                  : o.family == "E" ? uint64_t(blocks) * o.iterations * 16384
                  : mode == 3 ? 0
                              : uint64_t(o.iterations) * (o.family == "B" ? 8 : 1) * q * o.requests;
  std::cout << std::setprecision(17) << "{\"status\":\"measured\",\"family\":\"" << o.family
            << "\",\"protocol\":\"" << o.protocol << "\",\"iterations\":" << o.iterations
            << ",\"filler\":" << o.filler << ",\"stages\":" << o.stages
            << ",\"consumer\":" << o.consumer << ",\"requests\":" << o.requests
            << ",\"pitch\":" << o.pitch << ",\"source\":\"" << o.source
            << "\",\"control\":" << (o.control ? "true" : "false")
            << ",\"read\":" << (o.read ? "true" : "false")
            << ",\"capture\":" << (o.capture ? "true" : "false") << ",\"blocks\":" << blocks
            << ",\"threads\":" << threads << ",\"slots\":" << slots << ",\"panels\":" << panels
            << ",\"global_allocation_bytes\":" << gwords * 4
            << ",\"sm_count\":" << prop.multiProcessorCount << ",\"l2_bytes\":" << prop.l2CacheSize
            << ",\"work_bytes\":" << work << ",\"consumer_flop\":"
            << uint64_t(2) * 128 * o.consumer * o.iterations * (o.family == "B" ? 8 : 1)
            << ",\"elapsed\":" << elapsed << ",\"unit\":\""
            << (all ? "globaltimer_ns/GPU" : "clock64_cycle/CTA")
            << "\",\"registers_per_thread\":" << attr.numRegs
            << ",\"local_bytes_per_thread\":" << attr.localSizeBytes
            << ",\"static_smem_bytes\":" << attr.sharedSizeBytes
            << ",\"dynamic_smem_bytes\":" << smem
            << ",\"occupancy_limit_ctas_per_sm\":" << occupancy
            << ",\"warmup_converged\":" << (converged ? "true" : "false") << ",\"warmup\":[";
  for (size_t i = 0; i < warm.size(); ++i)
    std::cout << (i ? "," : "") << warm[i];
  std::cout << "],\"stamps\":[";
  for (int i = 0; i < blocks; ++i) {
    auto s = times[i];
    std::cout << (i ? "," : "") << '[' << s.begin_cycle << ',' << s.end_cycle << ',' << s.begin_ns
              << ',' << s.end_ns << ',' << s.sm << ',' << s.wait_position << ',' << s.wait_return
              << ',' << s.release << ',' << s.complete << ']';
  }
  std::cout << "]}\n";
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
