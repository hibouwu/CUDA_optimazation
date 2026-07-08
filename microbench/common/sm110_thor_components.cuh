#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

#define SM110_CHECK_CUDA(call)                                               \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                << " code=" << static_cast<int>(err__)                      \
                << " name=" << cudaGetErrorName(err__)                      \
                << " msg=" << cudaGetErrorString(err__) << std::endl;       \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#if defined(__CUDA_ARCH_FEAT_SM110_ALL)
#define SM110_HAS_TCGEN05 1
#else
#define SM110_HAS_TCGEN05 0
#endif

#ifndef TC3_SM110_HOST_HAS_TCGEN05
#define TC3_SM110_HOST_HAS_TCGEN05 0
#endif

namespace sm110_thor_components {

struct TmemProbeShape {
  static constexpr int kThreads = 128;
  static constexpr unsigned int kTmemColumns = 128;
};

struct WorkTile {
  int tile_id;
  int valid;
};

__host__ __device__ constexpr bool tcgen05_available() {
#if defined(__CUDA_ARCH__)
  return SM110_HAS_TCGEN05 != 0;
#else
  return TC3_SM110_HOST_HAS_TCGEN05 != 0;
#endif
}

__device__ __forceinline__ unsigned int smem_u32_ptr(const void* p) {
  return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

inline bool report_cuda(const char* label, cudaError_t status) {
  if (status == cudaSuccess) {
    std::cout << label << ": ok\n";
    return true;
  }
  std::cerr << label << ": code=" << static_cast<int>(status)
            << " name=" << cudaGetErrorName(status)
            << " msg=" << cudaGetErrorString(status) << '\n';
  return false;
}

inline bool print_device_or_skip() {
  SM110_CHECK_CUDA(cudaFree(0));

  int device = 0;
  SM110_CHECK_CUDA(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  SM110_CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "GPU: " << prop.name << '\n';
  std::cout << "compute capability: " << prop.major << "." << prop.minor
            << '\n';

  if (prop.major != 11) {
    std::cout << "Not SM110-class device. Skip TCGen05/TMEM probe.\n";
    return false;
  }
  if (!tcgen05_available()) {
    std::cout << "This binary was not built with sm110a TCGen05 enabled. "
                 "Skip TCGen05/TMEM probe.\n";
    return false;
  }
  return true;
}

template <typename Shape = TmemProbeShape>
__device__ __forceinline__ void tmem_alloc(unsigned int* tmem_base) {
#if SM110_HAS_TCGEN05
  if (threadIdx.x == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_u32_ptr(tmem_base)), "r"(Shape::kTmemColumns)
        : "memory");
  }
#endif
}

template <typename Shape = TmemProbeShape>
__device__ __forceinline__ void tmem_dealloc(unsigned int tmem_base) {
#if SM110_HAS_TCGEN05
  if (threadIdx.x == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(tmem_base), "r"(Shape::kTmemColumns)
                 : "memory");
  }
#endif
}

__device__ __forceinline__ void tmem_store_load_one(unsigned int tmem_base,
                                                    float* out,
                                                    int index) {
#if SM110_HAS_TCGEN05
  if (threadIdx.x == 0) {
    const unsigned int one_bits = 0x3f800000u;
    asm volatile("tcgen05.st.sync.aligned.16x64b.x1.b32 [%0], {%1};"
                 :
                 : "r"(tmem_base), "r"(one_bits)
                 : "memory");

    unsigned int loaded = 0u;
    asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%0}, [%1];"
                 : "=r"(loaded)
                 : "r"(tmem_base)
                 : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    float value = 0.0f;
    *reinterpret_cast<unsigned int*>(&value) = loaded;
    out[index] = value;
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = 0.0f;
  }
#endif
}

template <typename Shape = TmemProbeShape>
__global__ void tcgen05_tmem_probe_kernel(float* out) {
  __shared__ unsigned int tmem_base;
  (void)tmem_base;

#if SM110_HAS_TCGEN05
  tmem_alloc<Shape>(&tmem_base);
  __syncthreads();
  tmem_store_load_one(tmem_base, out, 0);
  __syncthreads();
  tmem_dealloc<Shape>(tmem_base);
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = 0.0f;
  }
#endif
}

__device__ __forceinline__ WorkTile static_fetch(int worker_id,
                                                 int iter,
                                                 int total_tiles) {
  const int tile_id = worker_id + iter * gridDim.x;
  return {tile_id, tile_id < total_tiles ? 1 : 0};
}

__device__ __forceinline__ WorkTile dynamic_fetch(int* work_counter,
                                                  int total_tiles) {
  const int tile_id = atomicAdd(work_counter, 1);
  return {tile_id, tile_id < total_tiles ? 1 : 0};
}

template <typename Shape = TmemProbeShape>
__global__ void clc_static_tmem_probe_kernel(float* out, int total_tiles) {
  __shared__ unsigned int tmem_base;
  __shared__ WorkTile work;
  (void)tmem_base;
  (void)work;

#if SM110_HAS_TCGEN05
  tmem_alloc<Shape>(&tmem_base);
  __syncthreads();

  for (int iter = 0;; ++iter) {
    if (threadIdx.x == 0) {
      work = static_fetch(static_cast<int>(blockIdx.x), iter, total_tiles);
    }
    __syncthreads();
    if (!work.valid) break;
    tmem_store_load_one(tmem_base, out, work.tile_id);
    __syncthreads();
  }

  tmem_dealloc<Shape>(tmem_base);
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = 0.0f;
  }
#endif
}

template <typename Shape = TmemProbeShape>
__global__ void clc_dynamic_tmem_probe_kernel(float* out,
                                              int total_tiles,
                                              int* work_counter) {
  __shared__ unsigned int tmem_base;
  __shared__ WorkTile work;
  (void)tmem_base;
  (void)work;

#if SM110_HAS_TCGEN05
  tmem_alloc<Shape>(&tmem_base);
  __syncthreads();

  while (true) {
    if (threadIdx.x == 0) {
      work = dynamic_fetch(work_counter, total_tiles);
    }
    __syncthreads();
    if (!work.valid) break;
    tmem_store_load_one(tmem_base, out, work.tile_id);
    __syncthreads();
  }

  tmem_dealloc<Shape>(tmem_base);
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = 0.0f;
  }
#endif
}

inline bool matched_one(float value) {
  return std::abs(value - 1.0f) < 1e-6f;
}

}  // namespace sm110_thor_components

