#pragma once
// Derived S18 device forms for finite short validation.
// Original feasibility source remains frozen. Pressure has per-kernel maxnreg32.
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

struct AuxiliaryStamp {
  unsigned long long begin_ns, end_ns, begin_cycle, end_cycle;
  unsigned begin_smid, end_smid;
};
__device__ __forceinline__ unsigned long long aux_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__device__ __forceinline__ unsigned aux_smid() {
  unsigned value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}
__device__ __forceinline__ void aux_begin(AuxiliaryStamp* stamp) {
  __syncthreads();
  if (threadIdx.x == 0) {
    stamp->begin_smid = aux_smid();
    stamp->begin_ns = aux_ns();
    stamp->begin_cycle = clock64();
  }
  __syncthreads();
}
__device__ __forceinline__ void aux_end(AuxiliaryStamp* stamp) {
  __syncthreads();
  if (threadIdx.x == 0) {
    stamp->end_cycle = clock64();
    stamp->end_ns = aux_ns();
    stamp->end_smid = aux_smid();
  }
}
__device__ __forceinline__ unsigned aux_initial(unsigned seed, unsigned t, unsigned j) {
  return seed + 65537u * (t + 1u) + 257u * (j + 1u);
}

template<int Words, bool ExplicitLocal>
__device__ __forceinline__ void aux_words(const unsigned* inputs, unsigned* output,
    unsigned* smem_output, unsigned smem_words, unsigned seed, int iterations,
    AuxiliaryStamp* stamp) {
  extern __shared__ unsigned dynamic_words[];
  // The volatile array is instantiated only for explicit-local targets.
  using Word = typename std::conditional<ExplicitLocal, volatile unsigned, unsigned>::type;
  Word values[Words];
  const unsigned t = threadIdx.x;
  if constexpr (ExplicitLocal) {
    #pragma unroll 1
    for (int j = 0; j < Words; ++j) values[j] = inputs[t * Words + j];
  } else {
    #pragma unroll
    for (int j = 0; j < Words; ++j) values[j] = inputs[t * Words + j];
  }
  for (unsigned j = t; j < smem_words; j += blockDim.x)
    dynamic_words[j] = aux_initial(seed, 0, j);
  aux_begin(stamp);
  #pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    if constexpr (ExplicitLocal) {
      // Runtime indexing with an intact loop keeps an addressable private array.
      // Actual LDL/STL and loop placement must still be checked after compilation.
      #pragma unroll 1
      for (int j = 0; j < Words; ++j) {
        unsigned value = values[j];
        asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;" : "+r"(value));
        values[j] = value;
      }
    } else {
      #pragma unroll
      for (int j = 0; j < Words; ++j) {
        unsigned value = values[j];
        asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;" : "+r"(value));
        values[j] = value;
      }
    }
  }
  if constexpr (ExplicitLocal) {
    #pragma unroll 1
    for (int j = 0; j < Words; ++j) output[t * Words + j] = values[j];
  } else {
    #pragma unroll
    for (int j = 0; j < Words; ++j) output[t * Words + j] = values[j];
  }
  aux_end(stamp);
  // Untimed payload drain proves the dynamic-SMEM allocation is used.
  __syncthreads();
  for (unsigned j = t; j < smem_words; j += blockDim.x) smem_output[j] = dynamic_words[j];
}
#define WORD_TARGET(Name, Words, Local) \
extern "C" __global__ void Name(const unsigned* inputs, unsigned* output, \
 unsigned* smem_output, unsigned smem_words, unsigned seed, int iterations, AuxiliaryStamp* stamp) { \
 aux_words<Words, Local>(inputs, output, smem_output, smem_words, seed, iterations, stamp); }

extern "C" __global__ void __maxnreg__(32) s18_pressure32(const unsigned* inputs, unsigned* output, unsigned* smem_output, unsigned smem_words, unsigned seed, int iterations, AuxiliaryStamp* stamp) { aux_words<32,false>(inputs,output,smem_output,smem_words,seed,iterations,stamp); }
extern "C" __global__ void __maxnreg__(32) s18_pressure128(const unsigned* inputs, unsigned* output, unsigned* smem_output, unsigned smem_words, unsigned seed, int iterations, AuxiliaryStamp* stamp) { aux_words<128,false>(inputs,output,smem_output,smem_words,seed,iterations,stamp); }
WORD_TARGET(s18_capacity32, 32, false)
WORD_TARGET(s18_capacity64, 64, false)
WORD_TARGET(s18_capacity128, 128, false)
WORD_TARGET(s18_local32, 32, true)
WORD_TARGET(s18_local128, 128, true)

template<int Operation, int Streams>
__device__ __forceinline__ void aux_service(unsigned seed, int iterations,
    unsigned long long* output, AuxiliaryStamp* stamp) {
  unsigned long long state[Streams], constants[Streams];
  unsigned factors[Streams], salts[Streams];
  const unsigned t = threadIdx.x;
  #pragma unroll
  for (int j = 0; j < Streams; ++j) {
    if constexpr (Operation == 0) {
      state[j] = (static_cast<unsigned long long>(aux_initial(seed, t, j)) << 32)
                   | aux_initial(seed, t, j + 17);
      constants[j] = ((static_cast<unsigned long long>(seed) << 32)
                      + 65537ull * (t + 1) + 257ull * (j + 1)) | 1ull;
    } else if constexpr (Operation == 1) {
      state[j] = aux_initial(seed, t, j);
      factors[j] = aux_initial(seed ^ 0xa5a5a5a5u, t, j) | 1u;
      constants[j] = (static_cast<unsigned long long>(aux_initial(seed ^ 0x5a5a5a5au, t, j)) << 32)
                     | aux_initial(seed ^ 0xc3c3c3c3u, t, j);
    } else {
      state[j] = aux_initial(seed, t, j) & 1023u;
      salts[j] = (aux_initial(seed ^ 0x9e3779b9u, t, j) & 1023u) | 1u;
    }
  }
  aux_begin(stamp);
  #pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    #pragma unroll
    for (int j = 0; j < Streams; ++j) {
      if constexpr (Operation == 0) {
        asm volatile("add.u64 %0, %0, %1;" : "+l"(state[j]) : "l"(constants[j]));
      } else if constexpr (Operation == 1) {
        unsigned x = static_cast<unsigned>(state[j]);
        asm volatile("mad.wide.u32 %0, %1, %2, %3;" : "=l"(state[j])
                     : "r"(x), "r"(factors[j]), "l"(constants[j]));
      } else {
        unsigned previous = static_cast<unsigned>(state[j]);
        if (i != 0) previous = Operation == 2 ? previous & 1023u : (previous >> 13) & 1023u;
        unsigned q = (previous + salts[j] + static_cast<unsigned>(i)) & 1023u;
        if constexpr (Operation == 2) {
          unsigned bits = 0x3f800000u | (q << 13);
          unsigned short result;
          asm volatile("{ .reg .f32 f; mov.b32 f, %1; cvt.rn.f16.f32 %0, f; }"
                       : "=h"(result) : "r"(bits));
          state[j] = result;
        } else {
          unsigned short bits = static_cast<unsigned short>(0x3c00u | q);
          unsigned result;
          asm volatile("{ .reg .f32 f; cvt.f32.f16 f, %1; mov.b32 %0, f; }"
                       : "=r"(result) : "h"(bits));
          state[j] = result;
        }
      }
    }
  }
  #pragma unroll
  for (int j = 0; j < Streams; ++j) output[t * Streams + j] = state[j];
  aux_end(stamp);
}
#define SERVICE_TARGET(Name, Operation, Streams) \
extern "C" __global__ void Name(unsigned seed, int iterations, unsigned long long* output, AuxiliaryStamp* stamp) { \
 aux_service<Operation, Streams>(seed, iterations, output, stamp); }
SERVICE_TARGET(s18_add64_s1, 0, 1)
SERVICE_TARGET(s18_add64_s4, 0, 4)
SERVICE_TARGET(s18_madwide_s1, 1, 1)
SERVICE_TARGET(s18_madwide_s4, 1, 4)
SERVICE_TARGET(s18_cvt_f16_f32_s1, 2, 1)
SERVICE_TARGET(s18_cvt_f16_f32_s4, 2, 4)
SERVICE_TARGET(s18_cvt_f32_f16_s1, 3, 1)
SERVICE_TARGET(s18_cvt_f32_f16_s4, 3, 4)

template<bool Shared, bool SameWord>
__device__ __forceinline__ void aux_atomic(unsigned* global_words, unsigned* output,
    unsigned seed, int iterations, AuxiliaryStamp* stamp) {
  __shared__ unsigned shared_words[Shared ? 128 : 1];
  const unsigned t = threadIdx.x;
  if constexpr (Shared) shared_words[t] = aux_initial(seed, 0, t);
  // Global targets must be initialized by the host before this launch.
  unsigned index = SameWord ? 0 : t;
  unsigned increment = 1u + t % 7u;
  unsigned address = 0;
  if constexpr (Shared) address = static_cast<unsigned>(__cvta_generic_to_shared(shared_words + index));
  aux_begin(stamp);
  #pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    unsigned unused;
    if constexpr (Shared) {
      asm volatile("atom.relaxed.cta.shared::cta.add.u32 %0, [%1], %2;"
                   : "=r"(unused) : "r"(address), "r"(increment) : "memory");
    } else {
      asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;"
                   : "=r"(unused) : "l"(global_words + index), "r"(increment) : "memory");
    }
  }
  __syncthreads();
  if (!SameWord || t == 0) output[t] = Shared ? shared_words[t] : global_words[t];
  aux_end(stamp);
}
#define ATOMIC_TARGET(Name, Shared, Same) \
extern "C" __global__ void Name(unsigned* words, unsigned* output, unsigned seed, int iterations, AuxiliaryStamp* stamp) { \
 aux_atomic<Shared, Same>(words, output, seed, iterations, stamp); }
ATOMIC_TARGET(s18_atomic_global_same, false, true)
ATOMIC_TARGET(s18_atomic_global_independent, false, false)
ATOMIC_TARGET(s18_atomic_shared_same, true, true)
ATOMIC_TARGET(s18_atomic_shared_independent, true, false)
