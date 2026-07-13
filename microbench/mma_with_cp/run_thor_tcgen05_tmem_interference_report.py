#!/usr/bin/env python3
import argparse
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import run_thor_tcgen05_cp_mma_report as base


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "benchmark_src_tmem_interference"
BUILD_DIR = ROOT / "build_tmem_interference"
REPORT_PATH = ROOT / "tmem_interference_report.txt"

DEFAULT_SHAPES = ["m128n256"]
DEFAULT_PRECISIONS = ["FP4", "BF16"]
DEFAULT_NOISE = [1, 2, 4, 8]
DEFAULT_REPEATS = 50
K_BLOCKS = 16


CU_TEMPLATE = r'''
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                           \
  cudaError_t err__ = (call);                                           \
  if (err__ != cudaSuccess) {                                           \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                 cudaGetErrorString(err__));                            \
    std::exit(1);                                                       \
  }                                                                     \
} while (0)

static constexpr long long kMacPerInst = {mac_per_inst}LL;
static constexpr int kBlockThreads = 128;
static constexpr int kWarpsPerBlock = 4;
static constexpr int kTmemAllocCols = 512;
static constexpr int kMainloopKBlocks = {mainloop_k_blocks};
static constexpr int kNoiseCpPerMma = {noise_cp_per_mma};
static constexpr int kRequiredCpPerMma = {required_cp_per_mma};
static constexpr int kCommitArriveCount = {commit_arrive_count};
static constexpr int kInitialBarrierArriveCount = {initial_barrier_arrive_count};
static constexpr int kMmaAPanelBytes = {mma_a_panel_bytes};
static constexpr int kMmaBPanelBytes = {mma_b_panel_bytes};
static constexpr int kDescLeading = {desc_leading};
static constexpr int kDescStride = {desc_stride};
static constexpr int kEffectiveBytesPerCp = {effective_bytes_per_cp};
static constexpr char kCaseId[] = "{case_id}";
static constexpr char kCaseLabel[] = "{case_label}";
static constexpr char kMmaPath[] = "{mma_path}";
static constexpr char kIssueMode[] = "{issue_mode}";
static constexpr char kPrecision[] = "{precision}";
static constexpr char kShape[] = "{shape_label}";
static constexpr char kCpSuffix[] = "{cp_suffix}";

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool same_issuer_thread() {
  return threadIdx.x == 0;
}

__device__ __forceinline__ bool split_cp_issuer_thread() {
  return threadIdx.x == 0;
}

__device__ __forceinline__ bool split_mma_issuer_thread() {
  return threadIdx.x == 32;
}

__device__ __forceinline__ bool split_issuer_thread() {
  return split_cp_issuer_thread() || split_mma_issuer_thread();
}

__device__ __forceinline__ uint64_t make_smem_desc(void const* ptr, uint32_t leading_u128, uint32_t stride_u128) {
  uint32_t addr = smem_u32(ptr);
  uint64_t desc = 0;
  desc |= uint64_t((addr >> 4) & 0x3fff);
  desc |= uint64_t(leading_u128 & 0x3fff) << 16;
  desc |= uint64_t(stride_u128 & 0x3fff) << 32;
  desc |= uint64_t(1) << 46;
  return desc;
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier, uint32_t arrive_count) {
  uint32_t addr = smem_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(arrive_count));
}

__device__ __forceinline__ void barrier_wait(uint64_t* barrier, uint32_t phase) {
  uint32_t addr = smem_u32(barrier);
  uint32_t ticks = 0x989680;
  asm volatile(
      "{ .reg .pred p; wait_loop: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
      "@p bra wait_done; bra wait_loop; wait_done: }"
      :: "r"(addr), "r"(phase), "r"(ticks));
}

__device__ __forceinline__ void commit_and_wait_from(uint64_t* barrier, uint32_t phase, bool should_commit) {
  uint32_t bar_addr = smem_u32(barrier);
  if (should_commit) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar_addr));
  }
  if (threadIdx.x == 0) {
    barrier_wait(barrier, phase);
  }
  __syncthreads();
}

__device__ __forceinline__ uint64_t make_idesc_bf16() {
  uint32_t d = 0;
  d |= 1u << 4;
  d |= 1u << 7;
  d |= 1u << 10;
  d |= {n_desc_units}u << 17;
  d |= {m_desc_units}u << 24;
  return uint64_t(d) << 32;
}

__device__ __forceinline__ uint64_t make_idesc_fp8() {
  uint32_t d = 0;
  d |= 1u << 4;
  d |= {n_desc_units}u << 17;
  d |= {m_desc_units}u << 24;
  return uint64_t(d) << 32;
}

__device__ __forceinline__ uint64_t make_idesc_fp4() {
  uint32_t d = 0;
  d |= 5u << 7;
  d |= 5u << 10;
  d |= {n_desc_units}u << 17;
  d |= 0u << 23;
  d |= {m_desc_units}u << 24;
  return uint64_t(d) << 32;
}

__device__ __forceinline__ void issue_ss_mma(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint64_t idesc,
    uint32_t tsfa, uint32_t tsfb, uint32_t scale) {
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
    {ss_mma_asm}
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(uint32_t(idesc >> 32)), "r"(scale),
       {extra_operands});
}

__device__ __forceinline__ void issue_ts_mma(
    uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint64_t idesc,
    uint32_t tsfa, uint32_t tsfb, uint32_t scale) {
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
    {ts_mma_asm}
    :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(uint32_t(idesc >> 32)), "r"(scale),
       {extra_operands});
}

__device__ __forceinline__ void issue_cp(uint32_t taddr, uint64_t s_desc) {
  asm volatile("{cp_asm}" :: "r"(taddr), "l"(s_desc) : "memory");
}

__global__ __launch_bounds__(128, 1)
void tcgen05_kernel(int iters, unsigned long long* cycles_out) {
  __shared__ alignas(16) uint8_t smem_a[{smem_a_bytes}];
  __shared__ alignas(16) uint8_t smem_b[{smem_b_bytes}];
  __shared__ alignas(8) uint64_t done_barrier;
  __shared__ uint32_t tmem_base;

  for (int i = threadIdx.x; i < int(sizeof(smem_a)); i += blockDim.x) {
    smem_a[i] = uint8_t(i * 13 + 1);
  }
  for (int i = threadIdx.x; i < int(sizeof(smem_b)); i += blockDim.x) {
    smem_b[i] = uint8_t(i * 17 + 3);
  }

  if (threadIdx.x == 0) {
    barrier_init(&done_barrier, kInitialBarrierArriveCount);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    uint32_t dst = smem_u32(&tmem_base);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(dst), "r"(kTmemAllocCols));
  }
  __syncthreads();

  uint64_t a_desc = make_smem_desc(smem_a, kDescLeading, kDescStride);
  uint64_t b_desc = make_smem_desc(smem_b, kDescLeading, kDescStride);
  uint64_t idesc = {idesc_func}();
  uint32_t d_tmem = tmem_base;
  uint32_t a_tmem0 = tmem_base + 256;
  uint32_t a_tmem1 = tmem_base + 320;
{tmem_setup}

  uint32_t phase = 0;
{pre_timing_body}

  __syncthreads();
  unsigned long long start = clock64();

{timed_body}

  unsigned long long stop = clock64();
  if (threadIdx.x == 0) {
    cycles_out[blockIdx.x] = stop - start;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_base), "r"(kTmemAllocCols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
  }
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? std::atoi(argv[1]) : 10000;
  double freq_hz = argc > 2 ? std::atof(argv[2]) : 1575000000.0;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount;
  int active_blocks = prop.multiProcessorCount;

  unsigned long long* d_cycles = nullptr;
  unsigned long long* h_cycles = new unsigned long long[blocks];
  CUDA_CHECK(cudaMalloc(&d_cycles, blocks * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_cycles, 0, blocks * sizeof(unsigned long long)));

  tcgen05_kernel<<<blocks, kBlockThreads>>>(iters, d_cycles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_cycles, d_cycles, blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  unsigned long long max_cycles = 0;
  for (int i = 0; i < blocks; ++i) {
    max_cycles = max_cycles > h_cycles[i] ? max_cycles : h_cycles[i];
  }

  long long mma_instruction_count =
      static_cast<long long>(active_blocks) * static_cast<long long>(kMainloopKBlocks) * static_cast<long long>(iters);
  long long required_cp_instruction_count =
      static_cast<long long>(active_blocks) * static_cast<long long>(kMainloopKBlocks) *
      static_cast<long long>(kRequiredCpPerMma) * static_cast<long long>(iters);
  long long noise_cp_instruction_count =
      static_cast<long long>(active_blocks) * static_cast<long long>(kMainloopKBlocks) *
      static_cast<long long>(kNoiseCpPerMma) * static_cast<long long>(iters);
  long long cp_instruction_count = required_cp_instruction_count + noise_cp_instruction_count;
  long long k_groups = static_cast<long long>(active_blocks) * static_cast<long long>(iters);
  double elapsed_seconds = double(max_cycles) / freq_hz;
  double tflops = 2.0 * double(kMacPerInst) * double(mma_instruction_count) / elapsed_seconds / 1.0e12;
  double bytes_per_cycle = cp_instruction_count > 0
      ? double(cp_instruction_count) * double(kEffectiveBytesPerCp) / double(max_cycles)
      : 0.0;
  double cycles_per_cp = cp_instruction_count > 0
      ? double(max_cycles) / double(cp_instruction_count)
      : 0.0;
  double cycles_per_cta_iter = double(max_cycles) / double(iters);
  double cycles_per_mma = double(max_cycles) / double(static_cast<long long>(iters) * kMainloopKBlocks);

  std::printf("case_id=%s\n", kCaseId);
  std::printf("case_label=%s\n", kCaseLabel);
  std::printf("mma_path=%s\n", kMmaPath);
  std::printf("issue_mode=%s\n", kIssueMode);
  std::printf("precision=%s\n", kPrecision);
  std::printf("shape=%s\n", kShape);
  std::printf("cp_suffix=%s\n", kCpSuffix);
  std::printf("sm_count=%d\n", prop.multiProcessorCount);
  std::printf("active_blocks=%d\n", active_blocks);
  std::printf("block_threads=%d\n", kBlockThreads);
  std::printf("warps_per_block=%d\n", kWarpsPerBlock);
  std::printf("commit_arrive_count=%d\n", kCommitArriveCount);
  std::printf("iters=%d\n", iters);
  std::printf("k_blocks=%d\n", kMainloopKBlocks);
  std::printf("noise_cp_per_mma=%d\n", kNoiseCpPerMma);
  std::printf("required_cp_per_mma=%d\n", kRequiredCpPerMma);
  std::printf("cycles=%llu\n", max_cycles);
  std::printf("mma_instruction_count=%lld\n", mma_instruction_count);
  std::printf("required_cp_instruction_count=%lld\n", required_cp_instruction_count);
  std::printf("noise_cp_instruction_count=%lld\n", noise_cp_instruction_count);
  std::printf("cp_instruction_count=%lld\n", cp_instruction_count);
  std::printf("k_groups=%lld\n", k_groups);
  std::printf("effective_bytes_per_cp=%d\n", kEffectiveBytesPerCp);
  std::printf("thor_tflops=%.6f\n", tflops);
  std::printf("bytes_per_cycle=%.6f\n", bytes_per_cycle);
  std::printf("cycles_per_cp=%.6f\n", cycles_per_cp);
  std::printf("cycles_per_cta_iter=%.6f\n", cycles_per_cta_iter);
  std::printf("cycles_per_mma=%.6f\n", cycles_per_mma);

  CUDA_CHECK(cudaFree(d_cycles));
  delete[] h_cycles;
  return 0;
}
'''


def log(message):
    print(message, flush=True)


def split_words(value):
    value = value.replace(",", " ")
    return [x.strip() for x in value.split() if x.strip()]


def ensure_dirs():
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


def matrix_panel_bytes(precision, extent, k):
    return base.matrix_panel_bytes(precision, extent, k)


def make_cases(noise_values):
    cases = []
    cases.append({
        "case_id": "ss_mainloop_k16_base",
        "label": "SS mainloop K16 baseline",
        "path": "SS",
        "mode": "same",
        "noise": 0,
        "baseline": None,
    })
    cases.append({
        "case_id": "ss_mainloop_k16_split_base",
        "label": "SS mainloop K16 split-issuer control",
        "path": "SS",
        "mode": "split",
        "noise": 0,
        "baseline": None,
    })
    cases.append({
        "case_id": "ts_mainloop_a2_k16_base",
        "label": "TS A2 mainloop K16 baseline",
        "path": "TS",
        "mode": "same",
        "noise": 0,
        "baseline": None,
    })
    cases.append({
        "case_id": "ts_mainloop_a2_k16_split_base",
        "label": "TS A2 mainloop K16 split-issuer control",
        "path": "TS",
        "mode": "split",
        "noise": 0,
        "baseline": None,
    })
    for noise in noise_values:
        cases.append({
            "case_id": f"ss_mainloop_k16_cp_noise_same_n{noise}",
            "label": f"SS mainloop K16 + cp noise same issuer n{noise}",
            "path": "SS",
            "mode": "same",
            "noise": noise,
            "baseline": "ss_mainloop_k16_base",
        })
        cases.append({
            "case_id": f"ss_mainloop_k16_cp_noise_split_n{noise}",
            "label": f"SS mainloop K16 + cp noise split issuer n{noise}",
            "path": "SS",
            "mode": "split",
            "noise": noise,
            "baseline": "ss_mainloop_k16_split_base",
        })
        cases.append({
            "case_id": f"ts_mainloop_a2_k16_extra_cp_same_n{noise}",
            "label": f"TS A2 mainloop K16 + extra cp same issuer n{noise}",
            "path": "TS",
            "mode": "same",
            "noise": noise,
            "baseline": "ts_mainloop_a2_k16_base",
        })
        cases.append({
            "case_id": f"ts_mainloop_a2_k16_extra_cp_split_n{noise}",
            "label": f"TS A2 mainloop K16 + extra cp split issuer n{noise}",
            "path": "TS",
            "mode": "split",
            "noise": noise,
            "baseline": "ts_mainloop_a2_k16_split_base",
        })
    return cases


def case_by_id(cases):
    return {case["case_id"]: case for case in cases}


def pre_timing_body(spec):
    if spec["path"] != "TS":
        return ""
    if spec["mode"] == "same":
        return """
  uint64_t a_desc_first = make_smem_desc(smem_a, kDescLeading, kDescStride);
  if (same_issuer_thread()) {
    issue_cp(a_tmem0, a_desc_first);
  }
  commit_and_wait_from(&done_barrier, phase, same_issuer_thread());
  phase ^= 1;
"""
    return """
  uint64_t a_desc_first = make_smem_desc(smem_a, kDescLeading, kDescStride);
  if (split_cp_issuer_thread()) {
    issue_cp(a_tmem0, a_desc_first);
  }
  commit_and_wait_from(&done_barrier, phase, split_cp_issuer_thread());
  phase ^= 1;
  if (threadIdx.x == 0) {
    barrier_init(&done_barrier, kCommitArriveCount);
  }
  __syncthreads();
  phase = 0;
"""


def timed_body(spec):
    path = spec["path"]
    mode = spec["mode"]
    if path == "SS" and mode == "same":
        return """
  for (int i = 0; i < iters; ++i) {
    for (int k_block = 0; k_block < kMainloopKBlocks; ++k_block) {
      uint32_t scale = ((i == 0) && (k_block == 0)) ? 0u : 1u;
      uint64_t a_desc_k = make_smem_desc(
          smem_a + k_block * uint32_t(kMmaAPanelBytes), kDescLeading, kDescStride);
      uint64_t b_desc_k = make_smem_desc(
          smem_b + k_block * uint32_t(kMmaBPanelBytes), kDescLeading, kDescStride);
      if (same_issuer_thread()) {
        for (int noise = 0; noise < kNoiseCpPerMma; ++noise) {
          uint32_t dst = (noise & 1) ? a_tmem1 : a_tmem0;
          issue_cp(dst, a_desc_k);
        }
        issue_ss_mma(d_tmem, a_desc_k, b_desc_k, idesc, tsfa, tsfb, scale);
      }
    }
    commit_and_wait_from(&done_barrier, phase, same_issuer_thread());
    phase ^= 1;
  }
"""
    if path == "SS" and mode == "split":
        return """
  for (int i = 0; i < iters; ++i) {
    for (int k_block = 0; k_block < kMainloopKBlocks; ++k_block) {
      uint32_t scale = ((i == 0) && (k_block == 0)) ? 0u : 1u;
      uint64_t a_desc_k = make_smem_desc(
          smem_a + k_block * uint32_t(kMmaAPanelBytes), kDescLeading, kDescStride);
      uint64_t b_desc_k = make_smem_desc(
          smem_b + k_block * uint32_t(kMmaBPanelBytes), kDescLeading, kDescStride);
      if (split_cp_issuer_thread()) {
        for (int noise = 0; noise < kNoiseCpPerMma; ++noise) {
          uint32_t dst = (noise & 1) ? a_tmem1 : a_tmem0;
          issue_cp(dst, a_desc_k);
        }
      }
      if (split_mma_issuer_thread()) {
        issue_ss_mma(d_tmem, a_desc_k, b_desc_k, idesc, tsfa, tsfb, scale);
      }
    }
    commit_and_wait_from(&done_barrier, phase, split_issuer_thread());
    phase ^= 1;
  }
"""
    if path == "TS" and mode == "same":
        return """
  for (int i = 0; i < iters; ++i) {
    for (int k_block = 0; k_block < kMainloopKBlocks; ++k_block) {
      int next_k_block = (k_block + 1 == kMainloopKBlocks) ? 0 : (k_block + 1);
      int global_k_block = i * kMainloopKBlocks + k_block;
      uint32_t current = (global_k_block & 1) ? a_tmem1 : a_tmem0;
      uint32_t next = ((global_k_block + 1) & 1) ? a_tmem1 : a_tmem0;
      uint32_t scale = ((i == 0) && (k_block == 0)) ? 0u : 1u;
      uint64_t a_desc_next = make_smem_desc(
          smem_a + next_k_block * uint32_t(kMmaAPanelBytes), kDescLeading, kDescStride);
      uint64_t b_desc_k = make_smem_desc(
          smem_b + k_block * uint32_t(kMmaBPanelBytes), kDescLeading, kDescStride);
      if (same_issuer_thread()) {
        for (int copy = 0; copy < kRequiredCpPerMma + kNoiseCpPerMma; ++copy) {
          issue_cp(next, a_desc_next);
        }
        issue_ts_mma(d_tmem, current, b_desc_k, idesc, tsfa, tsfb, scale);
      }
      commit_and_wait_from(&done_barrier, phase, same_issuer_thread());
      phase ^= 1;
    }
  }
"""
    if path == "TS" and mode == "split":
        return """
  for (int i = 0; i < iters; ++i) {
    for (int k_block = 0; k_block < kMainloopKBlocks; ++k_block) {
      int next_k_block = (k_block + 1 == kMainloopKBlocks) ? 0 : (k_block + 1);
      int global_k_block = i * kMainloopKBlocks + k_block;
      uint32_t current = (global_k_block & 1) ? a_tmem1 : a_tmem0;
      uint32_t next = ((global_k_block + 1) & 1) ? a_tmem1 : a_tmem0;
      uint32_t scale = ((i == 0) && (k_block == 0)) ? 0u : 1u;
      uint64_t a_desc_next = make_smem_desc(
          smem_a + next_k_block * uint32_t(kMmaAPanelBytes), kDescLeading, kDescStride);
      uint64_t b_desc_k = make_smem_desc(
          smem_b + k_block * uint32_t(kMmaBPanelBytes), kDescLeading, kDescStride);
      if (split_cp_issuer_thread()) {
        for (int copy = 0; copy < kRequiredCpPerMma + kNoiseCpPerMma; ++copy) {
          issue_cp(next, a_desc_next);
        }
      }
      if (split_mma_issuer_thread()) {
        issue_ts_mma(d_tmem, current, b_desc_k, idesc, tsfa, tsfb, scale);
      }
      commit_and_wait_from(&done_barrier, phase, split_issuer_thread());
      phase ^= 1;
    }
  }
"""
    raise KeyError((path, mode))


def make_cfg(spec, precision, shape):
    p = base.PRECISIONS[precision]
    s = base.SHAPES[shape]
    shape_k = s["k_by_precision"][precision]
    mma_a_panel_bytes = matrix_panel_bytes(precision, s["m"], shape_k)
    mma_b_panel_bytes = matrix_panel_bytes(precision, s["n"], shape_k)
    smem_a_bytes = max(32768, mma_a_panel_bytes * K_BLOCKS)
    smem_b_bytes = max(32768, mma_b_panel_bytes * K_BLOCKS)
    is_split = spec["mode"] == "split"
    is_ts = spec["path"] == "TS"
    required_cp_per_mma = 1 if is_ts else 0
    needs_prefill = is_ts
    return {
        "case_id": spec["case_id"],
        "case_label": spec["label"],
        "mma_path": spec["path"],
        "issue_mode": spec["mode"],
        "precision": precision,
        "shape_label": f"{s['label']}K{shape_k}",
        "shape_k": shape_k,
        "m_desc_units": s["m_desc_units"],
        "n_desc_units": s["n_desc_units"],
        "mac_per_inst": s["m"] * s["n"] * shape_k,
        "mainloop_k_blocks": K_BLOCKS,
        "noise_cp_per_mma": spec["noise"],
        "required_cp_per_mma": required_cp_per_mma,
        "commit_arrive_count": 2 if is_split else 1,
        "initial_barrier_arrive_count": 1 if needs_prefill else (2 if is_split else 1),
        "mma_a_panel_bytes": mma_a_panel_bytes,
        "mma_b_panel_bytes": mma_b_panel_bytes,
        "smem_a_bytes": smem_a_bytes,
        "smem_b_bytes": smem_b_bytes,
        "desc_leading": p["desc_leading"],
        "desc_stride": p["desc_stride"],
        "ss_mma_asm": p["ss_mma_asm"],
        "ts_mma_asm": p["ts_mma_asm"],
        "idesc_func": p["idesc_func"],
        "extra_operands": p["extra_operands"],
        "tmem_setup": p["tmem_setup"] or "uint32_t tsfa = 0; uint32_t tsfb = 0;",
        "cp_suffix": p["cp_suffix"],
        "cp_asm": f"tcgen05.cp.cta_group::1.{p['cp_suffix']} [%0], %1;",
        "effective_bytes_per_cp": p["effective_bytes_per_cp"],
        "pre_timing_body": pre_timing_body(spec),
        "timed_body": timed_body(spec),
    }


def render_template(cfg):
    text = CU_TEMPLATE
    for key, value in cfg.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def source_name(case_id, precision, shape):
    return f"tcgen05_tmem_interf_{case_id}_{shape}_{precision.lower()}_benchmark.cu"


def binary_name(case_id, precision, shape):
    return f"tcgen05_tmem_interf_{case_id}_{shape}_{precision.lower()}_benchmark"


def iter_keys(cases, shapes, precisions):
    for spec in cases:
        for shape in shapes:
            for precision in precisions:
                yield spec, precision, shape


def generate_sources(cases, shapes, precisions):
    paths = {}
    for spec, precision, shape in iter_keys(cases, shapes, precisions):
        cfg = make_cfg(spec, precision, shape)
        src = SRC_DIR / source_name(spec["case_id"], precision, shape)
        src.write_text(render_template(cfg))
        paths[(spec["case_id"], precision, shape)] = src
    return paths


def compile_sources(srcs, ccbin=None):
    bins = {}
    for key, src in srcs.items():
        case_id, precision, shape = key
        binary = BUILD_DIR / binary_name(case_id, precision, shape)
        cmd = [
            "nvcc",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-gencode",
            "arch=compute_110a,code=sm_110a",
            src,
            "-o",
            binary,
        ]
        if ccbin:
            cmd[1:1] = ["-ccbin", ccbin]
        base.run(cmd)
        bins[key] = binary
    return bins


def inspect_instructions(cases, srcs, bins):
    cases_map = case_by_id(cases)
    checks = {}
    for key in srcs:
        case_id, precision, shape = key
        spec = cases_map[case_id]
        cfg = make_cfg(spec, precision, shape)
        ret, sass = base.run(["cuobjdump", "--dump-sass", bins[key]], capture=True, check=False, echo=False)
        mma_lines = [line.strip() for line in sass.splitlines() if re.search(r"\bUTC[A-Z0-9.]*MMA", line)]
        cp_lines = [line.strip() for line in sass.splitlines() if re.search(r"\bUTCCP(?:\.|\s)", line)]
        expected_mma = base.PRECISIONS[precision]["sass_instruction"].split(".")[0]
        needs_cp = spec["path"] == "TS" or spec["noise"] > 0
        expected_cp_tokens = base.cp_sass_tokens(cfg["cp_suffix"]) if needs_cp else ()
        mma_ok = any(expected_mma in line for line in mma_lines)
        cp_ok = (not needs_cp) or any(base.has_all_tokens(line, expected_cp_tokens) for line in cp_lines)
        checks[key] = {
            "status": "ok" if ret == 0 and mma_ok and cp_ok else "check_failed",
            "expected_mma_sass": base.PRECISIONS[precision]["sass_instruction"],
            "expected_cp_sass": " ".join(expected_cp_tokens),
            "mma_sample": mma_lines[:1],
            "cp_sample": cp_lines[:1],
        }
    return checks


def parse_result(text):
    raw = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            raw[key.strip()] = value.strip()
    required = [
        "case_id", "case_label", "mma_path", "issue_mode", "precision", "shape",
        "sm_count", "active_blocks", "iters", "k_blocks", "noise_cp_per_mma",
        "required_cp_per_mma", "cycles", "mma_instruction_count",
        "required_cp_instruction_count", "noise_cp_instruction_count",
        "cp_instruction_count", "thor_tflops", "bytes_per_cycle",
        "cycles_per_cp", "cycles_per_cta_iter", "cycles_per_mma",
    ]
    missing = [field for field in required if field not in raw]
    if missing:
        raise RuntimeError(f"missing benchmark fields: {missing}\n{text}")
    result = dict(raw)
    for key in [
        "sm_count", "active_blocks", "iters", "k_blocks", "noise_cp_per_mma",
        "required_cp_per_mma", "cycles", "mma_instruction_count",
        "required_cp_instruction_count", "noise_cp_instruction_count",
        "cp_instruction_count",
    ]:
        result[key] = int(result[key])
    for key in ["thor_tflops", "bytes_per_cycle", "cycles_per_cp", "cycles_per_cta_iter", "cycles_per_mma"]:
        result[key] = float(result[key])
    return result


def median_value(values):
    return float(statistics.median(values))


def summarize_samples(samples):
    first = samples[0]
    summary = dict(first)
    median_keys = [
        "cycles",
        "thor_tflops",
        "bytes_per_cycle",
        "cycles_per_cp",
        "cycles_per_cta_iter",
        "cycles_per_mma",
    ]
    for key in median_keys:
        summary[key] = median_value([sample[key] for sample in samples])
    summary["repeat_count"] = len(samples)
    summary["thor_tflops_min"] = min(sample["thor_tflops"] for sample in samples)
    summary["thor_tflops_max"] = max(sample["thor_tflops"] for sample in samples)
    summary["cycles_per_mma_min"] = min(sample["cycles_per_mma"] for sample in samples)
    summary["cycles_per_mma_max"] = max(sample["cycles_per_mma"] for sample in samples)
    return summary


def run_benchmarks(bins, iters, freq_hz, repeats):
    results = {}
    total = len(bins)
    for idx, (key, binary) in enumerate(bins.items(), 1):
        log(f"Running {idx}/{total}: {key} x {repeats}")
        samples = []
        for _ in range(repeats):
            _, out = base.run([binary, str(iters), str(freq_hz)])
            samples.append(parse_result(out))
        results[key] = summarize_samples(samples)
    return results


def official_tflops(precision, active_blocks, sm_count):
    return base.official_tflops(precision, active_blocks, sm_count)


def ratio(num, den):
    return 0.0 if den == 0 else num / den


def write_report(cases, shapes, precisions, results, checks, dev_name, cc, sm_count, freq_hz, iters, repeats, noise_values):
    cases_map = case_by_id(cases)
    lines = []
    lines.append("Thor tcgen05 TMEM interference microbenchmark")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    lines.append("Device")
    lines.append(f"GPU: {dev_name}")
    lines.append(f"Compute Capability: {cc}")
    lines.append(f"SM count: {sm_count}")
    lines.append(f"Frequency: {freq_hz / 1e9:.3f} GHz")
    lines.append(f"Iters: {iters}")
    lines.append(f"Repeats: {repeats}")
    lines.append("Reported values: median across repeats; min/max columns show the TFLOP/s sample range.")
    lines.append(f"Shapes: {', '.join(base.SHAPES[s]['label'] for s in shapes)}")
    lines.append(f"Precisions: {', '.join(precisions)}")
    lines.append(f"Noise cp per MMA: {', '.join(str(x) for x in noise_values)}")
    lines.append("")
    lines.append("Case semantics")
    lines.append("- SS same baseline: one issuer sends 16 SS MMA instructions, then waits once.")
    lines.append("- SS same noise: the same issuer sends N unrelated tcgen05.cp writes into unused A slots before each SS MMA.")
    lines.append("- SS split control: threadIdx.x==32 sends SS MMA; threadIdx.x==0 only participates in commit. This isolates split-issuer wait overhead.")
    lines.append("- SS split noise: threadIdx.x==0 sends the unrelated cp noise while threadIdx.x==32 sends SS MMA.")
    lines.append("- TS same baseline: one issuer sends cp(next A slot) plus TS MMA(current A slot), then waits.")
    lines.append("- TS same noise: the same issuer sends extra copies into the next A slot before TS MMA.")
    lines.append("- TS split control: threadIdx.x==0 sends the required cp(next), threadIdx.x==32 sends TS MMA(current).")
    lines.append("- TS split noise: threadIdx.x==0 sends required cp plus extra cp writes to next A while threadIdx.x==32 sends TS MMA.")
    lines.append("")
    lines.append("Important interpretation note")
    lines.append("This is a resource interference test, not a production GEMM mainloop. SS noise writes to TMEM slots that SS MMA does not consume. TS extra noise writes the next A slot, so it measures extra producer-side TMEM write pressure while the current A slot is read by TS MMA.")
    lines.append("")
    lines.append("SASS check")
    for spec, precision, shape in iter_keys(cases, shapes, precisions):
        key = (spec["case_id"], precision, shape)
        c = checks[key]
        lines.append(
            f"- {spec['case_id']} {precision} {base.SHAPES[shape]['label']}: "
            f"{c['status']}; MMA={c['expected_mma_sass']}; cp={c['expected_cp_sass'] or '-'}"
        )
    lines.append("")
    lines.append("Throughput and slowdown")
    lines.append("|Case|Precision|Shape|Path|Mode|Noise cp/MMA|TFLOP/s|TFLOP/s min|TFLOP/s max|Peak Ratio|cycles/CTA iter|cycles/MMA|cp inst|bytes/cycle|Slowdown vs control|")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for spec, precision, shape in iter_keys(cases, shapes, precisions):
        key = (spec["case_id"], precision, shape)
        r = results[key]
        peak = official_tflops(precision, r["active_blocks"], r["sm_count"])
        if spec["baseline"]:
            base_r = results[(spec["baseline"], precision, shape)]
            slowdown = ratio(r["thor_tflops"], base_r["thor_tflops"])
        else:
            slowdown = 1.0
        lines.append(
            f"|{spec['case_id']}|{precision}|{base.SHAPES[shape]['label']}|"
            f"{r['mma_path']}|{r['issue_mode']}|{r['noise_cp_per_mma']}|"
            f"{r['thor_tflops']:.3f}|{r['thor_tflops_min']:.3f}|{r['thor_tflops_max']:.3f}|"
            f"{100.0 * r['thor_tflops'] / peak:.2f}%|"
            f"{r['cycles_per_cta_iter']:.3f}|{r['cycles_per_mma']:.3f}|"
            f"{r['cp_instruction_count']}|{r['bytes_per_cycle']:.3f}|{slowdown:.3f}x|"
        )
    lines.append("")
    lines.append("Compact comparison by noise")
    lines.append("|Path|Mode|Precision|Shape|Noise cp/MMA|TFLOP/s ratio|cycle ratio|extra cycles/MMA|")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for spec, precision, shape in iter_keys(cases, shapes, precisions):
        if not spec["baseline"]:
            continue
        r = results[(spec["case_id"], precision, shape)]
        b = results[(spec["baseline"], precision, shape)]
        lines.append(
            f"|{spec['path']}|{spec['mode']}|{precision}|{base.SHAPES[shape]['label']}|"
            f"{spec['noise']}|{ratio(r['thor_tflops'], b['thor_tflops']):.3f}x|"
            f"{ratio(r['cycles_per_mma'], b['cycles_per_mma']):.3f}x|"
            f"{r['cycles_per_mma'] - b['cycles_per_mma']:.3f}|"
        )
    lines.append("")
    lines.append("Suggested NCU counters")
    lines.append("- sm__inst_executed_pipe_tensor* confirms the MMA issue rate.")
    lines.append("- sm__inst_executed_pipe_tmem* confirms the extra TMEM-side traffic.")
    lines.append("- smsp__average_warps_issue_stalled_dispatch_stall* separates issuer/dispatch pressure.")
    lines.append("- smsp__average_warps_issue_stalled_math_pipe_throttle* and *_short_scoreboard* help identify tensor/TMEM backend contention.")
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"\nWrote report: {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate, build, run, and report Thor tcgen05 TMEM interference benchmarks."
    )
    parser.add_argument("--iters", type=int, default=10000, help="benchmark iterations")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="runs per benchmark case; use 100 for tighter stability checks")
    parser.add_argument("--shapes", default=" ".join(DEFAULT_SHAPES), help="space/comma-separated shapes")
    parser.add_argument("--precisions", default=" ".join(DEFAULT_PRECISIONS), help="space/comma-separated precisions")
    parser.add_argument("--noise", default=" ".join(str(x) for x in DEFAULT_NOISE), help="space/comma-separated noise cp counts")
    parser.add_argument("--generate-only", action="store_true", help="only generate CUDA sources")
    parser.add_argument("--skip-run", action="store_true", help="compile and inspect, but do not run benchmarks")
    parser.add_argument("--ccbin", default=os.environ.get("TCGEN05_CCBIN"), help="optional host compiler for nvcc")
    args = parser.parse_args()

    os.chdir(ROOT)
    ensure_dirs()
    shapes = split_words(args.shapes)
    precisions = [x.upper() for x in split_words(args.precisions)]
    noise_values = [int(x) for x in split_words(args.noise)]
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    for shape in shapes:
        if shape not in base.SHAPES:
            raise ValueError(f"unknown shape: {shape}")
    for precision in precisions:
        if precision not in base.PRECISIONS:
            raise ValueError(f"unknown precision: {precision}")
    for noise in noise_values:
        if noise < 1:
            raise ValueError("noise values must be positive; baselines are generated automatically")

    cases = make_cases(noise_values)
    freq_hz = base.read_freq_hz()

    log("Generating TMEM interference CUDA sources")
    srcs = generate_sources(cases, shapes, precisions)
    if args.generate_only:
        for src in srcs.values():
            log(f"generated: {src}")
        return

    log("Compiling TMEM interference benchmarks")
    bins = compile_sources(srcs, args.ccbin)

    log("Inspecting tcgen05 MMA/cp SASS")
    checks = inspect_instructions(cases, srcs, bins)
    if args.skip_run:
        for key in srcs:
            log(f"{key}: {checks[key]['status']}")
        return

    dev_name, cc, sm_count = base.device_info()
    log(f"Running {len(cases)} cases x {len(precisions)} precisions x {len(shapes)} shapes x {args.repeats} repeats")
    results = run_benchmarks(bins, args.iters, freq_hz, args.repeats)
    write_report(cases, shapes, precisions, results, checks, dev_name, cc, sm_count, freq_hz, args.iters, args.repeats, noise_values)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
