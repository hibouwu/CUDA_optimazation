#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "benchmark_src"
BUILD_DIR = ROOT / "build"
REPORT_PATH = ROOT / "分析报告.txt"

OFFICIAL_TFLOPS = {
    "FP4": 1035.0,
    "FP8": 517.0,
    "BF16": 258.5,
}

SHAPES = {
    "m128n64": {
        "label": "M128N64",
        "m": 128,
        "n": 64,
        "m_desc_units": 8,
        "n_desc_units": 8,
        "k_by_precision": {"BF16": 16, "FP8": 32, "FP4": 64},
    },
    "m128n128": {
        "label": "M128N128",
        "m": 128,
        "n": 128,
        "m_desc_units": 8,
        "n_desc_units": 16,
        "k_by_precision": {"BF16": 16, "FP8": 32, "FP4": 64},
    },
    "m128n256": {
        "label": "M128N256",
        "m": 128,
        "n": 256,
        "m_desc_units": 8,
        "n_desc_units": 32,
        "k_by_precision": {"BF16": 16, "FP8": 32, "FP4": 64},
    },
}

CASES = {
    "ss_mma_only": {
        "label": "SS MMA-only",
        "needs_mma": True,
        "needs_cp": False,
        "reports_tflops": True,
        "reports_cp": False,
    },
    "ts_mma_only": {
        "label": "TS MMA-only",
        "needs_mma": True,
        "needs_cp": True,
        "reports_tflops": True,
        "reports_cp": False,
    },
    "tcgen05_cp_only": {
        "label": "tcgen05.cp-only",
        "needs_mma": False,
        "needs_cp": True,
        "reports_tflops": False,
        "reports_cp": True,
    },
    "ts_cp_mma_serial_a1": {
        "label": "TS CP+MMA Serial A1",
        "needs_mma": True,
        "needs_cp": True,
        "reports_tflops": True,
        "reports_cp": False,
    },
    "ts_cp_mma_overlap_a2": {
        "label": "TS CP+MMA Overlap A2",
        "needs_mma": True,
        "needs_cp": True,
        "reports_tflops": True,
        "reports_cp": False,
    },
}

PRECISIONS = {
    "BF16": {
        "kernel_kind": "bf16",
        "dense_ptx_instruction": "tcgen05.mma.cta_group::1.kind::f16",
        "sass_instruction": "UTCHMMA",
        "ss_mma_asm": (
            '"tcgen05.mma.cta_group::1.kind::f16 '
            '[%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"'
        ),
        "ts_mma_asm": (
            '"tcgen05.mma.cta_group::1.kind::f16 '
            '[%0], [%1], %2, %3, {%5,%6,%7,%8}, p; }"'
        ),
        "idesc_func": "make_idesc_bf16",
        "desc_leading": 16,
        "desc_stride": 8,
        "k_inst": 16,
        "extra_operands": '"r"(0), "r"(0), "r"(0), "r"(0)',
        "tmem_setup": "",
        "cp_suffix": "128x128b",
        "effective_bytes_per_cp": 2048,
    },
    "FP8": {
        "kernel_kind": "fp8",
        "dense_ptx_instruction": "tcgen05.mma.cta_group::1.kind::f8f6f4",
        "sass_instruction": "UTCQMMA",
        "ss_mma_asm": (
            '"tcgen05.mma.cta_group::1.kind::f8f6f4 '
            '[%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"'
        ),
        "ts_mma_asm": (
            '"tcgen05.mma.cta_group::1.kind::f8f6f4 '
            '[%0], [%1], %2, %3, {%5,%6,%7,%8}, p; }"'
        ),
        "idesc_func": "make_idesc_fp8",
        "desc_leading": 8,
        "desc_stride": 4,
        "k_inst": 32,
        "extra_operands": '"r"(0), "r"(0), "r"(0), "r"(0)',
        "tmem_setup": "",
        "cp_suffix": "128x128b",
        "effective_bytes_per_cp": 2048,
    },
    "FP4": {
        "kernel_kind": "fp4",
        "dense_ptx_instruction": "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16",
        "sass_instruction": "UTCOMMA.4X",
        "ss_mma_asm": (
            '"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 '
            '[%0], %1, %2, %3, [%5], [%6], p; }"'
        ),
        "ts_mma_asm": (
            '"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 '
            '[%0], [%1], %2, %3, [%5], [%6], p; }"'
        ),
        "idesc_func": "make_idesc_fp4",
        "desc_leading": 4,
        "desc_stride": 2,
        "k_inst": 64,
        "extra_operands": '"r"(tsfa), "r"(tsfb)',
        "tmem_setup": "uint32_t tsfa = tmem_base + 384; uint32_t tsfb = tmem_base + 448;",
        "cp_suffix": "128x128b.b8x16.b4x16_p64",
        "effective_bytes_per_cp": 2048,
    },
}

CASE_ORDER = [
    "ss_mma_only",
    "ts_mma_only",
    "tcgen05_cp_only",
    "ts_cp_mma_serial_a1",
    "ts_cp_mma_overlap_a2",
]
PRECISION_ORDER = ["FP4", "FP8", "BF16"]
SHAPE_ORDER = ["m128n64", "m128n128", "m128n256"]
REPORT_SHAPE_ORDER = ["m128n256", "m128n128", "m128n64"]

BLOCK_THREADS = 128
WARPS_PER_BLOCK = 4
ISSUERS_PER_BLOCK = 1

CP_SASS_SHAPE_TOKENS = {
    "4x256b": "4dp256bit",
    "32x128b": "32dp128bit",
    "64x128b": "64dp128bit",
    "128x128b": "128dp128bit",
    "128x256b": "128dp256bit",
}

CP_SASS_DECODE_TOKENS = {
    "b8x16.b4x16_p64": "U4x16P64",
    "b8x16.b6x16_p32": "U6x16P32",
}


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
static constexpr int kInstK = {shape_k};
static constexpr int kBlockThreads = 128;
static constexpr int kWarpsPerBlock = 4;
static constexpr int kIssuerCount = 1;
static constexpr int kCpInstructionsPerTile = {cp_instructions_per_tile};
static constexpr int kEffectiveBytesPerCp = {effective_bytes_per_cp};
static constexpr char kCaseId[] = "{case_id}";
static constexpr char kCaseLabel[] = "{case_label}";
static constexpr char kPrecision[] = "{precision}";
static constexpr char kShape[] = "{shape_label}";
static constexpr char kCpSuffix[] = "{cp_suffix}";

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool issuer_thread() {
  return threadIdx.x == 0;
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

__device__ __forceinline__ void commit_and_wait(uint64_t* barrier, uint32_t phase) {
  uint32_t bar_addr = smem_u32(barrier);
  if (issuer_thread()) {
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
  __shared__ alignas(16) uint8_t smem_a[32768];
  __shared__ alignas(16) uint8_t smem_b[32768];
  __shared__ alignas(8) uint64_t done_barrier;
  __shared__ uint32_t tmem_base;

  for (int i = threadIdx.x; i < int(sizeof(smem_a)); i += blockDim.x) {
    smem_a[i] = uint8_t(i * 13 + 1);
    smem_b[i] = uint8_t(i * 17 + 3);
  }

  if (threadIdx.x == 0) {
    barrier_init(&done_barrier, kIssuerCount);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    uint32_t dst = smem_u32(&tmem_base);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(dst), "r"(512));
  }
  __syncthreads();

  uint64_t a_desc = make_smem_desc(smem_a, {desc_leading}, {desc_stride});
  uint64_t b_desc = make_smem_desc(smem_b, {desc_leading}, {desc_stride});
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
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_base), "r"(512));
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

  long long mma_instruction_count = {mma_instruction_count_expr};
  long long cp_instruction_count = {cp_instruction_count_expr};
  long long processed_tiles = {processed_tiles_expr};
  double elapsed_seconds = double(max_cycles) / freq_hz;
  double tflops = mma_instruction_count > 0
      ? 2.0 * double(kMacPerInst) * double(mma_instruction_count) / elapsed_seconds / 1.0e12
      : 0.0;
  double bytes_per_cycle = cp_instruction_count > 0
      ? double(cp_instruction_count) * double(kEffectiveBytesPerCp) / double(max_cycles)
      : 0.0;
  double cycles_per_cp = cp_instruction_count > 0
      ? double(max_cycles) / double(cp_instruction_count)
      : 0.0;
  double cycles_per_tile = processed_tiles > 0
      ? double(max_cycles) / double(processed_tiles)
      : 0.0;

  std::printf("case_id=%s\n", kCaseId);
  std::printf("case_label=%s\n", kCaseLabel);
  std::printf("precision=%s\n", kPrecision);
  std::printf("shape=%s\n", kShape);
  std::printf("cp_suffix=%s\n", kCpSuffix);
  std::printf("sm_count=%d\n", prop.multiProcessorCount);
  std::printf("active_blocks=%d\n", active_blocks);
  std::printf("block_threads=%d\n", kBlockThreads);
  std::printf("warps_per_block=%d\n", kWarpsPerBlock);
  std::printf("issuer_count=%d\n", kIssuerCount);
  std::printf("iters=%d\n", iters);
  std::printf("cycles=%llu\n", max_cycles);
  std::printf("mma_instruction_count=%lld\n", mma_instruction_count);
  std::printf("cp_instruction_count=%lld\n", cp_instruction_count);
  std::printf("processed_tiles=%lld\n", processed_tiles);
  std::printf("effective_bytes_per_cp=%d\n", kEffectiveBytesPerCp);
  std::printf("thor_tflops=%.6f\n", tflops);
  std::printf("bytes_per_cycle=%.6f\n", bytes_per_cycle);
  std::printf("cycles_per_cp=%.6f\n", cycles_per_cp);
  std::printf("cycles_per_tile=%.6f\n", cycles_per_tile);

  CUDA_CHECK(cudaFree(d_cycles));
  delete[] h_cycles;
  return 0;
}
'''


def log(message):
    print(message, flush=True)


def run(cmd, *, cwd=ROOT, capture=True, check=True, echo=True):
    log("+ " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    out = proc.stdout or ""
    if echo and capture and out:
        print(out, end="")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(str(x) for x in cmd)}\n{out}")
    return proc.returncode, out


def read_freq_hz():
    path = Path("/sys/class/devfreq/gpu-gpc-0/cur_freq")
    if path.exists():
        try:
            return int(path.read_text().strip())
        except ValueError:
            pass
    return 1575000000


def device_info():
    code = (
        "import torch\n"
        "p=torch.cuda.get_device_properties(0)\n"
        "print(f'{p.name}|{p.major}.{p.minor}|{p.multi_processor_count}')\n"
    )
    _, out = run(["python3", "-c", code])
    name, cc, sm_count = out.strip().splitlines()[-1].split("|")
    return name, cc, int(sm_count)


def ensure_dirs():
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


def case_body(case_id):
    if case_id == "ss_mma_only":
        return "", """
  for (int i = 0; i < iters; ++i) {
    uint32_t scale = (i == 0) ? 0u : 1u;
    if (issuer_thread()) {
      issue_ss_mma(d_tmem, a_desc, b_desc, idesc, tsfa, tsfb, scale);
    }
  }
  commit_and_wait(&done_barrier, phase);
  phase ^= 1;
"""
    if case_id == "ts_mma_only":
        pre = """
  if (issuer_thread()) {
    issue_cp(a_tmem0, a_desc);
  }
  commit_and_wait(&done_barrier, phase);
  phase ^= 1;
"""
        body = """
  for (int i = 0; i < iters; ++i) {
    uint32_t scale = (i == 0) ? 0u : 1u;
    if (issuer_thread()) {
      issue_ts_mma(d_tmem, a_tmem0, b_desc, idesc, tsfa, tsfb, scale);
    }
  }
  commit_and_wait(&done_barrier, phase);
  phase ^= 1;
"""
        return pre, body
    if case_id == "tcgen05_cp_only":
        return "", """
  for (int i = 0; i < iters; ++i) {
    uint32_t dst = (i & 1) ? a_tmem1 : a_tmem0;
    if (issuer_thread()) {
      issue_cp(dst, a_desc);
    }
  }
  commit_and_wait(&done_barrier, phase);
  phase ^= 1;
"""
    if case_id == "ts_cp_mma_serial_a1":
        return "", """
  for (int i = 0; i < iters; ++i) {
    uint32_t scale = (i == 0) ? 0u : 1u;
    if (issuer_thread()) {
      issue_cp(a_tmem0, a_desc);
    }
    commit_and_wait(&done_barrier, phase);
    phase ^= 1;
    if (issuer_thread()) {
      issue_ts_mma(d_tmem, a_tmem0, b_desc, idesc, tsfa, tsfb, scale);
    }
    commit_and_wait(&done_barrier, phase);
    phase ^= 1;
  }
"""
    if case_id == "ts_cp_mma_overlap_a2":
        pre = """
  if (issuer_thread()) {
    issue_cp(a_tmem0, a_desc);
  }
  commit_and_wait(&done_barrier, phase);
  phase ^= 1;
"""
        body = """
  for (int i = 0; i < iters; ++i) {
    uint32_t scale = (i == 0) ? 0u : 1u;
    uint32_t current = (i & 1) ? a_tmem1 : a_tmem0;
    uint32_t next = (i & 1) ? a_tmem0 : a_tmem1;
    if (issuer_thread()) {
      issue_cp(next, a_desc);
      issue_ts_mma(d_tmem, current, b_desc, idesc, tsfa, tsfb, scale);
    }
    commit_and_wait(&done_barrier, phase);
    phase ^= 1;
  }
"""
        return pre, body
    raise KeyError(case_id)


def make_cfg(case_id, precision, shape):
    p = PRECISIONS[precision]
    s = SHAPES[shape]
    shape_k = s["k_by_precision"][precision]
    shape_label = f"{s['label']}K{shape_k}"
    pre_body, timed_body = case_body(case_id)
    cp_insts_per_tile = ISSUERS_PER_BLOCK if CASES[case_id]["needs_cp"] else 0
    cfg = {
        "case_id": case_id,
        "case_label": CASES[case_id]["label"],
        "precision": precision,
        "shape": shape,
        "shape_label": shape_label,
        "shape_m": s["m"],
        "shape_n": s["n"],
        "shape_k": shape_k,
        "m_desc_units": s["m_desc_units"],
        "n_desc_units": s["n_desc_units"],
        "mac_per_inst": s["m"] * s["n"] * shape_k,
        "kernel_kind": p["kernel_kind"],
        "sass_instruction": p["sass_instruction"],
        "ss_mma_asm": p["ss_mma_asm"],
        "ts_mma_asm": p["ts_mma_asm"],
        "idesc_func": p["idesc_func"],
        "desc_leading": p["desc_leading"],
        "desc_stride": p["desc_stride"],
        "extra_operands": p["extra_operands"],
        "tmem_setup": p["tmem_setup"] or "uint32_t tsfa = 0; uint32_t tsfb = 0;",
        "cp_suffix": p["cp_suffix"],
        "cp_asm": f"tcgen05.cp.cta_group::1.{p['cp_suffix']} [%0], %1;",
        "effective_bytes_per_cp": p["effective_bytes_per_cp"],
        "cp_instructions_per_tile": cp_insts_per_tile,
        "pre_timing_body": pre_body,
        "timed_body": timed_body,
        "mma_instruction_count_expr": (
            "static_cast<long long>(active_blocks) * static_cast<long long>(kIssuerCount) * static_cast<long long>(iters)"
            if CASES[case_id]["reports_tflops"] else "0LL"
        ),
        "cp_instruction_count_expr": (
            "static_cast<long long>(active_blocks) * static_cast<long long>(kIssuerCount) * static_cast<long long>(iters)"
            if CASES[case_id]["reports_cp"] or case_id.startswith("ts_cp_mma") else "0LL"
        ),
        "processed_tiles_expr": (
            "static_cast<long long>(active_blocks) * static_cast<long long>(kIssuerCount) * static_cast<long long>(iters)"
            if case_id.startswith("ts_cp_mma") else "0LL"
        ),
    }
    return cfg


def iter_keys():
    for case_id in CASE_ORDER:
        for shape in SHAPE_ORDER:
            for precision in PRECISION_ORDER:
                yield case_id, precision, shape


def report_keys():
    for precision in PRECISION_ORDER:
        for shape in REPORT_SHAPE_ORDER:
            yield precision, shape


def source_name(case_id, precision, shape):
    prefix = case_id if case_id.startswith("tcgen05_") else f"tcgen05_{case_id}"
    return f"{prefix}_{shape}_{precision.lower()}_benchmark.cu"


def binary_name(case_id, precision, shape):
    prefix = case_id if case_id.startswith("tcgen05_") else f"tcgen05_{case_id}"
    return f"{prefix}_{shape}_{precision.lower()}_benchmark"


def render_template(cfg):
    text = CU_TEMPLATE
    for key, value in cfg.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def generate_sources():
    paths = {}
    for case_id, precision, shape in iter_keys():
        cfg = make_cfg(case_id, precision, shape)
        src = SRC_DIR / source_name(case_id, precision, shape)
        src.write_text(render_template(cfg))
        paths[(case_id, precision, shape)] = src
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
        run(cmd)
        bins[key] = binary
    return bins


def cp_sass_tokens(cp_suffix):
    parts = cp_suffix.split(".")
    tokens = ["UTCCP"]
    shape_token = CP_SASS_SHAPE_TOKENS.get(parts[0])
    if shape_token:
        tokens.append(shape_token)
    decode_suffix = ".".join(parts[1:])
    decode_token = CP_SASS_DECODE_TOKENS.get(decode_suffix)
    if decode_token:
        tokens.append(decode_token)
    return tuple(token.upper() for token in tokens)


def has_all_tokens(line, tokens):
    upper = line.upper()
    return all(token in upper for token in tokens)


def inspect_instructions(srcs, bins):
    checks = {}
    for key in srcs:
        case_id, precision, shape = key
        cfg = make_cfg(case_id, precision, shape)
        ret, sass = run(["cuobjdump", "--dump-sass", bins[key]], capture=True, check=False, echo=False)
        mma_lines = [line.strip() for line in sass.splitlines() if re.search(r"\bUTC[A-Z0-9.]*MMA", line)]
        cp_lines = [line.strip() for line in sass.splitlines() if re.search(r"\bUTCCP(?:\.|\s)", line)]
        expected = cfg["sass_instruction"].split(".")[0]
        needs_mma = CASES[case_id]["needs_mma"]
        needs_cp = CASES[case_id]["needs_cp"]
        expected_cp_tokens = cp_sass_tokens(cfg["cp_suffix"]) if needs_cp else ()
        mma_ok = (not needs_mma) or any(expected in line for line in mma_lines)
        cp_ok = (not needs_cp) or any(has_all_tokens(line, expected_cp_tokens) for line in cp_lines)
        status = "ok" if ret == 0 and mma_ok and cp_ok else "check_failed"
        checks[key] = {
            "status": status,
            "expected_mma_sass": cfg["sass_instruction"] if needs_mma else "",
            "expected_cp_sass": " ".join(expected_cp_tokens),
            "mma_sample": mma_lines[:2],
            "cp_sample": cp_lines[:2],
            "cp_suffix": cfg["cp_suffix"] if needs_cp else "",
        }
    return checks


def parse_result(text):
    result = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    required = [
        "case_id", "case_label", "precision", "shape", "sm_count",
        "active_blocks", "block_threads", "warps_per_block", "issuer_count", "iters",
        "cycles", "mma_instruction_count", "cp_instruction_count",
        "processed_tiles", "effective_bytes_per_cp", "thor_tflops",
        "bytes_per_cycle", "cycles_per_cp", "cycles_per_tile",
    ]
    missing = [k for k in required if k not in result]
    if missing:
        raise RuntimeError(f"missing benchmark fields: {missing}\n{text}")
    return {
        "case_id": result["case_id"],
        "case_label": result["case_label"],
        "precision": result["precision"],
        "shape": result["shape"],
        "cp_suffix": result.get("cp_suffix", ""),
        "sm_count": int(result["sm_count"]),
        "active_blocks": int(result["active_blocks"]),
        "block_threads": int(result["block_threads"]),
        "warps_per_block": int(result["warps_per_block"]),
        "issuer_count": int(result["issuer_count"]),
        "iters": int(result["iters"]),
        "cycles": int(result["cycles"]),
        "mma_instruction_count": int(result["mma_instruction_count"]),
        "cp_instruction_count": int(result["cp_instruction_count"]),
        "processed_tiles": int(result["processed_tiles"]),
        "effective_bytes_per_cp": int(result["effective_bytes_per_cp"]),
        "thor_tflops": float(result["thor_tflops"]),
        "bytes_per_cycle": float(result["bytes_per_cycle"]),
        "cycles_per_cp": float(result["cycles_per_cp"]),
        "cycles_per_tile": float(result["cycles_per_tile"]),
    }


def run_benchmarks(bins, iters, freq_hz):
    results = {}
    for key, binary in bins.items():
        _, out = run([binary, str(iters), str(freq_hz)])
        results[key] = parse_result(out)
    return results


def official_tflops(precision, active_blocks, sm_count):
    return OFFICIAL_TFLOPS[precision] * float(active_blocks) / float(sm_count)


def speedup(num, den):
    if den == 0:
        return 0.0
    return num / den


def write_report(results, checks, dev_name, cc, sm_count, freq_hz, iters):
    lines = []
    lines.append("Thor tcgen05 cp + MMA pipeline 微基准报告")
    lines.append(f"生成时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    lines.append("设备信息")
    lines.append(f"GPU: {dev_name}")
    lines.append(f"Compute Capability: {cc}")
    lines.append(f"SM 数量: {sm_count}")
    lines.append(f"频率: {freq_hz / 1e9:.3f} GHz")
    lines.append(f"压测 iters: {iters}")
    lines.append("")
    lines.append("测试维度")
    lines.append("Case: SS MMA-only, TS MMA-only, tcgen05.cp-only, TS CP+MMA Serial A1, TS CP+MMA Overlap A2")
    lines.append("Precision/K: BF16 K=16, FP8 K=32, FP4 K=64")
    lines.append("Shape: M128N64, M128N128, M128N256")
    lines.append("Launch: 每 SM 一个 CTA，每 CTA 128 个线程 / 4 个 warp，其中仅 threadIdx.x==0 发射 tcgen05 指令")
    lines.append("")
    lines.append("反汇编检查")
    for key in iter_keys():
        c = checks[key]
        case_id, precision, shape = key
        cfg = make_cfg(case_id, precision, shape)
        lines.append(
            f"{CASES[case_id]['label']} {cfg['shape_label']} {precision}: "
            f"check={c['status']}；MMA SASS={c['expected_mma_sass'] or '-'}；"
            f"cp PTX={c['cp_suffix'] or '-'}；cp SASS={c['expected_cp_sass'] or '-'}。"
        )
        if c["mma_sample"]:
            lines.append(f"  MMA sample: {c['mma_sample'][0]}")
        if c["cp_sample"]:
            lines.append(f"  cp sample: {c['cp_sample'][0]}")
    lines.append("")
    lines.append("MMA-only TFLOP/s 与 Peak Ratio")
    lines.append("|Precision|Shape|K|SS MMA-only TFLOP/s|TS MMA-only TFLOP/s|SS Peak Ratio|TS Peak Ratio|")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for precision, shape in report_keys():
        cfg = make_cfg("ss_mma_only", precision, shape)
        ss = results[("ss_mma_only", precision, shape)]
        ts = results[("ts_mma_only", precision, shape)]
        ss_peak = official_tflops(precision, ss["active_blocks"], ss["sm_count"])
        ts_peak = official_tflops(precision, ts["active_blocks"], ts["sm_count"])
        lines.append(
            f"|{precision}|{SHAPES[shape]['label']}|{cfg['shape_k']}|"
            f"{ss['thor_tflops']:.3f}|{ts['thor_tflops']:.3f}|"
            f"{100.0 * ss['thor_tflops'] / ss_peak:.2f}%|{100.0 * ts['thor_tflops'] / ts_peak:.2f}%|"
        )
    lines.append("")
    lines.append("tcgen05.cp-only")
    lines.append("|Precision|Shape|cp suffix|effective bytes/cp|cp instruction count|elapsed cycles|bytes/cycle|cycles/cp|")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for precision, shape in report_keys():
        r = results[("tcgen05_cp_only", precision, shape)]
        lines.append(
            f"|{precision}|{SHAPES[shape]['label']}|{r['cp_suffix']}|{r['effective_bytes_per_cp']}|"
            f"{r['cp_instruction_count']}|{r['cycles']}|{r['bytes_per_cycle']:.3f}|{r['cycles_per_cp']:.3f}|"
        )
    lines.append("")
    lines.append("CP+MMA pipeline")
    lines.append("|Precision|Shape|Serial A1 TFLOP/s|Serial A1 cycles/tile|Overlap A2 TFLOP/s|Overlap A2 cycles/tile|Overlap Gain|")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for precision, shape in report_keys():
        serial = results[("ts_cp_mma_serial_a1", precision, shape)]
        overlap = results[("ts_cp_mma_overlap_a2", precision, shape)]
        gain = speedup(overlap["thor_tflops"], serial["thor_tflops"])
        lines.append(
            f"|{precision}|{SHAPES[shape]['label']}|"
            f"{serial['thor_tflops']:.3f}|{serial['cycles_per_tile']:.3f}|"
            f"{overlap['thor_tflops']:.3f}|{overlap['cycles_per_tile']:.3f}|{gain:.3f}x|"
        )
    lines.append("")
    lines.append("图 5 speedup 数据源")
    columns = [
        ("BF16", "m128n64"), ("BF16", "m128n128"), ("BF16", "m128n256"),
        ("FP8", "m128n64"), ("FP8", "m128n128"), ("FP8", "m128n256"),
        ("FP4", "m128n64"), ("FP4", "m128n128"), ("FP4", "m128n256"),
    ]
    header = "|Case|" + "|".join(f"{p}-{SHAPES[s]['label'].replace('M128', '')}" for p, s in columns) + "|"
    lines.append(header)
    lines.append("|---|" + "|".join("---:" for _ in columns) + "|")
    for case_id in ["ss_mma_only", "ts_mma_only", "ts_cp_mma_serial_a1", "ts_cp_mma_overlap_a2"]:
        cells = []
        for precision, shape in columns:
            base = results[("ss_mma_only", precision, shape)]["thor_tflops"]
            val = results[(case_id, precision, shape)]["thor_tflops"]
            cells.append(f"{speedup(val, base):.2f}x")
        lines.append(f"|{CASES[case_id]['label']}|" + "|".join(cells) + "|")
    lines.append("")
    lines.append("说明")
    lines.append("1. cp suffix 和 effective bytes/cp 来自脚本配置；低精度 packed copy 的有效字节口径需要结合最终 SASS/NCU 再确认。")
    lines.append("2. Serial A1 在每轮 cp 和 MMA 后都等待完成；Overlap A2 先预填一个 A slot，再交错发 cp(next) 和 mma(current)。")
    lines.append("3. TFLOP/s 只按 MMA 数学计算量计算；cp-only 使用 bytes/cycle 和 cycles/cp。")
    lines.append("4. 生成源码中的 TS/cp PTX 语法依赖 CUDA 13.x tcgen05 支持，首次运行后需要以 cuobjdump 和 NCU 计数器校验。")
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"\n写入报告: {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Generate, build, run, and report Thor tcgen05 cp+mma pipeline benchmarks.")
    parser.add_argument("--iters", type=int, default=10000, help="benchmark iterations")
    parser.add_argument("--generate-only", action="store_true", help="only generate CUDA sources")
    parser.add_argument("--skip-run", action="store_true", help="compile and inspect, but do not run benchmarks")
    parser.add_argument("--ccbin", default=os.environ.get("TCGEN05_CCBIN"), help="optional host compiler for nvcc")
    args = parser.parse_args()

    os.chdir(ROOT)
    ensure_dirs()
    freq_hz = read_freq_hz()

    log("生成 benchmark 源码")
    srcs = generate_sources()
    if args.generate_only:
        for src in srcs.values():
            log(f"生成: {src}")
        return

    log("编译 benchmark")
    bins = compile_sources(srcs, args.ccbin)

    log("检查 tcgen05 MMA/cp 指令")
    checks = inspect_instructions(srcs, bins)
    if args.skip_run:
        for key in iter_keys():
            log(f"{key}: {checks[key]['status']}")
        return

    dev_name, cc, sm_count = device_info()

    log("运行 5 case × 3 precision × 3 shape 压测")
    results = run_benchmarks(bins, args.iters, freq_hz)

    write_report(results, checks, dev_name, cc, sm_count, freq_hz, args.iters)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
