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
CUTLASS_DIR = Path("/opt/pytorch/ao/third_party/cutlass")

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

LAUNCHES = {
    "single_warp_block": {
        "label": "SingleWarpBlock",
        "block_threads": 32,
        "warps_per_block": 1,
        "block_count_expr": "1",
        "active_blocks_expr": "1",
    },
    "full_sm_4warp_block": {
        "label": "FullSM4WarpBlock",
        "block_threads": 128,
        "warps_per_block": 4,
        "block_count_expr": "prop.multiProcessorCount",
        "active_blocks_expr": "prop.multiProcessorCount",
    },
}

PRECISION_ORDER = ["FP4", "FP8", "BF16"]
SHAPE_ORDER = ["m128n64", "m128n128", "m128n256"]
REPORT_SHAPE_ORDER = ["m128n256", "m128n128", "m128n64"]
LAUNCH_ORDER = ["single_warp_block", "full_sm_4warp_block"]

PRECISIONS = {
    "BF16": {
        "kernel_kind": "bf16",
        "dense_ptx_instruction": "tcgen05.mma.cta_group::1.kind::f16",
        "sass_instruction": "UTCHMMA",
        "dense_mma_asm": (
            '"tcgen05.mma.cta_group::1.kind::f16 '
            '[%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"'
        ),
        "idesc_func": "make_idesc_bf16",
        "desc_leading": 16,
        "desc_stride": 8,
        "k_inst": 16,
        "dense_extra_operands": '"r"(0), "r"(0), "r"(0), "r"(0)',
        "dense_tmem_setup": "",
    },
    "FP8": {
        "kernel_kind": "fp8",
        "dense_ptx_instruction": "tcgen05.mma.cta_group::1.kind::f8f6f4",
        "sass_instruction": "UTCQMMA",
        "dense_mma_asm": (
            '"tcgen05.mma.cta_group::1.kind::f8f6f4 '
            '[%0], %1, %2, %3, {%5,%6,%7,%8}, p; }"'
        ),
        "idesc_func": "make_idesc_fp8",
        "desc_leading": 8,
        "desc_stride": 4,
        "k_inst": 32,
        "dense_extra_operands": '"r"(0), "r"(0), "r"(0), "r"(0)',
        "dense_tmem_setup": "",
    },
    "FP4": {
        "kernel_kind": "fp4",
        "dense_ptx_instruction": "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16",
        "sass_instruction": "UTCOMMA.4X",
        "dense_mma_asm": (
            '"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 '
            '[%0], %1, %2, %3, [%5], [%6], p; }"'
        ),
        "idesc_func": "make_idesc_fp4",
        "desc_leading": 4,
        "desc_stride": 2,
        "k_inst": 64,
        "dense_extra_operands": '"r"(tsfa), "r"(tsfb)',
        "dense_tmem_setup": "uint32_t tsfa = tmem_base + 256; uint32_t tsfb = tmem_base + 384;",
    },
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
static constexpr char kPrecision[] = "{precision}";
static constexpr char kMode[] = "{mode}";
static constexpr char kShape[] = "{shape_label}";
static constexpr char kLaunch[] = "{launch_label}";
static constexpr int kBlockThreads = {block_threads};
static constexpr int kWarpsPerBlock = {warps_per_block};

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool warp_leader() {
  return (threadIdx.x & 31) == 0;
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

__device__ __forceinline__ uint64_t make_idesc_bf16() {
  uint32_t d = 0;
  d |= 1u << 4;        // C format F32
  d |= 1u << 7;        // A BF16
  d |= 1u << 10;       // B BF16
  d |= {n_desc_units}u << 17;      // N = {shape_n}
  d |= {m_desc_units}u << 24;       // M = {shape_m}
  return uint64_t(d) << 32;
}

__device__ __forceinline__ uint64_t make_idesc_fp8() {
  uint32_t d = 0;
  d |= 1u << 4;        // C format F32, A/B E4M3
  d |= {n_desc_units}u << 17;      // N = {shape_n}
  d |= {m_desc_units}u << 24;       // M = {shape_m}
  return uint64_t(d) << 32;
}

__device__ __forceinline__ uint64_t make_idesc_fp4() {
  uint32_t d = 0;
  d |= 5u << 7;        // A E2M1
  d |= 5u << 10;       // B E2M1
  d |= {n_desc_units}u << 17;      // N = {shape_n}
  d |= 0u << 23;       // UE4M3 scale format
  d |= {m_desc_units}u << 24;       // M = {shape_m}
  return uint64_t(d) << 32;
}

__global__ __launch_bounds__({block_threads}, 1)
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
    barrier_init(&done_barrier, kWarpsPerBlock);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    uint32_t dst = smem_u32(&tmem_base);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(dst), "r"(512));
  }
  __syncthreads();

  uint64_t desc_a = make_smem_desc(smem_a, {desc_leading}, {desc_stride});
  uint64_t desc_b = make_smem_desc(smem_b, {desc_leading}, {desc_stride});
  uint64_t idesc = {idesc_func}();

  __syncthreads();
  unsigned long long start = clock64();

  for (int i = 0; i < iters; ++i) {
    uint32_t scale = (i == 0) ? 0u : 1u;
    uint32_t tmem_c = tmem_base;
    {tmem_setup}
    if (warp_leader()) {
      asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
        {mma_asm}
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idesc >> 32)), "r"(scale),
           {extra_operands});
    }
  }

  uint32_t bar_addr = smem_u32(&done_barrier);
  if (warp_leader()) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar_addr));
  }
  if (threadIdx.x == 0) {
    barrier_wait(&done_barrier, 0);
  }
  __syncthreads();

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
  int blocks = {block_count_expr};
  int active_blocks = {active_blocks_expr};

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

  double inst_per_active_block = double(kWarpsPerBlock) * double(iters);
  double macs_per_active_block = inst_per_active_block * double(kMacPerInst);
  double macs_per_cycle_per_active_block = macs_per_active_block / double(max_cycles);
  double thor_macs_per_cycle = macs_per_cycle_per_active_block * double(active_blocks);
  double thor_macs_per_second = thor_macs_per_cycle * freq_hz;
  double thor_tflops = 2.0 * thor_macs_per_second / 1.0e12;

  std::printf("mode=%s\n", kMode);
  std::printf("precision=%s\n", kPrecision);
  std::printf("shape=%s\n", kShape);
  std::printf("launch=%s\n", kLaunch);
  std::printf("sm_count=%d\n", prop.multiProcessorCount);
  std::printf("active_blocks=%d\n", active_blocks);
  std::printf("block_threads=%d\n", kBlockThreads);
  std::printf("warps_per_block=%d\n", kWarpsPerBlock);
  std::printf("iters=%d\n", iters);
  std::printf("cycles=%llu\n", max_cycles);
  std::printf("macs_per_cycle_per_active_block=%.6f\n", macs_per_cycle_per_active_block);
  std::printf("thor_macs_per_cycle=%.6f\n", thor_macs_per_cycle);
  std::printf("thor_tflops=%.6f\n", thor_tflops);

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


def make_benchmark_cfg(precision, shape, launch):
    base = PRECISIONS[precision]
    shape_cfg = SHAPES[shape]
    launch_cfg = LAUNCHES[launch]
    shape_k = shape_cfg.get("k_by_precision", {}).get(precision, base["k_inst"])
    shape_label = f"{shape_cfg['label']}K{shape_k}"
    dense_mac_per_inst = shape_cfg["m"] * shape_cfg["n"] * shape_k
    cfg = {
        "mode": "dense",
        "mode_label": "Dense",
        "precision": precision,
        "shape": shape,
        "shape_base_label": shape_cfg["label"],
        "shape_label": shape_label,
        "shape_m": shape_cfg["m"],
        "shape_n": shape_cfg["n"],
        "shape_k": shape_k,
        "m_desc_units": shape_cfg["m_desc_units"],
        "n_desc_units": shape_cfg["n_desc_units"],
        "launch": launch,
        "launch_label": launch_cfg["label"],
        "block_threads": launch_cfg["block_threads"],
        "warps_per_block": launch_cfg["warps_per_block"],
        "block_count_expr": launch_cfg["block_count_expr"],
        "active_blocks_expr": launch_cfg["active_blocks_expr"],
        "kernel_kind": base["kernel_kind"],
        "ptx_instruction": base["dense_ptx_instruction"],
        "sass_instruction": base["sass_instruction"],
        "mma_asm": base["dense_mma_asm"],
        "idesc_func": base["idesc_func"],
        "desc_leading": base["desc_leading"],
        "desc_stride": base["desc_stride"],
        "mac_per_inst": dense_mac_per_inst,
        "extra_operands": base["dense_extra_operands"],
        "tmem_setup": base["dense_tmem_setup"],
    }
    return cfg


def benchmark_keys():
    for launch in LAUNCH_ORDER:
        for shape in SHAPE_ORDER:
            for precision in PRECISION_ORDER:
                yield precision, shape, launch


def report_keys():
    for precision in PRECISION_ORDER:
        for launch in LAUNCH_ORDER:
            for shape in REPORT_SHAPE_ORDER:
                yield precision, shape, launch


def key_label(key):
    precision, shape, launch = key
    cfg = make_benchmark_cfg(precision, shape, launch)
    return f"{LAUNCHES[launch]['label']} {cfg['shape_label']} {precision}"


def generate_sources():
    paths = {}
    for precision, shape, launch in benchmark_keys():
        cfg = make_benchmark_cfg(precision, shape, launch)
        src = SRC_DIR / f"tcgen05_{launch}_{shape}_{precision.lower()}_benchmark.cu"
        text = CU_TEMPLATE
        replacements = {k: str(v) for k, v in cfg.items()}
        for key, value in replacements.items():
            text = text.replace("{" + key + "}", value)
        src.write_text(text)
        paths[(precision, shape, launch)] = src
    return paths


def compile_sources(srcs):
    if not CUTLASS_DIR.exists():
        raise RuntimeError(f"CUTLASS include path not found: {CUTLASS_DIR}")

    bins = {}
    for key, src in srcs.items():
        precision, shape, launch = key
        binary = BUILD_DIR / f"tcgen05_{launch}_{shape}_{precision.lower()}_benchmark"
        cmd = [
            "nvcc",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-gencode",
            "arch=compute_110a,code=sm_110a",
            "-I" + str(CUTLASS_DIR / "include"),
            src,
            "-o",
            binary,
        ]
        run(cmd)
        bins[key] = binary
    return bins


def inspect_mma_instructions(srcs, bins):
    checks = {}
    for key, src in srcs.items():
        precision, shape, launch = key
        cfg = make_benchmark_cfg(precision, shape, launch)
        source_text = src.read_text()
        source_has_sparse = "tcgen05.mma.sp" in source_text or ".mma.sp." in source_text
        ret, sass = run(["cuobjdump", "--dump-sass", bins[key]], capture=True, check=False, echo=False)
        expected = cfg["sass_instruction"].split(".")[0]
        sass_lines = []
        for line in sass.splitlines():
            if re.search(r"\bUTC[A-Z0-9.]*MMA", line):
                sass_lines.append(line.strip())
        expected_sass_found = any(expected in line for line in sass_lines)
        status = "dense" if (ret == 0 and not source_has_sparse and expected_sass_found) else "check_failed"
        checks[key] = {
            "ptx_instruction": cfg["ptx_instruction"],
            "expected_sass_instruction": cfg["sass_instruction"],
            "source_has_sparse": source_has_sparse,
            "expected_sass_found": expected_sass_found,
            "sass_sample": sass_lines[:3],
            "status": status,
        }
    return checks


def parse_plain_result(text):
    result = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    required = [
        "mode", "precision", "shape", "launch", "sm_count", "active_blocks",
        "block_threads", "warps_per_block", "iters", "cycles",
        "macs_per_cycle_per_active_block", "thor_macs_per_cycle", "thor_tflops",
    ]
    missing = [k for k in required if k not in result]
    if missing:
        raise RuntimeError(f"missing benchmark fields: {missing}\n{text}")
    return {
        "mode": result["mode"],
        "precision": result["precision"],
        "shape": result["shape"],
        "launch": result["launch"],
        "sm_count": int(result["sm_count"]),
        "active_blocks": int(result["active_blocks"]),
        "block_threads": int(result["block_threads"]),
        "warps_per_block": int(result["warps_per_block"]),
        "iters": int(result["iters"]),
        "cycles": int(result["cycles"]),
        "macs_per_cycle_per_active_block": float(result["macs_per_cycle_per_active_block"]),
        "thor_macs_per_cycle": float(result["thor_macs_per_cycle"]),
        "thor_tflops": float(result["thor_tflops"]),
    }


def run_plain_benchmarks(bins, iters, freq_hz):
    results = {}
    for key, binary in bins.items():
        _, out = run([binary, str(iters), str(freq_hz)])
        results[key] = parse_plain_result(out)
    return results


def official_tflops(precision, active_blocks, sm_count):
    return OFFICIAL_TFLOPS[precision] * float(active_blocks) / float(sm_count)


def write_report(results, instruction_checks, dev_name, cc, sm_count, freq_hz, iters):
    lines = []
    lines.append("Thor tcgen05 dense 计算能力实测分析报告")
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
    lines.append("Mode: Dense only")
    lines.append("Precision/K: K 随 M/N shape 显式配置；当前 dense cta_group::1: BF16 K=16, FP8 K=32, FP4 K=64")
    lines.append("Shape(M*N*K): shape 输出会按 precision 附加 K，例如 M128N256K64 / M128N256K32 / M128N256K16")
    lines.append("Launch: SingleWarpBlock = <<<1, 32>>>；FullSM4WarpBlock = <<<SM数, 128>>>")
    lines.append("")
    lines.append("NVIDIA 官方 Thor dense 峰值")
    lines.append("FP4: 1035 TFLOPS")
    lines.append("FP8: 517 TFLOPS")
    lines.append("BF16/FP16: 258.5 TFLOPS")
    lines.append("SingleWarpBlock 的官方对比按 active_blocks / SM数 缩放；shape 不缩放官方峰值。")
    lines.append("")
    lines.append("MMA 指令确认")
    lines.append("说明: M256N128 对当前 cta_group::1 单 CTA 模板不是通用合法 shape，FP8 会触发 illegal instruction；需要 cta_group::2/cluster 版本另测。本脚本当前使用 M128N256 覆盖 256 维。")
    lines.append("本测试只包含 dense tcgen05 MMA，源码不包含 tcgen05.mma.sp；SASS 检查确认命中对应精度的 UTC* MMA 指令。")
    for key in benchmark_keys():
        c = instruction_checks[key]
        lines.append(
            f"{key_label(key)}: PTX inline asm = {c['ptx_instruction']}；"
            f"SASS = {c['expected_sass_instruction']}；check = {c['status']}。"
        )
        if c["sass_sample"]:
            lines.append(f"  SASS sample: {c['sass_sample'][0]}")
    lines.append("")
    lines.append("实测汇总")
    lines.append(
        "| **精度** | WarpNum | Launch | **矩阵形状(M*N*K)** | **计算量(MAC/inst)** | "
        "**实际测试/TFLOP/s** | **理论峰值/TFLOP/s** | **比率** |"
    )
    lines.append(
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"
    )
    previous_precision = None
    for key in report_keys():
        precision, shape, launch = key
        cfg = make_benchmark_cfg(precision, shape, launch)
        r = results[key]
        official = official_tflops(precision, r["active_blocks"], r["sm_count"])
        ratio = 100.0 * r["thor_tflops"] / official
        precision_cell = f"**{precision}**" if precision != previous_precision else ""
        previous_precision = precision
        lines.append(
            f"| {precision_cell} | "
            f"{r['warps_per_block']} | "
            f"{LAUNCHES[launch]['label']} | "
            f"{cfg['shape_label']} | "
            f"{cfg['mac_per_inst']} | "
            f"{r['thor_tflops']:.3f} | "
            f"{official:.3f} | "
            f"{ratio:.2f}% |"
        )
    lines.append("")
    lines.append("分析")
    for key in benchmark_keys():
        precision, shape, launch = key
        r = results[key]
        official = official_tflops(precision, r["active_blocks"], r["sm_count"])
        ratio = 100.0 * r["thor_tflops"] / official
        lines.append(
            f"{key_label(key)} K={make_benchmark_cfg(precision, shape, launch)['shape_k']}: cycles={r['cycles']}，"
            f"实测 {r['thor_tflops']:.3f} TFLOP/s，约为该 launch 缩放后官方 {official:.3f} TFLOP/s 的 {ratio:.2f}%。"
            f" MAC/cycle/active-block={r['macs_per_cycle_per_active_block']:.2f}。"
        )
    lines.append("")
    lines.append("说明")
    lines.append("1. 测试 CUDA 源码由本脚本生成到 ./benchmark_src，每个 launch/shape/precision 组合一个 .cu。")
    lines.append("2. 计时区间只循环发 dense tcgen05.mma，SMEM 初始化、TMEM alloc/dealloc 不计入 clock64 周期。")
    lines.append("3. K 不做独立 sweep，由 shape 的 k_by_precision 显式配置；MAC 数按 M*N*K 计算。")
    lines.append("4. TFLOP/s 按 1 MAC = 2 FLOP 换算，频率取 /sys/class/devfreq/gpu-gpc-0/cur_freq。")
    lines.append("5. SingleWarpBlock 使用一个 block、一个 warp；FullSM4WarpBlock 使用每 SM 一个 block、每 block 四个 warp。")
    lines.append("6. SASS 中 BF16/FP8/FP4 分别应出现 UTCHMMA/UTCQMMA/UTCOMMA，对应 tcgen05 MMA 的机器指令形态。")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"\n写入报告: {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Generate, build, run, and report Thor tcgen05 compute power benchmarks.")
    parser.add_argument("--iters", type=int, default=10000, help="benchmark iterations")
    args = parser.parse_args()

    os.chdir(ROOT)
    ensure_dirs()
    freq_hz = read_freq_hz()
    dev_name, cc, sm_count = device_info()

    log("生成 benchmark_src/*.cu")
    srcs = generate_sources()

    log("编译 benchmark")
    bins = compile_sources(srcs)

    log("检查 dense tcgen05 指令")
    instruction_checks = inspect_mma_instructions(srcs, bins)

    log("运行 launch/shape/precision 三维 dense 压测")
    results = run_plain_benchmarks(bins, args.iters, freq_hz)

    write_report(results, instruction_checks, dev_name, cc, sm_count, freq_hz, args.iters)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
