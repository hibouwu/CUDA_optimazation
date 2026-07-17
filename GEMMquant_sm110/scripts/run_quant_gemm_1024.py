#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path


CSV_HEADER = [
    "Precision",
    "Stage",
    "BackendId",
    "BackendLabel",
    "N",
    "Reference",
    "TimeMs",
    "GFLOPS",
    "RatioToReference",
    "Matched",
    "Status",
    "Reason",
    "Trial",
]


PLANNED_BACKENDS = [
    {
        "precision": "NVFP4",
        "stage": "q0",
        "backend_id": "nvfp4_q0_tc6_fused_epilogue",
        "label": "reuse GEMMsm110 tc6 fused NVFP4 epilogue",
        "runner": "gemm_sm110",
        "source_backend": "tc6",
        "implemented": True,
    },
    {
        "precision": "NVFP4",
        "stage": "q1",
        "backend_id": "nvfp4_q1_native_fp4_mainloop",
        "label": "planned native FP4/NVFP4 mainloop",
        "implemented": False,
        "reason": "planned: replace FP16 mainloop + requant with native FP4/NVFP4 path",
    },
    {
        "precision": "NVFP4",
        "stage": "q2",
        "backend_id": "nvfp4_q2_cutlass_72b_nvfp4",
        "label": "CUTLASS 72b block-scaled NVFP4 Tensor Core",
        "runner": "cutlass_fp4_72b",
        "implemented": True,
        "swizzle": 2,
        "iterations": 100,
    },
    {
        "precision": "MXFP4",
        "stage": "q0",
        "backend_id": "mxfp4_q0_cutlass_72a_mxfp4_bf16",
        "label": "CUTLASS 72a MXFP4 BF16 default swizzle",
        "runner": "cutlass_mxfp4_72a",
        "implemented": True,
        "swizzle": 2,
        "iterations": 100,
    },
    {
        "precision": "MXFP4",
        "stage": "q1",
        "backend_id": "mxfp4_q1_cutlass_72a_swizzle1",
        "label": "CUTLASS 72a MXFP4 BF16 swizzle tuned",
        "runner": "cutlass_mxfp4_72a",
        "implemented": True,
        "swizzle": 1,
        "iterations": 100,
    },
    {
        "precision": "FP8",
        "stage": "q0",
        "backend_id": "fp8_q0_cuda_naive",
        "label": "FP8 q0 naive CUDA baseline",
        "runner": "quant_bench",
        "source_backend": "fp8_q0_cuda_naive",
        "implemented": True,
    },
    {
        "precision": "FP8",
        "stage": "q1",
        "backend_id": "fp8_q1_cuda_vec4cols",
        "label": "FP8 q1 register-tiled four-column CUDA baseline",
        "runner": "quant_bench",
        "source_backend": "fp8_q1_cuda_vec4cols",
        "implemented": True,
    },
    {
        "precision": "FP8",
        "stage": "q2",
        "backend_id": "fp8_q2_cuda_vec8cols",
        "label": "FP8 q2 register-tiled eight-column CUDA baseline",
        "runner": "quant_bench",
        "source_backend": "fp8_q2_cuda_vec8cols",
        "implemented": True,
    },
    {
        "precision": "FP8",
        "stage": "q3",
        "backend_id": "fp8_q3_cuda_vec16cols",
        "label": "FP8 q3 register-tiled sixteen-column CUDA baseline",
        "runner": "quant_bench",
        "source_backend": "fp8_q3_cuda_vec16cols",
        "implemented": True,
    },
    {
        "precision": "FP8",
        "stage": "q4",
        "backend_id": "fp8_q4_mma_m16n8k32_global",
        "label": "FP8 q4 warp MMA M16N8K32 global-load Tensor Core",
        "runner": "quant_bench",
        "source_backend": "fp8_q4_mma_m16n8k32_global",
        "implemented": True,
    },
    {
        "precision": "FP8",
        "stage": "q5",
        "backend_id": "fp8_q5_mma_m16n8k32_smem64",
        "label": "FP8 q5 warp MMA M16N8K32 shared 64x64 tile",
        "runner": "quant_bench",
        "source_backend": "fp8_q5_mma_m16n8k32_smem64",
        "implemented": True,
    },
    {
        "precision": "FP8",
        "stage": "q6",
        "backend_id": "fp8_q6_mma_m16n8k32_smem64x128",
        "label": "FP8 q6 warp MMA M16N8K32 shared 64x128 tile",
        "runner": "quant_bench",
        "source_backend": "fp8_q6_mma_m16n8k32_smem64x128",
        "implemented": True,
    },
    {
        "precision": "FP8",
        "stage": "q7",
        "backend_id": "fp8_q7_mma_m16n8k32_smem128x64",
        "label": "FP8 q7 warp MMA M16N8K32 shared 128x64 tile",
        "runner": "quant_bench",
        "source_backend": "fp8_q7_mma_m16n8k32_smem128x64",
        "implemented": True,
    },
    {
        "precision": "FP8",
        "stage": "q8",
        "backend_id": "fp8_q8_cublaslt_matmul",
        "label": "FP8 q8 cuBLASLt matmul backend",
        "runner": "quant_bench",
        "source_backend": "fp8_q8_cublaslt_matmul",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q0",
        "backend_id": "int8_q0_cuda_naive",
        "label": "INT8 q0 naive CUDA baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q0_cuda_naive",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q1",
        "backend_id": "int8_q1_cuda_vec4cols",
        "label": "INT8 q1 register-tiled four-column CUDA baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q1_cuda_vec4cols",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q2",
        "backend_id": "int8_q2_cuda_vec8cols",
        "label": "INT8 q2 register-tiled eight-column CUDA baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q2_cuda_vec8cols",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q3",
        "backend_id": "int8_q3_cuda_vec16cols",
        "label": "INT8 q3 register-tiled sixteen-column CUDA baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q3_cuda_vec16cols",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q4",
        "backend_id": "int8_q4_wmma_m16n16k16",
        "label": "INT8 q4 WMMA M16N16K16 Tensor Core baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q4_wmma_m16n16k16",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q5",
        "backend_id": "int8_q5_wmma_m16n16k16_8warp",
        "label": "INT8 q5 WMMA M16N16K16 8-warps-per-block baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q5_wmma_m16n16k16_8warp",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q6",
        "backend_id": "int8_q6_wmma_m32n8k16",
        "label": "INT8 q6 WMMA M32N8K16 Tensor Core baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q6_wmma_m32n8k16",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q7",
        "backend_id": "int8_q7_wmma_m8n32k16",
        "label": "INT8 q7 WMMA M8N32K16 Tensor Core baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q7_wmma_m8n32k16",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q8",
        "backend_id": "int8_q8_wmma_m32n64k16_smem",
        "label": "INT8 q8 WMMA M32N64K16 shared-memory staged baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q8_wmma_m32n64k16_smem",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q9",
        "backend_id": "int8_q9_wmma_m32n32k16_reuse_a",
        "label": "INT8 q9 WMMA M32N32K16 reuse-A baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q9_wmma_m32n32k16_reuse_a",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q10",
        "backend_id": "int8_q10_wmma_m32n64k16_reuse_a",
        "label": "INT8 q10 WMMA M32N64K16 reuse-A baseline",
        "runner": "quant_bench",
        "source_backend": "int8_q10_wmma_m32n64k16_reuse_a",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q11",
        "backend_id": "int8_q11_wmma_m32n128k16_reuse_a",
        "label": "INT8 q11 WMMA M32N128K16 reuse-A pressure test",
        "runner": "quant_bench",
        "source_backend": "int8_q11_wmma_m32n128k16_reuse_a",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q12",
        "backend_id": "int8_q12_wmma_m128n64k16_4warp_reuse_a",
        "label": "INT8 q12 WMMA M128N64K16 4-warp reuse-A CTA",
        "runner": "quant_bench",
        "source_backend": "int8_q12_wmma_m128n64k16_4warp_reuse_a",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q13",
        "backend_id": "int8_q13_wmma_m128n128k16_8warp_reuse_a",
        "label": "INT8 q13 WMMA M128N128K16 8-warp reuse-A CTA",
        "runner": "quant_bench",
        "source_backend": "int8_q13_wmma_m128n128k16_8warp_reuse_a",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q14",
        "backend_id": "int8_q14_wmma_m256n64k16_8warp_reuse_a",
        "label": "INT8 q14 WMMA M256N64K16 8-warp reuse-A CTA",
        "runner": "quant_bench",
        "source_backend": "int8_q14_wmma_m256n64k16_8warp_reuse_a",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q15",
        "backend_id": "int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol",
        "label": "INT8 q15 WMMA M128N64K16 4-warp reuse-A B-col layout",
        "runner": "quant_bench",
        "source_backend": "int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q16",
        "backend_id": "int8_q16_wmma_m128n128k16_8warp_reuse_a_bcol",
        "label": "INT8 q16 WMMA M128N128K16 8-warp reuse-A B-col layout",
        "runner": "quant_bench",
        "source_backend": "int8_q16_wmma_m128n128k16_8warp_reuse_a_bcol",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q17",
        "backend_id": "int8_q17_wmma_m256n64k16_8warp_reuse_a_bcol",
        "label": "INT8 q17 WMMA M256N64K16 8-warp reuse-A B-col layout",
        "runner": "quant_bench",
        "source_backend": "int8_q17_wmma_m256n64k16_8warp_reuse_a_bcol",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q18",
        "backend_id": "int8_q18_mma_m16n8k32_smem64",
        "label": "INT8 q18 inline MMA M16N8K32 shared 64x64 tile",
        "runner": "quant_bench",
        "source_backend": "int8_q18_mma_m16n8k32_smem64",
        "implemented": True,
    },
    {
        "precision": "INT8",
        "stage": "q19",
        "backend_id": "int8_q19_cublas_gemmex",
        "label": "INT8 q19 cuBLAS GemmEx backend",
        "runner": "quant_bench",
        "source_backend": "int8_q19_cublas_gemmex",
        "implemented": True,
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_gemm_sm110_csv(path: Path, backend_id: str):
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("BackendId") == backend_id:
                return row
    return None


def build_quant_bench(root: Path, raw_root: Path) -> bool:
    build_script = root / "GEMMquant_sm110/build_and_run.sh"
    build_log = raw_root / "build_quant_gemm_sm110_bench.txt"
    if not build_script.exists():
        build_log.write_text(f"missing build script: {build_script}\n")
        return False
    build_log.parent.mkdir(parents=True, exist_ok=True)
    with build_log.open("w") as stdout:
        proc = subprocess.run(
            [str(build_script), "build-only"],
            cwd=root,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode == 0


def build_fp4_cutlass(root: Path, raw_root: Path) -> bool:
    build_script = root / "GEMMquant_sm110/build_and_run.sh"
    build_log = raw_root / "build_cutlass_72b_nvfp4_sm110.txt"
    if not build_script.exists():
        build_log.write_text(f"missing build script: {build_script}\n")
        return False
    build_log.parent.mkdir(parents=True, exist_ok=True)
    with build_log.open("w") as stdout:
        proc = subprocess.run(
            [str(build_script), "build-fp4-cutlass"],
            cwd=root,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode == 0


def missing_row(entry, n: int, trial: int, reason: str):
    return {
        "Precision": entry["precision"],
        "Stage": entry["stage"],
        "BackendId": entry["backend_id"],
        "BackendLabel": entry["label"],
        "N": str(n),
        "Reference": "cuBLAS/cuBLASLt precision reference",
        "TimeMs": "",
        "GFLOPS": "",
        "RatioToReference": "",
        "Matched": "0",
        "Status": "missing",
        "Reason": reason,
        "Trial": str(trial),
    }


def parse_cutlass_fp4_stdout(text: str):
    passed = "Disposition: Passed" in text
    time_match = re.search(r"Avg runtime:\s*([0-9.eE+-]+)\s*ms", text)
    gflops_match = re.search(r"GFLOPS:\s*([0-9.eE+-]+)", text)
    time_ms = float(time_match.group(1)) if time_match else 0.0
    gflops = float(gflops_match.group(1)) if gflops_match else 0.0
    return passed, time_ms, gflops


def run_gemm_sm110_backend(entry, bench_bin: Path, raw_root: Path, n: int, trial: int, timeout: int):
    run_dir = raw_root / f"N{n}" / f"trial_{trial}" / entry["backend_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "stdout.txt"
    with stdout_path.open("w") as stdout:
        proc = subprocess.run(
            ["timeout", "--foreground", f"{timeout}s", str(bench_bin), str(n), entry["source_backend"]],
            cwd=run_dir,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        return missing_row(
            entry,
            n,
            trial,
            f"source backend exited with status {proc.returncode}; see {stdout_path}",
        )

    source_csv = run_dir / "sgemm_sm110_benchmark.csv"
    source_row = parse_gemm_sm110_csv(source_csv, entry["source_backend"])
    if source_row is None:
        return missing_row(entry, n, trial, f"source CSV row missing in {source_csv}")

    matched = source_row.get("Matched", "0")
    return {
        "Precision": entry["precision"],
        "Stage": entry["stage"],
        "BackendId": entry["backend_id"],
        "BackendLabel": entry["label"],
        "N": source_row.get("N", str(n)),
        "Reference": source_row.get("Reference", ""),
        "TimeMs": source_row.get("TimeMs", "0"),
        "GFLOPS": source_row.get("GFLOPS", "0"),
        "RatioToReference": source_row.get("RatioToReference", "0"),
        "Matched": matched,
        "Status": "ok" if matched == "1" else "failed",
        "Reason": "" if matched == "1" else "source backend did not match reference",
        "Trial": str(trial),
    }


def run_cutlass_fp4_backend(
    entry,
    cutlass_bin: Path,
    bench_bin: Path,
    raw_root: Path,
    n: int,
    trial: int,
    timeout: int,
):
    run_dir = raw_root / f"N{n}" / f"trial_{trial}" / entry["backend_id"]
    reference_dir = run_dir / "reference_cublas_tc"
    run_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    if not bench_bin.exists():
        return missing_row(
            entry,
            n,
            trial,
            f"missing source benchmark binary for reference: {bench_bin}",
        )

    reference_stdout_path = reference_dir / "stdout.txt"
    with reference_stdout_path.open("w") as stdout:
        ref_proc = subprocess.run(
            ["timeout", "--foreground", f"{timeout}s", str(bench_bin), str(n), "cublas_tc"],
            cwd=reference_dir,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if ref_proc.returncode != 0:
        return missing_row(
            entry,
            n,
            trial,
            f"reference cublas_tc exited with status {ref_proc.returncode}; see {reference_stdout_path}",
        )

    reference_csv = reference_dir / "sgemm_sm110_benchmark.csv"
    reference_row = parse_gemm_sm110_csv(reference_csv, "cublas_tc")
    if reference_row is None:
        return missing_row(entry, n, trial, f"reference CSV row missing in {reference_csv}")
    try:
        reference_gflops = float(reference_row.get("GFLOPS", "0"))
    except ValueError:
        reference_gflops = 0.0
    if reference_gflops <= 0.0:
        return missing_row(entry, n, trial, f"invalid reference GFLOPS in {reference_csv}")

    stdout_path = run_dir / "stdout.txt"
    cmd = [
        "timeout",
        "--foreground",
        f"{timeout}s",
        str(cutlass_bin),
        f"--m={n}",
        f"--n={n}",
        f"--k={n}",
        f"--iterations={entry.get('iterations', 100)}",
        f"--swizzle={entry.get('swizzle', 2)}",
    ]
    with stdout_path.open("w") as stdout:
        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        return missing_row(
            entry,
            n,
            trial,
            f"CUTLASS FP4 backend exited with status {proc.returncode}; see {stdout_path}",
        )

    passed, time_ms, gflops = parse_cutlass_fp4_stdout(stdout_path.read_text())
    if gflops <= 0.0:
        return missing_row(entry, n, trial, f"CUTLASS FP4 metrics missing in {stdout_path}")

    ratio = gflops / reference_gflops
    matched = "1" if passed else "0"
    return {
        "Precision": entry["precision"],
        "Stage": entry["stage"],
        "BackendId": entry["backend_id"],
        "BackendLabel": entry["label"],
        "N": str(n),
        "Reference": "cuBLAS Tensor Core fp16->fp32",
        "TimeMs": f"{time_ms:.6g}",
        "GFLOPS": f"{gflops:.6g}",
        "RatioToReference": f"{ratio:.6g}",
        "Matched": matched,
        "Status": "ok" if matched == "1" else "failed",
        "Reason": "" if matched == "1" else "CUTLASS reference comparison failed",
        "Trial": str(trial),
    }


def run_quant_bench_backend(entry, bench_bin: Path, raw_root: Path, n: int, trial: int, timeout: int):
    run_dir = raw_root / f"N{n}" / f"trial_{trial}" / entry["backend_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "stdout.txt"
    with stdout_path.open("w") as stdout:
        proc = subprocess.run(
            ["timeout", "--foreground", f"{timeout}s", str(bench_bin), str(n), entry["source_backend"]],
            cwd=run_dir,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        return missing_row(
            entry,
            n,
            trial,
            f"quant bench exited with status {proc.returncode}; see {stdout_path}",
        )

    source_csv = run_dir / "quant_sm110_benchmark.csv"
    source_row = parse_gemm_sm110_csv(source_csv, entry["source_backend"])
    if source_row is None:
        return missing_row(entry, n, trial, f"source CSV row missing in {source_csv}")

    matched = source_row.get("Matched", "0")
    return {
        "Precision": entry["precision"],
        "Stage": entry["stage"],
        "BackendId": entry["backend_id"],
        "BackendLabel": entry["label"],
        "N": source_row.get("N", str(n)),
        "Reference": source_row.get("Reference", ""),
        "TimeMs": source_row.get("TimeMs", ""),
        "GFLOPS": source_row.get("GFLOPS", ""),
        "RatioToReference": source_row.get("RatioToReference", ""),
        "Matched": matched,
        "Status": "ok" if matched == "1" else "failed",
        "Reason": "" if matched == "1" else "source backend did not match reference",
        "Trial": str(trial),
    }


def main():
    parser = argparse.ArgumentParser(description="Run SM110 quantized GEMM 1024 harness")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--bench-bin", type=Path, default=None)
    parser.add_argument("--quant-bench-bin", type=Path, default=None)
    parser.add_argument("--cutlass-fp4-bin", type=Path, default=None)
    parser.add_argument("--cutlass-mxfp4-bin", type=Path, default=None)
    args = parser.parse_args()

    root = repo_root()
    out_csv = args.out or root / "results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv"
    raw_dir = args.raw_dir or root / "results/quant_gemm_sm110/raw"
    bench_bin = args.bench_bin or root / "GEMMsm110/build/gemm_sm110_bench"
    quant_bench_bin = args.quant_bench_bin or root / "GEMMquant_sm110/build/quant_gemm_sm110_bench"
    cutlass_fp4_bin = args.cutlass_fp4_bin or root / "GEMMquant_sm110/build/cutlass_72b_nvfp4_nvfp4_sm110"
    cutlass_mxfp4_bin = args.cutlass_mxfp4_bin or root / "GEMMquant_sm110/build/cutlass_72a_mxfp4_bf16_sm110"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    need_quant_bench = any(
        entry.get("implemented") and entry.get("runner") == "quant_bench"
        for entry in PLANNED_BACKENDS
    )
    if need_quant_bench and not quant_bench_bin.exists():
        build_quant_bench(root, raw_dir)
    need_cutlass_fp4 = any(
        entry.get("implemented")
        and entry.get("runner") in {"cutlass_fp4_72b", "cutlass_mxfp4_72a"}
        for entry in PLANNED_BACKENDS
    )
    if need_cutlass_fp4 and (not cutlass_fp4_bin.exists() or not cutlass_mxfp4_bin.exists()):
        build_fp4_cutlass(root, raw_dir)

    rows = []
    for trial in range(1, args.trials + 1):
        for entry in PLANNED_BACKENDS:
            if entry.get("implemented"):
                if entry.get("runner") == "gemm_sm110" and not bench_bin.exists():
                    rows.append(
                        missing_row(
                            entry,
                            args.n,
                            trial,
                            f"missing source benchmark binary: {bench_bin}",
                        )
                    )
                elif entry.get("runner") == "quant_bench" and not quant_bench_bin.exists():
                    rows.append(
                        missing_row(
                            entry,
                            args.n,
                            trial,
                            f"missing quant benchmark binary: {quant_bench_bin}",
                        )
                    )
                elif entry.get("runner") == "quant_bench":
                    rows.append(run_quant_bench_backend(entry, quant_bench_bin, raw_dir, args.n, trial, args.timeout))
                elif entry.get("runner") == "cutlass_fp4_72b" and not cutlass_fp4_bin.exists():
                    rows.append(
                        missing_row(
                            entry,
                            args.n,
                            trial,
                            f"missing CUTLASS FP4 binary: {cutlass_fp4_bin}",
                        )
                    )
                elif entry.get("runner") == "cutlass_fp4_72b":
                    rows.append(
                        run_cutlass_fp4_backend(
                            entry,
                            cutlass_fp4_bin,
                            bench_bin,
                            raw_dir,
                            args.n,
                            trial,
                            args.timeout,
                        )
                    )
                elif entry.get("runner") == "cutlass_mxfp4_72a" and not cutlass_mxfp4_bin.exists():
                    rows.append(
                        missing_row(
                            entry,
                            args.n,
                            trial,
                            f"missing CUTLASS MXFP4 binary: {cutlass_mxfp4_bin}",
                        )
                    )
                elif entry.get("runner") == "cutlass_mxfp4_72a":
                    rows.append(
                        run_cutlass_fp4_backend(
                            entry,
                            cutlass_mxfp4_bin,
                            bench_bin,
                            raw_dir,
                            args.n,
                            trial,
                            args.timeout,
                        )
                    )
                else:
                    rows.append(run_gemm_sm110_backend(entry, bench_bin, raw_dir, args.n, trial, args.timeout))
            else:
                rows.append(missing_row(entry, args.n, trial, entry["reason"]))

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
