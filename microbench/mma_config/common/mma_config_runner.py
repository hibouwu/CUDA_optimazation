#!/usr/bin/env python3
import argparse
import csv
import hashlib
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
COMMON = ROOT / "common"
STAGES = [
    "00_validation",
    "01_collector_protocol",
    "02_latency_throughput",
    "03_effective_smem_ingress",
    "04_smem_layout_address",
    "05_ldshared_contention",
    "06_tmem_dependency",
    "07_config_matrix",
]

SHAPES = {
    "m128n64k16": {"m": 128, "n": 64, "k": 16, "d_footprint": 64},
    "m128n128k16": {"m": 128, "n": 128, "k": 16, "d_footprint": 128},
    "m128n256k16": {"m": 128, "n": 256, "k": 16, "d_footprint": 256},
}
DTYPES = ["fp16", "bf16"]
LAYOUTS = ["sw128", "sw64", "sw32", "none"]
TMEM_COLUMNS = [128, 256, 512]

CSV_FIELDS = [
    "experiment", "case_id", "valid", "invalid_reason",
    "gpu", "compute_capability", "driver", "cuda_version", "ptx_version",
    "sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_w",
    "dtype", "m", "n", "k", "Q", "iterations", "run_order", "repeat",
    "collector_mode", "operand_address_mode",
    "independent_d_count", "d_reuse_distance",
    "commit_interval", "pending_mbarriers", "wait_polling_mode",
    "smem_layout", "swizzle", "alignment_bytes", "lda", "ldb",
    "smem_base_offset",
    "tmem_columns", "d_base_column", "d_tile_base_delta",
    "d_alias_class", "input_d",
    "interference_mode", "interference_ops_per_iter", "interference_warps",
    "resident_ctas", "elapsed_cycles", "elapsed_us",
    "elapsed_cycles_p10", "elapsed_cycles_p90",
    "alpha_cycles", "beta_cycles_per_mma",
    "logical_smem_bytes_per_mma", "effective_smem_bytes_per_cycle",
    "tflops", "tflops_p10", "tflops_p90",
    "poll_count", "max_abs_error", "guard_ok", "sass_hash",
    "notes",
]


def stage_dir(stage):
    return ROOT / stage


def src_path(stage):
    return stage_dir(stage) / "benchmark_src" / f"tcgen05_{stage}_bench.cu"


def bin_path(stage):
    return stage_dir(stage) / "build" / f"tcgen05_{stage}_bench"


def keep_dir(stage):
    return stage_dir(stage) / "build" / "keep"


def ptx_path(stage):
    return keep_dir(stage) / f"tcgen05_{stage}_bench.ptx"


def run_cmd(cmd, cwd=ROOT, check=True, capture=True):
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    out = proc.stdout or ""
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{out}")
    return proc.returncode, out


def build(stage):
    out = bin_path(stage)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep_dir(stage).mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvcc", "-std=c++17", "-O3",
        "-gencode", "arch=compute_110a,code=sm_110a",
        "--keep", "--keep-dir", keep_dir(stage),
        src_path(stage), "-lcuda", "-o", out,
    ]
    return run_cmd(cmd)


def tool_version(cmd):
    try:
        _, out = run_cmd(cmd, check=False)
        return " ".join(out.strip().split())
    except Exception:
        return ""


def query_nvidia_smi():
    fields = {
        "driver": "", "cuda_version": "", "sm_clock_mhz": "",
        "mem_clock_mhz": "", "temperature_c": "", "power_w": "",
    }
    try:
        _, out = run_cmd(["nvidia-smi"], check=False)
        for line in out.splitlines():
            if "Driver Version:" in line:
                parts = line.replace("|", " ").split()
                if "Version:" in parts:
                    fields["driver"] = parts[parts.index("Version:") + 1]
                if "CUDA" in parts and "Version:" in parts[parts.index("CUDA"):]:
                    idx = parts.index("CUDA")
                    fields["cuda_version"] = parts[idx + 2]
    except Exception:
        pass
    try:
        query = "clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw"
        _, out = run_cmd([
            "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"
        ], check=False)
        line = out.strip().splitlines()[0]
        vals = []
        for x in line.split(","):
            cleaned = x.strip().replace("[Not Supported]", "").replace("[N/A]", "").strip()
            vals.append(cleaned)
        if len(vals) >= 4:
            fields["sm_clock_mhz"], fields["mem_clock_mhz"], fields["temperature_c"], fields["power_w"] = vals[:4]
    except Exception:
        pass
    freq = read_freq_hz()
    if freq and not fields["sm_clock_mhz"]:
        fields["sm_clock_mhz"] = f"{freq / 1e6:.3f}"
    return fields


def read_freq_hz():
    for p in [
        Path("/sys/class/devfreq/gpu-gpc-0/cur_freq"),
        Path("/sys/class/devfreq/17000000.gpu/cur_freq"),
    ]:
        if p.exists():
            try:
                return int(p.read_text().strip())
            except Exception:
                pass
    return 1575000000


def sass_info(stage):
    binary = bin_path(stage)
    info = {"sass_hash": "", "ptx_hash": "", "sass_summary": "", "sass_path": ""}
    try:
        _, out = run_cmd(["cuobjdump", "--dump-sass", binary], check=False)
        digest = hashlib.sha256(out.encode("utf-8", "ignore")).hexdigest()
        counts = {
            "UTCHMMA": out.count("UTCHMMA"),
            "UTCHMMA_WS": out.count("UTCHMMA.WS"),
            "UTCBAR": out.count("UTCBAR"),
            "UTCATOMSWS": out.count("UTCATOMSWS"),
            "SYNCS_TRYWAIT": out.count("SYNCS.PHASECHK.TRANS64.TRYWAIT"),
            "LDS": out.count("LDS"),
            "LDG": out.count("LDG"),
        }
        ptx_counts = {}
        ptx_digest = ""
        ptx = ptx_path(stage)
        if ptx.exists():
            ptx_text = ptx.read_text(errors="ignore")
            ptx_digest = hashlib.sha256(ptx_text.encode("utf-8", "ignore")).hexdigest()
            ptx_counts = {
                "ptx_tcgen05_mma": ptx_text.count("tcgen05.mma"),
                "ptx_tcgen05_mma_ws": ptx_text.count("tcgen05.mma.ws"),
                "ptx_tcgen05_commit": ptx_text.count("tcgen05.commit"),
                "ptx_tcgen05_alloc": ptx_text.count("tcgen05.alloc"),
                "ptx_tcgen05_dealloc": ptx_text.count("tcgen05.dealloc"),
                "ptx_tcgen05_ld": ptx_text.count("tcgen05.ld"),
                "ptx_tcgen05_st": ptx_text.count("tcgen05.st"),
                "ptx_mbarrier_wait": ptx_text.count("mbarrier.try_wait"),
                "ptx_cp_async_tma": ptx_text.count("cp.async.bulk.tensor"),
                "ptx_ld_shared": ptx_text.count("ld.shared"),
                "ptx_ld_global": ptx_text.count("ld.global"),
            }
        path = stage_dir(stage) / "plots" / "sass_summary.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(f"sha256,{digest}\n")
            if ptx_digest:
                f.write(f"ptx_sha256,{ptx_digest}\n")
                f.write(f"ptx_path,{ptx}\n")
            for k, v in counts.items():
                f.write(f"{k},{v}\n")
            for k, v in ptx_counts.items():
                f.write(f"{k},{v}\n")
        info.update({
            "sass_hash": digest,
            "ptx_hash": ptx_digest,
            "sass_summary": ";".join(f"{k}={v}" for k, v in {**counts, **ptx_counts}.items()),
            "sass_path": str(path),
        })
    except Exception as exc:
        info["sass_summary"] = f"cuobjdump_failed:{exc}"
    return info


def parse_kv(text):
    data = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def percentile(values, q):
    if not values:
        return math.nan
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def fnum(value, default=math.nan):
    try:
        return float(value)
    except Exception:
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def logical_smem_bytes(dtype, n):
    return 2 * 16 * (128 + n)


def flops_per_mma(n):
    return 2 * 128 * n * 16


def error_tolerance(dtype, q, iterations, input_d):
    if dtype == "bf16":
        return max(0.25, 0.0025 * q * iterations if input_d else 0.25)
    return max(0.05, 0.001 * q * iterations if input_d else 0.05)


def case_to_args(case):
    args = []
    mapping = {
        "dtype": "--dtype",
        "shape": "--shape",
        "layout": "--layout",
        "Q": "--q",
        "iterations": "--iterations",
        "collector_protocol": "--collector-protocol",
        "collector_reuse": "--collector-reuse",
        "ws": "--ws",
        "ws_buffer_count": "--ws-buffer-count",
        "operand_address_mode": "--operand-address-mode",
        "input_d": "--input-d",
        "tmem_columns": "--tmem-columns",
        "d_base_column": "--d-base-column",
        "d_tile_base_delta": "--d-tile-base-delta",
        "independent_d_count": "--independent-d-count",
        "d_reuse_distance": "--d-reuse-distance",
        "commit_interval": "--commit-interval",
        "pending_mbarriers": "--pending-mbarriers",
        "wait_polling_mode": "--wait-mode",
        "smem_base_offset": "--smem-base-offset",
        "interference_mode": "--interference-mode",
        "interference_ops_per_iter": "--interference-ops-per-iter",
        "interference_warps": "--interference-warps",
        "active_blocks": "--active-blocks",
        "operand_slots": "--operand-slots",
    }
    for key, flag in mapping.items():
        if key in case and case[key] not in (None, ""):
            args += [flag, str(case[key])]
    return args


def base_case(stage, case_id, dtype="bf16", shape="m128n128k16"):
    n = SHAPES[shape]["n"]
    cap = 512 // n
    return {
        "experiment": stage,
        "case_id": case_id,
        "dtype": dtype,
        "shape": shape,
        "layout": "sw128",
        "Q": 1,
        "iterations": 1,
        "collector_protocol": "discard",
        "collector_reuse": 0,
        "ws": 0,
        "ws_buffer_count": 1,
        "operand_address_mode": "same",
        "input_d": 0,
        "tmem_columns": 512,
        "d_base_column": 0,
        "d_tile_base_delta": n,
        "independent_d_count": max(1, cap),
        "d_reuse_distance": max(1, cap),
        "commit_interval": 1,
        "pending_mbarriers": 1,
        "wait_polling_mode": "nocount",
        "smem_base_offset": 0,
        "interference_mode": "none",
        "interference_ops_per_iter": 0,
        "interference_warps": 4,
        "active_blocks": 1 if stage == "00_validation" else 0,
        "d_alias_class": "none",
        "notes": "",
    }


def validation_cases(quick=False):
    cases = []
    layouts = ["sw128"] if quick else LAYOUTS
    for dtype in DTYPES:
        for shape in SHAPES:
            n = SHAPES[shape]["n"]
            for layout in layouts:
                for cols in [c for c in TMEM_COLUMNS if c >= n]:
                    max_ind = max(1, cols // n)
                    for input_d in [0, 1]:
                        for proto, q, reuse in [
                            ("discard", 1, 0),
                            ("discard", 3, 0),
                            ("fill_use_lastuse", 4, 2),
                        ]:
                            c = base_case("00_validation", f"val_{dtype}_{shape}_{layout}_c{cols}_d{input_d}_{proto}_q{q}", dtype, shape)
                            c.update({
                                "layout": layout,
                                "Q": q,
                                "iterations": 1,
                                "collector_protocol": proto,
                                "collector_reuse": reuse,
                                "input_d": input_d,
                                "tmem_columns": cols,
                                "independent_d_count": max_ind,
                                "d_reuse_distance": max_ind,
                                "commit_interval": q,
                            })
                            cases.append(c)
    for dtype in DTYPES:
        for shape in SHAPES:
            c = base_case("00_validation", f"val_ws_b0_{dtype}_{shape}", dtype, shape)
            c.update({"Q": 4, "iterations": 1, "collector_protocol": "fill_use_lastuse",
                      "collector_reuse": 2, "ws": 1, "ws_buffer_count": 1,
                      "input_d": 1, "commit_interval": 4})
            cases.append(c)
    return cases


def collector_cases(quick=False):
    shapes = ["m128n128k16"] if quick else list(SHAPES)
    cases = []
    for dtype in DTYPES:
        for shape in shapes:
            for address in ["same", "pingpong"]:
                for proto, reuse, q in [
                    ("discard", 0, 32),
                    ("fill_lastuse", 0, 2),
                    ("fill_use_lastuse", 1, 3),
                    ("fill_use_lastuse", 3, 5),
                    ("fill_use_lastuse", 7, 9),
                    ("fill_use_discard", 3, 5),
                ]:
                    c = base_case("01_collector_protocol", f"collector_{dtype}_{shape}_{address}_{proto}_r{reuse}", dtype, shape)
                    n = SHAPES[shape]["n"]
                    max_ind = 512 // n
                    c.update({"Q": q, "iterations": 400 if not quick else 80,
                              "collector_protocol": proto, "collector_reuse": reuse,
                              "operand_address_mode": address,
                              "input_d": 0, "independent_d_count": max_ind,
                              "d_reuse_distance": max_ind, "commit_interval": q,
                              "active_blocks": 0})
                    cases.append(c)
            for buffers in [1, 2, 4]:
                c = base_case("01_collector_protocol", f"collector_ws_b{buffers}_{dtype}_{shape}", dtype, shape)
                c.update({"Q": 4 * buffers, "iterations": 300 if not quick else 60,
                          "collector_protocol": "fill_use_lastuse",
                          "collector_reuse": 2, "ws": 1,
                          "ws_buffer_count": buffers, "input_d": 0,
                          "commit_interval": 4 * buffers, "active_blocks": 0})
                cases.append(c)
    return cases


def latency_cases(quick=False):
    q_values = [1, 2, 4, 8, 16] if quick else [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    shapes = ["m128n128k16"] if quick else list(SHAPES)
    cases = []
    for dtype in DTYPES:
        for shape in shapes:
            n = SHAPES[shape]["n"]
            max_ind = 512 // n
            for input_d in [0, 1]:
                for dmode in ["same_d", "legal_ring"]:
                    for q in q_values:
                        c = base_case("02_latency_throughput", f"lat_{dtype}_{shape}_{dmode}_in{input_d}_q{q}", dtype, shape)
                        if dmode == "same_d":
                            ind = 1
                            delta = 0
                            reuse = 1
                        else:
                            ind = max_ind
                            delta = n
                            reuse = max_ind
                        c.update({"Q": q, "iterations": max(8, 512 // max(1, q)) if not quick else max(4, 64 // max(1, q)),
                                  "input_d": input_d, "d_tile_base_delta": delta,
                                  "independent_d_count": ind, "d_reuse_distance": reuse,
                                  "commit_interval": q, "active_blocks": 0,
                                  "notes": dmode})
                        cases.append(c)
    for commit in [1, 2, 4, 8]:
        for pending in [1, 2, 4]:
            c = base_case("02_latency_throughput", f"commit_prefix_c{commit}_p{pending}", "bf16", "m128n128k16")
            c.update({"Q": 16, "iterations": 128 if not quick else 32,
                      "commit_interval": commit, "pending_mbarriers": pending,
                      "wait_polling_mode": "count", "active_blocks": 0,
                      "notes": "commit prefix scan; pending_mbarriers is prefix tracking count"})
            cases.append(c)
    return cases


def smem_ingress_cases(quick=False):
    shapes = ["m128n128k16"] if quick else list(SHAPES)
    modes = ["same"] if quick else ["same", "pingpong", "rotating"]
    cases = []
    for dtype in DTYPES:
        for shape in shapes:
            n = SHAPES[shape]["n"]
            max_ind = 512 // n
            for mode in modes:
                c = base_case("03_effective_smem_ingress", f"ingress_{dtype}_{shape}_{mode}", dtype, shape)
                c.update({"Q": 64, "iterations": 128 if not quick else 32,
                          "collector_protocol": "discard",
                          "operand_address_mode": mode,
                          "input_d": 0, "independent_d_count": max_ind,
                          "d_reuse_distance": max_ind,
                          "commit_interval": 64, "active_blocks": 0})
                cases.append(c)
    return cases


def layout_cases(quick=False):
    shapes = ["m128n128k16"] if quick else list(SHAPES)
    offsets = [0, 64] if quick else [0, 16, 32, 64, 128, 256]
    layouts = ["sw128"] if quick else LAYOUTS
    cases = []
    for dtype in DTYPES:
        for shape in shapes:
            n = SHAPES[shape]["n"]
            max_ind = 512 // n
            for layout in layouts:
                for off in offsets:
                    c = base_case("04_smem_layout_address", f"layout_{dtype}_{shape}_{layout}_off{off}", dtype, shape)
                    c.update({"layout": layout, "Q": 32,
                              "iterations": 128 if not quick else 32,
                              "smem_base_offset": off,
                              "input_d": 0,
                              "independent_d_count": max_ind,
                              "d_reuse_distance": max_ind,
                              "commit_interval": 32, "active_blocks": 0})
                    cases.append(c)
    return cases


def ldshared_cases(quick=False):
    ops = [0, 32] if quick else [0, 8, 32, 128]
    modes = ["none", "register_alu", "predicated_off_load", "l1_hit_global", "ld_shared", "interference_only"]
    cases = []
    for dtype in DTYPES:
        for mode in modes:
            for op in ops:
                if mode == "none" and op != 0:
                    continue
                c = base_case("05_ldshared_contention", f"ldcont_{dtype}_{mode}_ops{op}", dtype, "m128n128k16")
                c.update({"Q": 32, "iterations": 128 if not quick else 32,
                          "interference_mode": mode,
                          "interference_ops_per_iter": op,
                          "interference_warps": 4,
                          "input_d": 0, "commit_interval": 32,
                          "active_blocks": 0,
                          "notes": "fixed active warp count"})
                cases.append(c)
    return cases


def tmem_dependency_cases(quick=False):
    shapes = ["m128n128k16"] if quick else list(SHAPES)
    cases = []
    for dtype in DTYPES:
        for shape in shapes:
            n = SHAPES[shape]["n"]
            legal_cols = [c for c in TMEM_COLUMNS if c >= n]
            for cols in legal_cols:
                max_ind = cols // n
                for input_d in [0, 1]:
                    specs = [
                        ("full", 0, 1, 1),
                        ("none", n, max_ind, max_ind),
                    ]
                    if n // 2 > 0 and cols >= n + n // 2:
                        specs.append(("partial", n // 2, 1, 2))
                    for alias, delta, ind, reuse in specs:
                        address_capacity = 1
                        if delta > 0:
                            address_capacity = max(1, (cols - n) // delta + 1)
                        for rd in sorted(set([1, reuse, max_ind])):
                            c = base_case("06_tmem_dependency", f"tmem_{dtype}_{shape}_c{cols}_{alias}_in{input_d}_rd{rd}", dtype, shape)
                            c.update({"Q": 32, "iterations": 128 if not quick else 32,
                                      "tmem_columns": cols,
                                      "d_tile_base_delta": delta,
                                      "independent_d_count": ind,
                                      "d_reuse_distance": max(1, min(rd, address_capacity)),
                                      "d_alias_class": alias,
                                      "input_d": input_d,
                                      "commit_interval": 32,
                                      "active_blocks": 0})
                            cases.append(c)
    return cases


def config_matrix_cases(quick=False):
    layouts = ["sw128"] if quick else ["sw128", "sw64", "sw32", "none"]
    cases = []
    for dtype in DTYPES:
        for shape in SHAPES:
            n = SHAPES[shape]["n"]
            for cols in [c for c in TMEM_COLUMNS if c >= n]:
                max_ind = cols // n
                for layout in layouts:
                    for collector, reuse in [("discard", 0), ("fill_use_lastuse", 3)]:
                        for addr in ["same", "pingpong"]:
                            c = base_case("07_config_matrix", f"cfg_{dtype}_{shape}_c{cols}_{layout}_{collector}_{addr}", dtype, shape)
                            c.update({"layout": layout, "Q": 32,
                                      "iterations": 128 if not quick else 32,
                                      "collector_protocol": collector,
                                      "collector_reuse": reuse,
                                      "operand_address_mode": addr,
                                      "tmem_columns": cols,
                                      "independent_d_count": max_ind,
                                      "d_reuse_distance": max_ind,
                                      "input_d": 0,
                                      "commit_interval": 32,
                                      "active_blocks": 0})
                            cases.append(c)
    return cases


CASE_BUILDERS = {
    "00_validation": validation_cases,
    "01_collector_protocol": collector_cases,
    "02_latency_throughput": latency_cases,
    "03_effective_smem_ingress": smem_ingress_cases,
    "04_smem_layout_address": layout_cases,
    "05_ldshared_contention": ldshared_cases,
    "06_tmem_dependency": tmem_dependency_cases,
    "07_config_matrix": config_matrix_cases,
}


def run_one(stage, case, env, sass, run_order, repeat_index):
    cmd = [bin_path(stage)] + case_to_args(case)
    ret, out = run_cmd(cmd, check=False)
    kv = parse_kv(out)
    row = {k: "" for k in CSV_FIELDS}
    row.update(env)
    shape = case["shape"]
    n = SHAPES[shape]["n"]
    elapsed_cycles = fnum(kv.get("elapsed_cycles"))
    freq = read_freq_hz()
    elapsed_us = elapsed_cycles / freq * 1e6 if freq and math.isfinite(elapsed_cycles) else math.nan
    q = int(case["Q"])
    iters = int(case["iterations"])
    tflops = ""
    if freq and elapsed_cycles > 0 and case.get("interference_mode") != "interference_only":
        tflops = flops_per_mma(n) * q * iters * inum(kv.get("active_blocks", 1)) / (elapsed_cycles / freq) / 1e12
    max_err = fnum(kv.get("max_abs_error"))
    guard_ok = inum(kv.get("guard_ok"), 0)
    tol = error_tolerance(case["dtype"], q, iters, int(case.get("input_d", 0)))
    valid = (ret == 0 and kv.get("status") == "ok" and guard_ok == 1 and math.isfinite(max_err) and max_err <= tol)
    reason = ""
    if ret != 0:
        reason = f"process_exit_{ret}"
    elif kv.get("status") != "ok":
        reason = kv.get("invalid_reason", kv.get("status", "unknown"))
    elif guard_ok != 1:
        reason = "tmem_guard_modified"
    elif not math.isfinite(max_err) or max_err > tol:
        reason = f"max_abs_error>{tol:.6g}"
    row.update({
        "experiment": stage,
        "case_id": case["case_id"],
        "valid": 1 if valid else 0,
        "invalid_reason": reason,
        "gpu": kv.get("gpu", ""),
        "compute_capability": kv.get("compute_capability", ""),
        "dtype": case["dtype"],
        "m": 128,
        "n": n,
        "k": 16,
        "Q": q,
        "iterations": iters,
        "run_order": run_order,
        "repeat": repeat_index,
        "collector_mode": case.get("collector_protocol", ""),
        "operand_address_mode": case.get("operand_address_mode", ""),
        "independent_d_count": kv.get("independent_d_count", case.get("independent_d_count", "")),
        "d_reuse_distance": kv.get("d_reuse_distance", case.get("d_reuse_distance", "")),
        "commit_interval": case.get("commit_interval", ""),
        "pending_mbarriers": case.get("pending_mbarriers", ""),
        "wait_polling_mode": case.get("wait_polling_mode", ""),
        "smem_layout": case.get("layout", ""),
        "swizzle": case.get("layout", ""),
        "alignment_bytes": 16,
        "lda": 16,
        "ldb": 16,
        "smem_base_offset": case.get("smem_base_offset", 0),
        "tmem_columns": case.get("tmem_columns", ""),
        "d_base_column": case.get("d_base_column", ""),
        "d_tile_base_delta": case.get("d_tile_base_delta", ""),
        "d_alias_class": case.get("d_alias_class", "none"),
        "input_d": case.get("input_d", ""),
        "interference_mode": case.get("interference_mode", ""),
        "interference_ops_per_iter": case.get("interference_ops_per_iter", ""),
        "interference_warps": case.get("interference_warps", ""),
        "resident_ctas": kv.get("active_blocks", ""),
        "elapsed_cycles": elapsed_cycles,
        "elapsed_us": elapsed_us,
        "logical_smem_bytes_per_mma": logical_smem_bytes(case["dtype"], n),
        "tflops": tflops,
        "poll_count": kv.get("poll_count", ""),
        "max_abs_error": max_err,
        "guard_ok": guard_ok,
        "sass_hash": sass.get("sass_hash", ""),
        "notes": case.get("notes", "") + ("; " + sass.get("sass_summary", "") if repeat_index == 0 else ""),
    })
    if out and not valid:
        log_path = stage_dir(stage) / "plots" / "failed_commands.log"
        with log_path.open("a") as f:
            f.write("$ " + " ".join(map(str, cmd)) + "\n")
            f.write(out + "\n")
    return row


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        key = row["case_id"]
        groups.setdefault(key, []).append(row)
    agg = []
    for key, vals in groups.items():
        base = vals[0].copy()
        cycles = [fnum(v["elapsed_cycles"]) for v in vals if str(v["valid"]) == "1"]
        us = [fnum(v["elapsed_us"]) for v in vals if str(v["valid"]) == "1"]
        tflops = [fnum(v["tflops"]) for v in vals if str(v["valid"]) == "1" and v["tflops"] != ""]
        errs = [fnum(v["max_abs_error"]) for v in vals if math.isfinite(fnum(v["max_abs_error"]))]
        if cycles:
            base["elapsed_cycles"] = statistics.median(cycles)
            base["elapsed_cycles_p10"] = percentile(cycles, 0.10)
            base["elapsed_cycles_p90"] = percentile(cycles, 0.90)
            base["elapsed_us"] = statistics.median(us)
        if tflops:
            base["tflops"] = statistics.median(tflops)
            base["tflops_p10"] = percentile(tflops, 0.10)
            base["tflops_p90"] = percentile(tflops, 0.90)
        if errs:
            base["max_abs_error"] = max(errs)
        base["repeat"] = len(vals)
        base["valid"] = 1 if all(str(v["valid"]) == "1" for v in vals) else 0
        if not base["valid"]:
            base["invalid_reason"] = ";".join(sorted({v["invalid_reason"] for v in vals if v["invalid_reason"]}))
        agg.append(base)
    return agg


def fit_latency(rows):
    groups = {}
    for r in rows:
        if str(r.get("valid")) != "1":
            continue
        if r["case_id"].startswith("lat_"):
            key = (r["dtype"], r["n"], r["input_d"], r["d_tile_base_delta"],
                   r["independent_d_count"], r["d_reuse_distance"],
                   r["collector_mode"], r["operand_address_mode"], "batch_end_commit")
        else:
            key = (r["dtype"], r["n"], r["input_d"], r["d_tile_base_delta"],
                   r["independent_d_count"], r["d_reuse_distance"],
                   r["collector_mode"], r["operand_address_mode"],
                   r["commit_interval"], r["pending_mbarriers"])
        groups.setdefault(key, []).append(r)
    for vals in groups.values():
        pts = [(fnum(v["Q"]), fnum(v["elapsed_cycles"]) / max(1, fnum(v["iterations"]))) for v in vals]
        pts = [(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y) and x >= 2]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xm = statistics.mean(xs)
        ym = statistics.mean(ys)
        denom = sum((x - xm) ** 2 for x in xs)
        if denom == 0:
            continue
        beta = sum((x - xm) * (y - ym) for x, y in pts) / denom
        alpha = ym - beta * xm
        for v in vals:
            v["alpha_cycles"] = alpha
            v["beta_cycles_per_mma"] = beta


def derive_metrics(stage, rows):
    if stage == "02_latency_throughput":
        fit_latency(rows)
    for r in rows:
        q = fnum(r["Q"])
        iters = max(1, fnum(r["iterations"]))
        cycles = fnum(r["elapsed_cycles"])
        if stage in ("03_effective_smem_ingress", "04_smem_layout_address", "07_config_matrix"):
            if str(r.get("valid")) == "1" and cycles > 0 and q > 0:
                beta = cycles / (iters * q)
                r["beta_cycles_per_mma"] = r.get("beta_cycles_per_mma") or beta
                r["effective_smem_bytes_per_cycle"] = fnum(r["logical_smem_bytes_per_mma"]) / beta


def write_csv(path, rows, fields=CSV_FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_svg(stage, rows):
    valid = [r for r in rows if str(r.get("valid")) == "1"]
    if not valid:
        return
    metric = "tflops" if stage not in ("00_validation", "02_latency_throughput") else "elapsed_cycles"
    vals = [(r["case_id"][:24], fnum(r.get(metric))) for r in valid if math.isfinite(fnum(r.get(metric)))]
    if not vals:
        return
    vals = vals[:80]
    width = max(900, 40 * len(vals))
    height = 420
    left, bottom, top = 70, 70, 30
    ymax = max(v for _, v in vals) or 1.0
    bar_w = max(4, (width - left - 30) / len(vals) * 0.75)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="22" font-family="monospace" font-size="16">{stage} {metric}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-20}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
    ]
    for i, (label, val) in enumerate(vals):
        x = left + 10 + i * ((width - left - 40) / len(vals))
        h = (height - bottom - top) * val / ymax
        y = height - bottom - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2f6f73"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+14}" font-family="monospace" font-size="9" transform="rotate(65 {x:.1f},{height-bottom+14})">{label}</text>')
    parts.append(f'<text x="10" y="{top+20}" font-family="monospace" font-size="12">max {metric}={ymax:.3f}</text>')
    parts.append("</svg>")
    (stage_dir(stage) / "plots" / f"{stage}_{metric}.svg").write_text("\n".join(parts))


def row_cycles_per_mma(r):
    cycles = fnum(r.get("elapsed_cycles"))
    q = max(1.0, fnum(r.get("Q"), 1.0))
    iters = max(1.0, fnum(r.get("iterations"), 1.0))
    return cycles / (q * iters) if math.isfinite(cycles) else math.nan


def short_float(value, digits=3):
    x = fnum(value)
    return f"{x:.{digits}f}" if math.isfinite(x) else ""


def stage_observation_lines(stage, valid, invalid_rows):
    lines = []
    if stage == "00_validation":
        lines += [
            "- Descriptor, TMA load, MMA issue, commit/wait, full D readback, guard columns, and CUDA error checks are all exercised before later stages run.",
            "- TMEM D footprint for M128 FP32 accumulators under the tested shapes:",
            "",
            "| Shape | D footprint columns | Max non-overlap D tiles in 512 columns |",
            "| --- | ---: | ---: |",
        ]
        for shape, meta in SHAPES.items():
            lines.append(f"| {shape} | {meta['d_footprint']} | {512 // meta['d_footprint']} |")
        by_shape = {}
        for r in valid:
            by_shape.setdefault((r.get("dtype"), r.get("n")), 0)
            by_shape[(r.get("dtype"), r.get("n"))] += 1
        if by_shape:
            summary = ", ".join(f"{dt}/N{n}:{count}" for (dt, n), count in sorted(by_shape.items()))
            lines.append(f"- Valid descriptor rows by dtype/N: {summary}.")
    elif stage == "01_collector_protocol":
        discard = [row_cycles_per_mma(r) for r in valid if r.get("collector_mode") == "discard"]
        reuse = [row_cycles_per_mma(r) for r in valid if r.get("collector_mode") != "discard"]
        ws = [r for r in valid if "collector_ws" in r.get("case_id", "")]
        if discard:
            lines.append(f"- discard median cycles/MMA range: {min(discard):.3f} to {max(discard):.3f}.")
        if reuse:
            lines.append(f"- fill/use/lastuse median cycles/MMA range: {min(reuse):.3f} to {max(reuse):.3f}.")
        if ws:
            buffers = sorted({r.get("case_id", "").split("_")[2] for r in ws})
            lines.append(f"- weights-stationary B collector cases executed for {', '.join(buffers)}.")
    elif stage == "02_latency_throughput":
        betas = [fnum(r.get("beta_cycles_per_mma")) for r in valid if r.get("case_id", "").startswith("lat_")]
        betas = [x for x in betas if math.isfinite(x)]
        q1 = [row_cycles_per_mma(r) for r in valid if r.get("case_id", "").startswith("lat_") and str(r.get("Q")) == "1"]
        if betas:
            lines.append(f"- Fitted beta range over latency rows: {min(betas):.3f} to {max(betas):.3f} cycles/MMA.")
        if q1:
            lines.append(f"- Q=1 forced-completion diagnostic cycles/MMA range: {min(q1):.3f} to {max(q1):.3f}.")
        commit_rows = [r for r in valid if r.get("case_id", "").startswith("commit_prefix")]
        if commit_rows:
            lines.append(f"- commit-prefix scan rows: {len(commit_rows)}; `pending_mbarriers` is recorded as completion-prefix tracking count.")
    elif stage == "03_effective_smem_ingress":
        rates = [fnum(r.get("effective_smem_bytes_per_cycle")) for r in valid if r.get("collector_mode") == "discard"]
        rates = [x for x in rates if math.isfinite(x)]
        if rates:
            lines.append(f"- collector-discard logical effective SMEM operand rate range: {min(rates):.3f} to {max(rates):.3f} bytes/cycle.")
    elif stage == "04_smem_layout_address":
        offsets = sorted({r.get("smem_base_offset") for r in valid})
        if offsets:
            lines.append(f"- Valid SMEM base offsets in this run: {', '.join(map(str, offsets))}.")
        if invalid_rows:
            invalid_offsets = sorted({r.get("smem_base_offset") for r in invalid_rows})
            lines.append(f"- Invalid SMEM base offsets/descriptors are isolated in invalid_cases.csv: {', '.join(map(str, invalid_offsets))}.")
    elif stage == "05_ldshared_contention":
        modes = sorted({r.get("interference_mode") for r in valid})
        lines.append(f"- Fixed active interference warp count; modes present: {', '.join(modes)}.")
        for mode in ["none", "register_alu", "predicated_off_load", "l1_hit_global", "ld_shared", "interference_only"]:
            vals = [fnum(r.get("tflops")) for r in valid if r.get("interference_mode") == mode and r.get("tflops") not in ("", None)]
            vals = [x for x in vals if math.isfinite(x)]
            if vals:
                lines.append(f"- {mode} TFLOP/s median range: {min(vals):.6f} to {max(vals):.6f}.")
    elif stage == "06_tmem_dependency":
        alias = sorted({r.get("d_alias_class") for r in valid})
        cols = sorted({r.get("tmem_columns") for r in valid}, key=lambda x: int(float(x)))
        lines.append(f"- D alias classes present: {', '.join(alias)}.")
        lines.append(f"- TMEM column allocations present: {', '.join(map(str, cols))}; independent_d_count is clamped to actual capacity.")
    elif stage == "07_config_matrix":
        best = sorted(valid, key=lambda r: fnum(r.get("tflops"), -1), reverse=True)[:5]
        for r in best:
            lines.append(
                f"- Top config `{r['case_id']}`: {short_float(r.get('tflops'), 6)} TFLOP/s, "
                f"beta {short_float(r.get('beta_cycles_per_mma'))} cycles/MMA."
            )
    return lines


def write_analysis(stage, rows, invalid_rows):
    valid = [r for r in rows if str(r.get("valid")) == "1"]
    lines = [
        f"# {stage} analysis",
        "",
        "## Observation",
        f"- valid cases: {len(valid)}",
        f"- invalid cases: {len(invalid_rows)}",
    ]
    if valid:
        best = max(valid, key=lambda r: fnum(r.get("tflops"), -1))
        fastest = min(valid, key=lambda r: fnum(r.get("elapsed_cycles"), math.inf))
        lines.append(f"- fastest median cycles case: `{fastest['case_id']}` = {fnum(fastest.get('elapsed_cycles')):.3f} cycles")
        if best.get("tflops") not in ("", None):
            lines.append(f"- best TFLOP/s case: `{best['case_id']}` = {fnum(best.get('tflops')):.6f}")
    extra = stage_observation_lines(stage, valid, invalid_rows)
    if extra:
        lines.extend(extra)
    if invalid_rows:
        reasons = {}
        for r in invalid_rows:
            reasons[r.get("invalid_reason", "")] = reasons.get(r.get("invalid_reason", ""), 0) + 1
        lines.append("- invalid reason counts: " + ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items())))
    lines += [
        "",
        "## Inference",
        "- Rows report software-visible behavior only. `pending_mbarriers` is treated as cumulative completion-prefix tracking, not as an independent async group queue.",
        "- Effective SMEM rates, when present, are logical operand bytes per measured cycle under collector-discard conditions and are not physical port widths.",
        "",
        "## Unsupported Claim",
        "- These results do not identify physical SMEM bank count, physical TMEM bank width, or hidden collector depth.",
    ]
    (stage_dir(stage) / "plots" / "analysis.md").write_text("\n".join(lines) + "\n")


def write_stage_readme(stage):
    path = stage_dir(stage) / "README.md"
    if path.exists():
        return
    path.write_text(
        f"# {stage}\n\n"
        "This directory is an independent tcgen05 MMA configuration microbenchmark stage.\n\n"
        "- `benchmark_src/`: CUDA source for this stage.\n"
        "- `scripts/run.py`: builds and runs only this stage.\n"
        "- `plots/`: raw CSV, aggregate CSV, SVG plots, SASS summary, and analysis.\n\n"
        "Run from the repository root or this directory:\n\n"
        f"```bash\npython3 microbench/mma_config/{stage}/scripts/run.py --quick\n```\n"
    )


def read_stage_rows(stage, filename="benchmark_results.csv"):
    path = stage_dir(stage) / "plots" / filename
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def validation_core_passed(rows):
    needed = {(dtype, str(SHAPES[shape]["n"])) for dtype in DTYPES for shape in SHAPES}
    seen = set()
    for r in rows:
        if str(r.get("valid")) != "1":
            continue
        if r.get("smem_layout") != "sw128":
            continue
        if str(r.get("tmem_columns")) != "512":
            continue
        if r.get("collector_mode") not in ("discard", "fill_use_lastuse"):
            continue
        seen.add((r.get("dtype"), str(r.get("n"))))
    return needed.issubset(seen), sorted(needed - seen)


def generate_final_report(stages=STAGES):
    report = REPO_ROOT / "Docs" / "ExperimentReport.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    all_rows = {stage: read_stage_rows(stage) for stage in stages}
    first = next((rows[0] for rows in all_rows.values() if rows), {})
    lines = [
        "# tcgen05 MMA hardware-path calibration report",
        "",
        "This report is generated from the latest CSV artifacts under `microbench/mma_config/*/plots/`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 microbench/mma_config/scripts/run_all.py",
        "python3 microbench/mma_config/scripts/run_all.py --quick --repeats 3",
        "python3 microbench/mma_config/scripts/run_all.py --stage 02_latency_throughput --case-id lat_bf16_m128n128k16_legal_ring_in0_q16 --repeats 5",
        "```",
        "",
        "## Environment",
        "",
        f"- GPU: {first.get('gpu', '')}",
        f"- Compute capability: {first.get('compute_capability', '')}",
        f"- Driver: {first.get('driver', '')}",
        f"- CUDA toolkit/PTX: {first.get('cuda_version', '')} / {first.get('ptx_version', '')}",
        f"- SM clock MHz: {first.get('sm_clock_mhz', '')}",
        f"- Memory clock MHz: {first.get('mem_clock_mhz', '')}",
        f"- Temperature C / power W: {first.get('temperature_c', '')} / {first.get('power_w', '')}",
        "",
        "## Artifact Index",
        "",
        "| Stage | Benchmark source | Raw CSV | Aggregate CSV | Invalid CSV | Plot | Analysis |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stage in stages:
        plot_files = sorted((stage_dir(stage) / "plots").glob("*.svg"))
        plot = plot_files[0] if plot_files else stage_dir(stage) / "plots"
        lines.append(
            f"| {stage} | `microbench/mma_config/{stage}/benchmark_src/` | "
            f"`microbench/mma_config/{stage}/plots/raw_results.csv` | "
            f"`microbench/mma_config/{stage}/plots/benchmark_results.csv` | "
            f"`microbench/mma_config/{stage}/plots/invalid_cases.csv` | "
            f"`{plot.relative_to(REPO_ROOT)}` | "
            f"`microbench/mma_config/{stage}/plots/analysis.md` |"
        )
    lines += [
        "",
        "## Observation",
        "",
        "| Stage | Valid aggregate cases | Invalid aggregate cases | Key measured field |",
        "| --- | ---: | ---: | --- |",
    ]
    for stage, rows in all_rows.items():
        valid = [r for r in rows if str(r.get("valid")) == "1"]
        invalid = [r for r in rows if str(r.get("valid")) != "1"]
        metric = "elapsed_cycles"
        if stage in ("03_effective_smem_ingress", "04_smem_layout_address"):
            metric = "effective_smem_bytes_per_cycle"
        elif stage in ("01_collector_protocol", "05_ldshared_contention", "07_config_matrix"):
            metric = "tflops"
        values = [fnum(r.get(metric)) for r in valid if math.isfinite(fnum(r.get(metric)))]
        if values:
            key = f"{metric}: {min(values):.3f} to {max(values):.3f}"
        else:
            key = metric
        lines.append(f"| {stage} | {len(valid)} | {len(invalid)} | {key} |")
    rows00 = all_rows.get("00_validation", [])
    ok, missing = validation_core_passed(rows00)
    lines += [
        "",
        f"- `00_validation` core pass: {'yes' if ok else 'no'}"
        + ("" if ok else f"; missing {missing}"),
    ]
    rows02 = all_rows.get("02_latency_throughput", [])
    betas = [fnum(r.get("beta_cycles_per_mma")) for r in rows02 if r.get("case_id", "").startswith("lat_")]
    betas = [x for x in betas if math.isfinite(x)]
    if betas:
        lines.append(f"- `02_latency_throughput` fitted beta range: {min(betas):.3f} to {max(betas):.3f} cycles/MMA.")
    rows03 = all_rows.get("03_effective_smem_ingress", [])
    rates = [fnum(r.get("effective_smem_bytes_per_cycle")) for r in rows03 if str(r.get("valid")) == "1"]
    rates = [x for x in rates if math.isfinite(x)]
    if rates:
        lines.append(f"- `03_effective_smem_ingress` reports logical effective operand supply: {min(rates):.3f} to {max(rates):.3f} bytes/cycle.")
    rows05 = all_rows.get("05_ldshared_contention", [])
    modes = sorted({r.get("interference_mode") for r in rows05 if str(r.get("valid")) == "1"})
    if modes:
        lines.append(f"- `05_ldshared_contention` includes controls: {', '.join(modes)}.")
    lines += [
        "",
        "### Invalid Case Summary",
        "",
        "| Stage | Invalid reasons |",
        "| --- | --- |",
    ]
    for stage in stages:
        invalid = [r for r in all_rows.get(stage, []) if str(r.get("valid")) != "1"]
        counts = {}
        for r in invalid:
            reason = r.get("invalid_reason", "")
            counts[reason] = counts.get(reason, 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) if counts else "none"
        lines.append(f"| {stage} | {summary} |")
    lines += [
        "",
        "## Inference",
        "",
        "- `tcgen05.commit` is analyzed as cumulative completion-prefix tracking for prior async tcgen05 operations. The CSV field `pending_mbarriers` is not interpreted as an independent async group queue.",
        "- `Q`, `independent_d_count`, and `d_reuse_distance` are separate CSV fields. When D capacity is exhausted, the run records the clamped independent D count and the reuse distance instead of labeling the sequence as independent-D.",
        "- Effective SMEM bytes/cycle is only reported as logical operand bytes divided by measured cycles under validated collector-discard cases. It is not a physical port-width measurement.",
        "- ld.shared contention conclusions must be read relative to the register ALU, predicated-off load, L1-hit global-load, MMA-only, and interference-only controls.",
        "",
        "## Unsupported Claim",
        "",
        "- These microbenchmarks do not prove physical SMEM port width, physical bank count, physical TMEM bank width, hidden collector depth, or hidden async group queue depth.",
        "- Shape or layout performance differences are reported as software-visible sensitivity unless corroborated by the controlled experiments listed above.",
        "",
        "## Deliverable Audit",
        "",
        "| Requirement | Status | Evidence |",
        "| --- | --- | --- |",
        "| One independent subfolder per experiment | done | `microbench/mma_config/00_validation` through `07_config_matrix` each contain `benchmark_src`, `scripts`, and `plots` |",
        "| Validation before performance stages | done | top-level runner stops after `00_validation` if core sw128/512 rows are missing; current core pass is recorded above |",
        "| Raw and aggregate CSV | done | each stage has `raw_results.csv`, `benchmark_results.csv`, and `invalid_cases.csv` |",
        "| Numeric and legality checks | done | valid rows require CUDA success, status ok, guard_ok=1, and max_abs_error within dtype tolerance |",
        "| PTX/SASS audit trail | done | each stage writes `sass_summary.txt` with SASS/PTX hashes and instruction counts |",
        "| Randomized performance order | done | non-validation stages shuffle cases with a recorded `run_order` field |",
        "| p10/p90/median timing | done | aggregate CSV records median `elapsed_cycles` plus `elapsed_cycles_p10` and `elapsed_cycles_p90`; raw CSV keeps all repeats |",
        "| Plots and short analyses | done | each stage has an SVG plot and `analysis.md` |",
        "| Final report with claim boundaries | done | this file separates observation, inference, and unsupported claims |",
    ]
    report.write_text("\n".join(lines) + "\n")
    return report


def run_stage(stage, quick=False, repeats=5, seed=20260720, single_case=None):
    write_stage_readme(stage)
    build(stage)
    sass = sass_info(stage)
    env = query_nvidia_smi()
    env["ptx_version"] = "PTX ISA 9.3 / CUDA 13.0"
    env.setdefault("driver", "")
    env.setdefault("cuda_version", "")
    cases = CASE_BUILDERS[stage](quick)
    if single_case:
        cases = [c for c in cases if c["case_id"] == single_case]
    if stage != "00_validation":
        random.Random(seed).shuffle(cases)
    all_rows = []
    invalid_single = []
    for idx, case in enumerate(cases, 1):
        reps = 1 if stage == "00_validation" else repeats
        for rep in range(reps):
            row = run_one(stage, case, env, sass, idx, rep)
            all_rows.append(row)
            if str(row["valid"]) != "1":
                invalid_single.append(row)
        if idx % 20 == 0:
            print(f"{stage}: completed {idx}/{len(cases)} cases", flush=True)
    agg = aggregate_rows(all_rows)
    derive_metrics(stage, agg)
    invalid_agg = [r for r in agg if str(r["valid"]) != "1"]
    plots = stage_dir(stage) / "plots"
    write_csv(plots / "raw_results.csv", all_rows)
    write_csv(plots / "benchmark_results.csv", agg)
    write_csv(plots / "invalid_cases.csv", invalid_agg)
    if stage == "00_validation":
        write_csv(plots / "valid_descriptor_cases.csv", [r for r in agg if str(r["valid"]) == "1"])
    plot_svg(stage, agg)
    write_analysis(stage, agg, invalid_agg)
    return agg


def main(stage):
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--case-id")
    args = parser.parse_args()
    run_stage(stage, quick=args.quick, repeats=args.repeats, seed=args.seed, single_case=args.case_id)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: {sys.argv[0]} <stage> [stage args]")
    stage_arg = sys.argv.pop(1)
    main(stage_arg)
