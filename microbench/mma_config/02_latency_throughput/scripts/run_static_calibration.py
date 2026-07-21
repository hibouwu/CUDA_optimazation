#!/usr/bin/env python3
import argparse
import csv
import hashlib
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "02_latency_throughput"
SRC = STAGE / "benchmark_src" / "tcgen05_02_static_calibration_bench.cu"
BUILD = STAGE / "build" / "static_calibration"
PLOTS = STAGE / "plots"
STATIC_BUILD_VERSION = "v3"

DTYPE_ID = {"fp16": 0, "bf16": 1}
MODE_ID = {"empty": 0, "commit_wait": 1, "forced_wait": 2, "batch": 3, "cta_sync": 4}
DMODE_ID = {"same": 0, "ring": 1}
ADDR_ID = {"same": 0, "pingpong": 1, "rotating": 2}
COLLECTOR_ID = {
    "discard": 0,
    "fill_lastuse": 1,
    "fill_use_lastuse": 2,
    "fill_use_discard": 3,
}
INTERFERENCE_ID = {
    "none": 0,
    "register_alu": 1,
    "predicated_off_load": 2,
    "l1_hit_global": 3,
    "ld_shared": 4,
    "interference_only": 5,
}
FIELDS = [
    "phase", "case_id", "valid", "invalid_reason",
    "gpu", "compute_capability", "driver", "cuda_version",
    "sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_w", "telemetry_status",
    "dtype", "n", "q", "iterations", "repeat", "run_order",
    "mode", "d_mode", "addr_mode", "collector", "input_d",
    "wait_hint", "active_blocks_request", "launch_blocks",
    "interference", "interference_ops",
    "clock64_cycles", "clock64_cycles_p10", "clock64_cycles_p50",
    "clock64_cycles_p90", "clock64_cycles_cv", "cycles_per_mma",
    "event_ms", "event_ms_p10", "event_ms_p50", "event_ms_p90",
    "event_ms_cv", "clock64_full_grid_tflops", "event_wall_tflops",
    "poll_count", "max_abs_error", "guard_ok",
    "sass_hash", "sass_utchmma", "sass_trywait", "sass_syncthreads",
    "sass_ld_shared", "sass_ld_global", "sass_utcld", "sass_utcst",
    "notes",
]


def run(cmd, check=True):
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{proc.stdout}")
    return proc.returncode, proc.stdout or ""


def query_env():
    env = {
        "driver": "",
        "cuda_version": "",
        "sm_clock_mhz": "",
        "mem_clock_mhz": "",
        "temperature_c": "",
        "power_w": "",
        "telemetry_status": "",
    }
    try:
        _, out = run(["nvidia-smi"], check=False)
        for line in out.splitlines():
            if "Driver Version:" in line:
                parts = line.replace("|", " ").split()
                if "Version:" in parts:
                    env["driver"] = parts[parts.index("Version:") + 1]
                if "CUDA" in parts:
                    idx = parts.index("CUDA")
                    if idx + 2 < len(parts):
                        env["cuda_version"] = parts[idx + 2]
    except Exception:
        pass
    statuses = []
    try:
        _, out = run([
            "nvidia-smi",
            "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], check=False)
        vals = [x.strip() for x in out.splitlines()[0].split(",")] if out.splitlines() else []
        keys = ["sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_w"]
        for key, val in zip(keys, vals):
            if val and val != "[N/A]":
                env[key] = val
            else:
                statuses.append(f"{key}:unavailable")
    except Exception:
        statuses.append("nvidia-smi-query:failed")
    if not env["sm_clock_mhz"]:
        hz = read_freq_hz()
        if hz > 0:
            env["sm_clock_mhz"] = f"{hz / 1e6:.3f}"
            statuses = [s for s in statuses if s != "sm_clock_mhz:unavailable"]
            statuses.append("sm_clock_mhz:sysfs_or_default")
    env["telemetry_status"] = ";".join(statuses) if statuses else "ok"
    return env


def read_freq_hz():
    for path in [
        Path("/sys/class/devfreq/gpu-gpc-0/cur_freq"),
        Path("/sys/class/devfreq/17000000.gpu/cur_freq"),
    ]:
        try:
            return int(path.read_text().strip())
        except Exception:
            pass
    return 1_575_000_000


def flops_per_mma(n):
    return 2 * 128 * n * 16


def case_key(case):
    parts = [
        STATIC_BUILD_VERSION, case["phase"], case["dtype"], f"n{case['n']}", f"q{case['q']}",
        case["mode"], case["d_mode"], f"in{case['input_d']}",
        case["addr_mode"], case["collector"], f"h{case['wait_hint']}",
        f"b{case['active_blocks']}", case["interference"], f"ops{case['interference_ops']}",
    ]
    return "_".join(str(x) for x in parts)


def build_case(case):
    key = case_key(case)
    out = BUILD / key / "bench"
    keep = BUILD / key / "keep"
    if out.exists() and out.stat().st_mtime >= SRC.stat().st_mtime:
        return out
    keep.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvcc", "-std=c++17", "-O3",
        "-gencode", "arch=compute_110a,code=sm_110a",
        "--keep", "--keep-dir", keep,
        "-DSTATIC_SINGLE_CASE",
        f"-DCFG_DTYPE={DTYPE_ID[case['dtype']]}",
        f"-DCFG_N={case['n']}",
        f"-DCFG_Q={case['q']}",
        f"-DCFG_MODE={MODE_ID[case['mode']]}",
        f"-DCFG_DMODE={DMODE_ID[case['d_mode']]}",
        f"-DCFG_INPUT_D={case['input_d']}",
        f"-DCFG_ADDR={ADDR_ID[case['addr_mode']]}",
        f"-DCFG_COLLECTOR={COLLECTOR_ID[case['collector']]}",
        f"-DCFG_WAIT_HINT={case['wait_hint']}",
        f"-DCFG_INTERFERENCE={INTERFERENCE_ID[case['interference']]}",
        SRC, "-lcuda", "-o", out,
    ]
    run(cmd)
    return out


def sass_summary(binary, case):
    _, sass = run(["cuobjdump", "--dump-sass", binary], check=False)
    digest = hashlib.sha256(sass.encode("utf-8", "ignore")).hexdigest()
    info = {
        "sass_hash": digest,
        "sass_utchmma": sass.count("UTCHMMA"),
        "sass_trywait": sass.count("SYNCS.PHASECHK.TRANS64.TRYWAIT"),
        "sass_syncthreads": sass.count("BAR.SYNC") + sass.count("BSYNC"),
        "sass_ld_shared": sass.count("LDS"),
        "sass_ld_global": sass.count("LDG"),
        "sass_utcld": sass.count("UTCLD") + sass.count("UTCULD"),
        "sass_utcst": sass.count("UTCST") + sass.count("UTCUST"),
    }
    path = PLOTS / "static_sass"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{case_key(case)}.sass.txt").write_text(sass)
    return info


def parse_kv(text):
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def fnum(x, default=math.nan):
    try:
        return float(x)
    except Exception:
        return default


def percentile(vals, q):
    if not vals:
        return math.nan
    xs = sorted(vals)
    idx = (len(xs) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[int(lo)]
    return xs[int(lo)] * (hi - idx) + xs[int(hi)] * (idx - lo)


def cv(vals):
    good = [v for v in vals if math.isfinite(v)]
    if len(good) < 2:
        return ""
    mean = statistics.mean(good)
    if mean == 0:
        return ""
    return statistics.stdev(good) / mean


def validate(case, kv, ret):
    if ret != 0:
        return False, f"process_exit_{ret}"
    if kv.get("status") != "ok":
        return False, kv.get("status", "missing_status")
    if kv.get("guard_ok") != "1":
        return False, "tmem_guard_modified"
    err = fnum(kv.get("max_abs_error"))
    if case["mode"] in ("empty", "commit_wait", "cta_sync") or case["interference"] == "interference_only":
        return True, ""
    tol = 0.02 if case["dtype"] == "fp16" else 0.08
    if not math.isfinite(err) or err > tol:
        return False, f"max_abs_error>{tol}"
    return True, ""


def run_one(case, repeat, order, env):
    binary = build_case(case)
    sass = sass_summary(binary, case)
    cmd = [
        binary,
        "--dtype", case["dtype"],
        "--n", case["n"],
        "--q", case["q"],
        "--iterations", case["iterations"],
        "--mode", case["mode"],
        "--d-mode", case["d_mode"],
        "--input-d", case["input_d"],
        "--addr-mode", case["addr_mode"],
        "--collector", case["collector"],
        "--wait-hint", case["wait_hint"],
        "--active-blocks", case["active_blocks"],
        "--interference", case["interference"],
        "--interference-ops", case["interference_ops"],
    ]
    ret, out = run(cmd, check=False)
    kv = parse_kv(out)
    valid, reason = validate(case, kv, ret)
    cycles = fnum(kv.get("elapsed_cycles"))
    event_ms = fnum(kv.get("event_ms"))
    launch_blocks = int(fnum(kv.get("launch_blocks"), 0))
    mma_count = case["q"] * case["iterations"] * max(1, launch_blocks)
    if case["mode"] in ("empty", "commit_wait", "cta_sync") or case["interference"] == "interference_only":
        mma_count = 0
    freq = read_freq_hz()
    clock_tflops = ""
    event_tflops = ""
    cycles_per_mma = ""
    if mma_count and cycles > 0:
        cycles_per_mma = cycles / (case["q"] * case["iterations"])
        clock_tflops = flops_per_mma(case["n"]) * mma_count / (cycles / freq) / 1e12
    if mma_count and event_ms > 0:
        event_tflops = flops_per_mma(case["n"]) * mma_count / (event_ms / 1e3) / 1e12
    row = {k: "" for k in FIELDS}
    row.update(env)
    row.update(sass)
    row.update({
        "phase": case["phase"],
        "case_id": case_key(case),
        "valid": 1 if valid else 0,
        "invalid_reason": reason,
        "gpu": kv.get("gpu", ""),
        "compute_capability": kv.get("compute_capability", ""),
        "dtype": case["dtype"],
        "n": case["n"],
        "q": case["q"],
        "iterations": case["iterations"],
        "repeat": repeat,
        "run_order": order,
        "mode": case["mode"],
        "d_mode": case["d_mode"],
        "addr_mode": case["addr_mode"],
        "collector": case["collector"],
        "input_d": case["input_d"],
        "wait_hint": case["wait_hint"],
        "active_blocks_request": case["active_blocks"],
        "launch_blocks": launch_blocks,
        "interference": case["interference"],
        "interference_ops": case["interference_ops"],
        "clock64_cycles": cycles,
        "cycles_per_mma": cycles_per_mma,
        "event_ms": event_ms,
        "clock64_full_grid_tflops": clock_tflops,
        "event_wall_tflops": event_tflops,
        "poll_count": kv.get("poll_count", ""),
        "max_abs_error": kv.get("max_abs_error", ""),
        "guard_ok": kv.get("guard_ok", ""),
        "notes": "static single-case binary; CUDA event includes setup/readback, clock64 covers timed window",
    })
    if not valid:
        failed = PLOTS / "static_failed_commands.log"
        with failed.open("a") as f:
            f.write("$ " + " ".join(map(str, cmd)) + "\n")
            f.write(out + "\n")
    return row


def aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["case_id"], []).append(r)
    out = []
    for vals in groups.values():
        base = vals[0].copy()
        good = [v for v in vals if str(v["valid"]) == "1"]
        cyc = [fnum(v["clock64_cycles"]) for v in good]
        ev = [fnum(v["event_ms"]) for v in good]
        tclk = [fnum(v["clock64_full_grid_tflops"]) for v in good if v["clock64_full_grid_tflops"] != ""]
        tev = [fnum(v["event_wall_tflops"]) for v in good if v["event_wall_tflops"] != ""]
        cpm = [fnum(v["cycles_per_mma"]) for v in good if v["cycles_per_mma"] != ""]
        if cyc:
            base["clock64_cycles_p10"] = percentile(cyc, 0.1)
            base["clock64_cycles_p50"] = statistics.median(cyc)
            base["clock64_cycles_p90"] = percentile(cyc, 0.9)
            base["clock64_cycles_cv"] = cv(cyc)
            base["clock64_cycles"] = statistics.median(cyc)
        if ev:
            base["event_ms_p10"] = percentile(ev, 0.1)
            base["event_ms_p50"] = statistics.median(ev)
            base["event_ms_p90"] = percentile(ev, 0.9)
            base["event_ms_cv"] = cv(ev)
            base["event_ms"] = statistics.median(ev)
        if cpm:
            base["cycles_per_mma"] = statistics.median(cpm)
        if tclk:
            base["clock64_full_grid_tflops"] = statistics.median(tclk)
        if tev:
            base["event_wall_tflops"] = statistics.median(tev)
        base["repeat"] = len(vals)
        base["valid"] = 1 if len(good) == len(vals) else 0
        if not base["valid"]:
            base["invalid_reason"] = ";".join(sorted({v["invalid_reason"] for v in vals if v["invalid_reason"]}))
        out.append(base)
    return out


def calibration_cases(quick=False):
    q_values = [1, 4, 16] if quick else [1, 2, 4, 8, 16, 32, 64]
    hints = [0, 32, 0x989680]
    cases = []
    for dtype in ["bf16", "fp16"]:
        for n in [128, 256]:
            for active in [1, 0]:
                for hint in hints:
                    for q in q_values:
                        for d_mode, input_d in [("same", 0), ("same", 1), ("ring", 0)]:
                            cases.append({
                                "phase": "calibration",
                                "dtype": dtype,
                                "n": n,
                                "q": q,
                                "iterations": 500 if not quick else 120,
                                "mode": "batch",
                                "d_mode": d_mode,
                                "input_d": input_d,
                                "addr_mode": "same",
                                "collector": "discard",
                                "wait_hint": hint,
                                "active_blocks": active,
                                "interference": "none",
                                "interference_ops": 0,
                            })
                    for mode in ["empty", "commit_wait", "forced_wait"]:
                        cases.append({
                            "phase": "control",
                            "dtype": dtype,
                            "n": n,
                            "q": 1 if mode != "empty" else max(q_values),
                            "iterations": 1000 if not quick else 200,
                            "mode": mode,
                            "d_mode": "same",
                            "input_d": 0,
                            "addr_mode": "same",
                            "collector": "discard",
                            "wait_hint": hint,
                            "active_blocks": active,
                            "interference": "none",
                            "interference_ops": 0,
                        })
    return cases


def collector_cases(quick=False):
    q = 16 if quick else 32
    cases = []
    for dtype in ["bf16", "fp16"]:
        for n in [128, 256]:
            for collector in ["discard", "fill_lastuse", "fill_use_lastuse", "fill_use_discard"]:
                for addr in ["same", "pingpong"]:
                    cases.append({
                        "phase": "collector",
                        "dtype": dtype,
                        "n": n,
                        "q": q,
                        "iterations": 500 if not quick else 120,
                        "mode": "batch",
                        "d_mode": "ring",
                        "input_d": 1,
                        "addr_mode": addr,
                        "collector": collector,
                        "wait_hint": 0,
                        "active_blocks": 0,
                        "interference": "none",
                        "interference_ops": 0,
                    })
    return cases


def ingress_cases(quick=False):
    q = 16 if quick else 64
    cases = []
    for dtype in ["bf16", "fp16"]:
        for n in [128, 256]:
            for addr in ["same", "pingpong", "rotating"]:
                cases.append({
                    "phase": "ingress",
                    "dtype": dtype,
                    "n": n,
                    "q": q,
                    "iterations": 500 if not quick else 120,
                    "mode": "batch",
                    "d_mode": "ring",
                    "input_d": 1,
                    "addr_mode": addr,
                    "collector": "discard",
                    "wait_hint": 0,
                    "active_blocks": 0,
                    "interference": "none",
                    "interference_ops": 0,
                })
    return cases


def ldshared_cases(quick=False):
    ops_values = [0, 32] if quick else [0, 8, 32, 128]
    modes = ["none", "register_alu", "predicated_off_load", "l1_hit_global", "ld_shared", "interference_only"]
    cases = []
    for mode in modes:
        for ops in ops_values:
            if mode == "none" and ops != 0:
                continue
            cases.append({
                "phase": "ldshared",
                "dtype": "bf16",
                "n": 128,
                "q": 32 if not quick else 16,
                "iterations": 500 if not quick else 120,
                "mode": "batch",
                "d_mode": "ring",
                "input_d": 1,
                "addr_mode": "same",
                "collector": "discard",
                "wait_hint": 0,
                "active_blocks": 0,
                "interference": mode,
                "interference_ops": ops,
            })
    return cases


def tmem_cases(quick=False):
    cases = []
    for input_d in [0, 1]:
        for d_mode in ["same", "ring"]:
            for n in [128, 256]:
                cases.append({
                    "phase": "tmem",
                    "dtype": "bf16",
                    "n": n,
                    "q": 32 if not quick else 16,
                    "iterations": 500 if not quick else 120,
                    "mode": "batch",
                    "d_mode": d_mode,
                    "input_d": input_d,
                    "addr_mode": "same",
                    "collector": "discard",
                    "wait_hint": 0,
                    "active_blocks": 0,
                    "interference": "none",
                    "interference_ops": 0,
                })
    return cases


def make_cases(matrix, quick):
    if matrix == "calibration":
        return calibration_cases(quick)
    if matrix == "collector":
        return collector_cases(quick)
    if matrix == "ingress":
        return ingress_cases(quick)
    if matrix == "ldshared":
        return ldshared_cases(quick)
    if matrix == "tmem":
        return tmem_cases(quick)
    cases = []
    for name in ["calibration", "collector", "ingress", "ldshared", "tmem"]:
        cases.extend(make_cases(name, quick))
    return cases


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(rows, matrix):
    agg = aggregate(rows)
    valid = [r for r in agg if str(r["valid"]) == "1"]
    lines = [
        f"# 静态 {matrix} 分析",
        "",
        "## 观察",
        f"- aggregate cases: {len(agg)}",
        f"- valid aggregate cases: {len(valid)}",
        f"- invalid aggregate cases: {len(agg) - len(valid)}",
    ]
    k4 = [r for r in valid if r["phase"] == "calibration" and r["dtype"] == "bf16" and str(r["n"]) == "128" and str(r["q"]) == "4" and r["d_mode"] == "same" and str(r["input_d"]) == "0" and str(r["active_blocks_request"]) == "0" and str(r["wait_hint"]) == "0"]
    if k4:
        lines.append(f"- BF16 N128 Q4 same-D input_d=0 full-grid median cycles/MMA: {fnum(k4[0]['cycles_per_mma']):.3f}；可信 K4 reference 约为 146.132 cycles/MMA。")
    cal = [fnum(r["cycles_per_mma"]) for r in valid if r["phase"] == "calibration" and r["cycles_per_mma"] != ""]
    if cal:
        lines.append(f"- calibration cycles/MMA 范围: {min(cal):.3f} 到 {max(cal):.3f}.")
    telemetry = sorted({r.get("telemetry_status", "") for r in valid if r.get("telemetry_status", "")})
    if telemetry:
        lines.append(f"- telemetry status: {'; '.join(telemetry)}.")
    ld = [r for r in valid if r["phase"] == "ldshared"]
    if ld:
        none = [fnum(r["cycles_per_mma"]) for r in ld if r["interference"] == "none"]
        lines.append(f"- ld.shared contention control 已记录；MMA-only cycles/MMA baseline median: {statistics.median(none):.3f}.")
    lines += [
        "",
        "## 推断",
        "- 静态 single-case binary 将 runtime collector/shape/D dispatch 和 descriptor construction 移出 timed loop。",
        "- CUDA event wall time 只作为端到端 launch/setup/readback 交叉检查；clock64 cycles 是校准用 timed-window 指标。",
        "- logical effective bytes/cycle 只有在比较 same/pingpong/rotating address mode、shape scaling 和 control-subtracted cycles 后才可解释。",
        "",
        "## 不支持的说法",
        "- 这些数据本身不能识别物理 SMEM port width、SMEM bank count、TMEM bank width 或 hidden collector depth。",
    ]
    (PLOTS / f"static_{matrix}_analysis.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=["calibration", "collector", "ingress", "ldshared", "tmem", "all"], default="calibration")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--case-id")
    args = ap.parse_args()
    env = query_env()
    cases = make_cases(args.matrix, args.quick)
    if args.case_id:
        cases = [c for c in cases if case_key(c) == args.case_id]
        if not cases:
            raise SystemExit(f"--case-id not found in static {args.matrix} matrix: {args.case_id}")
    random.Random(args.seed).shuffle(cases)
    rows = []
    total = len(cases)
    for idx, case in enumerate(cases, 1):
        for rep in range(args.repeats):
            rows.append(run_one(case, rep, idx, env))
        if idx % 20 == 0 or idx == total:
            print(f"static {args.matrix}: completed {idx}/{total}", flush=True)
    raw_path = PLOTS / f"static_{args.matrix}_raw.csv"
    agg_path = PLOTS / f"static_{args.matrix}_benchmark.csv"
    invalid_path = PLOTS / f"static_{args.matrix}_invalid.csv"
    if args.case_id:
        safe_case = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.case_id)
        raw_path = PLOTS / f"static_{args.matrix}_{safe_case}_raw.csv"
        agg_path = PLOTS / f"static_{args.matrix}_{safe_case}_benchmark.csv"
        invalid_path = PLOTS / f"static_{args.matrix}_{safe_case}_invalid.csv"
    all_raw = rows
    write_csv(raw_path, all_raw)
    write_csv(agg_path, aggregate(all_raw))
    write_csv(invalid_path, [r for r in aggregate(all_raw) if str(r["valid"]) != "1"])
    if not args.case_id:
        write_analysis(all_raw, args.matrix)
    print(f"wrote {agg_path}")


if __name__ == "__main__":
    main()
