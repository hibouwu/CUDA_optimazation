#!/usr/bin/env python3
"""CPU re-check of a V01 run: identities, full outputs, timing windows, segments, SASS evidence
and prediction errors e = (predicted - measured) / measured."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

NAMES = ("lds_ffma", "small_tc", "target_tc")


def value(row, col, a):
    """Same dyadic coordinate input as probes/r00_common.hpp (seed 17)."""
    return ((row * (7 if a else 5) + col * (13 if a else 11) + 17 * (3 if a else 5)) % 17 - 8) / 32


def table(k_total):
    """17 x 17 exact reference; every input is periodic with period 17 in row, column and k."""
    a = np.array([[value(r, k, True) for k in range(17)] for r in range(17)])
    b = np.array([[value(k, c, False) for c in range(17)] for k in range(17)])
    full, rest = divmod(k_total, 17)
    return full * (a @ b) + a[:, :rest] @ b[:rest, :]


def reference_block(kind, length):
    """Expected per-CTA output of one combination kernel, in stored order."""
    if kind == 0:
        t = table(length)
        q = np.arange(2048)
        return t[((q // 16) * 4 + (q % 16) // 4) % 17, (q % 4) % 17]
    m, n = (128, 256) if kind == 2 else (64, 64)
    t = table(length * 64)
    return t[np.ix_(np.arange(m) % 17, np.arange(n) % 17)].reshape(-1)


def work(case, blocks):
    if case["kind"] == "cutlass":
        m, n, k = case["m"], case["n"], case["k"]
        return dict(flop=2 * m * n * k, input_bytes=(m + n) * k * 2, output_bytes=m * n * 4)
    k, kind = case["length"], case["kind"]
    if kind == 0:
        return dict(
            flop=128 * k * 32 * blocks,
            input_bytes=128 * k * 32 * blocks,
            output_bytes=2048 * 4 * blocks,
        )
    m, n = (128, 256) if kind == 2 else (64, 64)
    return dict(
        flop=2 * m * n * 64 * k * blocks,
        input_bytes=(m + n) * 128 * k * blocks,
        output_bytes=m * n * 4 * blocks,
    )


def function_sass(folder, prefix):
    parts = (folder / "build/v01.sass").read_text().split("Function : ")
    return next(p for p in parts if p.startswith(prefix))


def audit_target(folder, destination):
    """Instruction evidence for target_async; writes the function SASS and the main loop."""
    part = function_sass(folder, "_Z12target_async")
    name = part.splitlines()[0].strip()
    lines = [
        re.sub(r"\s*/\* 0x[0-9a-f]+ \*/", "", l.strip())
        for l in part.splitlines()
        if re.search(r"/\*[0-9a-f]{4}\*/", l)
    ]
    (destination / "target_async.sass").write_text("\n".join(lines) + "\n")
    log = (folder / "build/compile.log").read_text()
    resources = re.search(
        r"Compiling entry function '" + re.escape(name) + r"'.*?Used (\d+) registers", log, re.S
    )
    spills = re.search(
        r"Function properties for " + re.escape(name) + r"\n\s*(\d+) bytes stack frame, "
        r"(\d+) bytes spill stores, (\d+) bytes spill loads",
        log,
    )
    count = lambda pattern: sum(bool(re.search(pattern, l)) for l in lines)
    # Main loop: from the first in-loop wait1 back to its loop head (the backward branch).
    wait1 = [i for i, l in enumerate(lines) if "WARPGROUP.DEPBAR.LE gsb0, 0x1" in l]
    loop = []
    for i, l in enumerate(lines):
        m = re.search(r"BRA (0x[0-9a-f]+)", l)
        if m and wait1 and i > wait1[0]:
            target = int(m.group(1), 16)
            pcs = [int(re.match(r"/\*([0-9a-f]+)\*/", x).group(1), 16) for x in lines]
            head = pcs.index(target)
            if head < wait1[0] < i:
                loop = lines[head : i + 1]
                break
    (destination / "target_mainloop.sass").write_text("\n".join(loop) + "\n")
    return dict(
        function=name,
        registers=int(resources.group(1)) if resources else None,
        spill_bytes=(int(spills.group(2)) + int(spills.group(3))) if spills else None,
        hgmma_64x256x16=count(r"HGMMA\.64x256x16\.F32"),
        hgmma_commit_gsb0=count(r"HGMMA\.64x256x16\.F32.*gsb0"),
        warpgroup_arrive=count(r"WARPGROUP\.ARRIVE"),
        wait1=len(wait1),
        wait0=count(r"WARPGROUP\.DEPBAR\.LE gsb0, 0x0"),
        empty_release_arrive=count(r"SYNCS\.ARRIVE\.TRANS64\.A1T0"),
        full_barrier_try_wait=count(r"SYNCS\.PHASECHK\.TRANS64\.TRYWAIT"),
        tma_load=count(r"UTMALDG"),
        tma_store=count(r"UTMASTG"),
        sts64=count(r"STS\.64"),
        mainloop_instructions=len(loop),
        mainloop_hgmma=sum("HGMMA" in l for l in loop),
        mainloop_wait1=sum("DEPBAR.LE gsb0, 0x1" in l for l in loop),
        mainloop_wait0=sum("DEPBAR.LE gsb0, 0x0" in l for l in loop),
        c7520_serialized=any("C7520" in l and name in l for l in log.splitlines()),
        c7517_injected_wait=any("C7517" in l and name in l for l in log.splitlines()),
        c7519_injected_arrive=any("C7519" in l and name in l for l in log.splitlines()),
    )


def segments(record):
    """Per-CTA clock64 segments of the target kernel: fill, main loop, output."""
    rows = []
    for (bn, en, bc, ec, sm), (t1, t2a, t2b) in zip(record["stamps"], record["phase_stamps"]):
        t2 = max(t2a, t2b)  # later of the two consumers' wait_group 0 returns
        if not bc <= t1 <= t2 <= ec:
            raise ValueError("phase ordering")
        rows.append((t1 - bc, t2 - t1, ec - t2, ec - bc, en - bn, sm, bn, en))
    return rows


def relaunch_gaps(record):
    """For all-GPU runs: gap between consecutive CTAs on one SM (globaltimer, ns)."""
    by_sm = {}
    for bn, en, bc, ec, sm in record["stamps"]:
        by_sm.setdefault(sm, []).append((bn, en))
    gaps = []
    for spans in by_sm.values():
        spans.sort()
        gaps += [b[0] - a[1] for a, b in zip(spans, spans[1:])]
    return gaps, len(by_sm), max(len(s) for s in by_sm.values())


def check_identities(folder):
    for name, digest in json.loads((folder / "source_hashes.json").read_text()).items():
        if hashlib.sha256((folder / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("source identity mismatch " + name)
    for name, digest in json.loads((folder / "build/binary_hashes.json").read_text()).items():
        if hashlib.sha256((folder / "build" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("binary identity mismatch " + name)
    protocol = json.loads((folder / "protocol.json").read_text())
    prediction_sha = None
    if (folder / "predictions.json").exists():
        prediction_sha = hashlib.sha256((folder / "predictions.json").read_bytes()).hexdigest()
        if prediction_sha != protocol["prediction_sha256"]:
            raise ValueError("prediction file changed after the run started")
    return protocol, prediction_sha


def load(folder, prediction_sha):
    groups, max_error = {}, 0.0
    for file in sorted((folder / "samples").glob("*/*/result.json")):
        r = json.loads(file.read_text())
        if r.get("prediction_sha256") != prediction_sha:
            raise ValueError("sample prediction identity mismatch " + str(file))
        case = r["configuration"]
        if case["kind"] == "cutlass":
            t = table(case["k"])
            idx = np.array(r["checked_indices"], dtype=np.int64)
            ref = t[(idx // case["n"]) % 17, (idx % case["n"]) % 17]
            error = float(np.max(np.abs(np.array(r["checked_values"], dtype=float) - ref)))
            elapsed, unit = r["elapsed_ms"], "cuda_event_ms/GPU"
        else:
            with gzip.open(file.parent / "output.f32.gz", "rb") as f:
                data = f.read()
            if hashlib.sha256(data).hexdigest() != r["output_sha256"]:
                raise ValueError("output identity mismatch " + str(file))
            values = np.frombuffer(data, dtype="<f4").reshape(r["blocks"], -1)
            if not np.all(np.isfinite(values)):
                raise ValueError("nonfinite saved output")
            error = float(np.max(np.abs(values - reference_block(case["kind"], case["length"]))))
            elapsed, unit = r["elapsed"], r["unit"]
            stamps = r["stamps"]
            if case["scope"] == "one_cta":
                window = stamps[0][3] - stamps[0][2]
            else:
                window = max(s[1] for s in stamps) - min(s[0] for s in stamps)
            if window != elapsed:
                raise ValueError("timing envelope mismatch " + str(file))
        if error > 1e-5:
            raise ValueError(f"CPU reference mismatch {file}: {error}")
        max_error = max(max_error, error)
        groups.setdefault(case["id"], []).append((r, elapsed, unit, error))
    return groups, max_error


def summarize(groups, predictions):
    rows = []
    for case_id, records in sorted(groups.items()):
        rs = [x[0] for x in records]
        values = [x[1] for x in records]
        case = rs[0]["configuration"]
        median = statistics.median(values)
        prediction = predictions.get(case_id)
        predicted = prediction["predicted_time"] if prediction else None
        row = dict(
            case_id=case_id,
            role=rs[0]["trial_role"],
            processes=len(values),
            median=median,
            minimum=min(values),
            maximum=max(values),
            cv=statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None,
            unit=records[0][2],
            predicted=predicted,
            error=(predicted - median) / median if predicted is not None else None,
            alternative=(
                prediction.get("alternative", {}).get("predicted_time") if prediction else None
            ),
            cpu_max_error=max(x[3] for x in records),
            warmup_all_converged=all(r.get("warmup_converged", False) for r in rs),
            pipeline_mode=rs[0].get("wgmma_pipeline_mode", "cutlass"),
            registers=rs[0].get("registers_per_thread"),
            **work(case, rs[0].get("blocks", 1)),
        )
        if row["alternative"] is not None:
            row["alternative_error"] = (row["alternative"] - median) / median
        if case["kind"] == 2:
            seg = [segments(r) for r in rs]
            # One CTA: its own segments.  All-GPU: median over all CTAs of each process.
            per_process = [[statistics.median(s[i] for s in cta) for i in range(5)] for cta in seg]
            fill, mainloop, out, total, total_ns = (
                statistics.median(p[i] for p in per_process) for i in range(5)
            )
            row.update(
                fill_cycles=fill,
                mainloop_cycles=mainloop,
                output_cycles=out,
                mainloop_cycles_per_ktile=mainloop / case["length"],
                cta_window_cycles=total,
            )
            if case["scope"] == "all_gpu":
                clocks = [s[3] / s[4] for cta in seg for s in cta if s[4] > 0]
                gaps, sms, depth = relaunch_gaps(rs[len(rs) // 2])
                row.update(
                    cta_window_ns=total_ns,
                    sm_clock_ghz_median=statistics.median(clocks),
                    relaunch_gap_ns_median=statistics.median(gaps) if gaps else None,
                    sms_used=sms,
                    ctas_per_sm_max=depth,
                )
        if prediction and prediction.get("segments") and case["scope"] == "one_cta":
            row.update({f"predicted_{k}": v for k, v in prediction["segments"].items()})
        rows.append(row)
    return rows


def fmt(x, digits=1):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def report(destination, folder, environment, protocol, machine, rows, max_error):
    text = [
        "# V01 留出验证结果",
        "",
        f"设备 {environment['hostname']}，{environment['gpu_uuid']}，"
        f"Slurm {environment['slurm_job_id']}，"
        f"CUDA 12.9 / sm_90a。模式：{protocol['role']}。CPU完整重算最大误差：{max_error}。",
        "",
    ]
    if protocol.get("prediction_sha256"):
        text += [
            f"预测文件SHA256 `{protocol['prediction_sha256']}`，"
            f"冻结于 {protocol['prediction_created_utc']}；"
            f"预测后探针源码改变：{protocol['probe_changed_after_prediction']}。",
            "",
        ]
    if machine:
        text += [
            "## target_async 机器码",
            "",
            f"{machine['registers']} registers/thread，spill {machine['spill_bytes']} B；"
            f"C7520={machine['c7520_serialized']}，C7517={machine['c7517_injected_wait']}，"
            f"C7519={machine['c7519_injected_arrive']}。",
            f"静态计数：HGMMA.64x256x16 {machine['hgmma_64x256x16']}（其中带gsb0提交 "
            f"{machine['hgmma_commit_gsb0']}），WARPGROUP.ARRIVE {machine['warpgroup_arrive']}，"
            f"wait1 {machine['wait1']}，wait0 {machine['wait0']}，空槽释放arrive "
            f"{machine['empty_release_arrive']}，UTMALDG {machine['tma_load']}，UTMASTG "
            f"{machine['tma_store']}，STS.64 {machine['sts64']}。",
            f"主循环体（`target_mainloop.sass`）{machine['mainloop_instructions']}条：HGMMA "
            f"{machine['mainloop_hgmma']}，wait1 {machine['mainloop_wait1']}，wait0 "
            f"{machine['mainloop_wait0']}。",
            "",
        ]
    text += [
        "## 计时与预测",
        "",
        "| 配置 | 进程 | 中位数 | 范围 | CV | 单位 | 预测 | 误差e |",
        "|---|---:|---:|---|---:|---|---:|---:|",
    ]
    for r in rows:
        cv = f"{r['cv'] * 100:.2f}%" if r["cv"] is not None else "–"
        err = f"{r['error'] * 100:+.1f}%" if r["error"] is not None else "–"
        digits = 5 if r["unit"].startswith("cuda") else 0
        text.append(
            f"| {r['case_id']} | {r['processes']} | {fmt(r['median'], digits)} | "
            f"{fmt(r['minimum'], digits)}–{fmt(r['maximum'], digits)} | {cv} | {r['unit']} | "
            f"{fmt(r['predicted'], digits)} | {err} |"
        )
    alt = [r for r in rows if r.get("alternative") is not None]
    if alt:
        text += ["", "CUTLASS的R00-A锚定备选模型：", ""]
        for r in alt:
            text.append(
                f"- {r['case_id']}：备选预测 {r['alternative']:.5f} ms，误差 "
                f"{r['alternative_error'] * 100:+.1f}%。"
            )
    target = [r for r in rows if "fill_cycles" in r]
    if target:
        text += [
            "",
            "## 目标组合分段（clock64 cycle，同CTA；整卡为各CTA中位数）",
            "",
            "| 配置 | 预填 | 主循环 | 主循环/Ktile | 输出 | CTA窗口 | 整卡附加 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in target:
            extra = ""
            if "sm_clock_ghz_median" in r:
                extra = (
                    f"SM时钟≈{r['sm_clock_ghz_median']:.3f} GHz；CTA窗口 {r['cta_window_ns']:.0f} ns；"
                    f"同SM相邻CTA间隔中位 {fmt(r['relaunch_gap_ns_median'], 0)} ns；"
                    f"{r['sms_used']} SM×最多{r['ctas_per_sm_max']} CTA"
                )
            text.append(
                f"| {r['case_id']} | {r['fill_cycles']:.0f} | {r['mainloop_cycles']:.0f} | "
                f"{r['mainloop_cycles_per_ktile']:.1f} | {r['output_cycles']:.0f} | "
                f"{r['cta_window_cycles']:.0f} | {extra} |"
            )
        pred = [r for r in target if "predicted_fill" in r]
        if pred:
            text += ["", "单CTA预测分段（预填 / 主循环 / 输出）：", ""]
            for r in pred:
                text.append(
                    f"- {r['case_id']}：{r['predicted_fill']:.0f} / {r['predicted_mainloop']:.0f} / "
                    f"{r['predicted_output']:.0f}"
                )
    text += [
        "",
        "整卡SM时钟为各CTA clock64窗口/globaltimer窗口之比的中位数，globaltimer分辨率限制其精度。"
        "完整输出见各进程`output.f32.gz`；SASS为`build/v01.sass`、`target_async.sass`；"
        "CUTLASS的SASS为`build/r00_cutlass.sass`；CPU入口为归档`source/analyze_v01.py`。",
        "",
    ]
    (destination / "report.md").write_text("\n".join(text))


def analyze(folder, output=None):
    folder = Path(folder)
    destination = Path(output) if output else folder
    destination.mkdir(parents=True, exist_ok=True)
    protocol, prediction_sha = check_identities(folder)
    environment = json.loads((folder / "environment.json").read_text())
    predictions = {}
    if prediction_sha:
        plan = json.loads((folder / "predictions.json").read_text())
        predictions = {c["case_id"]: c for c in plan["cases"]}
    groups, max_error = load(folder, prediction_sha)
    rows = summarize(groups, predictions)
    has_target = "_Z12target_async" in (folder / "build/v01.sass").read_text()
    machine = audit_target(folder, destination) if has_target else None
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (destination / "cases.csv").open("w") as out:
            w = csv.DictWriter(out, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    errors = [abs(r["error"]) for r in rows if r["error"] is not None]
    summary = dict(
        cpu_max_error=max_error,
        prediction_sha256=prediction_sha,
        machine_code=machine,
        cases=rows,
        absolute_error_median=statistics.median(errors) if errors else None,
        absolute_error_max=max(errors) if errors else None,
    )
    (destination / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    report(destination, folder, environment, protocol, machine, rows, max_error)
    (destination / "analyzer_identity.json").write_text(
        json.dumps(
            dict(
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                python=sys.version,
                input=str(folder.resolve()),
            ),
            indent=2,
        )
        + "\n"
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    analyze(args.input, args.output)
