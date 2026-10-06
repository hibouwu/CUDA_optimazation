#!/usr/bin/env python3
"""Recompute R00 work/time statistics and generate a readable report."""

from __future__ import annotations
import argparse
import array
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def recompute(record):
    if record["kind"] == "gemm":
        work = 2 * record["m"] * record["n"] * record["k"]
        elapsed = record["elapsed_ms"]
        if work <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid work or time")
        rate = work / (elapsed * 1e6)
        unit = "GFLOP/s/GPU"
    else:
        work = (
            2
            * 64
            * record["n"]
            * record["k"]
            * record["chains"]
            * record["per_chain_batch"]
            * record["groups"]
            * record["iterations"]
            * record["blocks"]
        )
        stamps = record["stamps"]
        if len(stamps) != record["blocks"]:
            raise ValueError("missing CTA timestamps")
        if record["scope"] == "one_cta":
            if record["unit"] != "clock64_cycle/CTA":
                raise ValueError("clock domain mismatch")
            elapsed = stamps[0][3] - stamps[0][2]
            unit = "FLOP/clock64_cycle/CTA"
        else:
            if record["unit"] != "globaltimer_ns/GPU":
                raise ValueError("clock domain mismatch")
            elapsed = max(s[1] for s in stamps) - min(s[0] for s in stamps)
            unit = "GFLOP/s/GPU"
        if elapsed != record["elapsed"]:
            raise ValueError("timing window does not match raw stamps")
        if work <= 0 or elapsed <= 0:
            raise ValueError("invalid work or time")
        rate = work / elapsed
    if work != record["work_flop"] or elapsed <= 0:
        raise ValueError("invalid work or time")
    return work, elapsed, rate, unit


def analyze(root):
    environment = json.loads((root / "environment.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    rows = []
    sources = []
    for case in cases:
        values = []
        times = []
        records = []
        for path in sorted((root / "samples" / case["id"]).glob("trial-*/result.json")):
            record = json.loads(path.read_text())
            if record["trial_role"] == "pilot":
                continue
            work, elapsed, rate, unit = recompute(record)
            if record["status"] != "measured":
                raise ValueError("failed numerical witness")
            if case["kind"] == "wgmma":
                expected_elements = (
                    record["blocks"] * record["threads"] * record["chains"] * (record["n"] // 2)
                )
                if record["output_elements"] != expected_elements:
                    raise ValueError("output shape mismatch")
                with gzip.open(path.parent / "output.f32.gz", "rb") as stream:
                    digest = hashlib.sha256()
                    size = 0
                    expected = (
                        0.0
                        if record["input_profile"] == "uniform_paired_sign_bounded"
                        else record["iterations"] * record["per_chain_batch"] * record["k"] / 256.0
                    )
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                        if block != struct.pack("<f", expected) * (len(block) // 4):
                            output_values = array.array("f")
                            output_values.frombytes(block)
                            if any(
                                abs(value - expected) > 1e-4 or not math.isfinite(value)
                                for value in output_values
                            ):
                                raise ValueError("independent WGMMA output mismatch")
                if (
                    size != 4 * record["output_elements"]
                    or digest.hexdigest() != record["output_sha256"]
                ):
                    raise ValueError("damaged WGMMA raw output")
            values.append(rate)
            times.append(elapsed)
            records.append(record)
            sources.append(
                dict(
                    case_id=case["id"],
                    path=str(path.relative_to(root)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        if not records:
            rows.append(dict(case_id=case["id"], count=0, status="not_run"))
            continue
        cv = statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None
        spill = case["kind"] == "wgmma" and any(r["local_bytes_per_thread"] for r in records)
        status = (
            "spill_condition"
            if spill
            else ("unstable" if cv is not None and cv > 0.05 else "measured")
        )
        rows.append(
            dict(
                case_id=case["id"],
                count=len(values),
                status=status,
                median_rate=statistics.median(values),
                min_rate=min(values),
                max_rate=max(values),
                unit=unit,
                median_time=statistics.median(times),
                time_unit="ms" if case["kind"] == "gemm" else records[0]["unit"],
                cv=cv,
                max_error=max(
                    r.get("max_output_error", r.get("max_absolute_error", 0)) for r in records
                ),
            )
        )
    fields = [
        "case_id",
        "status",
        "count",
        "median_time",
        "time_unit",
        "median_rate",
        "min_rate",
        "max_rate",
        "unit",
        "cv",
        "max_error",
    ]
    with (root / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    (root / "summary.json").write_text(
        json.dumps(dict(mode=environment["mode"], cases=rows), indent=2) + "\n"
    )
    heading = "R00 最小验证" if environment["mode"] == "smoke" else "R00 实测结果"
    report = [
        "# " + heading,
        "",
        f"设备：{environment['hostname']}，{environment['gpu_uuid']}；Slurm {environment['slurm_job_id']}；CUTLASS {environment['cutlass_tag']}。",
        "",
        "计时和工作率分别重算；单 CTA 使用 clock64，整卡使用 globaltimer 或 CUDA event。表中 CV 是跨进程工作率的样本标准差/均值。频率未知；不把 ns 换算成已测周期。",
        "",
        "| 配置 | 状态 | 进程数 | 中位时间 | 中位工作率 | CV |",
        "|---|---|---:|---|---|---:|",
    ]
    for row in rows:
        if not row["count"]:
            report.append(f"| {row['case_id']} | not_run | 0 | — | — | — |")
            continue
        cv = "—" if row["cv"] is None else f"{row['cv']*100:.3f}%"
        report.append(
            f"| {row['case_id']} | {row['status']} | {row['count']} | {row['median_time']:.6g} {row['time_unit']} | {row['median_rate']:.6g} {row['unit']} | {cv} |"
        )
    report += ["", "## 一条结果的重算", ""]
    first = next((p for p in sorted((root / "samples").glob("*/trial-*/result.json"))), None)
    if first:
        record = json.loads(first.read_text())
        work, elapsed, rate, unit = recompute(record)
        report += [
            f"原始记录：[{first.parent.parent.name}/{first.parent.name}]({first.relative_to(root)})。",
            "",
            f"工作量 {work} FLOP，时间 {elapsed} {'ms' if record['kind']=='gemm' else record['unit']}，工作率 {rate:.9g} {unit}。",
        ]
    report += [
        "",
        "## 适用范围",
        "",
        "- 小尺寸完成全输出检查，大尺寸保存4096个位置与实际值；相对L2误差仅基于这些样本，不是全矩阵Frobenius误差。",
        "- 数值检查使用有限、可精确表示的有符号坐标输入；任务容差未给定。实现见证通过不表示任意输入或任务精度通过。",
        "- FP8 GEMM采用TN列主序适配，B和D布局与主实例不同；预打包不在窗口内。输入scale为1，输出BF16。",
        "- WGMMA短检查使用坐标输入；正式FP16/BF16使用均匀输入，FP8使用正负A配对抵消的有界输入。",
        "- 寄存器、SMEM、编译日志、SASS及全部原始样本在本目录；local/spill条件单列。普通依赖库kernel的SASS未由本探针导出。",
        "- 驱逐准备和重复调用只描述准备方式，没有物理缓存命中率或HBM流量证据。",
        "",
        "[统计表](cases.csv) · [来源](sources.json) · [环境](environment.json) · [源码身份](source_hashes.json)",
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        directory = root / "plots"
        directory.mkdir(exist_ok=True)
        for unit in sorted({r["unit"] for r in rows if r["count"]}):
            selected = [r for r in rows if r["count"] and r["unit"] == unit]
            figure, axis = plt.subplots(
                figsize=(max(8, len(selected) * 0.7), 5), constrained_layout=True
            )
            y = [r["median_rate"] for r in selected]
            axis.bar(
                range(len(y)),
                y,
                yerr=[
                    [r["median_rate"] - r["min_rate"] for r in selected],
                    [r["max_rate"] - r["median_rate"] for r in selected],
                ],
                capsize=3,
            )
            axis.set_xticks(range(len(y)))
            axis.set_xticklabels(
                [r["case_id"] for r in selected], rotation=65, ha="right", fontsize=8
            )
            axis.set_ylabel(unit)
            axis.set_title("Median; error bars: min/max across processes")
            name = "cta.png" if "cycle" in unit else "gpu.png"
            figure.savefig(directory / name, dpi=150)
            plt.close(figure)
            report += ["", f"![{unit}](plots/{name})"]
    except ImportError:
        report += ["", "绘图依赖matplotlib未安装；可在有该依赖的CPU环境重新运行analyze.py。"]
    (root / "report.md").write_text("\n".join(report) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    analyze(parser.parse_args().input.resolve())
