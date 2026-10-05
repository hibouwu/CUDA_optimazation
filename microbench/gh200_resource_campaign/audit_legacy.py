#!/usr/bin/env python3
"""Read-only GH200 archive audit; never upgrades missing evidence to a pass."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

FAMILIES = {
    "initial": ("probe.cu", "probe", 32, 7, (128, 512, 2048)),
    "sustained": ("stress.cu", "stress", 42, 5, None),
    "audit": ("audit.cu", "audit", 7, 12, (8192, 32768, 65536)),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, context, tol=1e-8):
    require(math.isfinite(float(a)) and math.isfinite(float(b)), context + ": nonfinite")
    require(math.isclose(a, b, rel_tol=tol, abs_tol=1e-10), context + ": mismatch")


def operation_work(case):
    """Derive work from instruction shape and participation, not recorded work."""
    threads = int(case["threads"])
    require(threads > 0 and threads % 32 == 0, "invalid participation")
    ptx = case["ptx"]
    if ptx.startswith("fma."):
        return 2 * threads * (2 if case["kind"].endswith("x2") else 1)
    match = re.search(r"\.m(\d+)n(\d+)k(\d+)\.", ptx)
    require(match is not None, "matrix shape missing")
    m, n, k = map(int, match.groups())
    width = 128 if ptx.startswith("wgmma.") else 32
    require(threads % width == 0, "partial collective")
    return 2 * m * n * k * (threads // width)


def check_trial(row, case, family, sms):
    it = int(row["iterations"])
    require(it >= 0 and row["max_abs_error"] == 0, "iterations/correctness")
    blocks = int(row.get("blocks", 1))
    require(blocks > 0, "nonpositive grid")
    expected = blocks * it * case["batch"] * case["chains"] * operation_work(case)
    require(row["work_flop"] == expected, "issued work mismatch")
    require(case["work_per_collective"] == operation_work(case), "manifest work mismatch")
    if family == "initial":
        require(row["cycles"] > 0, "nonpositive cycles")
        require(row["ptx_collectives"] == it * case["batch"] * case["chains"],
                "collective count mismatch")
        require(row["smid"] >= 0, "negative SM ID")
        require(math.isfinite(row["event_ms"]) and row["event_ms"] > 0, "invalid CUDA event")
        return expected / row["cycles"]

    require(row["scope"] in ("single_cta", "full_gpu"), "unknown scope")
    require(row["threads"] == case["threads"], "thread count mismatch")
    require(row["kind"] == case["kind"], "kind mismatch")
    details = row["blocks_detail"]
    require(len(details) == blocks, "CTA record count mismatch")
    ids = {r["smid"] for r in details}
    # Physical SM IDs need not be contiguous; only check nonnegative identities.
    require(all(i >= 0 for i in ids), "negative SM ID")
    require(len(ids) == row["unique_sms"], "SM coverage field mismatch")
    if row["scope"] == "single_cta":
        require(blocks == 1 and len(ids) == 1, "single CTA scope mismatch")
    else:
        require(blocks == sms * min(4, row["occupancy_limit_ctas_per_sm"]),
                "grid occupancy contract mismatch")
        require(len(ids) == sms, "full GPU missing SM coverage")
    for b in details:
        require(b["cycles"] > 0 and b["stop_ns"] > b["start_ns"], "invalid CTA interval")
    first = min(b["start_ns"] for b in details)
    last = max(b["stop_ns"] for b in details)
    require(row["start_ns"] == first and row["stop_ns"] == last, "grid interval mismatch")
    require(row["max_block_cycles"] == max(b["cycles"] for b in details),
            "max CTA cycle mismatch")
    require(math.isfinite(row["event_ms"]) and row["event_ms"] > 0, "invalid CUDA event")
    require((last - first) / 1e6 <= row["event_ms"] * 1.05, "device interval exceeds event")
    require(row["host_stop_unix_ns"] > row["host_start_unix_ns"], "invalid host interval")
    if family == "sustained":
        close(expected / (last - first) / 1000, row["tflops"], "TFLOP/s")
        return expected / row["max_block_cycles"] if row["scope"] == "single_cta" else None
    spans = {}
    for b in details:
        require(b["cycles"] == b["stop_cycle"] - b["start_cycle"],
                "local clock difference mismatch")
        lo, hi = spans.get(b["smid"], (b["start_cycle"], b["stop_cycle"]))
        spans[b["smid"]] = min(lo, b["start_cycle"]), max(hi, b["stop_cycle"])
    total = sum(hi - lo for lo, hi in spans.values())
    require(total == row["summed_sm_span_cycles"], "SM-span denominator mismatch")
    close(expected / total, row["flop_per_sm_cycle"], "SM-span rate", tol=1e-10)
    return expected / total


def check_source(case, source, sass):
    name = case["name"]
    match = re.search(r"__global__ void " + re.escape(name) + r"\(.*?\n}\n", source, re.S)
    require(match is not None, name + ": source function absent")
    body = match.group()
    require(case["ptx"] in body, name + ": PTX form mismatch")
    require(re.search(r"q\s*<\s*" + str(case["batch"]) + r"\b", body) is not None,
            name + ": batch mismatch")
    require(re.search(r"c\s*<\s*" + str(case["chains"]) + r"\b", body) is not None,
            name + ": chain count mismatch")
    require("clock64()" in body and "__syncthreads()" in body, name + ": timer/drain missing")
    symbol = f"_Z{len(name)}{name}iP6ResultPd"
    parts = re.split(r"Function\s*:\s*", sass)[1:]
    found = [x for x in parts if x.splitlines()[0].strip() == symbol]
    require(len(found) == 1, name + ": SASS function absent/ambiguous")
    token = ("HGMMA" if case["kind"].startswith("wgmma") else
             "DMMA" if case["kind"] == "mma_f64" else
             "HMMA" if case["kind"].startswith("mma_") else
             "FFMA" if case["kind"] == "f32" else
             "DFMA" if case["kind"] == "f64" else "HFMA2")
    require(re.search(r"\b" + token + r"(?:\.|\s)", found[0]) is not None,
            name + ": SASS operation absent")
    instructions = []
    for line in found[0].splitlines():
        inst = re.match(r"\s*/\*([0-9a-f]+)\*/(.*?) /\*", line)
        if inst:
            instructions.append((int(inst[1], 16), inst[2]))
    timers = [pc for pc, op in instructions if "SR_CLOCKLO" in op]
    require(len(timers) == 2, name + ": clock boundaries ambiguous")
    loops = 0
    for pc, op in instructions:
        branch = re.search(r"\bBRA\s+0x([0-9a-f]+)", op)
        if branch and timers[0] < int(branch[1], 16) < pc < timers[1]:
            lo = int(branch[1], 16)
            loop = "\n".join(text for addr, text in instructions if lo <= addr <= pc)
            require(re.search(r"\b" + token + r"(?:\.|\s)", loop) is not None,
                    name + ": compute absent in timed loop; possible hoist")
            loops += 1
    require(loops > 0, name + ": no timed compute loop")
    if case["kind"].startswith("wgmma"):
        waits = re.findall(r"WARPGROUP\.DEPBAR[^;]*;", found[0])
        pending = case.get("pending_after_wait", 0)
        require(any(re.search(r",\s*(?:0x)?" + str(pending) + r"\s*;", w) for w in waits),
                name + ": selected WGMMA wait absent")
        require(any(re.search(r",\s*(?:0x)?0\s*;", w) for w in waits),
                name + ": final WGMMA drain absent")
    return token


def audit_run(run_dir, family):
    source_name, binary_name, case_count, repeats, lengths = FAMILIES[family]
    result = {"run_id": run_dir.name, "family": family, "findings": [], "checked": {}}
    findings = result["findings"]
    def note(severity, code, message):
        findings.append({"severity": severity, "code": code, "message": message})
    try:
        for name in (source_name, "raw.jsonl", "summary.json", "sass.txt", "SHA256SUMS",
                     "compile_command.txt", "compile.log", "environment.txt", "environment_after.txt"):
            require((run_dir / name).is_file(), "missing artifact: " + name)
        source = (run_dir / source_name).read_text()
        sass = (run_dir / "sass.txt").read_text()
        summary = json.loads((run_dir / "summary.json").read_text())
        lines = [json.loads(x) for x in (run_dir / "raw.jsonl").read_text().splitlines() if x.strip()]
        device, rows = lines[0], lines[1:]
        require("GH200" in device["device"] and device["sms"] > 0, "device mismatch")
        require(summary["device"] == device, "summary device mismatch")
        command = (run_dir / "compile_command.txt").read_text()
        require("compute_90a,code=sm_90a" in command and source_name in command,
                "compile target/source mismatch")
        for line in (run_dir / "SHA256SUMS").read_text().splitlines():
            sha, name = line.split(maxsplit=1)
            target = run_dir / name.lstrip("*")
            require(target.resolve().parent == run_dir.resolve(), "hash path escapes run")
            if not target.is_file():
                note("gap", "missing_hashed_artifact", name)
            else:
                require(digest(target) == sha, "SHA256SUMS mismatch: " + name)
        for name, sha in summary.get("file_sha256", {}).items():
            require(digest(run_dir / name) == sha, "summary hash mismatch: " + name)
        if "raw_sha256" in summary:
            require(digest(run_dir / "raw.jsonl") == summary["raw_sha256"], "raw hash mismatch")
        if (run_dir / binary_name).is_file():
            header = (run_dir / binary_name).read_bytes()[:64]
            require(header[:4] == b"\x7fELF" and int.from_bytes(header[18:20], "little") == 183,
                    "binary is not archived AArch64 ELF")
        if family == "initial":
            cases = summary["cases"]
            note("gap", "no_original_case_manifest",
                 "原批次未归档独立 cases.json；本次用 summary 中的配置逐项核对归档源码，不伪造历史 manifest。")
        else:
            cases = json.loads((run_dir / ("audit_cases.json" if family == "audit" else "stress_cases.json")).read_text())["cases"]
        require(len(cases) == case_count, "case count mismatch")
        specs = {c["name"]: c for c in cases}
        require(len(specs) == case_count, "duplicate case ID")
        for case in cases:
            check_source(case, source, sass)
        groups, controls = defaultdict(list), defaultdict(list)
        for row in rows:
            require(row["case"] in specs, "unknown case")
            check_trial(row, specs[row["case"]], family, device["sms"])
            if family == "initial":
                groups[row["case"], row["iterations"]].append(row)
            elif family == "sustained":
                groups[row["case"], row["scope"]].append(row)
            elif row["phase"] == "measure":
                groups[row["case"], row["scope"], row["iterations"]].append(row)
            elif row["phase"] == "empty_control":
                require(row["iterations"] == 0, "nonempty control")
                controls[row["case"]].append(row)
            else:
                require(row["phase"] in ("warmup", "transition_warmup"), "unknown phase")
        expected_groups = (case_count * len(lengths) if family == "initial" else
                           case_count * 2 if family == "sustained" else case_count * 2 * len(lengths))
        require(len(groups) == expected_groups, "incomplete case/scope/length matrix")
        for key, samples in groups.items():
            ids = [r["round"] if family == "audit" else r["repeat"] for r in samples]
            require(len(samples) == repeats and set(ids) == set(range(repeats)),
                    "missing/duplicate repeat: " + str(key))
        if family == "initial":
            for c in cases:
                pts = [(it, statistics.median(r["cycles"] for r in groups[c["name"], it])) for it in lengths]
                slope, intercept = statistics.linear_regression(*zip(*pts))
                close(slope, c["slope_cycles_per_iteration"], "fitted slope")
                close(intercept, c["intercept_cycles"], "fitted intercept")
                close(operation_work(c) * c["batch"] * c["chains"] / slope,
                      c["cta_flop_per_cycle"], "fitted FLOP/cycle")
        else:
            require(len(summary["cases"]) == expected_groups, "summary row cardinality")
            summary_keys = [(r["case"], r["scope"], r["iterations"]) if family == "audit"
                            else (r["case"], r["scope"]) for r in summary["cases"]]
            require(set(summary_keys) == set(groups), "summary keys incomplete/duplicated")
            for record in summary["cases"]:
                key = ((record["case"], record["scope"], record["iterations"])
                       if family == "audit" else (record["case"], record["scope"]))
                require(key in groups, "summary case absent from raw")
                values = []
                for row in groups[key]:
                    if family == "audit":
                        values.append(row["work_flop"] / row["summed_sm_span_cycles"])
                    else:
                        denom = (row["max_block_cycles"] if row["scope"] == "single_cta" else
                                 (row["stop_ns"] - row["start_ns"]) * summary["reference_clock_hz"] / 1e9)
                        values.append(row["work_flop"] / denom)
                for field, value in (("median", statistics.median(values)), ("min", min(values)), ("max", max(values))):
                    summary_field = field if family == "audit" else "flop_per_cycle_" + field
                    close(value, record[summary_field], "summary " + summary_field)
        if family == "audit":
            require(len(controls) == case_count, "missing control cases")
            for samples in controls.values():
                require(len(samples) == repeats and {x["round"] for x in samples} == set(range(repeats)),
                        "control repeat mismatch")
            for name, state in summary["warmups"].items():
                samples = [r["event_ms"] for r in rows if r["phase"] == "warmup" and r["case"] == name]
                cv = statistics.stdev(samples[-5:]) / statistics.mean(samples[-5:])
                stable = len(samples) >= 8 and cv <= 0.01
                require(stable == state["stable"] and len(samples) == state["launches"], "warmup state mismatch")
                if not stable:
                    note("gap", "warmup_not_stable", name + ": 完整 kernel 时长预热未达阈值。")
            for pair in summary["paired_comparisons"]:
                a, b = ((0, 3) if pair["comparison"] == "wait3/wait0" else (3, 7))
                first = {r["round"]: r["flop_per_sm_cycle"] for r in groups[f"wgmma_f16_c2_q16_t128_p{a}", pair["scope"], 65536]}
                ratios = [r["flop_per_sm_cycle"] / first[r["round"]] for r in groups[f"wgmma_f16_c2_q16_t128_p{b}", pair["scope"], 65536]]
                close(statistics.median(ratios), pair["median_ratio"], "paired wait ratio")
        ncu_files = list(run_dir.glob("ncu_*.csv"))
        denied = any("ERR_NVGPUCTRPERM" in p.read_text() for p in ncu_files)
        note("gap", "counter_permission_denied" if denied else "no_counter_evidence",
             "没有可用动态硬件计数器证据；SASS presence 不能代替动态计数或缓存流量。")
        note("gap", "not_external_trials",
             f"{repeats} 次记录来自同一进程；未满足 Thor 式至少 10 次外部独立进程采样。")
        missing = [x for x in ("run_spec.json", "campaign_status.json", "progress.jsonl", "COMPLETE")
                   if not (run_dir / x).is_file()]
        if missing:
            note("gap", "missing_frozen_run_contract", "旧批次未生成：" + ", ".join(missing))
        telemetry = run_dir / "telemetry.csv"
        count = max(0, len(telemetry.read_text().splitlines()) - 1) if telemetry.exists() else 0
        if count < 2:
            note("gap", "insufficient_telemetry", f"仅 {count} 条遥测，不能证明整个测量窗口的频率/热状态。")
        note("limitation", "constant_operands", "性能探针使用固定简单输入；独立非均匀正确性补测不扩展其性能适用域。")
        note("limitation", "not_bare_instruction_latency", "窗口包含控制与同步，不是纯硬件指令延迟/启动间隔。")
        result["checked"] = {"cases": case_count, "formal_measurements": sum(map(len, groups.values())),
                             "timed_compute_loops_checked": True,
                             "all_raw_records": len(rows), "empty_controls": sum(map(len, controls.values())),
                             "telemetry_samples": count, "source_sha256": digest(run_dir / source_name),
                             "raw_sha256": digest(run_dir / "raw.jsonl"), "binary_present": (run_dir / binary_name).exists()}
    except (ValueError, KeyError, TypeError, OSError, StopIteration, ZeroDivisionError) as e:
        note("error", "integrity_failure", str(e))
    result["integrity_pass"] = not any(x["severity"] == "error" for x in findings)
    result["qualified_like_thor"] = result["integrity_pass"] and not any(x["severity"] == "gap" for x in findings)
    result["model_use"] = "conditional_empirical_only" if result["integrity_pass"] else "quarantine"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=Path("microbench/gh200_l0/results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "audit output already exists; choose a new run ID")
    audits = [audit_run(args.archive_root / ("20260930-" + family), family) for family in FAMILIES]
    args.output.mkdir(parents=True)
    result = {"schema_version": 1, "audit_scope": "Thor checks adapted to ROMEO; no GPU execution",
              "auditor_sha256": digest(Path(__file__)), "runs": audits}
    (args.output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    report = ["# GH200 历史微基准独立复审", "",
              "按 Thor 的证据项目复核，保留 ROMEO 的平台差异。完整性通过不等于完成正式资格检查。", "",
              "| 批次 | 原始数值与工件完整性 | 正式资格 | 可用范围 |", "|---|---|---|---|"]
    for a in audits:
        report.append(f"| {a['run_id']} | {'通过' if a['integrity_pass'] else '失败'} | {'通过' if a['qualified_like_thor'] else '未满足'} | {a['model_use']} |")
    for a in audits:
        report += ["", "## " + a["run_id"], "", "核对统计：" + json.dumps(a["checked"], ensure_ascii=False), ""]
        report += [f"- {f['severity']} / {f['code']}：{f['message']}" for f in a["findings"]]
    report += ["", "原有源码、原始数据和报告未被改写。正式重测需使用新的配置清单、运行编号、外部重复与结果状态；不能补造旧批次的预先冻结记录。"]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(args.output), "runs": [
        {k: a[k] for k in ("run_id", "integrity_pass", "qualified_like_thor", "model_use")} for a in audits]}, ensure_ascii=False))
    return 0 if all(a["integrity_pass"] for a in audits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
