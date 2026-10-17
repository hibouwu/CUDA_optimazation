#!/usr/bin/env python3
"""回放固定大工作集global读写旧证据，比较两种纯读写外推形式。"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


SUITE = Path("gh200_resource_campaign/20261001-resource-suite-v2")
RUN = SUITE / "global_duplex/formal-b3-v1"
REPORT = SUITE / "analysis/global-duplex-formal-b3-v1-r2"
FROZEN = SUITE / "implementation/s13-report-teaching-r2/frozen"
COMPLETE_SHA256 = "9ecb5211da1ec20422bd5cafd2b4a38f1ea35c51cec0558899a8135e6c5eff0a"
STATS_HELPER_SHA256 = "6fdbca5788dc19257be79dad32a8ccbc226eff1ec47fe1de8fabd8e56d50e5fb"
MODES = ("read_ca", "read_cg", "write", "dependent_copy", "independent_r1_w1",
         "independent_r2_w1", "independent_r4_w1", "independent_r1_w2", "independent_r1_w4")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(results, include_small):
    root = results / RUN
    report = results / REPORT
    require(digest(root / "COMPLETE") == COMPLETE_SHA256, "固定COMPLETE身份不符")
    complete = read_json(root / "COMPLETE")
    for name, key in (("summary.json", "summary_sha256"),
                      ("measurement_manifest.json", "measurement_manifest_sha256"),
                      ("reviews/C/review.json", "C_review_sha256")):
        require(digest(root / name) == complete[key], "COMPLETE封套不符: " + name)
    review = read_json(root / "reviews/C/review.json")
    require(review["status"] == "pass" and review["qualification"]["C_complete"], "原独立C未通过")
    for name, expected in review["gate_files"].items():
        require(digest(root / name) == expected, "原C gate身份不符: " + name)
    finalize_path = results / SUITE / "implementation/s13-finalize/completion-collection.json"
    finalize = read_json(finalize_path)
    require(finalize["status"] == "original_runtime_strict_finalize_exit0_COMPLETE_collected" and
            finalize["files_sha256"]["COMPLETE"] == COMPLETE_SHA256, "原strict finalize收据不符")
    helper = results / FROZEN / "report_async_copy_v1.py"
    require(digest(helper) == STATS_HELPER_SHA256, "冻结精确统计helper身份不符")
    spec = importlib.util.spec_from_file_location("frozen_global_stats", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exact_stats = module.stats
    summary = read_json(root / "summary.json")
    run_spec = read_json(root / "run_spec.json")
    manifest = read_json(root / "measurement_manifest.json")
    contract_path = root / run_spec["contract_path"]
    require(contract_path.resolve().is_relative_to(root), "契约路径逃离原run")
    contract = read_json(contract_path)
    require(contract["adapter_id"] == "global_duplex_formal_v1" and len(contract["cases"]) == 18,
            "固定18条件契约不符")
    cases = {case["id"]: case["parameters"] for case in contract["cases"]}
    original_cases = {case["case_id"]: case for case in summary["cases"]}
    csv_cases = {row["case_id"]: row for row in read_csv(report / "cases.csv")}
    csv_trials = {(row["case_id"], int(row["batch"]), int(row["trial"])): row
                  for row in read_csv(report / "trials.csv")}
    sizes = ("large", "small") if include_small else ("large",)
    selected = [mode + "_" + size for size in sizes for mode in MODES]
    require(set(cases) == set(original_cases) == set(csv_cases), "原条件覆盖不符")
    profiles, samples, process_intervals, seen_pids = [], [], [], set()
    for case_id in selected:
        parameters = cases[case_id]
        original = original_cases[case_id]
        require(original["status"] == "stable" and original["exportable"] and
                len(original["batches"]) == 1, "原条件不合格或批次数不符")
        batch = original["batches"][0]
        require(batch["complete"] and batch["batch"] == 0 and len(batch["samples"]) == 10,
                "原十进程批次不完整")
        rates, case_samples, identities = [], [], set()
        require({sample["trial"] for sample in batch["samples"]} == set(range(10)), "trial覆盖不符")
        for original_sample in batch["samples"]:
            trial = original_sample["trial"]
            receipt_path = root / original_sample["receipt"]
            require(receipt_path.resolve().is_relative_to(root), "receipt路径逃离原run")
            raw_path = receipt_path.with_name("raw.jsonl")
            raw_relative = raw_path.relative_to(root).as_posix()
            require(digest(raw_path) == original_sample["raw_sha256"] == manifest[raw_relative] and
                    digest(receipt_path) == manifest[original_sample["receipt"]], "raw/receipt SHA不符")
            receipt = read_json(receipt_path)
            require(receipt["returncode"] == 0 and receipt["status"] == "valid" and
                    receipt["cleanup_confirmed"] and not receipt["timed_out"], "旧进程失败")
            require(receipt["pid"] not in seen_pids, "重复PID")
            seen_pids.add(receipt["pid"])
            process_intervals.append((receipt["host_start_ns"], receipt["host_stop_ns"]))
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            require(len(rows) == 2 and rows[0] == summary["device"] and rows[1]["type"] == "trial",
                    "raw设备/样本格式不符")
            device, raw = rows
            require(raw["case_id"] == case_id and raw["iterations"] == 16 and
                    raw["threads"] == 256 and raw["scope"] == "all_gpu" and
                    raw["seed"] == 3 + 19 * trial, "冻结运行条件不符")
            blocks = device["sms"] * min(4, raw["occupancy_limit_ctas_per_sm"])
            requested = (4 * device["l2_cache_bytes"] if parameters["working_set_class"] == "large"
                         else device["l2_cache_bytes"] // 4)
            quantum = blocks * 256 * 16
            array_bytes = ((requested + quantum - 1) // quantum) * quantum
            reads, writes = parameters["read_requests_per_group"], parameters["write_requests_per_group"]
            read_bytes, write_bytes = 16 * array_bytes * reads, 16 * array_bytes * writes
            aggregate = array_bytes * (bool(reads) + bool(writes))
            require(raw["blocks"] == blocks and raw["requested_array_bytes"] == requested and
                    raw["array_bytes"] == array_bytes and raw["aggregate_array_bytes"] == aggregate and
                    raw["read_requests_per_group"] == reads and raw["write_requests_per_group"] == writes,
                    "实际工作集或读写比例不符")
            require(raw["read_payload_bytes"] == read_bytes and raw["write_payload_bytes"] == write_bytes and
                    raw["work_count"] == read_bytes + write_bytes, "方向逻辑请求量不符")
            expected_checks = (blocks * 256 if reads else 0) + (array_bytes // 4 if writes else 0)
            require(raw["errors"] == 0 and raw["correctness"]["checked_elements"] == expected_checks and
                    raw["warmup_converged"] and original_sample["warmup_converged"],
                    "旧正确性状态/声明检查量/暖机不符")
            details = raw["blocks_detail"]
            require(len(details) == blocks and {cta["block_id"] for cta in details} == set(range(blocks)) and
                    len({cta["smid"] for cta in details}) == device["sms"], "CTA/SM覆盖不符")
            require(all(cta["stop_cycle"] > cta["start_cycle"] > 0 and
                        cta["stop_ns"] > cta["start_ns"] > 0 for cta in details), "CTA计时不符")
            begin = min(cta["start_ns"] for cta in details)
            end = max(cta["stop_ns"] for cta in details)
            require(raw["start_ns"] == begin and raw["stop_ns"] == end, "globaltimer包络不符")
            rate = Fraction(read_bytes + write_bytes, end - begin)
            old = csv_trials[case_id, 0, trial]
            require(float(rate) == original_sample["value"] == float(old["value"]) and
                    end - begin == int(old["elapsed_ns"]) and int(old["work_count"]) == read_bytes + write_bytes,
                    "旧逐样本CSV或原rate不符")
            resources = tuple(raw[key] for key in ("registers_per_thread", "static_smem_bytes",
                                                   "dynamic_smem_bytes", "local_size_bytes",
                                                   "occupancy_limit_ctas_per_sm"))
            require(resources[1:4] == (1024, 0, 0), "drain资源或spill不符")
            identities.add((array_bytes, aggregate, blocks, resources))
            sample = dict(case_id=case_id, trial=trial, seed=raw["seed"], read_payload_bytes=read_bytes,
                          write_payload_bytes=write_bytes, total_payload_bytes=read_bytes + write_bytes,
                          elapsed_ns=end - begin, requested_GBps=float(rate),
                          warmup_windows=len(raw["warmup_samples_ns"]), declared_checks=expected_checks,
                          raw=raw_relative, raw_sha256=original_sample["raw_sha256"],
                          receipt_sha256=manifest[original_sample["receipt"]])
            case_samples.append(sample)
            rates.append(rate)
        require(len(identities) == 1, "同条件资源/工作集漂移")
        statistics = exact_stats(rates)
        old = csv_cases[case_id]
        for field in ("samples", "mean", "median", "minimum", "maximum", "cv"):
            require(statistics[field] == float(old[field]), "原CSV统计不符: " + field)
        require(statistics["cv_squared_exact"] == old["cv_squared_exact"] and
                Fraction(statistics["cv_squared_exact"]) <= Fraction(1, 400), "精确CV资格不符")
        array_bytes, aggregate, blocks, resources = next(iter(identities))
        first = case_samples[0]
        profiles.append(dict(
            case_id=case_id, working_set=parameters["working_set_class"], parameters=parameters,
            actual_array_bytes=array_bytes, aggregate_array_bytes=aggregate, blocks=blocks,
            iterations=16, threads=256, resources=dict(zip(
                ("registers_per_thread", "static_smem_bytes", "dynamic_smem_bytes", "local_size_bytes",
                 "occupancy_limit_ctas_per_sm"), resources)),
            read_payload_bytes=first["read_payload_bytes"], write_payload_bytes=first["write_payload_bytes"],
            total_payload_bytes=first["total_payload_bytes"], original_statistics=statistics,
            mean_rate_equivalent_ns=first["total_payload_bytes"] / statistics["mean"]))
        samples.extend(case_samples)
    process_intervals.sort()
    require(all(first[1] <= second[0] for first, second in zip(process_intervals, process_intervals[1:])),
            "原进程时间区间重叠")
    by_case = {profile["case_id"]: profile for profile in profiles}
    predictions = []
    for profile in profiles:
        size = profile["working_set"]
        read_rate = by_case["read_cg_" + size]["original_statistics"]["mean"]
        write_rate = by_case["write_" + size]["original_statistics"]["mean"]
        read_ns = profile["read_payload_bytes"] / read_rate
        write_ns = profile["write_payload_bytes"] / write_rate
        reference = profile["mean_rate_equivalent_ns"]
        predictions.append(dict(
            case_id=profile["case_id"], read_baseline="read_cg_" + size, write_baseline="write_" + size,
            read_baseline_mean_requested_GBps=read_rate, write_baseline_mean_requested_GBps=write_rate,
            reference_mean_rate_equivalent_ns=reference,
            serialized_sum_ns=read_ns + write_ns, serialized_sum_relative_error=(read_ns + write_ns) / reference - 1,
            complete_overlap_max_ns=max(read_ns, write_ns),
            complete_overlap_max_relative_error=max(read_ns, write_ns) / reference - 1))
    source = run_spec["source_path"]
    identities = {name: digest(root / name) for name in (
        "COMPLETE", "reviews/C/review.json", "summary.json", "measurement_manifest.json", "run_spec.json",
        "snapshot/manifest.json", source, "binary/probe", "build/sass.txt", "build/sass_audit.json")}
    return dict(
        schema="gh200-hbm-old-evidence-replay/1", status="verified", profiles=profiles,
        samples=samples, pure_endpoint_predictions=predictions, input_identities=identities,
        report_identities={name: digest(report / name) for name in ("cases.csv", "trials.csv")},
        statistics_helper_sha256=STATS_HELPER_SHA256, finalize_receipt_sha256=digest(finalize_path),
        original_qualification=review["qualification"], original_summary_qualification=summary["qualification"],
        gpu_uuid=summary["device"]["uuid"],
        scope="9个large条件；small只有在--include-small时纳入同协议对照",
        timing="入口取时后CTA gate；read含四word checksum drain；含写配置含MEMBAR.SC.GPU及CTA排空；globaltimer全grid包络",
        mixing="同一线程每group先r次read并即时消费，再w次独立store；copy写实际load值；并非分离读写流",
        mean_rate_time="Q/各独立进程请求速率的算术均值；是mean-rate等效窗，非原elapsed_ns算术均值",
        prediction_scope="sum和max是固定纯端点外推对照；不拟合，不赋予物理调度或HBM峰值语义",
        correctness_scope="承接原error_count_only host检查；旧raw未保存完整输出值，本轮仅核声明检查量与零错误状态",
        no_full_manifest_rescan=True, physical_HBM_bytes_proven=False, cache_residency_proven=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-small", action="store_true", help="增加9个small条件作协议对照")
    args = parser.parse_args()
    require(not sys.flags.optimize, "须保留冻结helper检查，不能使用-O/-OO")
    sys.dont_write_bytecode = True
    results = args.results_root.resolve(strict=True)
    output = args.output.resolve()
    protected = results / SUITE
    require(output != protected and protected not in output.parents, "输出不能位于所读旧归档内")
    output.mkdir(parents=True, exist_ok=False)
    try:
        evidence = replay(results, args.include_small)
        save_json(output / "evidence.json", evidence)
        save_csv(output / "samples.csv", evidence["samples"])
        save_csv(output / "predictions.csv", evidence["pure_endpoint_predictions"])
        save_json(output / "summary.json", dict(
            schema="gh200-hbm-evidence-summary/1", status="verified", results_root=str(results),
            replay_source_sha256=digest(Path(__file__)), include_small=args.include_small,
            large_conditions=9, small_conditions=9 if args.include_small else 0,
            raw_processes=len(evidence["samples"]), profiles=len(evidence["profiles"]),
            original_C_COMPLETE=True, new_GPU_executions=0, new_fits=0,
            artifacts={name: digest(output / name) for name in ("evidence.json", "samples.csv", "predictions.csv")},
            limitation="原合格逻辑请求服务；不是物理HBM带宽。Q/mean-rate等效窗分开报告sum/max外推误差。"))
    except Exception as error:
        save_json(output / "failure.json", dict(status="failed", error=str(error)))
        raise
    print(f"已回放{len(evidence['profiles'])}条件、{len(evidence['samples'])}旧进程：{output / 'summary.json'}")


if __name__ == "__main__":
    main()
