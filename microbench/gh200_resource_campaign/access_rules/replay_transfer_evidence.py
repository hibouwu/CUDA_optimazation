#!/usr/bin/env python3
"""CPU 回放有限异步搬运旧证据；运输率保持其原协议与完成边界。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st
import sys
import tarfile


SUITE = Path("gh200_resource_campaign/20261001-resource-suite-v2")
TENSOR = Path("gh200_resource_campaign/access_rules/20261014-tma-tensor-discrimination-001")
S16 = SUITE / "implementation/s16-main-execution/published-v1"
ASYNC = SUITE / "async_copy/formal-b3-v2"
ASYNC_REPORT = SUITE / "analysis/async-copy-formal-b3-v2"
TENSOR_ARCHIVE_SHA = "05f8585c918e405597bf0b5210e08b36abeabfb636ae62604356efa1a355d0e4"


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


def identity(root, names):
    return {name: digest(root / name) for name in names}


def replay_tensor(results):
    root = results / TENSOR
    returned = root / "returned"
    receipt = read_json(root / "receipt.json")
    require(receipt["returncode"] == 0 and receipt["sha256"] == TENSOR_ARCHIVE_SHA and
            digest(root / "output.tar.gz") == TENSOR_ARCHIVE_SHA and
            receipt["bytes"] == (root / "output.tar.gz").stat().st_size, "tensor 输出归档不符")
    with tarfile.open(root / "output.tar.gz") as archive:
        for name in ("binding.json", "plan.json", "invocations.json"):
            require(archive.extractfile("./" + name).read() == (returned / name).read_bytes(),
                    "tensor 返回索引与归档不符: " + name)
    binding = read_json(returned / "binding.json")
    for name, expected in binding["files"].items():
        require(digest(root / name) == expected and digest(returned / name) == expected,
                "tensor 冻结文件不符: " + name)
    plan = read_json(returned / "plan.json")
    cases = {case["id"]: case["arguments"] for case in plan["conditions"]}
    invocations = read_json(returned / "invocations.json")
    require(len(cases) == 24 and len(invocations) == 168 and plan["processes"] == 7,
            "tensor 冻结覆盖不符")
    seen, processes, events_count, order_edges = set(), [], 0, 0
    for invocation in invocations:
        case, repeat = invocation["id"], invocation["repeat"]
        require(case in cases and repeat in range(7) and (case, repeat) not in seen,
                "tensor 条件或进程覆盖不符")
        seen.add((case, repeat))
        relative = f"samples/r{repeat:02d}-{case}.json"
        path = returned / relative
        require(digest(path) == invocation["sha256"] and invocation["returncode"] == 0,
                "tensor raw 身份或进程返回码不符")
        sample = read_json(path)
        require(sample["schema"] == "cfg-b-tensor-probe/1" and
                sample["gpu_uuid"] == plan["gpu_uuid"] and
                sample["errors"] == sample["checked"] == sample["verify"] == 0,
                "tensor raw 模式不符")
        arguments = cases[case]
        expected_args = [str(value) for pair in arguments.items()
                         for value in ("--" + pair[0], pair[1])]
        require(invocation["command"][1:] == expected_args, "tensor 命令参数不符")
        for key, value in arguments.items():
            field = "active_groups" if key == "active" else key.replace("-", "_")
            require(sample[field] == value, "tensor raw 条件不符: " + field)
        ctas = sample["ctas_data"]
        require(len(ctas) == sample["ctas"] == len({cta["smid"] for cta in ctas}),
                "tensor CTA/SM 映射不符")
        require(sample["smem"] == 132096 and sample["registers"] == 40 and
                sample["local_bytes"] == 0 and sample["max_active_ctas_per_sm"] == 1,
                "tensor 资源不符")
        local_edges = 0
        if sample["trace"]:
            events = {(event["cta"], event["group"]): event["points"]
                      for event in sample["group_events"]}
            require(len(events) == len(sample["group_events"]) == sample["ctas"] * sample["groups"],
                    "tensor 逐组记录覆盖不符")
            for cta in ctas:
                for group in range(sample["groups"]):
                    points = events[cta["cta"], group]
                    require(len(points) == 5, "tensor 五端点缺失")
                    for unit in (0, 1):
                        values = [cta["begin"][unit]] + [point[unit] for point in points] + [cta["end"][unit]]
                        require(all(type(value) is int for value in values) and
                                values == sorted(values), "tensor 端点顺序不符")
                        # Q 是软件未归还许可数；物理目的缓冲固定四槽。
                        for distance in (sample["active_groups"], 4):
                            if group + distance < sample["groups"]:
                                require(points[4][unit] <= events[cta["cta"], group + distance][0][unit],
                                        "tensor 信用/目的槽提前复用")
                                local_edges += 1
            events_count += len(events)
        else:
            require(not sample["group_events"], "tensor 未记录模式含逐组戳")
        order_edges += local_edges
        processes.append(dict(case=case, repeat=repeat, raw=relative,
                              sha256=invocation["sha256"], status="verified",
                              trace=sample["trace"], observed_sms=len(ctas),
                              descriptor_a_sha256=hashlib.sha256(bytes.fromhex(sample["descriptor_a"])).hexdigest(),
                              descriptor_b_sha256=hashlib.sha256(bytes.fromhex(sample["descriptor_b"])).hexdigest(),
                              lifecycle_order_edges=local_edges))
    require(len(seen) == 168, "tensor 进程缺失")
    # 只执行已冻结的读取与评分公式；不调用候选模型或重新选择价格。
    source = returned / "source/score_tma_tensor_discrimination.py"
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(source.parent))
        spec = importlib.util.spec_from_file_location("frozen_tensor_score", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score = module.score(returned)
    finally:
        sys.path[:] = original_path
    require(score == read_json(root / "score.json"), "tensor 旧评分重放不符")
    with tarfile.open(root / "evidence/native-input.tar.gz") as archive:
        native_source = archive.extractfile("source/probes/tma_tensor_multicta.cu").read()
    return dict(
        family="cfg_b_tensor", original_qualification="冻结价格集合均失败；旧观测转为开发证据",
        scope="12资源条件×trace开关×7独立进程；三请求A16KiB+B0/B1各8KiB",
        identities=identity(root, ("binding.json", "plan.json", "score.json", "build/probe",
                                   "evidence/probe.sass", "evidence/native-input.tar.gz")),
        native_probe_source_sha256=hashlib.sha256(native_source).hexdigest(),
        archive_sha256=TENSOR_ARCHIVE_SHA, gpu_uuid=plan["gpu_uuid"],
        native_correctness=read_json(root / "evidence/native-check.json"),
        resource_conditions=12, processes=processes, group_events=events_count,
        lifecycle_order_edges=order_edges,
        endpoints=["issue_begin", "issue_end", "wait_begin", "wait_end", "released"],
        endpoint_scope="同issuer线程调用端点；released是软件许可标记；verify=0无实际消费者读取/CTA发布",
        observations=score["observations"],
        old_score_replayed=True, old_score=dict(
            primary_phase_response=score["primary_phase_response"],
            untraced_phase_response=score["untraced_phase_response"],
            candidates={name: dict(conclusion=value["conclusion"],
                                  passing_parameter_indices=value["passing_parameter_indices"],
                                  best_joint_max_error=value["best_joint_diagnostic"]["max_absolute_error"])
                        for name, value in score["candidates"].items()}),
        transfer_limit="局部wait不是裸TMA延迟；没有物理信用返回/投递完成戳；不导出通用路径价")


def replay_s16(results):
    root = results / S16
    final = read_json(root / "publication-final-state.json")
    require(final["publication_state"] == "published_after_independent_bridge_review" and
            final["eligible_conditioned_observations"] == 32 and
            final["retained_capacity_exclusions"] == 4, "S16 原发布资格不符")
    for entry in (final["parameters"], final["prepared_manifest"],
                  final["publication_review"], final["actual_C_review"]):
        require(digest(root / entry["path"]) == entry["sha256"], "S16 发布封套身份不符")
    for name, entry in read_json(root / "publication-manifest.json").items():
        require(digest(root / name) == entry["sha256"] and
                (root / name).stat().st_size == entry["bytes"], "S16 发布工件不符: " + name)
    parameters = read_json(root / "qualified-parameters.json")
    selected = {f"{direction}_16kib_s{stages}_r{requests}_{scope}"
                for direction in ("gmem_to_smem", "smem_to_gmem")
                for stages, requests in ((1, 1), (1, 4), (2, 2))
                for scope in ("one_cta", "all_gpu")}
    samples = defaultdict(list)
    for row in read_csv(root / "samples.csv"):
        if row["case_id"] not in selected:
            continue
        blocks, iterations, requests = [int(row[key]) for key in ("blocks", "iterations", "requests_per_item")]
        work = blocks * iterations * requests * 16384
        denominator = int(row["denominator"])
        require(work == int(row["work_count_bytes"]) and denominator > 0 and
                int(row["payload_bytes"]) == 16384 and row["warmup_converged"] == "True",
                "S16 工作量/分母不符")
        value = work / denominator  # B/ns 数值等于 GB/s；单CTA为B/cycle。
        require(value == float(row["value"]), "S16 已发布样本rate不符")
        samples[row["case_id"]].append(dict(trial=int(row["trial"]), seed=int(row["seed"]),
                                            work_bytes=work, denominator=denominator,
                                            value=value, raw_sha256=row["raw_sha256"]))
    profiles = []
    for parameter in parameters["parameters"]:
        case = parameter["case_id"]
        if case not in selected:
            continue
        rows = samples[case]
        require(len(rows) == 10 and {row["trial"] for row in rows} == set(range(10)),
                "S16 选定样本覆盖不符")
        require(st.median(row["value"] for row in rows) == parameter["value"], "S16 原中位数不符")
        source_run = (root / parameter["source_run"]).resolve()
        require(digest(source_run / "summary.json") == parameter["source_summary_sha256"] and
                digest(source_run / "source-manifest.json") == parameter["source_manifest_sha256"],
                "S16 原审查输入身份不符")
        profiles.append(dict(parameter=parameter, published_samples_recomputed=rows))
    require(len(profiles) == 12 and set(samples) == selected, "S16 固定12坐标缺失")
    manifest = read_json(source_run / "source-manifest.json")
    probe_source = "microbench/gh200_resource_campaign/probes/tma_stage_request_formal.cu"
    for name in ("bin/probe", probe_source):
        require(digest(source_run / "repo" / name) == manifest[name], "S16 原源码/二进制不符")
    return dict(
        family="s16_bulk", original_qualification=final, profiles=profiles,
        identities=identity(root, ("publication-final-state.json", "qualified-parameters.json", "samples.csv")),
        source_sha256=manifest[probe_source], binary_sha256=manifest["bin/probe"],
        replay_scope="12发布坐标、120样本的B×I×R×16384/分母/中位数；未重新打开全payload packs",
        prior_full_value_review_retained=True, original_GPU_terminal=final["original_GPU_terminal"],
        endpoints="准备后→prefill/稳态/完整tail排空及必要CTA门槛之后；G2S mbarrier，S2G full wait",
        transfer_limit="只复用限定完整运输协议率；软件S/R不是物理在途量；无独立信用归还时戳")


def replay_cp_async(results):
    root = results / ASYNC
    report = results / ASYNC_REPORT
    complete = read_json(root / "COMPLETE")
    for name, field in (("summary.json", "summary_sha256"),
                        ("measurement_manifest.json", "measurement_manifest_sha256"),
                        ("reviews/C/review.json", "C_review_sha256")):
        require(digest(root / name) == complete[field], "cp.async COMPLETE身份不符")
    review = read_json(root / "reviews/C/review.json")
    require(review["status"] == "pass", "cp.async 原C审查不合格")
    summary = read_json(root / "summary.json")
    spec = read_json(root / "run_spec.json")
    manifest = read_json(root / "measurement_manifest.json")
    require(digest(root / "binary/probe") == spec["binary_sha256"] == summary["binary_sha256"] and
            digest(root / "build/sass.txt") == spec["sass_sha256"] and
            digest(root / "snapshot/manifest.json") == spec["snapshot_manifest_sha256"],
            "cp.async 构建身份不符")
    source = spec["source_path"]
    snapshot = read_json(root / "snapshot/manifest.json")
    require(digest(root / source) == snapshot[source.removeprefix("snapshot/")],
            "cp.async 冻结probe源码不符")
    selected = {f"{modifier}_w16_stages{stages}_{scope}" for modifier in ("ca", "cg")
                for stages in (1, 2, 4) for scope in ("one_cta", "all_gpu")}
    cases = {case["case_id"]: case for case in summary["cases"]}
    published = {row["case_id"]: row for row in read_csv(report / "cases.csv")}
    rows = defaultdict(list)
    for row in read_csv(report / "trials.csv"):
        case = row["case_id"]
        if case not in selected:
            continue
        raw = root / row["raw"]
        receipt_path = root / row["receipt"]
        require(digest(raw) == row["raw_sha256"] == manifest[row["raw"]] and
                digest(receipt_path) == manifest[row["receipt"]], "cp.async raw/receipt身份不符")
        receipt = read_json(receipt_path)
        require(receipt["returncode"] == 0 and receipt["status"] == "valid" and
                not receipt["timed_out"] and receipt["cleanup_confirmed"], "cp.async 原进程失败")
        records = [json.loads(line) for line in raw.read_text().splitlines()]
        require(len(records) == 2 and records[0]["type"] == "device" and records[1]["type"] == "trial",
                "cp.async raw格式不符")
        device, sample = records
        require(device["uuid"] == summary["device"]["uuid"] and sample["case_id"] == case and
                sample["errors"] == 0 and sample["warmup_converged"] and
                sample["iterations"] == 8192 and sample["threads"] == 128 and
                sample["request_bytes_per_thread"] == 16 and sample["global_array_bytes"] == 8388608,
                "cp.async 条件或正确性状态不符")
        work = sample["blocks"] * 8192 * 128 * 8 * 16
        require(all(sample[key] == work for key in ("work_count", "read_payload_bytes",
                                                  "write_payload_bytes", "consumer_read_bytes")),
                "cp.async 工作量不符")
        details = sample["blocks_detail"]
        require(len(details) == sample["blocks"], "cp.async CTA覆盖不符")
        if sample["scope"] == "one_cta":
            require(len(details) == 1, "cp.async 单CTA数量不符")
            denominator = details[0]["stop_cycle"] - details[0]["start_cycle"]
        else:
            denominator = max(cta["stop_ns"] for cta in details) - min(cta["start_ns"] for cta in details)
        require(denominator > 0, "cp.async 分母须为正")
        value = work / denominator
        require(denominator == int(row["denominator"]) and work == int(row["Q_transport_bytes"]) and
                value == float(row["value"]) == receipt["validation"]["value"], "cp.async 分母或rate不符")
        rows[case].append(dict(trial=int(row["trial"]), work_bytes=work, denominator=denominator,
                               value=value, raw=row["raw"], raw_sha256=row["raw_sha256"]))
    profiles = []
    for case in sorted(selected):
        samples = rows[case]
        require(len(samples) == 10 and {sample["trial"] for sample in samples} == set(range(10)),
                "cp.async 固定样本覆盖不符")
        median = st.median(sample["value"] for sample in samples)
        original = cases[case]
        require(median == original["merged"]["median"] == float(published[case]["median"]) and
                original["status"] == "stable" and original["exportable"], "cp.async 原中位数/资格不符")
        profiles.append(dict(case_id=case, median=median, unit=original["unit"],
                             original_statistics=original["merged"], condition=published[case], samples=samples))
    return dict(
        family="cp_async", original_qualification="原COMPLETE及独立C通过；summary的pending_C旧标签原样保留",
        original_summary_qualification=summary["qualification"], profiles=profiles, original_C_review=review,
        identities=identity(root, ("COMPLETE", "summary.json", "measurement_manifest.json", "run_spec.json")),
        report_identities=identity(report, ("cases.csv", "trials.csv")),
        source_sha256=digest(root / source), binary_sha256=spec["binary_sha256"], sass_sha256=spec["sass_sha256"],
        gpu_uuid=summary["device"]["uuid"], replay_scope="固定16B ca/cg×S1/2/4×两scope；120份raw的Q和分母",
        endpoints="准备后→prefill/稳态/drain/相邻线程shared消费及CTA同步完成",
        transfer_limit="复合复制+消费率；没有目的ready/软件slot释放独立戳；8MiB共享源不授HBM或L2物理流量资格")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="显式指定总results目录")
    parser.add_argument("--output", type=Path, required=True, help="必须尚不存在的新目录")
    args = parser.parse_args()
    require(not sys.flags.optimize, "回放须保留冻结函数的assert，不能使用-O/-OO")
    sys.dont_write_bytecode = True
    results = args.results_root.resolve(strict=True)
    output = args.output.resolve()
    protected = (results / SUITE, results / TENSOR)
    require(all(output != root and root not in output.parents for root in protected),
            "输出不能位于本次读取的旧归档内")
    output.mkdir(parents=True, exist_ok=False)
    families = []
    try:
        for function, filename in ((replay_tensor, "tensor.json"), (replay_s16, "s16.json"),
                                   (replay_cp_async, "cp_async.json")):
            evidence = function(results)
            save_json(output / filename, evidence)
            families.append(dict(family=evidence["family"], file=filename, sha256=digest(output / filename)))
        save_json(output / "summary.json", dict(
            schema="gh200-transfer-evidence-replay/1", status="verified", results_root=str(results),
            replay_source_sha256=digest(Path(__file__)), families=families,
            tensor_resource_conditions=12, tensor_processes=168,
            s16_coordinates=12, s16_published_samples=120, cp_async_coordinates=12, cp_async_raw_processes=120,
            scope="有限旧证据回放；没有GPU、没有新拟合、没有通用搬运路径价格"))
    except Exception as error:
        save_json(output / "failure.json", dict(status="failed", error=str(error), completed_families=families))
        raise
    print(f"三个旧证据族回放通过：{output / 'summary.json'}")


if __name__ == "__main__":
    main()
