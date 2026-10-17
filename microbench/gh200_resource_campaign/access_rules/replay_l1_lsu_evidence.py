#!/usr/bin/env python3
"""有限 CPU 回放：R01 global 链37进程、EXP13纯读40进程、EXP18 local/spill40进程。"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys


ACCESS = Path("gh200_resource_campaign/access_rules")
SUITE = Path("gh200_resource_campaign/20261001-resource-suite-v2")
R01 = ACCESS / "20261006-r01r06-unrolled/r01"
HBM = ACCESS / "20261017-HBM-01-replay-001"
GLOBAL = SUITE / "global_duplex/formal-b3-v1"
S18 = SUITE / "parallel/compute_onchip/s18-integration-v2/operations-v1"
LOCAL = S18 / "deployment/repo/results" / SUITE / "auxiliary/formal-v2"
LOCAL_CASES = ("local_explicit_words32", "local_explicit_words128",
               "spill_pressure_words32", "spill_pressure_words128")
READ_CASES = tuple(mode + "_" + size for size in ("small", "large")
                   for mode in ("read_ca", "read_cg"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def bind(root, name, expected):
    path = root / name
    require(path.resolve().is_relative_to(root), "输入路径逃离归档: " + str(path))
    require(sha(path) == expected, "SHA不符: " + str(path))
    return expected


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def pure_functions(path, names, namespace):
    """只抽冻结纯函数，避免旧评分主入口及模块级注册。"""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in selected} == set(names), "冻结纯函数缺失")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def exact_cv_squared(values):
    values = [Fraction(value) for value in values]
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1) / mean ** 2


def replay_global_chain(results):
    root = results / R01
    identity = {}
    identity["source_hashes.json"] = bind(root, "source_hashes.json",
        "06877158b07523fc0f47af7050139878e46d53f9afd8f3d3a7d887491451690b")
    for name, expected in read(root / "source_hashes.json").items():
        identity["source/" + name] = bind(root, "source/" + name, expected)
    identity["rules.json"] = bind(root, "rules.json",
        "65b6222c5f77aa59ab379ad35e1ef2fa99a569cd0ce681f52e9e8380486967fe")
    identity["build/r01"] = bind(root, "build/r01", read(root / "build/binary_hash.json"))
    identity["build/sass.txt"] = bind(root, "build/sass.txt",
        "12be2a36de45e03fe725463ee67f7be8d3f8957cb36ad93fb4101e013544f032")
    helper = module(root / "source/analyze_r01.py", "frozen_lsu_r01")
    rules, old_sass = read(root / "rules.json"), read(root / "sass_check.json")
    old_cases = {row["case"]: row for row in rules["cases"]}
    environment = read(root / "environment.json")
    functions = helper.sass_functions((root / "build/sass.txt").read_text())
    sass = []
    for name, instructions in functions.items():
        if not re.search(r"scalar_chainILi[34]ELi[14]ELb[01]", name):
            continue
        window, body = helper.timed_loop(instructions)
        info = dict(function=name, **helper.loop_summary(window, body))
        streams = int(re.search(r"ELi([14])ELb", name)[1])
        require(info["loop_opcodes"]["LDG.E.STRONG.GPU"] == 64 * streams and
                info["loop_opcodes"]["IMAD.WIDE.U32"] == 64 * streams, "global链操作数不符")
        if streams == 1:
            info["chain_links"] = helper.check_chain(body, ("LDG", "IMAD"))
        info["status"] = "passed"
        require(info == next(row for row in old_sass["functions"] if row["function"] == name),
                "global链原SASS评分不符")
        sass.append(info)
    require(len(sass) == 6, "global链六个机器码体不完整")
    rows, samples = [], []
    for size in ("small", "large"):
        for warps, streams, steps in ((1, 1, 128), (1, 1, 512), (1, 1, 2048),
                                      (4, 1, 2048), (4, 4, 2048)):
            op = "global_" + size
            case_id = f"{op}_w{warps}_s{streams}_{steps}"
            paths = sorted((root / "samples" / case_id).glob("*/result.json"))
            count = 10 if (size, warps, steps) == ("large", 1, 128) else 3
            require([path.parent.name for path in paths] == [f"trial-{n:02}" for n in range(count)],
                    "global链固定进程覆盖不符")
            records = []
            for path in paths:
                raw = read(path)
                active = 1 if warps == 1 else 128
                work = active * streams * steps * 4
                require(raw["status"] == "measured" and raw["trial_role"] == "formal" and
                        raw["case_id"] == case_id and raw["op"] == op and raw["steps"] == steps and
                        raw["warps"] == warps and raw["streams"] == streams and raw["unroll"] == 64 and
                        raw["iterations"] == steps // 64 and raw["control"] == 0 and
                        raw["active_lanes"] == active and raw["requested_bytes"] == work and
                        raw["local_bytes_per_thread"] == 0 and raw["warmup_converged"] and
                        raw["gpu_uuid"] == environment["gpu_uuid"], "global链条件/工作量/资格不符")
                require(raw["workset_bytes"] == (raw["l2_bytes"] // 4 if size == "small" else 4 * raw["l2_bytes"])
                        and raw["prepare"] == "full_linear_cg_read_before_each_window", "链工作集/预处理不符")
                require(read(path.parent / "command.json")[1:] == ["--op", op, "--warps", str(warps),
                    "--streams", str(streams), "--steps", str(steps), "--control", "0"], "链命令不符")
                stdout = read(path.parent / "stdout.json")
                require(all(raw[key] == value for key, value in stdout.items()) and
                        (path.parent / "stderr.log").read_text() == "", "链stdout/result不符")
                checked = helper.replay(path.parent, raw)
                stamp = raw["stamps"][0]
                require(len(raw["stamps"]) == 1 and stamp[4] == stamp[5] and
                        stamp[3] - stamp[2] == raw["elapsed"] and stamp[1] > stamp[0], "链窗口不符")
                samples.append(dict(case_id=case_id, trial=path.parent.name, checked_values=checked,
                    elapsed_cycles=raw["elapsed"], elapsed_ns=stamp[1] - stamp[0], requested_bytes=work,
                    hashes={p.name: sha(p) for p in sorted(path.parent.iterdir()) if p.is_file()}))
                records.append(raw)
            computed = helper.stats([raw["elapsed"] for raw in records])
            original = old_cases[case_id]
            require(all(original[key] == value for key, value in computed.items()) and
                    original["work"] == work and original["work_per_cycle"] == work / computed["mean"] and
                    original["cycles_per_step"] == computed["mean"] / steps, "链原统计不符")
            rows.append(original)
    relations, pairs = [], []
    for op in ("global_small", "global_large"):
        relation = rules["fits"][op]
        selected = sorted((row for row in rows if row["op"] == op and row["warps"] == 1), key=lambda row: row["steps"])
        require([row["mean"] for row in selected] == relation["mean_cycles"] and
                [row["count"] for row in selected] == relation["counts"], "原固定斜率输入不符")
        residuals = [row["mean"] - (relation["intercept_cycles"] + relation["cycles_per_step"] * row["steps"])
                     for row in selected]
        require(residuals == relation["residual_cycles"] and relation["status"] == "observed_linear" and all(
            abs(error) <= max(64, 3 * row["sd"], .01 * row["mean"]) for row, error in zip(selected, residuals)),
            "原固定关系残差/门槛不符")
        relations.append(dict(op=op, **relation))
        left, right = [row for row in rows if row["op"] == op and row["warps"] == 4]
        a = [sample["elapsed_cycles"] for sample in samples if sample["case_id"] == left["case"]]
        b = [sample["elapsed_cycles"] for sample in samples if sample["case_id"] == right["case"]]
        deltas = [y - x for x, y in zip(a, b)]
        pair = dict(op=op, matched_processes=3, s1_mean_cycle=left["mean"], s4_mean_cycle=right["mean"],
            mean_delta_cycle=statistics.mean(deltas), standard_error_delta_cycle=statistics.stdev(deltas) / 3 ** .5,
            work_rate_gain=4 * left["mean"] / right["mean"], raw_deltas_cycle=deltas)
        require(pair == next(value for value in rules["independent_stream_pairs"] if value["op"] == op),
                "原1/4流配对评分不符")
        pairs.append(pair)
    require(len(samples) == 37, "链37进程不符")
    return dict(identity=identity, environment=environment, profiles=rows, samples=samples, sass=sass,
        original_fixed_relations=relations, original_pairs=pairs, original_scores_exact=True,
        timing="same-CTA clock64窗；含IMAD地址依赖、64步展开、最终STS消费者和CTA barrier",
        boundary="每条流仍是.cg指针链；4warp×4条独立链不等于独立地址加载饱和；无.ca对照/无L1驻留证明",
        correctness="逐值重放output、starts、edges见证；原runner measured状态继承，旧格式无独立exit-code收据")


def compare_cache_sass(text):
    """仅允许匹配纯读体中一个LDG cache语义位不同。"""
    functions = {}
    for part in text.split("Function : ")[1:]:
        name = part.splitlines()[0].strip()
        lines = part.splitlines()
        instructions = []
        for i, line in enumerate(lines):
            match = re.search(r"/\*([0-9a-f]+)\*/\s*(.*?)\s*/\* 0x([0-9a-f]{16}) \*/", line)
            if match:
                high = re.search(r"/\* 0x([0-9a-f]{16}) \*/", lines[i + 1])
                require(high is not None, "不完整128bit SASS")
                instructions.append((match[1], match[2].strip(), match[3], high[1]))
        functions[name] = instructions
    ca, cg = [functions["gd_read_" + policy + "_r1_w0"] for policy in ("ca", "cg")]
    require(len(ca) == len(cg) == 184, "纯读184指令不符")
    differences = [(a, b) for a, b in zip(ca, cg) if a != b]
    require(len(differences) == 1, "纯读体还有cache策略之外的机器码差异")
    a, b = differences[0]
    require(a[0] == b[0] == "0410" and a[1].replace("STRONG.SM", "STRONG.GPU") == b[1] and
            a[1].startswith("LDG.E.128.STRONG.SM ") and a[2] == b[2] and
            int(a[3], 16) ^ int(b[3], 16) == 0x4000, "cache语义/编码差异不符")
    return dict(instruction_count=184, differing_instructions=1, difference_pc="0x410",
                ca_instruction=a[1], cg_instruction=b[1], high64_xor="0x4000", matched_body=True)


def replay_cache_read(results):
    archive, root = results / HBM, results / GLOBAL
    bind(archive, "replay/summary.json", "7937e7d7cbc6a422ba034e8bf3292a07a99b96dc9f0fb18e2a424c92d2262ec0")
    summary = read(archive / "replay/summary.json")
    bind(archive, "replay/evidence.json", summary["artifacts"]["evidence.json"])
    bind(archive, "source/replay_hbm_evidence.py", summary["replay_source_sha256"])
    helper = module(archive / "source/replay_hbm_evidence.py", "frozen_lsu_hbm")
    old = read(archive / "replay/evidence.json")
    for name, expected in old["input_identities"].items():
        bind(root, name, expected)
    complete, review = read(root / "COMPLETE"), read(root / "reviews/C/review.json")
    require(complete["C_review_sha256"] == old["input_identities"]["reviews/C/review.json"] and
            review["status"] == "pass" and review["qualification"]["C_complete"], "原C资格不符")
    stats_path = results / helper.FROZEN / "report_async_copy_v1.py"
    require(sha(stats_path) == old["statistics_helper_sha256"], "精确统计helper身份不符")
    exact_stats = module(stats_path, "frozen_lsu_exact_stats").stats
    profiles = [row for row in old["profiles"] if row["case_id"] in READ_CASES]
    originals = [row for row in old["samples"] if row["case_id"] in READ_CASES]
    require(len(profiles) == 4 and len(originals) == 40, "四纯读条件覆盖不符")
    samples = []
    for profile in profiles:
        rates = []
        selected = sorted((row for row in originals if row["case_id"] == profile["case_id"]), key=lambda row: row["trial"])
        require([row["trial"] for row in selected] == list(range(10)), "纯读十进程不符")
        for sample in selected:
            raw_path = root / sample["raw"]
            bind(root, sample["raw"], sample["raw_sha256"])
            require(sha(raw_path.with_name("receipt.json")) == sample["receipt_sha256"], "纯读receipt SHA不符")
            receipt = read(raw_path.with_name("receipt.json"))
            device, raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
            require(receipt["returncode"] == 0 and receipt["status"] == "valid" and
                    not receipt["timed_out"] and receipt["cleanup_confirmed"], "纯读进程失败")
            require(device["uuid"] == old["gpu_uuid"] and raw["case_id"] == sample["case_id"] and
                    raw["iterations"] == 16 and raw["seed"] == 3 + 19 * sample["trial"] and
                    raw["threads"] == 256 and raw["blocks"] == 528 and raw["errors"] == 0 and
                    raw["warmup_converged"] and raw["write_payload_bytes"] == 0, "纯读条件/资格不符")
            array_bytes = 17301504 if profile["working_set"] == "small" else 253034496
            work = array_bytes * 16
            require(raw["array_bytes"] == raw["aggregate_array_bytes"] == array_bytes and
                    raw["read_payload_bytes"] == raw["work_count"] == work and
                    raw["correctness"]["checked_elements"] == 528 * 256, "纯读工作量/host检查量不符")
            require(all(raw[key] == value for key, value in profile["resources"].items()) and
                    profile["resources"]["registers_per_thread"] == 28, "纯读匹配资源不符")
            details = raw["blocks_detail"]
            require(len(details) == 528 and {b["block_id"] for b in details} == set(range(528)) and
                    len({b["smid"] for b in details}) == 132 and all(
                    b["stop_cycle"] > b["start_cycle"] and b["stop_ns"] > b["start_ns"] for b in details),
                    "纯读CTA覆盖/窗口不符")
            begin, end = min(b["start_ns"] for b in details), max(b["stop_ns"] for b in details)
            require((raw["start_ns"], raw["stop_ns"]) == (begin, end) and
                    sample["elapsed_ns"] == end - begin and sample["read_payload_bytes"] == work,
                    "纯读包络/原样本不符")
            rate = Fraction(work, end - begin)
            require(float(rate) == sample["requested_GBps"], "纯读原rate不符")
            rates.append(rate)
            samples.append(sample)
        require(exact_stats(rates) == profile["original_statistics"], "纯读原精确评分不符")
    sass = compare_cache_sass((root / "build/sass.txt").read_text())
    return dict(identity=old["input_identities"], reused_archive=str(HBM),
        reused_source_sha256=summary["replay_source_sha256"], profiles=profiles, samples=samples, sass=sass,
        original_scores_exact=True, original_C_COMPLETE=True,
        timing=old["timing"], correctness=old["correctness_scope"],
        boundary="只匹配读.ca/.cg；全部写是.wb，没有写cache策略对照。固定顺序地址与即时checksum，不是指针依赖链或裸LDG饱和；不证明L1命中/驻留或物理带宽")


def replay_local_spill(results):
    root, published = results / LOCAL, results / S18 / "job733338-results/published-v1"
    identity = {}
    for name, expected in {
        "COMPLETE": "73f598125c3dbea0b17e59face71357580b0ccba12144c396348c091e77f1a43",
        "reviews/C/review.json": "2279bfcd7d31944242bb40f31d7c08115a904e796dc6aac44bfcc130a9a9b937",
        "snapshot/manifest.json": "e17934b5afdb433867b8e7dcac9565c203a574870b0761810d9a00c4a4cc7a86",
        "resolved_cases.json": "9073161d8a86e75d5b6e868b00bc4146e4492190f194708dea53d647a3458488",
        "binary/probe": "fc31f0d69d1f54c4c1c9fed8b2d545857e7096a1a2dfef4b3e85395b735e987c",
        "build/sass.txt": "e7518825aec0d0e09be6f3607a69d5b3f2b25b5077f3ab7f3a90c15ffda62e2d",
    }.items():
        identity[name] = bind(root, name, expected)
    complete, review = read(root / "COMPLETE"), read(root / "reviews/C/review.json")
    require(review["status"] == "pass" and complete["C_review_sha256"] == identity["reviews/C/review.json"], "S18 C资格不符")
    for name, key in (("summary.json", "summary_sha256"), ("measurement_manifest.json", "measurement_manifest_sha256")):
        identity[name] = bind(root, name, complete[key])
    for name, expected in review["gate_files"].items():
        bind(root, name, expected)
    for name, expected in {
        "manifest.json": "a3539d729f53017cd0737be8207a640e6fb6448ba077bce869f322c8b3d8a2d6",
        "result-table.json": "f6bf27deae15cbf15aa56cde22e889fa8afb8a4598695d43824ee68f99b5aad9",
        "sample-table.json": "955e74244b5ff9d5de83f785975386350f7fabfed8368b524d3b25b50104f85f",
    }.items():
        identity["published/" + name] = bind(published, name, expected)
    publication = read(published / "manifest.json")
    require(publication["status"] == "published_C_qualified_complete_and_relocated_strict_pass", "S18发布/迁移资格不符")
    code = root / "snapshot/repo/microbench/gh200_resource_campaign"
    snapshot = read(root / "snapshot/manifest.json")
    needed = ("common/suite_io.py", "auditors/auxiliary_sass_v1.py", "contracts/auxiliary_sass_baseline_v1.draft.json",
              "common/auxiliary_device_v1.cuh", "probes/auxiliary_formal_candidate_v1.cu")
    for name in needed:
        relative = "repo/microbench/gh200_resource_campaign/" + name
        identity["snapshot/" + relative] = bind(root / "snapshot", relative, snapshot[relative])
    frozen_stats = pure_functions(code / "common/suite_io.py", ("cv", "stats"), {"require": require, "math": math})
    stats = frozen_stats["stats"]
    sass_audit = pure_functions(code / "auditors/auxiliary_sass_v1.py", ("audit_sass",),
        dict(require=require, hashlib=hashlib, json=json, re=re,
             BASELINE_PATH=code / "contracts/auxiliary_sass_baseline_v1.draft.json"))["audit_sass"]
    all_sass = sass_audit((root / "build/sass.txt").read_text(), {})
    require(all_sass == read(root / "build/sass_audit.json"), "S18原全SASS比较不符")
    records = {row["case_id"]: row for row in read(published / "result-table.json")["records"]}
    original_samples = read(published / "sample-table.json")["samples"]
    summary_cases = read(root / "summary.json")["cases"]
    original_cases = {row["case_id"]: row for row in summary_cases}
    case_indices = {row["case_id"]: index for index, row in enumerate(summary_cases)}
    legacy_differences = read(root / "reviews/C/evidence/local-independent-proof.json")["local_summary_differences"]
    resolution = {row["case_id"]: row for row in read(root / "resolved_cases.json")["cases"]}
    manifest = read(root / "measurement_manifest.json")
    profiles, samples, diagnostics = [], [], []
    for case_id in LOCAL_CASES:
        record = records[case_id]
        selected = sorted((row for row in original_samples if row["case_id"] == case_id), key=lambda row: row["trial"])
        require([row["trial"] for row in selected] == list(range(10)) and record["qualified_parameter_export"] and
                record["status"] == "stable", "S18固定十进程/原发布资格不符")
        values, rates, windows = [], [], []
        for sample in selected:
            relative = "batches/" + case_id + f"/batch_00/trial_{sample['trial']:02}/attempt_00/raw.jsonl"
            raw_path = root / relative
            require(sha(raw_path) == sample["raw"]["sha256"] == manifest[relative] and
                    sha(raw_path.with_name("receipt.json")) == sample["receipt"]["sha256"] ==
                    manifest[relative.replace("raw.jsonl", "receipt.json")], "S18 raw/receipt绑定不符")
            device, raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
            receipt = read(raw_path.with_name("receipt.json"))
            n, words = record["N"], record["parameters"]["words_per_thread"]
            q = 128 * words * n
            require(receipt["returncode"] == 0 and receipt["status"] == "valid" and
                    not receipt["timed_out"] and receipt["cleanup_confirmed"] and
                    receipt["binary_sha256"] == identity["binary/probe"] and
                    receipt["relative_command"] == ["binary/probe", case_id, str(n), str(sample["seed"])], "S18命令/进程不符")
            require(device["uuid"] == "GPU-565e9222-0d4c-d7b1-4908-73a9d422a257" and
                    raw["case_id"] == case_id and raw["iterations"] == n == resolution[case_id]["resolved_iterations"] and
                    raw["seed"] == sample["seed"] == 3 + 19 * sample["trial"] and raw["errors"] == 0 and
                    raw["threads"] == 128 and raw["blocks"] == 1 and raw["scope"] == "one_cta" and
                    raw["correctness"]["checked_elements"] == 2 * 128 * words + 32 and
                    raw["logical_target_requests"] == raw["work_count"] == q and
                    raw["logical_local_loop_bytes"] == (8 * q if case_id.startswith("local") else 0) and
                    raw["final_output_artifacts"] == [] and raw["read_payload_bytes"] == raw["write_payload_bytes"] == 0,
                    "S18条件/工作量/原host检查不符")
            require(all(raw[key] == value for key, value in record["resource_identity"].items()), "S18实际资源不符")
            b = raw["blocks_detail"][0]
            c, ns = b["stop_cycle"] - b["start_cycle"], b["stop_ns"] - b["start_ns"]
            require(len(raw["blocks_detail"]) == 1 and b["block_id"] == 0 and
                    raw["begin_smid"] == raw["end_smid"] == b["smid"] and c > 0 and ns >= 10000 and
                    raw["event_ms"] * 1e6 >= ns and raw["complete_loop_cycles"] == sample["C"] == c and
                    raw["complete_loop_cycles_per_iteration"] == sample["C_over_N"] == c / n and
                    sample["Q"] == q and sample["Q_over_C"] == q / c and sample["inner_ns"] == ns and
                    sample["soft_target_met"] == (ns >= 20000000), "S18原窗口/逐样本评分不符")
            warm = raw["warmup_samples_ns"]
            require(8 <= len(warm) <= 30 and raw["warmup_converged"] and
                    exact_cv_squared(warm[-5:]) <= Fraction(1, 2500) and all(
                    exact_cv_squared(warm[i-5:i]) > Fraction(1, 2500) for i in range(8, len(warm))), "S18首个收敛预热不符")
            old_sample = original_cases[case_id]["batches"][0]["samples"][sample["trial"]]
            warm_cv = frozen_stats["cv"](warm[-5:])
            if warm_cv != old_sample["warmup_cv"]:
                legacy = [item for item in legacy_differences if item["path"] ==
                    f"/cases/{case_indices[case_id]}/batches/0/samples/{sample['trial']}/warmup_cv"]
                require(len(legacy) == 1 and legacy[0]["recorded"] == old_sample["warmup_cv"] and
                        legacy[0]["local"] == warm_cv and legacy[0]["ULP_diagnostic"] == 1,
                        "出现原封存诊断之外的CV反例；本入口不授通用容差")
                diagnostics.append(dict(case_id=case_id, trial=sample["trial"], field="warmup_cv",
                    original=old_sample["warmup_cv"], replay=warm_cv, ulp_difference=1,
                    original_threshold_pass=old_sample["warmup_cv"] <= .02, exact_threshold_pass=True,
                    original_diagnostic=legacy[0], exact_cv_squared=str(exact_cv_squared(warm[-5:]))))
            values.append(c / n); rates.append(q / c); windows.append(ns)
            samples.append(dict(case_id=case_id, trial=sample["trial"], N=n, C=c, Q=q, C_over_N=c/n,
                Q_over_C=q/c, inner_ns=ns, soft_target_met=ns >= 20000000, raw=relative,
                raw_sha256=sample["raw"]["sha256"], receipt_sha256=sample["receipt"]["sha256"],
                declared_host_checked_elements=raw["correctness"]["checked_elements"]))
        require(stats(values) == record["primary"] and stats(rates) == record["companion"] and
                record["window_min_ns"] == min(windows) and record["window_max_ns"] == max(windows) and
                record["soft_target_met_count"] == sum(ns >= 20000000 for ns in windows) and
                exact_cv_squared(values) <= Fraction(1, 400) and exact_cv_squared(rates) <= Fraction(1, 400),
                "S18原双统计/精确5%门槛不符")
        profiles.append(record)
    identity["build/compile.stderr"] = bind(root, "build/compile.stderr", manifest["build/compile.stderr"])
    ptxas = []
    compile_log = (root / "build/compile.stderr").read_text()
    for profile in profiles:
        symbol = profile["resource_identity"]["kernel_symbol"]
        match = re.search(r"Compiling entry function '" + symbol +
                          r"'.*?(?=ptxas info\s*: Compiling entry function|\Z)", compile_log, re.S)
        require(match is not None, "缺少原ptxas目标")
        counts = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", match[0])
        require(counts is not None, "缺少原ptxas stack/spill声明")
        ptxas.append(dict(kernel_symbol=symbol, stack_frame_bytes=int(counts[1]),
                          static_spill_store_bytes=int(counts[2]), static_spill_load_bytes=int(counts[3])))
    return dict(identity=identity, profiles=profiles, samples=samples, ptxas=ptxas,
        sass=[row for row in all_sass if row["kernel_symbol"] in {p["resource_identity"]["kernel_symbol"] for p in profiles}],
        original_C_COMPLETE=True, original_primary_companion_exact=True, cv_diagnostics=diagnostics,
        all_selected_fields_bitwise_exact=not diagnostics, generic_float_tolerance_added=False,
        original_local_strict_failure_preserved="原全27条件x86严格回放6个CV字段1ULP而拒绝；原ARM严格、finalize及迁移回放通过",
        timing="same-CTA clock64窗；包含local递推循环、循环后global output drain和CTA barrier；C/N完整迭代，Q/C逻辑递推量",
        correctness="承接原C及逐进程全host检查；正式40raw不保留输出数组，不能逐值重新重放；pilot/preflight完整输出资格继承",
        boundary="显式local与spill为不同循环组织/寄存器条件；不按二者比值定价spill，也不把logical_local_loop_bytes或ptxas静态spill字节当运行物理流量")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not sys.flags.optimize and sys.byteorder == "little", "冻结见证重放要求普通模式及little endian")
    sys.dont_write_bytecode = True
    results, output = args.results_root.resolve(strict=True), args.output.resolve()
    require(not output.is_relative_to(results / SUITE) and not output.is_relative_to(results / R01) and
            not output.is_relative_to(results / HBM), "输出不能位于所读旧归档内")
    output.mkdir(parents=True, exist_ok=False)
    try:
        evidence = {"r01": replay_global_chain(results), "cache_read": replay_cache_read(results),
                    "local_spill": replay_local_spill(results)}
        for name, value in evidence.items():
            save(output / (name + ".json"), value)
        counts = {name: dict(conditions=len(value["profiles"]), processes=len(value["samples"]))
                  for name, value in evidence.items()}
        require(counts == {"r01": {"conditions": 10, "processes": 37}, "cache_read": {"conditions": 4, "processes": 40},
                           "local_spill": {"conditions": 4, "processes": 40}}, "有限范围计数不符")
        save(output / "summary.json", dict(schema="gh200-l1-lsu-evidence-replay/1", status="verified",
            results_root=str(results), replay_source_sha256=sha(Path(__file__)), families=counts,
            total_conditions=18, total_processes=117,
            r01_full_output_values_replayed=sum(row["checked_values"] for row in evidence["r01"]["samples"]),
            local_cv_diagnostics=evidence["local_spill"]["cv_diagnostics"], new_GPU_executions=0, new_fits=0,
            L1_residency_proven=False, physical_LSU_bandwidth_proven=False, generic_spill_price_exported=False,
            artifacts={name + ".json": sha(output / (name + ".json")) for name in evidence}))
    except Exception as error:
        save(output / "failure.json", dict(status="failed", error=str(error)))
        raise
    print(f"已回放18条件117正式进程：{output / 'summary.json'}")


if __name__ == "__main__":
    main()
