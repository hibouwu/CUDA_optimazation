#!/usr/bin/env python3
"""仅用 CPU 回放三轮冻结的 barrier 返回观察，不生成同步服务价格。"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / "results/gh200_resource_campaign/access_rules"
GPU_UUID = "GPU-43269fbc-449d-3e0f-908a-9c81229546d3"
GROUPS = (
    ("20261015-barrier-return-development-001", 18,
     "c36ad4b218101f254af32b49c32c8ac4b8ddb4011083ddc6fcaeadbf527066ce"),
    ("20261015-barrier-return-development-002", 16,
     "95249a2ba55369475b7a3516f0394d17905e713a3a78f691db4437b2ce3286f0"),
    ("20261015-barrier-return-development-003", 12,
     "223679877ba8ebcfc61d7e61147c059eef4a38b1edb9e5afc90f38db1289853a"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def archive_files(path):
    # 只读成员，不向文件系统解包。
    with tarfile.open(path) as archive:
        return {member.name.removeprefix("./"): archive.extractfile(member).read()
                for member in archive if member.isfile()}


def replay_sample(sample):
    require(sample["schema"] == "barrier-return/1", "raw schema 不符")
    require(sample["ctas"] == 1 and sample["rounds"] == 32, "冻结工作量不符")
    require(sample["shared_bytes"] == 215040 and sample["occupancy"] == 1,
            "驻留资源不符")
    arrivers, waiters = sample["arrivers"], sample["waiters"]
    require(arrivers in (1, 128) and waiters in (1, 128), "参与者数量不符")
    require(sample["checked"] == 32 * (arrivers + waiters), "checked 计数不符")
    require(len(sample["stamps"]) == 32 * 256, "线程戳数量不符")
    records = sample["cta_records"]
    require(len(records) == 1 and len(records[0]) == 5, "CTA 端点数量不符")
    smid, begin, end, begin_ns, end_ns = records[0]
    require(all(type(value) is int for value in records[0]), "CTA 戳须为整数")
    require(smid >= 0 and end > begin > 0 and end_ns > begin_ns > 0, "CTA 时间顺序不符")
    windows, entry_classes = [], Counter()
    for round_index in range(32):
        stamps = sample["stamps"][round_index * 256:(round_index + 1) * 256]
        require(all(len(pair) == 2 and all(type(value) is int for value in pair)
                    for pair in stamps), "线程戳须为整数对")
        arrivals = stamps[128:128 + arrivers]
        waits = stamps[:waiters]
        require(all(begin <= first <= last <= end and first > 0
                    for first, last in arrivals + waits), "参与线程时间顺序不符")
        for thread, pair in enumerate(stamps):
            if not (thread < waiters or 128 <= thread < 128 + arrivers):
                require(pair == [0, 0], "非参与线程戳非零")
        last_before = max(pair[0] for pair in arrivals)
        last_after = max(pair[1] for pair in arrivals)
        wait_before, wait_after = waits[0]
        require(min(pair[1] for pair in waits) >= last_before, "必要信号前错误返回")
        if wait_before > last_after:
            entry_class = "after_all_signal_stamps"
        elif wait_before < min(pair[0] for pair in arrivals):
            entry_class = "before_all_signal_stamps"
        else:
            entry_class = "overlapping_signal_stamps"
        entry_classes[entry_class] += 1
        # 最后必要信号只能定位在 [max(before), max(after)]，不取中点。
        windows.append(dict(
            cta=0, round=round_index, wait_cycles=wait_after - wait_before,
            post_signal_lower=wait_after - last_after,
            post_signal_upper=wait_after - last_before,
            wait_begin_to_last_signal=last_before - wait_before,
            arrival_bracket=last_after - last_before,
            arrival_skew=last_before - min(pair[0] for pair in arrivals),
            wait_return_skew=max(pair[1] for pair in waits) - min(pair[1] for pair in waits)))
    metrics = {key: statistics.median(window[key] for window in windows)
               for key in windows[0] if key not in ("cta", "round")}
    metrics["effective_ghz"] = (end - begin) / (end_ns - begin_ns)
    gap = sample["gap_cycles"]
    if gap == 512:
        require(entry_classes == {"after_all_signal_stamps": 32}, "+512 起点分类不符")
    elif gap == -512:
        require(entry_classes == {"before_all_signal_stamps": 32}, "-512 起点分类不符")
    else:
        require(gap == 0, "冻结 gap 不符")
    return windows, metrics, entry_classes, smid


def replay_group(name, expected_cases, expected_archive_sha, results=RESULTS):
    root = results / name
    returned = root / "returned"
    output_bytes = (root / "output.tar.gz").read_bytes()
    receipt = read_json(root / "receipt.json")
    require(sha256(output_bytes) == expected_archive_sha, f"{name}: 输出归档 SHA 不符")
    require(receipt == dict(returncode=0, bytes=len(output_bytes), sha256=expected_archive_sha),
            f"{name}: 归档 receipt 不符")
    output_files = archive_files(root / "output.tar.gz")
    input_files = archive_files(root / "input.tar.gz")
    binding = read_json(root / "binding.json")
    require((root / "binding.json").read_bytes() == (returned / "binding.json").read_bytes(),
            f"{name}: 返回 binding 不符")
    for relative, expected_sha in binding["files"].items():
        require(sha256((root / relative).read_bytes()) == expected_sha and
                sha256((returned / relative).read_bytes()) == expected_sha and
                sha256(input_files[relative]) == expected_sha,
                f"{name}: 冻结来源不符: {relative}")
    for relative in ("binding.json", "plan.json", "invocations.json", "build/probe",
                     "build/probe.sass", "build/resources.txt", "build/compile.log",
                     "build/nvcc-version.txt", "build/nvidia-smi.txt"):
        require(output_files[relative] == (returned / relative).read_bytes(),
                f"{name}: 展开归档不符: {relative}")
    plan = read_json(root / "plan.json")
    conditions = {case["id"]: case["arguments"] for case in plan["conditions"]}
    require(len(conditions) == expected_cases and plan["repeats"] == 3, "冻结条件数不符")
    old_analysis = read_json(root / "analysis.json")
    old_rows = {(row["case"], row["trial"]): row for row in old_analysis["rows"]}
    old_summaries = {row["case"]: row for row in old_analysis["summary"]}
    invocations = read_json(returned / "invocations.json")
    require(len(invocations) == expected_cases * 3 and len(old_rows) == len(invocations),
            "冻结进程数不符")
    rows, raw_checks, seen, smids = {}, [], set(), set()
    for record in invocations:
        case, trial = record["case"], record["trial"]
        require(case in conditions and trial in (0, 1, 2) and (case, trial) not in seen,
                "条件/进程重复或超出冻结范围")
        seen.add((case, trial))
        raw = (returned / record["raw"]).read_bytes()
        require(raw == output_files[record["raw"]] and sha256(raw) == record["sha256"],
                f"{name}/{case}/{trial}: raw SHA 不符")
        require(record["returncode"] == 0, f"{name}/{case}/{trial}: 进程失败")
        arguments = conditions[case]
        command = ["build/probe"] + [str(value) for pair in arguments.items() for value in pair]
        require(record["command"] == command, "命令与冻结参数不符")
        sample = json.loads(gzip.decompress(raw))
        for option, value in arguments.items():
            require(sample[option.removeprefix("--").replace("-", "_")] == value,
                    f"{name}/{case}: raw 条件不符: {option}")
        windows, metrics, entry_classes, smid = replay_sample(sample)
        old = old_rows[case, trial]
        require(old["raw"] == record["raw"] and old["sha256"] == record["sha256"],
                "原分析 raw 身份不符")
        condition = {option.removeprefix("--").replace("-", "_"): value
                     for option, value in arguments.items()}
        require(old["condition"] == condition,
                "原分析条件不符")
        require(old["windows"] == windows and old["metrics"] == metrics,
                f"{name}/{case}/{trial}: 独立公式与原分析不符")
        smids.add(smid)
        rows.setdefault(case, []).append(dict(trial=trial, metrics=metrics,
                                             entry_classes=entry_classes))
        raw_checks.append(dict(group=name, case=case, trial=trial, raw=record["raw"],
                               sha256=record["sha256"], status="verified", rounds=32,
                               checked_participants=sample["checked"], smid=smid,
                               entry_relative_to_signal_stamps=dict(entry_classes)))
    profiles = []
    require(set(rows) == set(conditions) == set(old_summaries), "条件覆盖不符")
    for case in sorted(rows):
        processes = sorted(rows[case], key=lambda row: row["trial"])
        require([row["trial"] for row in processes] == [0, 1, 2], "进程覆盖不符")
        medians = {key: statistics.median(row["metrics"][key] for row in processes)
                   for key in processes[0]["metrics"]}
        condition = {option.removeprefix("--").replace("-", "_"): value
                     for option, value in conditions[case].items()}
        require(old_summaries[case]["condition"] == condition, "条件摘要元数据不符")
        require(medians == old_summaries[case]["metrics"], "条件摘要与原分析不符")
        classes = Counter()
        for row in processes:
            classes.update(row["entry_classes"])
        profiles.append(dict(
            group=name, case=case, condition=old_summaries[case]["condition"],
            qualification="development_observed_call_profile", processes=3, rounds_per_process=32,
            wait_call_cycles=dict(median=medians["wait_cycles"],
                                  process_medians=[row["metrics"]["wait_cycles"] for row in processes]),
            last_signal_to_return_cycles=dict(
                median_bounds=[medians["post_signal_lower"], medians["post_signal_upper"]],
                process_median_bounds=[[row["metrics"]["post_signal_lower"],
                                        row["metrics"]["post_signal_upper"]] for row in processes]),
            entry_relative_to_signal_stamps=dict(classes)))
    command_text = (root / "command.json").read_text()
    require(f"export CUDA_VISIBLE_DEVICES={GPU_UUID}" in command_text,
            "归档启动命令 GPU 选择不符")
    identity = dict(
        group=name, archive_sha256=expected_archive_sha,
        source_sha256=sha256((returned / "source/probe.cu").read_bytes()),
        binary_sha256=sha256((returned / "build/probe").read_bytes()),
        sass_sha256=sha256((returned / "build/probe.sass").read_bytes()),
        binding_sha256=sha256((root / "binding.json").read_bytes()),
        original_analysis_sha256=sha256((root / "analysis.json").read_bytes()),
        bound_files=len(binding["files"]), actual_smids=sorted(smids),
        command_gpu_uuid=GPU_UUID,
        gpu_binding_scope="归档命令选择的 UUID；probe raw 本身未输出 UUID",
        warmup="每进程两次同条件调用；没有收敛检测",
        threads=256, dynamic_shared_bytes=215040, occupancy=1,
        static_shared_bytes=dict(ptxas_user=16, resource_total=1040),
        timing="同 SM clock64 cycle；整调用双时钟比值不用于换算局部窗口")
    return profiles, raw_checks, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS,
                        help="三份原始归档所在目录；运行保存副本时显式指定")
    parser.add_argument("--output", type=Path, required=True, help="必须尚不存在的新目录")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profiles, raw_checks, identities = [], [], []
    try:
        for name, cases, archive_sha in GROUPS:
            group_profiles, group_checks, identity = replay_group(name, cases, archive_sha, args.results_root)
            profiles.extend(group_profiles)
            raw_checks.extend(group_checks)
            identities.append(identity)
        require(len(profiles) == 46 and len(raw_checks) == 138 and
                sum(row["rounds"] for row in raw_checks) == 4416, "总覆盖计数不符")
        save_json(output / "conditions.json", profiles)
        save_json(output / "raw_checks.json", raw_checks)
        save_json(output / "summary.json", dict(
            schema="gh200-barrier-return-replay/1", status="verified",
            replay_source=str(Path(__file__).resolve()),
            replay_source_sha256=sha256(Path(__file__).read_bytes()),
            qualification="development_observed_call_profiles_only",
            conditions=46, raw_processes=138, rounds=4416, identities=identities,
            statistics="每进程32轮取中位，再跨三个独立进程取中位；区间两端分别保留",
            boundaries=dict(
                wait_call="等待前戳到等待后戳，可能包含后继事务登记",
                last_signal_to_return="最后必要信号位于[max(before),max(after)]；输出其到返回的上下界",
                entry_class="调用起点相对信号戳；未观察动态快/慢分支",
                exclusions="无实际 TMA/WGMMA；非硬件 ready 时戳、非可迁移返回价格；不拟合")))
    except Exception as error:
        save_json(output / "failure.json", dict(status="failed", error=str(error),
                                               verified_groups=[row["group"] for row in identities]))
        raise
    print(f"已验证 46 条件、138 进程、4416 轮：{output / 'summary.json'}")


if __name__ == "__main__":
    main()
