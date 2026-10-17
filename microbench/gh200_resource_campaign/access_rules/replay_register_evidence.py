#!/usr/bin/env python3
"""有限回放 R02 源供给/消费与 R01 FFMA 依赖链的 473 条旧记录。"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import statistics
import sys


BASE = Path("gh200_resource_campaign/access_rules")
R02 = BASE / "20261008-R02-job736989-v8"
FRAME = BASE / "20261008-R02-frame-job737122-v1"
R01 = BASE / "20261006-r01r06-unrolled/r01"
SOURCE_MANIFESTS = {
    R02: "bd1a5b4435ed83df964e3aad10555a874e93224641e60a2da6140cbfa152ba47",
    FRAME: "956b945140f2471cc2e58eadcec4e95ae82eaa93a65eae0df536bdaaaf00f19c",
    R01: "06877158b07523fc0f47af7050139878e46d53f9afd8f3d3a7d887491451690b",
}
R02_RULES = "949cd3a89606eab0294782fade1e8ffb6b161a4225e5c29197ce636389de377d"
R02_ORIGINAL_RULES = "ead41bb62a861eaf8f25663353b0a17c9e8ecca75d5044c2e00315b451a7dd46"
FRAME_RULES = "c564230b49287435798e0f450f31eb0f131c6d75b90ebfa63ce80b5182359c6d"
FRAME_HELPER = "1b4e650558044876906fd77c8b1affc8408366a4ca4f39759b401aa02448319d"
R01_RULES = "65b6222c5f77aa59ab379ad35e1ef2fa99a569cd0ce681f52e9e8380486967fe"


def require(condition, message):
    if not condition:
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
    require(path.resolve().is_relative_to(root), "输入路径逃离归档")
    actual = sha(path)
    require(actual == expected, "SHA不符: " + str(path))
    return actual


def save(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_identity(results, relative):
    root = results / relative
    bind(root, "source_hashes.json", SOURCE_MANIFESTS[relative])
    for name, expected in read(root / "source_hashes.json").items():
        bind(root, name if relative != R01 else "source/" + name, expected)
    if relative == R01:
        binary = bind(root, "build/r01", read(root / "build/binary_hash.json"))
        sass = bind(root, "build/sass.txt", "12be2a36de45e03fe725463ee67f7be8d3f8957cb36ad93fb4101e013544f032")
    else:
        identities = read(root / "build/binary.json")
        binary = bind(root, "build/r02", identities["sha256"])
        sass = bind(root, "build/sass.txt", identities["sass_sha256"])
    return dict(source_manifest_sha256=SOURCE_MANIFESTS[relative], binary_sha256=binary,
                sass_sha256=sass, environment=read(root / "environment.json"),
                source_files=read(root / "source_hashes.json"))


def compare(original, replayed, location=""):
    """原评分对象精确相等。"""
    require(original == replayed, "原评分不相等: " + location)


def r02_raw_identity(root, expected_count):
    checks = []
    for path in sorted((root / "samples").glob("*/*/result.json")):
        record = read(path)
        require(record["returncode"] == record["check_errors"] == record["local_bytes"] == 0,
                "旧进程/正确性/无spill状态失败")
        require(record["case_id"] == path.parent.parent.name and record["label"] == path.parent.name,
                "目录与条件不符")
        files = {"result.json": sha(path)}
        for name in record["files"]:
            files[name + ".gz"] = sha(path.parent / (name + ".gz"))
        checks.append(dict(case_id=record["case_id"], label=record["label"],
                           progress_cycles=record["progress_cycles"], consumed_cycles=record["consumed_cycles"],
                           warmup_converged=record["warmup_converged"], hashes=files))
    require(len(checks) == expected_count, "固定记录数不符")
    return checks


def replay_r02(results, output):
    root = results / R02
    identity = source_identity(results, R02)
    bind(root, "analysis/rules.json", R02_ORIGINAL_RULES)
    bind(root, "reanalysis/with-carry-controls-v1/rules.json", R02_RULES)
    module = load_module(root / "source/analyze_r02.py", "frozen_r02_analysis")
    module.analyze(root, output=output / "r02-analysis")
    old = read(root / "reanalysis/with-carry-controls-v1/rules.json")
    new = read(output / "r02-analysis/rules.json")
    compare(old, new)
    compare(read(root / "reanalysis/with-carry-controls-v1/checks.json"),
            read(output / "r02-analysis/checks.json"))
    original = read(root / "analysis/rules.json")
    compare(original["rules"], old["rules"], "/380_vs_382/rules")
    require((original["processes"], original["checked_u64_words"]) == (380, 1367296) and
            (new["processes"], new["checked_u64_words"]) == (382, 1379840) and
            new["resource_conditions"] == 32 and new["qualified_progress_conditions"] == 28 and
            new["qualified_consumer_conditions"] == 32, "原资格/380与382计数不符")
    raw_checks = r02_raw_identity(root, 382)
    carry = [row for row in raw_checks if row["label"] == "carry-check"]
    require(len(carry) == 2, "固定两条carry控制不符")
    return dict(schema="gh200-register-r02-replay/1", status="verified", identity=identity,
                original_rules_sha256=R02_ORIGINAL_RULES, current_rules_sha256=R02_RULES,
                rows=new["rules"], processes=382, raw_checks=raw_checks, original_scores_exact=True,
                checked_u64_words=new["checked_u64_words"], original_380_service_rows_unchanged=True,
                original_processes=380, carry_controls=carry, carry_controls_priced=False,
                qualification="32个SASS条件服务；28生产推进/32最终消费合格；两carry只校验u64进位",
                limitations=["t1为指令流推进，不等于所有结果可用", "8+4P只描述指定FADD消费者",
                             "banked为LDS地址控制，不是物理RF bank测量", "不导出通用RF源池倍率/端口带宽"],
                physical_RF_bandwidth_proven=False, new_fits=0, new_GPU_executions=0)


def replay_frame(results, output):
    root = results / FRAME
    identity = source_identity(results, FRAME)
    bind(root, "reanalysis/runner-chain-check.py", FRAME_HELPER)
    bind(root, "reanalysis/local-independent/rules.json", FRAME_RULES)
    previous = {name: sys.modules.get(name) for name in ("run_r02", "analyze_r02")}
    try:
        for name in previous:
            sys.modules[name] = load_module(root / "source" / (name + ".py"), name)
        helper = load_module(root / "reanalysis/runner-chain-check.py", "frozen_frame_chain_check")
        helper.analyze(root, output / "frame-analysis")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    new = read(output / "frame-analysis/rules.json")
    compare(read(root / "reanalysis/local-independent/rules.json"), new)
    compare(read(root / "reanalysis/local-independent/sass-check.json"),
            read(output / "frame-analysis/sass-check.json"))
    require(new["processes"] == 82 and new["checked_values"] == 514304 and len(new["conditions"]) == 8,
            "固定帧计数不符")
    require(all(row["registers"] == [40] and row["qualified"] for row in new["conditions"]), "固定帧资格不符")
    plain = [row for row in new["conditions"] if row["trace"] == 0]
    require([row["cycles"] for row in plain] == [9880, 9880, 10400, 10400], "固定帧plain窗口不符")
    return dict(schema="gh200-register-fixed-frame-replay/1", status="verified", identity=identity,
                helper_sha256=FRAME_HELPER, original_rules_sha256=FRAME_RULES,
                rows=new["conditions"], processes=82, checked_u64_words=514304,
                raw_checks=r02_raw_identity(root, 82), original_scores_exact=True,
                plain_rotation_over_reuse=10400 / 9880, logical_FLOP_per_window=new["flop"],
                qualification="同kernel40reg/thread仍有5.26%条件差；两个branch body排程/源字段仍不同",
                limitations=["不同GPU上的旧22%不能作同卡前后差分", "物理目的寄存器编号数量不等于逻辑累加链数",
                             "不把reuse标记统计当作operand cache命中或物理RF端口价格"],
                new_fits=0, new_GPU_executions=0)


def replay_ffma(results):
    root = results / R01
    identity = source_identity(results, R01)
    bind(root, "rules.json", R01_RULES)
    module = load_module(root / "source/analyze_r01.py", "frozen_r01_analysis")
    rules = read(root / "rules.json")
    old_cases = {row["case"]: row for row in rules["cases"]}
    old_sass = read(root / "sass_check.json")
    functions = module.sass_functions((root / "build/sass.txt").read_text())
    name = next(name for name in functions if "scalar_chainILi0ELi1ELb0" in name)
    window, body = module.timed_loop(functions[name])
    info = dict(function=name, **module.loop_summary(window, body),
                chain_links=module.check_chain(body, ("FFMA",)), status="passed")
    require(info == next(row for row in old_sass["functions"] if row["function"] == name), "FFMA原SASS分析不符")
    require(info["chain_links"] == 64 and info["loop_opcodes"]["HFMA2.MMA"] == 1,
            "FFMA64步/准备指令不符")
    rows, raw_checks = [], []
    for steps in (128, 512, 2048):
        case_id = f"ffma_w1_s1_{steps}"
        paths = sorted((root / "samples" / case_id).glob("*/result.json"))
        require([p.parent.name for p in paths] == [f"trial-{t:02}" for t in range(3)], "FFMA三进程覆盖不符")
        records = []
        for path in paths:
            record = read(path)
            require(record["status"] == "measured" and record["trial_role"] == "formal" and
                    record["case_id"] == case_id and record["op"] == "ffma" and
                    record["steps"] == steps and record["iterations"] == steps // 64 and
                    record["warps"] == record["streams"] == 1 and record["unroll"] == 64 and
                    record["control"] == record["local_bytes_per_thread"] == 0 and
                    record["gpu_uuid"] == identity["environment"]["gpu_uuid"] and record["warmup_converged"],
                    "FFMA原条件或资格不符")
            require(read(path.parent / "command.json")[1:] == ["--op", "ffma", "--warps", "1",
                    "--streams", "1", "--steps", str(steps), "--control", "0"], "FFMA命令不符")
            stdout = read(path.parent / "stdout.json")
            require(all(record[key] == value for key, value in stdout.items()) and
                    (path.parent / "stderr.log").read_text() == "", "FFMA stdout/result不符")
            checked = module.replay(path.parent, record)
            stamp = record["stamps"][0]
            require(len(record["stamps"]) == 1 and stamp[4] == stamp[5] == 124 and
                    stamp[3] - stamp[2] == record["elapsed"] and record["work_flop"] == 2 * 32 * steps,
                    "FFMA原窗口或工作量不符")
            raw_checks.append(dict(case_id=case_id, trial=path.parent.name, checked_values=checked,
                elapsed_cycles=record["elapsed"], hashes={p.name: sha(p) for p in sorted(path.parent.iterdir()) if p.is_file()}))
            records.append(record)
        stats = module.stats([record["elapsed"] for record in records])
        old = old_cases[case_id]
        require(all(old[key] == value for key, value in stats.items()) and
                old["work"] == 2 * 32 * steps and old["work_per_cycle"] == old["work"] / stats["mean"],
                "FFMA原统计不符")
        rows.append(dict(case_id=case_id, steps=steps, **stats))
    relation = rules["fits"]["ffma"]
    require(relation["intercept_cycles"] == 265 and relation["cycles_per_step"] == 4.265625 and
            relation["residual_cycles"] == [0, 0, 0] and relation["counts"] == [3, 3, 3], "FFMA原关系不符")
    require([row["mean"] for row in rows] == relation["mean_cycles"] and all(
        row["mean"] == relation["intercept_cycles"] + relation["cycles_per_step"] * row["steps"] for row in rows),
        "FFMA固定关系未复现")
    return dict(schema="gh200-register-ffma-chain-replay/1", status="verified", identity=identity,
                original_rules_sha256=R01_RULES, rows=rows, original_relation=relation,
                processes=9, checked_values=sum(row["checked_values"] for row in raw_checks),
                raw_checks=raw_checks, sass=info, fixed_relation_exact=True,
                qualification="正式U64单warp依赖序列4.265625cycle/步；含HFMA2.MMA准备、控制与最终消费者",
                limitations=["不重新拟合；仅验证原固定关系", "4.00是b(U)假设下弱身份U32对照外推，不能替换正式斜率",
                             "不把依赖链斜率作为通用RF读写价格"], new_fits=0, new_GPU_executions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    results = args.results_root.resolve(strict=True)
    output = args.output.resolve()
    for relative in (R02, FRAME, R01):
        require(not output.is_relative_to(results / relative), "输出不能位于所读旧归档内")
    output.mkdir(parents=True, exist_ok=False)
    try:
        save(output / "r02.json", replay_r02(results, output))
        save(output / "frame.json", replay_frame(results, output))
        save(output / "ffma.json", replay_ffma(results))
        save(output / "summary.json", dict(
            schema="gh200-register-evidence-summary/1", status="verified", results_root=str(results),
            replay_source_sha256=sha(Path(__file__)), r02_processes=382, fixed_frame_processes=82,
            ffma_processes=9, total_processes=473, carry_controls_priced=False, new_fits=0, new_GPU_executions=0,
            artifacts={p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
            endpoint="固定RF源/消费与FFMA依赖旧证据完成；不授予通用RF bank/reuse倍率或裸延迟资格"))
    except Exception as error:
        save(output / "failure.json", dict(status="failed", error=str(error)))
        raise
    print(f"已回放473旧记录：{output / 'summary.json'}")


if __name__ == "__main__":
    main()
