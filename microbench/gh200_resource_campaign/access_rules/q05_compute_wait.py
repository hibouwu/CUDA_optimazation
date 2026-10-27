#!/usr/bin/env python3
"""驻留计算wait0／wait1配对的准备、构建与全输出评分。"""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import statistics
import tarfile
import tempfile

import cfg_b_compute_chain as chain
from l2_saturation import command_record, digest, read_json, require, save_json


HERE = Path(__file__).resolve().parent
SCHEMA = "gh200-q05-compute-wait-plan/1"
SEEDS = (3, 22, 41)


def cases():
    return [dict(id=f"k{kt}_wait{retain}", ktiles=kt, wait_retain=retain,
                 trace=False, split="heldout" if kt == 32 else "dev",
                 wgmma_count=8*kt, work_flop=2*128*128*64*kt)
            for kt in (4, 13, 32) for retain in (0, 1)]


def prepare(root, headers):
    root.mkdir(parents=True, exist_ok=False)
    source = root/"source"
    source.mkdir()
    for name in ("q05_compute_wait.py", "cfg_b_compute_chain.py", "l2_saturation.py"):
        shutil.copy2(HERE/name, source/name)
    for name in ("cfg_b_compute_chain.cu", "r00_common.hpp"):
        shutil.copy2(HERE/"probes"/name, source/name)
    shutil.copy2(headers, source/"cutlass-3.9.2-include.tar.gz")
    save_json(root/"plan.json", dict(schema=SCHEMA, cases=cases(), seeds=list(SEEDS),
        threads=128, slots=6, dynamic_smem_bytes=196608, warmup_limit=60,
        development_conditions=4, heldout_conditions=2, dev_processes=12, heldout_processes=6,
        maximum_candidates=3, maximum_development_conditions=8, maximum_heldout_conditions=4,
        maximum_gpu_seconds=600, maximum_repairs=2, prediction=None, GPU_execution=False,
        prediction_forms=dict(unit="clock64_cycle/CTA", compute_cycles=512, commit_lead_cycles=256,
            first_commit_wait=False, tail_wait_retain=0,
            fixed_intercepts=dict(T_wait1="a1+512*K", T_wait0="a0+512*K",
                length_increment_4_to_13="4608 in both arms"),
            per_group_lag=dict(lag_bounds="0<d<512", T_wait1="a+512*K+d",
                T_wait0="a+512*K+(K-1)*d", wait0_minus_wait1="(K-2)*d",
                length_increment_wait1_4_to_13="4608", length_increment_wait0_4_to_13="4608+9*d"),
            fitted_parameters=None, predictions_k32=None),
        qualification="development_localization", source_sha256={p.name:digest(p) for p in source.iterdir()}))
    for split in ("dev", "heldout"):
        commands = []
        selected = [case for case in cases() if case["split"] == split]
        for trial, seed in enumerate(SEEDS):
            for index in range(len(selected)):
                case = selected[(index+trial) % len(selected)]
                commands.append(dict(case_id=case["id"], seed=seed,
                    folder=f"{split}/{case['id']}/seed{seed}",
                    argv=["build/probe", "--ktiles", str(case["ktiles"]),
                          "--wait-retain", str(case["wait_retain"]), "--trace", "0",
                          "--seed", str(seed), "--warmup-limit", "60"]))
        save_json(root/("commands.json" if split == "dev" else "heldout-commands.json"),
            dict(split=split, executed=False, freeze_required=split == "heldout", commands=commands))
    baseline = root/"baseline"
    baseline.mkdir()
    old = HERE.parents[2]/"results/gh200_resource_campaign/access_rules/Q14-composition/COMPUTE-01-04-resident-chain/20261017-COMPUTE-01-execution-002"
    files = {"source.cu":old/"source/cfg_b_compute_chain.cu",
             "target-cuda12.9.sass":old/"build/disassemble/stdout",
             "target-identity.json":old/"build/identity.json",
             "target-review.json":old/"build/target-review.json",
             "compiler-version.txt":old/"build/version/stdout",
             "compile-stderr.txt":old/"build/compile/stderr"}
    for name, path in files.items():
        shutil.copy2(path, baseline/name)
    save_json(baseline/"manifest.json", {name:dict(original=str(path),sha256=digest(baseline/name))
                                        for name,path in files.items()})
    (root/"README.md").write_text("# Q05 计算等待连接\n\n"
        "复用A的六槽驻留体，仅改变后续Ktile提交后的wait保留组数0／1。"
        "首次提交均无wait，末尾均wait0；布局、8条WGMMA/Ktile、完整累加器及消费者保持。\n\n"
        "4／13 Ktile×两wait各3seed，共12开发进程；32×两wait共6进程只准备，"
        "开发形成共同长度规则并冻结后才执行。当前没有有依据的幅度预测，按开发定位报告。"
        "两个无定参响应形式保存在plan.json的prediction_forms；开发只估计相应组合量。"
        "commands.json为开发命令，heldout-commands.json标记freeze_required。\n\n"
        "source/q05_compute_wait.py compile --output <包>只构建和反汇编；"
        "score --output <包> --split dev复用原独立整数参考检查全输出。"
        "目标机执行及单卡锁由主集成管理。\n")
    print(f"PREPARED {root}: 开发12、留出6进程；GPU执行0")


def load_plan(root):
    plan = read_json(root/"plan.json")
    require(plan["schema"] == SCHEMA and plan["cases"] == cases(), "Q05条件清单不符")
    for name, expected in plan["source_sha256"].items():
        require(digest(root/"source"/name) == expected, f"冻结源码变更：{name}")
    return plan


def compile_probe(root, nvcc):
    load_plan(root)
    compiler = shutil.which(nvcc)
    require(compiler is not None, "nvcc未找到")
    compiler = Path(compiler).resolve()
    build = root/"build"
    build.mkdir(exist_ok=False)
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-lineinfo",
             "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"]
    command_record(build/"version", [compiler, "--version"], 30)
    with tempfile.TemporaryDirectory(prefix="q05-compute-headers-") as temporary:
        with tarfile.open(root/"source/cutlass-3.9.2-include.tar.gz") as archive:
            archive.extractall(temporary, filter="data")
        command_record(build/"compile", [compiler, *flags, "-I"+temporary+"/include",
                       root/"source/cfg_b_compute_chain.cu", "-o", build/"probe"], 180)
    command_record(build/"disassemble", [compiler.parent/"cuobjdump", "--dump-sass", build/"probe"], 60)
    save_json(build/"identity.json", dict(plan_sha256=digest(root/"plan.json"),
        probe_sha256=digest(build/"probe"), sass_sha256=digest(build/"disassemble/stdout"),
        flags=flags, compiler=str(compiler), static_instruction_review="待目标审核"))
    print(f"COMPILED {build/'probe'}；GPU执行0")


def validate(folder, row, case, seed):
    require(row["conditions"]["wait_groups"] == case["wait_retain"] and
            row["conditions"]["prologue_wait"] is False, "循环wait或首次提交不符")
    require(row["resources"]["occupancy_ctas_per_sm"] == 1, "驻留容量不符")
    # wait立即数已核对；其余原合同和全输出参考复用A的校验实现。
    base = deepcopy(row)
    base["conditions"]["wait_groups"] = 1
    validated = chain.validate(folder, base, case, seed)
    require(row["performance_eligible"] is True and validated["performance_eligible"], "性能资格未通过")
    return validated


def heldout_freeze(root, path):
    require(path is not None, "先冻结开发规则及两个32 Ktile预测")
    freeze = read_json(path)
    require(freeze["schema"] == "gh200-q05-compute-wait-freeze/1" and
        freeze["plan_sha256"] == digest(root/"plan.json") and
        freeze["build_identity_sha256"] == digest(root/"build/identity.json") and
        freeze["dev_score_sha256"] == digest(root/"score.json"), "开发冻结身份不符")
    require(freeze["criterion"] == dict(unit="clock64_cycle/CTA", max_abs_relative_error=.05,
        statistic="median_of_3_eligible_processes"), "原5%窗口门槛不符")
    require(set(freeze["predictions_cycle"]) == {"k32_wait0", "k32_wait1"} and
        all(math.isfinite(v) and v > 0 for v in freeze["predictions_cycle"].values()), "32预测不完整")
    return dict(sha256=digest(path), contents=freeze)


def score(root, split, freeze_path):
    plan = load_plan(root)
    freeze = heldout_freeze(root, freeze_path) if split == "heldout" else None
    rows = []
    selected = [case for case in plan["cases"] if case["split"] == split]
    for case in selected:
        for seed in SEEDS:
            folder = root/split/case["id"]/f"seed{seed}"
            raw = read_json(folder/"stdout")
            validated = validate(folder, raw, case, seed)
            rows.append(dict(case_id=case["id"], ktiles=case["ktiles"], wait_retain=case["wait_retain"],
                seed=seed, raw_sha256=digest(folder/"stdout"), **validated))
    require(len({row["device_uuid"] for row in rows}) == 1, "GPU身份变化")
    require(all(row["resources"] == rows[0]["resources"] for row in rows), "配对资源变化")
    aggregates = []
    for case in selected:
        records = [row for row in rows if row["case_id"] == case["id"]]
        aggregates.append(dict(case_id=case["id"], **{unit:dict(
            median=statistics.median(row["timing"][key] for row in records),
            minimum=min(row["timing"][key] for row in records),
            maximum=max(row["timing"][key] for row in records))
            for unit,key in (("cycles","elapsed_cycles"),("ns","elapsed_ns"))}))
    wait_responses = []
    for kt in sorted({case["ktiles"] for case in selected}):
        for seed in SEEDS:
            pair = {row["wait_retain"]:row for row in rows if row["ktiles"] == kt and row["seed"] == seed}
            wait_responses.append(dict(ktiles=kt, seed=seed, **{unit:dict(
                wait0_minus_wait1=pair[0]["timing"][key]-pair[1]["timing"][key],
                response_pct=100*(pair[0]["timing"][key]/pair[1]["timing"][key]-1))
                for unit,key in (("cycles","elapsed_cycles"),("ns","elapsed_ns"))}))
    length_responses = []
    if split == "dev":
        for retain in (0, 1):
            for seed in SEEDS:
                pair = {row["ktiles"]:row for row in rows if row["wait_retain"] == retain and row["seed"] == seed}
                length_responses.append(dict(wait_retain=retain, seed=seed, **{unit:dict(
                    k13_minus_k4=pair[13]["timing"][key]-pair[4]["timing"][key],
                    per_added_ktile=(pair[13]["timing"][key]-pair[4]["timing"][key])/9)
                    for unit,key in (("cycles","elapsed_cycles"),("ns","elapsed_ns"))}))
    prediction_checks = None
    if freeze is not None:
        prediction_checks = []
        for item in aggregates:
            predicted = freeze["contents"]["predictions_cycle"][item["case_id"]]
            error = abs(predicted/item["cycles"]["median"]-1)
            prediction_checks.append(dict(case_id=item["case_id"], predicted_cycles=predicted,
                abs_relative_error=error, passed=error <= .05))
    report = dict(schema="gh200-q05-compute-wait-score/1", split=split, rows=rows,
        aggregates=aggregates, wait_responses=wait_responses, length_responses=length_responses,
        all_performance_eligible=True, full_output_elements=len(rows)*16384,
        qualification="development_localization" if split == "dev" else "frozen_window_predictions",
        freeze=freeze, prediction_checks=prediction_checks, independent_C2=False,
        plan_sha256=digest(root/"plan.json"), build_identity_sha256=digest(root/"build/identity.json"))
    save_json(root/("score.json" if split == "dev" else "heldout-score.json"), report)
    print(f"SCORED {split}: {len(rows)}进程／{len(rows)*16384}完整输出元素")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "compile", "score"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            child.add_argument("--headers", type=Path, required=True)
        elif name == "compile":
            child.add_argument("--nvcc", default="nvcc")
        else:
            child.add_argument("--split", choices=("dev", "heldout"), default="dev")
            child.add_argument("--dev-freeze", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "prepare":
        prepare(root, args.headers.resolve())
    elif args.command == "compile":
        compile_probe(root, args.nvcc)
    else:
        score(root, args.split, args.dev_freeze)


if __name__ == "__main__":
    main()
