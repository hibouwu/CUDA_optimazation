#!/usr/bin/env python3
"""单 CTA L1/LSU 指针加载的固定条件准备、编译与显式采样。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import time

from l2_saturation import (command_record, digest, exclusive_lock, inspect_allocated_gpu,
                           read_json, require, save_json)

HERE = Path(__file__).resolve().parent
PROBE = HERE / "probes/l1_lsu.cu"
if not PROBE.is_file():
    PROBE = HERE / "l1_lsu.cu"  # prepare 包内保存为平铺源码；仍可独立复跑 CPU 检查。
THREADS = 128
ROUNDS = 256
SEEDS = (3, 22, 41)
SCHEMA = "gh200-l1-lsu-plan/1"
PROCESS_TIMEOUT = 180


def work_oracle(working_set_bytes, chains, rounds=ROUNDS):
    require(working_set_bytes in (16*1024, 64*1024, 128*1024, 256*1024), "工作集超出固定范围")
    require(chains in (1, 4) and type(chains) is int, "J 必须为 1 或 4")
    require(type(rounds) is int and 1 <= rounds <= ROUNDS, "遍历轮数超出范围")
    batches = working_set_bytes // (16*THREADS)
    return dict(vector_nodes=working_set_bytes//16, nodes_per_thread=batches,
                nodes_per_chain=batches//chains, groups_per_sweep=batches//chains,
                loads_per_thread=batches*rounds, logical_read_bytes=working_set_bytes*rounds,
                warm_logical_read_bytes=working_set_bytes)


def checksum_oracle(working_set_bytes, seed, thread, rounds):
    """按线程的完整节点集合求闭式和；不模拟设备指针追踪或 J 批处理。"""
    work = work_oracle(working_set_bytes, 1, rounds)
    require(type(thread) is int and 0 <= thread < THREADS, "线程索引超出范围")
    require(type(seed) is int and 0 <= seed <= 0xFFFFFFFF, "seed 超出 uint32")
    count = work["nodes_per_thread"]
    index_sum = THREADS*count*(count-1)//2 + thread*count
    # next_index 的集合与原节点索引集合相同；payload 是 word 1/2/3。
    return ((index_sum + 17*(12*index_sum + 6*count) + 3*count*seed)*rounds) & 0xFFFFFFFF


def make_cases():
    result = []
    for kib in (16, 256):
        for policy in ("ca", "cg"):
            for chains in (1, 4):
                result.append(dict(working_set_kib=kib, requested_carveout_percent=100,
                                   policy=policy, chains=chains, split="dev"))
    for carveout in (0, 100):
        for policy in ("ca", "cg"):
            result.append(dict(working_set_kib=64, requested_carveout_percent=carveout,
                               policy=policy, chains=4, split="dev"))
    for carveout in (0, 100):
        for policy in ("ca", "cg"):
            for chains in (1, 4):
                result.append(dict(working_set_kib=128, requested_carveout_percent=carveout,
                                   policy=policy, chains=chains, split="heldout"))
    for case in result:
        kib, hint, policy, chains = (case[key] for key in
                                    ("working_set_kib", "requested_carveout_percent", "policy", "chains"))
        case.update(id=f"w{kib}_h{hint}_{policy}_j{chains}", working_set_bytes=kib*1024,
                    rounds=ROUNDS, seeds=list(SEEDS), work=work_oracle(kib*1024, chains))
    return result


def prepare(root):
    """只冻结文本、条件和身份；不查询 CUDA、设备或编译器。"""
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    shutil.copy2(PROBE, source / "l1_lsu.cu")
    for name in ("l1_lsu_probe.py", "test_l1_lsu_probe.py", "l2_saturation.py"):
        shutil.copy2(HERE/name, source/name)
    save_json(root / "plan.json", dict(schema=SCHEMA, cases=make_cases(), threads=THREADS,
        rounds=ROUNDS, independent_processes_per_case=len(SEEDS), dev_processes=36,
        heldout_processes=24, source_sha256={p.name: digest(p) for p in source.iterdir()},
        warmup="每次 kernel 内完整遍历本 W 一次；无跨 kernel L1 暖态主张",
        timing_boundary="内窗含起始 CTA barrier、指针加载、checksum、shared drain、结束 CTA barrier；event 含暖读与诊断",
        actual_carveout=None, cache_residency_proven=False, physical_cache_bytes_proven=False,
        state="prepared", GPU_execution=False))
    print(f"PREPARED {root}: 12 开发/36 进程，8 留出/24 进程；未运行 GPU")


def load_plan(root):
    plan = read_json(root / "plan.json")
    require(plan.get("schema") == SCHEMA and plan.get("cases") == make_cases(), "固定条件矩阵不符")
    for name, checksum in plan["source_sha256"].items():
        require(digest(root / "source" / name) == checksum, f"准备包源码变更：{name}")
    require(digest(Path(__file__)) == plan["source_sha256"]["l1_lsu_probe.py"], "请执行准备包的 source/l1_lsu_probe.py")
    return plan


def compile_probe(root, nvcc, extra_flags=()):
    load_plan(root)
    compiler = shutil.which(nvcc)
    require(compiler is not None, "nvcc 未找到")
    build = root / "build"
    build.mkdir(exist_ok=False)
    command_record(build/"version", [compiler, "--version"], 30)
    flags = ["-std=c++17", "-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v", *extra_flags]
    command_record(build/"compile", [compiler, *flags, root/"source/l1_lsu.cu", "-o", build/"probe"], 180)
    command_record(build/"disassemble", [Path(compiler).resolve().parent/"cuobjdump", "--dump-sass", build/"probe"], 60)
    save_json(build/"identity.json", dict(plan_sha256=digest(root/"plan.json"),
        probe_sha256=digest(build/"probe"), source_sha256=digest(root/"source/l1_lsu.cu"),
        sass_sha256=digest(build/"disassemble/stdout"), compiler=str(Path(compiler).resolve()), flags=flags,
        static_instruction_review="待结合 SASS 核对 .ca/.cg 与 J4 实际发出组织"))
    print(f"COMPILED {build/'probe'}；未运行 GPU")


def verify_build(root):
    identity = read_json(root/"build/identity.json")
    for key, path in (("plan_sha256", root/"plan.json"), ("probe_sha256", root/"build/probe"),
                      ("source_sha256", root/"source/l1_lsu.cu"), ("sass_sha256", root/"build/disassemble/stdout")):
        require(identity[key] == digest(path), f"构建身份变更：{key}")
    return identity


def validate(row, case, seed):
    require(row.get("schema") == "gh200-l1-lsu/1" and row.get("status") == "measured", "probe 状态不符")
    for key in ("working_set_bytes", "chains", "policy", "requested_carveout_percent", "rounds"):
        require(row.get(key) == case[key], f"测量条件不符：{key}")
    require(row.get("seed") == seed and row.get("actual_carveout", "missing") is None, "seed 或 carveout 证据边界不符")
    require(row.get("logical_read_bytes") == case["work"]["logical_read_bytes"], "正式逻辑请求量不符")
    device, resources = row["device"], row["resources"]
    require("GH200" in device["name"] and device["sms"] == 132 and device["major"] == 9 and device["minor"] == 0,
            "目标设备不符")
    require(str(device["uuid"]).startswith("GPU-"), "设备 UUID 缺失")
    require(resources["grid_ctas"] == 1 and resources["threads"] == THREADS
            and resources["dynamic_smem_bytes"] == 0 and resources["local_size_bytes"] == 0,
            "单 CTA / 无动态 SMEM / 无 local memory 条件不符")
    trials = row["trials"]
    require([trial["rounds"] for trial in trials] == [1, 2, ROUNDS], "短校验及正式窗口不齐")
    for trial in trials:
        rounds = trial["rounds"]
        require(trial["logical_read_bytes"] == case["working_set_bytes"]*rounds, "窗口逻辑请求量不符")
        require(trial["correctness"] == dict(checked_elements=2*THREADS, errors=0), "设备 checksum 资格失败")
        for name, multiplier in (("warm_checksums", 1), ("timed_checksums", rounds)):
            actual = trial[name]
            expected = [checksum_oracle(case["working_set_bytes"], seed, thread, multiplier) for thread in range(THREADS)]
            require(actual == expected, f"{name} 与每线程独立参考不符")
        require(all(type(trial[key]) is int for key in
                    ("begin_cycle", "end_cycle", "begin_ns", "end_ns", "smid_start", "smid_end")), "原始计时须为整数")
        require(trial["end_cycle"] > trial["begin_cycle"] and trial["end_ns"] > trial["begin_ns"], "原始窗口非正")
        require(trial["smid_start"] == trial["smid_end"], "CTA 的 SMID 前后改变")
        require(math.isfinite(trial["event_ms"]) and trial["event_ms"] > 0, "完整 event 窗口无效")
    formal = trials[-1]
    return dict(device_uuid=device["uuid"], smid=formal["smid_start"],
        elapsed_cycle=formal["end_cycle"]-formal["begin_cycle"], elapsed_ns=formal["end_ns"]-formal["begin_ns"],
        event_ms=formal["event_ms"], logical_read_bytes=case["work"]["logical_read_bytes"],
        independently_checked_elements=3*2*THREADS, actual_carveout=None,
        cache_residency_proven=False, physical_cache_bytes_proven=False)


def frozen_predictions(root, path):
    """只绑定调用方冻结预测的原文与 SHA；不为其虚构数值或新模型 schema。"""
    require(path is not None, "heldout 需要 --frozen-predictions")
    value = read_json(path)
    require(isinstance(value, dict) and bool(value), "冻结预测须为非空 JSON 对象")
    summary_path = root/"dev/summary.json"
    development = read_json(summary_path)
    require(development.get("complete") is True and development.get("independent_processes") == 36
            and development.get("plan_sha256") == digest(root/"plan.json")
            and development.get("build_identity_sha256") == digest(root/"build/identity.json"), "开发采样尚未完整或身份不符")
    return dict(path=str(path), sha256=digest(path), development_summary_sha256=digest(summary_path))


def run_scan(root, split, allow_gpu, predictions):
    require(allow_gpu, "run 需要显式 --allow-gpu；prepare/compile 不调用 CUDA")
    plan = load_plan(root)
    verify_build(root)
    freeze = frozen_predictions(root, predictions) if split == "heldout" else None
    directory = root/split
    directory.mkdir(exist_ok=False)
    # 先写冻结身份，再查分配或执行探针；中断后保留目录，本入口不隐式复用旧进程。
    save_json(directory/"binding.json", dict(plan_sha256=digest(root/"plan.json"), frozen_predictions=freeze))
    with exclusive_lock(root/".run.lock"):
        allocated = inspect_allocated_gpu(directory/"allocation")
        binding = root/"device-binding.json"
        expected_device = dict(uuid=allocated["uuid"], name=allocated["name"])
        if binding.exists():
            require(read_json(binding) == expected_device, "开发与留出设备不同")
        else:
            save_json(binding, expected_device)
        results = []
        with exclusive_lock(Path("/tmp")/f"gh200-measurement-{allocated['uuid']}.lock"):
            started = time.monotonic()
            for process_index, seed in enumerate(SEEDS):
                cases = [case for case in plan["cases"] if case["split"] == split]
                random.Random(20261017+process_index).shuffle(cases)
                for case in cases:
                    require(time.monotonic()-started < 600, "本次 split 的 10 分钟采样预算耗尽；已保存完成进程")
                    folder = directory/case["id"]/f"process_{process_index:02}"
                    argv = [root/"build/probe", "--working-set-kib", case["working_set_kib"],
                            "--chains", case["chains"], "--policy", case["policy"],
                            "--carveout", case["requested_carveout_percent"], "--seed", seed]
                    command_record(folder, argv, PROCESS_TIMEOUT)
                    result = validate(json.loads((folder/"stdout").read_text()), case, seed)
                    require(result["device_uuid"] == allocated["uuid"], "probe 与分配设备不同")
                    save_json(folder/"validated.json", result)
                    results.append(dict(case_id=case["id"], seed=seed, path=str(folder.relative_to(root)),
                                        validated_sha256=digest(folder/"validated.json"), **result))
                    print(f"VALID {case['id']} seed={seed}", flush=True)
        save_json(directory/"summary.json", dict(schema="gh200-l1-lsu-summary/1", split=split, complete=True,
            plan_sha256=digest(root/"plan.json"), build_identity_sha256=digest(root/"build/identity.json"),
            device_uuid=allocated["uuid"], frozen_predictions=freeze, independent_processes=len(results),
            results=results, model_fit_performed=False, actual_carveout=None,
            cache_residency_proven=False, physical_cache_bytes_proven=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)
    for name in ("prepare", "compile", "run"):
        child = steps.add_parser(name)
        child.add_argument("--output", required=True, type=Path)
        if name == "compile":
            child.add_argument("--nvcc", default="nvcc")
            child.add_argument("--nvcc-flag", action="append", default=[], help="显式保存的本地编译补充参数；可重复")
        if name == "run":
            child.add_argument("--split", choices=("dev", "heldout"), default="dev")
            child.add_argument("--allow-gpu", action="store_true")
            child.add_argument("--frozen-predictions", type=Path, help="留出前人工冻结的非空预测 JSON；保存原文 SHA")
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.step == "prepare": prepare(root)
    elif args.step == "compile": compile_probe(root, args.nvcc, args.nvcc_flag)
    else: run_scan(root, args.split, args.allow_gpu, args.frozen_predictions.resolve() if args.frozen_predictions else None)


if __name__ == "__main__":
    main()
