#!/usr/bin/env python3
"""cfg_b 驻留计算短链的有限准备、构建与采样入口。"""
import argparse
import json
import math
import random
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

from l2_saturation import (command_record, digest, exclusive_lock, inspect_allocated_gpu,
                           read_json, require, save_json)

HERE = Path(__file__).resolve().parent
SCHEMA = "gh200-cfg-b-compute-chain-plan/1"
SEEDS = (3, 22, 41)


def cases():
    return [dict(id=f"k{kt}_{'trace' if trace else 'plain'}", ktiles=kt, trace=trace,
                 split="heldout" if kt == 32 else "dev", wgmma_count=8*kt,
                 work_flop=2*128*128*64*kt)
            for kt in (4, 13, 32) for trace in (False, True)]


def output_coordinates():
    """独立枚举每线程两个M64片段的64个FP32输出。"""
    thread, m64, reg = np.indices((128, 2, 64))
    row = 64*m64 + 16*(thread//32) + (thread % 32)//4 + 8*((reg//2) % 2)
    col = 2*(thread % 4) + reg % 2 + 8*(reg//4)
    return row.ravel(), col.ravel()


def reference(ktiles, seed):
    """整数矩阵乘与槽使用次数求和；最后一次除1024，避开设备的片段循环。"""
    rows = np.arange(128, dtype=np.int64)[:, None]
    inner = np.arange(64, dtype=np.int64)[None, :]
    numerator = np.zeros((128, 128), dtype=np.int64)
    for slot in range(6):
        count = (ktiles + 5 - slot)//6
        a = (7*rows + 13*inner + 3*slot + 3*(seed % 17)) % 17 - 8
        b = (11*rows + 5*inner + 7*slot + 5*(seed % 17)) % 17 - 8
        numerator += count*(a @ b.T)
    row, col = output_coordinates()
    return (numerator[row, col]/1024).astype(np.float32)


def verify_output(path, ktiles, seed):
    require(path.stat().st_size == 16384*4, "完整累加器输出长度不符")
    actual = np.fromfile(path, dtype="<f4")
    expected = reference(ktiles, seed)
    require(np.isfinite(actual).all() and np.array_equal(actual, expected),
            "独立整数参考与完整累加器输出不符")
    return dict(checked_elements=len(actual), sha256=digest(path), max_abs_error=0.)


def resident_hypothesis(ktiles):
    """复用当前引擎：零供给成本、一个输出、1 GHz数值坐标转成cycle。"""
    import supply_model
    result = supply_model.shared_stage_events(ktiles, 6, 1, service_ns=0.,
        read_latency_ns=0., compute_ns=512., commit_lead_ns=256., output_counts=[1])
    actor = result["ctas"][0]
    return dict(qualification="工作假设，未由实际短链辨识", unit="cycle",
                compute_cycles=512., commit_lead_cycles=256.,
                cycle_timeline={key: actor[key] for key in
                    ("starts", "commits", "completed", "wait1_return", "released",
                     "mma_returns", "main_ends")})


def prepare(root, cutlass):
    version = (cutlass/"include/cutlass/version.h").read_text()
    import re
    values = [int(re.search(r"#define\s+CUTLASS_"+key+r"\s+(\d+)", version)[1])
              for key in ("MAJOR", "MINOR", "PATCH")]
    require(values == [3, 9, 2], "本探针固定CUTLASS 3.9.2")
    root.mkdir(parents=True, exist_ok=False)
    source = root/"source"
    source.mkdir()
    for name in ("cfg_b_compute_chain.py", "l2_saturation.py", "supply_model.py"):
        shutil.copy2(HERE/name, source/name)
    for name in ("cfg_b_compute_chain.cu", "r00_common.hpp"):
        shutil.copy2(HERE/"probes"/name, source/name)
    # 只保存头文件压缩包；构建时临时解压，避免新增长期可索引的build树。
    with tarfile.open(source/"cutlass-3.9.2-include.tar.gz", "w:gz") as archive:
        archive.add(cutlass/"include", arcname="include")
    save_json(root/"plan.json", dict(schema=SCHEMA, cases=cases(), seeds=list(SEEDS),
        threads=128, slots=6, dynamic_smem_bytes=196608, warmup_limit=60,
        dev_processes=12, heldout_processes=6, GPU_execution=False,
        source_sha256={p.name: digest(p) for p in source.iterdir()}))
    save_json(root/"resident-hypothesis.json", dict(engine_sha256=digest(source/"supply_model.py"),
        input="全部驻留输入已就绪；零供给/控制/输出成本；不是完整探针窗口",
        conditions={str(kt): resident_hypothesis(kt) for kt in (4, 13, 32)}))
    print(f"PREPARED {root}: 3长度×plain/trace；开发12、留出6进程；未运行GPU")


def load_plan(root):
    plan = read_json(root/"plan.json")
    require(plan["schema"] == SCHEMA and plan["cases"] == cases(), "条件清单不符")
    for name, checksum in plan["source_sha256"].items():
        require(digest(root/"source"/name) == checksum, f"准备包源码变更：{name}")
    require(digest(Path(__file__)) == plan["source_sha256"]["cfg_b_compute_chain.py"],
            "请执行准备包source/cfg_b_compute_chain.py")
    return plan


def compile_probe(root, nvcc):
    load_plan(root)
    compiler = shutil.which(nvcc)
    require(compiler is not None, "nvcc未找到")
    build = root/"build"
    build.mkdir(exist_ok=False)
    command_record(build/"version", [compiler, "--version"], 30)
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-lineinfo",
             "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"]
    with tempfile.TemporaryDirectory(prefix="cfg-b-compute-headers-") as temporary:
        with tarfile.open(root/"source/cutlass-3.9.2-include.tar.gz") as archive:
            archive.extractall(temporary, filter="data")
        command_record(build/"compile", [compiler, *flags, "-I"+temporary+"/include",
                       root/"source/cfg_b_compute_chain.cu", "-o", build/"probe"], 180)
    disassembler = Path(compiler).resolve().parent/"cuobjdump"
    command_record(build/"disassemble", [disassembler, "--dump-sass", build/"probe"], 60)
    save_json(build/"identity.json", dict(plan_sha256=digest(root/"plan.json"),
        probe_sha256=digest(build/"probe"), sass_sha256=digest(build/"disassemble/stdout"),
        flags=flags, compiler=str(Path(compiler).resolve()), static_instruction_review="待核对"))
    print(f"COMPILED {build/'probe'}；未运行GPU")


def validate(folder, row, case, seed):
    require(row.get("schema") == "gh200-cfg-b-compute-chain/1" and
            row.get("status") == "measured", "probe状态不符")
    require(all(row.get(key) == case[key] for key in
                ("ktiles", "trace", "wgmma_count", "work_flop")) and
            row.get("seed") == seed and row.get("samples") == 1, "测量条件不符")
    device, resources = row["device"], row["resources"]
    require("GH200" in device["name"] and device["sms"] == 132 and
            device["major"] == 9 and device["minor"] == 0, "目标设备不符")
    require(resources["threads"] == 128 and resources["grid_ctas"] == 1 and
            resources["dynamic_smem_bytes"] == 196608 and resources["local_size_bytes"] == 0,
            "计算资源或spill不符")
    conditions = dict(slots=6, software_tile=[128,128,64], instruction_shape=[64,128,16],
        a_major="K", b_major="MN", smem_swizzle_bytes=128, commit_per_ktile=1,
        wait_groups=1, tail_wait_groups=0)
    require(all(row["conditions"].get(key) == value for key, value in conditions.items()),
            "布局或group组织不符")
    timing = row["timing"]
    for unit in ("cycle", "ns"):
        begin, end = timing[f"begin_{unit}"], timing[f"end_{unit}"]
        require(type(begin) is int and type(end) is int and end > begin, "原始计时无效")
        elapsed_key = "elapsed_cycles" if unit == "cycle" else "elapsed_ns"
        require(timing[elapsed_key] == end-begin, "原始计时不能复算")
    require(timing["smid_start"] == timing["smid_end"], "CTA的SM身份变化")
    trace = row["trace_detail"]
    if case["trace"]:
        require(trace["k_tile_index"] == case["ktiles"]-1, "trace位置不符")
        names = ("issue_before", "commit_after", "wait1_after", "tail_wait0_after")
        for unit in ("cycle", "ns"):
            points = [timing[f"begin_{unit}"]] + [trace[n][unit] for n in names] + [timing[f"end_{unit}"]]
            require(all(type(v) is int for v in points) and points == sorted(points),
                    "trace不满足发出/提交/等待/排空的程序顺序")
    else:
        require(trace is None, "plain包含trace")
    require(row["output_file"] == "output.f32" and row["output_elements"] == 16384 and
            row["output_bytes"] == 65536, "输出见证不符")
    witness = verify_output(folder/"output.f32", case["ktiles"], seed)
    require(row["correctness"]["checked_elements"] == 16384 and
            row["correctness"]["errors"] == 0 and row["correctness"]["max_abs_error"] == 0,
            "host完整参考失败")
    require([s["ktiles"] for s in row["short_checks"]] == [1,2] and
            all(s["correctness"]["errors"] == 0 and s["correctness"]["checked_elements"] == 16384
                for s in row["short_checks"]),
            "短正确性检查不完整")
    warmup = row["warmup"]
    windows = warmup["elapsed_ns"]
    require(warmup["limit"] == 60 and 8 <= len(windows) <= 60 and
            len(warmup["elapsed_cycles"]) == len(windows) and min(windows) > 0, "暖机记录不符")
    tail = windows[-5:]
    mean = sum(tail)/5
    cv = math.sqrt(sum((v-mean)**2 for v in tail)/4)/mean
    require(warmup["converged"] == (cv <= .02) and
            math.isclose(cv, warmup["cv"], rel_tol=1e-7, abs_tol=1e-12), "暖机收敛无法复算")
    return dict(device_uuid=device["uuid"], timing=timing, trace_detail=trace,
                resources=resources, output_witness=witness, warmup_cv=cv,
                performance_eligible=warmup["converged"], work_flop=case["work_flop"],
                flop_per_window_cycle=case["work_flop"]/timing["elapsed_cycles"])


def verify_freeze(root, freeze_path):
    require(freeze_path is not None, "先冻结开发解释及32 Ktile预测，再采留出")
    freeze = read_json(freeze_path)
    require(freeze["schema"] == "gh200-cfg-b-compute-chain-freeze/1" and
            freeze["plan_sha256"] == digest(root/"plan.json") and
            freeze["build_identity_sha256"] == digest(root/"build/identity.json") and
            freeze["dev_summary_sha256"] == digest(root/"dev/summary.json"), "开发冻结身份不符")
    require(read_json(root/"dev/summary.json")["all_performance_eligible"] is True,
            "开发数据尚未齐备")
    require(freeze["criterion"] == dict(unit="clock64_cycle/CTA",
        max_abs_relative_error=.05, statistic="median_of_3_eligible_processes"),
        "留出判定与本轮协议不符")
    parameters, predictions = freeze["candidate_parameters"], freeze["predictions_cycle"]
    require(parameters and set(parameters) <= {"nominal", "conditional"} and
            set(parameters) == set(predictions), "冻结候选与预测不符")
    for name, values in parameters.items():
        a, q = values["a_cycle"], values["q_cycle"]
        require(math.isfinite(a) and a >= 0 and math.isfinite(q) and q > 0 and
                (name != "nominal" or q == 512), "窗口解释参数不符")
        require(math.isclose(predictions[name]["k32_plain"], a+32*q,
                            rel_tol=1e-12, abs_tol=1e-9), "缺失或错误的32 Ktile预测")
    return dict(sha256=digest(freeze_path), contents=freeze)


def bind_freeze(directory, freeze):
    path = directory/"dev-freeze.json"
    if path.exists():
        require(read_json(path) == freeze, "留出已开始，开发冻结身份发生变化")
    else:
        save_json(path, freeze)


def run(root, split, freeze_path):
    plan = load_plan(root)
    identity = read_json(root/"build/identity.json")
    require(identity["plan_sha256"] == digest(root/"plan.json") and
            identity["probe_sha256"] == digest(root/"build/probe") and
            identity["sass_sha256"] == digest(root/"build/disassemble/stdout"), "构建身份变更")
    freeze = None
    if split == "heldout":
        freeze = verify_freeze(root, freeze_path)
    directory = root/split
    directory.mkdir(exist_ok=True)
    with exclusive_lock(root/".run.lock"):
        if freeze is not None:
            bind_freeze(directory, freeze)
        environment = inspect_allocated_gpu(root/"runs"/f"{split}-{time.time_ns()}")
        uuid = environment["uuid"]
        binding = root/"device-binding.json"
        if binding.exists():
            require(read_json(binding)["uuid"] == uuid, "准备包已有采样GPU身份不同")
        else:
            save_json(binding, dict(uuid=uuid))
        results = []
        with exclusive_lock(Path("/tmp")/f"gh200-measurement-{uuid}.lock"):
            for process, seed in enumerate(SEEDS):
                ordered = [case for case in plan["cases"] if case["split"] == split]
                random.Random(20261017+process).shuffle(ordered)
                for case in ordered:
                    folder = directory/case["id"]/f"process_{process:02}"
                    argv = list(map(str, [root/"build/probe", "--ktiles", case["ktiles"],
                        "--seed", seed, "--trace", int(case["trace"]), "--warmup-limit", 60]))
                    if folder.exists():
                        receipt = read_json(folder/"receipt.json")
                        require(receipt["argv"] == argv and receipt["returncode"] == 0 and
                                receipt["stdout_sha256"] == digest(folder/"stdout") and
                                receipt["stderr_sha256"] == digest(folder/"stderr"), "旧采样身份变更")
                        result = validate(folder, json.loads((folder/"stdout").read_text()), case, seed)
                        require(read_json(folder/"validated.json") == result, "旧采样回放不一致")
                    else:
                        command_record(folder, argv, 180)
                        result = validate(folder, json.loads((folder/"stdout").read_text()), case, seed)
                        save_json(folder/"validated.json", result)
                    require(result["device_uuid"] == uuid, "probe与分配GPU身份不同")
                    results.append(dict(case_id=case["id"], seed=seed,
                        path=str(folder.relative_to(root)), **result))
                    print(f"VALID {case['id']} seed={seed}", flush=True)
        summary = dict(schema="gh200-cfg-b-compute-chain-summary/1", complete=True, split=split,
            plan_sha256=digest(root/"plan.json"), build_identity_sha256=digest(root/"build/identity.json"),
            device_uuid=uuid, freeze=freeze, results=results,
            all_performance_eligible=all(r["performance_eligible"] for r in results),
            model_fit_performed=False)
        path = directory/"summary.json"
        if path.exists():
            require(read_json(path) == summary, "已有汇总与回放不一致")
        else:
            save_json(path, summary)
        print(f"SAVED {path}: {len(results)}进程")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "compile", "run"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            child.add_argument("--cutlass-root", type=Path, required=True)
        elif name == "compile":
            child.add_argument("--nvcc", default="nvcc")
        else:
            child.add_argument("--split", choices=("dev", "heldout"), required=True)
            child.add_argument("--dev-freeze", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "prepare":
        prepare(root, args.cutlass_root.resolve())
    elif args.command == "compile":
        compile_probe(root, args.nvcc)
    else:
        run(root, args.split, args.dev_freeze)


if __name__ == "__main__":
    main()
