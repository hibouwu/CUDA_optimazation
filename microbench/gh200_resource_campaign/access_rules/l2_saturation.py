#!/usr/bin/env python3
"""准备、编译和运行固定范围的 GH200 L2 请求服务扫描。"""
from __future__ import annotations

import argparse
import array
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PROBE = HERE / "probes/l2_saturation.cu"
ARRAY_BYTES = 17_301_504
THREADS = 128
ITERATIONS = 16
SEEDS = (3, 22, 41)
WARMUP_LIMIT = 60
PROCESS_TIMEOUT = 180
PLAN_SCHEMA = "gh200-l2-saturation-plan/1"
FREEZE_SCHEMA = "gh200-l2-saturation-dev-freeze/1"


def digest(path):
    with Path(path).open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    """证据使用独占创建；失败目录和旧结果保留。"""
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def work_oracle(active_ctas, chains, mode, array_bytes=ARRAY_BYTES, iterations=ITERATIONS):
    """每轮遍历固定数组一次；chains 只改变批量，不改变请求量。"""
    require(type(active_ctas) is int and 1 <= active_ctas <= 132, "active_ctas 超出范围")
    require(chains in (1, 4, 8), "chains 必须为 1、4 或 8")
    require(mode in ("read", "write", "mixed"), "未知读写模式")
    require(type(iterations) is int and 1 <= iterations <= ITERATIONS, "迭代次数超出范围")
    require(type(array_bytes) is int and array_bytes > 0 and array_bytes % 16 == 0,
            "数组须为完整 16 B 向量")
    vectors = array_bytes // 16
    total_threads = active_ctas * THREADS
    require(vectors % (total_threads * chains) == 0, "数组不能整除参与线程及请求批次")
    read = array_bytes * iterations if mode in ("read", "mixed") else 0
    write = array_bytes * iterations if mode in ("write", "mixed") else 0
    return dict(total_threads=total_threads, vectors=vectors,
                vectors_per_thread=vectors // total_threads,
                batches_per_thread=vectors // (total_threads * chains),
                read_payload_bytes=read, write_payload_bytes=write, work_count=read + write,
                aggregate_array_bytes=array_bytes * (2 if mode == "mixed" else 1))


def read_checksum_oracle(array_bytes, active_ctas, iterations, seed, thread):
    """对每线程拥有的算术序列求和，独立于设备的批处理循环。"""
    work = work_oracle(active_ctas, 1, "read", array_bytes, iterations)
    total_threads = work["total_threads"]
    require(0 <= thread < total_threads, "checksum 线程索引超出范围")
    groups = work["vectors_per_thread"]
    # 一个 uint4 的 word 索引之和为 16 * vector_index + 6。
    index_sum = 8 * total_threads * groups * (groups - 1) + (16 * thread + 6) * groups
    return ((17 * index_sum + 4 * groups * seed) * iterations) & 0xFFFFFFFF


def make_cases():
    cases = []
    for active in (1, 8, 32, 132):
        for chains in (1, 4, 8):
            cases.append(dict(id=f"read_a{active}_j{chains}", split="dev",
                              active_ctas=active, chains=chains, mode="read"))
    for mode in ("write", "mixed"):
        for active in (32, 132):
            cases.append(dict(id=f"{mode}_a{active}_j4", split="dev",
                              active_ctas=active, chains=4, mode=mode))
    for active in (16, 66):
        for chains in (4, 8):
            cases.append(dict(id=f"read_a{active}_j{chains}", split="heldout",
                              active_ctas=active, chains=chains, mode="read"))
    for case in cases:
        case.update(array_bytes=ARRAY_BYTES, iterations=ITERATIONS, samples=1,
                    warmup_limit=WARMUP_LIMIT, seeds=list(SEEDS))
        case["work"] = work_oracle(case["active_ctas"], case["chains"], case["mode"])
    return cases


def prepare(root):
    require(PROBE.is_file(), "探针源码尚未完成")
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    shutil.copy2(PROBE, source / "l2_saturation.cu")
    shutil.copy2(Path(__file__), source / "l2_saturation.py")
    shutil.copy2(HERE / "l2_service.py", source / "l2_service.py")
    cases = make_cases()
    plan = dict(schema=PLAN_SCHEMA, device="GH200 CC9.0, 132 SM",
                threads=THREADS, array_bytes=ARRAY_BYTES, iterations=ITERATIONS,
                independent_processes_per_case=len(SEEDS), warmup_limit=WARMUP_LIMIT,
                process_timeout_seconds=PROCESS_TIMEOUT, cases=cases,
                dev_processes=48, heldout_processes=12,
                source_sha256={p.name: digest(p) for p in source.iterdir()},
                cache_residency_proven=False, physical_hbm_bytes_proven=False,
                state="prepared", GPU_execution=False)
    save_json(root / "plan.json", plan)
    print(f"PREPARED {root}: 16 开发条件／48 进程，4 留出条件／12 进程；未运行 GPU")


def load_plan(root):
    plan = read_json(root / "plan.json")
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("cases") == make_cases(),
            "条件清单与本轮固定矩阵不符")
    for name, expected in plan["source_sha256"].items():
        require(digest(root / "source" / name) == expected, f"准备包源码变更：{name}")
    require(digest(Path(__file__)) == plan["source_sha256"]["l2_saturation.py"],
            "runner 与准备包身份不符，请执行准备包的 source/l2_saturation.py")
    return plan


def command_record(folder, argv, timeout):
    """独占目录记录单次命令；超时终止整个子进程组并保留原输出。"""
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "command.json", dict(argv=list(map(str, argv)),
                                            cwd=str(folder), timeout_seconds=timeout))
    start = time.time_ns()
    timed_out = False
    process = None
    code = None
    failure = None
    with (folder / "stdout").open("xb") as output, (folder / "stderr").open("xb") as errors:
        try:
            process = subprocess.Popen(list(map(str, argv)), cwd=folder, stdout=output,
                                       stderr=errors, start_new_session=True)
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
        except BaseException as error:
            failure = error
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    receipt = dict(argv=list(map(str, argv)), pid=process.pid if process is not None else None,
                   returncode=code, failure=str(failure) if failure is not None else None,
                   timed_out=timed_out, start_ns=start, stop_ns=time.time_ns(),
                   stdout_sha256=digest(folder / "stdout"),
                   stderr_sha256=digest(folder / "stderr"))
    save_json(folder / "receipt.json", receipt)
    if failure is not None:
        raise failure
    require(code == 0 and not timed_out, f"命令失败，证据保留：{folder}")
    return receipt


def compile_probe(root, nvcc):
    load_plan(root)
    compiler = shutil.which(nvcc)
    require(compiler is not None, f"未找到 nvcc：{nvcc}")
    build = root / "build"
    build.mkdir(exist_ok=False)
    command_record(build / "version", [compiler, "--version"], 30)
    executable = build / "probe"
    flags = ["-std=c++17", "-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a",
             "-Xptxas=-v"]
    command_record(build / "compile", [compiler, *flags, root / "source/l2_saturation.cu",
                                       "-o", executable], PROCESS_TIMEOUT)
    disassembler = Path(compiler).resolve().parent / "cuobjdump"
    require(disassembler.is_file(), "编译工具链缺少 cuobjdump")
    command_record(build / "disassemble", [disassembler, "--dump-sass", executable], 60)
    save_json(build / "identity.json", dict(plan_sha256=digest(root / "plan.json"),
              probe_sha256=digest(executable), source_sha256=digest(root / "source/l2_saturation.cu"),
              sass_sha256=digest(build / "disassemble/stdout"), compiler=str(Path(compiler).resolve()),
              flags=flags, static_instruction_review="待结合 SASS 单独核对"))
    print(f"COMPILED {executable}；未运行 GPU")


def verify_build(root):
    identity = read_json(root / "build/identity.json")
    require(identity["plan_sha256"] == digest(root / "plan.json"), "编译的条件清单变更")
    for key, path in (("probe_sha256", root / "build/probe"),
                      ("source_sha256", root / "source/l2_saturation.cu"),
                      ("sass_sha256", root / "build/disassemble/stdout")):
        require(identity[key] == digest(path), f"编译身份变更：{key}")
    return identity


def verify_write_witness(folder, row, seed):
    path = folder / "output.u32"
    require(path.is_file() and path.stat().st_size == row["array_bytes"], "完整写出见证缺失或长度不符")
    index = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            values = array.array("I")
            values.frombytes(chunk)
            require(values.itemsize == 4, "host uint32 宽度不符")
            if sys.byteorder != "little":
                values.byteswap()
            for value in values:
                expected = (29 * index + seed) & 0xFFFFFFFF
                require(value == expected, f"写出见证错误，word={index}")
                index += 1
    return dict(path="output.u32", bytes=path.stat().st_size, sha256=digest(path),
                independently_checked_words=index)


def validate_probe(folder, row, case, seed):
    """原始整数计时、工作量、完整写出与每线程读 checksum 独立复算。"""
    require(row.get("schema") == "gh200-l2-saturation/1", "探针 schema 不符")
    require(row.get("status") == "measured" and row.get("cache_residency_proven") is False
            and row.get("physical_hbm_bytes_proven") is False, "探针状态或证据资格不符")
    expected = {name: case[name] for name in
                ("active_ctas", "chains", "mode", "array_bytes", "iterations", "samples")}
    expected["seed"] = seed
    require(all(row.get(name) == value for name, value in expected.items()), "探针条件与冻结清单不符")
    work = work_oracle(case["active_ctas"], case["chains"], case["mode"],
                       case["array_bytes"], case["iterations"])
    for name in ("read_payload_bytes", "write_payload_bytes", "work_count"):
        require(type(row.get(name)) is int and row[name] == work[name], f"请求量不符：{name}")
    require(row["aggregate_array_bytes"] == work["aggregate_array_bytes"], "实际地址并集不符")
    device = row["device"]
    require(device.get("sms") == 132 and "GH200" in device.get("name", "")
            and str(device.get("uuid", "")).startswith("GPU-"), "目标设备身份不符")
    require(device["l2_cache_bytes"] > work["aggregate_array_bytes"], "源／目的地址并集不再小于L2")
    resources = row["resources"]
    require(resources["threads"] == THREADS and resources["grid_ctas"] == case["active_ctas"],
            "实际grid或线程数量不符")
    require(resources["occupancy_ctas_per_sm"] == 1 and resources["local_size_bytes"] == 0,
            "实际驻留不为1或出现local使用")
    for field in ("registers_per_thread", "dynamic_smem_bytes", "static_smem_bytes"):
        require(type(resources[field]) is int and resources[field] >= 0, f"资源字段不符：{field}")
    require(len(row["trials"]) == 1, "每独立进程必须只有一个正式sample")
    trial = row["trials"][0]
    require(trial["iterations"] == case["iterations"], "正式sample迭代数不符")
    blocks = trial["blocks_detail"]
    require(len(blocks) == resources["grid_ctas"], "grid CTA记录不完整")
    require(all(type(block.get("active")) is bool for block in blocks), "缺少CTA活动标识")
    require(len({block["smid_start"] for block in blocks}) == resources["grid_ctas"],
            "合作式grid的CTA未分别位于不同SM")
    active = [block for block in blocks if block["active"]]
    require(len(active) == case["active_ctas"], "实际活跃CTA数量不符")
    require(len({block["smid_start"] for block in active}) == case["active_ctas"],
            "多个活跃CTA驻留于同一个SM")
    for block in blocks:
        require(type(block["smid_start"]) is int and 0 <= block["smid_start"] < 0xFFFFFFFF
                and block["smid_start"] == block["smid_end"], "SM身份错误或迁移")
        for field in ("begin_ns", "end_ns", "begin_cycle", "end_cycle"):
            require(type(block[field]) is int and block[field] >= 0, "计时字段必须为非负整数")
        if block["active"]:
            require(block["end_ns"] > block["begin_ns"]
                    and block["end_cycle"] > block["begin_cycle"], "计时窗口无效")
        else:
            require(all(block[field] == 0 for field in
                        ("begin_ns", "end_ns", "begin_cycle", "end_cycle")), "inactive CTA被计时")
    begin_ns = min(block["begin_ns"] for block in active)
    end_ns = max(block["end_ns"] for block in active)
    elapsed_ns = end_ns - begin_ns
    common_overlap_ns = min(b["end_ns"] for b in active) - max(b["begin_ns"] for b in active)
    require(trial["all_active_windows_overlap"] == (common_overlap_ns > 0), "实际重叠标识不符")
    require(type(trial["elapsed_ns"]) is int and trial["elapsed_ns"] == elapsed_ns,
            "globaltimer包络无法从CTA记录精确重算")
    event_ms = trial["event_ms"]
    require(isinstance(event_ms, (float, int)) and math.isfinite(event_ms) and event_ms > 0
            and elapsed_ns <= event_ms * 1.05e6, "CUDA event与内部包络不符")
    correctness = trial["correctness"]
    require(correctness["errors"] == 0, "探针正确性检查失败")
    read_checksums = trial["read_checksums"]
    expected_read_checks = work["total_threads"] if work["read_payload_bytes"] else 0
    require(correctness["read_checked"] == expected_read_checks
            and len(read_checksums) == expected_read_checks, "读checksum覆盖不完整")
    for thread, value in enumerate(read_checksums):
        require(type(value) is int and value == read_checksum_oracle(
                    case["array_bytes"], case["active_ctas"], case["iterations"], seed, thread),
                f"独立读checksum复算失败，thread={thread}")
    expected_write_checks = case["array_bytes"] // 4 if work["write_payload_bytes"] else 0
    require(correctness["write_checked"] == expected_write_checks, "写结果覆盖不完整")
    witness = None
    if work["write_payload_bytes"]:
        require(trial["output_file"] == "output.u32" and trial["output_bytes"] == case["array_bytes"],
                "完整写出见证身份不符")
        witness = verify_write_witness(folder, row, seed)
    warmup = row["warmup"]
    windows = warmup["elapsed_ns"]
    require(warmup["limit"] == case["warmup_limit"], "预热上限与清单不符")
    require(8 <= len(windows) <= WARMUP_LIMIT
            and all(type(value) is int and value > 0 for value in windows), "预热窗口数量或计时不符")
    tail = windows[-5:]
    mean = sum(tail) / len(tail)
    cv = math.sqrt(sum((value - mean) ** 2 for value in tail) / (len(tail) - 1)) / mean
    require(type(warmup["converged"]) is bool and warmup["converged"] == (cv <= 0.02),
            "预热收敛标识与末5窗口不符")
    require(math.isclose(warmup["cv"], cv, rel_tol=1e-7, abs_tol=1e-12), "预热CV不能独立复算")
    short_check = row["short_check"]
    require(short_check["iterations"] == [1, 2] and short_check["errors"] == 0
            and short_check["read_checked"] == expected_read_checks * 2
            and short_check["write_checked"] == expected_write_checks * 2, "短检查覆盖不符")
    return dict(device_uuid=device["uuid"], device=device, resources=resources,
                elapsed_ns=elapsed_ns, begin_ns=begin_ns, end_ns=end_ns,
                read_payload_bytes=work["read_payload_bytes"],
                write_payload_bytes=work["write_payload_bytes"], work_count=work["work_count"],
                rate_numerator=work["work_count"], rate_denominator_ns=elapsed_ns,
                requested_GB_per_s=work["work_count"] / elapsed_ns,
                start_span_ns=max(b["begin_ns"] for b in active) - begin_ns,
                common_overlap_ns=common_overlap_ns,
                active_smids=sorted(b["smid_start"] for b in active),
                read_checksums_verified=expected_read_checks, write_witness=witness,
                warmup_windows=len(windows), warmup_cv=cv, warmup_converged=warmup["converged"],
                performance_eligible=warmup["converged"] and common_overlap_ns > 0,
                cache_residency_proven=False, physical_hbm_bytes_proven=False)


def verify_dev_freeze(root, path):
    require(path is not None, "留出执行须显式 --dev-freeze，先冻结开发服务身份")
    freeze = read_json(path)
    require(freeze.get("schema") == FREEZE_SCHEMA, "开发冻结身份 schema 不符")
    require(freeze.get("plan_sha256") == digest(root / "plan.json"), "冻结条件清单不符")
    require(freeze.get("build_identity_sha256") == digest(root / "build/identity.json"),
            "冻结二进制身份不符")
    summary = root / "dev/summary.json"
    require(summary.is_file() and freeze.get("dev_summary_sha256") == digest(summary),
            "开发采样尚未完成或冻结开发结果不符")
    model = (path.parent / freeze["service_file"]).resolve()
    require(model.is_file() and digest(model) == freeze.get("service_sha256"),
            "冻结服务文件缺失或哈希不符")
    dev = read_json(summary)
    require(dev.get("complete") is True and dev.get("all_performance_eligible") is True,
            "开发结果尚未完整或存在未收敛／未重叠条件")
    return dict(path=str(path), sha256=digest(path), service_sha256=digest(model))


@contextlib.contextmanager
def exclusive_lock(path):
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def inspect_allocated_gpu(folder):
    """只在实际分配的计算节点查询GPU；保存当前作业而非复用旧查询。"""
    job = os.environ.get("SLURM_JOB_ID", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(re.fullmatch(r"[0-9]+", job) is not None, "run需要有效SLURM_JOB_ID")
    require(visible and "," not in visible and visible != "-1", "需要一个分配的可见CUDA设备")
    command_record(folder / "allocation", ["scontrol", "show", "job", "-o", job], 15)
    job_info = (folder / "allocation/stdout").read_text()
    require(re.search(r"\bJobState=RUNNING\b", job_info), "Slurm作业未运行")
    require(re.search(r"UserId=[^ ]*\(" + str(os.geteuid()) + r"\)", job_info),
            "Slurm作业所有者不符")
    require(re.search(r"(?:gres/gpu=1(?:[,\s]|$)|gres/gpu:[^=, ]+=1(?:[,\s]|$))", job_info),
            "需要恰好一个已分配的Slurm GPU")
    nodes = re.search(r"\bNodeList=([^ ]+)", job_info)
    require(nodes is not None, "Slurm节点列表缺失")
    command_record(folder / "allocated-hosts", ["scontrol", "show", "hostnames", nodes[1]], 15)
    hosts = (folder / "allocated-hosts/stdout").read_text().split()
    require(os.uname().nodename.split(".")[0] in [host.split(".")[0] for host in hosts],
            "runner不在作业分配的计算节点上")
    command_record(folder / "device-query", ["nvidia-smi", "-i", visible,
                   "--query-gpu=uuid,name", "--format=csv,noheader"], 30)
    lines = (folder / "device-query/stdout").read_text().strip().splitlines()
    require(len(lines) == 1 and "GH200" in lines[0], "需要单个GH200 GPU")
    uuid, name = (value.strip() for value in lines[0].split(",", 1))
    require(uuid.startswith("GPU-") and all(c.isalnum() or c == "-" for c in uuid), "非法设备UUID")
    identity = dict(job_id=job, host=os.uname().nodename, cuda_visible_devices=visible,
                    uuid=uuid, name=name, uid=os.geteuid())
    save_json(folder / "identity.json", identity)
    return identity


def run_scan(root, split, dev_freeze):
    plan = load_plan(root)
    verify_build(root)
    freeze = verify_dev_freeze(root, dev_freeze) if split == "heldout" else None
    cases = [c for c in plan["cases"] if c["split"] == split]
    directory = root / split
    directory.mkdir(exist_ok=True)
    with exclusive_lock(root / ".run.lock"):
        environment = inspect_allocated_gpu(root / "runs" / f"{split}-{time.time_ns()}")
        uuid = environment["uuid"]
        device_binding = dict(uuid=uuid, name=environment["name"])
        if (root / "device-binding.json").exists():
            require(read_json(root / "device-binding.json") == device_binding, "GPU身份与本准备包既有采样不符")
        else:
            save_json(root / "device-binding.json", device_binding)
        results = []
        with exclusive_lock(Path("/tmp") / f"gh200-measurement-{uuid}.lock"):
            for process_index, seed in enumerate(SEEDS):
                ordered = list(cases)
                random.Random(20261017 + process_index).shuffle(ordered)
                for case in ordered:
                    folder = directory / case["id"] / f"process_{process_index:02}"
                    argv = [root / "build/probe", "--active-ctas", case["active_ctas"],
                            "--chains", case["chains"], "--mode", case["mode"],
                            "--array-bytes", case["array_bytes"], "--iterations", case["iterations"],
                            "--seed", seed, "--samples", 1, "--warmup-limit", WARMUP_LIMIT]
                    argv = list(map(str, argv))
                    if folder.exists():
                        result = read_json(folder / "validated.json")
                        receipt = read_json(folder / "receipt.json")
                        require(receipt["argv"] == argv and receipt["returncode"] == 0 and
                                receipt["stdout_sha256"] == digest(folder / "stdout") and
                                receipt["stderr_sha256"] == digest(folder / "stderr"), "旧采样身份变更")
                        replayed = validate_probe(folder, json.loads((folder / "stdout").read_text()), case, seed)
                        require(replayed == result, "旧采样重放不一致")
                        require(result["device_uuid"] == uuid, "旧采样与当前分配GPU身份不符")
                    else:
                        command_record(folder, argv, PROCESS_TIMEOUT)
                        row = json.loads((folder / "stdout").read_text())
                        try:
                            result = validate_probe(folder, row, case, seed)
                        except (KeyError, ValueError, OSError) as error:
                            save_json(folder / "validation-error.json", dict(error=str(error)))
                            raise
                        require(result["device_uuid"] == uuid, "probe 与查询的设备身份不同")
                        save_json(folder / "validated.json", result)
                    results.append(dict(case_id=case["id"], seed=seed,
                                        path=str(folder.relative_to(root)),
                                        validated_sha256=digest(folder / "validated.json"), **result))
                    print(f"VALID {case['id']} seed={seed}", flush=True)
        summary = dict(schema="gh200-l2-saturation-summary/1", split=split, complete=True,
                       plan_sha256=digest(root / "plan.json"),
                       build_identity_sha256=digest(root / "build/identity.json"),
                       device_uuid=uuid, dev_freeze=freeze, results=results,
                       independent_processes=len(results), model_fit_performed=False,
                       all_warmups_converged=all(r["warmup_converged"] for r in results),
                       all_performance_eligible=all(r["performance_eligible"] for r in results),
                       cache_residency_proven=False, physical_hbm_bytes_proven=False)
        summary_path = directory / "summary.json"
        if summary_path.exists():
            require(read_json(summary_path) == summary, "已有汇总与重放不一致")
        else:
            save_json(summary_path, summary)
        unqualified = sum(not result["performance_eligible"] for result in results)
        print(f"COMPLETE {split}: {len(results)} 独立进程，{unqualified} 个未具备性能资格；"
              "未拟合或自动运行其他 split")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="step", required=True)
    for command in ("prepare", "compile", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output", required=True, type=Path)
        if command == "compile":
            sub.add_argument("--nvcc", default="nvcc", help="本地编译器路径；编译不运行GPU")
        if command == "run":
            sub.add_argument("--split", choices=("dev", "heldout"), default="dev")
            sub.add_argument("--dev-freeze", type=Path, help=f"留出所需的 {FREEZE_SCHEMA} 文件")
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.step == "prepare":
        prepare(root)
    elif args.step == "compile":
        compile_probe(root, args.nvcc)
    else:
        run_scan(root, args.split, args.dev_freeze.resolve() if args.dev_freeze else None)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, NotImplementedError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
