"""Q02 三种行距的现有源服务隔离回放；CPU、固定参数、最多 33 次调用。"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import signal
import statistics as st
import subprocess
import sys
import time


CONDITIONS = (("base", 2304), ("B32", 2320), ("B80", 2344))
MODULES = ("q02_source_isolation.py", "tma_stateful_source.py", "tma_gpc_service.py",
           "tma_tensor_multicta.py", "cfg_b_program.py", "case_inputs.py",
           "register_quota.py", "supply_model.py", "v08_model.py", "v06_model.py")


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    repo = Path(__file__).resolve().parents[3]
    base = repo / "results/gh200_resource_campaign/access_rules"
    old = base / "Q02-tma-request/request-shape-response/20261017-Q02-tma-stride-001"
    measured = base / "Q02-tma-request/request-shape-response/20261017-Q02-tma-stride-002"
    sharing = base / "Q02-tma-request/gemm-source-reuse/20261015-tma-source-sharing-development-001"
    state = base / "Q02-tma-request/gemm-source-reuse/20261015-tma-gemm-source-development-001/domain-layout/replay.json"
    package = args.package.resolve()
    package.mkdir(exist_ok=False)
    (package / "source").mkdir()
    inputs = package / "inputs"
    inputs.mkdir()
    for name in MODULES:
        shutil.copy2(Path(__file__).parent / name, package / "source" / name)
    originals = {
        "program.json": old / "program.json", "worklist.txt": old / "worklist.txt",
        "q4-seed3.ndjson": measured / "runs/formal/q4-seed3/stdout",
        "sharing-plan.json": sharing / "plan.json",
        "sharing-predictions.json": sharing / "predictions.json",
        "state-prices.json": state,
    }
    for name, path in originals.items():
        shutil.copy2(path, inputs / name)
    save(package / "plan.json", dict(
        qualification="已失败基线的固定参数开发定位；不恢复候选资格",
        conditions=[dict(name=name, ldb=ldb) for name, ldb in CONDITIONS],
        allocation=dict(lda=4096, alloc_ldb=2360, m=2304, n=2304, k=4096),
        active_groups=4, groups=64, source_ring=64, physical_slots=4,
        warmup_work_indices=[0, 1] * 5, target_work_index=1,
        simulated_calls_limit=33, per_call_timeout_seconds=60, round_timeout_seconds=600,
        model_gap_ns=100000,
        source="current ProgramGemmSource, original domain-state/GPC prices and arbitration",
        control="trace1-derived nine control prices retained as empirical replacements; trace0 reuse at wait return, consumer=0",
        addresses_and_placement="Q02 paired Q4 seed3 process; common allocations and CTA-to-SM mapping",
        scope="cfg_b G5 工作列表前两轮的纯 TMA 探针；无 WGMMA、输出或第三轮",
        input_sources={name: str(path.relative_to(repo)) for name, path in originals.items()}))
    files = list((package / "source").iterdir()) + list(inputs.iterdir()) + [package / "plan.json"]
    save(package / "binding.json", {str(p.relative_to(package)): digest(p) for p in files})
    (package / "README.md").write_text(
        "# Q02 当前源服务隔离回放\n\n"
        "只回放 Q4 base/B32/B80，各 10 次交替暖机和 1 次 work_index=1 目标，共 33 次上限。"
        "复用当前 ProgramGemmSource、原两域状态价格与GPC仲裁；九项trace控制价格保留为经验替代，plain逻辑复用接wait返回、consumer置0。"
        "实际地址与放置取 Q02 同一 seed3 配对进程；A/B 初始化覆盖完整固定分配。\n\n"
        "这是已失败基线的开发定位，不恢复候选资格。输出比较源就绪、目的投递、槽释放与最终窗口，"
        "并保存 32/128 B 地址覆盖、local/remote/pending/HBM 计数和合并数。"
        "旧控制价格来自 trace1，本次连接 trace0；绝对时间只报告诊断值，主要核对行距响应传递。\n\n"
        "```sh\npython source/q02_source_isolation.py run --package .\n```\n\n"
        "每次模拟限 60 秒，全轮限 600 秒，超限保留已完成调用，不自动重试。"
        "CPU 连接检查不执行完整模拟；模型回放从 base 开始。\n")
    print(package)


def modules():
    import cfg_b_program
    import tma_tensor_multicta
    from tma_stateful_source import ProgramGemmSource, StatefulTensorSource
    return cfg_b_program, tma_tensor_multicta, ProgramGemmSource, StatefulTensorSource


def condition_inputs(package, ldb):
    cfg_b_program, _, _, _ = modules()
    original = load(package / "inputs/program.json")
    case = {key: copy.deepcopy(value) for key, value in original["case"].items()
            if key not in cfg_b_program.case_inputs.SETUP_PROGRAM or key in ("stages", "tile", "cluster")}
    setup = copy.deepcopy(original["setup"])
    case["ldb"] = setup["ldb"] = ldb
    program = cfg_b_program.build_program(case, setup)
    tokens = [int(x) for x in (package / "inputs/worklist.txt").read_text().split()]
    assert tokens[:4] == [2304, 2304, 4096, 132]
    coordinates = [tokens[i:i+2] for i in range(4, len(tokens), 2)]
    assert [[list(w[j]) for w in program["work"]] for j in (0, 1)] == [coordinates[:132], coordinates[132:]]
    records = [json.loads(line) for line in (package / "inputs/q4-seed3.ndjson").read_text().splitlines() if line.strip()]
    sample = next(row for row in records if row["ldb"] == ldb)
    assert all(sample[key] == value for key, value in dict(
        active_groups=4, groups=64, ring=64, lda=4096, alloc_ldb=2360,
        trace=0, warmup=10, warmup_work_offset=0, work_index=1,
        source_phase_stride=0, source_tile_shift=0, release_delay_ns=0).items())
    addresses = {key: sample[field] for key, field in dict(A="source_a_address", B="source_b_address",
        records="records_address", ctas="ctas_address", worklist="worklist_address", gate="gate_address").items()}
    assert all(all(row[field] == sample[field] for field in (
        "source_a_address", "source_b_address", "records_address", "ctas_address", "worklist_address", "gate_address"))
        for row in records)
    old = load(package / "inputs/sharing-plan.json")
    sm_groups = dict(zip(old["cta_sms"], old["cta_groups"]))
    ctas = sorted(sample["ctas_data"], key=lambda r: r["cta"])
    groups = [sm_groups[row["smid"]] for row in ctas]
    assert [row["tile"] for row in ctas] == coordinates[132:]
    frozen = load(package / "inputs/sharing-predictions.json")
    return program, sample, addresses, groups, frozen["prices_ns"], frozen["controller"], load(package / "inputs/state-prices.json")["prices"]


def row_coverage(program, cta, work_index, k, atom, addresses, granule):
    output = program["outputs"][cta][work_index]
    ids = output["operands"]["A"]["source_atoms"] + output["operands"]["B"]["source_atoms"]
    source = program["source_atoms"][ids[atom]]
    origin = (source["origin"][0], source["origin"][1] + 64*k)
    stride, shape = source["stride_bytes"], source["shape"]
    axis = 0 if atom == 0 else 1
    first = addresses[source["operand"]] + source["base_offset_bytes"] + sum(a*b for a, b in zip(origin, stride))
    length = shape[1-axis]*2
    return sum((first+row*stride[axis]+length-1)//granule - (first+row*stride[axis])//granule + 1
               for row in range(shape[axis]))


class ProbeConnection:
    """只转换程序请求编号；源服务仍由原 ProgramGemmSource.input 提供。"""
    def __init__(self, program, service):
        self.program, self.service = program, service
        self.version = service.version
        self.work_index = 0
        self.capture = False
        self.extra_max = 0.

    @property
    def count(self):
        return self.service.count

    @property
    def work(self):
        return self.service.work

    @property
    def busy(self):
        return self.service.busy

    def submit(self, t, cta, key, weight, partition_demand=None):
        _, k, operand = key
        atom = ("A", "B0", "B1").index(operand)
        result = self.service.input(t, cta, self.work_index*self.service.ktiles+k, atom, weight)
        core = self.service.core_pending[self.service.cta_groups[cta], result["source_key"]]
        extra = result["source_ready"] - (core["source_ready"]-self.service.base)
        self.extra_max = max(self.extra_max, extra)
        if self.capture:
            result.update(core_ready_ns=core["source_ready"]-self.service.base, data_extra_ns=extra,
                cover32=row_coverage(self.program, cta, self.work_index, k, atom, self.service.addresses, 32),
                cover128=row_coverage(self.program, cta, self.work_index, k, atom, self.service.addresses, 128))
        return result


def make_service(package, ldb):
    _, _, ProgramGemmSource, _ = modules()
    program, sample, addresses, groups, prices, control, state_prices = condition_inputs(package, ldb)
    service = ProgramGemmSource(program, groups, prices, output_base=0,
        **state_prices, addresses=addresses, trace_stores=False, record_diagnostics=False)
    return program, sample, service, ProbeConnection(program, service), prices, control


def check(package):
    _, _, _, StatefulTensorSource = modules()
    rows = []
    for name, ldb in CONDITIONS:
        program, _, service, connection, prices, _ = make_service(package, ldb)
        StatefulTensorSource.begin(service, 1, 100000., groups=64, probe_metadata=True)
        connection.work_index = 1
        requests = []
        for atom, (operand, weight) in enumerate((("A", 1.), ("B0", .5), ("B1", .5))):
            response = connection.submit(0., 0, (0, 0, operand), weight)
            expected = row_coverage(program, 0, 1, 0, atom, service.addresses, 128)
            assert len(service.line_sets[response["source_key"]]) == expected
            assert response["source_key"][2] == 0 and service.work_index == 1
            requests.append(dict(atom=operand, cover32=row_coverage(program, 0, 1, 0, atom, service.addresses, 32), cover128=expected))
        expected_b = {2304: (256, 64), 2320: (256, 112), 2344: (288, 120)}[ldb]
        assert [(r["cover32"], r["cover128"]) for r in requests] == [(512, 128), expected_b, expected_b]
        rows.append(dict(condition=name, ldb=ldb, prices=prices, requests=requests,
            work_rounds=[list(program["work"][0][j]) for j in (0, 1)]))
    save(package / "cpu-check.json", dict(status="pass", full_simulated_calls=0,
        checks="first-two-round worklist, actual placement/allocation, adapter phase/key, A/B row coverage", rows=rows))


def timeout(signum, frame):
    raise TimeoutError("single model call exceeded 60 seconds")


def worker(args):
    package = args.package.resolve()
    _, tensor, _, StatefulTensorSource = modules()
    name, ldb = next((n, l) for n, l in CONDITIONS if n == args.condition)
    out = package / "replay" / name
    out.mkdir()
    program, sample, service, connection, prices, control = make_service(package, ldb)
    service.store(0., None, service.addresses["A"], sample["allocated_a_elements"]*2)
    service.store(0., None, service.addresses["B"], sample["allocated_b_elements"]*2)
    p = tensor.program(132, 4, "independent", groups=64, source_ring=64, ldb=ldb)
    p["source_shape"] = dict(m=2304, n=2304, k=4096)
    p["source_strides_elements"] = dict(A=4096, B=ldb)
    p["source_allocation_bytes"] = dict(A=sample["allocated_a_elements"]*2, B=sample["allocated_b_elements"]*2)
    p["initial_offsets_ns"] = [0.]*132
    signal.signal(signal.SIGALRM, timeout)
    history, base = [], 100000.
    for launch in range(11):
        work_index = launch % 2 if launch < 10 else 1
        StatefulTensorSource.begin(service, work_index, base, groups=64, probe_metadata=True)
        connection.work_index, connection.capture, connection.extra_max = work_index, launch == 10, 0.
        before = {k: getattr(service, k) for k in ("hbm_lines", "remote_lines", "local_lines", "pending_lines")}
        start = time.monotonic()
        signal.alarm(60)
        try:
            result = tensor.simulate(p, "shared_source_fifo", source_service=connection,
                service=prices[0], latency=prices[1], destination=prices[2], issue=0.,
                control=control["control_ns"], consumer=0., trace=launch == 10,
                memory_observer=service.memory_observer)
        except TimeoutError as error:
            save(out / "status.json", dict(status="timeout", completed_calls=len(history), failed_launch=launch, error=str(error)))
            return 2
        finally:
            signal.alarm(0)
        row = dict(launch=launch, work_index=work_index, cpu_seconds=time.monotonic()-start,
            final_median_ns=st.median(result["cta_times"]), final_envelope_ns=result["time"],
            final_span_ns=result["cta_span"], merged=result["merges"],
            max_data_extra_ns=connection.extra_max,
            **{k:getattr(service,k)-before[k] for k in before})
        history.append(row)
        save(out / "history.json", history)
        print(name, launch, json.dumps(row, ensure_ascii=False), flush=True)
        base += result["time"] + 100000.
    requests, events = result["request_events"], result["group_events"]
    final = dict(condition=name, ldb=ldb, qualification="已失败基线的固定参数开发定位",
        model=dict(state_prices=load(package / "inputs/state-prices.json")["prices"], prices_ns=prices, controller={**control, "consumer_ns": 0.}),
        source_ready_median_ns=st.median(r["source_ready"] for r in requests),
        source_ready_last_ns=max(r["source_ready"] for r in requests),
        delivery_last_ns=max(r["ready"] for r in requests),
        destination_queue_median_ns=st.median(r["destination_start"]-r["source_ready"] for r in requests),
        source_to_ready_median_ns=st.median(r["ready"]-r["source_ready"] for r in requests),
        cover32_total=sum(r["cover32"] for r in requests), cover128_total=sum(r["cover128"] for r in requests),
        final_median_ns=st.median(result["cta_times"]), final_envelope_ns=result["time"], final_span_ns=result["cta_span"],
        target=history[-1], cta_times_ns=result["cta_times"],
        issue_to_wait_return_median_ns=st.median(e["wait_return"]-e["issue"] for e in events),
        request_count=len(requests), group_count=len(events), addresses=service.addresses,
        cta_gpcs=service.cta_groups)
    save(out / "summary.json", final)
    with gzip.open(out / "target-events.json.gz", "wt") as f:
        json.dump(dict(request_events=requests, group_events=events), f, separators=(",", ":"))
    save(out / "status.json", dict(status="complete", completed_calls=11))
    return 0


def score(package):
    rows = []
    for name, _ in CONDITIONS:
        path = package / "replay" / name / "summary.json"
        if path.exists():
            rows.append(load(path))
    comparisons = []
    if rows and rows[0]["condition"] == "base":
        for row in rows[1:]:
            comparisons.append(dict(condition=row["condition"], **{
                key+"_response_pct": (row[key]/rows[0][key]-1)*100
                for key in ("source_ready_last_ns", "delivery_last_ns", "final_median_ns", "final_envelope_ns", "cover32_total", "cover128_total")}))
    save(package / "score.json", dict(qualification="已失败基线定位；开发数据回放，不恢复候选资格",
        complete_conditions=len(rows), comparisons=comparisons, rows=rows))


def run(args):
    package = args.package.resolve()
    for name, sha in load(package / "binding.json").items():
        assert digest(package / name) == sha, name
    check(package)
    (package / "replay").mkdir()
    start = time.monotonic()
    status = []
    for name, _ in CONDITIONS:
        remaining = 600-(time.monotonic()-start)
        if remaining <= 0:
            break
        command = [sys.executable, str(package / "source/q02_source_isolation.py"), "_worker", "--package", str(package), "--condition", name]
        with (package / "replay" / (name+".stdout")).open("w") as stdout, (package / "replay" / (name+".stderr")).open("w") as stderr:
            try:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=remaining)
                status.append(dict(condition=name, returncode=completed.returncode))
            except subprocess.TimeoutExpired:
                status.append(dict(condition=name, status="round_timeout"))
                break
        if completed.returncode:
            break
    score(package)
    histories = list((package / "replay").glob("*/history.json"))
    save(package / "execution.json", dict(cpu_seconds=time.monotonic()-start,
        completed_calls=sum(len(load(p)) for p in histories), new_gpu_processes=0, conditions=status))
    print(json.dumps(load(package / "execution.json"), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "_worker"))
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--condition", choices=[name for name, _ in CONDITIONS])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        run(args)
    else:
        sys.exit(worker(args))


if __name__ == "__main__":
    main()
