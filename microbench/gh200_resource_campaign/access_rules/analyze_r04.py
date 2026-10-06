#!/usr/bin/env python3
"""CPU-only recomputation of all R04 outputs, fixed work and paired statistics."""

import argparse
import array
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import re
import sys


def half_checksum(base, count):
    def floor_prefix(value, step):
        if value < 0:
            return 0
        q, r = divmod(value, step)
        return step * q * (q - 1) // 2 + q * (r + 1)

    def rounded_prefix(value, step):
        if value < 0:
            return 0
        half = step // 2
        ties = (value - half) // (2 * step) + 1 if value >= half else 0
        return floor_prefix(value + half, step) - floor_prefix(half - 1, step) - ties

    first, last, total = base + 1, base + count, 0
    while first <= last:
        n = first.bit_length() - 1
        end = min(last, (1 << (n + 1)) - 1)
        items = end - first + 1
        constant = (n + 7) * 1024 - 1024
        if n <= 10:
            total += items * constant + ((first + end) * items // 2) * (1 << (10 - n))
        else:
            step = 1 << (n - 10)
            total += items * constant + rounded_prefix(end, step) - rounded_prefix(first - 1, step)
        first = end + 1
    return total & 65535


def expected(record):
    f = record["family"]
    o = record["order"]
    a = record["a_count"] * record["repeats"]
    b = record["b_count"] * record["repeats"]
    for t in range(record["threads"]):
        for j in range(140):
            x = 0.0
            if j < 128 and f < 2 and (f == 0 or t < 128) and o != 3:
                x = (record["repeats"] % 2) * record["a_count"] / 16
            if 128 <= j < 132 and f != 3 and (f != 1 or t >= 128) and o != 2:
                x = (
                    (record["repeats"] % 2)
                    * (record["b_count"] / 4)
                    * (1 + t % 7)
                    * (j - 127)
                    / 1024
                )
            if 132 <= j < 136 and f >= 2 and o != 3:
                x = (t * 4 + j - 132 + 1) / 32
            if j >= 136 and f == 3 and o != 2:
                x = half_checksum((t + 1 + j - 136) * 8, record["repeats"] * record["b_count"] // 4)
            yield x


def work(r):
    f = r["family"]
    a = r["a_count"] * r["repeats"]
    b = r["b_count"] * r["repeats"]
    # A count is WG collective instructions for MMA, per-warp LDS otherwise.
    return dict(
        a_wg_instructions=a if f < 2 else 0,
        a_warp_instructions=4 * a if f >= 2 else 0,
        a_flop=524288 * a if f < 2 else 0,
        a_read_bytes=128 * 16 * a if f >= 2 else 0,
        b_warp_instructions=4 * b,
        b_flop=128 * 2 * b if f != 3 else 0,
        b_conversions=128 * b if f == 3 else 0,
        b_prepare_fadd=128 * b if f == 3 else 0,
        b_checksum_integer_ops=256 * b if f == 3 else 0,
    )


KERNEL = re.compile(r"_Z5jointILi(\d)ELi(\d)|_Z11cross_jointILi(\d)E")


def sass_ops(part):
    """(pc, opcode text) for every instruction of one cuobjdump function."""
    ops = []
    for line in part.splitlines():
        m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*/\*", line)
        if m:
            ops.append((int(m[1], 16), m[2].strip()))
    return ops


def wgmma_groups(ops):
    """Split the WGMMA stream into groups: each ends at an HGMMA carrying gsb0 (commit)."""
    groups, current = [], 0
    for _, op in ops:
        if "HGMMA.64x256x16.F32" in op:
            current += 1
            if "gsb0" in op:
                groups.append(current)
                current = 0
    return groups


def audit_sass(folder):
    text = (folder / "build/r04.sass").read_text()
    log = (folder / "build/compile.log").read_text().splitlines()
    rows = []
    for part in text.split("Function : "):
        match = KERNEL.match(part)
        if not match:
            continue
        if match.group(3) is not None:
            family, order = 1, int(match.group(3))
        else:
            family, order = int(match.group(1)), int(match.group(2))
        name = part.splitlines()[0].strip()
        lines = part.splitlines()
        clocks = [i for i, l in enumerate(lines) if "SR_CLOCKLO" in l]
        if len(clocks) != 2:
            raise ValueError("expected exactly one clock64 start/end site " + name)
        # cross_joint places the warpgroup-1 branch after the end stamp, so it is audited to
        # the end of the function; the generic kernel is audited between the two stamps.
        body = lines[clocks[0] :] if family == 1 else lines[clocks[0] : clocks[1] + 1]
        ops = sass_ops("\n".join(body))
        targets = [i for i, l in enumerate(body) if "HGMMA.64x256x16.F32" in l]
        zero_between = sum(
            any("WARPGROUP.DEPBAR.LE" in l and "0x0" in l for l in body[a + 1 : b])
            for a, b in zip(targets, targets[1:])
        )
        groups = wgmma_groups(ops)
        rows.append(
            dict(
                family=family,
                order=order,
                function=name,
                clock_start_site=lines[clocks[0]].strip(),
                clock_end_site=lines[clocks[1]].strip(),
                target_hgmma_static=len(targets),
                hgmma_per_commit_group=sorted(set(groups)),
                warpgroup_arrive_static=sum(op == "WARPGROUP.ARRIVE" for _, op in ops),
                wait1_static=sum("WARPGROUP.DEPBAR.LE gsb0, 0x1" in op for _, op in ops),
                wait0_static=sum("WARPGROUP.DEPBAR.LE gsb0, 0x0" in op for _, op in ops),
                helper_hgmma_static=sum("HGMMA.64x8x16" in l for l in body),
                lds128_static=sum("LDS.128" in l for l in body),
                ffma_static=sum(" FFMA " in l for l in body),
                cvt_static=sum("F2F.F16.F32" in l for l in body),
                wait0_between_target_sites=zero_between,
                compiler_serialization_warning=any("C7520" in l and name in l for l in log),
                compiler_injected_wait=any("C7517" in l and name in l for l in log),
                compiler_injected_arrive=any("C7519" in l and name in l for l in log),
                dynamic_count_basis=(
                    "source loops and active warp/WG predicates; "
                    "static sites alone are not dynamic counts"
                ),
            )
        )
    if len(rows) != 16:
        raise ValueError("expected sixteen compiled family/order kernels")
    return rows


def audit_dynamic_work(folder):
    rows = []
    for part in (Path(folder) / "build/r04.sass").read_text().split("Function : "):
        match = re.match(r"_Z5jointILi([23])ELi(\d)", part)
        if not match:
            continue
        family, order = map(int, match.groups())
        ops = []
        for line in part.splitlines():
            m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*/\*", line)
            if m:
                ops.append((int(m[1], 16), m[2].strip()))
        clocks = [pc for pc, op in ops if "SR_CLOCKLO" in op]
        start, end = clocks[0], clocks[-1]
        loops = []
        for pc, op in ops:
            m = re.search(r"\bBRA(?:\.[A-Z]+)?\s+0x([0-9a-f]+)", op)
            if m and start <= int(m[1], 16) < pc <= end:
                loops.append((int(m[1], 16), pc))
        inners = [
            (a, b)
            for a, b in loops
            if not any(a <= c < d <= b and (a, b) != (c, d) for c, d in loops)
        ]
        proof = []
        for lo, hi in inners:
            body = [(pc, op) for pc, op in ops if lo <= pc <= hi]
            loads = [(pc, op) for pc, op in body if "LDS.128" in op and re.search(r"\[R\d+", op)]
            bsites = [
                (pc, op) for pc, op in body if ("FFMA" if family == 2 else "F2F.F16.F32") in op
            ]
            signature = (len(loads), len(bsites))
            if order == 1:
                trip = {(1, 4): 32, (4, 4): 20, (16, 4): 8}.get(signature)
                if trip is None:
                    raise ValueError("mixed loop opcode count " + str((family, order, signature)))
                if not any(
                    ("ISETP" in op) and re.search(r"0x" + format(trip, "x") + r"\b", op)
                    for pc, op in body
                ):
                    raise ValueError("mixed loop counter bound " + str((family, signature)))
            else:
                expected = (
                    [(1, 0), (0, 4)] if order == 0 else ([(1, 0)] if order == 2 else [(0, 4)])
                )
                if signature not in expected:
                    raise ValueError(
                        "serial/baseline loop target count " + str((family, order, signature))
                    )
                trip = "n_A" if signature[0] else "n_B/4"
            targets = loads + bsites
            if any(op.startswith("@") for pc, op in targets):
                raise ValueError("predicated target inside counted loop")
            if any(sum(a <= pc <= b for a, b in loops) < 2 for pc, op in targets):
                raise ValueError("target not nested in action and repeat loops")
            proof.append(
                dict(
                    begin=hex(lo),
                    backedge=hex(hi),
                    trip=trip,
                    lds128_sites=len(loads),
                    b_sites=len(bsites),
                    target_sequence=[
                        "A" if "LDS.128" in op else "B" for pc, op in body if (pc, op) in targets
                    ],
                    counter_updates_and_tests=[
                        dict(pc=hex(pc), op=op)
                        for pc, op in body
                        if any(x in op for x in ("ISETP", "IADD", "VIADD", "BRA"))
                    ],
                    counter_tail=[dict(pc=hex(pc), op=op) for pc, op in body[-7:]],
                    target_sites=[dict(pc=hex(pc), op=op) for pc, op in targets],
                )
            )
        if len(proof) != (3 if order == 1 else (2 if order == 0 else 1)):
            raise ValueError("missing counted loop " + str((family, order)))
        rows.append(
            dict(
                family=family,
                order=order,
                status="CFG_counted",
                loops=proof,
                excluded=(
                    "uniform shared timestamp reads and output drain are control, not LDS demand"
                ),
            )
        )
    if len(rows) != 8:
        raise ValueError("eight LDS/CVT kernels required in module")
    return rows


def analyze(folder, output=None):
    folder = Path(folder)
    destination = Path(output) if output else folder
    destination.mkdir(parents=True, exist_ok=True)
    machine = audit_sass(folder)
    gpu_source = (folder / "source/probes/r04.cu").read_text()
    corrected = "ld.volatile.shared.v4.b32" in gpu_source and "0f3b800000" in gpu_source
    dynamic = audit_dynamic_work(folder) if corrected else []
    (destination / "dynamic-work-cfg.json").write_text(json.dumps(dynamic, indent=2) + "\n")
    machine_lookup = {(r["family"], r["order"]): r for r in machine}
    identities = json.loads((folder / "source_hashes.json").read_text())
    for name, digest in identities.items():
        if hashlib.sha256((folder / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("source hash mismatch " + name)
    digest = json.loads((folder / "build/binary_hash.json").read_text())["r04"]
    if hashlib.sha256((folder / "build/r04").read_bytes()).hexdigest() != digest:
        raise ValueError("binary hash mismatch")
    rows = []
    summary = []
    groups = {}
    max_error = 0
    for file in sorted((folder / "samples").glob("*/*/result.json")):
        r = json.loads(file.read_text())
        if r["family"] >= 2 and not corrected:
            raise ValueError("invalid_dynamic_work: old fixed LDS/CVT record " + str(file))
        raw = file.parent / "output.f32"
        if hashlib.sha256(raw.read_bytes()).hexdigest() != r["output_sha256"]:
            raise ValueError("output hash mismatch")
        values = array.array("f")
        values.frombytes(raw.read_bytes())
        refs = list(expected(r))
        if len(values) != len(refs) or len(values) != r["output_elements"]:
            raise ValueError("output length mismatch")
        if not all(math.isfinite(x) for x in values):
            raise ValueError("nonfinite saved output")
        error = max(abs(x - y) for x, y in zip(values, refs))
        max_error = max(error, max_error)
        if error > 1e-4:
            raise ValueError(f"CPU output mismatch {file}: {error}")
        if r["trial_role"] != "formal":
            continue
        c = r["configuration"]
        groups.setdefault(c["id"], []).append(r)
        rows.append(
            dict(
                case_id=c["id"],
                trial=file.parent.name,
                elapsed_cycles=r["elapsed"],
                elapsed_ns=r["elapsed_ns"],
                cpu_error=error,
                **work(r),
            )
        )
    groups_meta = {
        case: dict(family=rs[0]["family"], order=rs[0]["order"]) for case, rs in groups.items()
    }
    for case, rs in groups.items():
        vals = [r["elapsed"] for r in rs]
        mean = statistics.mean(vals)
        cv = statistics.stdev(vals) / mean if len(vals) > 1 else None
        code = machine_lookup[(rs[0]["family"], rs[0]["order"])]
        summary.append(
            dict(
                pipeline_mode=rs[0].get(
                    "wgmma_pipeline_mode",
                    (
                        "compiler_serialized"
                        if code["compiler_serialization_warning"]
                        else "previous_source_protocol"
                    ),
                ),
                compiler_serialization_warning=code["compiler_serialization_warning"],
                case_id=case,
                processes=len(vals),
                median=statistics.median(vals),
                minimum=min(vals),
                maximum=max(vals),
                cv=cv,
                warmup_all_converged=all(r["warmup_converged"] for r in rs),
                registers=rs[0]["registers_per_thread"],
                local_bytes=rs[0]["local_bytes_per_thread"],
                **work(rs[0]),
            )
        )
    for name, data in [("samples.csv", rows), ("cases.csv", summary)]:
        if data:
            with (destination / name).open("w") as out:
                w = csv.DictWriter(out, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    lookup = {s["case_id"]: s for s in summary}
    pairs = []
    for name in ("wgmma_ffma_same", "wgmma_ffma_cross", "lds_ffma", "lds_cvt"):
        for a, b in [(80, 80), (32, 128), (128, 32)]:
            s = lookup.get(f"{name}_{a}:{b}_serial")
            i = lookup.get(f"{name}_{a}:{b}_interleaved")
            if not s or not i:
                continue
            ta = (
                lookup[name + "_a_only"]["median"] * a / 160 if name + "_a_only" in lookup else None
            )
            tb = (
                lookup[name + "_b_only"]["median"] * b / 160 if name + "_b_only" in lookup else None
            )
            pairs.append(
                dict(
                    family=name,
                    a=a,
                    b=b,
                    serial=s["median"],
                    interleaved=i["median"],
                    paired_difference=i["median"] - s["median"],
                    scaled_baseline_overlap=(
                        max(ta, tb) if ta is not None and tb is not None else None
                    ),
                    scaled_baseline_sum=ta + tb if ta is not None and tb is not None else None,
                    baseline_status="scaled; count linearity not independently checked",
                )
            )
    rules = dict(
        status="observation",
        cpu_max_error=max_error,
        environment=json.loads((folder / "environment.json").read_text()),
        pairs=pairs,
        cases=summary,
        dependencies=["R03 matched LDS baseline", "R01 matched dependency context"],
        machine_code_audit=machine,
        dynamic_work_cfg=dynamic,
        scope="independent joint services; does not model real FP8 partial-sum promotion",
    )
    (destination / "rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        (destination / "plots").mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, name in zip(
            axes.flat, ("wgmma_ffma_same", "wgmma_ffma_cross", "lds_ffma", "lds_cvt")
        ):
            data = [r for r in pairs if r["family"] == name]
            if not data:
                continue
            x = list(range(len(data)))
            ax.bar([i - 0.18 for i in x], [r["serial"] for r in data], width=0.36, label="serial")
            ax.bar(
                [i + 0.18 for i in x],
                [r["interleaved"] for r in data],
                width=0.36,
                label="interleaved",
            )
            ax.set_xticks(x, [f"{r['a']}:{r['b']}" for r in data])
            ax.set_title(name)
            ax.set_ylabel("SM clock64 cycles")
            ax.legend()
        fig.tight_layout()
        fig.savefig(destination / "plots/joint_windows.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass

    text = [
        "# R04 联合服务结果",
        "",
        f"CPU重算最大误差：{max_error}。正式配置：{len(summary)}。",
        "",
        "计时为单CTA的clock64，窗口含循环、分支、同步与结果排空。串行点先完成A（WGMMA时wait0），"
        "再执行独立B；交错点各自合法完成后结束。跨warpgroup点每单元末尾CTA同步，串行点另有阶段同步。",
        "",
        "A-only/B-only各160次；max/sum按次数缩放，只作比较，未独立证明线性。CVT的B含FADD准备、"
        "RNE转换和16bit校验和消费，不是裸CVT速率。",
        "",
        "## WGMMA机器码",
        "",
        "| family/order | HGMMA静态 | 每commit组 | ARRIVE | wait1 | wait0 | C7520 | C7517 |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    present = {(s["family"], s["order"]) for s in (groups_meta.values())}
    for m in machine:
        if m["family"] >= 2 or (m["family"], m["order"]) not in present:
            continue
        text.append(
            f"| {m['family']}/{m['order']} | {m['target_hgmma_static']} | "
            f"{m['hgmma_per_commit_group']} | {m['warpgroup_arrive_static']} | "
            f"{m['wait1_static']} | {m['wait0_static']} | "
            f"{m['compiler_serialization_warning']} | {m['compiler_injected_wait']} |"
        )
    text += [
        "",
        "## 配对与max/sum",
        "",
        "| 配对 | A:B | 串行 | 交错 | 交错−串行 | 缩放max | 缩放sum | 交错/max−1 | 交错/sum−1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pairs:
        overlap, total = r["scaled_baseline_overlap"], r["scaled_baseline_sum"]
        cells = ["missing"] * 4
        if overlap is not None:
            cells = [
                f"{overlap:.1f}",
                f"{total:.1f}",
                f"{r['interleaved'] / overlap - 1:+.4f}",
                f"{r['interleaved'] / total - 1:+.4f}",
            ]
        text.append(
            f"| {r['family']} | {r['a']}:{r['b']} | {r['serial']:.1f} | {r['interleaved']:.1f} | "
            f"{r['paired_difference']:.1f} | " + " | ".join(cells) + " |"
        )
    if (destination / "plots/joint_windows.png").exists():
        text += ["", "![Joint windows](plots/joint_windows.png)", ""]
    text += [
        "",
        "| 配置 | 模式 | 进程 | 中位数cycle | CV | registers | local B/thread | 预热收敛 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for s in summary:
        cv = f"{s['cv']:.2e}" if s["cv"] is not None else "未估计"
        text.append(
            f"| {s['case_id']} | {s['pipeline_mode']} | {s['processes']} | {s['median']} | {cv} | "
            f"{s['registers']} | {s['local_bytes']} | {s['warmup_all_converged']} |"
        )
    text += [
        "",
        "SASS：`build/r04.sass`；编译日志：`build/compile.log`；完整输出与SHA256在每进程目录；"
        "CPU入口为归档`source/analyze_r04.py`。不作物理端口归因。",
        "",
    ]
    (destination / "report.md").write_text("\n".join(text))
    (destination / "analyzer_identity.json").write_text(
        json.dumps(
            dict(
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                python=sys.version,
                input=str(folder.resolve()),
            ),
            indent=2,
        )
        + "\n"
    )
    return rules


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    analyze(args.input, args.output)
