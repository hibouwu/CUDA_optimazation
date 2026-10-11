#!/usr/bin/env python3
"""R05 exact short trace checks, formal numeric checks and offline reports."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct

_supply_reference_cache = {}


def words(path, code="I"):
    raw = path.read_bytes()
    return struct.unpack("<" + code * (len(raw) // 4), raw)


def filler_value(start, count):
    # Compose the affine LCG modulo2^32, without a count-long CPU loop.
    mul, add = 1, 0
    step_mul, step_add = 1664525, 1013904223
    while count:
        if count & 1:
            mul, add = mul * step_mul & 0xFFFFFFFF, (add * step_mul + step_add) & 0xFFFFFFFF
        step_mul, step_add = (
            step_mul * step_mul & 0xFFFFFFFF,
            step_add * (step_mul + 1) & 0xFFFFFFFF,
        )
        count //= 2
    return (start * mul + add) & 0xFFFFFFFF


def check(r, directory):
    for name, digest in r["artifacts"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("artifact hash mismatch")
    family, proto, count = r["family"], r["protocol"], r["iterations"]
    checked = 0
    if family == "D":
        if r["occupancy_limit_ctas_per_sm"] != 1:
            raise ValueError("D occupancy limit")
        raw = (directory / "final.u32").read_bytes()
        panels, slots = r["panels"], r["slots"]
        blocks = r["blocks"]
        key = (panels, slots, blocks, count, r["source"])
        expected = _supply_reference_cache.get(key)
        if expected is None:
            tiles = []
            for block in range(blocks):
                panel = (block * slots if r["source"] == "independent" else 0) + (count - 1) % slots
                if block and r["source"] == "shared":
                    tiles.append(tiles[0])
                    continue
                rows = []
                for which, height in enumerate((128, 256)):
                    offset = 0 if which == 0 else panels * 4096
                    for y in range(height):
                        values = []
                        for physical in range(32):
                            logical_x = ((physical // 4) ^ (y % 8)) * 4 + physical % 4
                            index = offset + panel * height * 32 + y * 32 + logical_x
                            values.append((17 * index + 17) & 0xFFFFFFFF)
                        rows.append(struct.pack("<32I", *values))
                tiles.append(b"".join(rows))
            expected = b"".join(tiles)
            _supply_reference_cache[key] = expected
        if raw != expected:
            raise ValueError("D complete physical SW128 tile mismatch")
        checked = len(raw) // 4
    elif family == "E":
        raw = (directory / "global.u32").read_bytes()
        pitch = r["pitch"] // 2
        tile_bytes = 128 * r["pitch"]
        if len(raw) != r["blocks"] * 8 * tile_bytes:
            raise ValueError("E output length")
        poison = struct.pack("<I", 0xDEADBEEF) * (tile_bytes // 4)
        padding = [0xBEEF if x % 2 == 0 else 0xDEAD for x in range(64, pitch)]
        for block in range(r["blocks"]):
            rows = [
                struct.pack(
                    "<" + "H" * pitch,
                    *([(17 * x + 31 * y + 151 * block + 17) & 65535 for x in range(64)] + padding),
                )
                for y in range(128)
            ]
            tile = b"".join(rows)
            for slot in range(8):
                begin = (block * 8 + slot) * tile_bytes
                if raw[begin : begin + tile_bytes] != (tile if slot < count else poison):
                    raise ValueError(f"E valid region or padding block{block} slot{slot}")
        checked = len(raw) // 2
    elif proto == "wg":
        v = words(directory / "consumed.f32", "f")
        integers = (
            words(directory / "integers.u32") if (directory / "integers.u32").exists() else None
        )
        if len(v) != 16512:
            raise ValueError("WG output size")
        for lane in range(128):
            for reg in range(128):
                row = (lane // 32) * 16 + (lane % 32) // 4 + ((reg // 2) % 2) * 8
                col = (lane % 4) * 2 + reg % 2 + (reg // 4) * 8
                expected = (
                    0
                    if r["control"]
                    else sum(
                        (1 + (row + 2 * k) % 7) * (1 + (col + 3 * k) % 11) / 512 for k in range(16)
                    )
                    * count
                )
                if v[lane * 128 + reg] != expected:
                    raise ValueError("WG accumulator mismatch")
                checked += 1
            expected_fill = (
                filler_value(integers[lane * 2], r["filler"])
                if integers is not None
                else lane + 17 + count * r["filler"]
            )
            if integers is not None and integers[lane * 2 + 1] != expected_fill:
                raise ValueError("WG actual integer output")
            expected_fill = struct.unpack("<f", struct.pack("<f", expected_fill))[0]
            if v[16384 + lane] != expected_fill:
                raise ValueError("WG filler")
    else:
        q = 2048 if proto == "cp" else 16384
        w = q // 4
        requests = 8 if family == "B" else 1
        v = words(directory / "consumed.f32", "f")
        for lane, observed in enumerate(v):
            index = lane * 4 if proto == "cp" else lane
            val = ((29 * index + 17) if r["control"] else (17 * index + 17)) & 15
            expected = (
                0 if proto == "tout" else ((val - 8) / 32 + r["consumer"] / 1024) * count * requests
            )
            if observed != expected:
                raise ValueError(f"{proto} consumer lane{lane}: {observed} != {expected}")
            checked += 1
        if proto == "tout":
            v = words(directory / "global.u32")
            for slot in range(8):
                last = slot + (count - 1 - slot) // 8 * 8
                for request in range(r["requests"]):
                    for word in range(w):
                        at = (slot * r["requests"] + request) * w + word
                        expected = (
                            (29 * word + 31 * last + 17) & 0xFFFFFFFF
                            if last >= 0 and not r["control"]
                            else (17 * at + 17) & 0xFFFFFFFF
                        )
                        if v[at] != expected:
                            raise ValueError("TMA output old-generation check")
                        checked += 1
        if r["capture"]:
            v = words(directory / "trace.u32")
            if proto == "tout":
                for word in range(w):
                    if v[word] != (~(29 * word + 31 * (count - 1) + 17)) & 0xFFFFFFFF:
                        raise ValueError("source replacement")
                    checked += 1
                filler = v[w]
            else:
                for i in range(count * requests):
                    src = (i % 8) * w
                    for word in range(w):
                        expected = (29 * word + 17) if r["control"] else 17 * (src + word) + 17
                        if v[i * w + word] != expected & 0xFFFFFFFF:
                            raise ValueError(f"input trace {i} {word}")
                        checked += 1
                filler = v[count * requests * w]
            expected_fill = (
                filler_value(17, count * requests * r["filler"])
                if r.get("filler_instruction") == "dependent_IMAD_LCG"
                else 17 + count * requests * r["filler"]
            )
            if filler != expected_fill:
                raise ValueError("filler count")
    if r["work_bytes"] != (
        0
        if r["control"]
        else (
            r["blocks"] * count * 49152
            if family == "D"
            else (
                r["blocks"] * count * 16384
                if family == "E"
                else (
                    0
                    if proto == "wg"
                    else count
                    * (8 if family == "B" else 1)
                    * (2048 if proto == "cp" else 16384)
                    * r["requests"]
                )
            )
        )
    ):
        raise ValueError("work count")
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    a = p.parse_args()
    root = a.input
    env = json.loads((root / "environment.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    groups = {c["id"]: [] for c in cases}
    checked = 0
    for relative, digest in json.loads((root / "source_hashes.json").read_text()).items():
        if hashlib.sha256((root / "source" / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("archived source hash mismatch")
    if hashlib.sha256((root / "build/r05").read_bytes()).hexdigest() != json.loads(
        (root / "build/binary_hash.json").read_text()
    ):
        raise ValueError("binary hash mismatch")
    for f in sorted((root / "samples").glob("*/*/result.json")):
        r = json.loads(f.read_text())
        checked += check(r, f.parent)
        if r["role"].startswith("trial-"):
            groups[r["case_id"]].append(r)
    report = [
        "# R05 异步完成与复用",
        f"\nGPU `{env['gpu_uuid']}`，模式 `{env['mode']}`。CPU重算通过 {checked} 个输出。",
        "\nclock64只在同CTA相减；整卡只报告globaltimer包络ns。小源表示实际地址复访，不证明物理L2命中。stage改变活动槽数；D的occupancy=1是驻留上限。A的等待位置相对请求准备起点（WG为MMA调用前）；传输坐标包含地址准备/源发布/发出，显式等待段为返回减进入。",
        "\n| 配置 | 进程 | 窗口均值 | 单位 | CV | 状态 |",
        "|---|---:|---:|---|---:|---|",
    ]
    rules = []
    for c in cases:
        rows = groups[c["id"]]
        if not rows:
            continue
        times = [r["elapsed"] for r in rows]
        mean = statistics.mean(times)
        cv = statistics.stdev(times) / mean if len(times) > 1 else None
        rule = dict(
            configuration=c,
            processes=len(rows),
            mean_window=mean,
            unit=rows[0]["unit"],
            cv=cv,
            status=(
                "short_check"
                if env["mode"] == "smoke"
                else (
                    "observed"
                    if cv is not None and cv <= 0.05 and all(r["warmup_converged"] for r in rows)
                    else "unstable"
                )
            ),
            gpu_uuid=env["gpu_uuid"],
        )
        work_row = rows[0]
        per_sequence = 8 if c["family"] == "B" else 1
        iterations = work_row["iterations"]
        active_filler_lanes = 1 if c["protocol"] == "tout" else 128
        rule["work_counts_per_process"] = dict(
            requested_bytes=work_row["work_bytes"],
            wgmma_instructions=iterations if c["protocol"] == "wg" and not c["control"] else 0,
            wgmma_flop=(
                2 * 64 * 256 * 16 * iterations if c["protocol"] == "wg" and not c["control"] else 0
            ),
            consumer_flop=work_row["consumer_flop"],
            filler_thread_ops=active_filler_lanes * c["filler"] * iterations * per_sequence,
            filler_warp_instructions=(1 if c["protocol"] == "tout" else 4)
            * c["filler"]
            * iterations
            * per_sequence,
        )
        if c["family"] == "A" and rows[0].get("filler_instruction") != "dependent_IMAD_LCG":
            rule["status"] = "invalid_sass_schedule"
            rule["exclusion_reason"] = (
                "early WGMMA wait / contracted integer additions; A replaced by corrected separate run"
            )
        rule["end_event"] = rows[0].get(
            "end_event",
            (
                "bulk_full_completion_and_CTA_join"
                if c["family"] in ("C", "E") or c["protocol"] == "tout"
                else (
                    "acquire_and_consumer_release_join"
                    if c["family"] == "D"
                    else (
                        "covered_WGMMA_wait_and_accumulator_sink"
                        if c["protocol"] == "wg"
                        else "input_acquire_consumer_sink_and_CTA_join"
                    )
                )
            ),
        )
        if c["family"] == "D":
            rates = [49152 * r["iterations"] / (s[1] - s[0]) for r in rows for s in r["stamps"]]
            rule.update(
                cta_rate_median=statistics.median(rates),
                cta_rate_range=[min(rates), max(rates)],
                actual_sm_coverage_per_process=[len(set(s[4] for s in r["stamps"])) for r in rows],
                gb_s=[r["work_bytes"] / r["elapsed"] for r in rows],
                demand_bytes_per_cycle=48,
                source_visited_bytes=49152
                * min(rows[0]["slots"], rows[0]["iterations"])
                * (rows[0]["blocks"] if c["source"] == "independent" else 1),
                revisit_ktile_count=rows[0]["slots"],
            )
        if c["family"] == "A":
            rule.update(
                mean_wait_position_cycles=statistics.mean(
                    r["stamps"][0][5] / r["iterations"] for r in rows
                ),
                mean_wait_return_cycles=statistics.mean(
                    r["stamps"][0][6] / r["iterations"] for r in rows
                ),
            )
        if c["family"] == "C":
            rule.update(
                last_source_release_cycles=[r["stamps"][0][7] for r in rows],
                last_full_completion_cycles=[r["stamps"][0][8] for r in rows],
            )
        rules.append(rule)
        report.append(
            f"| {c['id']} | {len(rows)} | {mean:.2f} | {rule['unit']} | {cv if cv is not None else 'NA'} | {rule['status']} |"
        )
    for r in rules:
        if r["configuration"]["family"] == "D":
            report.append(
                f"\n`{r['configuration']['id']}`：CTA供给中位数 {r['cta_rate_median']:.3f} B/cycle，范围 {r['cta_rate_range']}，实际SM覆盖 {r['actual_sm_coverage_per_process']}，实际访问 {r['source_visited_bytes']} B，复访 {r['revisit_ktile_count']} Ktile。"
            )
    if any(r["status"] == "invalid_sass_schedule" for r in rules):
        report.append(
            "\n原A组因SASS等待提前或整数加法合并而失效；本报告保留原始样本，只使用另一个修正版A归档。"
        )
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    for family in ("A", "B", "E"):
        chosen = [
            r
            for r in rules
            if r["configuration"]["family"] == family
            and not r["configuration"]["control"]
            and r["status"] != "invalid_sass_schedule"
        ]
        if not chosen:
            continue
        xmax = 512 if family == "A" else 4 if family == "B" else 256
        ymax = (
            max(r["mean_window"] / r["configuration"].get("normalization", 1) for r in chosen) * 1.1
        )
        svg = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="950" height="480" viewBox="0 0 950 480">',
            '<rect width="950" height="480" fill="white"/>',
            f'<text x="40" y="30" font-size="18">R05-{family} complete window ({chosen[0]["unit"]})</text>',
        ]
        for tick in range(6):
            y = 410 - tick * 65
            value = ymax * tick / 5
            svg.append(
                f'<path d="M90 {y} H650" stroke="#ddd"/><text x="5" y="{y+4}" font-size="12">{value:.0f}</text>'
            )
        series = {}
        for r in chosen:
            c = r["configuration"]
            key = (
                c["protocol"]
                if family == "A"
                else c["protocol"] + "_c" + str(c["consumer"]) if family == "B" else "pitch"
            )
            x = c["filler"] if family == "A" else c["stages"] if family == "B" else c["pitch"]
            series.setdefault(key, []).append((x, r["mean_window"]))
        colors = ["#2563eb", "#dc2626", "#059669", "#9333ea"]
        for i, (key, points) in enumerate(series.items()):
            points.sort()
            coords = " ".join(f"{90+x/xmax*560},{410-value/ymax*325}" for x, value in points)
            svg.append(
                f'<polyline points="{coords}" fill="none" stroke="{colors[i%4]}" stroke-width="2"/>'
            )
            svg.append(
                f'<text x="680" y="{70+30*i}" fill="{colors[i%4]}" font-size="14">{key}</text>'
            )
            for x, value in points:
                svg.append(
                    f'<circle cx="{90+x/xmax*560}" cy="{410-value/ymax*325}" r="3" fill="{colors[i%4]}"/><text x="{90+x/xmax*560}" y="440" font-size="12">{x}</text>'
                )
        svg.append("</svg>")
        (plots / f"{family.lower()}-window.svg").write_text("\n".join(svg))
        report.append(f"\n![R05-{family}窗口](plots/{family.lower()}-window.svg)")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
            for key, points in series.items():
                ax.plot([v[0] for v in points], [v[1] for v in points], marker="o", label=key)
            ax.set(
                xlabel=(
                    "Dependent IMAD operations"
                    if family == "A"
                    else "Active stages" if family == "B" else "Global row pitch (B)"
                ),
                ylabel=chosen[0]["unit"],
                title=f"GH200 R05-{family}: complete window",
            )
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.savefig(plots / f"{family.lower()}-window.svg")
            fig.savefig(plots / f"{family.lower()}-window.png", dpi=150)
            plt.close(fig)
            if family == "A":
                fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
                for key in series:
                    selected = sorted(
                        [r for r in chosen if r["configuration"]["protocol"] == key],
                        key=lambda r: r["configuration"]["filler"],
                    )
                    xs = [r["configuration"]["filler"] for r in selected]
                    line = ax.plot(
                        xs,
                        [r["mean_wait_position_cycles"] for r in selected],
                        marker="o",
                        label=key + " wait entered",
                    )[0]
                    ax.plot(
                        xs,
                        [r["mean_wait_return_cycles"] for r in selected],
                        marker="x",
                        linestyle="--",
                        color=line.get_color(),
                        label=key + " wait returned",
                    )
                ax.set(
                    xlabel="Dependent IMAD operations",
                    ylabel="Mean cycles from request preparation",
                    title="GH200 R05-A: wait entry and return",
                )
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7, ncol=2)
                fig.savefig(plots / "a-wait-interval.svg")
                fig.savefig(plots / "a-wait-interval.png", dpi=150)
                plt.close(fig)
                report.append("\n![A等待位置与返回](plots/a-wait-interval.svg)")
        except ImportError:
            pass
    with (root / "cases.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "processes", "mean_window", "unit", "cv", "status"])
        for r in rules:
            writer.writerow(
                [
                    r["configuration"]["id"],
                    r["processes"],
                    r["mean_window"],
                    r["unit"],
                    r["cv"],
                    r["status"],
                ]
            )
    (root / "rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    (root / "report.md").write_text("\n".join(report) + "\n")
    print(root / "report.md")


if __name__ == "__main__":
    main()
