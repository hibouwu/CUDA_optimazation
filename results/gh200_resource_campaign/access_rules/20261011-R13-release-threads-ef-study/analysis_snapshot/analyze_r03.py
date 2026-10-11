#!/usr/bin/env python3
"""R03 offline exact-output and service-rate recomputation (no GPU)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct


def check(record, path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["output_sha256"]:
        raise ValueError("output hash changed")
    values = struct.unpack("<" + "I" * (len(raw) // 4), raw)
    op, width, warps, count, seed = [
        record[k] for k in ("op", "width", "warps", "iterations", "seed")
    ]
    words = 128 if op >= 2 else 32 * width // 4
    if len(values) != (warps * 32 if op in (0, 2) else warps * 8 * words):
        raise ValueError("output length")
    for index, value in enumerate(values):
        if op in (1, 3):
            expected = (17 * index + 31 * count + seed) & 0xFFFFFFFF
        else:
            warp, lane = divmod(index, 32)
            expected = 0
            # Each round visits every physical slot once; same sum for all rounds.
            for slot in range(8):
                for reg in range(4 if op == 2 else width // 4):
                    word = (warp * 8 + slot) * words + (
                        lane // 4 * 4 + lane % 4 + reg * 32
                        if op == 2
                        else lane * (width // 4) + reg
                    )
                    expected += 17 * word + seed
            expected = expected * count & 0xFFFFFFFF
        if expected != value:
            raise ValueError(f"exact check {path} index {index}")
    if record["work_bytes"] != words * 4 * warps * 8 * count:
        raise ValueError("work count")
    return len(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--sectors", help="JSON lane byte addresses")
    p.add_argument("--access-width", type=int, default=4)
    a = p.parse_args()
    if a.sectors:
        addresses = json.loads(a.sectors)
        covered = sorted(
            {
                sector
                for address in addresses
                for sector in range(address // 32, (address + a.access_width - 1) // 32 + 1)
            }
        )
        print(json.dumps(dict(sectors=covered, count=len(covered))))
        return
    if a.input is None:
        p.error("--input or --sectors required")
    root = a.input
    for relative, digest in json.loads((root / "source_hashes.json").read_text()).items():
        if hashlib.sha256((root / "source" / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("archived source hash mismatch")
    if hashlib.sha256((root / "build/r03").read_bytes()).hexdigest() != json.loads(
        (root / "build/binary_hash.json").read_text()
    ):
        raise ValueError("binary hash mismatch")
    env = json.loads((root / "environment.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    groups = {c["id"]: [] for c in cases}
    checked = 0
    for f in sorted((root / "samples").glob("*/*/result.json")):
        r = json.loads(f.read_text())
        checked += check(r, f.parent / "output.u32")
        if r["role"].startswith("trial-"):
            groups[r["case_id"]].append(r)
    rules = []
    report = [
        "# R03 片上访问并发",
        f"\nGPU `{env['gpu_uuid']}`，模式 `{env['mode']}`。CPU逐项重算通过 {checked} 个输出。",
        "\n窗口包含8次访问/轮、load归约、地址更新和CTA会合；B/cycle为请求字节/CTA clock64，不归因为端口或bank速率。",
        "\n| 配置 | 进程 | B/cycle均值 | CV | 末5次预热收敛 |",
        "|---|---:|---:|---:|---|",
    ]
    for c in cases:
        rows = groups[c["id"]]
        rates = [r["work_bytes"] / r["elapsed"] for r in rows]
        mean = statistics.mean(rates)
        cv = statistics.stdev(rates) / mean if len(rates) > 1 else None
        rule = dict(
            configuration=c,
            processes=len(rows),
            mean_requested_bytes_per_cycle=mean,
            cv=cv,
            status=(
                "observed"
                if cv is not None and cv <= 0.05 and all(r["warmup_converged"] for r in rows)
                else "short_check" if env["mode"] == "smoke" else "unstable"
            ),
            end_event=rows[0]["end_event"],
            gpu_uuid=env["gpu_uuid"],
        )
        rules.append(rule)
        report.append(
            f"| {c['id']} | {len(rows)} | {mean:.3f} | {cv if cv is not None else 'NA'} | {all(r['warmup_converged'] for r in rows)} |"
        )
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2"]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">',
        '<rect width="900" height="500" fill="white"/>',
        '<text x="50" y="28" font-size="18">R03 requested bytes / CTA clock64 cycle</text>',
    ]
    for y in range(0, 151, 25):
        py = 430 - y * 2.5
        svg.append(
            f'<path d="M60 {py} H620" stroke="#ddd"/><text x="20" y="{py+4}" font-size="12">{y}</text>'
        )
    for warp in (1, 4, 8):
        x = 60 + (warp - 1) * 80
        svg.append(f'<text x="{x}" y="455" font-size="12">{warp} warps</text>')
    series = {}
    for rule in rules:
        c = rule["configuration"]
        series.setdefault((c["op"], c["width"]), []).append(
            (c["warps"], rule["mean_requested_bytes_per_cycle"])
        )
    for i, ((op, width), points) in enumerate(series.items()):
        coords = " ".join(f"{60+(w-1)*80},{430-rate*2.5}" for w, rate in points)
        color = colors[i]
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.append(
            f'<text x="650" y="{75+i*30}" fill="{color}" font-size="14">{("load","store","ldmatrix","stmatrix")[op]} {width} B/lane</text>'
        )
    svg.append("</svg>")
    (plots / "service.svg").write_text("\n".join(svg))
    report.append("\n![R03请求工作率](plots/service.svg)")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
        for (op, width), points in series.items():
            selected = [
                r
                for r in rules
                if r["configuration"]["op"] == op and r["configuration"]["width"] == width
            ]
            ax.errorbar(
                [v[0] for v in points],
                [v[1] for v in points],
                yerr=[r["mean_requested_bytes_per_cycle"] * (r["cv"] or 0) for r in selected],
                marker="o",
                capsize=3,
                label=f'{("load","store","ldmatrix.x4","stmatrix.x4")[op]} {width}B/lane',
            )
        ax.set(
            xlabel="Participating warps / CTA",
            ylabel="Requested B / CTA clock64 cycle",
            xticks=[1, 4, 8],
            title="GH200 R03: access + consumer + CTA join",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(plots / "service.svg")
        fig.savefig(plots / "service.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    (root / "rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    (root / "report.md").write_text("\n".join(report) + "\n")
    with (root / "cases.csv").open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["id", "processes", "mean_requested_bytes_per_cycle", "cv", "status"]
        )
        w.writeheader()
        for r in rules:
            w.writerow(dict(id=r["configuration"]["id"], **{k: r[k] for k in w.fieldnames[1:]}))
    print(root / "report.md")


if __name__ == "__main__":
    main()
