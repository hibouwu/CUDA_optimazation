#!/usr/bin/env python3
import csv
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "l1_bandwidth"
RESULTS = ROOT / "results"
MODES = ["read-ca", "read-cg", "write-wb", "write-cg"]


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout


def header():
    return run([str(BIN), "--csv-header"]).strip().split(",")


def run_row(mode, bytes_per_cta):
    out = run([
        str(BIN),
        "--mode", mode,
        "--bytes-per-cta", str(bytes_per_cta),
        "--iters", "4096",
        "--warmup-rounds", "2",
        "--threads", "256",
        "--csv",
    ]).strip()
    return next(csv.DictReader([",".join(header()), out]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_design_review():
    text = """# L1 设计对抗式审查

## 攻击点

1. **`.ca` 读可能没有 L1 hit。**

   设计响应：每个 CTA 的 working set 默认会提升到 32 KiB，以覆盖 256 线程 * 8 unroll 的首轮唯一访问，并在同一个 kernel 中先用 `.ca` 预热；NCU 必须验证 L1TEX lookup-hit bytes 接近期望请求，且 LTS bytes 显著低于逻辑请求。

2. **block 不一定一一分布到 SM。**

   设计响应：默认 blocks=SM count；运行报告记录 blocks 和 SM count。若 NCU 显示 waves/SM 异常或 L1 hit 不成立，则该结论不通过。

3. **写路径不能叫纯 L1 cache write bandwidth。**

   设计响应：`write-wb` 只解释为 L1TEX global-store front-end / end-to-end store path。

4. **编译器可能改变 cache op 或访问宽度。**

   设计响应：使用 inline PTX `ld.global.ca/cg.v4.u32` 和 `st.global.wb/cg.v4.u32`；SASS 摘要必须显示 128-bit global ops。

## 设计结论

设计可以运行；通过条件是 NCU hit/miss/traffic 支持 `read-ca` 是 L1-hit，SASS 保持 128-bit op，并且写路径按 store path 限定表述。
"""
    (RESULTS / "design_adversarial_review.md").write_text(text)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_design_review()
    rows = []
    for bytes_per_cta in [4096, 8192, 16384, 32768, 65536]:
        for mode in MODES:
            for _ in range(3):
                rows.append(run_row(mode, bytes_per_cta))
    write_csv(RESULTS / "l1_capacity_sweep.csv", rows)

    summary = []
    groups = {}
    for row in rows:
        groups.setdefault((row["mode"], row["bytes_per_cta"]), []).append(float(row["bytes_per_cycle"]))
    for (mode, bytes_per_cta), values in sorted(groups.items()):
        summary.append({
            "mode": mode,
            "bytes_per_cta": bytes_per_cta,
            "bytes_per_cycle_median": f"{statistics.median(values):.6f}",
            "bytes_per_cycle_min": f"{min(values):.6f}",
            "bytes_per_cycle_max": f"{max(values):.6f}",
            "repeat_count": str(len(values)),
        })
    write_csv(RESULTS / "l1_capacity_sweep_summary.csv", summary)

    base = {r["mode"]: r for r in summary if r["bytes_per_cta"] == "32768"}
    lines = [
        "# L1 validation report",
        "",
        "## 32 KiB per CTA baseline",
        "",
        "|mode|median B/cycle|range B/cycle|",
        "|---|---:|---:|",
    ]
    for mode in MODES:
        r = base[mode]
        lines.append(f"|{mode}|{float(r['bytes_per_cycle_median']):.3f}|{float(r['bytes_per_cycle_min']):.3f}-{float(r['bytes_per_cycle_max']):.3f}|")
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")

    review = [
        "# L1 运行对抗式审查",
        "",
        "未通过：app sweep 已完成，但还需要 NCU 验证 L1TEX hit/miss、LTS traffic 和 SASS cache op 后才能通过。",
    ]
    (RESULTS / "adversarial_review.md").write_text("\n".join(review) + "\n")
    print(f"Wrote {RESULTS / 'l1_capacity_sweep.csv'}")


if __name__ == "__main__":
    main()
