#!/usr/bin/env python3
import csv
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "gmem_dram_bandwidth"
RESULTS = ROOT / "results"
MODES = ["read-stream", "write-stream", "copy-stream"]


def run(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout


def header():
    return run([str(BIN), "--csv-header"]).strip().split(",")


def run_row(mode, bytes_):
    out = run([
        str(BIN), "--mode", mode, "--bytes", str(bytes_),
        "--iters", "4096", "--warmup-iters", "32",
        "--blocks-per-sm", "4", "--threads", "256", "--csv",
    ]).strip()
    return next(csv.DictReader([",".join(header()), out]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "design_adversarial_review.md").write_text("""# GMEM/DRAM 设计对抗式审查

## 攻击点

1. **工作集不够大导致 L2 hit。**

   设计响应：正式 baseline 使用 256 MiB，远大于 32 MiB L2，并报告 64/128/256/512 MiB 容量 sweep。NCU 必须检查 LTS miss-sector 或 DRAM bytes。

2. **L1 影响读结果。**

   设计响应：所有 global load/store 使用 `.cg`，SASS 必须显示 `STRONG.GPU` 128-bit op。

3. **写路径可能只停在 L2。**

   设计响应：write/copy stop 前执行 `__threadfence()`，并用大工作集制造 dirty eviction。若无 `dram__bytes*`，只能以 LTS write miss/sector 作为 DRAM-path proxy，不能声称直接 DRAM byte counter 已验证。

4. **多 wave 计时高估。**

   设计响应：使用 occupancy 检查，默认 blocks/SM=4 不超过可驻留上限，grid 为单 resident wave。
""")

    rows = []
    for bytes_ in [64 << 20, 128 << 20, 256 << 20, 512 << 20]:
        for mode in MODES:
            for _ in range(3):
                rows.append(run_row(mode, bytes_))
    write_csv(RESULTS / "gmem_capacity_sweep.csv", rows)

    summary = []
    groups = {}
    for row in rows:
        groups.setdefault((row["mode"], row["working_set_mib"]), []).append(float(row["bytes_per_cycle"]))
    for (mode, mib), values in sorted(groups.items()):
        summary.append({
            "mode": mode,
            "working_set_mib": mib,
            "bytes_per_cycle_median": f"{statistics.median(values):.6f}",
            "bytes_per_cycle_min": f"{min(values):.6f}",
            "bytes_per_cycle_max": f"{max(values):.6f}",
            "repeat_count": str(len(values)),
        })
    write_csv(RESULTS / "gmem_capacity_sweep_summary.csv", summary)

    base = {r["mode"]: r for r in summary if r["working_set_mib"] == "256.000"}
    lines = ["# GMEM/DRAM validation report", "", "## 256 MiB baseline", "", "|mode|median B/cycle|range B/cycle|", "|---|---:|---:|"]
    for mode in MODES:
        r = base[mode]
        lines.append(f"|{mode}|{float(r['bytes_per_cycle_median']):.3f}|{float(r['bytes_per_cycle_min']):.3f}-{float(r['bytes_per_cycle_max']):.3f}|")
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "adversarial_review.md").write_text("# GMEM/DRAM 运行对抗式审查\n\n未通过：app sweep 已完成，但还需要 NCU LTS/DRAM traffic 验证。\n")
    print(f"Wrote {RESULTS / 'gmem_capacity_sweep.csv'}")


if __name__ == "__main__":
    main()
