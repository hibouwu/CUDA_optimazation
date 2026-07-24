#!/usr/bin/env python3
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "dsmem_bandwidth"
RESULTS = ROOT / "results"


MODES = ["local-read", "local-write", "remote-read", "remote-write"]


def run_cmd(args, **kwargs):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, **kwargs).stdout


def run_row(mode, cluster_size, clusters=None, iters=4096, threads=256, shared_bytes=32768):
    cmd = [
        str(BIN),
        "--mode", mode,
        "--cluster-size", str(cluster_size),
        "--iters", str(iters),
        "--warmup-iters", "64",
        "--threads", str(threads),
        "--shared-bytes", str(shared_bytes),
        "--csv",
    ]
    if clusters is not None:
        cmd.extend(["--clusters", str(clusters)])
    out = run_cmd(cmd)
    header = run_cmd([str(BIN), "--csv-header"]).strip().split(",")
    row = next(csv.DictReader([",".join(header), out.strip()]))
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["mode"], row["cluster_size"], row["clusters"])
        grouped.setdefault(key, []).append(float(row["bytes_per_cycle"]))
    summary = []
    for (mode, cluster_size, clusters), values in sorted(grouped.items()):
        summary.append({
            "mode": mode,
            "cluster_size": cluster_size,
            "clusters": clusters,
            "bytes_per_cycle_median": f"{statistics.median(values):.6f}",
            "bytes_per_cycle_min": f"{min(values):.6f}",
            "bytes_per_cycle_max": f"{max(values):.6f}",
            "repeat_count": str(len(values)),
        })
    return summary


def write_design_review():
    text = """# DSMEM 设计对抗式审查

## 目标

测 local shared 与同 cluster 内 remote DSMEM 的 read/write 请求吞吐。每个线程每次发 16 B `uint4` 请求，报告全 GPU 请求字节 / 最大 CTA `clock64()` 周期。

## 主要攻击点与设计响应

1. **多 wave launch 会高估吞吐。**

   设计响应：默认 `--clusters 0` 使用保守的一 CTA/SM 规则，即 `SM_count / cluster_size`；超过该值会被拒绝，除非显式 `--allow-waves`。CUDA 返回的 `cudaOccupancyMaxActiveClusters` 只记录，不作为默认，因为 remote DSMEM 在多 CTA/SM 下会进入多 wave 或失败区。

2. **remote 访问可能退化成本地 shared 访问。**

   设计响应：remote 模式强制 `cluster_size >= 2`，并用 `cluster.map_shared_rank(smem, (rank + 1) % cluster_size)` 映射邻居 CTA 的 dynamic shared memory。运行后必须用 NCU `mem_dshared` byte counters 验证。

3. **shared load/store 可能被 scalarize，导致 16 B/op 统计不干净。**

   设计响应：第一版使用 volatile `uint4` 访问并保留 SASS 摘要；运行审查必须确认是否存在宽 LDS/STS 或解释 scalarization。若 SASS 不满足 128-bit 请求假设，需要改 PTX 或重定义 bytes/op 后重跑。

4. **写入路径没有等完成就停表。**

   设计响应：`local-write` 在 stop 前 `__syncthreads()`；`remote-write` 在 stop 前 `cluster.sync()`，因此报告的是包含 completion boundary 的端到端 store-path throughput。

5. **local shared 和 remote DSMEM 的 NCU 计数能力不同。**

   设计响应：remote DSMEM 使用 `l1tex__t_bytes_pipe_lsu_mem_dshared*` 计算利用率和上限；local shared 若本机没有 byte counter，用 `l1tex__data_pipe_lsu_wavefronts_mem_shared*` 和 bank conflict 作为利用率 proxy，不写成 NCU byte-verified bandwidth。

## 设计审查结论

设计可以进入第一轮运行，但运行审查必须检查 SASS 宽度、NCU dshared bytes/expected、occupancy、bank conflicts 和 write completion 边界。
"""
    (RESULTS / "design_adversarial_review.md").write_text(text)


def write_report(summary_rows):
    by_mode = {row["mode"]: row for row in summary_rows if row["cluster_size"] == "2" and row["clusters"] == "10"}
    lines = [
        "# DSMEM validation report",
        "",
        "## Cluster size 2 full-GPU baseline",
        "",
        "|mode|median B/cycle|range B/cycle|repeats|",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        row = by_mode.get(mode)
        if row:
            lines.append(
                f"|{mode}|{float(row['bytes_per_cycle_median']):.3f}|"
                f"{float(row['bytes_per_cycle_min']):.3f}-{float(row['bytes_per_cycle_max']):.3f}|"
                f"{row['repeat_count']}|"
            )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `dsmem_bandwidth.csv`",
        "- `dsmem_cluster_sweep.csv`",
        "- `dsmem_cluster_sweep_summary.csv`",
        "- `sass_summary.txt`",
        "- `design_adversarial_review.md`",
        "- `adversarial_review.md`",
    ])
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")


def write_run_review(summary_rows):
    by_mode = {row["mode"]: row for row in summary_rows if row["cluster_size"] == "2" and row["clusters"] == "10"}
    remote_read = by_mode.get("remote-read", {})
    remote_write = by_mode.get("remote-write", {})
    local_read = by_mode.get("local-read", {})
    local_write = by_mode.get("local-write", {})
    sass = (RESULTS / "sass_summary.txt").read_text() if (RESULTS / "sass_summary.txt").exists() else ""
    has_lds = "LDS" in sass
    has_sts = "STS" in sass
    text = f"""# DSMEM 运行对抗式审查

## 当前结果

cluster_size=2, clusters=10 full-GPU baseline:

|模式|median B/cycle|
|---|---:|
|local-read|{float(local_read.get('bytes_per_cycle_median', 0.0)):.3f}|
|local-write|{float(local_write.get('bytes_per_cycle_median', 0.0)):.3f}|
|remote-read|{float(remote_read.get('bytes_per_cycle_median', 0.0)):.3f}|
|remote-write|{float(remote_write.get('bytes_per_cycle_median', 0.0)):.3f}|

## 审查

1. **occupancy / 多 wave 风险**

   CSV 记录 `resident_cluster_limit`、`cuda_max_active_clusters`、`clusters` 和 `occupancy_limited`。默认运行没有 `--allow-waves`，如果 `occupancy_limited=0`，per-CTA `clock64()` 不应因多 wave 高估。

2. **remote 是否真走 DSMEM**

   app 侧只能证明使用了 `map_shared_rank`；最终必须由 NCU `mem_dshared` byte counters 证明。没有 NCU 前，remote 数字只能作为候选结果。

3. **SASS load/store 宽度**

   SASS 摘要中 `LDS` present = `{has_lds}`，`STS` present = `{has_sts}`。需要人工检查是否为期望的 vector 宽度；若 scalarized，则必须修改或把 bytes/op 改成实际请求。

4. **写入完成边界**

   `remote-write` stop 前有 `cluster.sync()`；`local-write` stop 前有 `__syncthreads()`。这些数字是端到端 store-path，不是纯端口峰值。

## 当前审查状态

未通过：还需要 NCU dshared byte counter 验证和 SASS 宽度确认。通过条件是 NCU summary 显示 remote-read/remote-write 的 dshared bytes 接近期望请求字节，且 SASS/计数器没有推翻 16 B/op 假设。
"""
    (RESULTS / "adversarial_review.md").write_text(text)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_design_review()

    rows = []
    for cluster_size, clusters_list in [(2, [1, 2, 4, 8, 10])]:
        for clusters in clusters_list:
            for mode in MODES:
                for _ in range(3):
                    rows.append(run_row(mode, cluster_size, clusters=clusters))

    write_csv(RESULTS / "dsmem_cluster_sweep.csv", rows)
    summary_rows = summarize(rows)
    write_csv(RESULTS / "dsmem_cluster_sweep_summary.csv", summary_rows)
    write_report(summary_rows)
    write_run_review(summary_rows)
    print(f"Wrote {RESULTS / 'dsmem_cluster_sweep.csv'}")


if __name__ == "__main__":
    main()
