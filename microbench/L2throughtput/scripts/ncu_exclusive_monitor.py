#!/usr/bin/env python3
import argparse
import csv
import fcntl
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "l2_throughput"
BUILD_SCRIPT = ROOT / "build_and_run.sh"
RESULT_DIR = ROOT / "results" / "ncu"
LOCK_PATH = RESULT_DIR / "ncu_exclusive_monitor.lock"

MODES = ("read-unique", "write-unique")
CANDIDATE_METRICS = [
    "gpu__time_duration.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st_lookup_hit.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st_lookup_miss.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st.sum",
    "lts__t_bytes.sum",
    "lts__t_bytes.sum.per_second",
    "lts__t_bytes.sum.per_cycle_elapsed",
    "lts__t_bytes.sum.pct_of_peak_sustained_elapsed",
    "lts__t_request_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.max.pct_of_peak_sustained_elapsed",
    "lts__throughput.sum.pct_of_peak_sustained_elapsed",
    "lts__t_requests_lookup_hit.sum",
    "lts__t_requests_lookup_miss.sum",
    "lts__t_requests_op_read.sum",
    "lts__t_requests_op_read_lookup_hit.sum",
    "lts__t_requests_op_read_lookup_miss.sum",
    "lts__t_requests_op_write.sum",
    "lts__t_requests_op_write_lookup_hit.sum",
    "lts__t_requests_op_write_lookup_miss.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_read.sum.per_second",
    "lts__t_sectors_op_read.sum.per_cycle_elapsed",
    "lts__t_sectors_op_read.sum.pct_of_peak_sustained_elapsed",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_write.sum.per_second",
    "lts__t_sectors_op_write.sum.per_cycle_elapsed",
    "lts__t_sectors_op_write.sum.pct_of_peak_sustained_elapsed",
    "lts__t_sectors_op_write_lookup_hit.sum",
    "lts__t_sectors_op_write_lookup_miss.sum",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
]


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run(cmd, *, cwd=ROOT, check=True):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{proc.stdout}"
        )
    return proc


def append_log(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def build_binary():
    run([str(BUILD_SCRIPT), "build-only"])


def ps_rows():
    proc = run(["ps", "-eo", "pid,ppid,user,stat,comm,args"], check=False)
    return proc.stdout.splitlines()[1:]


def profiler_blockers():
    blockers = []
    self_pid = os.getpid()
    parent_pid = os.getppid()
    profiler_execs = {
        "ncu",
        "nvprof",
        "nv-nsight-cu-cli",
        "nsys",
        "dcgmi",
        "nv-hostengine",
    }
    profiler_scripts = {
        "ncu_profile.py",
    }
    for line in ps_rows():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid = int(parts[0])
        if pid in (self_pid, parent_pid):
            continue
        comm = parts[4]
        args = parts[5]
        if "ncu_exclusive_monitor.py" in args:
            continue
        tokens = [token.strip("'\"") for token in args.split()]
        basenames = [os.path.basename(token) for token in tokens[:4]]
        is_profiler = (
            comm in profiler_execs or
            any(name in profiler_execs for name in basenames) or
            any(name in profiler_scripts for name in basenames)
        )
        if is_profiler:
            blockers.append(line)
    return blockers


def compute_blockers():
    proc = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    blockers = []
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line or "No running processes found" in line:
            continue
        blockers.append(line)
    return blockers


def exclusive_blockers():
    blockers = []
    profilers = profiler_blockers()
    if profilers:
        blockers.append("profiler processes:\n" + "\n".join(profilers))
    compute = compute_blockers()
    if compute:
        blockers.append("compute processes:\n" + "\n".join(compute))
    return blockers


def app_probe(mode, args):
    cmd = [
        str(BIN),
        "--mode", mode,
        "--iters", str(args.iters),
        "--warmup-iters", str(args.warmup_iters),
        "--blocks-per-sm", str(args.blocks_per_sm),
        "--threads", str(args.threads),
        "--bytes", str(args.bytes),
        "--csv",
    ]
    proc = run(cmd)
    rows = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"unexpected app probe output for {mode}:\n{proc.stdout}")
    header = run([str(BIN), "--csv-header"]).stdout.strip()
    return next(csv.DictReader([header, rows[0]]))


def query_supported_metrics():
    proc = run(
        [
            "ncu",
            "--query-metrics",
            "--query-metrics-mode",
            "all",
            "--query-metrics-collection",
            "profiling",
        ],
        check=False,
    )
    if proc.returncode != 0:
        return None, proc.stdout

    supported = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("Device "):
            continue
        token = stripped.split(None, 1)[0]
        if re.match(r"^(gpu__|l1tex__|lts__|dram__)", token):
            supported.add(token)
    return supported, proc.stdout


def ncu_cmd(mode, args, report_base, metrics):
    return [
        "ncu",
        "--csv",
        "--page", "raw",
        "--cache-control", "none",
        "--clock-control", "none",
        "--launch-skip", "2",
        "--launch-count", "1",
        "--metrics", ",".join(metrics),
        "--force-overwrite",
        "-o", str(report_base),
        str(BIN),
        "--mode", mode,
        "--iters", str(args.iters),
        "--warmup-iters", str(args.warmup_iters),
        "--blocks-per-sm", str(args.blocks_per_sm),
        "--threads", str(args.threads),
        "--bytes", str(args.bytes),
        "--csv",
    ]


def parse_metric_csv(text):
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            header_idx = i
            break
    if header_idx is None:
        return parse_wide_metric_csv(lines)

    metrics = {}
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        name = row.get("Metric Name")
        value = row.get("Metric Value")
        if not name or value is None:
            continue
        unit = row.get("Metric Unit", "")
        clean = value.replace(",", "").replace("%", "").strip()
        try:
            parsed = float(clean)
        except ValueError:
            continue
        metrics[name] = {"value": parsed, "unit": unit}
    if metrics:
        return metrics

    return parse_wide_metric_csv(lines)


def parse_wide_metric_csv(lines):
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('"ID","Process ID"') or stripped.startswith("ID,Process ID"):
            header_idx = i
            break
    if header_idx is None or header_idx + 2 >= len(lines):
        return {}

    try:
        header = next(csv.reader([lines[header_idx]]))
        units = next(csv.reader([lines[header_idx + 1]]))
        values = next(csv.reader([lines[header_idx + 2]]))
    except csv.Error:
        return {}

    metrics = {}
    for idx, name in enumerate(header):
        if not re.match(r"^(gpu__|l1tex__|lts__|dram__)", name):
            continue
        if idx >= len(values):
            continue
        clean = values[idx].replace(",", "").replace("%", "").strip()
        if clean == "":
            continue
        try:
            parsed = float(clean)
        except ValueError:
            continue
        unit = units[idx] if idx < len(units) else ""
        metrics[name] = {"value": parsed, "unit": unit}
    return metrics


def metric(metrics, name, default=0.0):
    item = metrics.get(name)
    return default if item is None else item["value"]


def metric_or_none(metrics, name):
    item = metrics.get(name)
    return None if item is None else item["value"]


def ratio(value, expected):
    if value is None or not expected:
        return None
    return value / expected


def fmt(value, digits=3):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def run_ncu_once(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    supported, query_text = query_supported_metrics()
    query_log = RESULT_DIR / "query_metrics.log"
    if supported is None:
        selected_metrics = list(CANDIDATE_METRICS)
        query_log.write_text(
            "NCU metric 查询失败；将使用全部候选 metrics。\n\n"
            + query_text
        )
    else:
        selected_metrics = [name for name in CANDIDATE_METRICS if name in supported]
        missing_candidates = [
            name for name in CANDIDATE_METRICS
            if name not in supported
        ]
        query_log.write_text(
            "NCU metric 查询成功。\n\n"
            "支持的候选 metrics：\n"
            + "\n".join(f"- {name}" for name in selected_metrics)
            + "\n\n缺失的候选 metrics：\n"
            + "\n".join(f"- {name}" for name in missing_candidates)
            + "\n"
        )
    if not selected_metrics:
        raise RuntimeError(
            "NCU query succeeded but none of the candidate metrics were supported; "
            f"query log: {query_log}"
        )

    app_rows = {}
    metric_rows = {}
    for mode in MODES:
        blockers = exclusive_blockers()
        if blockers:
            raise RuntimeError(
                "exclusive pre-check failed before NCU run:\n" + "\n\n".join(blockers)
            )

        app_rows[mode] = app_probe(mode, args)
        blockers = exclusive_blockers()
        if blockers:
            raise RuntimeError(
                "exclusive pre-check failed after app probe and before NCU run:\n"
                + "\n\n".join(blockers)
            )
        report_base = RESULT_DIR / f"{mode.replace('-', '_')}_validation"
        cmd = ncu_cmd(mode, args, report_base, selected_metrics)
        proc = run(cmd, check=False)
        log_path = RESULT_DIR / f"{mode.replace('-', '_')}_ncu_raw.csv"
        log_path.write_text(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"NCU failed for {mode} with rc={proc.returncode}; raw log: {log_path}\n"
                f"{proc.stdout}"
            )
        collected_metrics = parse_metric_csv(proc.stdout)
        if not collected_metrics:
            raise RuntimeError(f"failed to parse NCU metrics for {mode}; raw log: {log_path}")
        metric_rows[mode] = collected_metrics
    return app_rows, metric_rows


def write_summary(app_rows, metric_rows, idle_confirm_seconds=None, idle_age=None):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = RESULT_DIR / "ncu_l2_validation_summary.csv"
    fields = [
        "mode",
        "expected_requested_bytes",
        "clock_bytes_per_cycle",
        "gpu_time_ns",
        "l1tex_global_ld_bytes",
        "l1tex_global_ld_miss_bytes",
        "l1tex_l2_to_l1_ld_bytes",
        "l1tex_global_st_bytes",
        "l1tex_global_st_miss_bytes",
        "l1tex_l1_to_l2_st_bytes",
        "lts_bytes",
        "lts_bytes_per_second",
        "lts_bytes_per_cycle_elapsed",
        "lts_bytes_pct_peak_elapsed",
        "lts_request_hit_rate_pct",
        "lts_throughput_avg_pct_peak_elapsed",
        "lts_throughput_max_pct_peak_elapsed",
        "lts_throughput_sum_pct_peak_elapsed",
        "lts_requests_hit",
        "lts_requests_miss",
        "lts_read_requests",
        "lts_read_hit",
        "lts_read_miss",
        "lts_write_requests",
        "lts_write_hit",
        "lts_write_miss",
        "lts_read_sectors",
        "lts_read_sectors_per_second",
        "lts_read_sectors_per_cycle_elapsed",
        "lts_read_sectors_pct_peak_elapsed",
        "lts_read_sector_hit",
        "lts_read_sector_miss",
        "lts_write_sectors",
        "lts_write_sectors_per_second",
        "lts_write_sectors_per_cycle_elapsed",
        "lts_write_sectors_pct_peak_elapsed",
        "lts_write_sector_hit",
        "lts_write_sector_miss",
        "lts_read_sector_bytes",
        "lts_write_sector_bytes",
        "dram_bytes",
        "dram_read_bytes",
        "dram_write_bytes",
        "l1tex_ld_bytes_to_expected",
        "l1tex_st_bytes_to_expected",
        "lts_bytes_to_expected",
        "lts_read_sector_bytes_to_expected",
        "lts_write_sector_bytes_to_expected",
        "dram_bytes_to_expected",
        "dram_read_bytes_to_expected",
        "dram_write_bytes_to_expected",
        "missing_metric_count",
        "missing_metrics",
    ]
    rows = []
    for mode in MODES:
        app = app_rows[mode]
        metrics = metric_rows[mode]
        expected = float(app["requested_bytes"])
        lts_read_sectors = metric_or_none(metrics, "lts__t_sectors_op_read.sum")
        lts_write_sectors = metric_or_none(metrics, "lts__t_sectors_op_write.sum")
        lts_read_sector_bytes = (
            lts_read_sectors * 32.0 if lts_read_sectors is not None else None
        )
        lts_write_sector_bytes = (
            lts_write_sectors * 32.0 if lts_write_sectors is not None else None
        )
        missing_metrics = [
            name for name in CANDIDATE_METRICS
            if name not in metrics
        ]
        row = {
            "mode": mode,
            "expected_requested_bytes": expected,
            "clock_bytes_per_cycle": float(app["bytes_per_cycle"]),
            "gpu_time_ns": metric(metrics, "gpu__time_duration.sum"),
            "l1tex_global_ld_bytes": metric(metrics, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"),
            "l1tex_global_ld_miss_bytes": metric(metrics, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum"),
            "l1tex_l2_to_l1_ld_bytes": metric(metrics, "l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum"),
            "l1tex_global_st_bytes": metric(metrics, "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"),
            "l1tex_global_st_miss_bytes": metric(metrics, "l1tex__t_bytes_pipe_lsu_mem_global_op_st_lookup_miss.sum"),
            "l1tex_l1_to_l2_st_bytes": metric(metrics, "l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st.sum"),
            "lts_bytes": metric(metrics, "lts__t_bytes.sum"),
            "lts_bytes_per_second": metric_or_none(metrics, "lts__t_bytes.sum.per_second"),
            "lts_bytes_per_cycle_elapsed": metric_or_none(metrics, "lts__t_bytes.sum.per_cycle_elapsed"),
            "lts_bytes_pct_peak_elapsed": metric_or_none(metrics, "lts__t_bytes.sum.pct_of_peak_sustained_elapsed"),
            "lts_request_hit_rate_pct": metric(metrics, "lts__t_request_hit_rate.pct"),
            "lts_throughput_avg_pct_peak_elapsed": metric_or_none(metrics, "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
            "lts_throughput_max_pct_peak_elapsed": metric_or_none(metrics, "lts__throughput.max.pct_of_peak_sustained_elapsed"),
            "lts_throughput_sum_pct_peak_elapsed": metric_or_none(metrics, "lts__throughput.sum.pct_of_peak_sustained_elapsed"),
            "lts_requests_hit": metric(metrics, "lts__t_requests_lookup_hit.sum"),
            "lts_requests_miss": metric(metrics, "lts__t_requests_lookup_miss.sum"),
            "lts_read_requests": metric(metrics, "lts__t_requests_op_read.sum"),
            "lts_read_hit": metric(metrics, "lts__t_requests_op_read_lookup_hit.sum"),
            "lts_read_miss": metric(metrics, "lts__t_requests_op_read_lookup_miss.sum"),
            "lts_write_requests": metric(metrics, "lts__t_requests_op_write.sum"),
            "lts_write_hit": metric(metrics, "lts__t_requests_op_write_lookup_hit.sum"),
            "lts_write_miss": metric(metrics, "lts__t_requests_op_write_lookup_miss.sum"),
            "lts_read_sectors": lts_read_sectors,
            "lts_read_sectors_per_second": metric_or_none(metrics, "lts__t_sectors_op_read.sum.per_second"),
            "lts_read_sectors_per_cycle_elapsed": metric_or_none(metrics, "lts__t_sectors_op_read.sum.per_cycle_elapsed"),
            "lts_read_sectors_pct_peak_elapsed": metric_or_none(metrics, "lts__t_sectors_op_read.sum.pct_of_peak_sustained_elapsed"),
            "lts_read_sector_hit": metric_or_none(metrics, "lts__t_sectors_op_read_lookup_hit.sum"),
            "lts_read_sector_miss": metric_or_none(metrics, "lts__t_sectors_op_read_lookup_miss.sum"),
            "lts_write_sectors": lts_write_sectors,
            "lts_write_sectors_per_second": metric_or_none(metrics, "lts__t_sectors_op_write.sum.per_second"),
            "lts_write_sectors_per_cycle_elapsed": metric_or_none(metrics, "lts__t_sectors_op_write.sum.per_cycle_elapsed"),
            "lts_write_sectors_pct_peak_elapsed": metric_or_none(metrics, "lts__t_sectors_op_write.sum.pct_of_peak_sustained_elapsed"),
            "lts_write_sector_hit": metric_or_none(metrics, "lts__t_sectors_op_write_lookup_hit.sum"),
            "lts_write_sector_miss": metric_or_none(metrics, "lts__t_sectors_op_write_lookup_miss.sum"),
            "lts_read_sector_bytes": lts_read_sector_bytes,
            "lts_write_sector_bytes": lts_write_sector_bytes,
            "dram_bytes": metric_or_none(metrics, "dram__bytes.sum"),
            "dram_read_bytes": metric_or_none(metrics, "dram__bytes_read.sum"),
            "dram_write_bytes": metric_or_none(metrics, "dram__bytes_write.sum"),
            "missing_metric_count": len(missing_metrics),
            "missing_metrics": ";".join(missing_metrics),
        }
        row["l1tex_ld_bytes_to_expected"] = (
            row["l1tex_global_ld_bytes"] / expected if expected else 0.0
        )
        row["l1tex_st_bytes_to_expected"] = (
            row["l1tex_global_st_bytes"] / expected if expected else 0.0
        )
        row["lts_bytes_to_expected"] = row["lts_bytes"] / expected if expected else 0.0
        row["lts_read_sector_bytes_to_expected"] = ratio(
            row["lts_read_sector_bytes"], expected
        )
        row["lts_write_sector_bytes_to_expected"] = ratio(
            row["lts_write_sector_bytes"], expected
        )
        row["dram_bytes_to_expected"] = ratio(row["dram_bytes"], expected)
        row["dram_read_bytes_to_expected"] = ratio(row["dram_read_bytes"], expected)
        row["dram_write_bytes_to_expected"] = ratio(row["dram_write_bytes"], expected)
        rows.append(row)

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = RESULT_DIR / "ncu_l2_validation_report.md"
    if idle_age is None:
        idle_lines = [
            "本次运行只在 exclusive pre-check 确认没有其他 profiler 进程、",
            "且 `nvidia-smi` 中没有其他 compute 进程后启动。",
        ]
    else:
        idle_lines = [
            f"本次运行由 exclusive monitor 启动；启动前已观察到 `{idle_age:.1f}s` GPU idle，",
            "期间没有 profiler 进程，也没有 `nvidia-smi` 可见的其他 compute 进程。",
        ]
        if idle_confirm_seconds is not None:
            idle_lines.append(
                f"配置的 idle-confirm 阈值是 `{idle_confirm_seconds:.1f}s`。"
            )

    lines = [
        "# NCU L2 验证报告",
        "",
        f"生成时间：{now()}",
        "",
        *idle_lines,
        "",
        "启动 benchmark 前已用 `ncu --query-metrics --query-metrics-mode all` 检查 metric 支持情况；",
        "不支持的候选 metric 会明确列出，不会被解释成 0。",
        "",
        "|模式|预期请求字节|app clock B/cycle|LTS throughput avg %peak elapsed|LTS throughput max %peak elapsed|LTS bytes %peak elapsed|LTS bytes/cycle elapsed|LTS read sectors %peak elapsed|LTS write sectors %peak elapsed|LTS hit rate|缺失 metric 数|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['mode']}|{row['expected_requested_bytes']:.0f}|"
            f"{row['clock_bytes_per_cycle']:.3f}|"
            f"{fmt(row['lts_throughput_avg_pct_peak_elapsed'])}|"
            f"{fmt(row['lts_throughput_max_pct_peak_elapsed'])}|"
            f"{fmt(row['lts_bytes_pct_peak_elapsed'])}|"
            f"{fmt(row['lts_bytes_per_cycle_elapsed'])}|"
            f"{fmt(row['lts_read_sectors_pct_peak_elapsed'])}|"
            f"{fmt(row['lts_write_sectors_pct_peak_elapsed'])}|"
            f"{row['lts_request_hit_rate_pct']:.2f}%|"
            f"{row['missing_metric_count']}|"
        )
    lines.extend([
        "",
        "`app clock B/cycle` 来自 NCU 启动前同参数 app probe 的 `clock64()` 计时；NCU counter 用来验证 L1TEX/LTS 流量和 hit/miss 情况。",
        "`%peak elapsed` 是 NCU 按 elapsed cycles 计算的 pct_of_peak_sustained_elapsed；它是 NCU 内部 peak 定义下的利用率，不等同于本 microbench 的理论硬件峰值。",
        "",
        "## L1/LTS/DRAM 流量归一化",
        "",
        "|模式|L1TEX LD/预期|L1TEX ST/预期|LTS bytes/预期|LTS read sector B/预期|LTS write sector B/预期|DRAM read B/预期|DRAM write B/预期|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"|{row['mode']}|"
            f"{row['l1tex_ld_bytes_to_expected']:.3f}|"
            f"{row['l1tex_st_bytes_to_expected']:.3f}|"
            f"{row['lts_bytes_to_expected']:.3f}|"
            f"{fmt(row['lts_read_sector_bytes_to_expected'])}|"
            f"{fmt(row['lts_write_sector_bytes_to_expected'])}|"
            f"{fmt(row['dram_read_bytes_to_expected'])}|"
            f"{fmt(row['dram_write_bytes_to_expected'])}|"
        )
    lines.extend([
        "",
        "## peak model 与 measured sustained",
        "",
        "|项目|数值|含义|",
        "|---|---:|---|",
        "|L2 read peak model|约 1024 B/cycle/GPU|由 NCU peak 利用率口径反推的模型峰值|",
        "|measured L2-hit read sustained|取 `read-unique` 的 app clock B/cycle|本次 app clock 实测持续值|",
        "|L2 write peak model|约 512 B/cycle/GPU|由 NCU write-sector peak 口径反推的模型峰值|",
        "|measured global-store sustained|取 `write-unique` 的 app clock B/cycle|本次 app clock 实测端到端 store-path 值|",
        "",
        "`1024/512 B/cycle` 只能作为 model peak 写入结论；当前 microbench 没有实测打满这两个数。",
        "",
        "## 产物",
        "",
        "- `read_unique_ncu_raw.csv`",
        "- `write_unique_ncu_raw.csv`",
        "- `read_unique_validation.ncu-rep`",
        "- `write_unique_validation.ncu-rep`",
        "- `query_metrics.log`",
        "- `ncu_l2_validation_summary.csv`",
        "",
    ])
    report.write_text("\n".join(lines))
    return summary_csv, report


def monitor(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    status_log = RESULT_DIR / "ncu_monitor.log"
    lock_f = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"another monitor already holds {LOCK_PATH}")

    build_binary()
    attempt = 0
    idle_since = None
    while True:
        attempt += 1
        blockers = exclusive_blockers()
        if blockers:
            idle_since = None
            append_log(
                status_log,
                f"[{now()}] attempt={attempt} blocked\n" + "\n\n".join(blockers) + "\n",
            )
            print(f"[{now()}] blocked; retrying in {args.poll_seconds}s", flush=True)
        else:
            if idle_since is None:
                idle_since = time.monotonic()
            idle_age = time.monotonic() - idle_since
            if idle_age < args.idle_confirm_seconds:
                append_log(
                    status_log,
                    f"[{now()}] attempt={attempt} exclusive candidate; "
                    f"idle_for={idle_age:.1f}s/"
                    f"{args.idle_confirm_seconds:.1f}s\n",
                )
                print(
                    f"[{now()}] exclusive candidate for {idle_age:.1f}s; "
                    f"waiting for {args.idle_confirm_seconds:.1f}s idle",
                    flush=True,
                )
            else:
                append_log(
                    status_log,
                    f"[{now()}] attempt={attempt} exclusive window found "
                    f"after {idle_age:.1f}s idle\n",
                )
                print(f"[{now()}] exclusive window found; running NCU", flush=True)
                try:
                    app_rows, metric_rows = run_ncu_once(args)
                    summary_csv, report = write_summary(
                        app_rows,
                        metric_rows,
                        args.idle_confirm_seconds,
                        idle_age,
                    )
                    append_log(status_log, f"[{now()}] NCU validation complete: {report}\n")
                    print(f"[{now()}] NCU validation complete: {report}", flush=True)
                    return 0
                except Exception as exc:
                    idle_since = None
                    append_log(status_log, f"[{now()}] NCU run failed: {exc}\n")
                    print(f"[{now()}] NCU run failed; retrying in {args.poll_seconds}s", flush=True)

        if args.max_attempts and attempt >= args.max_attempts:
            append_log(status_log, f"[{now()}] max attempts reached; exiting\n")
            return 2
        time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--iters", type=int, default=2048)
    parser.add_argument("--warmup-iters", type=int, default=64)
    parser.add_argument("--blocks-per-sm", type=int, default=4)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--idle-confirm-seconds", type=float, default=0.0)
    args = parser.parse_args()
    return monitor(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
