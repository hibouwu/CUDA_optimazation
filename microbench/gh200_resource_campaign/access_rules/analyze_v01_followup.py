#!/usr/bin/env python3
"""CPU replay and Chinese report for the finite V01 layout/new-length follow-up."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import struct

from analyze_v01 import analyze


def main(root):
    result = analyze(root / "fresh-lengths", root / "fresh-review")
    records = json.loads((root / "layout-summary.json").read_text())
    binary_sha = hashlib.sha256((root / "build/wgmma_layout").read_bytes()).hexdigest()
    for f in sorted((root / "layout-samples").glob("*/*/result.json")):
        r = json.loads(f.read_text())
        if r["binary_sha256"] != binary_sha:
            raise ValueError("layout binary identity")
        with gzip.open(f.parent / "output.f32.gz", "rb") as source:
            data = source.read()
        if hashlib.sha256(data).hexdigest() != r["output_sha256"]:
            raise ValueError("layout output identity")
        if len(data) != r["blocks"] * 256 * 128 * 4:
            raise ValueError("layout output length")
        if any(x[0] != 1024.0 for x in struct.iter_unpack("<f", data)):
            raise ValueError("layout CPU reference")
        if r["work_flop"] != 17179869184 * r["blocks"]:
            raise ValueError("layout work")
        window = r["stamps"][0][3] - r["stamps"][0][2] if r["scope"] == "one_cta" else max(s[1] for s in r["stamps"]) - min(s[0] for s in r["stamps"])
        if window != r["elapsed"]:
            raise ValueError("layout time")
    lines = ["# V01：布局对照与新长度验证", "",
             "2026-10-06；GH200，GPU-7c184a2e-41ea-3d2b-df5b-1699c95fc1fe；Slurm 735203，CUDA12.9/sm90a。",
             "",
             "本轮4个布局条件、26正式进程；3个新长度条件、23正式进程。旧数据保留。",
             "布局探针CPU检查全部保存输出；受控组合CPU检查完整输出；完整CUTLASS只检查保存的4096个坐标样本。所有这些检查误差均为0，任务精度容差仍未定义。",
             "",
             "## 1. K与MN布局", "",
             "FP16 SS m64n256k16，2 warpgroup/CTA，128累加寄存器/线程，batch16、commit/wait1、最终wait0；同GPU配对运行。",
             "| 范围 | B major | 进程 | 完成窗口中位数 | CV |", "|---|---|---:|---:|---:|"]
    for scope in ("one_cta", "all_gpu"):
        for major in ("K", "MN"):
            rows = [r for r in records if r["scope"] == scope and r["b_major"] == major]
            values = [r["elapsed"] for r in rows]
            unit = "cycle" if scope == "one_cta" else "ns"
            lines.append(f"| {scope} | {major} | {len(rows)} | {statistics.median(values):.0f} {unit} | {100*statistics.stdev(values)/statistics.mean(values):.4f}% |")
    lines += ["", "MN版16条HGMMA带`.tnspB`，K版没有；两版均154 registers/thread、无spill。非均匀坐标短检查由探针执行，正式均匀输入完整输出另经CPU重算。",
              "单CTA的MN完成时间只多10 cycle，整卡中位数相同。这个有限对照没有显示孤立MN WGMMA足以解释完整kernel约24%的偏差；它不能排除完整kernel里的地址组织、搬运、同步与布局共同作用。",
              "", "## 2. 采样前冻结的新点", "",
              "预测文件SHA256：`" + result["prediction_sha256"] + "`。新点复用旧V01/CUTLASS二进制并核对UUID、二进制与探针哈希；预测后探针未改变。",
              "| 新点 | 实测中位数 | 资源模型 | 误差 | 条件仿射模型 | 误差 | CV |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["cases"]:
        scale = 1000 if r["unit"].startswith("cuda") else 1
        unit = "µs" if scale == 1000 else ("cycle" if "clock64" in r["unit"] else "ns")
        lines.append(f"| {r['case_id']} | {r['median']*scale:.3f} {unit} | {r['predicted']*scale:.3f} | {100*r['error']:+.3f}% | {r['alternative']*scale:.3f} | {100*r['alternative_error']:+.3f}% | {100*r['cv']:.3f}% |")
    plan = json.loads((root / "fresh-lengths/predictions.json").read_text())
    known = plan["calibration"]["known_medians"]
    parameters = []
    for kind, first, second, x1, x2, unit, conditions in [
        ("target_one_cta", "target_tc_k19_one_cta", "target_tc_k47_one_cta", 19, 47, "cycle/CTA", "128x256x64,384线程,4stage,K/K,单CTA,FP32输出"),
        ("target_all_gpu", "target_tc_k19_all_gpu", "target_tc_k47_all_gpu", 19, 47, "ns/GPU", "同受控组合,grid528,共享输入,repeat,1CTA/SM上限"),
        ("cutlass_fixed", "cutlass_2048_2048_2048", "cutlass_2048_2048_8192", 32, 128, "ms/GPU", "M=N=2048,FP16输入,FP32输出,128x256x64,cluster2x1,4stage,repeat"),
    ]:
        slope = (known[second] - known[first]) / (x2 - x1)
        intercept = known[first] - slope * x1
        parameters.append(dict(name=kind, formula="T=a+b*Ktile", a=intercept, b=slope, unit=unit,
                               status="conditional_new_point_checked",
                               conditions=conditions, calibration_ktile=[x1,x2],
                               validation_ktile=64 if kind == "cutlass_fixed" else 31,
                               ktile_domain=dict(interval=[x1,x2], integer=True,
                                                 residue_mod4=None if kind == "cutlass_fixed" else 3),
                               scope="conditional interpolation; no extrapolation or cross-configuration transfer",
                               gpu_uuid=plan["gpu_uuid"], binary_hashes=plan["binary_hashes"],
                               prediction_sha256=result["prediction_sha256"]))
    (root / "conditional_parameters.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False) + "\n")
    lines += ["", "## 3. 可以用于什么", "",
              "资源模型的新单CTA误差−0.632%，整卡−8.077%，完整CUTLASS−23.957%；完整kernel仍超过最大20%目标。不能把备选模型通过写成资源模型通过。",
              "条件仿射模型的三个新点误差均小于1%。它提供固定配置下沿K长度变化的经验预测；不解释资源竞争，也不能用于tile搜索、跨M/N、cluster、stage、布局、设备或缓存条件的迁移。受控组合的校准和验证K均为3 mod 4，其他余数类未验证。",
              "", "完整CUTLASS的条件式（Ktile=K/64）：", "",
              "```text", f"T ≈ {parameters[2]['a']*1000:.6f} + {parameters[2]['b']*1000:.6f} × Ktile  µs", "```", "",
              "K=4096时，Ktile=64，预测53.744 µs，实测53.472 µs；两端K=2048/8192用于校准，4096未参与拟合。固定项吸收了该条件下的启动/预填/输出等成本，斜率吸收主循环和随K变化的成本，不能将其拆称为裸硬件服务。",
              "", "## 4. 手算与复现", "",
              "受控单CTA Ktile=31：工作量=2×128×256×64×31=130023424 FLOP。中位完成窗口39628 cycle，对应约3281.100 FLOP/cycle；窗口包含预填与完整输出，不能与孤立计算速率4095 FLOP/cycle直接等同。",
              "", "```bash",
              "python3 microbench/gh200_resource_campaign/access_rules/analyze_v01_followup.py \\",
              "  --input results/gh200_resource_campaign/access_rules/20261006-v01-followup",
              "```", "",
              "原始记录：`layout-samples/`、`fresh-lengths/samples/`；预测：`fresh-lengths/predictions.json`；完整SASS与编译日志：`build/`及`fresh-lengths/build/`；条件参数：`conditional_parameters.json`。",
              "对应独立增量审查为`independent-review-B.md`；审查结论与测量结果分别保留。", "",
              "![三种新点的主预测与条件仿射预测误差](prediction_errors.png)", ""]
    (root / "report.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = result["cases"]
    xs = list(range(len(rows)))
    figure, ax = plt.subplots(figsize=(9, 4.5))
    for offset, field, label, color in [(-0.18, "error", "Resource model", "#496bb4"),
                                       (0.18, "alternative_error", "Conditional affine model", "#459e80")]:
        bars = ax.bar([x+offset for x in xs], [100*r[field] for r in rows], 0.34,
                      label=label, color=color)
        for bar, r in zip(bars, rows):
            value = 100 * r[field]
            ax.text(bar.get_x()+bar.get_width()/2, value-0.6 if value < -2 else value+0.7,
                    f"{value:+.2f}%", ha="center", va="top" if value < -2 else "bottom", fontsize=9)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(-20, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs, ["CUTLASS K=4096", "Target Ktile=31 / GPU", "Target Ktile=31 / CTA"])
    ax.set_ylabel("Prediction error (%)")
    ax.set_title("New points: predictions frozen before measurement")
    ax.set_ylim(-29, 6)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(root / "prediction_errors.png", dpi=160)
    plt.close(figure)
    (root / "analysis_identity.json").write_text(json.dumps(dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_prediction_sha256=result["prediction_sha256"],
        parameter_sha256=hashlib.sha256((root / "conditional_parameters.json").read_bytes()).hexdigest(),
    ), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    main(parser.parse_args().input.resolve())
