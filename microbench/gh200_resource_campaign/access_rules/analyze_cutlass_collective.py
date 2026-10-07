#!/usr/bin/env python3
"""Replay the fixed CUTLASS phase model, saved outputs, diagnostics and new validation."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from analyze_v01 import table
from cutlass_collective_model import collect, predict


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    validation = root / "collective-validation"
    plan_file = validation / "predictions.json"
    plan = json.loads(plan_file.read_text())
    digest = sha(plan_file)
    protocol = json.loads((validation / "protocol.json").read_text())
    if digest != protocol["prediction_sha256"]:
        raise ValueError("prediction identity")
    if sha(validation / "build/r00_cutlass") != plan["binary_sha256"]:
        raise ValueError("validation binary")
    for name, value in json.loads((validation / "source_hashes.json").read_text()).items():
        if sha(validation / "source" / name) != value:
            raise ValueError("validation source " + name)
    medians, evidence = collect(root)
    if evidence != plan["calibration_evidence"]:
        raise ValueError("calibration identity")
    rows = []
    for case in plan["cases"]:
        if predict(plan["parameters"], case["k"]) != case:
            raise ValueError("frozen prediction recomputation")
        values = []
        for f in sorted(validation.glob("samples/" + case["case_id"] + "/*/result.json")):
            r = json.loads(f.read_text())
            if r["prediction_sha256"] != digest or r["binary_sha256"] != plan["binary_sha256"]:
                raise ValueError("sample identity")
            if r["work_flop"] != 2 * case["m"] * case["n"] * case["k"]:
                raise ValueError("work count")
            with gzip.open(f.parent / "full_output.f32.gz", "rb") as src:
                data = src.read()
            if hashlib.sha256(data).hexdigest() != r["full_output_sha256"]:
                raise ValueError("output identity")
            output = np.frombuffer(data, dtype="<f4").reshape(case["m"], case["n"])
            expected = table(case["k"])[np.ix_(np.arange(case["m"]) % 17, np.arange(case["n"]) % 17)]
            if not np.all(output == expected):
                raise ValueError("full output correctness")
            values.append(r["elapsed_ms"] * 1e6)
        if len(values) != 10:
            raise ValueError("missing validation samples")
        median = statistics.median(values)
        rows.append(dict(case_id=case["case_id"], k=case["k"], processes=len(values),
                         predicted_ns=case["predicted_ns"], measured_ns=median,
                         min_ns=min(values), max_ns=max(values), sd_ns=statistics.stdev(values),
                         cv=statistics.stdev(values)/statistics.mean(values),
                         relative_error=(case["predicted_ns"]-median)/median,
                         checked_elements=len(values)*case["m"]*case["n"], output_error=0.0))
    build_rows = []
    for folder, names in (("build-pair", ("original", "NDEBUG")), ("phase-pair", ("original", "phases"))):
        for k in (2048, 8192):
            for name in names:
                values = [json.loads(f.read_text())["elapsed_ms"]*1e6 for f in
                          sorted((root / folder).glob(f"samples/{name}_k{k}/*/result.json"))]
                if len(values) != 10:
                    raise ValueError("missing paired diagnostic")
                build_rows.append(dict(pair=folder,variant=name,k=k,median_ns=statistics.median(values),
                                       cv=statistics.stdev(values)/statistics.mean(values)))
    out = root / "published"
    out.mkdir(exist_ok=True)
    summary = dict(prediction_sha256=digest,cases=rows,paired_builds=build_rows,
                   calibration_medians=medians,parameters=plan["parameters"],conditions=plan["conditions"],
                   limits=plan["limits"],error_median=statistics.median(abs(r["relative_error"]) for r in rows),
                   error_max=max(abs(r["relative_error"]) for r in rows),
                   numeric_gate_pass=all(abs(r["relative_error"])<=0.2 and r["cv"]<=0.05 for r in rows),
                   qualification="matched aggregate collective model; independent review governs semantic acceptance")
    (out / "parameters.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (out / "cases.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=["# 固定CUTLASS：阶段服务与新尺寸预测", "",
          "FP16输入、FP32累加/输出；M=N=2048，tile128×256×64，cluster2×1，4stage，128CTA、1波。",
          "GPU-7c184a2e-41ea-3d2b-df5b-1699c95fc1fe，CUDA12.9/sm90a；repeat准备、坐标dyadic输入seed17。",
          "", "## 实验问题", "",
          "原资源模型低估完整kernel约24%。本轮检查是否遗漏了实际编译条件、组合服务、周期/时间换算与完整完成边界。",
          "原版C7510、逐条wait0保留。只加NDEBUG的版本用于成对诊断，不替代原预测对象。",
          "单独双warpgroup single_wait0仍接近计算峰值；NDEBUG使实际GEMM快约5%–6%，这两项均不足以单独解释全部差距。",
          "", "## 阶段与计算", "",
          "标记0：collective入口；1：mma_tail后；3：store_tail后的输出源复用。标记2为辅助进度，未用于预测。",
          "3不证明完整GMEM写回；CUDA event负责完整完成。20个阶段诊断保存的完整矩阵均通过CPU校验。",
          "主循环窗口包含日志0的固定观测成本及实际取数、计算、控制、同步；成对测量量化了插桩扰动。",
          "", "```text",
          "Cmain(Ktile) = main_start + Ktile × (core_floor + joint_increment)",
          "T = Cmain / ratio_main(Ktile) + output_source + wave_slack + completion_edge", "```", "",
          f"计算下界为{plan['parameters']['core_floor_cycles_per_tile']:.3f} cycle/Ktile；",
          f"匹配collective净增量为{plan['parameters']['joint_increment_cycles_per_tile']:.3f} cycle/Ktile。",
          "两项相加等于阶段校准的有效斜率，净增量是聚合条件项，不能据此识别裸端口或物理争用。",
          "collective已含搬运，不再加R05独立搬运时间。周期/ns比值来自同一消费者窗口，不作频率遥测。",
          "completion_edge由CUDA event时间减消费者包络得到，同时含初始化/派发和源复用到完整完成的剩余部分，不称为纯launch latency。",
          "", "## 新点与验收", "",
          f"预测SHA `{digest}` 在K5120/6144测量前冻结；新点不参与校准。每点10独立进程，完整输出总计83886080个值，误差0。",
          "验证程序只在计时后主机保存D；其整个device_kernel SASS与原版逐字一致。",
          "| K | 预测µs | 实测µs | 范围µs | CV | 误差 |", "|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        text.append(f"| {r['k']} | {r['predicted_ns']/1000:.3f} | {r['measured_ns']/1000:.3f} | {r['min_ns']/1000:.3f}–{r['max_ns']/1000:.3f} | {100*r['cv']:.2f}% | {100*r['relative_error']:+.3f}% |")
    text += ["", "这是固定配置的分阶段、条件化服务模型，不是无需组合校准的通用原子资源预测器。",
             "K须为256的倍数且在2048–8192之间；不向其他M/N、tile、stage、cluster、GPU、输入分布或构建选项迁移。",
             "", "## 手算一个点", "",
             "K5120对应80个Ktile。把冻结parameters.json里的阶段参数代入，得到64497.267 ns；目标工作量2×2048×2048×5120=42949672960 FLOP。",
             "此点实测64960.003 ns，预测误差(64497.267−64960.003)/64960.003≈−0.712%。名义GEMM FLOP不含实现辅助指令；输出阶段服务已包含这些实现成本。",
             "", "## 复现", "", "```bash",
             "python3 source/analyze_cutlass_collective.py --input /移动后的归档", "```", "",
             "原始记录在collective-validation/samples与phase-pair/samples；预测为collective-validation/predictions.json。",
             "独立审查：independent-collective-review-A.md；旧结果和旧预测均保留。", "",
             "图中误差条为10个独立进程的样本标准差，不是预测区间；实测柱高为中位数。", "",
             "![新尺寸预测与实测](validation.png)", ""]
    (out / "report.md").write_text("\n".join(text))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(len(rows))
    ax.bar(x-.18,[r['predicted_ns']/1000 for r in rows],.36,label='Frozen phase model',color='#496bb4')
    ax.bar(x+.18,[r['measured_ns']/1000 for r in rows],.36,yerr=[r['sd_ns']/1000 for r in rows],capsize=4,label='Measured median (SD bars)',color='#459e80')
    ax.set_xticks(x,[f"2048 x 2048 x {r['k']}" for r in rows]);ax.set_ylabel('CUDA-event completion time (us)')
    ax.legend();ax.set_ylim(0,90);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'validation.png',dpi=160);plt.close(fig)
    (out/'analyzer_identity.json').write_text(json.dumps({'script_sha256':sha(Path(__file__)),'prediction_sha256':digest},indent=2)+'\n')


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    main(p.parse_args().input.resolve())
