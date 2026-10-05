#!/usr/bin/env python3
"""Fit cycle costs from repeated, single-CTA observations; never infer GPU peak."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics as st


def fit(points):
    xs, ys = zip(*points)
    slope, intercept = st.linear_regression(xs, ys)
    total = sum((y-st.mean(ys))**2 for y in ys)
    residual = sum((y-intercept-slope*x)**2 for x,y in points)
    return slope, intercept, 1-residual/total if total else 1.0


def analyze(root):
    manifest = json.loads((Path(__file__).parent/'cases.json').read_text())
    rows = [json.loads(x) for x in (root/'raw.jsonl').read_text().splitlines()]
    device, data = rows[0], rows[1:]
    assert len(data)==len(manifest)*3*7, 'incomplete trial matrix'
    assert all(r['max_abs_error']==0 and r['cycles']>0 for r in data)
    sass = (root/'sass.txt').read_text()
    functions = re.split(r'Function\s*:\s*',sass)[1:]
    rng = random.Random(20260930)
    summaries=[]
    for c in manifest:
        cr = [r for r in data if r['case']==c['name']]
        groups=collections.defaultdict(list)
        for r in cr:
            groups[r['iterations']].append(r['cycles'])
        assert set(groups)=={128,512,2048} and all(len(v)==7 for v in groups.values())
        assert all(len({r['repeat'] for r in cr if r['iterations']==n})==7 for n in groups)
        points=[(n,st.median(v)) for n,v in sorted(groups.items())]
        slope,intercept,r2=fit(points)
        denom=c['batch']*c['chains']
        boot=[]
        for _ in range(400):
            bs=[(n,st.median(rng.choices(v,k=len(v)))) for n,v in sorted(groups.items())]
            boot.append(fit(bs)[0]/denom)
        boot.sort()
        symbol=f"_Z{len(c['name'])}{c['name']}iP6ResultPd"
        body=[f for f in functions if f.splitlines()[0].strip()==symbol]
        assert len(body)==1, (c['name'],'missing/ambiguous SASS')
        token='HGMMA' if c['kind'].startswith('wgmma') else 'DMMA' if c['kind']=='mma_f64' else 'HMMA' if c['kind'].startswith('mma_') else 'FFMA' if c['kind']=='f32' else 'DFMA' if c['kind']=='f64' else 'HFMA2'
        count=len(re.findall(r'\b'+token+r'(?:\.|\s)',body[0]))
        assert count>0, (c['name'],token,'absent')
        summaries.append(dict(**c,scope='single_cta',slope_cycles_per_iteration=slope,
            intercept_cycles=intercept,cycles_per_ptx_collective=slope/denom,
            bootstrap_95_percentile=[boot[9],boot[389]],r_squared=r2,
            cta_flop_per_cycle=c['work_per_collective']*denom/slope,
            registers_per_thread=cr[0]['registers_per_thread'],static_smem_bytes=cr[0]['static_smem_bytes'],
            observed_smids=sorted({r['smid'] for r in cr}),sass_token=token,sass_token_count=count,
            median_points=points,
            sample_cycles_min=min(r['cycles'] for r in cr),sample_cycles_max=max(r['cycles'] for r in cr)))
    files=['raw.jsonl','sass.txt','probe.cu','compile.log','environment.txt','telemetry.csv']
    compile_log=(root/'compile.log').read_text()
    telemetry_samples=max(0,len((root/'telemetry.csv').read_text().splitlines())-1)
    result=dict(device=device,trial_count=len(data),case_count=len(summaries),
        compiler_injected_warpgroup_arrive='C7519' in compile_log,
        telemetry_samples=telemetry_samples,
        limitation='Single CTA, selected forms, constant finite operands. Slope includes loop cost and WGMMA batch commit/wait; no isolated hardware latency or GPU roof claim.',
        file_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},cases=summaries)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# GH200 FMA / MMA 首批测量','',f"设备：{device['device']}，SM 数：{device['sms']}。",'',
           f"{len(summaries)} 个编译配置 × 3 个循环长度 × 7 次记录，共 {len(data)} 次实测；每个长度先执行 2 次预热。全部输出逐元素检查有限性和误差。",'',
           '下表来自 `cycles = intercept + slope × iterations` 的中位数拟合。每条 FMA 按整个 warp 执行计量，MMA 按 warp/warpgroup 的一条逻辑 PTX collective 计量；不能横向当作同工作量指令比较。','',
           '| 配置 | 独立链 | 每链 batch | cycles/PTX collective | 单 CTA FLOP/cycle | R² | 寄存器/线程 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['kind']} | {s['chains']} | {s['batch']} | {s['cycles_per_ptx_collective']:.3f} | {s['cta_flop_per_cycle']:.2f} | {s['r_squared']:.6f} | {s['registers_per_thread']} |")
    lines += ['', '计时从寄存器/SMEM 操作数就绪后开始，结束前通过结果归约、volatile SMEM 写入和 CTA barrier 排空依赖；这些固定尾部开销由截距吸收。循环控制仍在斜率内。WGMMA 每个 batch 执行 commit + wait_group 0，其耗时在斜率内。', '',
              '每个试验只启动一个 CTA。未测整卡饱和、多个 CTA 的竞争、一般输入分布或全部合法形状；恒定输入的正确性不替代任意矩阵布局验证。WGMMA 当前仅 SS、K-major、无 swizzle。未修改设备时钟，频率与功耗记录见 telemetry.csv。', '',
              '编译日志记录 ptxas C7519：编译器为 WGMMA 寄存器使用插入了 warpgroup.arrive。结果属于该生成代码的服务成本，不能解释为剥离所有控制指令后的硬件启动间隔。', '',
              f'本次 telemetry.csv 只有 {telemetry_samples} 条样本，不能证明计时期间频率稳定；不使用该快照将周期换算为实测 FLOP/s。', '',
              '拟合使用 3 个循环长度、每点 7 次；bootstrap 区间仅刻画本次样本内波动，不覆盖系统误差或跨日期变化。', '',
              '原始数据与源码：raw.jsonl、probe.cu、compile_command.txt、compile.log、sass.txt、environment.txt、environment_after.txt、telemetry.csv、summary.json。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(trials=len(data),cases=len(summaries),min_r_squared=min(s['r_squared'] for s in summaries))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path)
    analyze(p.parse_args().results)
