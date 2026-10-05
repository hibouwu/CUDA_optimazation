#!/usr/bin/env python3
"""Audit and summarize sustained-issue tests using observed grid makespan."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import statistics as st


def analyze(root, reference_clock_hz=1_980_000_000):
    assert math.isfinite(reference_clock_hz) and reference_clock_hz>0
    manifest=json.loads((root/'stress_cases.json').read_text())['cases']
    specs={c['name']:c for c in manifest}
    lines=(root/'raw.jsonl').read_text().splitlines()
    device=json.loads(lines[0]);rows=[json.loads(s) for s in lines[1:]]
    assert len(rows)==len(specs)*2*5, 'incomplete matrix'
    groups=collections.defaultdict(list)
    blocks_by_case=re.split(r'Function\s*:\s*',(root/'sass.txt').read_text())[1:]
    sass_map={b.splitlines()[0].strip():b for b in blocks_by_case}
    results=[]
    for r in rows:
        c=specs[r['case']]
        assert r['scope'] in ['single_cta','full_gpu']
        assert r['max_abs_error']==0 and r['threads']==c['threads']
        assert len(r['blocks_detail'])==r['blocks']
        smids={b['smid'] for b in r['blocks_detail']}
        assert len(smids)==r['unique_sms']
        assert r['unique_sms']==(device['sms'] if r['scope']=='full_gpu' else 1)
        assert r['start_ns']==min(b['start_ns'] for b in r['blocks_detail'])
        assert r['stop_ns']==max(b['stop_ns'] for b in r['blocks_detail'])
        assert all(b['stop_ns']>b['start_ns'] and b['cycles']>0 for b in r['blocks_detail'])
        work=r['blocks']*r['iterations']*c['batch']*c['chains']*c['work_per_collective']
        assert work==r['work_flop']
        rate=work/(r['stop_ns']-r['start_ns'])/1000
        assert math.isclose(rate,r['tflops'],rel_tol=1e-8)
        # CUDA event includes init/writeback; accept small timer granularity differences.
        assert (r['stop_ns']-r['start_ns'])/1e6 <= r['event_ms']*1.05
        groups[r['case'],r['scope']].append(r)
    assert len(groups)==len(specs)*2
    for (name,scope),g in sorted(groups.items()):
        assert len(g)==5 and {r['repeat'] for r in g}==set(range(5))
        c=specs[name];rates=[r['tflops'] for r in g]
        body=sass_map[f'_Z{len(name)}{name}iP6ResultPd']
        token='HGMMA' if c['kind'].startswith('wgmma') else 'DMMA' if c['kind']=='mma_f64' else 'HMMA' if c['kind'].startswith('mma_') else 'FFMA' if c['kind']=='f32' else 'DFMA' if c['kind']=='f64' else 'HFMA2'
        assert re.search(r'\b'+token+r'(?:\.|\s)',body)
        waits=re.findall(r'WARPGROUP\.DEPBAR[^;]*;',body)
        if c['kind'].startswith('wgmma'):
            n=c['pending_after_wait']
            assert any(re.search(r',\s*(?:0x)?'+str(n)+r'\s*;',w) for w in waits), (name,waits)
        reference_cycles=[(r['stop_ns']-r['start_ns'])*reference_clock_hz/1e9 for r in g]
        cycle_counts=([r['max_block_cycles'] for r in g] if scope=='single_cta' else reference_cycles)
        cycle_rates=[r['work_flop']/n for r,n in zip(g,cycle_counts)]
        results.append(dict(case=name,kind=c['kind'],scope=scope,threads=c['threads'],
            chains=c['chains'],batch=c['batch'],pending_after_wait=c['pending_after_wait'],
            blocks=g[0]['blocks'],registers_per_thread=g[0]['registers_per_thread'],
            iterations=g[0]['iterations'],tflops_median=st.median(rates),
            tflops_min=min(rates),tflops_max=max(rates),cv=st.stdev(rates)/st.mean(rates),
            duration_ms_median=st.median((r['stop_ns']-r['start_ns'])/1e6 for r in g),
            single_cta_flop_per_cycle=(st.median(r['work_flop']/r['max_block_cycles'] for r in g) if scope=='single_cta' else None),
            cycle_basis='clock64_local_sm' if scope=='single_cta' else 'fixed_reference_clock',
            flop_per_cycle_median=st.median(cycle_rates),flop_per_cycle_min=min(cycle_rates),flop_per_cycle_max=max(cycle_rates),
            timed_cycles_median=st.median(cycle_counts),reference_clock_hz=reference_clock_hz if scope=='full_gpu' else None,
            observed_sms=g[0]['unique_sms'],sass_token=token,sass_wait_instructions=waits))
    telemetry=list(csv.DictReader((root/'telemetry.csv').open()))
    telemetry=[{k.strip():v.strip() for k,v in r.items()} for r in telemetry]
    metrics={}
    for k in ['clocks.current.sm [MHz]','temperature.gpu','power.draw [W]','utilization.gpu [%]']:
        values=[]
        for r in telemetry:
            try:values.append(float(r[k].split()[0]))
            except (ValueError,KeyError):pass
        if values:metrics[k]=dict(min=min(values),median=st.median(values),max=max(values))
    result=dict(device=device,measured_launches=len(rows),configurations=len(results),
        reference_clock_hz=reference_clock_hz,
        cycle_convention='Single CTA uses measured clock64; full GPU uses grid makespan times fixed reference clock, not a sum/difference of clocks from different SMs.',
        timed_gpu_seconds=sum((r['stop_ns']-r['start_ns'])/1e9 for r in rows),
        telemetry_samples=len(telemetry),telemetry_whole_run=metrics,
        limitation='Finite constant operands, 5 sequential repetitions, unlocked clock, no HBM input traffic. Full-GPU rates use observed global-timer makespan, not multiplication of per-CTA rates.',
        file_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['stress.cu','stress_cases.json','raw.jsonl','sass.txt','compile.log','telemetry.csv']},cases=results)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    report=['# GH200 持续发射压测','',f"设备：{device['device']}；{device['sms']} SM。{len(results)} 个配置/范围组合，每个 5 次，共 {len(rows)} 次正式运行。",'',
        f"单 CTA 使用本地 SM 的实测 clock64 周期；整卡统一使用 {reference_clock_hz/1e9:g} GHz 参考频率归一化，即 reference cycles = grid 时间窗口 × 参考频率。该参考值取自本次遥测的典型频率，不表示每个试验都锁定在此频率。遥测共 {len(telemetry)} 条。",'',
        'FMA / mma.sync 使用 8 条独立累加链；WGMMA 使用 2 条，每链每批 16 条，wait_group 分别为 0/3/7，循环结束统一 wait_group 0。操作数已在寄存器或 SMEM，不包含 HBM 输入搬运。', '',
        '单 CTA 配置测 32/128/256 线程（WGMMA 为 128/256）；全 GPU 配置使用 SM 数 × min(4, occupancy API 上限) 个 CTA。后者逐次核验所有 SM 均被访问，完整每 CTA 时间戳和 SM ID 保存于 raw.jsonl。该布局不保证每个 SM 的 CTA 数完全相等。', '',
        '## 各计算形式的最高实测配置','',
        '这里只在本批有限配置中取最高中位数，不代表硬件峰值。不同精度和指令形状对应不同工作量。', '',
        '| 计算形式 | 线程/CTA | CTA 数 | wait_group | 整卡 FLOP/reference cycle | 最小–最大 | 窗口 reference cycles |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for kind in dict.fromkeys(c['kind'] for c in manifest):
        best=max((r for r in results if r['kind']==kind and r['scope']=='full_gpu'),key=lambda r:r['flop_per_cycle_median'])
        report.append(f"| {kind} | {best['threads']} | {best['blocks']} | {best['pending_after_wait'] if kind.startswith('wgmma') else '—'} | {best['flop_per_cycle_median']:,.2f} | {best['flop_per_cycle_min']:,.2f}–{best['flop_per_cycle_max']:,.2f} | {best['timed_cycles_median']:,.0f} |")
    report+=['','## 单 CTA 的实测周期口径','','以下值直接由工作量除以该 CTA 的 clock64 周期得到，不经过参考频率换算。','',
             '| 计算形式 | 线程/CTA | wait_group | FLOP/clock64 cycle |','|---|---:|---:|---:|']
    for r in results:
        if r['scope']=='single_cta':
            report.append(f"| {r['kind']} | {r['threads']} | {r['pending_after_wait'] if r['kind'].startswith('wgmma') else '—'} | {r['flop_per_cycle_median']:,.2f} |")
    report+=['','## WGMMA 等待深度对照','','每组线程数与执行范围固定，其余条件一致。wait_group N 允许最近至多 N 个已提交组仍未完成；不是测得硬件队列深度。','',
             '单 CTA 行单位为 FLOP/clock64 cycle；整卡行单位为 FLOP/reference cycle。两种分母不能混用。','',
             '| 精度 | 范围 | 线程/CTA | wait 0 | wait 3 | wait 7 |',
             '|---|---|---:|---:|---:|---:|']
    for kind in ['wgmma_f16','wgmma_bf16']:
        for scope in ['single_cta','full_gpu']:
            for threads in [128,256]:
                rs=sorted((r for r in results if (r['kind'],r['scope'],r['threads'])==(kind,scope,threads)),key=lambda r:r['pending_after_wait'])
                report.append(f"| {kind} | {scope} | {threads} | "+' | '.join(f"{r['flop_per_cycle_median']:,.2f}" for r in rs)+' |')
    report+=['','## 全程遥测','','范围包含预热、单 CTA、整卡和间歇；不能直接归因于某个测试配置。','',
             '| 指标 | 最小 | 中位数 | 最大 |','|---|---:|---:|---:|']
    for key,vals in metrics.items():
        report.append(f"| {key} | {vals['min']:.2f} | {vals['median']:.2f} | {vals['max']:.2f} |")
    report+=['','## 解释边界','',
             '- 整卡参考周期使用最早 CTA 开始到最晚 CTA 完成的时间窗口，计入最终排空；没有跨 SM 相减 clock64，也没有将多个 SM 的周期相加。',
             '- 原始时间戳及原吞吐字段保留用于审计，报告统一展示周期。整卡参考周期不等于实际累计 SM 时钟周期。',
             '- 这是持续计算吞吐实验，包含循环、发射和必要同步；不等同于纯发射端口速率、完整 GEMM 或长期热稳定性测试。',
             '- 数值逐元素检查有限性和精确值，全部通过；编译器插入的 WGMMA 同步仍属于生成代码成本。SASS 已核验相应 wait_group 值。',
             '- 试验顺序随机化，单配置 5 次连续重复；功耗和温度统计覆盖整个运行，不自动归因于某一个指令形式。未修改频率和功率上限。',
             '- 常量输入不覆盖所有数据相关功耗或数值行为；有限配置中的最好结果不是物理上限。', '',
             '所有配置统计见 summary.json，原始记录见 raw.jsonl，编译及实现证据见 stress.cu / compile.log / sass.txt / SHA256SUMS。']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({k:result[k] for k in ['measured_launches','configurations','timed_gpu_seconds','telemetry_samples','telemetry_whole_run']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path)
    p.add_argument('--reference-clock-hz',type=float,default=1_980_000_000)
    args=p.parse_args()
    analyze(args.results,args.reference_clock_hz)
