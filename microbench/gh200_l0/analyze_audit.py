#!/usr/bin/env python3
"""Audited per-SM cycle statistics. Raw clocks are only combined within one SM."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics as st


def bootstrap_median(values,seed=20260930):
    rng=random.Random(seed)
    samples=sorted(st.median(rng.choices(values,k=len(values))) for _ in range(2000))
    return [samples[49],samples[1949]]


def analyze(root):
    config=json.loads((root/'audit_cases.json').read_text());specs={c['name']:c for c in config['cases']}
    lines=[json.loads(l) for l in (root/'raw.jsonl').open()];device,rows=lines[0],lines[1:]
    groups=collections.defaultdict(list);empty=collections.defaultdict(list)
    for r in rows:
        c=specs[r['case']];assert r['max_abs_error']==0
        spans={}
        assert len(r['blocks_detail'])==r['blocks']
        for b in r['blocks_detail']:
            assert b['cycles']==b['stop_cycle']-b['start_cycle'] and b['cycles']>0
            pair=spans.setdefault(b['smid'],[b['start_cycle'],b['stop_cycle']])
            pair[0]=min(pair[0],b['start_cycle']);pair[1]=max(pair[1],b['stop_cycle'])
        total=sum(hi-lo for lo,hi in spans.values())
        assert total==r['summed_sm_span_cycles'] and len(spans)==r['unique_sms']
        assert r['work_flop']==r['blocks']*r['iterations']*c['batch']*c['chains']*c['work_per_collective']
        assert math.isclose(r['flop_per_sm_cycle'],r['work_flop']/total,rel_tol=1e-10,abs_tol=1e-12)
        if r['iterations'] and r['scope']=='full_gpu':assert len(spans)==device['sms']
        if r['phase']=='measure':groups[r['case'],r['scope'],r['iterations']].append(r)
        if r['phase']=='empty_control':empty[r['case']].append(r)
    assert len(groups)==42 and sum(map(len,groups.values()))==504
    for g in groups.values():assert len(g)==12 and {r['round'] for r in g}==set(range(12))
    assert len(empty)==7 and all(len(g)==12 for g in empty.values())
    sass=(root/'sass.txt').read_text()
    parts=re.split(r'Function\s*:\s*',sass)[1:];functions={p.splitlines()[0].strip():p for p in parts}
    sass_checks={}
    for name,c in specs.items():
        body=functions[f'_Z{len(name)}{name}iP6ResultPd']
        token='HGMMA' if c['kind'].startswith('wgmma') else 'HMMA' if c['kind'].startswith('mma') else 'FFMA' if c['kind']=='f32' else 'DFMA' if c['kind']=='f64' else 'HFMA2'
        assert re.search(r'\b'+token+r'(?:\.|\s)',body)
        sass_checks[name]=dict(token=token,static_count=len(re.findall(r'\b'+token+r'(?:\.|\s)',body)))
        if c['kind'].startswith('wgmma'):
            waits=re.findall(r'WARPGROUP\.DEPBAR[^;]*;',body)
            assert any(re.search(r',\s*0x'+str(c['pending_after_wait'])+r'\s*;',w) for w in waits)
            assert any(re.search(r',\s*0x0\s*;',w) for w in waits)
            sass_checks[name]['wait_instructions']=waits
    warmups={name:dict(stable=bool(int(ok)),launches=int(n)) for name,ok,n in re.findall(r'warmup case=(\S+) stable=(\d) launches=(\d+)',(root/'run.stderr').read_text())}
    assert set(warmups)==set(specs)
    summaries=[]
    for (name,scope,it),g in sorted(groups.items()):
        v=[r['flop_per_sm_cycle'] for r in g]
        block_cycles=[r['max_block_cycles'] for r in g]
        empty_med=st.median(r['max_block_cycles'] for r in empty[name])
        summaries.append(dict(case=name,kind=specs[name]['kind'],scope=scope,iterations=it,n=12,
            unit='FLOP/clock64 cycle/CTA' if scope=='single_cta' else 'FLOP/summed SM span cycle',
            median=st.median(v),mean=st.mean(v),min=min(v),max=max(v),cv=st.stdev(v)/st.mean(v),
            median_bootstrap95=bootstrap_median(v),median_max_cta_cycles=st.median(block_cycles),
            empty_control_cycles=empty_med,empty_to_short_window_ratio=empty_med/st.median(block_cycles),
            warmup=warmups[name]))
    pairs=[]
    for scope in ['single_cta','full_gpu']:
        for a,b in [(0,3),(3,7)]:
            ga=groups[f'wgmma_f16_c2_q16_t128_p{a}',scope,65536]
            gb=groups[f'wgmma_f16_c2_q16_t128_p{b}',scope,65536]
            av={r['round']:r['flop_per_sm_cycle'] for r in ga}
            ratios=[r['flop_per_sm_cycle']/av[r['round']] for r in gb]
            pairs.append(dict(scope=scope,comparison=f'wait{b}/wait{a}',median_ratio=st.median(ratios),bootstrap95=bootstrap_median(ratios)))
    fits=[]
    for name in specs:
        pts=[(it,st.median(r['max_block_cycles'] for r in groups[name,'single_cta',it])) for it in config['lengths']]
        slope,intercept=st.linear_regression(*zip(*pts));ybar=st.mean(y for _,y in pts)
        r2=1-sum((y-intercept-slope*x)**2 for x,y in pts)/sum((y-ybar)**2 for _,y in pts)
        throughputs=[st.median(r['flop_per_sm_cycle'] for r in groups[name,'single_cta',it]) for it in config['lengths']]
        fits.append(dict(case=name,slope_cycles_per_iteration=slope,intercept_cycles=intercept,r_squared=r2,
                         throughput_length_spread=max(throughputs)/min(throughputs)-1))
    ncu={p:dict(exit_code=int((root/f'ncu_{p}.exit').read_text()),counter_permission_denied='ERR_NVGPUCTRPERM' in (root/f'ncu_{p}.csv').read_text()) for p in ['fma','wgmma']}
    summary=dict(device=device,measured_launches=504,empty_controls=84,warmup_records=sum(1 for r in rows if 'warmup' in r['phase']),
        cycle_definition='For each SM, span=max(stop_cycle)-min(start_cycle) over its CTAs. Rate=sum(work)/sum(SM spans). No clock differences across SMs; full-GPU value is mean service per SM cycle, not GPU makespan throughput.',
        unit_limit='Local clock64 includes scheduling and sharing time. Mean SM-span rate is not the service of a completely occupied SM at every instant.',
        warmups=warmups,ncu=ncu,sass=sass_checks,cases=summaries,paired_comparisons=pairs,length_fits=fits,
        raw_sha256=hashlib.sha256((root/'raw.jsonl').read_bytes()).hexdigest())
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# GH200 微基准审查复测','',f"{device['device']}，{device['sms']} SM。预先指定 7 个代表配置，2 个范围，3 个固定循环长度，12 轮随机交错；504 次正式测量、84 次空窗口对照。",'',
      '## 周期定义','',
      '单 CTA 使用其 clock64 周期。整卡试验先对同一个 SM 上的 CTA 取最早 start_cycle 到最晚 stop_cycle，再将各 SM 的覆盖周期相加，计算总 FLOP / 总 SM 覆盖周期。它表达本次负载下每 SM 周期的平均服务率，包含 SM 内间隙；不是整卡全局完成周期，也没有用 1.98 GHz 代替实测周期。','',
      '## 65536 轮循环结果','',
      '| 配置 | 单 CTA FLOP/cycle | 全 GPU 运行下 FLOP/SM覆盖周期 | 全 GPU 样本最小–最大 |',
      '|---|---:|---:|---:|']
    for name in specs:
        single=next(r for r in summaries if (r['case'],r['scope'],r['iterations'])==(name,'single_cta',65536))
        full=next(r for r in summaries if (r['case'],r['scope'],r['iterations'])==(name,'full_gpu',65536))
        report.append(f"| {name} | {single['median']:.3f} | {full['median']:.3f} | {full['min']:.3f}–{full['max']:.3f} |")
    report+=['','## 配对比较','','每轮在同一轮号中配对；区间为对 12 个轮次比值重采样的中位数 percentile 区间，仅描述本次运行的不确定性。','',
      '重复值接近一致时，bootstrap 区间会退化或在显示精度下等宽；它不代表系统误差为零，也不能替代时钟、布局与执行边界验证。','',
      '| 范围 | 比较 | 比值中位数 | bootstrap 95% |','|---|---|---:|---:|']
    for p in pairs:report.append(f"| {p['scope']} | {p['comparison']} | {p['median_ratio']:.6f} | {p['bootstrap95'][0]:.6f}–{p['bootstrap95'][1]:.6f} |")
    report+=['','## 计时开销与长度敏感性','','空窗口保持相同计时、排空/归约与同步结构，但没有计算工作。它只用于量级检查，不能直接扣除所有循环或等待开销。','',
      '| 配置 | 空窗口中位 cycles | 相对最短单 CTA 窗口 | 三种长度吞吐最大变化 |','|---|---:|---:|---:|']
    for f in fits:
        c=next(r for r in summaries if (r['case'],r['scope'],r['iterations'])==(f['case'],'single_cta',8192))
        report.append(f"| {f['case']} | {c['empty_control_cycles']:.0f} | {100*c['empty_to_short_window_ratio']:.4f}% | {100*f['throughput_length_spread']:.4f}% |")
    report+=['','## 限制与验证状态','',
      '- 所有正式及控制输出逐元素检查有限性与精确值，均通过；时钟、工作量、SM 覆盖及样本矩阵独立重算通过。',
      '- WGMMA wait 3 的完整 kernel 时间预热检查在 30 次上限内未达到最后 5 次 CV≤1% 的预设阈值，保留 failed 状态；其周期结果不能自动升级为时间/功耗稳态结论。',
      '- Nsight Compute 返回 ERR_NVGPUCTRPERM；本账户没有 GPU 性能计数器权限。目标 SASS 和等待指令已检查，但动态指令计数仍未验证，不声称 profiler 闭环。',
      '- kernel 参数、循环长度固定；配置和长度每轮随机交错；不同配置前先执行一次短预热。12 轮来自同一个进程与分配，不等于 12 次独立机器会话。',
      '- 旧的固定参考周期结果仍可用于归一化比较，但不作为实测 SM 周期能力。',
      '- 复测只覆盖预先指定的代表配置，未重新验证所有数据类型和形状；没有拆出纯指令延迟、独立发射端口能力或完全排除输入数据模式影响。', '',
      '另有独立的 [非均匀输入验证](../20260930-audit-validation/validation.jsonl)：FP32 精确计数输入、FP16 MMA 非均匀片段、FP16 WGMMA 非均匀 SMEM 布局，分别执行 1/3/17 轮，共 9 项均逐元素零误差通过。这些是正确性补测，不是非均匀输入的性能数据。','',
      '原始记录 raw.jsonl、预热 run.stderr、SASS sass.txt、计数器拒绝 ncu_*.csv、完整统计 summary.json。']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(measured=504,controls=84,warmups=summary['warmups'],pairs=pairs,max_empty_fraction=max(r['empty_to_short_window_ratio'] for r in summaries))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path)
    analyze(parser.parse_args().results)
