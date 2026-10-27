#!/usr/bin/env python3
"""Recheck R10 responses in direct native-cycle and globaltimer windows."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from analyze_r18 import replay
from analyze_r10_supply_events import GPU,span
from supply_model import cfg_b_source_program


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False)
    inputs={}
    def read(path):
        inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    old_path=root/'Q03-source-service/R10-layout-cache/20261010-R10-raster-pitch/reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json'
    old=read(old_path)
    data={};checks=0
    for run_name in ('Q03-source-service/R10-layout-cache/20261010-R10-raster-pitch','Q03-source-service/R10-layout-cache/20261010-R10-source-pressure'):
        run=root/run_name
        rows=read(run/'cases.json')
        setups={v['case']:v['setup'] for v in read(run/'static_setup.json')}
        for row in rows:
            if row.get('set')=='reference':continue
            setup=setups[row['id']]
            if setup['gpu_uuid']!=GPU or setup['stages']!=6:
                raise ValueError('R10 comparison is confined to original same-card stage6 cases')
            program=cfg_b_source_program(row,setup)
            processes=[]
            for trial in range(10):
                record=read(run/f'samples/{row["id"]}/dual-{trial:02d}.json')
                raw=run/record['raw']
                inputs[str(raw.resolve())]=hashlib.sha256(raw.read_bytes()).hexdigest()
                if record['returncode'] or inputs[str(raw.resolve())]!=record['raw_sha256']:
                    raise ValueError('raw process identity or execution failed')
                observed=replay(run,record,row,retain_roles=True)
                windows={}
                for actor in observed['ctas']:
                    cta=actor['cta'];coords=program['cta_output_coords'][cta]
                    if actor['work']!=[tuple(c) for c in coords]:
                        raise ValueError('actual work does not match source program')
                    for j,coord in enumerate(coords):
                        role,ordinal=j%2,j//2
                        cy=actor['roles'][role]['tiles'][ordinal]
                        ns=actor['roles'][role]['tiles_ns'][ordinal]
                        if cy!=actor['tiles'][j] or ns!=actor['tiles_ns'][j]:
                            raise ValueError('direct same-role endpoints differ from prior merged view')
                        windows[(cta,j)]=dict(cta=cta,j=j,T=len(coords),coord=coord,role=role+1,sm=actor['sm'],
                            L_cycles=cy[1]-cy[0],L_ns=ns[1]-ns[0])
                processes.append(dict(trial=trial,elapsed_us=observed['elapsed_us'],windows=windows))
                checks+=1
            data[row['id']]=processes
    pairs=dict(along_m=('r10_raster_along_m_p0','r10_raster_along_m_p32'),
        along_n=('r10_raster_along_n_p0','r10_raster_along_n_p32'))
    pairs.update({s:(f'r10_source_{s}_p0',f'r10_source_{s}_p32') for s in ('single36','single126','persistent132')})
    fields=('L_ns_base','L_ns_variant','L_ns_delta','L_ns_ratio',
        'L_cycles_base','L_cycles_variant','L_cycles_delta','L_cycles_ratio')
    groups=[];all_windows=[]
    for previous in old['comparisons']:
        label,j,T=previous['pair'],previous['j'],previous['T']
        base,variant=pairs[label];trials=[]
        for trial in range(10):
            a,b=data[base][trial],data[variant][trial]
            if a['windows'].keys()!=b['windows'].keys():raise ValueError('paired work counts differ')
            windows=[]
            for key,x in a['windows'].items():
                y=b['windows'][key]
                if any(x[k]!=y[k] for k in ('coord','T','role')):raise ValueError('paired work/role differs')
                if (x['j'],x['T'])!=(j,T) or (label not in ('along_m','along_n') and x['cta']>=36):continue
                w=dict(pair=label,trial=trial,cta=x['cta'],j=j,T=T,coord=x['coord'],role=x['role'],
                    base_sm=x['sm'],variant_sm=y['sm'],same_sm=x['sm']==y['sm'])
                for unit in ('ns','cycles'):
                    p,q=x['L_'+unit],y['L_'+unit]
                    if min(p,q)<=0:raise ValueError('positive direct windows required')
                    w.update({f'L_{unit}_base':p,f'L_{unit}_variant':q,
                        f'L_{unit}_delta':q-p,f'L_{unit}_ratio':q/p-1})
                w['cycle_per_ns_proxy_base']=x['L_cycles']/x['L_ns']
                w['cycle_per_ns_proxy_variant']=y['L_cycles']/y['L_ns']
                w['proxy_ratio']=w['cycle_per_ns_proxy_variant']/w['cycle_per_ns_proxy_base']-1
                if abs((1+w['L_ns_ratio'])*(1+w['proxy_ratio'])-(1+w['L_cycles_ratio']))>1e-12:
                    raise ValueError('per-window multiplicative identity does not close')
                windows.append(w)
            all_windows.extend(windows)
            trials.append(dict(trial=trial,**{f:statistics.median(w[f] for w in windows) for f in fields},
                proxy_ratio_median=statistics.median(w['proxy_ratio'] for w in windows),
                same_sm=sum(w['same_sm'] for w in windows),windows=len(windows),
                event_ratio=b['elapsed_us']/a['elapsed_us']-1,
                signs={u:dict(positive=sum(w[f'L_{u}_delta']>0 for w in windows),
                    negative=sum(w[f'L_{u}_delta']<0 for w in windows),zero=sum(w[f'L_{u}_delta']==0 for w in windows))
                    for u in ('ns','cycles')}))
        current={f:span([t[f] for t in trials]) for f in fields}
        if current!=previous['observed'] or [{k:v for k,v in t.items() if k in (*fields,'trial')} for t in trials]!=previous['trials']:
            raise ValueError('native/ns observations changed from previous independent replay')
        groups.append(dict(pair=label,j=j,T=T,observed=current,trials=trials,
            proxy_ratio=span([t['proxy_ratio_median'] for t in trials]),
            positive_median_trials={u:sum(t[f'L_{u}_ratio']>0 for t in trials) for u in ('ns','cycles')},
            same_sm_window_pairs=sum(t['same_sm'] for t in trials),window_pairs=sum(t['windows'] for t in trials)))
    for name in ('analyze_r10_clock_consistency.py','analyze_r18.py','analyze_r10_supply_events.py',
            'supply_model.py','v06_model.py','v06_run.py'):
        path=Path(__file__).with_name(name)
        inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(scope='Post-reveal direct clock-domain diagnosis; no new fit, correction or observation qualification.',
        raw_processes_replayed=checks,numerically_checked_values=checks*4096,
        exact_previous_groups=len(groups),window_pairs=len(all_windows),groups=groups,input_sha256=inputs,
        limits=['Each native-cycle window uses one CTA/role/SM within one invocation; no cross-SM clock subtraction.',
            'B0/B32 process pairs retain their original trial and SM identities; no filtering or re-pairing.',
            'Each globaltimer interval uses its directly saved FIRST/MAIN endpoints, without interpolation.',
            'Cycle/ns is an endpoint ratio proxy, not a physical clock or DVFS measurement.',
            'The per-window multiplicative identity is checked before medians; medians do not generally obey it.',
            'Clock-domain agreement does not identify source versus destination, retirement, scheduling or observer causality.',
            'The new first-output control failed in a distinct protocol; it neither qualifies these old cuts nor automatically invalidates them.',
            'All old failures remain. These data cannot set a free-prediction frequency, repair failed gates or calibrate new prices.'])
    (output/'B-r10-clock-consistency.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with gzip.open(output/'B-r10-clock-window-pairs.json.gz','wt') as stream:
        json.dump(all_windows,stream,separators=(',',':'),allow_nan=False)
    lines=['# R10：native cycles与直接ns窗口的响应','',
        '100进程、409600数值复算；17组原汇总及每trial值精确复现。每个窗口取同CTA同role的FIRST/MAIN；不以目标cycle/ns换算预测。','',
        '| 配对 j/T | ns响应中位[范围] | cycle响应中位[范围] | cycle正响应trial数 | 同SM窗口配对 |',
        '|---|---:|---:|---:|---:|']
    for g in groups:
        values=[]
        for u in ('ns','cycles'):
            v=g['observed']['L_'+u+'_ratio']
            values.append(f'{100*v["median"]:.3f}% [{100*v["minimum"]:.3f},{100*v["maximum"]:.3f}]')
        lines.append(f'| {g["pair"]} {g["j"]}/{g["T"]} | '+' | '.join(values)+
            f' | {g["positive_median_trials"]["cycles"]}/10 | {g["same_sm_window_pairs"]}/{g["window_pairs"]} |')
    lines+=['','每个原窗口的native/ns/表观斜率比、SM和符号均保存于gzip JSON；全部原配对保留。',
        '表观斜率比的乘法恒等式仅在逐窗口检查，不把不同中位数拼成时间线，也不命名为物理频率变化。',
        '这里区分强cycle响应与只在ns中出现的小响应，不定位硬件资源，不修复观察资格，不给新价格。','']
    (output/'B-r10-clock-consistency.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),processes=checks,exact_previous_groups=len(groups),window_pairs=len(all_windows))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve())
