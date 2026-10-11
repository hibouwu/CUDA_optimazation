#!/usr/bin/env python3
"""R13 independent output reference, lifecycle trace checks and conditional rates."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import struct
from functools import lru_cache

def half_bits(x):return struct.unpack('<H',struct.pack('<e',x))[0]
def input_value(row,k,a):return (((row*7+k*13) if a else (k*5+row*11))%17-8)/32

def sw128(offset):
    # CuTe Swizzle<3,4,3>: byte address bits [9:7] xor into [6:4].
    return offset ^ ((offset & 0x380)>>3)

@lru_cache(maxsize=2)
def final_input(n):
    values=[0]*((128+n)*64)
    for a,outer,base in ((True,128,0),(False,n,128*64)):
        for row in range(outer):
            for k in range(64):
                linear=(row*64+k) if a else (row%64+64*k+4096*(row//64))
                physical=sw128(linear*2)//2
                values[base+physical]=half_bits(input_value(row,31*64+k,a))
    return struct.pack('<'+'H'*len(values),*values)

@lru_cache(maxsize=6)
def accum_reference(n,consumer):
    if consumer!=2:
        return [input_value(t%128,31*64,True)+64/1024 if consumer==1 else (t+129)/1024 for t in range(256)]
    out=[]
    for t in range(256):
        local=t%128
        for j in range(n//2):
            row=(t//128)*64+(local//32)*16+(local%32)//4+((j//2)%2)*8
            col=(local%4)*2+j%2+(j//4)*8
            terms=[int(input_value(row,k,True)*32)*int(input_value(col,k,False)*32) for k in range(17)]
            out.append((sum(terms)*(2048//17)+sum(terms[:2048%17]))/1024)
    return out

def binary(folder,row,name):
    raw=gzip.open(folder/(name+'.gz'),'rb').read();meta=row['files'][name]
    if len(raw)!=meta['bytes'] or hashlib.sha256(raw).hexdigest()!=meta['sha256']:
        raise ValueError('witness identity mismatch')
    return raw

def check(folder,row):
    n=row['n'];blocks=row['blocks'];consumer=row['consumer'];stages=row['stages']
    assert row['b_major']=='MN' and row['b_global_layout']=='row_major_K_N' and row['ldb']==n
    assert row['wgmma_wait']==(0 if stages==1 else 1) if consumer==2 else row['wgmma_wait']==-1
    assert row['tiles']==32 and row['reserved_slots']==4 and row['threads']==384
    assert row['tile_bytes']==(128+n)*128 and row['input_bytes']==blocks*32*(128+n)*128
    ref=accum_reference(n,consumer);raw=binary(folder,row,'accum.f32')
    packed=struct.pack('<'+'f'*len(ref),*ref)
    if raw!=packed*blocks:raise ValueError('accumulator CPU mismatch')
    raw_input=binary(folder,row,'last-input.u32')
    if raw_input!=final_input(n)*blocks:raise ValueError('final TMA tile/SW128 CPU mismatch')
    stamps=row['stamps'];assert len(stamps)==blocks
    for bc,ec,bn,en,first,last in stamps:assert bc<ec and bn<en and first==last
    elapsed=stamps[0][1]-stamps[0][0] if row['scope']=='one_cta' else max(s[3] for s in stamps)-min(s[2] for s in stamps)
    assert elapsed==row['elapsed']
    if row['variant'].startswith('trace'):
        raw=binary(folder,row,'trace.u64');assert len(raw)==blocks*32*2*5*8
        events=list(struct.iter_unpack('<QQQQQ',raw))
        for block in range(blocks):
            tiles=events[block*64:(block+1)*64]
            if row.get('trace_cta0_only',False) and block!=0:
                assert all(e==(0,0,0,0,0) for e in tiles)
                continue
            selected=row['trace_tile']
            for t in range(32):
                pair=tiles[t*2:t*2+2]
                if t!=selected:
                    assert all(e==(0,0,0,0,0) for e in pair)
                    continue
                if row['trace_pair']==2:
                    assert block==0 and selected+stages<32
                    assert pair[1]==(0,0,0,0,0) # one producer record, no invented WG timestamp
                    issue,wait,consume,slot_reusable_observed,refill_issue=pair[0]
                    assert issue==wait==consume==0
                    assert stamps[0][0]<=slot_reusable_observed<=refill_issue<=stamps[0][1]
                    assert slot_reusable_observed>0
                else:
                    for issue,wait,consume,retire_arrive_start,refill in pair:
                        assert refill==0
                        if row['trace_pair']==0:
                            assert 0<issue<=wait and consume==retire_arrive_start==0
                        elif row['trace_pair']==1:
                            assert 0<consume<=retire_arrive_start and issue==wait==0
                        else:raise ValueError('invalid trace event pair')
    return len(ref)*blocks

def covered_cycles(row):
    by_sm={}
    for bc,ec,bn,en,first,last in row['stamps']:
        if first!=last or ec<=bc:raise ValueError('invalid same-SM window')
        by_sm.setdefault(first,[]).append((bc,ec))
    # Subtract clocks only within each SM, then sum coverage spans (including gaps).
    spans={str(sm):max(end for begin,end in windows)-min(begin for begin,end in windows)
           for sm,windows in by_sm.items()}
    total=sum(spans.values())
    return dict(trial=row['trial'],sm_count=len(spans),ctas=row['blocks'],
                covered_cycles_by_sm=spans,total_SM_covered_cycles=total,
                requested_bytes=row['input_bytes'],B_per_SM_covered_cycle=row['input_bytes']/total)

def archive_gate(root):
    for name,digest in json.loads((root/'source_hashes.json').read_text()).items():
        if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('source identity mismatch: '+name)
    for name,digest in json.loads((root/'build/binary_hashes.json').read_text()).items():
        if hashlib.sha256((root/'build'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('binary identity mismatch: '+name)
    return {c['id']:c for c in json.loads((root/'cases.json').read_text())}

def plot_report(destination,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,n in zip(axes,(128,256)):
        for consumer,label in ((0,'wait-retire'),(1,'64 dependent FFMA'),(2,'WGMMA')):
            selected=sorted((r for r in rows if r['n']==n and r['consumer']==consumer),
                            key=lambda r:r['stages'])
            if selected:
                ax.errorbar([r['stages'] for r in selected],
                    [r['mean_B_per_SM_covered_cycle'] for r in selected],
                    yerr=[[r['mean_B_per_SM_covered_cycle']-r['min_B_per_SM_covered_cycle'] for r in selected],
                          [r['max_B_per_SM_covered_cycle']-r['mean_B_per_SM_covered_cycle'] for r in selected]],
                    fmt='o-',label=label)
        ax.set_title(f'128 x {n} x 64, A K / B MN')
        ax.set_xlabel('active slots (four reserved)');ax.set_ylabel('requested B / SM-covered cycle')
        ax.set_xticks([1,2,4]);ax.grid(alpha=.25);ax.legend(fontsize=8)
    figure.tight_layout();figure.savefig(destination/'supply.png',dpi=160);plt.close(figure)

def analyze(root,output=None):
    destination=root if output is None else output.resolve()
    if output is not None:destination.mkdir(parents=True,exist_ok=False)
    planned=archive_gate(root)
    groups={};checked=0;env=json.loads((root/'environment.json').read_text())
    for p in (root/'samples').glob('*/*/result.json'):
        row=json.loads(p.read_text());assert row['gpu_uuid']==env['gpu_uuid']
        if row['configuration']!=planned.get(row['configuration']['id']):
            raise ValueError('unplanned or changed coordinate')
        if row['status']!='measured':raise ValueError('invalid numeric status')
        checked+=check(p.parent,row);groups.setdefault(row['configuration']['id'],[]).append(row)
    if set(groups)!=set(planned):raise ValueError('missing planned cases')
    scopes={r['scope'] for rs in groups.values() for r in rs}
    if len(scopes)!=1:raise ValueError('mixed one-CTA / all-GPU scope')
    pairs={r['trace_pair'] for rs in groups.values() for r in rs if r['variant']!='plain'}
    if len(pairs)!=1:raise ValueError('mixed ready / retire diagnostic runs')
    rows=[];coverage={};pair_records={}
    for key,rs in sorted(groups.items()):
        plain=[r for r in rs if r['variant']=='plain'];trace=[r for r in rs if r['variant'].startswith('trace')]
        expected=10 if plain[0]['scope']=='all_gpu' else len(plain)
        if expected not in (3,10):raise ValueError('invalid process count')
        for subset in (plain,trace):
            if len(subset)!=expected or sorted(z['trial'] for z in subset)!=list(range(expected)):
                raise ValueError('incomplete or duplicate paired trials')
        if len({z['trace_tile'] for z in rs})!=1:raise ValueError('mixed selected tiles')
        x=[r['elapsed'] for r in plain];y=[r['elapsed'] for r in trace];r=plain[0]
        rate=r['input_bytes']/statistics.mean(x)
        per_process=[covered_cycles(record) for record in sorted(plain,key=lambda z:z['trial'])]
        coverage[key]=per_process
        rates=[record['B_per_SM_covered_cycle'] for record in per_process]
        pooled=sum(z['requested_bytes'] for z in per_process)/sum(z['total_SM_covered_cycles'] for z in per_process)
        plain_warm=all(z['warmup_converged'] for z in plain)
        trace_warm=all(z['warmup_converged'] for z in trace)
        plain_cv=statistics.stdev(x)/statistics.mean(x)
        eligible=plain_warm and plain_cv<=.05
        observer_ok=trace_warm and abs(statistics.mean(y)/statistics.mean(x)-1)<=.05
        trace_by_trial={z['trial']:z for z in trace}
        pairs_for_case=[dict(trial=z['trial'],plain_elapsed=z['elapsed'],
                            trace_elapsed=trace_by_trial[z['trial']]['elapsed'],
                            relative_change=trace_by_trial[z['trial']]['elapsed']/z['elapsed']-1)
                        for z in sorted(plain,key=lambda z:z['trial'])]
        pair_records[key]=pairs_for_case
        pair_changes=[z['relative_change'] for z in pairs_for_case]
        rows.append(dict(parameter_qualification='plain_conditional_observation' if eligible else 'plain_with_stability_limits',
          observer_qualification='within_5pct_mean' if observer_ok else 'observer_requires_reduction',
          plain_warmup_all=plain_warm,trace_warmup_all=trace_warm,
          pair_relative_mean=statistics.mean(pair_changes),pair_relative_median=statistics.median(pair_changes),
          pair_relative_min=min(pair_changes),pair_relative_max=max(pair_changes),
          pairs_over_5pct=sum(abs(v)>.05 for v in pair_changes),case=key,n=r['n'],stages=r['stages'],consumer=r['consumer'],scope=r['scope'],
          processes=len(x),mean=statistics.mean(x),cv=statistics.stdev(x)/statistics.mean(x),
          trace_fraction=statistics.mean(y)/statistics.mean(x)-1,
          condition_rate=rate,unit='B/cycle/CTA' if r['scope']=='one_cta' else 'GB/s/GPU',
          mean_B_per_SM_covered_cycle=statistics.mean(rates),
          median_B_per_SM_covered_cycle=statistics.median(rates),
          min_B_per_SM_covered_cycle=min(rates),max_B_per_SM_covered_cycle=max(rates),
          cv_B_per_SM_covered_cycle=statistics.stdev(rates)/statistics.mean(rates) if len(rates)>1 else 0,
          pooled_B_per_SM_covered_cycle=pooled,
          warmup_all=all(z['warmup_converged'] for z in rs)))
    (destination/'cases.csv').write_text('')
    if rows:
        with (destination/'cases.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (destination/'paired_observer.json').write_text(json.dumps(pair_records,indent=2)+'\n')
    (destination/'plain_SM_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    summary=dict(cases=rows,cpu_checked_elements=checked,
      coverage_scope='plain processes only; per SM max(end_cycle)-min(begin_cycle); no cross-SM subtraction; spans include gaps',
      qualification='conditional completed 32-tile sequence; wait-return is observed, not internal completion',
      trace_mode='single selected tile and compile-time event pair; v10 reuse observes CTA0 producer only',
      trace_fields=(['unobserved','unobserved','unobserved','slot_reusable_observed','refill_issue_before_A']
                    if next(iter(pairs))==2 else ['issue','wait_return','consume','retire_arrive_start','refill']),
      trace_semantics=(
        'producer word3 after successful empty acquire; word4 before next A TMA; '
        'stores after A/B issue; observed boundaries, not internal completion instant'
        if next(iter(pairs))==2 else
        'word3 is before consumer empty.arrive; both WG arrivals gate producer empty wait; '
        'ready/retire pairs do not timestamp producer reusability or refill'),
      trace_over_5pct=[r['case'] for r in rows if abs(r['trace_fraction'])>.05])
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    parameters=dict(environment=env,scope=next(iter(scopes)),trace_pair=next(iter(pairs)),
      conditions=dict(tiles=32,reserved_slots=4,threads=384,a_major='K',b_major='MN',
                      input='repeat, shared source, finite dyadic',completion='consumer retirement and final CTA publication'),
      cases=rows,limits=['No internal completion instant', 'No physical cache/HBM bandwidth',
                        'Coverage spans include gaps; no cross-SM clock subtraction',
                        'Ready/retire runs not combined as one event timeline',
                        'Mean observer gate is frozen; individual pairs can exceed 5 percent',
                        'Small differences near plain process variability are unresolved, not causal effects'])
    (destination/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    lines=['# R13 条件供给与退役', '',
           '单独分析一个UUID、scope和事件对。所有plain进程分别按SM重算覆盖跨度；trace只检查扰动与观察事件。', '',
           '|配置|plain进程|B/SM-covered-cycle均值|最小–最大|CV|trace均值扰动|逐pair扰动范围|超5%对数|plain资格|observer资格|',
           '|---|---:|---:|---|---:|---:|---|---:|---|---|']
    for r in rows:
        lines.append(f"|{r['case']}|{r['processes']}|{r['mean_B_per_SM_covered_cycle']:.3f}|{r['min_B_per_SM_covered_cycle']:.3f}–{r['max_B_per_SM_covered_cycle']:.3f}|{r['cv_B_per_SM_covered_cycle']:.2%}|{r['trace_fraction']:.2%}|{r['pair_relative_min']:.2%}–{r['pair_relative_max']:.2%}|{r['pairs_over_5pct']}|{r['parameter_qualification']}|{r['observer_qualification']}|")
    lines+=['','该参数描述32tile请求/消费/退休完整序列，不能当物理供给上限。退役arrive-start不是槽释放完成，wait返回不是内部完成。', '',
            'plain_SM_coverage.json保留每个plain进程和每个SM的分母；pooled与进程均值分别给出。paired_observer.json保存逐trial配对；observer不合格不删除正确plain服务，均值门槛和单pair范围分别保留。', '',
            f'完整CPU输出复核元素数：{checked}。']
    try:
        plot_report(destination,rows);lines+=['','![供给曲线](supply.png)']
    except ImportError:lines+=['','matplotlib不可用，保留全部数值表。']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

def cpu_check():
    assert len(final_input(128))==32768 and len(final_input(256))==49152
    for n in (128,256):
        ref=accum_reference(n,2);assert len(ref)==256*n//2
        for row,col in ((0,0),(63,n-1),(127,31)):
            expected=sum(input_value(row,k,True)*input_value(col,k,False) for k in range(2048))
            assert expected*1024==int(expected*1024)
        assert sorted(sw128(i*2)//2 for i in range(n*64))==list(range(n*64))
        assert sorted(sw128(2*(c%64+64*k+4096*(c//64)))//2
                      for c in range(n) for k in range(64))==list(range(n*64))
    assert accum_reference(128,1)[0]==input_value(0,31*64,True)+64/1024
    for stages in (1,2,4):
        issued=set(range(stages));retired=set()
        for tile in range(32):
            assert tile in issued
            if stages==1:retired.add(tile)
            elif tile>0:retired.add(tile-1)
            next_tile=tile+stages-1 if stages>1 else tile+1
            if next_tile<32 and next_tile not in issued:
                assert next_tile-stages in retired
                issued.add(next_tile)
        retired.add(31)
        assert retired==set(range(32))
    sample=dict(trial=0,blocks=3,input_bytes=400,stamps=[
        [100,140,1,2,0,0],[160,200,3,4,0,0],[900,920,1,2,1,1]])
    covered=covered_cycles(sample)
    assert covered['total_SM_covered_cycles']==120
    assert covered['covered_cycles_by_sm']=={'0':100,'1':20}
    assert covered['B_per_SM_covered_cycle']==400/120
    print('CPU reference: exact dyadic dot, fragment size, SW128 permutation and final tile passed')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path)
    p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.cpu_check:cpu_check()
    elif a.input:analyze(a.input.resolve(),a.output)
    else:p.error('--input or --cpu-check required')
