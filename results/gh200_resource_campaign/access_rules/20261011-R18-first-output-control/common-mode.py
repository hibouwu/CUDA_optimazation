#!/usr/bin/env python3
"""Posthoc all-CTA endpoint diagnosis. Never repairs the frozen control gates."""
import gzip,hashlib,json,statistics,sys
from pathlib import Path
r=Path(__file__).resolve().parent
out=r/'reanalysis/main-common-mode-v1'
out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(r/'analysis_snapshot'))
from analyze_r18_first_output import SELECTED_CTAS
from v08_model import scheduled_work
manifest=json.loads((r/'manifest.json').read_text())
for name,h in manifest['analysis_sha256'].items():assert sha(r/'analysis_snapshot'/name)==h
original=json.loads((r/'analysis.json').read_text());assert original['common_L_control_passed'] is False
records=json.loads((r/'returned/records.json').read_text())
assert [{k:p[k] for k in ('id','trial','flag')} for p in records]==manifest['samples']
work=scheduled_work('cfg_a',960,2816,[132,1,1],1)
processes=[];inputs={str(r/'manifest.json'):sha(r/'manifest.json'),str(r/'analysis.json'):sha(r/'analysis.json'),str(Path(__file__)):sha(Path(__file__))}
for row in records:
    path=r/'returned'/row['raw'];assert not row['returncode'] and sha(path)==row['sha256'];inputs[str(path)]=sha(path)
    with gzip.open(path,'rt') as stream:es={x['event']:x for line in stream for x in [json.loads(line)]}
    setup,call=es['setup'],es['call'];trace=call['trace']
    assert setup['gpu_uuid']==manifest['gpu_uuid'] and setup['grid']==[132,1,1] and setup['trace_words']==1318
    assert setup['input_map_m']==1024 and len(trace)==132*1318
    assert all(type(v) is int and 0<=v<2**64 for v in trace)
    local=[]
    for cta in range(132):
        base=cta*1318;assert trace[base+10]==0 and trace[base+2]>0
        for role in (1,2):
            head=base+16+(role-1)*640
            assert trace[base+10+role]==trace[base+2]
            assert trace[base+6+(role-1)*3]==len(work[cta])
            assert trace[head+4:head+6]==[v+1 for v in work[cta][0]]
            first_c,main_c=trace[head:head+2];first_ns,main_ns=trace[head+6:head+8]
            assert main_c>first_c>0 and main_ns>first_ns>0
            local.append(dict(cta=cta,role=role,sm=trace[base+2]-1,selected=cta in SELECTED_CTAS,
                cycles=main_c-first_c,ns=main_ns-first_ns))
    processes.append(dict(id=row['id'],trial=row['trial'],flag=row['flag'],local=local))
def summary(values):return dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
pairs=[]
for trial in range(10):
    arms=[next(p for p in processes if p['trial']==trial and p['flag']==flag) for flag in (0,1)]
    deltas=[]
    for a,b in zip(arms[0]['local'],arms[1]['local']):
        assert (a['cta'],a['role'])==(b['cta'],b['role'])
        deltas.append(dict(cta=a['cta'],role=a['role'],selected=a['selected'],sm_changed=a['sm']!=b['sm'],
            ns=b['ns']-a['ns'],cycles=b['cycles']-a['cycles'],
            apparent_cycle_per_ns_ratio=(b['cycles']/b['ns'])/(a['cycles']/a['ns'])-1))
    groups=[]
    for selected in (True,False):
        rows=[p for p in deltas if p['selected']==selected]
        groups.append(dict(selected=selected,roles=len(rows),ns=summary([p['ns'] for p in rows]),
            cycles=summary([p['cycles'] for p in rows]),
            apparent_cycle_per_ns_ratio=summary([p['apparent_cycle_per_ns_ratio'] for p in rows])))
    pairs.append(dict(trial=trial,groups=groups,sm_changes=sum(p['sm_changed'] for p in deltas),deltas=deltas))
within=[]
for flag in (0,1):
    rows=[p for p in processes if p['flag']==flag]
    for index in range(264):
        vals=[p['local'][index]['ns'] for p in rows]
        within.append(dict(flag=flag,cta=rows[0]['local'][index]['cta'],role=rows[0]['local'][index]['role'],
            ns=summary(vals),all45_absolute_pair_differences=summary([abs(a-b) for i,a in enumerate(vals) for b in vals[:i]])))
result=dict(processes=processes,pairs=pairs,within_arm=within,input_sha256=inputs,
    common_L_control_passed=False,internal_endpoints_qualified=False,
    limits=['Posthoc spatial diagnosis, not a new frozen control or expanded critical-CTA candidate set.',
        'Unselected CTAs share device, input and dependencies; they are not an isolated untreated experiment.',
        'All durations subtract only within the same invocation and SM; paired differences compare durations.',
        'Apparent cycle/ns ratios are interval observations, not hardware frequency or clock-read enclosure proof.',
        'No centering, compensation, outlier deletion or change to original gates.'])
(out/'common-mode.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 首输出共同变化的事后诊断','','原20样本和24角色门槛失败不变。检查原trace已保存的全132CTA首输出，不新增观测或关键候选。','','| trial | 选定24角色 Δns中位 | 其余240角色 Δns中位 | 选定 Δcycle中位 | 其余 Δcycle中位 |','|---|---:|---:|---:|---:|']
for p in pairs:
    a,b=p['groups'];lines.append(f"| {p['trial']} | {a['ns']['median']} | {b['ns']['median']} | {a['cycles']['median']} | {b['cycles']['median']} |")
lines += ['', '全部264角色在每个原配对中SM不变。trial3/8/9的大幅ns变化伴随接近零的cycle变化，并同时出现在未新增五戳的CTA上。它限制了只用新增读取局部开销解释失败的说法；仍不能区分时钟演化、端点读取偏差、历史或全局间接影响。', '其余CTA并非隔离的未处理对照；不能相减它们的中位数修复原门槛，也不能将表观cycle/ns比直接作为物理DVFS或预测输入。', '逐角色、逐trial及每臂45对差全部保存，未删除异常值。']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print('20 processes; 5280 local role windows; 2640 paired duration differences; frozen failure unchanged')
