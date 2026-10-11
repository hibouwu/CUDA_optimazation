#!/usr/bin/env python3
"""Compare constant-L rate with per-CTA varying-rate endpoint consistency."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_r13_sm import select_attempts
from analyze_r18 import replay
from r09_clock_brackets import local_clock_witness
from r09_time_contract import constant_rate_slack


def audit(ctas):
    rows=[]
    for c in ctas:
        points=[(c['entry_c'],c['entry_ns']),(c['prod_c'],c['prod_ns'])]
        labels=['entry','producer'];L=[]
        for role,r in enumerate(c['roles']):
            if len(r['tiles'])!=1:
                raise ValueError('two-output original-role mapping required')
            cyc,ns=r['tiles'][0],r['tiles_ns'][0]
            points.extend(zip(cyc,ns))
            labels.extend(str(role)+'_'+name for name in ('FIRST','MAIN','PERMIT','DONE'))
            points.append((r['final_c'],r['final_ns']));labels.append(str(role)+'_final')
            L.append((cyc[1]-cyc[0],ns[1]-ns[0]))
        witness=local_clock_witness(points,declared_rate_cap_ghz=1.98)
        rows.append(dict(cta=c['cta'],sm=c['sm'],constant_L_endpoint_slack_ns=constant_rate_slack(L)['necessary_endpoint_slack_ns'],
                         local_variable_rate_witness=witness,point_labels=labels))
    return dict(ctas=rows,constant_L_endpoint_slack_median_ns=st.median(r['constant_L_endpoint_slack_ns'] for r in rows),
        variable_rate_endpoint_slack_median_ns=st.median(r['local_variable_rate_witness']['minimum_endpoint_slack_ns'] for r in rows),
        variable_rate_endpoint_slack_maximum_ns=max(r['local_variable_rate_witness']['minimum_endpoint_slack_ns'] for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve()
    root=base/'20261010-R09-R13-R18-failure-controls'
    processes=[];provenance={}
    for row in json.loads((root/'cases.json').read_text()):
        if row['id'] not in ('r13_stage_k4096_s4','r13_stage_k4096_s6'):continue
        selected,failed,_=select_attempts(root,row['id'],('dual',))
        if failed or len(selected)!=10:raise ValueError('all declared stage records required')
        for item in selected:
            rec=item['record'];observed=replay(root,rec,row,retain_roles=True)
            processes.append(dict(case=row['id'],trial=rec['trial'],stage=row.get('stages',6),
                                  evidence='development repeated stage process',**audit(observed['ctas'])))
            provenance[str(root/rec['raw'])]=rec['raw_sha256']
    smoke=base/'20261011-R09-prefix-host-check-r5/returned'
    raw=smoke/'dual.jsonl.gz'
    import gzip
    with gzip.open(raw,'rt') as stream:
        events={x['event']:x for line in stream if line.strip() for x in [json.loads(line)]}
    setup=events['setup'];row={k:setup[k] for k in ('config','m','n','k','lda','ldb','ldd','zero_m','zero_n',
        'storage_m','storage_n','swizzle','alloc_lda','alloc_ldb','input_mode','seed')}
    row.update(id='plain-prefix-smoke-dual',kind='ordinary',warmup_policy='common-sw1-16')
    digest=hashlib.sha256(raw.read_bytes()).hexdigest();record=dict(raw=raw.name,raw_sha256=digest,variant='dual',trial=0)
    observed=replay(smoke,record,row,retain_roles=True)
    processes.append(dict(case=row['id'],trial=0,stage=6,evidence='single correctness smoke; no formal timing',**audit(observed['ctas'])))
    provenance[str(raw)]=digest
    result=dict(scope='Per-CTA endpoint existence checks only; all observations remain development or smoke, no clock fitting/prediction.',
        declared_rate_cap_ghz=1.98,cap_scope='Frozen-r2 declared upper rate, not a newly proved physical counter limit.',
        processes=processes,input_sha256=provenance,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implications=['Constant L conversion and arbitrary bounded local f(t) are distinct hypotheses.',
            'A low local endpoint allowance can coexist with a large constant-rate allowance; therefore the latter does not force large local timer errors.',
            'Each CTA has its own observation-conditioned witness. This does not fit a common across-SM trajectory or identify physical DVFS/gating.',
            'Witnesses use target endpoint observations only to prove compatibility. They are forbidden inputs to a free predictor.',
            'Cross-SM nested-interval counterexamples and observation-protocol qualification gaps remain unchanged.'])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'local-clock-consistency.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for stage in (4,6):
        xs=[p for p in processes if p['stage']==stage and p['evidence'].startswith('development')]
        print(stage,'constant endpoint slack median',st.median(p['constant_L_endpoint_slack_median_ns'] for p in xs),
              'variable endpoint slack process median/max',st.median(p['variable_rate_endpoint_slack_median_ns'] for p in xs),max(p['variable_rate_endpoint_slack_maximum_ns'] for p in xs))
    p=processes[-1];print('smoke',p['constant_L_endpoint_slack_median_ns'],p['variable_rate_endpoint_slack_median_ns'],p['variable_rate_endpoint_slack_maximum_ns'])


if __name__=='__main__':main()
