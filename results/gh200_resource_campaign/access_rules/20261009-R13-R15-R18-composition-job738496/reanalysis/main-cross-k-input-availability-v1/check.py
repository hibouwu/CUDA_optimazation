#!/usr/bin/env python3
"""Audit input availability without choosing a new reference or rescoring."""
from pathlib import Path
import gzip,hashlib,json
out=Path(__file__).resolve().parent;run=out.parents[1]
prior=run/'reanalysis/main-cfg-c-cross-k-v1/report.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads(prior.read_text());inputs={str(prior):sha(prior)}
for p,h in base['input_sha256'].items():assert sha(Path(p))==h
processes={}
for p in base['processes']:
    case=p['case'];trial=p['trial']
    matches=[]
    for path in (run/'samples'/case).glob('dual-*.json'):
        record=json.loads(path.read_text())
        if record['trial']==trial:matches.append((path,record))
    assert len(matches)==1
    path,record=matches[0];inputs[str(path)]=sha(path)
    assert inputs[str(path)]==base['input_sha256'][str(path)]
    raw=run/record['raw'];inputs[str(raw)]=sha(raw);assert inputs[str(raw)]==record['raw_sha256']
    with gzip.open(raw,'rt') as f:packets=[json.loads(line) for line in f]
    events={p['event']:p for p in packets};assert len(events)==len(packets)
    setup,call=events['setup'],events['call'];count=call['warmup_calls']
    assert count==len(call['warmup_us']) and record['host_start_ns']<record['host_stop_ns']
    processes[case,trial]=dict(case=case,trial=trial,host_start_ns=record['host_start_ns'],host_stop_ns=record['host_stop_ns'],
        warmup_calls=count,lda=setup['lda'],alloc_lda=setup['alloc_lda'],ldb=setup['ldb'],alloc_ldb=setup['alloc_ldb'])
pairs=[]
for pitch in ('aligned','a16','b16'):
    for trial in range(10):
        a=processes[f'cfg_c_r13_comp_k1024_{pitch}',trial];b=processes[f'cfg_c_r13_comp_k4096_{pitch}',trial]
        gap=b['host_start_ns']-a['host_stop_ns']
        previous=[v for (case,t),v in processes.items() if case==a['case'] and v['host_stop_ns']<b['host_start_ns']]
        pairs.append(dict(pitch=pitch,trial=trial,reference=a,target=b,reference_completed_before_target_start=gap>0,
            start_minus_reference_stop_ns=gap,warmup_counts_equal=a['warmup_calls']==b['warmup_calls'],
            prior_same_pitch_reference_count=len(previous)))
report=dict(input_sha256=inputs,pairs=pairs,total_pairs=30,
    chosen_reference_unavailable=sum(not p['reference_completed_before_target_start'] for p in pairs),
    mismatched_warmup_pairs=sum(not p['warmup_counts_equal'] for p in pairs),
    targets_with_no_prior_same_pitch_reference=sum(p['prior_same_pitch_reference_count']==0 for p in pairs),
    original_window_errors_unchanged=True,observation_qualified=False,
    limits=['Host process intervals establish availability only, not device clock alignment or causal pairing.',
        'Reference identity remains original same-trial pairing; no substitute reference, deletion or rescore.',
        'Prior reference counts do not establish local observation validity or static-input prediction.',
        'Adaptive history and A pitch differences prevent isolated K attribution.'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','pairs','limits')}))
