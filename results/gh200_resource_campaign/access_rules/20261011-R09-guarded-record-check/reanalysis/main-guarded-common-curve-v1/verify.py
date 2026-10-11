#!/usr/bin/env python3
"""Raw two-SM equal-timer counterexample plus saved global certificate replay."""
from pathlib import Path
import gzip,hashlib,json,subprocess,sys
out=Path(__file__).resolve().parent;run=out.parents[1]
a=run/'reanalysis/A-guarded-common-curve-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
check=subprocess.run([sys.executable,str(a/'verify.py')],check=True,capture_output=True,text=True)
with gzip.open(a/'audit.json.gz','rt') as f:report=json.load(f)
raw=run/'returned/raw/guarded_bracket.jsonl.gz'
assert sha(raw)==report['input_sha256'][str(raw)]
with gzip.open(raw,'rt') as f:packets=[json.loads(line) for line in f]
assert len({p['event'] for p in packets})==len(packets)
events={p['event']:p for p in packets};trace=events['call']['trace']
by_sm={}
for cta in range(132):
    b=trace[cta*1552:(cta+1)*1552];sm=b[2]-1
    assert sm not in by_sm and b[11]==b[12]==b[2]
    points=[]
    for role in (1,2):
        t=b[16+(role-1)*768:16+(role-1)*768+12]
        points.extend([dict(role=role,event='FIRST',before=t[0],timer=t[6],after=t[10]),
                       dict(role=role,event='MAIN',before=t[1],timer=t[7],after=t[11])])
    assert [[p['before'],p['timer'],p['after']] for p in points]==report['original_stamps'][str(sm)]
    by_sm[sm]=dict(cta=cta,points=points)
# The archived zero-allowance negative cycle identifies these four raw endpoints.
windows=[]
for sm,begin_index,end_index in ((57,1,3),(68,2,3)):
    p=by_sm[sm];begin=p['points'][begin_index];end=p['points'][end_index]
    lower=end['before']-begin['after'];upper=end['after']-begin['before']
    assert 0<lower<=upper
    windows.append(dict(sm=sm,cta=p['cta'],begin=begin,end=end,cycle_increment_interval=[lower,upper]))
assert windows[0]['begin']['timer']==windows[1]['begin']['timer']
assert windows[0]['end']['timer']==windows[1]['end']['timer']
gap=windows[1]['cycle_increment_interval'][0]-windows[0]['cycle_increment_interval'][1]
assert gap==375
certificate=report['branches']['uncapped']['zero_certificate']
assert certificate['negative_cycle_bound_scaled_cycles']==-gap
result=dict(raw_sha256=sha(raw),A_report_sha256=sha(a/'audit.json.gz'),
    verification_script_sha256=sha(a/'verify.py'),certificate_check=json.loads(check.stdout),
    raw_stamps_matched=528,equal_timer_windows=windows,
    common_timer_interval_ns=windows[0]['end']['timer']-windows[0]['begin']['timer'],
    disjoint_increment_gap_cycles=gap,
    minimum_integer_extra_endpoint_allowance_ns={k:v['minimum_integer_additional_endpoint_allowance_ns'] for k,v in report['branches'].items()},
    observation_qualified=False,
    limits=['Posthoc witness explaining the existing conditional contradiction; not an independent holdout.',
        'Assumes each timer sample is inside its before/after clock bracket and zero additional timer shift.',
        'Only same-SM cycle differences used. Common F implies equal increments between identical global times.',
        'Does not establish SM physical frequency differences, read enclosure, timer precision, or service prices.'])
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(certificate_check=result['certificate_check'],timer_interval_ns=result['common_timer_interval_ns'],increments=[v['cycle_increment_interval'] for v in windows],gap_cycles=gap)))
