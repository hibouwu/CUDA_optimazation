#!/usr/bin/env python3
"""Conditional arithmetic on the smoke records, never observer qualification."""
import hashlib,json,statistics,subprocess,sys
from fractions import Fraction
from pathlib import Path
r=Path(__file__).resolve().parent
out=r/'reanalysis/conditional-brackets-v1';out.mkdir(parents=True,exist_ok=False)
subprocess.run([sys.executable,str(r/'verify.py')],check=True,stdout=subprocess.DEVNULL)
sys.path.insert(0,str(r/'analysis_snapshot'))
from r09_clock_brackets import common_rate_interval
report=json.loads((r/'verification.json').read_text());assert report['all_records_valid']
records={v['arm']:v['decoded']['records'] for v in report['results']}
pairs=[];deltas=[]
for cta in range(132):
    rows=sorted([v for v in records['guarded_bracket'] if v['cta']==cta],key=lambda v:v['role'])
    assert [v['role'] for v in rows]==[1,2] and rows[0]['sm']==rows[1]['sm']
    windows=[(v['window']['begin'],v['window']['end']) for v in rows]
    result=common_rate_interval(windows,interval_tolerance_ns=128)
    # Recompute interval inequalities and exact optimum without the helper.
    bounds=[(max(0,e[0]-b[2]),e[2]-b[0],e[1]-b[1]) for b,e in windows]
    assert all(n>=3200 for lo,hi,n in bounds)
    lower=max(Fraction(lo,n+128) for lo,hi,n in bounds)
    upper=min(Fraction(hi,n-128) for lo,hi,n in bounds)
    minimum=max([Fraction(0)]+[Fraction(ni*loj-nj*hii,loj+hii) for loi,hii,ni in bounds for loj,hij,nj in bounds if loj])
    assert (lower<=upper)==result['feasible']
    assert minimum==Fraction(*result['minimum_interval_slack_fraction'])
    exact_lower=max(Fraction(lo)/(Fraction(n)+minimum) for lo,hi,n in bounds)
    exact_upper=min(Fraction(hi)/(Fraction(n)-minimum) for lo,hi,n in bounds if n>minimum)
    assert exact_lower<=exact_upper
    pairs.append(dict(cta=cta,sm=rows[0]['sm'],roles=[1,2],windows=windows,result=result,
        independently_checked=True,lower_at_128=[lower.numerator,lower.denominator],upper_at_128=[upper.numerator,upper.denominator]))
for a,b in zip(records['matched_dual'],records['guarded_bracket']):
    assert (a['cta'],a['role'])==(b['cta'],b['role'])
    deltas.append(dict(cta=a['cta'],role=a['role'],sm_changed=a['sm']!=b['sm'],
        ns=b['window']['ns']-a['window']['ns'],cycles=b['window']['cycles']-a['window']['cycles']))
def stats(v):return dict(minimum=min(v),median=statistics.median(v),maximum=max(v))
result=dict(total_interval_allowance_ns=128,cta_pairs=pairs,feasible_pairs=sum(p['result']['feasible'] for p in pairs),
    minimum_total_interval_slack_ns=stats([p['result']['minimum_interval_slack_ns'] for p in pairs]),
    smoke_observer_deltas=deltas,phase_deltas=[dict(role=role,ns=stats([p['ns'] for p in deltas if p['role']==role]),cycles=stats([p['cycles'] for p in deltas if p['role']==role])) for role in (1,2)],
    input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'manifest.json',r/'verification.json',r/'verify.py',r/'analysis_snapshot/r09_clock_brackets.py',Path(__file__))},
    observation_qualified=False,original_phenomenon_inference='inconclusive',
    limits=['Assumes globaltimer read occurs inside the two CLOCK reads; arithmetic does not prove hardware enclosure.',
        'Each CTA may choose its own positive constant rate; no native clocks are subtracted across SMs.',
        '128ns is total interval allowance, not each endpoint. Required slack is a diagnostic, never a revised budget.',
        'One smoke process per arm is not the ten-process reproduction/perturbation study; spatial qualification remains unresolved.',
        'No physical DVFS conclusion, source service price, new holdout or cfg_a/c transfer.'])
(out/'conditional-brackets.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(feasible_pairs=result['feasible_pairs'],slack=result['minimum_total_interval_slack_ns'],phases=result['phase_deltas'])))
