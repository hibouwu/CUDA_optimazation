#!/usr/bin/env python3
"""Exact empirical minimax bounds for one multiplier of a prior event."""
from fractions import Fraction
import gzip,hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parent
source=root/'event-scores.json.gz'
with gzip.open(source,'rt') as stream:rows=json.load(stream)
switched=[r for r in rows if (r['previous_swizzle'],r['current_swizzle'])==(1,8)]
groups={'all_first_switches':switched,
        'prior_twenty': [r for r in switched if r['dataset']=='prior_development_run'],
        'common_history_ten': [r for r in switched if r['dataset']=='cross_run_development']}
results={}
for name,values in groups.items():
    q=[Fraction(r['previous_event_us'])/Fraction(r['actual_event_us']) for r in values]
    low,high=min(q),max(q)
    multiplier=2/(low+high);floor=(high-low)/(high+low)
    lower=Fraction(9,10)/low;upper=Fraction(11,10)/high
    assert max(abs(multiplier*x-1) for x in q)==floor
    assert multiplier*low-1==-floor and multiplier*high-1==floor
    results[name]=dict(n=len(q),minimum_q=str(low),maximum_q=str(high),
        optimum_multiplier=str(multiplier),maximum_relative_floor=str(floor),floor_float=float(floor),
        ten_percent_multiplier_lower=str(lower),ten_percent_multiplier_upper=str(upper),
        ten_percent_feasible=lower<=upper,
        minimum_q_witness=values[q.index(low)],maximum_q_witness=values[q.index(high)])
assert [results[k]['n'] for k in groups]==[30,20,10]
assert not results['common_history_ten']['ten_percent_feasible']
result=dict(schema='gh200-r09-prior-event-multiplier-bound/1',
    predictor_class='prediction=r*last_completed_prefix_event, one positive r for the entire declared first-switch cohort',
    proof='For q=prefix/target, error is r*q-1. Extreme q values require r>=0.9/q_min and r<=1.1/q_max at 10%. Their intersection is empty for the common-history cohort. Minimax occurs where the two extreme errors are opposite.',
    groups=results,input_sha256={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
    analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limits=['Empirical exact bound on the archived observations; not a population confidence bound or hardware noise limit.',
        'The result rules out this homogeneous one-multiplier class, not richer histories, nonlinear checkpoint rules, or separate service/time states.',
        'All common-history witnesses come from one fixed input/protocol/kernel batch. Cross-run identity is not needed for this bound.',
        'No oracle multiplier replaces the unchanged failed development-transfer coefficient.'])
(root/'multiplier-bound.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:{q:v[q] for q in ('n','floor_float','ten_percent_feasible')} for k,v in results.items()},indent=2))
