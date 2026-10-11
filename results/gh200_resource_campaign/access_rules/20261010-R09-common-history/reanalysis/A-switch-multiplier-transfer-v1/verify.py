#!/usr/bin/env python3
"""Recheck all archive identities and exact scores/bounds without audit imports."""
from fractions import Fraction
import gzip,hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parent
report=json.loads((root/'transfer.json').read_text())
for filename,digest in report['input_sha256'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
assert hashlib.sha256((root/'audit.py').read_bytes()).hexdigest()==report['analysis_sha256']
with gzip.open(root/'event-scores.json.gz','rt') as stream:rows=json.load(stream)
ratio=Fraction(report['coefficient']['value'])
assert len(rows)==1200 and len({(r['dataset'],r['case'],r['trial'],r['target_index']) for r in rows})==1200
for row in rows:
    p,y=Fraction(row['previous_event_us']),Fraction(row['actual_event_us'])
    expected=ratio if (row['previous_swizzle'],row['current_swizzle'])==(1,8) else Fraction(1)
    assert Fraction(row['multiplier'])==expected
    assert Fraction(row['prediction_us'])==expected*p
    assert Fraction(row['error'])==expected*p/y-1
    assert row['error_float']==float(Fraction(row['error']))
bound=json.loads((root/'multiplier-bound.json').read_text())
assert hashlib.sha256((root/'bound.py').read_bytes()).hexdigest()==bound['analysis_sha256']
assert hashlib.sha256((root/'event-scores.json.gz').read_bytes()).hexdigest()==next(iter(bound['input_sha256'].values()))
for key,group in bound['groups'].items():
    values=[r for r in rows if (r['previous_swizzle'],r['current_swizzle'])==(1,8)
            and (key=='all_first_switches' or r['dataset']==('prior_development_run' if key=='prior_twenty' else 'cross_run_development'))]
    q=[Fraction(r['previous_event_us'])/Fraction(r['actual_event_us']) for r in values]
    lo,hi=min(q),max(q);r=Fraction(group['optimum_multiplier']);floor=Fraction(group['maximum_relative_floor'])
    assert len(q)==group['n'] and r==2/(lo+hi) and floor==(hi-lo)/(hi+lo)
    assert max(abs(r*x-1) for x in q)==floor
    assert Fraction(group['ten_percent_multiplier_lower'])==Fraction(9,10)/lo
    assert Fraction(group['ten_percent_multiplier_upper'])==Fraction(11,10)/hi
    assert group['ten_percent_feasible']==(Fraction(9,10)/lo<=Fraction(11,10)/hi)
    for witness,expected_q in ((group['minimum_q_witness'],lo),(group['maximum_q_witness'],hi)):
        assert witness in values
        assert Fraction(witness['previous_event_us'])/Fraction(witness['actual_event_us'])==expected_q
assert bound['groups']['common_history_ten']['floor_float']>.1
result=dict(input_hashes=len(report['input_sha256']),exact_scores=len(rows),bound_groups=len(bound['groups']),
    common_history_bound=bound['groups']['common_history_ten']['maximum_relative_floor'],passed=True)
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
