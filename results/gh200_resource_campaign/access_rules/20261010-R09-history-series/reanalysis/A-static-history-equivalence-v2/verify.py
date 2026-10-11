#!/usr/bin/env python3
"""Verify archived hashes and exact forecast/bound arithmetic, without model calls."""
import argparse,csv,gzip,hashlib,json
from fractions import Fraction
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
a=parser.parse_args();root=a.archive
report=json.loads((root/'equivalence.json').read_text())
checked=0
for name,digest in report['input_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    checked+=1
assert hashlib.sha256((root/'audit.py').read_bytes()).hexdigest()==report['analysis_sha256']
sig=json.load(gzip.open(root/'static-signature.json.gz','rt'))
packed=json.dumps(sig,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
assert hashlib.sha256(packed).hexdigest()==report['signature_sha256']
forecasts=json.load(gzip.open(root/'event-identity.json.gz','rt'))
assert len(forecasts)==1160 and len({(r['case'],r['trial'],r['target_index']) for r in forecasts})==1160
previous=next(Path(p) for p in report['input_sha256'] if p.endswith('/A-history-forecast-v1/call-forecasts.csv'))
with previous.open() as stream:
    old={(r['case'],int(r['trial']),int(r['target_index'])):r for r in csv.DictReader(stream) if r['method']=='previous_call'}
switch=[]
for row in forecasts:
    expected=old[row['case'],row['trial'],row['target_index']]
    assert row['static_normalization_ratio']=='1'
    assert row['prediction_us']==float(expected['event_prediction_us'])
    assert row['actual_us']==float(expected['event_actual_us'])
    assert row['error']==float(expected['free_event_error'])==row['prediction_us']/row['actual_us']-1
    if row['target_index']==0 and row['swizzle']==8:
        switch.append(Fraction(str(row['prediction_us']))/Fraction(str(row['actual_us'])))
lo,hi=min(switch),max(switch)
r=2/(lo+hi);floor=(hi-lo)/(hi+lo)
assert len(switch)==20
assert str(r)==report['first_switch_oracle']['fitted_static_ratio']
assert str(floor)==report['first_switch_oracle']['maximum_relative_floor']
assert max(abs(r*x-1) for x in switch)==floor
assert r*lo-1==-floor and r*hi-1==floor
assert all(not s['card_supported'] for s in report['support'])
result=dict(input_hashes=checked,exact_forecast_identities=len(forecasts),
    switch_bound_terms=len(switch),exact_lower_bound=str(floor),passed=True,
    scope='Hash/arithmetic check; extraction of complete static consumption remains audited against frozen source.')
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
