from pathlib import Path
from fractions import Fraction as F
import hashlib,json
out=Path(__file__).resolve().parent
other=out.parent/'B-20261011-input-latency-v1'
a=json.loads((other/'B-input-latency.json').read_text())
b=json.loads((out/'rerun/B-input-latency.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for report in (a,b):
 for path,digest in report['input_sha256'].items():assert sha(Path(path))==digest
assert {k:v for k,v in a.items() if k!='input_sha256'}=={k:v for k,v in b.items() if k!='input_sha256'}
common=set(a['input_sha256'])&set(b['input_sha256'])
assert all(a['input_sha256'][k]==b['input_sha256'][k] for k in common)
assert {Path(k).name:v for k,v in a['input_sha256'].items() if k not in common}=={Path(k).name:v for k,v in b['input_sha256'].items() if k not in common}
f=lambda x:F(str(x))
checks=[]
for case in b['results']:
 V=H=None
 for model in case['models']:
  demands=model['first_copy_admissions'];assert [d['kind'] for d in demands]==['issue_a','issue_b0','issue_b1']
  shift=f(model['shift_ns']);tau=f(case['prescribed_tau_ns'])
  ready=[f(d['source_ready_ns']) for d in demands]
  offsets=[f(d['arrival_ns'])-shift for d in demands]
  assert ready==sorted(ready) and offsets==sorted(offsets)
  # FIFO recurrence C_i=max(C_(i-1),R_i,delta+tau+a_i)+duration_i.
  # Expand constants and the coefficient-one predecessor independently.
  constant=F(0);linear=None
  for d,R,offset in zip(demands,ready,offsets):
   duration=f(d['work'])*f(b['parameters']['destination_service_ns'])
   constant=max(constant,R)+duration
   linear=(offset if linear is None else max(linear,offset))+duration
  if V is None:V,H=constant,linear
  assert abs(V-constant)<F(1,10**9) and abs(H-linear)<F(1,10**9)
  expected=max(constant,shift+tau+linear)
  assert abs(expected-f(model['FIRST_ns']))<F(1,10**9)
  assert abs(constant-f(case['source_suffix_ns']))<F(1,10**9)
  assert abs(linear-f(case['own_path_suffix_ns']))<F(1,10**9)
 # For x=tau+H, max(V,x+delta)-max(V,x) is 0, x+delta-V, or delta.
 # In the middle case V-delta < x < V, hence strictly between 0 and delta.
 checks.append(dict(family=case['family'],pitch_bytes=case['pitch_bytes'],constant_fraction=str(V),linear_offset_fraction=str(H),piecewise_response=['0 if x+delta<=V','x+delta-V if x<V<x+delta','delta if V<=x'],bound_for_all_nonnegative_tau=True))
assert len(b['observed_trials'])==20
for trial in b['observed_trials']:
 assert 0<=trial['producer_shift_ns']<b['constant_floor_bound']['delta_domain_open_upper_ns']
 assert trial['S_change_ns']>0 and trial['S_change_cycles']>0
regression=json.loads((out/'default-regression.json').read_text())
assert len(regression['checks'])==12 and regression['total_demands']==248832
result=dict(common_hashes=len(common),module_path_only_differences=len(a['input_sha256'])-len(common),rerun_scenarios=12,default_complete_dictionaries_equal=12,demands_per_replay=248832,independent_FIFO_checks=checks,observed_positive_S_pairs=20,observation_qualified=False,new_numeric_replay=False)
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='independent_FIFO_checks'}))
