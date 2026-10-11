#!/usr/bin/env python3
"""Compare independent rerun and verify subscription margins without fitting."""
from pathlib import Path
import hashlib,json
out=Path(__file__).resolve().parent
other=out.parent/'B-20261011-startup-absorption-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((other/'B-startup-absorption.json').read_text())
b=json.loads((out/'B-startup-absorption.json').read_text())
for report in (a,b):
 for name,digest in report['input_sha256'].items():assert sha(Path(name))==digest
assert {k:v for k,v in a.items() if k!='input_sha256'}=={k:v for k,v in b.items() if k!='input_sha256'}
common=set(a['input_sha256'])&set(b['input_sha256'])
assert all(a['input_sha256'][k]==b['input_sha256'][k] for k in common)
old_only=set(a['input_sha256'])-common;new_only=set(b['input_sha256'])-common
assert {Path(k).name:v for k,v in a['input_sha256'].items() if k in old_only}=={Path(k).name:v for k,v in b['input_sha256'].items() if k in new_only}
checks=[]
for r in b['results']:
 witnesses=r['initial_subscriptions'];assert len(witnesses)==18
 assert {(w['group'],w['kind']) for w in witnesses}=={(g,k) for g in range(6) for k in ('issue_a','issue_b0','issue_b1')}
 assert {w['owner_cta'] for w in witnesses}=={1,54}
 residual=[]
 for w in witnesses:
  assert abs(w['shifted_arrival_ns']-w['arrival_ns']-512)<1e-9
  residual.append(w['owner_join_end_ns']-w['shifted_arrival_ns'])
 assert min(residual)>0 and r['producer_first_refill_slack_ns']>512
 assert r['input_sources_byte_equal']
 assert all(v==0 for v in r['maximum_all_actor_progress_delta_ns'].values())
 assert max(r['common_translation_maximum_error_ns'].values())<=1e-9
 d=r['first_output_delta']
 assert [d[k] for k in ('full_ready_ns','FIRST_ns','MAIN_ns','L_ns')]==[0]*4
 assert d['producer_to_MAIN_ns']==-512
 assert abs(d['issue_end_to_full_ns']+512)<1e-9
 assert r['previous_zero_shift_windows_reproduced']==432
 checks.append(dict(family=r['family'],pitch_bytes=r['pitch_bytes'],minimum_remaining_subscription_margin_ns=min(residual),remaining_refill_margin_ns=r['producer_first_refill_slack_ns']-512,later_output_changes=len(r['output_completion_changes'])))
obs=b['reused_R10_first_output_observations']
assert [(r['pitch_bytes'],r['prefill_plus_L_ns_delta']['median']) for r in obs]==[(0,816),(32,288)]
assert all(r['prefill_plus_L_ns_delta']['process_min']>=0 for r in obs)
assert all(r['bounds']['issue_to_ready_ns']['process_lower']>0 for r in b['related_R13_issue_to_ready_bounds'])
result=dict(source_report_sha256=sha(other/'B-startup-absorption.json'),rerun_report_sha256=sha(out/'B-startup-absorption.json'),common_input_hashes=len(common),path_only_module_differences=len(old_only),simulated_runs=b['simulated_runs'],simulated_logical_demands=b['simulated_logical_demands'],zero_shift_windows_verified=2592,checks=checks,new_numeric_replay=False,observation_qualified=False)
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
