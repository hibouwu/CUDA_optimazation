"""Inverse matrix tiling and element-address checks, independent of model helpers."""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,hashlib,json
ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'B-boundary-binding.json').read_text())
M,N,K=2600,3000,2000
expected={}
for n in range(24):
 for m in range(21):
  linear=n*21+m;cta=linear%132;ordinal=linear//132
  expected[cta,ordinal]=(m,n)
counts=Counter(cta for cta,j in expected)
assert counts==Counter({cta:4 if cta<108 else 3 for cta in range(132)})
# Explicit valid FP16 elements, not source_granules or source-box row-span helpers.
A=[]
for q in range(32):
 A.append({(2*(row*K+inner))//128 for row in range(M) for inner in range(64*q,min(K,64*(q+1)))})
assert all(len(A[q]&A[q+1])==1950 for q in range(31))
assert len(A[-1]&A[0])==1950
results=[]
for case in report['cases']:
 data=json.loads(gzip.decompress((ROOT/case['ledger_file']).read_bytes()))
 seen={};classes=Counter();columns_by_ordinal=defaultdict(set)
 for output in data['outputs']:
  cta,j=output['cta'],output['output_ordinal'];coord=tuple(output['coord'])
  assert coord==expected[cta,j] and (cta,j) not in seen
  seen[cta,j]=coord
  assert output['consumer_warp_group']==j%2
  classes[j,coord[0]==20,coord[1]==23]+=1
  columns_by_ordinal[j].update(range(128*coord[1],min(N,128*(coord[1]+1))))
  for q,group in enumerate(output['groups']):
   global_group=32*j+q;state=group['slot_state']
   generation,slot=divmod(global_group,6)
   assert state==dict(group=global_group,slot=slot,generation=generation,
    full_wait_phase=generation%2,empty_wait_phase=1-generation%2,
    previous_group=global_group-6 if generation else None)
 assert seen==expected
 assert [dict(output_ordinal=j,M_tail=m,N_tail=n,count=v) for (j,m,n),v in sorted(classes.items())]==case['output_classes']
 ldb=int(case['case'].split('ldb')[1]);B={}
 for j,columns in columns_by_ordinal.items():
  for q in range(32):
   B[j*32+q]={(2*(inner*ldb+col))//128 for inner in range(64*q,min(K,64*(q+1))) for col in columns}
 for row in case['global_K_cohort_coverage']:
  g=row['global_group'];current=A[g%32] if row['operand']=='A' else B[g]
  assert row['units128']==len(current)
  if g<127:
   following=A[(g+1)%32] if row['operand']=='A' else B[g+1]
   assert row['next_group_intersection128']==len(current&following)
  else:assert row['next_group_intersection128'] is None
 for window in case['trace_windows']:
  cta=window['cta'];assert window['derived_output_ordinal']==2
  assert window['derived_coord']==list(expected[cta,2])==[cta+12,12]
  assert window['source_valid_bytes']==2*128*K*2
  assert window['destination_fill_bytes']==32*32768-2*128*K*2
  assert window['first_ring_state']['group']==64 and window['first_ring_state']['slot']==4
  assert window['last_ring_state']['group']==95 and window['last_ring_state']['slot']==5
 results.append(dict(case=case['case'],inverse_output_bindings=504,ring_groups=504*32,
  coverage_cohorts=128*2,consecutive_coverage_intersections=127*2,
  sampled_windows_MN_interior_K_tail=True,A_last_to_first_K_overlap_units128=1950))
(ROOT/'independent-binding-verification.json').write_text(json.dumps(dict(cases=results,
 assumed_source_contract='Pinned archived SS pingpong, swizzle1 AlongM, 132 CTA and stage6; source derivation is checked by main replay.',
 element_enumeration=True,no_schedule_or_granule_model_helpers=True,
 qualification=False,physical_inflight_sharing_proved=False,timing_support=False),indent=2)+'\n')
print(json.dumps(results,indent=2))
