#!/usr/bin/env python3
"""Independently check saved affine proofs and bind all stamps to raw packets."""
from pathlib import Path
from fractions import Fraction
import gzip, hashlib, json
out=Path(__file__).resolve().parent
run=out.parents[1]
a=run/'reanalysis/A-guarded-robustness-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with gzip.open(a/'robustness.json.gz','rt') as f:r=json.load(f)
for name,digest in r['input_sha256'].items():assert sha(Path(name))==digest
raw=run/'returned/raw/guarded_bracket.jsonl.gz'
with gzip.open(raw,'rt') as f:events={p['event']:p for p in map(json.loads,f)}
trace=events['call']['trace'];stamps={}
for cta in range(132):
 b=trace[cta*1552:(cta+1)*1552];sm=b[2]-1
 assert sm not in stamps and b[11]==b[12]==b[2]
 points=[]
 for role in (1,2):
  t=b[16+(role-1)*768:16+(role-1)*768+12]
  points.extend([[t[0],t[6],t[10]],[t[1],t[7],t[11]]])
 stamps[str(sm)]=points
assert stamps==r['original_stamps']
certificates=0; edges_checked=0
for case in r['cases']:
 lower=Fraction(0)
 for step in case['separating_cycles']:
  g=step['graph'];c=step['certificate'];ids=c['negative_cycle_edge_ids']
  edges=[g['edges'][i] for i in ids]
  assert all(edges[i][1]==edges[(i+1)%len(edges)][0] for i in range(len(edges)))
  bound=sum(e[2] for e in edges);assert bound<0
  relaxed=set(g['cycle_relaxed_edge_ids']);count=sum(i in relaxed for i in ids)
  k=Fraction(*g['extra_clock_cycles_fraction']);assert k==lower
  lower=k-Fraction(bound,g['cycle_scale']*count)
  assert lower==Fraction(*step['necessary_clock_cycles_fraction'])
  certificates+=1
 g=case['feasible_graph'];p=case['feasible_certificate']['potential_scaled_cycles'];scale=g['cycle_scale']
 assert Fraction(*case['minimum_extra_clock_cycles_fraction'])==lower
 assert all(p[v]-p[u]<=b for u,v,b in g['edges'])
 edges_checked+=len(g['edges']);certificates+=1
 # Direct original interval intersection, independent of graph builder/inverter.
 times=g['time_nodes_ns'];pos={t:i for i,t in enumerate(times)}
 eps=case['endpoint_slack_ns'];origin=g['timer_origin_ns']
 for sm,points in stamps.items():
  offset=p[g['sm_nodes'][sm]];co=g['clock_origins'][sm]
  for before,n,after in points:
   lo=(before-co-lower)*scale+offset;hi=(after-co+lower)*scale+offset
   assert max(lo,p[pos[n-origin-eps]])<=min(hi,p[pos[n-origin+eps]])
 # Monotonicity and rate cap are also checked directly.
 cap=None if case['cap'] is None else Fraction(str(case['cap']))
 for i in range(len(times)-1):
  dc=Fraction(p[i+1]-p[i],scale);assert dc>=0
  assert cap is None or dc<=cap*(times[i+1]-times[i])
inner=stamps['39'];outer=stamps['115']
begin_margin=inner[0][1]-outer[0][1];end_margin=outer[3][1]-inner[3][1]
inside=inner[3][0]-inner[0][2];outside=outer[3][2]-outer[0][0]
assert (begin_margin,end_margin,inside,outside)==(96,224,66787,66367)
assert min(begin_margin,end_margin)>=2*32 and Fraction(inside-outside,4)==105
assert all(c['minimum_extra_clock_cycles_fraction']==[105,1] for c in r['cases'] if c['endpoint_slack_ns']==32)
result=dict(raw_sha256=sha(raw),report_sha256=sha(a/'robustness.json.gz'),input_hashes_verified=len(r['input_sha256']),raw_stamps_verified=528,cases_verified=20,certificates_verified=certificates,feasible_edges_verified=edges_checked,original_interval_intersections=10560,solver_called=False,graph_builder_called=False,four_point_witness=dict(inner_sm=39,outer_sm=115,begin_margin_ns=begin_margin,end_margin_ns=end_margin,inner_min_cycles=inside,outer_max_cycles=outside,epsilon_ns=32,minimum_kappa_cycles=105),observation_qualified=False)
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
