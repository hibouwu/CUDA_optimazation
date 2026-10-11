from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,statistics,re
out=Path(__file__).resolve().parent;run=out.parents[1];repo=run.parents[3]
inputs={}
def sha(p):inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return inputs[str(p)]
m=json.loads((run/'manifest.json').read_text());build=Path(m['build_run']);sources=json.loads((build/'source_hashes.json').read_text())
for name in ('source/probes/r18.cu','source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp','source/cutlass/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl'):
 assert sha(build/name)==sources[name]
sass=build/'returned/build/cfg_c_s4_dual.sass';sha(sass)
shapes=set(re.findall(r'HGMMA\.([A-Za-z0-9.]+)',sass.read_text()))
assert shapes=={'64x128x16.F32'}
sha(repo/'Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R00-anchor-target.md')
work=2*256*128*4096;instructions=(256//64)*(128//128)*(4096//16)
assert instructions==1024 and instructions*(2*64*128*16)==work
reference=F(work,4096);assert reference==65536
rows=[]
for record in json.loads((run/'returned/records.json').read_text()):
 raw=run/'returned'/record['raw'];assert sha(raw)==record['sha256']
 with gzip.open(raw,'rt') as f:events={v['event']:v for v in map(json.loads,f)}
 setup=events['setup'];t=events['call']['trace']
 assert setup['tile']==[256,128,64] and setup['k']==4096 and setup['stages']==4
 for cta in range(132):
  h=cta*1296;assert t[h+11]==t[h+12]==t[h+2]
  assert t[h+6]==t[h+9]
  for j in range(t[h+6]):
   a=h+16+j*10;b=a+640;assert t[a+4:a+6]==t[b+4:b+6]
   span=max(t[a+1],t[b+1])-min(t[a],t[b])
   assert span>0
   error=(reference-span)/span
   rows.append(dict(pair=record['pair'],arm=record['arm'],cta=cta,ordinal=j,work=[v-1 for v in t[a+4:a+6]],first=[t[a],t[b]],main=[t[a+1],t[b+1]],span_cycles=span,reference_cycles=int(reference),excess_cycles=int(span-reference),reference_relative_error=str(error)))
assert len(rows)==4320
values=[r['span_cycles'] for r in rows];errors=[abs(F(r['reference_relative_error'])) for r in rows]
with gzip.open(out/'windows.json.gz','wt') as f:json.dump(rows,f)
report=dict(input_sha256=inputs,target_work_flop=work,aggregate_m64n128k16_instructions=1024,instructions_per_wg=512,conditional_capacity_flop_per_SM_cycle=4096,conditional_reference_cycles=65536,windows=4320,span_min_median_max=[min(values),statistics.median(values),max(values)],reference_absolute_relative_error_median=float(statistics.median(errors)),reference_absolute_relative_error_max=float(max(errors)),below_reference=sum(v<reference for v in values),shape_in_sass=sorted(shapes),observation_qualified=False,prediction_qualified=False,limits=['Reference selected in development after cycle diagnostics; no independent held-out performance claim.', '4096 is a nominal conditional capacity reference, not a same-device calibrated price or a proved universal latency bound.', 'Historical R00-B N128/2WG uses K-major B and another GPU; later MN check uses N256. Conditions are not silently transferred.', 'Envelope is same-CTA min WG FIRST to max WG MAIN; includes all MMA plus intervening software/waits, not bare instruction service.', 'Does not predict producer prefill, W/E/GAP/TAIL, critical CTA, D visibility or whole CUDA event.', 'No cycle-to-ns conversion or physical frequency inference.'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}))
