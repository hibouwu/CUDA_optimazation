"""Replay the frozen fixed-device integration diagnostic; never qualifies timing."""
from pathlib import Path
import gzip,hashlib,json,statistics,sys
import numpy as np
r=Path(__file__).resolve().parent;d=r/'returned'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'manifest.json').read_text());build=Path(m['build_run'])
assert sha(build/'build-audit.json')==m['build_audit_sha256']
assert sha(build/'returned/build/cfg_c_s4_dual')==sha(d/'build/cfg_c_s4_dual')==m['binary_sha256']
receipt=json.loads((r/'receipt.json').read_text())
assert receipt['returncode']==0 and sha(r/'output.tar.gz')==receipt['sha256']
for name,h in json.loads((r/'freeze.json').read_text())['files'].items():assert sha(r/name)==sha(d/name)==h,name
for name,h in m['analysis_sha256'].items():assert sha(r/'analysis_snapshot'/name)==h,name
sys.path.insert(0,str(r/'analysis_snapshot'))
sys.path.insert(0,str(r))
from analyze_r18 import reference
from decode_cfg_c_gt_bracket import decode
from validate_allocation import validate
validate((d/'allocation.txt').read_text())
context=json.loads((d/'assigned-context.json').read_text());assert context['CUDA_VISIBLE_DEVICES']
records=json.loads((d/'records.json').read_text())
assert [{k:row[k] for k in ('id','flag','command')} for row in records]==m['samples']
assert json.loads((d/'complete.json').read_text())==dict(manifest_sha256=sha(r/'manifest.json'),records=3)
expected=np.array([[reference(i,j,4096,17) for j in range(17)] for i in range(17)],dtype='<f4')
data=np.fromfile(d/'full-d.bin',dtype='<f4').reshape(2304,3072)
full_ok=bool(np.all(np.isfinite(data)) and np.array_equal(data,expected[np.arange(2304)[:,None]%17,np.arange(3072)[None,:]%17]))
results=[];widths=[];bounds=[]
for row in records:
 raw=d/row['raw'];assert row['returncode']==0 and sha(raw)==row['sha256']
 with gzip.open(raw,'rt') as stream:packets=[json.loads(line) for line in stream]
 events={p['event']:p for p in packets};assert len(events)==len(packets)
 setup=events['setup'];assert setup['gpu_uuid']==m['gpu_uuid']
 assert setup['gt_bracket_protocol']=='cfg-c-gt-bracket-v1' and setup['gt_bracket_observer']==row['flag']
 if row['id']=='full_D':
  c=events['gt_bracket_check'];ok=c['elements']==data.size and c['comparison_flag']==1 and c['different']==c['nonfinite']==0
  results.append(dict(id=row['id'],raw_sha256=sha(raw),elements=int(data.size),off_on_bit_equal=ok,exact_reference=full_ok));continue
 c=events['check'];numeric=(c['status']=='ok' and c['samples']==4096 and len(set(c['checked_indices']))==len(c['checked_values'])==4096 and not c['nonfinite'] and not c['padding_errors'])
 numeric=bool(numeric and all(type(i) is int and 0<=i<2304*3072 and v==reference(i//3072,i%3072,4096,17) for i,v in zip(c['checked_indices'],c['checked_values'])))
 decoded=decode(setup,events['call'],m['gpu_uuid'])
 if row['flag']:
  for cta in decoded['records']:
   for role in cta['roles']:
    for ordinal,tile in enumerate(role['tiles']):
     identity=dict(cta=cta['cta'],sm=cta['sm'],thread=role['physical_thread'],output_ordinal=ordinal,work=tile['work'])
     bounds.append(dict(identity,conditional_elapsed_gt=tile['conditional_elapsed_gt'],post_gt_elapsed=tile['post_gt_elapsed'],post_gt_error_bound=tile['post_gt_error_bound']))
     for event,stamp in zip(['FIRST_MMA','MAIN_END'],tile['gt_brackets']):
      widths.append(dict(identity,event=event,width=stamp['gt_after']-stamp['gt_before'],**stamp))
 results.append(dict(id=row['id'],raw_sha256=sha(raw),record_valid=decoded['record_valid'],decoded=decoded,numeric_ok=numeric,
  sampled_values=4096,elapsed_us=events['call']['elapsed_us']))
values=[v['width'] for v in widths]
summary=dict(stamps=len(values),minimum=min(values) if values else None,median=statistics.median(values) if values else None,
 maximum=max(values) if values else None,greater_than_64=sum(x>64 for x in values),greater_than_256=sum(x>256 for x in values))
report=dict(gpu_uuid=m['gpu_uuid'],allocation=context,full_D_sha256=sha(d/'full-d.bin'),results=results,
 bracket_widths=summary,conditional_windows=bounds,full_D_reference_ok=full_ok,
 hardware_sampling_enclosure=False,physical_ns_accuracy=False,observation_qualified=False,
 limits='One full-D comparison and one warm call per arm. Widths only describe new read records under GT-domain assumptions. No perturbation/null distribution, model score or physical time calibration.')
(r/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
with gzip.GzipFile(filename=str(r/'stamps.json.gz'),mode='wb',mtime=0) as stream:stream.write(json.dumps(widths,sort_keys=True).encode())
print(json.dumps(dict(full_D_elements=int(data.size),full_D_reference_ok=full_ok,results=[{k:v for k,v in x.items() if k!='decoded'} for x in results],bracket_widths=summary),indent=2))
assert full_ok and results[0]['off_on_bit_equal']
assert all(x['record_valid'] and x['numeric_ok'] for x in results[1:])
