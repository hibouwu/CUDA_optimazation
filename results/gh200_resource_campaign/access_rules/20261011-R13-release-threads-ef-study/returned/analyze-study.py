from pathlib import Path
import gzip,hashlib,json,math,statistics,sys
root=Path(__file__).resolve().parent;m=json.loads((root/'manifest.json').read_text());data=root/'returned'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in json.loads((root/'freeze.json').read_text())['files'].items():
 if sha(root/name)!=digest:raise ValueError('frozen input changed: '+name)
for name,digest in m['analysis_sha256'].items():
 if sha(root/'analysis_snapshot'/name)!=digest:raise ValueError('frozen analysis changed: '+name)
if sha(root/'manifest.json')!=json.loads((data/'collection-complete.json').read_text())['manifest_sha256']:raise ValueError('manifest changed')
if sha(root/'cases.json')!=m['cases_sha256']:raise ValueError('cases changed')
for f,field in [('source_hashes.json','source_hashes_sha256'),('build-audit.json','build_audit_sha256')]:
 if sha(root/f)!=m[field]:raise ValueError('compiled source evidence changed')
sys.path.insert(0,str(root/'analysis_snapshot'))
from analyze_r18 import replay
from analyze_r13_release_interpretation import release_thread_process,paired_observer_gates
rows={r['id']:r for r in json.loads((root/'cases.json').read_text())};records=json.loads((data/'records.json').read_text())
if [r['id'] for r in records]!=[s['id'] for s in m['samples']][:len(records)]:raise ValueError('execution order differs from freeze')
expected={s['id']:s for s in m['samples']};observed=set();processes=[];refs=[];errors=[]
for rec in records:
 if rec['id'] not in expected or rec['id'] in observed:raise ValueError('unexpected or duplicate sample')
 observed.add(rec['id'])
 if any(rec[k]!=v for k,v in expected[rec['id']].items()):raise ValueError('sample schedule identity changed')
 p=dict(stages=rec['stage'],observer=rec['flag'],trial=rec['trial'],id=rec['id'],raw=rec['raw'])
 try:
  if rec['returncode']:raise ValueError('nonzero process exit')
  result=replay(data,rec,rows[rec['case']]);ev={e['event']:e for e in map(json.loads,gzip.open(data/rec['raw'],'rt'))};setup=ev['setup']
  if setup['gpu_uuid']!=m['gpu_uuid'] or setup['stages']!=rec['stage'] or setup['grid']!=[1,132,1] or setup['resolved_raster_order']!='along_n':raise ValueError('GPU/stage/grid/raster changed')
  if not math.isfinite(result['elapsed_us']) or result['elapsed_us']<=0:raise ValueError('invalid event time')
  p.update(gpu_uuid=setup['gpu_uuid'],elapsed_us=result['elapsed_us'],checked_values=result['checked_values'])
  if rec['kind']=='study':
   if setup['trace_version']!=m['protocol']:raise ValueError('trace protocol changed')
   s=result['sparse'];p.update(sm=s['sm'],work=s['work'],slot=s['slot'],phase=s['phase'],sparse=s,diagnostic=release_thread_process(s))
  else:p['position']=rec['position']
 except (ValueError,KeyError) as err:
  p['decode_error']=str(err);errors.append(dict(id=rec['id'],error=str(err)))
 (processes if rec['kind']=='study' else refs).append(p)
missing=sorted(expected.keys()-observed);gates=paired_observer_gates(processes);reference_gates=[]
for stage in (4,6):
 rs=[r for r in refs if r['stages']==stage and 'decode_error' not in r];times=[r['elapsed_us'] for r in rs];cv=statistics.pstdev(times)/statistics.fmean(times) if times else None
 pre=[r['elapsed_us'] for r in rs if r['position']=='pre'];post=[r['elapsed_us'] for r in rs if r['position']=='post'];off=[p['elapsed_us'] for p in processes if p['stages']==stage and p['observer']==0 and 'decode_error' not in p]
 reference_gates.append(dict(stage=stage,n=len(rs),CV=cv,passed=len(rs)==10 and cv<=m['reference_cv_limit'],pre_post_median_ratio=statistics.median(post)/statistics.median(pre)-1 if pre and post else None,off_plain_median_ratio=statistics.median(off)/statistics.median(times)-1 if off and times else None))
checks=json.loads((data/'full-D-checks.json').read_text());numeric=len(checks)==4 and all(c['elements']==4325376 for c in checks) and len({c['D_sha256'] for c in checks})==1
qualified=not errors and not missing and numeric and all(g['gate_passed'] for g in gates) and all(r['passed'] for r in reference_gates)
result=dict(manifest_sha256=sha(root/'manifest.json'),processes=processes,references=refs,errors=errors,missing=missing,full_D_passed=numeric,paired_observer_gates=gates,reference_gates=reference_gates,observation_qualified=qualified,limits=['Development observation only; no independent holdout or service price.','Post-last interval includes visibility, polling, scheduling, expect_tx and clock path; not bare latency.','All processes retained; conditional classification is not adopted when any gate fails.'])
(root/'study-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(observation_qualified=qualified,errors=errors,missing=missing,paired_observer_gates=gates,reference_gates=reference_gates),indent=2))
