"""Replay frozen complete point controls; keep read-interval diagnostics separate."""
from pathlib import Path
import gzip,hashlib,json,sys,statistics
from collections import Counter
root=Path(__file__).resolve().parent;data=root/'returned'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'manifest.json').read_text())
assert manifest['status']=='frozen'
for name,digest in json.loads((root/'freeze.json').read_text())['files'].items():
 assert sha(root/name)==sha(data/name)==digest,name
assert sha(Path(manifest['correctness_record']))==manifest['correctness_sha256']
correct=json.loads(Path(manifest['correctness_record']).read_text())
assert correct['full_D_reference_ok'] and correct['results'][0]['off_on_bit_equal']
build=Path(manifest['build_run'])
assert sha(build/'build-audit.json')==manifest['build_audit_sha256']
assert sha(build/'returned/build/cfg_c_s4_dual')==sha(data/'build/cfg_c_s4_dual')==manifest['binary_sha256']
receipt=json.loads((root/'receipt.json').read_text());assert receipt['returncode']==0 and sha(root/'output.tar.gz')==receipt['sha256']
assert json.loads((data/'complete.json').read_text())==dict(manifest_sha256=sha(root/'manifest.json'),records=40)
records=json.loads((data/'records.json').read_text())
assert [{key:record[key] for key in ('id','comparison','pair','arm','flag','command')} for record in records]==manifest['samples']
sys.path.insert(0,str(root/'analysis_snapshot'))
sys.path.insert(0,str(root))
from analyze_cfg_c_gt_control import process,evaluate
from validate_allocation import validate
validate((data/'allocation.txt').read_text())
context=json.loads((data/'assigned-context.json').read_text());assert context['CUDA_VISIBLE_DEVICES']
cohorts={name:[] for name in manifest['comparisons']};errors=[]
for record in records:
 path=data/record['raw'];assert record['returncode']==0 and sha(path)==record['sha256']
 with gzip.open(path,'rt') as stream:events=[json.loads(line) for line in stream]
 try:cohorts[record['comparison']].append(process(events,record,manifest['gpu_uuid']))
 except (ValueError,KeyError) as error:errors.append(dict(id=record['id'],error=str(error)))
results={};summaries={}
for name,rows in cohorts.items():
 if len(rows)!=20:continue
 sub=dict(manifest,comparison=name,samples=[s for s in manifest['samples'] if s['comparison']==name])
 result=evaluate(rows,sub);results[name]=result
 by_metric={}
 for metric in sorted({x['series'].split('/')[-1] for x in result['local']}):
  group=[x for x in result['local'] if x['series'].split('/')[-1]==metric]
  by_metric[metric]=dict(series=len(group),failed=sum(not x['passed'] for x in group),
   median_absolute_range=[min(x['median_absolute'] for x in group),max(x['median_absolute'] for x in group)],
   maximum_absolute_range=[min(x['maximum_absolute'] for x in group),max(x['maximum_absolute'] for x in group)])
 widths=[width for row in result['enabled_read_diagnostics'] for pair in row['read_widths'] for width in pair]
 summaries[name]=dict(processes_verified=len(rows),numeric_values_verified=4096*len(rows),series_per_process=len(rows[0]['series']),
  failed_point_series=sum(not x['passed'] for x in result['local']),by_metric=by_metric,
  event=result['event'],off_event_population_cv=result['off_event_population_cv'],
  necessary_control_passed=result.get('incremental_control_passed',result.get('baseline_consistency_passed')),
  sm_changed_series=sum(any(x['sm_changed']) for x in result['local']),
  enabled_read_stamps=len(widths),read_width_histogram=dict(sorted(Counter(widths).items())),
  observation_qualified=False)
output=dict(manifest_sha256=sha(root/'manifest.json'),allocation=context,record_errors=errors,cohorts=results,observation_qualified=False)
(root/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
summary=dict(manifest_sha256=sha(root/'manifest.json'),record_errors=errors,cohorts=summaries,observation_qualified=False)
(root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
if errors:raise SystemExit(2)
