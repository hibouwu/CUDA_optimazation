from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,statistics
out=Path(__file__).resolve().parent;run=out.parents[1]
inputs={};packets={}
def sha(p):inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return inputs[str(p)]
manifest=json.loads((run/'manifest.json').read_text());sha(run/'manifest.json')
build=Path(manifest['build_run']);source=build/'source/probes/r18.cu';source_hashes=json.loads((build/'source_hashes.json').read_text());assert sha(source)==source_hashes['source/probes/r18.cu'];sha(build/'source_hashes.json')
text=source.read_text();start=text.index('    auto timed =');end=text.index('      return ms;',start);assert text.index('cudaEventSynchronize(call_end)',start)<text.index('cudaEventElapsedTime(&ms, call_begin, call_end)',start)<end
assert text.index('warm.push_back(timed(warmup_launch_params));',end)<text.index('        set_target_epilogue();',text.index('      if (common_history)'))
for row in json.loads((run/'returned/records.json').read_text()):
 raw=run/'returned'/row['raw'];assert sha(raw)==row['sha256']
 with gzip.open(raw,'rt') as f:events={v['event']:v for v in map(json.loads,f)}
 s,c=events['setup'],events['call'];assert row['flag']==s['producer_tail_observer']==c['producer_tail_observer']==0
 assert s['gpu_uuid']==manifest['gpu_uuid'] and c['warmup_calls']==16 and len(c['warmup_us'])==16
 packets[row['pair'],row['arm']]=c
rows=[];summary=[];event_errors=[]
for pair in range(10):
 a,b=packets[pair,0],packets[pair,1]
 p0,p1=F(str(a['warmup_us'][-1])),F(str(b['warmup_us'][-1]))
 # Equal-work prefix event is the declared scalar state proxy, before target.
 predicted_rate_ratio=p0/p1
 prefix_changes=[F(str(y))/F(str(x))-1 for x,y in zip(a['warmup_us'],b['warmup_us'])]
 group=[]
 for cta in range(132):
  h=cta*1296;assert a['trace'][h+2]==b['trace'][h+2]
  for wg in (0,1):
   count=a['trace'][h+6+3*wg];assert count==b['trace'][h+6+3*wg]
   for j in range(count):
    k=h+16+wg*640+j*10;x=a['trace'][k:k+10];y=b['trace'][k:k+10];assert x[4:6]==y[4:6]
    c0,c1=x[1]-x[0],y[1]-y[0];n0,n1=x[7]-x[6],y[7]-y[6]
    observed=F(c1*n0,c0*n1)
    # This is deliberately conditional: both target cycle counts are used.
    predicted_ns=F(n0*c1,c0)/predicted_rate_ratio
    residual=predicted_ns-n1
    assert predicted_ns/F(n1)==observed/predicted_rate_ratio
    row=dict(pair=pair,cta=cta,wg=wg,ordinal=j,observed_rate_ratio=str(observed),prefix_rate_ratio=str(predicted_rate_ratio),conditional_ns_residual=str(residual),relative_residual=str(residual/n1))
    group.append(row);rows.append(row)
 for arm,call in ((0,a),(1,b)):
  error=F(str(call['warmup_us'][-1]))/F(str(call['elapsed_us']))-1
  event_errors.append(dict(pair=pair,arm=arm,last_prefix_event_us=call['warmup_us'][-1],target_event_us=call['elapsed_us'],relative_error=str(error)))
 summary.append(dict(pair=pair,prefix_event_changes=[float(x) for x in prefix_changes],last_prefix_rate_ratio=float(predicted_rate_ratio),groups=[dict(wg=wg,ordinal=j,windows=len(v),median_relative_residual=float(statistics.median(F(r['relative_residual']) for r in v)),median_absolute_ns_residual=float(statistics.median(abs(F(r['conditional_ns_residual'])) for r in v))) for wg in (0,1) for j in (0,1) if (v:=[r for r in group if r['wg']==wg and r['ordinal']==j])]))
assert len(rows)==4320
series={}
for r in rows:series.setdefault((r['cta'],r['wg'],r['ordinal']),[]).append(abs(F(r['conditional_ns_residual'])))
assert len(series)==432 and all(len(v)==10 for v in series.values())
failed=sum(statistics.median(v)>64 or max(v)>256 for v in series.values())
with gzip.open(out/'windows.json.gz','wt') as f:json.dump(rows,f)
errors=[abs(F(r['relative_error'])) for r in event_errors]
result=dict(input_sha256=inputs,window_pairs=4320,series=432,conditional_residual_series_above_original_64_256_scale=failed,event_last_prefix_errors=event_errors,event_median_abs_error=float(statistics.median(errors)),event_max_abs_error=float(max(errors)),pairs=summary,observation_qualified=False,prediction_qualified=False,limits=['One preselected zero-parameter last-prefix scalar only; no fitting or alternative state search.', 'Local residual explicitly uses both target cycle counts and arm0 target ns: conditional diagnosis, not a free prediction.', 'Reverse arm execution order is retained; no reference availability claim for local conditional calculation.', '64/256 ns used only as diagnostic scale; no corrected acceptance or baseline subtraction.', 'Prefix event includes entire kernel execution; inverse event is a hypothesis proxy, not measured physical frequency.'])
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','pairs','event_last_prefix_errors')}))
