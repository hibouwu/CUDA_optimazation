#!/usr/bin/env python3
"""Parameter-free causal-history baselines, not a new accepted clock model.

Event durations are returned to the host before the next recorded invocation.
Role cycle/rate checkpoints were read only after the whole recorded batch;
their component forecasts are offline causal-index diagnostics, not an online
state input demonstrated by this protocol. No target W/f fits coefficients.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st


METHODS=('hold_prefix','previous_call')


def causal_event_forecasts(prefix, targets, method):
    """Predict before appending each target observation; no future access."""
    if method not in METHODS or not prefix:
        raise ValueError('known baseline and nonempty prefix required')
    if any(not math.isfinite(v) or v<=0 for v in prefix+targets):
        raise ValueError('finite positive event durations required')
    checkpoint=prefix[-1];history=prefix[:];predictions=[]
    for actual in targets:
        predictions.append(checkpoint if method=='hold_prefix' else history[-1])
        history.append(actual)
    return predictions


def role_forecast(previous, actual):
    """Matched (cycles,ns); exact multiplicative cycle/conversion closure."""
    pc,pn=previous;tc,tn=actual
    if min(pc,pn,tc,tn)<=0:
        raise ValueError('positive native windows required')
    cycle=pc/tc-1
    conditional=(tc/tn)/(pc/pn)-1
    free=pn/tn-1
    closure=(1+cycle)*(1+conditional)-(1+free)
    if abs(closure)>1e-12:
        raise ValueError('cycle/conversion forecast identity did not close')
    return dict(cycle_forecast_error=cycle,conditional_conversion_error=conditional,
                free_role_window_error=free,closure=closure)


def stats(values):
    values=list(values)
    return dict(n=len(values),median=st.median(values),median_absolute=st.median(abs(v) for v in values),
                minimum=min(values),maximum=max(values),maximum_absolute=max(abs(v) for v in values),
                rms=math.sqrt(st.fmean(v*v for v in values)))


def summarize(rows,fields):
    return {key:stats(r[key] for r in rows) for key in fields if all(key in r for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    archive=base/'20261010-R09-history-series/reanalysis/A-time-contract-20261011-v3'
    sources=[archive/'series-replay.json',archive/'mirror-replay.json']
    old_series=base/'20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-history-series.json'
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources,old_series]}
    availability_sources=[]
    for run in ('20261010-R09-history-series','20261010-R09-mirror-history'):
        source=base/run/'source/probes/r18.cu';manifest=base/run/'source_hashes.json'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if json.loads(manifest.read_text())['source/probes/r18.cu']!=digest:
            raise ValueError('recorded runtime source identity changed')
        text=source.read_text()
        event=text.index('cudaEventElapsedTime(&ms, call_begin, call_end)')
        returned=text.index('return ms;',event)
        copied=text.index('cudaMemcpy(saved_history.data()')
        if not event<returned<copied:
            raise ValueError('event/component state availability needs new review')
        provenance[str(source)]=digest
        provenance[str(manifest)]=hashlib.sha256(manifest.read_bytes()).hexdigest()
        availability_sources.append(dict(path=str(source),sha256=digest,
            event_return_line=text[:returned].count('\n')+1,
            batch_device_to_host_line=text[:copied].count('\n')+1))
    # Single controls have no role prefix trace. Their already-replayed archive
    # retains the sixteen host event values that the compact replay omits.
    archived=json.loads(old_series.read_text())
    single={(p['case'],p['trial']):p for p in archived['processes'] if p['mode']=='single'}
    processes=[]
    for source in sources:
        data=json.loads(source.read_text())
        for p in data['processes']:
            p=dict(p,dataset='series' if source.name.startswith('series') else 'mirror')
            if p['mode']=='single':
                old=single[p['case'],p['trial']]
                if old['raw_sha256']!=p['raw_sha256'] or old['target']['event_us']!=p['target']['event_us']:
                    raise ValueError('single-control archive identity mismatch')
                p['prefix_event_us_list']=old['prefix_call_event_us']
            processes.append(p)
    if len(processes)!=80 or sum(p['mode']=='series' for p in processes)!=60:
        raise ValueError('all eighty history/control processes required')
    rows=[];component_windows=0;closure_max=0
    fields=['case','trial','dataset','source','slot','swizzle','method','target_index','cta','role','tiles',
            'cycle_forecast_error','conditional_conversion_error','free_role_window_error','closure']
    with gzip.open(out/'role-forecasts.csv.gz','wt') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for p in processes:
            if p['mode']=='series':
                calls=p['calls'];prefix=[c['event_us'] for c in calls[:16]];targets=[c['event_us'] for c in calls[16:]]
            else:
                prefix=p['prefix_event_us_list'];targets=[p['target']['event_us']]
            if len(prefix)!=16 or len(targets)!=(19 if p['mode']=='series' else 1):
                raise ValueError('recorded prefix/target count changed')
            for method in METHODS:
                predictions=causal_event_forecasts(prefix,targets,method)
                for j,(prediction,actual) in enumerate(zip(predictions,targets)):
                    row={k:p[k] for k in ('case','trial','dataset','source','slot','swizzle','mode')}
                    row.update(method=method,target_index=j,event_prediction_us=prediction,event_actual_us=actual,
                               free_event_error=prediction/actual-1,event_checkpoint_runtime_available=True,
                               component_checkpoint_runtime_available=False)
                    if p['mode']=='series':
                        previous=calls[15 if method=='hold_prefix' else 15+j];target=calls[16+j]
                        row['same_static_swizzle']=previous['swizzle']==target['swizzle']
                        row['same_source_allocation']=not (p['source']=='mirror' and (method=='hold_prefix' or j==0))
                        old={(r[0],r[1]):r for r in previous['role_windows']}
                        now={(r[0],r[1]):r for r in target['role_windows']}
                        if set(old)!=set(now) or len(now)!=264:
                            raise ValueError('all 132 CTA/two-role pairs required')
                        metrics=[]
                        for key,r in now.items():
                            before=old[key]
                            if before[2]!=r[2]:raise ValueError('matched role work count changed')
                            metric=role_forecast((before[4],before[5]),(r[4],r[5]));metrics.append(metric)
                            writer.writerow(dict(row,cta=r[0],role=r[1],tiles=r[2],**metric))
                            component_windows+=1;closure_max=max(closure_max,abs(metric['closure']))
                        row['component_windows']=len(metrics)
                        for key in ('cycle_forecast_error','conditional_conversion_error','free_role_window_error'):
                            row[key+'_role_median']=st.median(m[key] for m in metrics)
                            row[key+'_role_maximum_absolute']=max(abs(m[key]) for m in metrics)
                    rows.append(row)
    if len(rows)!=2320 or component_windows!=601920:
        raise ValueError('complete two-baseline record count changed')
    fields=['free_event_error','cycle_forecast_error_role_median','conditional_conversion_error_role_median',
            'free_role_window_error_role_median']
    grouped=defaultdict(list)
    for r in rows:grouped[r['case'],r['method']].append(r)
    summaries=[];process_summaries=[]
    for (case,method),group in grouped.items():
        segments={'all_targets':group,'first_target':[r for r in group if r['target_index']==0]}
        if any(r['target_index']>0 for r in group):
            segments['after_first_target']=[r for r in group if r['target_index']>0]
        summaries.append(dict(case=case,method=method,segments={name:summarize(values,fields) for name,values in segments.items()}))
        for trial in range(10):
            values=[r for r in group if r['trial']==trial]
            process_summaries.append(dict(case=case,method=method,trial=trial,metrics=summarize(values,fields)))
    result=dict(schema='gh200-r09-history-forecast/1',
        scope='Development forecast baselines; no parameter fit, new holdout, clock-rule adoption, or GPU work.',
        input_sha256=provenance,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        methods={'hold_prefix':'Hold last prefix event/role state for all targets. Entire trajectory forecast uses no target observation.',
                 'previous_call':'One-step forecast uses only the preceding completed invocation, never the current/future target.'},
        availability={'event':'Each CUDA event result is returned by timed() before the next invocation; no new device read is required.',
                      'components':'Role cycle/rate values are saved by D2D and read after the whole batch. These forecasts are offline causal-index diagnostics; an online checkpoint protocol has not been implemented/validated.'},
        availability_source_evidence=availability_sources,
        verification=dict(processes=len(processes),target_event_forecasts=len(rows),matched_role_forecasts=component_windows,
                          maximum_multiplicative_closure=closure_max),
        case_summaries=summaries,process_summaries=process_summaries,calls=rows,
        limits=['All switch, first, continuation, same/mirror and single/series conditions remain. Subsets are predefined by protocol, not error.',
            'Event forecasts predict complete recorded-ends event duration, not plain or a decomposed clock. No service/clock identification follows from a low total error.',
            'Conditional conversion error holds target observed cycles fixed. Cycle forecast error holds target conversion fixed; their multiplicative product closes the same-role free-window error.',
            'Per-role cycle residual contains native measured execution/endpoint effects; it is not identified as a physical service-price error.',
            'Role component medians do not multiply into a median total, and role windows do not algebraically close the full CUDA event.',
            'Q/S definition and all calibration coefficients remain unchanged; the whole cfg_a/b/c goal and independent validation requirement remain.'])
    (out/'history-forecast.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (out/'call-forecasts.csv').open('w') as stream:
        keys=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps(result['verification'],indent=2))


if __name__=='__main__':main()
