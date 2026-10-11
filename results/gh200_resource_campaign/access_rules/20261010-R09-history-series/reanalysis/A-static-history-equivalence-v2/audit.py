#!/usr/bin/env python3
"""Check frozen V09 r2 static equality, without running its time predictor."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def static_inputs(model, row, setup):
    """Values consumed by frozen r2, with coordinate labels retained separately."""
    cfg = row['config']
    tm, tn, tk = model.events.base.CONFIGS[cfg]['tile']
    cm, cn = model.events.base.CONFIGS[cfg]['cluster']
    work = model.events.scheduled_work(cfg, row['m'], row['n'], setup['grid'], row['swizzle'])
    requests = model.supply.supply_request_rows(row, setup)
    loops = [[] for _ in work]
    for request in requests:
        loops[request['cta']].append({key: request[key] for key in ('j', 'T', 'kt', 'compute_cycles', 'X')})
    fractions = [[Fraction(max(0, min(tm, row['m']-mi*tm))*max(0, min(tn, row['n']-ni*tn)), tm*tn)
                  for mi, ni in cta] for cta in work]
    tiles = sum(map(len, work))
    kt = row['k']//64
    signature = dict(config=cfg, schedule=model.events.base.CONFIGS[cfg]['schedule'],
        input_mode=row.get('input_mode', 'dyadic'), kt=kt,
        tensor_mean_cycles=str(Fraction(tiles*kt*2*tm*tn*tk, 4096*132)),
        source_mean_kib=str(Fraction(tiles*kt*(2*tm*tk//cn+2*tn*tk//cm), 1024*132)),
        zero_product_fraction=model.zero_product_fraction(row, work),
        first_output_bytes=str(sum(f[0]*tm*tn*4 for f in fractions if f)),
        single=max(map(len,work))==1,
        ctas=[dict(loops=loop, valid_fractions=[str(f) for f in valid])
              for loop, valid in zip(loops, fractions)])
    return signature, work, requests


def metrics(errors):
    return dict(n=len(errors), median=statistics.median(errors),
        median_absolute=statistics.median(abs(e) for e in errors),
        maximum_absolute=max(abs(e) for e in errors),
        rms=(statistics.fmean(e*e for e in errors))**.5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    base=args.base.resolve(); out=args.output.resolve()
    out.mkdir(exist_ok=True)
    if (out/'equivalence.json').exists():
        raise ValueError('existing report is immutable; use a new output directory')
    frozen=base/'20261009-V09-freeze-job738972-r2'
    expected={'frozen/predictions.json':'f4df39e00cf1d2ab61947c911e699bd5bea4e533609d93f726de43077515d81e',
              'frozen/manifest.json':'09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab'}
    provenance={}
    for name,digest in expected.items():
        path=frozen/name
        if sha(path)!=digest: raise ValueError('approved freeze identity changed')
        provenance[str(path)]=digest
    manifest=read(frozen/'source_hashes.json')
    approved_files=read(frozen/'frozen/manifest.json')['files']
    if sha(frozen/'source_hashes.json')!=approved_files['source_hashes.json']:
        raise ValueError('source manifest is not bound to approved freeze')
    if sha(frozen/'frozen/calibration.json')!=approved_files['frozen/calibration.json']:
        raise ValueError('calibration is not bound to approved manifest')
    sys.path.insert(0,str(frozen/'source'))
    import v09_model as model
    loaded=[]
    for module in list(sys.modules.values()):
        path=Path(getattr(module,'__file__','') or '/')
        if not path.is_relative_to(frozen/'source') or path.suffix!='.py': continue
        key=str(path.relative_to(frozen))
        if sha(path)!=manifest[key]: raise ValueError('frozen imported source changed')
        provenance[str(path)]=manifest[key]; loaded.append(key)
    provenance[str(frozen/'source_hashes.json')]=sha(frozen/'source_hashes.json')
    provenance[str(frozen/'frozen/calibration.json')]=sha(frozen/'frozen/calibration.json')
    calibration=read(frozen/'frozen/calibration.json')
    archive=base/'20261010-R09-history-series/reanalysis/A-time-contract-20261011-v3'
    old=base/'20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-history-series.json'
    provenance[str(old)]=sha(old)
    singles={(p['case'],p['trial']):p for p in read(old)['processes'] if p['mode']=='single'}
    signatures={}; coordinates={}; datasets=[]; supports=[]
    for run,filename in [('20261010-R09-history-series','series-replay.json'),
                         ('20261010-R09-mirror-history','mirror-replay.json')]:
        root=base/run
        rows={r['id']:r for r in read(root/'cases.json')}
        setups={s['case']:s['setup'] for s in read(root/'static_setup.json')}
        for path in (root/'cases.json',root/'static_setup.json',root/'environment.json',archive/filename):
            provenance[str(path)]=sha(path)
        data=read(archive/filename)
        for case,row in rows.items():
            setup=setups[case]
            if (row['config'],row['m'],row['n'],row['k'],setup['stages'],setup['tile'],setup['cluster'],setup['grid']) != (
                'cfg_b',6144,8192,32768,6,[128,128,64],[1,1],[132,1,1]):
                raise ValueError('declared history geometry changed')
            for swizzle in (1,row['swizzle']):
                key=(case,swizzle)
                if key in signatures: continue
                signature,work,requests=static_inputs(model,dict(row,swizzle=swizzle),setup)
                signatures[key]=signature;coordinates[key]=work
                if any(r['gpu_uuid']!=setup['gpu_uuid'] for r in requests):
                    raise ValueError('original device identity was not preserved')
            supports.append(dict(case=case, actual_gpu_uuid=setup['gpu_uuid'],
                frozen_gpu_uuid=calibration['supply']['cfg_b']['gpu_uuid'],
                card_supported=setup['gpu_uuid']==calibration['supply']['cfg_b']['gpu_uuid']))
        datasets.append((root,data,rows))
    unique={hashlib.sha256(packed(s)).hexdigest() for s in signatures.values()}
    if len(unique)!=1: raise ValueError('static equivalence rejected')
    reference=next(iter(signatures.values()))
    expected_counts=[len(c['loops']) for c in reference['ctas']]
    process_count=call_count=role_count=raw_count=0
    forecasts=[];groups=defaultdict(list);switch_ratios=[]
    for root,data,rows in datasets:
        for p in data['processes']:
            record_path=root/'samples'/p['case']/('ends-%02d.json'%p['trial'])
            record=read(record_path);raw=root/record['raw']
            if sha(raw)!=p['raw_sha256'] or record['raw_sha256']!=p['raw_sha256']:
                raise ValueError('replay/raw identity mismatch')
            provenance[str(record_path)]=sha(record_path);provenance[str(raw)]=p['raw_sha256'];raw_count+=1
            process_count+=1
            if p['mode']=='series':
                calls=p['calls']
                if len(calls)!=35: raise ValueError('complete prefix/target history required')
                for call in calls:
                    if call['swizzle'] not in (1,8): raise ValueError('unexpected swizzle')
                    roles=call['role_windows']
                    if len(roles)!=264 or len({(r[0],r[1]) for r in roles})!=264:
                        raise ValueError('incomplete role work')
                    for cta,role,tiles,*_ in roles:
                        if tiles!=(expected_counts[cta]+(role==0))//2:
                            raise ValueError('role workload does not match static schedule')
                        role_count+=1
                prefix=[c['event_us'] for c in calls[:16]]
                target=[c['event_us'] for c in calls[16:]]
                call_count+=35
            else:
                single=singles[p['case'],p['trial']]
                if single['raw_sha256']!=p['raw_sha256'] or single['target']['event_us']!=p['target']['event_us']:
                    raise ValueError('single-control prefix source mismatch')
                prefix=single['prefix_call_event_us'];target=[p['target']['event_us']];call_count+=1
            if len(prefix)!=16 or min(prefix+target)<=0: raise ValueError('invalid event history')
            previous=prefix[-1]
            for index,actual in enumerate(target):
                # This is an identity check. No time predictor is invoked and no
                # target duration enters the prediction before this row is scored.
                prediction=previous
                error=prediction/actual-1
                row=dict(case=p['case'],trial=p['trial'],mode=p['mode'],swizzle=p['swizzle'],
                    target_index=index,prediction_us=prediction,actual_us=actual,error=error,
                    static_normalization_ratio='1',signature_sha256=next(iter(unique)))
                forecasts.append(row);groups[p['case']].append(error)
                if index==0 and p['swizzle']==8: switch_ratios.append(Fraction(str(previous))/Fraction(str(actual)))
                previous=actual
    if (process_count,raw_count,call_count,role_count,len(forecasts))!=(80,80,2120,554400,1160):
        raise ValueError('complete history counts changed')
    previous_file=base/'20261010-R09-history-series/reanalysis/A-history-forecast-v1/call-forecasts.csv'
    provenance[str(previous_file)]=sha(previous_file)
    with previous_file.open() as stream:
        previous_rows={(r['case'],int(r['trial']),int(r['target_index'])):r
                       for r in csv.DictReader(stream) if r['method']=='previous_call'}
    if len(previous_rows)!=1160: raise ValueError('previous baseline archive incomplete')
    for row in forecasts:
        old_row=previous_rows[row['case'],row['trial'],row['target_index']]
        if any(row[new]!=float(old_row[old]) for new,old in (
                ('prediction_us','event_prediction_us'),('actual_us','event_actual_us'),('error','free_event_error'))):
            raise ValueError('exact previous-call forecast identity failed')
    # A global ratio fitted to these target observations is only an oracle lower
    # bound. It is not an available input, prediction, or accepted coefficient.
    lo,hi=min(switch_ratios),max(switch_ratios)
    ratio=2/(lo+hi);floor=(hi-lo)/(hi+lo)
    negative_checks=[]
    root,_,rows=datasets[0];row=next(iter(rows.values()))
    setup={s['case']:s['setup'] for s in read(root/'static_setup.json')}[row['id']]
    for field,value in [('ldb',row['ldb']+16),('k',row['k']//2)]:
        changed,_,_=static_inputs(model,dict(row,**{field:value}),setup)
        if changed==reference: raise ValueError('changed priced input was dropped')
        negative_checks.append(dict(field=field,value=value,distinguished=True))
    sw1=coordinates['r09_trajectory_series_sw1',1]
    sw8=coordinates['r09_trajectory_series_sw8',8]
    changed=sum(a!=b for ca,cb in zip(sw1,sw8) for a,b in zip(ca,cb))
    if changed==0: raise ValueError('coordinate intervention absent')
    result=dict(schema='gh200-r09-static-history-equivalence/1',
        scope='Frozen r2 static input equivalence and causal event identity; no cross-card time prediction.',
        verification=dict(processes=process_count,raw_hashes=raw_count,recorded_calls=call_count,
            role_work_counts=role_count,target_forecasts=len(forecasts),signature_count=len(signatures),
            unique_signatures=len(unique),previous_call_exact_matches=len(forecasts),changed_tile_coordinates=changed,
            tile_count_distribution=dict(Counter(expected_counts))),
        frozen_sources=sorted(loaded),input_sha256=provenance,analysis_sha256=sha(Path(__file__)),
        static_summary={k:v for k,v in reference.items() if k!='ctas'},
        request_X_distribution=[dict(X=list(x),n=n) for x,n in Counter(tuple(q['X']) for c in reference['ctas'] for q in c['loops']).items()],
        signature_sha256=next(iter(unique)),support=supports,negative_checks=negative_checks,
        event_metrics={case:metrics(errors) for case,errors in groups.items()},
        first_switch_oracle=dict(n=len(switch_ratios),relative_ratio_range=[str(lo),str(hi)],
            fitted_static_ratio=str(ratio),maximum_relative_floor=str(floor),
            floor_float=float(floor),metrics=metrics([float(ratio*x-1) for x in switch_ratios]),
            scope='Post-hoc lower bound over one common switch multiplier; not a fitted predictor.'),
        conclusion='For any unchanged r2 parameters and supported same-card invocation, identical consumed static inputs imply identical time outputs. T0(current)/T0(previous)=1, so static-normalized previous-event prediction is exactly previous_call. Actual history card mismatch remains rejected; no model predict or supply_predict was called.',
        boundaries=['Equality is specific to these full aligned sw1/sw8 and same/mirror inputs.',
            'The model omits coordinate/address order from pricing here; this does not prove hardware independence of order.',
            'Source definitions, coefficients, target numeric values, and old frozen qualification are unchanged.',
            'No future-input or current target W/f is used for a forecast; oracle uses targets only to state a retrospective bound.',
            'Unknown initial state, cfg_a/c histories, event-to-plain mapping, independent holdout and cross-card time rule remain unestablished.'])
    (out/'equivalence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with gzip.open(out/'static-signature.json.gz','wt') as stream: json.dump(reference,stream,sort_keys=True,allow_nan=False)
    with gzip.open(out/'event-identity.json.gz','wt') as stream: json.dump(forecasts,stream,allow_nan=False)
    print(json.dumps(dict(verification=result['verification'],oracle=result['first_switch_oracle'],negative_checks=negative_checks),indent=2))


if __name__=='__main__': main()
