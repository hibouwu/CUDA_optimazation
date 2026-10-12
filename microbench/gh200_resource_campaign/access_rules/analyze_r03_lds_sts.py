#!/usr/bin/env python3
"""Calibrate or independently score the ordinary SMEM hypothesis; never refit holdout."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from r03_lds_sts_model import demand, group_key, predict


def load(path):
    return json.loads(path.read_text())


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def old_trial(path):
    records=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    row=next(item for item in records if item.get('type')=='trial')
    detail=row['blocks_detail'][0]
    cycles=detail['stop_cycle']-detail['start_cycle']
    return dict(cycles=cycles,rate=row['work_count']/cycles,smid=detail['smid'],
        warmup_converged=row['warmup_converged'],elapsed_ns=row['stop_ns']-row['start_ns'])


def summarize(case,folder):
    records=[];failures=[]
    for trial in range(3):
        path=folder/'samples'/case['id']/f'trial-{trial:02}'
        if (path/'failure.json').exists():failures.append(load(path/'failure.json'))
        elif (path/'result.json').exists():records.append(load(path/'result.json'))
        else:failures.append(dict(trial=trial,error='missing prelisted process'))
    if failures or len(records)!=3:
        return dict(case=case['id'],status='failed',failures=failures)
    cycles=[r['elapsed_cycles'] for r in records]
    cv=statistics.stdev(cycles)/statistics.mean(cycles)
    return dict(case=case['id'],status='measured',cycles=cycles,median_cycles=statistics.median(cycles),
        cv=cv,range_cycles=max(cycles)-min(cycles),smids=[r['smid'] for r in records],
        registers=records[0]['registers'],dynamic_smem_bytes=records[0]['dynamic_smem_bytes'])


def fit_group(cases,rows):
    # Enumerate all floor breakpoints. Four aggregate parameters, no internal
    # port or latency identification; the fixed term stays in the full window.
    batches=np.array([demand(c)['total_per_round'] for c in cases],float)
    repeats=np.array([c['iterations'] for c in cases],float)
    observed=np.array([r['median_cycles'] for r in rows],float)
    best=None
    for threshold in sorted(set(batches))+[0.0]:
        low=batches<=threshold
        matrix=np.column_stack([np.ones(len(cases)),repeats*low,repeats*(~low),repeats*batches*(~low)])
        coefficient,_,rank,_=np.linalg.lstsq(matrix/observed[:,None],np.ones(len(cases)),rcond=None)
        fixed,floor,offset,price=map(float,coefficient)
        if rank<4 or price<0 or floor<=0:
            continue
        calculated=fixed+repeats*np.maximum(floor,offset+price*batches)
        if fixed < -256 or np.any(calculated<=0):
            continue
        mse=float(np.mean(((calculated-observed)/observed)**2))
        candidate=dict(fixed_cycles=fixed,floor_cycles_per_round=floor,offset_cycles_per_round=offset,
            cycles_per_hypothesized_batch=price,batch_span=[int(min(batches)),int(max(batches))],
            fit_relative_rms=math.sqrt(mse),design_rank=int(rank),threshold_hypothesis=float(threshold))
        if best is None or mse<best[0]:best=(mse,candidate)
    if best is None:
        return dict(status='not_estimable',reason='no full-rank nonnegative service envelope fit')
    return dict(status='fitted_conditional',**best[1])


def analyze(package,phase,output):
    output.mkdir(exist_ok=False,parents=True)
    folder=package/phase
    if phase=='development':
        cases=load(package/'development-cases.json')
        old=[]
        for path in sorted((folder/'compatibility').glob('*w*-*/measure.stdout')):
            try:old.append(dict(case=path.parent.name.rsplit('-',1)[0],**old_trial(path)))
            except Exception as error:old.append(dict(case=path.parent.name,status='failed',error=str(error)))
        expected={'read_w4_stride2':64,'read_w4_stride4':32,'read_w4_stride8':16,'read_w4_stride16':8,'read_w4_stride32':4,
            'read_w8_stride1':128,'read_w16_stride1':128,'write_w4_stride1':128}
        gates=[]
        for case,rate in expected.items():
            selected=[r['rate'] for r in old if r['case']==case and 'rate' in r]
            value=statistics.median(selected) if len(selected)==3 else None
            gates.append(dict(case=case,expected_old_trend=rate,measured_rate=value,
                passed=value is not None and abs(value/rate-1)<=.15))
        broadcast=[r['rate'] for r in old if r['case']=='read_w4_broadcast' and 'rate' in r]
        broad_median=statistics.median(broadcast) if len(broadcast)==10 else None
        gates.append(dict(case='read_w4_broadcast',old_observed_range=[88.51999087216949,110.68736516398039],
            measured_rate=broad_median,passed=broad_median is not None and 88.51999087216949*.85<=broad_median<=110.68736516398039*1.15))
        write(output/'old-compatibility.json',dict(rows=old,stride_gates=gates,
            broadcast_rates=broadcast,broadcast_cv=statistics.stdev(broadcast)/statistics.mean(broadcast) if len(broadcast)>1 else None,
            note='Old absolute values are window-specific and are not transferred as new prices.'))
        rows=[summarize(c,folder) for c in cases]
        write(output/'development-results.json',rows)
        grain=[load(path) for path in sorted(folder.glob('grain-*.stdout'))]
        cd=[value for row in grain for value in row['cycle_deltas']]
        nd=[value for row in grain for value in row['ns_deltas']]
        controls=[dict(iterations=c['iterations'],warps=c['warps'],**summarize(c,folder))
            for c in load(package/'control-cases.json')]
        write(output/'timing-controls.json',dict(counter_samples=grain,empty_and_loop_windows=controls,
            observed_cycle_step_gcd=math.gcd(*cd),minimum_cycle_read_spacing=min(cd),
            observed_positive_ns_step_gcd=math.gcd(*[x for x in nd if x>0]),zero_ns_pairs=sum(x==0 for x in nd),
            representation_granularity_cycles=1,
            interpretation='Observed counter-read spacing/gcd are protocol diagnostics, not a proof of intrinsic physical timer resolution. Loop control is not subtracted as hardware service.'))
        if not all(g['passed'] for g in gates):
            write(output/'decision.json',dict(status='stop_protocol_difference',candidate=False,
                reason='EXP09 compatible stride trend not reproduced; no unexplained-difference pricing'))
            return
        if any(r['status']!='measured' or r['cv']>.02 for r in rows):
            write(output/'decision.json',dict(status='stop_failed_development_gate',candidate=False,
                failed_cases=[r['case'] for r in rows if r['status']!='measured' or r.get('cv',0)>.02]))
            return
        groups={};tolerances={}
        for key in sorted({group_key(c) for c in cases}):
            selected=[(c,r) for c,r in zip(cases,rows) if group_key(c)==key]
            groups[key]=fit_group([c for c,r in selected],[r for c,r in selected])
            tolerances[key]=math.ceil(max(64,3*max(r['range_cycles'] for c,r in selected)))
        parameters=dict(schema='r03-conditional-smem/1',groups=groups,
            status='development_only_not_independently_qualified',iterations_span=[256,8192],
            absolute_cycle_tolerance=tolerances,physical_batches_measured=False,
            GPU_uuid=load(folder/'environment.json')['gpu_uuid'],
            unit='clock64 cycles / single CTA',protocol_sha256=hashlib.sha256((package/'protocol.json').read_bytes()).hexdigest(),
            scope='ordinary full-warp LDS/STS 32/64/128bit; same binary and GPU UUID; independent and explicit dependent read organizations',
            parameter_meaning='aggregate conditional cycle envelope; fixed/issue/consumer/loop costs remain included; no unique physical service decomposition')
        write(output/'parameters.json',parameters)
        failed=[key for key,value in groups.items() if value['status']!='fitted_conditional']
        if failed:
            write(output/'decision.json',dict(status='stop_not_estimable',candidate=False,groups=failed))
            return
    else:
        cases=load(package/'freeze/holdout-cases.json');parameters=load(package/'freeze/parameters.json')
        rows=[summarize(c,folder) for c in cases]
    comparison=[]
    for case,row in zip(cases,rows):
        result=dict(**row,group=group_key(case),layout=case['layout'],native=case['native'],
            predicted_cycles=None,signed_error=None,absolute_cycle_error=None)
        if row['status']=='measured':
            try:
                predicted=predict(case,parameters)['window_cycles']
                result.update(predicted_cycles=predicted,signed_error=predicted/row['median_cycles']-1,
                    absolute_cycle_error=abs(predicted-row['median_cycles']),
                    absolute_cycle_tolerance=parameters['absolute_cycle_tolerance'][group_key(case)],
                    local_absolute_cycle_pass=abs(predicted-row['median_cycles'])<=parameters['absolute_cycle_tolerance'][group_key(case)],
                    observed_cycles_per_target_instruction=row['median_cycles']/(case['iterations']*8*case['warps']))
            except Exception as error:result.update(status='prediction_failed',error=str(error))
        comparison.append(result)
    numeric=[r for r in comparison if r['signed_error'] is not None]
    all_numeric=len(numeric)==len(cases)
    relative=[abs(r['signed_error']) for r in numeric];absolute=[r['absolute_cycle_error'] for r in numeric]
    metrics=dict(prelisted_conditions=len(cases),numeric_conditions=len(numeric),
        median_absolute_relative_error=statistics.median(relative) if relative else None,
        maximum_absolute_relative_error=max(relative) if relative else None,
        median_absolute_cycle_error=statistics.median(absolute) if absolute else None,
        maximum_absolute_cycle_error=max(absolute) if absolute else None)
    passed=all_numeric and metrics['median_absolute_relative_error']<=.05 and metrics['maximum_absolute_relative_error']<=.10
    if phase=='holdout':passed=passed and all(r.get('local_absolute_cycle_pass',False) and r.get('cv',1)<=.02 for r in comparison)
    write(output/'comparison.json',dict(phase=phase,rows=comparison,metrics=metrics,passed=passed,
        fitting_allowed=phase=='development',physical_batches_measured=False))
    write(output/'decision.json',dict(status=('ready_to_freeze' if phase=='development' else 'passed_independent_holdout') if passed else 'rejected_conditional_form',
        candidate=passed,metrics=metrics,full_epilogue_claimed=False,physical_batches_measured=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--phase',choices=('development','holdout'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.package.resolve(),a.phase,a.output.resolve())
