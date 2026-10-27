#!/usr/bin/env python3
"""Independent arithmetic, prediction replay and price-source verification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

import numpy as np

import v09_long_k_rejected as v09_model


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(base, result_root, output):
    attribution=read(result_root/'attribution-v1/attribution.json')
    prices=read(result_root/'form1-v2/parameters.json')
    report=read(result_root/'form1-v2/comparison.json')
    calibration=read(base/'Q15-full-time/V09-component-validation/20261009-V09-freeze-job738972-r2/frozen/calibration.json')
    frozen=read(base/'Q15-full-time/V09-component-validation/20261009-V09-freeze-job738972-r2/frozen/predictions.json')['predictions']
    checkpoint=result_root/'checkpoint-v1/README.md'
    assert sha(checkpoint)==report['checkpoint_sha256']==prices['checkpoint_sha256']
    for artifact in (attribution,report):
        for path,digest in artifact['input_sha256'].items():
            assert sha(Path(path))==digest,path
    # Three independent per-process identities from the retained event partitions.
    partition_checks=0
    for case in attribution['cases']:
        for process in case['processes']:
            for version in process['versions'].values():
                dec=version['observed_last_decomposition']
                assert math.isclose(sum(dec.values()),process['observed_envelope_ns']-version['envelope_ns'],abs_tol=1e-5)
                for detail in version['details'].values():
                    actual=sum(detail['observed_parts_ns'].values())
                    predicted=sum(detail['predicted_parts_ns'].values())
                    assert math.isclose(actual,detail['actual_duration_ns'],abs_tol=1e-5)
                    assert math.isclose(predicted,detail['predicted_duration_ns'],abs_tol=1e-5)
                    assert math.isclose(sum(detail['signed_observed_minus_predicted_parts_ns'].values()),actual-predicted,abs_tol=1e-5)
                    partition_checks+=1
    # Recalculate beta without using the fitter or its precomputed training prices.
    by_id={c['case']:c for c in attribution['cases']}
    fit_checks=[]
    for fitted in [prices['full'],*prices['folds']]:
        if fitted['status']=='unidentified':
            continue
        omitted=fitted.get('omitted_shape')
        for cfg,sources in fitted['beta_sources'].items():
            values=[]
            for case_id in sources:
                case=by_id[case_id];row=case['row'];setup=case['setup']
                assert [row['m'],row['n']]!=omitted
                n=max(map(len,v09_model.events.scheduled_work(cfg,row['m'],row['n'],setup['grid'],row['swizzle'])))-1
                process_prices=[]
                for process in case['processes']:
                    q=process['versions']['measured_frequency_conditioned']
                    process_prices.append(q['groups']['max_tile_count']['signed_parts_residual_ns']['later_L']['median']
                        /n/(row['k']//64)*q['frequency_ghz'])
                values.append(st.median(process_prices))
            assert math.isclose(max(0,st.median(values)),fitted['beta'][cfg],abs_tol=1e-12)
        # Independently verify the two-coefficient constrained optimum (KKT).
        design=[];target=[]
        for case_id in fitted['clock_sources']:
            case=by_id[case_id];row=case['row'];assert [row['m'],row['n']]!=omitted
            pred=v09_model.predict_components(row,case['setup'],calibration)
            point=pred['static_clock_work'];window=case['summary']['envelope_median_us']
            frequency=case['summary']['versions']['measured_frequency_conditioned']['frequency_ghz']['median']
            x=window/calibration['clock']['tau_us'];g=1+math.expm1(-x)/x
            design.append([point['tensor_mean_cycles']/(1000*window*frequency),point['source_mean_kib']*g/(window*frequency)])
            target.append((calibration['clock']['coefficients'][0]-frequency)/frequency)
        matrix=np.asarray(design);vector=np.asarray(target);coef=np.asarray([fitted['clock']['coefficients'][1],fitted['clock']['coefficients'][4]])
        gradient=matrix.T@(matrix@coef-vector)
        assert np.linalg.matrix_rank(matrix)==2 and (coef>=0).all()
        scale=max(1,float(np.linalg.norm(matrix.T@vector)))
        assert all(abs(v)<=1e-7*scale if c>1e-10 else v>=-1e-7*scale for c,v in zip(coef,gradient))
        fit_checks.append(dict(omitted_shape=omitted,gradient=gradient.tolist(),rank=2))
    folds={tuple(f['heldout_shape']):f for f in prices['folds']}
    replay_checks=0;zero_checks=0;domain_checks=0
    for case in report['cases']:
        row=case['row'];setup=by_id[case['case']]['setup']
        for name in ('r2','cost_only','clock_only','combined','combined_shape_fold'):
            fitted=folds[(row['m'],row['n'])] if name=='combined_shape_fold' else prices['full']
            if fitted['status']=='unidentified':
                continue
            cal=json.loads(json.dumps(calibration))
            if name in ('clock_only','combined','combined_shape_fold'):
                cal['clock']=fitted['clock']
            beta=fitted['beta'][row['config']] if name in ('cost_only','combined','combined_shape_fold') else None
            pred=v09_model.predict_components(row,setup,cal,later_loop_extra_cycles_per_ktile=beta)
            saved=case['versions'][name]
            assert pred['plain_us']==saved['plain_us'] and pred['frequency_ghz']==saved['frequency_ghz']
            if name=='r2':
                plain=v09_model.predict_components(row,setup,cal,later_loop_extra_cycles_per_ktile=0.0)
                assert plain==pred
                zero_checks+=1
                if case['case'] in report['fixed_conditions']:
                    serialized=json.loads(json.dumps(pred))
                    assert all(frozen[case['case']].get(k)==v for k,v in serialized.items())
            replay_checks+=1
        if not domain_checks:
            for bad in (-1,True,float('nan'),float('inf'),'1'):
                try:v09_model.predict_components(row,setup,cal,later_loop_extra_cycles_per_ktile=bad)
                except ValueError:domain_checks+=1
                else:raise AssertionError('invalid static price accepted')
    # Static coverage is an eligibility witness, not an HBM miss observation.
    device=base.parent/'20261001-memory-c/device.jsonl'
    hardware=read(device)
    assert hardware['uuid_hex']=='43269fbc449d3e0f908a9c81229546d3'
    capacity=hardware['l2_cache_bytes'];features=[]
    for case in attribution['cases']:
        row=case['row'];unique=2*row['k']*(row['m']+row['n'])
        features.append(dict(case=case['case'],shape=[row['m'],row['n']],k=row['k'],logical_unique_A_B_bytes=unique,
            spill_fraction=max(0,1-capacity/unique),protocol='dual10',source=case['source']))
    independent=[f for f in features if f['shape']!=[6144,8192]]
    assert len(independent)==33 and all(f['spill_fraction']==0 for f in independent)
    ends=[]
    for run in ('Q09-frequency/R07-R09-inkernel-clock/20261009-R09-input-clock-calibration-job738376','Q09-frequency/R07-R09-inkernel-clock/20261009-R09-equal-work-job738459'):
        for row in read(base/run/'cases.json'):
            unique=2*row['k']*(row['m']+row['n'])
            if unique>capacity:
                ends.append(dict(run=run,case=row['id'],unique_bytes=unique,protocol='ends',role_L_observed=False))
    fixed=report['fixed_conditions'];assert len(fixed)==29
    assert report['original_rejected']==['v09d_cfg_c_G2_base']
    numerical_gates={}
    for name in ('cost_only','clock_only','combined','combined_shape_fold'):
        errors={c['case']:abs(c['versions'][name]['plain_error_pct']) for c in report['cases'] if c['case'] in fixed}
        before={c['case']:abs(c['versions']['r2']['plain_error_pct']) for c in report['cases'] if c['case'] in fixed}
        targets={c['case'] for c in report['cases'] if c['row']['group']=='G6_base'}
        other=set(fixed)-targets
        improved=all(errors[k]<before[k]-1e-9 for k in targets)
        gate=improved and st.median(errors[k] for k in other)<=st.median(before[k] for k in other)+1e-9 and max(errors[k] for k in other)<=max(before[k] for k in other)+1e-9
        gate=gate and all(errors[k]-before[k]<=2+1e-9 for k in fixed)
        gate=gate and all(errors[k]<=10+1e-9 for k in report['nominal_proposed_scope'] if before[k]<=10)
        assert gate==report['decisions'][name]['time_gate']['passed']==False
        numerical_gates[name]=dict(time_pass=False,every_G6_time_improved=improved)
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='verified_failed_development_forms',gpu_jobs=0,partition_identities_checked=partition_checks,
        fitted_source_and_KKT_checks=fit_checks,predictions_replayed=replay_checks,zero_price_exact_checks=zero_checks,
        bad_price_rejections=domain_checks,fixed_scored_rows=29,original_rejection_retained=True,
        form1_numerical_gates=numerical_gates,
        form2_price_design=dict(feature_rows=features,independent_rows_after_G6_shape_removed=len(independent),
            nonzero_independent_rows=0,design_rank_after_G6_shape_removed=0,
            L2_bytes=capacity,device_record_sha256=sha(device),ends_positive_features=ends,
            limitation='Whole input coverage does not prove miss/traffic. ends may constrain other whole-window hypotheses; no matched local dual service price is assigned.'),
        inputs={str(p):sha(p) for p in (result_root/'attribution-v1/attribution.json',result_root/'form1-v2/comparison.json',
            result_root/'form1-v2/parameters.json',checkpoint,device)},verifier_sha256=sha(Path(__file__)))
    (output/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'source').mkdir();(output/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('Verified',replay_checks,'predictions,',partition_checks,'partition identities; form2 independent dual rank 0.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--result-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verify(args.base.resolve(),args.result_root.resolve(),args.output.resolve())
