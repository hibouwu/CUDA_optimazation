#!/usr/bin/env python3
"""One predeclared serial-loop hypothesis; fixed V09 development diagnostics."""
import argparse
import copy
import json
from pathlib import Path
import statistics as st

import numpy as np
from scipy.optimize import lsq_linear

import analyze_v09_gate_audit as audit
from analyze_r13_sm import select_attempts
from analyze_r18 import replay
from analyze_v10_task_a import adoption, case_components, scope_reasons
from run_v09 import supply_identification, check_prediction_support
import v06_run
import v09_long_k_rejected as v09_model


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def calibration_rows(attribution, calibration):
    eligible = []
    excluded = {}
    for case in attribution['cases']:
        row, setup = case['row'], case['setup']
        reasons = scope_reasons(row, setup)
        prediction = v09_model.predict_components(row, setup, calibration)
        if not prediction['all_supply_supported']:
            reasons.append('supply unsupported')
        tm, tn, _ = v09_model.events.base.CONFIGS[row['config']]['tile']
        if any(mi*tm+tm>row['m'] or ni*tn+tn>row['n']
               for cta in v09_model.events.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'])
               for mi, ni in cta):
            reasons.append('partial or padded nominal output work')
        if reasons:
            excluded[case['case']] = reasons
            continue
        work = v09_model.events.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'])
        later_count = max(map(len, work))-1
        beta_values = []
        for process in case['processes']:
            version = process['versions']['measured_frequency_conditioned']
            if later_count:
                beta_values.append(version['groups']['max_tile_count']['signed_parts_residual_ns']['later_L']['median']
                    *version['frequency_ghz']/(later_count*(row['k']//64)))
        eligible.append(dict(case=case['case'],shape=[row['m'],row['n']],config=row['config'],
            mode=row['input_mode'],beta_case_cycles_per_ktile=st.median(beta_values) if beta_values else None,
            frequency_ghz=case['summary']['versions']['measured_frequency_conditioned']['frequency_ghz']['median'],
            window_us=case['summary']['envelope_median_us'],point=prediction['static_clock_work']))
    return eligible, excluded


def fit(training, calibration, omitted_shape=None):
    kept = [c for c in training if c['shape'] != omitted_shape]
    beta = {}
    beta_sources = {}
    for cfg in ('cfg_a','cfg_b','cfg_c'):
        values = [c for c in kept if c['config']==cfg and c['beta_case_cycles_per_ktile'] is not None]
        if not values:
            return dict(status='unidentified',reason='no independent later-loop price: '+cfg)
        beta[cfg] = max(0.0,st.median(c['beta_case_cycles_per_ktile'] for c in values))
        beta_sources[cfg] = [c['case'] for c in values]
    # Old MSE/rank/tau-profile diagnostics belong to the original seven-column
    # fit. Do not attach them to this new two-column conditional refit.
    clock = {key:copy.deepcopy(calibration['clock'][key]) for key in
             ('tau_us','coefficients','coefficient_order','constraints')}
    dyadic = [c for c in kept if c['mode']=='dyadic']
    design = np.array([[c['point']['tensor_mean_cycles']/(1000*c['window_us']),
        c['point']['source_mean_kib']/c['window_us']*v09_model.clock.duration_term(c['window_us'],'relaxation',clock['tau_us'])]
        for c in dyadic])
    rates = np.array([c['frequency_ghz'] for c in dyadic])
    rank = int(np.linalg.matrix_rank(design))
    if rank != 2:
        return dict(status='unidentified',reason='dyadic clock design rank '+str(rank))
    solution = lsq_linear(design/rates[:,None],(clock['coefficients'][0]-rates)/rates,
                         bounds=(0,np.inf),tol=1e-12,lsmr_tol=1e-12,max_iter=1000)
    if not solution.success or not np.isfinite(solution.x).all():
        raise ValueError('nonnegative clock fit failed')
    clock['coefficients'][1],clock['coefficients'][4] = map(float,solution.x)
    clock['training_window_us'] = [min(c['window_us'] for c in dyadic),max(c['window_us'] for c in dyadic)]
    return dict(status='development_fit',omitted_shape=omitted_shape,beta=beta,beta_sources=beta_sources,
        clock=clock,clock_sources=[c['case'] for c in dyadic],clock_design_rank=rank,
        clock_singular_values=np.linalg.svd(design,compute_uv=False).tolist(),
        constraint='Only dyadic d/e refitted; intercept, tau, zero/random coefficients frozen r2.')


def absolute_ns(ctas, prediction):
    errors = {name:[] for name in audit.COMPONENTS}
    for cta, p in zip(ctas,prediction['cta_events']):
        a,b = cta['tiles_ns'],p['events']
        if not a:
            continue
        windows = [('prefix',a[0][0]-cta['entry_ns'],p['supply']),
            ('prefill',a[0][0]-cta['prod_ns'],p['prefill']),
            ('first_L',a[0][1]-a[0][0],b[0][1]-b[0][0]),
            ('last_E',a[-1][3]-a[-1][2],b[-1][3]-b[-1][2]),
            ('tail',cta['end_ns']-max(t[3] for t in a),p['tail'])]
        windows += [('later_L',t[1]-t[0],u[1]-u[0]) for t,u in zip(a[1:],b[1:])]
        for name,actual,estimate in windows:
            errors[name].append(abs(estimate-actual))
    return {name:dict(windows=len(values),median_ns=st.median(values) if values else None,
        maximum_ns=max(values) if values else None) for name,values in errors.items()}


def summary(processes, prediction, plain, rate):
    ns = {name:dict(median_ns=st.median(p['absolute_ns'][name]['median_ns'] for p in processes
                  if p['absolute_ns'][name]['median_ns'] is not None),
                  maximum_ns=max(p['absolute_ns'][name]['maximum_ns'] for p in processes
                  if p['absolute_ns'][name]['maximum_ns'] is not None))
          for name in audit.COMPONENTS
          if any(p['absolute_ns'][name]['median_ns'] is not None for p in processes)}
    return dict(plain_us=prediction['plain_us'],plain_error_pct=100*(prediction['plain_us']/plain-1),
        frequency_ghz=prediction['frequency_ghz'],frequency_error_pct=100*(prediction['frequency_ghz']/rate-1),
        converted_cycle_error_pct=100*(prediction['plain_us']*prediction['frequency_ghz']/(plain*rate)-1),
        components=case_components(processes),absolute_components_ns=ns,
        critical=dict(candidate_count=len(prediction['critical_candidates']),representative=prediction['critical_cta'],
            set_hits=sum(p['critical']['frozen_set_intersects_exact_last'] for p in processes),
            representative_hits=sum(p['critical']['chosen_is_exact_last'] for p in processes),
            median_regret_pct=st.median(p['critical']['end_regret_percent'] for p in processes),
            maximum_regret_pct=max(p['critical']['end_regret_percent'] for p in processes)))


def run(base, attribution_path, checkpoint_path, output):
    attribution = read(attribution_path)
    checkpoint = read(checkpoint_path/'binding.json')
    if checkpoint['checkpoint_sha256'] != v06_run.sha(checkpoint_path/'README.md'):
        raise ValueError('pre-fit checkpoint changed')
    for path,digest in attribution['input_sha256'].items():
        if v06_run.sha(Path(path)) != digest:
            raise ValueError('attribution input changed: '+path)
    freeze = base/'Q15-full-time/V09-component-validation/20261009-V09-freeze-job738972-r2'
    observed = base/'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011'
    calibration = read(freeze/'frozen/calibration.json')
    frozen = read(freeze/'frozen/predictions.json')['predictions']
    manager = {c['case']:c for c in read(observed/'reanalysis/manager-results-v1/summary.json')['cases']}
    targets = [c for c in attribution['cases'] if c['row']['set']=='heldout']
    fixed = [c['case'] for c in targets if frozen[c['case']]['status']=='predicted']
    if len(fixed)!=29 or len(targets)!=30:
        raise ValueError('fixed comparison changed')
    output.mkdir(parents=True,exist_ok=False)
    training,excluded = calibration_rows(attribution,calibration)
    full = fit(training,calibration)
    shapes = sorted({tuple(c['row'][k] for k in ('m','n')) for c in targets})
    folds = {shape:fit(training,calibration,list(shape)) for shape in shapes}
    write(output/'parameters.json',dict(training=training,excluded_training=excluded,full=full,
        folds=[dict(heldout_shape=list(shape),**value) for shape,value in folds.items()],
        checkpoint_sha256=checkpoint['checkpoint_sha256']))
    if full['status']=='unidentified':
        raise ValueError(full['reason'])
    identity = supply_identification(freeze,calibration)
    reports = []
    hashes = {str(attribution_path):v06_run.sha(attribution_path),str(checkpoint_path/'README.md'):checkpoint['checkpoint_sha256'],
        str(freeze/'frozen/predictions.json'):v06_run.sha(freeze/'frozen/predictions.json'),
        str(observed/'reanalysis/manager-results-v1/summary.json'):v06_run.sha(observed/'reanalysis/manager-results-v1/summary.json')}
    missing_folds = []
    native_witnesses = []
    for case in targets:
        row,setup = case['row'],case['setup']
        selected,failed,_ = select_attempts(observed,row['id'],('dual',))
        traces = []
        for item in selected:
            record = item['record']
            traces.append(replay(observed,record,row,retain_roles=True)['ctas'])
            hashes[str(observed/item['path'])] = v06_run.sha(observed/item['path'])
            hashes[str(observed/record['raw'])] = record['raw_sha256']
        versions = {}
        fold = folds[(row['m'],row['n'])]
        for label,parameters,use_clock,use_cost in (
            ('r2',full,False,False),('cost_only',full,False,True),('clock_only',full,True,False),
            ('combined',full,True,True),('combined_shape_fold',fold,True,True)):
            if parameters['status']=='unidentified':
                versions[label] = dict(error=parameters['reason'],plain_error_pct=None,frequency_error_pct=None)
                missing_folds.append(dict(case=row['id'],reason=parameters['reason']))
                continue
            cal = copy.deepcopy(calibration)
            if use_clock:
                cal['clock'] = parameters['clock']
            extra = parameters['beta'][row['config']] if use_cost else None
            prediction = v09_model.predict_components(row,setup,cal,later_loop_extra_cycles_per_ktile=extra)
            if label=='r2' and row['id'] in fixed:
                serialized = json.loads(json.dumps(prediction))
                if any(frozen[row['id']].get(k)!=v for k,v in serialized.items()):
                    raise ValueError('default r2 prediction changed: '+row['id'])
            processes = [dict(trial=item['record']['trial'],**audit.process_audit(ctas,prediction),
                absolute_ns=absolute_ns(ctas,prediction)) for item,ctas in zip(selected,traces)]
            versions[label] = dict(**summary(processes,prediction,manager[row['id']]['observed_us'],
                manager[row['id']]['observed_frequency_ghz']),
                original_status=frozen[row['id']]['status'],scope_reasons=scope_reasons(row,setup),
                supply_support_failures=check_prediction_support(row,setup,cal,prediction,identity),processes=processes)
        if row['group']=='G6_base':
            for item,ctas in zip(selected,traces):
                n=max(len(c['work']) for c in ctas);cohort=[c for c in ctas if len(c['work'])==n]
                last=max(ctas,key=lambda c:c['end_ns'])
                by_tile=[]
                for j in range(n):
                    cy=[c['tiles'][j][1]-c['tiles'][j][0] for c in cohort]
                    ns=[c['tiles_ns'][j][1]-c['tiles_ns'][j][0] for c in cohort]
                    by_tile.append(dict(j=j,merged_L_cycles_median=st.median(cy),merged_L_cycles_max=max(cy),
                        merged_L_ns_median=st.median(ns),merged_L_ns_max=max(ns)))
                native_witnesses.append(dict(case=row['id'],trial=item['record']['trial'],last_cta=last['cta'],last_sm=last['sm'],
                    cohort_by_tile=by_tile,
                    last_role_L=[dict(role=r,j=j,cycles=t[1]-t[0],ns=u[1]-u[0])
                      for r,role in enumerate(last['roles']) for j,(t,u) in enumerate(zip(role['tiles'],role['tiles_ns']))]))
        reports.append(dict(case=row['id'],row=row,failed_attempts_retained=failed,versions=versions))
        print(row['id'],{k:round(v['plain_error_pct'],3) if v['plain_error_pct'] is not None else None for k,v in versions.items()},flush=True)
    baseline = {c['case']:abs(c['versions']['r2']['plain_error_pct']) for c in reports if c['case'] in fixed}
    target_ids = [c['case'] for c in reports if c['row']['group']=='G6_base']
    supported = [c['case'] for c in reports if c['case'] in fixed and not scope_reasons(c['row'],next(t['setup'] for t in targets if t['case']==c['case']))]
    decisions = {}
    for name in ('cost_only','clock_only','combined','combined_shape_fold'):
        errors = {c['case']:abs(c['versions'][name]['plain_error_pct']) if c['versions'][name]['plain_error_pct'] is not None else None
                  for c in reports if c['case'] in fixed}
        gate = adoption(baseline,errors,target_ids,supported)
        target_checks = []
        for case in reports:
            if case['case'] not in target_ids:
                continue
            before,after=case['versions']['r2'],case['versions'][name]
            frequency_improved = after['frequency_error_pct'] is not None and abs(after['frequency_error_pct'])<abs(before['frequency_error_pct'])
            component_checks = []
            if after.get('components'):
                for component in ('first_L','later_L'):
                    a,b=before['components'][component],after['components'][component]
                    component_checks.append(dict(component=component,
                        median_nonincrease=b['median_abs_pct']<=a['median_abs_pct']+1e-9,
                        worst_nonincrease=b['max_window_abs_pct']<=a['max_window_abs_pct']+1e-9,
                        numerical_5_10=b['median_abs_pct']<=5 and b['max_window_abs_pct']<=10))
            target_checks.append(dict(case=case['case'],frequency_improved=frequency_improved,components=component_checks))
        decisions[name] = dict(time_gate=gate,target_checks=target_checks,adopted=False,
            observation_qualified=False,critical_qualified=False,
            status='Development diagnostics only; original observation and critical qualification unchanged.')
    write(output/'comparison.json',dict(scope='Predeclared form 1. No GPU, validation freeze or qualification.',
        checkpoint_sha256=checkpoint['checkpoint_sha256'],gpu_jobs=0,fixed_conditions=fixed,
        original_rejected=[c['case'] for c in targets if c['case'] not in fixed],
        nominal_proposed_scope=supported,missing_folds=missing_folds,cases=reports,decisions=decisions,
        input_sha256=hashes,source_sha256={str(Path(m.__file__).resolve()):v06_run.sha(Path(m.__file__))
          for m in (v09_model,v09_model.clock,v09_model.supply,v09_model.output_model,audit)},
        analysis_sha256=v06_run.sha(Path(__file__))))
    write(output/'native-witnesses.json',dict(scope='Original dual observer. Same-role C/N and merged-domain summaries are diagnostics, not physical service prices.',cases=native_witnesses))
    (output/'source').mkdir()
    for p in (Path(__file__),Path(v09_model.__file__)):
        (output/'source'/p.name).write_bytes(p.read_bytes())
    print('Fixed 29 scored rows plus original rejection retained; no candidate adopted.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--attribution',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.base.resolve(),args.attribution.resolve(),args.checkpoint.resolve(),args.output.resolve())
