#!/usr/bin/env python3
"""CPU development ablation and failure diagnostics against frozen V09 r2.

Fits only pre-V09 calibration records. V09 heldout observations are used only for
scoring after parameters are fixed. Nothing is a new heldout validation.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import statistics as st
from unittest.mock import patch

import numpy as np
import v06_fit
import v06_run as common
import v08_fit
import v08_model as events
import v09_model as model
from analyze_r18 import replay
from analyze_r13_sm import select_attempts
from analyze_r13_supply import metrics
from run_v08 import boundary_kind
from run_v09 import supply_identification, check_prediction_support


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def calibration_cases(base):
    """50 source calibration conditions plus the R15 single-output high scale."""
    names=['20261009-R13-R15-R18-composition-job738496','20261009-R10-b-coverage-job738203',
           '20261009-R18-input-map-job738296','20261009-R18-input-map-n-job738307',
           '20261009-R18-first-fill-job738707','20261009-R18-first-fill-job738865',
           '20261009-R15-dual-roles-job738397']
    cases=[];rows={};provenance={}
    for name in names:
        root=base/name;setups={s['case']:s['setup'] for s in read(root/'static_setup.json')}
        assert read(root/'environment.json')['gpu'].startswith('GPU-43269fbc-')
        for r in read(root/'cases.json'):
            if name.endswith('738397') and r['k']!=4096:continue
            row=dict(r,kind=boundary_kind(r['config'],r['m'],r['n']))
            assert row['id'] not in rows
            rows[row['id']]=row;setup=setups[row['id']]
            feat=events.features(row,setup['grid']);schedule=events.base.CONFIGS[row['config']]['schedule']
            selected,failed,_=select_attempts(root,row['id'],('dual',))
            assert len(selected)==10
            intervals=[];last=[];tile_L={};max_cycles=[];envelopes=[]
            for item in selected:
                rec=item['record'];observed=replay(root,rec,row);ctas=observed['ctas']
                assert [[tuple(w) for w in c['work']] for c in ctas]==feat['work']
                intervals.append(v06_fit.intervals(observed,schedule)[0])
                max_cycles.append(max(c['end_c']-c['entry_c'] for c in ctas))
                envelopes.append((max(c['end_ns'] for c in ctas)-min(c['entry_ns'] for c in ctas))/1000)
                for c in ctas:
                    if len(c['tiles'])==feat['T']:last.append(c['tiles'][-1][3]-c['tiles'][-1][2])
                    for j,(t,w) in enumerate(zip(c['tiles'],c['work'])):
                        if j:tile_L.setdefault(events.tile_class(row['config'],row,*w),[]).append(t[1]-t[0])
                provenance[str(root/rec['raw'])]=rec['raw_sha256']
            keys=set().union(*(p for p in intervals))
            cases.append(dict(id=row['id'],config=row['config'],set='calib',grid=setup['grid'],
                T=feat['T'],kt=feat['kt'],q=feat['q'],fp=feat['fp'],boundary=feat['boundary'],
                intervals={k:st.median(p[k] for p in intervals if k in p) for k in keys},
                E_last=st.median(last),c_max_stamped=st.median(max_cycles),
                observed_envelope_us=st.median(envelopes),
                tile_L={k:dict(median=st.median(v),n=len(v)) for k,v in tile_L.items()}))
    return cases,rows,provenance


def fit_v08_forms(cases,rows,old_forms):
    # Structural choices only come from V08. No old-card fitted numeric parameter
    # or old ends/stamped kappa is borrowed.
    v08_fit.ROWS=rows;v08_fit._FEAT.clear();params={};notes={}
    for cfg in ('cfg_a','cfg_b','cfg_c'):
        choice=old_forms[cfg]['choice'];group=[c for c in cases if c['config']==cfg]
        ordinary=[c for c in group if not c['boundary']];border=[c for c in group if c['boundary']]
        p=v08_fit.fit_params(ordinary,cfg,choice)
        if border:v08_fit.fit_boundary(p,cfg,border,choice['boundary'])
        params[cfg]=p
        notes[cfg]=dict(choice=choice,calibration_cases=[c['id'] for c in ordinary],
            boundary_cases=[c['id'] for c in border],
            scope='V08 forms refitted to same-card dual cycle intervals; boundary columns retain V08 calibrated-range clamping.')
    return params,notes


def clock_inputs(row,work,cal):
    cfg=row['config'];tm,tn,tk=events.base.CONFIGS[cfg]['tile'];cm,cn=events.base.CONFIGS[cfg]['cluster']
    kt=row['k']//tk;tiles=sum(map(len,work))
    point=dict(input_mode=row.get('input_mode','dyadic'),tensor_mean_cycles=tiles*kt*2*tm*tn*tk/4096/132,
        source_mean_kib=tiles*kt*(2*tm*tk/cn+2*tn*tk/cm)/1024/132)
    clock=copy.deepcopy(cal['clock']);z=model.zero_product_fraction(row,work)
    if cal.get('zero_activity_mixing',False):
        mode=model.clock.MODES.index(point['input_mode']);zero=model.clock.MODES.index('zero')
        for offset in (1,1+len(model.clock.MODES)):
            clock['coefficients'][offset+mode]=(1-z)*clock['coefficients'][offset+mode]+z*clock['coefficients'][offset+zero]
    return point,clock


def aggregate_prediction(row,setup,params,cal,observed_f=None):
    cfg=row['config'];feat=events.features(row,setup['grid']);p=events.case_params(params,cfg,feat)
    cyc,ctas,_,boundary=events.critical_cycles(params,cfg,feat)
    invalid=[k for k in ['P0','S','E','E0','Elast','Etail'] if k in p and p[k]<0]
    if invalid or cyc<=0:return dict(status='invalid',reason='negative aggregate interval: '+','.join(invalid),plain_us=None)
    point,clock=clock_inputs(row,feat['work'],cal)
    solved=(dict(frequency_ghz=observed_f,window_us=cyc/(1000*observed_f)) if observed_f is not None else
        model.clock.solve_source_envelope(point,clock,lambda f:cyc/(1000*f)))
    f=solved['frequency_ghz'];transfer=cal['plain_transfer'][cfg];crit=max(range(len(ctas)),key=ctas.__getitem__)
    timeline=[]
    for classes in feat['classes']:
        _,part=events.cta_cycles(p,events.base.CONFIGS[cfg]['schedule'],len(classes),feat['kt'],detail=True,classes=classes,trace_events=True)
        if part:
            part={k:([[v/f for v in t] for t in value] if k=='events' else value/f) for k,value in part.items()}
        timeline.append(part)
    return dict(status='numeric',plain_us=transfer['F_us']+transfer['kappa']*solved['window_us'],
        **solved,cta_events=timeline,critical_cta=crit,
        critical_candidates=[i for i,c in enumerate(ctas) if abs(c-max(ctas))<1e-7],
        boundary_cycles=boundary,boundary_ns=boundary/f,scope='Same-card refitted V08 forms with shared V09 clock and plain/dual transfer; global boundary correction is not assigned to CTA timelines; development only.')



def component_scores(root,row,versions):
    selected,_,_=select_attempts(root,row['id'],('dual',))
    processes={name:[] for name in versions}
    for item in selected:
        ctas=replay(root,item['record'],row)['ctas']
        last=max(range(len(ctas)),key=lambda i:ctas[i]['end_ns'])
        for name,version in versions.items():
            p=version['prediction']
            if 'cta_events' not in p:continue
            errors={k:[] for k in ('prefix','prefill','first_L','later_L','first_E','last_E','tail')}
            assert len(p['cta_events'])==len(ctas)
            for c,pred in zip(ctas,p['cta_events']):
                if not c['tiles_ns']:continue
                actual=c['tiles_ns'];pe=pred['events']
                for key,a,b in [('prefix',actual[0][0]-c['entry_ns'],pred['supply']),
                    ('prefill',actual[0][0]-c['prod_ns'],pred['prefill']),
                    ('first_L',actual[0][1]-actual[0][0],pe[0][1]-pe[0][0]),
                    ('first_E',actual[0][3]-actual[0][2],pe[0][3]-pe[0][2]),
                    ('last_E',actual[-1][3]-actual[-1][2],pe[-1][3]-pe[-1][2]),
                    ('tail',c['end_ns']-max(t[3] for t in actual),pred['tail'])]:
                    if a>0:errors[key].append(b/a-1)
                errors['later_L'] += [(b[1]-b[0])/(a[1]-a[0])-1 for a,b in zip(actual[1:],pe[1:]) if a[1]>a[0]]
            processes[name].append(dict(errors={k:st.median(v) if v else None for k,v in errors.items()},
                candidate_hit=last in p['critical_candidates'],candidate_count=len(p['critical_candidates'])))
    for name,values in processes.items():
        versions[name]['components']={k:st.median(x['errors'][k] for x in values if x['errors'][k] is not None)
            for k in ('prefix','prefill','first_L','later_L','first_E','last_E','tail') if any(x['errors'][k] is not None for x in values)}
        versions[name]['critical_candidates_hit_rate']=st.fmean(x['candidate_hit'] for x in values) if values else None


def failure_calls(base):
    """Yield the original 300 dual calls; keep rejected predictions as rejected.

    Roles are retained by the existing decoder so local cycle/ns comparisons can
    use the same consumer endpoints. Observations are for diagnosis only.
    """
    frozen=base/'20261009-V09-freeze-job738972-r2'
    target=base/'20261009-V09-heldout-job739011'
    from run_v09 import verify_approved
    approved=verify_approved(frozen,
        'f4df39e00cf1d2ab61947c911e699bd5bea4e533609d93f726de43077515d81e',
        '09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab')
    calibration=read(frozen/'frozen/calibration.json')
    setups={s['case']:s['setup'] for s in read(frozen/'static_setup.json')}
    rows=[row for row in read(frozen/'cases.json') if row['set']=='heldout']
    for row in rows:
        selected,failed,_=select_attempts(target,row['id'],('dual',))
        if len(selected)!=10:raise ValueError('incomplete dual processes: '+row['id'])
        for item in selected:
            record=item['record']
            yield dict(row=row,setup=setups[row['id']],trial=record['trial'],
                record=record,record_path=item['path'],failed_attempts=failed,
                observed=replay(target,record,row,retain_roles=True),
                prediction=approved['predictions'][row['id']],calibration=calibration)


def reference_comparison(base,controls,out):
    """Compare relevant call/supply references; a new card is not old calibration."""
    old=base/'20261009-V09-heldout-job739011'
    rows=[r for r in read(controls/'cases.json') if r['set']=='reference']
    previous={r['id']:r for r in read(old/'cases.json')}
    comparisons=[]
    for row in rows:
        summaries={}
        for name,root,item in [('V09_reference',old,previous[row['id']]),('current',controls,row)]:
            selected,failed,_=select_attempts(root,row['id'],('plain','dual'))
            calls=[];plain=[]
            for chosen in selected:
                rec=chosen['record'];obs=replay(root,rec,item)
                if rec['variant']=='plain':
                    plain.append(obs['elapsed_us']);continue
                ctas=obs['ctas'];longest=max(len(c['tiles']) for c in ctas)
                window=(max(c['end_ns'] for c in ctas)-min(c['entry_ns'] for c in ctas))/1000
                calls.append(dict(trial=rec['trial'],raw=rec['raw'],raw_sha256=rec['raw_sha256'],
                    window_us=window,frequency_ghz=st.median((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns'])
                        for c in ctas if len(c['tiles'])==longest),
                    mainloop_ns_per_ktile=st.median((t[1]-t[0])/(row['k']//64) for c in ctas for t in c['tiles_ns']),
                    dual_event_minus_window_us=obs['elapsed_us']-window))
            if len(plain)!=10 or len(calls)!=10:raise ValueError('incomplete common reference')
            summaries[name]=dict(plain_us=st.median(plain),
                **{k:st.median(c[k] for c in calls) for k in ('window_us','frequency_ghz','mainloop_ns_per_ktile','dual_event_minus_window_us')},
                calls=calls,failed_attempts=failed)
        comparisons.append(dict(case=row['id'],summaries=summaries,
            relative_change={k:summaries['current'][k]/summaries['V09_reference'][k]-1
                for k in ('plain_us','window_us','frequency_ghz','mainloop_ns_per_ktile')},
            dual_event_minus_window_change_us=summaries['current']['dual_event_minus_window_us']-summaries['V09_reference']['dual_event_minus_window_us']))
    write(out/'reference-comparison.json',dict(previous_environment=read(old/'environment.json'),
        current_environment=read(controls/'environment.json'),cases=comparisons,
        scope='Paired-window clock, aligned mainloop and same-call event/envelope references. No physical HBM rate is inferred. Different-card ratios do not calibrate or validate the old predictor.'))


def failure_decomposition(base,out,controls=None):
    """One entry point; family analyzers share the original raw decoder."""
    import r09_input_source_clock
    import analyze_r13_supply
    import analyze_r18
    out.mkdir(parents=True,exist_ok=True)
    for analyzer in (r09_input_source_clock,analyze_r13_supply,analyze_r18):
        analyzer.failure_diagnostics(base,out,failure_calls)
        if controls is not None:analyzer.control_diagnostics(controls,out)
    if controls is not None:reference_comparison(base,controls,out)


def candidate_development(base,out,controls=None,swizzle_controls=None,stage_events=None,common_history=None,raster_pitch=None,stage_progress=None,history_series=None,source_pressure=None,mirror_history=None,base_shift=None,work_swap=None,producer_delay=None,first_supply=None):
    """Compare explicit candidate mechanisms without changing frozen predictions."""
    import r09_input_source_clock
    import analyze_r13_supply
    import analyze_r18
    out.mkdir(parents=True,exist_ok=True)
    r09_input_source_clock.response_development(base,out)
    analyze_r13_supply.finite_stage_diagnostics(base,out)
    analyze_r18.boundary_candidate_analysis(base,out)
    analyze_r18.first_cluster_development(base,out)
    analyze_r18.boundary_endpoint_distribution(base,out)
    analyze_r18.boundary_cluster_coupling(base,out)
    if controls is not None:
        analyze_r18.fill_discrimination(controls,out)
        reference_comparison(base,controls,out)
    if swizzle_controls is not None:
        r09_input_source_clock.swizzle_diagnostics(swizzle_controls,out)
    if stage_events is not None:
        analyze_r13_supply.stage_event_diagnostics(stage_events,out)
        analyze_r13_supply.stage_event_model_diagnostics(base,out)
    if common_history is not None:
        r09_input_source_clock.common_history_diagnostics(common_history,out)
        r09_input_source_clock.history_state_diagnostics(common_history,out)
    if raster_pitch is not None:
        import analyze_r10
        analyze_r10.raster_pitch_diagnostics(raster_pitch,out)
        analyze_r10.source_organization_diagnostics(raster_pitch,out)
    if source_pressure is not None:
        import analyze_r10
        analyze_r10.source_pressure_diagnostics(source_pressure,out)
        analyze_r10.shared_source_service_diagnostics(source_pressure,out)
    if base_shift is not None:
        import analyze_r10
        analyze_r10.base_shift_diagnostics(base_shift,out)
    if work_swap is not None:
        import analyze_r10
        analyze_r10.work_swap_diagnostics(work_swap,out)
    if producer_delay is not None:
        import analyze_r10
        analyze_r10.producer_delay_diagnostics(producer_delay,out)
    if first_supply is not None:
        analyze_r13_supply.first_supply_diagnostics(first_supply,out)
    if mirror_history is not None:
        r09_input_source_clock.mirror_history_diagnostics(mirror_history,out)
    if stage_progress is not None:
        analyze_r13_supply.stage_progress_diagnostics(stage_progress,out)
    if history_series is not None:
        r09_input_source_clock.history_series_diagnostics(history_series,out)
        r09_input_source_clock.history_localization_diagnostics(history_series,out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--analysis',choices=('ablation','failure-decomposition','candidate-development'),default='ablation')
    parser.add_argument('--controls',type=Path,help='local control run for the selected diagnostic analysis')
    parser.add_argument('--swizzle-controls',type=Path,help='R09 sw1/sw8 paired run for candidate development')
    parser.add_argument('--stage-events',type=Path,help='R13 stage4/6 sparse event run')
    parser.add_argument('--common-history',type=Path,help='R09 common/adaptive history paired run')
    parser.add_argument('--raster-pitch',type=Path,help='R10 fixed-geometry raster/B32 paired run')
    parser.add_argument('--source-pressure',type=Path,help='R10 single-output source group pressure pair')
    parser.add_argument('--mirror-history',type=Path,help='R09 same/mirror source prehistory intervention')
    parser.add_argument('--base-shift',type=Path,help='R10 B base offset by pitch paired run')
    parser.add_argument('--work-swap',type=Path,help='R10 CTA55/91 whole-work-sequence permutation by pitch')
    parser.add_argument('--producer-delay',type=Path,help='R10 CTA55 producer start delay by pitch')
    parser.add_argument('--first-supply',type=Path,help='R13 first TMA issue and consumer wait markers, matched observation switch')
    parser.add_argument('--stage-progress',type=Path,help='R13 consumer commit/loop progress run')
    parser.add_argument('--history-series',type=Path,help='R09 prefix/target trajectory run')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();base=args.base.resolve();out=args.output.resolve()
    if args.controls is not None and args.analysis=='ablation':
        parser.error('--controls is not used by ablation')
    if any(p is not None for p in (args.swizzle_controls,args.stage_events,args.common_history,args.raster_pitch,args.stage_progress,args.history_series,args.source_pressure,args.mirror_history,args.base_shift,args.work_swap,args.producer_delay,args.first_supply)) and args.analysis!='candidate-development':
        parser.error('candidate control runs require candidate-development')
    if args.analysis!='ablation':
        controls=args.controls.resolve() if args.controls else None
        if args.analysis=='failure-decomposition':
            failure_decomposition(base,out,controls)
        else:
            candidate_development(base,out,controls,
                args.swizzle_controls.resolve() if args.swizzle_controls else None,
                args.stage_events.resolve() if args.stage_events else None,
                args.common_history.resolve() if args.common_history else None,
                args.raster_pitch.resolve() if args.raster_pitch else None,
                args.stage_progress.resolve() if args.stage_progress else None,
                args.history_series.resolve() if args.history_series else None,
                args.source_pressure.resolve() if args.source_pressure else None,
                args.mirror_history.resolve() if args.mirror_history else None,
                args.base_shift.resolve() if args.base_shift else None,
                args.work_swap.resolve() if args.work_swap else None,
                args.producer_delay.resolve() if args.producer_delay else None,
                args.first_supply.resolve() if args.first_supply else None)
        return
    out.mkdir(parents=True,exist_ok=False)
    frozen=base/'20261009-V09-freeze-job738972-r2';target=base/'20261009-V09-heldout-job739011'
    cal=read(frozen/'frozen/calibration.json');old=read(base/'20261008-V08-job737322-v1/derived/calibration.json')
    cases,cal_rows,provenance=calibration_cases(base)
    assert len(cases)==51 and not any(k.startswith('v09d_') for k in cal_rows)
    params,notes=fit_v08_forms(cases,cal_rows,old['selection'])
    write(out/'baseline-calibration.json',dict(params=params,notes=notes,training=cases,input_sha256=provenance))
    # No heldout measurement is read until the baseline parameters are fixed above.
    actual={c['case']:c for c in read(target/'reanalysis/manager-results-v1/summary.json')['cases']}
    approved=read(frozen/'frozen/predictions.json')['predictions'];setups={s['case']:s['setup'] for s in read(frozen/'static_setup.json')}
    rows=[r for r in read(frozen/'cases.json') if r['set']=='heldout'];reports=[]
    identification=supply_identification(frozen,cal)
    for row in rows:
        a=actual[row['id']];setup=setups[row['id']];raw=model.predict_components(row,setup,cal)
        agg=aggregate_prediction(row,setup,params,cal)
        # An intermediate contrast changes only the output component. It explains
        # whether the full/baseline difference can be assigned to output rules.
        const=copy.deepcopy(cal);const.pop('output_rules',None)
        no_output=model.predict_components(row,setup,const)
        const_unsupported=check_prediction_support(row,setup,const,no_output,identification)
        def aggregate_supply(requests,ignored,frequency_ghz=None):
            p=params[row['config']];kt=row['k']//64
            values=[events.lmain(p,kt,q['j']==0)*(1+p.get('rho',{}).get(events.tile_class(row['config'],row,*q['coord']),0))/frequency_ghz for q in requests]
            return np.asarray(values),np.asarray([v>0 for v in values],dtype=bool)
        with patch.object(model.supply,'supply_predict',side_effect=aggregate_supply):
            simple_supply=model.predict_components(row,setup,cal)
        observed_f=a['observed_frequency_ghz']
        with patch.object(model.clock,'solve_source_envelope',side_effect=lambda point,clock,envelope:dict(frequency_ghz=observed_f,window_us=envelope(observed_f))):
            full_observed_f=model.predict_components(row,setup,cal)
            no_output_observed_f=model.predict_components(row,setup,const)
            with patch.object(model.supply,'supply_predict',side_effect=aggregate_supply):
                simple_supply_observed_f=model.predict_components(row,setup,cal)
        agg_observed_f=aggregate_prediction(row,setup,params,cal,observed_f)
        versions={}
        for name,p,diag in [('v09',raw,full_observed_f),('constant_output',no_output,no_output_observed_f),('aggregate_supply',simple_supply,simple_supply_observed_f),('v08_forms_new_clock',agg,agg_observed_f)]:
            numeric=p.get('plain_us') is not None
            versions[name]=dict(prediction=p,source_support=(not const_unsupported if name=='constant_output' else approved[row['id']]['status']=='predicted' if name=='v09' else bool(p.get('all_supply_supported',numeric))),relative_error=p['plain_us']/a['observed_us']-1 if numeric else None,
                observed_f_error=diag['plain_us']/a['observed_us']-1 if diag.get('plain_us') is not None else None)
        component_scores(target,row,versions)
        # Timelines are reproducible from the fixed parameters; keep the local
        # error summaries instead of four duplicate per-CTA traces in the report.
        for version in versions.values():version['prediction'].pop('cta_events',None)
        reports.append(dict(row=row,observed=a,frozen_status=approved[row['id']]['status'],versions=versions))
    common_cases=[r for r in reports if r['frozen_status']=='predicted' and all(v['relative_error'] is not None and v['source_support'] for v in r['versions'].values())]
    comparisons={}
    for name in reports[0]['versions']:
        comparisons[name]=dict(common_n=len(common_cases),supported_numeric=sum(r['versions'][name]['relative_error'] is not None and r['versions'][name]['source_support'] for r in reports),common=metrics([r['versions'][name]['relative_error'] for r in common_cases]),
            observed_f_common=metrics([r['versions'][name]['observed_f_error'] for r in common_cases]),
            component_errors={key:metrics([r['versions'][name]['components'][key] for r in common_cases if key in r['versions'][name]['components']]) for key in ('prefix','prefill','first_L','later_L','first_E','last_E','tail')},
            all_numeric_diagnostic=metrics([r['versions'][name]['relative_error'] for r in reports if r['versions'][name]['relative_error'] is not None]),
            invalid=[r['row']['id'] for r in reports if r['versions'][name]['relative_error'] is None],unsupported=[r['row']['id'] for r in reports if not r['versions'][name]['source_support']])
    write(out/'ablation.json',dict(scope='Post-V09 development comparison, not validation. V09 r2 params untouched; no V09 timings used in baseline fit.',
        baseline_evidence='50 pre-V09 supply cases plus the existing cfg_c K4096 high-scale R15 output case. V08 structural choices only; no old-card numeric parameter. Full V09 retains its original component-specific subsets.',
        common_protocol='Same software work, V09 clock/zero mixing, dual envelope and V09 fresh plain_transfer. V08 cycle parameters refitted from same-card dual records; old kappa is not reused.',
        shared_calibration_sha256=common.sha(frozen/'frozen/calibration.json'),
        support_scope='V09/constant_output use r2 LP support; aggregate variants report positive numerical candidates, not empirically validated coverage. Primary comparison keeps the original29 supported cases.',
        implementation_sha256={Path(m.__file__).name:common.sha(Path(m.__file__)) for m in [model,model.supply,model.clock,model.output_model,events,v08_fit,v06_fit]},
        analysis_sha256=common.sha(Path(__file__)),fit=notes,comparisons=comparisons,cases=reports))
    print(json.dumps(comparisons,indent=2),flush=True)


if __name__=='__main__':main()
