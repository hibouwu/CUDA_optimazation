#!/usr/bin/env python3
"""CPU development ablation: same-card V08 forms + V09 clock versus V09 r2.

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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();base=args.base.resolve();out=args.output.resolve()
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
