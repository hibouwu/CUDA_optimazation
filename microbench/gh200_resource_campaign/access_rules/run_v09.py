#!/usr/bin/env python3
"""V09: prepare, reference calibration + immutable freeze, then separately approved sampling.

Job 1 never launches a heldout GEMM. Setup only obtains static kernel/grid metadata.
Job 2 requires the two externally supplied approved SHA256 values and does not refit.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import random
import shutil
import statistics as st
import subprocess
import sys
import time

import run_r18
import run_v08
import v06_run as common
from analyze_r13_sm import select_attempts
from analyze_r18 import replay
import analyze_r13_supply as supply
import v09_fit
import v09_model

ROOT=Path(__file__).resolve().parent
VARIANTS=['plain','wide','stamped','dual']


def read(path):return json.loads(path.read_text())


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(root,heldout_path,inputs,cutlass):
    refs=read(inputs/'reference-cases.json');held=read(heldout_path)
    if any(r['set']!='calib' for r in refs) or any(r['set']!='heldout' for r in held):
        raise ValueError('reference/heldout sets must be explicit')
    if len({r['id'] for r in refs+held})!=len(refs+held):raise ValueError('duplicate case IDs')
    run_r18.prepare(root,cutlass,'v09',refs+held,dual_clock=True)
    for name in ('run_v08.py','v08_model.py','v09_model.py','v09_fit.py','run_v09.py',
                 'v06_fit.py','analyze_r13_supply.py','analyze_r13_sm.py',
                 'r09_input_source_clock.py','r09_v08_clock.py',
                 'supply_model.py','output_model.py','clock_model.py'):
        shutil.copy2(ROOT/name,root/'source'/name)
    shutil.copytree(inputs,root/'inputs')
    common.write_json(root/'source_hashes.json',{str(p.relative_to(root)):common.sha(p)
        for p in (root/'source').rglob('*') if p.is_file()})
    cfg=read(root/'run_config.json');cfg['source_commit']=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();write(root/'run_config.json',cfg)
    (root/'freeze.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 V06_BUILD_JOBS=3
python3 source/run_v08.py build --output "$PWD"
python3 source/run_v09.py calibrate-freeze --output "$PWD"
''')
    (root/'heldout.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
python3 source/run_v09.py heldout --output "$PWD" --approved-prediction-sha "$1" --approved-manifest-sha "$2"
''')
    for name in ('freeze.sh','heldout.sh'):(root/name).chmod(0o755)
    print('prepared',len(refs),'references and',len(held),'heldout conditions; no GPU work')


def sample(root,rows,keep_going):
    failures=[];disabled=set()
    for trial in range(10):
        group=rows[:];random.Random(20261009+trial).shuffle(group)
        for row in group:
            variants=VARIANTS[:];random.Random(f"{trial}-{row['id']}").shuffle(variants)
            for variant in variants:
                if (row['id'],variant) in disabled:continue
                try:run_v08.run_one(root,row,variant,trial)
                except (ValueError,subprocess.TimeoutExpired) as error:
                    failures.append(dict(case=row['id'],variant=variant,trial=trial,error=str(error)))
                    write(root/'sampling-failures.json',failures)
                    if not keep_going:raise
                    disabled.add((row['id'],variant))
        print('trial',trial,'complete',flush=True)
    return failures


def reference_metrics(observations):
    return {c['row']['id']:dict(
        plain_us=st.median(p['plain_event_ns'] for p in c['processes'])/1000,
        envelope_us=st.median(p['envelope_ns'] for p in c['processes'])/1000,
        effective_ghz=st.median(p['effective_ghz'] for p in c['processes']),
        event_extra_us=st.median(p['same_call_event_extra_ns'] for p in c['processes'])/1000) for c in observations['cases']}


def compare_references(actual,expected,policy):
    if set(actual)!=set(expected):raise ValueError('reference conditions differ')
    changes={case:{key:actual[case][key]/expected[case][key]-1 for key in limits}
             for case in actual for limits in [policy['reference_relative_limits']]}
    failures=[dict(case=case,metric=key,change=value) for case,values in changes.items()
              for key,value in values.items() if abs(value)>policy['reference_relative_limits'][key]]
    for case in actual:
        for key,limit in policy.get('reference_absolute_limits',{}).items():
            difference=actual[case][key]-expected[case][key]
            if abs(difference)>limit:failures.append(dict(case=case,metric=key,absolute_change=difference))
    return dict(changes=changes,failures=failures,compatible=not failures)


def fit_plain_transfer(observations):
    import numpy as np
    result={}
    for cfg in ('cfg_a','cfg_b','cfg_c'):
        cases=[c for c in observations['cases'] if c['row']['config']==cfg]
        x=np.array([st.median(p['envelope_ns'] for p in c['processes'])/1000 for c in cases])
        y=np.array([st.median(p['plain_event_ns'] for p in c['processes'])/1000 for c in cases])
        coef,_,rank,_=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)
        if rank!=2 or min(coef)<0:raise ValueError('invalid reference observer transfer: '+cfg)
        result[cfg]=dict(F_us=float(coef[0]),kappa=float(coef[1]),
            scope='Fresh reference plain event versus same-call dual envelope; no supply-prediction residual is fitted.')
    return result


def supply_identification(root,model):
    with gzip.open(root/'inputs/supply-training.json.gz','rt') as stream:training=json.load(stream)
    return {key:{cfg:supply.supply_identification_bounds(training[phase][cfg],model[key][cfg])
                 for cfg in ('cfg_a','cfg_b','cfg_c')}
            for phase,key in [('first','first_supply'),('later','supply')]}


def check_prediction_support(row,setup,model,prediction,identification):
    """Check equivalent parameters within the fitted calibration active pattern."""
    requests=supply.supply_request_rows(row,setup);unsupported=[]
    for phase,key in [('first','first_supply'),('later','supply')]:
        selected=[r for r in requests if (r['j']==0)==(phase=='first')]
        bounds=supply.supply_predict_bounds(selected,model[key][row['config']],
            prediction['frequency_ghz'],identification[key][row['config']])
        for request,bound in zip(selected,bounds):
            if bound['status']!='identified prediction':
                unsupported.append(dict(cta=request['cta'],j=request['j'],reason=bound['status']))
    return unsupported


def freeze_predictions(root,model,metrics,policy):
    rows=read(root/'cases.json');held=[r for r in rows if r['set']=='heldout']
    if any((root/'samples'/r['id']).exists() for r in held):raise ValueError('heldout samples predate freeze')
    setups={s['case']:s['setup'] for s in read(root/'static_setup.json')}
    predictions={r['id']:v09_model.predict_components(r,setups[r['id']],model) for r in held}
    identification=supply_identification(root,model)
    for row in held:
        p=predictions[row['id']]
        unsupported=check_prediction_support(row,setups[row['id']],model,p,identification)
        if unsupported:
            p['all_supply_supported']=False;p['unsupported_windows']=unsupported
    for case,p in list(predictions.items()):
        if not p['all_supply_supported']:
            predictions[case]=dict(status='unsupported',plain_us=None,
                unsupported_windows=p['unsupported_windows'],
                reason='Supply request is unsupported or equivalent fitted parameters give different service predictions; no numerical prediction is frozen for this condition.')
        else:
            if p['plain_us'] is None or p['plain_us']<=0:raise ValueError('plain prediction missing')
            p['status']='predicted'
    frozen=root/'frozen';frozen.mkdir(exist_ok=False)
    shutil.copy2(root/'environment.json',root/'freeze-environment.json')
    shutil.copy2(root/'static_setup.json',root/'freeze-setup.json')
    write(frozen/'calibration.json',model)
    result=dict(status='frozen',frozen_unix_ns=time.time_ns(),freeze_host=os.uname().nodename,gpu=read(root/'environment.json')['gpu'],
        predictions=predictions,reference_metrics=metrics,policy=policy,
        protocol='Same GPU across two allocations. Heldout GEMMs require user approval of these exact prediction and manifest SHA256 values.')
    write(frozen/'predictions.json',result)
    paths=['cases.json','run_config.json','source_hashes.json','build/binary_hashes.json','build/sass_hashes.json',
           'freeze-environment.json','freeze-setup.json','freeze.sh','heldout.sh','frozen/calibration.json','frozen/predictions.json']
    paths += [str(p.relative_to(root)) for p in (root/'inputs').rglob('*') if p.is_file()]
    paths += [name for name in ('python-runtime.json','build/nvcc-version.txt',
                               'build/commands.json','build/resources.json','freeze-revision.json',
                               'freeze-runtime.json') if (root/name).exists()]
    write(frozen/'manifest.json',dict(source_commit=read(root/'run_config.json')['source_commit'],
        files={name:common.sha(root/name) for name in sorted(paths)}))
    for path in frozen.iterdir():path.chmod(0o444)
    receipt=dict(prediction_sha256=common.sha(frozen/'predictions.json'),manifest_sha256=common.sha(frozen/'manifest.json'))
    write(root/'freeze-receipt.json',receipt)
    print(json.dumps(receipt),flush=True)
    return receipt


def verify_approved(root,prediction_sha,manifest_sha):
    frozen=root/'frozen'
    if common.sha(frozen/'predictions.json')!=prediction_sha or common.sha(frozen/'manifest.json')!=manifest_sha:
        raise ValueError('approved SHA256 mismatch')
    for name,digest in read(frozen/'manifest.json')['files'].items():
        if common.sha(root/name)!=digest:raise ValueError('frozen input changed: '+name)
    if any(p.stat().st_mode&0o222 for p in frozen.iterdir()):raise ValueError('frozen files must be read-only')
    common.verify(root)
    for name,digest in read(root/'build/sass_hashes.json').items():
        if common.sha(root/'build'/name)!=digest:raise ValueError('SASS changed')
    result=read(frozen/'predictions.json')
    if result['status']!='frozen' or result['frozen_unix_ns']>=time.time_ns():raise ValueError('invalid freeze time')
    return result


def score(root,predictions,policy,failures):
    cases=[]
    first_sample_ns=read(root/'prediction_binding.json')['first_sample_unix_ns']
    for row in read(root/'cases.json'):
        if row['set']!='heldout':continue
        selected,bad,_=select_attempts(root,row['id'],VARIANTS)
        times={v:[] for v in VARIANTS};details=[]
        errors=[f for f in failures if f['case']==row['id']]
        for item in selected:
            rec=item['record']
            if rec['host_start_ns']<first_sample_ns:raise ValueError('target process predates the frozen binding')
            try:observed=replay(root,rec,row)
            except ValueError as error:
                errors.append(dict(variant=rec['variant'],trial=rec['trial'],error=str(error)));continue
            times[rec['variant']].append(observed['elapsed_us'])
            if rec['variant']=='dual' and predictions[row['id']]['status']=='predicted':
                p=predictions[row['id']];ctas=observed['ctas'];last=max(range(len(ctas)),key=lambda i:ctas[i]['end_ns'])
                chosen=p['critical_cta'];relative={key:[] for key in ('prefix','prefill','first_L','later_L','last_E','tail')}
                for c,predicted in zip(ctas,p['cta_events']):
                    if not c['tiles_ns']:continue
                    actual=c['tiles_ns'];pe=predicted['events']
                    for key,a,b in [('prefix',actual[0][0]-c['entry_ns'],predicted['supply']),
                                    ('prefill',actual[0][0]-c['prod_ns'],predicted['prefill']),
                                    ('first_L',actual[0][1]-actual[0][0],pe[0][1]-pe[0][0]),
                                    ('last_E',actual[-1][3]-actual[-1][2],pe[-1][3]-pe[-1][2]),
                                    ('tail',c['end_ns']-max(t[3] for t in actual),predicted['tail'])]:
                        if a>0:relative[key].append(b/a-1)
                    relative['later_L'] += [(b[1]-b[0])/(a[1]-a[0])-1 for a,b in zip(actual[1:],pe[1:]) if a[1]>a[0]]
                details.append(dict(trial=rec['trial'],component_median_errors={k:st.median(v) if v else None for k,v in relative.items()},
                    selected_cta_shortfall=(ctas[chosen]['end_ns']-ctas[chosen]['entry_ns'])/(ctas[last]['end_ns']-ctas[last]['entry_ns'])-1,
                    last_in_exact_predicted_ties=last in p['critical_candidates'],predicted_tie_count=len(p['critical_candidates'])))
        plain_ok=len(times['plain'])==10 and not any(e.get('variant')=='plain' for e in errors)
        target=st.median(times['plain']) if plain_ok else None
        predicted=predictions[row['id']]['status']=='predicted'
        cases.append(dict(case=row['id'],plain_valid=plain_ok,prediction_supported=predicted,observed_plain_us=target,
            plain_process_cv=st.pstdev(times['plain'])/st.fmean(times['plain']) if times['plain'] else None,
            predicted_plain_us=predictions[row['id']]['plain_us'],
            relative_error=predictions[row['id']]['plain_us']/target-1 if plain_ok and predicted else None,
            processes={k:len(v) for k,v in times.items()},failed_attempts=bad,errors=errors,components=details))
    valid=[c for c in cases if c['plain_valid'] and c['prediction_supported']]
    metrics=supply.metrics([c['relative_error'] for c in valid]) if valid else None
    passed=bool(len(valid)==len(cases) and metrics['median_abs_pct']<=policy['median_percent'] and metrics['max_abs_pct']<=policy['maximum_percent'])
    write(root/'score.json',dict(cases=cases,total_conditions=len(cases),
        valid_plain_conditions=sum(c['plain_valid'] for c in cases),scored_predictions=len(valid),
        plain_metrics=metrics,complete_time_passed=passed,
        scope='No failed condition is removed from the declared set. Component errors, exact ties and representative-CTA shortfall remain separate from total time.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['prepare','calibrate-freeze','heldout'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--heldout',type=Path);parser.add_argument('--inputs',type=Path);parser.add_argument('--cutlass-root',type=Path)
    parser.add_argument('--approved-prediction-sha');parser.add_argument('--approved-manifest-sha')
    args=parser.parse_args();root=args.output.resolve()
    if args.step=='prepare':
        prepare(root,args.heldout.resolve(),args.inputs.resolve(),args.cutlass_root.resolve());return
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Slurm allocation required')
    rows=read(root/'cases.json');refs=[r for r in rows if r['set']=='calib']
    if args.step=='calibrate-freeze':
        common.verify(root);policy=read(root/'inputs/policy.json')
        if run_v08.identity()['gpu']!=policy['gpu']:raise ValueError('wrong reference GPU/environment')
        run_v08.setup(root)
        for row in refs:run_v08.run_one(root,dict(row,id='check_'+row['id']),'plain',0)
        sample(root,refs,False)
        observed=v09_fit.event_observations(root);metrics=reference_metrics(observed)
        bridge=compare_references(metrics,read(root/'inputs/reference-metrics.json'),policy)
        write(root/'reference-check.json',bridge)
        if not bridge['compatible']:raise ValueError('reference drift; no heldout sampled or freeze produced')
        model=read(root/'inputs/base-model.json')
        with gzip.open(root/'inputs/supply-training.json.gz','rt') as stream:training=json.load(stream)
        for cfg in ('cfg_a','cfg_b','cfg_c'):
            for phase,key in [('first','first_supply'),('later','supply')]:
                model[key][cfg]=supply.fit_supply(training[phase][cfg],unit='ns',calibration_clock='observed',phase=phase)
        if any(not model[key][cfg]['optimizer_success'] for key in ('first_supply','supply') for cfg in ('cfg_a','cfg_b','cfg_c')):
            raise ValueError('supply fit did not converge; freeze not produced')
        model['plain_transfer']=fit_plain_transfer(observed)
        freeze_predictions(root,model,metrics,policy)
    else:
        if not args.approved_prediction_sha or not args.approved_manifest_sha:parser.error('the two externally approved hashes are required')
        frozen=verify_approved(root,args.approved_prediction_sha,args.approved_manifest_sha)
        if run_v08.identity()['gpu']!=frozen['gpu']:raise ValueError('wrong reference GPU/environment')
        if (root/'samples').exists():raise ValueError('heldout job requires a fresh directory without earlier samples')
        run_v08.setup(root)
        if read(root/'static_setup.json')!=read(root/'freeze-setup.json'):raise ValueError('static kernel/grid/resource identity changed')
        sample(root,refs,False)
        bridge=compare_references(reference_metrics(v09_fit.event_observations(root)),frozen['reference_metrics'],frozen['policy'])
        write(root/'reference-check.json',bridge)
        if not bridge['compatible']:raise ValueError('reference drift; heldout not started')
        write(root/'prediction_binding.json',dict(prediction_sha256=args.approved_prediction_sha,manifest_sha256=args.approved_manifest_sha,first_sample_unix_ns=time.time_ns(),environment=run_v08.identity()))
        failures=sample(root,[r for r in rows if r['set']=='heldout'],True)
        score(root,frozen['predictions'],frozen['policy'],failures)


if __name__=='__main__':main()
