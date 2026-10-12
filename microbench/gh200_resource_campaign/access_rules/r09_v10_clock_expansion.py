#!/usr/bin/env python3
"""Bounded V10 clock training expansion; no GPU work or new clock formula."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
from unittest.mock import patch


PLAN_SHA = '3f1f34a0eec29fda257fce710bda0d84228a140b8677543a4fb76f65122cdd32'
REFERENCE_GPU = 'GPU-43269fbc-449d-3e0f-908a-9c81229546d3'
OUTSIDE_GROUPS = {'G3_A16', 'G5_B32', 'G5_B80', 'G5_B112', 'G3_stage4'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def snapshot_code(public_code, target):
    """Snapshot only local import dependencies, preserving their source identity."""
    target.mkdir()
    own = Path(__file__).resolve().parent
    pending = ['v09_model', 'analyze_v09_ablation', 'analyze_v09_gate_audit', 'r09_input_source_clock',
               'r09_v10_clock_expansion']
    sources = {}
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        root = own if name in ('r09_input_source_clock', 'r09_v10_clock_expansion') else public_code
        path = root / (name + '.py')
        if not path.is_file():
            continue
        data = path.read_bytes()
        sources[name] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
        (target / path.name).write_bytes(data)
        for node in ast.parse(data).body:
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module)
    return sources


def abs_metrics(errors):
    values = [abs(v) * 100 for v in errors]
    return dict(n=len(values), median_abs_pct=st.median(values), max_abs_pct=max(values))


def adoption(rows, candidate):
    comparable = [r for r in rows if r['frozen_status'] == 'predicted']
    targets = [r for r in comparable if r['group'] == 'G6_base']
    others = [r for r in comparable if r['group'] != 'G6_base']
    if len(comparable) != 29 or len(targets) != 3:
        raise ValueError('fixed 29-condition comparison and three G6 targets required')
    unavailable = [r['case'] for r in comparable if r['versions'][candidate].get('error')]
    if unavailable:
        return dict(adopt=False, unavailable=unavailable)
    regressions = []
    crossing = []
    for r in comparable:
        old = abs(r['versions']['r2']['plain_error']) * 100
        new = abs(r['versions'][candidate]['plain_error']) * 100
        if new - old > 2 + 1e-10:
            regressions.append(dict(case=r['case'], increased_pp=new-old))
        if r['in_proposed_scope'] and old <= 10 and new > 10:
            crossing.append(r['case'])
    target_checks = []
    for r in targets:
        a, b = r['versions']['r2'], r['versions'][candidate]
        target_checks.append(dict(case=r['case'],
            frequency_improved=abs(b['frequency_error']) < abs(a['frequency_error']),
            complete_time_improved=abs(b['plain_error']) < abs(a['plain_error'])))
    baseline = abs_metrics(r['versions']['r2']['plain_error'] for r in others)
    changed = abs_metrics(r['versions'][candidate]['plain_error'] for r in others)
    median_ok = changed['median_abs_pct'] <= baseline['median_abs_pct'] + 1e-10
    max_ok = changed['max_abs_pct'] <= baseline['max_abs_pct'] + 1e-10
    passed = (all(t['frequency_improved'] and t['complete_time_improved'] for t in target_checks)
              and median_ok and max_ok and not regressions and not crossing)
    return dict(adopt=passed, targets=target_checks, non_target_baseline=baseline,
                non_target_candidate=changed, non_target_median_not_increased=median_ok,
                non_target_max_not_increased=max_ok, regression_over_2pp=regressions,
                supported_10pct_crossings=crossing, unavailable=[])


def analyze(base, public_code, output):
    import numpy as np
    if output.exists():
        raise ValueError('output exists; old reports must not be overwritten')
    output.mkdir(parents=True)
    sources = snapshot_code(public_code, output/'source')
    plan_path = public_code/'PLAN.md'
    if not plan_path.exists():
        plan_path = base.parents[2]/'Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/PLAN.md'
    if sha(plan_path)!=PLAN_SHA:
        raise ValueError('bounded task PLAN identity changed')
    (output/'source/PLAN.md').write_bytes(plan_path.read_bytes())
    sys.path.insert(0, str(output/'source'))
    sys.path.insert(1, str(public_code))
    import r09_input_source_clock as fit
    import v09_model as model
    import analyze_v09_ablation as ablation
    from analyze_v09_gate_audit import process_audit, COMPONENTS

    inputs = {str(plan_path):PLAN_SHA}
    def bound(path):
        inputs[str(path)] = sha(path)
        return read(path)

    roots = dict(ends21=base/'20261009-R09-input-clock-calibration-job738376',
                 ends9=base/'20261009-R09-equal-work-job738459',
                 dual30=base/'20261009-V09-heldout-job739011',
                 frozen=base/'20261009-V09-freeze-job738972-r2')
    original = bound(roots['ends21']/'reanalysis/C-20261009-source-clock-dev-v2/source-clock.json')
    for root in roots.values():
        env = bound(root/'environment.json')
        if env['gpu'].split(',')[0] != REFERENCE_GPU:
            raise ValueError('all fitting/diagnostic records must remain on the reference GPU')
    cal = bound(roots['frozen']/'frozen/calibration.json')
    frozen = bound(roots['frozen']/'frozen/predictions.json')['predictions']
    scores = {s['case']:s for s in bound(roots['dual30']/'score.json')['cases']}
    failure = bound(roots['dual30']/'reanalysis/failure-decomposition-v1/A-clock-failure.json')
    processes = {}
    for p in failure['processes']:
        processes.setdefault(p['case'], []).append(p)
    rows = [r for r in bound(roots['frozen']/'cases.json') if r['set']=='heldout']
    setups = {s['case']:s['setup'] for s in bound(roots['frozen']/'static_setup.json')}
    target_rows = {r['id']:r for r in bound(roots['dual30']/'cases.json')}
    target_setups = {s['case']:s['setup'] for s in bound(roots['dual30']/'static_setup.json')}
    if len(rows)!=30 or len(processes)!=30 or any(len(v)!=10 for v in processes.values()):
        raise ValueError('all 30 conditions and their ten dual labels are required')
    for r in rows:
        if r != target_rows[r['id']] or setups[r['id']] != target_setups[r['id']]:
            raise ValueError('target row/setup differs from frozen input')
    points = []
    for name, expected in (('ends21',21),('ends9',9)):
        rs = bound(roots[name]/'cases.json')
        ss = {s['case']:s['setup'] for s in bound(roots[name]/'static_setup.json')}
        summary = bound(roots[name]/'analysis/summary.json')['cases']
        rs = [r for r in rs if name=='ends21' or r['k']==4096]
        if len(rs)!=expected:
            raise ValueError('ends training matrix changed')
        for r in rs:
            feat = model.events.features(r, ss[r['id']]['grid'])
            ledger = model.supply.supply_work_ledger(r, ss[r['id']])
            tm,tn,tk = model.events.base.CONFIGS[r['config']]['tile']
            obs = summary[r['id']]
            if (ledger['schema']!='gh200-supply-work-ledger/1'
                    or len(obs['per_process_ends'])!=10):
                raise ValueError('source ledger or ends observation count changed')
            if obs['ghz_ends'] != st.median(p['ghz'] for p in obs['per_process_ends']):
                raise ValueError('ends label aggregation changed')
            points.append(dict(case=r['id'], config=r['config'], m=r['m'],n=r['n'],k=r['k'],
                input_mode=r['input_mode'], protocol='ends', batch=name, weight=1.,
                zero_fraction=0., observed_window_us=obs['window_ends'], observed_ghz=obs['ghz_ends'],
                tensor_mean_cycles=ledger['totals']['output_tiles']*feat['kt']*2*tm*tn*tk/4096/132,
                source_mean_kib=ledger['totals']['source_payload_KiB']/132))
    for r in rows:
        work = model.events.scheduled_work(r['config'],r['m'],r['n'],setups[r['id']]['grid'],r['swizzle'])
        ledger = model.supply.supply_work_ledger(r,setups[r['id']])
        tm,tn,tk = model.events.base.CONFIGS[r['config']]['tile']
        z = model.zero_product_fraction(r,work) if cal['zero_activity_mixing'] else 0.
        weights = [float(r['input_mode']==m)*(1-z)+float(m=='zero')*z for m in fit.MODES]
        ps = processes[r['id']]
        points.append(dict(case=r['id'],config=r['config'],group=r['group'],m=r['m'],n=r['n'],k=r['k'],
            input_mode=r['input_mode'],protocol='dual',batch='dual30',weight=1.,zero_fraction=z,
            mode_weights=weights,
            observed_window_us=st.median(p['observed_window_us'] for p in ps),
            observed_ghz=st.median(p['observed_frequency_ghz'] for p in ps),
            tensor_mean_cycles=ledger['totals']['output_tiles']*(r['k']//tk)*2*tm*tn*tk/4096/132,
            source_mean_kib=ledger['totals']['source_payload_KiB']/132))
    cap = original['model']['constraints']['a_max_ghz']
    datasets = dict(ends21=points[:21],ends30=points[:30],dual30=points[30:],pooled60=points)
    models = {name:fit.fit_source_clock(ps,cap) for name,ps in datasets.items()}
    for key in ('tau_us','coefficients'):
        a,b = models['ends21'][key],original['model'][key]
        if isinstance(a,list):
            if any(not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10) for x,y in zip(a,b)):
                raise ValueError('original ends21 fit no longer reproduces its fixed baseline')
        elif not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('original ends21 tau no longer reproduces its fixed baseline')
    conditional = {}
    mixing_checked = 0
    for name,cm in models.items():
        conditional[name] = {}
        for protocol in ('ends','dual'):
            errors = []
            nonpositive = []
            for p in points:
                if p['protocol']!=protocol:
                    continue
                predicted = float(fit.source_design([p],cm['tau_us'])[0] @ cm['coefficients'])
                mixed = fit.mixed_failure_clock(cm,p['input_mode'],p['zero_fraction'])
                runtime = model.clock.source_frequency(p,p['observed_window_us'],mixed)['frequency_ghz']
                if not math.isclose(predicted,runtime,rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('fitting mode weights differ from frozen r2 runtime mixing')
                mixing_checked += 1
                errors.append(predicted/p['observed_ghz']-1)
                if predicted<=0:
                    nonpositive.append(dict(case=p['case'],frequency_ghz=predicted))
            conditional[name][protocol] = dict(abs_metrics(errors),nonpositive_conditional_values=nonpositive)
    dual_unmixed = [{k:v for k,v in p.items() if k!='mode_weights'} for p in datasets['dual30']]
    dual_identification = dict(direct_zero_conditions=0,
        unmixed_design_rank=int(np.linalg.matrix_rank(fit.source_design(dual_unmixed,models['dual30']['tau_us']))),
        conditional_mixed_design_rank=models['dual30']['rank'],
        status='Conditional fit uses the original r2 static partial-zero mixing; no direct dual all-zero calibration.',
        limitation='Conditional matrix rank is not an independent all-zero observation certification or an ends/dual correction. Tau profiles are sensitivity ranges, not confidence intervals.')
    folds = []
    fold_models = {}
    for shape in sorted({(p['m'],p['n']) for p in points}):
        train = [p for p in points if (p['m'],p['n'])!=shape]
        held = [p for p in points if (p['m'],p['n'])==shape]
        fm = fit.fit_source_clock(train,cap)
        fold_models[shape] = fm
        folds.append(dict(shape=list(shape),model=fm,train_cases=[p['case'] for p in train],
            held_cases=[p['case'] for p in held],train_k=sorted({p['k'] for p in train}),
            held_k=sorted({p['k'] for p in held}),
            long_k_extrapolation=any(p['k']>max(t['k'] for t in train) for p in held),
            scope='Clock component leave MxN shape out; all r2 event/supply/plain parameters remain frozen.'))
    comparisons = []
    predictions = []
    for r in rows:
        cid = r['id']
        actual = next(p for p in points[30:] if p['case']==cid)
        versions = {}
        choices = dict(r2=cal['clock'],ends30=models['ends30'],dual30=models['dual30'],
                       pooled60=models['pooled60'],clock_shape_fold=fold_models[(r['m'],r['n'])])
        for name,cm in choices.items():
            cc = dict(cal,clock=cm)
            try:
                pred = model.predict_components(r,setups[cid],cc)
                versions[name] = dict(prediction=pred, plain_error=pred['plain_us']/scores[cid]['observed_plain_us']-1,
                    frequency_error=pred['frequency_ghz']/actual['observed_ghz']-1,
                    envelope_error=pred['window_us']/actual['observed_window_us']-1)
            except (ValueError,RuntimeError) as error:
                versions[name] = dict(error=str(error))
        if frozen[cid]['status']=='predicted':
            for key in ('plain_us','frequency_ghz','window_us'):
                if not math.isclose(versions['r2']['prediction'][key],frozen[cid][key],rel_tol=1e-9,abs_tol=1e-9):
                    raise ValueError('current default model changed frozen r2 prediction: '+cid)
        usable = {n:v for n,v in versions.items() if 'prediction' in v}
        observed_calls = []
        original_replay = ablation.replay
        def capture_replay(root, record, row):
            observed = original_replay(root, record, row)
            cs = observed['ctas']
            active = [c for c in cs if c['tiles']]
            most = max(len(c['tiles']) for c in active)
            frequency = st.median((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns'])
                                  for c in active if len(c['tiles'])==most)
            window = (max(c['end_ns'] for c in cs)-min(c['entry_ns'] for c in cs))/1000
            label = next(p for p in processes[cid] if p['trial']==record['trial'])
            if (not math.isclose(frequency,label['observed_frequency_ghz'],rel_tol=1e-12)
                    or window!=label['observed_window_us']):
                raise ValueError('saved dual frequency/window label differs from raw replay')
            inputs[str(root/record['raw'])] = record['raw_sha256']
            observed_calls.append(dict(ctas=cs, max_cycles=max(c['end_c']-c['entry_c'] for c in active)))
            return observed
        with patch.object(ablation,'replay',side_effect=capture_replay):
            ablation.component_scores(roots['dual30'],r,usable)
        observed_max_c = st.median(o['max_cycles'] for o in observed_calls)
        for v in usable.values():
            pred = v['prediction']
            v['converted_envelope_cycle_proxy_error'] = pred['window_us']*1000*pred['frequency_ghz']/observed_max_c-1
            shortfalls = []
            for o in observed_calls:
                cs=o['ctas']; last=max(range(len(cs)),key=lambda i:cs[i]['end_ns'])
                chosen=pred['critical_cta']
                shortfalls.append((cs[chosen]['end_ns']-cs[chosen]['entry_ns'])/
                                  (cs[last]['end_ns']-cs[last]['entry_ns'])-1)
            v['representative_cta_duration_shortfall'] = st.median(shortfalls)
            audits = [process_audit(o['ctas'],pred) for o in observed_calls]
            v['component_absolute_errors'] = {}
            for key in COMPONENTS:
                valid = [a['components'][key] for a in audits if a['components'][key]['windows']]
                v['component_absolute_errors'][key] = dict(
                    valid_processes=len(valid),invalid_windows=sum(a['components'][key]['invalid_windows'] for a in audits),
                    median_of_process_median_abs_pct=st.median(a['median_abs_pct'] for a in valid) if valid else None,
                    maximum_abs_pct=max(a['max_abs_pct'] for a in valid) if valid else None)
            v['critical_absolute_errors'] = dict(
                exact_last_hit_rate=st.fmean(a['critical']['chosen_is_exact_last'] for a in audits),
                candidate_set_hit_rate=st.fmean(a['critical']['frozen_set_intersects_exact_last'] for a in audits),
                median_end_regret_ns=st.median(a['critical']['end_regret_ns'] for a in audits),
                maximum_end_regret_ns=max(a['critical']['end_regret_ns'] for a in audits),
                median_end_regret_percent=st.median(a['critical']['end_regret_percent'] for a in audits))
        outside = r['group'] in OUTSIDE_GROUPS
        comparisons.append(dict(case=cid,config=r['config'],group=r['group'],shape=[r['m'],r['n']],
            k=r['k'],frozen_status=frozen[cid]['status'],in_proposed_scope=not outside and frozen[cid]['status']=='predicted',
            outside_scope=outside,observed_plain_us=scores[cid]['observed_plain_us'],versions=versions))
        predictions.append(dict(case=cid,row=r,setup=setups[cid],
            formal_status='diagnostic_only' if outside or frozen[cid]['status']!='predicted' else 'candidate',
            prediction=versions['pooled60'].get('prediction'),error=versions['pooled60'].get('error')))
    if sum(r['outside_scope'] for r in comparisons)!=7:
        raise ValueError('seven out-of-scope diagnostics must remain in the fixed table')
    for path,digest in inputs.items():
        if sha(Path(path))!=digest:
            raise ValueError('input changed during fitting: '+path)
    for module in list(sys.modules.values()):
        file = getattr(module,'__file__',None)
        if not file:
            continue
        path = Path(file).resolve()
        if path.suffix!='.py' or path.parent not in (public_code,Path(__file__).resolve().parent):
            continue
        name=path.stem
        if name not in sources:
            sources[name]=dict(path=str(path),sha256=sha(path))
            (output/'source'/path.name).write_bytes(path.read_bytes())
    result = dict(version='R09-V10-clock-expansion/1',formula='unchanged source_frequency, seven coefficients + shared tau',
        weights='one equal-weight median frequency/window label per condition; minimize absolute GHz squared residuals',
        label='Per process median native-cycle/ns ratio over maximum-work CTAs; condition median over ten processes. W is the condition median global entry-to-end envelope.',
        gpu=REFERENCE_GPU,models=models,conditional_protocol_scores=conditional,folds=folds,
        runtime_zero_mixing_checks=mixing_checked,dual_identification=dual_identification,
        adoption={n:adoption(comparisons,n) for n in ('ends30','dual30','pooled60','clock_shape_fold')},
        comparison_conditions=29,declared_conditions=30,outside_scope_conditions=7,
        protocol_limit='Separate fits confound protocol and geometry/K distributions; no same-card matched ends/dual pairs. Cross-protocol errors do not identify observation bias.',
        source_work='gh200-supply-work-ledger/1, nominal source payload including fill; unchanged from r2',
        cycle_diagnostic='Predicted envelope_us * 1000 * frequency_GHz versus median maximum same-SM CTA cycle span; scopes differ, so this is a converted-envelope proxy, not measured physical cycles.',
        component_diagnostic='Take absolute error within each process/window before aggregation; retain invalid/absent windows. Legacy signed component medians are reported separately and never replace absolute errors.',
        plan_sha256=PLAN_SHA,sources=sources,inputs=inputs,
        gpu_jobs=0,new_holdout=False,complete_model_leave_shape_out=False)
    # The exported clock contains model parameters only; no scoring labels.
    model_fields = ('tau_us','coefficients','coefficient_order','constraints','training_window_us')
    fixed_clock = {key:models['pooled60'][key] for key in model_fields}
    exported_folds = [dict(heldout_shape=f['shape'],
        clock={key:f['model'][key] for key in model_fields},status='fitted_development_model',
        reason=None,long_k_extrapolation=f['long_k_extrapolation']) for f in folds]
    write(output/'clock-candidate.json',dict(clock=fixed_clock,protocol='ends30 + dual30, equal condition weights',
        adopted=result['adoption']['pooled60']['adopt'] and result['adoption']['clock_shape_fold']['adopt'],
        folds=exported_folds,protocol_identification=dual_identification,
        folds_file='analysis.json',provenance_file='manifest.json'))
    write(output/'training-points.json',points)
    write(output/'free-predictions.json',predictions)
    write(output/'comparisons.json',comparisons)
    write(output/'analysis.json',result)
    write(output/'manifest.json',dict(plan_sha256=PLAN_SHA,inputs=inputs,sources=sources,
        outputs={p.name:sha(p) for p in sorted(output.glob('*.json'))}))
    print(json.dumps(dict(output=str(output),adoption=result['adoption'],
        pooled_clock=fixed_clock,fold_count=len(folds)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--public-code',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.base.resolve(),args.public_code.resolve(),args.output.resolve())
