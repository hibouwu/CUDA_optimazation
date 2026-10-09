#!/usr/bin/env python3
"""V09 development composition. Currently calibrated only for cfg_b on GPU432.

Natural-ns event constants and conditional supply enter the existing CTA recurrence;
the R09 work/source clock closes the compute floor self-consistently. This module
does not read target timings, choose training cases, or freeze a validation.
"""
import math

import analyze_r13_supply as supply
import r09_input_source_clock as clock
import v08_model as events


def case_recurrence(row, setup, calibration):
    if row['config'] != 'cfg_b':
        raise ValueError('cfg_a/c event and supply calibration is not yet available')
    if math.prod(setup['grid']) != 132 or setup['stages'] != 6:
        raise ValueError('current supply calibration covers the full 132-SM stage-6 grid')
    work = events.scheduled_work('cfg_b', row['m'], row['n'], setup['grid'], row['swizzle'])
    requests = supply.fill_cta_features(row, setup)
    kt = row['k'] // 64
    ntiles = sum(map(len, work))
    # Same nominal scheduled source definition as R09 training. OOB/pitch energy
    # changes have not been separately calibrated; retain this transfer assumption.
    point = dict(input_mode=row.get('input_mode', 'dyadic'),
                 tensor_mean_cycles=ntiles * kt * 512 / 132,
                 source_mean_kib=ntiles * kt * 32 / 132)
    params = calibration['events_ns']
    fit = calibration['supply_ns']
    intercept = fit['window_ns']
    coefficients = [fit[key] for key in supply.FILL_FIELDS]
    costs = [[sum(x*q for x,q in zip(features, coefficients)) for features in cta]
             for cta in requests]

    def durations(frequency):
        return [events.cta_cycles(params, 'pingpong', len(cta), kt,
                  mainloops=[intercept + kt * max(512/frequency, service) for service in cta])
                for cta in costs]

    return point, work, durations


def predict(row, setup, calibration):
    point, work, durations = case_recurrence(row, setup, calibration)

    def envelope_us(frequency):
        return max(durations(frequency)) / 1000

    solved = clock.solve_source_envelope(point, calibration['clock'], envelope_us)
    cta_ns = durations(solved['frequency_ghz'])
    critical = max(range(len(cta_ns)), key=cta_ns.__getitem__)
    return dict(**solved, event_us=solved['window_us'] + calibration['fixed_event_extra_ns']/1000,
                critical_cta=critical, cta_duration_ns=cta_ns,
                critical_work=work[critical], static_clock_work=point)


def development(base, output):
    """Replay the declared 13-condition same-card development suite, without fitting."""
    import hashlib
    import json
    import statistics
    from pathlib import Path

    def load(path):
        return json.loads(path.read_text())

    source_run = base / '20261009-R10-b-coverage-job738203'
    supply_path = source_run / 'reanalysis/A-20261009-valid-fill-v4/summary.json'
    events_path = source_run / 'reanalysis/A-20261009-cta-composition-v3/summary.json'
    clock_path = base / '20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json'
    fitted_supply, fitted_events, fitted_clock = map(load, (supply_path, events_path, clock_path))
    if fitted_clock['environment']['gpu'] != fitted_events['gpu'] or fitted_supply['gpu'] != fitted_events['gpu']:
        raise ValueError('component calibration GPU identities differ')
    calibration = dict(events_ns=fitted_events['shared_parameters']['ns'],
                       supply_ns=fitted_supply['fits']['ns']['parameters'],
                       fixed_event_extra_ns=fitted_events['fixed_event_extra_ns'],
                       clock=fitted_clock['model'])
    roots = [source_run, base/'20261009-R18-input-map-job738296',
             base/'20261009-R18-input-map-n-job738307']
    reports = []
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (supply_path, events_path, clock_path)}
    for root in roots:
        setups = {s['case']: s['setup'] for s in load(root/'static_setup.json')}
        for filename in ('cases.json', 'static_setup.json', 'environment.json'):
            path = root/filename
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if load(root/'environment.json')['gpu'] != fitted_events['gpu']:
            raise ValueError('development comparison must use the fitted GPU')
        for row in load(root/'cases.json'):
            predicted = predict(row, setups[row['id']], calibration)
            observed = {}
            cta_diagnostics = []
            for variant in ('plain', 'dual'):
                paths = sorted((root/'samples'/row['id']).glob(variant+'-*.json'))
                records = [load(p) for p in paths]
                if len(records) != 10 or any(r['returncode'] for r in records):
                    raise ValueError('incomplete declared development records')
                observed[variant+'_us'] = statistics.median(r['elapsed_us'] for r in records)
                if variant == 'dual':
                    for record in records:
                        ctas = events.replay(root, record, row)['ctas']
                        actual_last = max(range(len(ctas)), key=lambda i: ctas[i]['end_ns'])
                        selected = predicted['critical_cta']
                        max_tiles = max(len(c['tiles']) for c in ctas)
                        cta_diagnostics.append(dict(
                            chosen_is_last=selected == actual_last,
                            chosen_exit_gap_ns=ctas[actual_last]['end_ns']-ctas[selected]['end_ns'],
                            chosen_duration_error=predicted['cta_duration_ns'][selected]/
                                (ctas[selected]['end_ns']-ctas[selected]['entry_ns'])-1,
                            observed_ghz=statistics.median((c['end_c']-c['entry_c'])/
                                (c['end_ns']-c['entry_ns']) for c in ctas if len(c['tiles'])==max_tiles)))
                        raw = root/record['raw']
                        hashes[str(raw)] = hashlib.sha256(raw.read_bytes()).hexdigest()
                hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            previous = fitted_events['scores']['shared']['ns']['valid_fill']['case_scores'][row['id']]
            observed['envelope_us'] = previous['observed_envelope_ns']/1000
            observed_ghz = statistics.median(d['observed_ghz'] for d in cta_diagnostics)
            _, _, durations = case_recurrence(row, setups[row['id']], calibration)
            observed_f_event = max(durations(observed_ghz))/1000 + calibration['fixed_event_extra_ns']/1000
            observed['critical_cohort_ghz'] = observed_ghz
            report = dict(case=row['id'], **predicted, observed=observed,
                measured_frequency_diagnostic_event_us=observed_f_event,
                measured_frequency_diagnostic_dual_error=observed_f_event/observed['dual_us']-1,
                critical_cta_diagnostic=dict(
                    chosen_is_last_count=sum(d['chosen_is_last'] for d in cta_diagnostics),
                    processes=len(cta_diagnostics),
                    exit_gap_ns_median=statistics.median(d['chosen_exit_gap_ns'] for d in cta_diagnostics),
                    exit_gap_ns_max=max(d['chosen_exit_gap_ns'] for d in cta_diagnostics),
                    chosen_duration_error_median=statistics.median(d['chosen_duration_error'] for d in cta_diagnostics)),
                frequency_error=predicted['frequency_ghz']/observed_ghz-1,
                dual_error=predicted['event_us']/observed['dual_us']-1,
                plain_error=predicted['event_us']/observed['plain_us']-1,
                envelope_error=predicted['window_us']/observed['envelope_us']-1,
                constant_1p6ghz_dual_error=previous['dual_event_error'])
            reports.append(report)
    scores = {key: supply.metrics([row[key] for row in reports]) for key in
              ('dual_error', 'plain_error', 'envelope_error', 'frequency_error', 'constant_1p6ghz_dual_error', 'measured_frequency_diagnostic_dual_error')}
    output.mkdir(parents=True, exist_ok=False)
    result = dict(protocol='Same-card all-data development composition, not a heldout validation.',
        calibration=calibration, gpu=fitted_events['gpu'], cases=reports, scores=scores,
        input_sha256=hashes,
        implementation_sha256={Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (supply, clock, events, events.base)},
        model_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Supply/event parameters were developed on these same 13 conditions.',
            'Only cfg_b, full 132-SM grid, stage 6, complete K tiles and aligned A pitch are calibrated.',
            'Clock comes from a different job on the same GPU; all 21 R09 cases trained it.',
            'Natural-ns event constants currently transfer across frequency and input conditions.',
            'Nominal source work, including scheduled OOB tiles, follows the R09 feature definition; fill/pitch energy is uncalibrated.',
            'No target time, frequency or measured CTA list enters predict(). Entry skew is zero.',
            'F is the pooled dual-event overhead; plain timing is a diagnostic, without a newly fitted observer correction.'])
    (output/'composition.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(scores, indent=2))




def zero_product_fraction(row, work):
    """Guaranteed zero-product share of nominal work, from maps and zero panels."""
    if row.get('input_mode')=='zero':return 1.
    tm,tn,tk=events.base.CONFIGS[row['config']]['tile']
    m=row.get('input_map_m') or row['m'];n=row.get('input_map_n') or row['n']
    if row.get('zero_m',-1)>=0:m=min(m,row['zero_m'])
    if row.get('zero_n',-1)>=0:n=min(n,row['zero_n'])
    useful=sum(max(0,min(tm,m-mi*tm))*max(0,min(tn,n-ni*tn))*row['k'] for c in work for mi,ni in c)
    nominal=sum(map(len,work))*tm*tn*events.cdiv(row['k'],tk)*tk
    return 1-useful/nominal


def predict_components(row, setup, calibration):
    """Three-config development composition with explicit support and time protocols.

    Numeric values outside supply support are diagnostics, not frozen predictions.
    A first-phase supply model can be supplied separately after calibration.
    """
    cfg = row['config']
    spec = events.base.CONFIGS[cfg]
    tm, tn, tk = spec['tile']
    cm, cn = spec['cluster']
    work = events.scheduled_work(cfg,row['m'],row['n'],setup['grid'],row['swizzle'])
    requests = supply.supply_request_rows(row,setup)
    kt = row['k']//64
    tiles = sum(map(len,work))
    point = dict(input_mode=row.get('input_mode','dyadic'),
        tensor_mean_cycles=tiles*kt*(2*tm*tn*tk/4096)/132,
        source_mean_kib=tiles*kt*(2*tm*tk/cn+2*tn*tk/cm)/1024/132)
    params = calibration['events_ns'][cfg]
    output_rule=calibration.get('output_rules',{}).get(cfg)
    fractions=[[max(0,min(tm,row['m']-mi*tm))*max(0,min(tn,row['n']-ni*tn))/(tm*tn)
                for mi,ni in cta] for cta in work]
    first_bytes=sum(f[0]*tm*tn*4 for f in fractions if f)
    single=max(map(len,work))==1
    later = calibration['supply'][cfg]
    first = calibration.get('first_supply',{}).get(cfg,later)

    def evaluate(frequency,details=False):
        loops = [[] for _ in work]
        supported = []
        for model, selected in ((first,[r for r in requests if r['j']==0]),
                                (later,[r for r in requests if r['j']>0])):
            values, ok = supply.supply_predict(selected,model,frequency_ghz=frequency)
            for request,value,valid in zip(selected,values,ok):
                loops[request['cta']].append((request['j'],float(value)))
                supported.append((request['cta'],request['j'],bool(valid)))
        durations=[];timelines=[]
        for cta,valid in zip(loops,fractions):
            p=dict(params);epilogues=None
            if output_rule and cta:
                if spec['schedule']!='cooperative':
                    raise ValueError('these output rules were calibrated for cooperative kernels')
                if single:
                    rule=output_rule['E0_single']
                    e0=max(rule['floor_ns'],rule['beta_ns_per_MiB']*first_bytes/2**20)+rule['R_ns']
                    p['Etail']=output_rule['Etail_single_mean_ns']
                else:
                    e0=output_rule['E0_multi']['effective_ns']
                    p['Etail']=(1-valid[-1])*output_rule['Etail_oob_median_ns']+valid[-1]*output_rule['v09_Etail_multi_median_ns']
                middle=output_rule['E_middle_ns']
                if middle is None:middle=output_rule['middle_padding_proxy']['ns']
                epilogues=[e0]
                for j,fraction in enumerate(valid[1:],1):
                    key='Elast' if j==len(valid)-1 else 'E_middle'
                    full=output_rule['Elast_ns'] if key=='Elast' else middle
                    epilogues.append((1-fraction)*output_rule[key+'_oob_ns']+fraction*full)
            value=events.cta_cycles(p,spec['schedule'],len(cta),kt,detail=details,
                mainloops=[v for _,v in sorted(cta)],epilogues=epilogues,trace_events=details)
            durations.append(value[0] if details else value)
            if details:timelines.append(value[1])
        return durations, supported, timelines

    model=calibration['clock']
    z=zero_product_fraction(row,work)
    if calibration.get('zero_activity_mixing',False):
        # A zero-new-parameter hypothesis; the all-zero endpoint has both inputs
        # zero, so this is not an established physical power law for partial zeros.
        model=dict(model,coefficients=model['coefficients'][:])
        mode=clock.MODES.index(point['input_mode']);zero=clock.MODES.index('zero')
        for offset in (1,1+len(clock.MODES)):
            model['coefficients'][offset+mode]=(1-z)*model['coefficients'][offset+mode]+z*model['coefficients'][offset+zero]
    solved = clock.solve_source_envelope(point,model,lambda f:max(evaluate(f)[0])/1000)
    durations,supported,timelines = evaluate(solved['frequency_ghz'],True)
    critical = max(range(len(durations)),key=durations.__getitem__)
    transfer = calibration.get('plain_transfer',{}).get(cfg)
    return dict(**solved,
        dual_event_us=solved['window_us']+calibration['dual_event_extra_ns'][cfg]/1000,
        plain_us=(transfer['F_us']+transfer['kappa']*solved['window_us']) if transfer else None,
        zero_product_fraction=z,zero_activity_mixing=bool(calibration.get('zero_activity_mixing',False)),
        all_supply_supported=all(ok for _,_,ok in supported),
        unsupported_windows=[dict(cta=c,j=j) for c,j,ok in supported if not ok],
        critical_cta=critical,critical_candidates=[i for i,v in enumerate(durations) if abs(v-max(durations))<1e-7],
        critical_work=work[critical],cta_duration_ns=durations,first_output_bytes=first_bytes,
        static_clock_work=point,cta_events=timelines)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    development(args.development_suite.resolve(), args.output.resolve())
