#!/usr/bin/env python3
"""R09 offline diagnostic of frozen V08 time conversion; never refit or rescore V08.

Read V08 samples and its frozen predictions. Write a new reanalysis directory with
frequency substitution, an additive time-error split, and the existing L2 controls.
All times are us, frequencies GHz, and cycles are local SM clock64 cycles.

--candidates --followup V08F --trend-run R09RUN compares clock forms using only
the original V08 time-calibration cases for fitting. C, F and kappa stay frozen.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import v08_model


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spread(values):
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values))


def clock_at_window(prediction, rule, window_us, log_limit=None):
    # Recover the traffic estimate actually used by the frozen prediction. This
    # avoids rerunning the old hash-order-dependent LRU estimate, and is NOT a
    # measurement of physical DRAM bytes. phi and mu remain frozen in this slice.
    p = prediction
    frozen_d = (rule['a'] - rule['b'] * p['phi'] * math.log(p['window_us'])
                - rule['d'] * p['mu'] - p['clock_ghz']) / rule['c']
    traffic_bytes = frozen_d * p['window_us'] * 1e6
    log_window = min(window_us, log_limit) if log_limit else window_us
    clock = (rule['a'] - rule['b'] * p['phi'] * math.log(log_window)
             - rule['c'] * traffic_bytes / (window_us * 1e6) - rule['d'] * p['mu'])
    return clock, traffic_bytes


def time_split(p, m):
    measured_f = m['ghz_ends']
    scale = measured_f * 1000
    replacement_us = p['fixed_us'] + p['converted_cycles'] / scale
    # This identity splits medians from separate plain/stamped/ends processes.
    # It is bookkeeping, not a causal decomposition of one physical invocation.
    terms = dict(
        frequency_us=p['converted_cycles'] / (1000 * p['clock_ghz'])
        - p['converted_cycles'] / scale,
        cycle_model_us=p['kappa'] * (p['critical_cycles'] - m['c_max_stamped']) / scale,
        kappa_transfer_us=(p['kappa'] * m['c_max_stamped'] - m['c_max_ends']) / scale,
        envelope_proxy_us=m['c_max_ends'] / scale - m['window_ends'],
        fixed_gap_us=p['fixed_us'] - (m['ends_us'] - m['window_ends']),
        ends_vs_plain_us=m['ends_us'] - m['plain_us'],
    )
    closure = sum(terms.values()) - (p['predicted_us'] - m['plain_us'])
    if abs(closure) > 1e-8:
        raise ValueError('time split does not close')
    return dict(**terms, closure_us=closure, actual_frequency_time_us=replacement_us,
                actual_frequency_error=replacement_us / m['plain_us'] - 1,
                frozen_error=p['predicted_us'] / m['plain_us'] - 1,
                observed_kappa=m['c_max_ends'] / m['c_max_stamped'],
                ends_event_minus_window_medians_us=m['ends_us'] - m['window_ends'],
                plain_minus_ends_window_us=m['plain_us'] - m['window_ends'])


def analyze(run, output):
    frozen_path = run / 'frozen/v08-predictions.json'
    frozen = read_json(frozen_path)
    # Only check the few inputs used below, without touching archived metadata.
    for relative, key in [('cases.json', 'cases_sha256'),
                          ('static_setup.json', 'static_setup_sha256'),
                          ('derived/summary.json', 'summary_sha256')]:
        if sha256(run / relative) != frozen[key]:
            raise ValueError('frozen input differs: ' + relative)
    official = read_json(run / 'reanalysis/validation-v1/validation.json')
    if official['prediction_sha256'] != sha256(frozen_path):
        raise ValueError('official result uses a different frozen prediction')
    rows = {r['id']: r for r in read_json(run / 'cases.json')}
    setups = {s['case']: s['setup'] for s in read_json(run / 'static_setup.json')}
    summary = read_json(run / 'derived/summary.json')
    calibration = frozen['calibration']
    cal_ranges = {}
    fixed = []
    for cfg, selection in calibration['selection'].items():
        cs = [summary[cid] for cid in selection['time_cases']]
        cal_ranges[cfg] = dict(
            window_us=spread([c['window_ends'] for c in cs]),
            plain_minus_ends_window_us=spread([c['plain_us'] - c['window_ends'] for c in cs]),
            ends_event_minus_window_medians_us=spread([c['ends_us'] - c['window_ends'] for c in cs]),
            frozen_F_us=calibration['time'][cfg]['F'],
            frozen_kappa=calibration['time'][cfg]['kappa'],
        )
        for c in cs:
            fixed.append(dict(case=c['id'], config=cfg, window_us=c['window_ends'],
                              plain_us=c['plain_us'], ends_us=c['ends_us'],
                              ends_event_minus_window_medians_us=c['ends_us'] - c['window_ends'],
                              plain_minus_ends_window_us=c['plain_us'] - c['window_ends']))

    cases, extrapolation, h06 = [], [], {}
    official_cases = {c['case']: c for c in official['cases']}
    for cid, p in frozen['predictions'].items():
        row = rows[cid]
        m = v08_model.summarize_case(run, row, setups[cid])
        split = time_split(p, m)
        if abs(split['frozen_error'] - official_cases[cid]['error']) > 1e-12:
            raise ValueError('raw replay differs from official time: ' + cid)
        for key in ('plain_us', 'c_max_stamped', 'c_max_ends', 'ghz_ends'):
            if abs(m[key] - official_cases[cid][key]) > 1e-9:
                raise ValueError('raw replay differs: ' + cid + '/' + key)
        cfg = row['config']
        rule = calibration['time'][cfg]['clock_rule']
        upper = cal_ranges[cfg]['window_us']['maximum']
        at_measured, traffic = clock_at_window(p, rule, m['window_ends'])
        capped, _ = clock_at_window(p, rule, m['window_ends'], upper)
        cases.append(dict(case=cid, config=cfg, m=row['m'], n=row['n'], k=row['k'],
                          plain_us=m['plain_us'], frozen_us=p['predicted_us'],
                          frozen_ghz=p['clock_ghz'], actual_ghz=m['ghz_ends'],
                          measured_window_us=m['window_ends'], frozen_window_us=p['window_us'],
                          clock_error=p['clock_ghz'] / m['ghz_ends'] - 1,
                          rule_at_measured_window_ghz=at_measured,
                          rule_at_measured_window_error=at_measured / m['ghz_ends'] - 1,
                          log_capped_at_calibration_ghz=capped,
                          log_cap_gain_ghz=capped - at_measured,
                          c_max_stamped=m['c_max_stamped'], c_max_ends=m['c_max_ends'],
                          converted_cycles=p['converted_cycles'], **split))
        if cid.endswith('_h06'):
            paired_gaps = []
            for path in sorted((run / 'samples' / cid).glob('ends-*.json')):
                record = read_json(path)
                if record['returncode']:
                    continue
                observed = v08_model.observe(run, record, row, setups[cid])
                ctas = observed['ctas']
                envelope = (max(c['end_ns'] for c in ctas if c['tiles'])
                            - min(c['entry_ns'] for c in ctas)) / 1000
                paired_gaps.append(observed['elapsed_us'] - envelope)
            h06[cid] = dict(prediction={k: v for k, v in p.items() if k != 'per_cta_cycles'},
                            measured={k: v for k, v in m.items() if not k.startswith('per_cta')},
                            paired_ends_event_minus_window_us=spread(paired_gaps),
                            implied_frozen_traffic_bytes=traffic, rule=rule)
            # A one-dimensional formula slice, not predictions for new shapes.
            # At constant estimated traffic, f(W) turns down after this maximum.
            turn_us = rule['c'] * traffic / (1e6 * rule['b'] * p['phi'])
            h06[cid]['fixed_traffic_turn_us'] = turn_us
            for window in (400, 600, 800, 1600, 3200):
                clock, _ = clock_at_window(p, rule, window)
                extrapolation.append(dict(case=cid, window_us=window, fixed_traffic_ghz=clock,
                                          log_penalty_ghz=rule['b'] * p['phi'] * math.log(window)))

    controls = []
    for cid, c in summary.items():
        if c['set'] != 'ctrl':
            continue
        warm = summary[cid.removesuffix('_evict')]
        controls.append(dict(case=cid, config=c['config'],
                             delta_P0_cycles=c['intervals']['P0'] - warm['intervals']['P0'],
                             delta_S_cycles=c['intervals']['S'] - warm['intervals']['S'],
                             delta_plain_us=c['plain_us'] - warm['plain_us']))

    result = dict(
        purpose='Post-measurement diagnostic only; frozen V08 score and decision unchanged.',
        run=str(run), frozen_sha256=sha256(frozen_path),
        diagnostic_source_sha256=sha256(Path(__file__)),
        official_complete_time=official['complete_time'], official_passed=official['passed'],
        calibration=cal_ranges, h06=h06, cases=cases, l2_controls=controls,
        actual_frequency_error=spread([c['actual_frequency_error'] for c in cases]),
        actual_frequency_absolute_error=spread([abs(c['actual_frequency_error']) for c in cases]),
        max_closure_us=max(abs(c['closure_us']) for c in cases),
        notes=[
            'Frequency is the process median of maximum-tile CTA cycle/ns ratios in ends.',
            'Cmax/fmedian is not the cross-SM envelope; envelope_proxy_us keeps that difference.',
            'plain, stamped and ends are separate processes; their differences include protocol/state variation.',
            'fixed_gap_us uses median(ends event) minus median(ends envelope), not median(event-envelope).',
            'h06 also reports paired per-process event-envelope gaps; neither gap is pure host launch.',
            'The extrapolation slice fixes frozen phi, mu and inferred traffic; it is not a new workload forecast.',
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / 'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for filename, table in [('time_split.csv', cases), ('clock_slice.csv', extrapolation),
                            ('calibration_fixed.csv', fixed), ('l2_controls.csv', controls)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    for c in cases:
        if c['case'].endswith('_h06'):
            print(c['case'], 'frozen %+0.3f%%' % (100 * c['frozen_error']),
                  'actual-frequency %+0.3f%%' % (100 * c['actual_frequency_error']),
                  'time %.3f us' % c['actual_frequency_time_us'])
    print('raw replay matches 36 official cases; max closure us:', result['max_closure_us'])
    print('output:', output)


def clock_point(row, observed, setup, calibration, prediction=None):
    """Fixed V08 cycle prediction and workload features; no measured cycles as inputs."""
    cfg = row['config']
    timing = calibration['time'][cfg]
    feat = v08_model.features(row, setup['grid'])
    if prediction is None:
        cycles, ctas, stages, _ = v08_model.critical_cycles(calibration['params'], cfg, feat)
        critical = max(range(len(ctas)), key=ctas.__getitem__)
        phi = sum(ctas) / (132 * cycles)
        mu = stages['mainloop'] / cycles
        traffic = v08_model.base.dram_bytes(cfg, row['m'], row['n'], row['k'], feat['work'])
        converted = timing['kappa'] * cycles
    else:
        # Exact frozen held-out features, including the old LRU traversal outcome.
        cycles, converted = prediction['critical_cycles'], prediction['converted_cycles']
        critical = prediction['critical_cta']
        phi, mu = prediction['phi'], prediction['mu']
        _, traffic = clock_at_window(prediction, timing['clock_rule'], prediction['window_us'])
    tm, tn, tk = v08_model.base.CONFIGS[cfg]['tile']
    lower_bound = 2 * tm * tn * tk / 4096  # FP16 Tensor compute cycles/Ktile/SM
    compute_per_tile = lower_bound * feat['kt']
    critical_tiles = len(feat['work'][critical])
    all_tiles = sum(map(len, feat['work']))  # includes scheduled OOB/padded work
    return dict(case=row['id'], config=cfg, m=row['m'], n=row['n'], k=row['k'],
                group=row.get('group', row['kind']), phi=phi, mu=mu, traffic_bytes=traffic,
                mu_tensor_critical=critical_tiles * compute_per_tile / cycles,
                mu_tensor_device=all_tiles * compute_per_tile / (132 * cycles),
                compute_cycles_per_kt=lower_bound, critical_tiles=critical_tiles, scheduled_tiles=all_tiles,
                cycles=cycles, converted_cycles=converted, fixed_us=timing['F'],
                observed_window_us=observed['window_ends'], observed_ghz=observed['ghz_ends'],
                plain_us=observed['plain_us'], cycle_error=cycles / observed['c_max_stamped'] - 1)


def duration_term(window, form, tau):
    if form in ('frozen', 'refit_ac', 'full_log'):
        return math.log(window)
    if form == 'no_duration':
        return 0.0
    # Mean of an exponential relaxation over a window, not its instantaneous endpoint.
    x = window / tau
    return 1 + math.expm1(-x) / x


def candidate_frequency(point, window, model):
    a, b, c, d = model['coefficients']
    return (a - b * point['phi'] * duration_term(window, model['form'], model.get('tau_us'))
            - c * point['traffic_bytes'] / (window * 1e6) - d * point[model.get('activity', 'mu')])


def fit_clock_candidate(points, form, frozen_rule, activity='mu'):
    import numpy as np

    if form == 'frozen':
        return dict(form=form, activity=activity, coefficients=[frozen_rule[k] for k in ('a', 'b', 'c', 'd')])
    y = np.array([p['observed_ghz'] for p in points])
    profile = []
    taus = np.logspace(-1, 5, 121) if form == 'relaxation' else [None]
    for tau in taus:
        X = np.array([[1, -p['phi'] * duration_term(p['observed_window_us'], form, tau),
                       -p['traffic_bytes'] / (p['observed_window_us'] * 1e6), -p[activity]] for p in points])
        columns = [0, 2] if form == 'refit_ac' else [0, 2, 3] if form == 'no_duration' else [0, 1, 2, 3]
        offset = X[:, 1] * frozen_rule['b'] + X[:, 3] * frozen_rule['d'] if form == 'refit_ac' else np.zeros(len(y))
        design = X[:, columns]
        fitted, _, rank, _ = np.linalg.lstsq(design, y - offset, rcond=None)
        if rank != len(columns):
            raise ValueError('clock design is rank deficient')
        coefficients = np.zeros(4)
        if form == 'refit_ac':
            coefficients[1], coefficients[3] = frozen_rule['b'], frozen_rule['d']
        coefficients[columns] = fitted
        residual = X @ coefficients - y
        profile.append(dict(tau_us=float(tau) if tau is not None else None,
                            coefficients=coefficients.tolist(), mse_ghz=float(np.mean(residual ** 2)),
                            observed_frequency_rms=float(np.sqrt(np.mean((residual / y) ** 2))),
                            normalized_condition=float(np.linalg.cond(design / np.linalg.norm(design, axis=0)))))
    # Tau is selected on training GHz residuals only, separately inside every fold.
    best = min(profile, key=lambda p: p['mse_ghz'])
    return dict(form=form, activity=activity, **best, tau_profile=profile if form == 'relaxation' else [])


def evaluate_clock(point, model):
    observed_fit = candidate_frequency(point, point['observed_window_us'], model)
    window = point['converted_cycles'] / 1600  # same initial value as the frozen predictor
    failure = None
    for _ in range(200):
        frequency = candidate_frequency(point, window, model)
        if not math.isfinite(frequency) or frequency <= 0:
            failure = 'nonpositive_frequency'
            break
        next_window = .5 * window + .5 * point['converted_cycles'] / (1000 * frequency)
        if abs(next_window / window - 1) < 1e-12:
            window = next_window
            break
        window = next_window
    if failure is None:
        frequency = candidate_frequency(point, window, model)
        closure = window * frequency * 1000 / point['converted_cycles'] - 1
        if abs(closure) > 1e-8:
            failure = 'fixed_point_not_converged'
    return dict(case=point['case'], config=point['config'], k=point['k'], group=point['group'],
                observed_fit_frequency_error=observed_fit / point['observed_ghz'] - 1,
                free_frequency_error=frequency / point['observed_ghz'] - 1 if failure is None else None,
                free_time_error=(point['fixed_us'] + window) / point['plain_us'] - 1 if failure is None else None,
                free_window_us=window if failure is None else None,
                free_frequency_ghz=frequency if failure is None else None,
                cycle_error=point['cycle_error'], solver_failure=failure)


def error_stats(rows):
    result = dict(n=len(rows), solver_failures=sum(r['solver_failure'] is not None for r in rows))
    for key in ('observed_fit_frequency_error', 'free_frequency_error', 'free_time_error', 'cycle_error'):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = dict(rms=math.sqrt(statistics.fmean(x*x for x in values)),
                           median_abs=statistics.median(abs(x) for x in values), maximum_abs=max(map(abs, values)),
                           mean_signed=statistics.fmean(values), n=len(values)) if values else None
    return result


def clock_candidates(run, followup, trend_run, output, activity=False):
    frozen_path = run / 'frozen/v08-predictions.json'
    frozen = read_json(frozen_path)
    calibration = frozen['calibration']
    rule = calibration['time']['cfg_a']['clock_rule']
    for relative, key in [('cases.json', 'cases_sha256'), ('static_setup.json', 'static_setup_sha256'),
                          ('derived/summary.json', 'summary_sha256')]:
        if sha256(run / relative) != frozen[key]:
            raise ValueError('frozen calibration input changed: ' + relative)
    rows = {r['id']: r for r in read_json(run / 'cases.json')}
    setups = {r['case']: r['setup'] for r in read_json(run / 'static_setup.json')}
    summary = read_json(run / 'derived/summary.json')
    train_ids = [cid for selection in calibration['selection'].values() for cid in selection['time_cases']]
    points = [clock_point(rows[cid], summary[cid], setups[cid], calibration) for cid in train_ids]
    specs = ({'aggregate': ('relaxation', 'mu'),
              'tensor_critical': ('relaxation', 'mu_tensor_critical'),
              'tensor_device': ('relaxation', 'mu_tensor_device')} if activity else
             {form: (form, 'mu') for form in ('refit_ac', 'full_log', 'no_duration', 'relaxation')})
    forms = tuple(specs)
    fits = dict(frozen=fit_clock_candidate(points, 'frozen', rule))
    fits.update({name: fit_clock_candidate(points, form, rule, key) for name, (form, key) in specs.items()})
    scores, folds, grouped = [], [], {}
    for scheme in ('geometry', 'K_band'):
        labels = {p['case']: f"{p['m']}x{p['n']}" if scheme == 'geometry' else
                  'K<=2048' if p['k'] <= 2048 else '2048<K<=8192' if p['k'] <= 8192 else 'K>8192' for p in points}
        for label in sorted(set(labels.values())):
            training = [p for p in points if labels[p['case']] != label]
            held = [p for p in points if labels[p['case']] == label]
            for name, (form, key) in specs.items():
                model = fit_clock_candidate(training, form, rule, key)
                evaluated = [dict(evaluate_clock(p, model), phase=scheme, fold=label, form=name) for p in held]
                scores.extend(evaluated)
                folds.append(dict(scheme=scheme, held_group=label, form=name, activity=key, train_n=len(training),
                                  coefficients=model['coefficients'], tau_us=model.get('tau_us'),
                                  normalized_condition=model.get('normalized_condition'), errors=error_stats(evaluated)))
        grouped[scheme] = {form: error_stats([r for r in scores if r['phase'] == scheme and r['form'] == form]) for form in forms}

    # Development extrapolation is read only after all calibration fits and folds.
    heldout = [clock_point(rows[cid], v08_model.summarize_case(run, rows[cid], setups[cid]),
                           setups[cid], calibration, prediction=p) for cid, p in frozen['predictions'].items()]
    frows = {r['id']: r for r in read_json(followup / 'cases.json')}
    fsetups = {r['case']: r['setup'] for r in read_json(followup / 'static_setup.json')}
    fsummary = read_json(followup / 'reanalysis/followup-v1/summary.json')
    followup_points = [clock_point(r, fsummary[cid], fsetups[cid], calibration) for cid, r in frows.items()]
    diagnostics = {}
    for phase, dataset in [('calibration', points), ('V08_heldout_development', heldout), ('V08F_development', followup_points)]:
        diagnostics[phase] = {}
        for form, model in fits.items():
            evaluated = [dict(evaluate_clock(p, model), phase=phase, fold='', form=form) for p in dataset]
            scores.extend(evaluated)
            diagnostics[phase][form] = dict(all=error_stats(evaluated),
                per_config={cfg: error_stats([r for r in evaluated if r['config'] == cfg]) for cfg in ('cfg_a', 'cfg_b', 'cfg_c')})
    trend = read_json(trend_run / 'analysis/summary.json')
    trend_cases = [{key: row[key] for key in ('id', 'config', 'plain_us', 'ghz_ends', 'c_max_ends', 'window_ends')}
                   for row in trend['cases'].values() if row.get('status') != 'numeric_error']
    result = dict(frozen_sha256=sha256(frozen_path), diagnostic_source_sha256=sha256(Path(__file__)),
        activity_comparison=activity,
        training_cases=train_ids, train_n=len(points), forms=fits, conditional_grouped_cv=grouped,
        folds=folds, diagnostics=diagnostics, input_points={'calibration': points, 'heldout': heldout, 'followup': followup_points},
        other_card_trend=dict(environment=trend['environment'], cases=trend_cases, predictions_not_evaluated=True),
        notes=[
            'Only the original 64 V08 time-calibration cases train clock coefficients and tau; all original exclusions remain.',
            'Cycle predictions, phi, F and kappa stay frozen; activity mode changes only the specified mu proxy.',
            'Tensor critical uses the software work count of the predicted critical CTA; tensor device sums all scheduled tiles over 132*Cmax.',
            'The 512/512/1024 cycle/Ktile lower bounds exclude supply waits; proxies are ideal demand fractions, not measured Tensor active counters.',
            'This is conditional clock cross-validation, not a full-cycle-model cross-validation; frozen C/F/kappa used the original calibration.',
            ('Activity mode refits only the existing bounded family inside every fold; frozen original coefficients are a separate diagnostic.' if activity else
             'Grouped CV baseline refits a/c on each training fold with old V03 b/d; the all-data frozen baseline is reported separately.'),
            'Fits minimize absolute GHz residuals using observed W; free prediction solves W=kappa*C/(1000*f(W)) without observed W or measured cycles.',
            'Relaxation g(W)=1-(tau/W)*(1-exp(-W/tau)) has a steady limit; it is a candidate mean-frequency shape, not a measured physical relaxation.',
            'The tau profile spans 0.1 to 100000 us; training residual alone selects tau separately inside each fold.',
            'No coefficient sign constraints or frequency clamps; failed free solutions remain counted.',
            'Calibration/followup traffic uses deterministic sorted LRU; frozen heldout traffic is recovered from the exact frozen predictions.',
            'V08 heldout and same-card V08F are development diagnostics, not new validation; the different-card R09 run is not fitted or pooled.',
        ])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'candidates.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    for scheme, values in grouped.items():
        for form, stats in values.items():
            print(scheme, form, 'free f RMS', stats['free_frequency_error']['rms'],
                  'free time RMS', stats['free_time_error']['rms'], 'solver failures', stats['solver_failures'])
    print('output:', output)


def input_clock_evidence(run, r10, r15, output):
    """Same GPU, distinct protocols: whole-CTA and local-stage rates stay separate."""
    from analyze_r18 import replay
    from analyze_r15_output_ns import direct_windows
    import gzip

    gpu = read_json(run / 'environment.json')['gpu'].split(',')[0]
    observations = []
    for experiment, root, variants in [('R09', run, ('ends',)), ('R10', r10, ('stamped', 'dual')),
                                        ('R15', r15, ('ends', 'global'))]:
        environment = read_json(root / 'environment.json')
        if environment['gpu'].split(',')[0] != gpu:
            raise ValueError('input evidence must remain on one GPU')
        setups = {s['case']: s['setup'] for s in read_json(root / 'static_setup.json')}
        for row in read_json(root / 'cases.json'):
            records = [read_json(p) for p in sorted((root / 'samples' / row['id']).glob('*.json'))]
            plain = [r['elapsed_us'] for r in records if not r['returncode'] and r['variant'] == 'plain']
            for variant in variants:
                per = []
                for rec in records:
                    if rec['returncode'] or rec['variant'] != variant:
                        continue
                    obs = (v08_model.observe(root, rec, row, setups[row['id']]) if variant == 'ends'
                           else replay(root, rec, row))
                    ctas = obs['ctas']
                    count = lambda c: c['tiles'] if isinstance(c['tiles'], int) else len(c['tiles'])
                    active = [c for c in ctas if count(c)]
                    T = max(map(count, active))
                    f = statistics.median((c['end_c'] - c['entry_c']) / (c['end_ns'] - c['entry_ns'])
                                          for c in active if count(c) == T)
                    item = dict(event_us=rec['elapsed_us'], cta_ghz=f, max_tiles=T,
                                window_us=(max(c['end_ns'] for c in active) - min(c['entry_ns'] for c in ctas)) / 1000)
                    if variant == 'dual':
                        item['later_mainloop_ghz'] = statistics.median(
                            (c[1] - c[0]) / (n[1] - n[0]) for t in active
                            for c, n in zip(t['tiles'][1:], t['tiles_ns'][1:]))
                    if variant == 'global':
                        with gzip.open(root / rec['raw'], 'rt') as stream:
                            events = {e['event']: e for line in stream if line.strip() for e in [json.loads(line)]}
                        ws = direct_windows(events['setup'], events['call'])
                        item['issuer_store_ghz'] = statistics.median(w['issuer_store_cycles'] / w['issuer_store_ns'] for w in ws)
                    per.append(item)
                observations.append(dict(experiment=experiment, source=str(root), case=row['id'],
                    config=row['config'], input_mode=row.get('input_mode', 'dyadic'), seed=row.get('seed', 17),
                    m=row['m'], n=row['n'], k=row['k'], variant=variant, processes=len(per),
                    **{key: statistics.median(p[key] for p in per) for key in per[0]},
                    plain_us=statistics.median(plain),
                    variant_over_plain=statistics.median(p['event_us'] for p in per) / statistics.median(plain) - 1,
                    cta_ghz_cv=statistics.pstdev(p['cta_ghz'] for p in per) / statistics.mean(p['cta_ghz'] for p in per)))
    result = dict(gpu=gpu, observations=observations, coefficients_fitted=False,
        notes=['Whole-CTA rates use entry to the consumer final endpoint; local rates use different endpoints and are never pooled with them.',
               'R10 stamped/dual are per-tile-instrumented protocols, not the R09/R15 ends protocol; wide is a scratch-matched plain control without clock records.',
               'Only R09 has three input modes, at one long-window geometry per config; R10/R15 add dyadic evidence only.',
               'Same GPU across different jobs does not establish identical preceding state or instrumentation costs.'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'input-clock-evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print('same-card evidence:', len(observations), 'separate protocol rows; no coefficient fit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--candidates', action='store_true')
    parser.add_argument('--activity', action='store_true', help='compare activity proxies within the existing bounded duration family')
    parser.add_argument('--input-evidence', action='store_true')
    parser.add_argument('--r10-run', type=Path)
    parser.add_argument('--r15-run', type=Path)
    parser.add_argument('--followup', type=Path)
    parser.add_argument('--trend-run', type=Path)
    args = parser.parse_args()
    if args.input_evidence:
        input_clock_evidence(args.run.resolve(), args.r10_run.resolve(), args.r15_run.resolve(), args.output.resolve())
    elif args.candidates or args.activity:
        clock_candidates(args.run.resolve(), args.followup.resolve(), args.trend_run.resolve(), args.output.resolve(), args.activity)
    else:
        analyze(args.run.resolve(), args.output.resolve())
