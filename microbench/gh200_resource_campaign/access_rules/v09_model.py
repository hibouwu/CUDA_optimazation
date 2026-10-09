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


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    development(args.development_suite.resolve(), args.output.resolve())
