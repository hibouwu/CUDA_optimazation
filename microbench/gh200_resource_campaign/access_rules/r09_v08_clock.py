#!/usr/bin/env python3
"""R09 offline diagnostic of frozen V08 time conversion; never refit or rescore V08.

Read V08 samples and its frozen predictions. Write a new reanalysis directory with
frequency substitution, an additive time-error split, and the existing L2 controls.
All times are us, frequencies GHz, and cycles are local SM clock64 cycles.
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.output.resolve())
