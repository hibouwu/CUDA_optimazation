#!/usr/bin/env python3
"""R09 diagnostic: can V03's measured-frequency residual be a fixed gap alone?

Replay the eleven V03 held-out traces without modifying its frozen score.
Write one new reanalysis directory. Times are us and frequency is GHz.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import v03_analyze


def analyze(run, output):
    frozen_path = run / 'source/v03-predictions.json'
    frozen = json.loads(frozen_path.read_text())
    rows = [r for r in v03_analyze.load_rows(run, 'main') if r['status'] == 'measured']
    with (run / 'cases.csv').open() as stream:
        old = {r['label']: r for r in csv.DictReader(stream) if r['group'] == 'heldout'}
    cases = []
    for p in frozen['heldout']:
        label = p['label']
        plain = [r for r in rows if r['case_id'] == 'plain_' + label]
        traced = [r for r in rows if r['case_id'] == 'trace_' + label]
        traces = [v03_analyze.process_metrics(run, r) for r in traced]
        for r in plain + traced:
            if r['check']['max_storage_reference_error'] != 0:
                raise ValueError('nonzero archived numeric error: ' + label)
        m = {key: median(t[key] for t in traces) for key in
             ('critical_cycles', 'ghz', 'envelope_us', 'host_gap_us', 'event_us')}
        plain_us = median(r['calls'][0]['elapsed_us'] for r in plain)
        fixed_us = p['event_us'] - p['cta_window_us']
        measured_clock_us = fixed_us + p['cta_cycles'] / (m['ghz'] * 1000)
        error = measured_clock_us / plain_us - 1
        for value, key in [(plain_us, 'measured_us'), (m['critical_cycles'], 'cycles_meas'),
                           (m['ghz'], 'ghz_meas'), (m['host_gap_us'], 'host_gap_meas'),
                           (error, 'error_with_measured_clock')]:
            if abs(value - float(old[label][key])) > 1e-9:
                raise ValueError('raw replay differs: ' + label + '/' + key)
        cases.append(dict(
            case=label, plain_processes=len(plain), trace_processes=len(traced),
            plain_us=plain_us, traced_us=m['event_us'], ghz=m['ghz'],
            frozen_fixed_us=fixed_us, measured_clock_us=measured_clock_us,
            measured_clock_error=error, measured_clock_residual_us=measured_clock_us - plain_us,
            cycle_model_difference_us=(p['cta_cycles'] - m['critical_cycles']) / (m['ghz'] * 1000),
            cycle_error=p['cta_cycles'] / m['critical_cycles'] - 1,
            paired_event_minus_envelope_us=m['host_gap_us'],
            gap_increase_over_R09_us=m['host_gap_us'] - frozen['v02_rules']['host_gap']['value'],
            median_cycle_over_frequency_minus_envelope_us=m['critical_cycles'] / (m['ghz'] * 1000)
            - m['envelope_us'],
            traced_minus_plain_us=m['event_us'] - plain_us,
        ))
    result = dict(
        purpose='Post-measurement R09 diagnostic; V03 frozen predictions and decision unchanged.',
        run=str(run), frozen_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        old_R09_host_gap_us=frozen['v02_rules']['host_gap']['value'], cases=cases,
        notes=[
            'V03 critical_cycles is a median over maximum-tile CTAs, not the V08 Cmax.',
            'Values are medians from separate processes, not an additive single-call causal split.',
            'The old fixed term also includes entry skew, post-store and tail, not only host_gap.',
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / 'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'residual.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print('raw replay matches all', len(cases), 'V03 held-out cases; output:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.output.resolve())
