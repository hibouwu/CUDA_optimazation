#!/usr/bin/env python3
"""Diagnose existing random-input output errors without changing their verdicts."""
import argparse
import csv
import gzip
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from analyze_r18 import random_references


def quantiles(values):
    return dict(zip(['min', 'p01', 'p05', 'median', 'p95', 'p99', 'max'],
                    np.quantile(values, [0, .01, .05, .5, .95, .99, 1]).tolist()))


def analyze(run, output):
    cases = {row['id']: row for row in json.loads((run / 'cases.json').read_text())}
    results = []
    references = {}
    output_groups = {}
    point_rows = []
    for path in sorted((run / 'samples').glob('*/*.json')):
        record = json.loads(path.read_text())
        case_id = record['case']
        if case_id not in cases and case_id.startswith('check_'):
            case_id = case_id[len('check_'):]
        case = cases[case_id]
        if case.get('input_mode') != 'random':
            continue
        raw = run / record['raw']
        raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
        if raw_sha != record['raw_sha256']:
            raise ValueError('raw hash mismatch: ' + str(raw))
        with gzip.open(raw, 'rt') as stream:
            events = {event['event']: event for line in stream if line.strip()
                      for event in [json.loads(line)]}
        setup, check = events['setup'], events['check']
        indices = check['checked_indices']
        values = np.asarray(check['checked_values'], dtype=np.float64)
        if len(indices) != len(set(indices)) or not np.isfinite(values).all():
            raise ValueError('duplicate indices or nonfinite outputs: ' + str(raw))
        for field in ['m', 'n', 'k', 'seed', 'input_mode']:
            if setup[field] != case[field]:
                raise ValueError('setup mismatch: ' + field)
        atol, coefficient = setup['check_atol'], setup['check_sum_abs_rtol']
        if (atol, coefficient) != (2**-20, 2**-21):
            raise ValueError('unexpected original tolerance')
        key = (case['m'], case['n'], case['k'], case['seed'], tuple(indices))
        if key not in references:
            ref, tolerance = random_references(indices, *key[:4])
            references[key] = np.asarray(ref), np.asarray(tolerance)
        ref, tolerance = references[key]
        # The imported function returns atol + coefficient * sum(abs(A * B)).
        sum_abs = (tolerance - atol) / coefficient
        error = values - ref
        abs_error = np.abs(error)
        ratio = abs_error / tolerance
        needed_coefficient = np.maximum(abs_error - atol, 0) / sum_abs
        label = record['case'] + '/' + path.stem
        fingerprint = hashlib.sha256(json.dumps([indices, values.tolist()],
                                                 separators=(',', ':')).encode()).hexdigest()
        output_groups.setdefault(fingerprint, []).append(label)
        worst = int(np.argmax(ratio))
        result = dict(
            process=label, raw=str(raw), raw_sha256=raw_sha,
            shape=[case['m'], case['n'], case['k']], seed=case['seed'],
            original_status=check['status'], returncode=record['returncode'],
            original_max_error_ratio=events['tolerance']['max_error_ratio'],
            original_max_storage_reference_error=check['max_storage_reference_error'],
            checked=len(indices), full_output=len(indices) == case['m'] * case['n'],
            nonfinite=check['nonfinite'], padding_errors=check['padding_errors'],
            positive_error=int(np.count_nonzero(error > 0)),
            negative_error=int(np.count_nonzero(error < 0)),
            zero_error=int(np.count_nonzero(error == 0)),
            error_toward_zero=int(np.count_nonzero(error * ref < 0)),
            error_away_from_zero=int(np.count_nonzero(error * ref > 0)),
            signed_error_mean=float(error.mean()), signed_error=quantiles(error),
            absolute_error=quantiles(abs_error), absolute_reference=quantiles(np.abs(ref)),
            sum_abs=quantiles(sum_abs), tolerance=quantiles(tolerance),
            error_ratio=quantiles(ratio), violations=int(np.count_nonzero(ratio > 1)),
            max_storage_error_replayed=float(np.max(np.abs(values - ref.astype(np.float32)))),
            error_reference_correlation=float(np.corrcoef(error, ref)[0, 1]),
            error_reference_zero_intercept_slope=float(np.dot(error, ref) / np.dot(ref, ref)),
            minimum_coefficient_for_these_outputs=float(needed_coefficient.max()),
            coefficient_factor_over_original=float(needed_coefficient.max() / coefficient),
            output_fingerprint=fingerprint,
            worst_ratio_point=dict(index=indices[worst], row=indices[worst] // case['n'],
                                   column=indices[worst] % case['n'], output=float(values[worst]),
                                   reference=float(ref[worst]), error=float(error[worst]),
                                   sum_abs=float(sum_abs[worst]), tolerance=float(tolerance[worst]),
                                   ratio=float(ratio[worst])))
        results.append(result)
        for q, index in enumerate(indices):
            point_rows.append([label, index, index // case['n'], index % case['n'],
                               values[q], ref[q], error[q], sum_abs[q], tolerance[q],
                               ratio[q], needed_coefficient[q]])
    source = Path(inspect.getsourcefile(random_references)).resolve()
    summary = dict(
        run=str(run), reference_function='analyze_r18.random_references',
        reference_source=str(source), reference_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_atol=2**-20, original_sum_abs_coefficient=2**-21,
        scope='Offline output diagnosis; original pass/fail and tolerances are unchanged.',
        limitations=[
            'The public CPU reference regenerates quantized FP16 values; device input buffers were not archived.',
            'Comparisons across records with different M/N or seeds do not isolate the effect of K.',
            'The minimum coefficient describes only these outputs and is not a proposed replacement tolerance.',
            'Output agreement across configurations does not identify an arithmetic mechanism.'],
        identical_output_groups=list(output_groups.values()), processes=results)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'diagnostic.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'points.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['process', 'index', 'row', 'column', 'output', 'reference',
                         'signed_error', 'sum_abs', 'original_tolerance', 'error_ratio',
                         'minimum_sum_abs_coefficient'])
        writer.writerows(point_rows)
    for row in results:
        print(row['process'], 'violations=', row['violations'],
              'max_ratio=', row['error_ratio']['max'],
              'min_coefficient=', row['minimum_coefficient_for_these_outputs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.output.resolve())
