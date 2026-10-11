#!/usr/bin/env python3
"""Eight-case, zero-new-parameter diagnostic of nominal zero-product activity mixing."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import statistics as st

from analyze_r18 import replay
import r09_input_source_clock as clock
import v06_run as common
import v08_model


def static_zero_activity(row, setup):
    """Count nominal scheduled Tensor work; zero thresholds must cover whole tiles."""
    cfg = v08_model.base.CONFIGS[row['config']]
    tm, tn, tk = cfg['tile']
    cm, cn = cfg['cluster']
    if row['input_mode'] != 'dyadic' or any(z >= 0 and z % t for z, t in ((row['zero_m'], tm), (row['zero_n'], tn))):
        raise ValueError('this diagnostic requires dyadic inputs with whole-tile zero regions')
    work = v08_model.scheduled_work(row['config'], row['m'], row['n'], setup['grid'], row['swizzle'])
    counts = [len(cta) for cta in work]
    zero_counts = [sum((row['zero_m'] >= 0 and mi * tm >= row['zero_m'])
                       or (row['zero_n'] >= 0 and ni * tn >= row['zero_n']) for mi, ni in cta) for cta in work]
    kt = (row['k'] + tk - 1) // tk
    compute_per_tile = kt * 2 * tm * tn * tk / 4096
    total = sum(counts) * compute_per_tile
    zero = sum(zero_counts) * compute_per_tile
    source_per_tile = kt * (2 * tm * tk / cn + 2 * tn * tk / cm) / 1024
    return dict(input_mode='dyadic', tensor_mean_cycles=total / 132, source_mean_kib=sum(counts) * source_per_tile / 132), dict(
        z=zero / total, scheduled_tiles=sum(counts), zero_tiles=sum(zero_counts), nominal_cycles=total,
        nominal_zero_cycles=zero, per_cta_tiles=counts, per_cta_zero_tiles=zero_counts), work


def interpolated_model(model, z):
    mixed = copy.deepcopy(model)
    # Change only the dyadic evaluation slot; all endpoint parameters stay frozen.
    mixed['coefficients'][1] = (1 - z) * model['coefficients'][1] + z * model['coefficients'][2]
    mixed['coefficients'][4] = (1 - z) * model['coefficients'][4] + z * model['coefficients'][5]
    return mixed


def error_scores(rows):
    return {name: dict(n=len(rows), rms=math.sqrt(st.fmean(r[name] ** 2 for r in rows)),
                       maximum_abs=max(abs(r[name]) for r in rows))
            for name in ('dyadic_error', 'mixed_error')}


def analyze(m_run, n_run, clock_path, composition_path, output):
    frozen_sha = clock.clock.sha256(clock_path)
    frozen = clock.clock.read_json(clock_path)
    model = frozen['model']
    original_model = copy.deepcopy(model)
    composition = clock.clock.read_json(composition_path)
    if composition['calibration']['clock']['coefficients'] != model['coefficients'] or composition['calibration']['clock']['tau_us'] != model['tau_us']:
        raise ValueError('root composition and frozen source-clock parameters differ')
    root_cases = {r['case']: r for r in composition['cases']}
    reports, processes, identities, static = [], [], {}, {}
    checked_records = 0
    for run in (m_run, n_run):
        common.verify(run)
        if clock.clock.read_json(run / 'environment.json')['gpu'] != frozen['environment']['gpu']:
            raise ValueError('mixed-activity diagnostic must remain on the reference GPU')
        setups = {s['case']: s['setup'] for s in clock.clock.read_json(run / 'static_setup.json')}
        for name in ('cases.json', 'static_setup.json', 'environment.json', 'source_hashes.json'):
            identities[str(run / name)] = clock.clock.sha256(run / name)
        for row in clock.clock.read_json(run / 'cases.json'):
            cid = row['id']
            point, activity, work = static_zero_activity(row, setups[cid])
            mixed = interpolated_model(model, activity['z'])
            records = [clock.clock.read_json(p) for p in sorted((run / 'samples' / cid).glob('*.json'))]
            for variant in ('plain', 'wide', 'stamped', 'dual'):
                chosen = [r for r in records if r['variant'] == variant]
                if len(chosen) != 10 or {r['trial'] for r in chosen} != set(range(10)):
                    raise ValueError('all declared processes must remain present')
            per = []
            for record in records:
                if record['returncode']:
                    raise ValueError('failed process in eight-case diagnostic')
                observed = replay(run, record, row)
                checked_records += 1
                identities[str(run / record['raw'])] = record['raw_sha256']
                if record['variant'] != 'dual':
                    continue
                ctas = observed['ctas']
                if [c['work'] for c in ctas] != work:
                    raise ValueError('actual traced work differs from the software z definition')
                most = max(len(c['tiles']) for c in ctas)
                frequency = st.median((c['end_c'] - c['entry_c']) / (c['end_ns'] - c['entry_ns'])
                                      for c in ctas if len(c['tiles']) == most)
                window = (max(c['end_ns'] for c in ctas) - min(c['entry_ns'] for c in ctas)) / 1000
                dyadic = clock.source_frequency(point, window, model)['frequency_ghz']
                value = clock.source_frequency(point, window, mixed)['frequency_ghz']
                per.append(dict(case=cid, axis='M' if row['zero_m'] >= 0 else 'N', k=row['k'], input_path=row['input_path'],
                    trial=record['trial'], z=activity['z'], window_us=window, observed_ghz=frequency,
                    dyadic_ghz=dyadic, mixed_ghz=value, dyadic_error=dyadic / frequency - 1, mixed_error=value / frequency - 1))
            window, frequency = st.median(p['window_us'] for p in per), st.median(p['observed_ghz'] for p in per)
            expected = root_cases[cid]['observed']
            if not math.isclose(window, expected['envelope_us'], abs_tol=1e-9) or not math.isclose(frequency, expected['critical_cohort_ghz'], abs_tol=1e-12):
                raise ValueError('independent observation differs from root v4 composition')
            dyadic = clock.source_frequency(point, window, model)['frequency_ghz']
            value = clock.source_frequency(point, window, mixed)['frequency_ghz']
            reports.append(dict(case=cid, axis=per[0]['axis'], k=row['k'], input_path=row['input_path'], z=activity['z'],
                scheduled_tiles=activity['scheduled_tiles'], zero_tiles=activity['zero_tiles'], **point,
                effective_d=mixed['coefficients'][1], effective_e=mixed['coefficients'][4],
                window_us=window, observed_ghz=frequency, dyadic_ghz=dyadic, mixed_ghz=value,
                dyadic_error=dyadic / frequency - 1, mixed_error=value / frequency - 1))
            processes.extend(per)
            static[cid] = activity
    if len(reports) != 8 or len(processes) != 80 or checked_records != 320:
        raise ValueError('all eight conditions are required')
    paths = []
    for axis in ('M', 'N'):
        for k in (1024, 4096):
            pair = {r['input_path']: r for r in reports if r['axis'] == axis and r['k'] == k}
            oob, address = pair['oob'], pair['address_zero']
            for key in ('z', 'scheduled_tiles', 'tensor_mean_cycles', 'source_mean_kib', 'effective_d', 'effective_e'):
                if oob[key] != address[key]:
                    raise ValueError('same-value/work paths must have identical z and mixed parameters')
            paths.append(dict(axis=axis, k=k, z=oob['z'], window_change=address['window_us'] / oob['window_us'] - 1,
                observed_frequency_change=address['observed_ghz'] / oob['observed_ghz'] - 1,
                dyadic_frequency_change=address['dyadic_ghz'] / oob['dyadic_ghz'] - 1,
                mixed_frequency_change=address['mixed_ghz'] / oob['mixed_ghz'] - 1))
    if model != original_model or clock.clock.sha256(clock_path) != frozen_sha:
        raise ValueError('original model changed')
    result = dict(protocol='Unvalidated zero-product activity mixture; no new fitted parameter or GPU work.',
        source_clock_path=str(clock_path), source_clock_sha256=frozen_sha, coefficients=model['coefficients'], tau_us=model['tau_us'],
        composition_path=str(composition_path), composition_sha256=clock.clock.sha256(composition_path),
        diagnostic_source_sha256=clock.clock.sha256(Path(__file__)), input_sha256=identities,
        rows=reports, static_zero_work=static, paths=paths, scores=error_scores(reports),
        by_k={str(k): error_scores([r for r in reports if r['k'] == k]) for k in (1024, 4096)},
        checks=dict(numeric_processes_replayed=checked_records, output_values_replayed=checked_records * 4096,
            dual_work_matches_software=True, observations_match_root_v4=True, all_eight_retained=True,
            same_z_Q_S_and_coefficients_for_paths=True, frozen_model_unchanged=True),
        notes=[
            'z weights all scheduled nominal Tensor cycles, including padded OOB output tiles; this batch has whole-tile zero thresholds.',
            'Only dyadic d/e are evaluated as (1-z)*dyadic+z*zero. a/tau/Q/S and all endpoint coefficients are unchanged; z is never fitted.',
            'The zero-product region zeros one operand panel, whereas R09 zero calibration zeros both inputs; interpolating e is an unvalidated activity hypothesis, not physical source power.',
            'Observed dual W and maximum-work CTA cycle/ns are used for both predictions. No free time or joint-composition prediction is made.',
            'Both paths share static z and nominal work. Their prediction difference comes only from observed W; no path correction is added.',
            'All eight known development cases are retained, including remaining K4096 underprediction. No new family, parameter, or follow-up measurement.',
        ])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'zero-activity-clock.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, table in [('cases.csv', reports), ('processes.csv', processes), ('paths.csv', paths)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print('scores', result['scores'], 'by K', result['by_k'])
    for r in reports:
        print(r['case'], 'z', r['z'], 'W', r['window_us'], 'f', r['observed_ghz'], 'errors', r['dyadic_error'], r['mixed_error'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m-run', type=Path, required=True)
    parser.add_argument('--n-run', type=Path, required=True)
    parser.add_argument('--clock', type=Path, required=True)
    parser.add_argument('--composition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.m_run.resolve(), args.n_run.resolve(), args.clock.resolve(), args.composition.resolve(), args.output.resolve())
