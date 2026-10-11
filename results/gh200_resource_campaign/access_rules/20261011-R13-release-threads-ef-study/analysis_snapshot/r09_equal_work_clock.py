#!/usr/bin/env python3
"""Offline R09 equal-work comparison; evaluate the archived source clock without fitting."""
import argparse
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path
import statistics as st

import r09_input_source_clock as clock
import v06_run as common
import v08_model
from r09_clock_input_compare import same_summary


def distribution(values):
    return dict(n=len(values), median=st.median(values), minimum=min(values), maximum=max(values),
                positive=sum(v > 0 for v in values))


def rms(values):
    return dict(n=len(values), rms=math.sqrt(st.fmean(v*v for v in values)), maximum_abs=max(map(abs, values)))


def work_point(row, setup):
    feat = v08_model.features(row, setup['grid'])
    cfg = v08_model.base.CONFIGS[row['config']]
    tm, tn, tk = cfg['tile']
    cm, cn = cfg['cluster']
    tiles = sum(map(len, feat['work']))
    tensor = tiles * feat['kt'] * 2 * tm * tn * tk / 4096
    source = tiles * feat['kt'] * (2 * tm * tk / cn + 2 * tn * tk / cm) / 1024
    return dict(input_mode=row['input_mode'], tensor_mean_cycles=tensor / 132, source_mean_kib=source / 132,
                tensor_cycles=tensor, source_kib=source, output_bytes=row['m'] * row['n'] * 4,
                input_bytes=(row['m'] + row['n']) * row['k'] * 2)


def analyze(run, reference_run, clock_path, output):
    common.verify(run)
    common.verify(reference_run)
    rows = {r['id']: r for r in clock.clock.read_json(run / 'cases.json')}
    setups = {s['case']: s['setup'] for s in clock.clock.read_json(run / 'static_setup.json')}
    summary = clock.clock.read_json(run / 'analysis/summary.json')
    reference = clock.clock.read_json(run / 'bridge_reference.json')
    archived_bridge = clock.clock.read_json(run / 'analysis/bridge_comparison.json')
    environment = summary['environment']
    if environment['gpu'] != reference['environment']['gpu']:
        raise ValueError('bridge is not same GPU')
    if clock.clock.sha256(reference_run / 'analysis/summary.json') != reference['summary_sha256']:
        raise ValueError('bridge reference summary changed')
    # 1. All nine cross-job bridge conditions; the archived scope sentence is stale.
    bridge = []
    keys = ('plain_us', 'ends_us', 'c_max_ends', 'ghz_ends', 'window_ends')
    for cid, old in reference['cases'].items():
        current = summary['cases'][cid]
        item = dict(case=cid, config=rows[cid]['config'], input_mode=rows[cid]['input_mode'])
        item.update({key + '_change': current[key] / old[key] - 1 for key in keys})
        item.update(old_plain_cv=old['plain_cv'], new_plain_cv=current['plain_cv'],
                    old_ends_cv=old['ends_cv'], new_ends_cv=current['ends_cv'])
        saved = next(r for r in archived_bridge['comparisons'] if r['case'] == cid)
        for key in keys:
            if not math.isclose(item[key + '_change'], saved['relative_change'][key], abs_tol=1e-12):
                raise ValueError('bridge numeric comparison changed')
        bridge.append(item)
    if len(bridge) != 9 or {r['config'] for r in bridge} != {'cfg_a', 'cfg_b', 'cfg_c'}:
        raise ValueError('all nine bridges required')

    replay = clock.clock.read_json(output / 'replay/summary.json')
    if not same_summary(summary['cases'], replay['cases']) or replay['prechecks_replayed'] != 18:
        raise ValueError('independent replay is missing or differs')
    for name, digest in clock.clock.read_json(run / 'build/sass_hashes.json').items():
        if clock.clock.sha256(run / 'build' / name) != digest:
            raise ValueError('SASS identity changed')
    records, checks, tolerances = {}, [], {}
    for path in sorted((run / 'samples').glob('*/*.json')):
        record = clock.clock.read_json(path)
        if record['returncode'] or clock.clock.sha256(run / record['raw']) != record['raw_sha256']:
            raise ValueError('process/raw identity failed')
        cid = record['case'].removeprefix('check_')
        with gzip.open(run / record['raw'], 'rt') as stream:
            for line in stream:
                event = json.loads(line)
                if event['event'] == 'check':
                    if event['status'] != 'ok' or event['nonfinite'] or event['padding_errors']:
                        raise ValueError('numeric check failed')
                    checks.append(record['case'])
                if event['event'] == 'tolerance':
                    tolerances.setdefault(cid, []).append(event['max_error_ratio'])
        if record['case'] in rows:
            key = (record['case'], record['variant'], record['trial'])
            if key in records:
                raise ValueError('duplicate formal process')
            records[key] = record
    if len(checks) != 378 or len(records) != 360:
        raise ValueError('all 18 prechecks and 360 formal processes required')

    # 2. Keep every cfg/mode pair and every trial, even when the model misses it.
    manifest = clock.clock.read_json(run / 'paired-work.json')
    frozen = clock.clock.read_json(clock_path)
    frozen_sha = clock.clock.sha256(clock_path)
    if frozen_sha != manifest['old_clock_model_sha256']:
        raise ValueError('source-clock parameters differ from the pre-run frozen choice')
    model = frozen['model']
    points = {cid: work_point(row, setups[cid]) for cid, row in rows.items()}
    per_ends = {cid: {p['trial']: p for p in replay['cases'][cid]['per_process_ends']} for cid in rows}
    pairs, groups = [], []
    changes = ('plain_time_change', 'ends_time_change', 'ends_C_change', 'ends_f_change', 'ends_window_change', 'clock_f_ratio_error')
    for low, high in manifest['pairs']:
        a, b = low['case'], high['case']
        ra, rb = rows[a], rows[b]
        if (ra['config'], ra['input_mode'], ra['k'], rb['k']) != (rb['config'], rb['input_mode'], 1024, 4096):
            raise ValueError('equal-work pair identity changed')
        for item in (low, high):
            for key in ('tensor_cycles', 'source_kib', 'output_bytes', 'input_bytes'):
                if points[item['case']][key] != item[key]:
                    raise ValueError('software demand differs from pre-run manifest')
        if points[a]['tensor_mean_cycles'] != points[b]['tensor_mean_cycles'] or points[a]['source_mean_kib'] != points[b]['source_mean_kib']:
            raise ValueError('paired Q/S differs')
        if set(per_ends[a]) != set(range(10)) or set(per_ends[b]) != set(range(10)):
            raise ValueError('ten complete paired ends trials required')
        group = []
        for trial in range(10):
            pa, pb = per_ends[a][trial], per_ends[b][trial]
            pair = dict(config=ra['config'], input_mode=ra['input_mode'], baseline_case=a, target_case=b, trial=trial)
            for variant in ('plain', 'ends'):
                x, y = records[a, variant, trial], records[b, variant, trial]
                pair[variant + '_time_change'] = y['elapsed_us'] / x['elapsed_us'] - 1
                pair[variant + '_process_distance_s'] = abs(y['host_start_ns'] - x['host_start_ns']) / 1e9
            for field, name in (('cycles', 'C'), ('ghz', 'f'), ('window_us', 'window')):
                pair['ends_' + name + '_change'] = pb[field] / pa[field] - 1
            fa = clock.source_frequency(points[a], pa['window_us'], model)['frequency_ghz']
            fb = clock.source_frequency(points[b], pb['window_us'], model)['frequency_ghz']
            pair['clock_f_ratio_error'] = (fb / fa) / (pb['ghz'] / pa['ghz']) - 1
            group.append(pair)
        pairs.extend(group)
        groups.append(dict(config=ra['config'], input_mode=ra['input_mode'], baseline_case=a, target_case=b,
            tensor_mean_cycles=points[a]['tensor_mean_cycles'], source_mean_kib=points[a]['source_mean_kib'],
            output_bytes_ratio=points[b]['output_bytes'] / points[a]['output_bytes'],
            input_bytes_ratio=points[b]['input_bytes'] / points[a]['input_bytes'],
            paired_changes={key: distribution([p[key] for p in group]) for key in changes}))
    if len(groups) != 9 or len(pairs) != 90:
        raise ValueError('all nine groups and ninety trial pairs required')

    # 3. Evaluate only the original source clock at observed W; never fit or solve old C.
    cases, clock_processes = [], []
    for cid, row in rows.items():
        measured = replay['cases'][cid]
        fitted = clock.source_frequency(points[cid], measured['window_ends'], model)
        cases.append(dict(case=cid, config=row['config'], m=row['m'], k=row['k'],
            **points[cid], observed_window_us=measured['window_ends'], observed_ghz=measured['ghz_ends'],
            predicted_ghz=fitted['frequency_ghz'], frequency_error=fitted['frequency_ghz'] / measured['ghz_ends'] - 1,
            max_error_ratio=max(tolerances.get(cid, [0])), tolerance_processes=len(tolerances.get(cid, []))))
        for trial, observed in sorted(per_ends[cid].items()):
            pred = clock.source_frequency(points[cid], observed['window_us'], model)['frequency_ghz']
            clock_processes.append(dict(case=cid, config=row['config'], input_mode=row['input_mode'], k=row['k'], trial=trial,
                observed_window_us=observed['window_us'], observed_ghz=observed['ghz'], predicted_ghz=pred,
                frequency_error=pred / observed['ghz'] - 1))
    if clock.clock.sha256(clock_path) != frozen_sha:
        raise ValueError('frozen clock file changed')
    result = dict(run=str(run), environment=environment, diagnostic_source_sha256=clock.clock.sha256(Path(__file__)),
        source_clock_path=str(clock_path), source_clock_sha256=frozen_sha, coefficients=model['coefficients'], tau_us=model['tau_us'],
        input_hashes={name: clock.clock.sha256(run / name) for name in ('cases.json', 'paired-work.json', 'analysis/summary.json',
            'analysis/bridge_comparison.json', 'bridge_reference.json', 'transfer-receipt.json')},
        checks=dict(prechecks=18, formal_processes=360, numeric_checks_ok=len(checks), all_raw_sha256_verified=True,
            all_cases_independently_replayed=True, random_processes=sum(map(len, tolerances.values())),
            max_error_ratio=max(v for values in tolerances.values() for v in values), paired_groups=9, paired_trials=90),
        bridge=dict(reference_run=str(reference_run), archived_scope=archived_bridge['scope'],
            effective_scope='All nine listed cfg_a/b/c 8192-square K1024 conditions; archived cfg_b-only wording is stale and unchanged.',
            cases=bridge, ranges={key: distribution([r[key] for r in bridge]) for key in bridge[0] if key.endswith('_change')}),
        groups=groups, clock_cases=cases,
        frequency_errors_by_k={str(k): rms([p['frequency_error'] for p in cases if p['k'] == k]) for k in (1024, 4096)},
        notes=[
            'All nine cfg/mode pairs and ten same-trial comparisons are retained. Same trial is a randomized round, not simultaneous execution.',
            'Q/S are identical within each pair; output bytes fall to one quarter while unique input footprint doubles. K, tile counts and revisit also change.',
            'The pre-run source-clock SHA is checked and no coefficients, tau, candidate family, C/F/kappa or proxy definitions are fitted here.',
            'Frequency evaluation uses observed ends W only. No free time prediction or target-C substitution is performed.',
            'Cmax, frequency aggregation and cross-CTA ends envelope have different scopes; paired C/f need not reconstruct the complete event ratio.',
            'Ordinary all-zero inputs are not R18 partial-zero or boundary activity patterns; this batch cannot establish their mechanism.',
            'No automatic bridge correction, old-data pooling, new GPU measurement, or physical output-power attribution.',
        ])
    if (output / 'equal-work-clock.json').exists():
        raise FileExistsError(output / 'equal-work-clock.json')
    (output / 'equal-work-clock.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, table in [('bridge.csv', bridge), ('paired-trials.csv', pairs), ('clock-cases.csv', cases), ('clock-processes.csv', clock_processes)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print('checks:', result['checks'])
    print('frozen clock errors:', result['frequency_errors_by_k'])
    for group in groups:
        print(group['config'], group['input_mode'], {key: value['median'] for key, value in group['paired_changes'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--clock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new C directory containing an independent r09_analyze replay/')
    args = parser.parse_args()
    analyze(args.run.resolve(), args.reference_run.resolve(), args.clock.resolve(), args.output.resolve())
