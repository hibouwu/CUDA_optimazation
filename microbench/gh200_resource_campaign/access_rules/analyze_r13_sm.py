#!/usr/bin/env python3
"""R13 cfg_a pitch pairs at requested SM counts 32/64/96/132 (CPU only).

CTA entry/final globaltimer windows measure observed CTA overlap, not mainloop
concurrency. Tile L remains a same-SM cycle interval including wait and drain.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics as stats

from analyze_r18 import replay
from v08_model import scheduled_work
from v06_run import sha


def select_attempts(root, case, variants=('plain', 'stamped')):
    """Retain every failure; never count retries as independent processes."""
    successes = defaultdict(list)
    failed, ignored = [], []
    for path in sorted((root / 'samples' / case).glob('*.json')):
        record = json.loads(path.read_text())
        if record['case'] != case:
            raise ValueError('sample case mismatch: ' + str(path))
        item = dict(path=str(path.relative_to(root)), record=record)
        if record['returncode']:
            failed.append(item)
        elif record['variant'] in variants:
            successes[record['variant'], record['trial']].append(item)
        else:
            ignored.append(item['path'])
    selected = []
    for key, attempts in sorted(successes.items()):
        if len(attempts) != 1:
            raise ValueError(f'{case} {key}: multiple successful attempts')
        selected.append(attempts[0])
    return selected, failed, ignored


def validate_setup(row, static, sample):
    for key in ('config', 'm', 'n', 'k', 'lda', 'ldb', 'ldd', 'storage_m',
                'storage_n', 'zero_m', 'zero_n', 'swizzle', 'evict', 'grid',
                'tile', 'cluster', 'stages', 'smem', 'max_active_ctas_per_sm',
                'sm_count', 'requested_sm_count', 'scheduler_sm_count', 'input_mode', 'seed'):
        if sample.get(key) != static.get(key):
            raise ValueError(row['id'] + ': sample/static setup mismatch: ' + key)
    actual, expected = sample.get('gpu_uuid'), static.get('gpu_uuid')
    if 'sm_count' in row and (not actual or not expected):
        raise ValueError(row['id'] + ': new SM sweep requires sample and static GPU UUID')
    if actual != expected:
        raise ValueError(row['id'] + ': sample/static GPU UUID mismatch')
    return actual or 'unknown'


def overlap_windows(ctas):
    """Union windows on each SM before sweeping half-open globaltimer intervals."""
    by_sm = defaultdict(list)
    for c in ctas:
        if c['end_ns'] <= c['entry_ns'] or c['end_c'] <= c['entry_c']:
            raise ValueError('nonpositive CTA observation window')
        by_sm[c['sm']].append((c['entry_ns'], c['end_ns']))
    unions, events = [], Counter()
    for sm, windows in sorted(by_sm.items()):
        merged = []
        for start, end in sorted(windows):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        unions.append(dict(sm=sm, intervals_ns=merged))
        for start, end in merged:
            events[start] += 1
            events[end] -= 1
    segments, histogram = [], Counter()
    times = sorted(events)
    active = 0
    for i, start in enumerate(times[:-1]):
        active += events[start]
        end = times[i + 1]
        histogram[active] += end - start
        segments.append(dict(start_ns=start, end_ns=end, observed_sms=active))
    span = times[-1] - times[0]
    return dict(scope='CTA entry to final consumer release; not mainloop concurrency or exact residency',
                distinct_smids=len(by_sm), peak_observed_sms=max(histogram),
                mean_observed_sms=sum(n * ns for n, ns in histogram.items()) / span,
                envelope_ns=span, duration_by_observed_sms_ns=dict(sorted(histogram.items())),
                sm_unions=unions, segments=segments)


def cross_scale(pair_tiles):
    """Difference of pitch penalties at identical (trial,j,T,mi,ni), 132 minus 96."""
    lookup = defaultdict(dict)
    for p in pair_tiles:
        if p['comparison_sm_count'] in (96, 132):
            lookup[p['pitch'], p['trial'], p['comparison_sm_count']][
                p['j'], p['T'], p['mi'], p['ni']] = p
    details, groups = [], []
    for pitch, trial in sorted({(p, t) for p, t, _ in lookup}):
        a, b = lookup.get((pitch, trial, 96)), lookup.get((pitch, trial, 132))
        if not a or not b:
            groups.append(dict(pitch=pitch, trial=trial, status='missing_96_or_132_pair',
                               j=None, T=None, matched_tiles=0, qualified=False))
            continue
        for j, total in sorted({k[:2] for k in a} | {k[:2] for k in b}):
            common = sorted(k for k in a.keys() & b.keys() if k[:2] == (j, total))
            current = []
            for key in common:
                low, high = a[key], b[key]
                item = dict(pitch=pitch, trial=trial, j=j, T=total, mi=key[2], ni=key[3],
                            cta96=low['cta'], cta132=high['cta'],
                            penalty96_per_kt=low['delta_per_kt'], penalty132_per_kt=high['delta_per_kt'],
                            delta_penalty_per_kt=high['delta_per_kt'] - low['delta_per_kt'],
                            delta_relative=high['relative'] - low['relative'],
                            qualified=low['qualified'] and high['qualified'])
                details.append(item)
                current.append(item)
            group = dict(pitch=pitch, trial=trial, j=j, T=total, matched_tiles=len(common),
                         status='matched_diagnostic' if common else 'no_common_coordinates',
                         qualified=bool(current) and all(x['qualified'] for x in current))
            if current:
                group.update(delta_penalty_per_kt=stats.mean(x['delta_penalty_per_kt'] for x in current),
                             delta_relative=stats.mean(x['delta_relative'] for x in current))
            groups.append(group)
    return details, groups


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output, case_ids=None):
    if output.exists():
        raise ValueError('output must be a new directory')
    rows = json.loads((root / 'cases.json').read_text())
    if case_ids:
        rows = [r for r in rows if r['id'] in case_ids]
        if {r['id'] for r in rows} != set(case_ids):
            raise ValueError('unknown requested case')
    if not rows:
        raise ValueError('no cases')
    setups = {x['case']: x['setup'] for x in json.loads((root / 'static_setup.json').read_text())}
    cases, processes, tiles, groups = [], [], [], []
    failed, ignored, inputs = [], [], []
    for row in rows:
        if (row['config'], row['m'], row['n'], row['k'], row['swizzle']) != ('cfg_a', 2304, 3072, 4096, 1):
            raise ValueError('R13 SM analysis requires cfg_a 2304x3072x4096 swizzle=1; use --cases for legacy runs')
        pitch = {(4096, 3072): 'aligned', (4104, 3072): 'a16', (4096, 3080): 'b16'}.get((row['lda'], row['ldb']))
        if pitch is None or row.get('pitch', pitch) != pitch:
            raise ValueError('unknown or inconsistent pitch')
        static = setups[row['id']]
        comparison_sm = row.get('sm_count') or static.get('scheduler_sm_count') or static['sm_count']
        case = dict(case=row['id'], pitch=pitch, comparison_sm_count=comparison_sm,
                    requested_sm_count=row.get('sm_count', 0), static_grid=static['grid'],
                    static_gpu_uuid=static.get('gpu_uuid', 'unknown'))
        selected, failures, omitted = select_attempts(root, row['id'])
        failed.extend(failures)
        ignored.extend(omitted)
        current_processes = []
        for item in selected:
            record = item['record']
            with gzip.open(root / record['raw'], 'rt') as stream:
                events = {event['event']: event for line in stream if line.strip() for event in [json.loads(line)]}
            setup = events['setup']
            uuid = validate_setup(row, static, setup)
            observed = replay(root, record, row)
            inputs.append(dict(path=item['path'], raw=record['raw'], raw_sha256=record['raw_sha256']))
            process = dict(case=row['id'], pitch=pitch, comparison_sm_count=comparison_sm,
                           variant=record['variant'], trial=record['trial'], attempt=record.get('attempt', 0),
                           actual_grid=setup['grid'], requested_sm_count=setup.get('requested_sm_count', 0),
                           scheduler_sm_count=setup.get('scheduler_sm_count', setup['sm_count']),
                           gpu_uuid=uuid, elapsed_us=observed['elapsed_us'], checked_values=observed['checked_values'],
                           raw=record['raw'], raw_sha256=record['raw_sha256'])
            if record['variant'] == 'stamped':
                ctas = observed['ctas']
                expected = scheduled_work(row['config'], row['m'], row['n'], setup['grid'], row['swizzle'])
                if [[tuple(coord) for coord in c['work']] for c in ctas] != expected:
                    raise ValueError('recorded coordinates differ from scheduler')
                process['overlap'] = overlap_windows(ctas)
                process['cta_windows'] = [dict(cta=c['cta'], sm=c['sm'], T=len(c['tiles']),
                                               entry_c=c['entry_c'], end_c=c['end_c'],
                                               entry_ns=c['entry_ns'], end_ns=c['end_ns'],
                                               window_ghz=(c['end_c'] - c['entry_c']) / (c['end_ns'] - c['entry_ns']))
                                          for c in ctas]
                process['tile_count_histogram'] = dict(Counter(len(c['tiles']) for c in ctas))
                per_group = defaultdict(list)
                for c in ctas:
                    total = len(c['tiles'])
                    for j, (event, (mi, ni)) in enumerate(zip(c['tiles'], c['work'])):
                        value = (event[1] - event[0]) / 64
                        tiles.append(dict(case=row['id'], pitch=pitch, comparison_sm_count=comparison_sm,
                                          trial=record['trial'], cta=c['cta'], sm=c['sm'], j=j, T=total,
                                          mi=mi, ni=ni, first_mma_c=event[0], main_end_c=event[1], L_per_kt=value))
                        per_group[j, total].append(value)
                for (j, total), values in sorted(per_group.items()):
                    groups.append(dict(case=row['id'], trial=record['trial'], j=j, T=total,
                                       windows=len(values), mean_L_per_kt=stats.mean(values)))
            current_processes.append(process)
            processes.append(process)
        values = {v: [p['elapsed_us'] for p in current_processes if p['variant'] == v] for v in ('plain', 'stamped')}
        trials = {v: {p['trial'] for p in current_processes if p['variant'] == v} for v in values}
        unresolved = sorted({(x['record']['variant'], x['record']['trial']) for x in failures
                             if x['record']['variant'] in trials
                             and x['record']['trial'] not in trials[x['record']['variant']]})
        for variant, measurements in values.items():
            case[variant + '_processes'] = len(measurements)
            case[variant + '_us'] = stats.median(measurements) if measurements else None
            case[variant + '_cv'] = stats.pstdev(measurements) / stats.mean(measurements) if len(measurements) > 1 else None
        case['perturbation'] = case['stamped_us'] / case['plain_us'] - 1 if all(values.values()) else None
        case['unresolved_failed_trials'] = [dict(variant=v, trial=t) for v, t in unresolved]
        case['plain_qualified'] = (len(values['plain']) >= 10 and case['plain_cv'] <= .05
                                   and not any(v == 'plain' for v, _ in unresolved))
        case['trace_qualified'] = (case['plain_qualified'] and len(values['stamped']) >= 10
                                   and trials['plain'] == trials['stamped'] and case['stamped_cv'] <= .05
                                   and abs(case['perturbation']) <= .05 and not unresolved)
        case['sampling'] = 'formal_count' if min(map(len, values.values())) >= 10 else 'pilot_or_incomplete_diagnostic'
        case['plain_only_trials'] = sorted(trials['plain'] - trials['stamped'])
        case['stamped_only_trials'] = sorted(trials['stamped'] - trials['plain'])
        cases.append(case)
    if len({c['static_gpu_uuid'] for c in cases}) > 1:
        raise ValueError('selected cases do not share one GPU identity')
    by_case = {c['case']: c for c in cases}
    for tile in tiles:
        tile['qualified'] = by_case[tile['case']]['trace_qualified']
    group_summary = []
    for case in cases:
        by_group = defaultdict(list)
        for group in groups:
            if group['case'] == case['case']:
                by_group[group['j'], group['T']].append(group)
        for (j, total), observations in sorted(by_group.items()):
            group_summary.append(dict(case=case['case'], j=j, T=total, processes=len(observations),
                                      median_process_mean_L_per_kt=stats.median(g['mean_L_per_kt'] for g in observations),
                                      qualified=case['trace_qualified']))

    case_map = {}
    for case in cases:
        key = case['comparison_sm_count'], case['pitch']
        if key in case_map:
            raise ValueError('multiple cases for one SM/pitch pair')
        case_map[key] = case
    tile_map = defaultdict(dict)
    for tile in tiles:
        tile_map[tile['case'], tile['trial']][tile['cta'], tile['j'], tile['T'], tile['mi'], tile['ni']] = tile
    pair_tiles, pair_groups, missing_pairs = [], [], []
    for (sm, pitch), case in sorted(case_map.items()):
        if pitch == 'aligned':
            continue
        control = case_map.get((sm, 'aligned'))
        if control is None:
            missing_pairs.append(dict(case=case['case'], reason='missing_aligned_case'))
            continue
        trials = sorted({t for c, t in tile_map if c in (case['case'], control['case'])})
        for trial in trials:
            a, b = tile_map.get((control['case'], trial)), tile_map.get((case['case'], trial))
            if not a or not b:
                missing_pairs.append(dict(case=case['case'], trial=trial, reason='missing_stamped_trial'))
                continue
            if a.keys() != b.keys():
                raise ValueError('same-SM pitch pair changed CTA/j/T/coordinates')
            current = defaultdict(list)
            for key in sorted(a):
                low, high = a[key], b[key]
                pair = dict(case=case['case'], baseline=control['case'], pitch=pitch, comparison_sm_count=sm,
                            trial=trial, cta=key[0], j=key[1], T=key[2], mi=key[3], ni=key[4],
                            baseline_L_per_kt=low['L_per_kt'], pitched_L_per_kt=high['L_per_kt'],
                            delta_per_kt=high['L_per_kt'] - low['L_per_kt'], relative=high['L_per_kt'] / low['L_per_kt'] - 1,
                            qualified=case['trace_qualified'] and control['trace_qualified'])
                pair_tiles.append(pair)
                current[key[1], key[2]].append(pair)
            for (j, total), pairs in sorted(current.items()):
                pair_groups.append(dict(case=case['case'], pitch=pitch, comparison_sm_count=sm,
                                        trial=trial, j=j, T=total, matched_tiles=len(pairs),
                                        mean_delta_per_kt=stats.mean(p['delta_per_kt'] for p in pairs),
                                        relative=stats.mean(p['pitched_L_per_kt'] for p in pairs) / stats.mean(p['baseline_L_per_kt'] for p in pairs) - 1,
                                        qualified=all(p['qualified'] for p in pairs)))
    pair_summary = []
    paired = defaultdict(list)
    for p in pair_groups:
        paired[p['case'], p['j'], p['T']].append(p)
    for (case, j, total), observations in sorted(paired.items()):
        pair_summary.append(dict(case=case, j=j, T=total, processes=len(observations),
                                 median_process_mean_delta_per_kt=stats.median(p['mean_delta_per_kt'] for p in observations),
                                 median_process_relative=stats.median(p['relative'] for p in observations),
                                 qualified=len(observations) >= 10 and all(p['qualified'] for p in observations)))
    cross_tiles, cross_groups = cross_scale(pair_tiles)
    cross_summary = []
    matched = defaultdict(list)
    for group in cross_groups:
        matched[group['pitch'], group['j'], group['T'], group['status']].append(group)
    for (pitch, j, total, status), observations in matched.items():
        summary = dict(pitch=pitch, j=j, T=total, status=status, processes=len(observations),
                       matched_tiles_per_process=sorted({g['matched_tiles'] for g in observations}),
                       qualified=len(observations) >= 10 and all(g['qualified'] for g in observations))
        if status == 'matched_diagnostic':
            summary.update(median_process_mean_delta_penalty_per_kt=stats.median(g['delta_penalty_per_kt'] for g in observations),
                           median_process_mean_delta_relative=stats.median(g['delta_relative'] for g in observations))
        cross_summary.append(summary)
    result = dict(run=str(root), analyzer_sha256=sha(Path(__file__)),
                  cases_sha256=sha(root / 'cases.json'), static_setup_sha256=sha(root / 'static_setup.json'),
                  scope='Matched pitch differences; grid response cannot isolate local/shared supply or prior cache history.',
                  overlap_scope='CTA entry/final globaltimer windows only; no per-tile cross-SM concurrency is inferred.',
                  clock_scope='window_ghz=(end_c-entry_c)/(end_ns-entry_ns) on the same CTA/SM; interval average only.',
                  aggregation='Arithmetic mean across CTA windows within each process and (j,T), then median across processes.',
                  qualification='At least 10 plain/stamped processes, each CV <=5%, paired trial sets, |stamped/plain-1| <=5%; pilot values remain diagnostic.',
                  cases=cases, mainloop_groups=group_summary, mainloop_process_groups=groups,
                  pitch_pairs=pair_summary, pitch_process_pairs=pair_groups, missing_pairs=missing_pairs,
                  cross_96_132=cross_summary, cross_process_groups=cross_groups,
                  failed_attempts=failed, ignored_successful_variants=ignored, selected_inputs=inputs,
                  checked_values=sum(p['checked_values'] for p in processes))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (output / 'processes.json').write_text(json.dumps(processes, indent=2, allow_nan=False) + '\n')
    write_csv(output / 'tiles.csv', tiles)
    write_csv(output / 'pairs.csv', pair_tiles)
    write_csv(output / 'cross-pairs.csv', cross_tiles)
    print(json.dumps(dict(cases=len(cases), processes=len(processes), checked_values=result['checked_values'],
                          failed_attempts=len(failed), trace_qualified=sum(c['trace_qualified'] for c in cases), output=str(output))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    analyze(args.run.resolve(), args.output.resolve(), args.cases)
