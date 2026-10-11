#!/usr/bin/env python3
"""Replay the R09 public batch, compare zero/dyadic, and carry old V08 h06 forward diagnostically."""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import r09_analyze
import v06_run as common
import v08_model as model


def distribution(values):
    return dict(n=len(values), median=st.median(values), minimum=min(values), maximum=max(values),
                cv=st.pstdev(values) / abs(st.mean(values)) if st.mean(values) else None)


def same_summary(a, b):
    # ARM and x86 variance evaluation can differ by a few float ulps.
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_summary(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_summary(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def analyze(run, v08, output):
    # This replays all 360 successful formal records with the public numeric check.
    r09_analyze.summarize_shared(run, output)
    summary = json.loads((output / 'summary.json').read_text())
    archived = json.loads((run / 'analysis/summary.json').read_text())
    if not same_summary(summary['cases'], archived['cases']):
        raise ValueError('replay differs from archived summary')
    for filename, digest in json.loads((run / 'build/sass_hashes.json').read_text()).items():
        if common.sha(run / 'build' / filename) != digest:
            raise ValueError('SASS identity changed')
    rows = {r['id']: r for r in json.loads((run / 'cases.json').read_text())}
    setups = {r['case']: r['setup'] for r in json.loads((run / 'static_setup.json').read_text())}
    sampling = json.loads((run / 'sampling.json').read_text())
    failed = {r['case'] for r in sampling['numeric_failed']}
    processes = []
    for cid, row in rows.items():
        if cid in failed:
            # Failed numeric raw data are diagnosed by r09_input_error.py, not timed here.
            continue
        for path in sorted((run / 'samples' / cid).glob('*.json')):
            record = json.loads(path.read_text())
            if record['returncode']:
                continue  # summarize_shared already requires ten successful processes per variant.
            variant = record['variant']
            p = dict(case=cid, config=row['config'], k=row['k'], input_mode=row['input_mode'],
                     variant=variant, trial=record['trial'], host_start_ns=record['host_start_ns'],
                     event_us=record['elapsed_us'])
            if variant != 'plain':
                ctas = model.observe(run, record, row, setups[cid])['ctas']
                count = lambda c: c['tiles'] if isinstance(c['tiles'], int) else len(c['tiles'])
                active = [c for c in ctas if count(c)]
                most = max(map(count, active))
                frequency = st.median((c['end_c'] - c['entry_c']) / (c['end_ns'] - c['entry_ns'])
                                      for c in active if count(c) == most)
                cycles = max(c['end_c'] - c['entry_c'] for c in active)
                envelope = (max(c['end_ns'] for c in active) - min(c['entry_ns'] for c in ctas)) / 1000
                p.update(cycles=cycles, ghz=frequency, envelope_us=envelope,
                         event_minus_envelope_us=p['event_us'] - envelope,
                         cycle_frequency_proxy_us=cycles / (1000 * frequency))
                if row['config'] == 'cfg_b' and row['k'] == 20480 and variant == 'stamped':
                    critical = max(active, key=lambda c: c['end_c'] - c['entry_c'])
                    tiles = critical['tiles']
                    split = dict(
                        P0=critical['prod_c'] - critical['entry_c'],
                        S=tiles[0][0] - critical['prod_c'], L0=tiles[0][1] - tiles[0][0],
                        later_L=sum(t[1] - t[0] for t in tiles[1:]),
                        between_mainloops=sum(b[0] - a[1] for a, b in zip(tiles, tiles[1:])),
                        after_last_mainloop=critical['end_c'] - tiles[-1][1])
                    if sum(split.values()) != cycles:
                        raise ValueError('critical CTA cycle split does not close')
                    p.update({'critical_' + key: value for key, value in split.items()})
            processes.append(p)

    by_case = {}
    for cid in sampling['sampled_cases']:
        by_case[cid] = {}
        for variant in ('plain', 'stamped', 'ends'):
            ps = [p for p in processes if p['case'] == cid and p['variant'] == variant]
            keys = ['event_us'] if variant == 'plain' else [
                'event_us', 'cycles', 'ghz', 'envelope_us', 'event_minus_envelope_us', 'cycle_frequency_proxy_us']
            by_case[cid][variant] = {key: distribution([p[key] for p in ps]) for key in keys}

    comparisons = []
    for cfg in ('cfg_a', 'cfg_b', 'cfg_c'):
        dyadic, zero = (f'{cfg}_r09_k20480_{mode}' for mode in ('dyadic', 'zero'))
        for variant in ('plain', 'stamped', 'ends'):
            d = {p['trial']: p for p in processes if p['case'] == dyadic and p['variant'] == variant}
            z = {p['trial']: p for p in processes if p['case'] == zero and p['variant'] == variant}
            keys = ['event_us'] if variant == 'plain' else ['event_us', 'cycles', 'ghz', 'envelope_us', 'cycle_frequency_proxy_us']
            for key in keys:
                changes = [z[t][key] / d[t][key] - 1 for t in d]
                comparisons.append(dict(config=cfg, variant=variant, metric=key,
                    ratio_of_medians=st.median(z[t][key] for t in d) / st.median(d[t][key] for t in d) - 1,
                    same_trial_changes=distribution(changes), positive_trials=sum(x > 0 for x in changes),
                    paired_process_distance_s=distribution([abs(z[t]['host_start_ns'] - d[t]['host_start_ns']) / 1e9 for t in d])))

    frozen_path = v08 / 'frozen/v08-predictions.json'
    frozen = json.loads(frozen_path.read_text())
    transported = []
    for cfg in ('cfg_a', 'cfg_b', 'cfg_c'):
        p = frozen['predictions'][cfg + '_h06']
        s = summary['cases'][cfg + '_r09_k20480_dyadic']
        actual_frequency_us = p['fixed_us'] + p['converted_cycles'] / (1000 * s['ghz_ends'])
        transported.append(dict(config=cfg, old_predicted_us=p['predicted_us'], new_plain_us=s['plain_us'],
            old_clock_ghz=p['clock_ghz'], new_clock_ghz=s['ghz_ends'],
            unchanged_prediction_error=p['predicted_us'] / s['plain_us'] - 1,
            clock_error=p['clock_ghz'] / s['ghz_ends'] - 1,
            cycle_error=p['critical_cycles'] / s['c_max_stamped'] - 1,
            actual_frequency_time_us=actual_frequency_us,
            actual_frequency_error=actual_frequency_us / s['plain_us'] - 1))
    critical_split = {}
    for mode in ('dyadic', 'zero'):
        ps = [p for p in processes if p['config'] == 'cfg_b' and p['k'] == 20480
              and p['input_mode'] == mode and p['variant'] == 'stamped']
        critical_split[mode] = {key: st.mean(p[key] for p in ps) for key in ps[0]
                               if key.startswith('critical_') or key == 'cycles'}
    critical_split['zero_minus_dyadic'] = {
        key: critical_split['zero'][key] - critical_split['dyadic'][key] for key in critical_split['dyadic']}
    result = dict(environment=summary['environment'], formal_processes=len(processes),
        archived_summary_reproduced=True, planned_cases=len(rows), measured_cases=len(by_case),
        numeric_failed=list(sorted(failed)), per_case=by_case, zero_vs_dyadic=comparisons,
        cfg_b_stamped_critical_cycle_means=critical_split,
        old_v08_frozen_sha256=common.sha(frozen_path), old_v08_gpu=frozen['gpu'],
        old_v08_predictions_on_new_gpu=transported,
        notes=[
            'Old V08 predictions are carried to a different GPU without refitting; this is not same-card validation.',
            'Same trial means a randomized sampling round, not simultaneous or adjacent calls.',
            'cycle_frequency_proxy_us divides within a process before taking the median; it is not median(C)/median(f).',
            'The proxy still mixes Cmax with maximum-tile-CTA median frequency; use the direct envelope for measured time.',
            'Per-tile stamped intervals are SM cycles; the run has no per-tile globaltimer or power measurement.',
            'Critical cycle split selects the maximum-cycle CTA in each stamped process; means preserve additive closure.',
        ])
    common.write_json(output / 'comparison.json', result)
    with (output / 'processes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for p in processes for k in p)))
        writer.writeheader()
        writer.writerows(processes)
    print('replayed formal processes:', len(processes), '; output:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--v08', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.v08.resolve(), args.output.resolve())
