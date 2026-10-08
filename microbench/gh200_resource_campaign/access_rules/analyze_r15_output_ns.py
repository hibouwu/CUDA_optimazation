#!/usr/bin/env python3
"""R15 single-tile issuer store windows from direct globaltimer endpoints.

Expected shared profile: r15-first-output-ns, unchanged 784 words/CTA,
head[13:15] = thread256 tile0 store-enter/store-return globaltimer.
No GPU runner or clock-to-time fallback; old V08 traces are not this measurement.
"""
import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_r15 import overlap_windows, sha
from analyze_r18 import replay

PROFILE = 'r15-first-output-ns'
WIDTH = 16 + 2 * 64 * 6
med = statistics.median


def direct_windows(setup, call):
    """Only this call's direct ns are used to place output windows across SMs."""
    if setup['trace_version'] != PROFILE:
        raise ValueError('direct issuer globaltimer profile required; no affine fallback')
    if setup['config'] != 'cfg_c' or setup['trace_words'] != WIDTH:
        raise ValueError('expected cfg_c with the unchanged 784-word layout')
    words = call['trace']
    if len(words) != setup['grid'][0] * setup['grid'][1] * setup['grid'][2] * WIDTH:
        raise ValueError('trace size differs from grid')
    origin = min(words[i + 1] for i in range(0, len(words), WIDTH))
    windows = []
    for i in range(0, len(words), WIDTH):
        h = words[i:i + 16]
        if h[6] != 1 or h[9] != 1:
            raise ValueError('this profile analysis requires exactly one tile per consumer')
        if h[2] != h[11] or h[2] != h[12] or h[10]:
            raise ValueError('CTA SM changed or trace overflowed')
        if not 0 < h[1] <= h[13] < h[14] <= h[8] or h[7] <= h[0]:
            raise ValueError('missing or unordered direct issuer timestamps')
        a, b = i + 16, i + 16 + 64 * 6
        role1, role2 = words[a:a + 6], words[b:b + 6]
        mi, ni = role2[4] - 1, role2[5] - 1
        if role1[4:] != role2[4:] or not (0 <= mi * 256 <= setup['m'] - 256 and 0 <= ni * 128 <= setup['n'] - 128):
            raise ValueError('expected matching, fully valid output tile coordinates')
        windows.append(dict(cta=i // WIDTH, sm=h[2] - 1, mi=mi, ni=ni,
            entry_ns=h[1] - origin, final_ns=h[8] - origin,
            issuer_store_enter_ns=h[13] - origin, issuer_store_return_ns=h[14] - origin,
            issuer_store_ns=h[14] - h[13], issuer_store_cycles=role2[3] - role2[2],
            merged_store_cycles=max(role1[3], role2[3]) - max(role1[2], role2[2]),
            tail_after_store_ns=h[8] - h[14],
            entry_cycle=h[0], final_cycle=h[7], permit_cycle=role2[2], done_cycle=role2[3]))
    return windows


def summarize_call(windows):
    direct = [(w['issuer_store_enter_ns'], w['issuer_store_return_ns']) for w in windows]
    rates = [(w['final_cycle'] - w['entry_cycle']) / (w['final_ns'] - w['entry_ns']) for w in windows]
    common_rate = med(rates)
    estimates = dict(direct=direct, affine=[], entry_anchor=[], final_anchor=[])
    for w, rate in zip(windows, rates):
        estimates['affine'].append(tuple(w['entry_ns'] + (w[k] - w['entry_cycle']) / rate for k in ('permit_cycle', 'done_cycle')))
        estimates['entry_anchor'].append(tuple(w['entry_ns'] + (w[k] - w['entry_cycle']) / common_rate for k in ('permit_cycle', 'done_cycle')))
        estimates['final_anchor'].append(tuple(w['final_ns'] - (w['final_cycle'] - w[k]) / common_rate for k in ('permit_cycle', 'done_cycle')))
    overlap = {name: overlap_windows(intervals) for name, intervals in estimates.items()}
    for i, w in enumerate(windows):
        w['overlap'] = {name: values[i] for name, values in overlap.items()}
    changes = defaultdict(int)
    for begin, end in direct:
        changes[begin] += 1
        changes[end] -= 1
    active = peak = 0
    for t in sorted(changes):
        active += changes[t]
        peak = max(peak, active)
    return dict(ctas=len(windows), unique_sms=len({w['sm'] for w in windows}), peak_overlap=peak,
        store_start_spread_ns=max(a for a, b in direct) - min(a for a, b in direct),
        **{key: med(w[key] for w in windows) for key in ('issuer_store_ns', 'issuer_store_cycles', 'merged_store_cycles', 'tail_after_store_ns')},
        **{'overlap_' + name: med(values) for name, values in overlap.items()}, windows=windows)


def analyze(root, output):
    cases, processes, failed = [], [], []
    for row in json.loads((root / 'cases.json').read_text()):
        times = {variant: [] for variant in ('plain', 'stamped', 'ends', 'global')}
        observations = []
        for path in sorted((root / 'samples' / row['id']).glob('*.json')):
            record = json.loads(path.read_text())
            if record['returncode']:
                failed.append(dict(path=str(path.relative_to(root)), returncode=record['returncode']))
                continue
            # Existing numerical/coordinate checks also support the new shared input controls.
            variant = record['variant']
            replay(root, record if variant in ('stamped', 'global') else dict(record, variant='plain'), row)
            times[variant].append(record['elapsed_us'])
            if variant == 'global':
                with gzip.open(root / record['raw'], 'rt') as stream:
                    events = {e['event']: e for line in stream if line.strip() for e in [json.loads(line)]}
                observation = summarize_call(direct_windows(events['setup'], events['call']))
                observation.update(case=row['id'], trial=record['trial'], raw=record['raw'])
                processes.append(observation)
                observations.append(observation)
        if not all(times.values()):
            raise ValueError('plain/stamped/ends/global samples required: ' + row['id'])
        keys = ('issuer_store_ns', 'issuer_store_cycles', 'merged_store_cycles', 'tail_after_store_ns',
                'store_start_spread_ns', 'peak_overlap', 'overlap_direct', 'overlap_affine', 'overlap_entry_anchor', 'overlap_final_anchor')
        cases.append(dict(case=row['id'], m=row['m'], n=row['n'], k=row['k'], processes=len(observations),
            elapsed_us={variant: med(values) for variant, values in times.items()},
            global_relative={variant: med(times['global']) / med(times[variant]) - 1 for variant in ('plain', 'stamped', 'ends')},
            cv={variant: statistics.pstdev(values) / statistics.mean(values) for variant, values in times.items()},
            median={key: med(o[key] for o in observations) for key in keys},
            process_range={key: [min(o[key] for o in observations), max(o[key] for o in observations)] for key in keys}))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(profile=PROFILE, input=str(root), cases=cases, processes=processes, failed=failed,
        analyzer_sha256=sha(__file__), scope='Direct same-call issuer store enter/return windows, not physical TMA occupancy or global destination completion; affine values are comparison estimates only.')
    (output / 'output-ns.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(cases)} cases, {len(processes)} direct-ns calls -> {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.input.resolve(), args.output.resolve())
