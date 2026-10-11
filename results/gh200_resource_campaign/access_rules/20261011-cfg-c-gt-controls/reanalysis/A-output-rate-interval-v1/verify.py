"""Independent verification of all raw-selected within-process output pairs."""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction as Q
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import tarfile


def require(value):
    if not value:
        raise ValueError('independent verification failed')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    frozen = read(args.inputs)
    for name, expected in frozen['run_files'].items():
        require(sha(args.root / name) == expected)
    for name, expected in frozen['program_files'].items():
        require(sha(args.inputs.parent / name) == expected)
    freeze = read(args.root / 'freeze.json')['files']
    for name, expected in freeze.items():
        require(sha(args.root / name) == expected)
    archive_counts = {}
    for name, base in [('input.tar.gz', args.root), ('output.tar.gz', args.root / 'returned')]:
        count = 0
        with tarfile.open(args.root / name) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    require(hashlib.sha256(archive.extractfile(member).read()).hexdigest() ==
                            sha(base / member.name.removeprefix('./')))
                    count += 1
        archive_counts[name] = count
    records = read(args.root / 'returned/records.json')
    enabled = [r for r in records if r['flag'] == 1]
    require(len(records) == 40 and [r['id'] for r in enabled] ==
            ['off_on_pair%02d_arm1' % n for n in range(10)])
    packets = {}
    for record in enabled:
        require(record['sha256'] == sha(args.root / 'returned' / record['raw']))
        with gzip.open(args.root / 'returned' / record['raw'], 'rt') as stream:
            setup, call, numeric = [json.loads(s) for s in stream]
        require(setup['gpu_uuid'] == frozen['gpu_uuid'] and setup['gt_bracket_observer'] ==
                call['gt_bracket_observer'] == 1)
        packets[record['id']] = call
    report = read(args.analysis / 'audit.json')
    rows = [json.loads(s) for s in (args.analysis / 'windows.jsonl').read_text().splitlines()]
    pairs = [json.loads(s) for s in (args.analysis / 'phase-pairs.jsonl').read_text().splitlines()]

    def get_stamp(call, cta, wg, ordinal, event):
        i = cta * 1296 + 16 + wg * 640 + ordinal * 10 + event
        j = cta * 9 + wg * 4 + ordinal * 2 + event
        return dict(wg=wg, physical_thread=128 * (wg + 1), trace_cycle_word=i,
            trace_gt_after_word=i + 6, shadow_gt_before_word=j,
            cycle=call['trace'][i], gt_before=call['gt_shadow'][j], gt_after=call['trace'][i + 6])

    def verify_row(row):
        call = packets[row['process']]
        cta, ordinal = row['cta'], row['output_ordinal']
        start = [get_stamp(call, cta, w, ordinal, 0) for w in (0, 1)]
        end = [get_stamp(call, cta, w, ordinal, 1) for w in (0, 1)]
        if row['scope'] == 'common':
            require(row['wg'] is None)
            a = start[0] if start[0]['cycle'] <= start[1]['cycle'] else start[1]
            b = end[0] if end[0]['cycle'] >= end[1]['cycle'] else end[1]
        else:
            require(row['scope'] == 'WG' and type(row['wg']) is int and row['wg'] in (0, 1))
            a, b = start[row['wg']], end[row['wg']]
        require(row['begin'] == a and row['end'] == b)
        require(row['sm'] == call['trace'][cta * 1296 + 2] - 1)
        C, lo, hi = b['cycle'] - a['cycle'], b['gt_before'] - a['gt_after'], b['gt_after'] - a['gt_before']
        require(0 < lo <= hi and C > 0 and row['cycles'] == C)
        require(row['elapsed_lower_ns'] == lo and row['elapsed_upper_ns'] == hi)
        require(Q(row['rate_lower']) == Q(C, hi) and Q(row['rate_upper']) == Q(C, lo))
        require(row['post_gt_elapsed_ns'] == b['gt_after'] - a['gt_after'])
        require(row['post_gt_minus_selected_elapsed_lower_ns'] == a['gt_before'] - a['gt_after'])
        require(row['post_gt_minus_selected_elapsed_upper_ns'] == b['gt_after'] - b['gt_before'])
        linear = 2 * (cta % 66) + cta // 66 + 132 * ordinal
        coord = [(linear // 2) % 9, 2 * (linear // 18) + cta // 66]
        require(linear < 216 and row['work'] == coord)
        for wg in (0, 1):
            offset = cta * 1296 + 16 + wg * 640 + ordinal * 10
            require(call['trace'][offset + 4:offset + 6] == [v + 1 for v in coord])

    for row in rows:
        verify_row(row)
    expected = set()
    for process in packets:
        for cta in range(132):
            count = 2 if 2 * (cta % 66) + cta // 66 < 84 else 1
            for ordinal in range(count):
                for scope, wg in [('WG', 0), ('WG', 1), ('common', None)]:
                    expected.add((process, scope, wg, cta, ordinal))
    require({(r['process'], r['scope'], r['wg'], r['cta'], r['output_ordinal']) for r in rows} == expected)
    require(len(rows) == len(expected) == 6480)

    def verify_pair(pair):
        a, b = [rows[pair[k]] for k in ('first_row', 'second_row')]
        for k in ('process', 'scope', 'wg', 'cta', 'sm'):
            require(a[k] == b[k] == pair[k])
        require(a['output_ordinal'] == 0 and b['output_ordinal'] == 1)
        low = max(Q(a['cycles'], a['elapsed_upper_ns']), Q(b['cycles'], b['elapsed_upper_ns']))
        high = min(Q(a['cycles'], a['elapsed_lower_ns']), Q(b['cycles'], b['elapsed_lower_ns']))
        require(Q(pair['shared_rate_lower']) == low and Q(pair['shared_rate_upper']) == high)
        require(pair['constant_average_rate_feasible'] is (low <= high))
        require(Q(pair['minimum_first_over_second_minus_one']) ==
                Q(a['cycles'] * b['elapsed_lower_ns'], b['cycles'] * a['elapsed_upper_ns']) - 1)

    for pair in pairs:
        verify_pair(pair)
    require(len(pairs) == 2520 and len({(p['process'], p['scope'], p['wg'], p['cta']) for p in pairs}) == 2520)
    require(all(not p['constant_average_rate_feasible'] for p in pairs))
    all_gaps = []
    require({r['process'] for r in report['per_process']} == set(packets))
    for summary in report['per_process']:
        group = [p for p in pairs if p['process'] == summary['process']]
        gaps = [Q(p['minimum_first_over_second_minus_one']) for p in group if p['scope'] == 'common']
        require(summary['first_second_pairs'] == len(group) == 252 and summary['common_pairs'] == len(gaps) == 84)
        require(summary['constant_average_rate_feasible_pairs'] == 0)
        require(summary['minimum_first_over_second_minus_one'] ==
                dict(minimum=str(min(gaps)), median=str(median(gaps)), maximum=str(max(gaps))))
        all_gaps.extend(gaps)
    require(len(all_gaps) == 840 and len(report['per_process']) == 10)
    require(report['common_minimum_first_over_second_minus_one'] ==
            dict(minimum=str(min(all_gaps)), median=str(median(all_gaps)), maximum=str(max(all_gaps))))
    widths = [r[k]['gt_after'] - r[k]['gt_before'] for r in rows if r['scope'] == 'WG' for k in ('begin', 'end')]
    require(report['stamp_width_histogram'] == {str(k): v for k, v in sorted(Counter(widths).items())})
    require(report['checked_run_SHA'] == len(frozen['run_files']) and report['frozen_files'] == len(freeze))
    require(report['archive_file_matches'] == archive_counts and report['job'] == 740161)
    require(report['enabled_processes'] == 10 and report['selected_windows'] == 6480 and
            report['first_second_pairs'] == 2520 and report['constant_average_rate_feasible_pairs'] == 0)
    require(report['WG_windows'] == report['WG_main_reader_intervals_matched'] == 4320 and
            report['common_windows'] == 2160 and report['stamps'] == 8640)
    for key in ['observation_qualified', 'prediction_qualified', 'hardware_sampling_enclosure',
                'physical_ns_accuracy', 'joint_feasibility_proved']:
        require(report[key] is False)
    wrong_label = deepcopy(next(r for r in rows if r['scope'] == 'common'))
    wrong_label['begin']['wg'] ^= 1
    wrong_bound = deepcopy(rows[0]); wrong_bound['elapsed_upper_ns'] += 32
    wrong_process = deepcopy(pairs[0]); wrong_process['process'] = 'off_on_pair01_arm1'
    wrong_feasible = deepcopy(pairs[0]); wrong_feasible['constant_average_rate_feasible'] = True
    rejected = 0
    for verifier, item in [(verify_row, wrong_label), (verify_row, wrong_bound),
                            (verify_pair, wrong_process), (verify_pair, wrong_feasible)]:
        try:
            verifier(item)
        except ValueError:
            rejected += 1
        else:
            raise ValueError('negative control accepted')
    print(json.dumps(dict(input_SHA=len(frozen['run_files']), frozen_files=len(freeze),
        archive_file_matches=archive_counts, processes=10, raw_selected_windows=len(rows),
        WG_stamps=len(widths), first_second_pairs=len(pairs), common_pairs=840,
        exact_arithmetic_recomputed=True, negative_controls_rejected=rejected,
        observation_qualified=False, prediction_qualified=False), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
