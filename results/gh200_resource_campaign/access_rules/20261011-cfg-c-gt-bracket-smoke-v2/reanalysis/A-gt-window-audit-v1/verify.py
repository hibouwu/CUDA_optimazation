"""Independent raw-index/integer-rational verification; imports no A analyzer."""
import argparse
from copy import deepcopy
from fractions import Fraction as Q
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
from statistics import median
import tarfile


def require(value):
    if not value:
        raise ValueError('independent verification failed')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--build-root', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--analysis', required=True, type=Path)
    args = parser.parse_args()
    inputs = read(args.inputs)
    checked = []
    for label, base in [('run', args.root), ('build', args.build_root)]:
        for name, expected in inputs[label + '_files'].items():
            require(digest(base / name) == expected)
            checked.append(dict(source=label, path=name, sha256=expected))
    for name, expected in inputs['program_files'].items():
        require(digest(args.inputs.parent / name) == expected)
    sources = read(args.build_root / 'source_hashes.json')
    for name, expected in sources.items():
        require(digest(args.build_root / name) == expected)
    freeze = read(args.root / 'freeze.json')['files']
    for name, expected in freeze.items():
        require(digest(args.root / name) == expected)
    archive_counts = {}
    for name, base in [('input.tar.gz', args.root), ('output.tar.gz', args.root / 'returned')]:
        n = 0
        with tarfile.open(args.root / name) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    require(hashlib.sha256(archive.extractfile(member).read()).hexdigest() ==
                            digest(base / member.name.removeprefix('./')))
                    n += 1
        archive_counts[name] = n
    with gzip.open(args.root / 'returned/raw/flag1.jsonl.gz', 'rt') as stream:
        setup, call, numeric = [json.loads(s) for s in stream]
    trace, shadow = call['trace'], call['gt_shadow']
    with gzip.open(args.root / 'returned/raw/flag0.jsonl.gz', 'rt') as stream:
        off_setup, off_call, off_numeric = [json.loads(s) for s in stream]
    require(off_setup['gt_bracket_observer'] == off_call['gt_bracket_observer'] == 0)
    require(len(off_call['gt_shadow']) == 1188 and not any(off_call['gt_shadow']))
    counts = Counter()
    for flag, packet in [(0, off_call), (1, call)]:
        for cta in range(132):
            first = 2 * (cta % 66) + cta // 66
            count = 2 if first < 84 else 1
            require(packet['trace'][1296 * cta + 6] == packet['trace'][1296 * cta + 9] == count)
            counts[(flag, count)] += 1
    rows = lines(args.analysis / 'windows.jsonl')
    report = read(args.analysis / 'audit.json')
    pairs = lines(args.analysis / 'phase-pairs.jsonl')
    require(report['identity']['checked_files'] == checked)
    require(report['identity']['archive_file_matches'] == archive_counts)
    require(report['identity']['build_source_files'] == len(sources) == 858)
    require(report['identity']['smoke_frozen_files'] == len(freeze))
    require(report['CTA_output_counts'] == {str(f): {str(n): counts[(f, n)] for n in (1, 2)} for f in (0, 1)})

    def raw_stamp(cta, wg, ordinal, event):
        i = 1296 * cta + 16 + 640 * wg + 10 * ordinal + event
        j = 9 * cta + 4 * wg + 2 * ordinal + event
        return dict(wg=wg, physical_thread=128 * (wg + 1), trace_cycle_word=i,
            trace_gt_after_word=i + 6, shadow_gt_before_word=j,
            cycle=trace[i], gt_before=shadow[j], gt_after=trace[i + 6])

    def verify_row(row):
        cta, ordinal = row['cta'], row['output_ordinal']
        require(row['sm'] == trace[1296 * cta + 2] - 1)
        if row['scope'] == 'WG':
            left = raw_stamp(cta, row['wg'], ordinal, 0)
            right = raw_stamp(cta, row['wg'], ordinal, 1)
        else:
            require(row['scope'] == 'common' and row['wg'] is None)
            starts = [raw_stamp(cta, w, ordinal, 0) for w in (0, 1)]
            ends = [raw_stamp(cta, w, ordinal, 1) for w in (0, 1)]
            start_wg = 0 if starts[0]['cycle'] <= starts[1]['cycle'] else 1
            end_wg = 0 if ends[0]['cycle'] >= ends[1]['cycle'] else 1
            left, right = starts[start_wg], ends[end_wg]
        require(row['begin'] == left and row['end'] == right)
        require(0 < left['gt_before'] <= left['gt_after'] < right['gt_before'] <= right['gt_after'])
        C = right['cycle'] - left['cycle']
        lo = right['gt_before'] - left['gt_after']
        hi = right['gt_after'] - left['gt_before']
        require(row['cycles'] == C and row['elapsed_lower_ns'] == lo and row['elapsed_upper_ns'] == hi)
        require(Q(row['rate_lower']) == Q(C, hi) and Q(row['rate_upper']) == Q(C, lo))
        require(row['post_gt_elapsed_ns'] == right['gt_after'] - left['gt_after'])
        require(row['post_gt_minus_selected_elapsed_lower_ns'] == left['gt_before'] - left['gt_after'])
        require(row['post_gt_minus_selected_elapsed_upper_ns'] == right['gt_after'] - right['gt_before'])
        by, bx = cta // 66, cta % 66
        linear = bx * 2 + by + ordinal * 132
        coord = [(linear // 2) % 9, 2 * (linear // 18) + by]
        require(linear < 216 and row['work'] == coord)

    for row in rows:
        verify_row(row)
    identities = {(r['scope'], r['wg'], r['cta'], r['output_ordinal']) for r in rows}
    expected = set()
    for cta in range(132):
        first = 2 * (cta % 66) + cta // 66
        for ordinal in range(2 if first < 84 else 1):
            for scope, wg in [('WG', 0), ('WG', 1), ('common', None)]:
                expected.add((scope, wg, cta, ordinal))
    require(identities == expected and len(rows) == len(expected) == 648)
    main_report = read(args.root / 'verification.json')
    main_rows = {(r['cta'], r['thread'], r['output_ordinal']): r for r in main_report['conditional_windows']}
    widths = []
    for row in rows:
        if row['scope'] != 'WG':
            continue
        other = main_rows[(row['cta'], 128 * (row['wg'] + 1), row['output_ordinal'])]
        require(other['conditional_elapsed_gt'] == [row['elapsed_lower_ns'], row['elapsed_upper_ns']])
        widths.extend(row[end]['gt_after'] - row[end]['gt_before'] for end in ('begin', 'end'))
    require(report['stamp_width_histogram'] == {str(k): v for k, v in sorted(Counter(widths).items())})

    def verify_pair(pair):
        a, b = [rows[pair[name]] for name in ('first_row', 'second_row')]
        require([a['output_ordinal'], b['output_ordinal']] == [0, 1])
        for key in ('cta', 'sm', 'scope', 'wg'):
            require(a[key] == b[key] == pair[key])
        C0, C1 = a['cycles'], b['cycles']
        low = max(Q(C0, a['elapsed_upper_ns']), Q(C1, b['elapsed_upper_ns']))
        high = min(Q(C0, a['elapsed_lower_ns']), Q(C1, b['elapsed_lower_ns']))
        require(Q(pair['shared_rate_lower']) == low and Q(pair['shared_rate_upper']) == high)
        require(pair['constant_average_rate_feasible'] is (low <= high))
        require(Q(pair['minimum_first_over_second_minus_one']) ==
                Q(C0 * b['elapsed_lower_ns'], C1 * a['elapsed_upper_ns']) - 1)

    for pair in pairs:
        verify_pair(pair)
    require(len(pairs) == 252 and len({(p['scope'], p['wg'], p['cta']) for p in pairs}) == 252)
    require(all(not p['constant_average_rate_feasible'] for p in pairs))
    common = [r for r in rows if r['scope'] == 'common']
    selectors = Counter((r['begin']['wg'], r['end']['wg']) for r in common)
    require(report['common_selector_counts'] == {str(k): v for k, v in sorted(selectors.items())})
    for summary in report['common_output_summaries']:
        group = [r for r in common if r['output_ordinal'] == summary['output_ordinal']]
        require(summary['windows'] == len(group))
        for name, values in [('cycles', [r['cycles'] for r in group]), ('interval_width_ns', [
                r['elapsed_upper_ns'] - r['elapsed_lower_ns'] for r in group])]:
            require(summary[name] == dict(minimum=min(values), median=median(values), maximum=max(values)))
        require(summary['elapsed_low_min'] == min(r['elapsed_lower_ns'] for r in group))
        require(summary['elapsed_high_max'] == max(r['elapsed_upper_ns'] for r in group))
        require(Q(summary['rate_low_min']) == min(Q(r['rate_lower']) for r in group))
        require(Q(summary['rate_high_max']) == max(Q(r['rate_upper']) for r in group))
    separation = [float(Q(p['minimum_first_over_second_minus_one'])) for p in pairs if p['scope'] == 'common']
    require(report['common_minimum_first_over_second_minus_one'] ==
            dict(minimum=min(separation), median=median(separation), maximum=max(separation)))
    for key in ('hardware_sampling_enclosure', 'physical_ns_accuracy', 'observation_qualified',
                'prediction_qualified', 'joint_feasibility_proved'):
        require(report[key] is False)
    require(report['gt_error_ns'] == 0 and report['job'] == 740160)
    proof = report['numeric_proof']
    require(proof['main_report_sha256'] == digest(args.root / 'verification.json') and
            proof['full_D_sha256'] == digest(args.root / 'returned/full-d.bin') and
            proof['full_D_elements'] == 7077888 and proof['status'].startswith('reused_main'))
    # Deliberately wrong labels can leave GT values tied and plausible arithmetic.
    wrong_label = deepcopy(common[0]); wrong_label['begin']['wg'] ^= 1
    wrong_bound = deepcopy(common[0]); wrong_bound['elapsed_lower_ns'] += 1
    wrong_sign = deepcopy(common[0]); wrong_sign['post_gt_minus_selected_elapsed_lower_ns'] *= -1
    wrong_pair = deepcopy(pairs[0]); wrong_pair['constant_average_rate_feasible'] = True
    negatives = 0
    for verifier, value in [(verify_row, wrong_label), (verify_row, wrong_bound),
                            (verify_row, wrong_sign), (verify_pair, wrong_pair)]:
        try:
            verifier(value)
        except ValueError:
            negatives += 1
        else:
            raise ValueError('negative control accepted')
    print(json.dumps(dict(input_SHA_checks=len(checked), frozen_smoke_files=len(freeze),
        build_source_SHA_checks=len(sources), archive_file_matches=archive_counts,
        WG_stamps=864, WG_main_intervals=432, raw_selected_windows=len(rows),
        phase_pairs=len(pairs), constant_rate_feasible_pairs=0, negative_controls_rejected=negatives,
        full_D_numeric_evidence='bound and reused; no independent full-D recomputation',
        observation_qualified=False, prediction_qualified=False), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
