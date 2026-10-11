"""Raw two-point witnesses against joint exact-GT-coordinate sampling boxes.

If a same-SM nondecreasing single-valued counter has C_a<C_b, its sampling
times must satisfy t_a<t_b. Closed boxes with lower_a>=upper_b contradict
this condition. No constant rate, frequency cap or fitted error budget is used.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def check(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contradicts_order(a, b):
    """A sufficient contradiction, never a proof of hardware sampling behavior."""
    check(a['gpu_uuid'] == b['gpu_uuid'] and a['sm'] == b['sm'], 'one GPU/SM domain required')
    check(0 < a['gt_before'] <= a['gt_after'] and 0 < b['gt_before'] <= b['gt_after'],
          'nonzero ordered GT boxes required')
    check(0 < a['cycle'] < b['cycle'], 'strict native cycle order required')
    return a['gt_before'] >= b['gt_after']


def analyze(base, inputs):
    for name, expected in inputs['files'].items():
        check(sha(base / name) == expected, 'source SHA mismatch: ' + name)
    for name, expected in inputs['program_files'].items():
        check(sha(Path(__file__).parent / name) == expected, 'program SHA mismatch: ' + name)
    witnesses, summaries = [], []
    widths = Counter()
    for run in inputs['runs']:
        root = base / run
        manifest = read(root / 'manifest.json')
        check(manifest['gpu_uuid'] == inputs['gpu_uuid'] and
              manifest['binary_sha256'] == inputs['binary_sha256'] == sha(root / 'build/cfg_c_s4_dual'),
              'GPU/binary identity mismatch')
        records = read(root / 'returned/records.json')
        enabled = [r for r in records if r['flag'] == 1]
        expected_ids = ['flag1'] if run.endswith('smoke-v2') else ['off_on_pair%02d_arm1' % n for n in range(10)]
        check([r['id'] for r in enabled] == expected_ids, 'all enabled records required')
        for record in enabled:
            raw = root / 'returned' / record['raw']
            check(record['returncode'] == 0 and sha(raw) == record['sha256'], 'raw identity mismatch')
            with gzip.open(raw, 'rt') as stream:
                setup, call, _ = [json.loads(line) for line in stream]
            check(setup['gpu_uuid'] == inputs['gpu_uuid'] and setup['trace_version'] == 'cfg-c-gt-bracket-v1'
                  and setup['grid'] == [66, 2, 1] and setup['sm_count'] == 132 and
                  setup['gt_bracket_observer'] == call['gt_bracket_observer'] == 1, 'wrong record protocol')
            trace, shadow = call['trace'], call['gt_shadow']
            check(len(trace) == 132 * 1296 and len(shadow) == 132 * 9, 'incomplete native arrays')
            sm_ids = [trace[cta * 1296 + 2] - 1 for cta in range(132)]
            check(sorted(sm_ids) == list(range(132)), 'expected one observed CTA per SM')
            ordered_pairs = 0
            conflict_ctas = set()
            conflicts_before = len(witnesses)
            for cta in range(132):
                count = trace[cta * 1296 + 6]
                check(count == trace[cta * 1296 + 9] and count in (1, 2), 'role counts differ')
                points = []
                for wg in (0, 1):
                    for ordinal in range(count):
                        for event in (0, 1):
                            i = cta * 1296 + 16 + wg * 640 + ordinal * 10 + event
                            j = cta * 9 + wg * 4 + ordinal * 2 + event
                            point = dict(gpu_uuid=inputs['gpu_uuid'], sm=sm_ids[cta], cta=cta,
                                wg=wg, output_ordinal=ordinal, event=('FIRST', 'MAIN')[event],
                                trace_cycle_word=i, trace_gt_after_word=i + 6, shadow_gt_before_word=j,
                                cycle=trace[i], gt_before=shadow[j], gt_after=trace[i + 6])
                            points.append(point)
                            widths[point['gt_after'] - point['gt_before']] += 1
                for a in points:
                    for b in points:
                        if a['cycle'] >= b['cycle']:
                            continue
                        ordered_pairs += 1
                        if contradicts_order(a, b):
                            conflict_ctas.add(cta)
                            witnesses.append(dict(run=run, process=record['id'], raw_sha256=record['sha256'],
                                earlier_cycle=a, later_cycle=b, delta_cycles=b['cycle'] - a['cycle'],
                                gt_box_separation=a['gt_before'] - b['gt_after'],
                                both_zero_width_same_coordinate=(a['gt_before'] == a['gt_after'] ==
                                    b['gt_before'] == b['gt_after'])))
            summaries.append(dict(run=run, process=record['id'], CTAs=132, stamps=864,
                ordered_same_SM_pairs=ordered_pairs, contradictory_pairs=len(witnesses) - conflicts_before,
                contradictory_CTAs=len(conflict_ctas), CTA_indices=sorted(conflict_ctas)))
    return dict(gpu_uuid=inputs['gpu_uuid'], binary_sha256=inputs['binary_sha256'],
        input_SHA=len(inputs['files']), processes=len(summaries), per_process=summaries,
        ordered_same_SM_pairs=sum(r['ordered_same_SM_pairs'] for r in summaries),
        contradictory_pairs=len(witnesses), contradictory_CTAs=sum(r['contradictory_CTAs'] for r in summaries),
        zero_width_equal_coordinate_pairs=sum(w['both_zero_width_same_coordinate'] for w in witnesses),
        strictly_reversed_GT_boxes=sum(w['gt_box_separation'] > 0 for w in witnesses),
        delta_cycle_histogram={str(k): v for k, v in sorted(Counter(w['delta_cycles'] for w in witnesses).items())},
        stamp_width_histogram={str(k): v for k, v in sorted(widths.items())},
        GT_coordinate_extra_error=0,
        tested_premises=['Each central CLOCK sample lies in its closed exact recorded GT box.',
            'The same-SM nonwrapping CLOCK values belong to one single-valued nondecreasing function of that sampling coordinate.'],
        proof='C_a<C_b implies t_a<t_b. Boxes imply t_a>=GT_a_before>=GT_b_after>=t_b, a contradiction.',
        hardware_sampling_enclosure_established=False, physical_GT_error_bound_established=False,
        observation_qualified=False, prediction_qualified=False, joint_feasibility_proved=False,
        limits='These are raw conditional counterexamples to the conjunction of exact-coordinate boxes and one monotone counter. They do not identify sampling reordering, physical GT error, clock malfunction or a necessary positive error budget. Prior marginal interval arithmetic is unchanged; zero recorded width is not a proved joint uncertainty bound.'), witnesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report, witnesses = analyze(args.base, read(args.inputs))
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (args.output / 'witnesses.jsonl').write_text(''.join(json.dumps(w, sort_keys=True) + '\n' for w in witnesses))
    print(json.dumps({k: report[k] for k in ('processes', 'ordered_same_SM_pairs', 'contradictory_pairs',
        'zero_width_equal_coordinate_pairs', 'delta_cycle_histogram')}))


if __name__ == '__main__':
    main()
