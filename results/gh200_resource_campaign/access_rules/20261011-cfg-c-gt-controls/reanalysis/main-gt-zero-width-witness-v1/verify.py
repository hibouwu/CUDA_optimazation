"""Independent direct raw extraction and complete two-point enumeration."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path


def require(value):
    if not value:
        raise ValueError('independent verification failed')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certifies(a, b):
    # Strictly different counter values require strictly ordered sample times,
    # even if the counter can have flat parts or jumps.
    return (a['gpu_uuid'] == b['gpu_uuid'] and a['sm'] == b['sm'] and
            a['cycle'] < b['cycle'] and a['gt_before'] >= b['gt_after'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--analysis', required=True, type=Path)
    args = parser.parse_args()
    inputs = read(args.inputs)
    for name, expected in inputs['files'].items():
        require(digest(args.base / name) == expected)
    for name, expected in inputs['program_files'].items():
        require(digest(args.inputs.parent / name) == expected)
    expected_witnesses = []
    process_summaries = []
    widths = Counter()
    for run in inputs['runs']:
        root = args.base / run
        records = read(root / 'returned/records.json')
        for rec in records:
            if rec['flag'] != 1:
                continue
            with gzip.open(root / 'returned' / rec['raw'], 'rt') as stream:
                setup, call, _ = [json.loads(s) for s in stream]
            require(digest(root / 'returned' / rec['raw']) == rec['sha256'])
            trace, shadow = call['trace'], call['gt_shadow']
            conflict_ctas = set()
            ordered_pairs = 0
            before = len(expected_witnesses)
            for cta in range(132):
                sm = trace[1296 * cta + 2] - 1
                require(trace[1296 * cta + 11] == trace[1296 * cta + 12] == sm + 1)
                count = 2 if 2 * (cta % 66) + cta // 66 < 84 else 1
                require(trace[1296 * cta + 6] == trace[1296 * cta + 9] == count)
                points = []
                for role in range(2):
                    for output in range(count):
                        for event, name in enumerate(('FIRST', 'MAIN')):
                            i = 1296 * cta + 16 + role * 640 + output * 10 + event
                            j = 9 * cta + role * 4 + output * 2 + event
                            point = dict(gpu_uuid=setup['gpu_uuid'], sm=sm, cta=cta, wg=role,
                                output_ordinal=output, event=name, trace_cycle_word=i,
                                trace_gt_after_word=i + 6, shadow_gt_before_word=j,
                                cycle=trace[i], gt_before=shadow[j], gt_after=trace[i + 6])
                            require(point['gt_before'] <= point['gt_after'])
                            widths[point['gt_after'] - point['gt_before']] += 1
                            points.append(point)
                points.sort(key=lambda p: p['cycle'])
                for j, later in enumerate(points):
                    for earlier in points[:j]:
                        if earlier['cycle'] == later['cycle']:
                            continue
                        ordered_pairs += 1
                        if certifies(earlier, later):
                            conflict_ctas.add(cta)
                            expected_witnesses.append(dict(run=run, process=rec['id'], raw_sha256=rec['sha256'],
                                earlier_cycle=earlier, later_cycle=later,
                                delta_cycles=later['cycle'] - earlier['cycle'],
                                gt_box_separation=earlier['gt_before'] - later['gt_after'],
                                both_zero_width_same_coordinate=(earlier['gt_before'] == earlier['gt_after'] ==
                                    later['gt_before'] == later['gt_after'])))
            process_summaries.append(dict(run=run, process=rec['id'], CTAs=132, stamps=864,
                ordered_same_SM_pairs=ordered_pairs, contradictory_pairs=len(expected_witnesses) - before,
                contradictory_CTAs=len(conflict_ctas), CTA_indices=sorted(conflict_ctas)))
    report = read(args.analysis / 'summary.json')
    witnesses = [json.loads(s) for s in (args.analysis / 'witnesses.jsonl').read_text().splitlines()]
    key = lambda w: (w['run'], w['process'], w['earlier_cycle']['cta'],
                     w['earlier_cycle']['cycle'], w['later_cycle']['cycle'])
    require(sorted(witnesses, key=key) == sorted(expected_witnesses, key=key))
    require(report['per_process'] == process_summaries and len(process_summaries) == 11)
    require(report['ordered_same_SM_pairs'] == sum(r['ordered_same_SM_pairs'] for r in process_summaries) == 29040)
    require(report['contradictory_pairs'] == report['contradictory_CTAs'] == len(witnesses) == 55)
    require(report['zero_width_equal_coordinate_pairs'] == 55 and report['strictly_reversed_GT_boxes'] == 0)
    delta = Counter(w['delta_cycles'] for w in witnesses)
    require(report['delta_cycle_histogram'] == {str(k): v for k, v in sorted(delta.items())})
    require(delta == {8: 54, 12: 1})
    require(report['stamp_width_histogram'] == {str(k): v for k, v in sorted(widths.items())})
    require(report['input_SHA'] == len(inputs['files']) and report['GT_coordinate_extra_error'] == 0)
    for name in ('hardware_sampling_enclosure_established', 'physical_GT_error_bound_established',
                 'observation_qualified', 'prediction_qualified', 'joint_feasibility_proved'):
        require(report[name] is False)
    # Cross-SM/GPU comparisons, equal cycles and genuinely ordered boxes do
    # not supply this particular contradiction certificate.
    a, b = [witnesses[0][name] for name in ('earlier_cycle', 'later_cycle')]
    require(certifies(a, b))
    negatives = []
    other_sm = deepcopy(b); other_sm['sm'] = (b['sm'] + 1) % 132
    other_gpu = deepcopy(b); other_gpu['gpu_uuid'] = 'different GPU'
    equal_cycle = deepcopy(b); equal_cycle['cycle'] = a['cycle']
    ordered_box = deepcopy(b); ordered_box['gt_after'] += 1
    negatives.extend((other_sm, other_gpu, equal_cycle, ordered_box))
    require(all(not certifies(a, point) for point in negatives))
    print(json.dumps(dict(input_SHA=len(inputs['files']), processes=11, raw_stamps=sum(widths.values()),
        ordered_same_SM_pairs=29040, counterexamples=55, zero_width_equal_coordinate_pairs=55,
        delta_cycle_histogram={str(k): v for k, v in sorted(delta.items())},
        negative_certificates_rejected=len(negatives),
        independently_enumerated=True, observation_qualified=False, prediction_qualified=False),
        indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
