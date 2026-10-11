"""Conditional first/second output-rate intervals in all 740161 enabled targets.

Reuses the previously audited raw-slot/GT interval primitives. No across-
process subtraction, precision-budget fit, observer gate or prediction score.
"""
import argparse
from collections import Counter
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import tarfile

from r09_gt_window_audit import check, dump, fixed_work, load, raw_packets, sha, stamp, window


def analyze(root, inputs):
    for name, expected in inputs['program_files'].items():
        check(sha(Path(__file__).parent / name) == expected, 'program SHA mismatch: ' + name)
    for name, expected in inputs['run_files'].items():
        check(sha(root / name) == expected, 'run input SHA mismatch: ' + name)
    freeze = load(root / 'freeze.json')['files']
    for name, expected in freeze.items():
        check(sha(root / name) == expected, 'frozen input mismatch: ' + name)
    archive_counts = {}
    for name, parent in (('input.tar.gz', root), ('output.tar.gz', root / 'returned')):
        matches = 0
        with tarfile.open(root / name) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    check(hashlib.sha256(archive.extractfile(member).read()).hexdigest() ==
                          sha(parent / member.name.removeprefix('./')), 'archive/extraction mismatch')
                    matches += 1
        archive_counts[name] = matches
    check(load(root / 'receipt.json')['sha256'] == sha(root / 'output.tar.gz') and
          load(root / 'receipt.json')['returncode'] == 0 and
          load(root / 'input-receipt.json')['input_sha256'] == sha(root / 'input.tar.gz'),
          'receipt mismatch')
    check(load(root / 'returned/assigned-context.json')['SLURM_JOB_ID'] == '740161', 'wrong job')
    manifest = load(root / 'manifest.json')
    summary = load(root / 'summary.json')
    check(manifest['gpu_uuid'] == inputs['gpu_uuid'] and
          manifest['binary_sha256'] == inputs['binary_sha256'] == sha(root / 'build/cfg_c_s4_dual'),
          'wrong GPU or binary')
    check(summary['record_errors'] == [] and summary['observation_qualified'] is False,
          'control records failed or qualification changed')
    for name, failed in (('off_on', 4912), ('off_off', 4729)):
        cohort = summary['cohorts'][name]
        check(cohort['processes_verified'] == 20 and cohort['failed_point_series'] == failed
              and cohort['series_per_process'] == 6372 and cohort['by_metric']['L']['failed'] == 432
              and cohort['necessary_control_passed'] is False, 'control boundary changed')
    records = load(root / 'returned/records.json')
    check(len(records) == 40 and len({r['id'] for r in records}) == 40, 'all forty records required')
    for record, frozen_record in zip(records, manifest['samples']):
        check(all(record[k] == frozen_record[k] for k in frozen_record) and record['returncode'] == 0
              and sha(root / 'returned' / record['raw']) == record['sha256'], 'record identity mismatch')
    enabled = [r for r in records if r['flag'] == 1]
    check([r['id'] for r in enabled] == ['off_on_pair%02d_arm1' % n for n in range(10)],
          'all ten enabled targets required')
    rows, pairs, process_summaries = [], [], []
    all_widths = []
    # Import only after checking the exact frozen structural reader and its dependencies.
    sys.path.insert(0, str(root / 'analysis_snapshot'))
    from decode_cfg_c_gt_bracket import decode
    for record in enabled:
        setup, call, _ = raw_packets(root / 'returned' / record['raw'])
        decoded = decode(setup, call, inputs['gpu_uuid'])
        check(decoded['record_valid'] is True, 'strict GT packet decoding failed')
        trace, shadow = call['trace'], call['gt_shadow']
        index = {}
        for cta in range(132):
            sm = trace[cta * 1296 + 2] - 1
            for ordinal, coord in enumerate(fixed_work(cta)):
                roles = []
                for wg in (0, 1):
                    first = stamp(trace, shadow, cta, wg, ordinal, 0)
                    last = stamp(trace, shadow, cta, wg, ordinal, 1)
                    value = window(inputs['gpu_uuid'], sm, cta, ordinal, coord, 'WG', wg, first, last)
                    for end in (first, last):
                        all_widths.append(end['gt_after'] - end['gt_before'])
                    actual = decoded['records'][cta]['roles'][wg]['tiles'][ordinal]
                    check(actual['conditional_elapsed_gt'] == [value['elapsed_lower_ns'],
                          value['elapsed_upper_ns']], 'frozen reader interval mismatch')
                    value['process'] = record['id']
                    index[('WG', wg, cta, ordinal)] = len(rows)
                    rows.append(value)
                    roles.append(value)
                first = min((v['begin'] for v in roles), key=lambda v: (v['cycle'], v['wg']))
                last = min((v['end'] for v in roles), key=lambda v: (-v['cycle'], v['wg']))
                value = window(inputs['gpu_uuid'], sm, cta, ordinal, coord, 'common', None, first, last)
                value['process'] = record['id']
                index[('common', None, cta, ordinal)] = len(rows)
                rows.append(value)
        local_pairs = []
        for scope, wg in (('WG', 0), ('WG', 1), ('common', None)):
            for cta in range(132):
                key = (scope, wg, cta)
                if key + (1,) not in index:
                    continue
                i, j = index[key + (0,)], index[key + (1,)]
                a, b = rows[i], rows[j]
                lo = max(Q(a['rate_lower']), Q(b['rate_lower']))
                hi = min(Q(a['rate_upper']), Q(b['rate_upper']))
                local_pairs.append(dict(process=record['id'], scope=scope, wg=wg, cta=cta, sm=a['sm'],
                    first_row=i, second_row=j, shared_rate_lower=str(lo), shared_rate_upper=str(hi),
                    constant_average_rate_feasible=(lo <= hi),
                    minimum_first_over_second_minus_one=str(Q(a['rate_lower']) / Q(b['rate_upper']) - 1)))
        pairs.extend(local_pairs)
        common_pairs = [p for p in local_pairs if p['scope'] == 'common']
        gap = [Q(p['minimum_first_over_second_minus_one']) for p in common_pairs]
        process_summaries.append(dict(process=record['id'], first_second_pairs=len(local_pairs),
            constant_average_rate_feasible_pairs=sum(p['constant_average_rate_feasible'] for p in local_pairs),
            common_pairs=len(common_pairs), minimum_first_over_second_minus_one=dict(
                minimum=str(min(gap)), median=str(median(gap)), maximum=str(max(gap)))))
    common_gaps = [Q(p['minimum_first_over_second_minus_one']) for p in pairs if p['scope'] == 'common']
    return dict(job=740161, gpu_uuid=inputs['gpu_uuid'], binary_sha256=inputs['binary_sha256'],
        checked_run_SHA=len(inputs['run_files']), frozen_files=len(freeze), archive_file_matches=archive_counts,
        enabled_processes=10, WG_windows=4320, common_windows=2160, selected_windows=len(rows),
        WG_main_reader_intervals_matched=4320, stamps=len(all_widths),
        stamp_width_histogram={str(k): v for k, v in sorted(Counter(all_widths).items())},
        first_second_pairs=len(pairs), common_first_second_pairs=len(common_gaps),
        constant_average_rate_feasible_pairs=sum(p['constant_average_rate_feasible'] for p in pairs),
        common_minimum_first_over_second_minus_one=dict(minimum=str(min(common_gaps)),
            median=str(median(common_gaps)), maximum=str(max(common_gaps))),
        per_process=process_summaries, gt_error_ns=0,
        numeric_evidence='Reuses hash-bound main full-D and 40x4096 exact sample verification; not recomputed by A.',
        observation_qualified=False, prediction_qualified=False, hardware_sampling_enclosure=False,
        physical_ns_accuracy=False, joint_feasibility_proved=False,
        limits='Conditional exact GT-coordinate interval arithmetic, assuming real sampling enclosure. A shared exact average rate is tested within each process/CTA, not across processes. No fitted read-error budget, physical-frequency attribution, model price or prediction. Original failed observer/null controls unchanged.'), rows, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report, rows, pairs = analyze(args.root, load(args.inputs))
    dump(args.output / 'audit.json', report)
    for name, values in (('windows.jsonl', rows), ('phase-pairs.jsonl', pairs)):
        (args.output / name).write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in values))
    print(json.dumps({k: report[k] for k in ('enabled_processes', 'selected_windows',
        'first_second_pairs', 'constant_average_rate_feasible_pairs', 'common_minimum_first_over_second_minus_one')}))


if __name__ == '__main__':
    main()
