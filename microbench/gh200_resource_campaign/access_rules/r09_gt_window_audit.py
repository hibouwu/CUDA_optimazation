"""Offline raw-slot audit of the fixed 740160 cfg_c GT-CLOCK-GT smoke.

Only the recorded GT coordinate, with explicitly zero additional GT error, is
used. Sampling enclosure and physical GT accuracy remain unproved premises.
This is neither an observer gate nor a prediction/calibration entry point.
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import median
import tarfile

from r09_gt_brackets import GtClockStamp, window_interval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def check(condition, label):
    if not condition:
        raise ValueError(label)


def identity(root, build, frozen):
    checked = []
    for name, expected in frozen['program_files'].items():
        check(sha(Path(__file__).parent / name) == expected, 'program identity mismatch: ' + name)
    for label, parent in (('run', root), ('build', build)):
        for name, expected in frozen[label + '_files'].items():
            actual = sha(parent / name)
            check(actual == expected, 'input SHA mismatch: ' + label + '/' + name)
            checked.append(dict(source=label, path=name, sha256=actual))
    build_audit = load(build / 'build-audit.json')
    source_hashes = load(build / 'source_hashes.json')
    check(sha(build / 'source_hashes.json') == build_audit['source_manifest_sha256'],
          'build source manifest mismatch')
    for name, expected in source_hashes.items():
        check(sha(build / name) == expected, 'build source mismatch: ' + name)
    freeze = load(root / 'freeze.json')
    for name, expected in freeze['files'].items():
        check(sha(root / name) == expected, 'smoke freeze mismatch: ' + name)
    archive_counts = {}
    for archive, extracted in (('input.tar.gz', root), ('output.tar.gz', root / 'returned')):
        count = 0
        with tarfile.open(root / archive, 'r:gz') as stream:
            for member in stream.getmembers():
                if not member.isfile():
                    continue
                name = member.name.removeprefix('./')
                data = stream.extractfile(member).read()
                check(hashlib.sha256(data).hexdigest() == sha(extracted / name),
                      'archive/extraction mismatch: ' + archive + '/' + name)
                count += 1
        archive_counts[archive] = count
    receipt = load(root / 'receipt.json')
    check(load(root / 'input-receipt.json')['input_sha256'] == sha(root / 'input.tar.gz'),
          'input archive receipt mismatch')
    check(receipt['returncode'] == 0 and receipt['bytes'] == (root / 'output.tar.gz').stat().st_size
          and receipt['sha256'] == sha(root / 'output.tar.gz'), 'return receipt mismatch')
    check(load(root / 'returned/assigned-context.json')['SLURM_JOB_ID'] == '740160',
          'wrong smoke job')
    manifest = load(root / 'manifest.json')
    check(manifest['binary_sha256'] == build_audit['binary_sha256'] ==
          sha(root / 'build/cfg_c_s4_dual') == sha(build / 'returned/build/cfg_c_s4_dual'),
          'build/binary identity mismatch')
    return dict(checked_files=checked, smoke_frozen_files=len(freeze['files']),
                build_source_files=len(source_hashes), archive_file_matches=archive_counts)


def raw_packets(path):
    with gzip.open(path, 'rt') as stream:
        packets = [json.loads(line) for line in stream if line.strip()]
    check([p['event'] for p in packets] == ['setup', 'call', 'check'],
          'exact setup/call/check record required')
    return packets


def fixed_work(cta):
    # sw1, along_m, cluster1x2, grid66x2, nine M tiles and 24 N tiles.
    by, bx = divmod(cta, 66)
    return [[(linear // 2) % 9, 2 * (linear // 18) + by]
            for linear in range(2 * bx + by, 216, 132)]


def stamp(trace, shadow, cta, wg, ordinal, event):
    cycle_word = cta * 1296 + 16 + wg * 640 + ordinal * 10 + event
    after_word = cycle_word + 6
    before_word = cta * 9 + wg * 4 + ordinal * 2 + event
    return dict(wg=wg, physical_thread=128 * (wg + 1),
                trace_cycle_word=cycle_word, trace_gt_after_word=after_word,
                shadow_gt_before_word=before_word, cycle=trace[cycle_word],
                gt_before=shadow[before_word], gt_after=trace[after_word])


def window(gpu, sm, cta, ordinal, work, scope, wg, begin, end):
    first = GtClockStamp(gpu, sm, begin['gt_before'], begin['cycle'], begin['gt_after'])
    last = GtClockStamp(gpu, sm, end['gt_before'], end['cycle'], end['gt_after'])
    bounds = window_interval(first, last, gt_error_ns=0)
    check(not bounds['elapsed_lower_open'], 'unexpected overlapping elapsed brackets')
    return dict(cta=cta, sm=sm, output_ordinal=ordinal, work=work, scope=scope, wg=wg,
                begin=begin, end=end, cycles=bounds['cycles'],
                elapsed_lower_ns=int(bounds['elapsed_lower_ns']),
                elapsed_upper_ns=int(bounds['elapsed_upper_ns']),
                rate_lower=str(bounds['rate_lower_cycle_per_ns']),
                rate_upper=str(bounds['rate_upper_cycle_per_ns']),
                post_gt_elapsed_ns=end['gt_after'] - begin['gt_after'],
                post_gt_minus_selected_elapsed_lower_ns=-bounds['begin_gt_width_ns'],
                post_gt_minus_selected_elapsed_upper_ns=bounds['end_gt_width_ns'])


def spread(values):
    return dict(minimum=min(values), median=median(values), maximum=max(values))


def audit(root, build, frozen):
    identities = identity(root, build, frozen)
    manifest = load(root / 'manifest.json')
    report = load(root / 'verification.json')
    gpu = manifest['gpu_uuid']
    check(gpu == frozen['gpu_uuid'] == report['gpu_uuid'], 'GPU mismatch')
    records = load(root / 'returned/records.json')
    check([r['id'] for r in records] == ['full_D', 'flag0', 'flag1'], 'unexpected arms')
    for record in records:
        check(record['returncode'] == 0 and sha(root / 'returned' / record['raw']) == record['sha256'],
              'raw return/SHA mismatch')
    numeric = report['results']
    check(report['full_D_reference_ok'] is True and numeric[0]['off_on_bit_equal'] is True
          and numeric[0]['exact_reference'] is True and numeric[0]['elements'] == 7077888,
          'existing full-D evidence failed')
    check(sha(root / 'returned/full-d.bin') == report['full_D_sha256'], 'full-D SHA mismatch')
    rows = []
    widths = []
    counts = Counter()
    main_windows = {(v['cta'], v['thread'], v['output_ordinal']): v
                    for v in report['conditional_windows']}
    check(len(main_windows) == 432, 'main window identities incomplete')
    for record, observed in zip(records[1:], numeric[1:]):
        setup, call, _ = raw_packets(root / 'returned' / record['raw'])
        check(observed['id'] == record['id'] and observed['raw_sha256'] == record['sha256']
              and observed['record_valid'] is True and observed['numeric_ok'] is True,
              'existing structural/numeric evidence failed')
        check(setup['gpu_uuid'] == gpu and setup['trace_version'] == 'cfg-c-gt-bracket-v1'
              and setup['grid'] == [66, 2, 1] and setup['tile'] == [256, 128, 64]
              and setup['cluster'] == [1, 2] and setup['stages'] == 4,
              'fixed geometry/version mismatch')
        flag = record['flag']
        check(type(flag) is int and flag in (0, 1) and setup['gt_bracket_observer'] == flag
              and call['gt_bracket_observer'] == flag, 'observer flag mismatch')
        trace, shadow = call['trace'], call['gt_shadow']
        check(len(trace) == 132 * 1296 and len(shadow) == 132 * 9 and
              all(type(v) is int and 0 <= v < 2**64 for v in trace + shadow), 'native words required')
        if not flag:
            check(not any(shadow), 'disabled shadow not zero')
        work_seen = []
        for cta in range(132):
            head = trace[cta * 1296:cta * 1296 + 16]
            work = fixed_work(cta)
            check(head[6] == head[9] == len(work) and head[10] == shadow[cta * 9 + 8] == 0
                  and 1 <= head[2] <= 132 and head[11] == head[12] == head[2], 'CTA identity/count mismatch')
            counts[(flag, len(work))] += 1
            work_seen.extend(work)
            for ordinal, coord in enumerate(work):
                role_windows = []
                for wg in (0, 1):
                    offset = cta * 1296 + 16 + wg * 640 + ordinal * 10
                    check(trace[offset + 4:offset + 6] == [v + 1 for v in coord], 'raw work mismatch')
                    if not flag:
                        continue
                    begin = stamp(trace, shadow, cta, wg, ordinal, 0)
                    end = stamp(trace, shadow, cta, wg, ordinal, 1)
                    row = window(gpu, head[2] - 1, cta, ordinal, coord, 'WG', wg, begin, end)
                    for value in (begin, end):
                        widths.append(value['gt_after'] - value['gt_before'])
                    other = main_windows[(cta, 128 * (wg + 1), ordinal)]
                    check([row['elapsed_lower_ns'], row['elapsed_upper_ns']] == other['conditional_elapsed_gt']
                          and row['post_gt_elapsed_ns'] == other['post_gt_elapsed']
                          and other['post_gt_error_bound'] == max(
                              -row['post_gt_minus_selected_elapsed_lower_ns'],
                              row['post_gt_minus_selected_elapsed_upper_ns']), 'main interval mismatch')
                    tile = observed['decoded']['records'][cta]['roles'][wg]['tiles'][ordinal]
                    check(tile['gt_brackets'] == [dict(gt_before=v['gt_before'], cycles=v['cycle'],
                          gt_after=v['gt_after']) for v in (begin, end)], 'main raw triples mismatch')
                    role_windows.append(row)
                    rows.append(row)
                if flag:
                    # Choose in the same native SM cycle domain. Carry those exact labels to GT.
                    begin = min((r['begin'] for r in role_windows), key=lambda v: (v['cycle'], v['wg']))
                    end = min((r['end'] for r in role_windows), key=lambda v: (-v['cycle'], v['wg']))
                    rows.append(window(gpu, head[2] - 1, cta, ordinal, coord, 'common', None, begin, end))
            for wg in (0, 1):
                unused_shadow = shadow[cta * 9 + wg * 4 + len(work) * 2:cta * 9 + (wg + 1) * 4]
                offset = cta * 1296 + 16 + wg * 640
                check(not any(unused_shadow) and not any(trace[offset + len(work) * 10:offset + 640]),
                      'unused raw slots not zero')
        check(sorted(work_seen) == [[i, j] for i in range(9) for j in range(24)], 'incomplete work cover')
    pairs = []
    indexed = {(r['scope'], r['wg'], r['cta'], r['output_ordinal']): (i, r) for i, r in enumerate(rows)}
    for scope, wg in (('WG', 0), ('WG', 1), ('common', None)):
        for cta in range(132):
            key = (scope, wg, cta)
            if key + (1,) not in indexed:
                continue
            i, first = indexed[key + (0,)]
            j, second = indexed[key + (1,)]
            lower = max(Fraction(first['rate_lower']), Fraction(second['rate_lower']))
            upper = min(Fraction(first['rate_upper']), Fraction(second['rate_upper']))
            pairs.append(dict(cta=cta, sm=first['sm'], scope=scope, wg=wg,
                first_row=i, second_row=j, shared_rate_lower=str(lower), shared_rate_upper=str(upper),
                constant_average_rate_feasible=(lower <= upper),
                minimum_first_over_second_minus_one=str(
                    Fraction(first['rate_lower']) / Fraction(second['rate_upper']) - 1)))
    common = [r for r in rows if r['scope'] == 'common']
    summaries = []
    for ordinal in (0, 1):
        selected = [r for r in common if r['output_ordinal'] == ordinal]
        summaries.append(dict(output_ordinal=ordinal, windows=len(selected),
            cycles=spread([r['cycles'] for r in selected]),
            elapsed_low_min=min(r['elapsed_lower_ns'] for r in selected),
            elapsed_high_max=max(r['elapsed_upper_ns'] for r in selected),
            rate_low_min=str(min(Fraction(r['rate_lower']) for r in selected)),
            rate_high_max=str(max(Fraction(r['rate_upper']) for r in selected)),
            interval_width_ns=spread([r['elapsed_upper_ns'] - r['elapsed_lower_ns'] for r in selected])))
    common_pairs = [p for p in pairs if p['scope'] == 'common']
    report_out = dict(job=740160, gpu_uuid=gpu, binary_sha256=manifest['binary_sha256'],
        identity=identities, gt_error_ns=0,
        conditional_premises='Actual central CLOCK sampling is enclosed by its two GT reads; exact recorded GT coordinate. No physical accuracy claim.',
        WG_windows=432, common_windows=len(common), stamps=len(widths),
        stamp_width_histogram={str(k): v for k, v in sorted(Counter(widths).items())},
        CTA_output_counts={str(flag): {str(n): counts[(flag, n)] for n in (1, 2)} for flag in (0, 1)},
        main_reader_triples_and_intervals_matched=432,
        common_selector_counts={str(k): v for k, v in sorted(Counter(
            (r['begin']['wg'], r['end']['wg']) for r in common).items())},
        common_output_summaries=summaries,
        first_second_pairs=len(pairs), constant_average_rate_feasible_pairs=sum(
            p['constant_average_rate_feasible'] for p in pairs),
        common_first_second_pairs=len(common_pairs),
        common_minimum_first_over_second_minus_one=spread([
            float(Fraction(p['minimum_first_over_second_minus_one'])) for p in common_pairs]),
        numeric_proof=dict(status='reused_main_full_D_and_sample_reference_checks_not_recomputed_by_A',
            full_D_elements=7077888, full_D_sha256=report['full_D_sha256'],
            main_report_sha256=sha(root / 'verification.json')),
        hardware_sampling_enclosure=False, physical_ns_accuracy=False,
        observation_qualified=False, prediction_qualified=False, joint_feasibility_proved=False,
        limits='One process per warm arm. Disjoint marginal rate intervals reject one shared average rate under stated premises; they do not identify physical frequency or assess observer/null distributions. No old-binary calibration or prediction is used.')
    return report_out, rows, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--build-root', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report, rows, pairs = audit(args.root, args.build_root, load(args.inputs))
    dump(args.output / 'audit.json', report)
    for name, values in (('windows.jsonl', rows), ('phase-pairs.jsonl', pairs)):
        (args.output / name).write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in values))
    print(json.dumps({k: report[k] for k in ('WG_windows', 'common_windows', 'stamps',
        'first_second_pairs', 'constant_average_rate_feasible_pairs')}))


if __name__ == '__main__':
    main()
