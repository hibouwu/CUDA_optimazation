#!/usr/bin/env python3
"""Recompute S13 logical read/write service from a sealed formal archive."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

from report_async_copy_v1 import read, require, sha, stats, table


def render(run, replay_path, out):
    run, replay_path, out = run.resolve(), replay_path.resolve(), out.resolve()
    require(not out.exists() and not out.is_relative_to(run)
            and not run.is_relative_to(out), 'new analysis directory outside run required')
    replay = read(replay_path)
    require(replay['status'] == 'pass_under_bounded_float_replay_v1'
            and not replay['failed_checks'], 'passing bounded replay required')
    inventory = {p.relative_to(run).as_posix(): sha(p)
                 for p in run.rglob('*') if p.is_file()}
    require(inventory == replay['input_artifacts_sha256'], 'replay input identity changed')
    spec = read(run / 'run_spec.json')
    contract_path = (run / spec['contract_path']).resolve()
    require(contract_path.is_relative_to(run), 'contract escapes run')
    contract, summary = read(contract_path), read(run / 'summary.json')
    require(contract['adapter_id'] == 'global_duplex_formal_v1'
            and len(contract['cases']) == 18 and summary['terminal'],
            'terminal S13 formal archive required')
    cases = {c['id']: c for c in contract['cases']}
    require({r['case_id'] for r in summary['cases']} == set(cases), 'case coverage')
    device = read(run / 'environment/device.json')
    trials, batches, totals, intervals = [], [], [], []
    used_raw, used_receipts = set(), set()
    for result in summary['cases']:
        case = cases[result['case_id']]
        p = case['parameters']
        accepted, all_case_rows = [], []
        for batch in result['batches']:
            values = []
            for sample in batch['samples']:
                receipt_path = (run / sample['receipt']).resolve()
                raw_path = receipt_path.parent / 'raw.jsonl'
                require(receipt_path.is_relative_to(run) and raw_path.is_relative_to(run),
                        'sample escapes run')
                require(raw_path not in used_raw and receipt_path not in used_receipts,
                        'duplicate sample')
                used_raw.add(raw_path)
                used_receipts.add(receipt_path)
                require(sha(raw_path) == sample['raw_sha256'], 'raw hash drift')
                rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
                require(len(rows) == 2 and rows[1]['type'] == 'trial', 'device/trial rows')
                raw, receipt = rows[1], read(receipt_path)
                require(raw['case_id'] == case['id'] and raw['iterations'] == 16
                        and raw['threads'] == 256 and raw['scope'] == 'all_gpu',
                        'frozen dimensions')
                require(raw['seed'] == 3 + 19 * sample['trial'] + 1009 * sample['batch'],
                        'seed schedule')
                blocks = device['sms'] * min(4, raw['occupancy_limit_ctas_per_sm'])
                requested = (device['l2_cache_bytes'] // 4 if p['working_set_class'] == 'small'
                             else 4 * device['l2_cache_bytes'])
                quantum = blocks * 256 * 16
                size = ((requested + quantum - 1) // quantum) * quantum
                reads, writes = p['read_requests_per_group'], p['write_requests_per_group']
                rd, wr = 16 * size * reads, 16 * size * writes
                checked = (blocks * 256 if reads else 0) + (size // 4 if writes else 0)
                require(raw['blocks'] == blocks and raw['array_bytes'] == size
                        and raw['aggregate_array_bytes'] == size * (bool(reads) + bool(writes)),
                        'aligned working set')
                require(raw['read_payload_bytes'] == rd and raw['write_payload_bytes'] == wr
                        and raw['work_count'] == rd + wr, 'independent directional work')
                require(raw['errors'] == 0 and raw['correctness']['checked_elements'] == checked,
                        'full declared check count')
                details = raw['blocks_detail']
                require(len(details) == blocks and len({b['smid'] for b in details}) == device['sms'],
                        'CTA/SM coverage')
                start = min(b['start_ns'] for b in details)
                stop = max(b['stop_ns'] for b in details)
                require(raw['start_ns'] == start and raw['stop_ns'] == stop and stop > start,
                        'globaltimer envelope')
                rate = Fraction(rd + wr, stop - start)
                require(float(rate) == sample['value']
                        and sample['unit'] == 'GB/s_requested_payload/GPU', 'rate/unit')
                row = {'case_id': case['id'], 'mode': p['mode'],
                       'read_cache_policy': p['read_cache_policy'],
                       'working_set_class': p['working_set_class'],
                       'read_ratio': reads, 'write_ratio': writes,
                       'batch': sample['batch'], 'trial': sample['trial'], 'seed': raw['seed'],
                       'blocks': blocks, 'array_bytes': size,
                       'aggregate_array_bytes': raw['aggregate_array_bytes'],
                       'read_payload_bytes': rd, 'write_payload_bytes': wr,
                       'work_count': rd + wr, 'start_ns': start, 'stop_ns': stop,
                       'elapsed_ns': stop - start, 'value': float(rate),
                       'read_GB_s': float(Fraction(rd, stop - start)),
                       'write_GB_s': float(Fraction(wr, stop - start)),
                       'exact_rate_fraction': str(rate), 'event_ms': raw['event_ms'],
                       'warmup_converged': sample['warmup_converged'],
                       'warmup_windows': len(raw['warmup_samples_ns']),
                       'checked_elements': checked, 'status': result['status'],
                       'raw': raw_path.relative_to(run).as_posix(), 'raw_sha256': sha(raw_path)}
                trials.append(row)
                all_case_rows.append(row)
                intervals.append((receipt['host_start_ns'], receipt['host_stop_ns']))
                if sample['warmup_converged']:
                    values.append(rate)
                    accepted.append(rate)
            bs = stats(values)
            batches.append({'case_id': case['id'], 'batch': batch['batch'],
                            'recorded_processes': len(batch['samples']),
                            'complete': batch['complete'], 'warmup_failed': batch['warmup_failed'],
                            **bs})
        merged = stats(accepted)
        if result['status'] == 'stable':
            require(batch['complete'] and Fraction(bs['cv_squared_exact']) <= Fraction(1, 400)
                    and Fraction(merged['cv_squared_exact']) <= Fraction(1, 400),
                    'exact stable threshold')
        require(result['merged'] is None if len(accepted) < 2
                else result['merged']['samples'] == len(accepted), 'merged count')
        totals.append({'case_id': case['id'], 'mode': p['mode'],
                       'working_set_class': p['working_set_class'],
                       'read_ratio': reads, 'write_ratio': writes,
                       'status': result['status'], 'exportable_after_C': result['exportable'],
                       'processes': len(all_case_rows), 'batches': len(result['batches']),
                       'unit': 'GB/s_requested_payload/GPU', **merged})
    intervals.sort()
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), 'process overlap')
    out.mkdir(parents=True)
    table(out / 'trials.csv', trials)
    table(out / 'batches.csv', batches)
    table(out / 'cases.csv', totals)
    example = next(r for r in trials if r['case_id'] == 'independent_r2_w1_large')
    text = (f"# S13 原始字段手算\n\n原始路径：`{example['raw']}`。\n\n"
            f"每 active array={example['array_bytes']} B；16轮，读写比2:1。\n\n"
            f"读请求=16×{example['array_bytes']}×2={example['read_payload_bytes']} B；"
            f"写请求=16×{example['array_bytes']}×1={example['write_payload_bytes']} B。\n\n"
            f"包络时间={example['stop_ns']}−{example['start_ns']}={example['elapsed_ns']} ns。\n\n"
            f"总Q=读请求+写请求={example['read_payload_bytes']}+{example['write_payload_bytes']}"
            f"={example['work_count']} B。\n\n"
            f"总逻辑请求速率={example['work_count']}/{example['elapsed_ns']}"
            f"={example['value']:.12g} GB/s（约分值{example['exact_rate_fraction']}）。"
            "采用指定配置的第一条记录，不挑选最快样本。逻辑请求量不能作为物理HBM流量。\n")
    (out / 'manual-example.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'gh200-global-duplex-v1'
    import matplotlib.pyplot as plt
    for size in ('small', 'large'):
        selected = [r for r in totals if r['working_set_class'] == size and r['samples']]
        fig, axis = plt.subplots(figsize=(11, 5), layout='constrained')
        axis.bar(range(len(selected)), [r['mean'] for r in selected],
                 yerr=[r['mean'] * r['cv'] for r in selected], capsize=3)
        axis.set_xticks(range(len(selected)), [r['case_id'].removesuffix('_' + size)
                                               for r in selected], rotation=35, ha='right')
        axis.set_ylabel('Logical requested GB/s (GPU globaltimer)')
        axis.set_title(size + ': mean; error bars = process sample SD; C review pending')
        fig.savefig(out / (size + '.svg'), metadata={'Date': None})
        plt.close(fig)
    provenance = {'status': 'analysis_pending_independent_C', 'run_id': run.name,
                  'replay_sha256': sha(replay_path), 'input_artifacts_sha256': inventory,
                  'report_source_sha256': sha(Path(__file__)),
                  'helper_source_sha256': sha(Path(__file__).with_name('report_async_copy_v1.py')),
                  'cases': len(totals), 'processes': len(trials),
                  'logical_bytes_only': True, 'physical_HBM_evidence': False}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    outputs = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / 'outputs.json').write_text(json.dumps(outputs, indent=2) + '\n')
    return {'cases': len(totals), 'processes': len(trials), 'out': str(out)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.replay, args.out), indent=2))
