#!/usr/bin/env python3
"""Render S04 v2 evidence outside its immutable run; never grant C qualification."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('artifact path outside run: ' + name)
    return path


def render(root, replay_path, output):
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output) or output.exists():
        raise ValueError('output must be a new directory outside the immutable run')
    replay = read(replay_path)
    if replay['status'] != 'pass_under_bounded_float_replay_v1' or replay['failed_checks']:
        raise ValueError('accepted explicit replay receipt required')
    actual = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
    if actual != replay['input_artifacts_sha256']:
        raise ValueError('complete replay input manifest differs')
    summary = read(root / 'summary.json')
    if summary['family'] != 'memory_baseline' or not summary['terminal']:
        raise ValueError('terminal S04 measurement archive required')
    records, examples = [], []
    for case in summary['cases']:
        for batch in case['batches']:
            for sample in batch['samples']:
                raw_path = safe(root, sample['receipt']).parent / 'raw.jsonl'
                if sha(raw_path) != sample['raw_sha256']:
                    raise ValueError('raw hash mismatch')
                rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
                raw = next(row for row in rows if row.get('type') == 'trial')
                blocks = raw['blocks_detail']
                cycles = blocks[0]['stop_cycle'] - blocks[0]['start_cycle'] if len(blocks) == 1 else None
                ns = max(b['stop_ns'] for b in blocks) - min(b['start_ns'] for b in blocks)
                if case['unit'] == 'B/clock64_cycle/CTA': value = raw['work_count'] / cycles
                elif case['unit'] == 'GB/s_requested_payload/GPU': value = raw['work_count'] / ns
                elif case['unit'] == 'cycles/window': value = cycles
                elif case['unit'] == 'ns/window': value = ns
                else: raise ValueError('unknown metric unit')
                if value != sample['value']:
                    raise ValueError('raw-derived metric differs: ' + case['case_id'])
                records.append({'case_id': case['case_id'], 'batch': sample['batch'], 'trial': sample['trial'],
                    'scope': case['scope'], 'unit': case['unit'], 'value': value,
                    'warmup_converged': sample['warmup_converged'], 'included_in_merged': sample['warmup_converged'],
                    'status': case['status'], 'raw': raw_path.relative_to(root).as_posix(), 'raw_sha256': sha(raw_path)})
                if sample['batch'] == 0 and sample['trial'] == 0:
                    examples.append({'case_id': case['case_id'], 'work_count': raw['work_count'], 'cycles': cycles,
                        'elapsed_ns': ns, 'metric': value, 'unit': case['unit'], 'raw': raw_path.relative_to(root).as_posix(),
                        'requested_working_set_bytes': raw['requested_working_set_bytes'],
                        'allocation_per_array_bytes': raw['allocation_per_array_bytes']})
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gh200-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'gh200-memory-v2'
    output.mkdir(parents=True)
    with (output / 'samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (output / 'worked_examples.json').write_text(json.dumps(examples, indent=2) + '\n')
    for name, prefix, unit in [('smem', 'smem_', 'Requested B / clock64 cycle / CTA'),
                                ('global', 'global_', 'Requested payload GB/s / GPU')]:
        cases = [c for c in summary['cases'] if c['case_id'].startswith(prefix)]
        fig, ax = plt.subplots(figsize=(10, 5))
        for y, case in enumerate(cases):
            values = [r['value'] for r in records if r['case_id'] == case['case_id'] and r['included_in_merged']]
            median = statistics.median(values)
            color = '#b85b19' if case['status'] != 'stable' else '#24678d'
            ax.scatter(values, [y] * len(values), s=12, alpha=.5, color=color)
            ax.errorbar(median, y, xerr=[[median-min(values)], [max(values)-median]], fmt='D', color=color, capsize=4)
        ax.set_yticks(range(len(cases)), [c['case_id'] + (' [unstable]' if c['status'] != 'stable' else '') for c in cases])
        ax.invert_yaxis(); ax.set_xlabel(unit); ax.grid(axis='x', alpha=.2)
        ax.set_title('GH200 S04: median, min/max and all accepted samples\nC review pending; cache residency and physical HBM traffic unproven')
        fig.tight_layout()
        fig.savefig(output / (name + '.svg'), metadata={'Date': None})
        fig.savefig(output / (name + '.png'), dpi=160)
        plt.close(fig)
    selected = [r for r in records if r['case_id'] == 'smem_read_stride1']
    fig, ax = plt.subplots(figsize=(9, 4))
    for batch in sorted({r['batch'] for r in selected}):
        rows = [r for r in selected if r['batch'] == batch]
        ax.scatter([r['batch'] * 10 + r['trial'] for r in rows], [r['value'] for r in rows], label='batch ' + str(batch))
    ax.set_xlabel('Process index within this configuration'); ax.set_ylabel('Requested B / clock64 cycle / CTA')
    ax.set_title('SMEM stride=1: all 30 process samples; no scalar qualification')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(output / 'stride1.png', dpi=160); fig.savefig(output / 'stride1.svg', metadata={'Date': None}); plt.close(fig)
    shutil.copyfile(__file__, output / Path(__file__).name)
    manifest = {'schema_version': 1, 'qualification': 'analysis_pending_independent_C_review',
        'run_name': root.name, 'summary_sha256': sha(root / 'summary.json'), 'replay_sha256': sha(replay_path),
        'raw_inventory_sha256': replay['input_manifest_sha256'], 'matplotlib_version': matplotlib.__version__,
        'sample_rows': len(records), 'merged_sample_rows': sum(r['included_in_merged'] for r in records),
        'error_bars': 'min/max of all warmup-converged process samples; not confidence intervals',
        'parameter_export': False,
        'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.run, args.replay, args.output)
