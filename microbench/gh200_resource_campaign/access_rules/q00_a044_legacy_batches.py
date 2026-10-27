#!/usr/bin/env python3
"""CPU: split legacy replay tasks into disjoint, time-budgeted packages."""
import argparse
import collections
from functools import lru_cache
import json
from pathlib import Path
import shutil
import struct
import tarfile

from q00_a044_remeasure import digest, read, save


@lru_cache(maxsize=256)
def records(path):
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    try:
        value = read(path)
    except json.JSONDecodeError:
        return []
    return [value] if isinstance(value, dict) else []


def duration(origin):
    path = Path(origin['record'])
    options = [path]
    if path.name == 'command.json':
        options += [path.parent / name for name in ('process.json', 'receipt.json', 'result.json')]
    for option in options:
        if not option.exists():
            continue
        for row in records(option):
            if option.suffix == '.jsonl' and (row.get('case') != origin['case'] or
                    row.get('variant') != origin['variant'] or row.get('trial') != origin['trial']):
                continue
            for start, stop in (('host_start_ns', 'host_stop_ns'),
                                ('started_unix_ns', 'ended_unix_ns'), ('start_ns', 'end_ns')):
                if row.get(start) and row.get(stop) and row[stop] > row[start]:
                    return (row[stop] - row[start]) / 1e9, str(option), output_bytes(option, row)
    return None, None, output_bytes(path, {})


def output_bytes(path, row):
    def uncompressed_size(file):
        if file.suffix == '.gz':
            with file.open('rb') as stream:
                stream.seek(-4, 2)
                return max(file.stat().st_size, struct.unpack('<I', stream.read(4))[0])
        return file.stat().st_size
    if row.get('raw'):
        for parent in path.parents:
            candidate = parent / row['raw']
            if candidate.is_file():
                return uncompressed_size(candidate)
    if path.name in ('command.json', 'result.json', 'process.json', 'measure.process.json'):
        return sum(uncompressed_size(p) for p in path.parent.iterdir() if p.is_file())
    return path.stat().st_size


def estimate(task):
    by_run = collections.defaultdict(list)
    for origin in task['origins']:
        by_run[origin['run']].append(origin)
    chosen = max(by_run.values(), key=len)
    assert len(chosen) == task['repeats']
    samples = []
    for origin in chosen:
        seconds, source, size = duration(origin)
        samples.append(dict(seconds=seconds, timing_source=source,
                            original_output_bytes=size, origin_record=origin['record']))
    measured = [row['seconds'] for row in samples if row['seconds'] is not None]
    missing = len(samples) - len(measured)
    # 未记主机时间的旧进程以 5 秒规划；已记时间取 1.5 倍。
    sampling = 1.5 * sum(measured) + 5 * missing
    # 原输出展开字节计三遍：写出、打包、解包校验；每遍按20MiB/s估计。
    packaging = .25 * len(samples) + 3 * sum(row['original_output_bytes'] for row in samples) / (20 * 1024 * 1024)
    return dict(task=task['id'], processes=task['repeats'], original_measured_seconds=sum(measured),
        measured_processes=len(measured), unknown_processes=missing,
        sampling_budget_seconds=sampling, output_budget_seconds=packaging,
        total_budget_seconds=sampling + packaging, samples=samples)


def main(args):
    source = args.package.resolve()
    manifest = read(source / 'manifest.json')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    estimates = {task['id']:estimate(task) for task in manifest['tasks']}
    tasks = sorted(manifest['tasks'], key=lambda task: (
        'V09-heldout-regression' not in task['stages'],
        tuple(task['stages']), task['id']))
    groups, current, budget = [], [], 0
    current_v09 = None
    for task in tasks:
        cost = estimates[task['id']]['total_budget_seconds']
        v09 = 'V09-heldout-regression' in task['stages']
        if cost > args.seconds:
            raise ValueError('单task超预算，保持次数不拆分：' + task['id'] + ' ' + str(cost))
        if current and (budget + cost > args.seconds or current_v09 != v09):
            groups.append(current)
            current, budget = [], 0
        current.append(task)
        budget += cost
        current_v09 = v09
    if current:
        groups.append(current)
    plan = []
    for index, group in enumerate(groups):
        identities = {value.split('/', 1)[1] for task in group for value in task['argv'] if value.startswith('assets/')}
        costs = [estimates[task['id']] for task in group]
        plan.append(dict(batch=f'legacy-{index:03}', task_ids=[task['id'] for task in group],
            stages=sorted(set().union(*(set(task['stages']) for task in group))),
            processes=sum(task['repeats'] for task in group), assets=len(identities),
            uncompressed_asset_bytes=sum((source / 'assets' / name).stat().st_size for name in identities),
            original_measured_seconds=sum(row['original_measured_seconds'] for row in costs),
            measured_processes=sum(row['measured_processes'] for row in costs),
            unknown_processes=sum(row['unknown_processes'] for row in costs),
            sampling_budget_seconds=sum(row['sampling_budget_seconds'] for row in costs),
            output_budget_seconds=sum(row['output_budget_seconds'] for row in costs),
            total_budget_seconds=sum(row['total_budget_seconds'] for row in costs)))
    assert len([task for group in groups for task in group]) == len(manifest['tasks'])
    assert len({task['id'] for group in groups for task in group}) == len(manifest['tasks'])
    totals = dict(batches=len(groups), tasks=len(tasks), processes=sum(task['repeats'] for task in tasks),
        measured_processes=sum(row['measured_processes'] for row in plan),
        unknown_processes=sum(row['unknown_processes'] for row in plan),
        original_measured_hours=sum(row['original_measured_seconds'] for row in plan) / 3600,
        planned_sampling_hours=sum(row['sampling_budget_seconds'] for row in plan) / 3600,
        planned_output_hours=sum(row['output_budget_seconds'] for row in plan) / 3600)
    save(output / 'plan.json', dict(source_manifest_sha256=digest(source / 'manifest.json'),
        per_batch_budget_seconds=args.seconds,
        budget_method='1.5 * recorded process wall + 5s per unknown process + .25s/process + 3 * original uncompressed output bytes / 20MiB/s; excludes network and allocation startup',
        totals=totals, batches=plan))
    save(output / 'runtime-evidence.json', estimates)
    print(json.dumps(totals), flush=True)
    if not args.materialize:
        return
    for row, group in zip(plan, groups):
        folder = output / row['batch']
        folder.mkdir()
        identities = {value.split('/', 1)[1] for task in group for value in task['argv'] if value.startswith('assets/')}
        selected = dict(manifest, tasks=group, assets={key:manifest['assets'][key] for key in identities},
                        parent_manifest_sha256=digest(source / 'manifest.json'), batch=row,
                        totals=dict(unique_conditions=len(group), scheduled_processes=row['processes']))
        save(folder / 'manifest.json', selected)
        (folder / 'assets').mkdir()
        for checksum in identities:
            shutil.copy2(source / 'assets' / checksum, folder / 'assets' / checksum)
        for name in ('q00_a044_remeasure.py', 'q00_a044_legacy.py', 'l2_saturation.py'):
            shutil.copy2(source / name, folder / name)
        (folder / 'README.md').write_text(
            '# ' + row['batch'] + '\n\n'
            + '来源实验：' + '、'.join(row['stages']) + '。\n\n'
            + f"本块 {len(group)} 个原程序与参数组合、{row['processes']} 个进程，保留每个组合的原独立次数。\n\n"
            + '测量原端点、时间和周期响应，检验旧条件在 a044 实际频率下是否保持。V 系列按旧冻结预测做回归；局部 R/B 实验按原数值与事件规则评分。\n\n'
            + f"可恢复的原进程主机时间合计 {row['original_measured_seconds']:.2f} 秒；{row['unknown_processes']} 个进程未记主机时间。本块采集与输出预算 {row['total_budget_seconds']:.2f} 秒，网络回传另计。\n\n"
            + '先读本块原始输出和有效频率，再比较原参照；采集成功、数值通过和机制支持分别记录。\n')
        save(folder / 'package-identity.json', {str(path.relative_to(folder)):digest(path)
            for path in folder.rglob('*') if path.is_file()})
        archive_path = folder.with_suffix('.tar.gz')
        with tarfile.open(archive_path, 'w:gz', compresslevel=1) as archive:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(folder)))
        row.update(archive_bytes=archive_path.stat().st_size, archive_sha256=digest(archive_path))
        save(output / (row['batch'] + '-receipt.json'), row)
        print(row['batch'], row['processes'], row['archive_bytes'], flush=True)
    save(output / 'packages.json', plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--materialize', action='store_true')
    main(parser.parse_args())
