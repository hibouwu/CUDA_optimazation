#!/usr/bin/env python3
"""恢复已核实的 R09 相邻配对与 R07 遥测协议，并剔除对应旧集中重复。"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from q00_a044_remeasure import digest, inspect_stdout, read, save
from q00_a044_legacy_stdout import Text, original_stdout

HERE = Path(__file__).resolve().parent
SOURCES = ('20261007-ndebug-clock-fixedcost', '20261007-r09-clock-stages',
           '20261010-R09-mirror-history')


def package(folder, manifest, asset_source, runner_source):
    folder.mkdir()
    (folder/'assets').mkdir()
    used = {value.split('/')[1] for task in manifest['tasks'] for value in task['argv']
            if value.startswith('assets/')}
    manifest['assets'] = {key: manifest['assets'][key] for key in used}
    manifest['totals'] = dict(unique_conditions=len({tuple(t['argv']) for t in manifest['tasks']}),
                             scheduled_processes=sum(t['repeats'] for t in manifest['tasks']))
    for key in used:
        shutil.copy2(asset_source/'assets'/key, folder/'assets'/key)
    for name in ('q00_a044_remeasure.py', 'l2_saturation.py'):
        shutil.copy2(runner_source/name, folder/name)
    save(folder/'manifest.json', manifest)
    save(folder/'package-identity.json', {str(p.relative_to(folder)): digest(p)
         for p in folder.rglob('*') if p.is_file()})
    archive = folder.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(folder)))
    return dict(batch=folder.name, processes=manifest['totals']['scheduled_processes'],
                tasks=len(manifest['tasks']), archive_bytes=archive.stat().st_size,
                archive_sha256=digest(archive))


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = read(args.parent/'manifest.json')
    groups = collections.defaultdict(dict)
    replaced = {}
    for task in parent['tasks']:
        origins = [origin for origin in task['origins'] if Path(origin['run']).name in SOURCES]
        if not origins:
            continue
        replaced[task['id']] = task['repeats']
        for origin in origins:
            record = read(Path(origin['record']))
            if isinstance(record, list):
                record = read(Path(origin['record']).parent/'result.json')
            assert record['returncode'] == 0
            groups[origin['run']][origin['record']] = (origin, record, task)
    counts = collections.Counter(task['id'] for rows in groups.values()
                                 for origin, record, task in rows.values())
    assert dict(counts) == replaced, (counts, replaced)
    ordered_packages, audit = [], []
    for run, rows in sorted(groups.items()):
        rows = sorted(rows.values(), key=lambda row: row[1].get('host_start_ns', row[1].get('started_unix_ns')))
        sequence = []
        for index, (origin, record, old_task) in enumerate(rows):
            task = copy.deepcopy(old_task)
            task.update(id=f'ordered-{Path(run).name}-{index:04d}', repeats=1, origins=[origin])
            task['protocol']['profile'] = Path(origin['original_argv'][0]).name.rsplit('_', 1)[-1]
            task['protocol']['original_sequence_index'] = index
            task['protocol']['original_sequence_run'] = run
            task['protocol']['replaces_legacy_task'] = old_task['id']
            task['protocol']['packed_full_output'] = True
            monitor = None
            if Path(run).name == SOURCES[0] and '--mode' in task['argv'] and task['argv'][task['argv'].index('--mode')+1] == 'loop':
                fields = read(Path(run)/'protocol-clock.json')['nvsmi_fields']
                monitor = dict(fields=fields, interval_ms=50, before_seconds=.5, after_seconds=.3,
                               source=run+'/source/r07_run.py:182-197')
                task['protocol']['external_nvsmi'] = monitor
            raw = original_stdout(origin)
            parsed = inspect_stdout(Text(raw), task['protocol'])
            assert parsed['valid'], (str(raw), parsed)
            start = record.get('host_start_ns', record.get('started_unix_ns'))
            stop = record.get('host_stop_ns', record.get('ended_unix_ns'))
            wall = (stop-start)/1e9
            budget = 1.5*wall+.25+3*len(Text(raw).read_text().encode())/(20*1024*1024)
            if monitor:
                budget += monitor['before_seconds']+monitor['after_seconds']
            sequence.append(dict(task=task, budget=budget, wall=wall,
                                 boundary=(record.get('set'), record.get('trial'))))
            audit.append(dict(record=origin['record'], original_stdout=str(raw),
                original_stdout_sha256=digest(raw), parser_status=parsed['status'],
                original_sequence_index=index, run=run, legacy_task=old_task['id'], external_nvsmi=monitor))
        # 分块只在原试次边界，保留同一试次内全部相邻条件。
        trial_groups = []
        for row in sequence:
            if not trial_groups or row['boundary'] != trial_groups[-1][0]['boundary']:
                trial_groups.append([])
            trial_groups[-1].append(row)
        chunks, chunk, current = [], [], 0
        for trial in trial_groups:
            cost = sum(row['budget'] for row in trial)
            if chunk and current+cost > 600:
                chunks.append(chunk)
                chunk, current = [], 0
            chunk.extend(trial)
            current += cost
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            assert sum(row['budget'] for row in chunk) <= 600
            name = f'ordered-{len(ordered_packages):02d}'
            manifest = dict(schema=parent['schema'], source=run, target_node=parent['target_node'],
                target_clock_mhz=parent['target_clock_mhz'], tasks=[row['task'] for row in chunk],
                assets=parent['assets'], continuous_telemetry=False,
                qualification='原顺序与外部观测协议回归；旧冻结结果保持原文件',
                sequence=dict(order='original host_start_ns/started_unix_ns',
                    first_index=chunk[0]['task']['protocol']['original_sequence_index'],
                    last_index=chunk[-1]['task']['protocol']['original_sequence_index']),
                runtime=dict(original_wall_seconds=sum(row['wall'] for row in chunk),
                             total_budget_seconds=sum(row['budget'] for row in chunk)))
            result = package(output/name, manifest, args.parent, HERE)
            result.update(source=run, **manifest['runtime'], **manifest['sequence'])
            ordered_packages.append(result)
    remaining = output/'legacy-batches-r4'
    remaining.mkdir()
    remainder_packages, affected = [], []
    for old in read(args.batches/'packages.json'):
        name = old['batch']
        folder = args.batches/name
        manifest = read(folder/'manifest.json')
        removed = [task for task in manifest['tasks'] if task['id'] in replaced]
        if not removed:
            (remaining/name).symlink_to(folder.resolve(), target_is_directory=True)
            (remaining/(name+'.tar.gz')).symlink_to(folder.with_suffix('.tar.gz').resolve())
            remainder_packages.append(old)
            continue
        kept = [task for task in manifest['tasks'] if task['id'] not in replaced]
        affected.append(dict(batch=name, removed_tasks=[task['id'] for task in removed],
                             removed_processes=sum(task['repeats'] for task in removed)))
        if not kept:
            continue
        manifest['tasks'] = kept
        manifest['batch'] = dict(manifest['batch'], task_ids=[task['id'] for task in kept],
                                 processes=sum(task['repeats'] for task in kept))
        manifest['ordered_replacement'] = '../packages.json'
        result = package(remaining/name, manifest, folder, folder)
        # 原预算是保守上界；移除任务后仍沿用它。
        remainder_packages.append(dict(old, **result, task_ids=[task['id'] for task in kept],
                                       assets=len(manifest['assets'])))
    save(output/'packages.json', ordered_packages)
    save(remaining/'packages.json', remainder_packages)
    save(output/'replaced-tasks.json', dict(replaced=replaced, affected_batches=affected))
    save(output/'original-stdout-audit.json', audit)
    save(output/'summary.json', dict(original_legacy_processes=parent['totals']['scheduled_processes'],
        replaced_conditions=len(replaced), replaced_processes=sum(replaced.values()),
        ordered_packages=len(ordered_packages), ordered_processes=sum(p['processes'] for p in ordered_packages),
        remainder_packages=len(remainder_packages),
        remainder_processes=sum(p['processes'] for p in remainder_packages),
        external_monitor_processes=sum(row['external_nvsmi'] is not None for row in audit),
        gpu_jobs_submitted=0))
    print(json.dumps(read(output/'summary.json'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--batches', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
