#!/usr/bin/env python3
"""逐进程展开新压缩输出，复用冻结资源 oracle，汇总本轮资格与旧参数对照。"""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

from q00_a044_packed_oracle import ResourceOracle, file_sha, read

sys.dont_write_bytecode = True


def unpack(folder, target):
    index = read(folder / 'output-archive.json')
    archive = folder / index['archive']
    assert file_sha(archive) == index['sha256']
    seen = set()
    with tarfile.open(archive) as packet:
        for member in packet:
            assert member.isfile() and member.name in index['members'] and member.name not in seen
            relative = Path(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with packet.extractfile(member) as source, path.open('xb') as output:
                shutil.copyfileobj(source, output, 1024*1024)
            expected = index['members'][member.name]
            assert path.stat().st_size == expected['size'] and file_sha(path) == expected['sha256']
            seen.add(member.name)
    assert seen == set(index['members'])
    receipt = read(folder / 'receipt.json')
    for name in ('stdout', 'stderr'):
        assert file_sha(target / name) == receipt[name + '_sha256']
    def unique(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result
            result[key] = value
        return result
    rows = [json.loads(line, object_pairs_hook=unique,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            for line in (target / 'stdout').read_text().splitlines()]
    return receipt, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True,
                        help='一个完整输出包展开后的目录，含原打包文件和 results 子目录')
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--statistics-source', type=Path)
    args = parser.parse_args()
    if args.statistics_source:
        spec = importlib.util.spec_from_file_location('statistics', args.statistics_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['statistics'] = module
    inputs = []
    for run in args.run:
        packages = [run] if (run / 'package-identity.json').is_file() else [p.parent for p in run.glob('*/package-identity.json')]
        captures = [p.parent for p in run.glob('*/capture-summary.json')]
        assert len(packages) == len(captures) == 1
        package, capture = packages[0], captures[0]
        manifest = read(package / 'manifest.json')
        assert manifest['subset']['family'] == args.family
        assert read(capture / 'request.json')['manifest_sha256'] == file_sha(package / 'manifest.json')
        for name, checksum in read(package / 'package-identity.json').items():
            assert file_sha(package / name) == checksum
        inputs.append(dict(package=package, capture=capture, manifest=manifest))
    inputs.sort(key=lambda item: item['manifest']['subset']['index'])
    assert [item['manifest']['subset']['index'] for item in inputs] == list(range(len(inputs)))
    assert all(item['manifest']['subset']['chunks'] == len(inputs) for item in inputs)
    task_ids = [t['id'] for item in inputs for t in item['manifest']['tasks']]
    assert len(set(task_ids)) == len(task_ids)
    family_info = next(f for f in read(args.inventory)['resources'] if f['id'] == args.family)
    original_manifest = Path(family_info['prepared_package']) / 'manifest.json'
    assert all(item['manifest']['source_manifest_sha256'] == file_sha(original_manifest) for item in inputs)
    original_tasks = {t['id']: t for t in read(original_manifest)['tasks'] if t['stages'] == ['RES-' + args.family]}
    assert set(original_tasks) <= set(task_ids)
    extra_queries = {'dsm_multicast': 2, 'fp32_combination': 1, 'bf16_wgmma_combination': 2}.get(args.family, 0)
    assert len(task_ids) == len(original_tasks) + extra_queries
    for dataset in inputs:
        for task in dataset['manifest']['tasks']:
            if task['id'] in original_tasks:
                assert task == original_tasks[task['id']]
            else:
                assert task['protocol']['original_kind'] == 'resources'
    parameters_path = Path(family_info['parameter_source'])
    assert file_sha(parameters_path) == family_info['parameter_source_sha256']
    old_parameters = read(parameters_path)
    published = {r['case_id']: r for r in old_parameters['parameters']}
    excluded = {r['case_id']: r for r in old_parameters.get('excluded', [])}
    roots = [Path(run['path']).resolve() for run in family_info['source_runs']]
    code_roots = [r.parent / 'repo' if args.family == 'tma_tensor_2d' else r / 'repo' for r in roots]
    oracle = ResourceOracle(args.family, code_roots[0])
    if args.family == 'tma_tensor_2d':
        for source in code_roots[1:]:
            for name in ('run_candidate.py', 'reference_rows.py',
                         'microbench/gh200_resource_campaign/auditors/observation.py',
                         'microbench/gh200_resource_campaign/auditors/tma_tensor_2d_validation_v1.py',
                         'microbench/gh200_resource_campaign/contracts/tma_tensor_2d.json',
                         'microbench/gh200_resource_campaign/contracts/protocol.json'):
                assert file_sha(source / name) == file_sha(code_roots[0] / name)
    from common.suite_io import stats
    args.scratch.mkdir(parents=True, exist_ok=True)
    device, samples, intervals = None, [], []
    for dataset in inputs:
        package, capture, manifest = dataset['package'], dataset['capture'], dataset['manifest']
        allocation = read(capture / 'capture-summary.json')['allocation']
        for task in manifest['tasks']:
            assert task['stages'] == ['RES-' + args.family] and task['repeats'] == 1
            folder = capture / 'runs' / task['id'] / '000'
            with tempfile.TemporaryDirectory(prefix=args.family+'-', dir=args.scratch) as temporary:
                target = Path(temporary)
                receipt, rows = unpack(folder, target)
                assert receipt['returncode'] == 0 and not receipt['timed_out']
                assert receipt['argv'][1:] == task['argv'][1:]
                assert Path(receipt['argv'][0]).name == Path(task['argv'][0]).name
                assert file_sha(package / task['argv'][0]) == Path(task['argv'][0]).name
                assert receipt['stop_ns'] > receipt['start_ns']
                assert rows[0]['uuid'] == allocation['uuid']
                if device is None:
                    device = rows[0]
                assert rows[0] == device
                point = task['origins'][0]['point']
                kind = task['protocol']['original_kind']
                sample = dict(task=task['id'], case_id=point.get('case_id'), kind=kind,
                    batch=point.get('sampling_batch', point.get('batch')),
                    trial=point.get('trial'), seed=int(task['argv'][-1]) if kind != 'resources' else None,
                    source_receipt=str(folder / 'receipt.json'), stdout_sha256=receipt['stdout_sha256'],
                    host_seconds=(receipt['stop_ns']-receipt['start_ns'])/1e9)
                intervals.append((receipt['start_ns'], receipt['stop_ns'], allocation['job_id'], receipt['pid']))
                try:
                    sample['result'] = oracle.check(target, task, rows)
                    sample['oracle_pass'] = True
                except (ValueError, AssertionError, KeyError) as error:
                    sample['oracle_pass'] = False
                    sample['oracle_error'] = type(error).__name__ + ': ' + str(error)
                samples.append(sample)
            if len(samples) % 50 == 0:
                print(args.family, len(samples), '/', len(task_ids), flush=True)
    assert len({(a, job, pid) for a, b, job, pid in intervals}) == len(intervals)
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    cases = []
    for case_id, case in oracle.cases.items():
        related = [s for s in samples if s['case_id'] == case_id]
        formal = [s for s in related if s['kind'] == 'formal']
        merged, batches = [], []
        for batch in sorted({s['batch'] for s in formal}):
            subset = [s for s in formal if s['batch'] == batch]
            values = [s['result']['value'] for s in subset if s['oracle_pass'] and s['result']['warmup_converged']]
            merged.extend(values)
            current = stats(values) if len(values) >= 2 else None
            stable = len(values) == oracle.protocol['trials_per_batch'] and current['cv'] <= oracle.protocol['formal_cv_limit'] and stats(merged)['cv'] <= oracle.protocol['formal_cv_limit']
            batches.append(dict(batch=batch, processes=len(subset), valid_warm=len(values), stats=current, stable=stable))
        new = stats(merged) if len(merged) >= 2 else None
        old = published.get(case_id)
        if old is None and 'value' in excluded.get(case_id, {}):
            old = excluded[case_id]
        unit = next((s['result']['unit'] for s in formal if s['oracle_pass']), None)
        if old and unit:
            assert unit == old['unit']
        cases.append(dict(case_id=case_id, conditions=case, unit=unit, processes=len(related),
            oracle_pass=sum(s['oracle_pass'] for s in related),
            warmup_failed=sum(s['oracle_pass'] and not s['result']['warmup_converged'] for s in formal),
            short_checks=sum(s['kind'] == 'short' for s in related),
            old_published=case_id in published, original_exclusion=excluded.get(case_id),
            old=old, new=new, batches=batches,
            fixed_replay_statistically_eligible=bool(batches and batches[-1]['stable'] and all(s['oracle_pass'] for s in related)),
            median_change_percent=(new['median']/old['value']-1)*100 if new and old else None))
    result = dict(family=args.family, device=device, oracle_roots=list(map(str, code_roots)),
        frozen_source_manifest_sha256=oracle.source_manifest_sha256,
        analysis_sha256=file_sha(Path(__file__)), oracle_adapter_sha256=file_sha(Path(__file__).with_name('q00_a044_packed_oracle.py')),
        old_parameters=dict(path=str(parameters_path), sha256=file_sha(parameters_path)),
        resources=oracle.resources, totals=dict(processes=len(samples),
            kinds=dict(collections.Counter(s['kind'] for s in samples)),
            oracle_pass=sum(s['oracle_pass'] for s in samples),
            warmup_failed=sum(c['warmup_failed'] for c in cases),
            eligible_cases=sum(c['fixed_replay_statistically_eligible'] for c in cases),
            published_eligible_cases=sum(c['old_published'] and c['fixed_replay_statistically_eligible'] for c in cases),
            process_seconds=sum(s['host_seconds'] for s in samples)), cases=cases, samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['totals'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
