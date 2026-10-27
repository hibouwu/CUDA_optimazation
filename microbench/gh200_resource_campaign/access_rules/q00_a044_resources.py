#!/usr/bin/env python3
"""从已发布资源家族恢复原调用；只准备 a044 新包，不提交 GPU 作业。"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from q00_a044_remeasure import digest, executable, read, save

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def packed_records(root):
    """逐包读取命令，不把历史 full-output 展开到磁盘。"""
    for label in ('legacy-capabilities', 'measured-capabilities', 'capability',
                  'legacy-capability', 'measured-capability'):
        path = root / label / 'receipt.json'
        if path.is_file():
            yield dict(receipt=read(path), point=dict(kind='resources', query=label),
                source=str(path), source_receipt_sha256=digest(path))
    for pack in sorted((root / 'packs').glob('*.tar.*')):
        receipt_bytes, point = None, {}
        # receipt 之后是完整输出；流式读到 receipt 即止，避免重复解压大输出。
        with tarfile.open(pack, mode='r|*') as archive:
            for member in archive:
                name = member.name.lstrip('./')
                if name in ('point.json', 'coordinate.json', 'context.json'):
                    value = json.load(archive.extractfile(member))
                    if isinstance(value, dict):
                        point.update(value)
                if name == 'receipt.json':
                    receipt_bytes = archive.extractfile(member).read()
                    break
        if receipt_bytes is None:
            continue
        yield dict(receipt=json.loads(receipt_bytes), point=point,
            source=str(pack) + '::receipt.json',
            source_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            source_pack_sha256=digest(pack))


def plain_records(root):
    # snapshot 内有旧预检副本，只读取这次正式 run 自己的 batches。
    for path in sorted((root / 'batches').rglob('receipt.json')):
        receipt = read(path)
        yield dict(receipt=receipt, point=dict(kind='formal',
            case_id=receipt.get('case_id'), trial=receipt.get('trial'),
            batch=receipt.get('batch'), seed=receipt.get('seed')),
            source=str(path), source_receipt_sha256=digest(path))


def supplemental_short_tasks(name, family_tasks, source_runs):
    """原正式包复用过的短检查，在新卡重新执行已有 profile。"""
    pairs = []
    if name == 'tma_tensor_2d':
        # 原 68 条件各有三个非均匀数值 profile；正式包只收了 pilot/formal。
        cases = {}
        for task in family_tasks:
            argv = task['argv']
            if len(argv) > 2 and argv[1] == 'formal-only':
                cases[argv[2]] = task
        for case, task in sorted(cases.items()):
            for profile, seed in [('tensor_short_1_seed0_v1', '0'),
                                  ('tensor_short_2_seed3_v1', '3'),
                                  ('tensor_short_33_seed4294967295_v1', '4294967295')]:
                pairs.append((case, profile, seed, task,
                    str(HERE.parent / 'probes/tma_tensor_2d.cu')))
    if name in ('tma_concurrency', 'fp32_combination'):
        for source_run in source_runs:
            prior = REPO / source_run['path'] / 'prior-short-reuse.json'
            if not prior.exists():
                continue
            old = read(prior)
            coordinates = ([(case, 'stage_short_1_2_5_33_v1', '3')
                            for case in old.get('case_ids', [])] if name == 'tma_concurrency'
                           else [(case, profile, '1') for case, profile in old.get('coordinates', [])])
            for case, profile, seed in coordinates:
                task = next(task for task in family_tasks if len(task['argv']) > 2
                            and task['argv'][1] == 'formal-only' and task['argv'][2] == case)
                pairs.append((case, profile, seed, task, str(prior)))
    result = []
    for case, profile, seed, example, source in pairs:
        argv = [example['argv'][0], 'validate-only', case, profile, seed]
        key = hashlib.sha256(json.dumps(argv).encode()).hexdigest()[:20]
        result.append(dict(id=name + '-short-' + key, stages=['RES-' + name], argv=argv,
            repeats=1, timeout_seconds=120,
            protocol=dict(profile='plain', original_kind='short', packed_full_output=True,
                stdout_format='resource_jsonl', cache_and_warmup='original_validation_profile',
                full_output='per_process_cwd'),
            origins=[dict(run=example['origins'][0]['run'], receipt=source,
                restored_prior_short=True, original_start_ns=0,
                point=dict(case_id=case, kind='short', profile=profile, seed=int(seed)))]))
    return result


def prepare(args):
    inventory = read(args.inventory)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks, coverage, omitted, assets = [], [], [], {}
    for family in inventory['resources']:
        name = family['id']
        if args.family and name not in args.family:
            continue
        print('prepare', name, flush=True)
        family_tasks = []
        parameter_data = read(REPO / family['parameter_source'])
        published_cases = {row.get('case_id', row.get('case', {}).get('id'))
                           for row in parameter_data.get('parameters', parameter_data.get('records', []))
                           if row.get('family', name) == name}
        published_cases.discard(None)
        if len(published_cases) != family['qualified_conditions']:
            raise ValueError('发布条件数未匹配：' + name)
        for run in family['source_runs']:
            root = REPO / run['path']
            binaries = {digest(REPO / path): REPO / path for path in run['binary_candidates']}
            records = packed_records(root) if (root / 'packs').exists() else plain_records(root)
            for record in records:
                receipt, point = record['receipt'], record['point']
                if 'kind' not in point and 'mode' in point:
                    point['kind'] = point['mode'].removesuffix('-only')
                argv = receipt.get('argv', [])
                if not argv:
                    omitted.append(dict(family=name, source=record['source'], reason='原记录无 argv'))
                    continue
                if receipt.get('returncode') != 0 or receipt.get('timed_out'):
                    omitted.append(dict(family=name, source=record['source'], reason='历史失败调用保留'))
                    continue
                wanted = receipt.get('binary_sha256') or point.get('binary_sha256')
                matching = [(checksum, path) for checksum, path in binaries.items()
                            if wanted == checksum or (wanted is None and path.name == Path(argv[0]).name)]
                if len(matching) != 1:
                    omitted.append(dict(family=name, source=record['source'], reason='未唯一匹配原二进制',
                        wanted_binary_sha256=wanted, original_argv=argv))
                    continue
                checksum, binary = matching[0]
                if not executable(binary):
                    raise ValueError('原二进制不是 ARM ELF：' + str(binary))
                assets[checksum] = dict(source=str(binary), kind='binary')
                # 这些资源探针在 cwd 写完整输出，原 argv 全部为标量参数。
                if any('/' in str(value) for value in argv[1:]):
                    omitted.append(dict(family=name, source=record['source'],
                        reason='需绑定原路径参数', original_argv=argv))
                    continue
                origin = dict(run=str(root), receipt=record['source'], original_argv=argv,
                    original_returncode=receipt['returncode'],
                    original_start_ns=receipt.get('host_start_ns'),
                    source_receipt_sha256=record['source_receipt_sha256'],
                    source_pack_sha256=record.get('source_pack_sha256'), point=point)
                key = hashlib.sha256(record['source'].encode()).hexdigest()[:20]
                task = dict(id=name + '-' + key, stages=['RES-' + name],
                    argv=['assets/' + checksum, *map(str, argv[1:])], repeats=1,
                    timeout_seconds=receipt.get('timeout_seconds', 120),
                    protocol=dict(profile='plain', cache_and_warmup='original_binary_and_argv',
                                  original_kind=point.get('kind'), full_output='per_process_cwd',
                                  packed_full_output=True, stdout_format='resource_jsonl',
                                  published_case=point.get('case_id') in published_cases),
                    origins=[origin])
                family_tasks.append(task)
        # 原 run 的调用顺序保留；短数值检查与 pilot 位于正式记录之前。
        family_tasks.sort(key=lambda task: task['origins'][0]['original_start_ns'] or 0)
        extra = supplemental_short_tasks(name, family_tasks, family['source_runs'])
        after_resources = max((index + 1 for index, task in enumerate(family_tasks)
                               if task['protocol']['original_kind'] == 'resources'), default=0)
        family_tasks[after_resources:after_resources] = extra
        tasks.extend(family_tasks)
        cases = {task['origins'][0]['point'].get('case_id') for task in family_tasks
                 if task['protocol']['original_kind'] == 'formal'} - {None}
        coverage.append(dict(family=name, qualified_conditions=family['qualified_conditions'],
            formal_case_ids=sorted(cases), published_case_ids=sorted(published_cases),
            missing_published_case_ids=sorted(published_cases - cases),
            scheduled_processes=len(family_tasks), restored_short_profiles=len(extra),
            parameter_source=family['parameter_source'],
            parameter_source_sha256=family['parameter_source_sha256'],
            kinds=dict(collections.Counter(t['protocol']['original_kind'] for t in family_tasks)),
            status='prepared' if family_tasks else 'unresolved'))
        print(name, len(family_tasks), 'calls', flush=True)
    manifest = dict(schema='gh200-a044-remeasure/1', source=str(args.inventory.resolve()),
        target_node='romeo-a044', target_clock_mhz=1980,
        qualification='原资源程序与过程调用重测；采集结果另用原数值审计与统计验收',
        deduplication='每份原 run receipt 一次；保留原独立次数、seed和调用顺序',
        tasks=tasks, assets=assets, coverage=coverage, omitted=omitted,
        totals=dict(families=len(coverage), qualified_conditions=sum(x['qualified_conditions'] for x in coverage),
            scheduled_processes=len(tasks), omitted_calls=len(omitted),
            restored_short_profiles=sum(x['restored_short_profiles'] for x in coverage),
            stages={x['family']: x['scheduled_processes'] for x in coverage}))
    save(output / 'manifest.json', manifest)
    for name in ('q00_a044_remeasure.py', 'q00_a044_resources.py', 'l2_saturation.py'):
        shutil.copy2(HERE / name, output / name)
    (output / 'assets').mkdir()
    for checksum, asset in assets.items():
        path = output / 'assets' / checksum
        shutil.copy2(asset['source'], path)
        path.chmod(path.stat().st_mode | 0o100)
    save(output / 'package-identity.json', {str(path.relative_to(output)): digest(path)
         for path in output.rglob('*') if path.is_file()})
    print(json.dumps(manifest['totals'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', action='append')
    prepare(parser.parse_args())
