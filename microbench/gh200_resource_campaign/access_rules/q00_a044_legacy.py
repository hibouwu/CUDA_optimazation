#!/usr/bin/env python3
"""从 R/B/V 已执行命令准备 a044 回归包，不拟合或改写历史冻结结果。"""
import argparse
import ast
import collections
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from q00_a044_remeasure import digest, executable, read, resolve_file, save

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXCLUDED = {'source', 'source-before', 'reanalysis', 'analysis', 'analysis-source',
            'source-analysis', 'spack-cache', '__pycache__', 'build', 'compile', 'frozen', 'freeze'}


def records(root, files, omitted):
    """只读进程记录；同目录的 command/result/process 合成一个调用。"""
    seen = set()
    malformed = set()
    def record_json(path):
        try:
            return read(path)
        except json.JSONDecodeError as error:
            if path not in malformed:
                omitted.append(dict(run=str(root.relative_to(REPO)), record=str(path.relative_to(REPO)),
                    record_sha256=digest(path), reason='原记录JSON损坏：' + str(error)))
                malformed.add(path)
            return None
    paths = sorted(path for path in files if path.suffix == '.json'
                   and not set(path.relative_to(root).parts[:-1]).intersection(EXCLUDED))
    for path in paths:
        if path.name == 'command.json':
            command = record_json(path)
            if isinstance(command, dict):
                command = command.get('argv', command.get('command'))
            if not isinstance(command, list) or not command:
                continue
            evidence = None
            for name in ('receipt.json', 'process.json', 'result.json'):
                sibling = path.parent / name
                if sibling.exists():
                    value = record_json(sibling)
                    if isinstance(value, dict) and ('returncode' in value or value.get('status') in ('measured', 'ok')):
                        evidence = value
                        break
            if evidence is None:
                continue
            seen.add(path.parent)
            yield path, command, evidence
    for path in paths:
        if path.parent in seen or path.name == 'command.json':
            continue
        value = record_json(path)
        if not isinstance(value, dict):
            continue
        command = value.get('command', value.get('argv'))
        if command is None and isinstance(value.get('binary'), str) and isinstance(value.get('args'), list):
            command = [value['binary'], *value['args']]
        if isinstance(command, list) and command and 'returncode' in value:
            # sample JSON 包含完整命令；同目录可能还有其他独立 variant。
            if not any(part in ('samples', 'layout-samples', 'measurements', 'batches', 'attempts')
                       for part in path.relative_to(root).parts) and not path.name.endswith('.process.json'):
                continue
            yield path, command, value
    if root.name in ('20261008-V06-job736868', '20261008-R17-job736958'):
        source = root / 'source/v06_run.py'
        tree = ast.parse(source.read_text())
        case_args = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'case_args')
        expected = 'return [x for key in ("m", "n", "k", "lda", "ldb", "ldd") for x in (f"--{key}", str(row[key]))]'
        assert ast.dump(case_args.body[0]) == ast.dump(ast.parse(expected).body[0])
        run_process = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_process')
        expected_command = 'cmd = [str(output / "build" / f"{row[\'config\']}_{variant}"), *case_args(row)]'
        assert any(ast.dump(node) == ast.dump(ast.parse(expected_command).body[0]) for node in run_process.body)
        cases = {row['id']:row for row in read(root / 'cases.json')}
        for path in sorted((root / 'samples').glob('*.jsonl')):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                row = cases[record['case']]
                raw = root / record['raw']
                if not raw.exists():
                    omitted.append(dict(run=str(root.relative_to(REPO)), record=str(path.relative_to(REPO)),
                        case=record['case'], reason='原jsonl对应raw缺失，未重建命令'))
                    continue
                with gzip.open(raw, 'rt') as stream:
                    setup = next(json.loads(text) for text in stream if json.loads(text).get('event') == 'setup')
                assert setup['config'] == row['config']
                assert all(setup[key] == row[key] for key in ('m', 'n', 'k', 'lda', 'ldb', 'ldd'))
                command = [str(root / 'build' / (row['config'] + '_' + record['variant']))]
                command += [x for key in ('m', 'n', 'k', 'lda', 'ldb', 'ldd') for x in ('--' + key, str(row[key]))]
                record['host_start_ns'] = record['start_ns']
                record['command_reconstruction'] = dict(source=str(source.relative_to(REPO)),
                    source_sha256=digest(source), cases_sha256=digest(root / 'cases.json'),
                    raw=str(raw.relative_to(REPO)), raw_sha256=digest(raw),
                    method='原run_process/case_args AST与原raw setup逐字段核对')
                yield path, command, record


def prepare(args):
    inventory = read(args.inventory)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = collections.defaultdict(set)
    replacements, review_sources = [], []
    for entry in inventory['rbv']:
        if entry['status'] == '历史替代':
            replacements.append(dict(experiment=entry['id'], covered_by=entry['covered_by'],
                                     reason=entry['coverage_note']))
        for source in entry.get('source_rounds', []):
            registered = (REPO / source['path']).resolve()
            roots[registered].add(entry['id'])
            if registered.name.endswith('-review-B'):
                measurements = registered.with_name(registered.name.removesuffix('-review-B'))
                if (measurements / 'samples').exists():
                    roots[measurements].add(entry['id'])
                    review_sources.append(dict(experiment=entry['id'], review=str(registered.relative_to(REPO)),
                                               measurements=str(measurements.relative_to(REPO))))
    for extra in args.recovered or []:
        label, directory = extra.split('=', 1)
        roots[Path(directory).resolve()].add(label)
    assets, grouped, coverage, omitted = {}, collections.defaultdict(list), [], []
    for root, labels in sorted(roots.items()):
        if not root.exists():
            omitted.append(dict(run=str(root), reason='登记目录不存在'))
            continue
        print('scan', root.name, flush=True)
        files = [path for path in root.rglob('*') if path.is_file()
                 and not set(path.relative_to(root).parts[:-1]).intersection({'spack-cache', '__pycache__'})]
        binaries = [path for path in files if path.suffix not in ('.sass', '.log', '.json', '.gz', '.py', '.cu', '.hpp')
                    and executable(path)]
        attempted, ready, excluded = 0, 0, 0
        seen = set()
        rows = read(root / 'cases.json') if (root / 'cases.json').exists() else []
        heldout_ids = {row['id'] for row in rows if isinstance(row, dict) and row.get('set') == 'heldout'} if isinstance(rows, list) else set()
        for path, command, record in records(root, files, omitted):
            command = list(map(str, command))
            identity = (tuple(command), record.get('host_start_ns', record.get('started_unix_ns')), str(path.parent))
            if identity in seen:
                continue
            seen.add(identity)
            attempted += 1
            origin = dict(run=str(root.relative_to(REPO)), record=str(path.relative_to(REPO)),
                record_sha256=digest(path), original_argv=command,
                original_returncode=record.get('returncode'), original_status=record.get('status'),
                original_start_ns=record.get('host_start_ns', record.get('started_unix_ns', 0)),
                case=record.get('case', record.get('case_id')), variant=record.get('variant'), trial=record.get('trial'))
            if record.get('command_reconstruction'):
                origin['command_reconstruction'] = record['command_reconstruction']
            if Path(command[0]).name in ('nvcc', 'cuobjdump'):
                omitted.append(dict(**origin, reason='历史构建或反汇编调用；复用原二进制'))
                excluded += 1
                continue
            relative = path.relative_to(root)
            if any('smoke' in part.lower() or 'pilot' in part.lower() for part in relative.parts[:-1]):
                omitted.append(dict(**origin, reason='历史冒烟或长度预选；正式条件单独重测'))
                excluded += 1
                continue
            if record.get('returncode', 0) != 0 or record.get('timed_out'):
                omitted.append(dict(**origin, reason='历史失败尝试保留；相同条件的成功正式进程按原次数重测'))
                excluded += 1
                continue
            try:
                binary, checksum = resolve_file(command[0], binaries, binary=True, context=path)
                assets[checksum] = dict(source=str(binary), kind='binary')
                argv, inputs = ['assets/' + checksum], {}
                for index, value in enumerate(command[1:], 1):
                    if '/' not in value or value.startswith('--'):
                        argv.append(value)
                        continue
                    try:
                        source, input_hash = resolve_file(value, files, context=path)
                    except ValueError:
                        raise ValueError('未找到原路径参数：' + value)
                    if source.suffix in ('.map', '.json', '.u16', '.f16', '.bin') and command[index - 1] not in ('--output', '--out'):
                        assets[input_hash] = dict(source=str(source), kind='input')
                        inputs[str(index)] = input_hash
                        argv.append('assets/' + input_hash)
                    else:
                        raise ValueError('路径参数用途需按原 runner 核对：' + value)
                key = hashlib.sha256(json.dumps(argv).encode()).hexdigest()
                task_labels = set(labels)
                if root.name == '20261009-V09-heldout-job739011' and origin['case'] in heldout_ids:
                    task_labels.add('V09-heldout-regression')
                grouped[key].append(dict(argv=argv, origin=origin, labels=task_labels, inputs=inputs,
                    timeout_seconds=record.get('timeout_seconds', 300)))
                ready += 1
            except (ValueError, OSError, KeyError) as error:
                omitted.append(dict(**origin, reason=str(error)))
        coverage.append(dict(run=str(root.relative_to(REPO)), experiments=sorted(labels),
            recorded_calls=attempted, ready_calls=ready, historical_exclusions=excluded,
            status='prepared' if ready and ready + excluded == attempted else 'partial' if ready else 'no_recovered_calls'))
    tasks = []
    for key, group in grouped.items():
        by_run = collections.defaultdict(list)
        for item in group:
            by_run[item['origin']['run']].append(item)
        selected = max(by_run.values(), key=len)
        sample = selected[0]
        tasks.append(dict(id='legacy-' + key[:20], stages=sorted(set().union(*(item['labels'] for item in group))),
            argv=sample['argv'], repeats=len(selected), timeout_seconds=max(item['timeout_seconds'] for item in group),
            protocol=dict(profile='plain', packed_full_output=True,
                original_kind='program_query' if sample['argv'][1:] == ['device'] else 'measurement',
                cache_and_warmup='original_binary_and_argv', input_sha256=sample['inputs']),
            origins=[item['origin'] for item in group]))
    tasks.sort(key=lambda task: (0 if 'V09-heldout-regression' in task['stages'] else 1,
        min(origin['original_start_ns'] or 0 for origin in task['origins']), task['id']))
    per_stage = collections.Counter()
    for task in tasks:
        for stage in task['stages']:
            per_stage[stage] += task['repeats']
    manifest = dict(schema='gh200-a044-remeasure/1', source=str(args.inventory.resolve()),
        target_node='romeo-a044', target_clock_mhz=1980,
        qualification='R/B/V原条件回归；旧冻结预测保持原文件，重新评分另行保存',
        deduplication='original binary SHA256 + argv; maximum successful formal repetitions in one source run',
        tasks=tasks, assets=assets, coverage=coverage, omitted=omitted, historical_replacements=replacements,
        review_measurement_sources=review_sources,
        totals=dict(source_runs=len(coverage), recorded_calls=sum(row['recorded_calls'] for row in coverage),
                    unique_conditions=len(tasks), scheduled_processes=sum(task['repeats'] for task in tasks),
                    omitted_calls=len(omitted), stages=dict(per_stage)))
    save(output / 'manifest.json', manifest)
    for name in ('q00_a044_remeasure.py', 'q00_a044_legacy.py', 'l2_saturation.py'):
        shutil.copy2(HERE / name, output / name)
    (output / 'assets').mkdir()
    for checksum, asset in assets.items():
        target = output / 'assets' / checksum
        shutil.copy2(asset['source'], target)
        if asset['kind'] == 'binary':
            target.chmod(target.stat().st_mode | 0o100)
    save(output / 'package-identity.json', {str(path.relative_to(output)):digest(path)
        for path in output.rglob('*') if path.is_file()})
    print(json.dumps(manifest['totals'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovered', action='append', help='EXPERIMENT=repo-local recovered source directory')
    prepare(parser.parse_args())
