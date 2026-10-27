#!/usr/bin/env python3
"""a044 重测包的阶段预算与旧卡端点比较；只在 CPU 上读取证据。"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def rows(text):
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def metrics(records):
    """按探针已定义的窗口取时间；不同 CTA 的时钟不相减。"""
    result = {}
    scalar = ('elapsed_us', 'elapsed_ns', 'elapsed_cycles', 'event_ms', 'cuda_event_ms')

    def one(value, prefix):
        for key in scalar:
            if isinstance(value.get(key), (int, float)):
                result[prefix+'.'+key] = value[key]
        if isinstance(value.get('elapsed'), (int, float)) and 'unit' in value:
            result[prefix+'.elapsed['+value['unit']+']'] = value['elapsed']
        for first, last, unit in (('begin_cycle', 'end_cycle', 'cycles'),
            ('start_cycle', 'stop_cycle', 'cycles'), ('begin_ns', 'end_ns', 'ns'),
            ('start_ns', 'stop_ns', 'ns')):
            if first in value and last in value:
                result[prefix+'.delta_'+unit] = value[last]-value[first]
        if 'elapsed_cycles' in value and value.get('elapsed_ns', 0) > 0:
            result[prefix+'.effective_ghz'] = value['elapsed_cycles']/value['elapsed_ns']

    for index, row in enumerate(records):
        prefix = f'row{index}'
        one(row, prefix)
        if isinstance(row.get('timing'), dict):
            one(row['timing'], prefix+'.timing')
        for trial, value in enumerate(row.get('trials', [])):
            one(value, prefix+f'.trial{trial}')
        groups = row.get('ctas_data') or row.get('blocks_detail') or []
        cycles, times, begins, ends = [], [], [], []
        for item in groups:
            if 'begin' in item and 'end' in item:
                begin_cycle, begin_ns = item['begin']
                end_cycle, end_ns = item['end']
            elif all(key in item for key in ('start_cycle', 'stop_cycle', 'start_ns', 'stop_ns')):
                begin_cycle, end_cycle = item['start_cycle'], item['stop_cycle']
                begin_ns, end_ns = item['start_ns'], item['stop_ns']
            else:
                continue
            cycles.append(end_cycle-begin_cycle)
            times.append(end_ns-begin_ns)
            begins.append(begin_ns)
            ends.append(end_ns)
        if cycles:
            result[prefix+'.cta_median_cycles'] = statistics.median(cycles)
            result[prefix+'.cta_median_ns'] = statistics.median(times)
            result[prefix+'.global_envelope_ns'] = max(ends)-min(begins)
            frequencies = [cycle/ns for cycle, ns in zip(cycles, times) if ns > 0]
            if frequencies:
                result[prefix+'.cta_median_effective_ghz'] = statistics.median(frequencies)
    return result


def device(records):
    ids = set()
    for row in records:
        for value in (row.get('gpu_uuid'), row.get('uuid'), row.get('device', {}).get('uuid')):
            if isinstance(value, str) and value.startswith('GPU-'):
                ids.add(value)
    return sorted(ids)


def load_sample(folder):
    receipt = read(folder/'receipt.json')
    if (folder/'stdout').exists():
        content = (folder/'stdout').read_bytes()
    else:
        archive = read(folder/'output-archive.json')
        archive_path = folder/archive['archive']
        if sha(archive_path) != archive['sha256']:
            raise ValueError('新输出归档 SHA256 不符：'+str(folder))
        with tarfile.open(archive_path) as stream:
            content = stream.extractfile('stdout').read()
        if hashlib.sha256(content).hexdigest() != archive['members']['stdout']['sha256']:
            raise ValueError('新 stdout 成员 SHA256 不符：'+str(folder))
    if hashlib.sha256(content).hexdigest() != receipt['stdout_sha256']:
        raise ValueError('stdout 与 receipt 不符：'+str(folder))
    return rows(content.decode())


def budget(package):
    manifest = read(package/'manifest.json')
    source = Path(manifest['source'])
    stages = collections.defaultdict(lambda: dict(processes=0, original_wall_seconds=0,
        original_output_bytes=0, largest_process_output_bytes=0, tasks=[], original_runs=set()))
    unique_wall, unique_bytes = 0, 0
    for task in manifest['tasks']:
        durations, sizes = [], []
        for origin in task['origins']:
            receipt_path = source/origin['receipt']
            receipt = read(receipt_path)
            durations.append((receipt['stop_ns']-receipt['start_ns'])/1e9)
            sizes.append(sum(path.stat().st_size for path in receipt_path.parent.rglob('*') if path.is_file()))
        wall = max(durations)*task['repeats']
        size = max(sizes)*task['repeats']
        unique_wall += wall
        unique_bytes += size
        for stage in task['stages']:
            item = stages[stage]
            item['processes'] += task['repeats']
            item['original_wall_seconds'] += wall
            item['original_output_bytes'] += size
            item['largest_process_output_bytes'] = max(item['largest_process_output_bytes'], max(sizes))
            item['tasks'].append(task['id'])
            item['original_runs'].update(origin['run'] for origin in task['origins'] if origin['run'].startswith(stage))
    for stage in stages.values():
        stage['original_runs'] = sorted(stage['original_runs'])
    return dict(manifest_sha256=sha(package/'manifest.json'), stages=dict(sorted(stages.items())),
        unique_processes=sum(task['repeats'] for task in manifest['tasks']),
        unique_original_wall_seconds=unique_wall, unique_original_output_bytes=unique_bytes,
        clock_condition='请求1980 MHz；逐样本读取实测频率',
        time_scope='原进程 wall time 的同条件最大值求和；压缩、传输和新环境差异另计')


def compare(package, run):
    manifest = read(package/'manifest.json')
    if read(run/'request.json')['manifest_sha256'] != sha(package/'manifest.json'):
        raise ValueError('执行记录与指定准备包不符')
    tasks = {task['id']:task for task in manifest['tasks']}
    source = Path(manifest['source'])
    baseline_cache, compared, failures = {}, [], []
    for status_path in sorted((run/'runs').glob('*/*/remeasure-status.json')):
        task_id = status_path.parents[1].name
        task = tasks[task_id]
        status = read(status_path)
        try:
            actual_rows = load_sample(status_path.parent)
            actual = metrics(actual_rows)
            if task_id not in baseline_cache:
                old = collections.defaultdict(list)
                seen = set()
                for origin in task['origins']:
                    receipt_path = source/origin['receipt']
                    receipt = read(receipt_path)
                    if receipt['stdout_sha256'] in seen:
                        continue
                    seen.add(receipt['stdout_sha256'])
                    old_rows = load_sample(receipt_path.parent)
                    identity = ','.join(device(old_rows)) or origin['run']
                    old[identity].append(dict(source=origin['receipt'], measurements=metrics(old_rows)))
                baseline_cache[task_id] = old
            comparisons = []
            for old_gpu, records in baseline_cache[task_id].items():
                for endpoint, measured in actual.items():
                    values = [record['measurements'][endpoint] for record in records
                              if endpoint in record['measurements']]
                    if not values:
                        continue
                    reference = statistics.median(values)
                    comparisons.append(dict(old_gpu=old_gpu, endpoint=endpoint, observed=measured,
                        old_median=reference, old_min=min(values), old_max=max(values),
                        relative_change=measured/reference-1 if reference else None,
                        old_samples=len(values)))
            compared.append(dict(task_id=task_id, repeat=status['repeat'], stages=task['stages'],
                current_gpu=device(actual_rows), capture_status=status, measurements=actual,
                comparisons=comparisons, origins=task['origins']))
        except (OSError, KeyError, ValueError) as error:
            failures.append(dict(task_id=task_id, folder=str(status_path.parent), error=str(error)))
    return dict(manifest_sha256=sha(package/'manifest.json'), run=str(run), compared=compared,
        failures=failures, compared_processes=len(compared),
        mechanism_scoring='原始窗口和跨卡变化；完整数值oracle及原冻结预测另按每项实验评分')


def unpack(wrapper, output):
    archive = wrapper/'output.tar.gz'
    checksum = sha(archive)
    if checksum != read(wrapper/'receipt.json')['sha256']:
        raise ValueError('回传归档 SHA256 不符')
    output.mkdir(parents=True, exist_ok=False)
    roots, count = set(), 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            parts = Path(member.name).parts
            if not parts or not parts[0].startswith('results-') or not member.isfile():
                continue
            roots.add(parts[0])
            if '..' in parts or len(roots) > 1:
                raise ValueError('归档结果路径不唯一：'+member.name)
            path = output.joinpath(*parts[1:])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as target:
                source = stream.extractfile(member)
                for block in iter(lambda: source.read(1024*1024), b''):
                    target.write(block)
            count += 1
    if not (output/'request.json').is_file():
        raise ValueError('归档没有本轮采集 request')
    result = dict(wrapper=str(wrapper), archive_sha256=checksum, files=count,
        result_prefix=next(iter(roots)), output=str(output),
        full_outputs='每进程 output.tar.gz 保持压缩，完整数值输出留给原 oracle')
    (output/'view-source.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('budget', 'compare', 'unpack'))
    parser.add_argument('--package', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command in ('compare', 'unpack') and args.run is None:
        parser.error(args.command+' 需要 --run')
    if args.command == 'unpack':
        print(json.dumps(unpack(args.run.resolve(),args.output.resolve()),ensure_ascii=False,indent=2))
        return
    if args.package is None:
        parser.error(args.command+' 需要 --package')
    result = budget(args.package.resolve()) if args.command == 'budget' else compare(
        args.package.resolve(), args.run.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('stages','compared')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
