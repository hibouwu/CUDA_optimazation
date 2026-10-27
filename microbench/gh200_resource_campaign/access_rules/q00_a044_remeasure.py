#!/usr/bin/env python3
"""把旧探针的成功调用整理为 a044 重测包；默认只预览，不提交作业。"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import sys
import tarfile
import subprocess
import time


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def executable(path):
    try:
        with path.open('rb') as stream:
            header = stream.read(20)
        return header[:4] == b'\x7fELF' and header[18:20] == b'\xb7\x00'
    except OSError:
        return False


def resolve_file(old_name, files, binary=False, context=None):
    """按最长完整路径后缀匹配；同后缀异内容时保留缺失，避免选错构建。"""
    parts = Path(old_name).parts
    candidates = [path for path in files if path.name == parts[-1]]
    if binary:
        candidates = [path for path in candidates if executable(path)]
    if not candidates:
        raise ValueError('没有归档文件：' + old_name)
    scores = []
    for path in candidates:
        score = 0
        for left, right in zip(reversed(parts), reversed(path.parts)):
            if left != right:
                break
            score += 1
        scores.append((score, path))
    best = max(score for score, _ in scores)
    selected = [path for score, path in scores if score == best]
    if context is not None and len(selected) > 1:
        # 同包的 returned、repair、baseline 各自携带构建；先保持采样目录归属。
        distances = [(len(path.parts) + len(context.parts) - 2 * len(Path(
            os.path.commonpath((path, context))).parts), path) for path in selected]
        nearest = min(distance for distance, _ in distances)
        selected = [path for distance, path in distances if distance == nearest]
    hashes = {digest(path) for path in selected}
    if len(hashes) != 1:
        raise ValueError('相同路径后缀对应不同文件：' + old_name)
    return selected[0], next(iter(hashes))


def declared_binary(root, old_name, output):
    plan_path = root/'plan.json'
    declaration = read(plan_path).get('binary_source', '') if plan_path.exists() else ''
    if ':' not in declaration:
        raise ValueError('没有归档文件：' + old_name)
    archive_name, member_name = declaration.split(':', 1)
    if Path(member_name).name != Path(old_name).name:
        raise ValueError('归档声明的二进制名称不匹配：' + old_name)
    archive_path = (root/archive_name).resolve()
    with tarfile.open(archive_path) as archive:
        member = next(member for member in archive if member.name.removeprefix('./') == member_name)
        content = archive.extractfile(member).read()
    checksum = hashlib.sha256(content).hexdigest()
    path = output/'recovered'/checksum
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    if not executable(path):
        raise ValueError('声明归档中的文件不是 ARM64 ELF')
    return path, checksum, dict(archive=str(archive_path), member=member_name,
                               archive_sha256=digest(archive_path))


def inspect_stdout(path, protocol):
    text = path.read_text()
    try:
        value = json.loads(text)
        rows = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError('stdout 没有 JSON 记录')
    metrics, errors, eligibility = [], [], []

    def visit(value, location=''):
        if isinstance(value, dict):
            if value.get('schema') == 'barrier-return/1':
                for index, cta in enumerate(value['cta_records']):
                    if cta[2] > cta[1]:
                        metrics.append(dict(field=location+f'.cta_records[{index}].elapsed_cycles', value=cta[2]-cta[1]))
            if (value.get('warmup_metric') == 'median_consumer_inc_cycles' and value.get('schema_version') == 2
                    or value.get('threads') == 384 and all(key in value for key in ('delay', 'before', 'stamps'))):
                for index, stamp in enumerate(value['stamps']):
                    if stamp[2] > stamp[1]:
                        metrics.append(dict(field=location+f'.stamps[{index}][2]-[1]', value=stamp[2]-stamp[1]))
            if isinstance(value.get('begin'), list) and isinstance(value.get('end'), list):
                for index, (begin, end) in enumerate(zip(value['begin'], value['end'])):
                    if end > begin:
                        metrics.append(dict(field=location+f'.end-minus-begin[{index}]', value=end-begin))
            for key, item in value.items():
                name = location+'.'+key
                if key in ('errors', 'nonfinite') or key.endswith('_errors'):
                    if isinstance(item, (int, float)) and item != 0:
                        errors.append(dict(field=name, value=item))
                if key in ('performance_eligible', 'warmup_converged', 'status', 'legal'):
                    eligibility.append(dict(field=name, value=item))
                if key in ('elapsed_us', 'elapsed_ns', 'elapsed_ms', 'elapsed_cycles', 'event_ms', 'cuda_event_ms',
                           'cycles', 'ns', 'duration_ns', 'progress_cycles', 'consumed_cycles',
                           'isolated_median_us', 'host_call_median_us', 'burst_us', 'graph_us',
                           'call_us', 'call_median_us', 'event_us', 'timed_us', 'span_ns', 'span_cycles'):
                    if isinstance(item, (int, float)) and math.isfinite(item) and item > 0:
                        metrics.append(dict(field=name, value=item))
                if key == 'elapsed' and value.get('unit') in ('clock64_cycle/CTA', 'cycle', 'cycles', 'ns', 'us',
                                                            'ns/GPU', 'globaltimer_ns/GPU'):
                    if isinstance(item, (int, float)) and math.isfinite(item) and item > 0:
                        metrics.append(dict(field=name, value=item))
                if key in ('cycle_deltas', 'ns_deltas') and isinstance(item, list):
                    for index, delta in enumerate(item):
                        if isinstance(delta, (int, float)) and math.isfinite(delta) and delta > 0:
                            metrics.append(dict(field=name+f'[{index}]', value=delta))
                if key not in ('warmup', 'warmup_ctas', 'warmup_samples_ns', 'warmup_us',
                               'warm_checksums', 'timed_checksums'):
                    visit(item, name)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    visit(item, location+f'[{index}]')

    visit(rows)
    kind = protocol.get('original_kind', '')
    query_only = any(row.get('type') == 'resources' for row in rows) or kind in (
        'resources', 'validate', 'validate-only', 'short', 'shortcheck', 'short-check', 'correctness', 'program_query')
    status = 'timed' if metrics else 'query_or_validation' if query_only else 'missing_measurements'
    return dict(records=len(rows), status=status, timing_values=len(metrics),
                timing_examples=metrics[:8], errors=errors, eligibility=eligibility,
                valid=not errors and status != 'missing_measurements')


def pack_outputs(folder):
    """只打包本次调用的输出；解码逐成员核验后移除相应裸文件。"""
    keep = {'command.json', 'receipt.json', 'remeasure-status.json', 'output-archive.json', 'output.tar.gz'}
    files = [path for path in folder.rglob('*') if path.is_file() and path.name not in keep]
    members = {str(path.relative_to(folder)):dict(sha256=digest(path), size=path.stat().st_size)
               for path in files}
    archive_path = folder/'output.tar.gz'
    with tarfile.open(archive_path, 'w:gz', compresslevel=1) as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(folder)), recursive=False)
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            checksum = hashlib.sha256()
            stream = archive.extractfile(member)
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(block)
            if checksum.hexdigest() != members[member.name]['sha256']:
                raise ValueError('输出归档解码校验失败：' + member.name)
    report = dict(archive='output.tar.gz', sha256=digest(archive_path),
                  bytes=archive_path.stat().st_size, members=members, decoded_verified=True)
    save(folder/'output-archive.json', report)
    for path in files:
        path.unlink()
    return dict(bytes=report['bytes'], members=len(members), sha256=report['sha256'])


def inspect_a044_allocation(folder):
    """本预约可分配整机四卡；每个探针进程只选择其中一个 UUID。"""
    from l2_saturation import command_record
    job = os.environ.get('SLURM_JOB_ID', '')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not job.isdigit() or not visible or ',' in visible or visible == '-1':
        raise ValueError('需要 Slurm 分配与单个可见 GPU')
    command_record(folder/'allocation', ['scontrol', 'show', 'job', '-o', job], 15)
    info = (folder/'allocation/stdout').read_text()
    for condition, reason in ((r'\bJobState=RUNNING\b', '作业未运行'),
        (r'UserId=[^ ]*\('+str(os.geteuid())+r'\)', '作业所有者不符'),
        (r'\bNodeList=romeo-a044\b', '作业节点不是 a044'),
        (r'\bReservation=compteurgpu_R302026\b', '不是本次专用预约')):
        if re.search(condition, info) is None:
            raise ValueError(reason)
    allocated = re.search(r'\bAllocTRES=([^ ]+)', info)
    gpu_count = re.search(r'(?:^|,)gres/gpu=(1|4)(?:,|$)', allocated[1]) if allocated else None
    if gpu_count is None:
        raise ValueError('本次 runner 需要一张 GPU 或整机四张 GPU 的分配')
    command_record(folder/'all-gpus', ['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], 30)
    all_rows = (folder/'all-gpus/stdout').read_text().strip().splitlines()
    if gpu_count[1] == '4' and len(all_rows) != 4:
        raise ValueError('四卡分配与实际 GPU 数量不符')
    command_record(folder/'device-query', ['nvidia-smi', '-i', visible,
        '--query-gpu=uuid,name', '--format=csv,noheader'], 30)
    rows = (folder/'device-query/stdout').read_text().strip().splitlines()
    if len(rows) != 1 or 'GH200' not in rows[0] or rows[0] not in all_rows:
        raise ValueError('需要本节点的一个 GH200 GPU')
    uuid, name = (part.strip() for part in rows[0].split(',', 1))
    identity = dict(job_id=job, host=os.uname().nodename, cuda_visible_devices=visible,
        uuid=uuid, name=name, uid=os.geteuid(), allocated_gpu_count=int(gpu_count[1]))
    save(folder/'identity.json', identity)
    return identity


def prepare(args):
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = sorted(path for path in source.glob('Q*/*/20*') if path.is_dir()
                   and args.since <= path.name[:8] <= args.until)
    records, omitted, coverage = [], [], []
    assets = {}
    stages = {}
    all_files = {root: [path for path in root.rglob('*') if path.is_file()] for root in roots}
    all_binaries = [path for paths in all_files.values() for path in paths if executable(path)]
    binary_names = {path.name for path in all_binaries}
    for root in roots:
        stage = root.relative_to(source).parts[0].split('-')[0]
        if args.stage and stage not in args.stage:
            continue
        files = all_files[root]
        seen_receipts = set()
        attempted, successful = 0, 0
        for receipt_path in sorted(path for path in files if path.name == 'receipt.json'):
            relative = receipt_path.relative_to(root)
            # 这些目录装的是上游证据副本，真正的上游根会单独进入清单。
            if any(part in {'baseline', 'calibration', 'reference', 'source', 'source-before',
                            'before', 'reanalysis', 'initial'} for part in relative.parts[:-1]):
                continue
            receipt = read(receipt_path)
            argv = receipt.get('argv', [])
            if not argv or not isinstance(argv, list):
                continue
            name = Path(str(argv[0])).name
            if name not in binary_names:
                continue
            identity = (receipt.get('start_ns'), receipt.get('stop_ns'),
                        receipt.get('stdout_sha256'), tuple(map(str, argv)))
            if identity in seen_receipts:
                continue
            seen_receipts.add(identity)
            attempted += 1
            origin = dict(run=str(root.relative_to(source)), receipt=str(receipt_path.relative_to(source)),
                          original_argv=argv, original_start_ns=receipt.get('start_ns'),
                          original_returncode=receipt.get('returncode'))
            if receipt.get('returncode') != 0 or receipt.get('timed_out') or receipt.get('failure'):
                omitted.append(dict(**origin, reason='旧调用失败；保留原失败，不纳入成功重测集合'))
                continue
            try:
                stdout = receipt_path.parent / 'stdout'
                if digest(stdout) != receipt['stdout_sha256']:
                    raise ValueError('原 stdout 的 SHA256 不符')
                binary_provenance = {}
                try:
                    binary, binary_hash = resolve_file(str(argv[0]), files, binary=True, context=receipt_path)
                except ValueError as error:
                    if not str(error).startswith('没有归档文件'):
                        raise
                    # 某些包复用另一个实验的原二进制；要求跨包路径含原实验名。
                    parts = Path(str(argv[0])).parts
                    named = [path for path in all_binaries if any(
                        part.startswith('202610') and part in path.parts for part in parts)]
                    try:
                        binary, binary_hash = resolve_file(str(argv[0]), named, binary=True)
                    except ValueError:
                        binary, binary_hash, binary_provenance = declared_binary(root, str(argv[0]), output)
                assets[binary_hash] = dict(source=str(binary), kind='binary', **binary_provenance)
                normalized = ['assets/' + binary_hash]
                inputs = {}
                for index, value in enumerate(argv[1:], 1):
                    value = str(value)
                    if argv[index - 1] == '--worklist':
                        path, checksum = resolve_file(value, files, context=receipt_path)
                        assets[checksum] = dict(source=str(path), kind='input')
                        normalized.append('assets/' + checksum)
                        inputs['--worklist'] = checksum
                    elif '/' in value and not value.startswith('--'):
                        raise ValueError('尚未登记的路径参数：' + value)
                    else:
                        normalized.append(value)
                command_path = receipt_path.parent / 'command.json'
                command = read(command_path) if command_path.exists() else {}
                protocol = dict(profile='plain', cache_and_warmup='binary_and_argv',
                                input_sha256=inputs, packed_full_output=True,
                                original_kind='program_query' if any(flag in argv for flag in (
                                    '--inspect-layout', '--inspect-program')) else 'measurement')
                key = hashlib.sha256(json.dumps([binary_hash, normalized[1:], protocol],
                    sort_keys=True).encode()).hexdigest()
                records.append(dict(key=key, stage=stage, argv=normalized, protocol=protocol,
                    timeout_seconds=command.get('timeout_seconds', 180), origin=origin,
                    original_stdout_sha256=receipt['stdout_sha256']))
                successful += 1
            except (OSError, KeyError, ValueError) as error:
                omitted.append(dict(**origin, reason=str(error)))
        coverage.append(dict(run=str(root.relative_to(source)), stage=stage,
            original_calls=attempted, ready_calls=successful,
            status='ready' if attempted and successful == attempted else
                   'partial' if successful else 'no_executed_probe_receipts' if not attempted else 'unresolved'))
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record['key']].append(record)
    tasks = []
    for key, group in sorted(grouped.items()):
        by_run = collections.defaultdict(list)
        for record in group:
            by_run[record['origin']['run']].append(record)
        # 相同调用在不同研究包中重复出现时，保留单包内最大的独立进程数。
        selected = max(by_run.values(), key=len)
        exemplar = selected[0]
        task = dict(id=key[:16], stages=sorted({row['stage'] for row in group}),
            argv=exemplar['argv'], protocol=exemplar['protocol'], repeats=len(selected),
            timeout_seconds=max(row['timeout_seconds'] for row in group),
            origins=[row['origin'] for row in group])
        tasks.append(task)
        for stage in task['stages']:
            stages[stage] = stages.get(stage, 0) + task['repeats']
    tasks.sort(key=lambda task: min(origin['original_start_ns'] or 0 for origin in task['origins']))
    manifest = dict(schema='gh200-a044-remeasure/1', source=str(source), since=args.since,
        until=args.until, target_node='romeo-a044', target_clock_mhz=1980,
        qualification='旧条件跨设备重测；原始采集与原实验评分分开',
        deduplication='binary SHA256 + argv + protocol; max per-run repetitions',
        tasks=tasks, assets=assets, coverage=coverage, omitted=omitted,
        totals=dict(experiments=len(coverage), original_calls=sum(row['original_calls'] for row in coverage),
                    ready_calls=len(records), unique_conditions=len(tasks),
                    scheduled_processes=sum(row['repeats'] for row in tasks),
                    omitted_calls=len(omitted), stages=stages))
    save(output/'manifest.json', manifest)
    shutil.copy2(Path(__file__), output/'q00_a044_remeasure.py')
    shutil.copy2(HERE/'l2_saturation.py', output/'l2_saturation.py')
    (output/'assets').mkdir()
    for checksum, asset in assets.items():
        path = output/'assets'/checksum
        shutil.copy2(asset['source'], path)
        if asset['kind'] == 'binary':
            path.chmod(path.stat().st_mode | 0o100)
    save(output/'package-identity.json', {str(path.relative_to(output)):digest(path)
         for path in output.rglob('*') if path.is_file()})
    print(json.dumps(manifest['totals'], ensure_ascii=False, indent=2))


def add_nvsmi_unix_time(csv_path):
    """按原 R07 协议在节点上把本地时间转换为 unix_ns。"""
    lines = csv_path.read_text().splitlines()
    if not lines:
        return
    out = [lines[0] + ', unix_ns']
    for line in lines[1:]:
        stamp = line.split(',')[0].strip()
        try:
            seconds = datetime.datetime.strptime(stamp, '%Y/%m/%d %H:%M:%S.%f').timestamp()
            out.append(f'{line}, {int(seconds * 1e9)}')
        except ValueError:
            out.append(f'{line}, ')
    csv_path.write_text('\n'.join(out) + '\n')


def recorded_process(command_record, folder, argv, timeout, protocol):
    monitor = protocol.get('external_nvsmi')
    if not monitor:
        return command_record(folder, argv, timeout)
    folder.parent.mkdir(parents=True, exist_ok=True)
    csv_path = folder.with_name(folder.name+'.nvsmi.csv')
    monitor_argv = ['nvidia-smi', '--query-gpu='+','.join(monitor['fields']), '--format=csv,nounits',
                    '-lms', str(monitor['interval_ms']), '-f', str(csv_path)]
    sampler = subprocess.Popen(monitor_argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(monitor['before_seconds'])
        return command_record(folder, argv, timeout)
    finally:
        time.sleep(monitor['after_seconds'])
        sampler.send_signal(signal.SIGINT)
        sampler.wait(timeout=10)
        if csv_path.exists():
            add_nvsmi_unix_time(csv_path)
            csv_path.rename(folder/'nvsmi.csv')
        save(folder/'external-protocol.json', dict(command=monitor_argv, **monitor))


def run(args):
    root = args.package.resolve()
    manifest = read(root/'manifest.json')
    tasks = [task for task in manifest['tasks'] if not args.stage or
             set(args.stage).intersection(task['stages'])]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    summary = dict(conditions=len(tasks), processes=sum(task['repeats'] for task in tasks),
                   stages=args.stage, execute=args.execute)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not args.execute:
        for task in tasks:
            print(task['id'], task['repeats'], ' '.join(task['argv']))
        return
    from l2_saturation import command_record, exclusive_lock
    if os.uname().nodename.split('.')[0] != manifest['target_node']:
        raise ValueError('执行节点须为 ' + manifest['target_node'])
    for relative, checksum in read(root/'package-identity.json').items():
        if digest(root/relative) != checksum:
            raise ValueError('准备包内容变化：' + relative)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    allocation = inspect_a044_allocation(destination/'allocation')
    save(destination/'request.json', dict(**summary, manifest_sha256=digest(root/'manifest.json')))
    outcomes = []
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        query = ['nvidia-smi', '-i', allocation['uuid'],
                 '--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu',
                 '--format=csv']
        command_record(destination/'environment-before', query, 30)
        with (destination/'telemetry.csv').open('wb') as telemetry:
            sampler = (subprocess.Popen(query+['--loop-ms=1000'], stdout=telemetry, stderr=subprocess.STDOUT)
                       if manifest.get('continuous_telemetry', True) else None)
            try:
                for task in tasks:
                    for repeat in range(task['repeats']):
                        folder = destination/'runs'/task['id']/f'{repeat:03d}'
                        argv = [str(root/value) if value.startswith('assets/') else
                                str(folder/value.removeprefix('outputs/')) if value.startswith('outputs/') else
                                value for value in task['argv']]
                        try:
                            receipt = recorded_process(command_record, folder, argv, task['timeout_seconds'], task['protocol'])
                            observation = inspect_stdout(folder/'stdout', task['protocol'])
                            result = dict(task=task['id'], repeat=repeat, captured=observation['valid'],
                                observation=observation, receipt_sha256=digest(folder/'receipt.json'),
                                returncode=receipt['returncode'])
                        except Exception as error:
                            result = dict(task=task['id'], repeat=repeat, captured=False, error=str(error))
                        if task['protocol'].get('packed_full_output'):
                            result['output_archive'] = pack_outputs(folder)
                        outcomes.append(result)
                        save(folder/'remeasure-status.json', result)
                        print(json.dumps(result, ensure_ascii=False), flush=True)
            finally:
                if sampler is not None:
                    sampler.terminate()
                    sampler.wait(timeout=10)
                command_record(destination/'environment-after', query, 30)
    save(destination/'capture-summary.json', dict(allocation=allocation, outcomes=outcomes,
        captured=sum(row['captured'] for row in outcomes), attempted=len(outcomes),
        experiment_validation='按各原实验逐项评分，采集成功不自动视为机制通过'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--source', type=Path, required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    prepare_parser.add_argument('--since', default='20261016')
    prepare_parser.add_argument('--until', default='20261023')
    prepare_parser.add_argument('--stage', action='append')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--package', type=Path, required=True)
    run_parser.add_argument('--output', type=Path)
    run_parser.add_argument('--stage', action='append')
    run_parser.add_argument('--limit', type=int)
    run_parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        if args.execute and args.output is None:
            parser.error('--execute 需要新的 --output 目录')
        run(args)


if __name__ == '__main__':
    main()
