#!/usr/bin/env python3
"""a044 定向 NCU：复用原二进制和参数，单独采集计数器。"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


TIME = ['gpu__time_duration.sum', 'sm__cycles_elapsed.avg.per_second']
TRAFFIC = ['dram__bytes_read.sum', 'dram__bytes_write.sum',
           'lts__t_sectors_op_read.sum', 'lts__t_sectors_op_write.sum',
           'lts__t_sector_hit_rate.pct']
TMA = ['sm__inst_executed_pipe_tma.sum', 'sm__pipe_tma_cycles_active.sum']
TENSOR = ['sm__inst_executed_pipe_tensor_op_gmma.sum', 'sm__pipe_tensor_cycles_active.sum']


def prepare(args):
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'assets').mkdir()
    points, assets = [], {}
    packages = {key:args.campaign / value for key, value in (
        ('resource', 'resource-first-six-package-r2'), ('mechanism', 'mechanism-package-r3'),
        ('legacy', 'legacy-package'))}
    manifests = {key:read(path / 'manifest.json') for key, path in packages.items()}

    def add(key, task_id, name, kernel, selection, metrics, expected):
        task = next(t for t in manifests[key]['tasks'] if t['id'] == task_id)
        for argument in task['argv']:
            if argument.startswith('assets/'):
                source = packages[key] / argument
                if argument not in assets:
                    shutil.copy2(source, folder / argument)
                    assets[argument] = dict(sha256=sha(source), source=str(source.resolve()))
        points.append(dict(name=name, original_task_id=task_id, argv=task['argv'], kernel=kernel,
            selection=selection, metrics=list(dict.fromkeys(TIME+metrics)), expected=expected,
            origins=task['origins'], original_manifest_sha256=sha(packages[key] / 'manifest.json')))

    resource = manifests['resource']['tasks']
    for case in ('global_read_cg_8m', 'global_read_cg_256m', 'global_write_256m'):
        task = next(t for t in resource if t['stages'] == ['RES-memory_baseline'] and t['argv'][1] == case)
        raw = Path(task['origins'][0]['receipt']).parent / 'raw.jsonl'
        record = next(json.loads(line) for line in raw.read_text().splitlines() if json.loads(line).get('type') == 'trial')
        expected = {key:record[key] for key in ('blocks', 'iterations', 'allocation_per_array_bytes',
                    'read_payload_bytes', 'write_payload_bytes')}
        expected.update(original_stdout=str(raw), original_stdout_sha256=sha(raw))
        add('resource', task['id'], case, 'global_path', 'resource_adaptive', TRAFFIC, expected)
    add('mechanism', '9033465f9d848ac5', 'tma_ab_a_b', 'tensor_probe', 'tma_fixed_11_44_66', TRAFFIC+TMA,
        dict(formal_invocations=[11,44,66], pair_indices=[0,3,5], groups=32, ctas=1,
             request_masks=[7,1,6], tensor_payload_bytes=[1048576,524288,524288],
             tensor_load_instructions=[96,32,64]))
    add('mechanism', 'fadbd61d541b934c', 'compute_k32_wait1', 'compute_chain', 'compute_adaptive', TENSOR,
        dict(ktiles=32, wgmma_count=256, short_check_launches=2, ctas=1))
    for k, task_id in ((1024, 'legacy-833c5af50b76086de440'), (4096, 'legacy-daa74650c112b035f101')):
        add('legacy', task_id, f'cfg_b_k{k}', 'device_kernel', 'gemm_adaptive', TRAFFIC+TMA+TENSOR,
            dict(m=2304, n=3072, k=k, output_tiles=432, ktiles=k//64,
                 input_unique_payload_bytes=2*(2304*k+k*3072),
                 input_requested_payload_bytes=432*(k//64)*32768,
                 output_payload_bytes=2304*3072*4, dtype_a_b='FP16', dtype_d='FP32'))
        points[-1]['origins'] = [o for o in points[-1]['origins']
            if o['run'].endswith('20261009-V09-heldout-job739011')]
        assert points[-1]['origins']
    save(folder / 'manifest.json', dict(schema='a044-targeted-ncu/1', points=points, assets=assets,
        clock_control='none', cache_control='none', replay_mode='kernel',
        note='NCU 独立运行，原二进制与 argv 不变；自适应预热按原 stdout 选正式启动。'))
    shutil.copy2(Path(__file__), folder / Path(__file__).name)
    save(folder / 'package-identity.json', {str(p.relative_to(folder)):sha(p)
        for p in folder.rglob('*') if p.is_file()})
    print(json.dumps(dict(points=len(points), binaries=len([p for p in assets if (folder/p).read_bytes()[:4]==b'\x7fELF']),
        expected_target_types=9), ensure_ascii=False))


def command(ncu, point, package, output, cache_control, metrics):
    argv = [str(package / value) if value.startswith('assets/') else value for value in point['argv']]
    flags = ['--kernel-name-base', 'function']
    if point['selection'] == 'tma_fixed_11_44_66':
        flags += ['--kernel-id', '::tensor_probe:^(11|44|66)$']
    else:
        flags += ['--kernel-name', point['kernel']]
    return [ncu, '--clock-control', 'none', '--cache-control', cache_control,
            '--replay-mode', 'kernel', *flags, '--metrics', ','.join(metrics),
            '--log-file', str(output / 'profiler.log'), '--export', str(output / 'profile'), *argv]


def process(command, folder, name, timeout):
    start = time.time_ns()
    with (folder / (name+'.stdout')).open('w') as out, (folder / (name+'.stderr')).open('w') as err:
        result = subprocess.run(command, cwd=folder, stdout=out, stderr=err, timeout=timeout)
    receipt = dict(command=command, returncode=result.returncode, host_start_ns=start, host_stop_ns=time.time_ns(),
        stdout_sha256=sha(folder / (name+'.stdout')), stderr_sha256=sha(folder / (name+'.stderr')))
    save(folder / (name+'.receipt.json'), receipt)
    return result.returncode


def identify_formal(point, stdout, metrics_csv):
    records = [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
    table = list(csv.DictReader(io.StringIO(metrics_csv.read_text())))
    profiles = [row for row in table if row.get('ID', '').isdigit()]
    # NCU --page raw 每行是一个 launch；非数字 ID 的一行记录单位。
    selection = point['selection']
    if selection == 'tma_fixed_11_44_66':
        formal = [row for row in records if row.get('pair_index') in (0,3,5)]
        expected_count = 3
        selected = profiles
        expected_pairs = [0,3,5]
        if [row['pair_index'] for row in formal] != expected_pairs:
            raise ValueError('原 TMA 输出缺少选定 pair_index')
        if [row['request_mask'] for row in formal] != [7,1,6]:
            raise ValueError('原 TMA 选定 pair 的 request_mask 不符')
        warmups = 10
    else:
        if selection == 'resource_adaptive':
            formal = next(row for row in records if row.get('type') == 'trial')
            warmups = len(formal['warmup_samples_ns'])
            expected_count = warmups + 1
        elif selection == 'compute_adaptive':
            formal = next(row for row in records if row.get('schema') == 'gh200-cfg-b-compute-chain/1')
            warmups = len(formal['warmup']['elapsed_ns'])
            expected_count = len(formal['short_checks']) + warmups + 1
        else:
            formal = next(row for row in records if row.get('event') == 'call')
            warmups = formal['warmup_calls']
            expected_count = warmups + 1
        selected = profiles[-1:]
    if len(profiles) != expected_count:
        raise ValueError(f'匹配 launch 数 {len(profiles)} 与原 stdout 推导的 {expected_count} 不符')
    return dict(warmup_calls=warmups, profiled_launches=len(profiles), selected_formal=selected,
                original_formal_record=formal)


def run(args):
    package = args.package.resolve()
    manifest = read(package / 'manifest.json')
    points = [p for p in manifest['points'] if not args.point or p['name'] in args.point]
    if args.point and set(args.point) != {p['name'] for p in points}:
        raise ValueError('未登记的 point')
    output = args.output.resolve()
    metrics_for = lambda point: args.metrics.split(',') if args.metrics else point['metrics']
    if not args.execute:
        for point in points:
            print(json.dumps(command(args.ncu, point, package, output / point['name'], args.cache_control,
                                     metrics_for(point)), ensure_ascii=False))
        return
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in read(package / 'package-identity.json').items():
        if sha(package / name) != expected:
            raise ValueError('包身份变化：' + name)
    if process([args.ncu, '--version'], output, 'version', 30):
        raise RuntimeError('NCU 版本查询失败')
    if process([args.ncu, '--query-metrics', '--query-metrics-mode', 'all'], output, 'supported-metrics', 90):
        raise RuntimeError('NCU 支持指标查询失败')
    if args.query_only:
        return
    supported = set(re.findall(r'\b(?:dram|lts|l1tex|sm|smsp|gpu)__[A-Za-z0-9_.]+',
                              (output / 'supported-metrics.stdout').read_text()))
    missing = sorted({metric for point in points for metric in metrics_for(point)} - supported)
    save(output / 'requested-metrics.json', dict(points={p['name']:metrics_for(p) for p in points}, missing=missing))
    if missing:
        raise ValueError('当前 NCU 支持列表缺少指标：' + ','.join(missing))
    for point in points:
        folder = output / point['name']
        folder.mkdir()
        cmd = command(args.ncu, point, package, folder, args.cache_control, metrics_for(point))
        status = process(cmd, folder, 'probe-and-profiler', 1200)
        report = folder / 'profile.ncu-rep'
        if not report.exists():
            raise RuntimeError('NCU 未生成报告：' + point['name'])
        if process([args.ncu, '--import', str(report), '--page', 'raw', '--csv', '--print-units', 'base'],
                   folder, 'metrics', 120):
            raise RuntimeError('NCU 报告导出失败：' + point['name'])
        identified = identify_formal(point, folder/'probe-and-profiler.stdout', folder/'metrics.stdout')
        save(folder / 'formal-selection.json', dict(**identified, returncode=status,
            cache_control=args.cache_control, clock_control='none',
            original_task_id=point['original_task_id'], report_sha256=sha(report)))
        print(point['name'], 'returncode', status, 'matched launches', identified['profiled_launches'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--campaign', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    launch = sub.add_parser('run')
    launch.add_argument('--package', type=Path, required=True)
    launch.add_argument('--output', type=Path, required=True)
    launch.add_argument('--point', action='append')
    launch.add_argument('--metrics')
    launch.add_argument('--cache-control', choices=('none', 'all'), default='none')
    launch.add_argument('--ncu', default='ncu')
    launch.add_argument('--execute', action='store_true')
    launch.add_argument('--query-only', action='store_true')
    args = parser.parse_args()
    prepare(args) if args.action == 'prepare' else run(args)

