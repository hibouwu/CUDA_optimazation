#!/usr/bin/env python3
"""从 a044 返回包读取正式 NCU 目标、单位和流量，保留原包身份。"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile


def read(path):
    return json.loads(path.read_text())


def analyze(campaign):
    points = {point['name']: point for point in read(campaign/'ncu-targeted-package/manifest.json')['points']}
    observations = []
    for run in sorted(campaign.glob('ncu-*')):
        archive = run/'output.tar.gz'
        if not archive.is_file():
            continue
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            selections = [(name, None) for name in names if name.endswith('/formal-selection.json')]
            recovery_path = run/'cpu-recovery/recovery.json'
            if recovery_path.is_file():
                recovery = read(recovery_path)
                if hashlib.sha256(archive.read_bytes()).hexdigest() != recovery['source_archive_sha256']:
                    raise ValueError('CPU 恢复记录与原归档不符：'+str(run))
                selections.append((recovery['member_directory']+'/formal-selection.json',
                                   run/'cpu-recovery/formal-selection.json'))
            for name, recovered in selections:
                folder = name.rsplit('/', 1)[0]
                point_name = folder.rsplit('/', 1)[-1]
                point = points[point_name]
                selection = read(recovered) if recovered else json.loads(tar.extractfile(name).read())
                command = json.loads(tar.extractfile(folder+'/probe-and-profiler.receipt.json').read())['command']
                requested_metrics = command[command.index('--metrics')+1].split(',')
                csv_rows = list(csv.DictReader(io.StringIO(tar.extractfile(folder+'/metrics.stdout').read().decode())))
                units = next(row for row in csv_rows if not row['ID'])
                formal = selection['original_formal_record']
                if not isinstance(formal, list):
                    formal = [formal]
                for selected, original in zip(selection['selected_formal'], formal, strict=True):
                    metrics = {key: float(selected[key].replace(',', '')) for key in requested_metrics
                               if key in selected and selected[key] not in ('n/a', '')}
                    if 'profiler__replayer_passes' in selected:
                        metrics['profiler__replayer_passes'] = int(selected['profiler__replayer_passes'])
                    observation = dict(run=run.name, point=point_name, archive=str(archive),
                        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), member=name,
                        ncu_launch_id=selected['ID'], kernel=selected['Kernel Name'],
                        warmup_calls=selection['warmup_calls'], profiled_launches=selection['profiled_launches'],
                        returncode=selection['returncode'], clock_control=selection['clock_control'],
                        cache_control=selection['cache_control'], metrics=metrics,
                        units={key: units[key] for key in metrics},
                        original_record={key: original[key] for key in ('case_id', 'pair_index', 'request_mask',
                            'errors', 'event_ms', 'warmup_converged', 'elapsed_ns', 'elapsed_cycles', 'wgmma_count')
                                         if key in original},
                        program_work=point['expected'])
                    if recovered:
                        observation['cpu_recovery'] = str(recovery_path)
                        observation['selection_file'] = str(recovered)
                    if 'lts__t_sectors_op_read.sum' in metrics:
                        observation['l2_read_sector_bytes'] = metrics['lts__t_sectors_op_read.sum']*32
                    if 'lts__t_sectors_op_write.sum' in metrics:
                        observation['l2_write_sector_bytes'] = metrics['lts__t_sectors_op_write.sum']*32
                    observations.append(observation)
    return dict(observations=observations, interpretation=dict(
        dram='DRAM 计数器的 byte，表示 L2 与 HBM 间的传输；此处操作数来自本地 cudaMalloc。',
        l2='L2 请求 sector 数乘 32 B，包含整个 kernel 的请求和记录开销。',
        gmma='查询表定义为 tensor pipe 的 warp instructions；程序 WGMMA 次数按 warp group 操作列出。',
        tma='查询表定义为 TMA pipe 的 warp instructions；程序 tensor 请求数另列。',
        timing='NCU 完整 kernel 时间与原局部计时并列；plain 性能使用独立正式测量。',
        replay='多 pass 且 cache-control none 时，按报告中的 replay 缓存条件解释流量，另用小指标组复核。'),
        metric_semantics='ncu-supported-query/metric-semantics.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.campaign)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    for row in result['observations']:
        print(json.dumps({key: row[key] for key in ('run', 'point', 'ncu_launch_id', 'metrics')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
