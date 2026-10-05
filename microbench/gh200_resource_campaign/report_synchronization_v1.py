#!/usr/bin/env python3
"""从S11真实样本生成同步、fence和逻辑warp阶段的分组结果。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(run, output, document):
    run, output, document = run.resolve(), output.resolve(), document.resolve()
    assert not output.exists() and not document.exists()
    summary = json.loads((run / 'summary.json').read_text())
    spec = json.loads((run / 'run_spec.json').read_text())
    contract = json.loads((run / spec['contract_path']).read_text())
    assert summary['terminal'] and summary['family'] == 'synchronization'
    assert spec['kind'] == 'formal' and not spec['fixture'] and len(contract['cases']) == 13
    configs = {c['id']: c for c in contract['cases']}
    rows, parameters, examples = [], [], []
    for case in summary['cases']:
        config = configs[case['case_id']]
        phases = config['iterations'] * 8
        local_rows, resources = [], []
        for batch in case['batches']:
            for sample in batch['samples']:
                raw = (run / sample['receipt']).parent / 'raw.jsonl'
                assert sha(raw) == sample['raw_sha256']
                trial = next(json.loads(line) for line in raw.read_text().splitlines() if json.loads(line)['type'] == 'trial')
                assert trial['iterations'] == config['iterations'] and trial['errors'] == 0 and trial['work_count'] == phases
                assert trial['read_payload_bytes'] == trial['write_payload_bytes'] == 0
                assert len(trial['blocks_detail']) == 1
                stamp = trial['blocks_detail'][0]
                cycles = stamp['stop_cycle'] - stamp['start_cycle']
                assert cycles > 0 and cycles / phases == sample['value']
                measured = trial['thread_results']
                assert len(measured) == config['threads']
                assert all(t['completed'] == phases and t['timeout'] == t['errors'] == 0 for t in measured)
                waits = sum(t['wait_attempts'] for t in measured)
                row = {'case_id': case['case_id'], 'batch': sample['batch'], 'trial': sample['trial'],
                       'iterations': config['iterations'], 'work_count': phases, 'clock64_cycles': cycles,
                       'value': cycles / phases, 'unit': case['unit'], 'warmup_converged': sample['warmup_converged'],
                       'measured_wait_attempts_total': waits,
                       'wait_attempts_per_phase_per_thread': waits / (phases * config['threads']),
                       'separate_arrival_spread_cycles': trial['arrival_diagnostic']['arrival_spread_cycles'],
                       'raw': str(raw), 'raw_sha256': sha(raw)}
                rows.append(row)
                local_rows.append(row)
                resources.append({key: trial[key] for key in ['registers_per_thread', 'static_smem_bytes', 'local_size_bytes']})
                if sample['batch'] == sample['trial'] == 0:
                    examples.append(row)
        accepted = [r for r in local_rows if r['warmup_converged']]
        values = [r['value'] for r in accepted]
        merged = case['merged']
        if merged:
            assert len(values) == merged['samples'] and statistics.median(values) == merged['median']
        else:
            assert len(values) < 2 and not case['exportable']
        assert resources and all(r == resources[0] for r in resources)
        parameters.append({'case_id': case['case_id'], 'unit': case['unit'], 'status': case['status'],
            'exportable': case['exportable'], 'value': merged['median'] if merged else None,
            'statistic': 'median' if merged else None, 'samples': len(values), 'batches': len(case['batches']),
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None,
            'cv': merged['cv'] if merged else None,
            'exclusion_reason': None if case['exportable'] else ('compiled_no_native_WARPSYNC' if config['parameters']['mode'] == 'warp' else case['status']),
            'conditions': {'iterations': config['iterations'], 'threads': config['threads'], 'blocks': 1,
                           'dynamic_smem_bytes': 0, **config['parameters'], **resources[0]},
            'separate_arrival_spread_cycles_median': statistics.median(r['separate_arrival_spread_cycles'] for r in accepted) if accepted else None,
            'wait_attempts_per_phase_per_thread_median': statistics.median(r['wait_attempts_per_phase_per_thread'] for r in accepted) if accepted else None,
            'scope': 'one_CTA_complete_phase_loop_with_drain_and_timeout_protocol',
            'source_summary_sha256': sha(run / 'summary.json')})
    output.mkdir(parents=True)
    with (output / 'samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'worked_examples.json').write_text(json.dumps(examples, indent=2) + '\n')
    (output / 'parameters.json').write_text(json.dumps({'schema_version': 1, 'kind': 'conditioned_S11_phase_services',
        'qualification': 'requires_independent_C_and_COMPLETE', 'run': str(run),
        'parameters': [p for p in parameters if p['exportable']], 'excluded': [p for p in parameters if not p['exportable']],
        'bare_barrier_latency_claimed': False, 'fence_async_completion_claimed': False}, indent=2) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gh200-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [('collective', [p for p in parameters if p['conditions']['mode'] in ['cta', 'mbarrier']], 'clock64 cycle / collective phase / CTA'),
              ('fence', [p for p in parameters if p['conditions']['mode'] in ['fence_cta', 'fence_gpu', 'proxy_async']], 'clock64 cycle / warp instruction / CTA'),
              ('logical_warp', [p for p in parameters if p['conditions']['mode'] == 'warp'], 'clock64 cycle / logical phase / CTA')]
    for name, group, unit in groups:
        fig, ax = plt.subplots(figsize=(10, max(3, .4 * len(group) + 1)))
        for i, p in enumerate(group):
            values = [r['value'] for r in rows if r['case_id'] == p['case_id'] and r['warmup_converged']]
            if values:
                ax.scatter(values, [i] * len(values), s=13, alpha=.45, color='#26718a')
            if p['value'] is None:
                ax.text(0, i, 'insufficient converged samples', fontsize=8)
                continue
            ax.errorbar(p['value'], i, xerr=[[p['value'] - min(values)], [max(values) - p['value']]], fmt='D', color='#183e52', capsize=3)
        ax.set_yticks(range(len(group)), [p['case_id'] for p in group])
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.set_title('S11: samples, median and min/max' + ('; nonexportable' if name == 'logical_warp' else ''))
        ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        fig.savefig(output / (name + '.png'), dpi=150)
        plt.close(fig)
    def table(group):
        lines = []
        for p in group:
            value = '—' if p['value'] is None else f"{p['value']:.6f}"
            extent = '—' if p['minimum'] is None else f"{p['minimum']:.6f}–{p['maximum']:.6f}"
            cv = '—' if p['cv'] is None else f"{p['cv'] * 100:.6f}%"
            lines.append(f"| {p['case_id']} | {p['samples']} | {value} | {extent} | {cv} | {p['status']} |")
        return '\n'.join(lines)
    sections = []
    header = '| 配置 | 收敛样本数 | 中位数 | 最小值–最大值 | CV | 状态 |\n|---|---:|---:|---:|---:|---|\n'
    for title, (name, group, unit) in zip(['CTA与mbarrier', '无未完成请求的fence', '不可导出的逻辑warp阶段'], groups):
        sections.append('## ' + title + '\n\n单位：`' + unit + '`。\n\n' + header + table(group) + '\n\n![' + title + '](' + str(output / (name + '.png')) + ')\n')
    example = next(x for x in examples if x['case_id'] == 'mbarrier_t128_aligned')
    initial = json.loads((run / 'environment/initial.json').read_text())
    text = f'''# EXP-11：GH200 同步与完成事件实测

这组实验比较齐步到达与固定整数工作导致的到达偏斜，测量一个CTA完整同步阶段的周期；另外测没有未完成内存请求时的fence指令序列。13配置共{len(rows)}个正式进程样本。原两个warp点没有编译出目标WARPSYNC，仍展示逻辑阶段观察，但不导出原生warp barrier参数。

## 探针和计时

同步点固定2048轮、每轮8阶段；fence点固定8192轮、每轮8个warp指令位置。CTA与mbarrier分别有128/256线程，warp与fence有32线程。偏斜条件是最后一个warp，或warp点的后16个lane，每阶段执行256条有依赖的整数MAD，其余参与者先到达。

计时窗口包含循环、固定工作、被测同步或fence、mbarrier等待与计数、超时保护及最终结果排空。输入初始化在窗口前，host回读在窗口外。起止时间由同一CTA的线程0用同SM的clock64记录。每个进程另做2阶段consumer正确性检查和1阶段到达诊断，两个额外launch都不贡献正式周期；不能把不同launch的clock64相减。

{''.join(sections)}
图中的点为预热收敛的独立进程，菱形为中位数，横线为全部合并样本的最小值到最大值，不是置信区间。所有批次与未收敛记录保留；稳定且允许导出的配置才进入参数文件。

## 从原始字段算一条结果

`mbarrier_t128_aligned` 的整个CTA共同完成 `2048×8={example['work_count']}` 个阶段，不能再乘128线程。第一条raw周期差为 **{example['clock64_cycles']:,} cycle**，结果是 `{example['clock64_cycles']}/{example['work_count']}={example['value']:.9f} cycle/phase/CTA`。见[原始记录]({example['raw']})。

该条记录的等待尝试总数为{example['measured_wait_attempts_total']:,}，除以阶段数和128线程，得到每线程每阶段平均{example['wait_attempts_per_phase_per_thread']:.6f}次。该数量包括成功与重复轮询，不能当成mbarrier操作数或物理队列深度。

## 如何用于模型

CTA/mbarrier的阶段时间包括参与者就绪、同步服务、等待和本循环处理。偏斜点还明确包含依赖整数工作；不能将其与齐步点的差全部解释为同步指令裸延迟。单阶段的到达时间差只来自另外的诊断launch，作为就绪条件的观测，不能直接从正式循环时间中扣除。

三个fence点均没有未完成内存请求，其结果只代表这段空请求条件下的序列。fence提供相应顺序关系，不是TMA、cp.async或WGMMA统一完成事件。两个warp点被编译为无目标WARPSYNC的逻辑循环，不授native warp barrier吞吐或延迟。

匹配参与线程、同步类型、偏斜工作、等待与超时协议、静态SMEM和完成边界后，可使用[条件参数]({output / 'parameters.json'})。这里仅测单CTA，不外推多CTA竞争、完整GEMM或硬件峰值。

## 正确性、来源与复现

job{initial['job']}在{initial['host']}上完成本次自己的13点检查，正式源的39个measured/correctness/arrival角色逐机器码和编译资源核对。三个角色的最终逐线程值、阶段计数、timeout/errors、wait_attempts、consumer检查和诊断时间均持久化，可离线按uint32递推独立复算。中间phase tokens未全部保存，逐phaseconsumer零错误来自执行检查，不能宣称所有中间状态全值重放。

[run归档]({run})、[逐样本CSV]({output / 'samples.csv'})和[手算字段]({output / 'worked_examples.json'})。独立C、原ARM严格封存、COMPLETE和换目录复核决定资格，报告脚本本身不授资格。

冻结入口重算：

```sh
python3 -B {run / 'snapshot/repo/microbench/gh200_resource_campaign/run_suite.py'} audit {run}
```

图表和说明用 `report_synchronization_v1.py --run <run> --output <新目录> --document <新文档>` 重生。重新采样在有效单GPU Slurm分配内完成自己的preflight与独立B后运行原合同；原始参数与旧设备B不能自动授权未来运行。
'''
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(text)
    manifest = {'run': str(run), 'summary_sha256': sha(run / 'summary.json'),
        'measurement_manifest_sha256': sha(run / 'measurement_manifest.json'),
        'script_sha256': sha(Path(__file__)), 'sample_count': len(rows),
        'parameter_count': sum(p['exportable'] for p in parameters), 'excluded_count': sum(not p['exportable'] for p in parameters),
        'source_raw_files': {r['raw']: r['raw_sha256'] for r in rows}, 'document': str(document),
        'outputs': {str(p): sha(p) for p in [*output.iterdir(), document]}, 'C_qualified_by_renderer': False}
    (output / 'analysis-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'samples': len(rows), 'exportable': manifest['parameter_count'], 'excluded': manifest['excluded_count']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--document', type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.output, args.document)
