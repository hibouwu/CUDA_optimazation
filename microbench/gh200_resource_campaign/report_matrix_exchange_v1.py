#!/usr/bin/env python3
"""从 S10 原始样本生成两类单位的结果、图表和中文说明；资格由独立 C 决定。"""
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
    assert summary['family'] == 'matrix_exchange' and summary['terminal']
    assert spec['kind'] == 'formal' and not spec['fixture'] and len(contract['cases']) == 22
    configs = {c['id']: c for c in contract['cases']}
    rows, parameters, examples = [], [], []
    for case in summary['cases']:
        config = configs[case['case_id']]
        p = config['parameters']
        shuffle = p['mode'] == 'shuffle'
        direction = 0 if shuffle else 8192 * 8 * p['matrices'] * 8 * 8 * 2
        read = direction if p['mode'] in ('load', 'roundtrip') else 0
        write = direction if p['mode'] in ('store', 'roundtrip') else 0
        work = 8192 * 8 * p['streams'] * (config['threads'] // 32) if shuffle else read + write
        case_rows = []
        resources = []
        for batch in case['batches']:
            for sample in batch['samples']:
                raw = (run / sample['receipt']).parent / 'raw.jsonl'
                assert sha(raw) == sample['raw_sha256']
                raw_rows = [json.loads(line) for line in raw.read_text().splitlines()]
                trial = next(x for x in raw_rows if x['type'] == 'trial')
                assert trial['iterations'] == 8192 and trial['errors'] == 0 and trial['short_check_errors'] == 0
                assert trial['work_count'] == work and trial['read_payload_bytes'] == read and trial['write_payload_bytes'] == write
                assert trial['nonuniform_short_checks'] == ([1, 3, 33] if shuffle else [1, 2])
                assert len(trial['blocks_detail']) == 1
                block = trial['blocks_detail'][0]
                cycles = block['stop_cycle'] - block['start_cycle']
                assert cycles > 0 and work / cycles == sample['value']
                row = {'case_id': case['case_id'], 'batch': sample['batch'], 'trial': sample['trial'],
                       'work_count': work, 'read_bytes': read, 'write_bytes': write,
                       'clock64_cycles': cycles, 'value': work / cycles, 'unit': case['unit'],
                       'warmup_converged': sample['warmup_converged'],
                       'raw': str(raw), 'raw_sha256': sha(raw)}
                rows.append(row)
                case_rows.append(row)
                resources.append({key: trial[key] for key in ('registers_per_thread', 'static_smem_bytes',
                    'dynamic_smem_bytes', 'local_size_bytes', 'occupancy_limit_ctas_per_sm')})
                if sample['batch'] == 0 and sample['trial'] == 0:
                    examples.append(row)
        accepted = [r['value'] for r in case_rows if r['warmup_converged']]
        merged = case['merged']
        if merged:
            assert len(accepted) == merged['samples'] and statistics.median(accepted) == merged['median']
        else:
            assert len(accepted) < 2 and not case['exportable']
        assert all(r == resources[0] for r in resources)
        parameters.append({'case_id': case['case_id'], 'status': case['status'], 'exportable': case['exportable'],
            'unit': case['unit'], 'value': merged['median'] if merged else None,
            'statistic': 'median' if merged else None, 'samples': len(accepted),
            'minimum': min(accepted) if accepted else None, 'maximum': max(accepted) if accepted else None,
            'cv': merged['cv'] if merged else None, 'batches': len(case['batches']),
            'exclusion_reason': None if case['exportable'] else case['status'],
            'conditions': {'iterations': 8192, 'threads': config['threads'], 'blocks': 1, **p, **resources[0]},
            'read_payload_bytes': read, 'write_payload_bytes': write, 'work_count': work,
            'scope': 'single_CTA_complete_matrix_or_shuffle_loop', 'source_summary_sha256': sha(run / 'summary.json')})
    output.mkdir(parents=True)
    with (output / 'samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'worked_examples.json').write_text(json.dumps(examples, indent=2) + '\n')
    (output / 'parameters.json').write_text(json.dumps({'schema_version': 1, 'kind': 'conditioned_S10_services',
        'qualification': 'requires_independent_C_and_COMPLETE', 'run': str(run),
        'parameters': [p for p in parameters if p['exportable']],
        'excluded': [p for p in parameters if not p['exportable']],
        'bare_instruction_latency_claimed': False, 'physical_bandwidth_claimed': False}, indent=2) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gh200-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [('matrix', [p for p in parameters if not p['case_id'].startswith('shuffle')], 'Logical B / clock64 cycle / CTA'),
              ('shuffle', [p for p in parameters if p['case_id'].startswith('shuffle')], 'Warp instructions / clock64 cycle / CTA')]
    for name, group, unit in groups:
        fig, ax = plt.subplots(figsize=(10, max(3, len(group) * .32 + 1)))
        for i, parameter in enumerate(group):
            values = [r['value'] for r in rows if r['case_id'] == parameter['case_id'] and r['warmup_converged']]
            if values:
                ax.scatter(values, [i] * len(values), s=12, alpha=.45, color='#26718a')
            if parameter['value'] is None:
                ax.text(0, i, 'insufficient converged samples', fontsize=8)
                continue
            ax.errorbar(parameter['value'], i,
                xerr=[[parameter['value'] - min(values)], [max(values) - parameter['value']]],
                fmt='D', color='#183e52', capsize=3)
        ax.set_yticks(range(len(group)), [p['case_id'] for p in group])
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.set_title('S10: all warmup-converged samples, median and min/max')
        ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        fig.savefig(output / (name + '.png'), dpi=150)
        plt.close(fig)
    def table(group):
        lines = []
        for p in group:
            median = '—' if p['value'] is None else f"{p['value']:.6f}"
            extent = '—' if p['minimum'] is None else f"{p['minimum']:.6f}–{p['maximum']:.6f}"
            cv = '—' if p['cv'] is None else f"{p['cv'] * 100:.6f}%"
            lines.append(f"| {p['case_id']} | {p['samples']} | {median} | {extent} | {cv} | {p['status']} |")
        return '\n'.join(lines)
    matrix_example = next(r for r in examples if r['case_id'] == 'load_x2_normal')
    shuffle_example = next(r for r in examples if r['case_id'] == 'shuffle_t128_streams4')
    initial = json.loads((run / 'environment/initial.json').read_text())
    header = '| 配置 | 收敛样本 | 中位数 | 最小值–最大值 | CV | 状态 |\n|---|---:|---:|---:|---:|---|\n'
    text = f'''# EXP-10：GH200 矩阵片段搬运与 warp 交换实测

这组实验回答两个问题：SMEM 与寄存器之间的矩阵片段搬运，在给定布局下每周期能服务多少逻辑字节；warp 内寄存器交换，在给定依赖链与并发流下每周期能执行多少条 warp 指令。22 配置共 {len(rows)} 个正式进程样本，两个问题采用不同单位，图表分别展示。

## 探针与计时

矩阵探针固定单 CTA、32 线程、m8n8.b16。比较 load、store、load→store，x1/x2/x4，正常与 `.trans`；源与目标每方向4096 B，八个槽各512 B，动态 SMEM 共8192 B。x2 表示两块矩阵，由完整 warp 共同访问，不能再乘32线程。

shuffle 探针固定单 CTA，32或128线程，1或4条独立流；每条流按 `(lane+1)%32` 读取前一状态，自身有依赖链。每轮八个操作位置，所有配置固定8192轮。

输入初始化在计时前；线程0记录同 SM 的 clock64，再经 CTA barrier 进入循环。窗口内包含地址计算、矩阵结果消费或 store 值生成、必要同步及结果排空；load→store 的 PTX 源码每对指令请求 `bar.warp.sync`，但本次编译的六个往返目标在对应位置降为 NOP，SASS 中没有 `WARPSYNC`，不能把它另算成实际同步成本。host回读与核验在窗口外。因此结果代表本完整循环的服务，不能直接当成裸指令延迟或物理端口带宽。

## 矩阵搬运结果

单位为 **逻辑 B/clock64 cycle/CTA**；往返配置的工作量为读取与写入之和。

{header}{table(groups[0][1])}

![矩阵搬运全部样本、中位数和范围]({output / 'matrix.png'})

## warp 交换结果

单位为 **warp指令/clock64 cycle/CTA**。一条完整 warp 的 `shfl.sync` 计为一条指令；128线程包含4个warp，不能把 lane 数当成指令数，也不计为 FLOP。

{header}{table(groups[1][1])}

![warp交换全部样本、中位数和范围]({output / 'shuffle.png'})

图中点为预热收敛的独立进程，菱形为中位数，横线为全部合并样本的最小值到最大值，不是置信区间。所有批次和未收敛记录均保存在归档中；只导出状态稳定且允许导出的配置。

## 从原始字段手算

`load_x2_normal` 每位置共同加载 `2×8×8×2=256 B`，总工作量 `8192×8×256={matrix_example['work_count']:,} B`。第一条 raw 的同 SM clock64 差为 **{matrix_example['clock64_cycles']:,} cycle**，所以结果是 `{matrix_example['work_count']}/{matrix_example['clock64_cycles']}={matrix_example['value']:.9f} B/cycle/CTA`。见[原始记录]({matrix_example['raw']})。

`shuffle_t128_streams4` 工作量为 `8192轮×8位置×4条流×4个warp={shuffle_example['work_count']:,} 条warp指令`。第一条 raw 周期差 **{shuffle_example['clock64_cycles']:,} cycle**，得到 `{shuffle_example['work_count']}/{shuffle_example['clock64_cycles']}={shuffle_example['value']:.9f} warp指令/cycle/CTA`。见[原始记录]({shuffle_example['raw']})。

load→store 的字节速率包含两个方向和它们之间的依赖、同步，不可把完成时间除二声称单向延迟。比较多流或多warp时，也要同时考虑寄存器、循环和CTA同步的变化。

## 正确性与适用范围

job{initial['job']} 在 {initial['host']} 上执行本次自己的22点检查。每个进程用非均匀输入验证矩阵1/2轮或shuffle 1/3/33步，长循环仍检查全部片段、校验和、输出及padding。SASS审查确认相关指令留在循环内。实际完整输出数组没有持久化，归档保存的是host核验结果和原始样本，不能宣称事后逐元素独立重算了长循环输出。

使用[条件参数]({output / 'parameters.json'})时匹配线程、warp、片段数、布局、独立流数、动态SMEM和完成边界。本组没有测Tensor Core直接消费这些片段、多个CTA竞争或全GPU服务，也没有测完整GEMM；物理端口峰值和bare latency不在结论范围内。独立C、原ARM严格封存与COMPLETE决定结果资格，报告脚本本身不授资格。

## 复现入口

[原始run]({run})、[逐样本CSV]({output / 'samples.csv'})、[手算字段]({output / 'worked_examples.json'})。离线审查使用归档冻结入口：

```sh
python3 -B {run / 'snapshot/repo/microbench/gh200_resource_campaign/run_suite.py'} audit {run}
```

图表和说明可用 `report_matrix_exchange_v1.py --run <run> --output <新目录> --document <新文档>` 重生。重新测量使用该run的冻结合同和源码，在有效单GPU Slurm分配内先完成自己的preflight与独立B，再正式运行；不能直接把历史B当成新设备的测量准入。
'''
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(text)
    manifest = {'run': str(run), 'summary_sha256': sha(run / 'summary.json'),
        'measurement_manifest_sha256': sha(run / 'measurement_manifest.json'),
        'script_sha256': sha(Path(__file__)), 'sample_count': len(rows), 'parameter_count': len(parameters),
        'source_raw_files': {r['raw']: r['raw_sha256'] for r in rows}, 'document': str(document),
        'outputs': {str(p): sha(p) for p in [*output.iterdir(), document]}, 'C_qualified_by_renderer': False}
    (output / 'analysis-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'samples': len(rows), 'configurations': len(parameters), 'document': str(document)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--document', type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.output, args.document)
