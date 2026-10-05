"""从通过离线重放的正式归档组生成表和图；C签署前参数仍为候选。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import statistics
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def render(run, replay_path, output, tables_only=False):
    run, output = Path(run), Path(output)
    replay = json.loads(Path(replay_path).read_text())
    assert replay['kind'] == 'offline_cohort_replay_pending_independent_C' and replay['CPU_fixture_only'] is False
    assert not output.exists()
    summary = json.loads((run / 'summary.json').read_text())
    device = json.loads((run / 'device.json').read_text())
    assert summary['case_results'] == replay['cases']
    contract_path = run.parent / 'repo/microbench/gh200_resource_campaign/contracts/tma_tensor_2d.json'
    contract = json.loads(contract_path.read_text())
    configs = {c['id']: c for c in contract['cases']}
    samples = []
    examples = {}
    for state_path in sorted(run.glob('index*-state.json')):
        state = json.loads(state_path.read_text())
        point = state['point']
        if point['mode'] != 'formal-only':
            continue
        archive = run / 'packs' / (state_path.name.split('-state.json')[0] + '.tar.xz')
        assert sha(archive) == state['pack_identity']['archive_sha256']
        raw = None
        with tarfile.open(archive, 'r|xz') as packet:
            for member in packet:
                if member.name == 'raw.jsonl':
                    raw = packet.extractfile(member).read()
                    break
        assert raw is not None and hashlib.sha256(raw).hexdigest() == state['pack_identity']['members']['raw.jsonl']['sha256']
        row = json.loads(raw.decode().splitlines()[1])
        case = configs[point['case_id']]
        w, h = case['parameters']['box_dim']
        amount = row['blocks'] * row['iterations'] * w * h * 2
        denominator = row['blocks_detail'][0]['stop_cycle'] - row['blocks_detail'][0]['start_cycle'] if case['scope'] == 'one_cta' else row['stop_ns'] - row['start_ns']
        assert amount == row['work_count'] and denominator > 0
        sample = {'case_id': case['id'], 'sampling_batch': point['sampling_batch'], 'trial': point['trial'],
            'seed': point['seed'], 'iterations': row['iterations'], 'blocks': row['blocks'],
            'payload_bytes': w * h * 2, 'work_count_bytes': amount, 'denominator': denominator,
            'unit': case['metric']['unit'], 'value': amount / denominator,
            'warmup_converged': row['warmup_converged'], 'post_timing_export_bytes': row['post_timing_G2S_export_bytes'],
            'pack_sha256': state['pack_identity']['archive_sha256'], 'raw_sha256': hashlib.sha256(raw).hexdigest()}
        samples.append(sample)
        examples.setdefault(case['scope'], sample)
        if (case['parameters']['direction'] == 'gmem_to_smem'
                and case['parameters']['layout'] == 'continuous_none'
                and w * h * 2 == 1024 and point['sampling_batch'] == 0 and point['trial'] == 0):
            examples[case['scope']] = sample
    parameters, excluded, table = [], [], []
    for key, result in replay['cases'].items():
        case = configs[key]
        accepted = [s for s in samples if s['case_id'] == key and s['warmup_converged']]
        values = [s['value'] for s in accepted]
        merged = result['merged']
        if merged:
            assert merged['samples'] == len(values) and merged['median'] == statistics.median(values)
        parameter = {'case_id': key, 'status': result['status'], 'unit': case['metric']['unit'],
            'value': merged['median'] if merged else None, 'statistic': 'median', 'samples': len(values),
            'cv': merged['cv'] if merged else None, 'scope': case['scope'],
            'conditions': {**case['parameters'], 'threads': 128, 'iterations': accepted[0]['iterations'] if accepted else None,
                           'blocks': accepted[0]['blocks'] if accepted else None},
            'qualification': 'candidate_requires_independent_C', 'physical_HBM_traffic_proven': False,
            'source_run': run.name, 'source_summary_sha256': sha(run / 'summary.json'),
            'source_device_sha256': sha(run / 'device.json'),
            'source_protocol_sha256': sha(run / 'protocol.json'),
            'source_manifest_sha256': sha(run / 'source-manifest.json'),
            'device_uuid': device['uuid']}
        (parameters if result['status'] == 'stable' else excluded).append(parameter)
        table.append({'case_id': key, 'status': result['status'], 'scope': case['scope'],
            'direction': case['parameters']['direction'], 'layout': case['parameters']['layout'],
            'payload_KiB': case['parameters']['payload_bytes'] / 1024,
            'iterations': parameter['conditions']['iterations'], 'blocks': parameter['conditions']['blocks'],
            'unit': parameter['unit'], 'median': parameter['value'], 'samples': len(values),
            'sample_standard_deviation': statistics.stdev(values) if len(values) >= 2 else None,
            'cv': parameter['cv'], 'qualified': False})
    output.mkdir()
    with (output / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader(); writer.writerows(samples)
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    (output / 'parameter-candidates.json').write_text(json.dumps({'kind': 'conditioned_TMA2D_service_candidates',
        'parameters': parameters, 'excluded': excluded, 'formal_C_required': True, 'qualified': False}, indent=2) + '\n')
    assert set(examples) == {configs[k]['scope'] for k in replay['cases']}
    (output / 'worked-example.json').write_text(json.dumps(examples, indent=2) + '\n')
    text = ['# TMA 2D 归档组候选结果', '',
            f"本组包含 {len(table)} 个配置、{len(samples)} 个正式进程样本；独立 C 审查完成前不授参数资格。", '',
            '单 CTA 的 B/clock64 cycle 与全 GPU 的 GB/s 分表使用。中位数来自全部预热收敛样本，CV 和图中标准差反映进程间离散程度。', '']
    for scope in ('one_cta', 'all_gpu'):
        if scope not in examples:
            continue
        text += [f'## {scope}', '', '| 方向 | 布局 | payload KiB | N | CTA 数 | 状态 | 中位数 | 单位 | CV | 有效样本数 |',
                 '|---|---|---:|---:|---:|---|---:|---|---:|---:|']
        for row in table:
            if row['scope'] == scope:
                median_text = '无有效统计' if row['median'] is None else format(row['median'], '.6g')
                cv_text = '无有效统计' if row['cv'] is None else format(row['cv'], '.3%')
                text.append(f"| {row['direction']} | {row['layout']} | {row['payload_KiB']:g} | {row['iterations']} | {row['blocks']} | {row['status']} | {median_text} | {row['unit']} | {cv_text} | {row['samples']} |")
        ex = examples[scope]
        text += ['', '真实样本算例：', '',
                 f"`{ex['case_id']}` 的第 0 批第 0 次进程完成 {ex['blocks']} × {ex['iterations']} × {ex['payload_bytes']} = {ex['work_count_bytes']} B。计时差为 {ex['denominator']}，因此指标为 {ex['value']:.9g} {ex['unit']}。", '',
                 f"循环结束后的 G2S 导出为 {ex['post_timing_export_bytes']} B，不计入循环运输量。raw SHA：`{ex['raw_sha256']}`。", '',
                 f"该进程预热收敛为 `{ex['warmup_converged']}`；未收敛样本保留在 samples.csv，不进入有效统计。", '',
                 *([] if tables_only else [f'![{scope}候选图]({scope}.png)']), '']
    (output / 'RESULTS.zh.md').write_text('\n'.join(text) + '\n')
    (output / 'sources.json').write_text(json.dumps({
        'run_leaf': run.name,
        'selected_case_count': replay['selected_case_count'],
        'formal_sample_count': replay['formal_sample_count'],
        'pilot_count': replay['pilot_count'],
        'qualified': False,
        'files': {name: sha(run / name) for name in
                  ('summary.json', 'device.json', 'protocol.json', 'source-manifest.json',
                   'artifact-manifest.json', 'case-selection.json')},
        'ordinary_replay_sha256': sha(replay_path),
        'report_script_sha256': sha(__file__),
        'contract_sha256': sha(contract_path)}, indent=2) + '\n')
    if tables_only:
        (output / 'report-manifest.json').write_text(json.dumps({p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in output.iterdir() if p.is_file()}, indent=2) + '\n')
        return
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gh200-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for scope in ('one_cta', 'all_gpu'):
        if scope not in examples:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, direction in zip(axes, ('gmem_to_smem', 'smem_to_gmem')):
            for layout in ('continuous_none', 'padding_none', 'continuous_sw128'):
                group = [p for p in parameters if p['scope'] == scope and p['conditions']['direction'] == direction and p['conditions']['layout'] == layout]
                group.sort(key=lambda p: p['conditions']['payload_bytes'])
                if not group: continue
                errors = [statistics.stdev(s['value'] for s in samples if s['case_id'] == p['case_id'] and s['warmup_converged']) for p in group]
                ax.errorbar([p['conditions']['payload_bytes'] / 1024 for p in group], [p['value'] for p in group], yerr=errors, marker='o', capsize=3, label=layout)
            ax.set_xscale('log', base=2); ax.set_xlabel('Completed payload / KiB')
            ax.set_ylabel('B / clock64 cycle / CTA' if scope == 'one_cta' else 'GB transport / s / GPU')
            ax.set_title(direction); ax.grid(alpha=.25); ax.legend()
        fig.suptitle('TMA 2D candidate cohort; error bars: sample standard deviation')
        fig.tight_layout(); fig.savefig(output / (scope + '.png'), dpi=160); plt.close(fig)
    (output / 'report-manifest.json').write_text(json.dumps({p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in output.iterdir() if p.is_file()}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True); parser.add_argument('--replay', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--tables-only', action='store_true', help='ARM CPU 作业用标准库先生成表与参数候选；取回本地后生成图')
    args = parser.parse_args(); render(args.run, args.replay, args.output, args.tables_only)
