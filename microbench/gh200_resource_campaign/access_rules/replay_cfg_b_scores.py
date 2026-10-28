#!/usr/bin/env python3
"""复核旧cfg_b十二行开发评分；只读已保存预测、观测和逐进程评分。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics as st


BASE = Path('gh200_resource_campaign/access_rules')
FULL = BASE / 'Q14-composition/cfg-b-composition/20261015-cfg-b-full-matrix-001'
PREFIX = BASE / 'Q01-scheduling/startup-prefix/20261016-cfg-b-prefix-001'
V09 = BASE / 'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011'
ANCHORS = {
    FULL / 'delivery-manifest.json': '595a97c68982ab539607209aa1e741cc58b9b0ed828c3fc229a80063de662d18',
    FULL / 'plan.json': 'aa20ba52b61469432851397e40385a2f42dc01cf044c51eb64cda19d8654e521',
    FULL / 'comparison-final.json': '62687d4c1b516c1724011c415e2520b7477df80d958b2a7162d0779960d0a96e',
    PREFIX / 'delivery-manifest.json': '6bc5765dce4311d5b799336652f4aba002cc71bbdaef5ec877117f0a95b84ae0',
    PREFIX / 'plan.json': '14e551ef5becb1eb5d4220fa5b6b5c7c19d5d0b9f553172088721811bc4d7d6e',
    PREFIX / 'binding.json': '41d7d34b4bbf4b190dfcb1b2e76ca147592138dd1528f880d22e090e9e998773',
    PREFIX / 'comparison.json': '6181e1626bfe69188551e91e233711288c6cb512332bf3dc354ac2948d622c6e',
    V09 / 'score.json': '37492c12cc165994587986a17db7ed9f1dce8e33648c4ede5e64265861be4b19',
    V09 / 'prediction_binding.json': '92123e07fd263bf46d55692bfd81943b96008068b561d672c9f6b74b22b6a96d',
    V09 / 'frozen/manifest.json': '09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab',
    V09 / 'frozen/predictions.json': 'f4df39e00cf1d2ab61947c911e699bd5bea4e533609d93f726de43077515d81e',
}
COMPONENTS = ('prefix', 'prefill', 'first_L', 'later_L', 'last_E', 'tail')
RESPONSES = (
    ('v09d_cfg_b_G3_stage4', 'v09d_cfg_b_G3_base'),
    ('v09d_cfg_b_G3_A16', 'v09d_cfg_b_G3_base'),
    ('v09d_cfg_b_G5_B32', 'v09d_cfg_b_G5_base'),
    ('v09d_cfg_b_G5_B80', 'v09d_cfg_b_G5_base'),
    ('v09d_cfg_b_G5_B112', 'v09d_cfg_b_G5_base'),
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def bind(path, expected, identities):
    require(sha(path) == expected, 'SHA不符: ' + str(path))
    identities[str(path)] = expected


def observation_mapping(path, calibration):
    # 只抽冻结纯映射类，不导入求解器、不执行旧analyze.evaluate。
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'V09Observation']
    require(len(classes) == 1, '冻结V09Observation缺失')
    namespace = {}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['V09Observation'](calibration)


def component_summary(trials):
    return {key: {field: st.median([trial['components'][key][field] for trial in trials
            if trial['components'][key][field] is not None])
        if any(trial['components'][key][field] is not None for trial in trials) else None
        for field in ('predicted_ns', 'observed_ns', 'error_pct')} for key in COMPONENTS}


def replay(results):
    identities = {}
    for relative, expected in ANCHORS.items():
        bind(results / relative, expected, identities)
    # 两份旧交付的有限文件清单；仅哈希压缩时间线，不解码raw或重新求解模型。
    for relative in (FULL, PREFIX):
        root = results / relative
        for name, expected in read(root / 'delivery-manifest.json')['files'].items():
            path = results.parent / name if name.startswith('results/') else root / name
            bind(path, expected, identities)
        for name, expected in read(root / 'binding.json')['files'].items():
            require(name.startswith('results/'), '旧绑定出现非归档输入')
            bind(results.parent / name, expected, identities)
    full, prefix, archive = results / FULL, results / PREFIX, results / V09
    frozen = read(archive / 'frozen/predictions.json')
    prediction_binding = read(archive / 'prediction_binding.json')
    require(prediction_binding['prediction_sha256'] == ANCHORS[V09 / 'frozen/predictions.json'] and
            prediction_binding['manifest_sha256'] == ANCHORS[V09 / 'frozen/manifest.json'], '原预测绑定不符')
    frozen_manifest = read(archive / 'frozen/manifest.json')['files']
    for name, expected in frozen_manifest.items():
        bind(archive / name, expected, identities)
    build_identities = {}
    for name, expected in read(archive / 'build/binary_hashes.json').items():
        if '/cfg_b' in name:
            bind(archive / name, expected, identities); build_identities[name] = expected
    for name, expected in read(archive / 'build/sass_hashes.json').items():
        if name.startswith('cfg_b'):
            bind(archive / 'build' / name, expected, identities); build_identities['build/' + name] = expected
    plan, registered = read(full / 'plan.json'), read(prefix / 'plan.json')
    before, after = read(full / 'comparison-final.json'), read(prefix / 'comparison.json')
    score = read(archive / 'score.json')
    names = [item['case']['id'] for item in plan['rows']]
    require(len(names) == len(set(names)) == 12 and before['conditions'] == before['completed_conditions'] ==
            after['conditions'] == 12, '原十二行覆盖不符')
    require([row['case'] for row in before['rows']] == [row['case'] for row in after['rows']] == names and
            {row['case'] for row in score['cases'] if row['case'].startswith('v09d_cfg_b_')} == set(names),
            '计划/两比较/739011评分覆盖不符')
    prefix_ns = registered['producer_prefix_ns']
    calibration = read(archive / 'frozen/calibration.json')
    require(prefix_ns == after['producer_prefix_ns'] == calibration['events_ns']['cfg_b']['P0'] == 1608.,
            '固定P0经验替代身份不符')
    policy = frozen['policy']
    require((policy['median_percent'], policy['maximum_percent']) == (5, 10), '原5%/10%门槛不符')
    mapper = observation_mapping(prefix / 'source/cfg_b_composition.py', calibration)
    by_before = {row['case']: row for row in before['rows']}
    by_after = {row['case']: row for row in after['rows']}
    by_score = {row['case']: row for row in score['cases']}
    # 先检查三类真实保存路径，再复核全部十二行。
    preflight = [full / 'output/v09d_cfg_b_G1_base/report.json',
                 prefix / 'output/v09d_cfg_b_G1_base/report.json',
                 prefix / 'output/v09d_cfg_b_G6_base/projection.json']
    require(read(preflight[0])['prediction']['plain_us'] == by_before[names[0]]['mechanism']['plain_us'] and
            read(preflight[1])['prediction']['plain_us'] == by_after[names[0]]['after']['plain_us'] and
            read(preflight[2])['after'] == by_after['v09d_cfg_b_G6_base']['after'], '三类真实保存路径不符')
    raw_evidence = read(full / 'raw-evidence.json')
    raw_cases = {row['case']: row for row in raw_evidence['cases']}
    rows, failed_attempts, critical_hits = [], {}, 0
    for item in plan['rows']:
        name = item['case']['id']
        old, new, original = by_before[name], by_after[name], by_score[name]
        report = read(full / 'output' / name / 'report.json')
        require(new['before'] == old['mechanism'] and report['case'] == name and
                old['shape'] == new['shape'] == [item['case'][key] for key in ('m', 'n', 'k')] and
                old['stages'] == new['stages'] == item['setup']['stages'], '旧行身份/原before不符: ' + name)
        require(old['v09_plain_us'] == item['aggregate']['plain_us'] == original['predicted_plain_us'] ==
                frozen['predictions'][name]['plain_us'], '原V09冻结预测不符: ' + name)
        observed_plain = st.median(record['elapsed_us'] for record in report['observed']['records']
                                   if record['variant'] == 'plain')
        require(observed_plain == original['observed_plain_us'] == old['observed_plain_us'] ==
                new['observed_plain_us'] == report['observed']['plain_us'], '保存plain观测中位数不符: ' + name)
        old_error = 100 * (report['prediction']['plain_us'] / observed_plain - 1)
        window = report['prediction']['window_ns'] + prefix_ns
        mapped = mapper.project(dict(window_us=window / 1000))
        new_error = 100 * (mapped['plain_us'] / observed_plain - 1)
        require(old_error == old['mechanism']['plain_error_pct'] == report['plain_error_pct'] and
                mapped['plain_us'] == new['after']['plain_us'] and new_error == new['after']['plain_error_pct'] and
                window == new['after']['window_ns'] and new['after']['phase_errors_pct'] == report['phase_errors_pct'],
                '保存预测的plain/包络重评分不符: ' + name)
        replayed = name in registered['first_cases']
        if replayed:
            saved = read(prefix / 'output' / name / 'report.json')
            require(saved['prediction']['plain_us'] == mapped['plain_us'] and saved['observed'] == report['observed'],
                    '原真实前缀report不符: ' + name)
        else:
            saved = read(prefix / 'output' / name / 'projection.json')
            require(saved['after'] == new['after'] and saved['new_model_replays'] == 0 and
                    saved['source_timeline_sha256'] == report['timeline_sha256'], '原projection不符: ' + name)
        candidates = new['after']['predicted_critical_candidates']
        require(candidates == old['mechanism']['predicted_critical_candidates'], '共同前缀改变原关键CTA候选')
        for version in (old['mechanism'], new['after']):
            trials = version['trials']
            require([trial['trial'] for trial in trials] == list(range(10)) and
                    component_summary(trials) == version['components'], '六分项原逐进程重聚合不符: ' + name)
        for a, b in zip(old['mechanism']['trials'], new['after']['trials']):
            require(a['trial'] == b['trial'] and a['raw_sha256'] == b['raw_sha256'] and
                    a['observed_last_cta'] == b['observed_last_cta'] and
                    a['last_in_predicted_ties'] == b['last_in_predicted_ties'] ==
                    (b['observed_last_cta'] in candidates), '关键CTA命中/旧观测身份不符: ' + name)
            critical_hits += b['last_in_predicted_ties']
        failed_attempts[name] = original['failed_attempts']
        require(len(original['failed_attempts']) == raw_cases[name]['failed_attempts'] ==
                report['observed']['failed_attempts'] and raw_cases[name]['selected_processes'] == {'plain':10,'dual':10},
                '原失败/成功选择计数不符: ' + name)
        rows.append(dict(case=name, before_plain_us=report['prediction']['plain_us'], after_plain_us=mapped['plain_us'],
            observed_plain_us=observed_plain, before_error_pct=old_error, after_error_pct=new_error,
            before_components=old['mechanism']['components'], after_components=new['after']['components'],
            predicted_critical_candidates=candidates,
            critical_cta_hits=sum(trial['last_in_predicted_ties'] for trial in new['after']['trials']),
            after_within_original_10pct=abs(new_error) <= policy['maximum_percent'],
            source_mode=new['mode'], failed_attempts=len(original['failed_attempts'])))
    responses = []
    for name, reference in RESPONSES:
        row, ref = by_after[name], by_after[reference]
        value = dict(case=name, reference=reference,
            observed_pct=100*(row['observed_plain_us']/ref['observed_plain_us']-1),
            before_pct=100*(row['before']['plain_us']/ref['before']['plain_us']-1),
            after_pct=100*(row['after']['plain_us']/ref['after']['plain_us']-1),
            predicted_absolute_difference_us=row['after']['plain_us']-ref['after']['plain_us'])
        require(value == next(saved for saved in after['responses'] if saved['case'] == name), '原干预响应不符')
        require(value['observed_pct'] == next(saved for saved in before['responses'] if saved['case'] == name)['observed_pct']
                and value['before_pct'] == next(saved for saved in before['responses'] if saved['case'] == name)['predicted_pct'],
                '原full-matrix响应不符')
        responses.append(value)
    metrics = dict(before_median=st.median(abs(row['before_error_pct']) for row in rows),
        before_max=max(abs(row['before_error_pct']) for row in rows),
        after_median=st.median(abs(row['after_error_pct']) for row in rows),
        after_max=max(abs(row['after_error_pct']) for row in rows))
    require(metrics == after['plain_absolute_error_pct'] and critical_hits == after['critical_cta_hits'] == 1,
            '原总体误差/关键CTA总命中不符')
    random = next(item for item in plan['rows'] if item['case']['id'] == 'v09d_cfg_b_G5_random_seed29')
    require(plan['random_frequency'] == '保留random行；不修改阶段模型支持域，显式采用V09统一阶段频率' and
            random['frequency_mode'] == '经验替代：V09调用频率用于所有阶段' and
            all(value == random['frequency_ghz'] for value in random['phase_frequency_ghz']), '原random fallback资格不符')
    return dict(identity=identities, build_identities=build_identities, rows=rows, responses=responses,
        plain_absolute_error_pct=metrics, critical_cta_hits=critical_hits, critical_cta_trials=120,
        component_process_scores_reaggregated=2*120*6, selected_successful_processes=240,
        original_failed_attempts=failed_attempts, preflight_paths=[str(path) for path in preflight],
        producer_prefix_ns=prefix_ns, formal_model_replays_in_original=5, derived_conditions_in_original=7,
        old_development_evidence_reused=True, full_combination_C2=False, full_combination_C3=False,
        original_thresholds_pct=dict(median=5, maximum=10),
        after_median_gate_passed=metrics['after_median'] <= 5, after_maximum_gate_passed=metrics['after_max'] <= 10,
        full_combination_passed=False, random_frequency_fallback=dict(case=random['case']['id'],
            qualification=random['frequency_mode'], frequency_ghz=random['frequency_ghz']),
        original_V09_global_status={key:score[key] for key in ('total_conditions','scored_predictions','plain_metrics','complete_time_passed')},
        qualification='旧开发评分及身份复核；保存plain重评分、逐进程六分项重聚合；逐CTA测量资格承接旧证据，不新增独立C2/C3',
        new_GPU_executions=0, new_model_replays=0, new_raw_decodes=0, new_fits=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    results, output = args.results_root.resolve(strict=True), args.output.resolve()
    require(all(not output.is_relative_to(results / root) for root in (FULL, PREFIX, V09)), '不能写入所读旧归档')
    output.mkdir(parents=True, exist_ok=False)
    try:
        evidence = replay(results)
        save(output / 'evidence.json', evidence)
        save(output / 'summary.json', dict(schema='gh200-cfg-b-old-score-replay/1', status='verified',
            replay_source_sha256=sha(Path(__file__)), conditions=len(evidence['rows']),
            plain_absolute_error_pct=evidence['plain_absolute_error_pct'], critical_cta_hits=evidence['critical_cta_hits'],
            critical_cta_trials=120, component_process_scores_reaggregated=evidence['component_process_scores_reaggregated'],
            selected_successful_processes=240,
            original_failed_attempts=sum(len(rows) for rows in evidence['original_failed_attempts'].values()),
            old_development_evidence_reused=True, full_combination_C2=False, full_combination_C3=False,
            original_thresholds_pct=evidence['original_thresholds_pct'],
            after_median_gate_passed=evidence['after_median_gate_passed'],
            after_maximum_gate_passed=evidence['after_maximum_gate_passed'], full_combination_passed=False,
            new_GPU_executions=0, new_model_replays=0, new_raw_decodes=0, new_fits=0,
            evidence_sha256=sha(output / 'evidence.json')))
    except Exception as error:
        save(output / 'failure.json', dict(status='failed', error=str(error)))
        raise
    print(f'已复核12行旧评分：{output / "summary.json"}')


if __name__ == '__main__':
    main()
