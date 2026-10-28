#!/usr/bin/env python3
"""回放旧 Q02 记录并检查短长度验证器；只在临时目录合成记录，不执行探针。"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import q02_tma_length as length
import q02_tma_stride as stride
from l2_saturation import digest, read_json, save_json


def frozen_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(folder):
    return [json.loads(line) for line in (folder/'stdout').read_text().splitlines()]


def write_rows(folder, task, root, rows):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'stdout').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (folder/'stderr').write_text('')
    receipt = dict(argv=stride.probe_command(root, task), returncode=0,
        timed_out=False, stdout_sha256=digest(folder/'stdout'), stderr_sha256=digest(folder/'stderr'))
    (folder/'receipt.json').write_text(json.dumps(receipt)+'\n')


def check_old_records(reference):
    """分别调用原冻结验证器与当前 stride；逐项比较已保存的原输出。"""
    uuid = read_json(reference/'plan.json')['gpu_uuid']
    old_stride = frozen_module('q02_length_original_stride', reference/'baseline/q02_tma_stride.py')
    replay = reference/'baseline/replay'
    old_processes = old_arms = 0
    for name, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
        for task in read_json(replay/name):
            folder = replay/'runs'/split/task['id']
            before = old_stride.validate_process(replay, folder, task, uuid)
            after = stride.validate_process(replay, folder, task, uuid)
            assert before == after == read_json(folder/'validated.json')
            old_processes += 1
            old_arms += len(after)
    assert (old_processes, old_arms) == (12, 36)

    old_single = frozen_module('q02_length_original_single', reference/'source/q02_tma_single_cta.py')
    old_single.stride = frozen_module('q02_length_single_stride', reference/'source/q02_tma_stride.py')
    single_processes = single_arms = 0
    for name, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
        for task in read_json(reference/name):
            folder = reference/'runs'/split/task['id']
            before = old_single.validate_pair(reference, folder, task, uuid)
            after = stride.validate_process(reference, folder, task, uuid)
            assert before == after == read_json(folder/'validated.json')
            assert all('groups' not in row for row in after)
            single_processes += 1
            single_arms += len(after)
    assert (single_processes, single_arms) == (6, 12)
    return dict(old_processes=old_processes, old_arms=old_arms,
                single_processes=single_processes, single_arms=single_arms)


def make_fixture(root, fixture, uuid):
    """冻结输入只读链接到原包；所有可写合成记录位于新 runs/allocation。"""
    fixture.mkdir()
    for source in root.iterdir():
        if source.name not in ('runs', 'allocation', 'score.json'):
            (fixture/source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    (fixture/'allocation').mkdir()
    save_json(fixture/'allocation/identity.json', dict(uuid=uuid))


def source_rows(reference, task, split):
    name = 'correctness_commands.json' if split == 'correctness' else 'commands.json'
    original_task = next(item for item in read_json(reference/name) if item['seed'] == task['seed'])
    rows = read_rows(reference/'runs'/split/original_task['id'])
    return {row['ldb']:row for row in rows}


def synthesize(root, reference, task, split, delta_cycles):
    originals = source_rows(reference, task, split)
    rows = []
    for index, ldb in enumerate(task['paired_ldbs']):
        row = copy.deepcopy(originals[ldb])
        row.update(groups=task['groups'], warmup_groups=64, pair_index=index,
                   paired_ldbs=task['paired_ldbs'],
                   checked=task['groups']*16384 if row['verify'] else 0)
        cta = row['ctas_data'][0]
        base_cycles = task['groups']*1000
        cycles = base_cycles + (delta_cycles if ldb == 2344 else 0)
        cta['end'] = [cta['begin'][0]+cycles, cta['begin'][1]+cycles//2]
        rows.append(row)
    folder = root/'runs'/split/task['id']
    write_rows(folder, task, root, rows)
    return folder, rows


def check_rejections(root, uuid):
    task = read_json(root/'correctness_commands.json')[0]
    folder = root/'runs/correctness'/task['id']
    original = read_rows(folder)
    checks = []
    for label in ('checked', 'groups', 'warmup_groups', 'same_wrong_sm'):
        rows = copy.deepcopy(original)
        if label == 'checked':
            rows[0]['checked'] += 16384
        elif label == 'groups':
            rows[0]['groups'] = 64
        elif label == 'warmup_groups':
            rows[0]['warmup_groups'] = task['groups']
        else:
            for row in rows:
                row['ctas_data'][0]['smid'] = 125
        write_rows(folder, task, root, rows)
        try:
            length.validate_pair(root, folder, task, uuid)
        except ValueError as error:
            checks.append(dict(case=label, rejected=True, reason=str(error)))
        else:
            raise AssertionError('接受了错误记录：'+label)
    write_rows(folder, task, root, original)
    return checks


def check_scores(root, reference):
    hypotheses = {row['id']:row['delta_cycles'] for row in read_json(root/'hypotheses.json')}
    assert hypotheses == {'fixed-boundary':{'16':295.0, '32':295.0},
                          'per-group':{'16':73.75, '32':147.5}}
    scenarios = [('fixed-boundary', {16:295, 32:295}),
                 ('per-group', {16:74, 32:148}),
                 ('neither', {16:1000, 32:1000})]
    scores = []
    for supported, deltas in scenarios:
        for task in read_json(root/'commands.json'):
            synthesize(root, reference, task, 'formal', deltas[task['groups']])
        report = length.summarize(root)
        assert report['numeric_arm_checks'] == report['formal_arm_observations'] == 12
        assert len(report['pairs']) == 6 and len(report['conditions']) == 4
        assert report['heldout_conditions'] == 4
        assert report['stable_windows'] is True and report['all_pair_deltas_positive'] is True
        expected_support = [] if supported == 'neither' else [supported]
        assert report['supported_candidates'] == expected_support
        assert report['independent_C2'] is bool(expected_support)
        for condition in report['conditions']:
            assert condition['passed'] is True
            for window in condition['windows'].values():
                assert window['spread'] == window['relative_spread'] == 0
                assert window['passed'] is True
        for measured in report['lengths']:
            assert measured['delta_cycles'] == [deltas[measured['groups']]]*3
            assert measured['median_delta_cycles'] == deltas[measured['groups']]
        for candidate in report['candidates']:
            expected_pass = candidate['id'] == supported
            assert candidate['passed'] is expected_pass
            assert candidate['conclusion'] == ('获得支持' if expected_pass else '被反证')
            for prediction in candidate['predictions']:
                groups = prediction['groups']
                expected = hypotheses[candidate['id']][str(groups)]
                error = abs(deltas[groups]-expected)/expected
                assert prediction['predicted_delta_cycles'] == expected
                assert prediction['observed_median_delta_cycles'] == deltas[groups]
                assert prediction['relative_error'] == error
                assert prediction['passed'] is (error <= 0.05)
        scores.append(dict(case=supported, supported_candidates=report['supported_candidates'],
                           independent_C2=report['independent_C2'], candidates=report['candidates']))

    # 保持每对增量为 295，仅让一个 seed 的两臂一起变慢，检查窗口资格独立生效。
    for task in read_json(root/'commands.json'):
        folder, rows = synthesize(root, reference, task, 'formal', 295)
        if task['seed'] == 41:
            for row in rows:
                cta = row['ctas_data'][0]
                cta['end'][0] += task['groups']*100
                cta['end'][1] += task['groups']*50
            write_rows(folder, task, root, rows)
    report = length.summarize(root)
    assert report['stable_windows'] is False and report['all_pair_deltas_positive'] is True
    assert report['supported_candidates'] == [] and report['independent_C2'] is False
    assert all(candidate['conclusion'] == '信息不足' for candidate in report['candidates'])
    fixed = next(candidate for candidate in report['candidates'] if candidate['id'] == 'fixed-boundary')
    assert all(prediction['passed'] for prediction in fixed['predictions'])
    scores.append(dict(case='unstable_windows', supported_candidates=[], independent_C2=False))
    return scores


def check(root, reference):
    counts = check_old_records(reference)
    plan = length.load_plan(root)
    uuid = plan['gpu_uuid']
    with tempfile.TemporaryDirectory(prefix='q02-length-cpu-') as temporary:
        fixture = Path(temporary)/'fixture'
        make_fixture(root, fixture, uuid)
        processes = arms = 0
        for name, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
            for task in read_json(fixture/name):
                assert task['groups'] in (16, 32)
                argv = task['argv']
                assert argv[argv.index('--groups')+1] == '64'
                assert int(argv[argv.index('--measure-groups')+1]) == task['groups']
                folder, rows = synthesize(fixture, reference, task, split, 295)
                checked = length.validate_pair(fixture, folder, task, uuid)
                assert len(checked) == 2
                assert all(row['groups'] == task['groups'] for row in checked)
                assert [row['checked_values'] for row in checked] == [row['checked'] for row in rows]
                processes += 1
                arms += len(checked)
        assert (processes, arms) == (12, 24)
        rejections = check_rejections(fixture, uuid)
        scores = check_scores(fixture, reference)
    return dict(schema='gh200-q02-tma-length-cpu-check/1', **counts,
                synthetic_processes=processes, synthetic_arms=arms,
                rejections=rejections, scores=scores, gpu_programs_executed=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    reference = (args.reference or root.parents[2]/'Q02-tma-request/request-shape-response/20261018-Q02-tma-single-cta-001').resolve()
    print(json.dumps(check(root, reference), ensure_ascii=False, indent=2))
