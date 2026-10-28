#!/usr/bin/env python3
"""CPU 回放与合成检查；只在临时目录写记录，不执行探针或构建命令。"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import q02_tma_active as active
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
    receipt = dict(argv=stride.probe_command(root, task), returncode=0, timed_out=False,
        stdout_sha256=digest(folder/'stdout'), stderr_sha256=digest(folder/'stderr'))
    (folder/'receipt.json').write_text(json.dumps(receipt)+'\n')


def check_replay(single_reference, length_reference):
    original_stride = frozen_module('q02_active_original_stride', single_reference/'baseline/q02_tma_stride.py')
    original_single = frozen_module('q02_active_original_single', single_reference/'source/q02_tma_single_cta.py')
    original_single.stride = frozen_module('q02_active_single_stride', single_reference/'source/q02_tma_stride.py')
    original_length_stride = frozen_module('q02_active_length_stride', length_reference/'source/q02_tma_stride.py')
    counts = {}
    for label, root, validator, expected in (
            ('original', single_reference/'baseline/replay', original_stride.validate_process, (12, 36)),
            ('single', single_reference, original_single.validate_pair, (6, 12)),
            ('length', length_reference, original_length_stride.validate_process, (12, 24))):
        uuid = read_json(single_reference/'plan.json')['gpu_uuid']
        processes = arms = 0
        for filename, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
            for task in read_json(root/filename):
                folder = root/'runs'/split/task['id']
                before = validator(root, folder, task, uuid)
                after = stride.validate_process(root, folder, task, uuid)
                assert before == after == read_json(folder/'validated.json')
                assert all('groups' not in row and 'warmup_active_groups' not in row for row in after)
                processes += 1
                arms += len(after)
        assert (processes, arms) == expected
        counts[label] = dict(processes=processes, arms=arms, exact_saved_output=True)
    return counts


def check_source(root, reference):
    source = (root/'source/tma_gemm_sources.cu').read_text()
    before = (root/'source-before/tma_gemm_sources.cu').read_text()
    prior = (reference/'source/tma_gemm_sources.cu').read_text()
    def kernel(text):
        return text[text.index('__global__ void tensor_probe'):text.index('static std::string descriptor_hex')]
    assert kernel(source) == kernel(before) == kernel(prior)
    assert '&iterations,&active,&verify' in source
    configured = source.index('configured_active=active')
    pair = source.index('for (size_t pair_index=0;')
    reset = source.index('active=configured_active;', pair)
    recorded = source.index('warmup_active_groups=active;', pair)
    warmup = source.index('for (int i=0; i<warmup;', pair)
    restored_work = source.index('work_index=measured_work_index;', warmup)
    measured = source.index('if (measure_active) active=measure_active;', restored_work)
    launch = source.index('launch();', measured)
    assert configured < pair < reset < recorded < warmup < restored_work < measured < launch
    assert source.count('active=configured_active;') == 1
    assert source.count('if (measure_active) active=measure_active;') == 1
    assert 'if (measure_active) std::cout<<",\\\"warmup_active_groups\\\":"<<warmup_active_groups;' in source
    return dict(kernel_body_exact=True, launch_uses_current_active=True,
                paired_warmup_reset_before_override=True, warmup_field_conditional=True)


def make_fixture(root, fixture, uuid):
    fixture.mkdir()
    for source in root.iterdir():
        if source.name not in ('build', 'runs', 'allocation', 'score.json'):
            (fixture/source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    # 只回放原构建记录供 CPU 校验器读取；不代表新 host 已在目标编译。
    shutil.copytree(root/'baseline/build', fixture/'build')
    (fixture/'build/target-review.json').write_text(json.dumps(active.target_review(fixture))+'\n')
    (fixture/'allocation').mkdir()
    save_json(fixture/'allocation/identity.json', dict(uuid=uuid))


def synthesize(root, reference, task, split, delta_cycles):
    filename = 'correctness_commands.json' if split == 'correctness' else 'commands.json'
    original_task = next(item for item in read_json(reference/filename)
                         if item['seed'] == task['seed'] and item['groups'] == 32)
    originals = {row['ldb']:row for row in read_rows(reference/'runs'/split/original_task['id'])}
    rows = []
    for index, ldb in enumerate(task['paired_ldbs']):
        row = copy.deepcopy(originals[ldb])
        row.update(active_groups=task['active_groups'], groups=32, warmup_active_groups=4,
                   warmup_groups=64, pair_index=index, paired_ldbs=task['paired_ldbs'],
                   checked=32*16384 if row['verify'] else 0)
        cta = row['ctas_data'][0]
        cycles = 32000 + (delta_cycles if ldb == 2344 else 0)
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
    for label in ('formal_active', 'warmup_active', 'groups', 'same_wrong_sm', 'checked'):
        rows = copy.deepcopy(original)
        if label == 'formal_active':
            rows[0]['active_groups'] = 4 if task['active_groups'] == 1 else 1
        elif label == 'warmup_active':
            rows[0]['warmup_active_groups'] = 1
        elif label == 'groups':
            rows[0]['groups'] = 64
        elif label == 'same_wrong_sm':
            for row in rows:
                row['ctas_data'][0]['smid'] = 125
        else:
            rows[0]['checked'] += 16384
        write_rows(folder, task, root, rows)
        try:
            active.validate_pair(root, folder, task, uuid)
        except ValueError as error:
            checks.append(dict(case=label, rejected=True, reason=str(error)))
        else:
            raise AssertionError('接受了错误记录：'+label)
    write_rows(folder, task, root, original)
    return checks


def check_scores(root, reference):
    hypotheses = {row['id']:row['delta_cycles'] for row in read_json(root/'hypotheses.json')}
    assert hypotheses == {'fixed-boundary':{'1':295.0, '4':295.0},
                          'per-group-ready':{'1':9440.0, '4':295.0}}
    scenarios = [('fixed-boundary', {1:295, 4:295}),
                 ('per-group-ready', {1:9440, 4:295}),
                 ('neither', {1:1000, 4:295})]
    scores = []
    for supported, deltas in scenarios:
        for task in read_json(root/'commands.json'):
            synthesize(root, reference, task, 'formal', deltas[task['active_groups']])
        report = active.summarize(root)
        assert report['numeric_arm_checks'] == report['formal_arm_observations'] == 12
        assert len(report['pairs']) == 6 and len(report['conditions']) == 4
        assert report['development_conditions'] == report['heldout_conditions'] == 2
        assert report['full_gemm_C2'] is report['full_gemm_C3'] is False
        assert report['stable_windows'] is report['all_pair_deltas_positive'] is True
        expected_support = [] if supported == 'neither' else [supported]
        assert report['supported_candidates'] == expected_support
        assert report['independent_C2'] is bool(expected_support)
        for condition in report['conditions']:
            assert condition['passed'] is True
            assert condition['role'] == ('new_intervention' if condition['active_groups'] == 1 else 'control')
            for window in condition['windows'].values():
                assert window['spread'] == window['relative_spread'] == 0
                assert window['passed'] is True
        for response in report['responses']:
            assert response['delta_cycles'] == [deltas[response['active_groups']]]*3
            assert response['median_delta_cycles'] == deltas[response['active_groups']]
        for candidate in report['candidates']:
            expected_pass = candidate['id'] == supported
            assert candidate['passed'] is expected_pass
            assert candidate['conclusion'] == ('获得支持' if expected_pass else '被反证')
            for prediction in candidate['predictions']:
                q = prediction['active_groups']
                expected = hypotheses[candidate['id']][str(q)]
                error = abs(deltas[q]-expected)/expected
                assert prediction['predicted_delta_cycles'] == expected
                assert prediction['observed_median_delta_cycles'] == deltas[q]
                assert prediction['relative_error'] == error
                assert prediction['passed'] is (error <= 0.05)
        scores.append(dict(case=supported, supported_candidates=expected_support,
                           simulated_independent_C2=report['independent_C2']))

    for case in ('unstable_cycles', 'unstable_ns', 'nonpositive_pair', 'q4_control_mismatch'):
        for task in read_json(root/'commands.json'):
            delta = 1000 if case == 'q4_control_mismatch' and task['active_groups'] == 4 else 295
            if case == 'nonpositive_pair' and task['seed'] == 41 and task['active_groups'] == 1:
                delta = -1
            folder, rows = synthesize(root, reference, task, 'formal', delta)
            if case.startswith('unstable_') and task['seed'] == 41 and task['active_groups'] == 1:
                unit, amount = (0, 3200) if case == 'unstable_cycles' else (1, 1600)
                for row in rows:
                    row['ctas_data'][0]['end'][unit] += amount
                write_rows(folder, task, root, rows)
        report = active.summarize(root)
        unstable = case.startswith('unstable_')
        assert report['stable_windows'] is (not unstable)
        assert report['all_pair_deltas_positive'] is (case != 'nonpositive_pair')
        assert report['supported_candidates'] == [] and report['independent_C2'] is False
        fixed = next(candidate for candidate in report['candidates'] if candidate['id'] == 'fixed-boundary')
        if case != 'q4_control_mismatch':
            assert all(prediction['passed'] for prediction in fixed['predictions'])
        assert all(candidate['conclusion'] == ('信息不足' if unstable else '被反证')
                   for candidate in report['candidates'])
        scores.append(dict(case=case, stable_windows=report['stable_windows'],
                           all_pair_deltas_positive=report['all_pair_deltas_positive'],
                           supported_candidates=[], simulated_independent_C2=False))
    return scores


def check(root, reference, single_reference):
    replay = check_replay(single_reference, reference)
    source = check_source(root, reference)
    plan = active.load_plan(root)
    assert plan['groups'] == 32 and plan['warmup_groups'] == 64 and plan['warmup_active_groups'] == 4
    assert plan['required_smid'] == 124 and plan['active_groups'] == [1, 4]
    with tempfile.TemporaryDirectory(prefix='q02-active-cpu-') as temporary:
        fixture = Path(temporary)/'fixture'
        make_fixture(root, fixture, plan['gpu_uuid'])
        processes = arms = 0
        conditions = set()
        for filename, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
            for task in read_json(fixture/filename):
                argv = dict(zip(task['argv'][1::2], task['argv'][2::2]))
                assert task['groups'] == 32 and task['active_groups'] in (1, 4)
                assert argv['--active'] == '4' and argv['--measure-active'] == str(task['active_groups'])
                assert argv['--groups'] == '64' and argv['--measure-groups'] == '32'
                assert argv['--warmup'] == '10' and argv['--trace'] == '0' and argv['--ctas'] == '1'
                folder, rows = synthesize(fixture, reference, task, split, 295)
                checked = active.validate_pair(fixture, folder, task, plan['gpu_uuid'])
                assert len(checked) == 2 and all(row['active_groups'] == task['active_groups'] for row in checked)
                assert [row['checked_values'] for row in checked] == [row['checked'] for row in rows]
                assert all(row['ctas'][0]['smid'] == 124 for row in checked)
                conditions.update((task['active_groups'], ldb) for ldb in task['paired_ldbs'])
                processes += 1
                arms += len(checked)
        assert (processes, arms) == (12, 24)
        assert conditions == {(1,2304), (1,2344), (4,2304), (4,2344)}
        rejections = check_rejections(fixture, plan['gpu_uuid'])
        scores = check_scores(fixture, reference)
    return dict(schema='gh200-q02-tma-active-cpu-check/1', cpu_simulation_only=True,
                replay=replay, source=source, synthetic_processes=processes, synthetic_arms=arms,
                conditions=sorted(conditions), rejections=rejections, simulated_scores=scores,
                gpu_programs_executed=0, compiler_programs_executed=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--single-reference', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    reference = (args.reference or root.parents[2]/'Q02-tma-request/request-shape-response/20261018-Q02-tma-length-001').resolve()
    single_reference = (args.single_reference or root.parents[2]/'Q02-tma-request/request-shape-response/20261018-Q02-tma-single-cta-001').resolve()
    print(json.dumps(check(root, reference, single_reference), ensure_ascii=False, indent=2))
