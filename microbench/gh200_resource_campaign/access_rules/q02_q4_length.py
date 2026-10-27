"""冻结REG52 tensor probe：Q4三个长度的两臂plain验证。"""
import argparse
from pathlib import Path
import statistics
import time

from l2_saturation import (command_record, exclusive_lock, inspect_allocated_gpu,
                           read_json, require, save_json)
from q02_q4_stride import (FLAGS, probe_command, relative_check, sha, spread,
                          validate_process as validate_stride_process,
                          validate_rows as validate_stride_rows)


SCHEMA = 'gh200-q02-q4-length-heldout/1'
GROUPS = (16, 32, 64)
SEEDS = (3, 22, 41)
NATIVE = '6843b28f37c885459c51d7e51dc6c574a8c27f433ca475ecbb341be25a937b61'
TOTAL_CHECKED = 11010048


def validation_plan(plan, groups):
    require(type(groups) is int and groups in GROUPS, '长度不在固定范围')
    view = dict(plan)
    view['processes'] = dict(plan['processes'], checked_per_verify_arm=16384*groups)
    return view


def validate_rows(raw, task, plan, source):
    groups = task['G']
    checked = validate_stride_rows(raw, task, validation_plan(plan, groups), source)
    for row in checked:
        row['G'] = groups
    return checked


def validate_process(root, task, plan, freeze):
    groups = task['G']
    checked = validate_stride_process(root, task, validation_plan(plan, groups), freeze)
    for row in checked:
        row['G'] = groups
    return checked


def load_inputs(root):
    binding = read_json(root/'binding.json')
    for name, expected in binding['files'].items():
        require(sha(root/name) == expected, '冻结输入变化：'+name)
    require(sha(__file__) == binding['files']['source/q02_q4_length.py'], '执行脚本不是冻结版本')
    plan, commands = read_json(root/'plan.json'), read_json(root/'commands.json')
    freeze, predictions = read_json(root/'freeze.json'), read_json(root/'predictions.json')
    require(plan['schema'] == SCHEMA and plan['groups'] == list(GROUPS) and
            plan['seeds'] == list(SEEDS), '登记范围不符')
    require(plan['processes'] == dict(verify=9, formal=9, arms_per_process=2,
                                     total_checked=TOTAL_CHECKED), '数值/正式进程预算不符')
    require(sha(root/'binding.json') == freeze['binding_sha256'], '冻结binding身份不符')
    for name in ('plan', 'commands', 'predictions'):
        require(sha(root/(name+'.json')) == freeze[name+'_sha256'], '冻结记录不符：'+name)
    require(sha(root/'build/probe') == freeze['native_sha256'] == plan['native_sha256'] == NATIVE,
            '保存REG52 native变化')
    require((root/'build/probe').stat().st_mode & 0o111, '保存binary无执行位')
    require((root/'worklist.txt').read_text().split() ==
            ['2304', '2304', '4096', '1', '0', '0', '7', '6'], '工作列表不符')
    expected = [(role, groups, seed, verify) for role, verify in (('verify', 1), ('formal', 0))
                for groups in (32, 16, 64) for seed in SEEDS]
    require([(task['role'], task['G'], task['seed'], task['verify']) for task in commands] == expected,
            '18进程清单或先数值后正式顺序不符')
    require(len({(task['role'], task['id']) for task in commands}) == 18, '进程目录重复')
    for task in commands:
        args = dict(zip(task['argv'][1::2], task['argv'][2::2]))
        groups = task['G']
        strides = [[4096, 2304], [4104, 2304]]
        if task['seed'] != 3:
            strides.reverse()
        require(task['groups'] == groups and task['Q'] == task['active_groups'] == 4 and
                task['paired_strides'] == strides and task['paired_request_masks'] == [7, 7] and
                task['paired_traces'] == task['paired_wait_delays'] == [0] and
                task['pair_indices'] == [0, 1], '配对清单不符')
        require(args['--paired-strides'] == ','.join(f'{a}:{b}' for a, b in strides) and
                args['--paired-request-masks'] == '7,7' and
                args['--paired-traces'] == args['--paired-wait-delays'] == '0', '配对命令不符')
        fixed = {'--ctas': 1, '--active': 4, '--groups': 64, '--ring': 64,
                 '--verify': task['verify'], '--trace': 0, '--seed': task['seed'],
                 '--warmup': 10, '--source-phase-stride': 0, '--source-tile-shift': 0,
                 '--work-index': 1, '--warmup-work-offset': 0, '--release-delay-ns': 0,
                 '--pollute-kib': 0, '--launch-skew-ns': 0, '--warmup-skew-ns': 0,
                 '--lda': 4096, '--ldb': 2304, '--alloc-lda': 4104, '--alloc-ldb': 2360,
                 '--measure-groups': groups, '--measure-active': 4, '--wait-delay-cycles': 0,
                 '--request-mask': 7}
        require(all(int(args[key]) == value for key, value in fixed.items()), '固定命令变化')
    require(set(predictions['windows_cycles']) == set(predictions['paired_cycles']) == {'16', '32', '64'}
            and all(set(value) == {'4096', '4104'} for value in predictions['windows_cycles'].values())
            and set(predictions['length_delta_cycles']) == {'4096', '4104'}, '冻结预测条件不完整')
    return plan, commands, freeze, predictions


def summarize(plan, predictions, rows):
    numeric = [row for row in rows if row['role'] == 'verify']
    formal = [row for row in rows if row['role'] == 'formal']
    expected = {(groups, seed, lda) for groups in GROUPS for seed in SEEDS for lda in (4096, 4104)}
    require(len(numeric) == len(formal) == 18 and
            {(row['G'], row['seed'], row['lda']) for row in numeric} == expected and
            {(row['G'], row['seed'], row['lda']) for row in formal} == expected,
            '三长度数值/正式两臂不完整')
    require(sum(row['checked_values'] for row in numeric) == TOTAL_CHECKED and
            all(row['checked_values'] == 0 for row in formal), '完整数值检查总量不符')
    by_condition = {(row['G'], row['seed'], row['lda']):row for row in formal}
    gates = plan['gates']
    windows, paired = {}, {}
    for groups in GROUPS:
        windows[str(groups)] = {str(lda):{unit:spread([
            by_condition[groups, seed, lda]['metrics'][unit] for seed in SEEDS],
            gates['window_relative_spread']) for unit in ('cycles', 'ns')} for lda in (4096, 4104)}
        differences = {unit:[by_condition[groups, seed, 4104]['metrics'][unit]-
                             by_condition[groups, seed, 4096]['metrics'][unit] for seed in SEEDS]
                       for unit in ('cycles', 'ns')}
        paired[str(groups)] = spread(differences['cycles'], gates['paired_cycle_relative_spread'])
        paired[str(groups)].update(ns_values=differences['ns'], ns_median=statistics.median(differences['ns']))
    bridge = {lda:{unit:relative_check(windows['32'][lda][unit]['median'], reference[unit],
                                     gates['reference_relative_error']) for unit in ('cycles', 'ns')}
              for lda, reference in plan['reference_windows'].items()}
    pair_bridge = {unit:relative_check(paired['32']['median'] if unit == 'cycles' else paired['32']['ns_median'],
                                      plan['reference_pair'][unit], gates['reference_relative_error'])
                   for unit in ('cycles', 'ns')}
    lengths = {}
    for lda in (4096, 4104):
        differences = {unit:[by_condition[64, seed, lda]['metrics'][unit]-
                             by_condition[16, seed, lda]['metrics'][unit] for seed in SEEDS]
                       for unit in ('cycles', 'ns')}
        lengths[str(lda)] = {unit:dict(values=values, median=statistics.median(values))
                            for unit, values in differences.items()}
    double_difference = {unit:dict(values=[
        (by_condition[64, seed, 4104]['metrics'][unit]-by_condition[64, seed, 4096]['metrics'][unit])-
        (by_condition[16, seed, 4104]['metrics'][unit]-by_condition[16, seed, 4096]['metrics'][unit])
        for seed in SEEDS]) for unit in ('cycles', 'ns')}
    for value in double_difference.values():
        value['median'] = statistics.median(value['values'])
    qualified = (all(value['passed'] for group in windows.values() for window in group.values()
                     for value in window.values()) and all(value['passed'] for value in paired.values()) and
                 all(value['passed'] for window in bridge.values() for value in window.values()) and
                 all(value['passed'] for value in pair_bridge.values()))
    checks = None
    if qualified:
        tolerance = gates['prediction_denominator_relative_error']
        checks = {}
        for groups in (16, 64):
            for lda in ('4096', '4104'):
                checks[f'G{groups}_A{lda}_full_cycles'] = relative_check(
                    windows[str(groups)][lda]['cycles']['median'], predictions['windows_cycles'][str(groups)][lda], tolerance)
        for groups in GROUPS:
            checks[f'G{groups}_paired_cycles'] = relative_check(paired[str(groups)]['median'],
                                                               predictions['paired_cycles'][str(groups)], tolerance)
        for lda in ('4096', '4104'):
            checks[f'A{lda}_G64_minus_G16_cycles'] = relative_check(lengths[lda]['cycles']['median'],
                                                                  predictions['length_delta_cycles'][lda], tolerance)
    passed = qualified and all(value['passed'] for value in checks.values())
    decision = plan['decisions']['qualification_failure'] if not qualified else (
        plan['decisions']['passed'] if passed else plan['decisions']['prediction_failure'])
    return dict(schema=SCHEMA, decision=decision, observation_qualified=qualified, passed=passed,
                qualification=plan['qualification'], numeric=numeric, formal=formal, total_checked=TOTAL_CHECKED,
                windows=windows, paired=paired, G32_bridge=bridge, G32_pair_bridge=pair_bridge,
                length_differences=lengths, double_difference_report_only=double_difference,
                model_checks=checks, model_check_count=len(checks) if checks is not None else 0,
                prediction_unit='CTA clock64 cycle', ns_role='repeatability and G32 bridge only; no cycle-model ns prediction',
                gates=gates, **FLAGS)


def score(root):
    plan, tasks, freeze, predictions = load_inputs(root)
    rows = []
    for task in tasks:
        rows.extend(validate_process(root, task, plan, freeze))
    return summarize(plan, predictions, rows)


def execute(root, output=None):
    plan, tasks, freeze, predictions = load_inputs(root)
    require(not (root/'runs'/'heldout').exists(), '本轮已有进程记录，不自动重采')
    destination = output if output is not None else root/'score-heldout.json'
    start = time.monotonic()
    rows = []
    try:
        environment = inspect_allocated_gpu(root/'allocation'/'heldout')
        require(environment['uuid'] == plan['gpu_uuid'], '分配卡不是冻结参考卡')
        with exclusive_lock(Path('/tmp')/f'gh200-measurement-{plan["gpu_uuid"]}.lock'):
            for task in tasks:
                if task['role'] == 'formal':
                    require(sum(row['checked_values'] for row in rows if row['role'] == 'verify') == TOTAL_CHECKED,
                            '完整数值检查尚未通过，不执行正式采样')
                remaining = 480.-(time.monotonic()-start)
                require(remaining > 0, '完整轮480秒预算耗尽，不补跑')
                folder = root/'runs'/'heldout'/task['role']/task['id']
                command_record(folder, probe_command(root, task), min(60., remaining))
                checked = validate_process(root, task, plan, freeze)
                rows.extend(checked)
                save_json(folder/'validated.json', checked)
                print(f'VALID {task["role"]} G{task["G"]} seed{task["seed"]}:2 arms', flush=True)
        report = summarize(plan, predictions, rows)
    except Exception as error:
        save_json(destination, dict(schema=SCHEMA, decision=plan['decisions']['qualification_failure'],
            observation_qualified=False, passed=False, error=str(error), completed_rows=rows, **FLAGS))
        raise
    save_json(destination, report)
    print(f'{report["decision"]}: passed={report["passed"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('execute', 'score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'execute':
        execute(root, args.output)
    else:
        require(args.output is not None, 'score需要--output')
        save_json(args.output, score(root))


if __name__ == '__main__':
    main()
