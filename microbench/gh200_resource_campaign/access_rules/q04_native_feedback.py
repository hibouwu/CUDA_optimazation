#!/usr/bin/env python3
"""原生驻留短链的固定反馈假设、冻结外推和同批对照。"""
import argparse
from pathlib import Path
import statistics
import time

from cfg_b_handoff_chain import timed_lifecycle, probe_command
from q04_handoff_wait import validate
from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json


def predict(root):
    plan = read_json(root/'plan.json')
    index = read_json(root/'calibration/index-score.json')
    length = read_json(root/'calibration/length-score.json')
    old = read_json(root/'calibration/zero-feedback-report.json')
    old_windows = {str(delay):{unit:next(row for row in index['conditions'] if
        row['case_id'] == f's6_k13_incremental_d{delay}')[unit]['median'] for unit in ('cycles', 'ns')}
        for delay in (0, 4096)}
    old_response = {unit:statistics.median(row[unit]['incremental_response'] for row in index['pairs'])
                    for unit in ('cycles', 'ns')}
    feedback = old_response['cycles']-old['comparisons'][0]['delay_response']['predicted_cycle']
    k4 = next(row for row in length['conditions'] if row['case_id'] == 's6_k4_incremental_d0')['cycles']['median']
    models, calls = {}, 0
    for name, parameter in (('empty_feedback', 'release_visibility_cycles'),
                            ('blocked_full_return', 'full_blocked_return_cycles')):
        timelines = {}
        for case in plan['cases']:
            timelines[case['id']] = timed_lifecycle(case['ktiles'], case['stages'], case['delay_cycles'],
                                                    **{parameter:feedback})
            calls += 1
        fixed = old_windows['0']['cycles']-timelines['control_d0']['local_end_cycle']
        predictions = {key:dict(window_cycle=value['local_end_cycle']+fixed, timeline=value)
                        for key,value in timelines.items()}
        short = timed_lifecycle(4, 6, 0, **{parameter:feedback})['local_end_cycle']+fixed
        calls += 1
        delta = predictions['control_d4096']['window_cycle']-predictions['control_d0']['window_cycle']
        old_checks = dict(k4_window_error=abs(short-k4)/k4,
            k13_window_error=abs(predictions['control_d0']['window_cycle']-old_windows['0']['cycles'])/old_windows['0']['cycles'],
            delay_response_error=abs(delta-old_response['cycles'])/old_response['cycles'])
        models[name] = dict(parameters={parameter:feedback, 'compute_cycles':512,
            'commit_lead_cycles':256, 'completion_lag_cycles':0}, fixed_window_cycle=fixed,
            predictions=predictions, old_checks=old_checks, old_constraints_passed=all(v <= .05 for v in old_checks.values()))
    first, second = models.values()
    equal = all(first['predictions'][case['id']]['window_cycle'] == second['predictions'][case['id']]['window_cycle']
                for case in plan['cases'])
    return dict(unit='SM clock64 cycle', feedback_cycle=feedback, models=models,
        model_calls=calls, numerical_parameter_searches=0, old_windows=old_windows, old_response=old_response,
        placements_observationally_equal=equal, cpu_gate=all(model['old_constraints_passed'] for model in models.values()),
        qualification='经验校准的局部反馈假设；位置未辨识，后续冻结cycle外推',
        calibration_sha256={str(path.relative_to(root)):digest(path) for path in sorted((root/'calibration').iterdir())})


def load(root):
    for name, checksum in read_json(root/'execution-binding.json')['files'].items():
        require(digest(root/name) == checksum, '冻结输入变化：'+name)
    plan, freeze = read_json(root/'plan.json'), read_json(root/'freeze.json')
    require(freeze['cpu_gate'] and digest(root/'reference/probe') == plan['probe_sha256'], '预测或原binary未通过')
    return plan, freeze


def control_checks(rows, freeze):
    checks = []
    for delay in (0, 4096):
        group = [row for row in rows if row['case_id'] == f'control_d{delay}']
        for unit in ('cycles', 'ns'):
            values = [row[unit] for row in group]
            observed = statistics.median(values)
            expected = freeze['old_windows'][str(delay)][unit]
            checks.append(dict(kind='control_window', delay=delay, unit=unit, observed=observed, expected=expected,
                passed=abs(observed-expected) <= .05*expected and max(values)-min(values) <= .05*observed))
    for unit in ('cycles', 'ns'):
        pairs = [next(row[unit] for row in rows if row['case_id'] == 'control_d4096' and row['seed'] == seed)-
                 next(row[unit] for row in rows if row['case_id'] == 'control_d0' and row['seed'] == seed)
                 for seed in (3, 22, 41)]
        observed = statistics.median(pairs)
        expected = freeze['old_response'][unit]
        checks.append(dict(kind='control_response', unit=unit, observed=observed, expected=expected,
                           passed=abs(observed-expected) <= .05*abs(expected)))
    return checks


def summarize(root):
    plan, freeze = load(root)
    rows = [validate(root/'runs'/task['case_id']/f"seed{task['seed']}",
        next(case for case in plan['cases'] if case['id'] == task['case_id']),task['seed'],plan['gpu_uuid'])
        for task in plan['order']]
    bridge = control_checks([row for row in rows if row['case_id'].startswith('control_')], freeze)
    stability, medians = [], {}
    for case in plan['cases']:
        group = [row for row in rows if row['case_id'] == case['id']]
        medians[case['id']] = statistics.median(row['cycles'] for row in group)
        for unit in ('cycles', 'ns'):
            values = [row[unit] for row in group]
            median = statistics.median(values)
            stability.append(dict(case_id=case['id'], unit=unit, median=median,
                relative_spread=(max(values)-min(values))/median, passed=max(values)-min(values) <= .05*median))
    qualified = all(check['passed'] for check in bridge+stability)
    models = {}
    for name, model in freeze['models'].items():
        predicted = {key:value['window_cycle'] for key,value in model['predictions'].items()}
        windows = [dict(case_id=case['id'], observed=medians[case['id']], predicted=predicted[case['id']],
            relative_error=abs(medians[case['id']]-predicted[case['id']])/predicted[case['id']],
            passed=abs(medians[case['id']]-predicted[case['id']]) <= .05*predicted[case['id']])
            for case in plan['cases'] if case['split'] == 'heldout']
        responses = []
        for label, after, before in (('stage4_delay','stage4_d4096','stage4_d0'),
                                     ('long_delay','long_d4096','long_d0'),
                                     ('length_growth','long_d0','control_d0'),
                                     ('stage_baseline','stage4_d0','control_d0')):
            expected = predicted[after]-predicted[before]
            paired = [next(row['cycles'] for row in rows if row['case_id'] == after and row['seed'] == seed)-
                      next(row['cycles'] for row in rows if row['case_id'] == before and row['seed'] == seed)
                      for seed in plan['seeds']]
            observed = statistics.median(paired)
            tolerance = .05*(abs(expected) if expected else freeze['old_windows']['0']['cycles'])
            responses.append(dict(name=label, observed_cycle=observed, predicted_cycle=expected,
                tolerance_cycle=tolerance, passed=abs(observed-expected) <= tolerance))
        models[name] = dict(windows=windows, responses=responses,
            passed=qualified and all(item['passed'] for item in windows+responses))
    retained = [name for name, result in models.items() if result['passed']]
    return dict(qualification='冻结后的原生程序局部cycle外推', qualified=qualified,
        decision='information_insufficient' if not qualified else 'retained_local_prediction_class' if retained else 'local_prediction_class_not_passed',
        retained=retained, models=models, bridge=bridge, stability=stability, rows=rows,
        output_elements_checked=sum(row['output_witness']['checked_elements'] for row in rows),
        placements_observationally_equal=freeze['placements_observationally_equal'],
        independent_local_cycle_validation=bool(retained), independent_C2=False, independent_C3=False,
        full_gemm_C2=False, full_gemm_C3=False)


def execute(root):
    plan, freeze = load(root)
    allocation = inspect_allocated_gpu(root/'allocation')
    require(allocation['uuid'] == plan['gpu_uuid'], '分配卡不符')
    completed, controls = 0, []
    started = time.monotonic()
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        try:
            for task in plan['order']:
                if task['split'] == 'heldout' and completed == 6:
                    checks = control_checks(controls, freeze)
                    save_json(root/'control-gate.json', checks)
                    require(all(check['passed'] for check in checks), '原生同批对照桥接失败，停止留出')
                require(time.monotonic()-started < 510, '本轮时间预算耗尽')
                case = next(case for case in plan['cases'] if case['id'] == task['case_id'])
                folder = root/'runs'/case['id']/f"seed{task['seed']}"
                command = probe_command(root/'reference/probe', case, task['seed']) + ['--index-mode','incremental']
                command_record(folder, command, 60)
                row = validate(folder, case, task['seed'], plan['gpu_uuid'])
                if task['split'] == 'control':
                    controls.append(row)
                completed += 1
                print(f"{case['id']} seed{task['seed']}: {row['cycles']} cycle", flush=True)
            save_json(root/'score.json', summarize(root))
        except Exception as error:
            save_json(root/'failure.json', dict(completed_processes=completed, error=str(error),
                elapsed_seconds=time.monotonic()-started))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('predict','execute','score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'predict':
        save_json(args.output, predict(root))
    elif args.command == 'score':
        save_json(args.output, summarize(root))
    else:
        execute(root)


if __name__ == '__main__':
    main()
