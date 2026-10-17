#!/usr/bin/env python3
"""L2-01 六个已登记形式的有限开发拟合、冻结和独立留出评分。"""
import argparse
import importlib.util
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import scipy
from scipy.linalg import null_space
from scipy.optimize import nnls

import l2_saturation as scan
from l2_service import FORMS, predict_read_window

B = 276824064
SELECTION = ['development_condition_MAPE', 'development_condition_max_relative_error',
             'parameter_count', 'form_name']


def constrained_linear_fit(matrix, target, constraints):
    """Three variables, five homogeneous inequalities: enumerate at most 32 faces.

    The feasibility tolerance is floating-point solver bookkeeping, never a data/error
    acceptance threshold. Parameters are clipped to nonnegative before actual prediction.
    """
    feasible = []
    for mask in range(1 << len(constraints)):
        active = constraints[[i for i in range(len(constraints)) if mask & (1 << i)]]
        basis = null_space(active) if len(active) else np.eye(3)
        if basis.shape[1]:
            value = basis @ np.linalg.lstsq(matrix @ basis, target, rcond=None)[0]
        else:
            value = np.zeros(3)
        if min(constraints @ value) >= -1e-10*max(1., float(np.linalg.norm(value))):
            feasible.append(np.maximum(value, 0.))
    return min(feasible, key=lambda p: float(np.sum((matrix @ p-target)**2)))


def frozen_module(path, name):
    # Importing the saved pure helpers must not create pycache in historical evidence.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_validation(actual, saved):
    """Only replayed warmup CV permits cross-CPU sqrt rounding; all other fields exact."""
    return actual.keys() == saved.keys() and all(
        math.isclose(value, saved[key], rel_tol=1e-12, abs_tol=0.) if key == 'warmup_cv'
        else value == saved[key] for key, value in actual.items())


def read_evidence(root, split):
    helper = frozen_module(root/'source/l2_saturation.py', '_l2_saved_scan')
    plan = helper.load_plan(root)
    helper.require(helper.digest(Path(scan.__file__)) == plan['source_sha256']['l2_saturation.py']
        and helper.digest(Path(sys.modules[predict_read_window.__module__].__file__)) ==
        plan['source_sha256']['l2_service.py'], '拟合调用的runner／数值内核不是本批冻结版本')
    build = helper.verify_build(root)
    summary = helper.read_json(root/split/'summary.json')
    helper.require(summary['complete'] and summary['split'] == split, '采样批次未完整')
    helper.require(summary['plan_sha256'] == helper.digest(root/'plan.json') and
        summary['build_identity_sha256'] == helper.digest(root/'build/identity.json'), '汇总身份不符')
    cases = [case for case in plan['cases'] if case['split'] == split]
    expected = {(case['id'], seed) for case in cases for seed in case['seeds']}
    helper.require(len(summary['results']) == len(expected) and
        {(row['case_id'], row['seed']) for row in summary['results']} == expected,
        '开发／留出进程覆盖不符')
    rows, identities = [], []
    by_case = {case['id']: case for case in cases}
    for item in summary['results']:
        case = by_case[item['case_id']]
        if case['mode'] != 'read':
            continue  # write/mixed retain their original qualification and never price read.
        folder = root/item['path']
        receipt = helper.read_json(folder/'receipt.json')
        argv = ['--active-ctas', str(case['active_ctas']), '--chains', str(case['chains']),
                '--mode', 'read', '--array-bytes', str(case['array_bytes']),
                '--iterations', str(case['iterations']), '--seed', str(item['seed']),
                '--samples', '1', '--warmup-limit', '60']
        helper.require(receipt['returncode'] == 0 and receipt['timed_out'] is False and
            receipt['argv'][1:] == argv and
            receipt['stdout_sha256'] == helper.digest(folder/'stdout') and
            receipt['stderr_sha256'] == helper.digest(folder/'stderr'), 'raw命令或返回身份不符')
        validated = helper.read_json(folder/'validated.json')
        helper.require(helper.digest(folder/'validated.json') == item['validated_sha256'],
                       'validated SHA不符')
        actual = helper.validate_probe(folder, helper.read_json(folder/'stdout'), case, item['seed'])
        helper.require(same_validation(actual, validated) and
            same_validation(actual, {key: item[key] for key in actual}), 'raw独立重放不一致')
        helper.require(actual['device_uuid'] == summary['device_uuid'], '同卡身份不符')
        rows.append(dict(case_id=case['id'], seed=item['seed'], S=len(actual['active_smids']),
            J=case['chains'], B=actual['read_payload_bytes'], elapsed_ns=actual['elapsed_ns'],
            performance_eligible=actual['performance_eligible']))
        identities.append(dict(case_id=case['id'], seed=item['seed'], path=item['path'],
            stdout_sha256=receipt['stdout_sha256'], receipt_sha256=helper.digest(folder/'receipt.json'),
            validated_sha256=item['validated_sha256']))
        if actual['warmup_cv'] != validated['warmup_cv']:
            identities[-1]['warmup_cv_replay'] = dict(saved=validated['warmup_cv'],
                replayed=actual['warmup_cv'], saved_hex=float(validated['warmup_cv']).hex(),
                replayed_hex=float(actual['warmup_cv']).hex(), relative_tolerance=1e-12,
                original_converged=validated['warmup_converged'],
                replayed_converged=actual['warmup_converged'])
    helper.require(all(row['B'] == B for row in rows), '逻辑读量不是固定16轮工作量')
    identity = dict(plan_sha256=helper.digest(root/'plan.json'),
        build_identity_sha256=helper.digest(root/'build/identity.json'),
        summary_sha256=helper.digest(root/split/'summary.json'), device_uuid=summary['device_uuid'],
        source_sha256=plan['source_sha256'], binary_sha256=build['probe_sha256'],
        sass_sha256=build['sass_sha256'], raw=identities)
    return helper, summary, rows, identity


def conditions(rows):
    result = []
    for case_id in sorted({row['case_id'] for row in rows}):
        selected = [row for row in rows if row['case_id'] == case_id]
        scan.require(len(selected) == 3 and {row['seed'] for row in selected} == set(scan.SEEDS),
                     '每条件须为三个独立进程')
        scan.require(len({(row['S'], row['J']) for row in selected}) == 1, '实际活动SM或J不一致')
        result.append(dict(case_id=case_id, S=selected[0]['S'], J=selected[0]['J'],
            mean_ns=sum(row['elapsed_ns'] for row in selected)/3,
            min_ns=min(row['elapsed_ns'] for row in selected),
            max_ns=max(row['elapsed_ns'] for row in selected)))
    return result


def monotonic_counterexamples(data):
    failures = []
    for left in data:
        for right in data:
            axis = ('S' if left['J'] == right['J'] and left['S'] < right['S'] else
                    'J' if left['S'] == right['S'] and left['J'] < right['J'] else None)
            if axis and left['max_ns'] < right['min_ns']:
                failures.append(dict(axis=axis, fewer=left['case_id'], more=right['case_id'],
                    fewer_interval_ns=[left['min_ns'], left['max_ns']],
                    more_interval_ns=[right['min_ns'], right['max_ns']]))
    return failures


def fit_form(data, family, jsat):
    """One registered form; max enumerates only its finite, ordered branch partitions."""
    scan.require((family, jsat) in FORMS and all(row['S'] > 0 and row['J'] > 0 and
        math.isfinite(row['mean_ns']) and row['mean_ns'] > 0 for row in data),
        '形式或正时间／活动规模不符')
    x = np.array([1/(row['S']*min(row['J'], jsat)) for row in data])
    y = np.array([row['mean_ns'] for row in data])
    scale, xmax = float(max(y)), float(max(x))
    z, target = x/xmax, y/scale
    if family == 'sum':
        matrix = np.column_stack((np.ones(len(x)), z))
        values, _ = nnls(matrix/target[:, None], np.ones(len(x)))
        if np.ptp(target) == 0:
            values = np.array([target[0], 0.])
        a, u, v = float(values[0]), float(values[1]), None
    else:
        candidates = []
        unique = np.unique(z)
        # At either extreme use the exact two/one-parameter reduction and preserve bounds.
        local, _ = nnls(np.column_stack((np.ones(len(x)), z))/target[:, None],
                        np.ones(len(x)))
        candidates.append(np.array([local[0], local[1], local[1]*min(z)/2]))
        floor = float(np.sum(1/target)/np.sum(1/target**2))
        candidates.append(np.array([0., floor/2, floor]))
        if np.ptp(target) == 0:
            candidates = candidates[-1:]
        for low, high in zip(unique[:-1], unique[1:]):
            is_local = z >= high
            matrix = np.column_stack((np.ones(len(x)), z*is_local, ~is_local))
            weighted = matrix/target[:, None]
            constraints = np.vstack((np.eye(3), [0., -low, 1.], [0., high, -1.]))
            candidates.append(constrained_linear_fit(weighted, np.ones(len(x)), constraints))
        best = min(candidates, key=lambda p: float(np.sum(
            ((p[0]+np.maximum(p[1]*z, p[2])-target)/target)**2)))
        if best[1] == 0 and best[0]+best[2] > 0:
            # The same optimal constant window has a finite all-plateau representative.
            # Reparameterize this exact prediction, rather than prefer a worse LS solution.
            constant = best[0]+best[2]
            finite = np.array([0., constant/2, constant])
            scan.require(np.array_equal(best[0]+np.maximum(best[1]*z, best[2]),
                finite[0]+np.maximum(finite[1]*z, finite[2])), '平台重参数改变最优预测')
            best = finite
        a, u, v = map(float, best)
    record = dict(name=f'{family}_jsat{jsat}', family=family, jsat=jsat,
                  parameter_count=2 if family == 'sum' else 3)
    if not math.isfinite(u) or u <= 0:
        return dict(record, status='unusable_zero_supply_slope', service=None,
                    identifiability='b1无限，现数值接口无可执行有限参数；不伪造有限价格')
    service = dict(family=family, jsat=jsat, logical_bytes=B,
                   b1_bytes_per_ns=B*xmax/(u*scale))
    if family == 'sum':
        service['floor_ns'] = a*scale
        identification = dict(rank=2, separately_identified=['floor_ns', 'b1_bytes_per_ns'],
            excluded_decomposition='固定B只识别c；不分别识别a和B/Bg')
    else:
        scan.require(v is not None and v > 0 and a >= 0, 'Hmax非正共享项或负固定项')
        service.update(a_ns=a*scale, shared_bytes_per_ns=B/(v*scale))
        local = u*z > v+1e-8*max(1., u, v)
        plateau = u*z < v-1e-8*max(1., u, v)
        matrix = np.column_stack((np.ones(len(x)), z*local, plateau))
        rank = int(np.linalg.matrix_rank(matrix))
        local_x = sorted(set(map(float, x[local])))
        identification = dict(rank=rank, strict_local_conditions=int(sum(local)),
            strict_plateau_conditions=int(sum(plateau)), distinct_local_x=local_x,
            separately_identified=(len(local_x) >= 2 and bool(any(plateau))),
            numeric_service='冻结的确定性代表；分支不足不声称唯一物理参数')
        if all(local):
            identification['bounds'] = dict(shared_bytes_per_ns_min=B/(u*scale*min(z)),
                shared_window_ns_max=u*scale*float(min(z)))
        elif all(plateau):
            identification['bounds'] = dict(identified_floor_ns=(a+v)*scale,
                a_ns_interval=[0., (a+v)*scale],
                constraint='u*x_max <= v, a+v=floor；u=B/b1, v=B/Bg')
        elif not identification['separately_identified']:
            identification['bounds'] = dict(nonnegative=['a', 'u=B/b1', 'v=B/Bg'],
                local_x=local_x, plateau_x=sorted(set(map(float, x[plateau]))),
                constraint='平台u*x<=v；局部u*x>=v；仅保存这些分支界限与拟合组合')
    prediction = [predict_read_window(B, row['S'], row['J'], service) for row in data]
    errors = [(pred-row['mean_ns'])/row['mean_ns'] for pred, row in zip(prediction, data)]
    return dict(record, status='fitted', service=service, identifiability=identification,
        relative_least_squares=sum(error**2 for error in errors),
        MAPE=sum(abs(error) for error in errors)/len(errors),
        max_relative_error=max(map(abs, errors)),
        development=[dict(row, predicted_ns=pred, relative_error=error)
                     for row, pred, error in zip(data, prediction, errors)])


def fit_all(data):
    counterexamples = monotonic_counterexamples(data)
    if counterexamples:
        return [dict(name=f'{family}_jsat{jsat}', family=family, jsat=jsat,
            status='refuted_stable_monotonic_reversal', service=None)
            for family, jsat in FORMS], None, counterexamples
    fits = [fit_form(data, family, jsat) for family, jsat in FORMS]
    usable = [fit for fit in fits if fit['status'] == 'fitted']
    selected = min(usable, key=lambda fit: (fit['MAPE'], fit['max_relative_error'],
        fit['parameter_count'], fit['name'])) if usable else None
    return fits, selected, []


def check_sass_review(path, identity):
    review = scan.read_json(path)
    scan.require(review['source_sha256'] == identity['source_sha256']['l2_saturation.cu'] and
                 review['sass_sha256'] == identity['sass_sha256'], '目标SASS审核身份不符')
    if review.get('qualified') is not True:
        scan.require(review.get('schema') == 'gh200-l2-target-review/1' and
                     'release 12.9' in review['compiler_version'], '不是目标工具链审核')
        functions = review['functions']
        scan.require(len(functions) == 9 and
            {(item['mode'], item['chains']) for item in functions} ==
            {(mode, chains) for mode in range(3) for chains in (1, 4, 8)}, '九个目标函数覆盖不符')
        for item in functions:
            scan.require(item['ldg128_count'] == (item['chains'] if item['mode'] in (0, 2) else 0)
                and item['stg128_count'] == (item['chains'] if item['mode'] in (1, 2) else 0)
                and item['local_count'] == 0, '目标访存宽度／数量／local审核失败')
    return review


def fit(root, output, review_path):
    helper, summary, rows, identity = read_evidence(root, 'dev')
    scan.require(len(rows) == 36 and all(row['performance_eligible'] for row in rows),
                 '12个读条件未完整具备性能资格')
    review = check_sass_review(review_path, identity)
    data = conditions(rows)
    fits, selected, reversals = fit_all(data)
    heldout = [case for case in helper.make_cases() if case['split'] == 'heldout']
    predictions = [dict(case_id=case['id'], S=case['active_ctas'], J=case['chains'],
        logical_bytes=B, seeds=case['seeds'], forms={item['name']:
            predict_read_window(B, case['active_ctas'], case['chains'], item['service'])
            if item['service'] else None for item in fits}) for case in heldout]
    identity.update(fitter_sha256=scan.digest(Path(__file__)), numpy_version=np.__version__,
        scipy_version=scipy.__version__, target_review_sha256=scan.digest(review_path))
    service = dict(schema='gh200-l2-condition-services/1', logical_bytes=B, identity=identity,
        selection_rule=SELECTION, selected=selected['name'] if selected else None,
        all_fits=fits, heldout_predictions=predictions, stable_reversals=reversals,
        development_raw=rows, target_review=review,
        qualification='固定B整窗开发拟合；未定局部通过阈值，不宣称独立C2/C3通过')
    output.mkdir(parents=True, exist_ok=False)
    scan.save_json(output/'service.json', service)
    scan.save_json(output/'predictions.json', predictions)
    shutil.copy2(Path(__file__), output/'l2_fit.py')
    for name in ('l2_saturation.py', 'l2_service.py'):
        shutil.copy2(root/'source'/name, output/name)
    shutil.copy2(review_path, output/'target-review.json')
    if selected is None:
        return dict(state='no_service_frozen', forms=6, stable_reversals=len(reversals))
    scan.save_json(output/'selected.service.json', selected['service'])
    # Retain the original runner's whole-development qualification gate.
    if not summary['all_performance_eligible']:
        return dict(state='read_fit_saved_but_original_freeze_gate_not_met', forms=6)
    freeze = dict(schema=helper.FREEZE_SCHEMA, plan_sha256=identity['plan_sha256'],
        build_identity_sha256=identity['build_identity_sha256'],
        dev_summary_sha256=identity['summary_sha256'], service_file='service.json',
        service_sha256=scan.digest(output/'service.json'),
        selected_service_file='selected.service.json',
        selected_service_sha256=scan.digest(output/'selected.service.json'),
        predictions_file='predictions.json', predictions_sha256=scan.digest(output/'predictions.json'),
        fitter_sha256=identity['fitter_sha256'], selection_rule=SELECTION)
    scan.save_json(output/'dev-freeze.json', freeze)
    helper.verify_dev_freeze(root, output/'dev-freeze.json')
    return dict(state='frozen', selected=selected['name'], forms=6, read_processes=36,
                heldout_conditions=4, freeze_sha256=scan.digest(output/'dev-freeze.json'))


def score(root, freeze_path, output):
    helper, summary, rows, identity = read_evidence(root, 'heldout')
    binding = helper.verify_dev_freeze(root, freeze_path)
    scan.require(summary['dev_freeze']['sha256'] == binding['sha256'] and
        summary['dev_freeze']['service_sha256'] == binding['service_sha256'], '留出未绑定本开发冻结')
    freeze = scan.read_json(freeze_path)
    folder = freeze_path.parent
    for field in ('selected_service', 'predictions'):
        scan.require(scan.digest(folder/freeze[field+'_file']) == freeze[field+'_sha256'],
                     '冻结预测／数值服务身份变更')
    service = scan.read_json(folder/freeze['service_file'])
    scan.require(service['identity']['device_uuid'] == identity['device_uuid'], '留出不是开发同卡')
    predictions = scan.read_json(folder/freeze['predictions_file'])
    scan.require(predictions == service['heldout_predictions'], '冻结预测副本不一致')
    selected = service['selected']
    predicted = {row['case_id']: row for row in predictions}
    def errors(row, observation):
        frozen = predicted[row['case_id']]
        scan.require((row['S'], row['J']) == (frozen['S'], frozen['J']), '留出实际S/J不在冻结坐标')
        return {name: dict(predicted_ns=value, error_ns=value-observation,
            absolute_error_ns=abs(value-observation),
            relative_error=(value-observation)/observation)
            for name, value in frozen['forms'].items() if value is not None}
    per_process = [dict(row, errors=errors(row, row['elapsed_ns'])) for row in rows]
    per_condition = [dict(row, errors=errors(row, row['mean_ns'])) for row in conditions(rows)]
    report = dict(schema='gh200-l2-heldout-score/1', selected=selected,
        selection_rule=SELECTION, development_freeze_sha256=binding['sha256'], identity=identity,
        raw_processes=per_process, conditions=per_condition,
        selected_condition_MAPE=sum(abs(row['errors'][selected]['relative_error'])
            for row in per_condition)/len(per_condition),
        selected_condition_max_relative_error=max(abs(row['errors'][selected]['relative_error'])
            for row in per_condition),
        performance_eligible=all(row['performance_eligible'] for row in rows),
        qualification='连续时间/相对误差与冻结候选区别；无本轮局部通过阈值；非独立C2/C3通过；不重选')
    output.mkdir(parents=True, exist_ok=False)
    scan.save_json(output/'score.json', report)
    shutil.copy2(Path(__file__), output/'l2_fit.py')
    return dict(state='scored_without_reselection', conditions=len(per_condition),
                read_processes=len(rows), selected=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='step', required=True)
    for name in ('fit', 'score'):
        command = commands.add_parser(name)
        command.add_argument('--input', required=True, type=Path)
        command.add_argument('--output', required=True, type=Path)
        if name == 'fit':
            command.add_argument('--sass-review', type=Path)
        else:
            command.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    root = args.input.resolve()
    if args.step == 'fit':
        result = fit(root, args.output.resolve(),
                     (args.sass_review or root/'build/target-review.json').resolve())
    else:
        result = score(root, args.freeze.resolve(), args.output.resolve())
    print(result)


if __name__ == '__main__':
    main()
