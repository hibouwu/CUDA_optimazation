#!/usr/bin/env python3
"""Diagnose the archived m11 development set; no GPU or holdout execution."""
import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.optimize import linprog


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def full_warp_demand(case):
    total = 0
    words_per_lane = case['width'] // 4
    for addresses in case['addresses_words']:
        for warp in range(case['warps']):
            banks = [set() for _ in range(32)]
            for address in addresses[warp * 32:(warp + 1) * 32]:
                for word in range(address, address + words_per_lane):
                    banks[word % 32].add(word)
            total += max(len(words) for words in banks)
    return total


def metrics(rows):
    errors = [abs(row['signed_error']) for row in rows]
    cycles = [row['absolute_cycle_error'] for row in rows]
    return dict(conditions=len(rows), median_absolute_relative_error=statistics.median(errors),
                maximum_absolute_relative_error=max(errors),
                median_absolute_cycle_error=statistics.median(cycles),
                maximum_absolute_cycle_error=max(cycles),
                above_5pct=sum(error > .05 for error in errors),
                above_10pct=sum(error > .10 for error in errors),
                local_absolute_failures=sum(not row['local_absolute_cycle_pass'] for row in rows))


def feature_group(case):
    return f"op{case['op']}_b{case['width']}_w{case['warps']}_d{int(case['dependent'])}"


def design(cases, demands, form, threshold, nscale, dscale):
    n = np.array([case['iterations'] for case in cases], float) / nscale
    d = np.array(demands, float) / dscale
    if form == 'F2':
        return np.column_stack([np.ones(len(cases)), n, n * d])
    low = np.array(demands) <= threshold
    return np.column_stack([np.ones(len(cases)), n * low, n * ~low, n * d * ~low])


def branch_constraints(demands, form, threshold, dscale):
    if form == 'F2':
        return np.empty((0, 3))
    rows = []
    for demand in sorted(set(demands)):
        # floor <= offset + price*D on high branch; reverse on low branch.
        row = np.array([0., 1., -1., -demand / dscale])
        if demand <= threshold:
            row = -row
        rows.append(row)
    return np.array(rows)


def parameter_bounds(form):
    if form == 'F2':
        return [(-256, None), (0, None), (0, None)]
    return [(-256, None), (0, None), (None, None), (0, None)]


def optimize_partition(cases, observed, demands, form, threshold, nscale, dscale,
                       limits=None, objective=None, additional_demands=()):
    matrix = design(cases, demands, form, threshold, nscale, dscale)
    branches = branch_constraints(list(demands) + list(additional_demands), form, threshold, dscale)
    normalized = matrix / observed[:, None]
    if limits is None:
        # Minimize the worst relative error, rather than let many accurate points
        # conceal an avoidable regression in the original least-squares fit.
        left = np.vstack([normalized, -normalized])
        constraints = np.column_stack([left, -np.ones(len(left))])
        if len(branches):
            constraints = np.vstack([constraints, np.column_stack([branches, np.zeros(len(branches))])])
        right = np.concatenate([np.ones(len(observed)), -np.ones(len(observed)), np.zeros(len(branches))])
        cost = np.zeros(matrix.shape[1] + 1)
        cost[-1] = 1
        bounds = parameter_bounds(form) + [(0, None)]
    else:
        constraints = np.vstack([normalized, -normalized, branches])
        right = np.concatenate([1 + limits, -1 + limits, np.zeros(len(branches))])
        cost = np.zeros(matrix.shape[1]) if objective is None else objective
        bounds = parameter_bounds(form)
    result = linprog(cost, A_ub=constraints, b_ub=right, bounds=bounds, method='highs')
    if result.status == 3:
        return dict(status='unbounded')
    if not result.success:
        return dict(status='infeasible', solver_status=int(result.status))
    coefficient = result.x[:matrix.shape[1]]
    return dict(status='solved', coefficient=coefficient,
                error_bound=float(result.x[-1]) if limits is None else None,
                matrix_rank=int(np.linalg.matrix_rank(normalized)),
                columns=int(matrix.shape[1]), objective=float(cost @ result.x))


def predict(cases, demands, fitted):
    coefficient = fitted['coefficient']
    n = np.array([case['iterations'] for case in cases], float) / fitted['nscale']
    d = np.array(demands, float) / fitted['dscale']
    if fitted['form'] == 'F2':
        return coefficient[0] + n * (coefficient[1] + coefficient[2] * d)
    return coefficient[0] + n * np.maximum(coefficient[1], coefficient[2] + coefficient[3] * d)


def fit(cases, observations, demands, form, absolute_tolerances=None):
    observed = np.array(observations, float)
    nscale = max(case['iterations'] for case in cases)
    dscale = max(demands)
    thresholds = [None] if form == 'F2' else [0] + sorted(set(demands))
    best = None
    locally_feasible = False
    for threshold in thresholds:
        result = optimize_partition(cases, observed, demands, form, threshold, nscale, dscale)
        if result['status'] != 'solved':
            continue
        result.update(form=form, threshold=threshold, nscale=nscale, dscale=dscale)
        calculated = predict(cases, demands, result)
        represented = design(cases, demands, form, threshold, nscale, dscale) @ result['coefficient']
        assert np.max(np.abs(calculated - represented)) < .01
        error = np.abs(calculated / observed - 1)
        assert abs(max(error) - result['error_bound']) < 1e-5
        result['training_maximum_absolute_relative_error'] = float(max(error))
        result['training_median_absolute_relative_error'] = float(np.median(error))
        result['training_demand_span'] = [min(demands), max(demands)]
        if best is None or (max(error), np.median(error)) < (
                best['training_maximum_absolute_relative_error'], best['training_median_absolute_relative_error']):
            best = result
        if absolute_tolerances is not None:
            limits = np.minimum(.10, np.array(absolute_tolerances) / observed)
            feasibility = optimize_partition(cases, observed, demands, form, threshold, nscale, dscale, limits=limits)
            locally_feasible |= feasibility['status'] == 'solved'
    assert best is not None
    best['original_local_and_10pct_gates_feasible'] = locally_feasible if absolute_tolerances is not None else None
    return best


def fitted_json(fitted):
    result = dict(fitted)
    result['coefficient'] = fitted['coefficient'].tolist()
    result['coefficient_scaling'] = 'fixed; Nscale*floor/offset/issue; Nscale*Dscale*price'
    return result


def scored_row(case, row, prediction):
    result = dict(row)
    result.update(case=case['id'], group=feature_group(case), layout=case['layout'],
                  predicted_cycles=float(prediction), signed_error=float(prediction / row['median_cycles'] - 1),
                  absolute_cycle_error=float(abs(prediction - row['median_cycles'])))
    result['local_absolute_cycle_pass'] = result['absolute_cycle_error'] <= row['absolute_cycle_tolerance']
    return result


def prediction_interval(cases, observations, demands, query, query_demand, fitted,
                        mode='near_training_optimum', absolute_tolerances=None):
    # Training-only parameter uncertainty at the best training error + numerical
    # slack. No use of the held-out observed time to select a parameter.
    observed = np.array(observations, float)
    form = fitted['form']
    thresholds = [None] if form == 'F2' else [0] + sorted(set(demands + [query_demand]))
    if mode == 'original_local_and_10pct':
        limits = np.minimum(.10, np.array(absolute_tolerances) / observed)
    elif mode == '10pct_only':
        limits = np.full(len(cases), .10)
    else:
        limits = np.full(len(cases), fitted['training_maximum_absolute_relative_error'] + 1e-8)
    lower, upper = None, None
    lower_unbounded = upper_unbounded = False
    lower_training_metrics = upper_training_metrics = None
    lower_parameter_witness = upper_parameter_witness = None
    for threshold in thresholds:
        objective = design([query], [query_demand], form, threshold, fitted['nscale'], fitted['dscale'])[0]
        for sign in (1, -1):
            result = optimize_partition(cases, observed, demands, form, threshold,
                                        fitted['nscale'], fitted['dscale'], limits=limits,
                                        objective=sign * objective, additional_demands=[query_demand])
            if result['status'] == 'unbounded':
                if sign == 1:
                    lower_unbounded = True
                else:
                    upper_unbounded = True
            elif result['status'] == 'solved':
                value = float(objective @ result['coefficient'])
                matrix = design(cases, demands, form, threshold, fitted['nscale'], fitted['dscale'])
                errors = np.abs(matrix @ result['coefficient'] / observed - 1)
                training_metrics = dict(maximum_absolute_relative_error=float(max(errors)),
                                        median_absolute_relative_error=float(np.median(errors)))
                witness = dict(coefficient=result['coefficient'].tolist(), threshold=threshold,
                               form=form, nscale=fitted['nscale'], dscale=fitted['dscale'])
                if sign == 1:
                    if lower is None or value < lower:
                        lower = value
                        lower_training_metrics = training_metrics
                        lower_parameter_witness = witness
                else:
                    if upper is None or value > upper:
                        upper = value
                        upper_training_metrics = training_metrics
                        upper_parameter_witness = witness
    return dict(lower_cycles=lower, upper_cycles=upper,
                lower_unbounded=lower_unbounded, upper_unbounded=upper_unbounded,
                training_constraints_feasible=lower is not None or lower_unbounded,
                mode=mode, lower_training_metrics=lower_training_metrics,
                upper_training_metrics=upper_training_metrics,
                lower_parameter_witness=lower_parameter_witness,
                upper_parameter_witness=upper_parameter_witness,
                observed_holdout_not_used=True,
                interpretation='conditional feasible parameter bounds, not confidence intervals; median 5% is reported for extrema but not imposed in the linear program')


def collision_bounds(cases, rows, demands):
    grouped = defaultdict(list)
    for case, row, demand in zip(cases, rows, demands):
        grouped[(feature_group(case), case['iterations'], demand)].append((case, row))
    output = []
    for (group, iterations, demand), pairs in grouped.items():
        if len(pairs) < 2:
            continue
        low = min(pairs, key=lambda pair: pair[1]['median_cycles'])
        high = max(pairs, key=lambda pair: pair[1]['median_cycles'])
        a, b = int(high[1]['median_cycles']), int(low[1]['median_cycles'])
        bound = Fraction(a - b, a + b)
        output.append(dict(group=group, iterations=iterations, demand=demand,
                           low_case=low[0]['id'], high_case=high[0]['id'],
                           low_cycles=b, high_cycles=a, minimax_relative_lower_bound=float(bound),
                           exact_lower_bound=str(bound)))
    return sorted(output, key=lambda row: row['minimax_relative_lower_bound'], reverse=True)


def convexity_witnesses(cases, rows, demands):
    grouped = defaultdict(list)
    for case, row, demand in zip(cases, rows, demands):
        if case['iterations'] == 8192:
            grouped[feature_group(case)].append((demand, case, row))
    witnesses = []
    for group, points in grouped.items():
        for d1, c1, r1 in points:
            for d2, c2, r2 in points:
                for d3, c3, r3 in points:
                    if not d1 < d2 < d3:
                        continue
                    w = Fraction(d3 - d2, d3 - d1)
                    y1, y2, y3 = [Fraction(int(row['median_cycles'])) for row in (r1, r2, r3)]
                    chord = w * y1 + (1 - w) * y3
                    if y2 <= chord:
                        continue
                    bound = (y2 - chord) / (y2 + chord)
                    local_upper = w * (y1 + r1['absolute_cycle_tolerance']) + (1 - w) * (y3 + r3['absolute_cycle_tolerance'])
                    local_lower = y2 - r2['absolute_cycle_tolerance']
                    witnesses.append(dict(group=group, cases=[c1['id'], c2['id'], c3['id']],
                                          demands=[d1, d2, d3], observed_cycles=[int(y1), int(y2), int(y3)],
                                          convexity_minimax_relative_lower_bound=float(bound), exact_lower_bound=str(bound),
                                          original_local_bounds_inconsistent=local_lower > local_upper,
                                          local_gap_cycles=float(local_lower - local_upper)))
    return sorted(witnesses, key=lambda row: row['convexity_minimax_relative_lower_bound'], reverse=True)


def main(input_root, output, preregistration):
    prereg = read_json(preregistration)
    assert prereg['gpu_submission_budget'] == 0
    for name, expected in prereg['input_sha256'].items():
        assert hashlib.sha256((input_root / name).read_bytes()).hexdigest() == expected, name
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('archived_m11_model', input_root / 'package/source/r03_lds_sts_model.py')
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    previous_model_module = sys.modules.get('r03_lds_sts_model')
    sys.modules['r03_lds_sts_model'] = archived
    analyzer_spec = importlib.util.spec_from_file_location(
        'archived_m11_analyzer', input_root / 'package/source/analyze_r03_lds_sts.py')
    archived_analyzer = importlib.util.module_from_spec(analyzer_spec)
    analyzer_spec.loader.exec_module(archived_analyzer)
    if previous_model_module is None:
        del sys.modules['r03_lds_sts_model']
    else:
        sys.modules['r03_lds_sts_model'] = previous_model_module
    cases = read_json(input_root / 'package/development-cases.json')
    old_rows = read_json(input_root / 'development-analysis/comparison.json')['rows']
    original_parameters = read_json(input_root / 'development-analysis/parameters.json')
    assert len(cases) == len(old_rows) == 540
    old_demands, new_demands = [], []
    for case, row in zip(cases, old_rows):
        assert case['id'] == row['case']
        raw = [read_json(input_root / 'package/development/samples' / case['id'] / f'trial-{i:02}/result.json') for i in range(3)]
        assert [item['elapsed_cycles'] for item in raw] == row['cycles']
        assert all(item['errors'] == 0 and item['smid'] == 124 for item in raw)
        original = archived.predict(case, original_parameters)['window_cycles']
        assert abs(original - row['predicted_cycles']) < 1e-6
        archived.validate_map(case)
        old_demands.append(archived.demand(case)['total_per_round'])
        new_demands.append(full_warp_demand(case))
    write_json(output / 'input-replay.json', dict(conditions=540, processes=1620,
               controls_excluded_from_model_fit=12, original_prediction_replay=True,
               numerical_errors=0, smids=[124], source_model_sha256=prereg['input_sha256']['package/source/r03_lds_sts_model.py']))
    grouped_indices = defaultdict(list)
    for i, case in enumerate(cases):
        grouped_indices[feature_group(case)].append(i)
    slices = []
    for group, indices in grouped_indices.items():
        for layout in ['all'] + sorted({cases[i]['layout'] for i in indices}):
            selected = [old_rows[i] for i in indices if layout == 'all' or cases[i]['layout'] == layout]
            slices.append(dict(group=group, layout=layout, **metrics(selected)))
    write_json(output / 'original-error-slices.json', slices)
    write_json(output / 'old-equal-input-bounds.json', collision_bounds(cases, old_rows, old_demands))
    write_json(output / 'new-equal-input-bounds.json', collision_bounds(cases, old_rows, new_demands))
    write_json(output / 'convexity-witnesses.json', convexity_witnesses(cases, old_rows, new_demands))
    original_nonbroadcast = []
    for group, indices in grouped_indices.items():
        indices = [i for i in indices if cases[i]['layout'] != 'broadcast']
        selected = [cases[i] for i in indices]
        fitted = fit(selected, [old_rows[i]['median_cycles'] for i in indices], [old_demands[i] for i in indices], 'F1')
        predictions = predict(selected, [old_demands[i] for i in indices], fitted)
        rows = [scored_row(cases[i], old_rows[i], p) for i, p in zip(indices, predictions)]
        legacy_parameters = archived_analyzer.fit_group(selected, [old_rows[i] for i in indices])
        assert legacy_parameters['status'] == 'fitted_conditional'
        legacy_predictions = [archived.predict(case, {'groups': {group: legacy_parameters}})['window_cycles']
                              for case in selected]
        legacy_rows = [scored_row(cases[i], old_rows[i], p) for i, p in zip(indices, legacy_predictions)]
        complete = grouped_indices[group]
        replay_parameters = archived_analyzer.fit_group([cases[i] for i in complete], [old_rows[i] for i in complete])
        replay_predictions = [archived.predict(cases[i], {'groups': {group: replay_parameters}})['window_cycles']
                              for i in complete]
        original_difference = max(abs(p - old_rows[i]['predicted_cycles']) for i, p in zip(complete, replay_predictions))
        assert original_difference < .1
        original_nonbroadcast.append(dict(group=group, before=metrics([old_rows[i] for i in indices]),
                                         refit=metrics(rows), fitted=fitted_json(fitted),
                                         same_least_squares_objective_refit=metrics(legacy_rows),
                                         original_full_fit_replay_maximum_cycle_difference=original_difference,
                                         original_solver_nonbroadcast_parameters=legacy_parameters))
    write_json(output / 'original-nonbroadcast-refit.json', dict(
        interpretation='diagnostic refit with old features and original form; not a third adoption candidate', groups=original_nonbroadcast))
    decisions = {}
    for form in ('F1', 'F2'):
        fits, scores, folds = {}, [], []
        for group, indices in grouped_indices.items():
            selected = [cases[i] for i in indices]
            demands = [new_demands[i] for i in indices]
            observations = [old_rows[i]['median_cycles'] for i in indices]
            tolerances = [old_rows[i]['absolute_cycle_tolerance'] for i in indices]
            fitted = fit(selected, observations, demands, form, tolerances)
            fits[group] = fitted_json(fitted)
            predictions = predict(selected, demands, fitted)
            scores.extend(scored_row(cases[i], old_rows[i], p) for i, p in zip(indices, predictions))
            for layout in sorted({case['layout'] for case in selected}):
                training = [i for i in indices if cases[i]['layout'] != layout]
                heldout = [i for i in indices if cases[i]['layout'] == layout]
                train_cases = [cases[i] for i in training]
                train_demands = [new_demands[i] for i in training]
                train_observed = [old_rows[i]['median_cycles'] for i in training]
                fold_fit = fit(train_cases, train_observed, train_demands, form)
                query_cases = [cases[i] for i in heldout]
                query_demands = [new_demands[i] for i in heldout]
                calculated = predict(query_cases, query_demands, fold_fit)
                fold_rows = [scored_row(cases[i], old_rows[i], p) for i, p in zip(heldout, calculated)]
                longest = max(heldout, key=lambda i: cases[i]['iterations'])
                interval = prediction_interval(train_cases, train_observed, train_demands,
                                               cases[longest], new_demands[longest], fold_fit)
                training_tolerances = [old_rows[i]['absolute_cycle_tolerance'] for i in training]
                gate_interval = prediction_interval(train_cases, train_observed, train_demands,
                                                   cases[longest], new_demands[longest], fold_fit,
                                                   mode='original_local_and_10pct',
                                                   absolute_tolerances=training_tolerances)
                relative_interval = prediction_interval(train_cases, train_observed, train_demands,
                                                       cases[longest], new_demands[longest], fold_fit,
                                                       mode='10pct_only')
                folds.append(dict(group=group, omitted_layout=layout, training_conditions=len(training),
                                  training_fit=fitted_json(fold_fit), heldout_rows=fold_rows,
                                  metrics=metrics(fold_rows), demand_extrapolation=any(
                                      d < min(train_demands) or d > max(train_demands) for d in query_demands),
                                  longest_query_case=cases[longest]['id'], training_only_prediction_interval=interval,
                                  original_gate_prediction_interval=gate_interval,
                                  relative_gate_prediction_interval=relative_interval))
        scores.sort(key=lambda row: row['case'])
        score_metrics = metrics(scores)
        relative_pass = score_metrics['median_absolute_relative_error'] <= .05 and score_metrics['maximum_absolute_relative_error'] <= .10
        local_feasible = all(value['original_local_and_10pct_gates_feasible'] for value in fits.values())
        decisions[form] = dict(development_metrics=score_metrics, relative_gates_passed=relative_pass,
                               original_local_gates_feasible=local_feasible,
                               infeasible_local_groups=[group for group, value in fits.items() if not value['original_local_and_10pct_gates_feasible']],
                               leave_layout_metrics=metrics([row for fold in folds for row in fold['heldout_rows']]),
                               demand_extrapolation_folds=sum(fold['demand_extrapolation'] for fold in folds),
                               status='development_candidate_only' if relative_pass and local_feasible else 'rejected_offline',
                               independent_holdout_passed=False, physical_batches_measured=False)
        write_json(output / f'{form}-parameters.json', fits)
        write_json(output / f'{form}-comparison.json', dict(rows=scores, metrics=score_metrics))
        write_json(output / f'{form}-leave-layout.json', dict(folds=folds, independence='development diagnostic only; all layouts already observed'))
        with (output / f'{form}-comparison.csv').open('w') as f:
            keys = ['case', 'group', 'layout', 'median_cycles', 'predicted_cycles', 'signed_error',
                    'absolute_cycle_error', 'absolute_cycle_tolerance', 'local_absolute_cycle_pass']
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in scores:
                writer.writerow({key: row[key] for key in keys})
    write_json(output / 'decision.json', dict(forms=decisions, gpu_submissions=0,
               holdout_executed=False, public_model_modified=False,
               maximum_forms=2, scope='all 540 original conditions, same GPU/protocol/window'))
    print(json.dumps(decisions, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preregistration', type=Path, required=True)
    args = parser.parse_args()
    main(args.input.resolve(), args.output.resolve(), args.preregistration.resolve())
