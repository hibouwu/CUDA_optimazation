"""Score the lane contrast against unchanged tolerances; never refit old forms."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from r03_lane_contrast_model import address_features, group_key, validate_plan


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize(case, measured, tolerance):
    records = []
    failures = []
    for trial in range(3):
        folder = measured / 'samples' / case['id'] / f'trial-{trial:02}'
        if not (folder / 'completion.json').exists():
            failures.append(dict(trial=trial, error='missing prelisted process'))
            continue
        settled = json.loads((folder / 'completion.json').read_text())
        if settled['status'] != 'measured':
            failures.append(settled)
        else:
            records.append(json.loads((folder / settled['result']).read_text()))
    row = dict(case=case['id'], group=group_key(case), width=case['width'],
               warps=case['warps'], dependent=case['dependent'], layout=case['layout'],
               iterations=case['iterations'], absolute_cycle_tolerance=tolerance,
               features=address_features(case), failures=failures)
    if failures or len(records) != 3:
        return dict(**row, status='failed')
    cycles = [record['elapsed_cycles'] for record in records]
    cv = statistics.stdev(cycles) / statistics.mean(cycles)
    return dict(**row, status='measured' if cv <= .02 else 'dispersion_failed',
                cycles=cycles, median_cycles=statistics.median(cycles), cv=cv,
                smids=[record['smid'] for record in records],
                cycles_per_warp_instruction=statistics.median(cycles) / (8 * case['iterations'] * case['warps']))


def diagnostic_score(rows, cases, prediction_name):
    scores = []
    for case, row in zip(cases, rows):
        if row['status'] != 'measured':
            continue
        predicted = case['development_diagnostic_predictions'][prediction_name]
        observed = row['median_cycles']
        error = predicted / observed - 1
        absolute = abs(predicted - observed)
        scores.append(dict(case=case['id'], predicted_cycles=predicted,
                           observed_cycles=observed, signed_relative_error=error,
                           absolute_cycle_error=absolute,
                           absolute_cycle_tolerance=row['absolute_cycle_tolerance'],
                           local_absolute_pass=absolute <= row['absolute_cycle_tolerance']))
    return dict(source='prelisted archived diagnostic predictions; not a qualified model',
                conditions=len(scores), rows=scores,
                median_absolute_relative_error=statistics.median(abs(row['signed_relative_error']) for row in scores) if scores else None,
                maximum_absolute_relative_error=max(abs(row['signed_relative_error']) for row in scores) if scores else None,
                median_absolute_cycle_error=statistics.median(row['absolute_cycle_error'] for row in scores) if scores else None,
                maximum_absolute_cycle_error=max(row['absolute_cycle_error'] for row in scores) if scores else None,
                local_absolute_failures=sum(not row['local_absolute_pass'] for row in scores),
                old_global_failure_not_overridden=True, fitting_performed=False)


def analyze(package, measured, output):
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((package / 'minimal-measurement-cases.json').read_text())
    pairs = validate_plan(cases)
    parameters = json.loads((package / 'old-development-parameters.json').read_text())
    tolerances = parameters['absolute_cycle_tolerance']
    rows = [summarize(case, measured, tolerances[group_key(case)]) for case in cases]
    write(output / 'conditions.json', rows)
    indexed = {row['case']: row for row in rows}
    pair_results = []
    for pair in pairs:
        first = indexed[pair['contiguous']]
        second = indexed[pair['interleaved']]
        result = dict(contiguous=first['case'], interleaved=second['case'],
                      width=first['width'], warps=first['warps'], dependent=first['dependent'],
                      iterations=first['iterations'], invariant=pair['invariant'],
                      absolute_cycle_tolerance=first['absolute_cycle_tolerance'])
        if first['status'] == second['status'] == 'measured':
            a, b = first['median_cycles'], second['median_cycles']
            tolerance = first['absolute_cycle_tolerance']
            low = max(a - tolerance, b - tolerance, .9 * a, .9 * b)
            high = min(a + tolerance, b + tolerance, 1.1 * a, 1.1 * b)
            result.update(status='measured', contiguous_cycles=a, interleaved_cycles=b,
                          cycle_difference=b - a, relative_change=b / a - 1,
                          common_prediction_joint_interval=[low, high],
                          common_prediction_joint_feasible=low <= high,
                          minimum_common_max_absolute_error=abs(a - b) / 2,
                          minimum_common_max_relative_error=abs(a - b) / (a + b),
                          permutation_difference_within_original_tolerance=abs(a - b) <= tolerance,
                          smids=[first['smids'], second['smids']])
        else:
            result.update(status='failed', component_status=[first['status'], second['status']])
        pair_results.append(result)
    write(output / 'layout-pairs.json', pair_results)
    old_rows = {row['case']: row for row in json.loads((package / 'old-development-results.json').read_text())}
    old_cases = {case['id']: case for case in json.loads((package / 'old-development-cases.json').read_text())}
    bridges = []
    for case, row in zip(cases, rows):
        if case['layout'] not in ('stride1', 'broadcast'):
            continue
        old_id = f"o0_b{case['width']}_w{case['warps']}_d{int(case['dependent'])}_{case['layout']}_s17_n{case['iterations']}"
        if case['addresses_words'] != old_cases[old_id]['addresses_words']:
            raise ValueError('bridge maps differ from historical maps')
        result = dict(case=case['id'], historical_case=old_id, status=row['status'], passed=False)
        if row['status'] == 'measured':
            old = old_rows[old_id]['median_cycles']
            difference = row['median_cycles'] - old
            result.update(historical_cycles=old, current_cycles=row['median_cycles'],
                          cycle_difference=difference, relative_change=row['median_cycles'] / old - 1,
                          absolute_cycle_tolerance=row['absolute_cycle_tolerance'],
                          passed=abs(difference) <= row['absolute_cycle_tolerance'] and abs(row['median_cycles'] / old - 1) <= .10)
        bridges.append(result)
    if len(bridges) != 20:
        raise ValueError('twenty prelisted bridges required')
    write(output / 'bridges.json', bridges)
    for name in ('F1_fullwarp_cycles', 'old_form_nonbroadcast_refit_cycles'):
        write(output / (name + '.json'), diagnostic_score(rows, cases, name))
    primary = next(pair for pair in pair_results if pair['width'] == 16 and pair['warps'] == 8
                   and not pair['dependent'] and pair['iterations'] == 8192 and 'samebank' in pair['contiguous'])
    negative = [pair for pair in pair_results if pair['width'] == 4]
    incompatible = [pair for pair in pair_results if pair['status'] == 'measured'
                    and not pair['common_prediction_joint_feasible']]
    valid = all(row['status'] == 'measured' for row in rows)
    if not valid:
        endpoint = 'measurement_or_dispersion_gate_failed'
    elif not all(row['passed'] for row in bridges):
        endpoint = 'historical_bridge_gate_failed'
    elif not primary['common_prediction_joint_feasible']:
        endpoint = 'full_warp_set_hypothesis_refuted_lane_organization_observed'
    elif incompatible:
        endpoint = 'other_lane_pairs_refute_full_warp_set_invariance'
    else:
        endpoint = 'all_pairs_jointly_compatible_in_this_panel_dependency_gap_remains'
    decision = dict(status=endpoint, prelisted_conditions=48, measured_conditions=sum(row['status'] == 'measured' for row in rows),
                    bridges_passed=sum(row['passed'] for row in bridges), primary_pair=primary,
                    negative_control_pairs=negative, candidate=False, fitting_performed=False,
                    all_incompatible_pairs=incompatible,
                    incompatible_4B_negative_controls=[pair for pair in incompatible if pair['width'] == 4],
                    incompatible_dependent_controls=[pair for pair in incompatible if pair['dependent']],
                    old_dependency_counterexample_retained=True,
                    dependency_gap_8B_1warp_not_covered_by_new_16B_8warp_control=True,
                    existing_failed_forms_not_requalified=True, freeze_allowed=False,
                    holdout_allowed=False, physical_batches_measured=False,
                    isolated_sync_cost_measured=False)
    write(output / 'decision.json', decision)
    with (output / 'conditions.csv').open('w') as handle:
        columns = ['case', 'status', 'width', 'warps', 'dependent', 'layout', 'iterations',
                   'median_cycles', 'cv', 'absolute_cycle_tolerance', 'cycles_per_warp_instruction']
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--measured', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.package.resolve(), args.measured.resolve(), args.output.resolve())
