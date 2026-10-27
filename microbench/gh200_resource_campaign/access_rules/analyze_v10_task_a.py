#!/usr/bin/env python3
"""Fixed V09 development comparison for PLAN task A. CPU only; no refitting.

The original 29 scored conditions remain in every comparison, including seven
outside the proposed scope. G2 stays a separate historical rejection. This
analysis never grants holdout, observation, or cross-card qualification.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics

import analyze_v09_gate_audit as audit
from analyze_r13_sm import select_attempts
from analyze_r18 import replay
from run_v09 import supply_identification, check_prediction_support
from v06_run import sha
import v09_model


DEFAULT_STAGES = {'cfg_a': 6, 'cfg_b': 6, 'cfg_c': 4}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scope_reasons(row, setup):
    """Static exclusions only. Calibration coverage and LP support are separate."""
    reasons = []
    for name in ('lda', 'ldb'):
        if row[name] * 2 % 128:
            reasons.append(name + ' is not 128 B aligned')
    if setup['stages'] != DEFAULT_STAGES[row['config']]:
        reasons.append('non-default stage count')
    return reasons


def adoption(baseline, candidate, targets, proposed_supported):
    """Errors are absolute percentages; missing rows fail instead of shrinking n."""
    fixed = set(baseline)
    targets = set(targets)
    if not targets or not targets <= fixed or not set(proposed_supported) <= fixed:
        raise ValueError('target/support identities must belong to the fixed comparison')
    violations = []
    missing = sorted(fixed - set(candidate))
    extra = sorted(set(candidate) - fixed)
    if missing or extra:
        violations.append(dict(reason='comparison identity mismatch', missing=missing, extra=extra))
    values = {}
    for case in sorted(fixed):
        old = baseline[case]
        new = candidate.get(case)
        if not isinstance(old, (int, float)) or not math.isfinite(old) or old < 0:
            raise ValueError('invalid baseline error: ' + case)
        if (isinstance(new, bool) or not isinstance(new, (int, float))
                or not math.isfinite(new) or new < 0):
            violations.append(dict(case=case, reason='missing or invalid candidate error'))
            continue
        values[case] = new
        if case in targets and new >= old:
            violations.append(dict(case=case, reason='target did not improve', old=old, new=new))
        if new - old > 2 + 1e-9:
            violations.append(dict(case=case, reason='regression exceeds 2 percentage points', old=old, new=new))
        if case in proposed_supported and old <= 10 and new > 10 + 1e-9:
            violations.append(dict(case=case, reason='supported condition crossed 10 percent', old=old, new=new))
    others = sorted(fixed - targets)
    other_summary = None
    if others and all(case in values for case in others):
        old = [baseline[case] for case in others]
        new = [values[case] for case in others]
        other_summary = dict(baseline_median=statistics.median(old),
            candidate_median=statistics.median(new), baseline_maximum=max(old), candidate_maximum=max(new))
        if statistics.median(new) > statistics.median(old) + 1e-9:
            violations.append(dict(reason='non-target median regressed'))
        if max(new) > max(old) + 1e-9:
            violations.append(dict(reason='non-target maximum regressed'))
    return dict(passed=not violations, fixed_conditions=len(fixed),
                targets=sorted(targets), non_target_errors=other_summary, violations=violations)


def case_components(processes):
    result = {}
    for name in audit.COMPONENTS:
        present = [p['components'][name] for p in processes
                   if p['components'][name]['median_abs_pct'] is not None]
        result[name] = dict(processes=len(present),
            median_abs_pct=statistics.median(p['median_abs_pct'] for p in present) if present else None,
            signed_median_pct=statistics.median(p['signed_median_pct'] for p in present) if present else None,
            max_window_abs_pct=max(p['max_abs_pct'] for p in present) if present else None)
    return result


def run(base, output, clock_path=None):
    freeze = base / 'Q15-full-time/V09-component-validation/20261009-V09-freeze-job738972-r2'
    observed = base / 'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011'
    paths = [freeze/'frozen/calibration.json', freeze/'frozen/predictions.json',
             freeze/'frozen/manifest.json', freeze/'cases.json', freeze/'static_setup.json',
             observed/'reanalysis/manager-results-v1/summary.json']
    calibration = read(paths[0])
    frozen = read(paths[1])['predictions']
    rows = [r for r in read(paths[3]) if r['set'] == 'heldout']
    setups = {s['case']: s['setup'] for s in read(paths[4])}
    observations = {c['case']: c for c in read(paths[5])['cases']}
    versions = {'r2': calibration}
    fold_clocks = {}
    if clock_path:
        clock_artifact = read(clock_path)
        candidate = copy.deepcopy(calibration)
        candidate['clock'] = clock_artifact['clock']
        versions['clock'] = candidate
        for fold in clock_artifact.get('folds',[]):
            shape = tuple(fold['heldout_shape'])
            if shape in fold_clocks:
                raise ValueError('duplicate clock shape fold')
            fold_clocks[shape] = fold
        if fold_clocks:
            versions['clock_shape_fold'] = None
        paths.append(clock_path)
    fixed = [r['id'] for r in rows if frozen[r['id']]['status'] == 'predicted']
    if len(rows) != 30 or len(fixed) != 29:
        raise ValueError('the frozen 30/29 comparison must remain intact')
    output.mkdir(parents=True, exist_ok=False)
    policy = dict(scope='Post-V09 development; no new validation', fixed_conditions=fixed,
        original_rejected=[r['id'] for r in rows if r['id'] not in fixed],
        proposed_supported=[r['id'] for r in rows if r['id'] in fixed and not scope_reasons(r,setups[r['id']])],
        exclusions={r['id']:scope_reasons(r,setups[r['id']]) for r in rows},
        clock_targets=[r['id'] for r in rows if r['group'] == 'G6_base'],
        boundary_targets=[r['id'] for r in rows if r['group'].startswith(('P1_', 'P2_'))],
        target_policy='Every predeclared target must improve; no averaging across configurations.',
        regression_budget_percentage_points=2,
        arithmetic_tolerance_percentage_points=1e-9,
        component_aggregation='Absolute window errors before CTA median, then process median.',
        missing_policy='Missing prediction fails the fixed comparison; no subset scoring.',
        full_goal='Component/critical observations and cross-card qualification remain outstanding.')
    write(output/'policy.json',policy)
    identification = supply_identification(freeze,calibration)
    reports = []
    baseline_errors = {}
    errors_by_version = {name:{} for name in versions}
    hashes = {str(p):sha(p) for p in paths}
    for row in rows:
        case = row['id']
        saved = frozen[case]
        result = dict(case=case, config=row['config'], group=row['group'],
            original_status=saved['status'], scope_reasons=policy['exclusions'][case], versions={})
        selected, failed, _ = select_attempts(observed,case,('dual',))
        if len(selected) != 10:
            raise ValueError('ten complete dual processes required: ' + case)
        traces = []
        for item in selected:
            record = item['record']
            traces.append(replay(observed,record,row)['ctas'])
            hashes[str(observed/item['path'])] = sha(observed/item['path'])
            hashes[str(observed/record['raw'])] = record['raw_sha256']
        result['failed_attempts_retained'] = failed
        for name, parameters in versions.items():
            if name == 'clock_shape_fold':
                fold = fold_clocks.get((row['m'],row['n']))
                if fold is None or fold.get('clock') is None:
                    result['versions'][name] = dict(error='missing/unidentified clock shape fold',
                        plain_error_pct=None,frequency_error_pct=None,
                        final_frequency_supply_identified=False)
                    if case in fixed:
                        errors_by_version[name][case] = None
                    continue
                parameters = copy.deepcopy(calibration)
                parameters['clock'] = fold['clock']
            prediction = v09_model.predict_components(row,setups[case],parameters)
            if name == 'r2' and saved['status'] == 'predicted':
                serialized = json.loads(json.dumps(prediction,allow_nan=False))
                mismatches = [k for k,v in serialized.items() if saved.get(k) != v]
                if mismatches:
                    raise ValueError('r2 frozen prediction changed: ' + case + ' ' + ','.join(mismatches))
            unsupported = check_prediction_support(row,setups[case],parameters,prediction,identification)
            processes = [audit.process_audit(ctas,prediction) for ctas in traces]
            plain_error = 100*(prediction['plain_us']/observations[case]['observed_us'] - 1)
            rate_error = 100*(prediction['frequency_ghz']/observations[case]['observed_frequency_ghz'] - 1)
            summary = dict(plain_us=prediction['plain_us'], plain_error_pct=plain_error,
                frequency_ghz=prediction['frequency_ghz'], frequency_error_pct=rate_error,
                converted_cycle_error_pct=100*((prediction['plain_us']*prediction['frequency_ghz'])/
                    (observations[case]['observed_us']*observations[case]['observed_frequency_ghz'])-1),
                final_frequency_supply_identified=not unsupported, unsupported_windows=unsupported,
                components=case_components(processes),
                critical=dict(candidate_count=len(prediction['critical_candidates']),
                    representative=prediction['critical_cta'],
                    set_hit_processes=sum(p['critical']['frozen_set_intersects_exact_last'] for p in processes),
                    representative_hit_processes=sum(p['critical']['chosen_is_exact_last'] for p in processes),
                    median_end_regret_percent=statistics.median(p['critical']['end_regret_percent'] for p in processes)),
                processes=processes)
            result['versions'][name] = summary
            if case in fixed:
                errors_by_version[name][case] = abs(plain_error)
                if name == 'r2':
                    baseline_errors[case] = abs(plain_error)
        reports.append(result)
    result = dict(scope=policy['scope'], policy=policy, input_sha256=hashes,
        implementation_sha256={str(Path(m.__file__)):sha(Path(m.__file__)) for m in
            (v09_model,v09_model.supply,v09_model.clock,v09_model.output_model,audit)},
        analysis_sha256=sha(Path(__file__)), r2_all_scored=dict(n=29,
            median_abs_pct=statistics.median(baseline_errors.values()), max_abs_pct=max(baseline_errors.values())),
        r2_proposed_scope=dict(n=len(policy['proposed_supported']),
            median_abs_pct=statistics.median(baseline_errors[c] for c in policy['proposed_supported']),
            max_abs_pct=max(baseline_errors[c] for c in policy['proposed_supported'])), cases=reports)
    if clock_path:
        result['clock_adoption'] = {}
        for name in versions:
            if name == 'r2':
                continue
            time_check = adoption(baseline_errors,errors_by_version[name],
                                  policy['clock_targets'],policy['proposed_supported'])
            targets = []
            for row in reports:
                if row['case'] not in policy['clock_targets']:
                    continue
                old = abs(row['versions']['r2']['frequency_error_pct'])
                value = row['versions'][name]['frequency_error_pct']
                targets.append(dict(case=row['case'],baseline_abs_pct=old,
                    candidate_abs_pct=abs(value) if value is not None else None,
                    improved=value is not None and abs(value) < old))
            rejected = [r['case'] for r in reports if r['case'] in policy['proposed_supported']
                        and not r['versions'][name]['final_frequency_supply_identified']]
            result['clock_adoption'][name] = dict(
                passed=time_check['passed'] and all(t['improved'] for t in targets) and not rejected,
                time=time_check,frequency_targets=targets,final_frequency_lp_rejections=rejected,
                scope='Sample-in development' if name=='clock' else 'New component leave-shape diagnostic')
    write(output/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('r2_all_scored','r2_proposed_scope')},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--clock',type=Path)
    args = parser.parse_args()
    run(args.base.resolve(),args.output.resolve(),args.clock.resolve() if args.clock else None)
