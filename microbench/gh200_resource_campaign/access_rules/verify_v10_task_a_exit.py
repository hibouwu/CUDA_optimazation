#!/usr/bin/env python3
"""Verify the task-A exit without fitting a price for a rejected boundary form."""
import argparse
import copy
import json
from pathlib import Path

import supply_model
import v09_model
from v06_run import sha


def verify(base, clock_path, comparison_path, plan_path, output):
    freeze = base/'Q15-full-time/V09-component-validation/20261009-V09-freeze-job738972-r2'
    paths = [freeze/'cases.json', freeze/'static_setup.json',
             freeze/'frozen/calibration.json', clock_path, comparison_path, plan_path]
    rows = json.loads(paths[0].read_text())
    setups = {r['case']:r['setup'] for r in json.loads(paths[1].read_text())}
    calibration = json.loads(paths[2].read_text())
    clock_artifact = json.loads(clock_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    if comparison['clock_adoption']['clock']['passed'] or comparison['clock_adoption']['clock_shape_fold']['passed']:
        raise ValueError('the proposed exit requires both clock adoption checks to fail')
    fold_clocks = {tuple(f['heldout_shape']):f['clock'] for f in clock_artifact['folds']}
    statements = []
    for row in rows:
        if row['set'] != 'heldout' or not (row['group']=='G6_base' or row['group'].startswith('P2_')):
            continue
        features = supply_model.boundary_fill_features(row,setups[row['id']])
        first = [f for f in features if f['j']==0]
        if any(f['operands'][op]['cluster_max'] != 0 for f in first for op in ('A','B')):
            raise ValueError('a supposedly invariant condition has first-cluster fill: ' + row['id'])
        work = v09_model.events.scheduled_work(row['config'],row['m'],row['n'],
                                              setups[row['id']]['grid'],row['swizzle'])
        extra = [0.]*len(work)
        checks = {}
        for name,clock in (('r2',calibration['clock']),('pooled60',clock_artifact['clock']),
                           ('clock_shape_fold',fold_clocks[(row['m'],row['n'])])):
            candidate = copy.deepcopy(calibration)
            candidate['clock'] = clock
            plain = v09_model.predict_components(row,setups[row['id']],candidate)
            combined = v09_model.predict_components(row,setups[row['id']],candidate,
                                                    first_loop_extra_ns=extra)
            if json.dumps(plain,sort_keys=True) != json.dumps(combined,sort_keys=True):
                raise ValueError('zero first-L correction changed the full prediction')
            checks[name] = dict(complete_prediction_equal=True,plain_us=plain['plain_us'],
                               frequency_ghz=plain['frequency_ghz'])
        statements.append(dict(case=row['id'], config=row['config'],
            first_ctas=len(first), first_fill_ctas=0,
            later_fill_tiles=sum(f['j']>0 and any(f['operands'][op]['cluster_max']>0 for op in ('A','B')) for f in features),
            price_independent_reason='Every first fill feature is zero, so beta times Ktiles times that feature is zero for every finite beta.',
            checks=checks))
    if len(statements) != 4:
        raise ValueError('all three G6 configurations and P2 must be retained')
    # In cfg_a's cooperative recurrence, each L is window + kt*max(C/f, service).
    # All other event and output terms are frequency-independent, the end scale
    # is 1+xk, and plain F+kappa*envelope preserves monotonicity for kappa>0.
    # Thus higher f cannot increase this frozen model's predicted plain time.
    a = next(r for r in comparison['cases'] if r['case']=='v09d_cfg_a_G6_base')
    old = a['versions']['r2']
    parameters = calibration['events_ns']['cfg_a']
    transfer = calibration['plain_transfer']['cfg_a']
    if (v09_model.events.base.CONFIGS['cfg_a']['schedule']!='cooperative'
            or parameters['xk'] < -1 or transfer['kappa'] <= 0
            or any(calibration[key]['cfg_a']['unit']!='ns' for key in ('supply','first_supply'))
            or old['frequency_error_pct'] >= 0 or old['plain_error_pct'] >= 0):
        raise ValueError('clock-only incompatibility assumptions changed')
    conflict = dict(case=a['case'], baseline_frequency_error_pct=old['frequency_error_pct'],
        baseline_plain_error_pct=old['plain_error_pct'], end_scale=1+parameters['xk'],
        plain_kappa=transfer['kappa'],
        proof='The baseline frequency and time are both below their observations. Improving frequency absolute error requires f_new > f_old. With frozen ns services, cooperative events and positive plain transfer, T(f_new) <= T(f_old) < T_observed, so the absolute time error cannot improve.',
        consequence='More clock training alone cannot meet both G6-a target checks; changing the service/event model or the adopted target rule requires a separately authorized task.',
        physical_claim=False)
    result = dict(scope='Post-V09 task-A development; not a validation or general impossibility claim.',
        input_sha256={str(p):sha(p) for p in paths},
        implementation_sha256={str(Path(m.__file__)):sha(Path(m.__file__)) for m in (v09_model,supply_model)},
        verifier_sha256=sha(Path(__file__)), gpu_jobs=0,
        terminal='no_freezable_candidate_under_the_declared_forms_and_adoption_rules',
        reason='The expanded-clock model fails G6-a and the shape fold fails G6-b; the sole boundary correction is exactly zero on every G6 and P2, so no boundary price can repair these failures.',
        invariants=statements, clock_target_incompatibility=conflict,
        clock_adoption=comparison['clock_adoption'],
        historical_scores_unchanged=True,new_holdout=False,cross_card_qualification=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(terminal=result['terminal'],conditions=len(statements),
                         exact_full_prediction_checks=3*len(statements),gpu_jobs=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--clock',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verify(args.base.resolve(),args.clock.resolve(),args.comparison.resolve(),args.plan.resolve(),args.output.resolve())
