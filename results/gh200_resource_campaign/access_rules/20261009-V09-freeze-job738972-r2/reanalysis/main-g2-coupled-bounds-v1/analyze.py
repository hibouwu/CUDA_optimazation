"""Conditional shared-parameter G2 bounds using the frozen V09 model, no samples."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import linprog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def read(relative):
        path = run / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    manifest = read('frozen/manifest.json')
    calibration = read('frozen/calibration.json')
    original = read('frozen/predictions.json')
    cases = read('cases.json')
    setups = read('freeze-setup.json')
    source_hashes = read('source_hashes.json')
    for name, digest in hashes.items():
        if name in manifest['files']:
            assert digest == manifest['files'][name], name
    sys.path.insert(0, str(run / 'source'))
    import v09_model as v09
    supply, clock, events = v09.supply, v09.clock, v09.events
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(run / 'source'):
            relative = str(Path(file).resolve().relative_to(run))
            digest = hashlib.sha256(Path(file).read_bytes()).hexdigest()
            assert source_hashes[relative] == digest
            assert hashes['source_hashes.json'] == manifest['files']['source_hashes.json']
            hashes[relative] = digest
    training_path = run / 'inputs/supply-training.json.gz'
    hashes['inputs/supply-training.json.gz'] = hashlib.sha256(training_path.read_bytes()).hexdigest()
    assert hashes['inputs/supply-training.json.gz'] == manifest['files']['inputs/supply-training.json.gz']
    with gzip.open(training_path, 'rt') as stream:
        training = json.load(stream)['first']['cfg_c']
    case = next(row for row in cases if row['id'] == 'v09d_cfg_c_G2_base')
    setup = next(row['setup'] for row in setups if row['case'] == case['id'])
    assert original['predictions'][case['id']]['status'] == 'unsupported'
    requests = supply.supply_request_rows(case, setup)
    work = events.scheduled_work('cfg_c', case['m'], case['n'], setup['grid'], case['swizzle'])
    assert len(requests) == len(work) == 64
    assert all(len(cta) == 1 for cta in work)
    assert all(row['j'] == 0 and row['T'] == 1 and row['kt'] == 16 for row in requests)
    model = calibration['first_supply']['cfg_c']
    identification = supply.supply_identification_bounds(training, model)
    constraints = identification['constraints']
    parameters = np.array(model['parameters'])
    projection = np.array(model['projection'])
    z = np.array([row['X'] for row in requests]) @ projection
    n = len(parameters)
    representative = v09.predict_components(case, setup, calibration)
    point = representative['static_clock_work']
    source_clock = copy.deepcopy(calibration['clock'])
    zero = representative['zero_product_fraction']
    if calibration['zero_activity_mixing']:
        mode_index = clock.MODES.index(point['input_mode'])
        zero_index = clock.MODES.index('zero')
        for offset in (1, 1 + len(clock.MODES)):
            source_clock['coefficients'][offset + mode_index] = (
                (1-zero) * source_clock['coefficients'][offset + mode_index]
                + zero * source_clock['coefficients'][offset + zero_index])

    # Frozen cooperative, one-output recurrence: D=(1+xk)*L+offset.
    p = dict(calibration['events_ns']['cfg_c'])
    rule = calibration['output_rules']['cfg_c']
    e0 = max(rule['E0_single']['floor_ns'], rule['E0_single']['beta_ns_per_MiB']
             * representative['first_output_bytes'] / 2**20) + rule['E0_single']['R_ns']
    p['Etail'] = rule['Etail_single_mean_ns']
    alpha = 1 + p['xk']
    offset_ns = alpha * (p['P0'] + p['S'] + p['w'] + e0 + p['Etail']) + p['x0'] + p['x1']
    assert alpha > 0
    for L in (0., 1., 9000., 1e6):
        actual = events.cta_cycles(p, 'cooperative', 1, 16, mainloops=[L], epilogues=[e0])
        assert abs(actual - (offset_ns + alpha*L)) <= 1e-8
    unique_z = np.unique(z, axis=0)
    compute = requests[0]['compute_cycles']
    assert all(row['compute_cycles'] == compute for row in requests)
    kt = 16
    options = dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)
    lp_calls = 0

    def solve_lp(objective, **kwargs):
        nonlocal lp_calls
        lp_calls += 1
        fit = linprog(objective, **kwargs, bounds=(0, None), method='highs', options=options)
        if not fit.success:
            raise ValueError(fit.message)
        return fit

    def envelope(frequency, which, details=False):
        # Shared theta for every CTA. Each max branch is linear in theta.
        branches = []
        for features in unique_z:
            branches.append((np.r_[alpha, np.zeros(n-1)], offset_ns + alpha*kt*compute/frequency))
            branches.append((np.r_[alpha, alpha*kt*features], offset_ns))
        if which == 'minimum':
            args_lp = dict(
                A_eq=[row+[0] for row in constraints['A_eq']], b_eq=constraints['b_eq'],
                A_ub=[row+[0] for row in constraints['A_ub']] + [np.r_[a, -1].tolist() for a,b in branches],
                b_ub=constraints['b_ub'] + [-b for a,b in branches])
            fit = solve_lp(np.r_[np.zeros(n), 1], **args_lp)
            value, theta = fit.fun, fit.x[:n]
        else:
            fits = [(solve_lp(-a, **constraints), b) for a,b in branches]
            fit, constant = max(fits, key=lambda pair: -pair[0].fun+pair[1])
            value, theta = -fit.fun+constant, fit.x
        actual = max(float(a@theta+b) for a,b in branches)
        assert abs(actual-value) < 1e-5
        return (value/1000, theta) if details else value/1000

    results = {}
    for which in ('minimum', 'maximum'):
        solved = clock.solve_source_envelope(point, source_clock, lambda f: envelope(f, which))
        value, theta = envelope(solved['frequency_ghz'], which, True)
        candidate = copy.deepcopy(calibration)
        candidate['first_supply']['cfg_c']['parameters'] = theta.tolist()
        replay = v09.predict_components(case, setup, candidate)
        assert abs(replay['window_us'] - solved['window_us']) < 1e-7
        assert abs(replay['frequency_ghz'] - solved['frequency_ghz']) < 1e-7
        eq_error = float(np.max(np.abs(np.array(constraints['A_eq'])@theta-constraints['b_eq'])))
        ub_error = float(max(0, np.max(np.array(constraints['A_ub'])@theta-constraints['b_ub'])))
        assert eq_error < 1e-7 and ub_error < 1e-7 and min(theta) >= 0
        # Independently verify unchanged fitted training values and branch inequalities.
        train_z = np.array([row['X'] for row in training]) @ projection
        floors = np.array([row['calibration_floor_ns'] for row in training])
        train_kt = np.array([row['kt'] for row in training])
        before = parameters[0]+train_kt*np.maximum(floors,train_z@parameters[1:])
        after = theta[0]+train_kt*np.maximum(floors,train_z@theta[1:])
        training_error = float(np.max(np.abs(before-after)))
        assert training_error < 1e-6
        loops, _ = supply.supply_predict(requests, candidate['first_supply']['cfg_c'], replay['frequency_ghz'])
        assert max(abs(np.array(replay['cta_duration_ns'])-(offset_ns+alpha*loops))) < 1e-7
        results[which] = dict(**solved, plain_us=replay['plain_us'], dual_event_us=replay['dual_event_us'],
            shared_parameters=theta.tolist(), critical_candidates=replay['critical_candidates'],
            L_ns=loops.tolist(), cta_duration_ns=replay['cta_duration_ns'],
            equality_residual_max=eq_error, inequality_violation_max=ub_error,
            training_prediction_change_ns_max=training_error,
            original_predictor_replay_window_us=replay['window_us'])
    lower, upper = results['minimum'], results['maximum']
    assert lower['window_us'] - 1e-7 <= representative['window_us'] <= upper['window_us'] + 1e-7
    # Original fixed-f bounds are a distinct check, not the coupled result.
    fixed = supply.supply_predict_bounds(requests, model, 1.86, identification)
    transfer = calibration['plain_transfer']['cfg_c']
    report = dict(case=case, input_sha256=hashes,
        original_frozen_result=original['predictions'][case['id']],
        interpretation='Conditional identified set within the frozen training active pattern. Numerical LP/root bounds, not confidence intervals or physical qualification.',
        scope='Only first-supply equivalent parameters vary jointly. Frozen events/output/clock/activity mixing/plain transfer stay fixed. No heldout samples read.',
        source_clock=dict(tau_us=source_clock['tau_us'],coefficients=source_clock['coefficients']),
        static_clock_work=point, zero_product_fraction=zero, training_rows=len(training),
        unique_request_features=unique_z.tolist(), recurrence=dict(alpha=alpha,offset_ns=offset_ns),
        identification=identification, coupled=results,
        representative={k:representative[k] for k in ('window_us','frequency_ghz','plain_us')},
        fixed_1p86ghz_L_bounds=fixed,
        relative_plain_interval_width=(upper['plain_us']/lower['plain_us']-1),
        original_plain_transfer=transfer, lp_calls=lp_calls,
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prediction_qualified=False, observation_qualified=False,
        limitations=['No parameter uncertainty from other components, model misspecification, clock measurement, observer transfer or cross-card transfer is covered.',
                    'Endpoint critical sets are algebraic ties in the old zero-entry-skew model, not observed or qualified last-CTA sets.',
                    'The frozen unsupported result is unchanged. No interval midpoint or endpoint is accepted as a point prediction.',
                    'This interval is retrospective development evidence, not a new independent holdout.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({which:{k:r[k] for k in ('window_us','frequency_ghz','plain_us','training_prediction_change_ns_max')} for which,r in results.items()},indent=2))


if __name__ == '__main__':
    main()
