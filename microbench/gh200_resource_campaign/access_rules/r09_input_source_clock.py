#!/usr/bin/env python3
"""R09 source-clock development fit: all 21 conditions from job738376 train.

f=a-d_mode*Qmean/(1000*W_us)-e_mode*Smean_KiB/W_us*g(W_us/tau_us).
No GPU work, measured-cycle prediction inputs, DRAM term, or extra clock family.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics

import r09_v08_clock as clock
import v08_model

MODES = clock.INPUT_MODES
COEFFICIENTS = ('a', 'd_dyadic', 'd_zero', 'd_random', 'e_dyadic', 'e_zero', 'e_random')


def source_points(run, calibration):
    rows = {r['id']: r for r in clock.read_json(run / 'cases.json')}
    setups = {s['case']: s['setup'] for s in clock.read_json(run / 'static_setup.json')}
    points = []
    for old in clock.input_clock_points(run, calibration):
        row = rows[old['case']]
        cfg = row['config']
        feat = v08_model.features(row, setups[row['id']]['grid'])
        tm, tn, tk = v08_model.base.CONFIGS[cfg]['tile']
        cm, cn = v08_model.base.CONFIGS[cfg]['cluster']
        tiles = sum(map(len, feat['work']))
        compute_per_kt = 2 * tm * tn * tk / 4096
        source_per_kt = (2 * tm * tk / cn + 2 * tn * tk / cm) / 1024
        if (compute_per_kt, source_per_kt) != {'cfg_a': (512, 24), 'cfg_b': (512, 32), 'cfg_c': (1024, 32)}[cfg]:
            raise ValueError('unexpected Tensor/source demand definition')
        point = {key: old[key] for key in ('case', 'config', 'm', 'n', 'k', 'input_mode',
            'observed_window_us', 'observed_ghz', 'plain_us', 'observed_ends_cycles',
            'cycles', 'converted_cycles', 'fixed_us', 'converted_cycle_error')}
        point.update(original_purpose=old['purpose'], scheduled_tiles=tiles, kt=feat['kt'],
            sm_count=132, compute_cycles_per_kt=compute_per_kt, source_kib_per_kt=source_per_kt,
            tensor_mean_cycles=tiles * feat['kt'] * compute_per_kt / 132,
            source_mean_kib=tiles * feat['kt'] * source_per_kt / 132)
        points.append(point)
    return points


def max_clock_cap(run):
    evidence = {}
    for name in ('nvidia-smi-before.txt', 'nvidia-smi-after-ctrl.txt'):
        section = (run / name).read_text().split('\n    Max Clocks\n', 1)[1].split('\n    Max Customer', 1)[0]
        values = {key: int(re.search(r'^\s+' + key + r'\s+:\s+(\d+) MHz$', section, re.M)[1])
                  for key in ('Graphics', 'SM')}
        evidence[name] = dict(sha256=clock.sha256(run / name), max_clocks_mhz=values)
    return min(v for e in evidence.values() for v in e['max_clocks_mhz'].values()) / 1000, evidence


def source_design(points, tau):
    import numpy as np

    return np.array([[1,
        *[-p['tensor_mean_cycles'] / (1000 * p['observed_window_us']) * (p['input_mode'] == mode) for mode in MODES],
        *[-p['source_mean_kib'] / p['observed_window_us']
          * clock.duration_term(p['observed_window_us'], 'relaxation', tau)
          * (p['input_mode'] == mode) for mode in MODES]] for p in points])


def fit_source_clock(points, cap):
    import numpy as np
    from scipy.optimize import lsq_linear

    y = np.array([p['observed_ghz'] for p in points])

    def fit_tau(tau):
        design = source_design(points, tau)
        solved = lsq_linear(design, y, bounds=([1e-6, 0, 0, 0, 0, 0, 0], [cap, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                            method='bvls', tol=1e-12, max_iter=1000)
        if not solved.success or solved.optimality > 1e-7:
            raise ValueError('bounded source-clock fit failed KKT check: ' + solved.message)
        rank = clock.input_design_rank(design)
        return dict(tau_us=float(tau), coefficients=solved.x.tolist(),
            mse_ghz=float(np.mean((design @ solved.x - y) ** 2)),
            kkt_max_abs=float(solved.optimality), active_mask=solved.active_mask.tolist(), **rank)

    coarse = np.logspace(-1, 5, 121)
    profile = [fit_tau(tau) for tau in coarse]
    index = min(range(len(profile)), key=lambda i: profile[i]['mse_ghz'])
    # Refine only the neighboring log-tau bracket of the best training grid point.
    fine = np.geomspace(coarse[max(0, index - 1)], coarse[min(len(coarse) - 1, index + 1)], 81)
    known = {p['tau_us'] for p in profile}
    profile += [fit_tau(tau) for tau in fine if float(tau) not in known]
    profile.sort(key=lambda p: p['tau_us'])
    best = min(profile, key=lambda p: p['mse_ghz'])
    if best['rank'] != 7:
        raise ValueError('source-clock best design is rank deficient')
    tau, coefficients = best['tau_us'], best['coefficients']
    derivative = []
    for p in points:
        x = p['observed_window_us'] / tau
        e = coefficients[4 + MODES.index(p['input_mode'])]
        derivative.append(e * p['source_mean_kib'] / p['observed_window_us']
                          * (-math.expm1(-x) - x * math.exp(-x)) / x)
    jacobian = np.column_stack((source_design(points, tau), derivative))
    near = [p for p in profile if p['mse_ghz'] <= 1.1 * best['mse_ghz']]
    return dict(**best, coefficient_order=COEFFICIENTS, tau_profile=profile,
        jacobian_with_log_tau=clock.input_design_rank(jacobian),
        constraints=dict(a_min_ghz=1e-6, a_max_ghz=cap, d_e_nonnegative=True),
        active_constraints=[name + ('=upper' if mask > 0 else '=lower')
                            for name, mask in zip(COEFFICIENTS, best['active_mask']) if mask],
        near_minimum=dict(criterion='Grid MSE <= 1.10 * minimum; sensitivity range, not a confidence interval.',
            tau_min_us=min(p['tau_us'] for p in near), tau_max_us=max(p['tau_us'] for p in near),
            coefficients={name: [min(p['coefficients'][i] for p in near), max(p['coefficients'][i] for p in near)]
                          for i, name in enumerate(COEFFICIENTS)}),
        training_window_us=[min(p['observed_window_us'] for p in points), max(p['observed_window_us'] for p in points)])


def source_frequency(point, window, model):
    a, *costs = model['coefficients']
    mode = MODES.index(point['input_mode'])
    tensor = costs[mode] * point['tensor_mean_cycles'] / (1000 * window)
    source = costs[mode + 3] * point['source_mean_kib'] / window * clock.duration_term(window, 'relaxation', model['tau_us'])
    return dict(frequency_ghz=a - tensor - source, tensor_penalty_ghz=tensor, source_penalty_ghz=source)


def solve_source_envelope(point, model, envelope_us, *, rtol=1e-10, max_iter=200):
    """Couple the static Q/S clock to a caller's NS-recursion envelope, in us.

    Requires continuous, positive, nonincreasing E(f) for 0<f<=a, with
    E(f)->infinity as f->0 (positive work with a compute floor suffices).
    Since g is concave and g(0)=0, g(W)/W decreases: F(W) is nondecreasing.
    Thus h(f)=F(E(f))-f strictly decreases, h(0+)=a>0, h(a)<=0.
    Bisect the finite frequency bracket (0,a]; never evaluate E at zero or
    a negative clock mapping. Callback monotonicity is the caller's contract,
    not a claim that every overlapping event recursion has been proved so.

    Reads only input_mode/tensor_mean_cycles/source_mean_kib from point.
    Closure is F(returned W)/returned f - 1; returned W equals E(returned f).
    """
    a, *costs = model['coefficients']
    values = [a, *costs, model['tau_us'], point['tensor_mean_cycles'], point['source_mean_kib']]
    if (not all(math.isfinite(v) for v in values) or a <= 0 or min(costs) < 0
            or model['tau_us'] <= 0 or min(point['tensor_mean_cycles'], point['source_mean_kib']) < 0
            or not math.isfinite(rtol) or rtol <= 0 or max_iter < 1):
        raise ValueError('invalid source-envelope coefficient/work/solver domain')

    def evaluate(frequency):
        window = float(envelope_us(frequency))
        if not math.isfinite(window) or window <= 0:
            raise ValueError('envelope callback must return finite positive us')
        mapped = source_frequency(point, window, model)['frequency_ghz']
        if not math.isfinite(mapped):
            raise ValueError('nonfinite source-clock mapping')
        return window, mapped / frequency - 1

    lo, hi = 0.0, a  # zero is a symbolic limiting endpoint, never a callback input
    window, closure = evaluate(hi)
    if abs(closure) <= rtol:
        return dict(window_us=window, frequency_ghz=hi, closure=closure)
    for _ in range(max_iter):
        frequency = (lo + hi) / 2
        if frequency <= 0 or frequency in (lo, hi):
            break
        window, closure = evaluate(frequency)
        if abs(closure) <= rtol:
            return dict(window_us=window, frequency_ghz=frequency, closure=closure)
        if closure > 0:
            lo = frequency
        else:
            hi = frequency
    raise RuntimeError('source-envelope root did not converge; check callback contract')


def solve_source_clock(point, model):
    """The unique positive root lies in [target/a, (target+e*S)/a]."""
    a, *costs = model['coefficients']
    if not (a > 0 and min(costs) >= 0 and model['tau_us'] > 0
            and point['converted_cycles'] > 0 and point['tensor_mean_cycles'] >= 0 and point['source_mean_kib'] >= 0):
        raise ValueError('source-clock coefficient/work domain violated')
    mode = MODES.index(point['input_mode'])
    target = (point['converted_cycles'] + costs[mode] * point['tensor_mean_cycles']) / 1000
    source = costs[mode + 3] * point['source_mean_kib']
    if not (a > 0 and target > 0 and source >= 0):
        raise ValueError('source-clock root outside declared coefficient/work domain')
    lo, hi = target / a, (target + source) / a
    for _ in range(100):
        mid = (lo + hi) / 2
        residual = a * mid - source * clock.duration_term(mid, 'relaxation', model['tau_us']) - target
        if residual < 0:
            lo = mid
        else:
            hi = mid
    window = (lo + hi) / 2
    frequency = point['converted_cycles'] / (1000 * window)
    closure = source_frequency(point, window, model)['frequency_ghz'] / frequency - 1
    if abs(closure) > 1e-8 or not 0 < frequency <= a + 1e-9:
        raise ValueError('source-clock positive root failed closure/cap check')
    return dict(window_us=window, frequency_ghz=frequency, closure=closure,
                old_model_transfer_us=point['fixed_us'] + window)


def evaluate_source_clock(point, model):
    fitted = source_frequency(point, point['observed_window_us'], model)
    root = solve_source_clock(point, model)
    lo, hi = model['training_window_us']
    return dict(case=point['case'], config=point['config'], m=point['m'], n=point['n'], k=point['k'],
        input_mode=point['input_mode'], original_purpose=point['original_purpose'],
        **{'observed_fit_' + key: value for key, value in fitted.items()},
        observed_fit_frequency_error=fitted['frequency_ghz'] / point['observed_ghz'] - 1,
        free_window_us=root['window_us'], free_frequency_ghz=root['frequency_ghz'], closure=root['closure'],
        free_frequency_error=root['frequency_ghz'] / point['observed_ghz'] - 1,
        predicted_window_within_training_envelope=lo <= root['window_us'] <= hi,
        old_model_transfer_us=root['old_model_transfer_us'],
        old_model_transfer_time_error=root['old_model_transfer_us'] / point['plain_us'] - 1,
        converted_cycle_error=point['converted_cycle_error'])


def source_stats(rows):
    result = dict(n=len(rows), predicted_windows_outside_training=sum(not r['predicted_window_within_training_envelope'] for r in rows))
    for key in ('observed_fit_frequency_error', 'free_frequency_error', 'old_model_transfer_time_error', 'converted_cycle_error'):
        values = [r[key] for r in rows]
        result[key] = dict(rms=math.sqrt(statistics.fmean(v*v for v in values)),
                           maximum_abs=max(map(abs, values)), mean_signed=statistics.fmean(values))
    return result


def analyze_source_clock(run, calibration_run, old_long, output):
    import v06_run as common

    common.verify(run)
    common.verify(old_long)
    environment = clock.read_json(run / 'environment.json')
    if environment['gpu'].split(',')[0] != clock.read_json(old_long / 'environment.json')['gpu'].split(',')[0]:
        raise ValueError('source-clock development diagnostics must remain on one GPU')
    if clock.read_json(run / 'run_config.json')['cases_sha256'] != clock.sha256(run / 'cases.json'):
        raise ValueError('source-clock training matrix changed')
    calibration = clock.read_json(calibration_run / 'frozen/v08-predictions.json')['calibration']
    points = source_points(run, calibration)
    if len(points) != 21 or Counter(p['original_purpose'] for p in points) != {'calibration': 18, 'bridge': 3}:
        raise ValueError('unexpected all-21 source-clock development matrix')
    if any(p['k'] != 1024 for p in points):
        raise ValueError('source-clock version is scoped to the fixed K=1024 matrix')
    cap, evidence = max_clock_cap(run)
    model = fit_source_clock(points, cap)
    scores = [dict(evaluate_source_clock(p, model), phase='all21_training', fold='') for p in points]
    training_scores = scores[:]
    folds = []
    for scheme, key in (('geometry', 'm'), ('config', 'config')):
        for label in sorted({p[key] for p in points}):
            train = [p for p in points if p[key] != label]
            held = [p for p in points if p[key] == label]
            fitted = fit_source_clock(train, cap)
            evaluated = [dict(evaluate_source_clock(p, fitted), phase=scheme, fold=str(label)) for p in held]
            scores.extend(evaluated)
            folds.append(dict(scheme=scheme, held_group=label, training_cases=[p['case'] for p in train],
                              held_cases=[p['case'] for p in held], model=fitted, errors=source_stats(evaluated)))
    # Old cfg_a/c long data are read only after this version's fits/folds are fixed.
    old_points = [p for p in source_points(old_long, calibration) if p['config'] != 'cfg_b']
    old_scores = [dict(evaluate_source_clock(p, model), phase='old_cfg_ac_long_cross_job_development', fold='') for p in old_points]
    scores.extend(old_scores)
    result = dict(version='R09-input-source-clock-dev-v1', run=str(run), environment=environment,
        diagnostic_source_sha256=clock.sha256(Path(__file__)),
        dependency_sha256={name: clock.sha256(Path(__file__).parent / name)
                           for name in ('r09_v08_clock.py', 'v08_model.py', 'v06_model.py')},
        input_hashes={name: clock.sha256(run / name) for name in ('cases.json', 'static_setup.json', 'run_config.json',
                                                               'analysis/summary.json', 'analysis/bridge_comparison.json')},
        calibration_source=str(calibration_run), calibration_sha256=clock.sha256(calibration_run / 'frozen/v08-predictions.json'),
        old_long_source=str(old_long), old_long_summary_sha256=clock.sha256(old_long / 'analysis/summary.json'),
        training_cases=[p['case'] for p in points], train_n=21,
        archived_purpose_counts=dict(Counter(p['original_purpose'] for p in points)),
        former_bridges_are_training=True, archived_purposes_unchanged=True,
        formula='f_GHz=a-d_mode*Q_mean_cycles/(1000*W_us)-e_mode*S_mean_KiB/W_us*g(W_us/tau_us)',
        units=dict(a='GHz', d_mode='dimensionless', e_mode='GHz*us/KiB', Q='nominal Tensor cycles per SM averaged over 132 SM',
                   S='multicast-adjusted logical source KiB per SM averaged over 132 SM', W='us', tau='us'),
        max_clock_evidence=evidence, model=model,
        positive_root_certificate=dict(
            h='a*W-e_mode*S*g(W/tau)-(converted_cycles+d_mode*Q)/1000',
            assumptions='a>0, d/e>=0, Q/S>=0, converted_cycles>0, tau>0',
            g='g(0)=0, 0<=g<1, g increasing and concave',
            proof='H(0)<0, H convex, H tends to positive infinity; exactly one positive crossing. H need not be monotone near zero.',
            bracket='[(converted_cycles+d_mode*Q)/(1000*a), ((converted_cycles+d_mode*Q)/1000+e_mode*S)/a]'),
        diagnostics=dict(all21_training=source_stats(training_scores),
            per_config={cfg: source_stats([r for r in training_scores if r['config'] == cfg]) for cfg in ('cfg_a', 'cfg_b', 'cfg_c')},
            old_cfg_ac_long_cross_job_development=source_stats(old_scores)),
        folds=folds, points=dict(all21_training=points, old_cfg_ac_long=old_points),
        empirical_domain=dict(gpu=environment['gpu'].split(',')[0], configs=['cfg_a', 'cfg_b', 'cfg_c'],
            input_modes=MODES, seed=17, k=1024, geometries={'cfg_a': [2048, 8192], 'cfg_b': [2048, 8192, 20480], 'cfg_c': [2048, 8192]}),
        accepted_as_new_same_card_time_constants=False,
        notes=[
            'This new development version trains on all 21 conditions from job738376, including the three original bridges. Archive purpose fields and original measurements are unchanged.',
            'Former bridges are no longer independent checks of this version. Six old cfg_a/c long cases are known cross-job development diagnostics, never training or tau selection.',
            'One candidate family only: instantaneous nominal Tensor work rate plus duration-filtered multicast-adjusted source request rate. No DRAM term, config offset, or per-input tau.',
            'Q/S use total scheduled tiles divided by all 132 SM, including idle-SM averaging. Conditional clock features use no target measured cycles or old cycle-model features.',
            'Activities are whole-device per-SM averages, whereas the target frequency aggregates maximum-work CTA cycle/ns and W is an ends envelope; these are conditional proxy scopes.',
            'Logical source requests derive from (2*tileM*tileK/clusterN + 2*tileN*tileK/clusterM). They are not physical L2/HBM bytes or measured power.',
            'Observed W is permitted only in fitting/conditional evaluation. Every free solution uses predicted W and old V08 converted cycles.',
            'All free time estimates retain old 099dda56 C/F/kappa and are old-model transfer diagnostics, never newly established a057 constants or a full-time validation pass.',
            'Tau uses the original 0.1..100000 us/121-point scan and an 81-point refinement inside the best training grid point neighboring bracket, separately inside every fold.',
            'Near-minimum profile ranges use MSE <= 1.10*minimum and are sensitivity ranges, not confidence intervals.',
            'Geometry/config folds are conditional development checks on already-known data; K is fixed and cannot be cross-validated.',
            'Positive roots are unique from the declared coefficient/work domain; no a>b constraint or post-prediction frequency clamp is needed.',
        ])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source-clock.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for filename, rows in [('source-clock-scores.csv', scores), ('source-clock-profile.csv', model['tau_profile'])]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('all21 source-clock training:', result['diagnostics']['all21_training'])
    print('tau us:', model['tau_us'], 'coefficients:', model['coefficients'])
    print('output:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--calibration-run', type=Path, required=True)
    parser.add_argument('--old-long-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze_source_clock(args.run.resolve(), args.calibration_run.resolve(), args.old_long_run.resolve(), args.output.resolve())
