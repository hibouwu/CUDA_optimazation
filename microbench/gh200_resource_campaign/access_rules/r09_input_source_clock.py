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
from clock_model import (MODES, COEFFICIENTS, source_frequency,
                         solve_source_envelope, solve_source_clock)
import v08_model


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
                           for name in ('clock_model.py', 'r09_v08_clock.py', 'v08_model.py', 'v06_model.py')},
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


def fit_tensor_only(points, cap):
    """Delete the source term; fit only the original observed-frequency targets."""
    import numpy as np
    from scipy.optimize import lsq_linear

    design = source_design(points, 1)[:, :4]
    y = np.asarray([p['observed_ghz'] for p in points])
    solved = lsq_linear(design, y, bounds=([1e-6, 0, 0, 0], [cap, np.inf, np.inf, np.inf]),
                        method='bvls', tol=1e-12, max_iter=1000)
    if not solved.success or solved.optimality > 1e-7:
        raise ValueError('tensor-only observed-frequency fit failed')
    return dict(coefficients=solved.x.tolist() + [0., 0., 0.], tau_us=1.,
                target='observed GHz, no total-time or measured-cycle target',
                rank=int(np.linalg.matrix_rank(design)), kkt_max_abs=float(solved.optimality))


def mixed_failure_clock(model, mode, zero_fraction, enabled=True):
    """Apply the same zero-work hypothesis as the frozen predictor, without mutation."""
    result = dict(model, coefficients=model['coefficients'][:])
    if enabled:
        index, zero = MODES.index(mode), MODES.index('zero')
        for offset in (1, 1 + len(MODES)):
            result['coefficients'][offset + index] = ((1-zero_fraction) * model['coefficients'][offset+index]
                                                     + zero_fraction * model['coefficients'][offset+zero])
    return result


def failure_stats(values):
    values = list(values)
    if not values:
        return dict(n=0)
    return dict(n=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values),
                rms=math.sqrt(statistics.fmean(x*x for x in values)), positive=sum(x > 0 for x in values))


def role_window_diagnostics(ctas):
    """Keep consumer roles separate; no cross-role max/min frequency ratios."""
    from collections import defaultdict
    windows = defaultdict(list)
    boundary = defaultdict(list)
    mismatch = Counter()
    quantum = 0
    for c in ctas:
        roles = c['roles']
        boundary['P0'].append((c['prod_c']-c['entry_c'], c['prod_ns']-c['entry_ns']))
        work_order = {tuple(coord): j for j, coord in enumerate(c.get('work', []))}
        for role_index, role in enumerate(roles):
            cyc, ns = role['tiles'], role['tiles_ns']
            for j, (a, b) in enumerate(zip(cyc, ns)):
                for value in b:
                    quantum = math.gcd(quantum, int(value))
                global_j = work_order[tuple(c['role_work'][role_index][j])] if work_order else j
                windows['first_L' if global_j == 0 else 'later_L'].append((a[1]-a[0], b[1]-b[0]))
                windows['E'].append((a[3]-a[2], b[3]-b[2]))
            if cyc:
                windows['role_active'].append((role['final_c']-cyc[0][0], role['final_ns']-ns[0][0]))
                windows['tail'].append((role['final_c']-cyc[-1][3], role['final_ns']-ns[-1][3]))
                boundary['S_to_role'].append((cyc[0][0]-c['prod_c'], ns[0][0]-c['prod_ns']))
        # Cooperative consumers share work. Pingpong roles own disjoint tiles.
        if c.get('role_work') and c['role_work'][0] == c['role_work'][1]:
            for left, right, ln, rn in zip(roles[0]['tiles'], roles[1]['tiles'],
                                           roles[0]['tiles_ns'], roles[1]['tiles_ns']):
                for index in range(4):
                    choose = min if index == 0 else max
                    cv, nv = (left[index], right[index]), (ln[index], rn[index])
                    cs = {i for i, x in enumerate(cv) if x == choose(cv)}
                    ns = {i for i, x in enumerate(nv) if x == choose(nv)}
                    mismatch['endpoints'] += 1
                    mismatch['different_role'] += not bool(cs & ns)

    def summarize(groups):
        result = {}
        for name, pairs in groups.items():
            # 3.2 us is a reporting cutoff, not a claim of hardware precision.
            usable = [(c, n) for c, n in pairs if c > 0 and n >= 3200]
            result[name] = dict(windows=len(pairs), frequency_windows=len(usable),
                cycle=failure_stats(c for c, n in pairs), ns=failure_stats(n for c, n in pairs),
                ghz=failure_stats(c/n for c, n in usable),
                duration_weighted_ghz=sum(c for c, n in usable)/sum(n for c, n in usable) if usable else None)
        return result
    return dict(same_role=summarize(windows), cross_role_boundary=summarize(boundary),
                merged_endpoint_selection=dict(mismatch), globaltimer_timestamp_gcd_ns=quantum,
                frequency_min_window_ns=3200)


def failure_component_errors(ctas, prediction):
    from collections import defaultdict
    errors = defaultdict(list)
    for c, p in zip(ctas, prediction['cta_events']):
        a = c['tiles_ns']
        if not a:
            continue
        e = p['events']
        pairs = [('prefix', a[0][0]-c['entry_ns'], p['supply']),
                 ('prefill', a[0][0]-c['prod_ns'], p['prefill']),
                 ('first_L', a[0][1]-a[0][0], e[0][1]-e[0][0]),
                 ('last_E', a[-1][3]-a[-1][2], e[-1][3]-e[-1][2]),
                 ('tail', c['end_ns']-max(t[3] for t in a), p['tail'])]
        pairs += [('later_L', x[1]-x[0], y[1]-y[0]) for x, y in zip(a[1:], e[1:])]
        for key, actual, predicted in pairs:
            if actual > 0:
                errors[key].append(predicted/actual-1)
    return {k: statistics.median(v) for k, v in errors.items()}


def observer_pair_inventory(base):
    """Ask whether an existing successful ends/dual pair isolates instrumentation."""
    from analyze_r13_sm import select_attempts
    fields = ('config', 'm', 'n', 'k', 'lda', 'ldb', 'ldd', 'storage_m', 'storage_n',
              'zero_m', 'zero_n', 'swizzle', 'input_mode', 'seed', 'grid', 'stages',
              'allocated_a_elements', 'allocated_b_elements', 'alloc_lda', 'alloc_ldb',
              'input_map_m', 'input_map_n', 'gpu_uuid')
    records = {'ends': [], 'dual': []}
    hashes = {}
    for root in sorted(Path(base).glob('20261009-*job*')):
        path = root/'static_setup.json'
        if not path.exists():
            continue
        hashes[str(path)] = clock.sha256(path)
        for item in clock.read_json(path):
            case = item['case']
            for kind in records:
                if not any((root/'samples'/case).glob(kind+'-*.json')):
                    continue
                selected, _, _ = select_attempts(root, case, (kind,))
                if not selected:
                    continue
                records[kind].append(dict(run=root.name, case=case, setup=item['setup'],
                                          successful_trials=len(selected)))
    matches, mismatches = [], []
    for ends in records['ends']:
        for dual in records['dual']:
            a, b = ends['setup'], dual['setup']
            if any(a.get(k) != b.get(k) for k in ('config','m','n','k','lda','ldb','ldd','input_mode')):
                continue
            different = [k for k in fields if a.get(k) != b.get(k)]
            pair = dict(ends_run=ends['run'], ends_case=ends['case'], dual_run=dual['run'],
                        dual_case=dual['case'], different_conditions=different)
            (mismatches if different else matches).append(pair)
    return dict(source_sha256=hashes, matching_fields=fields,
                available_cases={k:len(v) for k,v in records.items()}, exact_pairs=matches,
                near_pairs=mismatches, conclusion=('No exact-condition successful ends/dual pair; protocol transfer cannot be excluded offline.'
                    if not matches else 'Exact-condition pairs exist; inspect timing before requesting GPU comparison.'),
                limits='Repeated frozen/reference copies are not independent trials. Matching metadata alone does not establish matched build or warmup history.')


def failure_diagnostics(base, output, calls_factory):
    """V09 development diagnosis. Calls are decoded once by the common runner."""
    from collections import defaultdict
    from unittest.mock import patch
    import v09_model
    from analyze_r13_sm import select_attempts

    base, output = Path(base), Path(output)
    training_path = base/'20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json'
    training = clock.read_json(training_path)
    points = training['points']['all21_training']
    if len(points) != 21 or any(p['k'] != 1024 for p in points):
        raise ValueError('tensor-only comparison requires the original 21 training conditions')
    reduced = fit_tensor_only(points, training['model']['constraints']['a_max_ghz'])
    # The fit is complete before any target observation enters the analysis.
    def compare(points, full):
        return {name: failure_stats(source_frequency(p, p['observed_window_us'], model)['frequency_ghz']/p['observed_ghz']-1
                                   for p in points) for name, model in [('original', full), ('without_source', reduced)]}
    fit_comparison = dict(training=compare(points, training['model']))
    equal_path = base/'20261009-R09-equal-work-job738459/reanalysis/C-20261009-equal-work-clock-v1/equal-work-clock.json'
    equal = clock.read_json(equal_path)['clock_cases']
    fit_comparison['equal_work_K4096'] = compare([p for p in equal if p['k'] == 4096], training['model'])
    fit_comparison['equal_work_by_config'] = {cfg: compare([p for p in equal if p['k'] == 4096 and p['config'] == cfg], training['model'])
                                            for cfg in ('cfg_a','cfg_b','cfg_c')}
    root = base/'20261009-V09-heldout-job739011'
    processes, plain, numeric, frozen_cases = [], {}, {}, {}
    for call in calls_factory(base):
        row, setup, rec = call['row'], call['setup'], call['record']
        ctas, cal, frozen = call['observed']['ctas'], call['calibration'], call['prediction']
        cid = row['id']
        if cid not in numeric:
            numeric[cid] = frozen if frozen['status'] == 'predicted' else v09_model.predict_components(row, setup, cal)
            frozen_cases[cid] = frozen['status']
            selected, _, _ = select_attempts(root, cid, ('plain',))
            plain[cid] = {p['record']['trial']: p['record']['elapsed_us'] for p in selected}
        pred = numeric[cid]
        max_tiles = max(len(c['tiles']) for c in ctas)
        f = statistics.median((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns'])
                              for c in ctas if len(c['tiles']) == max_tiles)
        w = (max(c['end_ns'] for c in ctas)-min(c['entry_ns'] for c in ctas))/1000
        model = mixed_failure_clock(cal['clock'], pred['static_clock_work']['input_mode'],
                                    pred['zero_product_fraction'], cal.get('zero_activity_mixing', False))
        stripped = mixed_failure_clock(reduced, pred['static_clock_work']['input_mode'],
                                       pred['zero_product_fraction'], cal.get('zero_activity_mixing', False))
        conditional = source_frequency(pred['static_clock_work'], w, model)
        with patch.object(v09_model.clock, 'solve_source_envelope',
                          side_effect=lambda point, model, envelope: dict(frequency_ghz=f, window_us=envelope(f))):
            observed_f = v09_model.predict_components(row, setup, cal)
        entry = dict(case=cid, config=row['config'], group=row['group'], trial=rec['trial'],
                     frozen_status=frozen['status'], observed_frequency_ghz=f, observed_window_us=w,
                     conditional_frequency_ghz=conditional['frequency_ghz'],
                     conditional_frequency_error=conditional['frequency_ghz']/f-1,
                     without_source_frequency_error=source_frequency(pred['static_clock_work'], w, stripped)['frequency_ghz']/f-1,
                     free_frequency_error=pred['frequency_ghz']/f-1,
                     free_envelope_error=pred['window_us']/w-1,
                     observed_f_envelope_error=observed_f['window_us']/w-1,
                     free_plain_error=pred['plain_us']/plain[cid][rec['trial']]-1,
                     observed_f_plain_error=observed_f['plain_us']/plain[cid][rec['trial']]-1,
                     free_component_errors=failure_component_errors(ctas, pred),
                     observed_f_component_errors=failure_component_errors(ctas, observed_f),
                     local_windows=role_window_diagnostics(ctas),
                     raw_sha256=rec['raw_sha256'])
        processes.append(entry)
    groups = defaultdict(list)
    for p in processes:
        groups[p['case']].append(p)
    if len(groups) != 30 or any(len(g) != 10 for g in groups.values()):
        raise ValueError('all 30 conditions and ten successful dual processes are required')
    fields = [k for k in processes[0] if k.endswith('_error')]
    cases = []
    for cid, group in groups.items():
        item = dict(case=cid, config=group[0]['config'], group=group[0]['group'], frozen_status=frozen_cases[cid],
                    errors={key: failure_stats(p[key] for p in group) for key in fields},
                    observed_frequency_ghz=failure_stats(p['observed_frequency_ghz'] for p in group),
                    conditional_frequency_ghz=failure_stats(p['conditional_frequency_ghz'] for p in group))
        for variant in ('free_component_errors', 'observed_f_component_errors'):
            keys = set().union(*(p[variant] for p in group))
            item[variant] = {key: failure_stats(p[variant][key] for p in group if key in p[variant]) for key in sorted(keys)}
        item['local_windows'] = {phase: dict(
            ghz=failure_stats(p['local_windows']['same_role'][phase]['ghz']['median'] for p in group
                             if p['local_windows']['same_role'][phase]['frequency_windows']),
            maximum_deviation_from_call_f=max((abs(p['local_windows']['same_role'][phase]['ghz']['median']/p['observed_frequency_ghz']-1)
                                               for p in group if p['local_windows']['same_role'][phase]['frequency_windows']), default=None),
            usable_windows=sum(p['local_windows']['same_role'][phase]['frequency_windows'] for p in group))
            for phase in sorted(set().union(*(p['local_windows']['same_role'] for p in group)))}
        cases.append(item)
    fit_comparison['V09_by_group'] = {group: {name: failure_stats(c['errors'][key]['median'] for c in cases if c['group'] == group)
        for name,key in [('original','conditional_frequency_error'),('without_source','without_source_frequency_error')]}
        for group in sorted({c['group'] for c in cases})}
    result = dict(scope='Post-V09 development diagnosis; the rejected case is numeric diagnostic only. No frozen score changes.',
        provenance=dict(training=str(training_path), training_sha256=clock.sha256(training_path),
                        equal_work=str(equal_path), equal_work_sha256=clock.sha256(equal_path),
                        analysis_sha256=clock.sha256(Path(__file__))),
        tensor_only_model=reduced, fit_comparison=fit_comparison, cases=cases, processes=processes,
        observer_pair_inventory=observer_pair_inventory(base),
        notes=['All frequency fits use the original 21 measured-frequency targets; heldout observations never fit coefficients.',
               'Plain and dual are separate processes; trial-labelled plain errors are diagnostic comparisons, not same-call causal decompositions.',
               'Local frequency statistics use the same consumer role, except explicitly labelled cross_role_boundary.',
               'A 3.2 us reporting cutoff limits short-window ratio noise; timestamp gcd is an observed lattice, not hardware timer accuracy.',
               'Conditional negative frequency predictions remain visible; there is no clamp or historical minimum floor.'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'A-clock-failure.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def control_window(observed):
    """Use the same CTA scope for ends (counts) and dual (tile arrays)."""
    ctas = observed['ctas']
    counts = [c['tiles'] if isinstance(c['tiles'], int) else len(c['tiles']) for c in ctas]
    maximum = max(counts)
    active = [c for c, count in zip(ctas, counts) if count]
    frequency = statistics.median((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns'])
                                  for c, count in zip(ctas, counts) if count == maximum)
    envelope = (max(c['end_ns'] for c in active)-min(c['entry_ns'] for c in ctas))/1000
    return dict(frequency_ghz=frequency, window_us=envelope,
                max_cta_cycles=max(c['end_c']-c['entry_c'] for c in active),
                event_minus_window_us=observed['elapsed_us']-envelope,
                max_tiles=maximum, observed_sms=len({c['sm'] for c in active}))


def control_diagnostics(root, output):
    """Same-card protocol comparison, without transferring old fitted GHz constants."""
    from analyze_r13_sm import select_attempts
    from analyze_r18 import replay

    root, output = Path(root), Path(output)
    rows = [r for r in clock.read_json(root/'cases.json') if r['id'] in
            ('r09_protocol_k4096', 'r09_protocol_k32768')]
    if {r['k'] for r in rows} != {4096,32768} or len(rows) != 2:
        raise ValueError('both declared R09 protocol conditions are required')
    setups = {s['case']:s['setup'] for s in clock.read_json(root/'static_setup.json')}
    environment = clock.read_json(root/'environment.json')
    processes, pairs, failures, inputs = [], [], [], {}
    for name in ('cases.json','static_setup.json','environment.json','build/commands.json'):
        inputs[name] = clock.sha256(root/name)
    for row in rows:
        selected, failed, _ = select_attempts(root, row['id'], ('plain','ends','dual'))
        failures.extend(failed)
        records = {}
        for item in selected:
            rec = item['record']
            if rec['variant'] == 'ends':
                observed = v08_model.observe(root, rec, row, setups[row['id']])
            else:
                observed = replay(root, rec, row)
            entry = dict(case=row['id'], k=row['k'], variant=rec['variant'], trial=rec['trial'],
                         event_us=observed['elapsed_us'], host_start_ns=rec['host_start_ns'],
                         raw_sha256=rec['raw_sha256'])
            if rec['variant'] != 'plain':
                entry.update(control_window(observed))
            records[rec['variant'],rec['trial']] = entry
            processes.append(entry)
            inputs[item['path']] = clock.sha256(root/item['path'])
        expected = {(variant,trial) for variant in ('plain','ends','dual') for trial in range(10)}
        if set(records) != expected:
            raise ValueError('ten successful processes per plain/ends/dual condition required')
        for trial in range(10):
            plain, ends, dual = (records[v,trial] for v in ('plain','ends','dual'))
            pairs.append(dict(case=row['id'], k=row['k'], trial=trial,
                dual_over_ends_frequency=dual['frequency_ghz']/ends['frequency_ghz']-1,
                dual_over_ends_window=dual['window_us']/ends['window_us']-1,
                dual_over_ends_cycles=dual['max_cta_cycles']/ends['max_cta_cycles']-1,
                dual_over_ends_event=dual['event_us']/ends['event_us']-1,
                ends_over_plain_event=ends['event_us']/plain['event_us']-1,
                dual_over_plain_event=dual['event_us']/plain['event_us']-1,
                ends_dual_separation_s=abs(dual['host_start_ns']-ends['host_start_ns'])/1e9,
                plain_dual_separation_s=abs(dual['host_start_ns']-plain['host_start_ns'])/1e9))
    cases = []
    for row in rows:
        group = [p for p in pairs if p['case'] == row['id']]
        setup = setups[row['id']]
        item = dict(case=row['id'], k=row['k'], conditions={key:setup.get(key,row.get(key)) for key in
            ('gpu_uuid','m','n','k','lda','ldb','ldd','alloc_lda','alloc_ldb','allocated_a_elements',
             'allocated_b_elements','input_mode','seed','swizzle','grid','stages')},
            paired={key:failure_stats(p[key] for p in group) for key in group[0]
                    if key not in ('case','k','trial')}, variants={})
        for variant in ('plain','ends','dual'):
            group = [p for p in processes if p['case'] == row['id'] and p['variant'] == variant]
            fields = ('event_us',) if variant == 'plain' else ('event_us','frequency_ghz','window_us','max_cta_cycles','event_minus_window_us')
            item['variants'][variant] = {key:dict(failure_stats(p[key] for p in group),
                process_cv=statistics.pstdev(p[key] for p in group)/abs(statistics.fmean(p[key] for p in group)))
                for key in fields}
        cases.append(item)
    commands = clock.read_json(root/'build/commands.json')
    trace_words = {v:10 if any(flag in commands['cfg_b_'+v] for flag in
                                ('-DR18_MATCH_DUAL_LAYOUT','-DR18_DUAL_CLOCK')) else 6
                   for v in ('ends','dual')}
    result = dict(scope='Same-job protocol comparison on the recorded GPU; no old-card parameter fit or validation score.',
        run=str(root), environment=environment, input_sha256=inputs,
        analysis_sha256=clock.sha256(Path(__file__)), cases=cases, processes=processes, paired_trials=pairs,
        failed_attempts=failures, trace_words_per_tile=trace_words,
        notes=['Every successful observation independently replays numeric checks and its raw SHA; failed originals remain untouched.',
               'Frequency is the process median of maximum-work CTA local cycle/ns, not maximum cycles divided by the cross-SM envelope.',
               'Trial pairs are sequential processes. Separation and process spread are reported; differences include state drift.',
               'The two K values share allocated bytes and row strides but have different active input footprints.',
               'ends/dual may differ in scratch layout as well as timestamp sites; this measures the complete observer protocol.',
               'These measurements characterize the current card. They do not retrospectively validate a clock formula fitted on GPU-43269fbc.'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'A-controls.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--calibration-run', type=Path, required=True)
    parser.add_argument('--old-long-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze_source_clock(args.run.resolve(), args.calibration_run.resolve(), args.old_long_run.resolve(), args.output.resolve())
