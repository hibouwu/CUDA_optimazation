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
                rms=math.sqrt(statistics.fmean(x*x for x in values)), positive=int(sum(x > 0 for x in values)))


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


def reciprocal_frequency(point, window, model):
    """Conditional hypothesis: reciprocal response to the existing Q/S proxies."""
    mode = MODES.index(point['input_mode'])
    a, *costs = model['coefficients']
    denominator = (a + costs[mode]*point['tensor_mean_cycles']/(1000*window)
                   + costs[mode+3]*point['source_mean_kib']/window
                   * clock.duration_term(window, 'relaxation', model['tau_us']))
    return 1/denominator


def fit_reciprocal_clock(points, cap):
    """Same eight parameters; minimize measured GHz residuals, not 1/GHz."""
    import numpy as np
    from scipy.optimize import lsq_linear, least_squares

    y = np.asarray([p['observed_ghz'] for p in points])
    def fit_tau(tau):
        design = source_design(points, tau)
        design[:,1:] *= -1
        scales = np.maximum(np.max(np.abs(design),axis=0),1e-12)
        scaled = design/scales
        lower, upper = [1/cap,0,0,0,0,0,0], [np.inf]*7
        # Reciprocal targets initialize the solve only. The optimized residual
        # and every model comparison below are in GHz.
        initial = np.maximum(lower,lsq_linear(scaled,1/y,bounds=(lower,upper),method='bvls',tol=1e-12).x)
        solved = least_squares(lambda x:1/(scaled@x)-y,initial,
            jac=lambda x:-scaled/(scaled@x)[:,None]**2,
            bounds=(lower,upper),ftol=1e-13,xtol=1e-13,gtol=1e-11,max_nfev=1000)
        if not solved.success or solved.optimality > 1e-6:
            raise ValueError(f'reciprocal observed-frequency fit failed at tau={tau}: {solved.message}')
        return dict(tau_us=float(tau), coefficients=(solved.x/scales).tolist(),
                    mse_ghz=float(np.mean(solved.fun**2)), optimality=float(solved.optimality),
                    rank=int(np.linalg.matrix_rank(design)))
    coarse = np.logspace(-1,5,121)
    profile = [fit_tau(tau) for tau in coarse]
    best_index = min(range(len(profile)),key=lambda i:profile[i]['mse_ghz'])
    fine = np.geomspace(coarse[max(0,best_index-1)],coarse[min(120,best_index+1)],81)
    known = {p['tau_us'] for p in profile}
    profile += [fit_tau(t) for t in fine if float(t) not in known]
    best = min(profile,key=lambda p:p['mse_ghz'])
    return dict(**best, response='reciprocal', tau_profile=sorted(profile,key=lambda p:p['tau_us']),
                frequency_cap_ghz=cap, target='unweighted observed GHz residuals',
                parameter_units=['1/GHz','1/GHz^2','1/GHz^2','1/GHz^2',
                                 'us/(GHz KiB)','us/(GHz KiB)','us/(GHz KiB)'])


def static_output_kib(row):
    """Effective FP32 D bytes per SM; no observed epilogue or timing input."""
    return 4*row['m']*row['n']/(132*1024)


def static_revisit_features(row, setup):
    """Distinct source panels per software output ordinal, not physical traffic."""
    cfg = v08_model.base.CONFIGS[row['config']]
    tm,tn,tk = cfg['tile']
    cm,cn = cfg['cluster']
    work = v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row.get('raster_order','heuristic'))
    if row['m']%tm or row['n']%tn or row['k']%tk or any(
            mi*tm>=row['m'] or ni*tn>=row['n'] for c in work for mi,ni in c):
        raise ValueError('revisit definition here requires full in-bounds tiles without cluster padding')
    waves = []
    for j in range(max(map(len,work))):
        coords = [c[j] for c in work if len(c)>j]
        a,b = len({mi for mi,ni in coords}),len({ni for mi,ni in coords})
        # The actual R18 descriptor boxes: A splits M by cluster-N; B splits
        # K by cluster-M and its 128 columns comprise two 64-column boxes.
        a_boxes = {(mi*tm+(ni%cn)*(tm//cn),tm//cn) for mi,ni in coords}
        b_boxes = {(ni*tn+offset,(mi%cm)*(tk//cm),tk//cm)
                   for mi,ni in coords for offset in (0,64)}
        union_kib = 2*(len(a_boxes)*(tm//cn)*tk+len(b_boxes)*64*(tk//cm))/1024
        if union_kib != 2*tk*(tm*a+tn*b)/1024:
            raise ValueError('software wave does not contain complete multicast source-panel slices')
        waves.append(dict(j=j,cta_count=len(coords),unique_A=a,unique_B=b,
                          unique_A_boxes=len(a_boxes),unique_B_boxes=len(b_boxes),
                          unique_source_KiB_per_Ktile=union_kib))
    tiles = sum(map(len,work))
    kt = row['k']//tk
    return dict(output_tiles=tiles,cta_tile_min=min(map(len,work)),cta_tile_max=max(map(len,work)),
        tensor_mean_cycles=tiles*kt*2*tm*tn*tk/4096/132,
        nominal_source_mean_KiB=tiles*kt*2*tk*(tm/cn+tn/cm)/1024/132,
        output_mean_KiB=static_output_kib(row),
        unique_source_mean_KiB=kt*sum(w['unique_source_KiB_per_Ktile'] for w in waves)/132,
        panel_slots=sum(w['unique_A']+w['unique_B'] for w in waves),waves=waves,
        multicast_source_slices_verified=True,
        scope='Union distinct issued A/B source boxes within a common software output ordinal, then sum ordinals and Ktiles; multicast delivery is not counted again. No observed timing, cache residency, physical transfer count, or simultaneous issue is assumed.')


def swizzle_diagnostics(root, output):
    """A fixed-work sw1/sw8 contrast, without any old-card fitted parameters."""
    from analyze_r13_sm import select_attempts
    root,output = Path(root),Path(output)
    rows = {r['swizzle']:r for r in clock.read_json(root/'cases.json') if r['id'] in ('r09_g6_sw1','r09_g6_sw8')}
    if set(rows)!={1,8}:
        raise ValueError('both declared swizzle conditions required')
    setups = {s['case']:s['setup'] for s in clock.read_json(root/'static_setup.json')}
    fixed = ('gpu_uuid','m','n','k','lda','ldb','ldd','alloc_lda','alloc_ldb','allocated_a_elements',
             'allocated_b_elements','input_mode','seed','grid','stages','zero_m','zero_n','evict')
    if any(setups[rows[1]['id']].get(k)!=setups[rows[8]['id']].get(k) for k in fixed):
        raise ValueError('swizzle contrast changed another declared condition')
    features = {str(sw):static_revisit_features(row,setups[row['id']]) for sw,row in rows.items()}
    for key in ('output_tiles','cta_tile_min','cta_tile_max','tensor_mean_cycles','nominal_source_mean_KiB','output_mean_KiB'):
        if features['1'][key]!=features['8'][key]:
            raise ValueError('swizzle changed software work: '+key)
    processes,failures,records = [],[],{}
    hashes = {name:clock.sha256(root/name) for name in ('cases.json','static_setup.json','environment.json')}
    for sw,row in rows.items():
        selected,failed,_ = select_attempts(root,row['id'],('plain','ends'))
        failures.extend(failed)
        for item in selected:
            rec = item['record']
            obs = v08_model.observe(root,rec,row,setups[row['id']])
            point = dict(case=row['id'],swizzle=sw,variant=rec['variant'],trial=rec['trial'],
                         event_us=obs['elapsed_us'],host_start_ns=rec['host_start_ns'],raw_sha256=rec['raw_sha256'],
                         warmup_calls=len(rec['warmup']),warmup_total_us=sum(rec['warmup']))
            if rec['variant']=='ends':
                point.update(control_window(obs))
            records[sw,rec['variant'],rec['trial']] = point
            processes.append(point)
            hashes[item['path']] = clock.sha256(root/item['path'])
    if set(records)!={(sw,variant,trial) for sw in (1,8) for variant in ('plain','ends') for trial in range(10)}:
        raise ValueError('ten successful plain and ends trials for both swizzles required')
    pairs = []
    for trial in range(10):
        a,b = records[1,'ends',trial],records[8,'ends',trial]
        pairs.append(dict(trial=trial,frequency_change=b['frequency_ghz']/a['frequency_ghz']-1,
            window_change=b['window_us']/a['window_us']-1,
            max_cta_cycles_change=b['max_cta_cycles']/a['max_cta_cycles']-1,
            ends_event_change=b['event_us']/a['event_us']-1,
            plain_event_change=records[8,'plain',trial]['event_us']/records[1,'plain',trial]['event_us']-1,
            ends_pair_separation_s=abs(b['host_start_ns']-a['host_start_ns'])/1e9))
    summary = {key:failure_stats(p[key] for p in pairs) for key in pairs[0] if key!='trial'}
    variants = {}
    for sw in (1,8):
        variants[str(sw)] = {}
        for variant in ('plain','ends'):
            group = [p for p in processes if p['swizzle']==sw and p['variant']==variant]
            fields = (('event_us',) if variant=='plain' else ('event_us','frequency_ghz','window_us','max_cta_cycles')) + ('warmup_calls','warmup_total_us')
            variants[str(sw)][variant] = {key:dict(failure_stats(p[key] for p in group),
                process_cv=statistics.pstdev(p[key] for p in group)/statistics.fmean(p[key] for p in group)) for key in fields}
    result = dict(scope='Same-job sw1/sw8 order contrast; no old-card absolute clock or time validation.',
        run=str(root),environment=clock.read_json(root/'environment.json'),source_sha256=clock.sha256(Path(__file__)),
        input_sha256=hashes,static_features=features,variants=variants,paired_summary=summary,
        paired_trials=pairs,processes=processes,failed_attempts=failures,
        notes=['All numeric checks and raw SHA are replayed; only successful declared trials enter pairs.',
               'Software output ordinals do not prove simultaneous requests or cache hits; unique-source KiB is an opportunity proxy.',
               'Swizzle also changes per-CTA operand order; a frequency response would not by itself identify traffic or power.',
               'Trial pairs are sequential processes, so their dispersion includes intervening state changes.',
               'Adaptive warmup counts and cumulative durations are measured history diagnostics, not prediction inputs or controlled power doses.',
               'Q and nominal S remain exactly collinear within each configuration; this experiment changes order while holding both and D fixed.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'A-swizzle.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def common_history_diagnostics(root, output):
    """Compare the two target orders after a declared common or adaptive prefix."""
    from analyze_r13_sm import select_attempts
    import v06_run as common
    root,output = Path(root),Path(output)
    common.verify(root)
    rows = clock.read_json(root/'cases.json')
    policies = ('common-sw1-16','adaptive')
    if len(rows)!=4 or {(r['warmup_policy'],r['swizzle']) for r in rows}!={(p,s) for p in policies for s in (1,8)}:
        raise ValueError('four declared common/adaptive sw1/sw8 conditions required')
    setups = {s['case']:s['setup'] for s in clock.read_json(root/'static_setup.json')}
    fixed = ('gpu_uuid','config','m','n','k','lda','ldb','ldd','alloc_lda','alloc_ldb',
             'allocated_a_elements','allocated_b_elements','input_mode','seed','grid','stages',
             'zero_m','zero_n','evict')
    anchor = setups[rows[0]['id']]
    if any(any(setups[row['id']].get(k)!=anchor.get(k) for k in fixed) for row in rows):
        raise ValueError('common-history contrast changed geometry/device/input conditions')
    inputs = {name:clock.sha256(root/name) for name in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/binary_hashes.json')}
    processes,failures,records = [],[],{}
    for row in rows:
        selected,failed,_ = select_attempts(root,row['id'],('ends',))
        failures.extend(failed)
        for item in selected:
            rec = item['record']
            observed = v08_model.observe(root,rec,row,setups[row['id']])
            # observe replays the explicit prefix identity. Fixed history does
            # not apply the adaptive convergence filter, here or in the reader.
            warm = rec['warmup']
            largest = max(c['tiles'] for c in observed['ctas'])
            observer_windows = [(c['end_ns']-c['entry_ns'])/1000 for c in observed['ctas'] if c['tiles']==largest]
            p = dict(case=row['id'],policy=row['warmup_policy'],swizzle=row['swizzle'],trial=rec['trial'],
                event_us=observed['elapsed_us'],host_start_ns=rec['host_start_ns'],raw_sha256=rec['raw_sha256'],
                warmup_calls=len(warm),warmup_total_us=sum(warm),
                warmup_last5_cv=statistics.pstdev(warm[-5:])/statistics.fmean(warm[-5:]),
                frequency_observer_window_us=statistics.median(observer_windows),
                **control_window(observed))
            processes.append(p)
            records[p['policy'],p['swizzle'],p['trial']] = p
            inputs[item['path']] = clock.sha256(root/item['path'])
    if set(records)!={(p,s,t) for p in policies for s in (1,8) for t in range(10)}:
        raise ValueError('all forty successful ends processes are required, including high-CV fixed prefixes')
    fields = ('frequency_ghz','max_cta_cycles','window_us','event_us')
    by_policy,pairs = {},[]
    for policy in policies:
        group = []
        for trial in range(10):
            a,b = records[policy,1,trial],records[policy,8,trial]
            p = dict(policy=policy,trial=trial,first_swizzle=1 if a['host_start_ns']<b['host_start_ns'] else 8,
                     separation_s=abs(a['host_start_ns']-b['host_start_ns'])/1e9,
                     prefix_kernel_sum_change=b['warmup_total_us']/a['warmup_total_us']-1,
                     prefix_kernel_sum_difference_us=b['warmup_total_us']-a['warmup_total_us'],
                     **{key+'_change':b[key]/a[key]-1 for key in fields})
            group.append(p)
        pairs.extend(group)
        by_policy[policy] = dict(paired={key+'_change':failure_stats(p[key+'_change'] for p in group) for key in fields},
            prefix_kernel_sum_change=failure_stats(p['prefix_kernel_sum_change'] for p in group),
            prefix_kernel_sum_difference_us=failure_stats(p['prefix_kernel_sum_difference_us'] for p in group),
            order_counts={str(sw):sum(p['first_swizzle']==sw for p in group) for sw in (1,8)},conditions={})
        for sw in (1,8):
            conditions = [records[policy,sw,t] for t in range(10)]
            by_policy[policy]['conditions'][str(sw)] = dict(
                metrics={key:dict(failure_stats(p[key] for p in conditions),
                    process_cv=statistics.pstdev(p[key] for p in conditions)/statistics.fmean(p[key] for p in conditions))
                    for key in (*fields,'warmup_calls','warmup_total_us','warmup_last5_cv','frequency_observer_window_us')},
                warmup_last5_cv_gt_2pct=sum(p['warmup_last5_cv']>.02 for p in conditions))
    policy_pairs = []
    for sw in (1,8):
        for trial in range(10):
            a,b = records['adaptive',sw,trial],records['common-sw1-16',sw,trial]
            policy_pairs.append(dict(swizzle=sw,trial=trial,
                **{key+'_common_over_adaptive':b[key]/a[key]-1 for key in fields}))
    reference_path = root.parent/'20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-swizzle.json'
    reference = clock.read_json(reference_path)
    inputs[str(reference_path)] = clock.sha256(reference_path)
    reference_keys = dict(frequency_ghz='frequency_change',max_cta_cycles='max_cta_cycles_change',
                          window_us='window_change',event_us='ends_event_change')
    reference_comparison = dict(run=reference['run'],environment=reference['environment'],
        same_gpu=clock.read_json(root/'environment.json')['gpu']==reference['environment']['gpu'],
        scope='Cross-batch adaptive replication reference only, not a same-trial control or prediction score.',
        old_paired_medians={key:reference['paired_summary'][value]['median'] for key,value in reference_keys.items()},
        current_adaptive_medians={key:by_policy['adaptive']['paired'][key+'_change']['median'] for key in fields})
    result = dict(scope='New declared prehistory protocol; measured contrasts only, not new prediction accuracy.',
        run=str(root),environment=clock.read_json(root/'environment.json'),source_sha256=clock.sha256(Path(__file__)),
        input_sha256=inputs,conditions={k:anchor.get(k) for k in fixed},processes=processes,
        paired_trials=pairs,by_policy=by_policy,common_over_adaptive_trials=policy_pairs,
        common_over_adaptive={str(sw):{key+'_common_over_adaptive':failure_stats(p[key+'_common_over_adaptive'] for p in policy_pairs if p['swizzle']==sw)
                                      for key in fields} for sw in (1,8)},
        cross_batch_reference=reference_comparison,failed_attempts=failures,
        verification=dict(source_and_binary_hashes=True,successful_raw_replays=len(processes),fixed_prefix_cv_exclusions=0),
        timing_scopes=dict(frequency='Same-CTA clock64/globaltimer entry-to-final interval; process median over maximum-work CTAs, not a continuous frequency trace.',
            window='Target-call globaltimer envelope across all CTA entries/final records.',
            prefix='Sum of sixteen or adaptive measured warmup CUDA-event kernel intervals. Trace clears, synchronization and host gaps between them are excluded; no full-prefix wall-clock envelope was recorded.'),
        notes=['Common history executes exactly sixteen sw1 calls on the same initialized buffers before one target; its warmup CV is recorded, never filtered.',
               'Adaptive controls retain the existing 8..30 call convergence protocol. Common/adaptive contrasts are reported separately.',
               'Equal declared initialization and prefix work do not prove identical physical cache contents, temperatures, power states, or elapsed histories.',
               'All pairs are sequential processes; prior target effects and device-state variation are not eliminated by a shared trial label.',
               'No old-card frequency coefficients, timing fits, or V09 score changes enter this analysis.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'A-common-history.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def history_state_weights(policy, target_swizzle, prefix_us, window_us, tau_us):
    """Weights of (initial f, sw1 equilibrium, sw8 equilibrium) for one pole."""
    average = -math.expm1(-window_us/tau_us)/(window_us/tau_us)
    decay = math.exp(-prefix_us/tau_us)
    weights = [decay*average,0.,0.]
    prefix_swizzle = 1 if policy=='common-sw1-16' else target_swizzle
    weights[1 if prefix_swizzle==1 else 2] += (1-decay)*average
    weights[1 if target_swizzle==1 else 2] += 1-average
    return weights


def history_state_diagnostics(root, output):
    """Constrain one local state law; measured H/W are never predictor inputs."""
    import numpy as np
    from scipy.optimize import lsq_linear,linprog
    root,output = Path(root),Path(output)
    path = Path(output)/'A-common-history.json'
    data = clock.read_json(path)
    points = data['processes']
    uuid = data['environment']['gpu'].split(',')[0]
    smi_path = root/'nvidia-smi-before.txt'
    smi = smi_path.read_text().split(uuid,1)[1].split('\nGPU ',1)[0]
    maxima = smi.split('\n    Max Clocks\n',1)[1].split('\n    Max Customer',1)[0]
    cap = int(re.search(r'\bSM\s*:\s*(\d+) MHz',maxima)[1])/1000
    y = np.asarray([p['frequency_ghz'] for p in points])
    def design(tau):
        return np.asarray([history_state_weights(p['policy'],p['swizzle'],p['warmup_total_us'],
            p['frequency_observer_window_us'],tau) for p in points])
    def fit_tau(tau):
        matrix = design(tau)
        fit = lsq_linear(matrix,y,bounds=(0,cap),method='bvls',tol=1e-12)
        if not fit.success:
            raise ValueError('fixed-tau state fit failed')
        values = matrix@fit.x
        return dict(tau_us=float(tau),initial_and_equilibria_ghz=fit.x.tolist(),
                    mse_ghz=float(np.mean((values-y)**2)),errors=failure_stats(values/y-1),
                    active_bounds=fit.active_mask.tolist(),rank=int(np.linalg.matrix_rank(matrix)))
    coarse = np.logspace(2,7,161)  # 0.1 ms .. 10 s, a profile not a physical bound.
    profile = [fit_tau(tau) for tau in coarse]
    i = min(range(len(profile)),key=lambda i:profile[i]['mse_ghz'])
    fine = np.geomspace(coarse[max(0,i-1)],coarse[min(len(coarse)-1,i+1)],81)
    profile += [fit_tau(tau) for tau in fine if float(tau) not in {p['tau_us'] for p in profile}]
    best = min(profile,key=lambda p:p['mse_ghz'])
    predicted = design(best['tau_us'])@best['initial_and_equilibria_ghz']
    groups = {}
    for policy in ('adaptive','common-sw1-16'):
        for sw in (1,8):
            indices = [i for i,p in enumerate(points) if p['policy']==policy and p['swizzle']==sw]
            groups[policy+'/sw'+str(sw)] = failure_stats(predicted[indices]/y[indices]-1)
    def variable_initial_problem(tau):
        matrix = design(tau)
        # Eliminate each unobserved f0_i with the conservative 0..MaxClocks
        # bounds: equilibrium contribution <= y <= contribution+gain*cap.
        return dict(A_ub=np.concatenate((matrix[:,1:],-matrix[:,1:])),
                    b_ub=np.concatenate((y,-y+matrix[:,0]*cap)),bounds=[(0,cap)]*2,method='highs')
    feasible = []
    for tau in coarse:
        fit = linprog([0.,0.],**variable_initial_problem(tau))
        if fit.success:
            feasible.append(float(tau))
    bounds = []
    for tau in (50000.,100000.,1000000.,10000000.):
        problem = variable_initial_problem(tau)
        ranges = []
        for direction in np.eye(2):
            lo,hi = (linprog(sign*direction,**problem) for sign in (1,-1))
            ranges.append([float(lo.fun),float(-hi.fun)] if lo.success and hi.success else None)
        bounds.append(dict(tau_us=tau,equilibrium_sw1_sw8_ranges_ghz=ranges))
    near = [p['tau_us'] for p in profile if p['mse_ghz']<=best['mse_ghz']*1.1]
    result = dict(scope='Local state-identification diagnostic for two loads on GPU-ef8692f3; not a GEMM frequency predictor.',
        source_sha256=clock.sha256(Path(__file__)),input_sha256={str(path):clock.sha256(path),str(smi_path):clock.sha256(smi_path)},
        formula='df/dt=(f_eq(load)-f)/tau; target mean = f_eq(target)+(f_eq(prefix)+(f0-f_eq(prefix))*exp(-H/tau)-f_eq(target))*(1-exp(-W/tau))/(W/tau)',
        clock_cap_ghz=cap,parameter_order=['shared initial f0','sw1 load equilibrium','sw8 load equilibrium'],
        shared_initial=dict(best=best,by_group=groups,tau_profile=sorted(profile,key=lambda p:p['tau_us']),
            sensitivity_tau_us=[min(near),max(near)],sensitivity_rule='Profile MSE<=1.10*minimum, not a confidence interval.'),
        unknown_initial=dict(feasible_tau_grid_us=feasible,equilibrium_ranges=bounds,
            exact_fit=True if feasible else False,
            limiting_case='As tau grows, initial-state gains approach one; measured target f can be assigned to f0_i independently of equilibrium values. The finite scan supplies no identified upper time constant.'),
        process_diagnostics=[dict(case=p['case'],trial=p['trial'],observed_f_ghz=float(y[i]),
            shared_initial_predicted_f_ghz=float(predicted[i]),prefix_kernel_sum_us=p['warmup_total_us'],
            target_observer_window_us=p['frequency_observer_window_us']) for i,p in enumerate(points)],
        notes=['Only one first-order response family is tested. Its two load equilibria are local identification parameters, not per-condition prediction constants.',
               'H is a measured sum of kernel event intervals and misses resets, synchronization and host gaps. W is an observed median CTA interval. Neither is a formal static predictor input.',
               'The process frequency is a median of CTA ratios, not an exact sample of one continuous scalar trajectory.',
               'Shared initial state is an unmeasured assumption. Allowing bounded process initial states exposes nonidentifiability, not a preferred fitted model.',
               'No prefix clocks or wall-clock inter-call gaps were recorded. Fitted tau is therefore not an identified hardware time constant.',
               'Next measurement needs same-process prefix and target call trajectories, with one final host trace read and globaltimer endpoints preserving the inter-call spans.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'A-history-state.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def history_series_design(processes, tau_us):
    """One pole on the observed globaltimer axis; process initial states are nuisance."""
    import numpy as np
    matrix=[]
    for pindex,process in enumerate(processes):
        state=np.zeros(len(processes)+2);state[pindex]=1.
        calls=process['calls']
        for index,call in enumerate(calls):
            equilibrium=np.zeros(len(processes)+2)
            equilibrium[len(processes)+(call['swizzle']==8)]=1.
            # Median of affine interval means equals their median attenuation
            # times the common state difference, including a negative difference.
            attenuation=statistics.median(math.exp(-offset/tau_us)*(-math.expm1(-width/tau_us))/(width/tau_us)
                for offset,width in call['frequency_observers_us'])
            matrix.append(attenuation*state+(1-attenuation)*equilibrium)
            if index+1<len(calls):
                dwell=(calls[index+1]['first_entry_ns']-call['first_entry_ns'])/1000
                decay=math.exp(-dwell/tau_us)
                state=decay*state+(1-decay)*equilibrium
    return np.asarray(matrix)


def fit_history_series(processes, cap):
    """Fit observed GHz only; the last nine target calls are a temporal diagnostic."""
    import numpy as np
    from scipy.optimize import lsq_linear
    points=[(p,c) for p in processes for c in p['calls']]
    y=np.asarray([c['frequency_ghz'] for p,c in points])
    masks=dict(prefix_and_first10_targets=np.asarray([c['index']<26 for p,c in points]),
               all_calls=np.ones(len(points),dtype=bool))
    reports,predictions={},{}
    for name,mask in masks.items():
        def at_tau(tau):
            full=history_series_design(processes,tau);x=full[mask]
            fit=lsq_linear(x,y[mask],bounds=(0,cap),method='bvls',tol=1e-11)
            if not fit.success:raise ValueError('history-series state fit failed')
            error=x@fit.x-y[mask]
            singular=np.linalg.svd(x,compute_uv=False)
            return dict(tau_us=float(tau),coefficients_ghz=fit.x.tolist(),
                mse_ghz=float(np.mean(error**2)),active_bounds=fit.active_mask.tolist(),
                rank=int(np.linalg.matrix_rank(x)),singular_ratio=float(singular[-1]/singular[0]))
        coarse=np.logspace(2,7,61)
        profile=[at_tau(t) for t in coarse]
        i=min(range(len(profile)),key=lambda j:profile[j]['mse_ghz'])
        fine=np.geomspace(coarse[max(i-1,0)],coarse[min(i+1,len(coarse)-1)],21)
        profile.extend(at_tau(t) for t in fine if float(t) not in {p['tau_us'] for p in profile})
        best=min(profile,key=lambda p:p['mse_ghz'])
        predicted=history_series_design(processes,best['tau_us'])@best['coefficients_ghz']
        predictions[name]=predicted
        near=[p['tau_us'] for p in profile if p['mse_ghz']<=best['mse_ghz']*1.10]
        groups={}
        for sw in (1,8):
            for segment,select in [('prefix',lambda c:c['index']<16),
                    ('target_first10',lambda c:16<=c['index']<26),('target_last9',lambda c:c['index']>=26)]:
                chosen=[j for j,(p,c) in enumerate(points) if p['swizzle']==sw and select(c)]
                groups[f'sw{sw}/{segment}']=dict(relative=failure_stats(predicted[chosen]/y[chosen]-1),
                    residual_ghz=failure_stats(predicted[chosen]-y[chosen]))
        reports[name]=dict(best=best,by_segment=groups,tau_profile=sorted(profile,key=lambda p:p['tau_us']),
            sensitivity_tau_us=[min(near),max(near)],
            sensitivity_reaches_scan_edge=min(near)==float(coarse[0]) or max(near)==float(coarse[-1]),
            sensitivity_rule='Profile MSE <= 1.10*minimum is sensitivity, not a confidence interval.',
            parameter_order=[p['case']+'/trial'+str(p['trial'])+' initial f' for p in processes]+['sw1 equilibrium','sw8 equilibrium'])
    process_predictions=[];carry_errors=[];holdout_errors=[];gap_ratios=[];training_residual=[]
    cursor=0
    for p in processes:
        calls=[];carry=p['calls'][25]['frequency_ghz']
        for c in p['calls']:
            predicted=float(predictions['prefix_and_first10_targets'][cursor])
            full=float(predictions['all_calls'][cursor]);observed=c['frequency_ghz']
            calls.append(dict(index=c['index'],observed_ghz=observed,trained_ghz=predicted,
                full_fit_ghz=full,last9_holdout=c['index']>=26))
            if c['index']>=26:
                carry_errors.append(carry/observed-1);holdout_errors.append(predicted/observed-1)
            if c['index']>0:
                gap_ratios.append(c['globaltimer_gap_before_us']/c['window_us'])
                training_residual.append(predicted/observed-1)
            cursor+=1
        process_predictions.append(dict(case=p['case'],trial=p['trial'],swizzle=p['swizzle'],calls=calls))
    correlation=float(np.corrcoef(gap_ratios,training_residual)[0,1]) if np.std(gap_ratios)>0 and np.std(training_residual)>0 else None
    return dict(formula='df/dt=(f_eq(sw)-f)/tau, shared tau/equilibria with one initial state per process',
        fits=reports,process_predictions=process_predictions,
        last9_comparison=dict(state=failure_stats(holdout_errors),carry_last_training_call=failure_stats(carry_errors)),
        residual_vs_preceding_globaltimer_gap_ratio=correlation,
        scope='All time coordinates are observed diagnostics; no trace time, initial-state nuisance coefficient or local equilibrium is added to a formal predictor.',
        assumptions=['All CTA interval means are represented by one process scalar state; median attenuation preserves the actual maximum-work CTA observation intervals.',
            'The preceding load equilibrium remains active from its first CTA entry until the next call first entry, including the measured boundary gap. There is no independently identified idle-state response.',
            'The first 16 prefix calls and first 10 target calls alone fit the temporal holdout model; the final 9 target calls never select its parameters or tau. Full-call fit is descriptive only.',
            'Temporal holdout calls share a process and workload; they are mechanism diagnostics, not independent new-shape/GPU validation.',
            'A profile minimum, an active clock bound, or small residual alone does not identify a physical time constant; inspect boundary sensitivity, cross-process scatter, direction, and held-segment residuals.'])


def history_call_metrics(trajectory, event_gaps_us):
    """Preserve common ends-decoder windows for R09 history interventions."""
    calls=[]
    for j,old in enumerate(trajectory['calls']):
        c={k:v for k,v in old.items() if k!='ctas'}
        maximum=max(t['tiles'] for t in old['ctas'])
        observers=[t for t in old['ctas'] if t['tiles']==maximum]
        c.update(frequency_observers_us=[[(t['entry_ns']-old['first_entry_ns'])/1000,
            (t['end_ns']-t['entry_ns'])/1000] for t in observers],
            median_full_cta_cycles=statistics.median(t['end_c']-t['entry_c'] for t in observers),
            median_full_cta_window_us=statistics.median((t['end_ns']-t['entry_ns'])/1000 for t in observers),
            max_cycles_over_process_f_us=old['max_cta_cycles']/(1000*old['frequency_ghz']),
            local_frequency_ghz=failure_stats((t['end_c']-t['entry_c'])/(t['end_ns']-t['entry_ns']) for t in observers),
            event_gap_before_us=None if j==0 else event_gaps_us[j-1],
            globaltimer_gap_before_us=None if j==0 else trajectory['inter_call_globaltimer_us'][j-1])
        c['role_windows']=[]
        for t in old['ctas']:
            for role,r in enumerate(t['roles']):
                if not r['tiles']:continue
                cycles,ns=r['final_c']-t['entry_c'],r['final_ns']-t['entry_ns']
                if min(cycles,ns)<=0:raise ValueError('nonpositive same-role history window')
                c['role_windows'].append([t['cta'],role,r['tiles'],t['sm'],cycles,ns])
        calls.append(c)
    return calls


def history_series_diagnostics(root, output):
    """Same-process clock trajectories with a contemporaneous recording perturbation pair."""
    import gzip
    from analyze_r13_sm import select_attempts
    from analyze_r18 import replay
    import v06_run as common
    root,output=Path(root),Path(output)
    common.verify(root)
    rows=clock.read_json(root/'cases.json')
    if len(rows)!=4 or {(r.get('history_mode'),r['swizzle']) for r in rows}!={(m,s) for m in ('single','series') for s in (1,8)}:
        raise ValueError('four declared single/series sw1/sw8 conditions required')
    setups={s['case']:s['setup'] for s in clock.read_json(root/'static_setup.json')}
    fixed=('gpu_uuid','config','m','n','k','lda','ldb','ldd','alloc_lda','alloc_ldb','allocated_a_elements',
        'allocated_b_elements','input_mode','seed','grid','stages','zero_m','zero_n','evict','history_capacity_bytes')
    anchor=setups[rows[0]['id']]
    if any(any(setups[r['id']].get(k)!=anchor.get(k) for k in fixed) for r in rows):
        raise ValueError('history-series contrast changed geometry/input/device/allocation conditions')
    inputs={name:clock.sha256(root/name) for name in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/binary_hashes.json','nvidia-smi-before.txt')}
    processes,failures,records=[],[],{}
    for row in rows:
        selected,failed,_=select_attempts(root,row['id'],('ends',));failures.extend(failed)
        for item in selected:
            rec=item['record']
            decoded=replay(root,rec,row) if row['history_mode']=='series' else v08_model.observe(root,rec,row,setups[row['id']])
            with gzip.open(root/rec['raw'],'rt') as stream:
                call=next(ev for line in stream if line.strip() for ev in [json.loads(line)] if ev['event']=='call')
            p=dict(case=row['id'],mode=row['history_mode'],swizzle=row['swizzle'],trial=rec['trial'],
                protocol=call['history_protocol'],host_start_ns=rec['host_start_ns'],host_stop_ns=rec['host_stop_ns'],
                raw_sha256=rec['raw_sha256'],prefix_kernel_sum_us=sum(call['warmup_us']),
                prefix_call_event_us=call['warmup_us'],
                prefix_last3_over_first3_event=statistics.median(call['warmup_us'][-3:])/statistics.median(call['warmup_us'][:3])-1,
                prefix_event_us=call['history_prefix_event_us'],total_event_us=call['history_event_us'],
                warmup_last5_cv=statistics.pstdev(call['warmup_us'][-5:])/statistics.fmean(call['warmup_us'][-5:]),
                preceding_target_calls=call['preceding_target_calls'],event_gaps_us=call['history_inter_call_event_us'])
            if p['mode']=='series':
                trajectory=decoded['history_series'];p['calls']=[]
                p['calls']=history_call_metrics(trajectory,p['event_gaps_us'])
                p['target']=p['calls'][16]
                p['globaltimer_boundary_gap_fraction']=sum(trajectory['inter_call_globaltimer_us'])/(
                    (p['calls'][-1]['last_final_ns']-p['calls'][0]['first_entry_ns'])/1000)
            else:
                p['target']=dict(event_us=decoded['elapsed_us'],**control_window(decoded))
                maximum=max(t['tiles'] for t in decoded['ctas'])
                p['target']['median_full_cta_cycles']=statistics.median(t['end_c']-t['entry_c'] for t in decoded['ctas'] if t['tiles']==maximum)
                p['target']['median_full_cta_window_us']=statistics.median((t['end_ns']-t['entry_ns'])/1000 for t in decoded['ctas'] if t['tiles']==maximum)
                p['target']['max_cycles_over_process_f_us']=p['target']['max_cta_cycles']/(1000*p['target']['frequency_ghz'])
            p['event_gap_fraction']=sum(p['event_gaps_us'])/p['total_event_us']
            records[p['mode'],p['swizzle'],p['trial']]=p;processes.append(p)
            inputs[item['path']]=clock.sha256(root/item['path'])
    if set(records)!={(m,s,t) for m in ('single','series') for s in (1,8) for t in range(10)}:
        raise ValueError('all forty successful processes required; no fixed-prefix CV filtering')
    fields=('frequency_ghz','max_cta_cycles','median_full_cta_cycles','median_full_cta_window_us','max_cycles_over_process_f_us','window_us','event_us')
    pairs=[];pair_summary={}
    for sw in (1,8):
        group=[]
        for trial in range(10):
            single,series=records['single',sw,trial],records['series',sw,trial]
            group.append(dict(swizzle=sw,trial=trial,first_mode='single' if single['host_start_ns']<series['host_start_ns'] else 'series',
                separation_s=abs(single['host_start_ns']-series['host_start_ns'])/1e9,
                prefix_event_change=series['prefix_event_us']/single['prefix_event_us']-1,
                prefix_kernel_sum_change=series['prefix_kernel_sum_us']/single['prefix_kernel_sum_us']-1,
                **{key+'_change':series['target'][key]/single['target'][key]-1 for key in fields}))
        pairs.extend(group)
        pair_summary[str(sw)]=dict(metrics={key:failure_stats(p[key] for p in group) for key in group[0] if key.endswith('_change')},
            order_counts={m:sum(p['first_mode']==m for p in group) for m in ('single','series')})
    series=[p for p in processes if p['mode']=='series']
    trajectory_summary={};transitions=[]
    for sw in (1,8):
        group=[p for p in series if p['swizzle']==sw]
        trajectory_summary[str(sw)]=[]
        for j in range(35):
            samples=[p['calls'][j] for p in group]
            summary={key:dict(failure_stats(c[key] for c in samples),
                process_cv=statistics.pstdev(c[key] for c in samples)/statistics.fmean(c[key] for c in samples)) for key in fields}
            if j:summary.update({key:failure_stats(c[key] for c in samples) for key in ('event_gap_before_us','globaltimer_gap_before_us')})
            trajectory_summary[str(sw)].append(dict(index=j,phase=samples[0]['phase'],metrics=summary))
        for p in group:
            c=p['calls']
            p['within_target_variation']={key:dict(
                cv=statistics.pstdev(x[key] for x in c[16:])/statistics.fmean(x[key] for x in c[16:]),
                max_over_min=max(x[key] for x in c[16:])/min(x[key] for x in c[16:])-1) for key in fields}
            transitions.append(dict(swizzle=sw,trial=p['trial'],
                first_target_over_last_prefix={key:c[16][key]/c[15][key]-1 for key in fields},
                last3_target_over_first3_target={key:statistics.median(x[key] for x in c[-3:])/statistics.median(x[key] for x in c[16:19])-1 for key in fields},
                prefix_last3_over_first3={key:statistics.median(x[key] for x in c[13:16])/statistics.median(x[key] for x in c[:3])-1 for key in fields}))
    environment=clock.read_json(root/'environment.json');uuid=environment['gpu'].split(',')[0]
    smi=(root/'nvidia-smi-before.txt').read_text().split(uuid,1)[1].split('\nGPU ',1)[0]
    maxima=smi.split('\n    Max Clocks\n',1)[1].split('\n    Max Customer',1)[0]
    cap=int(re.search(r'\bSM\s*:\s*(\d+) MHz',maxima)[1])/1000
    result=dict(scope='New recorded prehistory trajectories and same-binary recording perturbation controls; not a new prediction score.',
        run=str(root),environment=environment,source_sha256=clock.sha256(Path(__file__)),input_sha256=inputs,
        conditions={k:anchor.get(k) for k in fixed},processes=processes,failed_attempts=failures,
        recording_perturbation=dict(comparison='series first target / contemporaneous single target',paired_trials=pairs,by_swizzle=pair_summary),
        prefix_event_trend={mode:dict(all=failure_stats(p['prefix_last3_over_first3_event'] for p in processes if p['mode']==mode),
            by_swizzle={str(sw):failure_stats(p['prefix_last3_over_first3_event'] for p in processes if p['mode']==mode and p['swizzle']==sw) for sw in (1,8)}) for mode in ('single','series')},
        trajectory_by_index=trajectory_summary,process_transitions=transitions,
        target_variation={str(sw):{key:{measure:failure_stats(p['within_target_variation'][key][measure] for p in series if p['swizzle']==sw)
            for measure in ('cv','max_over_min')} for key in fields} for sw in (1,8)},
        gap_summary=dict(event_gap_fraction={mode:failure_stats(p['event_gap_fraction'] for p in processes if p['mode']==mode) for mode in ('single','series')},
            series_globaltimer_boundary_gap_fraction=failure_stats(p['globaltimer_boundary_gap_fraction'] for p in series)),
        transition_summary={str(sw):{segment:{key:failure_stats(p[segment][key] for p in transitions if p['swizzle']==sw) for key in fields}
            for segment in ('first_target_over_last_prefix','last3_target_over_first3_target','prefix_last3_over_first3')} for sw in (1,8)},
        state_diagnostic=fit_history_series(series,cap),
        verification=dict(source_and_binary_hashes=True,successful_raw_replays=len(processes),fixed_prefix_cv_exclusions=0,
            series_calls=sum(len(p['calls']) for p in series)),
        notes=['All series contain sixteen sw1 prefix calls plus nineteen targets; their ordinary call summary is target19 after eighteen preceding targets.',
            'Single/series share allocation capacity and event objects. Series alone adds D2D heads recording; equal allocation does not prove equal cache contents or unperturbed history.',
            'Event gaps and globaltimer end-to-next-entry gaps have different boundaries; neither is asserted to be pure idle time.',
            'median_full_cta_window_us is median same-CTA C/f over maximum-work CTAs, exactly the corresponding globaltimer interval. max_cycles_over_process_f_us combines maximum C and process-median f and is only a conversion proxy, not the global envelope.',
            'Every series call has ends/work/SM checks; the process numeric output check covers only its last target.',
            'A sequential process pair shares a trial label, not a common initial thermal/power/cache state. Report dispersion and actual execution order before causal attribution.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'A-history-series.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot_history_series(result,output)
    return result


def plot_history_series(report, output):
    """Process trajectories and empirical spread; bands are not confidence intervals."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),sharex=True,layout='constrained')
    fields=[('frequency_ghz',1,'Effective SM clock (GHz)'),
            ('max_cta_cycles',1e6,'Maximum CTA cycles (million)'),
            ('median_full_cta_window_us',1000,'Matched full-work CTA window (ms)'),
            ('event_us',1000,'Kernel event time (ms)')]
    x=np.arange(1,36)
    for sw,color in [(1,'#2878b5'),(8,'#d97919')]:
        group=[p for p in report['processes'] if p['mode']=='series' and p['swizzle']==sw]
        for ax,(field,scale,label) in zip(axes.flat,fields):
            values=np.array([[c[field]/scale for c in p['calls']] for p in group])
            for trace in values:ax.plot(x,trace,color=color,alpha=.14,linewidth=.65)
            lo,hi=np.quantile(values,[.1,.9],axis=0)
            ax.fill_between(x,lo,hi,color=color,alpha=.13,linewidth=0)
            ax.plot(x,np.median(values,axis=0),color=color,linewidth=1.8,label=f'target sw{sw}')
            ax.set_ylabel(label,fontsize=9)
    for ax in axes.flat:
        ax.axvline(16.5,color='#777777',linestyle='--',linewidth=.9)
        ax.set_xlim(1,35);ax.set_xticks([1,8,16,24,35]);ax.grid(alpha=.18)
        ax.tick_params(labelsize=8)
    axes[0,0].legend(frameon=False,fontsize=8)
    for ax in axes[1]:ax.set_xlabel('Call index: 1-16 common sw1; 17-35 target',fontsize=9)
    fig.suptitle('GH200 history response: 10 independent processes per target\n'
                 'Thin lines: all processes; thick line: median; band: empirical 10-90%',fontsize=11)
    fig.savefig(Path(output)/'A-history-series.png',dpi=160)
    plt.close(fig)


def history_localization_diagnostics(root, output):
    """Localize archived same-role drift from the single shared series replay."""
    root,output=Path(root),Path(output)
    path=output/'A-history-series.json';data=clock.read_json(path)
    if data['input_sha256']['cases.json']!=clock.sha256(root/'cases.json'):
        raise ValueError('history-series report belongs to another case matrix')
    def stats(v):
        values=list(v);r=failure_stats(values)
        if values:r['cv']=statistics.pstdev(values)/statistics.fmean(values) if statistics.fmean(values) else None
        return r
    def onset(values,threshold):
        return next((j for j in range(3,len(values)-1) if values[j]>threshold and values[j+1]>threshold),None)
    processes,call_table,role_table=[],[],[]
    for source in data['processes']:
        p={k:source[k] for k in ('case','mode','swizzle','trial','host_start_ns','raw_sha256')}
        warm=source['prefix_call_event_us'];base_event=statistics.median(warm[:3])
        p['event_ratios']=[x/base_event-1 for x in warm]
        p['prefix_onsets']={}
        for threshold in (.03,.05,.10):
            j=onset(p['event_ratios'],threshold)
            p['prefix_onsets'][str(threshold)]=dict(index0=j,
                prior_kernel_us=sum(warm[:j]) if j is not None else None,
                event_elapsed_before_us=sum(warm[:j])+sum(source['event_gaps_us'][:j]) if j is not None else None)
        if p['mode']=='series':
            calls=source['calls'];roles=[]
            for call in calls:
                local={}
                for c,role,tiles,sm,C,ns in call['role_windows']:
                    if min(C,ns,tiles)<=0:raise ValueError('invalid same-role interval in shared report')
                    local[c,role]=dict(C=C,ns=ns,f=C/ns,sm=sm,tiles=tiles)
                roles.append(local)
            baseline={key:sorted((roles[j][key] for j in range(3)),key=lambda w:w['ns'])[1] for key in roles[0]}
            for j in range(16):
                if set(roles[j])!=set(baseline) or any(roles[j][k]['tiles']!=baseline[k]['tiles'] for k in baseline):
                    raise ValueError('same CTA/role did not retain its prefix work count')
            per_call=[];per_role_onsets=[]
            for key,b in baseline.items():
                r=dict(case=p['case'],swizzle=p['swizzle'],trial=p['trial'],cta=key[0],role=key[1],tiles=b['tiles'])
                for name in ('C','ns','f'):
                    r[name+'_onset_index0']=onset([roles[j][key][name]/b[name]-1 for j in range(16)],.05)
                per_role_onsets.append(r);role_table.append(r)
            for j,local in enumerate(roles):
                comparisons={name:[local[k][name]/b[name]-1 for k,b in baseline.items()] for name in ('C','ns','f')}
                residual=max(abs(math.log(local[k]['C']/b['C'])-math.log(local[k]['f']/b['f'])-math.log(local[k]['ns']/b['ns'])) for k,b in baseline.items())
                if residual>1e-12:raise ValueError('paired C=f*ns identity failed')
                r=dict(case=p['case'],swizzle=p['swizzle'],trial=p['trial'],index0=j,phase='prefix' if j<16 else 'target',
                    C_ratio=statistics.median(comparisons['C']),ns_ratio=statistics.median(comparisons['ns']),f_ratio=statistics.median(comparisons['f']),
                    role_ns_over5_fraction=sum(x>.05 for x in comparisons['ns'])/len(baseline),
                    same_sm_as_baseline_fraction=sum(local[k]['sm']==b['sm'] for k,b in baseline.items())/len(baseline),
                    prior_kernel_us=sum(c['event_us'] for c in calls[:j]),
                    elapsed_globaltimer_us=(calls[j]['first_entry_ns']-calls[0]['first_entry_ns'])/1000,
                    event_us=calls[j]['event_us'],local_ns_median=statistics.median(x['ns'] for x in local.values()),
                    local_f_median=statistics.median(x['f'] for x in local.values()),role_count=len(local))
                per_call.append(r);call_table.append(r)
            p['matched_role_calls']=per_call;p['matched_role_onsets']={}
            for threshold in (.03,.05,.10):
                p['matched_role_onsets'][str(threshold)]={}
                for name in ('C','ns','f'):
                    j=onset([c[name+'_ratio'] for c in per_call[:16]],threshold)
                    p['matched_role_onsets'][str(threshold)][name]=dict(index0=j,
                        prior_kernel_us=per_call[j]['prior_kernel_us'] if j is not None else None,
                        elapsed_globaltimer_us=per_call[j]['elapsed_globaltimer_us'] if j is not None else None)
            p['role_onset_spread']={name:dict(histogram=dict(Counter(str(r[name+'_onset_index0']) for r in per_role_onsets)),
                detected=sum(r[name+'_onset_index0'] is not None for r in per_role_onsets),total=len(per_role_onsets)) for name in ('C','ns','f')}
        processes.append(p)
    series=[p for p in processes if p['mode']=='series'];summary={}
    for key in ('0.03','0.05','0.1'):
        summary[key]={}
        for name in ('C','ns','f'):
            group=[p['matched_role_onsets'][key][name] for p in series]
            summary[key][name]=dict(index0=stats(p['index0'] for p in group if p['index0'] is not None),
                histogram=dict(Counter(str(p['index0']) for p in group)),
                prior_kernel_us=stats(p['prior_kernel_us'] for p in group if p['index0'] is not None),
                elapsed_globaltimer_us=stats(p['elapsed_globaltimer_us'] for p in group if p['index0'] is not None))
    event_summary={mode:{key:dict(index0=stats(p['prefix_onsets'][key]['index0'] for p in processes if p['mode']==mode and p['prefix_onsets'][key]['index0'] is not None),
        histogram=dict(Counter(str(p['prefix_onsets'][key]['index0']) for p in processes if p['mode']==mode)),
        prior_kernel_us=stats(p['prefix_onsets'][key]['prior_kernel_us'] for p in processes if p['mode']==mode and p['prefix_onsets'][key]['index0'] is not None))
        for key in ('0.03','0.05','0.1')} for mode in ('single','series')}
    paired=[]
    for p in series:
        on=p['matched_role_onsets']['0.05'];f,ns=on['f']['index0'],on['ns']['index0']
        paired.append(dict(case=p['case'],trial=p['trial'],f_index0=f,ns_index0=ns,
            f_to_ns_calls=ns-f if f is not None and ns is not None else None))
    result=dict(run=data['run'],environment=data['environment'],source_sha256=clock.sha256(Path(__file__)),
        input_sha256={str(path):clock.sha256(path)},parent_verification=data['verification'],
        matched_role_onset_summary=summary,event_onset_summary=event_summary,processes=processes,
        frequency_and_ns_onsets=paired,
        method=dict(baseline='For each logical CTA and consumer role, select the median-ns call among first three prefix calls and retain its paired C/f/ns.',
            window='Same CTA entry to the same consumer-role final endpoint. Shared decoder already checked role SM IDs; this function requires positive nonempty windows and unchanged prefix work. CTA index is software work identity, not a pinned physical SM. Absolute cycles across calls are never subtracted.',
            onset='First call index j>=3 with two consecutive same-role median ratios above threshold. Primary 5%; 3% and 10% are sensitivity, not physical cutoffs.',
            time='Prior kernel event sum and actual globaltimer elapsed-before-call are both reported. Fixed per-call work leaves repetition count and cumulative active time confounded.',
            limits='Ends do not localize first/later source panels inside the affected call. All raw decoding comes from A-history-series; no extra parser, clock fit, GPU task or current experiment-page edit.'))
    (output/'A-history-localization.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,table in [('calls',call_table),('role-onsets',role_table)]:
        with (output/f'A-history-localization-{name}.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    return result


def mirror_pair_metrics(same, mirror):
    """Match CTA/role work; compare 19 targets without treating target1 as steady state."""
    if len(same['calls'])!=35 or len(mirror['calls'])!=35:
        raise ValueError('mirror comparison requires all sixteen prefix and nineteen target calls')
    comparisons=[]
    for j,(a,b) in enumerate(zip(same['calls'],mirror['calls'])):
        windows=[]
        for c in (a,b):
            local={(x[0],x[1]):dict(tiles=x[2],sm=x[3],C=x[4],ns=x[5]) for x in c['role_windows']}
            if len(local)!=len(c['role_windows']) or any(min(w['tiles'],w['C'],w['ns'])<=0 for w in local.values()):
                raise ValueError('missing, duplicate or nonpositive same-role mirror windows')
            windows.append(local)
        left,right=windows
        if set(left)!=set(right) or any(left[k]['tiles']!=right[k]['tiles'] for k in left):
            raise ValueError('mirror comparison changed logical CTA/role work')
        ns=[right[k]['ns']/left[k]['ns']-1 for k in left]
        cycles=[right[k]['C']/left[k]['C']-1 for k in left]
        freq=[(right[k]['C']/right[k]['ns'])/(left[k]['C']/left[k]['ns'])-1 for k in left]
        q=dict(index=j,phase='prefix' if j<16 else 'target',target_index=None if j<16 else j-16,
            role_window_change=statistics.median(ns),role_cycles_change=statistics.median(cycles),
            role_frequency_change=statistics.median(freq),role_count=len(left),
            event_change=b['event_us']/a['event_us']-1,window_change=b['window_us']/a['window_us']-1,
            same_event_us=a['event_us'],mirror_event_us=b['event_us'],
            same_role_ns=statistics.median(w['ns'] for w in left.values()),
            mirror_role_ns=statistics.median(w['ns'] for w in right.values()))
        comparisons.append(q)
    indices=dict(prefix_all=range(16),prefix_first3=range(3),prefix_last3=range(13,16),
        target_first1=range(16,17),target_first3=range(16,19),target_middle3=range(24,27),
        target_last3=range(32,35),target_all=range(16,35))
    fields=('role_window_change','role_cycles_change','role_frequency_change','event_change','window_change')
    segments={name:{key:statistics.median(comparisons[j][key] for j in segment) for key in fields}
              for name,segment in indices.items()}
    return dict(calls=comparisons,segments=segments,
        prefix_event_span_change=mirror['prefix_event_us']/same['prefix_event_us']-1,
        prefix_kernel_sum_change=mirror['prefix_kernel_sum_us']/same['prefix_kernel_sum_us']-1,
        target_gap_reduction={key:abs(segments['target_first3'][key])-abs(segments['target_last3'][key]) for key in fields},
        note='Positive gap reduction means the absolute mirror/same difference shrank; it does not alone prove catch-up or identify a physical source mechanism.')


def mirror_nvml_snapshots(root):
    """Read only the two existing whole-batch snapshots for this exact GPU."""
    root=Path(root);uuid=clock.read_json(root/'environment.json')['gpu'].split(',')[0]
    wanted={'Clocks Event Reasons','Clocks Event Reasons Counters','Utilization','Temperature',
        'GPU Power Readings','GPU Memory Power Readings','Module Power Readings','Clocks','Max Clocks'}
    snapshots={}
    for name in ('nvidia-smi-before.txt','nvidia-smi-after.txt'):
        text=(root/name).read_text()
        blocks=[b for b in re.split(r'(?m)^GPU ',text)[1:] if uuid in b]
        if len(blocks)!=1:raise ValueError('snapshot does not identify exactly the run GPU')
        block=blocks[0];sections={};current=None
        for line in block.splitlines():
            if re.match(r'^    \S',line):
                current=line.strip() if ':' not in line else None
                if current in wanted:sections[current]={}
            elif current in wanted and ':' in line:
                key,value=line.split(':',1);sections[current][key.strip()]=value.strip()
        snapshots[name]=dict(timestamp=re.search(r'^Timestamp\s*:\s*(.+)$',text,re.M)[1],
            gpu_header=block.splitlines()[0],uuid=uuid,sha256=clock.sha256(root/name),sections=sections)
    before,after=[snapshots[name]['sections']['Clocks Event Reasons Counters'] for name in ('nvidia-smi-before.txt','nvidia-smi-after.txt')]
    delta={key:int(after[key].split()[0])-int(value.split()[0]) for key,value in before.items()}
    return dict(run=str(root),source_sha256=clock.sha256(Path(__file__)),snapshots=snapshots,counter_delta_us=delta,
        scope='Existing batch-before/after snapshots on the exact run GPU. Cumulative counters cover the entire batch including initialization, timed calls and numeric checks; no counter delta locates timed calls or proves the cause of a particular cycle/ns increase. Endpoint clocks, power and temperatures may be idle snapshots and do not describe intervening trajectories.')


def mirror_history_diagnostics(root, output):
    """R09 same/mirror source prehistory, allocation slot stratified, no clock fit."""
    import gzip
    from analyze_r13_sm import select_attempts
    from analyze_r18 import replay
    import v06_run as common
    root,output=Path(root),Path(output);common.verify(root)
    rows=clock.read_json(root/'cases.json')
    if len(rows)!=4 or {(r.get('history_source'),r.get('mirror_target_slot')) for r in rows}!={(s,t) for s in ('same','mirror') for t in (0,1)}:
        raise ValueError('same/mirror by both allocation slots required')
    if any(r.get('history_mode')!='series' or r['swizzle']!=1 or r['warmup_policy']!='common-sw1-16' for r in rows):
        raise ValueError('source contrast must retain the full sw1 history protocol')
    setups={r['case']:r['setup'] for r in clock.read_json(root/'static_setup.json')}
    fixed=('gpu_uuid','config','m','n','k','lda','ldb','ldd','alloc_lda','alloc_ldb','allocated_a_elements',
        'allocated_b_elements','input_mode','seed','grid','stages','zero_m','zero_n','evict','history_capacity_bytes','source_copies')
    anchor=setups[rows[0]['id']]
    if anchor.get('source_copies')!=2 or any(any(setups[r['id']].get(k)!=anchor.get(k) for k in fixed) for r in rows):
        raise ValueError('source-history comparison changed allocation/input/device conditions')
    inputs={name:clock.sha256(root/name) for name in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/binary_hashes.json')}
    processes=[];failures=[];records={}
    for row in rows:
        selected,failed,_=select_attempts(root,row['id'],('ends',));failures.extend(failed)
        for item in selected:
            record=item['record'];observed=replay(root,record,row)
            with gzip.open(root/record['raw'],'rt') as stream:
                call=next(e for line in stream if line.strip() for e in [json.loads(line)] if e['event']=='call')
            p=dict(case=row['id'],history_source=row['history_source'],target_slot=row['mirror_target_slot'],trial=record['trial'],
                host_start_ns=record['host_start_ns'],host_stop_ns=record['host_stop_ns'],raw_sha256=record['raw_sha256'],
                raw=record['raw'],record_path=item['path'],
                protocol=call['history_protocol'],preceding_target_calls=call['preceding_target_calls'],
                prefix_event_us=call['history_prefix_event_us'],prefix_kernel_sum_us=sum(call['warmup_us']),
                total_event_us=call['history_event_us'],prefix_call_event_us=call['warmup_us'],
                prefix_last5_cv=statistics.pstdev(call['warmup_us'][-5:])/statistics.fmean(call['warmup_us'][-5:]),
                source_identity={key:call[key] for key in ('history_source','mirror_target_slot','source_copies',
                    'target_a_ptr','target_b_ptr','prefix_a_ptr','prefix_b_ptr')},
                calls=history_call_metrics(observed['history_series'],call['history_inter_call_event_us']))
            fields=('median_full_cta_window_us','median_full_cta_cycles','max_cta_cycles','frequency_ghz','event_us')
            c=p['calls'];early={key:statistics.median(x[key] for x in c[:3]) for key in fields}
            p['relative_to_early_prefix']={segment:{key:statistics.median(c[j][key] for j in indices)/early[key]-1 for key in fields}
                for segment,indices in dict(prefix_last3=range(13,16),target_first1=range(16,17),
                    target_first3=range(16,19),target_middle3=range(24,27),target_last3=range(32,35)).items()}
            processes.append(p);records[p['history_source'],p['target_slot'],p['trial']]=p
            inputs[item['path']]=clock.sha256(root/item['path'])
    if set(records)!={(source,slot,trial) for source in ('same','mirror') for slot in (0,1) for trial in range(10)}:
        raise ValueError('ten successful full-history processes per source/slot required')
    pairs=[]
    for slot in (0,1):
        for trial in range(10):
            a,b=records['same',slot,trial],records['mirror',slot,trial]
            pairs.append(dict(target_slot=slot,trial=trial,
                first_source='same' if a['host_start_ns']<b['host_start_ns'] else 'mirror',
                separation_s=abs(a['host_start_ns']-b['host_start_ns'])/1e9,**mirror_pair_metrics(a,b)))
    fields=('role_window_change','role_cycles_change','role_frequency_change','event_change','window_change')
    strata={}
    for label,group in [('pooled',pairs),('slot0',[p for p in pairs if p['target_slot']==0]),('slot1',[p for p in pairs if p['target_slot']==1])]:
        strata[label]=dict(order_counts={s:sum(p['first_source']==s for p in group) for s in ('same','mirror')},
            prefix_event_span_change=failure_stats(p['prefix_event_span_change'] for p in group),
            prefix_kernel_sum_change=failure_stats(p['prefix_kernel_sum_change'] for p in group),
            segments={segment:{key:failure_stats(p['segments'][segment][key] for p in group) for key in fields}
                for segment in group[0]['segments']},
            by_call=[dict(index=j,phase='prefix' if j<16 else 'target',
                metrics={key:failure_stats(p['calls'][j][key] for p in group) for key in fields}) for j in range(35)],
            gap_reduction={key:failure_stats(p['target_gap_reduction'][key] for p in group) for key in fields})
    progression={}
    for source in ('same','mirror'):
        progression[source]={}
        for label,slot in (('pooled',None),('slot0',0),('slot1',1)):
            group=[p for p in processes if p['history_source']==source and (slot is None or p['target_slot']==slot)]
            progression[source][label]={segment:{key:failure_stats(p['relative_to_early_prefix'][segment][key] for p in group)
                for key in group[0]['relative_to_early_prefix'][segment]} for segment in group[0]['relative_to_early_prefix']}
    snapshots=mirror_nvml_snapshots(root)
    inputs.update({name:value['sha256'] for name,value in snapshots['snapshots'].items()})
    # Paired role ratios have been computed; keep their summaries and raw
    # locations instead of duplicating all 1,400 calls' role windows in JSON.
    for process in processes:
        for call in process['calls']:
            del call['role_windows']
    result=dict(scope='Same/mirror source intervention with nineteen-target trajectories; mechanism development, not prediction validation.',
        run=str(root),environment=clock.read_json(root/'environment.json'),source_sha256=clock.sha256(Path(__file__)),
        input_sha256=inputs,conditions={key:anchor.get(key) for key in fixed},processes=processes,
        paired_trials=pairs,strata=strata,relative_to_early_prefix=progression,failed_attempts=failures,
        batch_counter_delta_us=snapshots['counter_delta_us'],batch_counter_scope=snapshots['scope'],
        verification=dict(source_and_binary_hashes=True,successful_raw_replays=len(processes),series_calls=sum(len(p['calls']) for p in processes),
            fixed_prefix_cv_exclusions=0,same_different_source_pointer_checks='Shared validate_warmup compares each process internally; setup and measurement processes need not have equal absolute pointer values.'),
        interpretation=['Read both prefix pressure curves and paired prefix duration differences before assigning target differences to source-address history.',
            'The target source and swizzle stay fixed. Source allocation slot is a separate stratum; any slot-dependent response must be retained before pooling.',
            'Compare target1 and the first/middle/last three target calls. A target1 difference may be cold-source or descriptor first-use and does not by itself establish repeated-address degradation.',
            'Catch-up means a sustained target trajectory difference narrows, not merely equal final medians. Gap-reduction statistics preserve direction and process spread rather than supplying a binary acceptance gate.',
            'Unchanged target source after disjoint prehistory can support a device-wide-state explanation only conditionally on similar prefix pressure; a changed response establishes source identity involvement, not a named cache/TLB/HBM mechanism.',
            'Every local ratio matches logical CTA and consumer role work and uses within-call local endpoints. Absolute cycles across SMs or processes are never subtracted.',
            'Per-role windows are decoded once for paired metrics, then omitted from this report; every process retains raw and record paths plus raw SHA for further inspection.',
            'Within-process early-prefix ratios use the median of three per-call metrics; median_full_cta_window_us is a median of same-CTA C/f intervals, not maximum C divided by a cross-process frequency.',
            'New extra input capacity and initialization differ from 739579. Older runs are qualitative references only; no old absolute clock coefficient is used.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'A-mirror-history.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'A-mirror-nvml-snapshots.json').write_text(json.dumps(snapshots,indent=2,allow_nan=False)+'\n')
    with (output/'A-mirror-history-pairs.csv').open('w') as stream:
        rows=[dict(target_slot=p['target_slot'],trial=p['trial'],first_source=p['first_source'],**c) for p in pairs for c in p['calls']]
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def response_development(base, output):
    """Declared response/activity contrasts on one training card; development only."""
    from collections import defaultdict
    from r09_equal_work_clock import work_point
    import v09_model

    base, output = Path(base), Path(output)
    paths = dict(training=base/'20261009-R09-input-clock-calibration-job738376/reanalysis/C-20261009-source-clock-dev-v2/source-clock.json',
        equal_work=base/'20261009-R09-equal-work-job738459/reanalysis/C-20261009-equal-work-clock-v1/equal-work-clock.json',
        v09_calls=base/'20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/A-clock-failure.json',
        new_controls=base/'20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/A-controls.json')
    data = {k:clock.read_json(p) for k,p in paths.items()}
    training = [dict(p,output_mean_kib=static_output_kib(p)) for p in data['training']['points']['all21_training']]
    equal_root = base/'20261009-R09-equal-work-job738459'
    equal_rows = {r['id']:r for r in clock.read_json(equal_root/'cases.json')}
    cross_k = [dict(p,output_mean_kib=static_output_kib(equal_rows[p['case']]))
               for p in data['equal_work']['clock_cases'] if p['k']==4096]
    revisit_support = {}
    def attach_revisit(points,run):
        row_map = {r['id']:r for r in clock.read_json(run/'cases.json')}
        setup_map = {s['case']:s['setup'] for s in clock.read_json(run/'static_setup.json')}
        features = {}
        for cid in {p['case'] for p in points}:
            try:
                features[cid] = static_revisit_features(row_map[cid],setup_map[cid])
                revisit_support[cid] = dict(status='full-tile source union defined',
                    unique_source_mean_KiB=features[cid]['unique_source_mean_KiB'],
                    multicast_source_slices_verified=features[cid]['multicast_source_slices_verified'])
            except ValueError as error:
                features[cid] = None
                revisit_support[cid] = dict(status='unsupported',reason=str(error))
        for name in ('cases.json','static_setup.json','environment.json'):
            paths[str(run/name)] = run/name
        return [dict(p,revisit_mean_kib=(features[p['case']]['unique_source_mean_KiB'] if features[p['case']] else None)) for p in points]
    training = attach_revisit(training,base/'20261009-R09-input-clock-calibration-job738376')
    cross_k = attach_revisit(cross_k,equal_root)
    if any(p['revisit_mean_kib'] is None for p in training+cross_k):
        raise ValueError('U must support every original21 and added9 training point; no training rows may be silently omitted')
    gpu = data['training']['environment']['gpu'].split(',')[0]
    if data['equal_work']['environment']['gpu'].split(',')[0] != gpu:
        raise ValueError('cross-K training must use the same GPU')
    cap = data['training']['model']['constraints']['a_max_ghz']
    models = dict(linear21=dict(data['training']['model'],response='linear'),
                  reciprocal21=fit_reciprocal_clock(training,cap))
    def frequency(point, window, model):
        mixed = mixed_failure_clock(model,point['input_mode'],point.get('zero_fraction',0))
        if model['response']=='linear_output':
            point = dict(point,source_mean_kib=point['output_mean_kib'])
        if model['response']=='linear_revisit':
            if point['revisit_mean_kib'] is None:
                return None
            point = dict(point,source_mean_kib=point['revisit_mean_kib'])
        return (reciprocal_frequency(point,window,mixed) if model['response']=='reciprocal'
                else source_frequency(point,window,mixed)['frequency_ghz'])
    def evaluate(points):
        result = {}
        for name,model in models.items():
            groups = defaultdict(list)
            unsupported = set()
            for point in points:
                predicted = frequency(point,point['observed_window_us'],model)
                if predicted is None:
                    unsupported.add(point['case'])
                    continue
                groups[point['case']].append(predicted/point['observed_ghz']-1)
            by_case = {case:failure_stats(errors) for case,errors in groups.items()}
            result[name] = dict(summary=failure_stats(v['median'] for v in by_case.values()),cases=by_case,
                                unsupported_cases=sorted(unsupported),declared_cases=len({p['case'] for p in points}))
        return result
    preliminary = evaluate(cross_k)
    datasets = dict(training21=training,crossK4096=cross_k,
                    same_card_wideK1024=[dict(p,output_mean_kib=static_output_kib(p))
                                        for p in data['training']['points']['old_cfg_ac_long']])
    datasets['same_card_wideK1024'] = attach_revisit(datasets['same_card_wideK1024'],base/'20261009-R09-wide-input-job738197')
    target = base/'20261009-V09-heldout-job739011'
    rows = {r['id']:r for r in clock.read_json(target/'cases.json')}
    setups = {s['case']:s['setup'] for s in clock.read_json(target/'static_setup.json')}
    v09points = []
    for item in data['v09_calls']['processes']:
        row = rows[item['case']]
        point = work_point(row,setups[row['id']])
        work = v08_model.scheduled_work(row['config'],row['m'],row['n'],setups[row['id']]['grid'],row['swizzle'],row.get('raster_order','heuristic'))
        v09points.append(dict(point,case=row['id'],config=row['config'],group=row['group'],
            output_mean_kib=static_output_kib(row),
            zero_fraction=v09_model.zero_product_fraction(row,work),trial=item['trial'],
            observed_window_us=item['observed_window_us'],observed_ghz=item['observed_frequency_ghz']))
    v09points = attach_revisit(v09points,target)
    datasets['V09_all'] = v09points
    for group in sorted({p['group'] for p in v09points}):
        datasets['V09_'+group] = [p for p in v09points if p['group']==group]
    # More data is a separate development version, never a replacement for the
    # original-21 failure. Refit both laws to distinguish form from new evidence.
    original21_migration = {name:evaluate(datasets[name]) for name in
                            ('crossK4096','same_card_wideK1024','V09_G6_base')}
    add_cross_k = any(abs(case['median']) > .10 for group in original21_migration.values()
                      for case in group['reciprocal21']['cases'].values())
    if add_cross_k:
        models['linear30'] = dict(fit_source_clock(training+cross_k,cap),response='linear')
        models['reciprocal30'] = fit_reciprocal_clock(training+cross_k,cap)
    output_training = [dict(p,source_mean_kib=p['output_mean_kib']) for p in training+cross_k]
    models['linear_output30'] = dict(fit_source_clock(output_training,cap),response='linear_output',
                                     activity_input='static effective D KiB per SM, replacing source KiB')
    if 'linear30' not in models:
        models['linear30'] = dict(fit_source_clock(training+cross_k,cap),response='linear')
    for name,points in [('linear_revisit21',training),('linear_revisit30',training+cross_k)]:
        models[name] = dict(fit_source_clock([dict(p,source_mean_kib=p['revisit_mean_kib']) for p in points],cap),
                            response='linear_revisit',activity_input='static source-box union per software output ordinal')
    import numpy as np
    identification = {}
    revisit_identification = {}
    for name,points in [('original21',training),('expanded30',training+cross_k)]:
        work_design = np.asarray([[1,
            *[-p['tensor_mean_cycles']/(1000*p['observed_window_us'])*(p['input_mode']==mode) for mode in MODES],
            *[-p['output_mean_kib']/p['observed_window_us']*(p['input_mode']==mode) for mode in MODES]] for p in points])
        with_duration = source_design([dict(p,source_mean_kib=p['output_mean_kib']) for p in points],
                                      models['linear_output30']['tau_us'])
        identification[name] = dict(work_only=clock.input_design_rank(work_design),
            with_duration=clock.input_design_rank(with_duration),
            output_bytes_per_nominal_cycle=sorted({round(p['output_mean_kib']*1024/p['tensor_mean_cycles'],12) for p in points}))
        union_design = work_design.copy()
        union_design[:,4:] = np.asarray([[-p['revisit_mean_kib']/p['observed_window_us']*(p['input_mode']==mode)
                                          for mode in MODES] for p in points])
        union_model = models['linear_revisit21' if name=='original21' else 'linear_revisit30']
        revisit_identification[name] = dict(work_only=clock.input_design_rank(union_design),
            with_duration=clock.input_design_rank(source_design([dict(p,source_mean_kib=p['revisit_mean_kib']) for p in points],union_model['tau_us'])),
            training_count=len(points),unsupported_training_count=0)
    scores = {name:evaluate(points) for name,points in datasets.items()}
    common_scores = {name:evaluate([p for p in points if p['revisit_mean_kib'] is not None]) for name,points in datasets.items()}
    # Other cards are never pooled into fits or absolute-error scoreboards.
    controls = data['new_controls']
    control_root = Path(controls['run'])
    control_rows = {r['id']:r for r in clock.read_json(control_root/'cases.json')}
    control_setups = {s['case']:s['setup'] for s in clock.read_json(control_root/'static_setup.json')}
    trends = []
    for variant in ('ends','dual'):
        for trial in range(10):
            pair = [next(p for p in controls['processes'] if p['k']==k and p['variant']==variant and p['trial']==trial)
                    for k in (4096,32768)]
            observed = pair[1]['frequency_ghz']/pair[0]['frequency_ghz']
            points = [dict(work_point(control_rows[p['case']],control_setups[p['case']]),
                           output_mean_kib=static_output_kib(control_rows[p['case']]),
                           revisit_mean_kib=static_revisit_features(control_rows[p['case']],control_setups[p['case']])['unique_source_mean_KiB']) for p in pair]
            trends.append(dict(card=controls['environment']['gpu'],variant=variant,trial=trial,
                observed_high_over_low=observed,
                predicted_high_over_low={name:frequency(points[1],pair[1]['window_us'],model)/frequency(points[0],pair[0]['window_us'],model)
                    for name,model in models.items()}))
    older_root = base/'20261009-R09-clock-input-job738110'
    older_summary = clock.read_json(older_root/'analysis/summary.json')
    older_rows = {r['id']:r for r in clock.read_json(older_root/'cases.json')}
    older_setups = {s['case']:s['setup'] for s in clock.read_json(older_root/'static_setup.json')}
    other_card_input_trends = []
    for cfg in ('cfg_a','cfg_b','cfg_c'):
        cases = [f'{cfg}_r09_k20480_{mode}' for mode in ('dyadic','zero')]
        summary = [older_summary['cases'][cid] for cid in cases]
        points = [dict(work_point(older_rows[cid],older_setups[cid]),output_mean_kib=static_output_kib(older_rows[cid]),
                       revisit_mean_kib=static_revisit_features(older_rows[cid],older_setups[cid])['unique_source_mean_KiB']) for cid in cases]
        other_card_input_trends.append(dict(config=cfg,observed_zero_over_dyadic=summary[1]['ghz_ends']/summary[0]['ghz_ends'],
            predicted_zero_over_dyadic={name:frequency(points[1],summary[1]['window_ends'],model)/frequency(points[0],summary[0]['window_ends'],model)
                for name,model in models.items()}))
    for label,root,names in [('v09',target,('cases.json','static_setup.json','environment.json')),
                            ('equal',equal_root,('cases.json',)),
                            ('new_card',control_root,('cases.json','static_setup.json','environment.json')),
                            ('old_long',older_root,('cases.json','static_setup.json','environment.json','analysis/summary.json'))]:
        for name in names:
            paths[label+'/'+name] = root/name
    swizzle_path = base/'20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-swizzle.json'
    swizzle = clock.read_json(swizzle_path)
    paths['swizzle_diagnostic'] = swizzle_path
    sw_root = Path(swizzle['run'])
    sw_rows = {r['id']:r for r in clock.read_json(sw_root/'cases.json')}
    sw_setups = {s['case']:s['setup'] for s in clock.read_json(sw_root/'static_setup.json')}
    swizzle_trends = []
    for trial in range(10):
        pair = [next(p for p in swizzle['processes'] if p['swizzle']==sw and p['variant']=='ends' and p['trial']==trial) for sw in (1,8)]
        points = [dict(work_point(sw_rows[p['case']],sw_setups[p['case']]),
            output_mean_kib=static_output_kib(sw_rows[p['case']]),
            revisit_mean_kib=static_revisit_features(sw_rows[p['case']],sw_setups[p['case']])['unique_source_mean_KiB']) for p in pair]
        swizzle_trends.append(dict(trial=trial,observed_sw8_over_sw1=pair[1]['frequency_ghz']/pair[0]['frequency_ghz'],
            predicted_sw8_over_sw1={name:frequency(points[1],pair[1]['window_us'],model)/frequency(points[0],pair[0]['window_us'],model)
                for name,model in models.items()}))
    result = dict(scope='Conditional-frequency development, not a new validation or total-time fit.',
        source_sha256=clock.sha256(Path(__file__)),input_sha256={str(p):clock.sha256(p) for p in paths.values()},
        training_gpu=gpu,training_cases=[p['case'] for p in training],added_training_cases=[p['case'] for p in cross_k] if add_cross_k else [],
        additional_split_trigger='Original21 reciprocal maximum absolute error on known same-card K4096/wide-K1024/G6 groups >10%; a development choice, not a pass criterion.',
        original21_crossK_result=preliminary,original21_migration=original21_migration,
        models=models,scores=scores,common_revisit_supported_scores=common_scores,revisit_support=revisit_support,
        swizzle_trends=swizzle_trends,new_card_trends=trends,output_activity_identification=identification,
        revisit_identification=revisit_identification,
        old_long_card=older_summary['environment'],old_long_input_trends=other_card_input_trends,
        root_conditions=dict(
            positivity='a>=1/1.98 and d/e>=0 give 0<f<=1/a<=1.98 without any lower clamp.',
            monotonicity='For Q/S>=0, g(W/tau)/W and Q/W are nonincreasing, so reciprocal F(W) is nondecreasing.',
            closure='If E(f)>0 is continuous and nonincreasing for f>0 and E(f)->infinity as f->0, h(f)=F(E(f))-f strictly decreases; h(0+)=1/a>0 and h(1/a)<=0. Exactly one root in (0,1/a].',
            limitation='No proof of the event-envelope premise or full-time improvement is implied by a conditional fit.'),
        notes=['The reciprocal form keeps seven coefficients plus one shared tau; only one alternative response is tested.',
               'All fits minimize observed GHz residuals. Reciprocal-target linear least squares initializes the nonlinear solver only.',
               'Original21 and expanded30 use one GPU; V09 observations are never fitted. Expanded30 K4096 training scores are not migration tests.',
               'V09 dual and original ends frequency targets use related but different observation protocols.',
               'Job738110 is GPU-201f9d2a on a043, not the GPU-43269fbc training card; it is used only for input-response trends.',
               'New-card trends use observed windows as conditional diagnostics; they establish neither transferred absolute clocks nor predictive timing.',
               'Coefficient zero-fraction mixing is retained only as the same unverified boundary hypothesis; it is not a physical power law.'])
    result['notes'].append('linear_output30 replaces S with static effective FP32 D/132 in the original linear response; Q and D have input-specific nonnegative prices, no S+D added parameter. Original21 static D/Q is fixed, but the duration factor can supply algebraic rank without independent output-work intervention.')
    result['notes'].append('linear_revisit21/30 replace S by U without adding coefficients; all 21/30 training points must have a full-tile multicast-box union. Partial/padded V09 cases remain declared unsupported, and common-support baseline scores are separate. New-card swizzle observations never fit U coefficients; adaptive warmup remains confounded.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'A-response-candidates.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--calibration-run', type=Path, required=True)
    parser.add_argument('--old-long-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze_source_clock(args.run.resolve(), args.calibration_run.resolve(), args.old_long_run.resolve(), args.output.resolve())
