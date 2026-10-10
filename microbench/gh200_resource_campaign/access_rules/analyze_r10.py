#!/usr/bin/env python3
"""R10 CPU reference, stride/padding evidence and paired mainloop/time analysis."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from fractions import Fraction


def input_value(row, col, a, seed=17):
    return ((row * (7 if a else 5) + col * (13 if a else 11)
             + seed * (3 if a else 5)) % 17 - 8) / 32.0


def reference(row, col, k, seed=17):
    # Products repeat after17 K coordinates; exact dyadic sums allow direct regrouping.
    full, tail = divmod(k, 17)
    cycle = sum(input_value(row, i, True, seed) * input_value(i, col, False, seed)
                for i in range(17))
    return full * cycle + sum(input_value(row, i, True, seed)
                             * input_value(i, col, False, seed) for i in range(tail))


def cpu_check():
    for row, col in ((0, 0), (127, 128), (2599, 2999), (513, 1027)):
        assert reference(row, col, 2000) == sum(
            input_value(row, i, True) * input_value(i, col, False) for i in range(2000))
    for stride in (3000, 3008, 3072):
        sentinel = 0x3555
        storage = [sentinel] * (3 * stride)
        for row in range(3):
            for col in range(3000):
                storage[row * stride + col] = input_value(row, col, False)
        assert all(storage[row * stride + col] == sentinel
                   for row in range(3) for col in range(3000, stride))
        assert storage[2 * stride + 2999] == input_value(2, 2999, False)
    # Last Ktile is partial: keep complete mainloop/ceil(K/64), without subtracting c0.
    assert (2000 + 63) // 64 == 32
    for begin, end in ((100, 400), (950, 1650)):
        assert (end - begin) / 32 > 0
    print('CPU exact reference, B physical row addresses and padding witness passed; 9 cases.')
    assert [extra_coverage(k, 128) for k in B_PITCH_CASES] == [0, 56, 48, 32, 0]
    assert [extra_coverage(k, 32) for k in B_PITCH_CASES] == [0, 32, 0, 0, 0]
    example = {0:300, 16:440, 32:420}
    small, large = fit_coverage(example, 32), fit_coverage(example, 128)
    assert not small['parameters_unique'] and small['calibration_status'] == 'fail'
    assert all(small['predictions'][k]['ns_per_kt'] == 300 for k in (64, 128))
    assert large['parameters_unique'] and large['calibration_status'] == 'pass'
    assert large['q0_range'] == [300, 300] and large['q1_range'] == [2.5, 2.5]
    assert large['predictions'][64]['ns_per_kt'] == 380
    plateau = fit_coverage({0:300, 16:330, 32:300}, 128)
    assert not plateau['parameters_unique'] and plateau['loss_ns_squared'] == 0
    assert all(plateau['predictions'][k]['unique'] and plateau['predictions'][k]['ns_per_kt'] == 300 for k in (64, 128))
    assert fit_coverage({0:300, 16:300, 32:430}, 128)['calibration_status'] == 'fail'
    print('Coverage counts, nonunique parameters with unique heldout predictions, and failed candidates passed.')


def cv(values):
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None


def check(record):
    case = record['configuration']
    if record.get('preparation_protocol') != 'symmetric_scratch_memset_sync_v3':
        raise ValueError('requires symmetric preparation; analyze v2 with frozen v2 script')
    if record.get('scratch_bytes') != 1024 * 8 * 8:
        raise ValueError('plain/trace scratch sizes differ')
    witness = record['check']
    points = [divmod(index, case['n']) for index in witness['checked_indices']]
    values = witness['checked_values']
    if len(points) != 4096 or len(points) != len(values) or len(set(map(tuple, points))) != len(points):
        raise ValueError('missing/duplicate saved numeric coordinates')
    if witness['status'] != 'ok' or witness['nonfinite'] or witness['padding_errors']:
        raise ValueError('device numerical/padding check failed')
    for key in ('m', 'n', 'k', 'lda', 'ldb', 'ldd'):
        if record[key] != case[key]:
            raise ValueError('descriptor/layout record mismatch')
    if record['work_flop'] != 2 * case['m'] * case['n'] * case['k']:
        raise ValueError('GEMM work mismatch')
    maximum = 0.0
    for (row, col), value in zip(points, values):
        if not (0 <= row < case['m'] and 0 <= col < case['n']) or not math.isfinite(value):
            raise ValueError('invalid saved value')
        maximum = max(maximum, abs(value - reference(row, col, case['k'])))
    if maximum != 0:
        raise ValueError('saved GEMM sample mismatch')
    padding = record['padding']
    if padding['b_sentinel_fp16_bits'] != 0x7bff:
        raise ValueError('FP16 padding sentinel identity')
    if padding['errors'] != 0:
        raise ValueError('padding sentinel changed')
    if padding['checked_elements'] != (case['k'] * (case['ldb'] - case['n'])):
        raise ValueError('B padding coverage differs')
    warmup = record['warmup_us']
    if not 8 <= len(warmup) <= 30:
        raise ValueError('warmup window count')
    recomputed = statistics.pstdev(warmup[-5:]) / statistics.mean(warmup[-5:]) <= 0.02
    if recomputed != record['warmup_converged']:
        raise ValueError('warmup qualification mismatch')
    if record['warmup_converged'] is not True:
        return maximum, False
    return maximum, True


def analyze(run, set_name):
    run = Path(run)
    environment = json.loads((run / f'environment-{set_name}.json').read_text())
    for name, digest in json.loads((run / 'source_hashes.json').read_text()).items():
        if hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen source changed')
    for name, digest in json.loads((run / 'build/binary_hashes.json').read_text()).items():
        if hashlib.sha256((run / 'build' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('binary changed')
    rows = []
    for path in sorted((run / 'samples' / set_name).glob('*/*/*/result.json')):
        record = json.loads(path.read_text())
        binary = record['configuration']['config'] + '_' + record['mode']
        if record['binary_sha256'] != hashlib.sha256(
                (run / 'build' / binary).read_bytes()).hexdigest():
            raise ValueError('sample binary identity differs')
        error, stable = check(record)
        record['cpu_max_error'] = error
        record['window_converged'] = stable
        rows.append(record)
    cases = json.loads((run / f'protocol-{set_name}.json').read_text())['cases']
    result = []
    for case in cases:
        members = [r for r in rows if r['configuration'] == case]
        plain = sorted((r for r in members if r['mode'] == 'plain'), key=lambda r: r['trial'])
        trace = sorted((r for r in members if r['mode'] == 'trace'), key=lambda r: r['trial'])
        if len(plain) != 10 or len(trace) != 10:
            raise ValueError('ten processes each plain/trace required')
        if [r['trial'] for r in plain] != [r['trial'] for r in trace]:
            raise ValueError('pair identity mismatch')
        us = [r['elapsed_us'] for r in plain]
        t_us = [r['elapsed_us'] for r in trace]
        disturbance = [t / p - 1 for p, t in zip(us, t_us)]
        mains, observations, missing = [], [], []
        limited = case['config'] == 'cfg_b'
        if limited and any(r.get('trace_schema') != 4 for r in members):
            raise ValueError('cfg_b requires v4 limited trace; keep v3 analysis separate')
        sampled_tile_counts = []
        for process in trace:
            words = process['trace']
            if len(words) != process['grid_ctas'] * 8:
                raise ValueError('trace length')
            count = 0
            for cta in range(process['grid_ctas']):
                w = words[cta * 8:cta * 8 + 8]
                if limited and cta >= 4:
                    if any(w):
                        raise ValueError('unselected CTA has trace writes')
                if limited and cta < 4:
                    if w[7] != 4 or not w[4] or w[2] or w[3] or w[5]:
                        raise ValueError('selected CTA lacks consumer0 or has consumer1 writes')
                for group in (0, 1):
                    if limited and (cta >= 4 or group != 0):
                        missing.append(dict(trial=process['trial'], cta=cta, group=group,
                                            mainloop_cycles=None, reason='outside_sampling_scope'))
                        continue
                    if not w[4 + group]:
                        continue
                    if w[group * 2] <= 0 or w[group * 2 + 1] <= w[group * 2]:
                        raise ValueError('mainloop trace event order')
                    count += w[4 + group] if case['config'] == 'cfg_b' or group == 0 else 0
                    cycles = w[group * 2 + 1] - w[group * 2]
                    mains.append(cycles / 32)
                    observations.append(dict(trial=process['trial'], cta=cta, group=group,
                                             sm_id=w[6], last_tile_mainloop_cycles=cycles,
                                             cycles_per_ktile_including_fill_tail=cycles / 32))
            tile_m = 256 if case['config'] == 'cfg_c' else 128
            # Cluster padding may assign fully OOB tiles; retain measured count in diagnostics.
            sampled_tile_counts.append(dict(trial=process['trial'],
                                            selected_consumer_finished_tiles=count))
            process['finished_tiles'] = None if limited else count
            process['logical_tiles'] = math.ceil(case['m'] / tile_m) * math.ceil(case['n'] / 128)
        acceptable = max(abs(v) for v in disturbance) <= 0.05
        result.append(dict(**case, processes_plain=len(plain), processes_trace=len(trace),
                           plain_us_median=statistics.median(us), plain_cv=cv(us),
                           trace_us_median=statistics.median(t_us),
                           paired_trace_over_plain=disturbance,
                           trace_disturbance_pass=acceptable,
                           mainloop_cycles_per_ktile_median=statistics.median(mains),
                           mainloop_cycles_per_ktile_range=[min(mains), max(mains)],
                           stage_boundary='first full-barrier wait returned to mma_tail returned',
                           trace_schema=4 if limited else 2,
                           sampled_ctas=[0, 1, 2, 3] if limited else 'all',
                           sampled_consumers=[0] if limited else [0, 1],
                           sampled_tile='last', total_finished_tiles_available=not limited,
                           sampled_tile_counts=sampled_tile_counts,
                           missing_observations=missing,
                           ktile_count=32, subtract_prefill=False,
                           max_check_error=max(r['cpu_max_error'] for r in members),
                           warmup_all_converged=all(r['window_converged'] for r in members),
                           status='observed' if acceptable and all(r['window_converged']
                                                                    for r in members)
                           else 'trace_or_warmup_not_qualified', cta_observations=observations))
    destination = run / ('analysis-' + set_name)
    destination.mkdir(exist_ok=False)
    summary = dict(environment=environment, cases=result,
                   preparation_protocol='symmetric_scratch_memset_sync_v3',
                   scope='fixed dimensions/configuration; physical ldb only; no cache attribution')
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = ['id', 'config', 'ldb', 'plain_us_median', 'plain_cv', 'trace_us_median',
              'trace_disturbance_pass', 'mainloop_cycles_per_ktile_median', 'status']
    with (destination / 'cases.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result)
    print(destination / 'summary.json')


def joint_pitch(run, output):
    """Same-shape A/B factorial, with same-call cycle/ns endpoints in dual traces."""
    from collections import defaultdict
    import gzip
    from analyze_r18 import replay
    from analyze_r13_sm import select_attempts, validate_setup, overlap_windows
    import v06_run as common

    common.verify(run)
    rows=json.loads((run/'cases.json').read_text())
    setups={r['case']:r['setup'] for r in json.loads((run/'static_setup.json').read_text())}
    processes=[];tiles=[];failed=[];summaries=[]
    for row in rows:
        selected, failures, ignored=select_attempts(run,row['id'],('plain','wide','stamped','dual'))
        failed.extend(failures)
        if ignored:raise ValueError('unexpected variants in factorial run')
        times=defaultdict(list);scratch={}
        for item in selected:
            rec=item['record'];obs=replay(run,rec,row)
            with gzip.open(run/rec['raw'],'rt') as stream:
                events={v['event']:v for line in stream if line.strip() for v in [json.loads(line)]}
            setup=events['setup'];validate_setup(row,setups[row['id']],setup)
            scratch[rec['variant']]=setup['scratch_bytes'];times[rec['variant']].append(rec['elapsed_us'])
            process=dict(case=row['id'],a_pitched=(row['lda']*2)%128!=0,b_pitched=(row['ldb']*2)%128!=0,
                trial=rec['trial'],variant=rec['variant'],elapsed_us=rec['elapsed_us'])
            if rec['variant']=='dual':
                if setup['trace_version']!='r18-dual-clock-events':raise ValueError('direct dual-clock trace required')
                cycle_last=max(obs['ctas'],key=lambda c:c['end_c']-c['entry_c'])
                time_last=max(obs['ctas'],key=lambda c:c['end_ns'])
                process.update(critical_cycle_cta=cycle_last['cta'],critical_time_cta=time_last['cta'],
                    cycle_choice_lag_ns=time_last['end_ns']-cycle_last['end_ns'],
                    envelope_ns=time_last['end_ns']-min(c['entry_ns'] for c in obs['ctas']),
                    max_cta_cycles=cycle_last['end_c']-cycle_last['entry_c'])
                later=[];first=[];supply=[];last_epi=[];windows=[]
                for c in obs['ctas']:
                    supply.append(c['tiles_ns'][0][0]-c['entry_ns'])
                    last_epi.append(c['tiles_ns'][-1][3]-c['tiles_ns'][-1][2])
                    for j,(cycle,ns,coord) in enumerate(zip(c['tiles'],c['tiles_ns'],c['work'])):
                        value=dict(case=row['id'],trial=rec['trial'],cta=c['cta'],sm=c['sm'],j=j,T=len(c['tiles']),
                            entry_ns=c['entry_ns'],final_ns=c['end_ns'],
                            mi=coord[0],ni=coord[1],L_cycles=cycle[1]-cycle[0],L_ns=ns[1]-ns[0],
                            E_cycles=cycle[3]-cycle[2],E_ns=ns[3]-ns[2],first_mma_ns=ns[0],main_end_ns=ns[1],
                            epi_permit_ns=ns[2],epi_done_ns=ns[3])
                        tiles.append(value);(later if j else first).append(value)
                        windows.append(dict(sm=c['sm'],entry_ns=ns[0],end_ns=ns[1],entry_c=cycle[0],end_c=cycle[1]))
                kt=(row['k']+63)//64
                overlap=overlap_windows(windows)
                process.update(L_later_cycle_per_kt=statistics.fmean(t['L_cycles'] for t in later)/kt,
                    L_later_ns_per_kt=statistics.fmean(t['L_ns'] for t in later)/kt,
                    L_first_cycle_per_kt=statistics.median(t['L_cycles'] for t in first)/kt,
                    L_first_ns_per_kt=statistics.median(t['L_ns'] for t in first)/kt,
                    mainloop_cycle_per_ns=statistics.median(t['L_cycles']/t['L_ns'] for t in later),
                    supply_ns=statistics.median(supply),last_epilogue_ns=statistics.median(last_epi),
                    mainloop_mean_sm_windows=overlap['mean_observed_sms'],mainloop_peak_sm_windows=overlap['peak_observed_sms'])
            processes.append(process)
        if len({scratch[v] for v in ('wide','stamped','dual')})!=1:raise ValueError('scratch clearing differs across matched variants')
        if any(len(times[v])!=10 for v in ('plain','wide','stamped','dual')):raise ValueError('ten successful processes per variant required')
        med={v:statistics.median(t) for v,t in times.items()}
        paired={v:[dict(trial=t,relative=next(p['elapsed_us'] for p in processes if p['case']==row['id'] and p['variant']==v and p['trial']==t)/
                    next(p['elapsed_us'] for p in processes if p['case']==row['id'] and p['variant']=='wide' and p['trial']==t)-1)
                   for t in range(10)] for v in ('stamped','dual')}
        summaries.append(dict(case=row['id'],a_pitched=(row['lda']*2)%128!=0,b_pitched=(row['ldb']*2)%128!=0,
            input_allocation_elements=dict(A=setups[row['id']].get('allocated_a_elements',row['storage_m']*row['lda']),
                                           B=setups[row['id']].get('allocated_b_elements',row['k']*row['ldb'])),
            elapsed_us=med,cv={v:statistics.pstdev(t)/statistics.mean(t) for v,t in times.items()},
            disturbance={v:med[v]/med['wide']-1 for v in ('stamped','dual')},
            preparation_shift=med['wide']/med['plain']-1,scratch_bytes=scratch,paired_disturbance=paired))
    interactions=[]
    for variant in ('plain','wide','stamped','dual'):
        metric_names=['elapsed_us']+(['L_later_cycle_per_kt','L_later_ns_per_kt','L_first_cycle_per_kt','L_first_ns_per_kt'] if variant=='dual' else [])
        for trial in range(10):
            members={(p['a_pitched'],p['b_pitched']):p for p in processes if p['variant']==variant and p['trial']==trial}
            if set(members)!={(False,False),(True,False),(False,True),(True,True)}:
                raise ValueError('incomplete factorial trial')
            for metric in metric_names:
                z,a,b,ab=[members[key][metric] for key in [(False,False),(True,False),(False,True),(True,True)]]
                interactions.append(dict(variant=variant,trial=trial,metric=metric,aligned=z,A_only=a,B_only=b,both=ab,
                    A_delta=a-z,B_delta=b-z,both_delta=ab-z,interaction=ab-a-b+z))
    grouped=defaultdict(list)
    for r in interactions:grouped[r['variant'],r['metric']].append(r)
    contrasts=[dict(variant=v,metric=m,**{key:dict(median=statistics.median(r[key] for r in members),
        minimum=min(r[key] for r in members),maximum=max(r[key] for r in members))
        for key in ('aligned','A_only','B_only','both','A_delta','B_delta','both_delta','interaction')})
        for (v,m),members in grouped.items()]
    allocations={(s['input_allocation_elements']['A'],s['input_allocation_elements']['B']) for s in summaries}
    report=dict(scope='Fixed M/N/K, D pitch, values and cache protocol; A/B physical pitch factorial. Direct ns is not reconstructed from whole-call frequency.',
        input_allocation_equal=len(allocations)==1,
        environment=json.loads((run/'environment.json').read_text()),cases=summaries,processes=processes,
        contrasts=contrasts,trial_contrasts=interactions,failed_attempts=failed,
        caveats=['Mainloop windows include pipeline wait/drain; their SM overlap is not TMA occupancy.',
                 'Same trial refers to adjacent randomized processes, not simultaneous calls.',
                 'Observer qualification and interval perturbation must be assessed before adopting parameters.'],
        analyzer_sha256=common.sha(Path(__file__)),cases_sha256=common.sha(run/'cases.json'))
    output.mkdir(parents=True,exist_ok=False)
    (output/'joint-pitch.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'tiles.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(tiles[0]));writer.writeheader();writer.writerows(tiles)
    print(json.dumps(dict(cases=summaries,contrasts=contrasts),indent=2))


B_PITCH_CASES = {offset: f'cfg_b_bcurve_p{offset}' for offset in (0, 16, 32, 64, 128)}
LOCAL_ERROR_LIMIT = .05


def extra_coverage(offset, line_bytes):
    """Extra address-covering lines for 64 rows of 256 B; not physical traffic."""
    return sum(((row * (6144 + offset)) % line_bytes + 256 + line_bytes - 1) // line_bytes
               - 256 // line_bytes for row in range(64))


def fit_coverage(values, line_bytes):
    """Exact constrained least squares for this three-point, two-parameter curve.

    C is anchored at p0; 0 <= q0 <= C, q1 >= 0. Enumerate the flat,
    one-active and two-active regions and retain every minimizing parameter set.
    Fractions avoid optimizer initialization and numerical tie-breaking.
    """
    C, Y, Z = (Fraction(str(values[k])) for k in (0, 16, 32))
    regions = []
    if line_bytes == 32:
        if Y <= C:
            regions.append(('flat', [(0, 0), (C, 0), (0, C / 32)]))
        else:
            regions.append(('p16_active', [(0, Y / 32), (C, (Y - C) / 32)]))
    elif line_bytes == 128:
        regions.append(('flat', [(0, 0), (C, 0), (0, C / 56)]))
        u = min(max(Y, C), 7 * C / 6)
        bound = 7 * C - 6 * u
        regions.append(('p16_active', [(0, u / 56), (bound, (u - bound) / 56)]))
        q0, q1 = 7 * Z - 6 * Y, (Y - Z) / 8
        if 0 <= q0 <= C and q1 >= 0 and q0 + 48 * q1 >= C:
            regions.append(('both_active_interior', [(q0, q1)]))
        denominator = 56**2 + 48**2
        regions.append(('q0_zero', [(0, max(C / 48, (56 * Y + 48 * Z) / denominator))]))
        regions.append(('q0_C', [(C, max(0, (56 * (Y - C) + 48 * (Z - C)) / denominator))]))
    else:
        raise ValueError('only the declared 32 B and 128 B candidates are supported')

    def predict(q, offset):
        return max(C, q[0] + q[1] * extra_coverage(offset, line_bytes))

    def loss(q):
        return sum((predict(q, k) - y)**2 for k, y in zip((0, 16, 32), (C, Y, Z)))

    best = min(loss(vertices[0]) for _, vertices in regions)
    winners = [(name, vertices) for name, vertices in regions if loss(vertices[0]) == best]
    vertices = {tuple(map(Fraction, q)) for _, group in winners for q in group}
    predictions = {}
    for offset in B_PITCH_CASES:
        possible = [predict(q, offset) for q in vertices]
        lo, hi = min(possible), max(possible)
        predictions[offset] = dict(extra_lines=extra_coverage(offset, line_bytes),
            unique=lo == hi, ns_per_kt=float(lo) if lo == hi else None,
            interval_ns_per_kt=[float(lo), float(hi)])
    errors = {k: max(abs(v / values[k] - 1) for v in predictions[k]['interval_ns_per_kt']) for k in (0, 16, 32)}
    return dict(line_bytes=line_bytes, C_ns_per_kt=float(C), loss_ns_squared=float(best),
        parameters_unique=len(vertices) == 1,
        q0_range=[float(min(q[0] for q in vertices)), float(max(q[0] for q in vertices))],
        q1_range=[float(min(q[1] for q in vertices)), float(max(q[1] for q in vertices))],
        minimizer_sets=[dict(region=name, vertices=[dict(q0=float(a), q1=float(b),
            exact_q0=str(a), exact_q1=str(b)) for a, b in group]) for name, group in winners],
        calibration_max_absolute_relative=max(errors.values()),
        calibration_errors=errors, calibration_status='pass' if max(errors.values()) <= LOCAL_ERROR_LIMIT else 'fail',
        predictions=predictions)


def pitch_cases(run):
    rows = json.loads((run / 'cases.json').read_text())
    by_id = {r['id']: r for r in rows}
    if len(rows) != 5 or set(by_id) != set(B_PITCH_CASES.values()):
        raise ValueError('the declared five B-pitch cases are required')
    fixed = dict(config='cfg_b', m=2304, n=3072, k=1024, lda=1024, ldd=3072,
                 alloc_lda=1024, alloc_ldb=3136, storage_m=2304, storage_n=3072,
                 input_mode='dyadic', seed=17, swizzle=1, sm_count=0, evict=0, zero_m=-1, zero_n=-1)
    for offset, name in B_PITCH_CASES.items():
        row = by_id[name]
        if any(row.get(k) != v for k, v in fixed.items()) or row['ldb'] != 3072 + offset // 2:
            raise ValueError('geometry, preparation or allocation changed: ' + name)
        if row['set'] != ('calib' if offset <= 32 else 'heldout'):
            raise ValueError('calibration/heldout membership changed: ' + name)
    return by_id


def pitch_observations(run, rows):
    """Read only the explicitly requested case sample directories."""
    import gzip
    from analyze_r18 import replay
    from analyze_r13_sm import select_attempts, validate_setup
    import v06_run as common
    setups = {s['case']: s['setup'] for s in json.loads((run / 'static_setup.json').read_text())}
    gpu = json.loads((run / 'environment.json').read_text())['gpu'].split(',')[0]
    observations, hashes = {}, {}
    for row in rows:
        selected, failed, ignored = select_attempts(run, row['id'], ('plain', 'wide', 'stamped', 'dual'))
        if ignored:
            raise ValueError('unexpected variants: ' + row['id'])
        times = {v: {} for v in ('plain', 'wide', 'stamped', 'dual')}
        trials = []
        for item in selected:
            rec = item['record']
            obs = replay(run, rec, row)
            with gzip.open(run / rec['raw'], 'rt') as stream:
                events = {v['event']: v for line in stream if line.strip() for v in [json.loads(line)]}
            setup = events['setup']
            if validate_setup(row, setups[row['id']], setup) != gpu:
                raise ValueError('sample GPU differs from the frozen run')
            if (setup['allocated_a_elements'], setup['allocated_b_elements']) != (2304 * 1024, 1024 * 3136):
                raise ValueError('input allocation is not matched')
            times[rec['variant']][rec['trial']] = rec['elapsed_us']
            hashes[item['path']] = common.sha(run / item['path'])
            hashes[rec['raw']] = rec['raw_sha256']
            if rec['variant'] == 'dual':
                if setup['trace_version'] != 'r18-dual-clock-events':
                    raise ValueError('dual direct timestamps required')
                later = [t[1] - t[0] for c in obs['ctas'] for t in c['tiles_ns'][1:]]
                if not later:
                    raise ValueError('no later-tile mainloop windows')
                trials.append(dict(trial=rec['trial'], host_start_ns=rec['host_start_ns'], later_windows=len(later),
                    ns_per_kt=statistics.fmean(later) / 16))
        if any(set(t) != set(range(10)) for t in times.values()):
            raise ValueError('ten successful trials per variant required: ' + row['id'])
        paired = [times['dual'][t] / times['wide'][t] - 1 for t in range(10)]
        observations[row['id']] = dict(target_ns_per_kt=statistics.median(t['ns_per_kt'] for t in trials),
            trials=trials, failed_attempts=failed,
            dual_over_wide=dict(median=statistics.median(times['dual'].values()) / statistics.median(times['wide'].values()) - 1,
                                paired=paired),
            elapsed_cv={v:statistics.pstdev(t.values()) / statistics.mean(t.values()) for v, t in times.items()})
    return observations, hashes


def pitch_hashes(run):
    import analyze_r18, analyze_r13_sm, v06_model, v06_run
    inputs = ('cases.json', 'static_setup.json', 'environment.json', 'run_config.json',
              'source_hashes.json', 'build/binary_hashes.json', 'build/sass_hashes.json')
    return dict(inputs={name:v06_run.sha(run / name) for name in inputs},
                code={Path(p).name:v06_run.sha(Path(p)) for p in
                      (__file__, analyze_r18.__file__, analyze_r13_sm.__file__, v06_model.__file__, v06_run.__file__)})


def freeze_b_pitch(run, output):
    import v06_run as common
    common.verify(run)
    rows = pitch_cases(run)
    if any((run / 'samples' / B_PITCH_CASES[k]).exists() for k in (64, 128)):
        raise ValueError('heldout samples must not exist before freezing')
    observations, sample_hashes = pitch_observations(run, [rows[B_PITCH_CASES[k]] for k in (0, 16, 32)])
    values = {k:observations[B_PITCH_CASES[k]]['target_ns_per_kt'] for k in (0, 16, 32)}
    models = {f'cover{width}':fit_coverage(values, width) for width in (32, 128)}
    predictions = {B_PITCH_CASES[k]:dict(offset_bytes=k,
        candidates={name:m['predictions'][k] for name, m in models.items()}) for k in (64, 128)}
    model_definition = dict(form='max(C, q0 + q1 * extra_lines)', constraints='0 <= q0 <= C; q1 >= 0',
        C='p0 calibrated target', box_rows=64, box_row_bytes=256, address_base_alignment_bytes=128,
        fit='equal-case squared error on p0/p16/p32; exact active-region enumeration; retain all minimizers',
        candidates=models)
    result = dict(status='frozen', frozen_unix_ns=time.time_ns(),
        gpu=json.loads((run / 'environment.json').read_text())['gpu'], predictions=predictions,
        target='dual: per-trial mean of all later tile (MAIN_END_ns-FIRST_MMA_ns)/16, then median of 10 trials',
        model=model_definition, calibration=observations, calibration_sample_hashes=sample_hashes,
        hashes=pitch_hashes(run),
        model_sha256=hashlib.sha256(json.dumps(json.loads(json.dumps(model_definition)), sort_keys=True).encode()).hexdigest(),
        scoring=dict(max_absolute_relative_per_case=LOCAL_ERROR_LIMIT, ids=list(predictions),
            interval_policy='worst endpoint error; never select a parameter after seeing heldout',
            candidate_policy='report both; a calibration failure remains failed even if heldout fits',
            scope='local dual mainloop ns/Ktile only; no GEMM pass claim'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    output.chmod(0o444)
    print('frozen', output, common.sha(output))


def score_b_pitch(run, frozen_path, output):
    import v06_run as common
    common.verify(run)
    frozen = json.loads(frozen_path.read_text())
    if frozen['status'] != 'frozen' or frozen_path.stat().st_mode & 0o222:
        raise ValueError('read-only frozen predictions required')
    if hashlib.sha256(json.dumps(frozen['model'], sort_keys=True).encode()).hexdigest() != frozen['model_sha256']:
        raise ValueError('frozen model identity changed')
    if pitch_hashes(run) != frozen['hashes']:
        raise ValueError('frozen input/code identities changed')
    rows = pitch_cases(run)
    observations, sample_hashes = pitch_observations(run, [rows[B_PITCH_CASES[k]] for k in (64, 128)])
    if any(t['host_start_ns'] <= frozen['frozen_unix_ns'] for o in observations.values() for t in o['trials']):
        raise ValueError('heldout measurement predates the frozen prediction')
    reports = {}
    for model, fit in frozen['model']['candidates'].items():
        scores = []
        for name in frozen['scoring']['ids']:
            observed = observations[name]['target_ns_per_kt']
            prediction = frozen['predictions'][name]['candidates'][model]
            errors = [x / observed - 1 for x in prediction['interval_ns_per_kt']]
            scores.append(dict(case=name, observed_ns_per_kt=observed, prediction=prediction,
                relative_error_interval=errors, worst_absolute_relative=max(map(abs, errors))))
        heldout_pass = all(s['worst_absolute_relative'] <= frozen['scoring']['max_absolute_relative_per_case'] for s in scores)
        reports[model] = dict(calibration_status=fit['calibration_status'], scores=scores,
            heldout_pass=heldout_pass, local_target_pass=heldout_pass and fit['calibration_status'] == 'pass')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'b-pitch-score.json').write_text(json.dumps(dict(frozen_sha256=common.sha(frozen_path),
        candidates=reports, observations=observations, sample_hashes=sample_hashes,
        scope=frozen['scoring']['scope']), indent=2) + '\n')
    print(json.dumps(reports, indent=2))


def raster_pitch_diagnostics(root, output):
    """A fixed-geometry order/pitch interaction; no timing parameter fit."""
    from analyze_r13_sm import select_attempts
    from analyze_r13_supply import _failure_pair_windows, _failure_pair_summary
    from analyze_r18 import replay
    from r09_input_source_clock import control_window
    import v08_model
    import v06_run as common
    import supply_model

    root,output=Path(root),Path(output)
    common.verify(root)
    rows={r['id']:r for r in json.loads((root/'cases.json').read_text())}
    setups={r['case']:r['setup'] for r in json.loads((root/'static_setup.json').read_text())}
    controls=[r for r in rows.values() if r['set']=='ctrl']
    fixed=('config','m','n','k','lda','ldd','storage_m','storage_n','zero_m','zero_n',
           'alloc_lda','alloc_ldb','sm_count','swizzle','input_mode','seed','evict','warmup_policy')
    if len(controls)!=4 or any(any(r.get(k)!=controls[0].get(k) for k in fixed) for r in controls):
        raise ValueError('raster/pitch contrast changed another declared condition')
    def span(values):
        values=list(values)
        return dict(median=statistics.median(values),trial_min=min(values),trial_max=max(values))
    calls={};plain={};processes=[];hashes={};failed={};contexts={}
    for row in controls:
        name=row['id'];setup=setups[name]
        selected,failures,_=select_attempts(root,name,('plain','dual'))
        failed[name]=[item['path'] for item in failures]
        calls[name]={};plain[name]={}
        work=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row['raster_order'])
        contexts[name]=dict(row=row,setup=setup,work_sha256=hashlib.sha256(json.dumps(work).encode()).hexdigest(),
            waves=[dict(j=j,ctas=sum(len(c)>j for c in work),
                unique_A_panels=len({c[j][0] for c in work if len(c)>j}),
                unique_B_panels=len({c[j][1] for c in work if len(c)>j})) for j in range(max(map(len,work)))])
        for item in selected:
            rec=item['record'];obs=replay(root,rec,row)
            for path in (root/item['path'],root/rec['raw']):
                hashes[str(path)]=common.sha(path)
            if rec['variant']=='plain':plain[name][rec['trial']]=obs['elapsed_us']
            else:
                if [c['work'] for c in obs['ctas']]!=work:
                    raise ValueError('explicit raster mapping differs from traced coordinates')
                calls[name][rec['trial']]=dict(row=row,trial=rec['trial'],record=rec,observed=obs)
                processes.append(dict(case=name,trial=rec['trial'],variant='dual',event_us=obs['elapsed_us'],
                    **control_window(obs)))
        if len(plain[name])!=10 or plain[name].keys()!=calls[name].keys():
            raise ValueError('ten matched plain/dual calls required')
    comparisons=[];positions=[];paired=[];summaries={}
    for order in ('along_m','along_n'):
        a,b=(f'r10_raster_{order}_p{x}' for x in (0,32))
        summary,windows=_failure_pair_summary('R10-raster',list(calls[a].values()),list(calls[b].values()))
        comparisons.append(summary);positions.extend(windows)
        for trial in sorted(calls[a]):
            aw,bw=calls[a][trial]['observed'],calls[b][trial]['observed']
            current=_failure_pair_windows(aw['ctas'],bw['ctas'])
            for phase in ('all','first','later'):
                subset=[x for x in current if phase=='all' or (x['j']==0)==(phase=='first')]
                paired.append(dict(raster=order,trial=trial,phase=phase,windows=len(subset),
                    **{field:statistics.median(x[field] for x in subset) for field in
                       ('L_ns_delta','L_ns_ratio','L_cycles_delta','L_cycles_ratio')}))
        summaries[order]=dict(plain_pitch_change=span(plain[b][t]/plain[a][t]-1 for t in plain[a]),
            frequency_pitch_change=span(control_window(calls[b][t]['observed'])['frequency_ghz']/
                control_window(calls[a][t]['observed'])['frequency_ghz']-1 for t in plain[a]),
            plain_us={r:span(plain[r].values()) for r in (a,b)},
            dual_disturbance={r:span(calls[r][t]['observed']['elapsed_us']/plain[r][t]-1 for t in plain[r]) for r in (a,b)},
            first=summary['phases']['first'],later=summary['phases']['later'])
    interaction=[]
    for phase in ('all','first','later'):
        groups={(r['raster'],r['trial']):r for r in paired if r['phase']==phase}
        for trial in range(10):
            a,b=groups['along_m',trial],groups['along_n',trial]
            am,bm=(f'r10_raster_along_m_p{x}' for x in (0,32))
            an,bn=(f'r10_raster_along_n_p{x}' for x in (0,32))
            interaction.append(dict(phase=phase,trial=trial,
                plain_difference_of_pitch_differences_us=(plain[bn][trial]-plain[an][trial])-(plain[bm][trial]-plain[am][trial]),
                **{key:b[key]-a[key] for key in ('L_ns_delta','L_ns_ratio','L_cycles_delta','L_cycles_ratio')}))
    interaction_summary={phase:{key:span(r[key] for r in interaction if r['phase']==phase)
        for key in interaction[0] if key not in ('phase','trial')} for phase in ('all','first','later')}
    blindness={}
    for pitch in (0,32):
        names=[f'r10_raster_{order}_p{pitch}' for order in ('along_m','along_n')]
        features=[]
        for name in names:
            requests=supply_model.supply_request_rows(rows[name],setups[name])
            features.append([{k:r[k] for k in ('cta','j','T','kt','X','compute_cycles')} for r in requests])
        blindness[str(pitch)]=dict(current_supply_inputs_identical=features[0]==features[1],
            windows=len(features[0]),feature_sha256=[hashlib.sha256(json.dumps(f,sort_keys=True).encode()).hexdigest() for f in features],
            interpretation='Coordinate sequences differ, but the current seven-column service input and per-CTA counts are identical. '
                'Refitting these same fields cannot distinguish the observed order response at this pitch.')
    references={};previous=root.parent/'20261010-R13-stage-events'
    for row in rows.values():
        if row['set']!='reference':continue
        series={}
        for label,run in [('current',root),('previous_same_card',previous)]:
            selected,failures,_=select_attempts(run,row['id'],('plain',));values=[]
            if json.loads((run/'environment.json').read_text())['gpu']!=json.loads((root/'environment.json').read_text())['gpu']:
                raise ValueError('reference card changed')
            for item in selected:
                values.append(replay(run,item['record'],row)['elapsed_us'])
                for path in (run/item['path'],run/item['record']['raw']):hashes[str(path)]=common.sha(path)
            series[label]=span(values)
        references[row['id']]=dict(**series,median_change=series['current']['median']/series['previous_same_card']['median']-1)
    for name in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/resources.json'):
        hashes[str(root/name)]=common.sha(root/name)
    report=dict(scope='Fixed shape/K/capacity; within-raster pitch pairs precede cross-raster response differences. '
        'Work ordering and grid orientation change together; no physical cache/traffic attribution.',
        environment=json.loads((root/'environment.json').read_text()),contexts=contexts,comparisons=comparisons,
        summaries=summaries,processes=processes,paired_phase_trials=paired,references=references,current_model_blindness=blindness,
        interaction=interaction,interaction_summary=interaction_summary,failed_attempts=failed,
        input_sha256=hashes,analyzer_sha256=common.sha(Path(__file__)),
        limits=['First/later coordinate membership changes across raster orders; all phase includes all 432 output tiles.',
                'Trial numbers pair independent processes, not simultaneous calls; uncertainty units are processes.',
                'Within-raster windows match CTA, ordinal, total work and coordinates exactly.',
                'Software panel unions are not measured memory traffic or concurrent TMA requests.',
                'The original old-card B32 result must not be merged into a new-card parameter fit.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'R10-raster-pitch.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'R10-raster-pitch-positions.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(positions[0]));writer.writeheader();writer.writerows(positions)
    return report


def mainloop_overlap_features(windows):
    """Same-call L-window overlap, counting each other CTA at most once.

    These observed intervals diagnose a software-ordinal grouping; they are not
    TMA issue windows and must never be formal prediction inputs.
    """
    from collections import defaultdict

    by_cta=defaultdict(list)
    for window in windows:
        if window['end']<=window['begin']:
            raise ValueError('positive same-endpoint mainloop window required')
        by_cta[window['cta']].append(window)

    def union_length(intervals):
        total=0;last=None
        for lo,hi in sorted(intervals):
            if last is None or lo>last:
                total+=hi-lo;last=hi
            elif hi>last:
                total+=hi-last;last=hi
        return total

    result=[]
    for target in windows:
        totals=dict(other_cta=0,same_A=0,same_B=0,other_ordinal=0)
        for cta,peers in by_cta.items():
            if cta==target['cta']:continue
            intervals={key:[] for key in totals}
            for peer in peers:
                lo=max(target['begin'],peer['begin']);hi=min(target['end'],peer['end'])
                if hi<=lo:continue
                intervals['other_cta'].append((lo,hi))
                if peer['mi']==target['mi']:intervals['same_A'].append((lo,hi))
                if peer['ni']==target['ni']:intervals['same_B'].append((lo,hi))
                if peer['j']!=target['j']:intervals['other_ordinal'].append((lo,hi))
            for key in totals:totals[key]+=union_length(intervals[key])
        duration=target['end']-target['begin']
        result.append({key:value/duration for key,value in totals.items()})
    return result


def source_panel_features(work):
    """Logical source sharing by software ordinal, without observed timestamps."""
    from collections import Counter
    previous=[set(),set()];seen=[set(),set()];groups=[];windows=[]
    for j in range(max(map(len,work))):
        active=[(i,c[j]) for i,c in enumerate(work) if len(c)>j]
        counts=[Counter(c[o] for _,c in active) for o in (0,1)]
        current=[]
        for cta,coord in active:
            item=dict(cta=cta,j=j,T=len(work[cta]),mi=coord[0],ni=coord[1],active_ctas=len(active))
            for o,op in enumerate(('A','B')):
                item[op+'_peer']=counts[o][coord[o]]-1
                item[op+'_seen_previous_wave']=int(coord[o] in previous[o])
                item[op+'_seen_earlier_wave']=int(coord[o] in seen[o])
            item['active_times_B_peer']=len(active)*item['B_peer']
            current.append(item);windows.append(item)
        for T in sorted({x['T'] for x in current}):
            subset=[x for x in current if x['T']==T]
            summary=dict(j=j,T=T,windows=len(subset),active_ctas=len(active),
                unique_A_panels=len(counts[0]),unique_B_panels=len(counts[1]),
                unique_source_KiB_per_ktile=16*(len(counts[0])+len(counts[1])),
                unique_B_KiB_per_ktile=16*len(counts[1]),
                wave_B_ordered_request_pairs=sum(n*(n-1) for n in counts[1].values()))
            for field in ('A_peer','B_peer','active_times_B_peer'):
                values=[x[field] for x in subset]
                summary[field]=dict(median=statistics.median(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values))
            for field in ('A_seen_previous_wave','B_seen_previous_wave','A_seen_earlier_wave','B_seen_earlier_wave'):
                summary[field+'_fraction']=statistics.mean(x[field] for x in subset)
            groups.append(summary)
        for o in (0,1):previous[o]=set(counts[o]);seen[o]|=previous[o]
    return dict(groups=groups,windows=windows)


def source_organization_diagnostics(root, output):
    """Test static source-count candidates using existing pitch pairs; no fit."""
    from collections import defaultdict
    from analyze_r13_sm import select_attempts
    from analyze_r13_supply import _failure_pair_windows
    from analyze_r18 import replay
    import v08_model

    root,output=Path(root),Path(output)
    def read(path):return json.loads(path.read_text())
    def span(values):
        values=list(values)
        return dict(median=statistics.median(values),trial_min=min(values),trial_max=max(values))
    raster_path=output/'R10-raster-pitch.json'
    raster=read(raster_path)
    csv_path=output/'R10-raster-pitch-positions.csv'
    positions=list(csv.DictReader(csv_path.open()))
    cohorts={};observed={};overlap_trials=[];hashes={};static={}
    for name,context in raster['contexts'].items():
        row,setup=context['row'],context['setup']
        work=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row['raster_order'])
        static[name]=source_panel_features(work)
        features={(x['cta'],x['j']):x for x in static[name]['windows']}
        observed[name]={}
        selected,_,_=select_attempts(root,name,('dual',))
        for item in selected:
            record=item['record'];obs=replay(root,record,row);observed[name][record['trial']]=obs
            windows=[dict(cta=c['cta'],j=j,mi=coord[0],ni=coord[1],begin=t[0],end=t[1])
                for c in obs['ctas'] for j,(coord,t) in enumerate(zip(c['work'],c['tiles_ns']))]
            overlap=mainloop_overlap_features(windows)
            for j,T in sorted({(x['j'],len(work[x['cta']])) for x in windows}):
                selected_overlap=[z for x,z in zip(windows,overlap) if x['j']==j and len(work[x['cta']])==T]
                overlap_trials.append(dict(case=name,trial=record['trial'],j=j,T=T,windows=len(selected_overlap),
                    **{key:statistics.median(v[key] for v in selected_overlap) for key in ('other_cta','same_A','same_B','other_ordinal')}))
            for path in (root/item['path'],root/record['raw']):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if len(observed[name])!=10:raise ValueError('source organization needs ten dual processes')
        groups=[]
        for group in static[name]['groups']:
            subset=[r for r in positions if r['variant']==name and int(r['j'])==group['j'] and int(r['T'])==group['T']]
            trials=[r for r in overlap_trials if r['case']==name and r['j']==group['j'] and r['T']==group['T']]
            groups.append(dict(**group,pitch_response={k:span(float(r[k]) for r in subset) for k in
                ('L_ns_delta','L_cycles_delta','L_ns_ratio','L_cycles_ratio')} if subset else None,
                observed_overlap={k:span(r[k] for r in trials) for k in ('other_cta','same_A','same_B','other_ordinal')}))
        cohorts[name]=dict(row=row,gpu=setup['gpu_uuid'],groups=groups)

    # Exact static feature strata, not a comparison of median input vectors.
    m0,m32='r10_raster_along_m_p0','r10_raster_along_m_p32'
    feature_map={(x['cta'],x['j']):x for x in static[m32]['windows']}
    matched_trials=[];matched_strata=[]
    for trial in sorted(observed[m0]):
        paired=_failure_pair_windows(observed[m0][trial]['ctas'],observed[m32][trial]['ctas'])
        buckets=defaultdict(lambda:defaultdict(list))
        for r in paired:
            f=feature_map[r['cta'],r['j']]
            if r['j']==0 and f['B_peer']==17:
                key=(r['mi'],f['A_peer'],f['B_peer'],f['A_seen_earlier_wave'],f['B_seen_earlier_wave'])
                buckets[key][r['T']].append(r)
        differences=[]
        for key,values in sorted(buckets.items()):
            if set(values)!={3,4}:continue
            a,b=values[3],values[4]
            record=dict(trial=trial,mi=key[0],A_peer=key[1],B_peer=key[2],A_earlier=key[3],B_earlier=key[4],
                T3_windows=len(a),T4_windows=len(b))
            for field in ('L_cycles_delta','L_ns_delta'):
                record[field+'_T3']=statistics.median(x[field] for x in a)
                record[field+'_T4']=statistics.median(x[field] for x in b)
                record[field+'_T4_minus_T3']=record[field+'_T4']-record[field+'_T3']
            matched_strata.append(record);differences.append(record)
        matched_trials.append(dict(trial=trial,matched_M_strata=len(differences),
            **{field:statistics.median(x[field] for x in differences) for field in
                ('L_cycles_delta_T3','L_cycles_delta_T4','L_cycles_delta_T4_minus_T3',
                 'L_ns_delta_T3','L_ns_delta_T4','L_ns_delta_T4_minus_T3')}))
    matched=dict(strata=matched_strata,processes=matched_trials,
        summary={k:span(r[k] for r in matched_trials) for k in matched_trials[0] if k not in ('trial','matched_M_strata')},
        scope='Match mi, A/B peer counts, ordinal, active count and prior logical-wave membership exactly. '
            'B panel identities/addresses and future work differ. This tests count-only source rules, not arbitrary address/history rules.')

    old_base=root.parent/'20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1'
    old_path=old_base/'B-summary.json';old=read(old_path)
    old_csv=old_base/'B-paired-windows.csv';old_positions=list(csv.DictReader(old_csv.open()))
    historical=[]
    for folder,names in [
        ('20261009-R10-b-coverage-job738203',('cfg_b_bcurve_p32','cfg_b_bcurve_p64')),
        ('20261009-V09-heldout-job739011',('v09d_cfg_b_G5_B32','v09d_cfg_b_G5_B80','v09d_cfg_b_G5_B112'))]:
        run=root.parent/folder;rows={r['id']:r for r in read(run/'cases.json')}
        setups={r['case']:r['setup'] for r in read(run/'static_setup.json')}
        for name in names:
            row,setup=rows[name],setups[name]
            work=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row.get('raster_order','heuristic'))
            groups=[]
            for group in source_panel_features(work)['groups']:
                subset=[r for r in old_positions if r['variant']==name and int(r['j'])==group['j'] and int(r['T'])==group['T']]
                groups.append(dict(**group,pitch_response={k:span(float(r[k]) for r in subset) for k in
                    ('L_ns_delta','L_cycles_delta','L_ns_ratio','L_cycles_ratio')}))
            historical.append(dict(case=name,source=str(run),row=row,gpu=setup['gpu_uuid'],groups=groups))
        for path in (run/'cases.json',run/'static_setup.json'):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (raster_path,csv_path,old_path,old_csv):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    def group(case,j,T):return next(g for g in cohorts[case]['groups'] if g['j']==j and g['T']==T)
    first,last=group(m32,0,4),group(m32,3,4)
    nfirst=group('r10_raster_along_n_p32',0,4)
    report=dict(scope='Post-measurement structural tests, no new service fit or validation score. Old and new cards stay separate.',
        definitions=dict(panel='A panel identity=mi; B panel identity=ni at a fixed Ktile. '
                'Each full cfg_b panel is 16 KiB per Ktile; these are logical sources, not physical traffic.',
            active='C_j = number of CTAs with output ordinal j.',
            fanout='n_B(j,p) = count of current-ordinal requests for B panel p; peer=n_B-1.',
            active_sharing='P_i=C_j*(n_B(j,p_i)-1). Also report sum_p n_p*(n_p-1).',
            revisit='Membership in earlier software ordinals of the same invocation, not cache residency or coldness. '
                'Repeated warmup means a panel absent from earlier ordinals may already have been read in another invocation.',
            overlap='Duration-weighted count of other CTA L intervals. Union all qualifying windows of each peer CTA first. '
                'L overlap is not TMA issue overlap; observed timestamps are diagnostic only.'),
        current=cohorts,historical=historical,observed_overlap_processes=overlap_trials,matched_first_groups=matched,
        rejected_candidates=[
            dict(name='Unique B source bytes alone',reason='AlongM first has 8 B panels; AlongN has 24. '
                'A nondecreasing cost of only unique B bytes predicts the opposite pitch-response order.',
                examples=[first,nfirst]),
            dict(name='B fanout plus previous-wave membership alone',reason='AlongM first T4 and last T4 have peer=17 '
                'and no earlier-wave B occurrence, but pitch responses differ.',examples=[first,last]),
            dict(name='Monotone active-times-sharing response',reason='Median P is 612 for AlongM last T4, 660 for AlongN first T4. '
                'The measured penalty order is opposite. A nondecreasing single-feature response, including a nonnegative hinge, fails.',
                examples=[last,nfirst]),
            dict(name='Count-only two-resource service',formula='Delta=max(512,a*C+b*B_extra128*peer_B)-max(512,a*C), a,b>=0.',
                reason='Matched first-output strata share the exact count inputs, but future work and B identities differ. '
                    'The measured T4-minus-T3 response is reported above; do not infer impossibility from equal median inputs alone.')],
        retained_boundary='Keep active demand, operand-specific panel multiplicities, source identity and predecessor/successor work '
            'as separate computable inputs. Current evidence does not identify a transferable scalar service price; '
            'do not add direction coefficients or call overlap a predictor.',
        next_discriminator='Single-output 36/126 CTA cohorts keep the first two B panels and 18 users each while changing other '
            'source groups. They remove producer-next-output mixing present in grid36/132 persistent work. '
            'If both single cohorts lack a penalty while persistent132 has one, a narrow 126-to132 demand threshold remains possible.',
        input_sha256=hashes,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    (output/'R10-source-organization.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 源分组能解释方向，计数还不能成为供给规则','',
        '固定几何的方向对照先否定简单计数假设；旧R10与V09只检查趋势和静态归属，不混合跨卡定值。','',
        '| 分组（B32） | 活动CTA | 唯一B面板 | B peer中位 | C×peer中位 | 此前轮见过B的比例 | ΔL周期中位 / 十次范围 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for order in ('along_m','along_n'):
        for g in cohorts[f'r10_raster_{order}_p32']['groups']:
            d=g['pitch_response']['L_cycles_delta']
            lines.append(f'| {order}, j{g["j"]}/T{g["T"]} | {g["active_ctas"]} | {g["unique_B_panels"]} | '
                f'{g["B_peer"]["median"]:.1f} | {g["active_times_B_peer"]["median"]:.1f} | '
                f'{g["B_seen_earlier_wave_fraction"]:.2f} | {d["median"]:.1f} / {d["trial_min"]:.1f}～{d["trial_max"]:.1f} |')
    d=matched['summary']['L_cycles_delta_T4_minus_T3']
    lines+=['','唯一B源字节的方向相反；fan-out相同又不能解释首末轮响应；active×fan-out单变量仍把AlongM末轮和AlongN首轮的顺序预测反。'
        '这些反例不否定任意多变量访问模型，但否定直接给上述单变量配一个正价格或阈值。',
        '',f'首轮T3/T4进一步按相同mi、A/B peer数、历史成员标记逐窗口分层，'
        f'T4−T3的B32周期差中位为{d["median"]:.1f}，十进程范围{d["trial_min"]:.1f}～{d["trial_max"]:.1f}。'
        '两个集合仍使用不同B面板地址和后续工作；这限制的是仅依赖计数的服务形式，不能推广为任意源身份/历史函数的矛盾。',
        '','| 同调用实际L重叠（B32） | 其它CTA中位 | 同B peer中位 | 跨ordinal peer中位 |','|---|---:|---:|---:|']
    for name,j,T in [(m32,0,4),(m32,3,4),('r10_raster_along_n_p32',0,4)]:
        g=group(name,j,T);o=g['observed_overlap']
        lines.append(f'| {name}, j{j}/T{T} | {o["other_cta"]["median"]:.2f} | {o["same_B"]["median"]:.2f} | {o["other_ordinal"]["median"]:.2f} |')
    lines+=['','每个peer CTA先把重叠L窗口取并集，再计时长，避免pingpong输出重叠重复计数。'
        '同B使用者集中程度的方向仍存在；这不是TMA发起并发度，不能将实测时序带入预测。',
        '','最短下一步保留当前源分组向量，做相同前两B面板、各18个使用者的single-output 36/126 CTA配对。'
        '它能检验其它源组是否改变局部面板响应，避免persistent grid扫描的下一输出预取混杂。'
        '若两者都无罚而原persistent132有罚，仍保留126–132之间窄阈值的可能，不直接归因缓存或硬件队列。',
        '','目前没有拟合方向常数、自由非线性函数或新服务价格。JSON保留全部(j,T)静态量、逐进程重叠和精确匹配分层。','']
    (output/'R10-source-organization.md').write_text('\n'.join(lines))
    return report


def source_pressure_diagnostics(root, output):
    """Matched first B panels: one output at 36/126 CTAs, persistent at 132."""
    from analyze_r13_sm import select_attempts
    from analyze_r13_supply import _failure_pair_windows
    from analyze_r18 import replay
    import v08_model
    import v06_run as common

    root,output=Path(root),Path(output)
    common.verify(root)
    rows={r['id']:r for r in json.loads((root/'cases.json').read_text())}
    setups={r['case']:r['setup'] for r in json.loads((root/'static_setup.json').read_text())}
    def span(v):
        v=list(v)
        return dict(median=statistics.median(v),trial_min=min(v),trial_max=max(v))
    controls=[r for r in rows.values() if r['set']=='ctrl']
    fixed=('config','m','k','lda','ldd','storage_m','storage_n','zero_m','zero_n','swizzle','evict',
           'input_mode','seed','alloc_lda','alloc_ldb','input_map_m','input_map_n','raster_order','warmup_policy')
    if len(controls)!=6 or any(r.get(k)!=controls[0].get(k) for r in controls for k in fixed):
        raise ValueError('source pressure contrast changed an undeclared condition')
    calls={};plain={};processes=[];hashes={};failures={};contexts={}
    expected=[[m,n] for n in range(2) for m in range(18)]
    for row in rows.values():
        name=row['id'];setup=setups[name]
        selected,failed,_=select_attempts(root,name,tuple(row['variants']))
        failures[name]=[x['path'] for x in failed];calls[name]={};plain[name]={}
        work=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row.get('raster_order','heuristic'))
        if row['set']=='ctrl':
            if [list(w[0]) for w in work[:36]]!=expected:
                raise ValueError('selected first-output coordinates changed')
            if row['sm_count'] in (36,126) and any(len(w)!=1 for w in work):
                raise ValueError('single-output group has future persistent work')
            if len(work)!=row['sm_count'] or setup['initialized_input_columns']!=3072:
                raise ValueError('actual grid or initialized source region changed')
            contexts[name]=dict(row=row,setup=setup,cta_count=len(work),tile_counts=sorted(set(map(len,work))),
                selected_first_coordinates=expected,first_B_users={str(n):sum(w[0][1]==n for w in work) for n in (0,1)})
        for item in selected:
            rec=item['record'];obs=replay(root,rec,row)
            for path in (root/item['path'],root/rec['raw']):hashes[str(path)]=common.sha(path)
            if rec['variant']=='plain':plain[name][rec['trial']]=obs['elapsed_us'];continue
            if [c['work'] for c in obs['ctas']]!=work:raise ValueError('actual work differs from static scheduling')
            calls[name][rec['trial']]=obs
            wins=[dict(cta=c['cta'],j=j,mi=p[0],ni=p[1],begin=t[0],end=t[1])
                  for c in obs['ctas'] for j,(p,t) in enumerate(zip(c['work'],c['tiles_ns']))]
            overlaps=mainloop_overlap_features(wins)
            chosen=[(w,o) for w,o in zip(wins,overlaps) if w['cta']<36 and w['j']==0]
            processes.append(dict(case=name,trial=rec['trial'],event_us=obs['elapsed_us'],source=item['path'],
                observed_sms=len({c['sm'] for c in obs['ctas']}),
                selected_first_L_overlap={key:statistics.median(o[key] for w,o in chosen) for key in overlaps[0]}))
        if len(plain[name])!=10 or (row['set']=='ctrl' and calls[name].keys()!=plain[name].keys()):
            raise ValueError('ten successful processes for each declared variant required')
    paired=[];summaries={}
    for label in ('single36','single126','persistent132'):
        a,b=(f'r10_source_{label}_p{x}' for x in (0,32))
        for trial in sorted(calls[a]):
            windows=_failure_pair_windows(calls[a][trial]['ctas'],calls[b][trial]['ctas'])
            subset=[w for w in windows if w['cta']<36 and w['j']==0]
            if len(subset)!=36:raise ValueError('missing matched first B-panel users')
            fields=('L_ns_base','L_ns_variant','L_ns_delta','L_ns_ratio','L_cycles_base','L_cycles_variant',
                    'L_cycles_delta','L_cycles_ratio','L_rate_ratio','prefix_ns_delta','prefill_ns_delta','E_ns_delta')
            paired.append(dict(group=label,trial=trial,windows=36,
                plain_base_us=plain[a][trial],plain_variant_us=plain[b][trial],
                plain_pitch_ratio=plain[b][trial]/plain[a][trial]-1,
                **{key:statistics.median(w[key] for w in subset) for key in fields}))
        summaries[label]={key:span(r[key] for r in paired if r['group']==label)
                          for key in paired[-1] if key not in ('group','trial','windows')}
        summaries[label]['dual_disturbance']={name:span(calls[name][t]['elapsed_us']/plain[name][t]-1 for t in plain[name]) for name in (a,b)}
    interaction=[]
    for trial in range(10):
        groups={r['group']:r for r in paired if r['trial']==trial}
        for a,b in [('single36','single126'),('single126','persistent132')]:
            interaction.append(dict(comparison=b+' minus '+a,trial=trial,
                **{key:groups[b][key]-groups[a][key] for key in ('L_ns_delta','L_cycles_delta','L_ns_ratio','L_cycles_ratio')}))
    differences={label:{key:span(r[key] for r in interaction if r['comparison']==label) for key in interaction[0] if key not in ('comparison','trial')}
                 for label in sorted({r['comparison'] for r in interaction})}
    references={}
    previous=root.parent/'20261010-R10-raster-pitch'
    if json.loads((previous/'environment.json').read_text())['gpu']!=json.loads((root/'environment.json').read_text())['gpu']:
        raise ValueError('same-card reference identity changed')
    for row in rows.values():
        if row['set']!='reference':continue
        name=row['id'];selected,_,_=select_attempts(previous,name,('plain',))
        old=[replay(previous,r['record'],row)['elapsed_us'] for r in selected]
        for item in selected:
            for p in (previous/item['path'],previous/item['record']['raw']):hashes[str(p)]=common.sha(p)
        references[name]=dict(current=span(plain[name].values()),previous=span(old),median_ratio=statistics.median(plain[name].values())/statistics.median(old)-1)
    for name in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/resources.json'):
        hashes[str(root/name)]=common.sha(root/name)
    report=dict(scope='Three within-context pitch pairs (six conditions), developed after raster counterexample; no frozen validation score.',
        environment=json.loads((root/'environment.json').read_text()),contexts=contexts,summaries=summaries,
        paired_trials=paired,interaction=interaction,interaction_summary=differences,processes=processes,
        references=references,failed_attempts=failures,input_sha256=hashes,analyzer_sha256=common.sha(Path(__file__)),
        limits=['Each reported difference first pairs B0/B32 inside a fixed logical shape and execution organization.',
                'Across single-output groups, the selected 36 CTAs use the same current coordinates, source maps and 18 users per B panel.',
                'The groups change A-source multiplicity, other B panels, output writers, invocation work and frequency; no unique physical attribution.',
                '126-to-132 pressure threshold is not separated from persistent organization by these conditions.',
                'L-window overlap counts each other CTA at most once; it is not instantaneous TMA request concurrency.',
                'Cycle/ns is diagnostic only and is not admitted as a formal prediction input.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'R10-source-pressure.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--set', choices=('representatives', 'formal'), default='formal')
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--joint-pitch', action='store_true')
    parser.add_argument('--freeze-b-pitch', action='store_true')
    parser.add_argument('--score-b-pitch', action='store_true')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
    elif args.freeze_b_pitch:
        if args.input is None or args.output is None:parser.error('--freeze-b-pitch requires --input RUN --output FILE')
        freeze_b_pitch(args.input.resolve(), args.output.resolve())
    elif args.score_b_pitch:
        if args.input is None or args.output is None or args.predictions is None:
            parser.error('--score-b-pitch requires --input RUN --predictions FILE --output NEW_DIRECTORY')
        score_b_pitch(args.input.resolve(), args.predictions.resolve(), args.output.resolve())
    elif args.joint_pitch:
        if args.output is None:parser.error('--joint-pitch requires --output')
        joint_pitch(args.input.resolve(),args.output.resolve())
    else:
        analyze(args.input, args.set)
