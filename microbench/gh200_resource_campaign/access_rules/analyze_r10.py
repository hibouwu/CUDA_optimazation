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
