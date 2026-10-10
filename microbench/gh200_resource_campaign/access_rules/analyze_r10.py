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
from supply_model import source_sharing_delta_cycles as _source_service_value


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
        exact_response_counterexamples=[
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
        error_scale='Exact-response mismatches alone do not establish failure at the intended whole-L error target. '
            'The separately reported shared-service candidate checks full-window-normalized residuals.',
        retained_boundary='Keep active demand, operand-specific panel multiplicities, source identity and predecessor/successor work '
            'as separate computable inputs. Current evidence does not identify a transferable scalar service price; '
            'do not add direction coefficients or call overlap a predictor.',
        next_discriminator='Single-output 36/126 CTA cohorts keep the first two B panels and 18 users each while changing other '
            'source groups. They remove producer-next-output mixing present in grid36/132 persistent work. '
            'If both single cohorts lack a penalty while persistent132 has one, a narrow 126-to132 demand threshold remains possible.',
        input_sha256=hashes,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    (output/'R10-source-organization.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 源分组与计数候选的精确响应缺项','',
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
        '这些是精确响应的反例，不代表整窗误差一定超过5%/10%目标；共享两参数候选的容许区间另见R10-shared-service-candidate.md。',
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

def _source_polygon_vertices(constraints):
    """Exact vertices of the bounded two-parameter diagnostic region."""
    points=set()
    for i,(p,q,r) in enumerate(constraints):
        for u,v,w in constraints[i+1:]:
            det=p*v-q*u
            if not det:continue
            a,b=(r*v-q*w)/det,(p*w-r*u)/det
            if all(x*a+y*b<=z for x,y,z in constraints):points.add((a,b))
    return sorted(points)


def _source_service_branches(C,X):
    # Constraints are p*a+q*b<=r; the final tuple is delta/K's affine form.
    z=Fraction(0);floor=Fraction(512)
    return [([(C,X,floor)],(z,z,z)),
        ([(C,z,floor),(-C,-X,-floor)],(C,X,-floor)),
        ([(-C,z,-floor)],(z,X,z))]


def _source_service_clip(regions,C,Xs,kt,lower,upper):
    """Constrain the median response (one/two middle Xs) without choosing a fit."""
    from itertools import product
    C,kt,lower,upper=map(lambda x:Fraction(str(x)),(C,kt,lower,upper))
    Xs=[Fraction(str(x)) for x in Xs]
    output=[];seen=set()
    for region in regions:
        for branches in product(*[_source_service_branches(C,X) for X in Xs]):
            extra=[c for conditions,_ in branches for c in conditions]
            affine=[sum(b[1][i] for b in branches)*kt/len(Xs) for i in range(3)]
            p,q,r=affine
            conditions=region+extra+[(p,q,upper-r),(-p,-q,r-lower)]
            vertices=_source_polygon_vertices(conditions)
            key=tuple(vertices)
            if vertices and key not in seen:seen.add(key);output.append(conditions)
    return output


def _source_service_interval(regions,C,Xs,kt):
    """Include target max-branch crossings, not only training polygon corners."""
    from itertools import product
    C,kt=Fraction(str(C)),Fraction(str(kt));Xs=[Fraction(str(x)) for x in Xs]
    values=[]
    for region in regions:
        for branches in product(*[_source_service_branches(C,X) for X in Xs]):
            vertices=_source_polygon_vertices(region+[c for conditions,_ in branches for c in conditions])
            values.extend(sum(_source_service_value(a,b,C,X,kt) for X in Xs)/len(Xs) for a,b in vertices)
    if not values:raise ValueError('empty service parameter domain')
    return [float(min(values)),float(max(values))]


def shared_source_service_diagnostics(root, output):
    """Two single-output pairs train a shared response; older raster data checks it."""
    from collections import defaultdict
    import supply_model
    import v08_model

    root,output=Path(root),Path(output)
    pressure_path=output/'R10-source-pressure.json';raster_path=output/'R10-raster-pitch.json'
    position_path=output/'R10-raster-pitch-positions.csv'
    pressure=json.loads(pressure_path.read_text());raster=json.loads(raster_path.read_text())
    if pressure['environment']['gpu']!=raster['environment']['gpu']:raise ValueError('candidate migration changed GPU')
    F=lambda x:Fraction(str(x))
    initial=[[(-F(1),F(0),F(0)),(F(0),-F(1),F(0))]]
    empirical=initial;target5=initial;training=[]
    for label,C in (('single36',36),('single126',126)):
        samples=[r for r in pressure['paired_trials'] if r['group']==label]
        context=pressure['contexts'][f'r10_source_{label}_p32']
        requests=supply_model.static_tma_requests(context['row'],context['setup'])
        extras={tile['B']['source_KiB'][2] for tiles in requests['ctas'] for tile in tiles}
        if extras!={12.}:raise ValueError('expected same 12 KiB B extra128 coverage per request')
        X=12*17;kt=context['row']['k']//64
        lo,hi=min(r['L_cycles_delta'] for r in samples),max(r['L_cycles_delta'] for r in samples)
        empirical=_source_service_clip(empirical,C,[X],kt,lo,hi)
        # The 5% band constrains each process's paired delta relative to its
        # observed variant L; it is not a 5% error on the sometimes-zero delta.
        lo5=max(r['L_cycles_delta']-.05*r['L_cycles_variant'] for r in samples)
        hi5=min(r['L_cycles_delta']+.05*r['L_cycles_variant'] for r in samples)
        target5=_source_service_clip(target5,C,[X],kt,lo5,hi5)
        training.append(dict(group=label,C=C,X=X,kt=kt,processes=samples,
            empirical_delta_interval=[lo,hi],per_process_5pct_L_delta_interval=[lo5,hi5],
            baseline_empirical_upper=max(r['L_cycles_base'] for r in samples),
            baseline_5pct_upper=min(1.05*r['L_cycles_base'] for r in samples)))
    def add_baseline(regions,kind):
        bounded=[]
        for region in regions:
            conditions=region+[(F(r['C']*r['kt']),F(0),F(r[kind])) for r in training]
            if _source_polygon_vertices(conditions):bounded.append(conditions)
        return bounded
    families=dict(response_empirical=empirical,
        response_plus_baseline_empirical=add_baseline(empirical,'baseline_empirical_upper'),
        process_5pct_L_and_baseline=add_baseline(target5,'baseline_5pct_upper'))
    def describe(regions):
        vertices=sorted({point for region in regions for point in _source_polygon_vertices(region)})
        return dict(feasible=bool(vertices),parameter_ranges={name:[float(min(v[i] for v in vertices)),float(max(v[i] for v in vertices))]
            for i,name in enumerate(('a','b'))} if vertices else {},
            regions=[dict(vertices=[[float(a),float(b)] for a,b in _source_polygon_vertices(r)],
                exact_constraints=[[str(x) for x in c] for c in r]) for r in regions])
    descriptions={name:describe(regions) for name,regions in families.items()}

    tests=[]
    for trial in pressure['paired_trials']:
        if trial['group']=='persistent132':
            tests.append(dict(cohort='739585 persistent132 (not trained)',case='r10_source_persistent132_p32',
                trial=trial['trial'],j=0,T='3/4 selected first 36',C=132,Xs=[204.],kt=16,
                observed_delta=trial['L_cycles_delta'],observed_variant_L=trial['L_cycles_variant']))
    positions=list(csv.DictReader(position_path.open()))
    for case,context in raster['contexts'].items():
        if not case.endswith('p32'):continue
        row,setup=context['row'],context['setup']
        work=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row['raster_order'])
        features=source_panel_features(work)['windows']
        for r in [v for v in positions if v['variant']==case]:
            j,T=int(r['j']),int(r['T'])
            members=[v for v in features if v['j']==j and v['T']==T]
            xs=sorted(12*v['B_peer'] for v in members);n=len(xs)
            middle=[xs[(n-1)//2],xs[n//2]]
            tests.append(dict(cohort='739571 raster development transfer',case=case,trial=int(r['trial']),j=j,T=T,
                C=members[0]['active_ctas'],Xs=middle,kt=row['k']//64,
                observed_delta=float(r['L_cycles_delta']),observed_variant_L=float(r['L_cycles_variant'])))
    interval_cache={}
    for t in tests:
        key=(t['C'],tuple(t['Xs']),t['kt'])
        if key not in interval_cache:
            interval_cache[key]={name:_source_service_interval(regions,*key) for name,regions in families.items() if regions}
        t['predicted_delta_intervals']=interval_cache[key]
        t['conditional_L_error_intervals']={name:[(p-t['observed_delta'])/t['observed_variant_L'] for p in interval]
            for name,interval in t['predicted_delta_intervals'].items()}
    # Illustrative, training-only parameter choices. Every witness is scored as
    # one shared (a,b) across all positions; no target-wise parameter selection.
    calibration_points=sorted({point for region in families['response_plus_baseline_empirical']
        for point in _source_polygon_vertices(region)})
    witnesses=[]
    for a,b in calibration_points:
        per_cohort=defaultdict(list)
        for t in tests:
            prediction=sum(float(_source_service_value(a,b,F(t['C']),F(x),F(t['kt']))) for x in t['Xs'])/len(t['Xs'])
            per_cohort[t['cohort']].append(abs((prediction-t['observed_delta'])/t['observed_variant_L']))
        witnesses.append(dict(a=float(a),b=float(b),scores={name:dict(group_processes=len(values),
            median_abs_pct=100*statistics.median(values),max_abs_pct=100*max(values)) for name,values in per_cohort.items()}))
    grouped=defaultdict(list)
    for t in tests:grouped[t['cohort'],t['case'],t['j'],str(t['T'])].append(t)
    summaries=[]
    for (cohort,case,j,T),items in grouped.items():
        t=items[0]
        summaries.append(dict(cohort=cohort,case=case,j=j,T=T,C=t['C'],Xs=t['Xs'],kt=t['kt'],
            observed_delta=dict(median=statistics.median(r['observed_delta'] for r in items),
                trial_min=min(r['observed_delta'] for r in items),trial_max=max(r['observed_delta'] for r in items)),
            observed_variant_L_median=statistics.median(r['observed_variant_L'] for r in items),
            predicted_delta_intervals=t['predicted_delta_intervals'],
            conditional_L_error_envelopes={name:[min(r['conditional_L_error_intervals'][name][0] for r in items),
                max(r['conditional_L_error_intervals'][name][1] for r in items)] for name in families},
            zero_response_max_L_error_pct=100*max(abs(r['observed_delta'])/r['observed_variant_L'] for r in items)))
    prospective=[dict(active_ctas=C,logical_n=C//18*128,B_peer=17,kt=16,
        delta_interval=_source_service_interval(families['response_plus_baseline_empirical'],C,[204],16))
        for C in (54,72,90,108)]
    report=dict(scope='Post-measurement development transfer, not new heldout validation or full GEMM prediction.',
        formula='Delta_L=Kt*(max(512,a*C+b*B_extra128_KiB*peer_B)-max(512,a*C)); a,b>=0.',
        definitions=dict(C='software-ordinal active CTA count',B_extra128='extra 128-byte address coverage per B request, not traffic',
            peer_B='other current-ordinal requests for the same B panel',
            baseline='A necessary condition for nonnegative omitted startup is Kt*a*C<=observed aligned L. '
                'It does not establish a unique startup, a complete baseline fit or the physical identity of either coefficient.'),
        training=training,parameter_families=descriptions,transfer_groups=summaries,transfer_processes=tests,
        training_vertex_witnesses=witnesses,
        witness_selection='All vertices of the training-derived baseline-constrained domain are listed. '
            'A successful vertex was noticed after inspecting transfer scores: it is an existence example, '
            'not a preselected predictor or a physically preferred minimum-a rule.',
        prospective_single_output_sensitivity=prospective,
        calibration_next_step='First consider reclassifying the existing AlongN/B32 development data as extra calibration '
            'if its low-peer direction is the main uncertainty; then it ceases to be an independent migration check. '
            'If another measurement is needed, compare these fixed-map single-output pressure intervals and choose the '
            'condition that shrinks the needed migration prediction interval, not one that merely makes all parameters unique. '
            'No such measurement is launched here.',
        interpretation=['Empirical bands are observed trial extrema, not confidence intervals; the separate 5% family enforces every process.',
            'Transfer errors are grouped Δ residuals divided by measured variant L. Observed aligned L anchors this response diagnosis; '
            'it is not a predicted baseline or an end-to-end accuracy claim.',
            'Each group interval marginalizes over the same training domain. Best endpoints chosen separately across groups do not prove '
            'a single shared parameter pair passes. Witness rows always use one pair for every transfer position.',
            'A nonzero residual may be acceptable at the full-window error target; exact equality is not required.',
            'No 739571 or persistent132 observation trains the reported families or chooses the vertex witnesses.'],
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (pressure_path,raster_path,position_path,Path(supply_model.__file__))},
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'R10-shared-service-candidate.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 两个单输出对照约束共享供给候选','',
        '`ΔL=Kt·[max(512,aC+b·B_extra128·peer_B)−max(512,aC)]`，a、b非负。'
        '只用739585的single36/single126训练，B extra128均为12 KiB、peer均17。'
        '响应经验包络与每进程5%整L容许范围分列；额外施加非负启动项的必要baseline界，避免共同基线错误被相减消掉。','',
        '| 训练约束族 | a范围 | b范围 |','|---|---:|---:|']
    for name,d in descriptions.items():
        ranges=d['parameter_ranges'];lines.append(f'| {name} | {ranges.get("a")} | {ranges.get("b")} |')
    lines+=['','| 独立开发迁移组 | 实测ΔL中位 | baseline必要界后的ΔL区间 | 整L误差包络 |',
        '|---|---:|---:|---:|']
    for g in summaries:
        interval=g['predicted_delta_intervals']['response_plus_baseline_empirical']
        errors=g['conditional_L_error_envelopes']['response_plus_baseline_empirical']
        lines.append(f'| {g["case"]}, j{g["j"]}/T{g["T"]} | {g["observed_delta"]["median"]:.1f} | '
            f'{interval[0]:.1f}～{interval[1]:.1f} | {100*errors[0]:+.2f}%～{100*errors[1]:+.2f}% |')
    lines+=['','上述每组区间保留参数非唯一性，不允许各组挑不同参数后声称通过。'
        'JSON列出全部训练域顶点，每个例子用同一对参数评估全部迁移位置。'
        '其中好点是在看完迁移结果后注意到的，只说明域内存在满足误差尺度的例子，不是预先选中的预测器。'
        '最小a没有物理优先性；若以后采用此确定规则，须在新验证冻结前声明。'
        '这些误差用实测对齐L作锚，只衡量B32响应是否已足够解释整窗；不是完整L或总时间预测的留出成绩。','']
    lines+=['| 训练域中的共享参数例子 | persistent中位/最大误差 | raster中位/最大误差 |',
        '|---|---:|---:|']
    for witness in witnesses:
        p=witness['scores']['739585 persistent132 (not trained)'];r=witness['scores']['739571 raster development transfer']
        lines.append(f'| a={witness["a"]:.6f}, b={witness["b"]:.6f} | {p["median_abs_pct"]:.2f}% / {p["max_abs_pct"]:.2f}% | '
            f'{r["median_abs_pct"]:.2f}% / {r["max_abs_pct"]:.2f}% |')
    lines+=['','不能只展示好点：整个训练容许域对persistent的误差上界约12.69%，价格尚未识别到可保证迁移目标的程度。'
        '已有AlongN数据可以转为额外开发校准，代价是它不再作为独立迁移检查；若再测，应先比较下表对剩余预测区间的区分度。','',
        '| 同输入map、单输出的候选CTA数 | B32增量预测区间（cycle） |','|---:|---:|']
    for p in prospective:
        lo,hi=p['delta_interval'];lines.append(f'| {p["active_ctas"]} | {lo:.1f}～{hi:.1f} |')
    lines+=['','这些是已有模型域的前瞻区间，没有申请或运行新GPU测量。','']
    lines+=['逐CTA窗口的补充检查见 [R10-shared-service-windows.md](R10-shared-service-windows.md)。上述0.87%/7.12%只对应分组中位，不能表示所有CTA准确率。','']
    (output/'R10-shared-service-candidate.md').write_text('\n'.join(lines))
    windows=shared_source_window_diagnostics(root,output)
    failing={name:d['positions_with_median_target_lower_bound_over_10pct'] for name,d in windows['full_parameter_domain'].items()}
    lines.insert(2,'逐CTA补充检查已经改变适用结论：完整参数域对十进程中位目标仍有必超10%的位置，'
        +str(failing)+'。下文的好点只说明分组响应可拟合，不能说明所有CTA达到目标；应先处理位置状态缺项，暂不靠C90缩参数域。\n')
    (output/'R10-shared-service-candidate.md').write_text('\n'.join(lines))
    return report

def _source_service_window_errors(a,b,windows,features,kt):
    """One shared parameter pair; each CTA keeps its own static sharing count."""
    result=[]
    for window in windows:
        f=features[window['cta'],window['j']]
        if (f['mi'],f['ni'])!=(window['mi'],window['ni']):raise ValueError('static source/window identity mismatch')
        delta=float(_source_service_value(a,b,f['active_ctas'],12*f['B_peer'],kt))
        predicted=window['L_cycles_base']+delta
        result.append(dict(cta=window['cta'],j=window['j'],T=window['T'],mi=window['mi'],ni=window['ni'],
            active_ctas=f['active_ctas'],B_peer=f['B_peer'],observed_B0_L=window['L_cycles_base'],
            observed_B32_L=window['L_cycles_variant'],predicted_delta=delta,predicted_B32_L=predicted,
            relative_error=predicted/window['L_cycles_variant']-1))
    return result


def _source_interval_error_floor(interval,observed_delta,observed_variant_L):
    lo,hi=interval
    return max(lo-observed_delta,observed_delta-hi,0)/observed_variant_L


def shared_source_window_diagnostics(root, output):
    """Replay every CTA/output, retaining independent processes as the summary unit."""
    from collections import defaultdict
    from analyze_r18 import replay
    from analyze_r13_sm import select_attempts
    from analyze_r13_supply import _failure_pair_windows
    import v08_model

    root,output=Path(root),Path(output)
    candidate_path=output/'R10-shared-service-candidate.json'
    candidate=json.loads(candidate_path.read_text())
    domain=candidate['parameter_families']['response_plus_baseline_empirical']['regions']
    vertices=sorted({point for region in domain for point in _source_polygon_vertices(
        [[Fraction(x) for x in row] for row in region['exact_constraints']])})
    inputs=[];hashes={};gpu=None
    for folder,pairs in [
        ('20261010-R10-raster-pitch',[('739571 AlongM','r10_raster_along_m_p0','r10_raster_along_m_p32'),
                                     ('739571 AlongN','r10_raster_along_n_p0','r10_raster_along_n_p32')]),
        ('20261010-R10-source-pressure',[('739585 persistent132','r10_source_persistent132_p0','r10_source_persistent132_p32')])]:
        run=root.parent/folder
        rows={r['id']:r for r in json.loads((run/'cases.json').read_text())}
        setups={r['case']:r['setup'] for r in json.loads((run/'static_setup.json').read_text())}
        for cohort,base_name,variant_name in pairs:
            row,setup=rows[variant_name],setups[variant_name]
            if gpu is None:gpu=setup['gpu_uuid']
            if setup['gpu_uuid']!=gpu:raise ValueError('window-level development check changed GPU')
            work=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row.get('raster_order','heuristic'))
            features={(r['cta'],r['j']):r for r in source_panel_features(work)['windows']}
            calls={}
            for name in (base_name,variant_name):
                selected,_,_=select_attempts(run,name,('dual',));calls[name]={}
                for item in selected:
                    record=item['record'];obs=replay(run,record,rows[name]);calls[name][record['trial']]=obs
                    for path in (run/item['path'],run/record['raw']):
                        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                if len(calls[name])!=10:raise ValueError('window check requires ten dual calls')
            if calls[base_name].keys()!=calls[variant_name].keys():raise ValueError('window trial sets differ')
            for trial in sorted(calls[base_name]):
                base,variant=calls[base_name][trial],calls[variant_name][trial]
                windows=_failure_pair_windows(base['ctas'],variant['ctas'])
                ctas={label:{c['cta']:c for c in obs['ctas']} for label,obs in (('base',base),('variant',variant))}
                for w in windows:
                    w['base_sm']=ctas['base'][w['cta']]['sm'];w['variant_sm']=ctas['variant'][w['cta']]['sm']
                    if w['j']!=0:continue
                    for label in ('base','variant'):
                        c=ctas[label][w['cta']]
                        direct=c['tiles_ns'][0][1]-c['entry_ns']
                        if direct!=w['prefix_ns_'+label]+w['L_ns_'+label]:raise ValueError('entry-to-FIRST_MMA plus L does not close')
                        w['entry_to_main_end_ns_'+label]=direct
                    w['entry_to_main_end_ns_delta']=w['entry_to_main_end_ns_variant']-w['entry_to_main_end_ns_base']
                critical=max(variant['ctas'],key=lambda c:c['end_ns'])['cta']
                inputs.append(dict(cohort=cohort,case=variant_name,trial=trial,kt=row['k']//64,
                    critical_observed_time_cta=critical,windows=windows,features=features))
        for file in ('cases.json','static_setup.json','source_hashes.json'):
            path=run/file;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    def span(values):
        values=list(values)
        return dict(median=statistics.median(values),process_min=min(values),process_max=max(values))
    domain_constraints=[[[Fraction(x) for x in row] for row in region['exact_constraints']] for region in domain]
    relaxed_constraints=[[[Fraction(x) for x in row] for row in region['exact_constraints']]
        for region in candidate['parameter_families']['process_5pct_L_and_baseline']['regions']]
    intervals={};relaxed_intervals={};positions=defaultdict(list);process_bounds=[]
    for data in inputs:
        current=[]
        for w in data['windows']:
            f=data['features'][w['cta'],w['j']]
            key=f['active_ctas'],12*f['B_peer'],data['kt']
            if key not in intervals:intervals[key]=_source_service_interval(domain_constraints,key[0],[key[1]],key[2])
            if key not in relaxed_intervals:relaxed_intervals[key]=_source_service_interval(relaxed_constraints,key[0],[key[1]],key[2])
            lo,hi=intervals[key];delta=w['L_cycles_variant']-w['L_cycles_base']
            bound=_source_interval_error_floor((lo,hi),delta,w['L_cycles_variant'])
            entry=dict(cohort=data['cohort'],case=data['case'],trial=data['trial'],cta=w['cta'],j=w['j'],T=w['T'],
                mi=w['mi'],ni=w['ni'],C=f['active_ctas'],B_peer=f['B_peer'],kt=data['kt'],
                observed_delta=delta,observed_variant_L=w['L_cycles_variant'],prediction_interval=[lo,hi],
                unavoidable_error_pct=100*bound,relaxed_training_prediction_interval=relaxed_intervals[key],
                relaxed_training_error_floor_pct=100*_source_interval_error_floor(relaxed_intervals[key],delta,w['L_cycles_variant']))
            current.append(entry);positions[data['cohort'],w['cta'],w['j']].append(entry)
        process_bounds.append(dict(cohort=data['cohort'],trial=data['trial'],windows=len(current),
            maximum_unavoidable_error_pct=max(r['unavoidable_error_pct'] for r in current),
            fraction_unavoidable_over_10pct=sum(r['unavoidable_error_pct']>10 for r in current)/len(current),
            worst=max(current,key=lambda r:r['unavoidable_error_pct'])))
    position_medians=[]
    for (cohort,cta,j),values in sorted(positions.items()):
        first=values[0]
        if len(values)!=10 or len({(r['C'],r['B_peer'],r['mi'],r['ni']) for r in values})!=1:
            raise ValueError('position median needs ten matching static windows')
        delta=statistics.median(v['observed_delta'] for v in values)
        denominator=statistics.median(v['observed_variant_L'] for v in values)
        position_medians.append(dict(cohort=cohort,cta=cta,j=j,T=first['T'],mi=first['mi'],ni=first['ni'],
            C=first['C'],B_peer=first['B_peer'],kt=first['kt'],prediction_interval=first['prediction_interval'],
            observed_delta_median=delta,observed_delta_trial_min=min(r['observed_delta'] for r in values),
            observed_delta_trial_max=max(r['observed_delta'] for r in values),observed_variant_L_median=denominator,
            median_target_unavoidable_error_pct=100*_source_interval_error_floor(first['prediction_interval'],delta,denominator),
            per_process_unavoidable_error_pct=span(r['unavoidable_error_pct'] for r in values),
            relaxed_training_prediction_interval=first['relaxed_training_prediction_interval'],
            relaxed_training_median_target_error_floor_pct=100*_source_interval_error_floor(first['relaxed_training_prediction_interval'],delta,denominator),
            relaxed_training_per_process_error_floor_pct=span(r['relaxed_training_error_floor_pct'] for r in values)))
    domain_summary={}
    for cohort in sorted({p['cohort'] for p in position_medians}):
        members=[p for p in position_medians if p['cohort']==cohort]
        worst=max(members,key=lambda r:r['median_target_unavoidable_error_pct'])
        domain_summary[cohort]=dict(positions=len(members),
            positions_with_median_target_lower_bound_over_10pct=sum(p['median_target_unavoidable_error_pct']>10 for p in members),
            worst_position=worst,worst_position_processes=positions[cohort,worst['cta'],worst['j']],
            per_process_maximum_lower_bound=span(p['maximum_unavoidable_error_pct'] for p in process_bounds if p['cohort']==cohort))
        relaxed_worst=max(members,key=lambda r:r['relaxed_training_median_target_error_floor_pct'])
        domain_summary[cohort]['relaxed_training_5pct_domain']=dict(
            positions_with_median_target_lower_bound_over_10pct=sum(p['relaxed_training_median_target_error_floor_pct']>10 for p in members),
            worst_position=relaxed_worst)
    first_boundaries={}
    for cohort in ('739571 AlongM','739585 persistent132'):
        records=[]
        for data in [d for d in inputs if d['cohort']==cohort]:
            for w in data['windows']:
                f=data['features'][w['cta'],w['j']]
                if w['j']!=0 or w['T']!=3 or w['mi']!=1 or f['B_peer']!=17:continue
                records.append(dict(trial=data['trial'],cta=w['cta'],mi=w['mi'],ni=w['ni'],A_peer=f['A_peer'],B_peer=f['B_peer'],
                    prefix_delta_ns=w['prefix_ns_delta'],prefill_delta_ns=w['prefill_ns_delta'],L_delta_ns=w['L_ns_delta'],
                    entry_to_main_end_delta_ns=w['entry_to_main_end_ns_delta']))
        fields=('prefix_delta_ns','prefill_delta_ns','L_delta_ns','entry_to_main_end_delta_ns')
        position_summary={str(cta):{field:span(r[field] for r in records if r['cta']==cta) for field in fields}
            for cta in sorted({r['cta'] for r in records})}
        contrasts=[]
        for trial in range(10):
            a=next(r for r in records if r['cta']==55 and r['trial']==trial)
            b=next(r for r in records if r['cta']==91 and r['trial']==trial)
            contrasts.append(dict(trial=trial,**{field:b[field]-a[field] for field in fields}))
        first_boundaries[cohort]=dict(positions=position_summary,processes=records,CTA91_minus_CTA55=contrasts,
            contrast_summary={field:span(r[field] for r in contrasts) for field in fields},
            closure='Within each B0 and B32 call, (FIRST_MMA-entry)+(MAIN_END-FIRST_MMA)=MAIN_END-entry exactly. '
                'Pitch differences still compare independent processes; no cross-call event timeline is constructed.')
    witnesses=[]
    for a,b in vertices:
        processes=[]
        for data in inputs:
            errors=_source_service_window_errors(float(a),float(b),data['windows'],data['features'],data['kt'])
            critical=[r for r in errors if r['cta']==data['critical_observed_time_cta']]
            groups=defaultdict(list)
            for r in errors:groups[r['j'],r['T'],r['B_peer']].append(r)
            processes.append(dict(cohort=data['cohort'],case=data['case'],trial=data['trial'],windows=len(errors),
                median_absolute_error_pct=100*statistics.median(abs(r['relative_error']) for r in errors),
                maximum_absolute_error_pct=100*max(abs(r['relative_error']) for r in errors),
                fraction_over_10pct=sum(abs(r['relative_error'])>.1 for r in errors)/len(errors),
                worst_window=max(errors,key=lambda r:abs(r['relative_error'])),
                critical_observed_time_cta=data['critical_observed_time_cta'],
                critical_CTA_maximum_L_error_pct=100*max(abs(r['relative_error']) for r in critical),
                critical_CTA_sum_L_error_pct=100*(sum(r['predicted_B32_L'] for r in critical)/sum(r['observed_B32_L'] for r in critical)-1),
                groups=[dict(j=j,T=T,B_peer=peer,windows=len(members),
                    median_absolute_error_pct=100*statistics.median(abs(r['relative_error']) for r in members),
                    maximum_absolute_error_pct=100*max(abs(r['relative_error']) for r in members))
                    for (j,T,peer),members in sorted(groups.items())]))
        cohorts={}
        for name in sorted({p['cohort'] for p in processes}):
            members=[p for p in processes if p['cohort']==name]
            cohorts[name]=dict(processes=len(members),windows_per_process=sorted({p['windows'] for p in members}),
                median_absolute_error_pct=span(p['median_absolute_error_pct'] for p in members),
                maximum_absolute_error_pct=span(p['maximum_absolute_error_pct'] for p in members),
                fraction_over_10pct=span(p['fraction_over_10pct'] for p in members),
                critical_CTA_maximum_L_error_pct=span(p['critical_CTA_maximum_L_error_pct'] for p in members),
                critical_CTA_sum_L_error_pct=span(p['critical_CTA_sum_L_error_pct'] for p in members))
            coordinate_scores=[]
            for position in [p for p in position_medians if p['cohort']==name]:
                delta=float(_source_service_value(a,b,position['C'],12*position['B_peer'],position['kt']))
                error=100*(delta-position['observed_delta_median'])/position['observed_variant_L_median']
                coordinate_scores.append(dict(cta=position['cta'],j=position['j'],B_peer=position['B_peer'],
                    predicted_delta=delta,observed_delta_median=position['observed_delta_median'],relative_error_pct=error))
            cohorts[name]['position_median_target_errors']=dict(
                median_absolute_error_pct=statistics.median(abs(p['relative_error_pct']) for p in coordinate_scores),
                maximum_absolute_error_pct=max(abs(p['relative_error_pct']) for p in coordinate_scores),
                worst=max(coordinate_scores,key=lambda p:abs(p['relative_error_pct'])))
        witnesses.append(dict(a=float(a),b=float(b),exact_a=str(a),exact_b=str(b),cohorts=cohorts,processes=processes))
    placement={}
    for cohort in sorted({d['cohort'] for d in inputs}):
        placement[cohort]={}
        for cta in (55,91):
            matched=[dict(trial=d['trial'],**{k:w[k] for k in ('base_sm','variant_sm')})
                for d in inputs if d['cohort']==cohort for w in d['windows'] if w['cta']==cta and w['j']==0]
            placement[cohort][str(cta)]=dict(processes=matched,
                base_sms=sorted({r['base_sm'] for r in matched}),variant_sms=sorted({r['variant_sm'] for r in matched}))
    hashes[str(candidate_path)]=hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    report=dict(scope='Per-window response development check; every window uses observed B0 L plus the same shared (a,b) static delta. '
        'This remains an observed-baseline diagnostic, not full-kernel prediction or frozen validation.',
        gpu=gpu,parameter_domain='All vertices of response_plus_baseline_empirical, without CTA-specific selection.',
        input_process_pairs=len(inputs),input_windows=sum(len(d['windows']) for d in inputs),witnesses=witnesses,
        full_parameter_domain=domain_summary,process_domain_lower_bounds=process_bounds,selected_CTA_placement=placement,
        position_median_targets=position_medians,first_wait_boundary_checks=first_boundaries,
        interpretation=['The earlier 0.87%/7.12% values summarize grouped medians, not all CTA windows.',
            'prefix=FIRST_MMA-entry, whereas prefill=FIRST_MMA-producer_first_work; only the latter is model S. Source identity, arrival/pipeline state and hardware placement remain distinct possible omissions.',
            'Each process first reports median and maximum absolute error over all its CTA/output windows. '
            'Then process summaries are reduced; windows are not independent repetitions.',
            'Critical CTA is selected by observed B32 final globaltimer only. Its L-window errors do not estimate '
            'the complete CTA/kernel error because supply, epilogue, timing conversion and CTA selection remain observed.',
            'No parameter was selected separately by CTA, ordinal, raster, or process.',
            'The full-domain residual floor is distance(observed delta, predicted delta interval)/observed pitched L. '
            'It permits each window its own optimistic endpoint only to prove a necessary lower bound, never to construct a predictor.',
            'Position targets first take the median of paired delta across ten independent processes, then normalize by '
            'the median pitched L at that same CTA/j. Trial ranges are reported separately; 12960 windows are not independent replicates.'],
        input_sha256=hashes,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'R10-shared-service-windows.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 共享供给候选的逐CTA窗口检查','',
        '每个窗口使用自己的 `X=12×B_peer`，全部窗口共享同一对训练域顶点参数。'
        '预测为实测B0 L加静态Δ，对照同trial B32 L；仍是响应诊断，不是完整L或整核预测。'
        '先在每个进程内求中位、最大误差，再报告十进程范围，原分组0.87%/7.12%不能当作所有CTA准确率。','',
        '| a / b | 条件 | 进程中位误差的十次范围 | 进程最大误差的十次范围 | 超10%窗口比例范围 |',
        '|---|---|---:|---:|---:|']
    for witness in witnesses:
        for name,c in witness['cohorts'].items():
            med,maximum,frac=(c[k] for k in ('median_absolute_error_pct','maximum_absolute_error_pct','fraction_over_10pct'))
            lines.append(f'| {witness["a"]:.6f} / {witness["b"]:.6f} | {name} | '
                f'{med["process_min"]:.2f}%～{med["process_max"]:.2f}% | '
                f'{maximum["process_min"]:.2f}%～{maximum["process_max"]:.2f}% | '
                f'{100*frac["process_min"]:.2f}%～{100*frac["process_max"]:.2f}% |')
    lines+=['','## 完整参数域能否消除稳定位置误差？','',
        '| 条件 | 十进程中位目标仍必超10%的位置数 | 最差位置CTA/j/peer | ΔL预测区间 | 配对ΔL中位 / 十次范围 | 位置中位目标的误差下界 |',
        '|---|---:|---|---:|---:|---:|']
    for name,d in domain_summary.items():
        w=d['worst_position'];lo,hi=w['prediction_interval']
        lines.append(f'| {name} | {d["positions_with_median_target_lower_bound_over_10pct"]}/{d["positions"]} | '
            f'{w["cta"]}/{w["j"]}/{w["B_peer"]} | {lo:.1f}～{hi:.1f} | '
            f'{w["observed_delta_median"]:.1f} / {w["observed_delta_trial_min"]:.1f}～{w["observed_delta_trial_max"]:.1f} | '
            f'{w["median_target_unavoidable_error_pct"]:.2f}% |')
    lines+=['','即使把训练允许范围放宽到每进程5%整L误差并保留baseline必要界：','',
        '| 条件 | 位置中位目标仍必超10%的个数 | 最差位置中位目标下界 |','|---|---:|---:|']
    for name,d in domain_summary.items():
        relaxed=d['relaxed_training_5pct_domain']
        lines.append(f'| {name} | {relaxed["positions_with_median_target_lower_bound_over_10pct"]}/432 | '
            f'{relaxed["worst_position"]["relaxed_training_median_target_error_floor_pct"]:.2f}% |')
    lines+=['','这些下界遍历完整训练参数域，包含目标max分支的内部交点，不只检查八个顶点。'
        '每个位置先对十次配对响应取中位；同时保留单进程范围，避免用一次噪声极值判结构失败。'
        '允许每个窗口各自选最有利参数仍超10%，才说明缩小当前参数域无法救回该位置。',
        '','JSON保留每进程最差窗口、按(j,T,B_peer)的局部误差，以及实测末次完成CTA的L窗口误差。'
        '关键CTA由B32最终globaltimer确定，不能把这些L误差称作关键CTA总时间误差。','']
    lines+=['## CTA91的短L是否把成本移到了首段等待？','',
        '只检查同mi=1、T3、B peer=17的五个位置，并重点比较CTA91与55。'
        'S与L逐调用闭合；B0/B32仍是独立进程，以下是配对响应的差，不拼接不同调用的时间线。','',
        '| 条件 | CTA91−55的Δprefix | CTA91−55的Δprefill | CTA91−55的ΔL | CTA91−55的Δ(prefix+L) |',
        '|---|---:|---:|---:|---:|']
    for name,d in first_boundaries.items():
        cells=[]
        for field in ('prefix_delta_ns','prefill_delta_ns','L_delta_ns','entry_to_main_end_delta_ns'):
            v=d['contrast_summary'][field];cells.append(f'{v["median"]:+.0f} [{v["process_min"]:+.0f},{v["process_max"]:+.0f}]')
        lines.append(f'| {name} | '+' | '.join(cells)+' |')
    lines+=['','单位ns，括号为十进程范围。CTA91的首段等待确实增加，但只补偿部分L差；'
        '两批Δ(prefix+L)差的十次范围均与零分离，不能只靠移动FIRST_MMA边界解释。'
        '这继续约束源身份/到达历史/流水组织，不直接归因为地址bank或某级缓存。'
        '既然完整参数域在稳定位置上仍必超10%，此时补C90只缩参数域也不能救回，应先处理这个状态缺项。','']
    (output/'R10-shared-service-windows.md').write_text('\n'.join(lines))
    return report

def _base_shift_response_pairs(reference, shifted):
    """Two predeclared mappings; a +256 B origin matches the next 128-column panel."""
    by_work={(r['cta'],r['j'],r['T'],r['mi'],r['ni']):r for r in reference}
    by_coord={(r['mi'],r['ni']):r for r in reference}
    from collections import Counter,defaultdict
    by_sm=defaultdict(list)
    for r in reference:
        if r['base_sm']==r['variant_sm']:by_sm[r['variant_sm'],r['j']].append(r)
    shifted_sm_counts=Counter((r['variant_sm'],r['j']) for r in shifted if r['base_sm']==r['variant_sm'])
    fixed=[];neighbor=[];matched_neighbor=[];fixed_common=[];same_sm=[]
    for row in shifted:
        key=tuple(row[k] for k in ('cta','j','T','mi','ni'))
        ref=by_work[key];fixed.append((ref,row))
        next_panel=by_coord.get((row['mi'],row['ni']+1))
        if next_panel is not None:
            neighbor.append((next_panel,row))
            context=('j','T','active_ctas','A_peer','B_peer')
            if all(next_panel[k]==row[k] for k in context):
                matched_neighbor.append((next_panel,row));fixed_common.append((ref,row))
        if row['base_sm']==row['variant_sm']:
            sm_key=row['variant_sm'],row['j'];sm_refs=by_sm[sm_key]
            if len(sm_refs)==1 and shifted_sm_counts[sm_key]==1 and sm_refs[0]['T']==row['T']:
                same_sm.append((sm_refs[0],row))
    return dict(same_work=fixed,next_B_panel=neighbor,next_B_panel_matched_context=matched_neighbor,
        same_work_on_neighbor_subset=fixed_common,same_SM_and_ordinal=same_sm,
        same_work_first_T3_common=[(a,b) for a,b in fixed_common if b['j']==0 and b['T']==3],
        next_B_first_T3_same_context=[(a,b) for a,b in matched_neighbor if b['j']==0 and b['T']==3])


def _base_shift_pattern_metrics(pairs,field):
    """Pattern agreement per process; no timing parameter is fitted."""
    if not pairs:return dict(windows=0,median_signed_residual=None,median_absolute_residual=None,
        root_mean_square_residual=None,maximum_absolute_residual=None,correlation=None)
    x=[a[field] for a,b in pairs];y=[b[field] for a,b in pairs]
    residual=[b-a for a,b in zip(x,y)]
    mx,my=statistics.mean(x),statistics.mean(y)
    covariance=sum((a-mx)*(b-my) for a,b in zip(x,y))
    variance=math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return dict(windows=len(pairs),median_signed_residual=statistics.median(residual),
        median_absolute_residual=statistics.median(abs(v) for v in residual),
        root_mean_square_residual=math.sqrt(statistics.mean(v*v for v in residual)),
        maximum_absolute_residual=max(map(abs,residual)),correlation=covariance/variance if variance else None)


def base_shift_diagnostics(root, output):
    """Address-origin/pitch factorial, preserving all CTA positions and SM identity."""
    from collections import defaultdict
    from analyze_r18 import replay
    from analyze_r13_sm import select_attempts
    from analyze_r13_supply import _failure_pair_windows
    import v08_model
    import v06_run as common

    root,output=Path(root),Path(output)
    common.verify(root)
    rows={r['id']:r for r in json.loads((root/'cases.json').read_text())}
    setups={r['case']:r['setup'] for r in json.loads((root/'static_setup.json').read_text())}
    controls=[r for r in rows.values() if r['set']=='ctrl']
    variable={'id','group','ldb','b_offset_bytes'}
    if len(controls)!=4 or any({k:v for k,v in r.items() if k not in variable}!=
        {k:v for k,v in controls[0].items() if k not in variable} for r in controls):
        raise ValueError('base-shift controls changed a non-factor condition')
    if {(r['b_offset_bytes'],r['ldb']) for r in controls}!={(o,p) for o in (0,256) for p in (3072,3088)}:
        raise ValueError('base-shift factorial is incomplete')
    def span(v):
        v=list(v)
        return dict(median=statistics.median(v),process_min=min(v),process_max=max(v))
    calls={};plain={};hashes={};processes=[];failed={};contexts={};features=None
    for row in rows.values():
        name=row['id'];calls[name]={};plain[name]={}
        selected,failures,_=select_attempts(root,name,tuple(row['variants']))
        failed[name]=[x['path'] for x in failures]
        if row['set']=='ctrl':
            setup=setups[name]
            for field in ('b_offset_bytes','b_allocation_extra_bytes'):
                if setup.get(field)!=row[field]:raise ValueError('actual B pointer/allocation option changed')
            work=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row['raster_order'])
            current={(r['cta'],r['j']):r for r in source_panel_features(work)['windows']}
            if features is None:features=current
            if current!=features:raise ValueError('base shift changed static software work')
            contexts[name]=dict(row=row,setup=setup,work_sha256=hashlib.sha256(json.dumps(work).encode()).hexdigest())
        for item in selected:
            record=item['record'];obs=replay(root,record,row)
            for path in (root/item['path'],root/record['raw']):hashes[str(path)]=common.sha(path)
            if record['variant']=='plain':plain[name][record['trial']]=obs['elapsed_us']
            else:
                if [c['work'] for c in obs['ctas']]!=work:raise ValueError('base-shift trace/work mismatch')
                calls[name][record['trial']]=obs
                processes.append(dict(case=name,trial=record['trial'],elapsed_us=obs['elapsed_us'],
                    observed_sm_count=len({c['sm'] for c in obs['ctas']}),
                    cta_sm=[dict(cta=c['cta'],sm=c['sm']) for c in obs['ctas']],
                    address_metadata=obs.get('b_address')))
        if len(plain[name])!=10 or (row['set']=='ctrl' and calls[name].keys()!=plain[name].keys()):
            raise ValueError('base-shift analysis needs all ten declared successful processes')
    if len({setups[r['id']]['allocated_b_elements'] for r in controls})!=1:
        raise ValueError('base-shift total B capacity differs')
    if len({setups[r['id']]['gpu_uuid'] for r in controls})!=1:raise ValueError('factorial spans different cards')

    responses={};timing={};position_values=defaultdict(list);selected_values=[]
    fields=('L_cycles_delta','L_ns_delta','L_cycles_ratio','L_ns_ratio','L_rate_ratio',
        'prefix_ns_delta','prefill_ns_delta','prefix_plus_L_ns_delta','prefill_plus_L_ns_delta')
    for offset in (0,256):
        a,b=(f'r10_baseshift_o{offset}_p{pitch}' for pitch in (0,32))
        responses[offset]={}
        for trial in sorted(calls[a]):
            base,variant=calls[a][trial],calls[b][trial]
            windows=_failure_pair_windows(base['ctas'],variant['ctas'])
            maps={label:{c['cta']:c for c in obs['ctas']} for label,obs in (('base',base),('variant',variant))}
            for w in windows:
                f=features[w['cta'],w['j']]
                w.update(active_ctas=f['active_ctas'],A_peer=f['A_peer'],B_peer=f['B_peer'],
                    base_sm=maps['base'][w['cta']]['sm'],variant_sm=maps['variant'][w['cta']]['sm'])
                if w['j']==0:
                    for label in ('base','variant'):
                        c=maps[label][w['cta']]
                        if w['prefix_ns_'+label]+w['L_ns_'+label]!=c['tiles_ns'][0][1]-c['entry_ns']:
                            raise ValueError('prefix+L same-call closure failed')
                        if w['prefill_ns_'+label]+w['L_ns_'+label]!=c['tiles_ns'][0][1]-c['prod_ns']:
                            raise ValueError('prefill+L same-call closure failed')
                    w['prefix_plus_L_ns_delta']=w['prefix_ns_delta']+w['L_ns_delta']
                    w['prefill_plus_L_ns_delta']=w['prefill_ns_delta']+w['L_ns_delta']
                key=offset,w['cta'],w['j']
                position_values[key].append(dict(trial=trial,**w))
                if w['cta'] in (55,91) and w['j']==0:selected_values.append(dict(offset=offset,trial=trial,**w))
            responses[offset][trial]=windows
        timing[str(offset)]=dict(plain_pitch_change=span(plain[b][t]/plain[a][t]-1 for t in plain[a]),
            plain_pitch_delta_us=span(plain[b][t]-plain[a][t] for t in plain[a]),
            dual_disturbance={name:span(calls[name][t]['elapsed_us']/plain[name][t]-1 for t in plain[name]) for name in (a,b)})
    timing['base_shift_plain']={str(p):dict(relative=span(plain[f'r10_baseshift_o256_p{p}'][t]/plain[f'r10_baseshift_o0_p{p}'][t]-1 for t in range(10)),
        delta_us=span(plain[f'r10_baseshift_o256_p{p}'][t]-plain[f'r10_baseshift_o0_p{p}'][t] for t in range(10))) for p in (0,32)}

    timing['plain_offset_by_pitch_interaction_us']=span(
        (plain['r10_baseshift_o256_p32'][t]-plain['r10_baseshift_o256_p0'][t])-
        (plain['r10_baseshift_o0_p32'][t]-plain['r10_baseshift_o0_p0'][t]) for t in range(10))
    pattern_processes=[];interaction_values=defaultdict(list);selected_interactions=[]
    for trial in range(10):
        pairings=_base_shift_response_pairs(responses[0][trial],responses[256][trial])
        for mapping,pairs in pairings.items():
            for field in ('L_cycles_delta','L_ns_delta'):
                pattern_processes.append(dict(trial=trial,mapping=mapping,metric=field,
                    **_base_shift_pattern_metrics(pairs,field)))
        for a,b in pairings['same_work']:
            entry=dict(trial=trial,cta=b['cta'],j=b['j'],T=b['T'],mi=b['mi'],ni=b['ni'],
                offset0_base_sm=a['base_sm'],offset0_variant_sm=a['variant_sm'],
                offset256_base_sm=b['base_sm'],offset256_variant_sm=b['variant_sm'],
                **{field:b[field]-a[field] for field in fields if field in a})
            interaction_values[b['cta'],b['j']].append(entry)
            if b['cta'] in (55,91) and b['j']==0:selected_interactions.append(entry)
    pattern_summaries=[]
    for mapping,metric in sorted({(r['mapping'],r['metric']) for r in pattern_processes}):
        values=[r for r in pattern_processes if r['mapping']==mapping and r['metric']==metric]
        valid=[r for r in values if r['windows']]
        summary=dict(mapping=mapping,metric=metric,matched_window_counts=sorted({r['windows'] for r in values}),
            comparable_processes=len(valid),
            **{field:span(r[field] for r in valid) if valid else None for field in
                ('median_signed_residual','median_absolute_residual','root_mean_square_residual','maximum_absolute_residual')})
        correlations=[r['correlation'] for r in valid if r['correlation'] is not None]
        summary['correlation']=span(correlations) if correlations else None
        pattern_summaries.append(summary)
    position_summaries=[]
    for (offset,cta,j),values in sorted(position_values.items()):
        first=values[0]
        position_summaries.append(dict(offset=offset,cta=cta,j=j,T=first['T'],mi=first['mi'],ni=first['ni'],
            active_ctas=first['active_ctas'],A_peer=first['A_peer'],B_peer=first['B_peer'],
            base_sms=sorted({r['base_sm'] for r in values}),variant_sms=sorted({r['variant_sm'] for r in values}),
            **{field:span(r[field] for r in values) for field in fields if field in first}))
    interactions=[]
    for (cta,j),values in sorted(interaction_values.items()):
        first=values[0]
        interactions.append(dict(cta=cta,j=j,T=first['T'],mi=first['mi'],ni=first['ni'],
            **{field:span(r[field] for r in values) for field in fields if field in first}))
    references={}
    previous=root.parent/'20261010-R10-source-pressure'
    if json.loads((previous/'environment.json').read_text())['gpu']!=json.loads((root/'environment.json').read_text())['gpu']:
        raise ValueError('base-shift reference is on another GPU')
    for row in rows.values():
        if row['set']!='reference':continue
        name=row['id'];old=[]
        for item in select_attempts(previous,name,('plain',))[0]:
            old.append(replay(previous,item['record'],row)['elapsed_us'])
            for path in (previous/item['path'],previous/item['record']['raw']):hashes[str(path)]=common.sha(path)
        references[name]=dict(current=span(plain[name].values()),previous=span(old),
            median_change=statistics.median(plain[name].values())/statistics.median(old)-1)
    for filename in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/resources.json','build/sass_hashes.json'):
        path=root/filename;hashes[str(path)]=common.sha(path)
    origins_by_trial=[]
    for trial in range(10):
        members=[p for p in processes if p['trial']==trial]
        origins={p['case']:p['address_metadata']['b_buffer_ptr'] for p in members}
        origins_by_trial.append(dict(trial=trial,allocation_origins=origins,all_four_same=len(set(origins.values()))==1))
    common_origin_trials=[r['trial'] for r in origins_by_trial if r['all_four_same']]
    origins=sorted({p['address_metadata']['b_buffer_ptr'] for p in processes})
    placement_maps={tuple((r['cta'],r['sm']) for r in p['cta_sm']) for p in processes}
    address_comparison=dict(virtual_allocation_origins=origins,
        relative_to_minimum_origin_bytes=[v-origins[0] for v in origins],process_trial_origins=origins_by_trial,
        same_allocation_origin_trial_ids=common_origin_trials,
        same_origin_pattern_processes=[r for r in pattern_processes if r['trial'] in common_origin_trials],
        distinct_CTA_to_SM_maps=len(placement_maps),
        limitation='The full 40-process comparison remains primary. Common-VA-origin trials are separate diagnostics only; '
            'equal virtual origins across processes do not establish equal physical addresses.')
    consistent_interactions={}
    for label,subset in [('all',interactions),('first',[r for r in interactions if r['j']==0]),
                         ('later',[r for r in interactions if r['j']>0])]:
        consistent_interactions[label]=dict(positions=len(subset),
            positive_cycles_and_ns=sum(r['L_cycles_delta']['process_min']>0 and r['L_ns_delta']['process_min']>0 for r in subset),
            negative_cycles_and_ns=sum(r['L_cycles_delta']['process_max']<0 and r['L_ns_delta']['process_max']<0 for r in subset))
    report=dict(scope='B pitch-by-relative-origin factorial. Pitch responses are formed within each offset before comparing offsets; '
        'independent processes remain the repetition units, no new service fit.',
        environment=json.loads((root/'environment.json').read_text()),contexts=contexts,timing=timing,
        address_comparison=address_comparison,pattern_processes=pattern_processes,pattern_summaries=pattern_summaries,position_responses=position_summaries,
        same_position_interactions=interactions,consistent_interaction_signs=consistent_interactions,selected_CTA_processes=selected_values,
        selected_CTA_interactions=selected_interactions,process_placement_and_addresses=processes,references=references,
        interpretation=['The relative +256 B origin corresponds to +128 FP16 columns at fixed row pitch. '
            'Only (mi,ni) to (mi,ni+1) is matched; no column-tail wrap or M change.',
            'Adjacent panels can change CTA/ordinal/future work. Compare the matched j/T/active/A-peer/B-peer subset '
            'against identical-work mapping on exactly the same target subset.',
            'Same-SM mapping requires stable SMID inside each pitch pair and matches ordinal/T. '
            'If CTA-to-SM mapping stays fixed, CTA and SM explanations remain confounded.',
            'Matching relative offsets does not prove equal virtual allocation origins, physical addresses or bank selection.',
            'prefix=FIRST_MMA-entry, prefill=FIRST_MMA-producer_first_work; only prefill corresponds to model S. '
            'Both plus-L intervals close within each call; B0/B32 response differences still compare independent calls.',
            'A good pattern correlation identifies a correspondence, not a physical cache/queue/bank mechanism. '
            'No translation under +256 B only constrains this intervention, not all address dependencies.',
            'Consistent-sign position counts describe ten observed process pairs, not a multiple-testing correction or proof that every address effect is absent.'],
        input_sha256=hashes,analyzer_sha256=common.sha(Path(__file__)))
    output.mkdir(parents=True,exist_ok=True)
    (output/'R10-base-shift.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    common_fixed=next(x for x in pattern_summaries if x['mapping']=='same_work_on_neighbor_subset' and x['metric']=='L_cycles_delta')
    common_neighbor=next(x for x in pattern_summaries if x['mapping']=='next_B_panel_matched_context' and x['metric']=='L_cycles_delta')
    first_fixed=next(x for x in pattern_summaries if x['mapping']=='same_work_first_T3_common' and x['metric']=='L_cycles_delta')
    first_neighbor=next(x for x in pattern_summaries if x['mapping']=='next_B_first_T3_same_context' and x['metric']=='L_cycles_delta')
    disturbances=[v for offset in ('0','256') for v in timing[offset]['dual_disturbance'].values()]
    lines=['# B基址平移与行距响应','',
        '每个offset先作B32−B0，再比较offset256−offset0。所有逐位置结果先形成十个进程内的配对差，再跨进程汇总；不拟合新服务价格。','',
        f'相同270个目标窗口内，固定工作位置的周期响应相关中位为{common_fixed["correlation"]["median"]:.3f}、'
        f'RMS差{common_fixed["root_mean_square_residual"]["median"]:.1f}；相邻B面板对应为'
        f'{common_neighbor["correlation"]["median"]:.3f}、{common_neighbor["root_mean_square_residual"]["median"]:.1f}。'
        f'只看首轮T3的72窗口，相关仍分别为{first_fixed["correlation"]["median"]:.3f}/'
        f'{first_neighbor["correlation"]["median"]:.3f}，不是单靠首末轮平均差得到这种对应。'
        '本次+256 B干预没有表现为响应整体平移到相邻panel，但不能因此排除其它地址依赖或认定SM机制。','',
        f'plain中的B32减速，offset0/+256分别为{100*timing["0"]["plain_pitch_change"]["median"]:.2f}%/'
        f'{100*timing["256"]["plain_pitch_change"]["median"]:.2f}%。dual/plain中位范围'
        f'{100*min(v["median"] for v in disturbances):.2f}%～{100*max(v["median"] for v in disturbances):.2f}%，'
        f'单对最大{100*max(v["process_max"] for v in disturbances):.2f}%；局部数值属于这个观察协议，不直接定硬件价格。','',
        f'全部432个位置中，{consistent_interactions["all"]["positive_cycles_and_ns"]}个位置在十次比较里周期和ns交互均为正，'
        f'{consistent_interactions["all"]["negative_cycles_and_ns"]}个均为负；这是观察到的同号计数，不是多重检验后的显著性结论。'
        '因此不能把未发生整体平移写成基址条件完全不影响局部窗口。','',
        '| 对应关系 | 窗口数 | ΔL周期响应差的绝对中位 | ΔL周期响应差RMS | 响应相关系数 |',
        '|---|---:|---:|---:|---:|']
    for summary in pattern_summaries:
        if summary['metric']!='L_cycles_delta':continue
        med=summary['median_absolute_residual'];rms=summary['root_mean_square_residual'];corr=summary['correlation']
        if med is None:
            lines.append(f'| {summary["mapping"]} | 0 | 无唯一对应 | 无唯一对应 | 未定义 |');continue
        correlation=f'{corr["median"]:.3f} [{corr["process_min"]:.3f},{corr["process_max"]:.3f}]' if corr else '未定义'
        lines.append(f'| {summary["mapping"]} | {summary["matched_window_counts"]} | {med["median"]:.1f} | {rms["median"]:.1f} | {correlation} |')
    lines+=['','邻panel只在同一M比较ni+1，末列不绕回。matched-context与same-work-on-neighbor-subset使用完全相同目标窗口，'
        '防止仅靠选择不同窗口得到较好对应。JSON保留每个位置的周期/ns响应、SM分布、首段闭合和完整调用扰动。',
        '','| offset | CTA | ΔL cycles中位/范围 | Δprefix ns中位 | Δprefill ns中位 | Δ(prefix+L) ns中位 |','|---:|---:|---:|---:|---:|---:|']
    for r in position_summaries:
        if r['cta'] not in (55,91) or r['j']!=0:continue
        d=r['L_cycles_delta']
        lines.append(f'| {r["offset"]} | {r["cta"]} | {d["median"]:.1f} [{d["process_min"]:.1f},{d["process_max"]:.1f}] | '
            f'{r["prefix_ns_delta"]["median"]:.1f} | {r["prefill_ns_delta"]["median"]:.1f} | {r["prefix_plus_L_ns_delta"]["median"]:.1f} |')
    for trial in common_origin_trials:
        common=[r for r in pattern_processes if r['trial']==trial and r['metric']=='L_cycles_delta']
        a=next(r for r in common if r['mapping']=='same_work_on_neighbor_subset')
        b=next(r for r in common if r['mapping']=='next_B_panel_matched_context')
        lines+=['',f'四条件同VA原点的trial{trial}中，同270窗口的固定位置/邻panel RMS差为'
            f'{a["root_mean_square_residual"]:.1f}/{b["root_mean_square_residual"]:.1f} cycles。'
            '这是单次四进程组的另列诊断，不替代全批统计，也不证明同PA。']
    lines+=['',f'四十个dual进程出现{len(origins)}个VA分配原点，相对最小值的间隔为'
        f'{address_comparison["relative_to_minimum_origin_bytes"]} B；四条件同trial原点全相同的只有{common_origin_trials}。'
        '该子集结果另列于JSON，不替代全部进程比较，也不证明同PA。',
        '',f'全部进程只有{len(placement_maps)}种CTA→SM映射，因此同CTA与同SM对应无法分离。',
        '','基址只声明相对分配原点偏移；VA原点与PA对应未证明时，不解释为物理bank平移。'
        'CTA和SM映射若固定，也不能凭相同逻辑位置的响应直接区分软件位置与硬件派发。','']
    lines+=['[响应空间图](R10-base-shift-response.png)使用同一JSON，第三幅为先配对再取中位的offset交互，不直接相减两幅中位。','']
    (output/'R10-base-shift.md').write_text('\n'.join(lines))
    return report

def _work_swap_expected(work, enabled):
    """Swap the complete persistent sequences, leaving physical grid indices intact."""
    result=[list(c) for c in work]
    if enabled:
        result[55],result[91]=result[91],result[55]
    return result


def _work_swap_pitch_windows(base, variant, exclude_ctas=()):
    """Same physical CTA pitch pair, with normalized globaltimer endpoints."""
    from analyze_r13_supply import _failure_pair_windows
    windows=_failure_pair_windows([c for c in base['ctas'] if c['cta'] not in exclude_ctas],
        [c for c in variant['ctas'] if c['cta'] not in exclude_ctas])
    maps={label:{c['cta']:c for c in obs['ctas']} for label,obs in (('base',base),('variant',variant))}
    origins={label:min(c['entry_ns'] for c in obs['ctas']) for label,obs in (('base',base),('variant',variant))}
    for w in windows:
        endpoints={}
        for label in ('base','variant'):
            c=maps[label][w['cta']];j=w['j'];cycles,ns=c['tiles'][j],c['tiles_ns'][j]
            values=dict(entry_relative_ns=c['entry_ns']-origins[label],
                producer_relative_ns=c['prod_ns']-origins[label],entry_to_producer_ns=c['prod_ns']-c['entry_ns'],
                FIRST_relative_ns=ns[0]-origins[label],MAIN_END_relative_ns=ns[1]-origins[label],
                entry_to_FIRST_ns=ns[0]-c['entry_ns'],entry_to_MAIN_END_ns=ns[1]-c['entry_ns'],
                entry_to_FIRST_cycles=cycles[0]-c['entry_c'],entry_to_MAIN_END_cycles=cycles[1]-c['entry_c'])
            if values['entry_to_FIRST_ns']+w['L_ns_'+label]!=values['entry_to_MAIN_END_ns']:
                raise ValueError('work-swap entry/FIRST/L endpoints do not close')
            if values['entry_to_FIRST_cycles']+w['L_cycles_'+label]!=values['entry_to_MAIN_END_cycles']:
                raise ValueError('work-swap local cycle endpoints do not close')
            endpoints[label]=values;w[label+'_sm']=c['sm']
        for field in endpoints['base']:
            w[field+'_base']=endpoints['base'][field];w[field+'_variant']=endpoints['variant'][field]
            w[field+'_delta']=endpoints['variant'][field]-endpoints['base'][field]
    return windows


def _work_swap_correspondence(reference, swapped):
    physical={(r['cta'],r['j'],r['T']):r for r in reference}
    logical={(r['mi'],r['ni']):r for r in reference}
    return dict(physical_CTA=[(physical[r['cta'],r['j'],r['T']],r) for r in swapped],
        logical_work=[(logical[r['mi'],r['ni']],r) for r in swapped])


def work_swap_diagnostics(root, output):
    """Whole-sequence 55/91 intervention: physical position versus logical work."""
    from collections import defaultdict
    from analyze_r13_sm import select_attempts
    from analyze_r18 import replay
    from analyze_r13_supply import _failure_pair_windows
    import v08_model
    import v06_run as common

    root,output=Path(root),Path(output);common.verify(root)
    rows={r['id']:r for r in json.loads((root/'cases.json').read_text())}
    setups={r['case']:r['setup'] for r in json.loads((root/'static_setup.json').read_text())}
    controls=[r for r in rows.values() if r['set']=='ctrl'];ignored={'id','group','ldb','work_swap55_91'}
    if len(controls)!=4 or any({k:v for k,v in r.items() if k not in ignored}!=
        {k:v for k,v in controls[0].items() if k not in ignored} for r in controls):
        raise ValueError('work-swap factorial changed another control condition')
    if {(r['work_swap55_91'],r['ldb']) for r in controls}!={(s,p) for s in (0,1) for p in (3072,3088)}:
        raise ValueError('work-swap factors incomplete')
    def span(values):
        values=list(values)
        return dict(median=statistics.median(values),process_min=min(values),process_max=max(values))
    calls={};plain={};contexts={};processes=[];hashes={};failures={};original_work=None
    for row in rows.values():
        name=row['id'];calls[name]={};plain[name]={}
        if row['set']=='ctrl':
            setup=setups[name]
            if setup.get('work_swap55_91')!=row['work_swap55_91']:raise ValueError('actual work-swap flag changed')
            ordinary=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row['raster_order'])
            if original_work is None:original_work=ordinary
            if ordinary!=original_work:raise ValueError('work-swap base scheduling changed')
            work=_work_swap_expected(ordinary,row['work_swap55_91'])
            flattened=[tuple(p) for c in work for p in c]
            if len(flattened)!=len(set(flattened)) or set(flattened)!=set(tuple(p) for c in ordinary for p in c):
                raise ValueError('work-swap omitted or repeated output coordinates')
            for j in range(max(map(len,work))):
                if sorted(tuple(c[j]) for c in work if len(c)>j)!=sorted(tuple(c[j]) for c in ordinary if len(c)>j):
                    raise ValueError('work-swap changed a per-wave source workset')
            contexts[name]=dict(row=row,setup=setup,selected_sequences={str(c):work[c] for c in (55,91)},
                complete_work_sha256=hashlib.sha256(json.dumps(work).encode()).hexdigest(),
                wave_worksets_unchanged=True,unmodified_CTA_count=sum(a==b for a,b in zip(work,ordinary)))
        selected,failed,_=select_attempts(root,name,tuple(row['variants']));failures[name]=[i['path'] for i in failed]
        for item in selected:
            record=item['record'];obs=replay(root,record,row)
            for path in (root/item['path'],root/record['raw']):hashes[str(path)]=common.sha(path)
            if record['variant']=='plain':plain[name][record['trial']]=obs['elapsed_us']
            else:
                if [c['work'] for c in obs['ctas']]!=work:raise ValueError('actual CTA coordinates do not match whole-sequence swap')
                calls[name][record['trial']]=obs
                processes.append(dict(case=name,trial=record['trial'],elapsed_us=obs['elapsed_us'],
                    cta_sm=[dict(cta=c['cta'],sm=c['sm']) for c in obs['ctas']],
                    selected_work_and_SM=[dict(cta=c['cta'],sm=c['sm'],work=c['work']) for c in obs['ctas'] if c['cta'] in (55,91)],
                    address_metadata=obs.get('b_address')))
        if len(plain[name])!=10 or (row['set']=='ctrl' and calls[name].keys()!=plain[name].keys()):
            raise ValueError('ten complete work-swap processes required per variant')
    if len({setups[r['id']]['gpu_uuid'] for r in controls})!=1:raise ValueError('work-swap conditions span different cards')
    responses={};timing={};position_values=defaultdict(list);selected_windows=[]
    fields=('L_cycles_delta','L_ns_delta','L_rate_ratio','entry_relative_ns_delta','producer_relative_ns_delta',
        'entry_to_producer_ns_delta','FIRST_relative_ns_delta',
        'MAIN_END_relative_ns_delta','entry_to_FIRST_ns_delta','entry_to_MAIN_END_ns_delta',
        'entry_to_FIRST_cycles_delta','entry_to_MAIN_END_cycles_delta','prefill_ns_delta')
    for swap in (0,1):
        a,b=(f'r10_workswap_s{swap}_p{pitch}' for pitch in (0,32));responses[swap]={}
        for trial in sorted(calls[a]):
            windows=_work_swap_pitch_windows(calls[a][trial],calls[b][trial]);responses[swap][trial]=windows
            for w in windows:
                position_values[swap,w['cta'],w['j']].append(dict(trial=trial,**w))
                if w['cta'] in (55,91):selected_windows.append(dict(swap=swap,trial=trial,**w))
        timing[str(swap)]=dict(plain_pitch_ratio=span(plain[b][t]/plain[a][t]-1 for t in plain[a]),
            plain_pitch_delta_us=span(plain[b][t]-plain[a][t] for t in plain[a]),
            dual_disturbance={name:span(calls[name][t]['elapsed_us']/plain[name][t]-1 for t in plain[name]) for name in (a,b)})
    timing['plain_swap_effect']={str(p):dict(relative=span(plain[f'r10_workswap_s1_p{p}'][t]/plain[f'r10_workswap_s0_p{p}'][t]-1 for t in range(10)),
        delta_us=span(plain[f'r10_workswap_s1_p{p}'][t]-plain[f'r10_workswap_s0_p{p}'][t] for t in range(10))) for p in (0,32)}

    correspondence_processes=[];correspondence_position_values=defaultdict(list);unmodified_processes=[]
    metrics=('L_cycles_delta','L_ns_delta','entry_to_MAIN_END_ns_delta','entry_to_FIRST_ns_delta')
    for trial in range(10):
        paired=_work_swap_correspondence(responses[0][trial],responses[1][trial])
        for mapping,pairs in paired.items():
            changed=[(a,b) for a,b in pairs if b['cta'] in (55,91)]
            for field in metrics:
                for selection,subset in [('first',[p for p in changed if p[1]['j']==0]),('all_three',changed)]:
                    correspondence_processes.append(dict(trial=trial,mapping=mapping,selection=selection,metric=field,
                        **_base_shift_pattern_metrics(subset,field)))
            for a,b in changed:
                record=dict(trial=trial,reference_cta=a['cta'],swapped_cta=b['cta'],j=b['j'],T=b['T'],
                    reference_coord=[a['mi'],a['ni']],swapped_coord=[b['mi'],b['ni']],
                    reference_base_sm=a['base_sm'],reference_variant_sm=a['variant_sm'],
                    swapped_base_sm=b['base_sm'],swapped_variant_sm=b['variant_sm'])
                for field in fields:
                    if field in a:
                        record[field+'_reference']=a[field];record[field+'_swapped']=b[field];record[field+'_residual']=b[field]-a[field]
                correspondence_position_values[mapping,b['cta'],b['j']].append(record)
        unchanged=[(a,b) for a,b in paired['physical_CTA'] if b['cta'] not in (55,91)]
        unmodified_processes.append(dict(trial=trial,kind='pitch_response_interaction',windows=len(unchanged),
            **{field:_base_shift_pattern_metrics(unchanged,field) for field in metrics}))
        for pitch in (0,32):
            a,b=(calls[f'r10_workswap_s{s}_p{pitch}'][trial] for s in (0,1))
            changed=_work_swap_pitch_windows(a,b,exclude_ctas=(55,91))
            first=[w for w in changed if w['j']==0]
            unmodified_processes.append(dict(trial=trial,kind=f'direct_swap_pitch{pitch}',windows=len(changed),
                first_output={field:dict(median=statistics.median(w[field] for w in first),
                    maximum_absolute=max(abs(w[field]) for w in first)) for field in
                    ('entry_relative_ns_delta','producer_relative_ns_delta','FIRST_relative_ns_delta',
                     'L_cycles_delta','L_ns_delta','entry_to_MAIN_END_ns_delta','entry_to_FIRST_ns_delta','prefill_ns_delta')},
                **{field:dict(median=statistics.median(w[field] for w in changed),maximum_absolute=max(abs(w[field]) for w in changed))
                    for field in ('L_cycles_delta','L_ns_delta','L_cycles_ratio','L_ns_ratio')}))
    position_summaries=[]
    for (swap,cta,j),values in sorted(position_values.items()):
        w=values[0]
        position_summaries.append(dict(swap=swap,cta=cta,j=j,T=w['T'],mi=w['mi'],ni=w['ni'],
            base_sms=sorted({r['base_sm'] for r in values}),variant_sms=sorted({r['variant_sm'] for r in values}),
            **{field:span(r[field] for r in values) for field in fields if field in w}))
    correspondence_positions=[]
    for (mapping,cta,j),values in sorted(correspondence_position_values.items()):
        w=values[0]
        correspondence_positions.append(dict(mapping=mapping,swapped_cta=cta,j=j,reference_cta=w['reference_cta'],
            reference_coord=w['reference_coord'],swapped_coord=w['swapped_coord'],processes=values,
            **{field:span(r[field] for r in values) for field in w if field.endswith(('_reference','_swapped','_residual'))}))
    correspondence_summaries=[]
    for mapping,selection,metric in sorted({(r['mapping'],r['selection'],r['metric']) for r in correspondence_processes}):
        values=[r for r in correspondence_processes if (r['mapping'],r['selection'],r['metric'])==(mapping,selection,metric)]
        correspondence_summaries.append(dict(mapping=mapping,selection=selection,metric=metric,
            **{field:span(r[field] for r in values) for field in
                ('median_signed_residual','median_absolute_residual','root_mean_square_residual','maximum_absolute_residual')}))
    unmodified_summary={}
    for pitch in (0,32):
        values=[r for r in unmodified_processes if r['kind']==f'direct_swap_pitch{pitch}']
        unmodified_summary[str(pitch)]=dict(first_output={field:dict(
            process_medians=span(r['first_output'][field]['median'] for r in values),
            process_maximum_absolute=span(r['first_output'][field]['maximum_absolute'] for r in values))
            for field in values[0]['first_output']},
            all_outputs={field:span(r[field]['median'] for r in values)
                for field in ('L_cycles_delta','L_ns_delta','L_cycles_ratio','L_ns_ratio')})
    address_origins=[dict(trial=t,origins={r['case']:r['address_metadata']['b_buffer_ptr'] for r in processes if r['trial']==t}) for t in range(10)]
    address_summary=dict(virtual_origins=sorted({p['address_metadata']['b_buffer_ptr'] for p in processes}),
        per_trial=address_origins,same_origin_trials=[r['trial'] for r in address_origins if len(set(r['origins'].values()))==1],
        distinct_CTA_to_SM_maps=len({tuple((r['cta'],r['sm']) for r in p['cta_sm']) for p in processes}))
    references={};previous=root.parent/'20261010-R10-base-shift'
    if json.loads((previous/'environment.json').read_text())['gpu']!=json.loads((root/'environment.json').read_text())['gpu']:
        raise ValueError('work-swap references are on another card')
    for row in rows.values():
        if row['set']!='reference':continue
        name=row['id'];old=[]
        for item in select_attempts(previous,name,('plain',))[0]:
            old.append(replay(previous,item['record'],row)['elapsed_us'])
            for path in (previous/item['path'],previous/item['record']['raw']):hashes[str(path)]=common.sha(path)
        references[name]=dict(current=span(plain[name].values()),previous=span(old),
            median_change=statistics.median(plain[name].values())/statistics.median(old)-1)
    for filename in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/resources.json','build/sass_hashes.json'):
        path=root/filename;hashes[str(path)]=common.sha(path)
    for path in root.glob('correctness-*.jsonl'):hashes[str(path)]=common.sha(path)
    report=dict(scope='Physical CTA55/91 exchange complete logical work sequences. Within-mode pitch responses precede '
        'cross-mode comparisons; no new service price or hardware assignment is inferred from correlation.',
        environment=json.loads((root/'environment.json').read_text()),contexts=contexts,timing=timing,
        source_boundary=dict(same_binary_different_dynamic_path=True,
            plain_GEMM_swap_gate='0xa80 @!P0 BRA 0xb40 skips 0xa90..0xb30 in swap0; swap1 executes the 55/91 comparisons for all CTAs.',
            resources='Same grid [132,1,1] and 215040 B SMEM; resource and SASS files are hashed below.',
            remaining_confounders='Physical CTA, fixed SM, dispatch/arrival organization and the dual observer remain coupled.'),
        position_responses=position_summaries,selected_CTA_processes=selected_windows,
        correspondence_processes=correspondence_processes,correspondence_positions=correspondence_positions,
        correspondence_summaries=correspondence_summaries,unmodified_CTA_context=unmodified_processes,
        unmodified_CTA_summary=unmodified_summary,address_and_placement_summary=address_summary,
        process_placement_and_addresses=processes,references=references,failed_attempts=failures,
        interpretation=['Every observed CTA sequence must match the declared whole-sequence exchange. No missing/repeated outputs; '
            'each software-wave workset is unchanged.',
            'Only the six affected CTA/output windows judge physical-versus-logical correspondence. '
            'The unchanged 130 CTAs are context, not pooled to inflate the match.',
            'entry/FIRST/MAIN_END globaltimer endpoints are relative to each call\'s first CTA entry. '
            'entry-to-FIRST plus L closes to entry-to-MAIN_END within each call; local cycle differences never cross SMs.',
            'prefill is FIRST_MMA minus producer first-work only for first output; entry-to-FIRST is not model S.',
            'Following logical work exchanges B panels and D destinations together; it does not isolate B-address banking. '
            'Following physical CTA cannot identify an SM resource unless placement, temporal and observer alternatives are separately ruled out.',
            'Swap1 executes extra dynamic comparisons for all CTAs even in the same binary. The other 130 CTAs supply an internal '
            'control for entry/producer/FIRST/L/end movement and do not justify assuming zero observer or front-end effect.',
            'Pitch and swap comparisons span independent processes. VA metadata does not establish shared physical pages.'],
        input_sha256=hashes,analyzer_sha256=common.sha(Path(__file__)))
    output.mkdir(parents=True,exist_ok=True)
    (output/'R10-work-swap.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 持久工作序列55/91交换','',
        '先在每个swap模式内按物理CTA/j/T/坐标配对B32−B0，再分别与原物理位置、原逻辑工作序列对照。'
        '只有两个受干预CTA参与主要对应比较，另130个CTA单列为上下文。','',
        '| swap | 物理CTA/j | 逻辑坐标 | B0/B32 SM | ΔL cycles | ΔL ns | Δentry→FIRST ns | Δentry→MAIN_END ns |',
        '|---:|---|---|---|---:|---:|---:|---:|']
    for r in position_summaries:
        if r['cta'] not in (55,91):continue
        values=[]
        for field in ('L_cycles_delta','L_ns_delta','entry_to_FIRST_ns_delta','entry_to_MAIN_END_ns_delta'):
            v=r[field];values.append(f'{v["median"]:.1f} [{v["process_min"]:.1f},{v["process_max"]:.1f}]')
        lines.append(f'| {r["swap"]} | {r["cta"]}/{r["j"]} | ({r["mi"]},{r["ni"]}) | {r["base_sms"]}/{r["variant_sms"]} | '+' | '.join(values)+' |')
    lines+=['','括号是十进程观测范围，不是置信区间。entry与FIRST绝对位置按各自调用的最早CTA入口归一化；'
        '同CTA时长直接相减，不把不同SM的clock64拼成一条时轴。','',
        '| 对应方式 | 选择窗口 | 指标 | 响应差绝对中位 | 响应差RMS |','|---|---|---|---:|---:|']
    for r in correspondence_summaries:
        if r['metric'] not in ('L_cycles_delta','L_ns_delta','entry_to_MAIN_END_ns_delta'):continue
        lines.append(f'| {r["mapping"]} | {r["selection"]} | {r["metric"]} | '
            f'{r["median_absolute_residual"]["median"]:.1f} | {r["root_mean_square_residual"]["median"]:.1f} |')
    lines+=['','| 未交换130CTA首轮，swap1−swap0 | Δentry相对调用入口 ns | Δproducer ns | ΔFIRST ns | ΔL ns | Δentry→MAIN_END ns |',
        '|---|---:|---:|---:|---:|---:|']
    for pitch,summary in unmodified_summary.items():
        cells=[]
        for field in ('entry_relative_ns_delta','producer_relative_ns_delta','FIRST_relative_ns_delta','L_ns_delta','entry_to_MAIN_END_ns_delta'):
            v=summary['first_output'][field]['process_medians'];cells.append(f'{v["median"]:.1f} [{v["process_min"]:.1f},{v["process_max"]:.1f}]')
        lines.append(f'| B+{pitch} B | '+' | '.join(cells)+' |')
    lines+=['','swap1虽使用同binary，动态路径不同：plain GEMM的0xa80分支在swap0跳过0xa90～0xb30，swap1执行其中55/91比较，130个未交换CTA也经过。'
        '这些内部对照保留共同前端/入口变化，不能默认干预只影响两CTA。',
        '',f'四十dual进程有{len(address_summary["virtual_origins"])}种VA原点、'
        f'{address_summary["distinct_CTA_to_SM_maps"]}种CTA→SM映射；四条件同trial原点相同的序号为'
        f'{address_summary["same_origin_trials"]}。跨进程同VA也不证明同PA。',
        '','实际每CTA坐标、SM、无重复/遗漏与逐轮工作集合均检查；逐进程端点、130CTA上下文和完整plain/dual扰动保存在JSON。'
        '逻辑工作同时包含B源与D目的位置；即使响应随工作移动，也不能单独归因为B地址bank。','']
    (output/'R10-work-swap.md').write_text('\n'.join(lines))
    return report

def _producer_group_rank(ctas, target=55):
    """Software producer/FIRST markers in the same first-output B group; ties stay ties."""
    current=next(c for c in ctas if c['cta']==target);ni=current['work'][0][1]
    peers=[c for c in ctas if c['work'][0][1]==ni]
    result=dict(ni=ni,ctas=sorted(c['cta'] for c in peers))
    for name,get in [('entry',lambda c:c['entry_ns']),('producer',lambda c:c['prod_ns']),
                     ('FIRST',lambda c:c['tiles_ns'][0][0])]:
        value=get(current);others=[get(c) for c in peers if c['cta']!=target]
        result[name]=dict(rank_min=1+sum(get(c)<value for c in peers),rank_max=sum(get(c)<=value for c in peers),
            target_minus_peer_median_ns=value-statistics.median(others))
    return result


def producer_delay_diagnostics(root, output):
    """One CTA55 producer delay, with pitch interactions and same-call boundaries."""
    from collections import defaultdict
    from analyze_r13_sm import select_attempts
    from analyze_r18 import replay
    import v08_model
    import v06_run as common

    root,output=Path(root),Path(output);common.verify(root)
    rows={r['id']:r for r in json.loads((root/'cases.json').read_text())}
    setups={r['case']:r['setup'] for r in json.loads((root/'static_setup.json').read_text())}
    controls=[r for r in rows.values() if r['set']=='ctrl'];ignored={'id','group','ldb','producer_delay_ns'}
    if len(controls)!=4 or any({k:v for k,v in r.items() if k not in ignored}!=
        {k:v for k,v in controls[0].items() if k not in ignored} for r in controls):
        raise ValueError('producer delay changed another factorial condition')
    if {(r['producer_delay_ns'],r['ldb']) for r in controls}!={(d,p) for d in (0,512) for p in (3072,3088)}:
        raise ValueError('producer-delay factors incomplete')
    def span(v):
        v=list(v);return dict(median=statistics.median(v),process_min=min(v),process_max=max(v))
    calls={};plain={};processes=[];hashes={};contexts={};failures={};work=None
    for row in rows.values():
        name=row['id'];calls[name]={};plain[name]={}
        if row['set']=='ctrl':
            setup=setups[name]
            if setup.get('producer_delay_ns')!=row['producer_delay_ns']:raise ValueError('producer delay request changed')
            current=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row['raster_order'])
            if work is None:work=current
            if current!=work:raise ValueError('delay changed software work')
            contexts[name]=dict(row=row,setup=setup,work_sha256=hashlib.sha256(json.dumps(work).encode()).hexdigest())
        selected,failed,_=select_attempts(root,name,tuple(row['variants']));failures[name]=[r['path'] for r in failed]
        for item in selected:
            record=item['record'];obs=replay(root,record,row)
            for path in (root/item['path'],root/record['raw']):hashes[str(path)]=common.sha(path)
            if record['variant']=='plain':plain[name][record['trial']]=obs['elapsed_us'];continue
            if [c['work'] for c in obs['ctas']]!=work:raise ValueError('producer delay changed actual work mapping')
            calls[name][record['trial']]=obs;delay=obs['producer_delay']
            c=next(c for c in obs['ctas'] if c['cta']==55)
            if delay['cta']!=55 or delay['requested_ns']!=row['producer_delay_ns']:
                raise ValueError('delay trace identifies another CTA or request')
            ns=c['tiles_ns'][0];cycles=c['tiles'][0]
            post_gate_to_prod=c['prod_ns']-delay['end_ns'];S=ns[0]-c['prod_ns'];L=ns[1]-ns[0]
            if post_gate_to_prod<0 or post_gate_to_prod+S+L!=ns[1]-delay['end_ns']:
                raise ValueError('producer delay / prod / FIRST / MAIN_END do not close')
            processes.append(dict(case=name,trial=record['trial'],requested_ns=row['producer_delay_ns'],
                pitch_bytes=row['ldb']*2-6144,actual_delay=delay,elapsed_us=obs['elapsed_us'],
                target_sm=c['sm'],target_work=c['work'],delay_begin_from_entry_ns=delay['begin_ns']-c['entry_ns'],
                end_to_prod_ns=post_gate_to_prod,P0_ns=c['prod_ns']-c['entry_ns'],S_ns=S,L_ns=L,
                P0_cycles=c['prod_c']-c['entry_c'],S_cycles=cycles[0]-c['prod_c'],
                S_plus_L_cycles=cycles[1]-c['prod_c'],entry_to_MAIN_END_cycles=cycles[1]-c['entry_c'],
                S_plus_L_ns=S+L,entry_to_MAIN_END_ns=ns[1]-c['entry_ns'],end_to_FIRST_ns=ns[0]-delay['end_ns'],
                L_cycles=cycles[1]-cycles[0],source_group=_producer_group_rank(obs['ctas']),
                cta_sm=[dict(cta=z['cta'],sm=z['sm']) for z in obs['ctas']],address_metadata=obs.get('b_address')))
        if len(plain[name])!=10 or (row['set']=='ctrl' and calls[name].keys()!=plain[name].keys()):
            raise ValueError('ten complete producer-delay processes required')
    if len({setups[r['id']]['gpu_uuid'] for r in controls})!=1:raise ValueError('producer-delay conditions span cards')
    responses={};timing={};position_values=defaultdict(list);selected=[]
    fields=('L_cycles_delta','L_ns_delta','L_rate_ratio','entry_relative_ns_delta','producer_relative_ns_delta',
        'FIRST_relative_ns_delta','entry_to_producer_ns_delta','entry_to_FIRST_ns_delta','entry_to_MAIN_END_ns_delta',
        'prefill_ns_delta','prefill_plus_L_ns_delta')
    for delay in (0,512):
        a,b=(f'r10_proddelay_d{delay}_p{pitch}' for pitch in (0,32));responses[delay]={}
        for trial in range(10):
            windows=_work_swap_pitch_windows(calls[a][trial],calls[b][trial])
            for w in windows:
                if w['j']==0:w['prefill_plus_L_ns_delta']=w['prefill_ns_delta']+w['L_ns_delta']
                position_values[delay,w['cta'],w['j']].append(dict(trial=trial,**w))
                if w['cta'] in (55,91):selected.append(dict(delay=delay,trial=trial,**w))
            responses[delay][trial]=windows
        timing[str(delay)]=dict(plain_pitch_ratio=span(plain[b][t]/plain[a][t]-1 for t in range(10)),
            plain_pitch_delta_us=span(plain[b][t]-plain[a][t] for t in range(10)),
            dual_disturbance={name:span(calls[name][t]['elapsed_us']/plain[name][t]-1 for t in range(10)) for name in (a,b)})
    timing['plain_delay_effect']={str(p):dict(relative=span(plain[f'r10_proddelay_d512_p{p}'][t]/plain[f'r10_proddelay_d0_p{p}'][t]-1 for t in range(10)),
        delta_us=span(plain[f'r10_proddelay_d512_p{p}'][t]-plain[f'r10_proddelay_d0_p{p}'][t] for t in range(10))) for p in (0,32)}
    interactions=defaultdict(list);direct=[];background=[]
    first_B=work[55][0][1]
    for trial in range(10):
        reference={(w['cta'],w['j']):w for w in responses[0][trial]}
        for w in responses[512][trial]:
            a=reference[w['cta'],w['j']]
            interactions[w['cta'],w['j']].append(dict(trial=trial,**{k:w[k]-a[k] for k in fields if k in w}))
        for pitch in (0,32):
            a,b=(calls[f'r10_proddelay_d{d}_p{pitch}'][trial] for d in (0,512))
            windows=_work_swap_pitch_windows(a,b)
            for w in windows:
                if w['j']==0:w['prefill_plus_L_ns_delta']=w['prefill_ns_delta']+w['L_ns_delta']
                if w['cta'] in (55,91):direct.append(dict(pitch_bytes=pitch,trial=trial,**w))
            first=[w for w in windows if w['j']==0]
            for label,subset in [('same_B_partners',[w for w in first if w['ni']==first_B and w['cta']!=55]),
                                 ('other130',[w for w in first if w['cta'] not in (55,91)])]:
                background.append(dict(pitch_bytes=pitch,trial=trial,group=label,CTAs=len(subset),
                    **{field:statistics.median(w[field] for w in subset) for field in fields}))
    position_summaries=[]
    for (delay,cta,j),values in sorted(position_values.items()):
        w=values[0]
        position_summaries.append(dict(delay=delay,cta=cta,j=j,T=w['T'],coord=[w['mi'],w['ni']],
            base_sms=sorted({r['base_sm'] for r in values}),variant_sms=sorted({r['variant_sm'] for r in values}),
            **{k:span(r[k] for r in values) for k in fields if k in w}))
    interaction_summaries=[dict(cta=cta,j=j,**{k:span(r[k] for r in values) for k in values[0] if k!='trial'})
        for (cta,j),values in sorted(interactions.items())]
    direct_summaries=[]
    for pitch,cta,j in sorted({(r['pitch_bytes'],r['cta'],r['j']) for r in direct}):
        values=[r for r in direct if (r['pitch_bytes'],r['cta'],r['j'])==(pitch,cta,j)]
        direct_summaries.append(dict(pitch_bytes=pitch,cta=cta,j=j,
            **{k:span(r[k] for r in values) for k in fields if k in values[0]}))
    background_summaries=[]
    for pitch,group in sorted({(r['pitch_bytes'],r['group']) for r in background}):
        values=[r for r in background if (r['pitch_bytes'],r['group'])==(pitch,group)]
        background_summaries.append(dict(pitch_bytes=pitch,group=group,CTAs=values[0]['CTAs'],
            **{k:span(r[k] for r in values) for k in fields}))
    case_summaries=[]
    for name in contexts:
        values=[p for p in processes if p['case']==name]
        case_summaries.append(dict(case=name,requested_ns=values[0]['requested_ns'],pitch_bytes=values[0]['pitch_bytes'],
            actual_delay_ns=span(p['actual_delay']['duration_ns'] for p in values),
            actual_delay_cycles=span(p['actual_delay']['duration_cycles'] for p in values),
            target_sms=sorted({p['target_sm'] for p in values}),
            boundaries={k:span(p[k] for p in values) for k in
                ('delay_begin_from_entry_ns','end_to_prod_ns','P0_ns','S_ns','L_ns','S_plus_L_ns','entry_to_MAIN_END_ns','end_to_FIRST_ns','L_cycles','P0_cycles','S_cycles','S_plus_L_cycles','entry_to_MAIN_END_cycles')},
            ranks={event:{k:span(p['source_group'][event][k] for p in values) for k in
                ('rank_min','rank_max','target_minus_peer_median_ns')} for event in ('entry','producer','FIRST')}))
    gate_processes=[];gate_summaries=[]
    for pitch in (0,32):
        for trial in range(10):
            a=next(p for p in processes if p['requested_ns']==0 and p['pitch_bytes']==pitch and p['trial']==trial)
            b=next(p for p in processes if p['requested_ns']==512 and p['pitch_bytes']==pitch and p['trial']==trial)
            gate_processes.append(dict(pitch_bytes=pitch,trial=trial,
                duration_ns_delta=b['actual_delay']['duration_ns']-a['actual_delay']['duration_ns'],
                duration_cycles_delta=b['actual_delay']['duration_cycles']-a['actual_delay']['duration_cycles'],
                **{k+'_delta':b[k]-a[k] for k in ('P0_ns','end_to_prod_ns','S_ns','L_ns','S_plus_L_ns','entry_to_MAIN_END_ns','P0_cycles','S_cycles','S_plus_L_cycles','entry_to_MAIN_END_cycles')}))
        values=[p for p in gate_processes if p['pitch_bytes']==pitch]
        gate_summaries.append(dict(pitch_bytes=pitch,**{k:span(p[k] for p in values) for k in values[0] if k not in ('trial','pitch_bytes')}))
    references={};previous=root.parent/'20261010-R10-work-swap'
    if json.loads((previous/'environment.json').read_text())['gpu']!=json.loads((root/'environment.json').read_text())['gpu']:
        raise ValueError('producer-delay reference changed card')
    for row in rows.values():
        if row['set']!='reference':continue
        name=row['id'];old=[]
        for item in select_attempts(previous,name,('plain',))[0]:
            old.append(replay(previous,item['record'],row)['elapsed_us'])
            for path in (previous/item['path'],previous/item['record']['raw']):hashes[str(path)]=common.sha(path)
        references[name]=dict(current=span(plain[name].values()),previous=span(old),median_change=statistics.median(plain[name].values())/statistics.median(old)-1)
    for filename in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/resources.json','build/sass_hashes.json'):
        path=root/filename;hashes[str(path)]=common.sha(path)
    for path in sorted(root.glob('correctness-*.jsonl')):
        hashes[str(path)]=common.sha(path)
    origins=[dict(trial=t,by_case={p['case']:p['address_metadata']['b_buffer_ptr'] for p in processes if p['trial']==t}) for t in range(10)]
    report=dict(scope='CTA55 producer pre-first-load delay, no work or service-coefficient changes. '
        'Within-delay pitch pairs precede the delay-by-pitch interaction; processes remain the repetition units.',
        environment=json.loads((root/'environment.json').read_text()),contexts=contexts,case_summaries=case_summaries,
        processes=processes,timing=timing,measured_gate_effect_processes=gate_processes,measured_gate_effects=gate_summaries,
        position_pitch_responses=position_summaries,
        delay_by_pitch_interaction=interaction_summaries,selected_pitch_processes=selected,
        direct_delay_selected=direct,direct_delay_summaries=direct_summaries,
        background_processes=background,background_summaries=background_summaries,
        address_and_placement=dict(virtual_origins=sorted({p['address_metadata']['b_buffer_ptr'] for p in processes}),
            per_trial=origins,same_origin_trials=[r['trial'] for r in origins if len(set(r['by_case'].values()))==1],
            distinct_CTA_to_SM_maps=len({tuple((r['cta'],r['sm']) for r in p['cta_sm']) for p in processes})),
        references=references,failed_attempts=failures,
        source_boundary='The one-time gate is in the mainloop ProducerWarp after wait_on_dependent_grids and before the first while. '
            'Lane0 polls globaltimer, the warp synchronizes, and the original producer v06_begin follows. '
            'Tensor-map prefetch was issued before the gate, so the delay also grants prefetch completion opportunity.',
        interpretation=['P0=prod-entry includes the explicit wait; S=FIRST-prod and S+L exclude it. '
            'end-to-prod and end-to-FIRST isolate the post-gate software interval without calling it a bare TMA latency.',
            'Producer/FIRST group ranks use same-call globaltimer and retain tie intervals. '
            'They are software progress markers, not measured TMA request arrival order.',
            'The zero setting still executes the timer/do-while observer path. Report duration512-duration0 and P0 separately; '
            'end is before the stores/syncwarp, so end-to-prod remains part of the observation.',
            'begin/end cycle and ns reads are adjacent, not simultaneous; the short delay interval is not used to estimate frequency.',
            'A fixed max-plus dependency graph with unchanged edge costs and other root times cannot amplify a single-root delay. '
            'Positive (MAIN_END-prod) changes challenge a pure shift only under those assumptions; they do not reject all finite-stage models.',
            'The intervention tests participation of temporal state, including descriptor readiness, not a unique same-source queuing mechanism.',
            'Other130 excludes CTA55/91; same-B partners exclude only CTA55. Neither population is treated as independent process replicas.',
            'Pitch/delay pairs use independent processes; virtual-address metadata does not establish common physical pages.'],
        input_sha256=hashes,analyzer_sha256=common.sha(Path(__file__)))
    output.mkdir(parents=True,exist_ok=True)
    (output/'R10-producer-delay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# CTA55 producer起步延迟','',
        '同调用记录门前/门后、旧prod、FIRST与MAIN_END；P0含显式等待，S及S+L不含。'
        'descriptor prefetch位于门前，因此该干预检验时间状态参与，不唯一隔离同B请求到达顺序。',
        '每列独立取十个进程的中位数；时间闭合在逐进程检查，中位数各列不要求相加闭合。','',
        '| 条件 | 实际delay ns / cycles中位 | end→prod ns | P0 ns | S ns | L ns | S+L ns | entry→MAIN_END ns |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in case_summaries:
        b=r['boundaries'];lines.append(f'| {r["case"]} | {r["actual_delay_ns"]["median"]:.1f} / {r["actual_delay_cycles"]["median"]:.1f} | '
            +' | '.join(f'{b[k]["median"]:.1f}' for k in ('end_to_prod_ns','P0_ns','S_ns','L_ns','S_plus_L_ns','entry_to_MAIN_END_ns'))+' |')
    lines+=['','| B行距 | 实测duration512−duration0 ns | ΔP0 ns | Δend→prod ns |','|---:|---:|---:|---:|']
    for r in gate_summaries:
        cells=[]
        for field in ('duration_ns_delta','P0_ns_delta','end_to_prod_ns_delta'):
            v=r[field];cells.append(f'{v["median"]:.1f} [{v["process_min"]:.1f},{v["process_max"]:.1f}]')
        lines.append(f'| +{r["pitch_bytes"]} B | '+' | '.join(cells)+' |')
    lines+=['','| CTA55首输出直接延迟效应（512−0） | ΔS ns | ΔL ns | Δ(S+L) ns | Δ(entry→MAIN_END) ns |',
        '|---|---:|---:|---:|---:|']
    for r in gate_summaries:
        cells=[]
        for field in ('S_ns_delta','L_ns_delta','S_plus_L_ns_delta','entry_to_MAIN_END_ns_delta'):
            v=r[field];cells.append(f'{v["median"]:.1f} [{v["process_min"]:.1f},{v["process_max"]:.1f}]')
        lines.append(f'| B+{r["pitch_bytes"]} B | '+' | '.join(cells)+' |')
    lines+=['','直接延迟使两种行距的完整首输出窗口都增长。B+32 B的L缩短同时伴随S增长；'
        '即使排除显式等待，S+L仍增长。因此交互减小不是延迟带来加速，也不能把干预解释为仅增加固定启动项。']
    lines+=['','0档也有读表/do-while成本。end打点在stores与syncwarp之前，end→prod不能省略。',
        '','| 条件 | producer名次区间中位 | producer相对伙伴中位 ns | FIRST名次区间中位 | FIRST相对伙伴中位 ns |','|---|---:|---:|---:|---:|']
    for r in case_summaries:
        p,f=r['ranks']['producer'],r['ranks']['FIRST']
        lines.append(f'| {r["case"]} | {p["rank_min"]["median"]:.1f}～{p["rank_max"]["median"]:.1f} | '
            f'{p["target_minus_peer_median_ns"]["median"]:.1f} | {f["rank_min"]["median"]:.1f}～{f["rank_max"]["median"]:.1f} | '
            f'{f["target_minus_peer_median_ns"]["median"]:.1f} |')
    lines+=['','| CTA55首输出delay×pitch交互 | 中位 | 十进程范围 |','|---|---:|---:|']
    target=next(r for r in interaction_summaries if r['cta']==55 and r['j']==0)
    for k in ('L_cycles_delta','L_ns_delta','prefill_ns_delta','prefill_plus_L_ns_delta','entry_to_MAIN_END_ns_delta'):
        v=target[k];lines.append(f'| {k} | {v["median"]:.1f} | {v["process_min"]:.1f}～{v["process_max"]:.1f} |')
    lines+=['','在固定事件前驱、同一时钟域的固定边成本和其余根时刻不变时，max-plus递推满足单根延迟的增量不大于该延迟。'
        'B0除去实际producer起点变化后，MAIN_END仍多出以下区间，不能只将旧时间线平移；共享仲裁、事件顺序、局部成本和初态仍可能变化。'
        '这不排除所有有限stage模型，也不由跨进程配对宣称其余隐藏状态完全相同。','',
        '| 行距 | ΔP0 cycle | ΔS cycle | Δ(S+L) cycle | Δ入口→MAIN_END cycle |','|---|---:|---:|---:|---:|']
    for r in gate_summaries:
        cells=[]
        for field in ('P0_cycles_delta','S_cycles_delta','S_plus_L_cycles_delta','entry_to_MAIN_END_cycles_delta'):
            v=r[field];cells.append(f'{v["median"]:.1f} [{v["process_min"]:.1f},{v["process_max"]:.1f}]')
        lines.append(f'| +{r["pitch_bytes"]} B | '+' | '.join(cells)+' |')
    lines+=['','JSON保留全部位置行距响应、CTA55/91直接延迟对照、17个同B伙伴及其余130CTA的逐进程背景。'
        '不将短delay的cycle/ns比值作为频率，不用新数据拟合服务常数。','']
    (output/'R10-producer-delay.md').write_text('\n'.join(lines))
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
