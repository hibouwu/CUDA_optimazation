#!/usr/bin/env python3
"""R10 CPU reference, stride/padding evidence and paired mainloop/time analysis."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


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
                later=[];first=[];supply=[];last_epi=[];windows=[]
                for c in obs['ctas']:
                    supply.append(c['tiles_ns'][0][0]-c['entry_ns'])
                    last_epi.append(c['tiles_ns'][-1][3]-c['tiles_ns'][-1][2])
                    for j,(cycle,ns,coord) in enumerate(zip(c['tiles'],c['tiles_ns'],c['work'])):
                        value=dict(case=row['id'],trial=rec['trial'],cta=c['cta'],sm=c['sm'],j=j,T=len(c['tiles']),
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
        summaries.append(dict(case=row['id'],a_pitched=(row['lda']*2)%128!=0,b_pitched=(row['ldb']*2)%128!=0,
            elapsed_us=med,cv={v:statistics.pstdev(t)/statistics.mean(t) for v,t in times.items()},
            disturbance={v:med[v]/med['wide']-1 for v in ('stamped','dual')},
            preparation_shift=med['wide']/med['plain']-1,scratch_bytes=scratch))
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
    report=dict(scope='Fixed M/N/K, D pitch, values and cache protocol; A/B physical pitch factorial. Direct ns is not reconstructed from whole-call frequency.',
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--set', choices=('representatives', 'formal'), default='formal')
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--joint-pitch', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
    elif args.joint_pitch:
        if args.output is None:parser.error('--joint-pitch requires --output')
        joint_pitch(args.input.resolve(),args.output.resolve())
    else:
        analyze(args.input, args.set)
