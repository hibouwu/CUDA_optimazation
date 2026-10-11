"""Incremental tail observer control. Passing never qualifies underlying clocks."""
import math
import statistics

from analyze_r18 import reference
from decode_cfg_c_producer_tail import decode, equal


def process(events, sample, gpu_uuid):
    if len(events) != len({event['event'] for event in events}):
        raise ValueError('duplicate packet event')
    packets = {event['event']: event for event in events}
    setup, call, check = (packets[name] for name in ('setup', 'call', 'check'))
    if not equal(setup.get('producer_tail_observer'), sample['flag']):
        raise ValueError('manifest observer flag mismatch')
    decoded = decode(setup, call, gpu_uuid)
    if not decoded['record_valid']:
        raise ValueError('invalid complete CTA record')
    elapsed = call['elapsed_us']
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('invalid event duration')
    indices, values = check['checked_indices'], check['checked_values']
    if (check['samples'] != 4096 or len(indices) != 4096 or len(values) != 4096
            or len(set(indices)) != 4096 or check['nonfinite'] or check['padding_errors']):
        raise ValueError('incomplete numerical sample')
    for index, value in zip(indices, values):
        if type(index) is not int or not 0 <= index < 2304*3072:
            raise ValueError('invalid numerical index')
        if value != reference(index//3072, index%3072, 4096, 17):
            raise ValueError('numerical sample mismatch')
    series = {}
    sms = {}
    for record in decoded['records']:
        cta = record['cta']
        head = call['trace'][cta*1296:(cta+1)*1296]
        origin_ns, origin_cycles = head[1], head[0]
        sms[str(cta)] = record['sm']
        series[f'{cta}/producer'] = [head[13]-origin_ns, head[3]-origin_cycles]
        for role in record['roles']:
            wg = role['wg']
            series[f'{cta}/{wg}/FINAL'] = [role['final_ns']-origin_ns, role['final_cycles']-origin_cycles]
            for j, tile in enumerate(role['tiles']):
                coord = ','.join(map(str, tile['work']))
                prefix = f'{cta}/{wg}/{j}/{coord}/'
                for k, name in enumerate(('FIRST', 'MAIN', 'PERMIT', 'DONE')):
                    series[prefix+name] = [tile['ns'][k]-origin_ns, tile['cycles'][k]-origin_cycles]
                series[prefix+'L'] = [tile['ns'][1]-tile['ns'][0], tile['cycles'][1]-tile['cycles'][0]]
    return dict(id=sample['id'], pair=sample['pair'], flag=sample['flag'], elapsed_us=elapsed, series=series, sms=sms)


def metric(values, median_limit, maximum_limit):
    absolute = [abs(value) for value in values]
    median, maximum = statistics.median(absolute), max(absolute)
    return dict(signed=values, median_absolute=median, maximum_absolute=maximum,
                passed=median <= median_limit and maximum <= maximum_limit)


def evaluate(processes, manifest):
    expected = manifest['samples']
    if len(expected) != 20 or len(processes) != 20:
        raise ValueError('exactly twenty prescribed processes required')
    for i, (sample, actual) in enumerate(zip(expected, processes)):
        pair = i//2
        flag = (i%2) if pair%2 == 0 else (1-i%2)
        if not equal(sample['pair'], pair) or not equal(sample['flag'], flag):
            raise ValueError('prescribed alternating order required')
        if not all(equal(actual[k], sample[k]) for k in ('id', 'pair', 'flag')):
            raise ValueError('missing/reordered/mismatched process')
    keys = set(processes[0]['series'])
    if not keys or any(set(p['series']) != keys for p in processes):
        raise ValueError('complete matching role and output coordinates required')
    for p in processes:
        if set(p['sms']) != {str(cta) for cta in range(132)}:
            raise ValueError('complete CTA SM identities required')
    pairs = []
    for pair in range(10):
        rows = processes[pair*2:pair*2+2]
        pairs.append((next(p for p in rows if p['flag']==0), next(p for p in rows if p['flag']==1)))
    gates = manifest['gates']
    local = []
    for key in sorted(keys):
        delta = [b['series'][key][0]-a['series'][key][0] for a,b in pairs]
        local.append(dict(series=key, **metric(delta, gates['per_common_series_median_absolute_ns'],
            gates['per_common_series_maximum_absolute_ns']),
            native_cycle_deltas=[b['series'][key][1]-a['series'][key][1] for a,b in pairs],
            sm_changed=[a['sms'][key.split('/')[0]] != b['sms'][key.split('/')[0]] for a,b in pairs]))
    event = metric([b['elapsed_us']/a['elapsed_us']-1 for a,b in pairs],
                   gates['event_median_absolute_ratio'], gates['event_maximum_absolute_ratio'])
    off = [a['elapsed_us'] for a,b in pairs]
    cv = statistics.pstdev(off)/statistics.fmean(off)
    return dict(local=local, event=event, off_event_population_cv=cv,
        incremental_control_passed=all(row['passed'] for row in local) and event['passed'] and cv<=gates['off_event_population_cv'],
        observation_qualified=False, tail_precision_qualified=False, baseline_dual_qualified=False,
        limits='Incremental observer only. No tail off/on comparison, clock correction or pooled role acceptance.')
