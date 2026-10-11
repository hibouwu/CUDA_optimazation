"""Fixed GT observer controls; original point gates and separate read diagnostics."""
import math
import statistics
from analyze_r18 import reference
from decode_cfg_c_gt_bracket import decode
from decode_cfg_c_producer_tail import equal
from analyze_cfg_c_tail_control import evaluate as evaluate_point_controls


def process(events, sample, gpu_uuid):
    if len(events) != len({event['event'] for event in events}):
        raise ValueError('duplicate packet event')
    packets = {event['event']: event for event in events}
    setup, call, check = (packets[name] for name in ('setup', 'call', 'check'))
    if not equal(setup.get('gt_bracket_observer'), sample['flag']):
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
    common_origin = min(call['trace'][cta*1296+1] for cta in range(132))
    series = {}
    sms = {}
    for record in decoded['records']:
        cta = record['cta']
        head = call['trace'][cta*1296:(cta+1)*1296]
        origin_ns, origin_cycles = head[1], head[0]
        sms[str(cta)] = record['sm']
        series[f'{cta}/GLOBAL_ENTRY'] = [origin_ns-common_origin, None]
        series[f'{cta}/GLOBAL_PRODUCER'] = [head[13]-common_origin, None]
        series[f'{cta}/producer'] = [head[13]-origin_ns, head[3]-origin_cycles]
        for role in record['roles']:
            wg = role['wg']
            series[f'{cta}/{wg}/GLOBAL_FINAL'] = [role['final_ns']-common_origin, None]
            series[f'{cta}/{wg}/FINAL'] = [role['final_ns']-origin_ns, role['final_cycles']-origin_cycles]
            previous_ns, previous_cycles = head[13], head[3]
            for j, tile in enumerate(role['tiles']):
                coord = ','.join(map(str, tile['work']))
                prefix = f'{cta}/{wg}/{j}/{coord}/'
                for k, name in enumerate(('FIRST', 'MAIN', 'PERMIT', 'DONE')):
                    series[prefix+'GLOBAL_'+name] = [tile['ns'][k]-common_origin, None]
                    series[prefix+name] = [tile['ns'][k]-origin_ns, tile['cycles'][k]-origin_cycles]
                series[prefix+'L'] = [tile['ns'][1]-tile['ns'][0], tile['cycles'][1]-tile['cycles'][0]]
                for name, begin, end in (('W',1,2),('E',2,3)):
                    series[prefix+name] = [tile['ns'][end]-tile['ns'][begin], tile['cycles'][end]-tile['cycles'][begin]]
                series[prefix+'GAP'] = [tile['ns'][0]-previous_ns, tile['cycles'][0]-previous_cycles]
                previous_ns, previous_cycles = tile['ns'][3], tile['cycles'][3]
            series[f'{cta}/{wg}/TAIL'] = [role['final_ns']-previous_ns, role['final_cycles']-previous_cycles]
    brackets = {}
    for record in decoded['records']:
        for role in record['roles']:
            for ordinal, tile in enumerate(role['tiles']):
                key = f"{record['cta']}/{role['wg']}/{ordinal}/" + ','.join(map(str,tile['work'])) + '/L'
                if sample['flag']:
                    brackets[key] = dict(stamps=tile['gt_brackets'],
                        conditional_elapsed_gt=tile['conditional_elapsed_gt'],
                        post_gt_error_bound=tile['post_gt_error_bound'])
    if len(series) != 6372 or (sample['flag'] and len(brackets) != 432):
        raise ValueError('fixed complete series/window cardinality changed')
    result = dict(id=sample['id'], pair=sample['pair'], flag=sample['flag'], elapsed_us=elapsed,
        series=series, sms=sms, brackets=brackets)
    if 'arm' in sample: result['arm'] = sample['arm']
    return result


def evaluate(processes, manifest):
    result = evaluate_point_controls(processes, manifest)
    result.pop('tail_precision_qualified')
    result['gt_bracket_precision_qualified'] = False
    result['hardware_sampling_enclosure'] = False
    result['paired_interval_differences_available'] = False
    enabled = [row for row in processes if row['flag'] == 1]
    diagnostics = []
    if manifest['comparison'] == 'off_on':
        if len(enabled) != 10:
            raise ValueError('ten enabled targets required')
        keys = set(enabled[0]['brackets'])
        if len(keys) != 432 or any(set(row['brackets']) != keys for row in enabled):
            raise ValueError('complete matching bracket coordinates required')
        for key in sorted(keys):
            bounds = [row['brackets'][key]['post_gt_error_bound'] for row in enabled]
            widths = [[stamp['gt_after']-stamp['gt_before'] for stamp in row['brackets'][key]['stamps']] for row in enabled]
            diagnostics.append(dict(series=key,point_error_bounds=bounds,read_widths=widths,
                median_point_error_bound=statistics.median(bounds),maximum_point_error_bound=max(bounds),
                diagnostic_only=True))
    elif any(row['brackets'] for row in processes):
        raise ValueError('disabled targets must not have GT brackets')
    result['enabled_read_diagnostics'] = diagnostics
    result['limits'] = ('Original 6372 point-series gates remain unchanged. Read-width summaries are conditional GT-domain diagnostics only; disabled targets have no GT-before, so paired interval bounds cannot be supplied. No physical accuracy, sampling enclosure, perturbation correction or observation qualification.')
    return result
