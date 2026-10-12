"""Ordinary single-CTA SMEM demand hypothesis and conditional cycle model."""
import math


def validate_map(case):
    width = case['width'] // 4
    threads = case['warps'] * 32
    maps = case['addresses_words']
    if len(maps) != 8 or any(len(row) != threads for row in maps):
        raise ValueError('eight complete per-thread address rows required')
    owners = {}
    for row in maps:
        for tid, word in enumerate(row):
            warp = tid // 32
            if word % width or not warp*4096 <= word <= (warp+1)*4096-width:
                raise ValueError('unaligned/out-of-region vector')
            if case['op'] == 1:
                for at in range(word, word+width):
                    if at in owners and owners[at] != tid:
                        raise ValueError('store ownership changes between lanes')
                    owners[at] = tid


def demand(case):
    """128B lane-group decomposition is a tested hypothesis, not a counter."""
    validate_map(case)
    width = case['width'] // 4
    group_lanes = 32 // width
    batches = []
    unique = set()
    for row in case['addresses_words']:
        per_warp = []
        for warp in range(case['warps']):
            service = 0
            for begin in range(warp*32, (warp+1)*32, group_lanes):
                banks = [set() for _ in range(32)]
                for tid in range(begin, begin+group_lanes):
                    for word in range(row[tid], row[tid]+width):
                        banks[word % 32].add(word)
                        unique.add(word)
                service += max(map(len, banks))
            per_warp.append(service)
        batches.append(per_warp)
    return dict(hypothesized_batches=batches, total_per_round=sum(map(sum,batches)),
        requested_bytes_per_round=case['width']*32*case['warps']*8,
        distinct_address_bytes=4*len(unique), group_lanes=group_lanes,
        physical_batches_measured=False)


def group_key(case):
    return f"op{case['op']}_b{case['width']}_w{case['warps']}_d{int(case['dependent'])}"


def predict(case, parameters):
    if case['op'] not in (0,1) or case['width'] not in (4,8,16) or case['warps'] not in (1,4,8):
        raise ValueError('outside declared ordinary full-warp support')
    if not 256 <= case['iterations'] <= 8192:
        raise ValueError('window length outside declared development span')
    values = parameters['groups'][group_key(case)]
    batches = demand(case)['total_per_round']
    if not values['batch_span'][0] <= batches <= values['batch_span'][1]:
        raise ValueError('demand outside development span')
    per_round = max(values['floor_cycles_per_round'],
        values['offset_cycles_per_round'] + values['cycles_per_hypothesized_batch']*batches)
    result = values['fixed_cycles'] + case['iterations']*per_round
    if not math.isfinite(result) or result <= 0:
        raise ValueError('invalid conditional prediction')
    return dict(window_cycles=result, cycles_per_target_warp_instruction=result/
        (case['iterations']*8*case['warps']), hypothesized_demand=demand(case))
