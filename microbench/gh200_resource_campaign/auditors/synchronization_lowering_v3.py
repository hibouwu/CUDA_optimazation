"""S11 CUDA 12.9 lowering draft; no adapter registration or GPU qualification."""
from __future__ import annotations

import copy
import hashlib
import json
import re

from auditors import synchronization as original
from common.suite_io import require

ADAPTER_ID = 'synchronization_lowering_v3'
LOWERING_ID = 'cuda129_sm90a_synchronization_v3'
validate_device = original.validate_device


def validate_contract(contract):
    require(contract.get('adapter_id') == ADAPTER_ID, 'S11 lowering adapter')
    lowering = contract.get('lowering_revision', {})
    require(lowering.get('id') == LOWERING_ID, 'S11 lowering identity')
    require(lowering.get('warp_native_parameter_export') is False,
            'warp lowering cannot export a native barrier parameter')
    normalized = copy.deepcopy(contract)
    normalized['adapter_id'] = original.ADAPTER_ID
    for case in normalized['cases']:
        if case['parameters']['mode'] != 'warp':
            continue
        require(case['work_model'] == 'compiled_warp_phase_v3'
                and case['exportable'] is False, 'warp logical service only')
        require(case['metric']['unit'] == 'clock64_cycle/logical_phase/CTA',
                'warp metric must identify logical phase')
        case['work_model'] = 'synchronization_phase_v1'
        case['exportable'] = True
        case['metric']['unit'] = 'clock64_cycle/phase/CTA'
    original.validate_contract(normalized)
    kernels = contract['sass_contracts']['observed_kernels']
    require(set(kernels) == {c['id'] for c in contract['cases']},
            'exact measured kernel mapping required')
    return contract


def validate_trial(row, case, device, seed, protocol):
    # The device ABI and numerical oracle remain unchanged. The versioned case
    # supplies the logical-phase metric, so no raw performance field is relabeled.
    return original.validate_trial(row, case, device, seed, protocol)


def instructions(body):
    result = []
    for line in body.splitlines():
        match = re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?)(?:/\*|$)', line)
        if match:
            result.append((int(match[1], 16), ' '.join(match[2].split())))
    require(result and len({pc for pc, _ in result}) == len(result),
            'unique SASS addresses required')
    require(result == sorted(result), 'SASS addresses out of order')
    return result


def instruction_sha256(ops):
    encoded = json.dumps(ops, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def backward_branches(ops):
    result = []
    for pc, op in ops:
        # Hopper may encode a predicate operand after BRA (e.g. BRA P2, 0x2d0).
        match = re.search(r'\bBRA(?:\s+P\d+,)?\s+0x([0-9a-fA-F]+)', op)
        if match and int(match[1], 16) < pc:
            result.append((int(match[1], 16), pc))
    return result


def window_and_loop(ops, mode):
    timers = [pc for pc, op in ops if 'SR_GLOBALTIMERLO' in op]
    clocks = [pc for pc, op in ops if 'SR_CLOCKLO' in op]
    require(len(timers) == (18 if mode == 'mbarrier' else 2),
            'unexpected window/timeout timer count')
    require(len(clocks) == 2, 'exact clock64 boundary pair required')
    start, stop = timers[0], timers[-1]
    require(start < clocks[0] < clocks[1] < stop, 'boundary clock order')
    start_op = next(op for pc, op in ops if pc == start)
    require(start_op.startswith('@!P'), 'entry timer must be thread0-predicated')
    require(any('STS.128' in op for pc, op in ops if clocks[0] < pc < clocks[0] + 128),
            'entry timestamp shared store missing')
    require(any('BAR.SYNC' in op for pc, op in ops if start - 64 <= pc < start),
            'initial CTA barrier missing')
    require(any('BAR.SYNC' in op for pc, op in ops if clocks[1] - 64 <= pc < clocks[1]),
            'result drain CTA barrier missing')
    loops = [(lo, hi) for lo, hi in backward_branches(ops)
             if clocks[0] < lo < hi < clocks[1]]
    require(loops, 'no measured outer backedge')
    lo, hi = max(loops, key=lambda pair: pair[1] - pair[0])
    require(any('BAR.SYNC' in op for pc, op in ops if clocks[0] < pc < lo),
            'entry CTA gate missing')
    internal = [pc for pc in timers if start < pc < stop]
    require(all(lo <= pc <= hi for pc in internal), 'timeout timer outside loop')
    return {'start_ns_pc': start, 'stop_ns_pc': stop,
            'start_cycle_pc': clocks[0], 'stop_cycle_pc': clocks[1],
            'outer_loop': [lo, hi], 'timeout_timer_pcs': internal}


def check_demand(ops, boundary, case):
    mode = case['parameters']['mode']
    lo, hi = boundary['outer_loop']
    body = [op for pc, op in ops if lo <= pc <= hi]
    count = lambda token: sum(token in op for op in body)
    require(not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)', op) for _, op in ops),
            'local spill not allowed')
    mad = sum(bool(re.search(r'\bIMAD R2, R2, R\d+, 0x3c6ef35f', op)) for op in body)
    require(mad == (2048 if case['parameters']['skew'] else 0),
            'eight phases must retain256 dependent MADs each')
    result = {'dependent_MAD_static_per_iteration': mad,
              'WARPSYNC_static_per_iteration': count('WARPSYNC')}
    if mode == 'warp':
        require(count('WARPSYNC') == count('BAR.SYNC') == 0,
                'unexpected native warp synchronization lowering')
        result['native_warp_barrier_exportable'] = False
    elif mode == 'cta':
        require(count('BAR.SYNC.DEFER_BLOCKING 0x1') == 8,
                'eight target CTA barrier1 phases required')
    elif mode == 'mbarrier':
        require(count('SYNCS.ARRIVE.TRANS64.A1T0') == 8,
                'eight mbarrier arrivals required')
        require(count('SYNCS.PHASECHK.TRANS64.TRYWAIT') == 14
                and count('SYNCS.PHASECHK.TRANS64') == 28,
                'unknown mbarrier polling lowering')
        require(count('NANOSLEEP.SYNCS 0x40') == 14,
                'bounded try-wait hint lowering changed')
        require(count('0x3b9aca00') == 1 and count('ATOMS.EXCH') == 1,
                'one-second timeout comparison/collective abort store missing')
        require(count('BAR.SYNC') == 0, 'extra CTA barrier in measured mbarrier loop')
        result['static_phasecheck_sites'] = 28
        result['dynamic_wait_count_basis'] = 'raw per-thread wait_attempts only'
    else:
        token = 'MEMBAR.ALL.GPU' if mode == 'fence_gpu' else 'MEMBAR.ALL.CTA'
        require(count(token) == 8, 'eight fence issues required')
        require(count('FENCE.VIEW.ASYNC.S') == (8 if mode == 'proxy_async' else 0),
                'async proxy fence lowering changed')
    return result


def audit_sass(text, contract):
    validate_contract(contract)
    functions = re.split(r'Function\s*:\s*', text)[1:]
    evidence = []
    for case in contract['cases']:
        expected = contract['sass_contracts']['observed_kernels'][case['id']]
        matches = [f for f in functions if f.splitlines()[0].strip() == expected['symbol']]
        require(len(matches) == 1, 'missing/duplicate exact measured symbol')
        ops = instructions(matches[0])
        boundary = window_and_loop(ops, case['parameters']['mode'])
        demand = check_demand(ops, boundary, case)
        require(boundary == expected['boundaries'], 'target boundary map changed')
        # This first revision accepts only the observed target instruction stream.
        # Structural checks explain its semantics; the digest rejects unreviewed
        # branch/register/timeout changes that token counts alone would miss.
        require(instruction_sha256(ops) == expected['instructions_sha256'],
                'unreviewed CUDA12.9 measured instruction stream')
        evidence.append({'case_id': case['id'], 'symbol': expected['symbol'],
                         'boundaries': boundary, 'compiled_demand': demand})
    return {'lowering_id': LOWERING_ID, 'status': 'measured_sass_matches_observed_revision',
            'hardware_qualification': False, 'GPU_execution': False,
            'correctness_and_arrival_kernels_independently_qualified': False,
            'cases': evidence}
