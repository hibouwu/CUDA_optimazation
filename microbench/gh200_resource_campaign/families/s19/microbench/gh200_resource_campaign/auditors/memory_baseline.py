"""Independent S04 arithmetic and function-scoped SASS audit; never imports C++ formulas."""
from __future__ import annotations
import math
import re
from common.suite_io import cv, require
from auditors.observation import validate_observation, validate_timer_contract, TIMER_FLAG

ADAPTER_ID = 'memory_baseline_v2'


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, 'integer field: ' + name)
    return value


def validate_contract(contract):
    revised=validate_timer_contract(contract)
    require(contract.get('schema_version') == 2 and contract.get('family') == 'memory_baseline' and contract.get('adapter_id') == ADAPTER_ID, 'unknown family/schema/adapter')
    expected = [(f'smem_read_stride{s}', 'smem_read', 32768, 8192, s) for s in (1, 2, 4, 8, 16, 32)]
    expected += [('smem_write_stride1', 'smem_write', 32768, 8192, 1),
                 ('global_read_ca_8m', 'global_read_ca', 8*1024**2, 64, 1),
                 ('global_read_cg_8m', 'global_read_cg', 8*1024**2, 64, 1),
                 ('global_read_cg_256m', 'global_read_cg', 256*1024**2, 16, 1),
                 ('global_write_256m', 'global_write', 256*1024**2, 16, 1),
                 ('global_duplex_128m', 'global_duplex', 128*1024**2, 16, 1),
                 ('empty_one_cta', 'empty', 0, 1, 1), ('empty_all_gpu', 'empty', 0, 1, 1)]
    rows = [(c['id'], c['parameters']['mode'], c['parameters']['bytes'], c['iterations'], c['parameters']['stride']) for c in contract['cases']]
    require(len(rows) == len(expected) and set(rows) == set(expected), 'S04 finite matrix mismatch')
    require(contract['stage'] == 'S04', 'S04 stage identity')
    require(contract['build'] == {'compiler': 'nvcc', 'flags': ['-std=c++17', '-O3', '-lineinfo', '-gencode', 'arch=compute_90a,code=sm_90a', '-Xptxas=-v']+([TIMER_FLAG] if revised else []), 'include_dirs': []}, 'S04 build protocol mismatch')
    for case in contract['cases']:
        shared = case['parameters']['mode'].startswith('smem')
        empty = case['parameters']['mode'] == 'empty'
        one = shared or case['id'] == 'empty_one_cta'
        require(case['threads'] == 256 and case['scope'] == ('one_cta' if one else 'all_gpu'), 'case launch/scope')
        require(case['launch'] == ({'kind': 'one_cta'} if one else {'kind': 'sms_multiple', 'multiplier': 4}), 'case launch formula')
        require(case['work_model'] == ('empty_window' if empty else 'shared_scalar8' if shared else 'global_vector16'), 'unknown/wrong work model')
        require(case['work_unit'] == ('operation' if empty else 'byte'), 'work unit')
        expected_metric = {'numerator': 'elapsed_ns' if empty else 'work_count', 'denominator': 'one' if empty else 'cta_clock64_cycles' if shared else 'elapsed_ns', 'scale': 1, 'unit': 'ns/window' if empty else 'B/clock64_cycle/CTA' if shared else 'GB/s_requested_payload/GPU'}
        if revised and empty and one:
            expected_metric={'numerator':'cta_clock64_cycles','denominator':'one','scale':1,'unit':'cycles/window'}
        require(case['metric'] == expected_metric, 'metric quantity/unit/scale mismatch')
        require(case['coverage_policy'] == ('one_sm' if one else 'observed_subset' if empty else 'exact_sms'), 'coverage policy')
        require(case.get('exportable') is (not empty), 'empty windows are not throughput exports')
        require(case['capabilities'] == {'cc': '9.0', 'device_name_contains': 'GH200'}, 'capability requirements')
    return contract


def validate_device(device):
    require(device.get('schema_version') == 2 and device.get('type') == 'device', 'device schema')
    require(device.get('cc') == '9.0' and 'GH200' in device.get('name', ''), 'device target')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', device.get('uuid', '')) is not None, 'device UUID')
    integer(device.get('sms'), 'sms', 1)
    integer(device.get('driver_version'), 'driver_version', 1)
    require(device.get('runtime_version') == 12090, 'CUDA runtime must be 12.9')
    return device


def work(case, device):
    p = case['parameters']
    blocks = 1 if case['launch']['kind'] == 'one_cta' else device['sms'] * case['launch']['multiplier']
    threads = case['threads']
    if case['work_model'] == 'empty_window':
        return {'blocks': blocks, 'allocation': 0, 'read': 0, 'write': 0, 'operations': 0, 'checked': 0}
    if case['work_model'] == 'shared_scalar8':
        amount = blocks * threads * 8 * 4 * case['iterations']
        require(p['mode'] in ('smem_read', 'smem_write'), 'shared mode')
        return {'blocks': blocks, 'allocation': 32768, 'read': amount if p['mode'] == 'smem_read' else 0,
                'write': amount if p['mode'] == 'smem_write' else 0, 'operations': blocks * threads * 8 * case['iterations'], 'checked': blocks * threads}
    require(case['work_model'] == 'global_vector16', 'unknown work model')
    quantum = blocks * threads * 16
    allocation = ((p['bytes'] + quantum - 1) // quantum) * quantum
    read = allocation * case['iterations'] if p['mode'] != 'global_write' else 0
    write = allocation * case['iterations'] if p['mode'] in ('global_write', 'global_duplex') else 0
    return {'blocks': blocks, 'allocation': allocation, 'read': read, 'write': write,
            'operations': (read + write) // 16, 'checked': blocks * threads + (allocation // 4 if write else 0)}


def validate_trial(row, case, device, seed, protocol):
    # Preserve S04 family-specific identity, allocation and numerical evidence;
    # the shared observation validator owns versioned clock/warmup checks.
    require(type(row.get('schema_version')) is int and row['schema_version'] == 2
            and row.get('type') == 'trial', 'trial schema')
    for name in ('iterations', 'seed', 'threads', 'blocks'):
        integer(row.get(name), name, 1)
    expected = work(case, device)
    require(row.get('requested_working_set_bytes') == case['parameters']['bytes']
            and row.get('allocation_per_array_bytes') == expected['allocation']
            and row.get('stride') == case['parameters']['stride'], 'allocation/stride mismatch')
    empty = case['work_model'] == 'empty_window'
    shared = case['work_model'] == 'shared_scalar8'
    method = 'empty_window_timestamps' if empty else 'host_per_thread_address_checksum' if shared else 'host_checksum_and_optional_full_array_store_validation'
    inputs = 'no memory workload' if empty else 'shared[i]=17*i+3; write initialization poison' if shared else 'global[i]=17*i+seed modulo 2^32'
    expected['work'] = expected['read'] + expected['write']
    expected['correctness'] = {'method': method, 'checked_elements': expected['checked'], 'input_conditions': inputs}
    return validate_observation(row, case, device, seed, protocol, expected)


def timed_loops(text):
    instructions = []
    for line in text.splitlines():
        match = re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?)(?:/\*|$)', line)
        if match:
            instructions.append((int(match[1], 16), match[2].strip()))
    timers = [pc for pc, op in instructions if 'SR_GLOBALTIMERLO' in op]
    require(len(timers) == 2, 'SASS requires two globaltimer boundaries')
    loops = []
    for pc, op in instructions:
        branch = re.search(r'\bBRA\s+0x([0-9a-fA-F]+)', op)
        if branch:
            target = int(branch[1], 16)
            if timers[0] < target < pc < timers[1]:
                loops.append((target, pc, [(a, x) for a, x in instructions if target <= a <= pc]))
    require(loops, 'no timed backward branch: possible hoisted work')
    return loops


def audit_sass(text, contract):
    functions = re.split(r'Function\s*:\s*', text)[1:]
    findings = []
    for rule in contract['sass_contracts']:
        matches = [body for body in functions if rule['function'] in body.splitlines()[0]]
        require(len(matches) == 1, 'SASS function missing/ambiguous: ' + rule['function'])
        loops = timed_loops(matches[0])
        tokens = rule.get('loop_tokens', [rule.get('loop_token')])
        for start, stop, body in loops:
            counts = {token: sum(re.search(r'\b' + re.escape(token) + r'(?:\.|\s)', op) is not None for _, op in body) for token in tokens}
            require(all(count > 0 for count in counts.values()), 'memory instruction missing in timed loop')
            if 'count_per_unrolled_body' in rule:
                require(all(count == rule['count_per_unrolled_body'] for count in counts.values()), 'unrolled timed workload mismatch')
            findings.append({'function': rule['function'], 'start': start, 'stop': stop, 'counts': counts})
    return findings
