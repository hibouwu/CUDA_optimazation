"""S11 independent collective accounting, scalar oracle and evidence validation."""
from __future__ import annotations

import itertools
import re

from auditors.memory_baseline import validate_device, timed_loops
from auditors.observation import validate_observation
from common.suite_io import require

ADAPTER_ID = 'synchronization_v2'
MASK32 = (1 << 32) - 1
METHOD = 'lcg_and_collective_phase_count_v1'
INPUTS = 'x0=seed+17*thread modulo2^32; selected threads advance256 MADs per phase'
MODES = {'warp': 0, 'cta': 1, 'mbarrier': 2, 'fence_cta': 3,
         'fence_gpu': 4, 'proxy_async': 5}
PTX = {
    'warp': ['bar.warp.sync 0xffffffff;'],
    'cta': ['bar.sync 1;'],
    'mbarrier': ['mbarrier.arrive.release.cta.shared::cta.b64',
                 'mbarrier.try_wait.acquire.cta.shared::cta.b64'],
    'fence_cta': ['fence.acq_rel.cta;'],
    'fence_gpu': ['fence.acq_rel.gpu;'],
    'proxy_async': ['fence.proxy.async.shared::cta;'],
}


def validate_contract(contract):
    require(contract.get('schema_version') == 2 and contract.get('stage') == 'S11'
            and contract.get('family') == 'synchronization'
            and contract.get('adapter_id') == ADAPTER_ID, 'S11 identity')
    require(bool(contract.get('sass_contracts')), 'S11 SASS contract')
    observed = []
    expected = {('warp', 32, skew) for skew in (False, True)}
    expected |= set(itertools.product(('cta', 'mbarrier'), (128, 256), (False, True)))
    expected |= {(mode, 32, False) for mode in ('fence_cta', 'fence_gpu', 'proxy_async')}
    for case in contract['cases']:
        p = case['parameters']
        mode, threads, skew = p['mode'], case['threads'], p['skew']
        require(mode in MODES and type(threads) is int and type(skew) is bool,
                'mode/threads/skew type')
        barrier = MODES[mode] < 3
        name = (f'{mode}_t{threads}_' + ('fixed_work_skew' if skew else 'aligned')
                if barrier else f'{mode}_no_outstanding_work')
        require(case['id'] == name and case['scope'] == 'one_cta'
                and case['launch'] == {'kind': 'one_cta'}, 'case identity/launch')
        require(case['iterations'] == (2048 if barrier else 8192)
                and p['batch'] == 8 and p['ptx'] == PTX[mode], 'iteration/instruction')
        if barrier:
            require(p == {
                'mode': mode, 'ptx': PTX[mode], 'batch': 8, 'skew': skew,
                'skew_steps': 256 if skew else 0,
                'skew_participants': 'lane>=16' if mode == 'warp' else 'warp==last',
                'barrier_id': 1 if mode == 'cta' else None,
                'arrival_count': threads if mode == 'mbarrier' else None,
                'wait_timeout_ns_per_thread': 1000000000 if mode == 'mbarrier' else None,
            }, 'frozen barrier protocol')
        else:
            require(p == {'mode': mode, 'ptx': PTX[mode], 'batch': 8,
                          'outstanding_memory_requests': 0, 'skew': False},
                    'frozen fence condition')
        require(case['work_model'] == ('synchronization_phase_v1' if barrier
                                      else 'fence_warp_issue_v1')
                and case['work_unit'] == 'operation', 'collective work model')
        require(case['metric'] == {
            'numerator': 'cta_clock64_cycles', 'denominator': 'work_count', 'scale': 1,
            'unit': 'clock64_cycle/phase/CTA' if barrier
                    else 'clock64_cycle/warp_instruction/CTA',
        }, 'collective metric')
        require(case['coverage_policy'] == 'one_sm' and case['exportable'] is True,
                'one CTA coverage')
        require(case['capabilities'] == {'cc': '9.0', 'device_name_contains': 'GH200'},
                'target requirements')
        observed.append((mode, threads, skew))
    require(len(observed) == 13 and set(observed) == expected, 'finite 13-point S11 matrix')
    require(contract['build'] == {
        'compiler': 'nvcc',
        'flags': ['-std=c++17', '-O3', '-lineinfo', '-gencode',
                  'arch=compute_90a,code=sm_90a', '-Xptxas=-v'],
        'include_dirs': [],
    }, 'frozen S11 build')
    return contract


def advance(value, count):
    """Independent integer affine power under modulo2^32."""
    a, b = 1664525, 1013904223
    total_a, total_b = 1, 0
    while count:
        if count & 1:
            total_a, total_b = a * total_a & MASK32, (a * total_b + b) & MASK32
        a, b = a * a & MASK32, (a * b + b) & MASK32
        count //= 2
    return (total_a * value + total_b) & MASK32


def expected_value(case, seed, thread, phases):
    p = case['parameters']
    selected = (thread % 32 >= 16 if p['mode'] == 'warp'
                else thread // 32 == case['threads'] // 32 - 1)
    steps = 256 * phases if p['skew'] and selected else 0
    return advance((seed + 17 * thread) & MASK32, steps)


def thread_records(rows, case, seed, phases, smid, correctness=False, diagnostic=False):
    require(isinstance(rows, list) and len(rows) == case['threads'], 'thread cardinality')
    require(all(type(r.get('thread_id')) is int for r in rows)
            and {r['thread_id'] for r in rows} == set(range(case['threads'])),
            'unique exact thread identities')
    for row in rows:
        for key in ('value', 'completed', 'wait_attempts', 'timeout', 'errors', 'checked', 'smid'):
            require(type(row.get(key)) is int and row[key] >= 0, 'thread integer: ' + key)
        require(row['timeout'] == row['errors'] == 0, 'thread error/timeout')
        require(row['smid'] == smid and row['completed'] == phases, 'thread scope/phase count')
        require(row['value'] == expected_value(case, seed, row['thread_id'], phases),
                'independent fixed-work recurrence')
        require(row['checked'] == (phases if correctness else 0), 'consumer checks')
        if case['parameters']['mode'] == 'mbarrier':
            require(row['wait_attempts'] >= phases * (2 if correctness else 1),
                    'each mbarrier phase must have successful wait attempt')
        else:
            require(row['wait_attempts'] == 0, 'non-mbarrier wait attempts')
        if diagnostic:
            require(type(row.get('arrival_cycle')) is int
                    and type(row.get('departure_cycle')) is int
                    and 0 <= row['arrival_cycle'] < row['departure_cycle'],
                    'diagnostic local cycle interval')
    return rows


def validate_trial(row, case, device, seed, protocol):
    phases = case['iterations'] * 8
    expected = {
        'blocks': 1, 'work': phases, 'read': 0, 'write': 0, 'operations': phases,
        'correctness': {'method': METHOD, 'checked_elements': case['threads'],
                        'input_conditions': INPUTS},
    }
    require(row.get('timing_model') == 'synchronization_phase_window_v1'
            and row.get('phase') == 'measure', 'timing model/phase')
    require(row.get('async_completion_proven') is False, 'fence is not async completion')
    require(type(row.get('local_size_bytes')) is int and row['local_size_bytes'] == 0,
            'local memory/spill')
    require(type(row.get('registers_per_thread')) is int
            and 0 < row['registers_per_thread'] <= 255, 'register count')
    require(type(row.get('static_smem_bytes')) is int
            and 0 < row['static_smem_bytes'] <= device['smem_per_cta_optin_bytes'],
            'shared capacity')
    require(len(row.get('blocks_detail', [])) == 1, 'one CTA required')
    smid = row['blocks_detail'][0]['smid']
    thread_records(row.get('thread_results'), case, seed, phases, smid)
    check = row.get('correctness_validation', {})
    require(check.get('phases') == 2 and check.get('errors') == 0, 'short correctness scope')
    correctness_rows = check.get('threads', [])
    require(correctness_rows and type(correctness_rows[0].get('smid')) is int,
            'correctness launch SM')
    thread_records(correctness_rows, case, seed, 2, correctness_rows[0]['smid'],
                   correctness=True)
    diagnostic = row.get('arrival_diagnostic', {})
    require(diagnostic.get('separate_launch') is True and diagnostic.get('phases') == 1
            and diagnostic.get('clock_basis') == 'same_sm_clock64', 'separate diagnostic scope')
    rows = diagnostic.get('threads', [])
    # Another launch may run on another SM. Compare clocks only inside that CTA.
    require(rows and type(rows[0].get('smid')) is int, 'diagnostic SM')
    thread_records(rows, case, seed, 1, rows[0]['smid'], diagnostic=True)
    arrival = [r['arrival_cycle'] for r in rows]
    require(diagnostic.get('arrival_spread_cycles') == max(arrival) - min(arrival),
            'independent observed arrival spread')
    # Correctness was also a different CTA launch; its SM need not equal measured SM.
    # Its thread records must share one observed SM, not falsely share a timestamp origin.
    return validate_observation(row, case, device, seed, protocol, expected)


def audit_sass(text, contract):
    """Fail closed on unknown target lowering; CPU fixtures are not SASS proof.

    Public A freezes PTX semantics, not Hopper mnemonic guesses. Until actual
    CUDA12.9 SASS is independently mapped for every mode, do not authorize GPU
    collection by accepting token presence alone.
    """
    functions = re.split(r'Function\s*:\s*', text)[1:]
    evidence = []
    for case in contract['cases']:
        p = case['parameters']
        selector = (f'sync_measuredILi{MODES[p["mode"]]}ELi{case["threads"]}'
                    f'ELb{int(p["skew"])}E')
        matches = [f for f in functions if selector in f.splitlines()[0]]
        require(len(matches) == 1, 'missing measured synchronization specialization: ' + selector)
        body = matches[0]
        require(not re.search(r'\b(?:LDL|STL)(?:\.|\s)', body), 'synchronization local spill')
        loops = timed_loops(body)
        evidence.append({'case_id': case['id'], 'symbol_selector': selector,
                         'timed_backedges': [(lo, hi) for lo, hi, _ in loops]})
    require(False, 'S11 target opcode/count mapping awaits independent CUDA12.9 evidence; no SASS pass yet')
    return evidence
