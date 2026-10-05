"""S08 contract, raw and function/control-flow scoped SASS verification."""
from __future__ import annotations
import itertools
import re
from auditors.memory_baseline import validate_device, timed_loops
from auditors.observation import validate_observation
from common.suite_io import require

ADAPTER_ID = 'low_precision_v2'
KINDS = ('e4m3', 'e5m2', 's8', 'u8')
TIMING_MODEL = 'low_precision_start_gate_result_drain_v1'
METHOD = 'low_precision_full_output_integer_reference_v1'
INPUTS = 'formal uniform; separate seeded 1/2-iteration nonuniform full-output checks; fresh D0'
LOWERING_ID = 'cuda129_sm90a_fp8_mma_compiled_sequence_v1'
FRAGMENTS = {
    'mma_A': {'elements_per_lane': 16, 'row': 'lane//4+8*((element//4)%2)', 'col': '4*(lane%4)+element%4+16*(element//8)'},
    'mma_B': {'elements_per_lane': 8, 'row': '4*(lane%4)+element%4+16*(element//4)', 'col': 'lane//4'},
    'mma_D': {'elements_per_lane': 4, 'row': 'lane//4+8*(element//2)', 'col': '2*(lane%4)+element%2'},
    'wgmma_AB_smem': {'offset': '(outer%8)*16+(outer//8)*128+inner%16+(inner//16)*1024',
                      'element_bytes': 1, 'alignment_bytes': 128, 'leading_offset_bytes': 1024,
                      'stride_offset_bytes': 128, 'swizzle': 'none'},
    'wgmma_D': {'elements_per_thread': 32, 'row': '16*(thread//32)+(thread%32)//4+8*((element//2)%2)',
                'col': '2*(thread%4)+element%2+8*(element//4)'},
}


def validate_contract(contract):
    require(contract.get('schema_version') == 2 and contract.get('stage') == 'S08'
            and contract.get('family') == 'low_precision' and contract.get('adapter_id') == ADAPTER_ID, 'S08 identity')
    lowered_revision = contract.get('contract_revision') == 3
    require('contract_revision' not in contract or lowered_revision, 'unknown S08 revision')
    if lowered_revision:
        rule = contract.get('lowering_contract', {})
        require(rule.get('id') == LOWERING_ID and rule.get('logical_mma_per_warp_iteration') == 32
                and rule.get('observed_sass_per_warp_iteration') == {'F2FP.F16.INPUT.UNPACK_B': 12, 'HMMA.16816.F32': 2, 'FADD': 128}
                and rule.get('tensor_flop_per_warp_iteration') == 8192
                and rule.get('simt_add_operations_per_warp_iteration') == 4096
                and rule.get('logical_flop_per_warp_iteration') == 262144, 'approved FP8 lowering demand')
        require(contract.get('parent_contract') == {'path': 'microbench/gh200_resource_campaign/contracts/low_precision.json',
                'sha256': 'f8eb72346b1ee240845118ff869d41b6a6a8ed480d9a094078cbfac4c018d41a'}, 'lowering parent contract')
    found = []
    for case in contract['cases']:
        p = case['parameters']; path, kind, groups = p['path'], p['input_type'], p['groups']
        require(path in ('mma', 'wgmma') and kind in KINDS and type(groups) is int and groups in (1, 2), 'S08 dimensions')
        wg, floating = path == 'wgmma', kind in KINDS[:2]
        width, shape = (128, [64, 64, 32]) if wg else (32, [16, 8, 32])
        acc = 'f32' if floating else 's32'
        ptx = (f'wgmma.mma_async.sync.aligned.m64n64k32.{acc}.{kind}.{kind}' if wg else
               f'mma.sync.aligned.m16n8k32.row.col.{acc}.{kind}.{kind}.{acc}')
        require(case['scope'] in ('one_cta', 'all_gpu'), 'S08 scope')
        one = case['scope'] == 'one_cta'
        require(case['id'] == f'{path}_{kind}_g{groups}_{case["scope"]}'
                and case['threads'] == width * groups and type(case['iterations']) is int and case['iterations'] == 8192, 'case identity')
        require(p == {'path': path, 'input_type': kind, 'accumulator_type': acc, 'ptx': ptx, 'shape': shape,
                      'groups': groups, 'collective_width_threads': width, 'chains': 2, 'batch': 16,
                      'wait': 0 if wg else None, 'operand_source_form': 'SS' if wg else 'registers',
                      'saturating': False, 'uniform_input_value': 0.0625 if floating else 1,
                      'nonuniform_validation_iterations': [1, 2]}, 'frozen S08 parameters')
        require(p['saturating'] is False, 'saturation forbidden')
        require(case['launch'] == ({'kind': 'one_cta'} if one else {'kind': 'sms_capped_occupancy', 'maximum_ctas_per_sm': 4}), 'occupancy launch')
        unit = 'FLOP' if floating else 'OP'
        lowered = lowered_revision and path == 'mma' and floating
        require(case['work_model'] == ('ptx_logical_mma_lowering_v1' if lowered else 'dense_lowprecision_collective_v1')
                and case['work_unit'] == unit, 'work model/unit')
        metric_unit = f'{unit}/clock64_cycle/CTA' if one else f'G{unit}/s/GPU'
        if lowered:
            metric_unit = 'logical_' + metric_unit
            require(case.get('lowering_id') == LOWERING_ID, 'lowering case identity')
        require(case['metric'] == {'numerator': 'work_count', 'denominator': 'cta_clock64_cycles' if one else 'elapsed_ns',
                                  'scale': 1, 'unit': metric_unit}, 'metric unit/scope')
        require(case['coverage_policy'] == ('one_sm' if one else 'exact_sms') and case['exportable'] is (not lowered), 'coverage/export policy')
        require(case['capabilities'] == {'cc': '9.0', 'device_name_contains': 'GH200'}, 'target')
        found.append((path, kind, groups, case['scope']))
    required = set(itertools.product(('mma', 'wgmma'), KINDS, (1, 2), ('one_cta', 'all_gpu')))
    require(len(found) == 32 and set(found) == required, 'finite 32-point S08 matrix')
    require(contract['build'] == {'compiler': 'nvcc', 'flags': ['-std=c++17', '-O3', '-lineinfo', '-gencode',
                'arch=compute_90a,code=sm_90a', '-Xptxas=-v'], 'include_dirs': []}, 'build protocol')
    require(contract['fragments'] == FRAGMENTS, 'frozen fragment/layout rules')
    require(contract['correctness']['FP8_nonuniform'] == 'A(m,k)=((m+2*k+seed)%7-3)/8; B(k,n)=((n+3*k+seed)%7-3)/8; D0=(1+group+chain)/8'
            and contract['correctness']['S8_nonuniform'] == 'A(m,k)=(m+2*k+seed)%5-2; B(k,n)=(n+3*k+seed)%5-2; D0=1+group+chain'
            and contract['correctness']['U8_nonuniform'] == 'A(m,k)=(m+2*k+seed)%3; B(k,n)=(n+3*k+seed)%3; D0=1+group+chain', 'frozen nonuniform input')
    require(contract['numerical_bounds'] == {'formal_products_per_output': 4194304, 'FP8_formal_denominator': 256,
        'FP8_formal_integer_numerator_limit': 16777216, 'INT8_formal_output': 4194304,
        'INT8_max_nonuniform_abs_product': 4, 'signed_accumulation_limit': 2147483647}, 'numerical bounds')
    return contract


def work(case, device, occupancy):
    require(type(occupancy) is int and 1 <= occupancy <= 32, 'actual occupancy')
    p = case['parameters']; m, n, k = p['shape']
    require(case['threads'] == p['groups'] * p['collective_width_threads'], 'whole participating groups')
    blocks = 1 if case['scope'] == 'one_cta' else device['sms'] * min(4, occupancy)
    issued = blocks * p['groups'] * p['chains'] * p['batch'] * case['iterations']
    checked = blocks * p['groups'] * p['chains'] * m * n
    return {'blocks': blocks, 'work': issued * 2 * m * n * k, 'read': 0, 'write': 0, 'operations': issued,
            'correctness': {'method': METHOD, 'checked_elements': checked, 'input_conditions': INPUTS}}


def compiled_demand(case):
    kind = case['parameters']['input_type']
    require(case['work_model'] == 'ptx_logical_mma_lowering_v1' and case.get('lowering_id') == LOWERING_ID
            and kind in KINDS[:2] and case['parameters']['path'] == 'mma', 'lowering demand domain')
    return {'id': LOWERING_ID, 'basis': 'static_per_warp_iteration_requires_archived_sass_audit',
            'instruction_counts': {f'F2FP.F16.{kind.upper()}.UNPACK_B': 12, 'HMMA.16816.F32': 2, 'FADD': 128},
            'tensor_flop': 8192, 'simt_add_operations': 4096, 'logical_flop': 262144,
            'native_fp8_exportable': False, 'dynamic_counter_verified': False}


def validate_trial(row, case, device, seed, protocol):
    expected = work(case, device, row.get('occupancy_limit_ctas_per_sm'))
    p = case['parameters']; checked = expected['correctness']['checked_elements']
    for name in ('registers_per_thread', 'static_smem_bytes', 'local_size_bytes', 'output_elements_checked'):
        require(type(row.get(name)) is int and row[name] >= 0, 'integer resource field: ' + name)
    require(0 < row['registers_per_thread'] <= 255 and row['local_size_bytes'] == 0, 'registers/spill')
    required_smem = case['threads'] * 8 + 16 + (4096 if p['path'] == 'wgmma' else 0)
    require(row['static_smem_bytes'] >= required_smem, 'shared operands/drain/timer storage')
    for field in ('smem_per_sm_bytes', 'smem_per_cta_optin_bytes', 'registers_per_sm'):
        require(type(device.get(field)) is int and device[field] > 0, 'device resource: ' + field)
    occupancy = row['occupancy_limit_ctas_per_sm']
    require(row['static_smem_bytes'] <= device['smem_per_cta_optin_bytes']
            and occupancy * row['static_smem_bytes'] <= device['smem_per_sm_bytes'], 'shared capacity')
    require(occupancy * case['threads'] * row['registers_per_thread'] <= device['registers_per_sm'], 'register capacity')
    require(row.get('timing_model') == TIMING_MODEL and row.get('phase') == 'measure', 'timer/phase')
    require(row.get('input_encoding') == p['input_type'] and row.get('max_abs_error') == 0, 'input/error')
    require(row['output_elements_checked'] == checked, 'all outputs checked')
    require(row.get('nonuniform_validation') == {'iterations': [1, 2], 'checked_elements': checked * 2,
             'input_seed': seed, 'errors': 0, 'reference_model': 'low_precision_logical_integer_v1'}, 'nonuniform full output evidence')
    if case['work_model'] == 'ptx_logical_mma_lowering_v1':
        require(row.get('compiled_demand_contract') == compiled_demand(case), 'logical work/compiled demand separation')
    return validate_observation(row, case, device, seed, protocol, expected)


def validate_source_register_boundaries(source):
    """CPU structural check of the finite generated compiler-barrier contract.

    This does not replace inspection of newly compiled PTX/SASS. Every pair of
    unrolled loops explicitly binds all 2*32 registers at each WGMMA boundary.
    """
    evidence = []
    functions = re.split(r'__global__ void ', source)[1:]
    for kind, groups in itertools.product(KINDS, (1, 2)):
        symbol = f'lp_wgmma_{kind}_g{groups}'
        bodies = [f for f in functions if f.startswith(symbol + '(')]
        require(len(bodies) == 1, 'missing source specialization: ' + symbol)
        body = bodies[0]
        constraint = 'f' if kind in KINDS[:2] else 'r'
        boundary = (r'for\(int c=0;c<2;\+\+c\)\s*\{\s*#pragma unroll\s*'
                    r'for\(int j=0;j<32;\+\+j\)\s*asm volatile\(""\s*:\s*"\+'
                    + constraint + r'"\(d\[c\]\[j\]\)\s*::\s*"memory"\);\s*\}')
        matches = list(re.finditer(boundary, body))
        require(len(matches) == 2, 'all accumulator register boundaries required: ' + symbol)
        fence = body.index('asm volatile("wgmma.fence.sync.aligned;')
        wait = body.index('asm volatile("wgmma.wait_group.sync.aligned 0;')
        drain = body.index('double sum=0;')
        require(matches[0].end() < fence < wait < matches[1].start() < matches[1].end() < drain,
                'accumulator initialization/final-consumption ordering: ' + symbol)
        evidence.append({'symbol': symbol, 'constraint': '+' + constraint, 'registers_per_boundary': 64, 'boundaries': 2})
    return evidence


def decode_compute_opcode(op, path, kind):
    """Parse observed CUDA12.9/sm90a opcodes; reject mixed A/B and accumulator.

    Integer accumulator type is implicit in IMMA/IGMMA. FP8 warp MMA lowers to
    F2FP+HMMA and its input encoding must additionally be proved by conversions.
    """
    match = re.search(r'\b((?:QGMMA|IGMMA|HGMMA|HMMA|IMMA)(?:\.[A-Za-z0-9_]+)+)\s', op)
    require(match is not None, 'unknown matrix opcode')
    parts = match.group(1).split('.')
    floating = kind in KINDS[:2]
    if path == 'wgmma':
        prefix = 'QGMMA' if floating else 'IGMMA'
        expected_tail = ['F32', kind.upper(), kind.upper()] if floating else [kind.upper(), kind.upper()]
        require(parts[0] == prefix and parts[2:] == expected_tail, 'WGMMA A/B encoding or accumulator mismatch')
        shape = re.fullmatch(r'(\d+)x(\d+)x(\d+)', parts[1])
        require(shape is not None, 'unknown WGMMA SASS shape')
        m, n, k = map(int, shape.groups())
        require(m == 64 and 0 < n <= 64 and k == 32, 'wrong WGMMA shape')
    elif floating:
        require(parts == ['HMMA', '16816', 'F32'], 'FP8 warp lowering accumulator/shape mismatch')
        m, n, k = 16, 8, 16
    else:
        require(parts == ['IMMA', '16832', kind.upper(), kind.upper()], 'INT8 MMA A/B signedness or shape mismatch')
        m, n, k = 16, 8, 32
    return m, n, k


def audit_sass(text, contract):
    """Account SASS shapes rather than assuming one SASS per PTX instruction.

    Each specialization must contain the expected type-specific instructions in
    its rolled timed loop, proper start/drain barriers, and no local load/store.
    WGMMA completion and source form require the accompanying PTX as well.
    """
    functions = re.split(r'Function\s*:\s*', text)[1:]; evidence = []
    for path, kind, groups in itertools.product(('mma', 'wgmma'), KINDS, (1, 2)):
        symbol = f'lp_{path}_{kind}_g{groups}'
        matches = [body for body in functions if re.search(r'(?<![A-Za-z0-9_])' + symbol + r'(?:[^A-Za-z0-9_]|$)', body.splitlines()[0])
                   or symbol in body.splitlines()[0]]
        require(len(matches) == 1, 'missing/ambiguous S08 kernel: ' + symbol)
        function = matches[0]; instructions = []
        for line in function.splitlines():
            match = re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?)(?:/\*|$)', line)
            if match:
                instructions.append((int(match[1], 16), match[2].strip()))
        require(not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)', op) for _, op in instructions), 'S08 timed function spills')
        timers = [pc for pc, op in instructions if 'SR_GLOBALTIMERLO' in op]
        clocks = [pc for pc, op in instructions if 'SR_CLOCKLO' in op]
        require(len(timers) == len(clocks) == 2, 'S08 timer reads')
        barriers = [pc for pc, op in instructions if 'BAR.SYNC' in op]
        lowered = contract.get('contract_revision') == 3 and path == 'mma' and kind in KINDS[:2]
        if lowered:
            require(contract.get('lowering_contract', {}).get('id') == LOWERING_ID, 'unknown lowering acceptance rule')
        loops = timed_loops(function)
        seen = []
        for lo, hi, body in loops:
            arithmetic = [(pc, op) for pc, op in body if re.search(r'\b(?:QGMMA|IGMMA|HGMMA|HMMA|IMMA)(?:\.|\s)', op)]
            require(arithmetic, 'S08 arithmetic missing from timed back edge: ' + symbol)
            require(any(timers[0] < b < lo for b in barriers), 'S08 no barrier after start before loop')
            require(any(hi < b < timers[1] for b in barriers), 'S08 no post-loop drain barrier')
            if path == 'mma' and kind in KINDS[:2]:
                conversions = [re.search(r'\b(F2FP(?:\.[A-Za-z0-9_]+)+)\s', op).group(1)
                               for _, op in body if re.search(r'\bF2FP\.', op)]
                require(conversions and all(op == f'F2FP.F16.{kind.upper()}.UNPACK_B' for op in conversions),
                        'FP8 warp A/B conversion encoding mismatch')
            shape_work = 0
            for _, op in arithmetic:
                m, n, k = decode_compute_opcode(op, path, kind)
                shape_work += 2 * m * n * k
            logical_iteration_work = 16 * 2 * 2 * (64 if path == 'wgmma' else 16) * (64 if path == 'wgmma' else 8) * 32
            if lowered:
                adds = [op for _, op in body if re.search(r'\bFADD\s', op)]
                require(len(conversions) == 12 and len(arithmetic) == 2 and len(adds) == 128 and shape_work == 8192,
                        'FP8 lowering conversion/HMMA/FADD demand mismatch')
            else:
                require(shape_work == logical_iteration_work, 'SASS loop shape/count does not match batch16*chains2')
            if path == 'wgmma':
                # CUDA12.9 folds commit into the final matrix opcode's gsb0 operand.
                require(re.search(r',\s*gsb0\s*;', arithmetic[-1][1]) is not None, 'WGMMA final issue/commit group binding missing')
                require(any(re.search(r'WARPGROUP\.DEPBAR\.LE\s+gsb0,\s*0x0\s*;', op) for _, op in body), 'WGMMA wait0 missing')
                require(any(re.search(r'WARPGROUP\.DEPBAR\.LE\s+gsb0,\s*0x0\s*;', op)
                            for pc, op in instructions if hi < pc < timers[1]), 'WGMMA final wait0 missing')
                require(any('WARPGROUP.ARRIVE' in op for pc, op in instructions if pc < lo), 'WGMMA setup fence missing')
            seen.append({'loop_start': lo, 'loop_stop': hi, 'sass_compute_instructions': len(arithmetic),
                         'work_per_group_iteration_from_sass_shapes': shape_work,
                         'logical_flop_per_group_iteration': logical_iteration_work,
                         'lowering_id': LOWERING_ID if lowered else None,
                         'compiled_demand': {'F2FP': len(conversions), 'HMMA': len(arithmetic), 'FADD': len(adds)} if lowered else None,
                         'opcode_examples': sorted({op.split(';')[0].strip().split()[0] for _, op in arithmetic})})
        evidence.append({'symbol': symbol, 'path': path, 'type': kind, 'groups': groups, 'timed_loops': seen,
                         'count_basis': 'static SASS; B compares lowering with PTX batch16*chains2; not dynamic NCU count'})
    return evidence
