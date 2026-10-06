"""Bounded S08 model and pure checks; no GPU, filesystem, or qualification side effects."""
import itertools,json,re,hashlib
from auditors.low_precision_bounded_fp8_sass_baseline import BASELINE
from common.suite_io import require
from auditors.memory_baseline import validate_device,timed_loops
from auditors.observation import validate_observation
from auditors.low_precision import decode_compute_opcode
ADAPTER_ID='low_precision_bounded_fp8_v1'
METHOD='low_precision_bounded_fp8_full_output_v1'
INPUTS='bounded K-alternating FP8; D0=(1+2*group+chain)/8; nonuniform short retains original D0'
CASES = {'bounded_v1_wgmma_e4m3_g1_all_gpu': {'capabilities': {'cc': '9.0',
                                                       'device_name_contains': 'GH200'},
                                      'coverage_policy': 'exact_sms',
                                      'exportable': True,
                                      'id': 'bounded_v1_wgmma_e4m3_g1_all_gpu',
                                      'iterations': 8192,
                                      'launch': {'kind': 'sms_capped_occupancy',
                                                 'maximum_ctas_per_sm': 4},
                                      'metric': {'denominator': 'elapsed_ns',
                                                 'numerator': 'work_count',
                                                 'scale': 1,
                                                 'unit': 'GFLOP/s/GPU'},
                                      'original_case_id': 'wgmma_e4m3_g1_all_gpu',
                                      'parameters': {'accumulator_type': 'f32',
                                                     'batch': 16,
                                                     'bounded_D0': '(1+2*warpgroup_in_CTA+chain)/8',
                                                     'chains': 2,
                                                     'collective_width_threads': 128,
                                                     'formal_input_profile': 'nonzero_K_alternating_cancellation_v1',
                                                     'groups': 1,
                                                     'input_type': 'e4m3',
                                                     'nonuniform_D0': 'unchanged:(1+warpgroup_in_CTA+chain)/8',
                                                     'nonuniform_validation_iterations': [1,
                                                                                          2],
                                                     'operand_source_form': 'SS',
                                                     'path': 'wgmma',
                                                     'ptx': 'wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3',
                                                     'saturating': False,
                                                     'shape': [64, 64, 32],
                                                     'uniform_input_value': None,
                                                     'wait': 0},
                                      'scope': 'all_gpu',
                                      'threads': 128,
                                      'work_model': 'dense_lowprecision_collective_v1',
                                      'work_unit': 'FLOP'},
 'bounded_v1_wgmma_e4m3_g1_one_cta': {'capabilities': {'cc': '9.0',
                                                       'device_name_contains': 'GH200'},
                                      'coverage_policy': 'one_sm',
                                      'exportable': True,
                                      'id': 'bounded_v1_wgmma_e4m3_g1_one_cta',
                                      'iterations': 8192,
                                      'launch': {'kind': 'one_cta'},
                                      'metric': {'denominator': 'cta_clock64_cycles',
                                                 'numerator': 'work_count',
                                                 'scale': 1,
                                                 'unit': 'FLOP/clock64_cycle/CTA'},
                                      'original_case_id': 'wgmma_e4m3_g1_one_cta',
                                      'parameters': {'accumulator_type': 'f32',
                                                     'batch': 16,
                                                     'bounded_D0': '(1+2*warpgroup_in_CTA+chain)/8',
                                                     'chains': 2,
                                                     'collective_width_threads': 128,
                                                     'formal_input_profile': 'nonzero_K_alternating_cancellation_v1',
                                                     'groups': 1,
                                                     'input_type': 'e4m3',
                                                     'nonuniform_D0': 'unchanged:(1+warpgroup_in_CTA+chain)/8',
                                                     'nonuniform_validation_iterations': [1,
                                                                                          2],
                                                     'operand_source_form': 'SS',
                                                     'path': 'wgmma',
                                                     'ptx': 'wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3',
                                                     'saturating': False,
                                                     'shape': [64, 64, 32],
                                                     'uniform_input_value': None,
                                                     'wait': 0},
                                      'scope': 'one_cta',
                                      'threads': 128,
                                      'work_model': 'dense_lowprecision_collective_v1',
                                      'work_unit': 'FLOP'},
 'bounded_v1_wgmma_e4m3_g2_all_gpu': {'capabilities': {'cc': '9.0',
                                                       'device_name_contains': 'GH200'},
                                      'coverage_policy': 'exact_sms',
                                      'exportable': True,
                                      'id': 'bounded_v1_wgmma_e4m3_g2_all_gpu',
                                      'iterations': 8192,
                                      'launch': {'kind': 'sms_capped_occupancy',
                                                 'maximum_ctas_per_sm': 4},
                                      'metric': {'denominator': 'elapsed_ns',
                                                 'numerator': 'work_count',
                                                 'scale': 1,
                                                 'unit': 'GFLOP/s/GPU'},
                                      'original_case_id': 'wgmma_e4m3_g2_all_gpu',
                                      'parameters': {'accumulator_type': 'f32',
                                                     'batch': 16,
                                                     'bounded_D0': '(1+2*warpgroup_in_CTA+chain)/8',
                                                     'chains': 2,
                                                     'collective_width_threads': 128,
                                                     'formal_input_profile': 'nonzero_K_alternating_cancellation_v1',
                                                     'groups': 2,
                                                     'input_type': 'e4m3',
                                                     'nonuniform_D0': 'unchanged:(1+warpgroup_in_CTA+chain)/8',
                                                     'nonuniform_validation_iterations': [1,
                                                                                          2],
                                                     'operand_source_form': 'SS',
                                                     'path': 'wgmma',
                                                     'ptx': 'wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3',
                                                     'saturating': False,
                                                     'shape': [64, 64, 32],
                                                     'uniform_input_value': None,
                                                     'wait': 0},
                                      'scope': 'all_gpu',
                                      'threads': 256,
                                      'work_model': 'dense_lowprecision_collective_v1',
                                      'work_unit': 'FLOP'},
 'bounded_v1_wgmma_e4m3_g2_one_cta': {'capabilities': {'cc': '9.0',
                                                       'device_name_contains': 'GH200'},
                                      'coverage_policy': 'one_sm',
                                      'exportable': True,
                                      'id': 'bounded_v1_wgmma_e4m3_g2_one_cta',
                                      'iterations': 8192,
                                      'launch': {'kind': 'one_cta'},
                                      'metric': {'denominator': 'cta_clock64_cycles',
                                                 'numerator': 'work_count',
                                                 'scale': 1,
                                                 'unit': 'FLOP/clock64_cycle/CTA'},
                                      'original_case_id': 'wgmma_e4m3_g2_one_cta',
                                      'parameters': {'accumulator_type': 'f32',
                                                     'batch': 16,
                                                     'bounded_D0': '(1+2*warpgroup_in_CTA+chain)/8',
                                                     'chains': 2,
                                                     'collective_width_threads': 128,
                                                     'formal_input_profile': 'nonzero_K_alternating_cancellation_v1',
                                                     'groups': 2,
                                                     'input_type': 'e4m3',
                                                     'nonuniform_D0': 'unchanged:(1+warpgroup_in_CTA+chain)/8',
                                                     'nonuniform_validation_iterations': [1,
                                                                                          2],
                                                     'operand_source_form': 'SS',
                                                     'path': 'wgmma',
                                                     'ptx': 'wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3',
                                                     'saturating': False,
                                                     'shape': [64, 64, 32],
                                                     'uniform_input_value': None,
                                                     'wait': 0},
                                      'scope': 'one_cta',
                                      'threads': 256,
                                      'work_model': 'dense_lowprecision_collective_v1',
                                      'work_unit': 'FLOP'},
 'bounded_v1_wgmma_e5m2_g1_all_gpu': {'capabilities': {'cc': '9.0',
                                                       'device_name_contains': 'GH200'},
                                      'coverage_policy': 'exact_sms',
                                      'exportable': True,
                                      'id': 'bounded_v1_wgmma_e5m2_g1_all_gpu',
                                      'iterations': 8192,
                                      'launch': {'kind': 'sms_capped_occupancy',
                                                 'maximum_ctas_per_sm': 4},
                                      'metric': {'denominator': 'elapsed_ns',
                                                 'numerator': 'work_count',
                                                 'scale': 1,
                                                 'unit': 'GFLOP/s/GPU'},
                                      'original_case_id': 'wgmma_e5m2_g1_all_gpu',
                                      'parameters': {'accumulator_type': 'f32',
                                                     'batch': 16,
                                                     'bounded_D0': '(1+2*warpgroup_in_CTA+chain)/8',
                                                     'chains': 2,
                                                     'collective_width_threads': 128,
                                                     'formal_input_profile': 'nonzero_K_alternating_cancellation_v1',
                                                     'groups': 1,
                                                     'input_type': 'e5m2',
                                                     'nonuniform_D0': 'unchanged:(1+warpgroup_in_CTA+chain)/8',
                                                     'nonuniform_validation_iterations': [1,
                                                                                          2],
                                                     'operand_source_form': 'SS',
                                                     'path': 'wgmma',
                                                     'ptx': 'wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e5m2',
                                                     'saturating': False,
                                                     'shape': [64, 64, 32],
                                                     'uniform_input_value': None,
                                                     'wait': 0},
                                      'scope': 'all_gpu',
                                      'threads': 128,
                                      'work_model': 'dense_lowprecision_collective_v1',
                                      'work_unit': 'FLOP'},
 'bounded_v1_wgmma_e5m2_g1_one_cta': {'capabilities': {'cc': '9.0',
                                                       'device_name_contains': 'GH200'},
                                      'coverage_policy': 'one_sm',
                                      'exportable': True,
                                      'id': 'bounded_v1_wgmma_e5m2_g1_one_cta',
                                      'iterations': 8192,
                                      'launch': {'kind': 'one_cta'},
                                      'metric': {'denominator': 'cta_clock64_cycles',
                                                 'numerator': 'work_count',
                                                 'scale': 1,
                                                 'unit': 'FLOP/clock64_cycle/CTA'},
                                      'original_case_id': 'wgmma_e5m2_g1_one_cta',
                                      'parameters': {'accumulator_type': 'f32',
                                                     'batch': 16,
                                                     'bounded_D0': '(1+2*warpgroup_in_CTA+chain)/8',
                                                     'chains': 2,
                                                     'collective_width_threads': 128,
                                                     'formal_input_profile': 'nonzero_K_alternating_cancellation_v1',
                                                     'groups': 1,
                                                     'input_type': 'e5m2',
                                                     'nonuniform_D0': 'unchanged:(1+warpgroup_in_CTA+chain)/8',
                                                     'nonuniform_validation_iterations': [1,
                                                                                          2],
                                                     'operand_source_form': 'SS',
                                                     'path': 'wgmma',
                                                     'ptx': 'wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e5m2',
                                                     'saturating': False,
                                                     'shape': [64, 64, 32],
                                                     'uniform_input_value': None,
                                                     'wait': 0},
                                      'scope': 'one_cta',
                                      'threads': 128,
                                      'work_model': 'dense_lowprecision_collective_v1',
                                      'work_unit': 'FLOP'},
 'bounded_v1_wgmma_e5m2_g2_all_gpu': {'capabilities': {'cc': '9.0',
                                                       'device_name_contains': 'GH200'},
                                      'coverage_policy': 'exact_sms',
                                      'exportable': True,
                                      'id': 'bounded_v1_wgmma_e5m2_g2_all_gpu',
                                      'iterations': 8192,
                                      'launch': {'kind': 'sms_capped_occupancy',
                                                 'maximum_ctas_per_sm': 4},
                                      'metric': {'denominator': 'elapsed_ns',
                                                 'numerator': 'work_count',
                                                 'scale': 1,
                                                 'unit': 'GFLOP/s/GPU'},
                                      'original_case_id': 'wgmma_e5m2_g2_all_gpu',
                                      'parameters': {'accumulator_type': 'f32',
                                                     'batch': 16,
                                                     'bounded_D0': '(1+2*warpgroup_in_CTA+chain)/8',
                                                     'chains': 2,
                                                     'collective_width_threads': 128,
                                                     'formal_input_profile': 'nonzero_K_alternating_cancellation_v1',
                                                     'groups': 2,
                                                     'input_type': 'e5m2',
                                                     'nonuniform_D0': 'unchanged:(1+warpgroup_in_CTA+chain)/8',
                                                     'nonuniform_validation_iterations': [1,
                                                                                          2],
                                                     'operand_source_form': 'SS',
                                                     'path': 'wgmma',
                                                     'ptx': 'wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e5m2',
                                                     'saturating': False,
                                                     'shape': [64, 64, 32],
                                                     'uniform_input_value': None,
                                                     'wait': 0},
                                      'scope': 'all_gpu',
                                      'threads': 256,
                                      'work_model': 'dense_lowprecision_collective_v1',
                                      'work_unit': 'FLOP'},
 'bounded_v1_wgmma_e5m2_g2_one_cta': {'capabilities': {'cc': '9.0',
                                                       'device_name_contains': 'GH200'},
                                      'coverage_policy': 'one_sm',
                                      'exportable': True,
                                      'id': 'bounded_v1_wgmma_e5m2_g2_one_cta',
                                      'iterations': 8192,
                                      'launch': {'kind': 'one_cta'},
                                      'metric': {'denominator': 'cta_clock64_cycles',
                                                 'numerator': 'work_count',
                                                 'scale': 1,
                                                 'unit': 'FLOP/clock64_cycle/CTA'},
                                      'original_case_id': 'wgmma_e5m2_g2_one_cta',
                                      'parameters': {'accumulator_type': 'f32',
                                                     'batch': 16,
                                                     'bounded_D0': '(1+2*warpgroup_in_CTA+chain)/8',
                                                     'chains': 2,
                                                     'collective_width_threads': 128,
                                                     'formal_input_profile': 'nonzero_K_alternating_cancellation_v1',
                                                     'groups': 2,
                                                     'input_type': 'e5m2',
                                                     'nonuniform_D0': 'unchanged:(1+warpgroup_in_CTA+chain)/8',
                                                     'nonuniform_validation_iterations': [1,
                                                                                          2],
                                                     'operand_source_form': 'SS',
                                                     'path': 'wgmma',
                                                     'ptx': 'wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e5m2',
                                                     'saturating': False,
                                                     'shape': [64, 64, 32],
                                                     'uniform_input_value': None,
                                                     'wait': 0},
                                      'scope': 'one_cta',
                                      'threads': 256,
                                      'work_model': 'dense_lowprecision_collective_v1',
                                      'work_unit': 'FLOP'}}

CONTRACT_CORE = {'adapter_id': 'low_precision_bounded_fp8_v1',
 'build': {'compiler': 'nvcc',
           'flags': ['-std=c++17',
                     '-O3',
                     '-lineinfo',
                     '-gencode',
                     'arch=compute_90a,code=sm_90a',
                     '-Xptxas=-v'],
           'include_dirs': []},
 'dependencies': ['microbench/gh200_resource_campaign/common/probe_runtime.cuh',
                  'microbench/gh200_resource_campaign/common/low_precision_bounded_fp8_reference_v1.hpp'],
 'design': {'path': 'microbench/gh200_resource_campaign/contracts/low_precision_bounded_fp8_v1.draft.json',
            'sha256': '0b798237662235e17477239042481fe821e9ec0ba42ee57e2b718143b18b7377'},
 'family': 'low_precision_bounded_fp8',
 'fragments': {'mma_A': {'col': '4*(lane%4)+element%4+16*(element//8)',
                         'elements_per_lane': 16,
                         'row': 'lane//4+8*((element//4)%2)'},
               'mma_B': {'col': 'lane//4',
                         'elements_per_lane': 8,
                         'row': '4*(lane%4)+element%4+16*(element//4)'},
               'mma_D': {'col': '2*(lane%4)+element%2',
                         'elements_per_lane': 4,
                         'row': 'lane//4+8*(element//2)'},
               'wgmma_AB_smem': {'alignment_bytes': 128,
                                 'element_bytes': 1,
                                 'leading_offset_bytes': 1024,
                                 'offset': '(outer%8)*16+(outer//8)*128+inner%16+(inner//16)*1024',
                                 'stride_offset_bytes': 128,
                                 'swizzle': 'none'},
               'wgmma_D': {'col': '2*(thread%4)+element%2+8*(element//4)',
                           'elements_per_thread': 32,
                           'row': '16*(thread//32)+(thread%32)//4+8*((element//2)%2)'}},
 'parent_contract': {'path': 'microbench/gh200_resource_campaign/contracts/low_precision_lowering_v3.json',
                     'sha256': '24a46269a210033c79eb1e4d8e3d6b312442b7a441ee802b47e662e2fc52ea4c'},
 'schema_version': 2,
 'source': 'microbench/gh200_resource_campaign/probes/low_precision_bounded_fp8_v1.cu',
 'stage': 'S08'}

def case_identity(case):
    candidate={k:v for k,v in case.items() if k not in ('b3_resource_identity','b3_blocks')}
    require(json.dumps(candidate,sort_keys=True)==json.dumps(CASES.get(case.get('id')),sort_keys=True),'exact bounded case')
    return case['parameters']

def validate_contract(contract):
    require(json.dumps({k:contract.get(k) for k in CONTRACT_CORE},sort_keys=True)==json.dumps(CONTRACT_CORE,sort_keys=True),'frozen bounded contract/source/build/layout')
    require(contract.get('family')=='low_precision_bounded_fp8' and contract.get('adapter_id')==ADAPTER_ID and contract.get('stage')=='S08','bounded contract identity')
    require(len(contract['cases'])==8 and {c['id'] for c in contract['cases']}==set(CASES),'eight bounded cases')
    for case in contract['cases']:case_identity(case)
    return contract

def work(case,device,occupancy):
    p=case_identity(case);require(type(occupancy) is int and 1<=occupancy<=32,'occupancy')
    blocks=1 if case['scope']=='one_cta' else device['sms']*min(4,occupancy)
    issued=blocks*p['groups']*32*8192
    return {'blocks':blocks,'work':issued*262144,'operations':issued,'read':0,'write':0,'correctness':{'method':METHOD,'checked_elements':blocks*p['groups']*8192,'input_conditions':INPUTS}}

def validate_trial(row,case,device,seed,protocol):
    expected=work(case,device,row.get('occupancy_limit_ctas_per_sm'))
    require(row.get('input_profile_id')=='nonzero_K_alternating_cancellation_v1' and row.get('D0_rule')=='(1+2*group+chain)/8' and row.get('dynamic_counter_verified') is False,'bounded identity/count boundary')
    require(type(row.get('nominal_WGMMA_per_group_iteration')) is int and row['nominal_WGMMA_per_group_iteration']==32,'nominal work')
    positive=24 if case['parameters']['input_type']=='e4m3' else 44
    require(row.get('bounded_operand_bits')=={'A_positive':positive,'B_even':positive,'B_odd':positive+128},'bounded FP8 bits')
    for key,value in {'operand_source_form':'SS','groups_per_CTA':case['parameters']['groups'],'chains':2,'batch':16,'wait':0,'shape':[64,64,32]}.items():
        require(json.dumps(row.get(key))==json.dumps(value),'condition label: '+key)
    require(row.get('input_encoding')==case['parameters']['input_type'] and type(row.get('max_abs_error')) in (int,float) and row['max_abs_error']==0 and row.get('output_elements_checked')==expected['correctness']['checked_elements'],'bounded full outputs')
    require(row.get('timing_model')=='low_precision_start_gate_result_drain_v1' and row.get('phase')=='measure','timing')
    for name in ('registers_per_thread','static_smem_bytes','local_size_bytes'):require(type(row.get(name)) is int,'resource integer')
    regs=row['registers_per_thread'];smem=row['static_smem_bytes'];occ=row['occupancy_limit_ctas_per_sm']
    require(0<regs<=255 and row['local_size_bytes']==0 and 4096+case['threads']*8+16<=smem<=device['smem_per_cta_optin_bytes'],'resources')
    require(regs==148 and smem==4096+case['threads']*8+16 and occ*case['threads']<=2048,'compile730679 resource identity/thread bound')
    require(occ*smem<=device['smem_per_sm_bytes'] and occ*case['threads']*regs<=device['registers_per_sm'],'residency')
    return validate_observation(row,case,device,seed,protocol,expected)

def audit_sass(text, contract):
    """Account SASS shapes rather than assuming one SASS per PTX instruction.

    Each specialization must contain the expected type-specific instructions in
    its rolled timed loop, proper start/drain barriers, and no local load/store.
    WGMMA completion and source form require the accompanying PTX as well.
    """
    functions = re.split(r'Function\s*:\s*', text)[1:]; evidence = []
    for path, kind, groups in itertools.product(('wgmma',), ('e4m3','e5m2'), (1, 2)):
        symbol = f'lp_bounded_v1_{path}_{kind}_g{groups}'
        mangled='_Z'+str(len(symbol))+symbol+'ijbPN2gh5StampEPd'
        matches=[body for body in functions if body.splitlines()[0].strip()==mangled]
        require(len(matches) == 1, 'missing/ambiguous S08 kernel: ' + symbol)
        function = matches[0]; instructions = []
        for line in function.splitlines():
            match = re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?)(?:/\*|$)', line)
            if match:
                instructions.append((int(match[1], 16), match[2].strip()))
        identity={'instruction_count':len(instructions),'instructions_sha256':hashlib.sha256('\n'.join(f'{pc:04x}: {op}' for pc,op in instructions).encode()).hexdigest()}
        require(identity==BASELINE[mangled],'actual target instruction drift; independent B2 required')
        require(not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)', op) for _, op in instructions), 'S08 timed function spills')
        timers = [pc for pc, op in instructions if 'SR_GLOBALTIMERLO' in op]
        clocks = [pc for pc, op in instructions if 'SR_CLOCKLO' in op]
        require(len(timers) == len(clocks) == 2, 'S08 timer reads')
        barriers = [pc for pc, op in instructions if 'BAR.SYNC' in op]
        lowered = contract.get('contract_revision') == 3 and path == 'mma' and kind in ('e4m3','e5m2')
        if lowered:
            require(contract.get('lowering_contract', {}).get('id') == LOWERING_ID, 'unknown lowering acceptance rule')
        loops = timed_loops(function)
        require(len(loops)==1,'one measured rolled loop required')
        seen = []
        for lo, hi, body in loops:
            arithmetic = [(pc, op) for pc, op in body if re.search(r'\b(?:QGMMA|IGMMA|HGMMA|HMMA|IMMA)(?:\.|\s)', op)]
            require(not any(op.lstrip().startswith('@') for _,op in arithmetic),'predicated arithmetic rejected')
            require(arithmetic, 'S08 arithmetic missing from timed back edge: ' + symbol)
            require(any(timers[0] < b < lo for b in barriers), 'S08 no barrier after start before loop')
            require(any(hi < b < timers[1] for b in barriers), 'S08 no post-loop drain barrier')
            if path == 'mma' and kind in ('e4m3','e5m2'):
                conversions = [re.search(r'\b(F2FP(?:\.[A-Za-z0-9_]+)+)\s', op).group(1)
                               for _, op in body if re.search(r'\bF2FP\.', op)]
                require(conversions and all(op == f'F2FP.F16.{kind.upper()}.UNPACK_B' for op in conversions),
                        'FP8 warp A/B conversion encoding mismatch')
            chain_registers=[re.search(r'QGMMA\.[^ ]+ (R\d+), gdesc\[UR8\], (R\d+)',op).groups() for _,op in arithmetic]
            require(chain_registers==[('R56','R56'),('R24','R24')]*16,'two same-register chains, sixteen issues each')
            require(any(op=='UIADD3 UR4, UR4, 0x1, URZ ;' for _,op in body)
                    and any(op=='ISETP.LE.AND P0, PT, R0, UR4, PT ;' for _,op in body)
                    and any(op=='@!P0 BRA 0xe30 ;' for _,op in body),'rolled loop increments and runtime length bound')
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
        evidence.append({'symbol': symbol, 'instruction_identity':identity, 'path': path, 'type': kind, 'groups': groups, 'timed_loops': seen,
                         'count_basis': 'static SASS; B compares lowering with PTX batch16*chains2; not dynamic NCU count'})
    return evidence
