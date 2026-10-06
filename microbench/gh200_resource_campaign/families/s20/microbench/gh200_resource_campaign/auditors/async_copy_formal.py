"""S12 formal observations; transport payload is counted once, not read+write.

This module is not registered with the suite runner until independent B review.
The original source/SASS auditor and short-run snapshots remain unchanged.
"""
from auditors.async_copy import case_identity, validate_device, audit_sass
from auditors.observation import validate_observation
from common.suite_io import require

ADAPTER_ID = 'async_copy_v2'


def validate_contract(contract):
    require(contract.get('schema_version') == 2 and contract.get('stage') == 'S12'
            and contract.get('family') == 'async_copy'
            and contract.get('adapter_id') == ADAPTER_ID, 'S12 formal identity')
    coordinates = set()
    for case in contract['cases']:
        p = case_identity(case)
        coordinate = (p['ptx'], p['request_bytes_per_thread'], p['stages'], case['scope'])
        require(coordinate not in coordinates, 'duplicate S12 coordinate')
        coordinates.add(coordinate)
        one = case['scope'] == 'one_cta'
        require(case.get('work_model') == 'async_copy_transport_payload_v1'
                and case.get('work_unit') == 'byte' and case.get('exportable') is True,
                'transport payload work model')
        require(case.get('coverage_policy') == ('one_sm' if one else 'exact_sms'), 'SM coverage policy')
        require(case.get('metric') == {'numerator':'work_count',
                'denominator':'cta_clock64_cycles' if one else 'elapsed_ns', 'scale':1,
                'unit':'B_transport/clock64_cycle/CTA' if one else 'GB/s_transport/GPU'},
                'transport metric and clock domain')
    expected = {(f'cp.async.{cache}.shared.global', width, stages, scope)
                for cache, width in [('ca',4),('ca',8),('ca',16),('cg',16)]
                for stages in (1,2,4) for scope in ('one_cta','all_gpu')}
    require(coordinates == expected and len(contract['cases']) == 24, 'complete finite S12 matrix')
    require(contract['build'] == {'compiler':'nvcc','flags':['-std=c++17','-O3','-lineinfo',
            '-gencode','arch=compute_90a,code=sm_90a','-Xptxas=-v'],'include_dirs':[]}, 'S12 build')
    return contract


def work(case, blocks):
    p = case_identity(case)
    require(type(blocks) is int and blocks > 0, 'positive CTA count')
    requests = blocks * 128 * case['iterations'] * 8
    payload = requests * p['request_bytes_per_thread']
    # Each final stage retains one payload per producer thread. Formal runs also
    # check one modular checksum per consumer; short traces are a separate ABI.
    checked = blocks * 128 * (1 + p['stages'] * (p['request_bytes_per_thread']//4))
    return {'blocks':blocks, 'read':payload, 'write':payload, 'work':payload,
            'operations':requests,
            'correctness':{'method':'modular_checksum_and_all_final_slots',
              'checked_elements':checked,
              'input_conditions':'uint32(17*word_index+seed); neighbor=(thread+1)%128; poison reset each launch'}}


def validate_trial(row, case, device, seed, protocol):
    p = case_identity(case)
    for key in ('registers_per_thread','static_smem_bytes','dynamic_smem_bytes',
                'local_size_bytes','occupancy_limit_ctas_per_sm','consumer_read_bytes',
                'request_bytes_per_thread','stages','global_array_bytes'):
        require(type(row.get(key)) is int and row[key] >= 0, 'resource/demand integer: '+key)
    occupancy = row['occupancy_limit_ctas_per_sm']
    require(0 < occupancy <= 32 and 0 < row['registers_per_thread'] <= 255, 'resource limits')
    require(row['local_size_bytes'] == row['static_smem_bytes'] == 0, 'reviewed nonspilling target resources')
    shared = row['dynamic_smem_bytes'] + row['static_smem_bytes']
    require(row['dynamic_smem_bytes'] == p['shared_payload_bytes'], 'stage allocation')
    require(shared <= device['smem_per_cta_optin_bytes']
            and shared * occupancy <= device['smem_per_sm_bytes'], 'shared capacity bounds')
    require(128 * row['registers_per_thread'] * occupancy <= device['registers_per_sm'], 'register capacity bound')
    require(128 * occupancy <= 2048, 'SM90 resident thread capacity bound')
    blocks = 1 if case['scope'] == 'one_cta' else device['sms'] * min(4, occupancy)
    expected = work(case, blocks)
    require(row['consumer_read_bytes'] == expected['work'], 'consumer read demand separate from transport')
    require(all(row[k] == p[k] for k in ('request_bytes_per_thread','stages','global_array_bytes')), 'request conditions')
    return validate_observation(row, case, device, seed, protocol, expected)
