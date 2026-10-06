"""S09 audit: address enumeration and direction accounting independent of CUDA."""
from __future__ import annotations
import re
from auditors.memory_baseline import validate_device, timed_loops
from auditors.observation import validate_observation
from common.suite_io import require

ADAPTER_ID = 'shared_memory_v2'


def coordinates():
    return ([('read', 4, s, False) for s in (1, 2, 4, 8, 16, 32)]
            + [('read', w, 1, False) for w in (8, 16)]
            + [('write', w, 1, False) for w in (4, 8, 16)]
            + [('read', 4, 1, True)]
            + [('duplex', w, 1, False) for w in (4, 8, 16)])


def validate_contract(contract):
    require(contract['schema_version'] == 2 and contract['stage'] == 'S09'
            and contract['family'] == 'shared_memory' and contract['adapter_id'] == ADAPTER_ID, 'S09 identity')
    rows = []
    for c in contract['cases']:
        p = c['parameters']
        row = (p['mode'], p['access_bytes'], p['stride'], p['broadcast']);rows.append(row)
        require(type(p['broadcast']) is bool, 'broadcast must be boolean')
        name = f"{p['mode']}_w{p['access_bytes']}_" + ('broadcast' if p['broadcast'] else f"stride{p['stride']}")
        require(c['id'] == name and c['iterations'] == 8192 and c['threads'] == 256
                and c['scope'] == 'one_cta' and c['launch'] == {'kind': 'one_cta'}, 'case identity/launch')
        require(p == {'mode': row[0], 'access_bytes': row[1], 'stride': row[2], 'broadcast': row[3],
                      'accesses_per_thread_iteration': 8, 'array_words': 8192, 'arrays': 2,
                      'store_depends_on_load': False}, 'address/direction parameters')
        require(c['work_model'] == 'smem_vector_requests' and c['work_unit'] == 'byte', 'work model')
        require(c['coverage_policy'] == 'one_sm' and c['exportable'] is True, 'scope qualification')
        require(c['metric'] == {'numerator': 'work_count', 'denominator': 'cta_clock64_cycles',
                                'scale': 1, 'unit': 'B/clock64_cycle/CTA'}, 'metric')
        require(c['capabilities'] == {'cc': '9.0', 'device_name_contains': 'GH200'}, 'target')
    require(len(rows) == 15 and set(rows) == set(coordinates()), 'finite S09 matrix')
    require(contract['build'] == {'compiler': 'nvcc', 'flags': ['-std=c++17', '-O3', '-lineinfo', '-gencode',
                        'arch=compute_90a,code=sm_90a', '-Xptxas=-v'], 'include_dirs': []}, 'build protocol')
    return contract


def addresses(parameters):
    """Return word indices; repeated addresses intentionally remain repeated."""
    p = parameters; width = p['access_bytes'] // 4
    result = []
    for tid in range(256):
        for q in range(8):
            if p['broadcast']:
                base = (tid // 32) * 8 + q
            else:
                base = ((tid + q * 256) * p['stride'] * width) % p['array_words']
            require(base % width == 0 and base + width <= p['array_words'], 'word alignment/range')
            result.extend(range(base, base + width))
    return result


def work(case):
    p = case['parameters']; indices = addresses(p)
    direction_bytes = len(indices) * 4 * case['iterations']
    read = direction_bytes if p['mode'] in ('read', 'duplex') else 0
    write = direction_bytes if p['mode'] in ('write', 'duplex') else 0
    if write:
        require(len(set(indices)) == len(indices), 'overlapping shared stores')
    return {'blocks': 1, 'read': read, 'write': write, 'work': read + write,
            'operations': 256 * 8 * case['iterations'] * (int(read > 0) + int(write > 0)),
            'unique_bytes': len(set(indices)) * 4,
            'correctness': {'method': 'host_address_checksum_and_post_timer_per_word_store_check',
                            'checked_elements': 256 + (len(indices) if write else 0),
                            'input_conditions': 'read[i]=17*i+seed; write[i]=29*i+seed+7; poison; independent stores'}}


def validate_trial(row, case, device, seed, protocol):
    expected = work(case); p = case['parameters']
    require(row.get('dynamic_smem_bytes') == 65536, '64KiB allocation required')
    require(type(row.get('registers_per_thread')) is int and 0 < row['registers_per_thread'] <= 255, 'register count')
    require(type(row.get('occupancy_limit_ctas_per_sm')) is int and 0 < row['occupancy_limit_ctas_per_sm'] <= 32, 'occupancy upper')
    for key in ('smem_per_sm_bytes', 'smem_per_cta_optin_bytes', 'registers_per_sm'):
        require(type(device.get(key)) is int and device[key] > 0, 'missing device capacity: ' + key)
    occupancy = row['occupancy_limit_ctas_per_sm']
    require(row['dynamic_smem_bytes'] <= device['smem_per_cta_optin_bytes'], 'CTA shared capacity')
    # Necessary capacity bounds only; allocation granularity can reduce residency further.
    require(occupancy * row['dynamic_smem_bytes'] <= device['smem_per_sm_bytes'], 'occupancy exceeds shared capacity')
    require(occupancy * case['threads'] * row['registers_per_thread'] <= device['registers_per_sm'], 'occupancy exceeds register capacity')
    require(row.get('unique_bytes_per_direction_iteration') == expected['unique_bytes'], 'unique address accounting')
    require(row.get('access_bytes') == p['access_bytes'] and row.get('stride') == p['stride']
            and row.get('broadcast') is p['broadcast'], 'access condition')
    return validate_observation(row, case, device, seed, protocol, expected)


def audit_sass(text, contract):
    functions = re.split(r'Function\s*:\s*', text)[1:];out = []
    for width, mode, bcast in sorted({(p['parameters']['access_bytes'],
                                      {'read': 0, 'write': 1, 'duplex': 2}[p['parameters']['mode']],
                                      p['parameters']['broadcast']) for p in contract['cases']}):
        symbol = f'shared_accessILi{width}ELi{mode}ELb{int(bcast)}'
        matches = [x for x in functions if symbol in x.splitlines()[0]]
        require(len(matches) == 1, 'missing or ambiguous shared function: ' + symbol)
        for lo, hi, body in timed_loops(matches[0]):
            tokens = (['LDS'] if mode == 0 else ['STS'] if mode == 1 else ['LDS', 'STS'])
            for token in ('LDS', 'STS'):
                operations = [op for _, op in body if re.search(r'\b' + token + r'(?:\.|\s)', op)]
                require(len(operations) == (8 if token in tokens else 0), 'shared timed loop direction/access count')
                for op in operations:
                    expected_bits = width * 8
                    sizes = re.findall(r'\.(64|128)\b', op.split()[0])
                    actual_bits = int(sizes[0]) if sizes else 32
                    require(actual_bits == expected_bits, 'SASS access width mismatch')
            out.append({'symbol': symbol, 'start': lo, 'stop': hi, 'access_bytes': width, 'directions': tokens})
    return out
