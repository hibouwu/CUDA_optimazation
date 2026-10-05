"""S18 finite grouped short ABI, complete outputs and independent closed-form replay."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import re
from common.suite_io import require
from common import auxiliary_reference_v1 as ref
from auditors.async_copy import validate_device
from auditors.auxiliary_sass_v1 import audit_sass

ADAPTER_ID = 'auxiliary_short_v1'
ADAPTER_ABI_VERSION = 1
CODE = Path(__file__).resolve().parents[1]
MATRIX = json.loads((CODE / 'contracts/auxiliary_scope_v1.draft.json').read_text())
PAIRS = ((1, 0), (2, 3), (5, 4294967295))
SETMAX_SHA = '109a932bf4e503e2a8139c58e5c56eac0f9b9d088ae08a08064f57d71c6e99fa'

def coordinate(case):
    found = [point for point in MATRIX['points'] if point['id'] == case.get('id')]
    require(len(found) == 1 and json.dumps(case.get('parameters'), sort_keys=True) == json.dumps(found[0], sort_keys=True), 'exact frozen S18 coordinate')
    point = found[0]
    require(case.get('launch') == {'scope': 'one_cta', 'threads': point['threads']}, 'one CTA frozen launch')
    return point

def profile_check(profile):
    fixed = json.loads((CODE / 'contracts/auxiliary_validation_profiles_v1.draft.json').read_text())['profiles'][0]
    require(json.dumps(profile, sort_keys=True) == json.dumps(fixed, sort_keys=True), 'exact finite S18 short profile')

def validation_argv(binary, case, profile, seed):
    coordinate(case); profile_check(profile)
    require(type(seed) is int and seed == 3, 'top-level seed3 with three internal paired seeds')
    path = PurePosixPath(binary)
    require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == binary and '\\' not in binary and binary != '.', 'relative binary path')
    return [binary, 'validate-only', case['id'], profile['id'], '3']

def symbol(point):
    family = point['family']
    if family == 'capacity': return 's18_capacity' + str(point['live_u32_values'])
    if family == 'local_explicit': return 's18_local' + str(point['words_per_thread'])
    if family == 'spill_pressure': return 's18_pressure' + str(point['words_per_thread'])
    if family == 'auxiliary_service':
        names = {'add_u64': 's18_add64_s', 'mad_wide_u32': 's18_madwide_s',
                 'cvt_rn_f16_f32': 's18_cvt_f16_f32_s', 'cvt_f32_f16': 's18_cvt_f32_f16_s'}
        return names[point['operation']] + str(point['streams'])
    if family == 'atomic': return 's18_atomic_' + point['address_space'] + ('_same' if point['address_pattern'] == 'same_word' else '_independent')
    return 'aux_setmax_initial64_v1'

def layout(point, iterations):
    threads = point['threads']; family = point['family']; out = {}
    if point.get('dynamic_smem_bytes', 0): out['smem'] = [point['dynamic_smem_bytes'] // 4]
    if family in ('capacity', 'local_explicit', 'spill_pressure'):
        words = point.get('live_u32_values', point.get('words_per_thread'))
        out['values'] = [threads, words]; inputs = threads * words
    elif family == 'auxiliary_service': out['values'] = [threads, point['streams'], 2]; inputs = 1
    elif family == 'atomic': out['values'] = [1 if point['address_pattern'] == 'same_word' else threads]; inputs = 128
    else: out['values'] = [iterations, 2, 128, 60]; inputs = 128 * 60
    out.update(guards=[2, 8], inputs=[inputs], input_guards=[2, 8])
    if family != 'setmaxnreg_legality': out['stamps'] = [6, 2]
    return out

def work_extensions(point):
    family = point['family']; threads = point['threads']
    words = point.get('live_u32_values', point.get('words_per_thread', point.get('streams', 1)))
    unit = 'u32_recurrence' if family in ('capacity', 'local_explicit', 'spill_pressure') else 'PTX_' + point['operation'] if family == 'auxiliary_service' else 'u32_atomic_logical_update' if family == 'atomic' else 'setmax_legality_no_service'
    requests = [threads * words * iterations for iterations, _ in PAIRS] if family in ('capacity', 'local_explicit', 'spill_pressure', 'auxiliary_service') else [threads * iterations for iterations, _ in PAIRS] if family == 'atomic' else [None] * 3
    return {'logical_target_unit': unit, 'logical_target_requests': requests,
            'explicit_local_request_bytes': [threads * words * iterations * 8 if family == 'local_explicit' else 0 for iterations, _ in PAIRS],
            'logical_atomic_updates': [threads * iterations if family == 'atomic' else 0 for iterations, _ in PAIRS],
            'expected_native_atomic_updates_from_frozen_SASS': [(4 if point.get('address_pattern') == 'same_word' else 128) * iterations if family == 'atomic' else 0 for iterations, _ in PAIRS]}

def resources(device, row, point):
    validate_device(device)
    require(type(row.get('blocks')) is int and row['blocks'] == 1 and type(row.get('threads')) is int and row['threads'] == point['threads'] and row.get('scope') == 'one_cta', 'finite CTA geometry')
    resource = row['resource_identity']
    for key in ('registers_per_thread', 'static_smem_bytes', 'dynamic_smem_bytes', 'local_size_bytes', 'occupancy_limit_ctas_per_sm'):
        ref.unsigned(resource[key], 32, key)
    require(resource['kernel_symbol'] == symbol(point) and resource['registers_per_thread'] > 0 and resource['occupancy_limit_ctas_per_sm'] > 0, 'actual legal resource identity')
    require(resource['dynamic_smem_bytes'] == point.get('dynamic_smem_bytes', 0), 'actual requested dynamic SMEM')
    require(resource['registers_per_thread'] * point['threads'] <= device['registers_per_sm'], 'register capacity')
    require(resource['static_smem_bytes'] + resource['dynamic_smem_bytes'] <= device['smem_per_cta_optin_bytes'], 'actual CTA SMEM capacity')
    # Local/spill implementation is proved by full SASS and ptxas stack/spill,
    # independently of the runtime API local_size_bytes field.
    if point['family'] == 'spill_pressure': require(resource['registers_per_thread'] <= 32, 'pressure request32 and spill storage condition absent')
    if point['family'] == 'setmaxnreg_legality': require(resource['registers_per_thread'] == 64, 'setmax actual initial64 condition')
    carveout = point.get('carveout', 'default'); carveout = -1 if carveout == 'default' else carveout
    expected = {'occupancy_is_upper_bound': True, 'requested_carveout': carveout, 'actual_carveout': None,
                'requested_pressure_register_limit': 32 if point['family'] == 'spill_pressure' else 0,
                'setmax_offline_cubin_sha256': SETMAX_SHA if point['family'] == 'setmaxnreg_legality' else '',
                'no_formal_admission': True, 'timer_policy': 'numeric_short_positive_cycles_nondecreasing_ns_event_no_rate', 'CUDA_event_ms': resource['extensions'].get('CUDA_event_ms')}
    expected.update(work_extensions(point))
    require(resource['extensions'] == expected and type(resource['extensions']['requested_carveout']) is int, 'exact request/unknown actual carving/offline-image extensions')
    require(resource['extensions']['occupancy_is_upper_bound'] is True and resource['extensions']['no_formal_admission'] is True and type(resource['extensions']['requested_pressure_register_limit']) is int, 'resource extension strict types')
    events = resource['extensions']['CUDA_event_ms']
    require(isinstance(events, list) and len(events) == 3 and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in events), 'three finite nonnegative CUDA event extents; completion separately synchronized')
    for key in ('logical_target_requests', 'explicit_local_request_bytes', 'logical_atomic_updates', 'expected_native_atomic_updates_from_frozen_SASS'):
        require(all(value is None if point['family'] == 'setmaxnreg_legality' and key == 'logical_target_requests' else type(value) is int and value >= 0 for value in resource['extensions'][key]), 'lossless integer logical work counts')

def validate_validation(device, row, case, profile, seed):
    point = coordinate(case); profile_check(profile)
    require(all(type(row.get(key)) is int for key in ('schema_version', 'validation_schema_version', 'seed', 'errors')), 'integer validation schema and counters')
    require(type(seed) is int and seed == 3 and row['type'] == 'validation' and row['schema_version'] == 2 and row['validation_schema_version'] == 1 and row['case_id'] == case['id'] and row['profile_id'] == profile['id'] and type(row['seed']) is int and row['seed'] == 3 and type(row['errors']) is int and row['errors'] == 0, 'S18 validation identity')
    require(all(row[key] is False for key in ('performance_eligible', 'warmup_executed', 'pilot_executed')), 'short only, no hidden warmup/pilot')
    resources(device, row, point)
    require(len(row['target_launches']) == len(row['checks']) == 3, 'exact three paired launches')
    checked = 0
    for stage, (iterations, _) in enumerate(PAIRS):
        launch = {'launch_index': stage, 'iterations': iterations, 'input_profile': 'auxiliary_nonuniform_paired_seeds_v1', 'threads': point['threads'], 'blocks': 1}
        require(row['target_launches'][stage] == launch and all(type(row['target_launches'][stage][key]) is int for key in ('launch_index', 'iterations', 'threads', 'blocks')), 'fixed ordered launch geometry')
        check = row['checks'][stage]; shapes = layout(point, iterations); count = sum(math.prod(shape) for shape in shapes.values())
        require(type(check['launch_index']) is int and check['launch_index'] == stage and check['completed'] is True and type(check['errors']) is int and check['errors'] == 0 and check['comparison'] == 'exact' and check['tolerance_id'] is None and check['reference_model'] == profile['reference_identity']['model'] and check['reference_sha256'] == profile['reference_identity']['sha256'] and check['verified_CTA_ids'] == [0] and type(check['verified_CTA_ids'][0]) is int, 'full-reference/completion identity')
        require(type(check['expected_elements']) is int and type(check['checked_elements']) is int and check['expected_elements'] == check['checked_elements'] == count, 'complete independently derived word count')
        require(len(check['output_artifacts']) == len(shapes), 'complete output set')
        for item, (leaf, shape) in zip(check['output_artifacts'], shapes.items()):
            require(item['path'] == f'auxiliary_{stage}_{leaf}.u32le' and item['shape'] == shape and all(type(n) is int for n in item['shape']) and item['dtype'] == 'uint32' and item['evidence_kind'] == 'full_values' and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None, 'artifact path/shape/dtype identity')
        checked += count
    return {'status': 'pass', 'case_id': case['id'], 'profile_id': profile['id'], 'target_launches': 3,
            'checked_elements': checked, 'performance_eligible': False, 'output_evidence_kind': 'full_values'}

def audit_values(device, row, case, profile, seed, arrays):
    validate_validation(device, row, case, profile, seed); point = coordinate(case); family = point['family']
    required = {f'auxiliary_{stage}_{leaf}.u32le' for stage, (iterations, _) in enumerate(PAIRS) for leaf in layout(point, iterations)}
    require(set(arrays) == required, 'exact full-value component set')
    for stage, (iterations, local_seed) in enumerate(PAIRS):
        get = lambda leaf: arrays[f'auxiliary_{stage}_{leaf}.u32le']
        for leaf, shape in layout(point, iterations).items():
            values = get(leaf); require(len(values) == math.prod(shape), 'full array length')
            for value in values: ref.unsigned(value, 32, 'raw word')
        guards = [0xdac00000 + k for k in range(8)] + [0xdac10000 + k for k in range(8)]
        require(get('guards') == guards and get('input_guards') == guards, 'all input/output guards')
        if family in ('capacity', 'local_explicit', 'spill_pressure'):
            words = point.get('live_u32_values', point.get('words_per_thread'))
            expected = [ref.word_result(local_seed, thread, word, iterations) for thread in range(point['threads']) for word in range(words)]
            inputs = [ref.initial(local_seed, thread, word) for thread in range(point['threads']) for word in range(words)]
        elif family == 'auxiliary_service':
            expected = []
            for thread in range(point['threads']):
                for stream in range(point['streams']):
                    value = ref.service_result(point['operation'], local_seed, thread, stream, iterations); expected.extend((value & ref.MASK32, value >> 32))
            inputs = [0]
        elif family == 'atomic':
            same_word = point['address_pattern'] == 'same_word'; expected = ref.atomic_results(local_seed, 128, same_word, iterations)
            inputs = [ref.initial(local_seed, 0, word) for word in range(128)]
            if point['address_space'] == 'global':
                if same_word: inputs[0] = expected[0]
                else: inputs = expected[:]
        else:
            inputs = [ref.initial(local_seed, thread, word) for thread in range(128) for word in range(60)]
            expected = inputs * (2 * iterations)
        require(get('values') == expected and get('inputs') == inputs, 'all output values and immutable/atomic input state')
        if point.get('dynamic_smem_bytes', 0): require(get('smem') == [ref.initial(local_seed, 0, word) for word in range(point['dynamic_smem_bytes'] // 4)], 'complete untimed SMEM drain')
        if family != 'setmaxnreg_legality':
            words = get('stamps'); values = [words[2*i] | words[2*i+1] << 32 for i in range(6)]
            ns0, ns1, cycle0, cycle1, sm0, sm1 = values
            require(ns1 >= ns0 and cycle1 > cycle0 and all(value != 2**64-1 for value in values[:4]) and sm0 == sm1 and sm0 < 2**32-1, 'positive same physical SM CTA time')
    return {'status': 'pass', 'full_values_replayed': True, 'performance_eligible': False}

def validate_prior_evidence(evidence, request, artifacts):
    return {'status': 'insufficient', 'case_id': request.get('case_id', ''), 'missing_requirements': ['S18 full finite short evidence; feasibility compile cannot grant B3']}
