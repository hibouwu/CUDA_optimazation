"""Pure ABI1 validation adapter for the finite S08 short profile.

No subprocess, GPU, filesystem reads/writes, or implicit prior-evidence approval.
Full output comparison and persistence of full output values are separate claims.
"""
from __future__ import annotations

from pathlib import PurePosixPath
import re

from auditors.low_precision_bounded_fp8 import audit_sass, validate_device, case_identity
from common.suite_io import require

ADAPTER_ID = 'low_precision_bounded_fp8_validation_v1'
ADAPTER_ABI_VERSION = 1
PROFILE_ID = 'bounded_fp8_and_original_nonuniform_1_2_v1'
REFERENCE_MODEL = 'low_precision_bounded_fp8_logical_v1'
REFERENCE_PATH = 'microbench/gh200_resource_campaign/common/low_precision_bounded_fp8_reference_v1.hpp'
REFERENCE_SHA256 = '41b77e330956da7750ad499b0245e3225ba202ecdaad79d7646fa05929c3f489'
PLAN = [(1, 'nonzero_K_alternating_cancellation_v1'), (2, 'nonzero_K_alternating_cancellation_v1'), (1, 'original_seeded_nonuniform_v1'), (2, 'original_seeded_nonuniform_v1')]
REQUIRED_PROFILE_FIELDS = {
    'id', 'target_iterations', 'input_profiles', 'maximum_target_launches',
    'maximum_explicit_auxiliary_launches', 'completion_requirements',
    'reference_identity', 'declared_output_check', 'output_evidence_kind',
}


def relative_path(value):
    require(isinstance(value, str) and value, 'nonempty relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts
            and '\\' not in value and path.as_posix() == value and value != '.',
            'relative path escape/noncanonical path')
    return value


def validate_profile(profile):
    require(isinstance(profile, dict) and REQUIRED_PROFILE_FIELDS <= profile.keys(),
            'complete executable S08 profile required')
    require(profile['id'] == PROFILE_ID and profile['target_iterations'] == [1, 2]
            and profile['input_profiles'] == ['nonzero_K_alternating_cancellation_v1', 'original_seeded_nonuniform_v1'], 'finite S08 profile')
    require(profile['maximum_target_launches'] == 4
            and profile['maximum_explicit_auxiliary_launches'] == 0, 'bounded target launches')
    require(profile.get('launch_plan') == [
        {'launch_index': i, 'iterations': count, 'input_profile': inputs}
        for i, (count, inputs) in enumerate(PLAN)
    ], 'frozen launch order')
    require(profile['reference_identity'] == {
        'model': REFERENCE_MODEL, 'path': REFERENCE_PATH, 'sha256': REFERENCE_SHA256,
    }, 'frozen independent reference identity')
    check = profile['declared_output_check']
    require(isinstance(check, dict) and check.get('comparison') == 'exact' and check.get('tolerance_id') is None
            and check.get('sampling') is False
            and check.get('elements_per_launch') == 'blocks*groups*chains*M*N',
            'full exact output comparison')
    require(profile['output_evidence_kind'] == 'error_count_only'
            and profile.get('resource_extensions') == {}, 'declared evidence/resources')
    require(profile.get('performance_eligible') is False
            and profile.get('warmup_allowed') is False and profile.get('pilot_allowed') is False,
            'short-only profile')
    require(profile.get('diagnostic_failure_detail_limit') == 8, 'bounded failure diagnostics')
    require(profile.get('seed')==3,'fixed short seed')
    return profile


def validation_argv(binary_relative, case, profile, seed):
    validate_profile(profile)
    case_identity(case)
    require(type(seed) is int and seed == 3, 'uint32 seed')
    return [relative_path(binary_relative), 'validate-only', case['id'], profile['id'], str(seed)]


def validate_validation(device, row, case, profile, seed):
    validate_profile(profile)
    p = case_identity(case)
    validate_device(device)
    require(type(seed) is int and seed == 3, 'uint32 seed')
    require(isinstance(row, dict), 'validation object')
    require(row.get('schema_version') == 2 and row.get('validation_schema_version') == 1
            and row.get('type') == 'validation', 'validation row schema, not trial')
    require(row.get('case_id') == case['id'] and row.get('profile_id') == PROFILE_ID
            and type(row.get('seed')) is int and row['seed'] == seed, 'validation identity')
    require(type(row.get('errors')) is int and row['errors'] == 0, 'validation errors')
    for flag in ('performance_eligible', 'warmup_executed', 'pilot_executed'):
        require(row.get(flag) is False, 'forbidden performance work: ' + flag)
    resource = row.get('resource_identity', {})
    require(isinstance(resource, dict) and set(resource) == {'kernel_symbol', 'registers_per_thread', 'static_smem_bytes',
                             'dynamic_smem_bytes', 'local_size_bytes',
                             'occupancy_limit_ctas_per_sm', 'extensions'}, 'complete resource identity')
    require(resource['kernel_symbol'] == f'lp_bounded_v1_wgmma_{p["input_type"]}_g{p["groups"]}'
            and resource['extensions'] == {}, 'kernel or undeclared resource extension')
    for field in ('registers_per_thread', 'static_smem_bytes', 'dynamic_smem_bytes',
                  'local_size_bytes', 'occupancy_limit_ctas_per_sm'):
        require(type(resource[field]) is int and resource[field] >= 0, 'resource integer: ' + field)
    occupancy = resource['occupancy_limit_ctas_per_sm']
    for field in ('smem_per_cta_optin_bytes', 'smem_per_sm_bytes', 'registers_per_sm'):
        require(type(device.get(field)) is int and device[field] > 0, 'device capacity: ' + field)
    require(1 <= occupancy <= 32 and 0 < resource['registers_per_thread'] <= 255
            and resource['dynamic_smem_bytes'] == resource['local_size_bytes'] == 0,
            'legal nonspilling target resources')
    minimum_smem = case['threads'] * 8 + 16 + (4096 if p['path'] == 'wgmma' else 0)
    require(minimum_smem <= resource['static_smem_bytes'] <= device['smem_per_cta_optin_bytes'],
            'operand/drain shared allocation')
    require(occupancy * resource['static_smem_bytes'] <= device['smem_per_sm_bytes']
            and occupancy * case['threads'] * resource['registers_per_thread'] <= device['registers_per_sm'],
            'necessary residency capacity bounds')
    require(resource['registers_per_thread']==148 and resource['static_smem_bytes']==4096+case['threads']*8+16,'compile730679 exact registers/shared binding')
    require(occupancy*case['threads']<=2048,'SM thread residency bound')
    blocks = 1 if case['scope'] == 'one_cta' else device['sms'] * min(4, occupancy)
    require(type(row.get('blocks')) is int and row['blocks'] == blocks
            and row.get('threads') == case['threads'] and row.get('scope') == case['scope'],
            'preserve case launch geometry')
    launches, checks = row.get('target_launches'), row.get('checks')
    require(isinstance(launches, list) and isinstance(checks, list)
            and len(launches) == len(checks) == 4, 'complete four-launch profile')
    require(all(isinstance(c, dict) and type(c.get('launch_index')) is int for c in checks)
            and {c['launch_index'] for c in checks} == set(range(4)), 'unique check launch IDs')
    indexed = {c['launch_index']: c for c in checks}
    count = blocks * p['groups'] * p['chains'] * p['shape'][0] * p['shape'][1]
    for index, (iterations, inputs) in enumerate(PLAN):
        require(isinstance(launches[index], dict)
                and all(type(launches[index].get(field)) is int
                        for field in ('launch_index', 'iterations', 'threads', 'blocks')), 'launch integer types')
        require(launches[index] == {'launch_index': index, 'iterations': iterations,
                                   'input_profile': inputs, 'threads': case['threads'], 'blocks': blocks},
                'launch profile/iteration/participant mismatch')
        check = indexed[index]
        require(set(check) == {'launch_index', 'reference_model', 'reference_sha256', 'comparison',
                              'tolerance_id', 'checked_elements', 'expected_elements', 'errors',
                              'completed', 'verified_CTA_ids', 'output_artifacts'}, 'complete check fields')
        require(check['reference_model'] == REFERENCE_MODEL and check['reference_sha256'] == REFERENCE_SHA256
                and check['comparison'] == 'exact' and check['tolerance_id'] is None, 'exact independent reference')
        require(type(check['errors']) is int and check['errors'] == 0 and check['completed'] is True,
                'numerical/completion failure')
        require(type(check['checked_elements']) is int and type(check['expected_elements']) is int
                and check['checked_elements'] == check['expected_elements'] == count, 'all output elements')
        ids = check['verified_CTA_ids']
        require(isinstance(ids, list) and all(type(i) is int for i in ids)
                and len(ids) == blocks and set(ids) == set(range(blocks)), 'all launched CTAs, not all physical SMs')
        require(check['output_artifacts'] == [], 'error_count_only profile has no value files')
    return {'status': 'pass', 'case_id': case['id'], 'profile_id': PROFILE_ID,
            'target_launches': 4, 'checked_elements': count * 4,
            'output_evidence_kind': 'error_count_only', 'performance_eligible': False}


def validate_prior_evidence(evidence, request, frozen_refs):
    """No reuse mapping is approved by this minimal S08 diagnostic adapter.

    Validate supplied artifact identities before reporting lack of coverage;
    corruption cannot be hidden as an opportunity to rerun. A later separately
    reviewed mapper can add reusable results without changing old evidence.
    """
    require(isinstance(evidence, dict) and isinstance(request, dict)
            and isinstance(frozen_refs, dict), 'JSON evidence/request/ref dictionaries')
    case = request.get('case')
    require(isinstance(case, dict), 'requested case object')
    case_identity(case)
    declared = evidence.get('source_artifacts_sha256', {})
    require(isinstance(declared, dict), 'artifact identity map')
    require(not evidence or bool(declared), 'prior evidence must declare bound artifacts')
    for path, digest in declared.items():
        relative_path(path)
        require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest)
                and frozen_refs.get(path) == digest, 'prior artifact hash mismatch')
    return {'status': 'insufficient', 'case_id': case['id'],
            'missing_requirements': ['independently approved S08 semantic reuse mapping is not implemented by this diagnostic adapter']}
