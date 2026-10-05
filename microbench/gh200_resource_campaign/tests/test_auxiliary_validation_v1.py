"""CPU fixtures only. No CUDA entry, query or target launch is executed."""
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from auditors import auxiliary_validation_v1 as adapter
from common import auxiliary_reference_v1 as reference
from runners.validation_diagnostic import validation_envelope, manifest_check

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parents[1]
CONTRACT = json.loads((CODE / 'contracts/auxiliary_short_v1.draft.json').read_text())
PROFILE = json.loads((CODE / 'contracts/auxiliary_validation_profiles_v1.draft.json').read_text())['profiles'][0]
POLICY = json.loads((CODE / 'contracts/early_validation_v1.json').read_text())
DEVICE = {'schema_version': 2, 'type': 'device', 'uuid': 'GPU-43269fbc-449d-3e0f-908a-9c81229546d3',
          'name': 'NVIDIA GH200 120GB', 'cc': '9.0', 'sms': 132, 'driver_version': 13010,
          'runtime_version': 12090, 'global_memory_bytes': 102005473280, 'l2_cache_bytes': 62914560,
          'registers_per_sm': 65536, 'smem_per_sm_bytes': 233472, 'smem_per_cta_optin_bytes': 232448}

def fixture(case):
    point = case['parameters']; family = point['family']; rows = []; launches = []; arrays = {}
    for stage, (iterations, seed) in enumerate(adapter.PAIRS):
        data = {}
        if family in ('capacity', 'local_explicit', 'spill_pressure'):
            words = point.get('live_u32_values', point.get('words_per_thread'))
            data['values'] = [reference.word_result(seed, t, word, iterations) for t in range(point['threads']) for word in range(words)]
            data['inputs'] = [reference.initial(seed, t, word) for t in range(point['threads']) for word in range(words)]
        elif family == 'auxiliary_service':
            data['values'] = []
            for t in range(128):
                for stream in range(point['streams']):
                    value = reference.service_result(point['operation'], seed, t, stream, iterations)
                    data['values'].extend((value & 0xffffffff, value >> 32))
            data['inputs'] = [0]
        elif family == 'atomic':
            same = point['address_pattern'] == 'same_word'
            data['values'] = reference.atomic_results(seed, 128, same, iterations)
            data['inputs'] = [reference.initial(seed, 0, word) for word in range(128)]
            if point['address_space'] == 'global':
                if same: data['inputs'][0] = data['values'][0]
                else: data['inputs'] = data['values'][:]
        else:
            data['inputs'] = [reference.initial(seed, t, word) for t in range(128) for word in range(60)]
            data['values'] = data['inputs'] * (2 * iterations)
        guard = [0xdac00000 + n for n in range(8)] + [0xdac10000 + n for n in range(8)]
        data['guards'] = guard[:]; data['input_guards'] = guard[:]
        if point.get('dynamic_smem_bytes', 0): data['smem'] = [reference.initial(seed, 0, word) for word in range(point['dynamic_smem_bytes'] // 4)]
        if family != 'setmaxnreg_legality': data['stamps'] = [100, 0, 200, 0, 1000, 0, 2000, 0, 201, 0, 201, 0]
        artifacts = []
        for leaf, shape in adapter.layout(point, iterations).items():
            name = f'auxiliary_{stage}_{leaf}.u32le'; arrays[name] = data[leaf]
            artifacts.append({'path': name, 'sha256': '0' * 64, 'dtype': 'uint32', 'evidence_kind': 'full_values', 'shape': shape})
        count = sum(math.prod(shape) for shape in adapter.layout(point, iterations).values())
        rows.append({'launch_index': stage, 'reference_model': PROFILE['reference_identity']['model'],
                     'reference_sha256': PROFILE['reference_identity']['sha256'], 'comparison': 'exact',
                     'tolerance_id': None, 'completed': True, 'errors': 0, 'checked_elements': count,
                     'expected_elements': count, 'verified_CTA_ids': [0], 'output_artifacts': artifacts})
        launches.append({'launch_index': stage, 'iterations': iterations, 'input_profile': PROFILE['input_profiles'][0],
                         'threads': point['threads'], 'blocks': 1})
    local = point.get('words_per_thread', 0) * 4 if family == 'local_explicit' else 48 if family == 'spill_pressure' else 32 if family == 'setmaxnreg_legality' else 0
    carveout = point.get('carveout', 'default'); carveout = -1 if carveout == 'default' else carveout
    resource = {'kernel_symbol': adapter.symbol(point), 'registers_per_thread': 64 if family == 'setmaxnreg_legality' else 32,
                'static_smem_bytes': 512 if family == 'atomic' and point['address_space'] == 'shared' else 0,
                'dynamic_smem_bytes': point.get('dynamic_smem_bytes', 0), 'local_size_bytes': local,
                'occupancy_limit_ctas_per_sm': 1, 'extensions': {'occupancy_is_upper_bound': True,
                'requested_carveout': carveout, 'actual_carveout': None,
                'requested_pressure_register_limit': 32 if family == 'spill_pressure' else 0,
                'setmax_offline_cubin_sha256': adapter.SETMAX_SHA if family == 'setmaxnreg_legality' else '',
                'no_formal_admission': True, 'timer_policy': 'numeric_short_positive_cycles_nondecreasing_ns_event_no_rate', 'CUDA_event_ms': [0.001] * 3}}
    resource['extensions'].update(adapter.work_extensions(point))
    row = {'schema_version': 2, 'validation_schema_version': 1, 'type': 'validation', 'case_id': case['id'],
           'profile_id': PROFILE['id'], 'seed': 3, 'scope': 'one_cta', 'threads': point['threads'], 'blocks': 1,
           'errors': 0, 'performance_eligible': False, 'warmup_executed': False, 'pilot_executed': False,
           'target_launches': launches, 'checks': rows, 'resource_identity': resource}
    return row, arrays

class AuxiliaryValidationTests(unittest.TestCase):
    def test_all28_complete_fixture_and_existing_core_envelope(self):
        self.assertEqual(len(CONTRACT['cases']), 28)
        for case in CONTRACT['cases']:
            with self.subTest(case=case['id']):
                row, arrays = fixture(case)
                validation_envelope(row, case, PROFILE, 3, POLICY)
                self.assertFalse(adapter.validate_validation(DEVICE, row, case, PROFILE, 3)['performance_eligible'])
                self.assertTrue(adapter.audit_values(DEVICE, row, case, PROFILE, 3, arrays)['full_values_replayed'])
                self.assertEqual(adapter.validation_argv('binary/probe', case, PROFILE, 3),
                                 ['binary/probe', 'validate-only', case['id'], PROFILE['id'], '3'])

    def test_every_component_corruption_missing_and_shape_rejected(self):
        for name in ('resource_smem32k', 'local_explicit_words128', 'cvt_f32_f16_streams4',
                     'atomic_add_u32_global_same_word', 'setmaxnreg_dec32_inc64'):
            case = next(case for case in CONTRACT['cases'] if case['id'] == name)
            row, arrays = fixture(case)
            for key in arrays:
                changed = {name: values[:] for name, values in arrays.items()}
                if key.endswith('_stamps.u32le'): changed[key][-2] = 202
                else: changed[key][0] ^= 1
                with self.subTest(case=name, component=key), self.assertRaises(ValueError):
                    adapter.audit_values(DEVICE, row, case, PROFILE, 3, changed)
            bad = dict(arrays); bad.pop(next(iter(bad)))
            with self.assertRaises(ValueError): adapter.audit_values(DEVICE, row, case, PROFILE, 3, bad)
            wrong = copy.deepcopy(row); wrong['checks'][0]['output_artifacts'][0]['shape'][0] += 1
            with self.assertRaises(ValueError): adapter.validate_validation(DEVICE, wrong, case, PROFILE, 3)

    def test_resource_and_sameSM_time_fail_closed(self):
        case = CONTRACT['cases'][0]; row, arrays = fixture(case)
        for key, value in [('registers_per_thread', True), ('dynamic_smem_bytes', 4), ('occupancy_limit_ctas_per_sm', 0)]:
            changed = copy.deepcopy(row); changed['resource_identity'][key] = value
            with self.assertRaises(ValueError): adapter.validate_validation(DEVICE, changed, case, PROFILE, 3)
        for value in (False, -0.001, float('nan'), float('inf')):
            changed = copy.deepcopy(row); changed['resource_identity']['extensions']['CUDA_event_ms'][0] = value
            with self.assertRaises(ValueError): adapter.validate_validation(DEVICE, changed, case, PROFILE, 3)
        quantized = copy.deepcopy(row); quantized['resource_identity']['extensions']['CUDA_event_ms'][0] = 0
        valid_arrays = {key: value[:] for key, value in arrays.items()}; valid_arrays['auxiliary_0_stamps.u32le'][2] = 100
        self.assertFalse(adapter.audit_values(DEVICE, quantized, case, PROFILE, 3, valid_arrays)['performance_eligible'])
        changed = {key: value[:] for key, value in arrays.items()}; changed['auxiliary_0_stamps.u32le'][-2] = 202
        with self.assertRaises(ValueError): adapter.audit_values(DEVICE, row, case, PROFILE, 3, changed)
        for name, key, value in [('spill_pressure_words32', 'registers_per_thread', 33), ('setmaxnreg_dec32_inc64', 'registers_per_thread', 63), ('local_explicit_words32', 'local_size_bytes', -1)]:
            case = next(case for case in CONTRACT['cases'] if case['id'] == name); changed, _ = fixture(case); changed['resource_identity'][key] = value
            with self.assertRaises(ValueError): adapter.validate_validation(DEVICE, changed, case, PROFILE, 3)

    def test_CPP_direct_recurrence_matches_independent_closed_form(self):
        source = '#include "' + str(CODE / 'common/auxiliary_host_reference_v1.hpp') + '"\n#include <iostream>\nint main(){for(unsigned seed: {0u,3u,0xffffffffu})for(unsigned t:{0u,31u,127u})for(unsigned j=0;j<4;++j)for(unsigned n:{1u,2u,5u,33u}){std::cout<<auxiliary_host_reference::word(seed,t,j,n)<<" ";for(unsigned op=0;op<4;++op)std::cout<<auxiliary_host_reference::service(op,seed,t,j,n)<<" ";std::cout<<"\\n";}}\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'reference.cpp').write_text(source)
            compile_result = subprocess.run(['g++', '-std=c++17', str(root / 'reference.cpp'), '-o', str(root / 'reference')], capture_output=True)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run = subprocess.run([str(root / 'reference')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0); values = [list(map(int, line.split())) for line in run.stdout.splitlines()]
        index = 0
        for seed in (0, 3, 0xffffffff):
            for t in (0, 31, 127):
                for j in range(4):
                    for n in (1, 2, 5, 33):
                        expected = [reference.word_result(seed, t, j, n)] + [reference.service_result(op, seed, t, j, n) for op in ('add_u64', 'mad_wide_u32', 'cvt_rn_f16_f32', 'cvt_f32_f16')]
                        self.assertEqual(values[index], expected); index += 1

    def test_manifest_closure_and_undeclared_entry_refusal(self):
        manifest_check(REPO, CODE / 'contracts/auxiliary_short_v1.draft.json', CODE / 'contracts/auxiliary_validation_profiles_v1.draft.json', CODE / 'contracts/auxiliary_validation_adapter_v1.draft.json')
        case = CONTRACT['cases'][0]
        for binary in ('../probe', '/tmp/probe', 'binary\\probe'):
            with self.assertRaises(ValueError): adapter.validation_argv(binary, case, PROFILE, 3)
        with self.assertRaises(ValueError): adapter.validation_argv('binary/probe', case, PROFILE, True)

    def test_full19_code_comparison_and_bit_drift_refusal(self):
        suite = REPO / 'results/gh200_resource_campaign/20261001-resource-suite-v2'
        evidence = (suite / 'implementation/s18-short-v1/compile-r1/actual-job731459/diagnostics/sass.stdout').read_text()
        result = adapter.audit_sass(evidence, CONTRACT)
        self.assertEqual(len(result), 19)
        self.assertTrue(all(row['comparison_only'] and not row['GPU_admission'] for row in result))
        import re
        match = re.search(r'0x([0-9a-f]{16})', evidence)
        changed = evidence[:match.start(1)] + f'{int(match[1], 16) ^ 1:016x}' + evidence[match.end(1):]
        with self.assertRaises(ValueError): adapter.audit_sass(changed, CONTRACT)
        with self.assertRaises(ValueError): adapter.audit_sass(evidence + evidence, CONTRACT)

if __name__ == '__main__': unittest.main()
