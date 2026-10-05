"""S08 ABI1 finite short-profile tests. Fixtures confer no GPU qualification."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from auditors import low_precision_bounded_fp8_validation as adapter

source_text=(BASE/'probes/low_precision_bounded_fp8_v1.cu').read_text()
from types import SimpleNamespace
generator=SimpleNamespace(generate=lambda:source_text,HOST=source_text)

class LowPrecisionValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((BASE/'contracts/low_precision_bounded_fp8_v1.json').read_text())
        cls.profile = json.loads((BASE/'contracts/low_precision_bounded_fp8_validation_profiles_v1.json').read_text())['profiles'][0]
        cls.device = {
            'schema_version': 2, 'type': 'device', 'cc': '9.0', 'name': 'NVIDIA GH200 120GB',
            'uuid': 'GPU-00000000-0000-0000-0000-000000000001', 'sms': 132,
            'runtime_version': 12090, 'driver_version': 13010,
            'registers_per_sm': 65536, 'smem_per_sm_bytes': 233472,
            'smem_per_cta_optin_bytes': 232448,
        }

    def row(self, case):
        p = case['parameters']
        blocks = 1 if case['scope'] == 'one_cta' else self.device['sms'] * (2 if p['groups']==1 else 1)
        elements = blocks * p['groups'] * 2 * p['shape'][0] * p['shape'][1]
        return {
            'schema_version': 2, 'validation_schema_version': 1, 'type': 'validation',
            'case_id': case['id'], 'profile_id': adapter.PROFILE_ID, 'seed': 3,
            'scope': case['scope'], 'threads': case['threads'], 'blocks': blocks, 'errors': 0,
            'target_launches': [
                {'launch_index': i, 'iterations': count, 'input_profile': inputs,
                 'threads': case['threads'], 'blocks': blocks}
                for i, (count, inputs) in enumerate(adapter.PLAN)
            ],
            'checks': [
                {'launch_index': i, 'reference_model': adapter.REFERENCE_MODEL,
                 'reference_sha256': adapter.REFERENCE_SHA256, 'comparison': 'exact',
                 'tolerance_id': None, 'checked_elements': elements, 'expected_elements': elements,
                 'errors': 0, 'completed': True, 'verified_CTA_ids': list(range(blocks)),
                 'output_artifacts': []} for i in range(4)
            ],
            'resource_identity': {
                'kernel_symbol': f'lp_bounded_v1_wgmma_{p["input_type"]}_g{p["groups"]}',
                'registers_per_thread': 148,
                'static_smem_bytes': case['threads'] * 8 + 16 + (4096 if p['path'] == 'wgmma' else 0),
                'dynamic_smem_bytes': 0, 'local_size_bytes': 0,
                'occupancy_limit_ctas_per_sm': (2 if p['groups']==1 else 1), 'extensions': {},
            },
            'performance_eligible': False, 'warmup_executed': False, 'pilot_executed': False,
        }

    def validate(self, row, case=None, profile=None):
        return adapter.validate_validation(self.device, row, case or self.contract['cases'][0],
                                           profile or self.profile, 3)

    def test_complete_profile_reference_hash_and_exact_cli(self):
        adapter.validate_profile(self.profile)
        actual = hashlib.sha256((BASE/'common/low_precision_bounded_fp8_reference_v1.hpp').read_bytes()).hexdigest()
        self.assertEqual(actual, adapter.REFERENCE_SHA256)
        self.assertEqual(self.profile['reference_identity']['sha256'], actual)
        case = self.contract['cases'][0]
        self.assertEqual(adapter.validation_argv('binary/probe', case, self.profile, 3),
                         ['binary/probe', 'validate-only', case['id'], adapter.PROFILE_ID, '3'])
        self.assertIn('cannot support offline', self.profile['output_evidence_limit'])

    def test_all_8_short_geometry_variants_without_any_timing_fields(self):
        for case in self.contract['cases']:
            row = self.row(case)
            result = self.validate(row, case)
            self.assertEqual(result['target_launches'], 4)
            self.assertEqual(result['checked_elements'], row['checks'][0]['expected_elements'] * 4)
            self.assertFalse(result['performance_eligible'])
            self.assertEqual(result['output_evidence_kind'], 'error_count_only')
            self.assertNotIn('value', result)

    def test_unknown_profile_path_escape_seed_and_changed_target_rejected(self):
        case = self.contract['cases'][0]
        for path in ('/bin/probe', '../probe', 'binary/../probe', 'binary\\probe'):
            with self.assertRaises(ValueError):
                adapter.validation_argv(path, case, self.profile, 3)
        for seed in (-1, 0, 4, True, 1 << 32):
            with self.assertRaises(ValueError):
                adapter.validation_argv('binary/probe', case, self.profile, seed)
        for mutate in (
            lambda p: p.update(id='arbitrary'),
            lambda p: p.update(target_iterations=[8192]),
            lambda p: p.update(maximum_explicit_auxiliary_launches=1),
            lambda p: p['reference_identity'].update(sha256='0'*64),
            lambda p: p['declared_output_check'].update(comparison='profile_bounded'),
            lambda p: p.update(warmup_allowed=True),
        ):
            profile = deepcopy(self.profile)
            mutate(profile)
            with self.assertRaises(ValueError):
                adapter.validation_argv('binary/probe', case, profile, 3)
        case = deepcopy(case)
        case['parameters']['ptx'] = 'wrong.ptx'
        with self.assertRaises(ValueError):
            adapter.validation_argv('binary/probe', case, self.profile, 3)

    def test_missing_launch_bad_reference_nonfinite_and_sampled_outputs_rejected(self):
        for mutate in (
            lambda r: r.update(type='trial'),
            lambda r: r.update(errors=1),
            lambda r: r.update(warmup_executed=True),
            lambda r: r.update(pilot_executed=True),
            lambda r: r.update(performance_eligible=True),
            lambda r: r['target_launches'].pop(),
            lambda r: r['target_launches'][0].update(iterations=8192),
            lambda r: r['target_launches'][0].update(iterations=True),
            lambda r: r['checks'][0].update(launch_index=1),
            lambda r: r['checks'][0].update(reference_sha256='f'*64),
            lambda r: r['checks'][0].update(checked_elements=1),
            lambda r: r['checks'][0].update(errors=1),
            lambda r: r['checks'][0].update(completed=False),
            lambda r: r['checks'][0].update(tolerance_id='loose'),
            lambda r: r['checks'][0].update(verified_CTA_ids=[]),
            lambda r: r['checks'][0].update(output_artifacts=[{'path': 'sample.json'}]),
            lambda r: r['resource_identity'].update(local_size_bytes=4),
            lambda r: r['resource_identity'].update(extensions={'ignore_mismatch': True}),
            lambda r: r['resource_identity'].update(kernel_symbol='another_kernel'),
        ):
            row = self.row(self.contract['cases'][0])
            mutate(row)
            with self.assertRaises(ValueError):
                self.validate(row)

    def test_all_gpu_requires_all_launched_ctas_not_exact_physical_sm_coverage(self):
        case = next(c for c in self.contract['cases'] if c['scope'] == 'all_gpu')
        row = self.row(case)
        self.validate(row, case)  # no fabricated per-SM clock or exact_sms assertion
        row['checks'][0]['verified_CTA_ids'][-1] = 0
        with self.assertRaises(ValueError):
            self.validate(row, case)

    def test_prior_evidence_never_self_approves_reuse_or_hides_corruption(self):
        request = {'case': self.contract['cases'][0], 'profile': self.profile}
        result = adapter.validate_prior_evidence({}, request, {})
        self.assertEqual(result['status'], 'insufficient')
        evidence = {'source_artifacts_sha256': {'old/raw.jsonl': 'a'*64}}
        self.assertEqual(adapter.validate_prior_evidence(evidence, request,
                         {'old/raw.jsonl': 'a'*64})['status'], 'insufficient')
        with self.assertRaises(ValueError):
            adapter.validate_prior_evidence(evidence, request, {'old/raw.jsonl': 'b'*64})
        with self.assertRaises(ValueError):
            adapter.validate_prior_evidence({'raw': {'errors': 0}}, request, {})

    def test_generated_host_short_branch_returns_before_warmup_or_long_loops(self):
        source = generator.generate()
        begin = source.index('if(validate_only) {\n    std::vector<LowValidationCheck> checks;')
        end = source.index('// Formal entry requires', begin)
        branch = source[begin:end]
        self.assertIn('for(bool nonuniform:{false,true})', branch)
        self.assertIn('for(int length:{1,2})', branch)
        self.assertIn('return failed?2:0;', branch)
        self.assertNotIn('gh::warmup', branch)
        self.assertNotIn('gh::envelope', branch)
        self.assertNotIn('8192', branch)
        self.assertLess(end, source.index('gh::warmup'))
        self.assertIn('if(c.all_gpu&&!validate_only)', source)
        self.assertIn('if(validate_only)std::cerr<<message.str()', source)

    def test_actual_cpp_validation_emitter_matches_abi_and_failed_row(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('host C++ compiler unavailable')
        helper = generator.HOST.split('struct LowValidationCheck', 1)[1].split('int main(', 1)[0]
        helper = ('struct LowValidationCheck' + helper).replace('@REFERENCE_SHA256@', adapter.REFERENCE_SHA256)
        source = r'''
#include <iostream>
#include <string>
#include <vector>
namespace gh { using u64=unsigned long long;
std::string quote(const std::string& s){return "\""+s+"\"";} }
struct LowCase {const char* id;int kind,groups,width;bool all_gpu;};
struct cudaFuncAttributes {int numRegs;unsigned sharedSizeBytes,localSizeBytes;};
''' + helper + r'''
int main(){LowCase c{"bounded_v1_wgmma_e4m3_g1_one_cta",0,1,128,false};
cudaFuncAttributes attr{148,5136,0};std::vector<LowValidationCheck> checks;
for(int i=0;i<4;++i){LowValidationCheck x;x.index=i;x.iterations=i%2+1;
x.nonuniform=i>=2;x.completed=true;x.checked=x.expected=8192;x.cta_ids={0};checks.push_back(x);}
lp_emit_validation(c,3,1,attr,2,checks);
checks.resize(3);checks[2].errors=1;lp_emit_validation(c,3,1,attr,2,checks);}
'''
        with tempfile.TemporaryDirectory(prefix='gh200-validation-emitter-') as directory:
            root = Path(directory)
            (root/'test.cpp').write_text(source)
            subprocess.run([compiler, '-std=c++17', str(root/'test.cpp'), '-o', str(root/'test')],
                           check=True, capture_output=True, timeout=30)
            output = subprocess.run([str(root/'test')], check=True, text=True,
                                    capture_output=True, timeout=30).stdout.splitlines()
        good, failed = map(json.loads, output)
        case = next(c for c in self.contract['cases'] if c['id'] == good['case_id'])
        self.assertEqual(self.validate(good, case)['checked_elements'], 32768)
        self.assertEqual(failed['errors'], 1)
        self.assertEqual(len(failed['checks']), 3)
        with self.assertRaises(ValueError):
            self.validate(failed, case)


if __name__ == '__main__':
    unittest.main()
