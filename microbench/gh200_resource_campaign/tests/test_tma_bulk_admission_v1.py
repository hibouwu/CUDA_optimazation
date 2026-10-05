"""Resolved S14 lengths through the actual formal host and observation wrapper.

CPU resources are shim fixtures, never a substitute for qualified GPU B3.
"""
import copy
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_tma_bulk_formal_v1 import TmaBulkFormalTests, CONTRACT, PROTOCOL
from auditors import tma_bulk_admission_v1 as admission
from common.family_b3 import RESOURCE_FIELDS


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TmaBulkFormalTests.setUpClass()
        cls.host = TmaBulkFormalTests(methodName='runTest')

    @classmethod
    def tearDownClass(cls):
        TmaBulkFormalTests.tearDownClass()

    def fixture(self, case):
        process, rows, launches = self.host.run_host(case, length=128, seed=3)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(launches, ['128 3'] * 9)
        device, row = rows
        declared = {**copy.deepcopy(case), 'iterations': 128,
                    'b3_blocks': row['blocks'],
                    'b3_resource_identity': {key: row[key] for key in RESOURCE_FIELDS}}
        declared['b3_resource_identity'].update(kernel_symbol=row['kernel_symbol'], extensions={
            'validation_role': 'formal_final_transport', 'capture_enabled': False,
            **{key: row[key] for key in ('payload_bytes', 'global_slots_per_cta', 'global_allocation_bytes')}})
        return declared, device, row

    def test_all24_resolved_128_lengths_with_actual_host(self):
        for original in CONTRACT['cases']:
            with self.subTest(case=original['id']):
                case, device, row = self.fixture(original)
                admission.validate_trial(row, case, device, 3, PROTOCOL)

    def test_mismatched_lengths_resources_capture_and_work_rejected(self):
        case, device, row = self.fixture(CONTRACT['cases'][0])
        mutations = []
        for key, value in [('iterations', 32), ('iterations', True), ('work_count', row['work_count'] * 2),
                           ('capture_enabled', True), ('global_slots_per_cta', 16),
                           ('blocks', row['blocks'] + 1), ('registers_per_thread', row['registers_per_thread'] + 1)]:
            changed = copy.deepcopy(row); changed[key] = value; mutations.append((case, changed))
        changed_case = copy.deepcopy(case)
        changed_case['b3_resource_identity']['extensions']['validation_role'] = 'short_transport'
        mutations.append((changed_case, row))
        for declared, observed in mutations:
            with self.assertRaises(ValueError):
                admission.validate_trial(observed, declared, device, 3, PROTOCOL)

    def test_calibration_formula_floor_clamp_and_invalid_pilot(self):
        case, device, row = self.fixture(CONTRACT['cases'][0])
        pilot = {**row, 'iterations': 32}
        for milliseconds, expected in [(0.0005, 64000), (0.01, 64000), (0.02, 32000),
                                       (2.1, 304), (4.9, 130), (8.0, 128)]:
            resolved = admission.resolve_iterations({**pilot, 'event_ms': milliseconds}, case, device)
            self.assertEqual(resolved['resolved_iterations'], expected)
        for value in (0, -1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                admission.resolve_iterations({**pilot, 'event_ms': value}, case, device)
        with self.assertRaises(ValueError):
            admission.resolve_iterations(row, case, device)

    def test_resolved_contract_preserves_all24_semantics(self):
        contract = {**copy.deepcopy(CONTRACT), 'adapter_id': admission.ADAPTER_ID,
                    'execution_revision': admission.EXECUTION_REVISION,
                    'source': 'microbench/gh200_resource_campaign/probes/tma_bulk_formal_v1.cu',
                    'family_b3': {'CPU_fixture_only': True}}
        for case in contract['cases']:
            case['iterations'] = 128
        admission.validate_contract(contract)
        contract['cases'][0]['parameters']['copies_per_iteration'] = 2
        with self.assertRaises(ValueError):
            admission.validate_contract(contract)

    def test_real_compiled_sass_pilot_and_resolved_contracts(self):
        repo = ROOT.parents[1]
        suite = repo / 'results/gh200_resource_campaign/20261001-resource-suite-v2'
        coverage = json.loads((suite / 'reviews/evidence/S14-formal-short-B3/coverage.json').read_text())
        text = (repo / coverage['records'][0]['run_path'] / 'build/sass.stdout').read_text()
        contract = json.loads((ROOT / 'contracts/tma_bulk_admission_v1.draft.json').read_text())
        admission.audit_sass(text, contract)
        resolved = copy.deepcopy(contract)
        for case in resolved['cases']:
            case['iterations'] = 128
        admission.audit_sass(text, resolved)
        changed = re.sub(r'0x([0-9a-fA-F]{16})',
                         lambda match: '0x' + format(int(match[1], 16) ^ 1, '016x'), text, count=1)
        self.assertNotEqual(changed, text)
        with self.assertRaises(ValueError):
            admission.audit_sass(changed, contract)


if __name__ == '__main__':
    unittest.main()
