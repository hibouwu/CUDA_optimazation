import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auditors.shared_memory import validate_contract, work, addresses, validate_trial, audit_sass

ROOT = Path(__file__).resolve().parents[1]


class SharedMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((ROOT / 'contracts/shared_memory.json').read_text())
        cls.protocol = json.loads((ROOT / 'contracts/protocol.json').read_text())
        cls.device = {'sms': 132, 'smem_per_sm_bytes': 233472,
                      'smem_per_cta_optin_bytes': 232448, 'registers_per_sm': 65536}

    def test_all_fifteen_configs_and_alignment(self):
        validate_contract(self.contract)
        for case in self.contract['cases']:
            result = work(case)
            p = case['parameters']
            per_direction = 256 * 8 * p['access_bytes'] * 8192
            self.assertEqual(result['work'], per_direction * (2 if p['mode'] == 'duplex' else 1))
            if p['broadcast']:
                self.assertEqual(result['unique_bytes'], 256)

    def test_false_dependency_and_unit_rejected(self):
        for field, value in [('store_depends_on_load', True), ('arrays', 1)]:
            contract = copy.deepcopy(self.contract)
            contract['cases'][0]['parameters'][field] = value
            with self.assertRaises(ValueError):
                validate_contract(contract)
        contract = copy.deepcopy(self.contract)
        contract['cases'][0]['work_unit'] = 'FLOP'
        with self.assertRaises(ValueError):
            validate_contract(contract)

    def test_every_written_word_has_nonpoison_distinct_oracle(self):
        for case in self.contract['cases']:
            if case['parameters']['mode'] != 'read':
                indices = addresses(case['parameters'])
                self.assertEqual(len(indices), len(set(indices)))
                for seed in (3, 174, 999999):
                    self.assertTrue(all(29*i+seed+7 != 0xdeadbeef for i in indices))
                    self.assertTrue(all(29*i+seed+7 != 17*i+seed for i in indices))

    def test_realistic_row_and_tampering(self):
        case = self.contract['cases'][-1];p = case['parameters'];w = work(case)
        row = {'schema_version': 2, 'type': 'trial', 'case_id': case['id'], 'iterations': 8192,
               'seed': 3, 'threads': 256, 'blocks': 1, 'scope': 'one_cta', 'errors': 0,
               'correctness': w['correctness'], 'work_unit': 'byte', 'work_count': w['work'],
               'read_payload_bytes': w['read'], 'write_payload_bytes': w['write'],
               'start_ns': 1000, 'stop_ns': 1001000, 'event_ms': 1.1,
               'blocks_detail': [{'block_id': 0, 'smid': 137, 'start_ns': 1000, 'stop_ns': 1001000,
                                  'start_cycle': 4000, 'stop_cycle': 2004000}],
               'warmup_samples_ns': [1000000]*8, 'warmup_converged': True,
               'cache_residency_proven': False, 'physical_hbm_bytes_proven': False,
               'dynamic_smem_bytes': 65536, 'registers_per_thread': 48, 'occupancy_limit_ctas_per_sm': 3,
               'unique_bytes_per_direction_iteration': w['unique_bytes'], 'access_bytes': 16,
               'stride': 1, 'broadcast': False}
        self.assertGreater(validate_trial(row, case, self.device, 3, self.protocol)['value'], 0)
        for field in ('work_count', 'dynamic_smem_bytes', 'unique_bytes_per_direction_iteration', 'stop_ns'):
            bad = copy.deepcopy(row);bad[field] += 1
            with self.assertRaises(ValueError, msg=field):
                validate_trial(bad, case, self.device, 3, self.protocol)
        for field, value in [('occupancy_limit_ctas_per_sm', 32), ('registers_per_thread', 128)]:
            bad = copy.deepcopy(row);bad[field] = value
            with self.assertRaises(ValueError):
                validate_trial(bad, case, self.device, 3, self.protocol)
        bad = copy.deepcopy(row);bad['blocks_detail'][0]['block_id'] = 1
        with self.assertRaises(ValueError):
            validate_trial(bad, case, self.device, 3, self.protocol)

    def test_sass_rejects_access_in_forbidden_direction(self):
        for case, allowed, forbidden, mode in ((self.contract['cases'][0], 'LDS', 'STS', 0),
                                                (self.contract['cases'][8], 'STS', 'LDS', 1)):
            contract = {'cases': [case]}
            text = f'Function : shared_accessILi4ELi{mode}ELb0_suffix\n'
            body = [(16*i, f'{allowed} R0, [R2];') for i in range(8)]
            with patch('auditors.shared_memory.timed_loops', return_value=[(0, 256, body)]):
                self.assertEqual(len(audit_sass(text, contract)), 1)
            bad = body + [(144, f'{forbidden} R0, [R2];')]
            with patch('auditors.shared_memory.timed_loops', return_value=[(0, 256, bad)]):
                with self.assertRaises(ValueError):
                    audit_sass(text, contract)


if __name__ == '__main__':
    unittest.main()
