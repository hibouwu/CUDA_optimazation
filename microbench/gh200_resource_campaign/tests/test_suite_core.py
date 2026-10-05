"""CPU-only protocol negative tests; synthetic rows never count as GH200 evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common.suite_io import atomic_json, bounded, cv, file_lock, gate_mapping, process_ok, read_json, relative, sha, validate_gate
from common.suite_snapshot import freeze, check_snapshot
from auditors.memory_baseline import validate_contract, validate_trial, work, audit_sass
from runners.environment import Checkpoint, budget

CONTRACT = read_json(HERE / 'contracts/memory_baseline.json')
PROTOCOL = read_json(HERE / 'contracts/protocol.json')
DEVICE = {'schema_version': 2, 'type': 'device', 'uuid': 'GPU-12345678-1234-1234-1234-123456789abc', 'name': 'GH200 fixture', 'cc': '9.0', 'sms': 4, 'driver_version': 12090, 'runtime_version': 12090}


def raw(case):
    expected = work(case, DEVICE)
    empty = case['work_model'] == 'empty_window'
    shared = case['work_model'] == 'shared_scalar8'
    return {'schema_version': 2, 'type': 'trial', 'case_id': case['id'], 'iterations': case['iterations'], 'seed': 3,
            'threads': case['threads'], 'blocks': expected['blocks'], 'scope': case['scope'], 'errors': 0,
            'work_unit': case['work_unit'], 'work_count': expected['read'] + expected['write'], 'read_payload_bytes': expected['read'], 'write_payload_bytes': expected['write'],
            'requested_working_set_bytes': case['parameters']['bytes'], 'allocation_per_array_bytes': expected['allocation'], 'stride': case['parameters']['stride'],
            'cache_residency_proven': False, 'physical_hbm_bytes_proven': False, 'start_ns': 100, 'stop_ns': 1100, 'event_ms': .002,
            'warmup_samples_ns': [1000] * 8, 'warmup_converged': True,
            'correctness': {'method': 'empty_window_timestamps' if empty else 'host_per_thread_address_checksum' if shared else 'host_checksum_and_optional_full_array_store_validation', 'checked_elements': expected['checked'], 'input_conditions': 'no memory workload' if empty else 'shared[i]=17*i+3; write initialization poison' if shared else 'global[i]=17*i+seed modulo 2^32'},
            'blocks_detail': [{'block_id': i, 'smid': i % DEVICE['sms'], 'start_ns': 100, 'stop_ns': 1100, 'start_cycle': 200, 'stop_cycle': 2200} for i in range(expected['blocks'])]}


class ProtocolTests(unittest.TestCase):
    def test_all_fourteen_synthetic_rows_validate_arithmetic_only(self):
        validate_contract(CONTRACT)
        for case in CONTRACT['cases']:
            result = validate_trial(raw(case), case, DEVICE, 3, PROTOCOL)
            self.assertTrue(result['warmup_converged'])

    def test_wrong_work_bytes_units_timing_identity_rejected(self):
        case = CONTRACT['cases'][0]
        for key, value in [('work_count', 0), ('read_payload_bytes', 7), ('work_unit', 'FLOP'), ('seed', 9), ('iterations', 1), ('scope', 'all_gpu'), ('errors', 1), ('event_ms', .0001), ('start_ns', 101)]:
            with self.subTest(key=key):
                row = raw(case); row[key] = value
                with self.assertRaises(ValueError): validate_trial(row, case, DEVICE, 3, PROTOCOL)

    def test_duplicate_CTA_and_coverage_rejected(self):
        case = CONTRACT['cases'][7]
        for mutation in ('duplicate', 'missing_SM'):
            row = raw(case)
            if mutation == 'duplicate': row['blocks_detail'][1]['block_id'] = 0
            else:
                for block in row['blocks_detail']: block['smid'] = 0
            with self.assertRaises(ValueError): validate_trial(row, case, DEVICE, 3, PROTOCOL)

    def test_empty_grid_allows_observed_subset_without_export(self):
        case = CONTRACT['cases'][-1]; row = raw(case)
        for block in row['blocks_detail']: block['smid'] = 0
        self.assertEqual(validate_trial(row, case, DEVICE, 3, PROTOCOL)['value'], 1000)
        self.assertFalse(case['exportable'])

    def test_warmup_false_and_first_stop_consistency(self):
        case = CONTRACT['cases'][0]; row = raw(case)
        for samples, flag in [([1000]*7, True), ([1000]*9, True), ([1000]*8, False), ([1,100]*5, False)]:
            row['warmup_samples_ns'] = samples; row['warmup_converged'] = flag
            with self.assertRaises(ValueError): validate_trial(row, case, DEVICE, 3, PROTOCOL)
        row['warmup_samples_ns'] = [1,100]*15; row['warmup_converged'] = False
        self.assertFalse(validate_trial(row, case, DEVICE, 3, PROTOCOL)['warmup_converged'])

    def test_contract_unknown_model_scale_and_iteration_rejected(self):
        for field, value in [('work_model', 'eval(x)'), ('iterations', 8191), ('metric', {'numerator': 'work_count', 'denominator': 'elapsed_ns', 'scale': 1000, 'unit': 'TB/s'})]:
            contract = copy.deepcopy(CONTRACT); contract['cases'][0][field] = value
            with self.assertRaises(ValueError): validate_contract(contract)

    def test_sass_hoist_missing_loop_and_wrong_count_rejected(self):
        contract = {'sass_contracts': [{'function': 'probe', 'loop_token': 'LDS', 'count_per_unrolled_body': 1}]}
        valid = 'Function : probe\n/*0000*/ CS2R R2, SR_GLOBALTIMERLO ; /* x */\n/*0010*/ LDS R4, [R0] ; /* x */\n/*0020*/ BRA 0x10 ; /* x */\n/*0030*/ CS2R R4, SR_GLOBALTIMERLO ; /* x */'
        self.assertTrue(audit_sass(valid, contract))
        for bad in (valid.replace('LDS', 'MOV'), valid.replace('BRA 0x10', 'BRA 0x40'), valid.replace('/*0010*/ LDS', '/*0000*/ LDS')):
            with self.assertRaises(ValueError): audit_sass(bad, contract)

    def test_schema_rejects_nonfinite_and_duplicate_JSON(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'raw'
            for text in ('{"x":NaN}', '{"x":1,"x":2}'):
                path.write_text(text)
                with self.assertRaises(ValueError): read_json(path)

    def test_gate_accepts_only_two_explicit_forms_and_rejects_mutable(self):
        self.assertEqual(gate_mapping({'gate_files': {'a': '0'*64}}), {'a': '0'*64})
        self.assertEqual(gate_mapping({'gate_files': ['a'], 'gate_files_sha256': {'a': '0'*64}}), {'a': '0'*64})
        for review in ({'gate_files': ['a'], 'reviewed_files_sha256': {'a': '0'*64}}, {'gate_files': ['a'], 'gate_files_sha256': {'b': '0'*64}}, {'gate_files': {'implementation_status.json': '0'*64}}):
            with self.assertRaises(ValueError): gate_mapping(review)

    def test_actual_A_gates_validate_without_mutable_status_binding(self):
        repo = HERE.parents[1]; suite = repo / 'results/gh200_resource_campaign/20261001-resource-suite-v2'
        for stage, phase, name in [('S02','A','S02-A-review.json'), ('S04','A','S04-A-review.json'), ('S02','preflight-A','S02-preflight-A-review-r2.json')]:
            validate_gate(suite / 'reviews' / name, repo, stage, phase)

    def test_paths_reject_parent_absolute_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'x').write_text('x'); (root/'link').symlink_to(root/'x')
            for value in ('../x', '/x', 'link'):
                with self.assertRaises(ValueError): relative(root, value)

    def test_lock_conflict_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'lock'
            with file_lock(path):
                with self.assertRaises(BlockingIOError):
                    with file_lock(path): pass
            link = Path(temp)/'link'; link.symlink_to(path)
            with self.assertRaises(OSError):
                with file_lock(link): pass

    def test_timeout_cleans_own_process_and_keeps_failure_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = bounded([sys.executable, '-c', 'import time; print("before-timeout", flush=True); time.sleep(30)'], root, root/'stdout', root/'stderr', .1)
            self.assertTrue(receipt['timed_out']); self.assertTrue(receipt['cleanup_confirmed'])
            self.assertFalse(process_ok(receipt)); self.assertIn('before-timeout', (root/'stdout').read_text())

    def test_budget_checkpoint_has_cleanup_reserve(self):
        with patch('runners.environment.remaining_seconds', return_value=139):
            with self.assertRaises(Checkpoint): budget({'job':'1'}, 120)
        with patch('runners.environment.remaining_seconds', return_value=140): budget({'job':'1'}, 120)

    def test_snapshot_preserves_dependencies_and_detects_tamper(self):
        repo = HERE.parents[1]; suite = repo/'results/gh200_resource_campaign/20261001-resource-suite-v2'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Storage-only diagnostic snapshot: execution admission is tested separately.
            spec = freeze(repo, suite, root, HERE/'contracts/memory_baseline.json',
                          preflight=True, case_diagnostic=True)
            self.assertEqual(check_snapshot(root, spec)['family'], 'memory_baseline')
            (root/'snapshot/repo/microbench/gh200_resource_campaign/common/probe_runtime.cuh').write_text('tamper')
            with self.assertRaises(ValueError): check_snapshot(root, spec)


if __name__ == '__main__': unittest.main()
