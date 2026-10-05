"""S11 CPU evidence only; no CUDA launch or synchronization timing qualification."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from auditors import synchronization as audit


class SynchronizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((BASE / 'contracts/synchronization.json').read_text())
        cls.protocol = json.loads((BASE / 'contracts/protocol.json').read_text())
        cls.device = {
            'schema_version': 2, 'type': 'device', 'name': 'NVIDIA GH200', 'cc': '9.0',
            'uuid': 'GPU-00000000-0000-0000-0000-000000000001', 'sms': 132,
            'driver_version': 13010, 'runtime_version': 12090,
            'smem_per_cta_optin_bytes': 232448,
        }

    def test_contract_matrix_and_work_units(self):
        audit.validate_contract(self.contract)
        self.assertEqual(len(self.contract['cases']), 13)
        for case in self.contract['cases']:
            phases = case['iterations'] * case['parameters']['batch']
            self.assertEqual(phases, 16384 if case['parameters']['mode'] in
                             ('warp', 'cta', 'mbarrier') else 65536)

    def test_contract_wrong_arrivals_scope_unit_and_phase_count_rejected(self):
        mbarrier_index = next(i for i, c in enumerate(self.contract['cases'])
                             if c['parameters']['mode'] == 'mbarrier')
        for mutate in (
            lambda c: c['cases'].pop(),
            lambda c: c['cases'].append(deepcopy(c['cases'][0])),
            lambda c: c['cases'][0]['parameters'].update(batch=1),
            lambda c: c['cases'][0].update(work_unit='phase'),
            lambda c: c['cases'][0].update(scope='all_gpu'),
            lambda c: c['cases'][mbarrier_index]['parameters'].update(arrival_count=1),
            lambda c: c['cases'][mbarrier_index]['parameters'].update(wait_timeout_ns_per_thread=0),
        ):
            changed = deepcopy(self.contract)
            mutate(changed)
            with self.assertRaises(ValueError):
                audit.validate_contract(changed)

    def test_affine_oracle_against_literal_modular_recurrence(self):
        for seed in (0, 1, 0xffffffff):
            value = seed
            for n in range(1025):
                self.assertEqual(audit.advance(seed, n), value)
                value = (1664525 * value + 1013904223) % (1 << 32)

    def test_actual_cpp_reference_with_large_counts_and_seed_wrap(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('CPU compiler unavailable')
        source = r'''
#include <iostream>
#include "synchronization_reference.hpp"
int main() {
  unsigned seed, thread, threads; int mode, skew; unsigned long long phases;
  while (std::cin >> mode >> skew >> thread >> threads >> seed >> phases) {
    std::cout << synchronization_reference::expected(
      mode, bool(skew), thread, threads, seed, phases) << '\n';
  }
}
'''
        specs, expected = [], []
        for case in self.contract['cases']:
            for seed in (0, 3, 0xffffffff):
                for thread in (0, 15, 16, case['threads'] - 1):
                    for phases in (1, 2, case['iterations'] * 8):
                        specs.append((audit.MODES[case['parameters']['mode']],
                                      int(case['parameters']['skew']), thread,
                                      case['threads'], seed, phases))
                        expected.append(audit.expected_value(case, seed, thread, phases))
        with tempfile.TemporaryDirectory(prefix='gh200-sync-reference-') as directory:
            root = Path(directory)
            (root / 'test.cpp').write_text(source)
            subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(BASE / 'common'),
                            str(root / 'test.cpp'), '-o', str(root / 'test')],
                           check=True, capture_output=True, timeout=30)
            output = subprocess.run([str(root / 'test')],
                                    input='\n'.join(' '.join(map(str, s)) for s in specs),
                                    text=True, check=True, capture_output=True, timeout=30)
        self.assertEqual(list(map(int, output.stdout.split())), expected)

    def thread_rows(self, case, phases, smid, correctness=False, diagnostic=False):
        rows = []
        for t in range(case['threads']):
            mbarrier = case['parameters']['mode'] == 'mbarrier'
            row = {
                'thread_id': t, 'value': audit.expected_value(case, 3, t, phases),
                'completed': phases, 'wait_attempts': phases * (2 if correctness else 1)
                if mbarrier else 0,
                'timeout': 0, 'errors': 0, 'checked': phases if correctness else 0,
                'smid': smid,
            }
            if diagnostic:
                row.update(arrival_cycle=1000 + t * 2, departure_cycle=3000 + t)
            rows.append(row)
        return rows

    def trial(self, case):
        phases = case['iterations'] * 8
        return {
            'schema_version': 2, 'type': 'trial', 'case_id': case['id'],
            'iterations': case['iterations'], 'seed': 3, 'threads': case['threads'],
            'blocks': 1, 'scope': 'one_cta', 'errors': 0,
            'correctness': {'method': audit.METHOD, 'checked_elements': case['threads'],
                            'input_conditions': audit.INPUTS},
            'work_unit': 'operation', 'work_count': phases,
            'read_payload_bytes': 0, 'write_payload_bytes': 0,
            'start_ns': 1000, 'stop_ns': 2000, 'event_ms': 0.002,
            'warmup_samples_ns': [1000] * 8, 'warmup_converged': True,
            'blocks_detail': [{'block_id': 0, 'smid': 124, 'start_ns': 1000,
                              'stop_ns': 2000, 'start_cycle': 10, 'stop_cycle': 32778}],
            'cache_residency_proven': False, 'physical_hbm_bytes_proven': False,
            'async_completion_proven': False, 'phase': 'measure',
            'timing_model': 'synchronization_phase_window_v1',
            'registers_per_thread': 32, 'static_smem_bytes': 4096, 'local_size_bytes': 0,
            'thread_results': self.thread_rows(case, phases, 124),
            'correctness_validation': {'phases': 2, 'errors': 0,
                                       'threads': self.thread_rows(case, 2, 125, correctness=True)},
            'arrival_diagnostic': {
                'separate_launch': True, 'phases': 1, 'clock_basis': 'same_sm_clock64',
                'arrival_spread_cycles': (case['threads'] - 1) * 2,
                'threads': self.thread_rows(case, 1, 126, diagnostic=True),
            },
        }

    def test_all_cases_accept_separate_launch_smid_and_collective_counts(self):
        for case in self.contract['cases']:
            result = audit.validate_trial(self.trial(case), case, self.device, 3, self.protocol)
            self.assertEqual(result['unit'], case['metric']['unit'])
            self.assertEqual(result['value'], 32768 / (case['iterations'] * 8))

    def test_bad_thread_counts_timeout_recurrence_and_diagnostics_rejected(self):
        case = next(c for c in self.contract['cases']
                    if c['parameters']['mode'] == 'mbarrier' and c['parameters']['skew'])
        for mutate in (
            lambda r: r.update(work_count=r['work_count'] * case['threads']),
            lambda r: r['thread_results'][0].update(wait_attempts=0),
            lambda r: r['thread_results'][0].update(timeout=1),
            lambda r: r['thread_results'][0].update(completed=1),
            lambda r: r['thread_results'][-1].update(value=0),
            lambda r: r['thread_results'].pop(),
            lambda r: r['arrival_diagnostic'].update(separate_launch=False),
            lambda r: r['arrival_diagnostic'].update(arrival_spread_cycles=0),
            lambda r: r['arrival_diagnostic']['threads'][0].update(smid=2),
            lambda r: r['correctness_validation']['threads'][0].update(checked=1),
            lambda r: r.update(async_completion_proven=True),
        ):
            row = self.trial(case)
            mutate(row)
            with self.assertRaises(ValueError):
                audit.validate_trial(row, case, self.device, 3, self.protocol)

    def test_abort_protocol_model_all_arrival_orders_join_same_final_barrier(self):
        # Model the actual entry/wait abort checks. This is scheduling logic,
        # not a proof of CUDA memory ordering or of the ISA implementation.
        n, phases = 3, 3
        for order in itertools.permutations(range(n)):
            for abort_at in range(24):
                phase = [0] * n
                state = ['entry'] * n
                arrivals = [set() for _ in range(phases)]
                abort = False
                for step in range(60):
                    if step == abort_at:
                        abort = True
                    t = order[step % n]
                    if state[t] == 'joined':
                        continue
                    if abort:
                        state[t] = 'joined'
                    elif state[t] == 'entry':
                        arrivals[phase[t]].add(t)
                        state[t] = 'waiting'
                    elif len(arrivals[phase[t]]) == n:
                        phase[t] += 1
                        state[t] = 'joined' if phase[t] == phases else 'entry'
                self.assertTrue(all(s == 'joined' for s in state))

    def test_source_timeout_exit_and_diagnostic_separation(self):
        source = (BASE / 'probes/synchronization.cu').read_text()
        target = source.split('bool target_sync(', 1)[1].split('template<int Mode, int Threads', 1)[0]
        body = source.split('void sync_body(', 1)[1].split('using SyncKernel', 1)[0]
        self.assertIn('atomicExch(abort_flag, 1u)', target)
        self.assertGreaterEqual(target.count('atomicAdd(abort_flag, 0u)'), 2)
        self.assertNotIn('__syncthreads()', target)
        self.assertNotIn('return;', body)
        self.assertIn('if (!abort_flag)', body)
        self.assertIn('if constexpr (Purpose == 2) result.arrival_cycle', body)
        self.assertIn('if constexpr (Purpose == 2) result.departure_cycle', body)
        self.assertIn('Purpose == 0 ? 8 : 1', body)
        self.assertIn('execute(c.correctness, 2, 1)', source)
        self.assertIn('execute(c.arrival, 1, 2)', source)

    def test_missing_sass_evidence_is_not_gpu_or_hardware_pass(self):
        with self.assertRaises(ValueError):
            audit.audit_sass('', self.contract)


if __name__ == '__main__':
    unittest.main()
