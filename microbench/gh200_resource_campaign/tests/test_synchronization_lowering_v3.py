"""CPU checks against archived target SASS; never runs a GPU kernel."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
sys.path.insert(0, str(BASE))
from auditors import synchronization_lowering_v3 as audit


class LoweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((BASE / 'contracts/synchronization_lowering_v3.json').read_text())
        cls.sass = (REPO / cls.contract['sass_contracts']['observed_sass_path']).read_text()

    def mutate(self, case_id, old, new):
        symbol = self.contract['sass_contracts']['observed_kernels'][case_id]['symbol']
        chunks = self.sass.split('Function : ')
        index = next(i for i, body in enumerate(chunks)
                     if body.splitlines()[0].strip() == symbol)
        self.assertIn(old, chunks[index])
        chunks[index] = chunks[index].replace(old, new, 1)
        return 'Function : '.join(chunks)

    def rejects(self, text, phrase):
        with self.assertRaisesRegex(ValueError, phrase):
            audit.audit_sass(text, self.contract)

    def test_all13_actual_measured_functions(self):
        result = audit.audit_sass(self.sass, self.contract)
        self.assertEqual(len(result['cases']), 13)
        self.assertFalse(result['hardware_qualification'])
        for row in result['cases']:
            timers = row['boundaries']['timeout_timer_pcs']
            self.assertEqual(len(timers), 16 if row['case_id'].startswith('mbarrier') else 0)
            if row['case_id'].startswith('warp'):
                self.assertEqual(row['compiled_demand']['WARPSYNC_static_per_iteration'], 0)
                self.assertFalse(row['compiled_demand']['native_warp_barrier_exportable'])

    def test_warp_export_and_metric_cannot_revert(self):
        for field, value in [('exportable', True), ('work_model', 'synchronization_phase_v1')]:
            bad = deepcopy(self.contract)
            bad['cases'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'logical service'):
                audit.validate_contract(bad)
        bad = deepcopy(self.contract)
        bad['cases'][0]['metric']['unit'] = 'clock64_cycle/phase/CTA'
        with self.assertRaisesRegex(ValueError, 'logical phase'):
            audit.validate_contract(bad)

    def test_missing_outer_timer_rejected(self):
        bad = self.mutate('cta_t128_aligned', 'SR_GLOBALTIMERLO', 'SR_CLOCKLO')
        self.rejects(bad, 'timer count')

    def test_missing_timeout_timer_rejected(self):
        bad = self.mutate('mbarrier_t128_aligned',
                          'CS2R R8, SR_GLOBALTIMERLO', 'CS2R R8, SR_CLOCKLO')
        self.rejects(bad, 'timer count')

    def test_timeout_limit_and_abort_must_remain(self):
        for old, new in [('0x3b9aca00', '0x3b9aca01'), ('ATOMS.EXCH', 'ATOMS.ADD')]:
            bad = self.mutate('mbarrier_t128_aligned', old, new)
            self.rejects(bad, 'timeout comparison/collective abort')

    def test_target_arrival_and_wait_must_remain(self):
        for old, new, message in [
            ('SYNCS.ARRIVE.TRANS64.A1T0', 'SYNCS.EXCH.64', 'arrivals'),
            ('SYNCS.PHASECHK.TRANS64.TRYWAIT', 'SYNCS.PHASECHK.TRANS64', 'polling')]:
            self.rejects(self.mutate('mbarrier_t128_aligned', old, new), message)

    def test_shared_parser_not_used_for18_timers(self):
        symbol = self.contract['sass_contracts']['observed_kernels']['mbarrier_t128_aligned']['symbol']
        body = next(x for x in self.sass.split('Function : ')
                    if x.splitlines()[0].strip() == symbol)
        from auditors.memory_baseline import timed_loops
        with self.assertRaisesRegex(ValueError, 'two globaltimer'):
            timed_loops(body)
        boundary = audit.window_and_loop(audit.instructions(body), 'mbarrier')
        self.assertEqual(boundary['outer_loop'], [0x2d0, 0x15f0])

    def test_predicate_operand_backedge_and_changed_target(self):
        self.assertIn((0x2d0, 0x15f0), audit.backward_branches([
            (0x15f0, '@P0 BRA P2, 0x2d0 ;')]))
        bad = self.mutate('mbarrier_t128_aligned', 'BRA P2, 0x2d0', 'BRA P2, 0x2e0')
        self.rejects(bad, 'boundary map changed')

    def test_instruction_digest_catches_register_mutation(self):
        bad = self.mutate('cta_t128_aligned', 'ISETP.NE.AND P0', 'ISETP.NE.AND P2')
        self.rejects(bad, 'unreviewed CUDA12.9')

    def test_cta_fence_and_proxy_demand(self):
        self.rejects(self.mutate('cta_t128_aligned',
                                'BAR.SYNC.DEFER_BLOCKING 0x1', 'NOP'), 'barrier1')
        self.rejects(self.mutate('fence_gpu_no_outstanding_work',
                                'MEMBAR.ALL.GPU', 'MEMBAR.ALL.CTA'), 'fence issues')
        self.rejects(self.mutate('proxy_async_no_outstanding_work',
                                'FENCE.VIEW.ASYNC.S', 'NOP'), 'proxy fence')

    def test_duplicate_or_missing_function_rejected(self):
        first = self.contract['sass_contracts']['observed_kernels']['warp_t32_aligned']['symbol']
        self.rejects(self.sass.replace(first, first + '_unknown'), 'exact measured symbol')
        self.rejects(self.sass + '\nFunction : ' + first + '\n', 'exact measured symbol')


if __name__ == '__main__':
    unittest.main()
