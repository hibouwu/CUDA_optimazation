"""完整窗口的单位、外部依赖、覆盖和无隐含FIFO行为。"""
import copy
import unittest

from sm_joint_service import joint_service


class SmJointServiceTests(unittest.TestCase):
    def contract(self):
        return dict(pair='probe_a_b', org='cross_group_concurrent', cycles=120,
            work=dict(a_actions=128, b_actions=128, units=1),
            roles=dict(threads=256, a_group=0, b_group=1),
            consumers=dict(per_unit=2, method='retained values and peer-ready reads'),
            barriers=dict(per_unit=2, participants=256),
            sass=dict(kernel='frozen_kernel', sha256='fixture-sass'),
            identity=dict(profile_sha256='fixture-profile', source_sha256='fixture-source'),
            covers=['probe.actions', 'probe.consumers', 'probe.barriers', 'probe.drain'],
            boundary='all initial operands ready to consumed results and required async drain',
            qualification='conditional full window; no cfg_b migration')

    def test_complete_window_uses_latest_external_ready_and_cycles_per_ghz(self):
        result = joint_service(self.contract(), [3, 15, 7], 2.0)
        self.assertEqual((result['begin_ns'], result['end_ns']), (15, 75))
        self.assertEqual(result['covers'], self.contract()['covers'])
        self.assertEqual(result['boundary'], self.contract()['boundary'])
        self.assertFalse({'commit', 'compute_complete', 'release', 'service_start'} & result.keys())

    def test_calls_have_no_fifo_or_cross_call_state(self):
        first = joint_service(self.contract(), [100], 1)
        second = joint_service(self.contract(), [2], 1)
        self.assertEqual((first['begin_ns'], first['end_ns']), (100, 220))
        self.assertEqual((second['begin_ns'], second['end_ns']), (2, 122))
        self.assertEqual(joint_service(self.contract(), [2], 1), second)

    def test_results_do_not_mutate_contract_or_add_a_difference_price(self):
        contract = self.contract()
        original = copy.deepcopy(contract)
        result = joint_service(contract, [0], 1)
        result['covers'].append('another window')
        result['identity']['profile_sha256'] = 'changed'
        self.assertEqual(contract, original)
        self.assertEqual(result['end_ns'], contract['cycles'])

    def test_incomplete_contract_is_not_a_bare_opcode_price(self):
        for field in ('work', 'roles', 'consumers', 'barriers', 'sass', 'identity', 'covers', 'boundary'):
            contract = self.contract()
            del contract[field]
            with self.assertRaisesRegex(ValueError, '完整'):
                joint_service(contract, [0], 1)

    def test_invalid_unit_or_external_ready_is_rejected(self):
        for frequency in (0, -1, float('inf'), float('nan'), True):
            with self.assertRaisesRegex(ValueError, '频率'):
                joint_service(self.contract(), [0], frequency)
        for ready in ([], [-1], [float('nan')], [float('inf')]):
            with self.assertRaisesRegex(ValueError, '前驱'):
                joint_service(self.contract(), ready, 1)
        contract = self.contract()
        contract['cycles'] = -120
        with self.assertRaisesRegex(ValueError, 'cycle'):
            joint_service(contract, [0], 1)


if __name__ == '__main__':
    unittest.main()
