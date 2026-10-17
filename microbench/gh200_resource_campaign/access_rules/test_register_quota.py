"""cfg_b 逐 warp 配额守恒；无 GPU、时价或程序图依赖。"""
import unittest

from register_quota import CfgBRegisterQuota, cfg_b_quota_requirements


class RegisterQuotaTests(unittest.TestCase):
    def assert_conserved(self, quota):
        state = quota.snapshot()
        allocated = sum(32*warp['registers_per_thread'] for warp in state['warps'])
        self.assertEqual(allocated, state['allocated_registers'])
        self.assertEqual(allocated + state['available_registers'], 64512)
        self.assertGreaterEqual(state['available_registers'], 0)

    def test_fixed_demands_match_initial_and_final_cta_budget(self):
        requirements = cfg_b_quota_requirements()
        self.assertEqual(requirements['threads'], 384)
        self.assertEqual(len(requirements['warps']), 12)
        self.assertEqual(sum(warp['released_registers'] for warp in requirements['warps']), 16384)
        self.assertEqual(sum(warp['needed_registers'] for warp in requirements['warps']), 16384)
        self.assertEqual(requirements['initial_pool_registers'], 64512)
        self.assertEqual(requirements['steady_allocated_registers'], 64512)
        quota = CfgBRegisterQuota()
        self.assertFalse(quota.startup_complete)
        self.assertEqual(quota.available_registers, 0)
        self.assert_conserved(quota)

    def test_initial_consumer_requests_fail_without_state_change_or_sm_borrow(self):
        quota = CfgBRegisterQuota(cta=7)
        before = quota.snapshot()
        for warp in range(4, 12):
            event = quota.acquire_consumer(warp)
            self.assertFalse(event['success'])
            self.assertEqual((event['cta'], event['needed_registers'], event['available_registers']),
                             (7, 2048, 0))
            self.assertEqual(quota.snapshot(), before)

    def test_partial_producer_release_can_supply_two_consumer_warps(self):
        quota = CfgBRegisterQuota()
        released = quota.release_producer(0)
        self.assertEqual(released['released_registers'], 4096)
        self.assertFalse(quota.snapshot()['warpgroups'][0]['instruction_complete'])
        for warp in (4, 8):
            self.assertTrue(quota.acquire_consumer(warp)['success'])
            self.assert_conserved(quota)
        self.assertEqual(quota.available_registers, 0)
        before = quota.snapshot()
        self.assertFalse(quota.acquire_consumer(5)['success'])
        self.assertEqual(quota.snapshot(), before)
        self.assertEqual([warp['registers_per_thread'] for warp in before['warps'][:4]],
                         [40, 168, 168, 168])

    def test_caller_orders_can_finish_the_same_quota_without_arbitration_rule(self):
        final_states = []
        for consumer_order in (range(4, 12), range(11, 3, -1), (4, 8, 5, 9, 6, 10, 7, 11)):
            quota = CfgBRegisterQuota()
            for warp in range(4):
                quota.release_producer(warp)
                self.assert_conserved(quota)
            self.assertEqual(quota.available_registers, 16384)
            for warp in consumer_order:
                self.assertTrue(quota.try_set(warp, 232)['success'])
                self.assert_conserved(quota)
            state = quota.snapshot()
            self.assertTrue(state['startup_complete'])
            self.assertTrue(all(group['instruction_complete'] for group in state['warpgroups']))
            self.assertEqual(state['available_registers'], 0)
            self.assertEqual([event['sequence'] for event in state['events']], list(range(12)))
            final_states.append(state['warps'])
        self.assertEqual(final_states[0], final_states[1])
        self.assertEqual(final_states[0], final_states[2])

    def test_repeated_confirmation_does_not_reallocate_across_later_work(self):
        quota = CfgBRegisterQuota()
        for warp in range(4):
            quota.release_producer(warp)
        for warp in range(4, 12):
            quota.acquire_consumer(warp)
        before = quota.snapshot()
        for _ in range(3):
            for warp in range(4, 12):
                event = quota.acquire_consumer(warp)
                self.assertTrue(event['success'])
                self.assertTrue(event['already_satisfied'])
                self.assertEqual(event['needed_registers'], 0)
        self.assertEqual(quota.snapshot(), before)

    def test_release_does_not_supply_another_cta_pool(self):
        first, second = CfgBRegisterQuota(0), CfgBRegisterQuota(1)
        first.release_producer(0)
        before = second.snapshot()
        self.assertFalse(second.acquire_consumer(4)['success'])
        self.assertEqual(second.snapshot(), before)
        self.assertEqual(first.available_registers, 4096)


if __name__ == '__main__':
    unittest.main()
