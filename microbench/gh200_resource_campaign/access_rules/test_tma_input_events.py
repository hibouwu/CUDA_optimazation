import unittest
from tma_input_events import Costs, GroupCosts, simulate


class InputEventsTest(unittest.TestCase):
    def test_serial_known_answer(self):
        result = simulate('s14', 3, 1, 1, Costs(10, 20, 2, 5))
        self.assertEqual(result['cycles'], 105)

    def test_credit_return_precedes_software_release(self):
        r = simulate('s16', 4, 2, 2, Costs(10, 20, 1, 8), 1, trace=True)
        req = [e for e in r['events'] if e['kind'] == 'request']
        groups = [e for e in r['events'] if e['kind'] == 'group']
        self.assertEqual(req[1]['admit'], req[0]['ready'])
        self.assertLess(req[1]['admit'], groups[0]['slot_release'])
        self.assertEqual(r['max_inflight'], 1)
        for g in groups:
            self.assertGreaterEqual(g['wait_return'], g['ready'])
        for e in req:
            if e['group'] >= 2:
                previous = groups[e['group'] - 2]
                self.assertGreaterEqual(e['eligible'], previous['slot_release'])

    def test_r05_drains_every_eight(self):
        costs = Costs(10, 70, 2, 4, 3)
        one = simulate('r05', 8, 4, 1, costs, 2, consumer=16)
        two = simulate('r05', 16, 4, 1, costs, 2, consumer=16, trace=True)
        self.assertEqual(two['cycles'], 2 * one['cycles'])
        req8 = next(e for e in two['events'] if e['kind'] == 'request' and e['group'] == 8)
        group7 = next(e for e in two['events'] if e['kind'] == 'group' and e['group'] == 7)
        self.assertEqual(req8['eligible'], group7['slot_release'])

    def test_inactive_cap_equivalent_and_group_ready(self):
        args = ('s16', 9, 2, 4, Costs(10, 90, 1, 8))
        a = simulate(*args, trace=True)
        b = simulate(*args, credits=100, trace=True)
        self.assertEqual(a, b)
        for g in (e for e in a['events'] if e['kind'] == 'group'):
            requests = [e for e in a['events'] if e['kind'] == 'request' and e['group'] == g['group']]
            self.assertEqual(g['ready'], max(e['ready'] for e in requests))

    def test_initial_fill_and_final_drain(self):
        r = simulate('s16', 2, 4, 1, Costs(10, 20), trace=True)
        self.assertEqual(r['cycles'], 40)
        self.assertEqual(len([e for e in r['events'] if e['kind'] == 'group']), 2)

    def test_service_interval_and_completion_latency_are_distinct(self):
        costs = Costs(10, 90)
        pipelined = simulate('s16', 8, 4, 1, costs)
        serialized = simulate('s16', 8, 4, 1, costs, credits=1)
        self.assertEqual(pipelined['cycles'], 230)
        self.assertEqual(serialized['cycles'], 800)

    def test_explicit_group_cost_can_overlap_completion(self):
        # Five cycles after issue and five before wait fit inside latency 30.
        hidden = simulate('s14', 1, 1, 1, Costs(10, 20),
                          group_costs=GroupCosts(5, 5))
        visible = simulate('s14', 1, 1, 1, Costs(10, 20),
                           group_costs=GroupCosts(25, 10))
        self.assertEqual(hidden['cycles'], 30)
        self.assertEqual(visible['cycles'], 35)
        with self.assertRaises(ValueError):
            GroupCosts(-1, 0)

    def test_invalid_inputs(self):
        for kwargs in [dict(credits=0), dict(consumer=-1)]:
            with self.assertRaises(ValueError):
                simulate('s16', 8, 2, 1, Costs(1, 2), **kwargs)
        with self.assertRaises(ValueError):
            simulate('r05', 9, 2, 1, Costs(1, 2))
        with self.assertRaises(ValueError):
            Costs(float('nan'), 1)


if __name__ == '__main__':
    unittest.main()
