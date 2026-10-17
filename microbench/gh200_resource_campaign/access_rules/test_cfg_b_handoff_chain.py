"""B/C真实计算交接的独立数值、代次和观察边界检查。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import cfg_b_handoff_chain as handoff
import cfg_b_compute_chain as resident
from mbarrier_service import BarrierPhase


class HandoffChainTests(unittest.TestCase):
    def test_finite_conditions_and_same_binary_pair(self):
        rows = handoff.cases(2048)
        self.assertEqual(len(rows), 14)
        self.assertEqual(sum(row['split'] == 'dev' for row in rows)*3, 30)
        self.assertEqual(sum(row['split'] == 'heldout' for row in rows)*3, 12)
        for row in rows:
            self.assertEqual(row['wgmma_count'], 8*row['ktiles'])
            self.assertEqual(row['work_flop'], 2097152*row['ktiles'])
            if row['variant'] == 'C':
                self.assertGreater(row['ktiles'], row['stages'])
                pair = next(other for other in rows if other['variant'] == 'B' and
                    all(other[key] == row[key] for key in ('stages', 'ktiles', 'trace')))
                before = handoff.probe_command('/probe', pair, 3)
                after = handoff.probe_command('/probe', row, 3)
                position = before.index('--delay-cycles')+1
                self.assertEqual([(i, x, y) for i, (x, y) in enumerate(zip(before, after)) if x != y],
                                 [(position, '0', '2048')])

    def test_six_slot_numeric_reference_matches_resident_a(self):
        for length in (1, 2, 4, 7, 13, 32):
            for seed in handoff.SEEDS:
                np.testing.assert_array_equal(handoff.reference(length, 6, seed),
                                              resident.reference(length, seed))

    def test_four_slot_reference_matches_scalar_integer_dot(self):
        row, column = resident.output_coordinates()
        for length in (4, 5, 13, 32):
            actual = handoff.reference(length, 4, 22)
            for position in (0, 63, 64, 8191, 16383):
                total = 0
                for tile in range(length):
                    for inner in range(64):
                        a = (7*int(row[position])+13*inner+3*(tile % 4)+3*(22 % 17)) % 17-8
                        b = (11*int(column[position])+5*inner+7*(tile % 4)+5*(22 % 17)) % 17-8
                        total += a*b
                self.assertEqual(actual[position], total/1024)

    def test_first_reuse_three_generations_and_last_release(self):
        for stages in (4, 6):
            for length in (4, 5, 7, 13, 32):
                program = handoff.lifecycle(length, stages, 2048)
                nodes = program['nodes']
                for value in nodes:
                    self.assertTrue(all(predecessor < value['id'] for predecessor in value['predecessors']))
                acquire = [node for node in nodes if node['kind'] == 'producer_acquire']
                releases = [node for node in nodes if node['kind'] == 'empty_release']
                self.assertEqual(len(acquire), length)
                self.assertEqual(len(releases), length)
                by_tile = {node['tile']: node for node in releases}
                for value in acquire:
                    tile = value['tile']
                    expected = [] if tile < stages else [by_tile[tile-stages]['id']]
                    self.assertTrue(all(predecessor in value['predecessors'] for predecessor in expected))
                    if tile:
                        last_full = next(node for node in nodes if node['kind'] == 'full_arrive' and
                                         node['tile'] == tile-1)
                        self.assertIn(last_full['id'], value['predecessors'])
                for value in releases:
                    predecessor = nodes[value['predecessors'][0]]
                    self.assertEqual(predecessor['kind'], 'tail_wait0' if value['tile'] == length-1 else 'wait1')
                    self.assertEqual(value['arrivals'], 128)
                for slot in range(stages):
                    count = (length+stages-1-slot)//stages
                    self.assertEqual(program['empty'][slot]['completed_phases'], count)
                    self.assertEqual(program['full'][slot]['completed_phases'], count)

    def test_only_first_reuse_gets_controlled_delay(self):
        for stages in (4, 6):
            before = handoff.lifecycle(13, stages, 0)
            after = handoff.lifecycle(13, stages, 2048)
            delays = [node for node in after['nodes'] if node['kind'] == 'declared_delay']
            self.assertEqual([(node['tile'], node['generation'], node['delay_cycles']) for node in delays],
                             [(stages, 1, 2048)])
            self.assertFalse(any(node['kind'] == 'declared_delay' for node in before['nodes']))
        self.assertFalse(any(node['kind'] == 'declared_delay'
                             for node in handoff.lifecycle(4, 6, 2048)['nodes']))

    def test_missing_last_arriver_keeps_producer_blocked(self):
        barrier = BarrierPhase(128)
        self.assertTrue(barrier.wait_complete(1))
        for _ in range(127):
            barrier.arrive()
        self.assertFalse(barrier.wait_complete(0))
        barrier.arrive()
        self.assertTrue(barrier.wait_complete(0))

    def test_phase_alias_keeps_absolute_generation_witness(self):
        nodes = handoff.lifecycle(13, 6, 0)['nodes']
        first, third = [next(node for node in nodes if node['kind'] == 'full_arrive' and node['tile'] == tile)
                        for tile in (0, 12)]
        self.assertEqual(first['slot'], third['slot'])
        self.assertEqual(first['full_wait_phase'], third['full_wait_phase'])
        self.assertNotEqual(first['generation'], third['generation'])

    def test_first_try_success_does_not_determine_entry_state(self):
        pair = lambda value: dict(cycle=value, ns=value)
        producer = dict(full_before=pair(100), full_after=pair(120))
        consumer = dict(wait_before=pair(90), wait_after=pair(130), first_try_success=True)
        self.assertEqual(handoff.classify_wait(producer, consumer, False), 'entered_before_signal')
        consumer['wait_before'] = pair(110)
        self.assertEqual(handoff.classify_wait(producer, consumer, False), 'overlap')
        consumer['wait_before'] = pair(125)
        self.assertEqual(handoff.classify_wait(producer, consumer, False), 'ready_before_enter')
        self.assertEqual(handoff.classify_wait({}, {}, True), 'ready_before_window')

    def test_full_output_detects_wrong_ring_and_last_element(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'output.f32'
            expected = handoff.reference(13, 4, 41)
            expected.astype('<f4').tofile(path)
            self.assertEqual(handoff.verify_output(path, 13, 4, 41)['checked_elements'], 16384)
            with self.assertRaisesRegex(ValueError, '完整累加器'):
                handoff.verify_output(path, 13, 6, 41)
            expected[-1] += 1/1024
            expected.astype('<f4').tofile(path)
            with self.assertRaisesRegex(ValueError, '完整累加器'):
                handoff.verify_output(path, 13, 4, 41)


if __name__ == '__main__':
    unittest.main()
