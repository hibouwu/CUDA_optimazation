"""B/C真实计算交接的独立数值、代次和观察边界检查。"""
import tempfile
import unittest
import subprocess
from pathlib import Path

import numpy as np

import cfg_b_handoff_chain as handoff
import cfg_b_compute_chain as resident
from mbarrier_service import BarrierPhase


class HandoffChainTests(unittest.TestCase):
    def test_normal_wait0_waits_current_and_releases_previous_slot(self):
        wait0 = handoff.timed_lifecycle(4, 6, 0, completion_lag_cycles=95.,
                                      release_visibility_cycles=275., wait_retain=0)
        wait1 = handoff.timed_lifecycle(4, 6, 0, completion_lag_cycles=95.,
                                      release_visibility_cycles=275.)
        self.assertEqual(wait0['local_end_cycle'], 2608.)
        self.assertEqual(wait0['local_end_cycle']-wait1['local_end_cycle'], 190.)
        rows = wait0['timeline']
        self.assertIsNone(rows[0]['wait0_return_cycle'])
        self.assertEqual(rows[1]['wait0_return_cycle'], rows[1]['compute_complete_cycle'])
        self.assertEqual(rows[0]['release_emit_cycle'], rows[1]['wait0_return_cycle'])
        graph = handoff.lifecycle(4, 6, 0, wait_retain=0)
        for node in graph['nodes']:
            if node['kind'] == 'wait0':
                current_complete = next(n['id'] for n in graph['nodes']
                    if n['kind'] == 'group_complete' and n['tile'] == node['tile'])
                self.assertIn(current_complete, node['predecessors'])

    def test_timed_full_cost_uses_entry_state_and_feeds_compute_and_release(self):
        baseline = handoff.timed_lifecycle(13, 6, 4096)
        delayed = handoff.timed_lifecycle(13, 6, 4096, full_blocked_return_cycles=275)
        self.assertEqual(delayed['timeline'][6]['full_wait_path'], 'blocked')
        self.assertEqual(delayed['timeline'][6]['full_return_cycle'],
                         delayed['timeline'][6]['full_eligibility_cycle'] + 275)
        self.assertEqual(delayed['timeline'][0]['full_return_cost_cycle'], 0)
        self.assertEqual(delayed['local_end_cycle']-baseline['local_end_cycle'], 275)
        self.assertGreater(delayed['timeline'][5]['release_emit_cycle'],
                           baseline['timeline'][5]['release_emit_cycle'])

    def test_timed_chain_wait1_releases_previous_and_tail_releases_last(self):
        report = handoff.timed_lifecycle(4, 6, 4096)
        rows = report['timeline']
        self.assertEqual([row['commit_cycle'] for row in rows], [256, 768, 1280, 1792])
        self.assertEqual([row['compute_complete_cycle'] for row in rows], [512, 1024, 1536, 2048])
        self.assertEqual([row['release_emit_cycle'] for row in rows], [768, 1280, 1792, 2048])
        self.assertIsNone(rows[0]['wait1_return_cycle'])
        self.assertTrue(all(row['full_publish_cycle'] == 0 and row['acquire_return_cycle'] is None
                            for row in rows))
        self.assertEqual(report['full_transaction_bytes'], 0)

    def test_timed_visibility_holds_producer_without_delaying_consumer_continuation(self):
        before = handoff.timed_lifecycle(13, 6, 0)['timeline']
        after = handoff.timed_lifecycle(13, 6, 0, release_visibility_cycles=4096)['timeline']
        self.assertEqual(after[0]['consumer_continue_cycle'], before[0]['consumer_continue_cycle'])
        self.assertEqual(after[0]['release_emit_cycle'], before[0]['release_emit_cycle'])
        self.assertEqual(after[6]['empty_ready_cycle'], after[0]['release_emit_cycle'] + 4096)
        self.assertEqual(after[6]['acquire_return_cycle'], after[6]['empty_ready_cycle'])
        self.assertGreater(after[6]['full_return_cycle'], before[6]['full_return_cycle'])

    def test_timed_producer_control_precedes_empty_and_delay_follows_acquire(self):
        rows = handoff.timed_lifecycle(13, 6, 4096, producer_advance_cycles=1000)['timeline']
        self.assertEqual(rows[6]['acquire_enter_cycle'], 1000)
        self.assertEqual(rows[6]['acquire_return_cycle'], max(1000, rows[0]['empty_complete_cycle']))
        self.assertEqual(rows[6]['full_publish_cycle'], rows[6]['acquire_return_cycle'] + 4096)
        self.assertEqual(rows[7]['acquire_enter_cycle'], rows[6]['full_publish_cycle'] + 1000)
        self.assertEqual(rows[7]['full_publish_cycle'], rows[7]['acquire_return_cycle'])

    def test_timed_lag_does_not_replace_service_frontier(self):
        report = handoff.timed_lifecycle(13, 6, 0, completion_lag_cycles=95)
        rows = report['timeline']
        self.assertEqual([row['service_end_cycle'] for row in rows], [512*(i+1) for i in range(13)])
        self.assertEqual(report['tail_wait0_return_cycle'], 13*512+95)
        self.assertEqual(rows[0]['release_emit_cycle'], 768)

    def test_timed_zero_control_matches_existing_compute_engine(self):
        import supply_model
        for stages in (4, 6):
            for length in (4, 13, 32):
                report = handoff.timed_lifecycle(length, stages, 0)
                actor = supply_model.shared_stage_events(length, stages, 1,
                    service_ns=0., read_latency_ns=0., compute_ns=512., commit_lead_ns=256.,
                    output_counts=[1])['ctas'][0]
                for local, original in (('compute_begin_cycle', 'starts'), ('commit_cycle', 'commits'),
                                         ('compute_complete_cycle', 'completed'), ('release_emit_cycle', 'released')):
                    self.assertEqual([row[local] for row in report['timeline']], actor[original])
                self.assertEqual(report['tail_wait0_return_cycle'], actor['main_ends'][0])

    def test_incremental_ring_matches_runtime_slot_generation_and_releases(self):
        source = (Path(__file__).parent/'probes/cfg_b_handoff_chain.cu').read_text()
        ring = 'struct RingIndex {' + source.split('struct RingIndex {', 1)[1].split('\n};', 1)[0] + '\n};'
        ring = ring.replace('__host__ __device__ ', '')
        program = '#include <iostream>\n' + ring + r'''
int main() {
  for (int stages : {4, 6}) for (int length : {1, 2, stages+1, 13, 32}) {
    RingIndex consumer{0, 0};
    int previous_slot = 0;
    for (int tile = 0; tile < length; ++tile) {
      std::cout << stages << ' ' << length << " C " << tile << ' '
                << consumer.slot << ' ' << consumer.generation << ' '
                << (consumer.generation & 1) << ' '
                << (tile ? previous_slot : -1) << '\n';
      previous_slot = consumer.slot;
      consumer.advance(stages);
    }
    std::cout << stages << ' ' << length << " T " << length-1 << ' '
              << previous_slot << " 0 0 " << previous_slot << '\n';
    RingIndex producer{0, 1};
    for (int tile = stages; tile < length; ++tile) {
      std::cout << stages << ' ' << length << " P " << tile << ' '
                << producer.slot << ' ' << producer.generation << ' '
                << ((producer.generation+1) & 1) << " -1\n";
      producer.advance(stages);
    }
  }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'ring.cpp').write_text(program)
            subprocess.run(['g++', '-std=c++17', str(path/'ring.cpp'), '-o', str(path/'ring')],
                           check=True, capture_output=True, text=True)
            lines = subprocess.run([str(path/'ring')], check=True, capture_output=True, text=True).stdout.splitlines()
        for line in lines:
            stages, length, role, tile, slot, generation, phase, release = line.split()
            stages, length, tile, slot, generation, phase, release = map(int,
                (stages, length, tile, slot, generation, phase, release))
            self.assertEqual(slot, tile % stages)
            if role == 'T':
                self.assertEqual((tile, release), (length-1, (length-1) % stages))
            else:
                self.assertEqual(generation, tile // stages)
                self.assertEqual(phase, ((tile // stages)+(role == 'P')) & 1)
                self.assertEqual(release, (tile-1) % stages if role == 'C' and tile else -1)

    def test_index_variant_keeps_one_kernel_and_shared_handoff_operations(self):
        source = (Path(__file__).parent/'probes/cfg_b_handoff_chain.cu').read_text()
        kernel = source.split('void handoff_chain(', 1)[1].split('\nstruct ChainOptions', 1)[0]
        self.assertEqual(source.count('void handoff_chain('), 1)
        self.assertEqual(kernel.count('else { slot = tile % stages; gen = tile / stages; }'), 2)
        self.assertEqual(kernel.count('index.advance(stages);'), 2)
        self.assertIn('previous_slot = slot;', kernel)
        self.assertEqual(kernel.count('if constexpr (IncrementalIndex) Barrier::arrive(empty + previous_slot, 0, 1);'), 1)
        self.assertEqual(kernel.count('if constexpr (IncrementalIndex || ConsumerOnlyIndex) Barrier::arrive(empty + previous_slot, 0, 1);'), 1)
        self.assertIn('Barrier::wait(empty + slot, (gen + 1) & 1);', kernel)
        self.assertIn('Barrier::try_wait(full + slot, gen & 1);', kernel)
        self.assertIn('if (generation[slot] != U32(gen)) ++errors;', kernel)
        self.assertIn('bool trace = false, incremental_index = false, consumer_only_index = false;', source)
        self.assertIn('key == "--index-mode"', source)

    def test_consumer_only_leaves_producer_and_tail_runtime(self):
        source = (Path(__file__).parent/'probes/cfg_b_handoff_chain.cu').read_text()
        producer = source.split('RingIndex index{0, 1};', 1)[1].split('} else if (threadIdx.x < ConsumerThreads)', 1)[0]
        consumer = source.split('RingIndex index{0, 0};', 1)[1].split('asm volatile("wgmma.wait_group.sync.aligned 0;', 1)[0]
        tail = source.split('asm volatile("wgmma.wait_group.sync.aligned 0;', 1)[1].split('float checksum = 0;', 1)[0]
        self.assertNotIn('ConsumerOnlyIndex', producer)
        self.assertIn('if constexpr (IncrementalIndex) { slot = index.slot; gen = index.generation; }', producer)
        self.assertIn('else { slot = tile % stages; gen = tile / stages; }', producer)
        self.assertIn('if constexpr (IncrementalIndex || ConsumerOnlyIndex) { slot = index.slot; gen = index.generation; }', consumer)
        self.assertIn('if constexpr (IncrementalIndex || ConsumerOnlyIndex) Barrier::arrive(empty + previous_slot, 0, 1);', consumer)
        self.assertNotIn('ConsumerOnlyIndex', tail)
        self.assertIn('else Barrier::arrive(empty + (ktiles - 1) % stages, 0, 1);', tail)
        self.assertIn('consumer_only_index = value == "consumer-only";', source)
        self.assertIn('run_chain<false, false, true>(options)', source)

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
