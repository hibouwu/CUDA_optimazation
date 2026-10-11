"""CPU checks for source boundaries which must survive main composition."""
from collections import defaultdict, deque
import copy
import gzip
import json
from pathlib import Path
import unittest
import analyze

BASE = Path(__file__).parent
RUN = BASE.parent.parent


def ancestors(graph, target):
    parents = defaultdict(set)
    for before, after, rule in graph['edges']:
        parents[after].add(before)
    result = set()
    queue = deque(parents[target])
    while queue:
        node = queue.popleft()
        if node not in result:
            result.add(node)
            queue.extend(parents[node])
    return result


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.load(gzip.open(RUN / 'reanalysis/B-20261011-boundary-binding-v1/cfg_b_ldb3000-persistent-ledger.json.gz', 'rt'))
        cls.rows = [analyze.bind_output(row, cls.original['shape']) for row in cls.original['outputs']]
        cls.cta0 = [row for row in cls.rows if row['cta'] == 0]
        cls.graph = analyze.make_cta_graph(cls.cta0)

    def test_source_subtile_order_is_not_B_coordinate_sort(self):
        row = self.cta0[0]
        self.assertEqual([b['origin'] for b in row['D_boxes_source_order']],
                         [[0,0],[64,0],[0,32],[64,32],[0,64],[64,64],[0,96],[64,96]])
        self.assertNotEqual(row['D_boxes_source_order'][1]['origin'], self.original['outputs'][0]['D_boxes'][1]['origin'])

    def test_total_valid_area_and_boundary_classes(self):
        boxes = [box for row in self.rows for box in row['D_boxes_source_order']]
        self.assertEqual(len(boxes), 4032)
        self.assertEqual(sum(box['global_valid_bytes'] for box in boxes), 31_200_000)
        self.assertEqual(sum(box['class'] == 'zero_valid' for box in boxes), 178)
        self.assertEqual(set(box['class'] for box in boxes), {'full', 'partial', 'zero_valid'})
        self.assertIn([40,24], [box['valid_shape'] for box in boxes])

    def test_zero_valid_protocol_is_retained_and_unpriced(self):
        boxes = [box for row in self.rows for box in row['D_boxes_source_order'] if box['class'] == 'zero_valid']
        for box in boxes:
            self.assertTrue(box['register_and_smem_preparation_required'])
            self.assertTrue(box['fence_and_sync_and_store_path_required'])
            self.assertTrue(box['read_completion_obligation_required'])
            self.assertFalse(box['global_valid_write_obligation_required'])
            self.assertIsNone(box['physical_smem_read_bytes'])
            self.assertIsNone(box['time_price'])

    def test_extra_pingpong_state_advance_and_order_phases(self):
        self.assertEqual([r['store_state']['initial_count'] for r in self.cta0], [0,8,16,24])
        self.assertEqual([r['store_state']['after_pingpong_extra_advance_count'] for r in self.cta0], [16,24,32,40])
        self.assertEqual([r['order_barrier']['wait_phase'] for r in self.cta0], [1,0,0,1])
        self.assertEqual([b['store_state_slot'] for b in self.cta0[2]['D_boxes_source_order']], [0,1,0,1,0,1,0,1])
        self.assertEqual(self.cta0[2]['D_boxes_source_order'][0]['store_state_generation'], 8)

    def test_T3_and_T4_graphs_have_no_cycle(self):
        for cta in [0,107,108,131]:
            rows = [row for row in self.rows if row['cta'] == cta]
            expected = 4 if cta < 108 else 3
            self.assertEqual(len(rows), expected)
            self.assertGreater(analyze.check_dag(analyze.make_cta_graph(rows))['nodes'], 0)

    def test_all128_explicit_tail_to_arrival_paths(self):
        row = self.cta0[0]
        self.assertEqual(len(row['thread_ports']),128)
        self.assertEqual([p['cta_thread'] for p in row['thread_ports']], list(range(128,256)))
        all_arrivals = ancestors(self.graph, 'cta0.j0.epilogue_order.arrive_all128')
        for t in range(128):
            self.assertIn(f'cta0.j0.store_tail.return.t{t}', all_arrivals)
            self.assertIn(f'cta0.j0.epilogue_order.arrive.t{t}', all_arrivals)

    def test_nonissuing_single_thread_can_arrive_before_final_read0(self):
        read = 'cta0.j0.q7.source_read_complete.t0'
        self.assertNotIn(read, ancestors(self.graph, 'cta0.j0.epilogue_order.arrive.t96'))
        self.assertIn(read, ancestors(self.graph, 'cta0.j0.epilogue_order.arrive_all128'))
        self.assertIn(read, ancestors(self.graph, 'cta0.j0.store_tail.return_all128'))
        self.assertIn('cta0.j0.q6.source_read_complete.t0', ancestors(self.graph, 'cta0.j0.epilogue_order.arrive.t96'))

    def test_read1_all_older_groups_but_latest_can_remain(self):
        r2s_q2 = ancestors(self.graph, 'cta0.j0.q2.r2s_ready_all128')
        self.assertIn('cta0.j0.q0.source_read_complete.t0', r2s_q2)
        self.assertNotIn('cta0.j0.q1.source_read_complete.t0', r2s_q2)
        wait1_q3 = ancestors(self.graph, 'cta0.j0.q3.read1.return.t0')
        for q in range(3):
            self.assertIn(f'cta0.j0.q{q}.source_read_complete.t0', wait1_q3)
        self.assertNotIn('cta0.j0.q3.source_read_complete.t0', wait1_q3)

    def test_read_completion_need_not_follow_commit(self):
        self.assertNotIn('cta0.j0.q0.commit.t0', ancestors(self.graph, 'cta0.j0.q0.source_read_complete.t0'))
        self.assertIn('cta0.j0.q0.issue.t0', ancestors(self.graph, 'cta0.j0.q0.source_read_complete.t0'))

    def test_next_WG_epilogue_wait_requires_all128_prior_arrivals(self):
        parents = ancestors(self.graph, 'cta0.j1.epilogue_order.wait.return.t0')
        for t in range(128):
            self.assertIn(f'cta0.j0.epilogue_order.arrive.t{t}', parents)
        self.assertIn('cta0.j1.mma_tail.return.t0', parents)
        entry = ancestors(self.graph, 'cta0.j1.D.entry_all128')
        self.assertIn('cta0.j1.mma_tail.return.t0', entry)
        self.assertIn('cta0.j1.epilogue_order.wait.return.t0', entry)

    def test_each_epilogue_wait_follows_its_own_mma_tail(self):
        for j in range(4):
            for t in [0,31,32,96,127]:
                wait = ancestors(self.graph, f'cta0.j{j}.epilogue_order.wait.return.t{t}')
                self.assertIn(f'cta0.j{j}.mma_tail.return.t{t}', wait)
        all_waits = ancestors(self.graph, 'cta0.j0.epilogue_order.wait.return_all128')
        for t in range(128):
            self.assertIn(f'cta0.j0.mma_tail.return.t{t}', all_waits)

    def test_same_WG_next_math_wait_is_per_thread_and_separate_from_epilogue(self):
        parents = ancestors(self.graph, 'cta0.j2.math_order.wait.return.t96')
        self.assertIn('cta0.j0.epilogue_order.arrive.t96', parents)
        self.assertNotIn('cta0.j0.epilogue_order.arrive.t0', parents)
        self.assertIn('cta0.j1.math_order.arrive_all128', parents)
        self.assertNotIn('cta0.j1.epilogue_order.arrive_all128', parents)
        self.assertNotIn('cta0.j0.q7.source_read_complete.t0', parents)
        all_permission = ancestors(self.graph, 'cta0.j2.same_wg_next_mma_permission_all128')
        for t in range(128):
            self.assertIn(f'cta0.j0.epilogue_order.arrive.t{t}', all_permission)

    def test_initial_next_WG_MMA_handoff_does_not_wait_for_D_tail(self):
        parents = ancestors(self.graph, 'cta0.j1.math_order.wait.return.t0')
        self.assertIn('cta0.j0.math_order.arrive_all128', parents)
        self.assertNotIn('cta0.j0.mma_tail.return_all128', parents)
        self.assertNotIn('cta0.j0.store_tail.return_all128', parents)
        self.assertNotIn('cta0.j0.q7.source_read_complete.t0', parents)
        self.assertIn('cta0.math_order.initial_phase_ready', parents)

    def test_D_visibility_is_separate_from_read0_and_handoff(self):
        visible = 'cta0.j0.q7.global_valid_writes_visible'
        self.assertNotIn(visible, ancestors(self.graph, 'cta0.j0.store_tail.return_all128'))
        self.assertNotIn(visible, ancestors(self.graph, 'cta0.j0.D.exit_all128'))
        self.assertNotIn('cta0.j0.store_tail.return_all128', ancestors(self.graph, visible))
        self.assertIn(visible, ancestors(self.graph, 'cta0.cuda_kernel_completion_requirement'))

    def test_producer_tail_is_not_global_D_completion(self):
        self.assertNotIn('cta0.input.producer_tail_return', ancestors(self.graph, 'cta0.j0.q7.global_valid_writes_visible'))
        self.assertNotIn('cta0.j0.q7.global_valid_writes_visible', ancestors(self.graph, 'cta0.input.producer_tail_return'))
        closed = ancestors(self.graph, 'cta0.source_lifetimes_closed')
        self.assertIn('cta0.input.producer_tail_return', closed)
        self.assertIn('cta0.j2.D.exit_all128', closed)
        self.assertIn('cta0.j3.D.exit_all128', closed)
        self.assertNotIn('cta0.j3.q7.global_valid_writes_visible', closed)

    def test_zero_valid_box_keeps_read_but_has_no_valid_write_node(self):
        target = next(row for row in self.rows if any(b['class'] == 'zero_valid' for b in row['D_boxes_source_order']))
        rows = [row for row in self.rows if row['cta'] == target['cta']]
        graph = analyze.make_cta_graph(rows)
        names = {node['id'] for node in graph['nodes']}
        for box in target['D_boxes_source_order']:
            prefix = f"cta{target['cta']}.j{target['output_ordinal']}.q{box['subtile_ordinal']}"
            self.assertIn(prefix + '.source_read_complete.t0', names)
            self.assertEqual(prefix + '.global_valid_writes_visible' in names, box['global_valid_write_obligation_required'])

    def test_invalid_geometry_and_incomplete_CTA_are_rejected(self):
        source = self.original['outputs'][0]
        with self.assertRaises(ValueError): analyze.bind_output(source, [2600,3008,2000])
        wrong = copy.deepcopy(source)
        wrong['consumer_warp_group'] = 1
        with self.assertRaises(ValueError): analyze.bind_output(wrong, self.original['shape'])
        wrong = copy.deepcopy(source)
        wrong['D_boxes'][0]['global_valid_bytes'] = 0
        with self.assertRaises(ValueError): analyze.bind_output(wrong, self.original['shape'])
        with self.assertRaises(ValueError): analyze.make_cta_graph(self.cta0[1:])
        with self.assertRaises(ValueError): analyze.make_cta_graph(self.cta0 + [self.rows[-1]])

    def test_no_duration_or_clock_is_supplied(self):
        self.assertTrue(all(node['time_price'] is None for node in self.graph['nodes']))
        text = json.dumps(self.graph)
        for word in ['GT_pre', 'GT_post', 'clock64', 'compute_ns', 'duration_ns']:
            self.assertNotIn(word, text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
