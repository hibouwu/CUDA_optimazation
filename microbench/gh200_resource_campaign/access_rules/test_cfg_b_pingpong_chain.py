"""E 真实三输出 pingpong 的独立整数、槽代次与程序依赖检查。"""
import unittest

import numpy as np

import cfg_b_pingpong_chain as pingpong
from mbarrier_service import BarrierPhase


def full_integer_inputs(seed):
    """直接遍历 r00 的完整 M/K/N 坐标，不压缩 17 周期。"""
    rows = np.arange(384, dtype=np.int64)[:, None]
    inner = np.arange(832, dtype=np.int64)
    columns = np.arange(128, dtype=np.int64)[None, :]
    a = (7*rows+13*inner[None, :]+3*seed) % 17-8
    b = (5*inner[:, None]+11*columns+5*seed) % 17-8
    return a, b


def ancestors(nodes, node):
    result = set()
    pending = list(nodes[node])
    while pending:
        previous = pending.pop()
        if previous not in result:
            result.add(previous)
            pending.extend(nodes[previous])
    return result


class PingpongChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = pingpong.program_reference()

    def test_two_shapes_one_cta_three_outputs_and_complete_work(self):
        cases = pingpong.cases()
        self.assertEqual(len(cases), 2)
        self.assertEqual({case['trace'] for case in cases}, {False, True})
        for case in cases:
            self.assertEqual({key: case[key] for key in
                ('m', 'n', 'k', 'stages', 'output_tiles', 'ktiles_per_output')},
                dict(m=384, n=128, k=832, stages=6, output_tiles=3, ktiles_per_output=13))
            self.assertEqual(case['split'], 'dev')
        self.assertEqual(self.program['program']['work'], [[(0, 0), (1, 0), (2, 0)]])
        self.assertEqual(self.program['counts'], dict(input_groups=39, input_requests=117,
            wgmma_instructions=312, output_atoms=24, input_logical_bytes=1277952,
            output_bytes=196608))
        outputs = self.program['program']['outputs'][0]
        self.assertEqual([output['consumer_warpgroup'] for output in outputs], [0, 1, 0])
        self.assertEqual([output['input_group_range'] for output in outputs],
                         [[0, 13], [13, 26], [26, 39]])
        self.assertEqual(sum(output['output']['nominal_subtiles'] for output in outputs), 24)

    def test_period_reference_equals_full_832_integer_matmul_and_boundary_dots(self):
        for seed in pingpong.SEEDS:
            a, b = full_integer_inputs(seed)
            expected = ((a @ b)/1024).astype(np.float32)
            actual = pingpong.reference(seed)
            self.assertEqual(actual.shape, (384, 128))
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, expected)
            for row, column in ((0, 0), (127, 63), (127, 64), (128, 127),
                                (255, 64), (256, 0), (383, 127)):
                numerator = 0
                for inner in range(832):
                    av = (7*row+13*inner+3*seed) % 17-8
                    bv = (5*inner+11*column+5*seed) % 17-8
                    numerator += av*bv
                self.assertEqual(actual[row, column], numerator/1024)

    def test_missing_b1_and_permuted_output_row_tiles_change_full_output(self):
        a, b = full_integer_inputs(22)
        expected = pingpong.reference(22)
        missing_b1 = b.copy()
        missing_b1[:, 64:] = 0
        wrong = ((a @ missing_b1)/1024).astype(np.float32)
        np.testing.assert_array_equal(wrong[:, :64], expected[:, :64])
        self.assertFalse(np.array_equal(wrong[:, 64:], expected[:, 64:]))
        permuted = np.concatenate((expected[128:256], expected[:128], expected[256:]))
        self.assertFalse(np.array_equal(permuted, expected))
        self.assertTrue(np.any(permuted[:128] != expected[:128]))
        self.assertTrue(np.any(permuted[128:256] != expected[128:256]))
        np.testing.assert_array_equal(permuted[256:], expected[256:])

    def test_all_input_groups_keep_absolute_slot_generation_across_outputs(self):
        groups = self.program['graph']['groups']
        self.assertEqual([group['group'] for group in groups], list(range(39)))
        for absolute, group in enumerate(groups):
            self.assertEqual(group['output'], absolute//13)
            self.assertEqual(group['k'], absolute % 13)
            self.assertEqual(group['consumer_warpgroup'], (absolute//13) % 2)
            self.assertEqual(group['slot'], absolute % 6)
            self.assertEqual(group['generation'], absolute//6)
            self.assertEqual(group['full_wait_phase'], (absolute//6) % 2)
            self.assertEqual(group['empty_wait_phase'], (absolute//6+1) % 2)
            self.assertEqual(group['previous_group'], absolute-6 if absolute >= 6 else None)
            self.assertEqual(group['expected_full_bytes'], 32768)
        self.assertEqual((groups[13]['slot'], groups[13]['generation']), (1, 2))
        self.assertEqual((groups[26]['slot'], groups[26]['generation']), (2, 4))
        self.assertEqual((groups[38]['slot'], groups[38]['generation']), (2, 6))
        self.assertEqual([sum(group['slot'] == slot for group in groups) for slot in range(6)],
                         [7, 7, 7, 6, 6, 6])

    def test_requests_use_full_matrix_origins_and_all_completions_gate_full(self):
        graph = self.program['graph']
        nodes = graph['nodes']
        for absolute in range(39):
            output, inner = divmod(absolute, 13)
            a, b0, b1 = graph['requests'][3*absolute:3*absolute+3]
            self.assertEqual([request['atom'] for request in (a, b0, b1)], ['A', 'B0', 'B1'])
            self.assertEqual([request['bytes'] for request in (a, b0, b1)], [16384, 8192, 8192])
            self.assertEqual(a['source']['origin'], [128*output, 64*inner])
            self.assertEqual(b0['source']['origin'], [0, 64*inner])
            self.assertEqual(b1['source']['origin'], [64, 64*inner])
            self.assertEqual(a['source']['extents'], [384, 832])
            self.assertEqual(a['source']['stride_bytes'], [1664, 2])
            for request in (b0, b1):
                # The ledger represents B in (N,K) coordinates; storage is row-major (K,N).
                self.assertEqual(request['source']['extents'], [128, 832])
                self.assertEqual(request['source']['stride_bytes'], [2, 256])
            for request in (a, b0, b1):
                self.assertEqual(request['valid_bytes'], request['bytes'])
                self.assertEqual(request['fill_bytes'], 0)
                self.assertEqual(request['destination']['slot'], absolute % 6)
                self.assertEqual(request['destination']['generation'], absolute//6)
                self.assertEqual(request['destination']['offset_in_operand_buffer'],
                                 16384*(absolute % 6)+(8192 if request is b1 else 0))
                self.assertIn(request['completion'], nodes[f'g{absolute}.full_ready'])
            self.assertIn(f'g{absolute}.expect_tx', nodes[f'g{absolute}.full_ready'])
            self.assertIn(f'g{absolute}.full_ready', nodes[f'g{absolute}.consumer_wait'])
            if absolute >= 6:
                self.assertIn(f'g{absolute-6}.release', nodes[f'g{absolute}.acquire'])
            if absolute:
                self.assertIn(f'g{absolute-1}.B1.issue', nodes[f'g{absolute}.acquire'])
            if inner == 12:
                self.assertEqual(nodes[f'g{absolute}.release'], [f'j{output}.wait0'])
                self.assertIn(f'g{absolute}.compute_complete', nodes[f'j{output}.wait0'])
            else:
                self.assertEqual(nodes[f'g{absolute}.release'], [f'g{absolute+1}.wait1'])
                self.assertIn(f'g{absolute}.compute_complete', nodes[f'g{absolute+1}.wait1'])

    def test_pingpong_math_handoff_precedes_tail_and_same_group_reuses_after_epilogue(self):
        nodes = self.program['graph']['nodes']
        self.assertIn('j0.math_handoff', nodes['j1.math_permit'])
        self.assertNotIn('j0.main_end', ancestors(nodes, 'j1.math_permit'))
        self.assertNotIn('j0.wait0', ancestors(nodes, 'j1.math_permit'))
        self.assertNotIn('g12.compute_complete', ancestors(nodes, 'j0.math_handoff'))
        self.assertIn('j1.math_handoff', nodes['j2.math_permit'])
        self.assertIn('j0.epilogue_handoff', nodes['j2.math_permit'])
        self.assertIn('j0.store_tail', ancestors(nodes, 'j2.math_permit'))
        for output in range(3):
            self.assertIn(f'j{output}.main_end', nodes[f'j{output}.epilogue_permit'])
            self.assertEqual(nodes[f'j{output}.epilogue_handoff'], [f'j{output}.store_tail'])
            if output:
                self.assertIn(f'j{output-1}.epilogue_handoff', nodes[f'j{output}.epilogue_permit'])
        self.assertEqual(set(nodes['producer_tail']), {'g38.B1.issue'} |
                         {f'g{absolute}.release' for absolute in range(33, 39)})
        self.assertEqual(set(nodes['program_end']), {'consumer_exit', 'producer_tail'})

    def test_two_order_stages_preserve_phases_and_need_all_128_arrivals(self):
        reference = pingpong.order_reference()
        expected = [(0, 0, 1), (1, 1, 0), (2, 0, 0)]
        self.assertEqual(len(reference['records']), 6)
        phases = [[0, 0], [0, 0]]
        generations = [[0, 0], [0, 0]]
        for output, group, wait_phase in expected:
            for stage, kind in enumerate(('math', 'epilogue')):
                record = reference['records'][2*output+stage]
                self.assertEqual({key: record[key] for key in ('output', 'group', 'stage',
                    'kind', 'wait_phase', 'signalling_group', 'arrivals')},
                    dict(output=output, group=group, stage=stage, kind=kind,
                         wait_phase=wait_phase, signalling_group=1-group, arrivals=128))
                self.assertNotEqual(phases[stage][group], wait_phase)
                target = 1-group
                barrier = BarrierPhase(128, phase=phases[stage][target])
                for _ in range(127):
                    self.assertFalse(barrier.arrive())
                self.assertFalse(barrier.wait_complete(phases[stage][target]))
                self.assertTrue(barrier.arrive())
                phases[stage][target] ^= 1
                generations[stage][target] += 1
                self.assertEqual(record['target_phase_after'], phases[stage][target])
                self.assertEqual(barrier.phase, phases[stage][target])
        for stage in range(2):
            for group in range(2):
                final = reference['barriers'][stage][group]
                self.assertEqual(final['phase'], phases[stage][group])
                self.assertEqual(final['completed_phases'], generations[stage][group])
                self.assertEqual(final['expected_arrivals'], 128)
                self.assertEqual(final['pending_arrivals'], 128)
                self.assertEqual(final['pending_tx_bytes'], 0)

    def test_eight_output_atoms_cover_each_source_slot_once_per_subtile_and_reuse(self):
        atoms = pingpong.output_slots()
        self.assertEqual(len(atoms), 8)
        covered = []
        for subtile, atom in enumerate(atoms):
            slot = subtile % 2
            self.assertEqual((atom['subtile'], atom['source_slot'], atom['bytes']),
                             (subtile, slot, 8192))
            addresses = atom['sts64_word_addresses']
            self.assertEqual(len(addresses), 8)
            self.assertTrue(all(len(instruction) == 128 for instruction in addresses))
            words = sorted(word for instruction in addresses for start in instruction
                           for word in (start, start+1))
            begin = (0x800+slot*8192)//4
            self.assertEqual(words, list(range(begin, begin+2048)))
            covered.append(set(words))
        self.assertTrue(covered[0].isdisjoint(covered[1]))
        for subtile in range(2, 8):
            self.assertEqual(covered[subtile], covered[subtile-2])
        self.assertEqual(sum(atom['bytes'] for atom in atoms)*3, 196608)
        # Three outputs consume 24 source-buffer turns without resetting the absolute count.
        turns = [(8*output+subtile, atom['source_slot'], (8*output+subtile)//2)
                 for output in range(3) for subtile, atom in enumerate(atoms)]
        self.assertEqual([turn[0] for turn in turns], list(range(24)))
        self.assertEqual(turns[8], (8, 0, 4))
        self.assertEqual(turns[16], (16, 0, 8))
        self.assertEqual(turns[-1], (23, 1, 11))


if __name__ == '__main__':
    unittest.main()
