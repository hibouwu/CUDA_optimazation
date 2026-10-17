"""D原生TMA短链的地址、数值与无时间依赖合同；不执行GPU。"""
import unittest

import numpy as np

import cfg_b_handoff_chain as handoff
import cfg_b_tma_chain as transfer
from mbarrier_service import BarrierPhase


def generation_matrix(ktiles, stages, seed, stale_operand=None):
    """独立整数点积；漏填槽时保留该物理槽的首代正值。"""
    rows = np.arange(128, dtype=np.int64)[:, None]
    inner = np.arange(64, dtype=np.int64)[None, :]
    a_base = 1+(7*rows+13*inner+3*(seed % 8)) % 8
    b_base = 1+(11*rows+5*inner+5*(seed % 8)) % 8
    total = np.zeros((128, 128), dtype=np.int64)
    for tile in range(ktiles):
        a_source = tile % stages if stale_operand == 'A' else tile
        b_source = tile % stages if stale_operand == 'B' else tile
        a = a_base+a_source
        b = b_base+b_source
        if stale_operand == 'B1':
            b[64:] = b_base[64:]+tile % stages
        total += a @ b.T
    return (total/1024).astype(np.float32)


def ancestors(nodes, node_id):
    found = set()
    pending = list(nodes[node_id]['predecessors'])
    while pending:
        predecessor = pending.pop()
        if predecessor not in found:
            found.add(predecessor)
            pending.extend(nodes[predecessor]['predecessors'])
    return found


class TmaChainTests(unittest.TestCase):
    def test_eight_shapes_and_exact_work(self):
        rows = transfer.cases()
        self.assertEqual(len(rows), 8)
        self.assertEqual({(row['stages'], row['ktiles'], row['trace']) for row in rows},
            {(stages, ktiles, trace) for stages, ktiles in ((6,4), (6,13), (6,32), (4,13))
             for trace in (False, True)})
        self.assertEqual(sum(row['split'] == 'dev' for row in rows)*3, 18)
        self.assertEqual(sum(row['split'] == 'heldout' for row in rows)*3, 6)
        for row in rows:
            self.assertEqual(row['wgmma_count'], 8*row['ktiles'])
            self.assertEqual(row['work_flop'], 2097152*row['ktiles'])

    def test_three_requests_and_distinct_global_destination_offsets(self):
        for stages in (4, 6):
            for profile, source_tiles in (('ring', stages), ('generation', 13)):
                for tile in range(13):
                    requests = transfer.requests(tile, stages, source_tiles, profile)
                    self.assertEqual([request['operand'] for request in requests], ['A','B0','B1'])
                    self.assertEqual([request['bytes'] for request in requests], [16384,8192,8192])
                    self.assertEqual(sum(request['bytes'] for request in requests), 32768)
                    source = tile % stages if profile == 'ring' else tile
                    a, b0, b1 = requests
                    for request in requests:
                        self.assertEqual(request['source_k_tile'], source)
                        self.assertEqual(request['slot'], tile % stages)
                    self.assertEqual(a['source_coord'], [64*source,0,0])
                    self.assertEqual(b0['source_coord'], [0,64*source,0])
                    self.assertEqual(b1['source_coord'], [64,64*source,0])
                    self.assertEqual(a['source_byte_offset'], 128*source)
                    self.assertEqual(b0['source_byte_offset'], 16384*source)
                    self.assertEqual(b1['source_byte_offset']-b0['source_byte_offset'], 128)
                    self.assertEqual(a['source_row_stride_bytes'], 128*source_tiles)
                    self.assertEqual(b0['source_row_stride_bytes'], 256)
                    self.assertEqual(b1['source_row_stride_bytes'], 256)
                    self.assertEqual(a['destination_byte_offset'], (tile % stages)*16384)
                    self.assertEqual(b0['destination_byte_offset'], 98304+(tile % stages)*16384)
                    self.assertEqual(b1['destination_byte_offset']-b0['destination_byte_offset'], 8192)

    def test_half_offsets_cover_each_source_once_and_split_b_columns(self):
        for stages in (4, 6):
            for profile, source_tiles in (('ring', stages), ('generation', 13)):
                a_offsets, b_offsets = [], []
                for tile in range(source_tiles):
                    a, b0, b1 = transfer.requests(tile, stages, source_tiles, profile)
                    a_actual = transfer.source_offsets(a, source_tiles)
                    b0_actual = transfer.source_offsets(b0, source_tiles)
                    b1_actual = transfer.source_offsets(b1, source_tiles)
                    source = tile
                    a_expected = 2*(np.arange(128)[:,None]*(64*source_tiles)+
                                    64*source+np.arange(64)[None,:])
                    b_expected = 2*((64*source+np.arange(64)[:,None])*128+
                                    np.arange(64)[None,:])
                    np.testing.assert_array_equal(a_actual, a_expected)
                    np.testing.assert_array_equal(b0_actual, b_expected)
                    np.testing.assert_array_equal(b1_actual, b_expected+128)
                    a_offsets.append(a_actual.reshape(-1))
                    b_offsets.extend((b0_actual.reshape(-1), b1_actual.reshape(-1)))
                expected = np.arange(128*64*source_tiles, dtype=np.int64)*2
                np.testing.assert_array_equal(np.sort(np.concatenate(a_offsets)), expected)
                np.testing.assert_array_equal(np.sort(np.concatenate(b_offsets)), expected)

    def test_ring_reference_is_identical_to_handoff(self):
        for case in transfer.cases():
            for seed in handoff.SEEDS:
                np.testing.assert_array_equal(
                    transfer.reference(case['ktiles'], case['stages'], seed, profile='ring'),
                    handoff.reference(case['ktiles'], case['stages'], seed))

    def test_positive_generation_reference_matches_scalar_integer_dot(self):
        row, column = handoff.output_coordinates()
        for stages in (4, 6):
            for length in (4, 13, 32):
                actual = transfer.reference(length, stages, 22, profile='generation')
                matrix = generation_matrix(length, stages, 22)
                np.testing.assert_array_equal(actual, matrix[row,column])
                for position in (0, 63, 64, 8191, 16383):
                    total = 0
                    for tile in range(length):
                        for inner in range(64):
                            a = 1+(7*int(row[position])+13*inner+3*(22 % 8)) % 8+tile
                            b = 1+(11*int(column[position])+5*inner+5*(22 % 8)) % 8+tile
                            total += a*b
                    self.assertEqual(actual[position], total/1024)

    def test_missing_a_b_or_b1_refill_strictly_lowers_positive_witness(self):
        for stages in (4, 6):
            correct = generation_matrix(13, stages, 41)
            for operand in ('A', 'B'):
                wrong = generation_matrix(13, stages, 41, stale_operand=operand)
                self.assertTrue(np.all(wrong < correct), operand)
            wrong_b1 = generation_matrix(13, stages, 41, stale_operand='B1')
            np.testing.assert_array_equal(wrong_b1[:,:64], correct[:,:64])
            self.assertTrue(np.all(wrong_b1[:,64:] < correct[:,64:]))

    def test_descriptor_metadata_is_three_dimensional(self):
        for source_tiles in (4, 6, 13, 32):
            descriptor = transfer.expected_descriptor(source_tiles)
            k = 64*source_tiles
            for operand, dims, strides, box in (
                    ('A', [k,128,1], [2*k,256*k], [64,128,1]),
                    ('B', [128,k,1], [256,256*k], [64,64,1])):
                fields = descriptor[operand]
                self.assertEqual(fields['global_dim'], dims)
                self.assertEqual(fields['byte_strides'], strides)
                self.assertEqual(fields['box'], box)
                self.assertEqual(fields['swizzle_bytes'], 128)
                self.assertEqual(fields['rank'], 3)
                self.assertEqual(fields['encoding_type'], 'UINT16')
                self.assertEqual(fields['numeric_type'], 'FP16')
                self.assertEqual(fields['element_strides'], [1,1,1])
                self.assertEqual(fields['interleave'], 'NONE')
                self.assertEqual(fields['l2_promotion_bytes'], 128)
                self.assertEqual(fields['oob_fill'], 'NONE')

    def test_all_ktile_transfers_gate_full_and_preserve_program_order(self):
        for stages, length in ((6,4), (6,13), (6,32), (4,13)):
            program = transfer.lifecycle(length, stages)
            nodes = program['nodes']
            by_kind = {}
            for node in nodes:
                self.assertTrue(all(previous < node['id'] for previous in node['predecessors']))
                self.assertEqual(node['slot'], node['tile'] % stages)
                self.assertEqual(node['generation'], node['tile']//stages)
                by_kind.setdefault(node['kind'], []).append(node)
            self.assertEqual(len(by_kind['tma_issue']), 3*length)
            self.assertEqual(len(by_kind['tma_complete']), 3*length)
            self.assertEqual(len(by_kind['producer_acquire']), length)
            self.assertEqual(len(by_kind['empty_release']), length)
            for tile in range(length):
                select = lambda kind: [node for node in by_kind[kind] if node['tile'] == tile]
                expect = select('expect_tx')[0]
                self.assertEqual(expect['bytes'], 32768)
                issues, completions = select('tma_issue'), select('tma_complete')
                self.assertEqual({node['operand'] for node in issues}, {'A','B0','B1'})
                for records in (issues, completions):
                    self.assertEqual({node['operand']:node['bytes'] for node in records},
                                     dict(A=16384, B0=8192, B1=8192))
                full = select('full_ready')[0]
                consumer = select('consumer_full_wait')[0]
                for issue in issues:
                    self.assertIn(expect['id'], ancestors(nodes, issue['id']))
                    self.assertTrue(issue['in_timed_window'])
                for completion in completions:
                    self.assertIn(completion['id'], ancestors(nodes, full['id']))
                self.assertIn(full['id'], ancestors(nodes, consumer['id']))
                acquire = select('producer_acquire')[0]
                if tile >= stages:
                    release = next(node for node in by_kind['empty_release'] if node['tile'] == tile-stages)
                    self.assertIn(release['id'], ancestors(nodes, acquire['id']))
                if tile:
                    prior_issues = [node for node in by_kind['tma_issue'] if node['tile'] == tile-1]
                    for prior_issue in prior_issues:
                        self.assertIn(prior_issue['id'], ancestors(nodes, acquire['id']))
                    prior_control = next(node for node in by_kind['commit' if tile == 1 else 'empty_release']
                                         if node['tile'] == (0 if tile == 1 else tile-2))
                    self.assertIn(prior_control['id'], ancestors(nodes, consumer['id']))
                release = select('empty_release')[0]
                self.assertEqual(release['arrivals'], 128)
                predecessor = nodes[release['predecessors'][0]]
                self.assertEqual(predecessor['kind'], 'tail_wait0' if tile == length-1 else 'wait1')
            if length == 4:
                self.assertEqual(len(by_kind['tma_issue']), 12)

    def test_full_requires_three_completions_and_empty_all_arrivers(self):
        for stages in (4, 6):
            full = [BarrierPhase(1) for _ in range(stages)]
            empty = [BarrierPhase(128) for _ in range(stages)]
            for tile in range(13):
                slot, generation = tile % stages, tile//stages
                self.assertTrue(empty[slot].wait_complete((generation+1) % 2))
                self.assertFalse(full[slot].arrive_and_expect_tx(32768))
                self.assertFalse(full[slot].complete_tx(16384))
                self.assertFalse(full[slot].wait_complete(generation % 2))
                self.assertFalse(full[slot].complete_tx(8192))
                self.assertFalse(full[slot].wait_complete(generation % 2))
                self.assertTrue(full[slot].complete_tx(8192))
                self.assertTrue(full[slot].wait_complete(generation % 2))
                for _ in range(127):
                    self.assertFalse(empty[slot].arrive())
                self.assertFalse(empty[slot].wait_complete((generation+2) % 2))
                self.assertTrue(empty[slot].arrive())
                self.assertTrue(empty[slot].wait_complete((generation+2) % 2))
                self.assertEqual(full[slot].completed_phases, generation+1)
                self.assertEqual(empty[slot].completed_phases, generation+1)


if __name__ == '__main__':
    unittest.main()
