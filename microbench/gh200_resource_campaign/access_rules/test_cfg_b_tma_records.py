"""D记录的合成窗口／事务／完整输出合同；不生成真实descriptor或执行GPU。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import cfg_b_tma_chain as transfer


def clock_pair(value):
    return dict(cycle=value, ns=value)


def synthetic_record(case, seed=3):
    """独立指定两角色时间，不调用被测DAG或时间服务。"""
    tiles = []
    for tile in range(case['ktiles']):
        start = 10000*(tile+1)
        producer = {name:clock_pair(start+offset) for name,offset in (
            ('acquire_before',-90), ('acquire_after',-80), ('expect_before',-70),
            ('expect_after',-60), ('request_A_before',-50), ('request_A_after',-40),
            ('request_B_pair_before',-30), ('request_B_pair_after',-20))}
        consumer = dict(wait_before=clock_pair(start-100), wait_after=clock_pair(start),
            issue_before=clock_pair(start+10), commit_after=clock_pair(start+20),
            first_try_success=True, first_try_success_warp_masks=[0xffffffff]*4,
            wait1_after=clock_pair(start+30) if tile else None,
            release_tile=tile-1 if tile else None,
            release_before=clock_pair(start+40) if tile else None,
            release_after=clock_pair(start+50) if tile else None,
            release_last_before=clock_pair(start+45) if tile else None,
            release_last_after=clock_pair(start+60) if tile else None)
        tiles.append(dict(tile=tile, slot=tile % case['stages'], source_tile=tile % case['stages'],
            generation=tile//case['stages'],
            full_wait_phase=(tile//case['stages']) % 2,
            empty_wait_phase=(tile//case['stages']+1) % 2,
            prefilled=False, producer=producer, consumer=consumer))
    last = 10000*case['ktiles']
    trace = dict(tiles=tiles, tail_wait0_after=clock_pair(last+70),
        tail_release_tile=case['ktiles']-1, tail_release_before=clock_pair(last+80),
        tail_release_after=clock_pair(last+90), tail_release_last_before=clock_pair(last+85),
        tail_release_last_after=clock_pair(last+100))
    witnesses = []
    for length in (1, 2, case['stages']+1, case['ktiles']):
        witnesses.append(dict(ktiles=length, source_profile='generation', correctness=dict(
            checked_elements=16384, errors=0, max_abs_error=0,
            generation_checked=128*length, generation_errors=0)))
    witnesses[-1].update(output_file='witness_output.f32', output_elements=16384, output_bytes=65536)
    end = last+1000
    return dict(schema='gh200-cfg-b-tma-chain/1', status='measured',
        **{key:case[key] for key in ('stages','ktiles','trace','wgmma_count','work_flop')},
        seed=seed, samples=1, source_profile='ring', source_tiles=case['stages'],
        source_k=64*case['stages'],
        device=dict(name='GH200', sms=132, major=9, minor=0, uuid='GPU-synthetic'),
        resources=dict(threads=256, grid_ctas=1, dynamic_smem_bytes=196608,
            static_smem_bytes=16384, registers_per_thread=192, local_size_bytes=0,
            occupancy_ctas_per_sm=1),
        conditions=dict(slots=6, active_slots=case['stages'], full_arrivals=1, empty_arrivals=128,
            signal_kind='arrive_and_expect_tx_then_native_tma', transaction_bytes=32768,
            prefilled_tiles=0, tma_requests_per_tile=3, tma_request_bytes=[16384,8192,8192],
            logical_tma_bytes=case['tma_bytes'], smem_swizzle_bytes=128,
            software_tile=[128,128,64], instruction_shape=[64,128,16],
            commit_per_ktile=1, wait_groups=1, tail_wait_groups=0),
        timing=dict(begin_cycle=1000, end_cycle=end, elapsed_cycles=end-1000,
            begin_ns=1000, end_ns=end, elapsed_ns=end-1000,
            smid_start=144, smid_end=144, event_ms=end/1000000),
        correctness=dict(checked_elements=16384, errors=0, max_abs_error=0,
            generation_checked=128*case['ktiles'], generation_errors=0),
        generation_witnesses=witnesses,
        # 仅供记录长度校验的合成opaque值，不代表可用于TMA的编码。
        descriptors=dict(bytes_each=128, a_hex='11'*128, b_hex='22'*128),
        trace_detail=trace if case['trace'] else None,
        output_file='output.f32', output_elements=16384, output_bytes=65536,
        warmup=dict(elapsed_ns=[end-1000]*8, elapsed_cycles=[end-1000]*8,
            limit=60, converged=True, cv=0))


def write_outputs(folder, case, seed=3):
    transfer.reference(case['ktiles'], case['stages'], seed, 'ring').astype('<f4').tofile(
        folder/'output.f32')
    transfer.reference(case['ktiles'], case['stages'], seed, 'generation').astype('<f4').tofile(
        folder/'witness_output.f32')


class TmaRecordTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.case = next(case for case in transfer.cases() if case['id'] == 's6_k13_trace')
        self.row = synthetic_record(self.case)
        write_outputs(self.folder, self.case)

    def test_normal_records_cover_all_eight_shapes(self):
        for case in transfer.cases():
            with self.subTest(case=case['id']):
                write_outputs(self.folder, case)
                checked = transfer.validate(self.folder, synthetic_record(case), case, 3)
                self.assertTrue(checked['performance_eligible'])
                self.assertEqual(checked['performance_output']['checked_elements'], 16384)
                self.assertEqual(checked['tma_requests'], 3*case['ktiles'])
                self.assertEqual(checked['tma_bytes'], 32768*case['ktiles'])
                self.assertEqual(len(checked['intervals']), case['ktiles'] if case['trace'] else 0)

    def test_four_ktile_prefill_or_missing_transaction_is_rejected(self):
        case = next(case for case in transfer.cases() if case['id'] == 's6_k4_trace')
        write_outputs(self.folder, case)
        for field,wrong in (('prefilled_tiles',4), ('transaction_bytes',24576),
                            ('tma_requests_per_tile',2)):
            with self.subTest(field=field):
                row = synthetic_record(case)
                row['conditions'][field] = wrong
                with self.assertRaisesRegex(ValueError, 'TMA／barrier／计算条件'):
                    transfer.validate(self.folder, row, case, 3)

    def test_wait_return_before_required_b_pair_is_rejected(self):
        point = self.row['trace_detail']['tiles'][0]
        before = point['producer']['request_B_pair_before']['cycle']
        point['consumer']['wait_after'] = clock_pair(before-1)
        with self.assertRaisesRegex(ValueError, '等待返回早于必要的B请求对'):
            transfer.validate(self.folder, self.row, self.case, 3)

    def test_b_pair_post_stamp_can_follow_wait_return_and_try_true_is_not_ready(self):
        point = self.row['trace_detail']['tiles'][0]
        returned = point['consumer']['wait_after']['cycle']
        point['producer']['request_B_pair_after'] = clock_pair(returned+5)
        checked = transfer.validate(self.folder, self.row, self.case, 3)
        first = checked['intervals'][0]
        self.assertTrue(first['first_try_success'])
        self.assertEqual(first['entry_state'], 'entered_before_B_pair')
        for unit in ('cycle','ns'):
            self.assertGreater(point['producer']['request_B_pair_after'][unit],
                               point['consumer']['wait_after'][unit])
            self.assertEqual(first['full_completion_bounds'][unit],
                [point['producer']['request_B_pair_before'][unit],
                 point['consumer']['wait_after'][unit]])

    def test_damaged_generation_output_is_rejected(self):
        path = self.folder/'witness_output.f32'
        values = np.fromfile(path, dtype='<f4')
        values[-1] += 1/1024
        values.tofile(path)
        with self.assertRaisesRegex(ValueError, '重新填槽输出'):
            transfer.validate(self.folder, self.row, self.case, 3)

    def test_acquire_before_last_of_128_arrivals_is_rejected(self):
        tiles = self.row['trace_detail']['tiles']
        # 先在窗口前段发完首六槽，保留producer顺序；反例只违反再取槽边界。
        for tile in range(6):
            for index,name in enumerate(('acquire_before','acquire_after','expect_before','expect_after',
                                         'request_A_before','request_A_after',
                                         'request_B_pair_before','request_B_pair_after')):
                tiles[tile]['producer'][name] = clock_pair(1100+100*tile+10*index)
        release = tiles[1]['consumer']
        producer = tiles[6]['producer']
        producer['acquire_before'] = deepcopy(release['release_before'])
        producer['acquire_after'] = clock_pair(release['release_last_before']['cycle']-1)
        with self.assertRaisesRegex(ValueError, '最后必要arrival'):
            transfer.validate(self.folder, self.row, self.case, 3)


if __name__ == '__main__':
    unittest.main()
