"""B/C记录合同的合成正例与已发现的四类反例；不执行GPU。"""
import tempfile
import unittest
from pathlib import Path

import cfg_b_handoff_chain as handoff


def clock_pair(value):
    return dict(cycle=value, ns=value)


def synthetic_record(case, seed=3):
    """用明确的角色时序生成记录，不调用被测生命周期或校验函数。"""
    tiles = []
    for tile in range(case['ktiles']):
        start = 10000 * (tile + 1)
        consumer = dict(wait_before=clock_pair(start), wait_after=clock_pair(start+10),
            issue_before=clock_pair(start+20), commit_after=clock_pair(start+30),
            first_try_success=True, first_try_success_warp_masks=[0xffffffff]*4,
            wait1_after=clock_pair(start+40) if tile else None,
            release_tile=tile-1 if tile else None,
            release_before=clock_pair(start+50) if tile else None,
            release_after=clock_pair(start+60) if tile else None,
            release_last_before=clock_pair(start+55) if tile else None,
            release_last_after=clock_pair(start+70) if tile else None)
        producer = None
        if tile >= case['stages']:
            delay = case['delay_cycles'] if tile == case['stages'] else 0
            producer = dict(acquire_before=clock_pair(start-delay-14),
                acquire_after=clock_pair(start-delay-12),
                full_before=clock_pair(start-2), full_after=clock_pair(start-1))
        tiles.append(dict(tile=tile, slot=tile % case['stages'],
            generation=tile//case['stages'], full_wait_phase=(tile//case['stages']) % 2,
            empty_wait_phase=(tile//case['stages']+1) % 2,
            prefilled=tile < case['stages'], producer=producer, consumer=consumer))
    last_start = 10000 * case['ktiles']
    trace = dict(tiles=tiles, tail_wait0_after=clock_pair(last_start+80),
        tail_release_tile=case['ktiles']-1, tail_release_before=clock_pair(last_start+90),
        tail_release_after=clock_pair(last_start+100),
        tail_release_last_before=clock_pair(last_start+95),
        tail_release_last_after=clock_pair(last_start+110))
    end = last_start+1000
    return dict(schema='gh200-cfg-b-handoff-chain/1', status='measured',
        **{key:case[key] for key in ('stages', 'ktiles', 'delay_cycles', 'trace',
                                    'wgmma_count', 'work_flop')}, seed=seed, samples=1,
        device=dict(name='GH200', sms=132, major=9, minor=0, uuid='GPU-synthetic'),
        resources=dict(threads=256, grid_ctas=1, dynamic_smem_bytes=196608,
            local_size_bytes=0, occupancy_ctas_per_sm=1,
            static_smem_bytes=16384, registers_per_thread=192),
        conditions=dict(slots=6, active_slots=case['stages'], full_arrivals=1,
            empty_arrivals=128, signal_kind='normal_arrive',
            prefilled_tiles=min(case['stages'], case['ktiles'])),
        timing=dict(begin_cycle=1000, end_cycle=end, elapsed_cycles=end-1000,
            begin_ns=1000, end_ns=end, elapsed_ns=end-1000,
            smid_start=144, smid_end=144, event_ms=end/1000000),
        correctness=dict(generation_checked=128*case['ktiles'], generation_errors=0,
            checked_elements=16384, errors=0, max_abs_error=0),
        trace_detail=trace if case['trace'] else None,
        output_file='output.f32', output_elements=16384, output_bytes=65536,
        short_checks=[dict(ktiles=length, correctness=dict(checked_elements=16384,
            errors=0, max_abs_error=0, generation_checked=128*length, generation_errors=0))
            for length in (1, 2, case['stages']+1)],
        warmup=dict(elapsed_ns=[end-1000]*8, elapsed_cycles=[end-1000]*8,
            limit=60, converged=True, cv=0))


class HandoffRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.case = next(case for case in handoff.cases(2048)
                         if case['id'] == 's6_k13_B_trace')
        self.row = synthetic_record(self.case)
        handoff.reference(13, 6, 3).astype('<f4').tofile(self.folder/'output.f32')

    def test_normal_records_cover_all_fourteen_shapes(self):
        for case in handoff.cases(2048):
            with self.subTest(case=case['id']):
                row = synthetic_record(case)
                handoff.reference(case['ktiles'], case['stages'], 3).astype('<f4').tofile(
                    self.folder/'output.f32')
                checked = handoff.validate(self.folder, row, case, 3)
                self.assertTrue(checked['performance_eligible'])
                self.assertEqual(checked['output_witness']['checked_elements'], 16384)
                self.assertEqual(len(checked['intervals']), case['ktiles'] if case['trace'] else 0)

    def test_missing_tail_is_rejected(self):
        del self.row['trace_detail']['tail_wait0_after']
        with self.assertRaisesRegex(KeyError, 'tail_wait0_after'):
            handoff.validate(self.folder, self.row, self.case, 3)

    def test_wrong_release_identity_is_rejected(self):
        self.row['trace_detail']['tiles'][2]['consumer']['release_tile'] = 0
        with self.assertRaisesRegex(ValueError, '释放的输入组'):
            handoff.validate(self.folder, self.row, self.case, 3)
        self.row['trace_detail']['tiles'][2]['consumer']['release_tile'] = 1
        self.row['trace_detail']['tail_release_tile'] = 0
        with self.assertRaisesRegex(ValueError, '尾部释放组'):
            handoff.validate(self.folder, self.row, self.case, 3)

    def test_acquire_before_last_required_arrival_is_rejected(self):
        trace = self.row['trace_detail']['tiles']
        # tile6重用tile0，由tile1释放；其它128名arrival的最后括号更晚。
        release = trace[1]['consumer']
        producer = trace[6]['producer']
        producer['acquire_before'] = clock_pair(release['release_before']['cycle']+1)
        producer['acquire_after'] = clock_pair(release['release_last_before']['cycle']-1)
        with self.assertRaisesRegex(ValueError, '最后必要arrival'):
            handoff.validate(self.folder, self.row, self.case, 3)

    def test_generation_error_is_rejected(self):
        self.row['correctness']['generation_errors'] = 1
        with self.assertRaisesRegex(ValueError, '完整结果或代次错误'):
            handoff.validate(self.folder, self.row, self.case, 3)


if __name__ == '__main__':
    unittest.main()
