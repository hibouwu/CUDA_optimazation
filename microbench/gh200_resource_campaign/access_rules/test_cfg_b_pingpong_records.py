"""E 记录的完整输出与必要跨 WG 端点合同；不运行 GPU。"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import cfg_b_pingpong_chain as pingpong


def clock_pair(value):
    return dict(cycle=3*value+5, ns=7*value+11)


def fixture(folder, trace, seed=22):
    """手写并发时间线与完整 K832 整数输出，不从被测程序图生成。"""
    rows = np.arange(384, dtype=np.int64)[:, None]
    inner = np.arange(832, dtype=np.int64)
    columns = np.arange(128, dtype=np.int64)[None, :]
    a = (7*rows+13*inner[None, :]+3*seed) % 17-8
    b = (5*inner[:, None]+11*columns+5*seed) % 17-8
    ((a @ b)/1024).astype('<f4').tofile(folder/'output.f32')
    # j1 begins before j0's leader math-arrive-after stamp and tail wait0 return.
    times = ((200, 300, 360, 400, 410, 460, 500, 520),
             (320, 480, 490, 510, 530, 570, 600, 620),
             (550, 700, 710, 730, 740, 780, 800, 820))
    events = ('first_mma', 'mma_return', 'math_arrive_after', 'main_end',
              'epilogue_permit', 'store_return', 'store_tail_return', 'epilogue_arrive_after')
    outputs = []
    for output, points in enumerate(times):
        outputs.append(dict(output=output, consumer_wg=output % 2,
            local_seq=output//2, coord=[output, 0],
            **{event: clock_pair(value) for event, value in zip(events, points)}))
    trace_detail = dict(entry=dict(smid=143, **clock_pair(100)),
        producer_first_work=clock_pair(120), outputs=outputs,
        finals=[dict(consumer_wg=0, tiles=2, smid=143, **clock_pair(900)),
                dict(consumer_wg=1, tiles=1, smid=143, **clock_pair(650))])
    return dict(schema='gh200-cfg-b-pingpong-chain/1', status='measured', trace=trace,
        seed=seed, conditions=dict(m=384, n=128, k=832, stages=6, output_tiles=3,
            ktiles_per_output=13, scheduler_sm_count=1),
        device=dict(name='NVIDIA GH200 480GB', sms=132, major=9, minor=0,
                    uuid='GPU-synthetic-e'),
        resources=dict(threads=384, grid=[1, 1, 1], local_size_bytes=0,
                       occupancy_ctas_per_sm=1),
        timing=dict(cuda_event_ms=.02),
        correctness=dict(checked_elements=49152, errors=0, max_abs_error=0,
                         guard_elements=32, guard_errors=0),
        guard_bits=[0x7fc12345]*32, output_file='output.f32', output_elements=49152,
        output_bytes=196608,
        warmup=dict(limit=60, elapsed_ms=[.02]*8, cv=0., converged=True),
        trace_detail=trace_detail if trace else None)


class PingpongRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.row = fixture(self.folder, True)
        self.case = next(case for case in pingpong.cases() if case['trace'])

    def validate(self, row=None, case=None):
        return pingpong.validate(self.folder, self.row if row is None else row,
                                 self.case if case is None else case, 22)

    def test_both_shapes_accept_complete_output_and_sparse_smid(self):
        for case in pingpong.cases():
            with self.subTest(trace=case['trace']):
                row = fixture(self.folder, case['trace'])
                result = self.validate(row, case)
                self.assertTrue(result['performance_eligible'])
                self.assertEqual(len(result['intervals']), 3 if case['trace'] else 0)
                self.assertEqual(result['device_uuid'], 'GPU-synthetic-e')
                self.assertEqual(result['timing'], dict(cuda_event_ms=.02))
                self.assertEqual(len(result['output_sha256']), 64)
                if case['trace']:
                    self.assertEqual(row['trace_detail']['entry']['smid'], 143)

    def test_guard_and_last_output_element_corruption_are_rejected(self):
        guard = copy.deepcopy(self.row)
        guard['guard_bits'][-1] = 0
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            self.validate(guard)
        output = np.fromfile(self.folder/'output.f32', dtype='<f4')
        output[-1] += 1/1024
        output.tofile(self.folder/'output.f32')
        with self.assertRaisesRegex(ValueError, '独立整数参考与完整输出'):
            self.validate()

    def test_wrong_output_role_coordinate_or_local_sequence_is_rejected(self):
        for key, value in (('consumer_wg', 0), ('coord', [0, 0]), ('local_seq', 1)):
            with self.subTest(field=key):
                wrong = copy.deepcopy(self.row)
                wrong['trace_detail']['outputs'][1][key] = value
                with self.assertRaisesRegex(ValueError, '工作列表'):
                    self.validate(wrong)

    def test_only_necessary_cross_output_edges_reject_early_work(self):
        early_math = copy.deepcopy(self.row)
        early_math['trace_detail']['outputs'][1]['first_mma'] = clock_pair(290)
        with self.assertRaisesRegex(ValueError, '必要数学交接'):
            self.validate(early_math)
        early_output = copy.deepcopy(self.row)
        item = early_output['trace_detail']['outputs'][1]
        item['main_end'] = clock_pair(495)
        item['epilogue_permit'] = clock_pair(499)
        with self.assertRaisesRegex(ValueError, '必要输出源释放'):
            self.validate(early_output)
        early_reentry = copy.deepcopy(self.row)
        early_reentry['trace_detail']['outputs'][2]['first_mma'] = clock_pair(499)
        with self.assertRaisesRegex(ValueError, '同一数学WG过早重入'):
            self.validate(early_reentry)

    def test_overlap_is_legal_and_mma_return_cannot_be_replaced_by_main_end(self):
        previous, following = self.row['trace_detail']['outputs'][:2]
        for unit in ('cycle', 'ns'):
            self.assertGreater(following['first_mma'][unit], previous['mma_return'][unit])
            self.assertLess(following['first_mma'][unit], previous['math_arrive_after'][unit])
            self.assertLess(following['first_mma'][unit], previous['main_end'][unit])
        result = self.validate()
        collapsed = copy.deepcopy(self.row)
        item = collapsed['trace_detail']['outputs'][0]
        item['mma_return'] = item['main_end'].copy()
        item['math_arrive_after'] = item['main_end'].copy()
        with self.assertRaisesRegex(ValueError, '必要数学交接'):
            self.validate(collapsed)
        interval = result['intervals'][0]
        self.assertEqual(interval['final_compute_upper'], previous['main_end'])
        self.assertEqual(interval['output_source_read_upper'], previous['store_tail_return'])
        self.assertEqual(set(interval), {'output', 'final_compute_upper', 'output_source_read_upper'})
        self.assertIn('全局输出在stream完成后校验', result['qualification'])

    def test_consumer_exit_preserves_role_smid_and_last_handoff(self):
        for key, value in (('consumer_wg', 1), ('tiles', 1), ('smid', 17)):
            with self.subTest(field=key):
                wrong = copy.deepcopy(self.row)
                wrong['trace_detail']['finals'][0][key] = value
                with self.assertRaisesRegex(ValueError, '消费者退出身份'):
                    self.validate(wrong)
        early = copy.deepcopy(self.row)
        early['trace_detail']['finals'][1].update(clock_pair(619))
        with self.assertRaisesRegex(ValueError, '退出早于最后交接'):
            self.validate(early)


if __name__ == '__main__':
    unittest.main()
