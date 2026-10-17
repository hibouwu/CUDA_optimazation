"""实际Ktile覆盖、独立全输出参考及当前复合服务的CPU检查。"""
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

import cfg_b_compute_chain as chain


class ComputeChainTests(unittest.TestCase):
    def synthetic_row(self, case, seed):
        trace = None
        if case['trace']:
            trace = dict(k_tile_index=case['ktiles']-1,
                issue_before=dict(cycle=3000,ns=2000), commit_after=dict(cycle=3500,ns=2250),
                wait1_after=dict(cycle=3700,ns=2350), tail_wait0_after=dict(cycle=3900,ns=2450))
        return dict(schema='gh200-cfg-b-compute-chain/1', status='measured',
            ktiles=case['ktiles'], trace=case['trace'], wgmma_count=case['wgmma_count'],
            work_flop=case['work_flop'], seed=seed, samples=1,
            device=dict(name='GH200',sms=132,major=9,minor=0,uuid='GPU-synthetic'),
            resources=dict(threads=128,grid_ctas=1,dynamic_smem_bytes=196608,
                           local_size_bytes=0,static_smem_bytes=512,registers_per_thread=192),
            conditions=dict(slots=6, software_tile=[128,128,64], instruction_shape=[64,128,16],
                a_major='K', b_major='MN', smem_swizzle_bytes=128,
                commit_per_ktile=1,wait_groups=1,tail_wait_groups=0),
            timing=dict(begin_cycle=1000,end_cycle=4000,begin_ns=1000,end_ns=2500,
                elapsed_cycles=3000,elapsed_ns=1500,smid_start=100,smid_end=100,event_ms=.01),
            trace_detail=trace,output_file='output.f32',output_elements=16384,output_bytes=65536,
            correctness=dict(checked_elements=16384,errors=0,max_abs_error=0.),
            short_checks=[dict(ktiles=kt,correctness=dict(checked_elements=16384,
                                                        errors=0,max_abs_error=0.)) for kt in (1,2)],
            warmup=dict(limit=60,elapsed_ns=[1500]*8,elapsed_cycles=[3000]*8,
                        converged=True,cv=0.))

    def test_output_fragments_cover_entire_tile_once(self):
        row, col = chain.output_coordinates()
        self.assertEqual(sorted((128*row+col).tolist()), list(range(16384)))
        self.assertEqual((int(row[0]), int(col[0])), (0,0))
        self.assertEqual((int(row[64]), int(col[64])), (64,0))

    def test_integer_matrix_reference_matches_independent_scalar_dot(self):
        row, col = chain.output_coordinates()
        for ktiles in (1,2,4,13,32):
            for seed in chain.SEEDS:
                actual = chain.reference(ktiles, seed)
                for position in (0,1,63,64,127,255,8191,16383):
                    total = 0
                    for tile in range(ktiles):
                        for k in range(64):
                            a = (7*int(row[position])+13*k+3*(tile % 6)+3*(seed % 17)) % 17-8
                            b = (11*int(col[position])+5*k+7*(tile % 6)+5*(seed % 17)) % 17-8
                            total += a*b
                    self.assertEqual(actual[position], total/1024)

    def test_input_witness_distinguishes_m_k_and_slot_errors(self):
        correct = chain.reference(13, 3)
        self.assertFalse(np.array_equal(correct, chain.reference(4, 3)))
        row, col = chain.output_coordinates()
        # 错将第二M64片段指向第一片段，会在完整参考中留下反例。
        matrix = np.empty((128,128), dtype=np.float32)
        matrix[row,col] = correct
        wrong = matrix[row % 64,col]
        self.assertGreater(np.count_nonzero(wrong != correct), 0)
        # 重复q0的K16片段，少算另外三个不同的K偏移。
        s = 0
        wrong_dot = 4*sum(((7*0+13*k+3*s+9) % 17-8)*
                          ((11*0+5*k+7*s+15) % 17-8) for k in range(16))/1024
        self.assertNotEqual(wrong_dot, chain.reference(1,3)[0])
        self.assertFalse(np.array_equal(chain.reference(6,3), 6*chain.reference(1,3)))

    def test_formal_work_and_dev_heldout_separation(self):
        cases = chain.cases()
        self.assertEqual(len(cases), 6)
        self.assertEqual(len({c['id'] for c in cases}), 6)
        self.assertEqual({c['ktiles'] for c in cases if c['split']=='dev'}, {4,13})
        self.assertEqual({c['ktiles'] for c in cases if c['split']=='heldout'}, {32})
        for case in cases:
            self.assertEqual(case['wgmma_count'], 8*case['ktiles'])
            self.assertEqual(case['work_flop'], 262144*case['wgmma_count'])
        self.assertEqual([c['work_flop'] for c in cases[::2]],
                         [8388608,27262976,67108864])

    def test_full_witness_detects_last_output_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'output.f32'
            data = chain.reference(32,41)
            data.astype('<f4').tofile(path)
            self.assertEqual(chain.verify_output(path,32,41)['checked_elements'], 16384)
            data[-1] += 1/1024
            data.astype('<f4').tofile(path)
            with self.assertRaisesRegex(ValueError,'完整累加器'):
                chain.verify_output(path,32,41)

    def test_saved_current_hypothesis_separates_commit_wait_and_tail(self):
        for kt in (4,13,32):
            row = chain.resident_hypothesis(kt)
            timeline = row['cycle_timeline']
            self.assertEqual(timeline['main_ends'], [512.*kt])
            self.assertEqual(timeline['mma_returns'], [512.*kt-256])
            self.assertEqual(timeline['commits'][0], 256.)
            self.assertEqual(timeline['wait1_return'][0], 256.)
            self.assertEqual(timeline['released'][-1], 512.*kt)
            for k in range(kt-1):
                self.assertEqual(timeline['released'][k], timeline['wait1_return'][k+1])
                self.assertGreaterEqual(timeline['released'][k], timeline['completed'][k])

    def test_cu_schema_nested_short_checks_and_trace_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for case in chain.cases()[:2]:
                chain.reference(case['ktiles'],3).astype('<f4').tofile(folder/'output.f32')
                row = self.synthetic_row(case,3)
                self.assertTrue(chain.validate(folder,row,case,3)['performance_eligible'])
                wrong = deepcopy(row)
                wrong['short_checks'][1]['correctness']['errors'] = 1
                with self.assertRaisesRegex(ValueError,'短正确性'):
                    chain.validate(folder,wrong,case,3)
                if case['trace']:
                    row['trace_detail']['wait1_after']['cycle'] = 3400
                    with self.assertRaisesRegex(ValueError,'程序顺序'):
                        chain.validate(folder,row,case,3)

    def test_heldout_freeze_is_bound_before_any_partial_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            first = dict(sha256='first', contents=dict(predictions_cycle={'conditional':{'k32_plain':17000}}))
            chain.bind_freeze(folder,first)
            chain.bind_freeze(folder,first)
            changed = deepcopy(first)
            changed['sha256'] = 'changed'
            changed['contents']['predictions_cycle']['conditional']['k32_plain'] = 18000
            with self.assertRaisesRegex(ValueError,'冻结身份发生变化'):
                chain.bind_freeze(folder,changed)

    def test_heldout_freeze_requires_the_actual_k32_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'build').mkdir()
            (root/'dev').mkdir()
            chain.save_json(root/'plan.json',dict(test=True))
            chain.save_json(root/'build/identity.json',dict(test=True))
            chain.save_json(root/'dev/summary.json',dict(all_performance_eligible=True))
            freeze = dict(schema='gh200-cfg-b-compute-chain-freeze/1',
                plan_sha256=chain.digest(root/'plan.json'),
                build_identity_sha256=chain.digest(root/'build/identity.json'),
                dev_summary_sha256=chain.digest(root/'dev/summary.json'),
                criterion=dict(unit='clock64_cycle/CTA',max_abs_relative_error=.05,
                               statistic='median_of_3_eligible_processes'),
                candidate_parameters={'nominal':dict(a_cycle=128.,q_cycle=512.)},
                predictions_cycle={'nominal':dict(k32_plain=16512.)})
            chain.save_json(root/'valid.json',freeze)
            self.assertEqual(chain.verify_freeze(root,root/'valid.json')['contents'], freeze)
            freeze['predictions_cycle']['nominal']['k32_plain'] = 0
            chain.save_json(root/'wrong.json',freeze)
            with self.assertRaisesRegex(ValueError,'32 Ktile预测'):
                chain.verify_freeze(root,root/'wrong.json')


if __name__ == '__main__':
    unittest.main()
