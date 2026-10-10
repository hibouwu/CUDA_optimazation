"""Check the measurement reductions and the stage max-branch diagnosis."""
import copy
import unittest

from analyze_r13_supply import (
    _failure_pair_summary, _failure_pair_windows, _failure_stage_response,
    _failure_k_scaling,
    _finite_stage_cycles,
)


def cta(index, length):
    return dict(cta=index, work=[[index, 0], [index, 1]], entry_ns=0, prod_ns=2,
        end_ns=2*length+40,
        tiles_ns=[[10, 10+length, 11+length, 15+length],
                  [20+length, 20+2*length, 21+2*length, 25+2*length]],
        tiles=[[20, 20+2*length, 22+2*length, 30+2*length],
               [40+2*length, 40+4*length, 42+4*length, 50+4*length]])


class PairedFailureTests(unittest.TestCase):
    def test_pairing_uses_identity_and_coordinates_not_list_order(self):
        a = [cta(0, 10), cta(1, 20)]
        b = [cta(1, 30), cta(0, 20)]
        rows = _failure_pair_windows(a, b)
        self.assertEqual([(r['cta'], r['j']) for r in rows], [(0,0),(0,1),(1,0),(1,1)])
        self.assertEqual([r['L_ns_delta'] for r in rows], [10,10,10,10])
        self.assertEqual(rows[0]['L_rate_ratio'], 0)
        b[0]['work'][1] = [99, 1]
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            _failure_pair_windows(a, b)

    def test_ratios_are_paired_before_reduction(self):
        a = [cta(0,1), cta(1,10), cta(2,100)]
        b = [cta(0,2), cta(1,100), cta(2,100)]
        control = [dict(trial=t,row=dict(id='base'),observed=dict(ctas=a)) for t in range(10)]
        variant = [dict(trial=t,row=dict(id='variant'),observed=dict(ctas=b)) for t in range(10)]
        summary, positions = _failure_pair_summary('synthetic', control, variant)
        self.assertEqual(summary['phases']['first']['L_ns_ratio']['median'], 1)
        self.assertEqual(summary['phases']['later']['L_ns_ratio']['median'], 1)
        self.assertEqual(len(positions), 20)
        # Ratio of condition medians would be 9, and is not our paired statistic.
        self.assertNotEqual(summary['phases']['first']['L_ns_ratio']['median'], 100/10-1)
        with self.assertRaisesRegex(ValueError, 'ten successful trials'):
            _failure_pair_summary('synthetic', control[:-1], variant)

    def test_compute_max_can_hide_a_changed_stage_feature(self):
        row = dict(id='base',config='cfg_b',m=2816,n=1536,k=4096,swizzle=1)
        setup = dict(grid=[1,132,1],stages=6,smem=215040,max_active_ctas_per_sm=1)
        model = dict(projection=[[1],[0],[0],[0],[0],[0],[0]], parameters=[10,.1])
        calibration = dict(first_supply=dict(cfg_b=model),supply=dict(cfg_b=model))
        prediction = dict(frequency_ghz=1.,window_us=1.,plain_us=1.,cta_duration_ns=[1.],cta_events=[])
        control = dict(row=row,setup=setup,calibration=calibration,prediction=prediction)
        variant = copy.deepcopy(control)
        variant['setup'].update(stages=4,smem=149504)
        result = _failure_stage_response(control, variant)
        first6, later6 = result['stage6']['positions']
        first4, later4 = result['stage4']['positions']
        self.assertEqual(first6['source_KiB'], first4['source_KiB'])
        self.assertNotEqual(later6['source_KiB'], later4['source_KiB'])
        self.assertEqual(later6['L_ns'], later4['L_ns'])
        self.assertEqual(later4['supply_active_windows'], 0)
        self.assertTrue(all(result['frozen_outputs_equal'].values()))

    def test_K_scaling_separates_fixed_from_per_tile_difference(self):
        def positions(value):
            return [dict(trial=t,j=0,T=2,windows=132,L_ns_delta=value,L_cycles_delta=2*value)
                for t in range(10)]
        fixed = _failure_k_scaling(positions(100),positions(100),16,64)['positions'][0]['statistics']
        sustained = _failure_k_scaling(positions(100),positions(400),16,64)['positions'][0]['statistics']
        self.assertEqual(fixed['fixed_only_residual_ns']['median'],0)
        self.assertNotEqual(fixed['per_ktile_only_residual_ns']['median'],0)
        self.assertNotEqual(sustained['fixed_only_residual_ns']['median'],0)
        self.assertEqual(sustained['per_ktile_only_residual_ns']['median'],0)

    def test_finite_slots_release_previous_group_after_wait1(self):
        result = _finite_stage_cycles(16,4,100,2200,512,detail=True)
        for i,issue in enumerate(result['issued']):
            self.assertEqual(result['ready'][i],issue+2200)
            if i>=4:
                self.assertGreaterEqual(issue,result['released'][i-4])
            self.assertGreaterEqual(result['released'][i],result['completed'][i])
            if i%16!=15:
                self.assertEqual(result['released'][i],max(result['first_mma_or_commit'][i+1],result['completed'][i]))
        self.assertEqual(result['released'][15],result['completed'][15])

    def test_pingpong_handoff_precedes_previous_tail_completion(self):
        result = _finite_stage_cycles(16,6,0,0,512,detail=True)
        self.assertLess(result['first_mma_or_commit'][16],result['completed'][15])
        self.assertEqual(result['windows'][0],16*512)
        # The second FIRST_MMA is before the previous last group completes.
        self.assertEqual(result['windows'][1],17*512)
        delayed = _finite_stage_cycles(16,6,0,200,512,detail=True)
        self.assertEqual(delayed['windows'][0],result['windows'][0])
        self.assertEqual(delayed['first_mma_or_commit'][0],200)


if __name__ == '__main__':
    unittest.main()
