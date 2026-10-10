"""Boundary diagnostics keep prediction ties and same-role time endpoints."""
import unittest
import copy
from analyze_r18 import boundary_geometry, critical_replacements, boundary_stage_rows, stage_process_summary, paired_boundary_windows, static_boundary_features, fit_fill_hypotheses

class FailureDiagnosticsTests(unittest.TestCase):
    def test_geometry_uses_both_logical_axes(self):
        row=dict(config='cfg_a',m=960,n=2816)
        self.assertEqual(boundary_geometry(row,(6,0)),'real')
        self.assertEqual(boundary_geometry(row,(7,0)),'partial')
        self.assertEqual(boundary_geometry(row,(8,0)),'oob')
        self.assertEqual(boundary_geometry(dict(config='cfg_c',m=4096,n=2240),(0,17)),'partial')

    def test_entry_and_internal_replacements_keep_frozen_ties(self):
        ctas=[dict(cta=i,sm=i,entry_ns=e,end_ns=e+d,entry_c=0,end_c=cy,work=[(i,0)])
              for i,(e,d,cy) in enumerate([(1000,100,110),(1050,80,120),(1000,100,100)])]
        pred=dict(cta_duration_ns=[100,90,100],critical_candidates=[0,2],critical_cta=0)
        value=critical_replacements(ctas,pred)
        self.assertEqual(value['observed_last'],[1])
        self.assertEqual(value['observed_longest_ns'],[0,2])
        self.assertEqual(value['observed_longest_cycles'],[1])
        self.assertEqual(value['entry_only_candidates'],[1])
        self.assertFalse(value['frozen_hit']);self.assertTrue(value['entry_only_hit'])
        self.assertFalse(value['internal_only_hit'])
        self.assertEqual(value['frozen_candidates'],[0,2])
        self.assertEqual(value['entry_only_envelope_ns'],140)
        self.assertEqual(value['internal_only_envelope_ns'],100)

    def test_stage_pairs_do_not_merge_roles(self):
        a=dict(tiles=[[10,30,35,45]],tiles_ns=[[100,140,145,165]],final_c=50,final_ns=170)
        b=dict(tiles=[[12,28,36,44]],tiles_ns=[[110,138,150,164]],final_c=49,final_ns=169)
        c=dict(cta=0,roles=[a,b],role_work=[[(0,0)],[(0,0)]])
        rows=boundary_stage_rows(dict(id='case',config='cfg_a',m=128,n=128),dict(trial=0,ctas=[c]))
        L=[r for r in rows if r['stage']=='L']
        self.assertEqual([(r['cycles'],r['ns']) for r in L],[(20,40),(16,28)])
        summary=stage_process_summary(rows)
        self.assertEqual(next(r for r in summary if r['stage']=='L' and r['role']==0)['ns'],40)
        self.assertEqual(next(r for r in summary if r['stage']=='L' and r['role']==0)['windows'],1)

    def test_map_pair_matches_coordinates_before_subtraction(self):
        role=dict(tiles=[[10,30,35,45]],tiles_ns=[[100,140,145,165]],final_c=50,final_ns=170)
        cta=dict(cta=0,work=[(7,0)],roles=[role,copy.deepcopy(role)],role_work=[[(7,0)],[(7,0)]])
        left=dict(trial=3,ctas=[cta]);right=copy.deepcopy(left)
        right['ctas'][0]['roles'][0]['tiles_ns'][0][1]+=10
        row=dict(id='p1',config='cfg_a',m=960,n=2816)
        pairs=paired_boundary_windows(row,left,right)
        target=next(p for p in pairs if p['role']==0 and p['stage']=='L')
        self.assertEqual(target['delta_ns'],10)
        self.assertEqual(target['geometry'],'partial')
        self.assertEqual(target['cluster_boundary'],'partial')
        self.assertEqual(target['delta_cycles'],0)
        right['ctas'][0]['role_work'][0][0]=(6,0)
        with self.assertRaisesRegex(ValueError,'work coordinates'):
            paired_boundary_windows(row,left,right)

    def test_static_cluster_features_reach_complete_partner_not_operand_mask(self):
        row=dict(id='p1',config='cfg_a',m=960,n=2816,k=4096,lda=4096,ldb=2816,
                 input_map_m=960,input_map_n=2816,storage_m=1024,storage_n=2816,swizzle=1)
        setup=dict(grid=[132,1,1],tile=[128,128,64],cluster=[2,1],stages=6)
        fs={tuple(f['coord']):f for f in static_boundary_features(row,setup)}
        real=fs[6,0]['operands']['A'];partial=fs[7,0]['operands']['A']
        self.assertEqual((real['own'],real['mask_max'],real['mask_sum']),(0,0,0))
        self.assertEqual((real['cluster_max'],real['cluster_sum'],real['cluster_any']),(8,8,1))
        self.assertEqual((partial['own'],partial['cluster_max'],partial['cluster_sum']),(8,8,8))
        cleared=static_boundary_features(dict(row,input_map_m=1024),setup)
        self.assertTrue(all(f['operands']['A']['cluster_any']==0 for f in cleared))

    def test_static_middle_map_separates_sum_from_max_and_state(self):
        row=dict(id='old',config='cfg_a',m=832,n=4096,k=4096,lda=4096,ldb=4096,
                 input_map_m=832,input_map_n=4096,storage_m=1024,storage_n=4096,swizzle=1)
        setup=dict(grid=[132,1,1],tile=[128,128,64],cluster=[2,1],stages=6)
        def full_oob(mapped):
            return next(f['operands']['A'] for f in static_boundary_features(dict(row,input_map_m=mapped),setup) if f['coord']==[7,0])
        a,b=full_oob(832),full_oob(896)
        self.assertEqual((a['own'],b['own']),(16,16))
        self.assertEqual((a['cluster_max'],b['cluster_max']),(16,16))
        self.assertEqual((a['cluster_sum'],b['cluster_sum']),(24,16))
        self.assertEqual((a['cluster_any'],b['cluster_any']),(1,1))

    def test_four_map_controls_keep_work_and_separate_cluster_hypotheses(self):
        import json
        from pathlib import Path
        import numpy as np
        from supply_model import boundary_fill_features
        rows=json.loads((Path(__file__).parent/'configs/r18-cluster-fill-discrimination.json').read_text())
        controls=[r for r in rows if r['set']=='ctrl']
        baseline={k:v for k,v in controls[0].items() if k not in ('id','input_map_m')}
        setup=dict(grid=[132,1,1],tile=[128,128,64],cluster=[2,1],stages=6)
        expected=[(16,24,1),(16,16,1),(8,8,1),(0,0,0)]
        work=[];design=[]
        for row,values in zip(controls,expected):
            self.assertEqual({k:v for k,v in row.items() if k not in ('id','input_map_m')},baseline)
            fs=boundary_fill_features(row,setup)
            work.append([(f['cta'],f['j'],tuple(f['coord'])) for f in fs])
            feature=next(f['operands']['A'] for f in fs if f['coord']==[6,0])
            vector=tuple(feature[k] for k in ('cluster_max','cluster_sum','cluster_any'))
            self.assertEqual(vector,values);design.append(vector)
        self.assertTrue(all(w==work[0] for w in work))
        self.assertEqual(np.linalg.matrix_rank(design),3)
        self.assertEqual(len(controls),4)

    def test_four_maps_separate_competing_fill_prices(self):
        patterns=[(16,24,1),(16,16,1),(8,8,1),(0,0,0)]
        for truth in ('cluster_max','cluster_sum','cluster_any'):
            rows=[]
            for a,b,c in patterns:
                row=dict(cluster_max=a,cluster_sum=b,cluster_any=c)
                row['ns_excess']=100*row[truth];rows.append(row)
            fitted={r['hypothesis']:r for r in fit_fill_hypotheses(rows)}
            self.assertAlmostEqual(fitted[truth]['rms_ns'],0)
            self.assertAlmostEqual(fitted[truth]['coefficients'][truth],100)
            for alternative in ('cluster_max','cluster_sum','cluster_any'):
                if alternative!=truth:self.assertGreater(fitted[alternative]['rms_ns'],0)
            self.assertEqual(fitted['mixed']['rank'],3)
            self.assertLess(fitted['mixed']['rms_ns'],1e-9)

if __name__=='__main__':unittest.main()
