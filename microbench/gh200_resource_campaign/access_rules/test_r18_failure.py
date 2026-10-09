"""Boundary diagnostics keep prediction ties and same-role time endpoints."""
import unittest
import copy
from analyze_r18 import boundary_geometry, critical_replacements, boundary_stage_rows, stage_process_summary, paired_boundary_windows

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

if __name__=='__main__':unittest.main()
