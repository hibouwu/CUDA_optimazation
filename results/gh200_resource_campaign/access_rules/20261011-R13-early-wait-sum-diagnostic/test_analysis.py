import copy
from fractions import Fraction
import itertools
import unittest
from analyze import block_contrasts, summarize, CONTRASTS


class JointDiagnosticTests(unittest.TestCase):
    def fixture(self):
        return [dict(stages=s,observer=f,sm=124,early_cycles=d,full_wait_cycles=q,outside_wait_cycles=r)
                for s,f,d,q,r in [(4,0,11500,None,None),(4,1,10739,4633,6106),
                                  (6,0,7982,None,None),(6,1,8582,1301,7281)]]

    def test_real_smoke_arithmetic_not_effect_estimate(self):
        result=block_contrasts(self.fixture())
        self.assertEqual([result[k] for k in CONTRASTS],[2157,3332,-1175,4507,3518,-1361,-761,600])
        # These values reproduce smoke arithmetic, not a valid randomized block.

    def test_reject_missing_duplicate_sm_and_missingness(self):
        original=self.fixture()
        bad=[original[:3],original+[original[0]]]
        for key,value in [('sm',125),('full_wait_cycles',0),('early_cycles',True)]:
            rows=copy.deepcopy(original);rows[0][key]=value;bad.append(rows)
        rows=copy.deepcopy(original);rows[1]['outside_wait_cycles']+=1;bad.append(rows)
        for rows in bad:
            with self.assertRaises(ValueError):block_contrasts(rows)

    def test_exact_tail_by_full_sign_enumeration(self):
        zero={name:0 for name in CONTRASTS}
        result=summarize([zero]*16)
        self.assertEqual(result['ranks_one_based'],[3,14])
        outside=sum(sum(bits)<3 or sum(bits)>13 for bits in itertools.product((0,1),repeat=16))
        self.assertEqual(Fraction(outside,2**16),Fraction(result['per_interval_tail_bound']))
        self.assertLessEqual(Fraction(result['family_tail_bound']),Fraction(1,20))
        with self.assertRaises(ValueError):summarize([zero]*15)

    def test_joint_median_not_sum_of_component_medians(self):
        blocks=[]
        for i in range(16):
            q,r = ((0,100) if i<6 else (100,0) if i<12 else (100,100))
            row={name:0 for name in CONTRASTS}
            row.update(stage_Q_on=q,stage_R_on=r,stage_D_on=q+r)
            blocks.append(row)
        result=summarize(blocks)['summary']
        self.assertEqual(result['stage_D_on']['median'],100)
        self.assertEqual(result['stage_Q_on']['median']+result['stage_R_on']['median'],200)


if __name__=='__main__':unittest.main()
