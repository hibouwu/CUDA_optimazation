import copy
from fractions import Fraction
import itertools
import unittest
from analyze import contrasts,summarize


class FeedbackContrastTests(unittest.TestCase):
    def block(self,x=100,y=20):
        rows=[]
        for stage in (4,6):
            for mode in (1,2,3):
                q2=50 if mode in (1,2) else None
                q16=(50+x if mode==1 else 50) if mode in (1,3) else None
                if mode==2:q2+=y
                rows.append(dict(stages=stage,observer=mode,sm=124,group2_wait_cycles=q2,
                    group16_wait_cycles=q16,full_wait_cycles=250 if mode==1 else None,
                    outside_wait_cycles=250 if mode==1 else None))
        return rows

    def test_joint_difference_and_missingness(self):
        result=contrasts(self.block())
        self.assertEqual([result['stage4_'+k] for k in ('L16','L2','F')],[100,-20,120])
        bad=[self.block()[:-1],self.block()+[self.block()[0]]]
        rows=self.block();rows[1]['full_wait_cycles']=50;bad.append(rows)
        rows=self.block();rows[2]['group2_wait_cycles']=0;bad.append(rows)
        rows=self.block();rows[2]['sm']=125;bad.append(rows)
        rows=self.block();rows[-1]=copy.deepcopy(rows[0]);bad.append(rows)
        for rows in bad:
            with self.assertRaises(ValueError):contrasts(rows)

    def test_joint_F_median_not_difference_of_medians(self):
        records=[]
        for i in range(16):records+=self.block(0 if i<9 else 100,100 if 2<=i<9 else 0)
        result=summarize(records)['summary']
        self.assertEqual(result['stage4_L16']['median'],0)
        self.assertEqual(result['stage4_L2']['median'],0)
        self.assertEqual(result['stage4_F']['median'],100)
        with self.assertRaises(ValueError):summarize(records[:-1])

    def test_conditional_rank_by_sign_enumeration(self):
        result=summarize(self.block()*16)
        self.assertEqual(result['ranks_one_based'],[3,14])
        rejected=sum(sum(signs)<3 or sum(signs)>13 for signs in itertools.product((0,1),repeat=16))
        self.assertEqual(Fraction(rejected,2**16),Fraction(result['per_interval_tail_bound']))
        self.assertEqual(Fraction(result['family_tail_bound']),Fraction(411,16384))


if __name__=='__main__':unittest.main()
