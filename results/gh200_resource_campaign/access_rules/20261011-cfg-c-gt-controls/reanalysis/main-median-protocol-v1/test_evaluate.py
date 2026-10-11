import copy
import unittest
from evaluate import evaluate


class MedianContractTests(unittest.TestCase):
    def setUp(self):
        self.protocol=dict(schema='gh200-median-equivalence-proposal/1',series=['L0','L1'],
            repeats_per_arm=64,family_failure_probability='1/20',median_difference_margin_ns=64)
        self.samples={cohort:{name:dict(arm0=[1000]*64,arm1=[1000]*64) for name in ('L0','L1')}
                      for cohort in ('off_on','off_off')}

    def test_equivalence_never_grants_qualification(self):
        result=evaluate(self.protocol,self.samples)
        self.assertTrue(result['all_statistical_intervals_equivalent'])
        self.assertFalse(result['observation_qualified'])
        self.assertFalse(result['physical_timestamp_accuracy_verified'])
        self.assertFalse(result['critical_cta_ordering_verified'])

    def test_zero_inside_wide_interval_is_not_equivalence(self):
        for row in self.samples['off_on'].values():
            row['arm1']=[500]*32+[1500]*32
        result=evaluate(self.protocol,self.samples)
        self.assertEqual(result['counts']['inconclusive'],2)
        self.assertFalse(result['all_statistical_intervals_equivalent'])

    def test_interval_outside_margin_rejects(self):
        self.samples['off_on']['L0']['arm1']=[1100]*64
        result=evaluate(self.protocol,self.samples)
        self.assertEqual(result['counts']['conditional non-equivalence'],1)

    def test_old_tail_failure_is_retained_when_medians_agree(self):
        self.samples['off_on']['L0']['arm1'][-1]=2000
        result=evaluate(self.protocol,self.samples)
        self.assertTrue(result['all_statistical_intervals_equivalent'])
        self.assertEqual(result['original_point_failures']['off_on'],1)

    def test_coordinate_family_cannot_shrink(self):
        del self.samples['off_on']['L0']
        with self.assertRaises(ValueError):evaluate(self.protocol,self.samples)

    def test_missing_repeat_cannot_be_filtered_out(self):
        self.samples['off_on']['L0']['arm1'].pop()
        with self.assertRaises(ValueError):evaluate(self.protocol,self.samples)

    def test_nan_and_bool_rejected(self):
        for bad in (float('nan'),True):
            changed=copy.deepcopy(self.samples)
            changed['off_on']['L0']['arm1'][0]=bad
            with self.assertRaises(ValueError):evaluate(self.protocol,changed)

    def test_n10_full_family_cannot_reuse_marginal_coverage(self):
        protocol=dict(self.protocol,series=[str(i) for i in range(432)],repeats_per_arm=10)
        samples={cohort:{name:dict(arm0=[1000]*10,arm1=[1000]*10) for name in protocol['series']}
                 for cohort in ('off_on','off_off')}
        result=evaluate(protocol,samples)
        self.assertEqual(result['counts']['insufficient simultaneous coverage'],864)
        self.assertIsNone(result['order_statistic_ranks'])


if __name__=='__main__':
    unittest.main()
