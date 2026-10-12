import math
import unittest

import clock_model
import r09_input_source_clock as fit
from r09_v10_clock_expansion import adoption


class ClockExpansionContract(unittest.TestCase):
    def test_static_zero_weights_match_runtime_mixing(self):
        model = dict(tau_us=200.,coefficients=[1.98,.2,.1,.3,.02,.003,.04])
        for mode in fit.MODES:
            for z in (0.,.03125,.453125,1.):
                p = dict(input_mode=mode,tensor_mean_cycles=25000.,source_mean_kib=1200.,
                         observed_window_us=30.,
                         mode_weights=[float(mode==m)*(1-z)+float(m=='zero')*z for m in fit.MODES])
                a = float(fit.source_design([p],model['tau_us'])[0] @ model['coefficients'])
                mixed = fit.mixed_failure_clock(model,mode,z)
                b = clock_model.source_frequency(p,p['observed_window_us'],mixed)['frequency_ghz']
                self.assertAlmostEqual(a,b,places=13)

    def test_invalid_weights_are_rejected(self):
        p = dict(input_mode='dyadic',tensor_mean_cycles=10.,source_mean_kib=2.,observed_window_us=5.)
        for weights in ([1.,1.,0.],[-1.,1.,1.],[math.nan,0.,0.],[1.,0.]):
            with self.assertRaises(ValueError):
                fit.source_design([dict(p,mode_weights=weights)],20.)

    def rows(self):
        rows=[]
        for i in range(29):
            rows.append(dict(case=str(i),frozen_status='predicted',
                group='G6_base' if i<3 else 'other',in_proposed_scope=True,
                versions=dict(r2=dict(plain_error=.06,frequency_error=.10),
                              candidate=dict(plain_error=.05 if i<3 else .06,frequency_error=.09))))
        return rows

    def test_each_target_requires_frequency_and_time_improvement(self):
        rows=self.rows()
        self.assertTrue(adoption(rows,'candidate')['adopt'])
        rows[0]['versions']['candidate']['frequency_error']=.11
        self.assertFalse(adoption(rows,'candidate')['adopt'])

    def test_absolute_error_cannot_cancel_by_sign(self):
        rows=self.rows()
        rows[5]['versions']['r2']['plain_error']=-.06
        rows[5]['versions']['candidate']['plain_error']=.09
        self.assertEqual(adoption(rows,'candidate')['regression_over_2pp'][0]['case'],'5')

    def test_individual_regression_cannot_hide_under_aggregate(self):
        rows=self.rows()
        rows[5]['versions']['r2']['plain_error']=.001
        rows[5]['versions']['candidate']['plain_error']=.0221
        result=adoption(rows,'candidate')
        self.assertTrue(result['non_target_median_not_increased'])
        self.assertTrue(result['non_target_max_not_increased'])
        self.assertFalse(result['adopt'])

    def test_ten_percent_crossing_is_a_separate_gate(self):
        rows=self.rows()
        rows[5]['versions']['r2']['plain_error']=.099
        rows[5]['versions']['candidate']['plain_error']=.101
        result=adoption(rows,'candidate')
        self.assertFalse(result['regression_over_2pp'])
        self.assertEqual(result['supported_10pct_crossings'],['5'])

    def test_unavailable_and_wrong_comparison_set_are_not_dropped(self):
        rows=self.rows()
        rows[5]['versions']['candidate']=dict(error='unavailable')
        self.assertEqual(adoption(rows,'candidate')['unavailable'],['5'])
        with self.assertRaises(ValueError):
            adoption(rows[:-1],'candidate')


if __name__=='__main__':
    unittest.main()
