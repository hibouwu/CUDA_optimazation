"""Check causal boundaries of the R09 failure decomposition."""
import copy
import unittest

from r09_input_source_clock import (fit_tensor_only, mixed_failure_clock,
                                    role_window_diagnostics, source_frequency, control_window)


class R09FailureDiagnostics(unittest.TestCase):
    def test_reduced_fit_ignores_time_error_and_recovers_frequency_targets(self):
        points = []
        for mode, d in [('dyadic', .2), ('zero', .1), ('random', .3)]:
            for rate in (.2, .7, 1.2):
                points.append(dict(input_mode=mode, tensor_mean_cycles=rate*1000*100,
                    source_mean_kib=100, observed_window_us=100, observed_ghz=1.9-d*rate,
                    plain_us=1., converted_cycles=1.))
        fitted = fit_tensor_only(points, 1.98)
        self.assertEqual(fitted['rank'], 4)
        for observed, expected in zip(fitted['coefficients'], [1.9,.2,.1,.3,0,0,0]):
            self.assertAlmostEqual(observed, expected, places=10)
        altered = [dict(p, plain_us=1e9, converted_cycles=1e9) for p in points]
        self.assertEqual(fitted['coefficients'], fit_tensor_only(altered, 1.98)['coefficients'])

    def test_zero_mixture_changes_only_selected_mode_and_preserves_model(self):
        model = dict(coefficients=[2.,.2,.1,.3,.04,.01,.05], tau_us=100)
        original = copy.deepcopy(model)
        mixed = mixed_failure_clock(model, 'dyadic', .25)
        self.assertEqual(model, original)
        self.assertAlmostEqual(mixed['coefficients'][1], .175)
        self.assertAlmostEqual(mixed['coefficients'][4], .0325)
        self.assertEqual(mixed['coefficients'][2:4], model['coefficients'][2:4])
        self.assertEqual(mixed_failure_clock(model, 'dyadic', .25, False), model)

    def test_same_role_frequency_does_not_use_merged_role_endpoints(self):
        # Role 0 wins the cycle endpoint; role 1 wins ns. Their individual
        # frequencies remain 2 and 1, instead of the merged ratio 8000/7000.
        roles = [dict(tiles=[[100,8100,8200,8300]], tiles_ns=[[1000,5000,5100,5200]],
                      final_c=8400, final_ns=5300),
                 dict(tiles=[[100,7100,7200,7300]], tiles_ns=[[1000,8000,8100,8200]],
                      final_c=7400, final_ns=8300)]
        cta = dict(entry_c=0, entry_ns=0, prod_c=50, prod_ns=500,
                   roles=roles, role_work=[[(0,0)],[(0,0)]])
        report = role_window_diagnostics([cta])
        self.assertEqual(report['same_role']['first_L']['ghz']['median'], 1.5)
        self.assertEqual(report['same_role']['first_L']['frequency_windows'], 2)
        self.assertEqual(report['same_role']['E']['frequency_windows'], 0)
        self.assertEqual(report['merged_endpoint_selection']['different_role'], 3)

        # Pingpong's second consumer starts with global tile 1, which belongs to
        # later L even though it is that consumer's first local record.
        cta.update(work=[(0,0),(0,1)], role_work=[[(0,0)],[(0,1)]])
        pingpong = role_window_diagnostics([cta])
        self.assertEqual(pingpong['same_role']['first_L']['ghz']['median'], 2.)
        self.assertEqual(pingpong['same_role']['later_L']['ghz']['median'], 1.)

    def test_conditional_frequency_remains_negative_when_formula_fails(self):
        point = dict(input_mode='dyadic', tensor_mean_cycles=10000, source_mean_kib=10000)
        model = dict(coefficients=[1.,1.,0,0,1.,0,0], tau_us=1.)
        self.assertLess(source_frequency(point, 1., model)['frequency_ghz'], 0)

    def test_protocol_frequency_uses_local_windows_not_global_envelope(self):
        ctas = [dict(entry_c=0,end_c=400,entry_ns=0,end_ns=200,tiles=2,sm=0),
                dict(entry_c=1000,end_c=1400,entry_ns=100,end_ns=300,tiles=2,sm=1)]
        result = control_window(dict(ctas=ctas,elapsed_us=4.3))
        self.assertEqual(result['frequency_ghz'], 2.)
        self.assertEqual(result['window_us'], .3)
        self.assertEqual(result['event_minus_window_us'], 4.)
        dual_ctas = [dict(c, tiles=[[0,1,2,3],[4,5,6,7]]) for c in ctas]
        self.assertEqual(result, control_window(dict(ctas=dual_ctas,elapsed_us=4.3)))


if __name__ == '__main__':
    unittest.main()
