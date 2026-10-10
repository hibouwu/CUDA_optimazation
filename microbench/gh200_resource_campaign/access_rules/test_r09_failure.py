"""Check causal boundaries of the R09 failure decomposition."""
import copy
import statistics
import unittest

from r09_input_source_clock import (fit_tensor_only, mixed_failure_clock,
                                    role_window_diagnostics, source_frequency, control_window,
                                    reciprocal_frequency, fit_reciprocal_clock, source_design, static_output_kib,
                                    static_revisit_features, history_state_weights, history_series_design, fit_history_series, history_localization_diagnostics)


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

    def test_reciprocal_response_is_positive_bounded_and_monotone_in_window(self):
        point = dict(input_mode='dyadic',tensor_mean_cycles=100000,source_mean_kib=10000)
        model = dict(coefficients=[1/1.98,.2,.1,.3,.04,.01,.05],tau_us=100)
        values = [reciprocal_frequency(point,w,model) for w in (.001,1,10,100,1000,1e6)]
        self.assertTrue(all(0 < f <= 1.98 for f in values))
        self.assertEqual(values, sorted(values))
        # A monotone positive envelope produces a decreasing closure residual.
        frequencies = [.01,.1,.5,1,1.5,1.98]
        residuals = [reciprocal_frequency(point,100/f+10,model)-f for f in frequencies]
        self.assertEqual(residuals, sorted(residuals,reverse=True))
        self.assertGreater(residuals[0],0)
        self.assertLessEqual(residuals[-1],0)

    def test_reciprocal_fit_optimizes_ghz_not_reciprocal_targets(self):
        import numpy as np
        from scipy.optimize import lsq_linear
        true = dict(coefficients=[.52,.04,.02,.08,.02,.003,.04],tau_us=100)
        points = []
        for mode in ('dyadic','zero','random'):
            for i,window in enumerate((10,100,1000)):
                for j,rate in enumerate((.2,1.1)):
                    p = dict(input_mode=mode,observed_window_us=window,
                             tensor_mean_cycles=1000*window*rate,source_mean_kib=window*(1+3*j))
                    p['observed_ghz'] = reciprocal_frequency(p,window,true) + (.02 if (i+j)%2 else -.02)
                    points.append(p)
        fitted = fit_reciprocal_clock(points,1.98)
        design = source_design(points,fitted['tau_us'])
        design[:,1:] *= -1
        y = np.asarray([p['observed_ghz'] for p in points])
        inverse_fit = lsq_linear(design,1/y,bounds=([1/1.98,0,0,0,0,0,0],[np.inf]*7),method='bvls')
        inverse_target_ghz_mse = float(np.mean((1/(design@inverse_fit.x)-y)**2))
        direct = np.asarray([reciprocal_frequency(p,p['observed_window_us'],fitted) for p in points])
        self.assertAlmostEqual(fitted['mse_ghz'],float(np.mean((direct-y)**2)),places=12)
        self.assertLess(fitted['mse_ghz'],inverse_target_ghz_mse-1e-9)
        self.assertTrue(all(0 < f <= 1.98 for f in direct))

    def test_output_proxy_uses_valid_extent_and_not_timing_or_k(self):
        row = dict(m=129,n=128,k=1024,observed_E_ns=100,observed_window_us=10)
        value = static_output_kib(row)
        self.assertEqual(value,4*129*128/(132*1024))
        self.assertEqual(value,static_output_kib(dict(row,k=32768,observed_E_ns=999999,observed_window_us=5000)))
        self.assertLess(value,4*256*128/(132*1024))

    def test_swizzle_changes_revisit_opportunity_without_changing_work(self):
        row = dict(config='cfg_b',m=6144,n=8192,k=32768,swizzle=1)
        a = static_revisit_features(row,dict(grid=[132,1,1]))
        b = static_revisit_features(dict(row,swizzle=8),dict(grid=[132,1,1]))
        for key in ('tensor_mean_cycles','nominal_source_mean_KiB','output_mean_KiB','output_tiles'):
            self.assertEqual(a[key],b[key])
        self.assertEqual((a['panel_slots'],b['panel_slots']),(1222,644))
        self.assertLess(b['unique_source_mean_KiB'],a['unique_source_mean_KiB'])
        with self.assertRaisesRegex(ValueError,'full in-bounds'):
            static_revisit_features(dict(row,n=8191),dict(grid=[132,1,1]))

    def test_revisit_union_counts_multicast_source_slices_once(self):
        a = static_revisit_features(dict(config='cfg_a',m=256,n=256,k=64,swizzle=1),dict(grid=[2,2,1]))
        c = static_revisit_features(dict(config='cfg_c',m=256,n=256,k=64,swizzle=1),dict(grid=[1,2,1]))
        self.assertEqual(a['waves'][0]['unique_source_KiB_per_Ktile'],64)
        self.assertEqual(c['waves'][0]['unique_source_KiB_per_Ktile'],64)
        self.assertEqual((a['waves'][0]['unique_A_boxes'],a['waves'][0]['unique_B_boxes']),(2,8))
        self.assertEqual((c['waves'][0]['unique_A_boxes'],c['waves'][0]['unique_B_boxes']),(2,4))
        # Individually full tiles may still require a padded cluster member.
        with self.assertRaisesRegex(ValueError,'cluster padding'):
            static_revisit_features(dict(config='cfg_c',m=512,n=3968,k=1024,swizzle=1),dict(grid=[66,2,1]))

    def test_history_state_uses_interval_mean_not_terminal_clock(self):
        import math
        H,W,tau=10000.,5000.,20000.
        common=history_state_weights('common-sw1-16',8,H,W,tau)
        self.assertAlmostEqual(sum(common),1.)
        self.assertTrue(all(v>=0 for v in common))
        mean_factor=-math.expm1(-W/tau)/(W/tau)
        self.assertAlmostEqual(common[0],math.exp(-H/tau)*mean_factor)
        self.assertAlmostEqual(common[2],1-mean_factor)
        adaptive=history_state_weights('adaptive',8,H,W,tau)
        self.assertEqual(adaptive[1],0.)
        self.assertAlmostEqual(adaptive[2],1-adaptive[0])


    def test_series_state_preserves_local_interval_means_and_observed_gaps(self):
        import math
        import numpy as np
        calls=[dict(index=0,swizzle=1,first_entry_ns=0,frequency_observers_us=[[0.,2.],[1.,2.]]),
               dict(index=1,swizzle=8,first_entry_ns=10000,frequency_observers_us=[[0.,2.]])]
        matrix=history_series_design([dict(calls=calls)],10.)
        self.assertTrue(np.allclose(matrix.sum(axis=1),1.))
        average=-math.expm1(-.2)/.2
        attenuation=statistics.median([average,math.exp(-.1)*average])
        self.assertAlmostEqual(matrix[0,0],attenuation)
        self.assertAlmostEqual(matrix[1,0],math.exp(-1)*average)
        self.assertAlmostEqual(matrix[1,1],(1-math.exp(-1))*average)
        self.assertAlmostEqual(matrix[1,2],1-average)
        longer=copy.deepcopy(calls);longer[1]['first_entry_ns']=20000
        self.assertLess(history_series_design([dict(calls=longer)],10.)[1,0],matrix[1,0])

    def test_series_temporal_holdout_does_not_fit_last_nine_targets(self):
        import numpy as np
        processes=[]
        for sw in (1,8):
            calls=[dict(index=j,swizzle=1 if j<16 else sw,first_entry_ns=j*1000000,
                frequency_observers_us=[[0.,900.]],window_us=900.,globaltimer_gap_before_us=100.) for j in range(35)]
            processes.append(dict(case='sw'+str(sw),trial=0,swizzle=sw,calls=calls))
        observed=history_series_design(processes,10000.)@np.asarray([1.8,1.7,1.,1.3])
        for c,y in zip([c for p in processes for c in p['calls']],observed):c['frequency_ghz']=float(y)
        original=fit_history_series(processes,1.98)
        changed=copy.deepcopy(processes)
        for p in changed:
            for c in p['calls'][26:]:c['frequency_ghz']*=.9
        altered=fit_history_series(changed,1.98)
        key='prefix_and_first10_targets'
        self.assertEqual(original['fits'][key]['best'],altered['fits'][key]['best'])
        self.assertEqual(original['fits'][key]['tau_profile'],altered['fits'][key]['tau_profile'])
        self.assertLess(original['last9_comparison']['state']['rms'],1e-6)
        self.assertGreater(altered['last9_comparison']['state']['rms'],.1)


    def test_history_localization_reuses_paired_role_windows_and_allows_sm_remapping(self):
        import hashlib,json,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'cases.json').write_text('[]')
            calls=[]
            for j in range(35):
                ns=100 if j<6 else 120
                calls.append(dict(event_us=ns/1000,first_entry_ns=j*1000,
                    role_windows=[[0,0,1,j%2,ns*2,ns],[0,1,1,j%2,ns*2,ns]]))
            process=dict(case='synthetic',mode='series',swizzle=1,trial=0,host_start_ns=0,raw_sha256='synthetic',
                prefix_call_event_us=[c['event_us'] for c in calls[:16]],event_gaps_us=[.9]*34,calls=calls)
            report=dict(run=str(root),environment={},processes=[process],verification={'synthetic':True},
                input_sha256={'cases.json':hashlib.sha256(b'[]').hexdigest()})
            path=root/'A-history-series.json';path.write_text(json.dumps(report))
            result=history_localization_diagnostics(root,root)
            onset=result['processes'][0]['matched_role_onsets']['0.05']
            self.assertEqual(onset['ns']['index0'],6)
            self.assertEqual(onset['C']['index0'],6)
            self.assertIsNone(onset['f']['index0'])
            self.assertEqual(result['processes'][0]['matched_role_calls'][6]['same_sm_as_baseline_fraction'],0)
            self.assertEqual(result['input_sha256'][str(path)],hashlib.sha256(path.read_bytes()).hexdigest())
            report['processes'][0]['calls'][3]['role_windows'][0][-1]=0
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'invalid same-role'):
                history_localization_diagnostics(root,root)


if __name__ == '__main__':
    unittest.main()
