import copy
import unittest

from analyze_cfg_c_tail_control import evaluate


class TailControl(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(samples=[], gates=dict(per_common_series_median_absolute_ns=64,
            per_common_series_maximum_absolute_ns=256,event_median_absolute_ratio=.02,
            event_maximum_absolute_ratio=.05,off_event_population_cv=.05))
        self.rows = []
        for pair in range(10):
            for flag in ((0,1) if pair%2==0 else (1,0)):
                sample=dict(id=f'{pair}/{flag}',pair=pair,flag=flag)
                self.manifest['samples'].append(sample)
                self.rows.append(dict(sample,elapsed_us=100.,sms={str(c):c for c in range(132)},
                    series={'0/0/0/0,0/L':[1000,1500],'0/0/0/0,0/FIRST':[500,750],
                            '0/0/0/0,0/MAIN':[1500,2250]}))

    def test_pass_never_qualifies_tail_or_baseline(self):
        r=evaluate(self.rows,self.manifest)
        self.assertTrue(r['incremental_control_passed'])
        for key in ('observation_qualified','tail_precision_qualified','baseline_dual_qualified'):
            self.assertFalse(r[key])

    def test_common_shift_is_caught_even_when_L_unchanged(self):
        for p in self.rows:
            if p['flag']:
                p['series']['0/0/0/0,0/FIRST'][0]+=65
                p['series']['0/0/0/0,0/MAIN'][0]+=65
        r=evaluate(self.rows,self.manifest)
        self.assertFalse(r['incremental_control_passed'])
        self.assertTrue(next(x for x in r['local'] if x['series'].endswith('/L'))['passed'])

    def test_one_outlier_fails_without_median_failure(self):
        self.rows[1]['series']['0/0/0/0,0/L'][0]+=257
        r=evaluate(self.rows,self.manifest)
        self.assertFalse(r['incremental_control_passed'])
        self.assertEqual(next(x for x in r['local'] if x['series'].endswith('/L'))['median_absolute'],0)

    def test_missing_reordered_bool_and_coordinate_mismatch_rejected(self):
        variants=[self.rows[:-1],list(reversed(self.rows))]
        bad=copy.deepcopy(self.rows);bad[0]['flag']=False;variants.append(bad)
        bad=copy.deepcopy(self.rows);bad[-1]['series'].pop('0/0/0/0,0/L');variants.append(bad)
        for rows in variants:
            with self.assertRaises(ValueError):evaluate(rows,self.manifest)

    def test_signed_cancellation_and_sm_changes_retained(self):
        for p in self.rows:
            if p['flag']:
                p['series']['0/0/0/0,0/L'][0]+=100 if p['pair']%2 else -100
                p['sms']['0']=1
        r=evaluate(self.rows,self.manifest)
        local=next(x for x in r['local'] if x['series'].endswith('/L'))
        self.assertEqual(local['median_absolute'],100)
        self.assertEqual(local['sm_changed'],[True]*10)
        self.assertFalse(r['incremental_control_passed'])

    def test_whole_cta_translation_is_caught_by_common_origin(self):
        for p in self.rows:
            p['series']['0/GLOBAL_ENTRY']=[80 if p['flag'] else 0,None]
        r=evaluate(self.rows,self.manifest)
        self.assertFalse(r['incremental_control_passed'])
        global_row=next(x for x in r['local'] if x['series']=='0/GLOBAL_ENTRY')
        self.assertEqual(global_row['native_cycle_deltas'],[None]*10)

    def test_opposed_endpoints_require_direct_window_gate(self):
        for p in self.rows:
            p['series']['0/0/0/0,0/PERMIT']=[1600,2400]
            p['series']['0/0/0/0,0/W']=[100,150]
            if p['flag']:
                p['series']['0/0/0/0,0/MAIN'][0]-=40
                p['series']['0/0/0/0,0/PERMIT'][0]+=40
                p['series']['0/0/0/0,0/W'][0]+=80
        r=evaluate(self.rows,self.manifest)
        by_name={x['series'].split('/')[-1]:x for x in r['local']}
        self.assertTrue(by_name['MAIN']['passed'] and by_name['PERMIT']['passed'])
        self.assertFalse(by_name['W']['passed'])
        self.assertFalse(r['incremental_control_passed'])

    def test_event_outlier_fails_even_with_local_agreement(self):
        self.rows[1]['elapsed_us']=106
        result=evaluate(self.rows,self.manifest)
        self.assertTrue(all(row['passed'] for row in result['local']))
        self.assertFalse(result['event']['passed'])
        self.assertFalse(result['incremental_control_passed'])

    def test_off_instability_fails_with_exact_pair_matching(self):
        for p in self.rows:p['elapsed_us']=100 if p['pair']%2 else 120
        r=evaluate(self.rows,self.manifest)
        self.assertTrue(r['event']['passed'])
        self.assertFalse(r['incremental_control_passed'])


if __name__=='__main__':unittest.main()
