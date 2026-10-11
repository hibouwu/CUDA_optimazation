"""Fixed comparison retention, regression guards, and first-L event placement."""
import json
from pathlib import Path
import unittest

import analyze_v10_task_a as task
import v09_model


class AdoptionChecks(unittest.TestCase):
    def test_missing_candidate_cannot_reduce_comparison(self):
        old = {'target':35., 'outside':20., 'inside':3.}
        new = {'target':5., 'inside':3.}
        result = task.adoption(old,new,['target'],['target','inside'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['fixed_conditions'],3)
        self.assertIn('comparison identity mismatch',[v['reason'] for v in result['violations']])

    def test_existing_failure_cannot_worsen_behind_unchanged_median(self):
        old = {'target':35., 'a':1., 'b':2., 'c':3., 'bad':20.}
        new = dict(old,target=5.,bad=21.)
        result = task.adoption(old,new,['target'],list(old))
        self.assertFalse(result['passed'])
        self.assertIn('non-target maximum regressed',[v['reason'] for v in result['violations']])

    def test_ten_percent_guard_is_not_replaced_by_two_point_budget(self):
        old = {'target':35., 'near':9., 'bad':20., 'a':1.}
        new = dict(old,target=5.,near=10.5,bad=19.)
        result = task.adoption(old,new,['target'],['target','near','a'])
        self.assertFalse(result['passed'])
        self.assertIn('supported condition crossed 10 percent',[v['reason'] for v in result['violations']])

    def test_every_target_must_improve(self):
        old = {'target_a':35., 'target_c':7., 'other':2.}
        new = {'target_a':5., 'target_c':8., 'other':2.}
        self.assertFalse(task.adoption(old,new,['target_a','target_c'],list(old))['passed'])

    def test_good_candidate_and_static_scope(self):
        old = {'target':35., 'outside':20., 'inside':3.}
        self.assertTrue(task.adoption(old,dict(old,target=5.),['target'],['target','inside'])['passed'])
        row=dict(config='cfg_b',lda=1024,ldb=1536)
        self.assertEqual(task.scope_reasons(row,dict(stages=6)),[])
        self.assertEqual(task.scope_reasons(dict(row,lda=1032),dict(stages=6)),['lda is not 128 B aligned'])
        self.assertEqual(task.scope_reasons(row,dict(stages=4)),['non-default stage count'])


class FirstResidualChecks(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[3]/'results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2'
        if not root.exists():
            self.skipTest('local frozen V09 archive required')
        self.calibration=json.loads((root/'frozen/calibration.json').read_text())
        self.calibration['clock']=dict(coefficients=[2.,0,0,0,0,0,0],tau_us=1.)
        self.calibration['zero_activity_mixing']=False
        self.row=next(r for r in json.loads((root/'cases.json').read_text()) if r['id']=='v09d_cfg_b_G3_base')
        self.setup=next(s['setup'] for s in json.loads((root/'static_setup.json').read_text()) if s['case']==self.row['id'])

    def test_zero_residual_preserves_complete_output(self):
        baseline=v09_model.predict_components(self.row,self.setup,self.calibration)
        extra=[0.]*len(baseline['cta_events'])
        self.assertEqual(v09_model.predict_components(self.row,self.setup,self.calibration,first_loop_extra_ns=extra),baseline)

    def test_residual_changes_first_L_not_later_L_or_prefill(self):
        baseline=v09_model.predict_components(self.row,self.setup,self.calibration)
        extra=[0.]*len(baseline['cta_events']);extra[0]=4096.
        changed=v09_model.predict_components(self.row,self.setup,self.calibration,first_loop_extra_ns=extra)
        old=baseline['cta_events'][0];new=changed['cta_events'][0]
        self.assertEqual(new['prefill'],old['prefill'])
        self.assertEqual(new['events'][0][0],old['events'][0][0])
        self.assertAlmostEqual(new['events'][0][1]-new['events'][0][0],old['events'][0][1]-old['events'][0][0]+4096.)
        for a,b in zip(old['events'][1:],new['events'][1:]):
            self.assertAlmostEqual(b[1]-b[0],a[1]-a[0])
        self.assertGreater(changed['cta_duration_ns'][0],baseline['cta_duration_ns'][0])
        self.assertEqual(changed['frequency_ghz'],2.)

    def test_invalid_residual_is_rejected(self):
        count=len(v09_model.events.scheduled_work(self.row['config'],self.row['m'],self.row['n'],self.setup['grid'],self.row['swizzle']))
        for extra in ([0.]*(count-1), [float('nan')]+[0.]*(count-1), [-1.]+[0.]*(count-1), [True]+[0.]*(count-1)):
            with self.subTest(extra=extra[:1]):
                with self.assertRaises(ValueError):
                    v09_model.predict_components(self.row,self.setup,self.calibration,first_loop_extra_ns=extra)


if __name__=='__main__':
    unittest.main()
