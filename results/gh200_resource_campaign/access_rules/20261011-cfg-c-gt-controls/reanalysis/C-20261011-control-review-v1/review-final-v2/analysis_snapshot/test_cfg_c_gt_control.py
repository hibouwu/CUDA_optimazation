import unittest
from analyze_cfg_c_gt_control import evaluate
import test_cfg_c_tail_control as tail_tests


class GtControl(unittest.TestCase):
    def inputs(self, comparison='off_on'):
        fixture=tail_tests.TailControl();fixture.setUp()
        manifest=fixture.manifest;manifest['comparison']=comparison
        rows=fixture.rows
        for sample,row in zip(manifest['samples'],rows):
            if comparison=='off_off':
                sample['arm']=sample['flag'];row['arm']=row['flag']
                sample['flag']=0;row['flag']=0
            row['brackets']={}
            if row['flag']:
                for i in range(432):
                    row['brackets'][str(i)]=dict(post_gt_error_bound=32,
                        stamps=[dict(gt_before=100,gt_after=132),dict(gt_before=200,gt_after=232)])
        return rows,manifest

    def test_both_controls_keep_qualification_false(self):
        for comparison in ('off_on','off_off'):
            rows,m=self.inputs(comparison);result=evaluate(rows,m)
            self.assertFalse(result['observation_qualified'])
            self.assertFalse(result['gt_bracket_precision_qualified'])
            self.assertFalse(result['paired_interval_differences_available'])
            self.assertEqual(len(result['enabled_read_diagnostics']),432 if comparison=='off_on' else 0)

    def test_read_bound_does_not_erase_point_failure(self):
        rows,m=self.inputs()
        for row in rows:
            if row['flag']:row['series']['0/0/0/0,0/L'][0]+=300
        r=evaluate(rows,m)
        self.assertFalse(r['incremental_control_passed'])
        self.assertEqual(r['enabled_read_diagnostics'][0]['maximum_point_error_bound'],32)

    def test_width_not_silently_turned_into_gate_or_clipped(self):
        rows,m=self.inputs();row=next(x for x in rows if x['flag'])
        row['brackets']['0']['post_gt_error_bound']=1024
        result=evaluate(rows,m)
        diagnostic=next(x for x in result['enabled_read_diagnostics'] if x['series']=='0')
        self.assertEqual(diagnostic['maximum_point_error_bound'],1024)
        self.assertTrue(diagnostic['diagnostic_only'])
        self.assertFalse(result['gt_bracket_precision_qualified'])

    def test_missing_enabled_window_and_disabled_brackets_rejected(self):
        rows,m=self.inputs();next(x for x in rows if x['flag'])['brackets'].pop('0')
        with self.assertRaises(ValueError):evaluate(rows,m)
        rows,m=self.inputs('off_off');rows[0]['brackets']['bad']={}
        with self.assertRaises(ValueError):evaluate(rows,m)


if __name__=='__main__':unittest.main()
