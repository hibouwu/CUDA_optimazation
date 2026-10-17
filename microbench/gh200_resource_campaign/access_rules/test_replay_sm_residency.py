"""R16原评分比较只承认两项已核CV显示差，不授通用容差。"""
import copy
import json
import math
import unittest

import cfg_b_composition as aggregate
import replay_sm_residency as replay


class ResidencyScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        results = aggregate.DEFAULT_ARCHIVE.parents[2]
        cls.original = json.loads((results / replay.R16 / 'analysis/rules.json').read_text())

    def with_known_differences(self):
        changed = copy.deepcopy(self.original)
        for row in changed['conditions']:
            if row['case'] in replay.KNOWN_CV_DIFFERENCES:
                row['cv'] = replay.KNOWN_CV_DIFFERENCES[row['case']][1]
        return changed

    def test_exact_original_has_no_diagnostic(self):
        self.assertEqual(replay.compare_r16_scores(self.original, copy.deepcopy(self.original)), [])

    def test_two_specific_cv_values_preserve_original_qualification(self):
        changed = self.with_known_differences()
        before = copy.deepcopy(changed)
        diagnostics = replay.compare_r16_scores(self.original, changed)
        self.assertEqual({row['case'] for row in diagnostics}, set(replay.KNOWN_CV_DIFFERENCES))
        self.assertTrue(all(row['old_qualified'] and row['new_qualified'] and
                            row['old_threshold_pass'] and row['new_threshold_pass'] for row in diagnostics))
        self.assertTrue(all(row['old_hex'] == row['old'].hex() and
                            row['new_hex'] == row['new'].hex() for row in diagnostics))
        self.assertEqual(changed, before)

    def test_another_one_ulp_cv_change_is_rejected(self):
        changed = self.with_known_differences()
        row = changed['conditions'][0]
        row['cv'] = math.nextafter(row['cv'], math.inf)
        with self.assertRaisesRegex(ValueError, '已核两项之外的CV差异'):
            replay.compare_r16_scores(self.original, changed)

    def test_known_cv_differences_do_not_allow_a_time_change(self):
        changed = self.with_known_differences()
        changed['conditions'][0]['ns'] += 1
        with self.assertRaisesRegex(ValueError, '原评分不符'):
            replay.compare_r16_scores(self.original, changed)


if __name__ == '__main__':
    unittest.main()
