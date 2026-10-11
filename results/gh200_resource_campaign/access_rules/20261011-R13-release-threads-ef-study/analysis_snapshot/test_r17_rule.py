"""Regression checks for measurement eligibility and first-tile composition."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import r17_rule as rule
import v06_model as model


class R17Checks(unittest.TestCase):
    def test_overperturbed_point_is_not_a_parameter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'samples').mkdir()
            rows = []
            for case, traced in [('good', 104.0), ('disturbed', 106.0)]:
                for variant, elapsed in [('plain', 100.0), ('stamped', traced)]:
                    for trial in range(10):
                        rows.append(dict(case=case, variant=variant, trial=trial,
                                         elapsed_us=elapsed, returncode=0, check='ok',
                                         padding_errors=0, last5_cv=0.0))
            (root / 'samples/r17.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with (root / 'classes.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=['case', 'config', 'round', 'm', 'cls', 'per_ktile'])
                writer.writeheader()
                for case, cycles in [('good', 600), ('disturbed', 900)]:
                    writer.writerow(dict(case=case, config='cfg_a', round=0, m=128,
                                         cls='oob', per_ktile=cycles))
            eligible, _ = rule.eligibility(root)
            self.assertFalse(eligible['disturbed']['qualified'])
            rates, points = rule.rule_rates(root, eligible)
            self.assertEqual(rates, {'oob': 600})
            self.assertEqual(points['oob'], [600])

    def test_extra_window_does_not_add_a_second_intercept(self):
        p = dict(P0=10, S=20, l0=470, l1=512, dL0=20, w=3,
                 E0=50, E=40, h=4, Etail=5, x0=0, x1=0, xk=0)
        extra = 6080
        changed = dict(p, dL0=p['dL0'] + extra)
        before = model.cta_cycles(p, 'cooperative', 2, 40)
        after = model.cta_cycles(changed, 'cooperative', 2, 40)
        self.assertEqual(after - before, extra)
        self.assertEqual(model.lmain(changed, 40, False), model.lmain(p, 40, False))

    def test_two_points_do_not_export_a_transfer_rule(self):
        pairs = [dict(kind='oob', m=1408, n=2304, kt=k, extra_cycles=5+3*k, case=str(k))
                 for k in [64, 160]]
        result = rule.fit_descriptions(pairs)[0]
        self.assertFalse(result['exportable'])
        self.assertEqual(result['status'], 'descriptive_fit_without_independent_check')
        self.assertEqual(result['a_cycles'], 5)
        self.assertEqual(result['b_cycles_per_ktile'], 3)


if __name__ == '__main__':
    unittest.main()
