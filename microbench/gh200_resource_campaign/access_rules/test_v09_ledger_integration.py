"""Reject clock-work basis changes and preserve the frozen r2 work definition."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import v09_model as model


class LedgerIntegrationTests(unittest.TestCase):
    def test_unknown_version_or_effective_byte_basis_is_rejected(self):
        row = dict(config='cfg_b', m=128, n=128, k=128, swizzle=1)
        setup = dict(grid=[1, 1, 1], stages=6)
        for schema, field in [('gh200-supply-work-ledger/2', 'totals.source_payload_KiB'),
                              ('gh200-supply-work-ledger/1', 'totals.source_valid_KiB')]:
            with self.subTest(schema=schema, field=field):
                with patch.object(model.supply, 'supply_work_ledger', return_value=dict(
                        schema=schema, legacy_clock_source_field=field)):
                    with self.assertRaisesRegex(ValueError, 'nominal source-payload'):
                        model.predict_components(row, setup, {})

    def test_frozen_clock_inputs_are_preserved(self):
        root = Path(__file__).resolve().parents[3]/'results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011'
        if not root.exists():
            self.skipTest('local frozen V09 archive required')
        frozen = json.loads((root/'frozen/predictions.json').read_text())['predictions']
        calibration = json.loads((root/'frozen/calibration.json').read_text())
        setups = {r['case']: r['setup'] for r in json.loads((root/'static_setup.json').read_text())}
        count = 0
        cfg_b_count = 0
        for row in json.loads((root/'cases.json').read_text()):
            if row['set'] != 'heldout' or frozen[row['id']]['status'] != 'predicted':
                continue
            with self.subTest(case=row['id']):
                prediction = model.predict_components(row, setups[row['id']], calibration)
                self.assertEqual(prediction['static_clock_work'], frozen[row['id']]['static_clock_work'])
                if row['config'] == 'cfg_b':
                    expected = {k: v for k, v in frozen[row['id']].items() if k != 'status'}
                    self.assertEqual(json.dumps(prediction, sort_keys=True), json.dumps(expected, sort_keys=True))
                    cfg_b_count += 1
            count += 1
        self.assertEqual(count, 29)
        self.assertEqual(cfg_b_count, 12)


if __name__ == '__main__':
    unittest.main()
