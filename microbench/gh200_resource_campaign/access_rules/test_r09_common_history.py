"""Fixed common history changes the intervention, not the convergence filter."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import analyze_r18
import run_r18
import run_v08


class CommonHistory(unittest.TestCase):
    def fixed(self, swizzle=8):
        row = dict(swizzle=swizzle,warmup_policy='common-sw1-16')
        setup = dict(warmup_policy='common-sw1-16',warmup_swizzle=1,warmup_fixed_calls=16)
        call = dict(setup,target_swizzle=swizzle,warmup_calls=16,warmup_us=[100.,200.]*8)
        return row,setup,call

    def test_fixed_prefix_keeps_unsettled_sample(self):
        row,setup,call = self.fixed()
        self.assertGreater(analyze_r18.validate_warmup(setup,call,row),.02)
        adaptive = dict(warmup_us=call['warmup_us'])
        with self.assertRaisesRegex(ValueError,'converge'):
            analyze_r18.validate_warmup({},adaptive,dict(swizzle=8))

    def test_prefix_count_and_identity_are_not_optional(self):
        row,setup,call = self.fixed()
        for bad in (dict(call,warmup_us=call['warmup_us'][:-1],warmup_calls=15),
                    dict(call,warmup_swizzle=8),dict(call,target_swizzle=1),
                    dict(call,warmup_policy='adaptive')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                analyze_r18.validate_warmup(setup,bad,row)

    def test_legacy_adaptive_metadata_still_replays(self):
        self.assertEqual(analyze_r18.validate_warmup({},dict(warmup_us=[100.]*8),dict(swizzle=1)),0)
        with self.assertRaisesRegex(ValueError,'converge'):
            analyze_r18.validate_warmup({},dict(warmup_us=[100.]*7),dict(swizzle=1))

    def test_four_ends_cases_and_balanced_pair_order(self):
        rows=json.loads((Path(__file__).parent/'configs/r09-common-history.json').read_text())
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['variants']==['ends'] for r in rows))
        for policy in ('adaptive','common-sw1-16'):
            pair=[r for r in rows if r['warmup_policy']==policy]
            first=[run_r18.trial_rows(pair,t)[0]['swizzle'] for t in range(10)]
            self.assertEqual(first.count(1),5)
            self.assertEqual(first.count(8),5)
        for row in rows:
            self.assertIn('--warmup-policy',run_r18.case_args(row))
            self.assertIn('--warmup-policy',run_v08.case_args(row))

    def test_fixed_history_does_not_retry_warmup_failure(self):
        for runner in (run_r18,run_v08):
            with self.subTest(runner=runner.__name__),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                folder=root/'samples'/'fixed'
                folder.mkdir(parents=True)
                (folder/'ends-00.json').write_text(json.dumps(dict(returncode=1,
                    stderr='last five warmups CV exceeds 2% after 30 calls')))
                with self.assertRaisesRegex(ValueError,'invalid prior process'):
                    runner.run_one(root,dict(id='fixed',warmup_policy='common-sw1-16'),'ends',0)
                self.assertEqual(len(list(folder.iterdir())),1)

    def test_public_reader_needs_no_site_packages(self):
        script = ('import sys; sys.path.insert(0,'+repr(str(Path(__file__).parent))+'); '
                  'import analyze_r18,run_r18,run_v08; '
                  "assert 'scipy' not in sys.modules; assert 'numpy' not in sys.modules")
        subprocess.run([sys.executable,'-I','-S','-c',script],check=True)


if __name__=='__main__':unittest.main()
