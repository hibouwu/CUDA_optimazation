"""六形式的独立数学 oracle、稳定反证及只读证据→冻结→评分回归。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import l2_fit as fit
import l2_saturation as scan
import numpy as np
from scipy.optimize import nnls


def analytic_time(S, J, family='sum', jsat=4):
    local = 276824064/(S*50*min(J, jsat))
    return 400+local if family == 'sum' else 250+max(local, 276824064/2200)


def development(family='sum'):
    return [dict(case_id=case['id'], S=case['active_ctas'], J=case['chains'],
        mean_ns=analytic_time(case['active_ctas'], case['chains'], family),
        min_ns=.999*analytic_time(case['active_ctas'], case['chains'], family),
        max_ns=1.001*analytic_time(case['active_ctas'], case['chains'], family))
        for case in scan.make_cases() if case['split'] == 'dev' and case['mode'] == 'read']


def target_review(root):
    build = scan.read_json(root/'build/identity.json')
    return dict(schema='gh200-l2-target-review/1', compiler_version='release 12.9',
        source_sha256=build['source_sha256'], sass_sha256=build['sass_sha256'],
        functions=[dict(symbol=f'm{mode}j{chains}', mode=mode, chains=chains,
            ldg128_count=chains if mode in (0, 2) else 0,
            stg128_count=chains if mode in (1, 2) else 0, local_count=0)
            for mode in range(3) for chains in (1, 4, 8)])


def saved_evidence(root, split, frozen=None):
    """CPU-only raw envelopes/checksums, plus the original runner's exact validated rows."""
    results = []
    for case in scan.make_cases():
        if case['split'] != split:
            continue
        for index, seed in enumerate(scan.SEEDS):
            path = f'{split}/{case["id"]}/process_{index:02}'
            if case['mode'] != 'read':
                # Non-read records are bound by the development summary, never read-fit inputs.
                results.append(dict(case_id=case['id'], seed=seed, path=path))
                continue
            elapsed = round(analytic_time(case['active_ctas'], case['chains'])*
                            (1.2 if split == 'heldout' else 1.))
            active = case['active_ctas']
            row = dict(schema='gh200-l2-saturation/1', status='measured',
                cache_residency_proven=False, physical_hbm_bytes_proven=False,
                **{key: case[key] for key in ('active_ctas', 'chains', 'mode', 'array_bytes',
                                               'iterations', 'samples')}, seed=seed,
                read_payload_bytes=276824064, write_payload_bytes=0, work_count=276824064,
                aggregate_array_bytes=17301504,
                device=dict(name='GH200', uuid='GPU-fixture', sms=132, l2_cache_bytes=62914560),
                resources=dict(threads=128, grid_ctas=active, occupancy_ctas_per_sm=1,
                    local_size_bytes=0, registers_per_thread=32, dynamic_smem_bytes=132096,
                    static_smem_bytes=0),
                short_check=dict(iterations=[1, 2], errors=0, read_checked=active*256, write_checked=0),
                warmup=dict(elapsed_ns=[elapsed]*8, cv=0., converged=True, limit=60),
                trials=[dict(iterations=16, elapsed_ns=elapsed, event_ms=(elapsed+10000)/1e6,
                    all_active_windows_overlap=True,
                    blocks_detail=[dict(active=True, smid_start=2*sm+200, smid_end=2*sm+200,
                        begin_ns=1000, end_ns=1000+elapsed, begin_cycle=500,
                        end_cycle=500+2*elapsed) for sm in range(active)],
                    correctness=dict(read_checked=active*128, write_checked=0, errors=0),
                    read_checksums=[scan.read_checksum_oracle(17301504, active, 16, seed, thread)
                                    for thread in range(active*128)])])
            folder = root/path
            folder.mkdir(parents=True)
            scan.save_json(folder/'stdout', row)
            (folder/'stderr').write_text('')
            argv = [str(root/'build/probe'), '--active-ctas', str(active), '--chains', str(case['chains']),
                '--mode', 'read', '--array-bytes', '17301504', '--iterations', '16',
                '--seed', str(seed), '--samples', '1', '--warmup-limit', '60']
            scan.save_json(folder/'receipt.json', dict(argv=argv, returncode=0, timed_out=False,
                stdout_sha256=scan.digest(folder/'stdout'), stderr_sha256=scan.digest(folder/'stderr')))
            validated = scan.validate_probe(folder, row, case, seed)
            scan.save_json(folder/'validated.json', validated)
            results.append(dict(case_id=case['id'], seed=seed, path=path,
                validated_sha256=scan.digest(folder/'validated.json'), **validated))
    scan.save_json(root/split/'summary.json', dict(complete=True, split=split, results=results,
        plan_sha256=scan.digest(root/'plan.json'),
        build_identity_sha256=scan.digest(root/'build/identity.json'), device_uuid='GPU-fixture',
        all_performance_eligible=True, dev_freeze=frozen))


class L2FitTests(unittest.TestCase):
    def test_hsum_recovers_shared_floor_and_supply_and_fixed_tie_rule(self):
        fits, selected, reversals = fit.fit_all(development())
        self.assertEqual(reversals, [])
        self.assertEqual(len(fits), 6)
        self.assertEqual(selected['name'], 'sum_jsat4')
        self.assertAlmostEqual(selected['service']['floor_ns'], 400, places=7)
        self.assertAlmostEqual(selected['service']['b1_bytes_per_ns'], 50, places=10)
        self.assertLess(selected['MAPE'], 1e-12)
        self.assertIn('不分别识别a和B/Bg', selected['identifiability']['excluded_decomposition'])

    def test_hmax_recovers_both_branches_and_three_identifiable_parameters(self):
        fitted = fit.fit_form(development('max'), 'max', 4)
        self.assertLess(fitted['max_relative_error'], 1e-10)
        self.assertAlmostEqual(fitted['service']['a_ns'], 250, places=5)
        self.assertAlmostEqual(fitted['service']['b1_bytes_per_ns'], 50, places=8)
        self.assertAlmostEqual(fitted['service']['shared_bytes_per_ns'], 2200, places=7)
        self.assertTrue(fitted['identifiability']['separately_identified'])

    def test_stable_disjoint_reversals_refute_before_any_fit(self):
        for axis in ('S', 'J'):
            data = development()
            first = next(row for row in data if row['S'] == 1 and row['J'] == 1)
            second = next(row for row in data if row['S'] == (8 if axis == 'S' else 1)
                          and row['J'] == (1 if axis == 'S' else 4))
            first.update(min_ns=10., mean_ns=11., max_ns=12.)
            second.update(min_ns=20., mean_ns=21., max_ns=22.)
            with patch.object(fit, 'fit_form', side_effect=AssertionError('must not fit')):
                forms, selected, reversals = fit.fit_all(data)
            self.assertIsNone(selected)
            self.assertEqual(len(forms), 6)
            self.assertTrue(any(row['axis'] == axis for row in reversals))
            self.assertTrue(all(row['status'] == 'refuted_stable_monotonic_reversal' for row in forms))

    def test_all_local_and_all_plateau_keep_identifiability_bounds(self):
        local = fit.fit_form(development(), 'max', 4)
        self.assertFalse(local['identifiability']['separately_identified'])
        self.assertGreater(local['identifiability']['bounds']['shared_bytes_per_ns_min'], 0)
        flat = development()
        for row in flat:
            row.update(mean_ns=1234., min_ns=1233., max_ns=1235.)
        plateau = fit.fit_form(flat, 'max', 4)
        self.assertFalse(plateau['identifiability']['separately_identified'])
        self.assertEqual(plateau['identifiability']['bounds']['a_ns_interval'], [0., 1234.])
        self.assertLess(plateau['max_relative_error'], 1e-12)
        self.assertEqual(fit.fit_form(flat, 'sum', 4)['status'], 'unusable_zero_supply_slope')

    def test_negative_time_and_unregistered_form_are_rejected(self):
        data = development()
        data[0]['mean_ns'] = -1
        with self.assertRaises(ValueError):
            fit.fit_form(data, 'sum', 4)
        with self.assertRaises(ValueError):
            fit.fit_form(development(), 'max', 2)

    def test_zero_slope_hmax_optimum_keeps_the_exact_finite_plateau_representative(self):
        data = development()
        for row in data:
            row.update(mean_ns=1000+.001*row['S'], min_ns=900., max_ns=1200.)
        forms, selected, reversals = fit.fit_all(data)
        self.assertEqual(reversals, [])
        self.assertIsNotNone(selected)
        self.assertEqual(selected['family'], 'max')
        for result in forms:
            if result['family'] == 'sum':
                self.assertEqual(result['status'], 'unusable_zero_supply_slope')
                continue
            self.assertEqual(result['status'], 'fitted')
            service = result['service']
            self.assertGreater(service['b1_bytes_per_ns'], 0)
            self.assertGreater(service['shared_bytes_per_ns'], 0)
            self.assertEqual(service['a_ns'], 0)
            self.assertFalse(result['identifiability']['separately_identified'])
            # Recover the former winning u=0 constant, including its rounding. The
            # representative must preserve it, not choose the nearby analytic floor.
            x = np.array([1/(row['S']*min(row['J'], result['jsat'])) for row in data])
            y = np.array([row['mean_ns'] for row in data])
            scale = max(y)
            target = y/scale
            zero_slope, _ = nnls(np.column_stack((np.ones(len(x)), x/max(x)))/
                                target[:, None], np.ones(len(x)))
            self.assertEqual(zero_slope[1], 0)
            self.assertEqual(service['shared_bytes_per_ns'], fit.B/(zero_slope[0]*scale))

    def test_saved_raw_freeze_and_heldout_score_do_not_reselect(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent/'samples'
            scan.prepare(root)
            build = root/'build'
            (build/'disassemble').mkdir(parents=True)
            (build/'probe').write_bytes(b'CPU fixture, never executed')
            (build/'disassemble/stdout').write_text('bound synthetic SASS')
            scan.save_json(build/'identity.json', dict(plan_sha256=scan.digest(root/'plan.json'),
                probe_sha256=scan.digest(build/'probe'),
                source_sha256=scan.digest(root/'source/l2_saturation.cu'),
                sass_sha256=scan.digest(build/'disassemble/stdout')))
            scan.save_json(build/'target-review.json', target_review(root))
            saved_evidence(root, 'dev')
            output = parent/'freeze'
            result = fit.fit(root, output, build/'target-review.json')
            self.assertEqual(result['state'], 'frozen')
            binding = scan.verify_dev_freeze(root, output/'dev-freeze.json')
            selected = scan.read_json(output/'selected.service.json')
            predictions = scan.read_json(output/'predictions.json')
            self.assertEqual(len(predictions), 4)
            self.assertEqual({(row['S'], row['J']) for row in predictions},
                             {(16, 4), (16, 8), (66, 4), (66, 8)})
            self.assertEqual(selected['logical_bytes'], 276824064)
            saved_evidence(root, 'heldout', binding)
            before = scan.digest(output/'dev-freeze.json')
            with patch.object(fit, 'fit_all', side_effect=AssertionError('heldout must not fit')):
                scored = fit.score(root, output/'dev-freeze.json', parent/'scored')
            self.assertEqual(scored['read_processes'], 12)
            report = scan.read_json(parent/'scored/score.json')
            self.assertEqual(len(report['conditions']), 4)
            self.assertAlmostEqual(report['selected_condition_MAPE'], 1/6, places=4)
            self.assertEqual(scan.digest(output/'dev-freeze.json'), before)
            with self.assertRaises(FileExistsError):
                fit.score(root, output/'dev-freeze.json', parent/'scored')
            # A changed raw byte is rejected before fitting or scoring can reuse it.
            raw = root/'heldout/read_a16_j4/process_00/stdout'
            raw.write_text(raw.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'raw命令或返回身份'):
                fit.read_evidence(root, 'heldout')

    def test_target_sass_counts_and_hashes_are_checked(self):
        identity = dict(source_sha256={'l2_saturation.cu': 'source'}, sass_sha256='sass')
        review = dict(schema='gh200-l2-target-review/1', compiler_version='release 12.9',
            source_sha256='source', sass_sha256='sass',
            functions=[dict(mode=mode, chains=chains, ldg128_count=chains if mode != 1 else 0,
                stg128_count=chains if mode != 0 else 0, local_count=0)
                for mode in range(3) for chains in (1, 4, 8)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'review.json'
            scan.save_json(path, review)
            self.assertEqual(fit.check_sass_review(path, identity), review)
            for field, value in (('local_count', 1), ('ldg128_count', 99)):
                wrong = copy.deepcopy(review)
                wrong['functions'][0][field] = value
                path.write_text(json.dumps(wrong))
                with self.assertRaisesRegex(ValueError, '目标访存'):
                    fit.check_sass_review(path, identity)
            review['sass_sha256'] = 'different'
            path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, '审核身份'):
                fit.check_sass_review(path, identity)

    def test_only_warmup_cv_allows_actual_cross_cpu_ulp_replay_difference(self):
        for replayed, arm_saved in ((.005325796842416135, .005325796842416137),
            (.0008044687633181553, .0008044687633181555),
            (.00014155842005871087, .0001415584200587109),
            (.00018874142414039598, .00018874142414039593)):
            actual = dict(warmup_cv=replayed, elapsed_ns=1000, performance_eligible=True)
            saved = dict(actual, warmup_cv=arm_saved)
            self.assertNotEqual(actual, saved)
            self.assertTrue(fit.same_validation(actual, saved))
            self.assertFalse(fit.same_validation(actual, dict(saved, warmup_cv=arm_saved*1.00000001)))
            self.assertFalse(fit.same_validation(actual, dict(saved, elapsed_ns=1001)))
            self.assertFalse(fit.same_validation(actual, dict(saved, performance_eligible=False)))
        self.assertFalse(fit.same_validation(dict(warmup_cv=0.), dict(warmup_cv=1e-300)))


if __name__ == '__main__':
    unittest.main()
