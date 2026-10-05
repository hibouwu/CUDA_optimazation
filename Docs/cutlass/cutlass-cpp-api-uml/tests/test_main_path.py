"""Scheme-review scope, source-locator and negative-case checks, not GPU proof."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_main_path as builder


class MainPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas = json.loads((ROOT / 'data/atlas.json').read_text())
        cls.overviews = json.loads((ROOT / 'data/overviews.json').read_text())
        cls.spec = json.loads((ROOT / 'data/main-path-spec.json').read_text())
        cls.published = json.loads((ROOT / 'data/main-path.json').read_text())

    def checked(self, spec=None, atlas=None):
        return builder.validate(ROOT, copy.deepcopy(spec or self.spec), copy.deepcopy(atlas or self.atlas), self.overviews)

    def test_all_five_steps_have_objects_conditions_and_inspection_questions(self):
        self.checked()
        self.assertEqual([s['id'] for s in self.spec['steps']], builder.STEP_IDS)
        for step in self.spec['steps']:
            self.assertTrue(step['input'] and step['responsibility'] and step['handoff'])
            self.assertTrue(step['conditions'] and step['boundary'])
            for q in step['questions']:
                self.assertTrue(q['question'] and q['evidence'])

    def test_same_fixture_and_original_source_graph(self):
        self.assertEqual(self.published['base_atlas_sha256'], builder.sha(ROOT / 'data/atlas.json'))
        self.assertEqual(self.published['fixture']['problem_mnkl'], [256, 256, 128, 1])
        for key in ('runtime_verified', 'actual_browser_verified', 'independent_use_review_passed', 'global_complete'):
            self.assertFalse(self.published[key])

    def test_sources_really_contain_selected_statements_and_offline_anchors(self):
        checked = self.checked()
        for e in checked['evidence']:
            self.assertTrue(e['source_sha256'])
            self.assertTrue(e['source_url'])

    def test_cannot_drop_failure_and_reuse_step(self):
        bad = copy.deepcopy(self.spec); bad['steps'].pop()
        with self.assertRaisesRegex(ValueError, 'five-step'):
            self.checked(bad)

    def test_cannot_switch_problem_midway_by_changing_fixture(self):
        bad = copy.deepcopy(self.spec); bad['fixture']['problem_mnkl'][0] = 512
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.checked(bad)

    def test_changed_compilation_recipe_is_not_silently_reused(self):
        atlas = copy.deepcopy(self.atlas)
        next(m for m in atlas['modules'] if m['module_id'] == self.spec['module_id'])['configuration']['binary_target'] = 'sm_90a'
        with self.assertRaisesRegex(ValueError, 'configuration drifted'):
            self.checked(atlas=atlas)

    def test_unknown_api_or_condition_link_cannot_be_added(self):
        bad = copy.deepcopy(self.spec); bad['steps'][0]['edge_refs'].append('fabricated.initialize.check')
        with self.assertRaisesRegex(ValueError, 'Unrecorded'):
            self.checked(bad)

    def test_mislocated_evidence_is_rejected(self):
        bad = copy.deepcopy(self.spec)
        e = next(e for e in bad['evidence'] if e['id'] == 'mma.wait')
        e['start_line'], e['end_line'] = 1, 3
        with self.assertRaisesRegex(ValueError, 'Evidence range'):
            self.checked(bad)

    def test_wrong_source_contents_cannot_be_called_the_saved_fixture(self):
        bad = copy.deepcopy(self.spec); bad['fixture']['example_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'saved source'):
            self.checked(bad)

    def test_core_distinctions_have_exact_original_relations(self):
        edges = {e['id']: e for e in self.atlas['edges']}
        expected = {
            'types.edge.selected_specialization': ('types.builder_bound', 'template_binds', 'types.builder_sm100'),
            'host.cuda.submit_device_kernel.294': ('host.cuda.launch_ex', 'launches', 'host.device_kernel'),
            'host.device.call_functor.123': ('host.device_kernel', 'calls', 'host.kernel.operator'),
            'host.app.sync.250': ('host.app.main', 'calls', 'host.cuda.stream_sync'),
        }
        used = {r for step in self.spec['steps'] for r in step['edge_refs']}
        for identifier, relation in expected.items():
            self.assertIn(identifier, used)
            self.assertEqual(tuple(edges[identifier][k] for k in ('source', 'relation', 'target')), relation)
        self.assertFalse(any(e['source'] == 'host.initialize' and e['target'] == 'host.can_implement' for e in self.atlas['edges']))
        self.assertFalse(any(e['source'] == 'host.update' and e['target'] == 'host.kernel.initialize_workspace' for e in self.atlas['edges']))

    def test_all_publication_inputs_still_match(self):
        for path, expected in self.published['input_sha256'].items():
            self.assertEqual(builder.sha(ROOT / path), expected, path)

    def test_other_scenarios_are_questions_not_claimed_implementations(self):
        self.assertEqual({s['title'] for s in self.spec['scenario_branches']}, {'Grouped GEMM', 'MoE', 'Block-Scaled', 'Attention'})
        self.assertIn('不是这些场景已经完成', self.spec['scenario_notice'])

    def test_three_review_cases_retain_conditional_judgments(self):
        self.checked()
        self.assertEqual({c['id'] for c in self.spec['review_cases']}, {'layout-contract', 'completion-contract', 'reuse-contract'})
        for case in self.spec['review_cases']:
            self.assertEqual(len(case['decisions']), 3)
            self.assertTrue(case['boundary'])
        self.assertIn('不是同事已经给出的实现', self.spec['review_case_notice'])

    def test_review_case_cannot_drop_scope_or_correctness_distinction(self):
        bad = copy.deepcopy(self.spec); bad['review_cases'][0]['decisions'] = bad['review_cases'][0]['decisions'][:1]
        with self.assertRaisesRegex(ValueError, 'distinguish scope'):
            self.checked(bad)

    def test_review_case_must_have_source_not_only_a_plausible_story(self):
        bad = copy.deepcopy(self.spec); bad['review_cases'][1]['evidence_ids'].append('unverified.runtime.event')
        with self.assertRaisesRegex(ValueError, 'unrecorded source'):
            self.checked(bad)

    def test_host_layout_result_matches_program_output_and_fixed_sources(self):
        probe = builder.load_host_probe(ROOT, self.atlas)
        self.assertEqual(probe['coordinates_checked'], 131072)
        self.assertEqual([x['a_offset_1_0'] for x in probe['cases']], [128, 136])
        self.assertEqual([x['b_offset_1_1'] for x in probe['cases']], [257, 265])
        self.assertGreater(probe['ignored_padding_mismatches'], 0)
        self.assertFalse(probe['gemm_runtime_verified'])

    def test_probe_is_host_only_even_though_source_suffix_is_cuda(self):
        source = (ROOT / 'verification/scheme-layout/layout_probe.cu').read_text()
        self.assertNotIn('<<<', source); self.assertNotIn('cudaLaunch', source)
        self.assertIn('TagToStrideA_t', source); self.assertIn('TagToStrideB_t', source)
        self.assertIn('mathematical_a(cutlass::MatrixCoord(m, k))', source)
        self.assertIn('mathematical_b(cutlass::MatrixCoord(k, n))', source)
        self.assertEqual(self.published['host_layout_probe']['scope'], 'host_layout_coordinate_mapping_only')


if __name__ == '__main__':
    unittest.main()
