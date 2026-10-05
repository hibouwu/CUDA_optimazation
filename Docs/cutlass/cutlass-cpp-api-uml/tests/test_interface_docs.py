"""Finite publication checks; no compiler/frontend or GPU execution claimed."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_interface_docs as docs


class ParameterPhaseTests(unittest.TestCase):
    def test_constexpr_does_not_turn_scalar_argument_into_template(self):
        node = {'signature': 'static constexpr int f(int x)', 'template_parameters': []}
        self.assertEqual(docs.callable_category(node, {'type': 'int'})[0], 'call_time')

    def test_shape_type_does_not_prove_all_extents_static(self):
        self.assertEqual(docs.callable_category({}, {'type': 'ProblemShape const&'})[0], 'mixed')

    def test_generic_value_can_carry_static_information_in_its_type(self):
        node = {'template_parameters': [{'parameters': [{'kind': 'type_parameter_declaration', 'name': 'Alpha'}]}]}
        self.assertEqual(docs.callable_category(node, {'type': 'Alpha const&'})[0], 'mixed')

    def test_tag_parameter_is_not_an_arbitrary_runtime_integer(self):
        self.assertEqual(docs.callable_category({}, {'type': 'prefer<1>'})[0], 'type_carried')

    def test_macro_parameter_is_preprocessor_text(self):
        node = {'id': 'macro', 'kind': 'api', 'entity_kind': 'macro_definition',
                'signature': '#define ALIGN(n) alignas(n)', 'parameters': []}
        out = docs.parameter_docs(ROOT, node, {'parameter_notes': {'n': '对齐文本'}})
        self.assertEqual(out['macros'][0]['phase'], 'preprocessor')
        self.assertFalse(out['arguments'])

    def test_missing_explanation_cannot_be_published_as_complete(self):
        node = {'id': 'f', 'kind': 'api', 'parameters': [{'name': 'value', 'raw': 'int value', 'type': 'int'}]}
        with self.assertRaisesRegex(ValueError, 'Missing function parameter'):
            docs.parameter_docs(ROOT, node, {})

    def test_enclosing_template_is_distinct_from_method_template(self):
        node = {'id': 'f', 'kind': 'api', 'line': 30, 'signature_range': {'start_line': 29},
                'template_parameters': [{'parameters': [{'name': 'T', 'raw': 'class T', 'start_line': 10, 'kind': 'type_parameter_declaration'}]},
                                        {'parameters': [{'name': 'U', 'raw': 'class U', 'start_line': 29, 'kind': 'type_parameter_declaration'}]}]}
        out = docs.parameter_docs(ROOT, node, {'template_notes': {'T': '类参数', 'U': '方法参数'}})
        self.assertEqual([p['ownership'] for p in out['templates']], ['所属类型或外层模板', '本接口模板'])

    def test_zero_unnamed_argument_has_independent_key(self):
        node = {'id': 'f', 'kind': 'api', 'parameters': [{'name': None, 'raw': 'int', 'type': 'int'}]}
        out = docs.parameter_docs(ROOT, node, {'parameter_notes': {'#0': '未命名选择输入'}})
        self.assertIsNone(out['arguments'][0]['name'])


class PublishedInterfaceDocsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas = json.loads((ROOT / 'data/atlas.json').read_text())
        cls.result = json.loads((ROOT / 'data/interface-docs.json').read_text())

    def test_scope_matches_every_published_api_type_external(self):
        expected = {n['id'] for n in self.atlas['nodes'] if n['kind'] in docs.KINDS}
        self.assertEqual(expected, set(self.result['nodes']))
        self.assertEqual(self.result['base_atlas_sha256'], docs.sha(ROOT / 'data/atlas.json'))
        self.assertFalse(self.result['global_complete'])

    def test_all_descriptions_and_parameters_nonempty(self):
        for entry in self.result['nodes'].values():
            for field in ('layer', 'summary', 'execution'):
                self.assertTrue(entry[field], (entry['id'], field))
            for group in entry['parameters'].values():
                for parameter in group:
                    self.assertTrue(parameter['description'])
                    self.assertTrue(parameter['phase_label'])

    def test_device_kernel_is_one_real_argument_not_two_parser_fragments(self):
        entry = self.result['nodes']['host.device_kernel']
        self.assertEqual([p['name'] for p in entry['parameters']['arguments']], ['params'])
        self.assertTrue(entry['parameter_correction'])

    def test_defaults_and_return_mutability_are_not_dropped(self):
        entry = self.result['nodes']['host.initialize']
        params = {p['name']: p for p in entry['parameters']['arguments']}
        self.assertEqual(params['workspace']['default'], 'nullptr')
        self.assertEqual(params['stream']['default'], 'nullptr')
        self.assertEqual(params['args']['type'], 'Arguments const&')


if __name__ == '__main__':
    unittest.main()
