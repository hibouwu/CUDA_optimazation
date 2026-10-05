"""Regression checks for the reviewer's independent validator (read-only)."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('elementwise_source_check', HERE / 'check_source.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class SourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = json.loads((HERE / 'relations.json').read_text())

    def verify(self, data):
        return checker.verify(data, ledger=False, atlas=False)

    def test_frozen_pass(self):
        self.assertEqual(self.verify(self.frozen)['status'], 'PASS')

    def test_dropped_call_obligation_fails(self):
        data = copy.deepcopy(self.frozen)
        items = data['coverage']['obligations']
        items.remove(next(o for o in items if o['category'] == 'call_expression'))
        with self.assertRaises(AssertionError): self.verify(data)

    def test_merged_same_line_x_call_fails(self):
        data = copy.deepcopy(self.frozen)
        pairs = [e for e in data['edges'] if e['relation'] == 'calls' and e['source_expression'] == 'x(i)']
        pairs[1]['callsite'] = copy.deepcopy(pairs[0]['callsite'])
        with self.assertRaises(AssertionError): self.verify(data)

    def test_decltype_cannot_be_runtime_call(self):
        data = copy.deepcopy(self.frozen)
        edge = next(e for e in data['edges'] if e.get('evaluation') == 'unevaluated_decltype')
        edge['relation'] = 'calls'; edge['callsite'] = edge['expression_range']
        with self.assertRaises((AssertionError, KeyError)): self.verify(data)

    def test_missing_default_argument_fails(self):
        data = copy.deepcopy(self.frozen)
        items = data['coverage']['obligations']
        items.remove(next(o for o in items if o['category'] == 'default_argument'))
        with self.assertRaises(AssertionError): self.verify(data)

    def test_byte_drift_fails(self):
        data = copy.deepcopy(self.frozen)
        edge = next(e for e in data['edges'] if e['relation'] == 'calls')
        edge['callsite']['start_byte'] += 1
        with self.assertRaises(AssertionError): self.verify(data)

    def test_missing_operator_obligation_fails(self):
        data = copy.deepcopy(self.frozen)
        items = data['coverage']['obligations']
        items.remove(next(o for o in items if o['category'] == 'operator_expression'))
        with self.assertRaises(AssertionError): self.verify(data)

    def test_fixed_initializer_is_not_symbolic(self):
        data = copy.deepcopy(self.frozen)
        node = next(n for n in data['nodes'] if n.get('semantic_class') == 'fixed_type_value_initialization')
        node['resolution'] = 'symbolic'
        with self.assertRaises(AssertionError): self.verify(data)

    def test_control_container_is_not_pending_api(self):
        data = copy.deepcopy(self.frozen)
        node = next(n for n in data['nodes'] if n.get('semantic_class') == 'source_control_container')
        node['resolution'] = 'symbolic'
        with self.assertRaises(AssertionError): self.verify(data)

    def test_missing_unreachable_expansion_fails(self):
        data = copy.deepcopy(self.frozen)
        macro = next(o for o in data['coverage']['obligations'] if o.get('macro_kind') == 'conditional_builtin_or_empty_expansion')
        macro['expansion_variants'].pop()
        with self.assertRaises(AssertionError): self.verify(data)

    def test_include_target_drift_fails(self):
        data = copy.deepcopy(self.frozen)
        include = next(o for o in data['coverage']['obligations'] if o['category'] == 'include')
        include['target_path'] = 'include/cute/algorithm/gemm.hpp'
        with self.assertRaises(AssertionError): self.verify(data)


if __name__ == '__main__':
    unittest.main()
