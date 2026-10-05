"""Independent array stage mutations. Captured dumps never touch artifacts.

The ledger stream is a labelled in-memory fixture from the three actual fixed
headers, not a read or rewrite of the global declarations.json. Each stage run
retains the draft's own source checker and the real canonical enrichment.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_atlas as atlas
from extract_declarations import Extractor
import stage_cute_array as stage

PATH = 'include/cute/container/array.hpp'
DRAFT = ROOT / 'data/module-drafts/cute_array/relations.json'


class ArrayStageIndependentReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = []
        for path in ('include/cute/config.hpp', PATH, 'include/cute/util/type_traits.hpp'):
            extractor = Extractor('8f50b052e1099fb982392a622caab69b97b63128')
            extractor.extract(path, (ROOT / 'snapshot' / path).read_bytes())
            cls.records.extend(extractor.occurrences)
        cls.authored = json.loads(DRAFT.read_text())
        # A broken unmodified stage must never make every rejection test look
        # successful for an unrelated early failure.
        cls.baseline = cls('runTest').run_stage()

    def occurrence(self, records, line, qualified=None):
        matches = [o for o in records if o['path'] == PATH
                   and o['signature_range']['start_line'] == line
                   and (qualified is None or o['qualified_name'] == qualified)]
        self.assertEqual(len(matches), 1, (line, qualified, len(matches)))
        return matches[0]

    def run_stage(self, ledger_mutation=None, draft_mutation=None, enriched_mutation=None):
        records = copy.deepcopy(self.records)
        draft = copy.deepcopy(self.authored)
        if ledger_mutation:
            ledger_mutation(records)
        if draft_mutation:
            draft_mutation(draft)
        captures = []
        original_read_text = Path.read_text
        original_enrich = stage.enrich_nodes

        def read_text(path, *args, **kwargs):
            if path == DRAFT:
                return json.dumps(draft)
            return original_read_text(path, *args, **kwargs)

        def fingerprint(path):
            if Path(path) == ROOT / 'data/declarations.json':
                return 'read-only-array-memory-ledger-fixture'
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        def enrich(nodes, scope, issues):
            output = original_enrich(nodes, scope, issues)
            if enriched_mutation:
                enriched_mutation(nodes)
            return output

        with patch.object(atlas, 'top_items', lambda *a, **kw: (('occurrences', o, True) for o in records)), \
             patch.object(stage, 'file_sha', fingerprint), patch.object(Path, 'read_text', read_text), \
             patch.object(stage, 'enrich_nodes', enrich), \
             patch.object(stage, 'dump', lambda p, d: captures.append((str(p), copy.deepcopy(d)))), \
             contextlib.redirect_stdout(io.StringIO()):
            stage.main()
        self.assertEqual(len(captures), 2)
        return captures[0][1]

    def rejected(self, **mutations):
        with self.assertRaises((ValueError, AssertionError)):
            self.run_stage(**mutations)

    def test_positive_actual_source_preserves_overloads_and_availability(self):
        data = self.baseline
        joined = data['canonical_integration']
        self.assertEqual(joined['physical_declarations'], 87)
        self.assertEqual(joined['bound_declarations'], 92)
        self.assertEqual(joined['current_manual_overrides'], 0)
        self.assertFalse(joined['actual_browser_acceptance'])
        self.assertFalse(joined['global_completion_claimed'])
        nodes = {n['id']: n for n in data['nodes']}
        std = nodes['array.type.std.tuple_size']
        cuda = nodes['array.type.cuda_std.tuple_size']
        self.assertNotEqual(std['entity_id'], cuda['entity_id'])
        self.assertNotEqual(nodes['array.api.fn_56']['entity_id'], nodes['array.api.fn_62']['entity_id'])
        self.assertEqual({o['start_line'] for o in std['source_declaration_occurrences']}, {433, 464})
        self.assertEqual(len({o['declaration_occurrence_id'] for o in std['source_declaration_occurrences']}), 2)
        self.assertNotIn('conditions', std)
        self.assertEqual(std['declaration_availability']['operator'], 'any_of')
        self.assertEqual(len(std['declaration_availability']['occurrence_condition_groups']), 2)
        self.assertNotEqual(*std['declaration_availability']['occurrence_condition_groups'])

    def test_wrong_qname_or_template_specialization_rejected(self):
        for qualified in ('wrong::array::operator[]', 'cute::array<T,0>::operator[]'):
            with self.subTest(qualified=qualified):
                self.rejected(ledger_mutation=lambda rs: self.occurrence(rs, 55).update(qualified_name=qualified))

    def test_const_overload_cannot_take_nonconst_source_signature(self):
        def mutate(rs):
            a, b = self.occurrence(rs, 55), self.occurrence(rs, 61)
            b.update(signature_range=copy.deepcopy(a['signature_range']), raw_signature=a['raw_signature'])
        self.rejected(ledger_mutation=mutate)

    def test_canonical_parameters_and_template_parameters_cannot_be_changed(self):
        mutations = [
            lambda o: o['parameters'][0].update(type='long'),
            lambda o: o['parameters'][0].update(name='different_pos'),
            lambda o: o['parameters'][0].update(index=17),
            lambda o: o['parameters'][0].update(default='1'),
            lambda o: o['parameters'][0].update(raw='long pos'),
            lambda o: o['parameters'][0].update(start_byte=o['parameters'][0]['start_byte'] + 1),
            lambda o: o['template_parameters'][0]['parameters'].pop(),
        ]
        for i, mutation in enumerate(mutations):
            with self.subTest(mutation=i):
                self.rejected(ledger_mutation=lambda rs: mutation(self.occurrence(rs, 55)))

    def test_canonical_return_qualifiers_attributes_and_body_range_cannot_change(self):
        mutations = [
            lambda o: o.update(return_type='const_reference'),
            lambda o: o.update(qualifiers=['constexpr', 'const']),
            lambda o: o.update(attributes=[]),
            lambda o: o['body_range'].update(end_byte=o['body_range']['end_byte'] - 1),
        ]
        for i, mutation in enumerate(mutations):
            with self.subTest(mutation=i):
                self.rejected(ledger_mutation=lambda rs: mutation(self.occurrence(rs, 55)))

    def test_canonical_branch_expression_origin_or_completeness_cannot_change(self):
        q = 'std::tuple_size<cute::array<T,N>>'
        mutations = [
            lambda o: o.update(preprocessor_conditions=[]),
            lambda o: o['preprocessor_conditions'][0].update(expression='defined(__CUDACC_RTC__)'),
            lambda o: o['preprocessor_conditions'][0].update(directive_path=PATH),
            lambda o: o['preprocessor_conditions'][0].update(directive_line=106),
        ]
        for i, mutation in enumerate(mutations):
            with self.subTest(mutation=i):
                self.rejected(ledger_mutation=lambda rs: mutation(self.occurrence(rs, 433, q)))

    def test_missing_duplicate_or_replaced_source_occurrence_rejected(self):
        q = 'std::tuple_size<cute::array<T,N>>'
        def missing(rs):
            rs.remove(self.occurrence(rs, 464, q))
        def duplicated(rs):
            rs.append(copy.deepcopy(self.occurrence(rs, 433, q)))
        def substituted(rs):
            rs.remove(self.occurrence(rs, 464, q))
            rs.append(copy.deepcopy(self.occurrence(rs, 433, q)))
        for mutation in (missing, duplicated, substituted):
            with self.subTest(mutation=mutation.__name__):
                self.rejected(ledger_mutation=mutation)

    def test_same_entity_two_positions_cannot_be_split_or_cross_namespace_merged(self):
        std = 'std::tuple_size<cute::array<T,N>>'
        cuda = 'cuda::std::tuple_size<cute::array<T,N>>'
        self.rejected(ledger_mutation=lambda rs: self.occurrence(rs, 464, std).update(entity_id='wrong_second_entity'))
        def merge(rs):
            self.occurrence(rs, 433, cuda)['entity_id'] = self.occurrence(rs, 433, std)['entity_id']
        self.rejected(ledger_mutation=merge)

    def test_two_operator_overloads_cannot_share_one_canonical_entity(self):
        def merge(rs):
            self.occurrence(rs, 61)['entity_id'] = self.occurrence(rs, 55)['entity_id']
        self.rejected(ledger_mutation=merge)

    def test_first_occurrence_conditions_cannot_replace_entity_or_availability(self):
        target = 'array.type.std.tuple_size'
        def misleading_field(nodes):
            nodes[target]['conditions'] = copy.deepcopy(nodes[target]['selected_declaration_conditions'])
        def single_branch(nodes):
            nodes[target]['declaration_availability']['occurrence_condition_groups'].pop()
        def intersection(nodes):
            nodes[target]['declaration_availability']['operator'] = 'all_of'
        for mutation in (misleading_field, single_branch, intersection):
            with self.subTest(mutation=mutation.__name__):
                self.rejected(enriched_mutation=mutation)

    def test_output_occurrence_and_authored_body_cannot_be_removed_or_changed(self):
        def missing_output(nodes):
            nodes['array.type.std.tuple_size']['source_declaration_occurrences'].pop()
        def wrong_body(draft):
            node = next(n for n in draft['nodes'] if n['id'] == 'array.api.fn_56')
            node['body_range']['raw'] = '{ return begin()[0]; }'
        self.rejected(enriched_mutation=missing_output)
        self.rejected(draft_mutation=wrong_body)

    def test_implicit_begin_and_end_cannot_swap_api_targets(self):
        def swapped(draft):
            first, second = [e for e in draft['edges'] if e['relation'] == 'implicit_calls']
            first['target'], second['target'] = second['target'], first['target']
        self.rejected(draft_mutation=swapped)

    def test_array_field_preserves_base_declarator_and_complete_type_separately(self):
        field = next(n for n in self.baseline['nodes'] if n['id'] == 'array.member.decl_194')
        self.assertEqual(field['declared_type'], 'element_type')
        self.assertEqual(field['declarator'], '__elems_[N]')
        self.assertEqual(atlas.LEXER.lex(field['declared_type_spelling'].encode())[0][-2].text, 'N')
        self.assertEqual(stage.token_spelling(field['declared_type_spelling']),
                         stage.token_spelling('element_type[N]'))
        self.assertIn('declaration_type_specifier', field['declared_type_role'])
        self.assertEqual(field['signature'], 'element_type __elems_[N];')

    def test_array_field_split_type_metadata_cannot_disagree_with_source(self):
        for field, value in [('declared_type', 'float'),
                             ('declarator', '__elems_[N+1]'),
                             ('declared_type_spelling', 'element_type[N+1]'),
                             ('declared_type_role', 'complete_type_only')]:
            with self.subTest(field=field):
                self.rejected(ledger_mutation=lambda rs: self.occurrence(rs, 194).update({field: value}))

    def test_return_cv_order_equivalence_does_not_erase_pointer_layers_or_metadata(self):
        data = self.run_stage(ledger_mutation=lambda rs: self.occurrence(rs, 99).update(return_type='T const*'))
        function = next(n for n in data['nodes'] if n['id'] == 'array.api.fn_100')
        self.assertEqual(function['return_cv_qualifiers'], ['const'])
        self.assertEqual(sorted(function['qualifiers']), ['const', 'constexpr'])
        for value in ('T* const', 'T*'):
            with self.subTest(return_type=value):
                self.rejected(ledger_mutation=lambda rs: self.occurrence(rs, 99).update(return_type=value))
        self.rejected(ledger_mutation=lambda rs: self.occurrence(rs, 99).update(return_cv_qualifiers=[]))

    def test_tuple_bridge_local_and_external_primary_paths_are_mutually_exclusive(self):
        nodes = {n['id']: n for n in self.authored['nodes']}
        source = (ROOT / 'snapshot' / PATH).read_bytes()
        expected_local = 'defined(CUTE_STL_NAMESPACE_IS_CUDA_STD) && !(__CUDACC_VER_MAJOR__ >= 13) && defined(__CUDACC_RTC__)'
        expected_external = 'defined(CUTE_STL_NAMESPACE_IS_CUDA_STD) && ((__CUDACC_VER_MAJOR__ >= 13) || !defined(__CUDACC_RTC__))'
        for line, target, primary_name in ((465, 'array.type.decl_457', 'std::tuple_size'),
                                           (470, 'array.type.decl_460', 'std::tuple_element')):
            edges = [e for e in self.authored['edges'] if e['relation'] == 'specializes'
                     and e['evidence'][0]['start_line'] == line]
            self.assertEqual(len(edges), 2)
            local = next(e for e in edges if e['target'] == target)
            external = next(e for e in edges if nodes[e['target']]['kind'] == 'external')
            self.assertEqual(local['condition'], expected_local)
            self.assertEqual(external['condition'], expected_external)
            self.assertEqual(nodes[target]['qualified_name'], primary_name)
            self.assertFalse(nodes[target]['definition'])
            for edge in edges:
                span = edge['evidence'][0]
                self.assertEqual(source[span['start_byte']:span['end_byte']].decode(), edge['source_expression'])
            for active in (False, True):
                for compiler_13 in (False, True):
                    for rtc in (False, True):
                        local_active = active and not compiler_13 and rtc
                        external_active = active and (compiler_13 or not rtc)
                        self.assertFalse(local_active and external_active)
                        self.assertEqual(local_active or external_active, active)


if __name__ == '__main__':
    unittest.main()
