"""Independent alignment-proof fixtures plus fixed-source integration checks."""
import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reconcile_candidates as recon

PATH = 'include/cute/alignment_reconcile_review.hpp'
PROVIDER = 'include/cute/container/alignment.hpp'


def location(path, raw, start, end):
    return {'path': path, 'start_byte': start, 'end_byte': end,
            'start_line': raw[:start].count(b'\n') + 1,
            'end_line': raw[:max(start, end - 1)].count(b'\n') + 1}


def fixture(extra=''):
    """Hand-constructed claims; no declaration/header extraction is called."""
    raw = ('#include "cute/container/alignment.hpp"\n' + extra +
           'template<int Alignment> struct Holder { CUTE_ALIGNAS(Alignment) int value; };\n').encode()
    a = raw.index(b'CUTE_ALIGNAS(Alignment)')
    b = a + len(b'CUTE_ALIGNAS(Alignment)')
    end = raw.index(b';', b) + 1
    name = raw.index(b'value', b)
    provider = (ROOT / 'snapshot' / PROVIDER).read_bytes()
    proof = recon.SourceProof(PROVIDER, provider)
    variants = []
    for d in proof.scan.definitions:
        if d['name'] != 'CUTE_ALIGNAS':
            continue
        x, y = d['body_byte_range']
        guards = []
        for g in proof.conditions_at(d['byte_range'][0]):
            directive = next(v for v in proof.directives if v['physical_range'] == g['span'])
            guards.append({'expression': g['expression'], 'directive': directive['directive'],
                           'constant_false': False,
                           'directive_span': location(PROVIDER, provider, *g['span'])})
        definition = {**location(PROVIDER, provider, d['byte_range'][0], d['byte_range'][1] - 1),
                      'name': 'CUTE_ALIGNAS', 'parameters': ['n'], 'body': provider[x:y].decode()}
        variants.append({'definition': definition, 'definition_source_sha256': recon.sha(provider),
                         'conditions': guards,
                         'expanded_spelling': definition['body'].replace('(n)', '(Alignment)')})
    attribute = {**location(PATH, raw, a, b), 'attribute_id': 'independent-attribute',
                 'kind': 'cute_requested_alignment', 'semantic_spelling': raw[a:b].decode(),
                 'alignment_expression': 'Alignment',
                 'expression_span': location(PATH, raw, a + len('CUTE_ALIGNAS('), b - 1),
                 'alignment_unit': 'bytes', 'value_evaluation': 'not_evaluated; compiler authoritative',
                 'conditions': [], 'conditional_expansions': variants,
                 'definition_include_chain': [{'path': PATH, 'start_line': 1, 'end_line': 1, 'target_path': PROVIDER}],
                 'owner_hint': {'kind': 'field', 'syntax_kind': 'field_declaration',
                                'declaration_span': location(PATH, raw, a, end)},
                 'owner_occurrence_id': 'occ-field', 'owner_entity_id': 'entity-field'}
    occurrence = {'id': 'occ-field', 'entity_id': 'entity-field', 'source_id': 'source-field',
                  'kind': 'member', 'name': 'value', 'qualified_name': 'Holder::value',
                  'range': [a, end], 'syntax': [a, end], 'signature': [a, end],
                  'declarator': [name, name + len('value')], 'body': None,
                  'raw_signature_sha256': recon.sha(raw[a:end]), 'parse_status': 'parsed',
                  'attributes': [], 'parameter_count': 0, 'conditions': [], 'condition_lines': [],
                  'scope_chain': [{'name': 'Holder', 'kind': 'struct_specifier', 'entity_id': 'holder', 'name_resolution': {}}],
                  'name_resolution': {}, 'qualified_name_resolution': 'source_name',
                  'alignment_specifiers': [attribute]}
    metadata = copy.deepcopy(attribute)
    metadata.pop('owner_occurrence_id'); metadata.pop('owner_entity_id')
    metadata.update(owner_occurrence_ids=['occ-field'], owner_entity_ids=['entity-field'])
    file_record = {'path': PATH, 'sha256': recon.sha(raw),
                   'syntax_analysis': {'header_analysis': {'alignment_attributes': [metadata]}}}
    return raw, occurrence, file_record, {PROVIDER: provider}


def mapped(raw, occurrences, file_record, sources):
    cs = recon.LEXER.scan_sources({PATH: raw})['candidates']
    candidate = next(c for c in cs if c['kind'] == 'macro_invocation' and c.get('origin') == 'source' and c['name'] == 'CUTE_ALIGNAS')
    r = recon.Reconciler(PATH, raw, cs, occurrences, file_record=file_record, source_reader=sources.__getitem__)
    return r.reconcile(candidate)


class AlignmentProofAdversarialTests(unittest.TestCase):
    def test_hand_built_source_proof_is_a_declaration_attribute_not_a_generated_api(self):
        raw, o, f, sources = fixture()
        row = mapped(raw, [o], f, sources)
        self.assertEqual(row['status'], 'classified_non_api', row['pending_reasons'])
        self.assertEqual(row['obligation'], 'phase1_declaration_attribute')
        self.assertEqual(row['occurrence_ids'], ['occ-field'])
        self.assertEqual(row['alignment_specifiers'][0]['alignment_expression'], 'Alignment')

    def test_tampered_semantic_claims_are_pending_even_if_both_copies_agree(self):
        mutations = {
            'expression': lambda a: a.update(alignment_expression='128'),
            'expression_range': lambda a: a['expression_span'].update(start_byte=a['expression_span']['start_byte'] + 1),
            'expression_line': lambda a: a['expression_span'].update(start_line=999),
            'attribute_line': lambda a: a.update(end_line=999),
            'owner_kind': lambda a: a['owner_hint'].update(kind='type'),
            'owner_range': lambda a: a['owner_hint']['declaration_span'].update(start_byte=0),
            'missing_branch': lambda a: a['conditional_expansions'].pop(),
            'changed_condition': lambda a: a['conditional_expansions'][0]['conditions'][0].update(expression='defined(__CUDA_ARCH__)'),
            'missing_condition': lambda a: a['conditional_expansions'][0].update(conditions=[]),
            'missing_condition_range': lambda a: a['conditional_expansions'][0]['conditions'][0].pop('directive_span'),
            'wrong_expansion': lambda a: a['conditional_expansions'][0].update(expanded_spelling='__align__(128)'),
            'wrong_body': lambda a: a['conditional_expansions'][0]['definition'].update(body='__align__(2*n)'),
            'definition_line': lambda a: a['conditional_expansions'][0]['definition'].update(start_line=54),
            'definition_hash': lambda a: a['conditional_expansions'][0].update(definition_source_sha256='0' * 64),
            'missing_include': lambda a: a.update(definition_include_chain=[]),
            'wrong_include_line': lambda a: a['definition_include_chain'][0].update(start_line=2, end_line=2),
            'wrong_include_target': lambda a: a['definition_include_chain'][0].update(target_path='include/cute/config.hpp'),
        }
        for name, mutate in mutations.items():
            raw, o, f, sources = fixture()
            mutate(o['alignment_specifiers'][0])
            mutate(f['syntax_analysis']['header_analysis']['alignment_attributes'][0])
            with self.subTest(mutation=name):
                row = mapped(raw, [o], f, sources)
                self.assertEqual(row['status'], 'pending', row)
                self.assertTrue(row['pending_reasons'])

    def test_owner_ids_missing_owner_and_scope_taint_are_not_accepted(self):
        for mode in ['wrong_id', 'wrong_entity', 'missing_owner', 'missing_file', 'scope_taint']:
            raw, o, f, sources = fixture()
            occurrences = [o]
            if mode == 'wrong_id': o['alignment_specifiers'][0]['owner_occurrence_id'] = 'other'
            elif mode == 'wrong_entity': o['alignment_specifiers'][0]['owner_entity_id'] = 'other'
            elif mode == 'missing_owner': occurrences = []
            elif mode == 'missing_file': f = {}
            else: o['scope_review_required'] = True
            with self.subTest(mode=mode):
                self.assertEqual(mapped(raw, occurrences, f, sources)['status'], 'pending')

    def test_stale_alignment_after_local_macro_changes_is_rejected(self):
        for extra in ['#undef CUTE_ALIGNAS\n', '#define CUTE_ALIGNAS(n) alignas(2*(n))\n',
                      '#if CHANGED\n#undef CUTE_ALIGNAS\n#endif\n', '#define __align__(n) ignored(n)\n']:
            raw, o, f, sources = fixture(extra)
            row = mapped(raw, [o], f, sources)
            with self.subTest(extra=extra):
                self.assertEqual(row['status'], 'pending')
                self.assertIn('alignment_local_macro_binding_changed', row['pending_reasons'])

    def test_sqlite_staging_retains_attribute_provenance_and_owner_ids(self):
        raw, o, f, sources = fixture()
        full = {**o, 'path': PATH, 'start_byte': o['range'][0], 'end_byte': o['range'][1],
                'declaration_occurrence_id': o['id'], 'source_occurrence_id': o['source_id'],
                'signature_range': location(PATH, raw, *o['signature']),
                'syntax_node_range': location(PATH, raw, *o['syntax']),
                'declarator_range': location(PATH, raw, *o['declarator']),
                'raw_signature': raw[slice(*o['signature'])].decode(), 'parameters': [], 'preprocessor_conditions': []}
        with tempfile.TemporaryDirectory(prefix='alignment-review-') as directory:
            path = Path(directory) / 'declarations.json'
            path.write_text(json.dumps({'occurrences': [full], 'diagnostics': [], 'files': [f]}))
            db = sqlite3.connect(':memory:')
            self.addCleanup(db.close)
            recon.stage_declarations(path, db)
            staged_f = json.loads(db.execute('SELECT value FROM files').fetchone()[0])
            staged_o = json.loads(db.execute('SELECT value FROM occurrences').fetchone()[0])
            self.assertEqual(staged_f['syntax_analysis'], f['syntax_analysis'])
            self.assertEqual(staged_o['alignment_specifiers'], o['alignment_specifiers'])
            row = mapped(raw, [staged_o], staged_f, sources)
            self.assertEqual(row['status'], 'classified_non_api', row['pending_reasons'])


class AlignmentFixedSourceReconciliation(unittest.TestCase):
    def test_all_sixteen_real_attributes_and_namespace_variants_reconcile(self):
        from extract_declarations import Extractor
        from test_cute_alignment_review import EXPECTED
        count = 0
        for path in sorted({v[0] for v in EXPECTED}):
            raw = (ROOT / 'snapshot' / path).read_bytes()
            extractor = Extractor('independent_alignment_reconciliation')
            file_record = extractor.extract(path, raw)
            occurrences = [recon.slim_occurrence(o) for o in extractor.occurrences]
            cs = recon.LEXER.scan_sources({path: raw})['candidates']
            candidates = [c for c in cs if c['kind'] == 'macro_invocation' and c.get('origin') == 'source' and c['name'] == 'CUTE_ALIGNAS']
            for candidate in candidates:
                checker = recon.Reconciler(path, raw, cs, occurrences, extractor.diagnostics,
                    file_record=file_record, source_reader=lambda p: (ROOT / 'snapshot' / p).read_bytes())
                row = checker.reconcile(candidate)
                self.assertEqual(row['status'], 'classified_non_api', (path, candidate['line_range'], row['pending_reasons']))
                self.assertEqual(row['obligation'], 'phase1_declaration_attribute')
                if path.startswith('include/cutlass/'):
                    self.assertEqual(len(row['occurrence_ids']), 2)
                    dropped = [o for o in occurrences if o['id'] != row['occurrence_ids'][0]]
                    broken = recon.Reconciler(path, raw, cs, dropped, extractor.diagnostics,
                        file_record=file_record, source_reader=lambda p: (ROOT / 'snapshot' / p).read_bytes()).reconcile(candidate)
                    self.assertEqual(broken['status'], 'pending')
                    self.assertIn('namespace_binding_instance_alternatives_incomplete', broken['pending_reasons'])
                count += 1
        self.assertEqual(count, 16)


if __name__ == '__main__':
    unittest.main()
