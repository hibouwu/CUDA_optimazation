"""Owner trust must follow mapped physical scope evidence, not node.has_error."""
import hashlib
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from extract_declarations import Extractor
from reconcile_candidates import Reconciler,slim_occurrence


class ScopeIntegrityIntegrationTests(unittest.TestCase):
    def test_unrepaired_pipeline_tail_is_not_a_clean_global_api(self):
        path='include/cutlass/pipeline/sm90_pipeline.hpp';raw=(ROOT/'snapshot'/path).read_bytes()
        x=Extractor('fixed');record=x.extract(path,raw,project_syntax=False)
        finding=next(d for d in x.diagnostics if d['category']=='scope_closure_mismatch')
        self.assertEqual(finding['scope_evidence']['expected_closing_brace']['start_line'],1388)
        function=next(o for o in x.occurrences if o['name']=='pipeline_init_wait')
        self.assertEqual(function['qualified_name'],'pipeline_init_wait')  # No invented prefix repair.
        self.assertEqual(function['local_syntax_parse_status'],'parsed')
        self.assertEqual(function['parse_status'],'scope_closure_mismatch')
        self.assertTrue(function['scope_review_required'])
        self.assertEqual(record['diagnostic_count'],len(x.diagnostics))

    def test_repaired_scope_report_uses_physical_bytes_and_hash(self):
        path='include/cutlass/pipeline/sm90_pipeline.hpp';raw=(ROOT/'snapshot'/path).read_bytes()
        x=Extractor('fixed');record=x.extract(path,raw)
        report=record['scope_integrity'];self.assertEqual(report['source_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(report['coordinate_space'],'mapped_source_bytes')
        scope=next(s for s in report['scopes']if s['name']=='cutlass')
        self.assertEqual(scope['expected_closing_brace']['start_byte'],47638)
        self.assertEqual(raw[scope['opening_brace']['start_byte']:scope['opening_brace']['end_byte']],b'{')
        self.assertFalse([o for o in x.occurrences if o.get('scope_review_required')])

    def test_uncertain_conditional_scope_is_not_claimed_as_source_damage(self):
        raw=b'namespace n { int f();\n#if FLAG\n}\n#else\n}\n#endif\n'
        x=Extractor('fixture');record=x.extract('fixture.hpp',raw)
        self.assertTrue(record['scope_integrity']['uncertainties'])
        self.assertFalse([d for d in x.diagnostics if d['category']=='scope_closure_mismatch'])
        function=next(o for o in x.occurrences if o['name']=='f')
        self.assertTrue(function['scope_review_required'])
        occurrences=[slim_occurrence(o)for o in x.occurrences]
        reconciler=Reconciler('fixture.hpp',raw,[],occurrences,x.diagnostics,file_record=record)
        slim=next(o for o in occurrences if o['name']=='f')
        self.assertIn('enclosing_scope_requires_source_review',reconciler.invalid_occurrence(slim))

    def test_macro_expansion_keeps_its_scope_evidence(self):
        raw=b'#define MAKE(N) namespace N { struct S {}; }\nMAKE(example)\n'
        x=Extractor('fixture');record=x.extract('fixture.hpp',raw)
        expansion=record['macro_expansions'][0]
        self.assertIn('scope_integrity',expansion)
        self.assertEqual(expansion['scope_integrity_source_kind'],'virtual_macro_expansion_with_invocation_mapping')
        self.assertTrue(expansion['scope_integrity']['scopes'])

    def test_macro_generated_tail_inherits_untrusted_callsite_scope(self):
        path='include/cutlass/pipeline/sm90_pipeline.hpp';raw=(ROOT/'snapshot'/path).read_bytes()
        raw=b'#define AUDIT_DECL() void audit_generated();\n'+raw.replace(b'}  // end namespace cutlass',b'AUDIT_DECL()\n}  // end namespace cutlass')
        x=Extractor('fixture');record=x.extract(path,raw,project_syntax=False,project_constraints=False,project_bitfields=False)
        generated=next(o for o in x.occurrences if o['name']=='audit_generated')
        self.assertTrue(generated['macro_origin'])
        self.assertTrue(generated['scope_review_required'])
        self.assertEqual(generated['parse_status'],'scope_closure_mismatch')
        self.assertEqual(x.entities[generated['entity_id']]['semantic_resolution'],'unverified_scope_identity_quarantined')
        self.assertTrue(any(e['status']=='expanded_scope_review_pending'for e in record['macro_expansions']))

    def test_uncertain_scope_covers_potential_tail_not_only_parsed_body(self):
        path='include/cutlass/pipeline/sm90_pipeline.hpp';raw=(ROOT/'snapshot'/path).read_bytes()
        raw=b'#define AUDIT_OPEN namespace audit_inner {\n#define AUDIT_CLOSE }\n'+raw.replace(b'namespace cutlass {',b'namespace cutlass {\nAUDIT_OPEN AUDIT_CLOSE',1)
        x=Extractor('fixture');record=x.extract(path,raw,project_syntax=False,project_constraints=False,project_bitfields=False)
        self.assertTrue(record['scope_integrity']['uncertainties'])
        tail=next(o for o in x.occurrences if o['name']=='pipeline_init_wait')
        self.assertTrue(tail['scope_review_required'])
        self.assertTrue(any(not s['proven_mismatch']for s in tail['scope_integrity_refs']))

    def test_untrusted_guessed_global_identity_does_not_merge_with_real_global(self):
        path='include/cutlass/pipeline/sm90_pipeline.hpp';raw=(ROOT/'snapshot'/path).read_bytes()
        raw=b'#define AUDIT_DECL() void audit_generated();\n'+raw.replace(b'}  // end namespace cutlass',b'AUDIT_DECL()\n}  // end namespace cutlass')+b'\nvoid audit_generated();\n'
        x=Extractor('fixture');x.extract(path,raw,project_syntax=False,project_constraints=False,project_bitfields=False)
        functions=[o for o in x.occurrences if o['name']=='audit_generated']
        self.assertEqual(len(functions),2)
        self.assertEqual(len({o['entity_id']for o in functions}),2)
        self.assertEqual(sum(bool(o.get('scope_review_required'))for o in functions),1)


if __name__=='__main__':unittest.main()
