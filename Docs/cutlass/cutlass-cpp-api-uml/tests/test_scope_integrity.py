"""Independent scope-integrity evidence; no extractor changes or source rewrite."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from tree_sitter import Language,Parser
import tree_sitter_cpp

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('declaration_scope_integrity',ROOT/'scripts/declaration_scope_integrity.py')
integrity=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(integrity)
LANGUAGE=Language(tree_sitter_cpp.language())
ANNOTATIONS=re.compile(rb'\b(?:CUTLASS_HOST_DEVICE|CUTLASS_HOST|CUTLASS_DEVICE|CUTE_HOST_DEVICE|CUTE_DEVICE|CUTLASS_PRAGMA_UNROLL|CUTLASS_PRAGMA_NO_UNROLL|CUTE_UNROLL|CUTE_NO_UNROLL)\b')

class ScopeIntegrityTests(unittest.TestCase):
    def analyze(self,source,**kwargs):
        if isinstance(source,str):source=source.encode()
        tree=Parser(LANGUAGE).parse(source)
        return integrity.analyze_scope_integrity('fixture.hpp',source,tree,**kwargs)

    def clang(self,source,define):
        compiler=shutil.which('clang++')
        if not compiler:self.skipTest('clang++ unavailable')
        result=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-x','c++','-',f'-DA={define}'],input=source,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_actual_sm90_unfixed_tree_detects735_instead_of1388(self):
        path='include/cutlass/pipeline/sm90_pipeline.hpp';source=(ROOT/'snapshot'/path).read_bytes()
        masked=ANNOTATIONS.sub(lambda m:b' '*len(m[0]),source)
        result=integrity.analyze_scope_integrity(path,source,Parser(LANGUAGE).parse(masked))
        findings=[d for d in result['diagnostics'] if d['kind']=='namespace_definition' and d['name']=='cutlass']
        self.assertEqual(len(findings),1);finding=findings[0]
        self.assertTrue(finding['blocks_phase_1'])
        self.assertEqual(finding['parsed_body_range']['end_line'],735)
        self.assertEqual(finding['expected_closing_brace']['start_line'],1388)
        self.assertEqual(finding['expected_closing_brace']['start_byte'],47638)
        self.assertEqual(finding['affected_tail']['start_byte'],25630)
        self.assertEqual(finding['affected_tail']['end_byte'],47639)
        self.assertEqual(finding['mismatch_direction'],'premature_close')
        self.assertFalse(result['uncertainties'])

    def test_actual_sm90_named_parser_projection_has_no_scope_mismatch(self):
        path='include/cutlass/pipeline/sm90_pipeline.hpp';source=(ROOT/'snapshot'/path).read_bytes()
        source=source.replace(b'ThisTemplateParameterExistsOnlyForDependentFalse* /* unused */ = nullptr',
                              b'ThisTemplateParameterExistsOnlyForDependentFalse* unused = nullptr')
        tree=Parser(LANGUAGE).parse(ANNOTATIONS.sub(lambda m:b' '*len(m[0]),source))
        result=integrity.analyze_scope_integrity(path,source,tree)
        self.assertEqual(result['summary']['blocking_mismatches'],0)
        self.assertEqual(result['summary']['uncertain_scopes'],0)
        self.assertEqual(result['summary']['recognized_scope_bodies'],24)

    def test_balanced_condition_branches_preserve_literal_closure(self):
        source='namespace n {\n#if A\nstruct TypeA { int x; };\n#else\nstruct TypeB { void f() {} };\n#endif\nunion U { int x; float y; };\n}'
        result=self.analyze(source)
        self.assertFalse(result['diagnostics']);self.assertFalse(result['uncertainties'])
        self.assertTrue(all(s['trust_scope_bounds'] for s in result['scopes']))
        self.clang(source,0);self.clang(source,1)

    def test_alternative_physical_closings_are_uncertain_not_source_error(self):
        source='namespace n {\n#if A\n}\n#else\n}\n#endif\n'
        result=self.analyze(source)
        self.assertFalse(result['diagnostics'])
        self.assertTrue(result['uncertainties'])
        n=next(s for s in result['scopes'] if s['name']=='n')
        self.assertEqual(len(n['candidate_closing_braces']),2)
        self.assertFalse(n['trust_scope_bounds'])
        self.assertTrue(all(not d['blocks_phase_1'] for d in result['uncertainties']))
        self.clang(source,0);self.clang(source,1)

    def test_correlated_conditions_not_assumed_independent_source_errors(self):
        source='namespace n {\n#if A\nnamespace inner {\n#endif\nstruct S {};\n#if A\n}\n#endif\n}\n'
        result=self.analyze(source)
        self.assertFalse(result['diagnostics']);self.assertTrue(result['uncertainties'])
        self.clang(source,0);self.clang(source,1)

    def test_scope_opening_inside_branch_keeps_branch_ownership(self):
        source='namespace outer {\n#if A\nstruct S { int x; };\n#else\nstruct T { int y; };\n#endif\n}\n'
        result=self.analyze(source)
        self.assertFalse(result['diagnostics']);self.assertFalse(result['uncertainties'])
        inner=[s for s in result['scopes'] if s['name'] in ('S','T')]
        self.assertEqual(len(inner),2)
        self.assertTrue(all(len(s['enclosing_conditions'])==1 for s in inner))
        self.assertNotEqual(inner[0]['enclosing_conditions'][0]['branch_id'],inner[1]['enclosing_conditions'][0]['branch_id'])

    def test_literals_comments_raw_strings_and_digit_separators_are_shielded(self):
        source='namespace n { const char* a=R"tag( } #endif { )tag"; char c=\'}\'; int x=12\'345; /* } */ struct S {}; // }\n}'
        result=self.analyze(source)
        self.assertFalse(result['diagnostics']);self.assertFalse(result['uncertainties'])
        self.assertEqual(next(s for s in result['scopes'] if s['name']=='n')['expected_closing_brace']['start_byte'],len(source.encode())-1)

    def test_continued_line_comments_and_define_bodies_are_not_source_braces(self):
        source='#define UNUSED(X) } \\\n{ X {\nnamespace n {\n// continued } \\\n} {\nstruct S {};\n}\n'
        result=self.analyze(source)
        self.assertFalse(result['diagnostics']);self.assertFalse(result['uncertainties'])

    def test_unexpanded_scope_macro_prevents_a_confident_physical_claim(self):
        source='#define CLOSE }\nnamespace n { CLOSE int f(); }\n'
        result=self.analyze(source)
        self.assertFalse(result['diagnostics']);self.assertTrue(result['uncertainties'])
        self.assertTrue(any(r['reason']=='unexpanded_scope_changing_macro' for u in result['uncertainties'] for r in u['reasons']))

    def test_external_scope_macro_can_be_supplied_by_caller(self):
        result=self.analyze('namespace n { EXTERNAL_CLOSE int f(); }',known_scope_macros=['EXTERNAL_CLOSE'])
        self.assertFalse(result['diagnostics']);self.assertTrue(result['uncertainties'])

    def test_parser_errors_alone_do_not_prove_scope_corruption(self):
        result=self.analyze('namespace n { void f(int* = nullptr) {} struct S { int x; }; }')
        self.assertFalse(result['diagnostics'])
        self.assertTrue(next(s for s in result['scopes'] if s['name']=='n')['parser_node_has_error'])
        self.assertTrue(next(s for s in result['scopes'] if s['name']=='n')['trust_scope_bounds'])

    def test_if_zero_branch_is_not_silently_dropped(self):
        result=self.analyze('namespace n {\n#if 0\n}\n#else\n}\n#endif\n')
        self.assertFalse(result['diagnostics']);self.assertTrue(result['uncertainties'])
        self.assertEqual(len(next(s for s in result['scopes'] if s['name']=='n')['candidate_closing_braces']),2)

    def test_malformed_preprocessing_is_uncertain_not_a_scope_error(self):
        result=self.analyze('namespace n {\n#if A\nstruct S {};\n}\n')
        self.assertFalse(result['diagnostics']);self.assertTrue(result['uncertainties'])

    def test_budget_exhaustion_retains_uncertainty(self):
        result=self.analyze('namespace n { struct S { int x; }; }',max_steps=1)
        self.assertFalse(result['diagnostics']);self.assertTrue(result['uncertainties'])

    def test_utf8_byte_ranges_and_root_node_entrypoint(self):
        source='// 中文\nnamespace n { struct S {}; }\n'.encode()
        result=integrity.analyze_scope_integrity('utf8.hpp',source,Parser(LANGUAGE).parse(source).root_node)
        self.assertFalse(result['diagnostics'])
        n=next(s for s in result['scopes'] if s['name']=='n')
        self.assertEqual(source[n['opening_brace']['start_byte']:n['opening_brace']['end_byte']],b'{')
        self.assertEqual(source[n['expected_closing_brace']['start_byte']:n['expected_closing_brace']['end_byte']],b'}')

    def test_different_coordinate_layer_is_not_misreported_as_source_damage(self):
        original=b'namespace n {}'
        tree=Parser(LANGUAGE).parse(b'namespace long_parser_only_name {}')
        result=integrity.analyze_scope_integrity('fixture.hpp',original,tree)
        self.assertFalse(result['diagnostics']);self.assertTrue(result['summary']['input_rejected'])

if __name__=='__main__':unittest.main()
