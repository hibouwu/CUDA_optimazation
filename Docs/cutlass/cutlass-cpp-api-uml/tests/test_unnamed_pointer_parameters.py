"""The fixed anonymous-pointer grammar gap must not invent parameter names."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from declaration_syntax import Source,unnamed_pointer_parameter_edits,analyze_syntax,project_for_parser
from extract_declarations import Extractor


class UnnamedPointerParameterTests(unittest.TestCase):
    def extract(self,text):
        x=Extractor('fixture');x.extract('fixture.hpp',text.encode());return x

    def edits(self,text):return unnamed_pointer_parameter_edits(Source('fixture.hpp',text.encode()))

    def assert_no_api_name_leak(self,x):
        # parser_projection_signature is explicitly a diagnostic spelling;
        # actual signatures, names, parameter/default data and identities must
        # remain free of names introduced only for grammar recovery.
        fields=['name','qualified_name','raw_signature','expanded_signature','parameters','template_parameters','return_type','scope_chain']
        leaked=[(o['qualified_name'],field)for o in x.occurrences for field in fields
                if '__codex_parser_'in json.dumps(o.get(field))]
        self.assertEqual(leaked,[])
        self.assertFalse('__codex_parser_'in json.dumps(x.entities))

    def test_function_pointer_default_stays_anonymous(self):
        source='namespace n { template<class T> void f(T* /* unused */ = nullptr) {} struct Later {}; }'
        x=self.extract(source);f=next(o for o in x.occurrences if o['name']=='f')
        self.assertFalse(x.diagnostics)
        self.assertEqual(f['qualified_name'],'n::f')
        self.assertEqual(f['parameters'][0]['name'],None)
        self.assertEqual(f['parameters'][0]['default'],'nullptr')
        self.assertEqual(f['parameters'][0]['type'],'T* /* unused */')
        self.assertTrue(any(o['qualified_name']=='n::Later'for o in x.occurrences))
        self.assert_no_api_name_leak(x)

    def test_fixed_pipeline_namespace_and_later_class_are_restored(self):
        path='include/cutlass/pipeline/sm90_pipeline.hpp';raw=(ROOT/'snapshot'/path).read_bytes()
        x=Extractor('fixed');record=x.extract(path,raw)
        self.assertEqual(len(x.occurrences),508);self.assertFalse(x.diagnostics)
        ns=next(o for o in x.occurrences if o['qualified_name']=='cutlass'and o['kind']=='namespace')
        self.assertEqual((ns['start_line'],ns['end_line']),(48,1388))
        expected={'cutlass::PipelineAsync::producer_tail','cutlass::PipelineAsync::producer_acquire','cutlass::pipeline_init_wait'}
        self.assertTrue(expected<={o['qualified_name']for o in x.occurrences})
        self.assertFalse([o for o in x.occurrences if o['qualified_name']in {'PipelineAsync::producer_tail','pipeline_init_wait'}])
        f=next(o for o in x.occurrences if o['name']=='producer_acquire'and o['signature_range']['start_line']<=731<=o['signature_range']['end_line'])
        self.assertIsNone(f['parameters'][1]['name']);self.assertEqual(f['parameters'][1]['default'],'nullptr')
        self.assertIn('ThisTemplateParameterExistsOnlyForDependentFalse* /* unused */ = nullptr',f['raw_signature'])
        self.assert_no_api_name_leak(x)
        self.assertTrue(any(e['kind']=='unnamed_pointer_parameter_default'for e in record['syntax_analysis']['projection_edits']))

    def test_cpp_compiler_accepts_the_unchanged_anonymous_parameter(self):
        compiler=shutil.which('clang++')
        if not compiler:self.skipTest('clang++ unavailable')
        source='namespace n { template<int> struct S; template<> struct S<0> { template<class T=int> void f(T* /* unused */ = nullptr) {} }; struct Later{}; } int main(){n::S<0>{}.f(); n::Later x;}'
        result=subprocess.run([compiler,'-std=c++17','-pedantic-errors','-fsyntax-only','-x','c++','-'],input=source,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(self.extract(source).diagnostics)

    def test_compound_assignment_is_not_repaired_into_a_parameter(self):
        for source in ['void f(){ int x=2; x*=3; }','void f(){g(x*=3);}','void f(T*=nullptr);']:
            self.assertFalse(self.edits(source),source)

    def test_named_pointer_and_macro_definition_are_not_changed(self):
        self.assertFalse(self.edits('template<class T> void f(T* value = nullptr);'))
        self.assertFalse(self.edits('#define FN(T) void f(T* = nullptr)\n'))
        self.assertFalse(self.edits('// void f(T* = nullptr)\nconst char* s="void f(T* = nullptr)";'))

    def test_nttp_and_invalid_function_pointer_defaults_not_classified_as_function_parameters(self):
        self.assertFalse(self.edits('template<int* = nullptr> struct S;'))
        self.assertFalse(self.edits('void (*p)(int* = nullptr);'))
        self.assertFalse(self.edits('void f(int* =);'))

    def test_two_defaults_do_not_merge_overloads_or_names(self):
        source='void f(int* = nullptr); void f(double* = nullptr);'
        x=self.extract(source);fs=[o for o in x.occurrences if o['name']=='f']
        self.assertEqual(len(fs),2);self.assertEqual(len({o['entity_id']for o in fs}),2)
        self.assertTrue(all(o['parameters'][0]['name']is None for o in fs))
        self.assertFalse(x.diagnostics)

    def test_constructor_and_cv_pointer_preserve_their_api(self):
        source='struct S { S(int* const /* unused */ = nullptr) {} };'
        x=self.extract(source);ctor=next(o for o in x.occurrences if o['kind']=='constructor')
        self.assertIsNone(ctor['parameters'][0]['name']);self.assertEqual(ctor['parameters'][0]['default'],'nullptr')
        self.assertIn('* const',ctor['parameters'][0]['type']);self.assertFalse(x.diagnostics)

    def test_source_map_with_utf8_and_nested_default_projection(self):
        source='// 中文\nnamespace n { template<class T> void f(T* /* p */ = nullptr, T x = {}) {} struct S {}; }'
        x=self.extract(source);f=next(o for o in x.occurrences if o['name']=='f')
        for p in f['parameters']:
            self.assertEqual(source.encode()[p['start_byte']:p['end_byte']].decode(),p['raw'])
        self.assertEqual(f['parameters'][1]['default'].strip(),'{}')
        self.assertFalse(x.diagnostics)


if __name__=='__main__':unittest.main()
