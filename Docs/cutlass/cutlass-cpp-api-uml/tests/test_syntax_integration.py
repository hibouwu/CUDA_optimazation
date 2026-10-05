from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from extract_declarations import Extractor


class SyntaxIntegrationTests(unittest.TestCase):
    def extract(self,s):
        e=Extractor('fixture');e.extract('fixture.hpp',s.encode());return e

    def test_conditional_alias_has_one_entity_two_target_variants(self):
        s='namespace n { using X =\n#if MODE\nint\n#else\nlong\n#endif\n; }'
        e=self.extract(s);aliases=[o for o in e.occurrences if o['kind']=='alias']
        self.assertEqual(len(aliases),2)
        self.assertEqual(len({o['entity_id']for o in aliases}),1)
        self.assertEqual(len({o['variant_id']for o in aliases}),2)
        self.assertEqual({o['target_type']for o in aliases},{'int','long'})
        self.assertTrue(all(o['qualified_name']=='n::X'for o in aliases))
        for o in aliases:
            r=o['signature_range'];self.assertEqual(s[r['start_byte']:r['end_byte']].rstrip(),o['raw_signature'])
        self.assertFalse(e.diagnostics)

    def test_default_restored_and_no_parser_name_in_api(self):
        s='template<class T> void f(T const& value = {});'
        e=self.extract(s);f=next(o for o in e.occurrences if o['name']=='f')
        self.assertEqual(f['parameters'][0]['default'],'{}')
        self.assertNotIn('__codex_parser',f['raw_signature'])
        self.assertNotIn('__codex_parser',f['expanded_signature'])
        self.assertFalse(e.diagnostics)

    def test_dependent_type_expression_not_false_local_function(self):
        e=self.extract('template<class T> auto f(){return typename T::X{};}')
        self.assertEqual([o['name']for o in e.occurrences],['f'])
        self.assertFalse(e.diagnostics)

    def test_constraint_and_default_projection_compose(self):
        s='template<class T, __CUTE_REQUIRES(sizeof(T)>1)> void f(T const& value = {});'
        e=self.extract(s);f=next(o for o in e.occurrences if o['name']=='f')
        self.assertEqual(f['raw_signature'],s)
        self.assertEqual(f['parameters'][0]['default'],'{}')
        self.assertIn('enable_if',f['expanded_signature'])
        self.assertNotIn('__codex_parser',f['expanded_signature'])
        self.assertFalse(e.diagnostics)

    def test_static_assert_macros_preserve_class_and_constraints(self):
        e=self.extract('template<class T> struct S { CUTE_STATIC_ASSERT_V(T::ok); using X=int; };')
        self.assertFalse(e.diagnostics)
        self.assertTrue(any(o['qualified_name']=='S::X'for o in e.occurrences))
        edit=next(x for x in e.files[0]['syntax_analysis']['projection_edits']if x['kind']=='static_assertion_macro')
        self.assertEqual(edit['expanded'],'static_assert(decltype(T::ok)::value)')
        self.assertTrue(edit['macro_definition']['path'].endswith('cute/config.hpp'))

    def test_conditional_constexpr_is_not_unconditional_api_qualifier(self):
        e=self.extract('CUTLASS_CONSTEXPR_IF_CXX17 int f(){return 1;}')
        f=next(o for o in e.occurrences if o['name']=='f')
        self.assertNotIn('constexpr',f.get('qualifiers',[]))
        self.assertEqual({v['expanded']for v in f['conditional_specifiers'][0]['variants']},{'constexpr',''})
        self.assertIn('CUTLASS_CONSTEXPR_IF_CXX17',f['raw_signature'])


if __name__=='__main__':unittest.main()
