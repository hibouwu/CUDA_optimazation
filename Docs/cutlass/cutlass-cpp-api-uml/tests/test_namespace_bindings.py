import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from extract_declarations import Extractor


class NamespaceBindingTests(unittest.TestCase):
    def extract(self,s):
        e=Extractor('fixture');e.extract('fixture.hpp',s.encode());return e

    def test_stl_both_literal_variants_and_provenance(self):
        e=self.extract('namespace CUTE_STL_NAMESPACE { struct X { void f(); }; }')
        fs=[o for o in e.occurrences if o['name']=='f']
        self.assertEqual({o['qualified_name']for o in fs},{'std::X::f','cuda::std::X::f'})
        self.assertEqual(len({o['declaration_occurrence_id']for o in fs}),2)
        self.assertTrue(all(o['preprocessor_conditions']for o in fs))
        self.assertFalse(e.diagnostics)
        for o in fs:
            scopes=o['scope_chain']
            self.assertTrue(any(s['name_resolution'].get('definition_chain')for s in scopes))

    def test_cutlass_default_and_parameterized_owner(self):
        e=self.extract('namespace cutlass { int f(); }')
        fs=[o for o in e.occurrences if o['name']=='f']
        self.assertEqual(len(fs),2)
        default=next(o for o in fs if o['qualified_name']=='cutlass::f')
        parameter=next(o for o in fs if o['qualified_name_resolution']=='parameterized_macro_binding')
        self.assertNotEqual(default['entity_id'],parameter['entity_id'])
        resolution=parameter['scope_chain'][0]['name_resolution']
        self.assertTrue(resolution['missing_bindings'])
        self.assertIn('rescan',resolution['scope_expression'])
        self.assertEqual(len(resolution['definition_chain']),3)
        self.assertFalse(e.diagnostics)

    def test_macros_inside_namespace_materialized_in_both_contexts(self):
        e=self.extract('#define MAKE(X) struct X {};\nnamespace CUTE_STL_NAMESPACE { MAKE(A) }')
        generated=[o for o in e.occurrences if o['name']=='A' and 'macro_origin'in o]
        self.assertEqual({o['qualified_name']for o in generated},{'std::A','cuda::std::A'})
        self.assertEqual(len({o['declaration_occurrence_id']for o in generated}),2)

    def test_define_after_namespace_does_not_retroactively_rename(self):
        e=self.extract('namespace API_NS { struct X {}; }\n#define API_NS std\n')
        x=next(o for o in e.occurrences if o['name']=='X')
        self.assertEqual(x['qualified_name'],'API_NS::X')

    def test_undef_restores_literal_namespace(self):
        e=self.extract('#define API_NS std\n#undef API_NS\nnamespace API_NS { struct X {}; }')
        self.assertEqual(next(o for o in e.occurrences if o['name']=='X')['qualified_name'],'API_NS::X')

    def test_false_undef_does_not_end_macro_lifetime(self):
        e=self.extract('#define API_NS std\n#if 0\n#undef API_NS\n#endif\nnamespace API_NS { struct X {}; }')
        self.assertEqual(next(o for o in e.occurrences if o['name']=='X')['qualified_name'],'std::X')

    def test_conditional_undef_is_not_assumed_unconditional(self):
        e=self.extract('#define API_NS std\n#if FLAG\n#undef API_NS\n#endif\nnamespace API_NS { struct X {}; }')
        self.assertTrue(any(d['category']=='namespace_macro_expansion_pending' for d in e.diagnostics))

    def test_commented_and_spliced_undef(self):
        e=self.extract('#define API_NS std\n/*\n#undef API_NS\n*/\nnamespace API_NS { struct X {}; }')
        self.assertEqual(next(o for o in e.occurrences if o['name']=='X')['qualified_name'],'std::X')
        e=self.extract('#define API_NS std\n#un\\\ndef API_NS\nnamespace API_NS { struct X {}; }')
        self.assertEqual(next(o for o in e.occurrences if o['name']=='X')['qualified_name'],'API_NS::X')


if __name__=='__main__':unittest.main()
