"""Source-position regression checks, including repeated text in one API."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from extract_declarations import Extractor


class SemanticProjectionTests(unittest.TestCase):
    def extract(self, source):
        result = Extractor('semantic-projection-regression')
        result.extract('fixture.hpp', source.encode())
        return result

    def test_identical_spelling_in_same_return_type_not_replaced(self):
        source = 'template<class T> auto f()->decltype((typename T::X{}, sizeof(typename T::X), "(typename T::X)"));'
        result = self.extract(source)
        f = next(o for o in result.occurrences if o['name'] == 'f')
        self.assertEqual(f['return_type'], 'decltype((typename T::X{}, sizeof(typename T::X), "(typename T::X)"))')
        entity = result.entities[f['entity_id']]
        self.assertNotIn('sizeoftypename', str(entity['identity_key']))
        self.assertFalse(result.diagnostics)

    def test_templated_conditional_alias_keeps_template_and_prefix(self):
        source = 'template<class T> using X=\n#if A\nT\n#else\nT*\n#endif\n;'
        result = self.extract(source)
        aliases = [o for o in result.occurrences if o['name'] == 'X']
        self.assertEqual(len(aliases), 2)
        self.assertEqual(len({o['entity_id'] for o in aliases}), 1)
        for occurrence in aliases:
            self.assertEqual(occurrence['raw_signature'], source)
            self.assertEqual(occurrence['template_parameters'][0]['parameters'][0]['name'], 'T')
        self.assertFalse(result.diagnostics)

    def test_nested_class_access_does_not_leak_outward(self):
        source = '#define MAKE(X) struct X {};\nstruct S {protected: struct N {private: MAKE(A)}; MAKE(B) public: MAKE(C)};'
        result = self.extract(source)
        actual = {o['qualified_name']: o['access'] for o in result.occurrences if 'macro_origin' in o}
        self.assertEqual(actual['S::N::A'], 'private')
        self.assertEqual(actual['S::B'], 'protected')
        self.assertEqual(actual['S::C'], 'public')
        self.assertFalse(result.diagnostics)

    def test_temporary_parameter_name_not_in_entity_identity(self):
        source = 'template<class T, __CUTE_REQUIRES(sizeof(T)>1)> void f(T x = {});'
        result = self.extract(source)
        self.assertNotIn('__codex_parser_', str(result.entities))
        self.assertFalse(result.diagnostics)

    def test_configuration_defining_header_uses_same_namespace_model(self):
        root = Path(__file__).resolve().parents[1]
        path = 'include/cutlass/detail/helper_macros.hpp'
        result = Extractor('fixed-source')
        result.extract(path, (root/'snapshot'/path).read_bytes())
        self.assertFalse([d for d in result.diagnostics if d['category']=='namespace_macro_expansion_pending'])
        namespaces = [o for o in result.occurrences if o['kind']=='namespace']
        self.assertTrue(any(o.get('name_resolution',{}).get('choice_key')=='custom_namespace' for o in namespaces))

    def test_conditional_access_is_not_false_public(self):
        source = 'struct S {\n#if A\nprivate:\n#else\npublic:\n#endif\nint x; protected: int y;};'
        result = self.extract(source)
        x = next(o for o in result.occurrences if o['name']=='x')
        y = next(o for o in result.occurrences if o['name']=='y')
        self.assertEqual(x['access'],'conditional')
        self.assertEqual([c['access']for c in x['access_resolution']['ordered_cases']],['public','private'])
        self.assertEqual(x['access_resolution']['ordered_cases'][1]['conditions'][0]['expression'],'A')
        self.assertEqual(y['access'],'protected')
        self.assertFalse(result.diagnostics)

    def test_conditional_access_reaches_macro_placeholder(self):
        source = '#define MAKE(X) struct X {};\nstruct S {\n#if A\nprivate:\n#endif\nMAKE(X)};'
        result = self.extract(source)
        x = next(o for o in result.occurrences if o['name']=='X')
        self.assertEqual(x['access'],'conditional')
        self.assertEqual(x['access_resolution']['otherwise'],'public')
        self.assertEqual(x['access_resolution']['ordered_cases'][0]['access'],'private')
        self.assertFalse(result.diagnostics)


if __name__ == '__main__':
    unittest.main()
