import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from macro_expansion import definitions,expand_file,substitute,arguments,constraint_expansions


class MacroTests(unittest.TestCase):
    def test_integral_constant_operators(self):
        path='include/cute/numeric/integral_constant.hpp'
        data=(ROOT/'snapshot'/path).read_bytes();x=expand_file(path,data)
        unary=[e for e in x['expansions']if e['name']=='CUTE_LEFT_UNARY_OP']
        binary=[e for e in x['expansions']if e['name']=='CUTE_BINARY_OP']
        self.assertEqual(len(unary),5)
        self.assertEqual(len(binary),18)
        self.assertEqual({e['bindings']['OP']for e in unary},{'+','-','~','!','*'})
        self.assertEqual(len({e['invocation_id']for e in unary+binary}),23)
        self.assertTrue(all('operator' in e['virtual_source']for e in unary+binary))

    def test_comments_and_strings_not_invocations(self):
        s=b'#define MAKE(X) struct X {};\n// MAKE(Fake)\nchar const* s="MAKE(Fake)";\nMAKE(Real)\n'
        x=expand_file('test.hpp',s)
        self.assertEqual(len(x['expansions']),1)
        self.assertIn('struct Real',x['expansions'][0]['virtual_source'])

    def test_pasting_and_stringification(self):
        d=definitions('t',b'#define MAKE(X) struct X ## _type { const char* name = #X; };')[0]
        s,b=substitute(d,['ABC'])
        self.assertIn('ABC_type',s)
        self.assertIn('"ABC"',s)

    def test_ambiguous_definition_not_arbitrarily_chosen(self):
        s=b'#if A\n#define M(X) struct X {};\n#else\n#define M(X) class X {};\n#endif\nM(Y)\n'
        x=expand_file('t',s)
        self.assertFalse(x['expansions'])
        self.assertTrue(x['diagnostics'])

    def test_macro_commas_use_parentheses_not_template_angles(self):
        self.assertEqual(arguments('(std::pair<A,B>,x)',0)[0],['std::pair<A','B>','x'])
        self.assertEqual(arguments('((a,b),x)',0)[0],['(a,b)','x'])

    def test_pasting_not_inside_string(self):
        d=definitions('t',b'#define M(X) struct X { const char* s=" ## "; };')[0]
        self.assertIn('" ## "',substitute(d,['Y'])[0])

    def test_stringification_preserves_spaces_inside_literal(self):
        d=definitions('t',b'#define M(X) struct Y { const char* s=#X; };')[0]
        self.assertIn('a  b',substitute(d,['"a  b"'])[0])

    def test_comment_fake_define_not_active(self):
        self.assertEqual(definitions('t',b'/*\n#define M(X) struct X {};\n*/\n'),[])

    def test_line_splicing_joins_identifier(self):
        d=definitions('t',b'#define M(X) struct NA\\\nME {};')[0]
        self.assertIn('NAME',substitute(d,['Y'])[0])

    def test_constraint_is_expanded_not_discarded(self):
        p='include/cute/numeric/integral_constant.hpp'
        s=(ROOT/'snapshot'/p).read_bytes()
        traits=(ROOT/'snapshot/include/cute/util/type_traits.hpp').read_bytes()
        edits=constraint_expansions(p,s,traits)
        self.assertGreater(len(edits),10)
        self.assertTrue(all('cute::enable_if' in x['expanded']for x in edits))
        self.assertTrue(all('is_std_integral' in x['expanded']for x in edits))


if __name__=='__main__':unittest.main()
