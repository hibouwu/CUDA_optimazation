"""Return-type cv and declarator shapes are not callable cv/ref qualifiers."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from extract_declarations import Extractor


class FunctionQualifierSeparationTests(unittest.TestCase):
    def extract(self,source):
        e=Extractor('fixture');e.extract('include/cute/qualifier_fixture.hpp',source.encode())
        self.assertFalse(e.diagnostics);return e.occurrences

    def test_return_const_never_implies_const_member_function(self):
        os=self.extract('struct S { const int* a(); int const* b() const; int* c() const; };')
        by_name={o['name']:o for o in os}
        self.assertEqual(by_name['a']['qualifiers'],[])
        self.assertEqual(by_name['b']['qualifiers'],['const'])
        self.assertEqual(by_name['c']['qualifiers'],['const'])
        self.assertEqual(by_name['a']['return_cv_qualifiers'],['const'])
        self.assertEqual(by_name['b']['return_cv_qualifiers'],['const'])
        self.assertEqual(by_name['c']['return_cv_qualifiers'],[])

    def test_storage_constexpr_and_callable_ref_qualifiers_are_retained(self):
        os=self.extract('struct S { constexpr int const* f() const & noexcept { return nullptr; } };')
        f=next(o for o in os if o['name']=='f')
        self.assertEqual(set(f['qualifiers']),{'constexpr','const','&','noexcept'})
        self.assertEqual(f['return_type'],'const int *')

    def test_every_written_function_parameter_has_its_own_index(self):
        os=self.extract('int f(int count, float const* ptr, ...);')
        self.assertEqual([p['index']for p in os[0]['parameters']],[0,1,2])

    def test_full_member_type_spelling_does_not_drop_array_or_pointer_shape(self):
        os=self.extract('struct S { int a[3]; int const* ptr = nullptr; int* const fixed = nullptr; };')
        ns={o['name']:o for o in os}
        self.assertEqual(ns['a']['declared_type'],'int')
        self.assertEqual(ns['a']['declarator'],'a[3]')
        self.assertEqual(ns['a']['declared_type_spelling'],'int [3]')
        self.assertEqual(ns['ptr']['declared_type_spelling'],'int const *')
        self.assertEqual(ns['fixed']['declared_type_spelling'],'int * const')
        self.assertEqual(ns['ptr']['initializer'],'nullptr')

    def test_array_const_data_accessor_has_one_callable_const(self):
        path='include/cute/container/array.hpp';e=Extractor('fixture');e.extract(path,(ROOT/'snapshot'/path).read_bytes())
        ds=[o for o in e.occurrences if o['name']=='data'and o['signature_range']['start_line']in(99,254)]
        self.assertEqual(len(ds),2)
        for d in ds:
            self.assertEqual(d['qualifiers'].count('const'),1)
            self.assertIn('constexpr',d['qualifiers']);self.assertEqual(d['return_cv_qualifiers'],['const'])


if __name__=='__main__':unittest.main()
