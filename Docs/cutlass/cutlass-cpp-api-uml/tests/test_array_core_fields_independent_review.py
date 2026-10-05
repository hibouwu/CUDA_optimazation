"""Independent bounded review of array-related core field changes.

Never write generator/data; use one fixed header and small in-memory fixtures.
The new spelling fields describe written declaration syntax, not fully deduced
C++ types or all cv levels. Clang verifies selected type correspondences.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from extract_declarations import Extractor

def compact(text):return ''.join(text.split())

class ArrayCoreFieldsIndependentReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path='include/cute/container/array.hpp'
        cls.raw=(ROOT/'snapshot'/cls.path).read_bytes()
        cls.array=Extractor('8f50b052e1099fb982392a622caab69b97b63128')
        cls.array.extract(cls.path,cls.raw)

    def extract(self,source):
        result=Extractor('independent-field-review')
        result.extract('include/cute/independent_field_review.hpp',source.encode())
        self.assertFalse(result.diagnostics,result.diagnostics)
        return result

    def semantic_fields_have_no_parser_names(self,result):
        fields=('name','qualified_name','raw_signature','return_type','return_cv_qualifiers','parameters',
                'declared_type','declared_type_spelling','declarator','initializer','qualifiers')
        for occurrence in result.occurrences:
            payload=json.dumps({k:occurrence.get(k)for k in fields},ensure_ascii=False)
            self.assertNotIn('__codex_parser_',payload)
            self.assertNotIn('__codex_named_parameter_',payload)

    def test_array_denominator_and_all_function_parameter_indices(self):
        self.assertFalse(self.array.diagnostics)
        self.assertEqual(len(self.array.occurrences),92)
        functions=[o for o in self.array.occurrences if o['kind']in('method','function','operator')]
        self.assertEqual(len(functions),52)
        self.assertEqual(sum(len(o['parameters'])for o in functions),19)
        for o in functions:
            self.assertEqual([p['index']for p in o['parameters']],list(range(len(o['parameters']))))
            self.assertIn('constexpr',o['qualifiers'])
            self.assertEqual(o['qualifiers'].count('const'),int('const'in o['qualifiers']))
            s=o['signature_range']
            self.assertEqual(self.raw[s['start_byte']:s['end_byte']].decode().rstrip(),o['raw_signature'])
            for p in o['parameters']:
                self.assertEqual(self.raw[p['start_byte']:p['end_byte']].decode(),p['raw'])
        self.assertEqual(sum('const'in o['qualifiers']for o in functions),22)
        self.assertNotIn('__codex_parser_',json.dumps(self.array.entities))

    def test_array_storage_keeps_base_and_complete_declarator(self):
        fields=[o for o in self.array.occurrences if o['kind']in('member','member_constant')]
        self.assertEqual(len(fields),1)
        member=fields[0]
        self.assertEqual(member['qualified_name'],'cute::array::__elems_')
        self.assertEqual(member['declared_type'],'element_type')
        self.assertEqual(member['declarator'],'__elems_[N]')
        self.assertEqual(compact(member['declared_type_spelling']),'element_type[N]')
        self.assertEqual(member['declared_type_role'],'declaration_type_specifier')
        self.assertIsNone(member['initializer'])
        self.assertEqual(self.raw[member['signature_range']['start_byte']:member['signature_range']['end_byte']].decode().rstrip(),member['raw_signature'])

    def test_array_const_pointer_return_and_free_get_are_not_same_qualifier(self):
        data=[o for o in self.array.occurrences if o['name']=='data'and 'const'in o['qualifiers']]
        self.assertEqual(len(data),2)
        for o in data:
            self.assertEqual(o['return_type'],'const T *')
            self.assertEqual(o['return_cv_qualifiers'],['const'])
            self.assertEqual(o['qualifiers'],['const','constexpr'])
        get=next(o for o in self.array.occurrences if o['name']=='get'and o['signature_range']['start_line']==412)
        self.assertEqual(get['return_type'],'const T &')
        self.assertEqual(get['return_cv_qualifiers'],['const'])
        self.assertEqual(get['qualifiers'],['constexpr'])
        self.semantic_fields_have_no_parser_names(self.array)

    def test_cv_receiver_overloads_keep_distinct_identity(self):
        result=self.extract('struct S { const int* f(); int* f() const; const volatile int* g() const volatile & noexcept; };')
        f=[o for o in result.occurrences if o['name']=='f']
        self.assertEqual(len({o['entity_id']for o in f}),2)
        self.assertEqual([o['qualifiers']for o in f],[[],['const']])
        self.assertEqual([o['return_cv_qualifiers']for o in f],[['const'],[]])
        g=next(o for o in result.occurrences if o['name']=='g')
        self.assertEqual(g['return_cv_qualifiers'],['const','volatile'])
        self.assertEqual(g['qualifiers'],['const','volatile','&','noexcept'])
        self.assertEqual(g['return_type'],'const volatile int *')

    def test_parameter_index_keeps_anonymous_defaults_and_variadic_sentinel(self):
        result=self.extract('struct S { const int* f(int* = nullptr, const float& named = {}, ... ) const; };')
        f=next(o for o in result.occurrences if o['name']=='f')
        self.assertEqual([p['index']for p in f['parameters']],[0,1,2])
        self.assertEqual([p['name']for p in f['parameters']],[None,'named',None])
        self.assertEqual([p['default']for p in f['parameters']],['nullptr','{}',None])
        self.assertEqual([compact(p['type'])for p in f['parameters']],['int*','constfloat&','...'])
        self.semantic_fields_have_no_parser_names(result)

    def test_multiple_field_declarators_do_not_share_initializer_or_pointer_shape(self):
        result=self.extract('struct S { int *pointer=nullptr, values[3]={1,2,3}; const int* const fixed=nullptr; int (*matrix)[3]=nullptr; int (&reference)[3]; };')
        by_name={o['name']:o for o in result.occurrences}
        expected={'pointer':('int*','nullptr'),'values':('int[3]','{1,2,3}'),
                  'fixed':('constint*const','nullptr'),'matrix':('int(*)[3]','nullptr'),'reference':('int(&)[3]',None)}
        for name,(tp,initializer)in expected.items():
            self.assertEqual(compact(by_name[name]['declared_type_spelling']),tp)
            self.assertEqual(by_name[name]['initializer'],initializer)
            self.assertNotIn(name,by_name[name]['declared_type_spelling'])
        self.assertEqual(by_name['pointer']['declared_type'],'int')
        self.assertEqual(by_name['values']['declarator'],'values[3]')

    def test_field_function_pointer_array_remains_a_field(self):
        result=self.extract('struct S { const int (*callback)(float x)=nullptr; int (*callbacks[2])(float x)={}; };')
        by_name={o['name']:o for o in result.occurrences}
        self.assertIn(by_name['callback']['kind'],('member','member_constant'))
        self.assertEqual(compact(by_name['callback']['declared_type_spelling']),'constint(*)(floatx)')
        self.assertEqual(compact(by_name['callbacks']['declared_type_spelling']),'int(*[2])(floatx)')
        self.assertEqual(by_name['callbacks']['initializer'],'{}')

    def test_projection_restores_repeated_typename_and_anonymous_bitfield(self):
        result=self.extract('template<class T> struct S { decltype(typename T::X{}) value{}; decltype((typename T::X{}, sizeof(typename T::X))) *pointer=nullptr; unsigned : 3; const int* f(int* = nullptr,T t = {}) const; };')
        by_name={o['name']:o for o in result.occurrences}
        self.assertEqual(by_name['value']['declared_type_spelling'],'decltype(typename T::X{})')
        self.assertEqual(by_name['pointer']['declared_type_spelling'],'decltype((typename T::X{}, sizeof(typename T::X))) *')
        self.assertEqual(by_name[None]['declared_type_spelling'],'unsigned')
        self.assertEqual(by_name[None]['bit_width'],'3')
        self.assertEqual(by_name['f']['parameters'][0]['name'],None)
        self.semantic_fields_have_no_parser_names(result)

    def test_macro_members_and_conditional_header_keep_new_fields(self):
        result=self.extract('#define FIELDS(N) int slots[N]={}; const int* ptr=nullptr;\nstruct S { FIELDS(3) }; struct U { static const int* f(\n#if A\nint x\n#else\nfloat y\n#endif\n) { return nullptr; } };')
        members={o['name']:o for o in result.occurrences if o.get('macro_origin')}
        self.assertEqual(compact(members['slots']['declared_type_spelling']),'int[3]')
        self.assertEqual(members['slots']['initializer'],'{}')
        functions=[o for o in result.occurrences if o['name']=='f']
        self.assertEqual(len(functions),2)
        self.assertEqual({f['parameters'][0]['name']for f in functions},{'x','y'})
        self.assertEqual({f['parameters'][0]['index']for f in functions},{0})
        for f in functions:
            self.assertEqual(f['qualifiers'],['static'])
            self.assertEqual(f['return_cv_qualifiers'],['const'])
            self.assertEqual(f['return_type'],'const int *')
        self.semantic_fields_have_no_parser_names(result)

    def test_adjacent_macro_and_conditional_header_is_an_explicit_pending_boundary(self):
        source='#define FIELDS(N) int slots[N]={}; const int* ptr=nullptr;\nstruct S { FIELDS(3) static const int* f(\n#if A\nint x\n#else\nfloat y\n#endif\n) { return nullptr; } };'
        result=Extractor('independent-pending-boundary')
        result.extract('include/cute/independent_field_review.hpp',source.encode())
        categories={d['category']for d in result.diagnostics}
        self.assertIn('conditional_function_variant_parse_pending',categories)
        self.assertIn('parse_error',categories)
        self.assertTrue(all(o['parse_status']!='parsed'for o in result.occurrences if o['name']=='f'))
        self.semantic_fields_have_no_parser_names(result)
        clang=shutil.which('clang++')
        if clang:
            for condition in(0,1):
                p=subprocess.run([clang,'-std=c++17','-fsyntax-only',f'-DA={condition}','-x','c++','-'],input=source,text=True,capture_output=True,timeout=30)
                self.assertEqual(p.returncode,0,p.stderr)

    def test_new_spelling_fields_are_not_deduced_type_or_all_cv_levels(self):
        result=self.extract('struct S { int* const p(); auto q() const -> const int*; static constexpr int values[2]={1,2}; };')
        by_name={o['name']:o for o in result.occurrences}
        self.assertEqual(by_name['p']['return_type'],'int * const')
        self.assertEqual(by_name['q']['return_type'],'const int*')
        self.assertEqual(by_name['p']['return_cv_qualifiers'],[])
        self.assertEqual(by_name['q']['return_cv_qualifiers'],[])
        self.assertEqual(by_name['q']['qualifiers'],['const'])
        self.assertEqual(compact(by_name['values']['declared_type_spelling']),'int[2]')
        self.assertIn('constexpr',by_name['values']['qualifiers'])

    def test_clang_field_type_correspondence_and_receiver_types(self):
        clang=shutil.which('clang++')
        if not clang:self.skipTest('Clang unavailable')
        source='''
struct S { const int* f(); int* f() const;
 const volatile int* g() const volatile & noexcept;
 int *pointer=nullptr, values[3]={1,2,3}; const int* const fixed=nullptr;
 int (*matrix)[3]=nullptr; int (&reference)[3];
 const int (*callback)(float x)=nullptr; int (*callbacks[2])(float x)={};
 static constexpr int constants[2]={1,2}; };
static_assert(__is_same(decltype(S::pointer), int*));
static_assert(__is_same(decltype(S::values), int[3]));
static_assert(__is_same(decltype(S::fixed), const int* const));
static_assert(__is_same(decltype(S::matrix), int(*)[3]));
static_assert(__is_same(decltype(S::reference), int(&)[3]));
static_assert(__is_same(decltype(S::callback), const int(*)(float)));
static_assert(__is_same(decltype(S::callbacks), int(*[2])(float)));
static_assert(__is_same(decltype(S::constants), const int[2]));
static_assert(__is_same(decltype(static_cast<const int*(S::*)()>(&S::f)),const int*(S::*)()));
static_assert(__is_same(decltype(static_cast<int*(S::*)() const>(&S::f)),int*(S::*)() const));
static_assert(__is_same(decltype(&S::g),const volatile int*(S::*)() const volatile & noexcept));
'''
        p=subprocess.run([clang,'-std=c++17','-fsyntax-only','-x','c++','-'],input=source,text=True,capture_output=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main()
