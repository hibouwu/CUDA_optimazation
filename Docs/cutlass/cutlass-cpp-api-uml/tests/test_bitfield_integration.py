"""End-to-end source identity and field contract checks for bitfield integration."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from extract_declarations import Extractor


class BitfieldIntegrationTests(unittest.TestCase):
    def extract(self,source,path='fixture.hpp'):
        result=Extractor('bitfield-integration')
        result.extract(path,source.encode()if isinstance(source,str)else source)
        return result

    def fields(self,result):
        return [o for o in result.occurrences if o.get('is_bitfield')]

    def test_anonymous_fields_do_not_merge_or_acquire_a_cpp_name(self):
        result=self.extract('struct S {unsigned a:14, :2, :0;};')
        fields=self.fields(result)
        self.assertEqual([f['name']for f in fields],['a',None,None])
        self.assertEqual(len({f['entity_id']for f in fields}),3)
        self.assertEqual([f['bit_width']for f in fields],['14','2','0'])
        self.assertTrue(all(f['initializer']is None for f in fields))
        self.assertNotIn('__codex_parser_',str(result.entities))
        self.assertFalse(result.diagnostics)

    def test_three_projection_layers_keep_physical_origin(self):
        source='template<class T,__CUTE_REQUIRES(sizeof(T)>1)> struct S {void f(T x={});unsigned a:1, // 注释\n :2;};'
        result=self.extract(source)
        self.assertFalse(result.diagnostics)
        fields=self.fields(result)
        self.assertEqual([f['name']for f in fields],['a',None])
        data=source.encode()
        for field in fields:
            width=field['bit_width_range'];colon=field['colon_range'];decl=field['declarator_range']
            self.assertEqual(data[width['start_byte']:width['end_byte']].decode(),field['bit_width'])
            self.assertEqual(data[colon['start_byte']:colon['end_byte']],b':')
            self.assertEqual(data[decl['start_byte']:decl['end_byte']].decode(),field['declarator'])
        for occurrence in result.occurrences:
            span=occurrence['signature_range']
            self.assertEqual(data[span['start_byte']:span['end_byte']].decode().rstrip(),occurrence['raw_signature'])
            self.assertNotIn('__codex_parser_',occurrence.get('expanded_signature',''))
        f=next(o for o in result.occurrences if o['name']=='f')
        self.assertEqual(f['parameters'][0]['default'],'{}')

    def test_bit_width_and_cpp20_initializer_are_separate(self):
        fields=self.fields(self.extract('struct S {unsigned a:3=1, b:4{2};};'))
        self.assertEqual([(f['bit_width'],f['initializer'])for f in fields],[('3','1'),('4','{2}')])

    def test_access_and_namespace_alternatives_are_retained(self):
        source='namespace cutlass {struct S {private: unsigned :1, x:2;};}'
        result=self.extract(source)
        fields=self.fields(result)
        self.assertEqual(len(fields),4)
        self.assertEqual({f['access']for f in fields},{'private'})
        self.assertEqual(len({f['bitfield_source_id']for f in fields}),2)
        self.assertEqual(len({f['entity_id']for f in fields}),4)
        self.assertFalse(result.diagnostics)

    def test_macro_generated_bitfield_keeps_invocation_and_virtual_origin(self):
        source='#define FIELDS unsigned a:1, :2;\nstruct S {FIELDS};'
        result=self.extract(source)
        # Object declaration macro is still an explicit unresolved obligation.
        self.assertTrue(result.diagnostics)
        source='#define FIELDS(N) unsigned N:1, :2;\nstruct S {FIELDS(x)};'
        result=self.extract(source)
        fields=self.fields(result)
        self.assertEqual(len(fields),2)
        self.assertTrue(all('macro_origin'in f and 'virtual_range'in f['bitfield_origin']['declarator']for f in fields))
        self.assertFalse(result.diagnostics)

    def test_fixed_descriptors_have_62_independent_fields(self):
        total=[]
        for path in ['include/cute/arch/mma_sm100_desc.hpp','include/cute/arch/mma_sm90_desc.hpp']:
            result=self.extract((ROOT/'snapshot'/path).read_bytes(),path)
            self.assertFalse(result.diagnostics)
            total.extend(self.fields(result))
        self.assertEqual(len(total),62)
        anonymous=[f for f in total if f['name']is None]
        self.assertEqual(len(anonymous),18)
        self.assertEqual(len({f['entity_id']for f in anonymous}),18)
        self.assertEqual(len({f['bitfield_source_id']for f in anonymous}),18)
        self.assertTrue(all(f['initializer']is None for f in total))


if __name__=='__main__':unittest.main()
