"""Whole declaration/body/identity checks beyond the independent header parser."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from extract_declarations import Extractor


class HeaderIntegrationTests(unittest.TestCase):
    def extract(self,source,path='fixture.hpp'):
        e=Extractor('header-integration');e.extract(path,source.encode()if isinstance(source,str)else source);return e

    def test_private_method_two_parameter_names_one_identity_and_shared_body(self):
        source='struct S {private: int\n#if A\nf(int x)\n#else\nf(int /*x*/)\n#endif\n{ return 42; }};'
        e=self.extract(source);fs=[o for o in e.occurrences if o['name']=='f']
        self.assertEqual(len(fs),2)
        self.assertEqual(len({f['entity_id']for f in fs}),1)
        self.assertEqual({f['access']for f in fs},{'private'})
        self.assertEqual({f['parameters'][0]['name']for f in fs},{'x',None})
        data=source.encode()
        for f in fs:
            sig=f['signature_range'];body=f['body_range']
            self.assertEqual(data[sig['start_byte']:sig['end_byte']].decode().rstrip(),f['raw_signature'])
            self.assertNotIn('return',f['raw_signature'])
            self.assertEqual(data[body['start_byte']:body['end_byte']],b'{ return 42; }')
            self.assertTrue(f['preprocessor_conditions'])
        self.assertFalse(e.diagnostics)

    def test_shared_body_local_type_is_visited_and_not_split_by_header_projection(self):
        source='int\n#if A\nf(int x)\n#else\nf(int /*x*/)\n#endif\n{ struct Local {int value;}; return sizeof(Local); }'
        e=self.extract(source)
        for name in ['Local','value']:
            matches=[o for o in e.occurrences if o['name']==name]
            self.assertEqual(len(matches),2)
            self.assertEqual(len({o['entity_id']for o in matches}),1)
            self.assertEqual(len({o['source_occurrence_id']for o in matches}),1)
        self.assertFalse(e.diagnostics)

    def test_fixed_cluster_initializer_retains_all_configurations_and_no_parse_gap(self):
        path='include/cutlass/cluster_launch.hpp';data=(ROOT/'snapshot'/path).read_bytes()
        e=self.extract(data,path);fs=[o for o in e.occurrences if o['name']=='init']
        self.assertEqual(len(fs),4) # two header choices, two namespace bindings
        self.assertEqual(len({o['entity_id']for o in fs}),2)
        self.assertEqual({o['parameters'][0]['name']for o in fs},{'kernel_function',None})
        self.assertFalse(e.diagnostics)
        for o in e.occurrences:
            if o.get('macro_origin'):continue
            sig=o['signature_range']
            self.assertEqual(data[sig['start_byte']:sig['end_byte']].decode().rstrip(),o['raw_signature'])

    def test_alignment_remains_cuda_spelling_and_source_expression(self):
        path='include/cutlass/platform/platform.h';data=(ROOT/'snapshot'/path).read_bytes();e=self.extract(data,path)
        types=[o for o in e.occurrences if o.get('alignment_specifiers')]
        self.assertEqual(len(types),26)
        self.assertEqual({o['alignment_specifiers'][0]['alignment_expression']for o in types},{str(2**n)for n in range(13)})
        for o in types:
            self.assertIn('__align__(',o['raw_signature'])
            self.assertNotIn('__attribute__',o['raw_signature'])
            a=o['alignment_specifiers'][0]
            self.assertEqual(data[a['start_byte']:a['end_byte']].decode(),a['semantic_spelling'])
        self.assertFalse([d for d in e.diagnostics if d['category']in {'parse_error','missing_syntax'}])

    def test_foreign_namespace_condition_line_not_shifted_by_class_fragment(self):
        source='#define FIELD(N) int N;\nnamespace cutlass {struct S {FIELD(x)};}'
        e=self.extract(source)
        xs=[o for o in e.occurrences if o['name']=='x']
        self.assertEqual(len(xs),2)
        for x in xs:
            foreign=[c for c in x['preprocessor_conditions']if c.get('directive_path')=='include/cutlass/detail/helper_macros.hpp']
            self.assertTrue(foreign)
            self.assertTrue(all(c['directive_line']>10 for c in foreign))
        self.assertFalse(e.diagnostics)


if __name__=='__main__':unittest.main()
