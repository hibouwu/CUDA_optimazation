"""Small deterministic checks for atlas-specific presentation invariants."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import build_atlas
from build_atlas import diagram_source,source_href,without_templates,auxiliary_path,validate_evidence,puml_label,retire_resolved_manual_identity


class AtlasBuildTests(unittest.TestCase):
    def test_each_edge_has_own_arrow_and_id(self):
        nodes={x:{'id':x,'kind':'api','name':x,'path':'include/test.hpp','line':1,'signature':'void '+x+'(int value);'}for x in ['f','g']}
        edges={'e1':{'id':'e1','source':'f','target':'g','relation':'calls','source_expression':'g(1)','evidence':[]},
               'e2':{'id':'e2','source':'f','target':'g','relation':'calls','source_expression':'g(2)','evidence':[]}}
        text=diagram_source('test',['f','g'],['e1','e2'],nodes,edges,True)
        self.assertEqual(text.count('n0 --> n1'),2)
        self.assertIn('void f(int value);',text);self.assertIn('g(1)',text);self.assertIn('g(2)',text)

    def test_attributes_and_macro_expansions_are_not_call_arrows(self):
        for relation in ('has_attribute','member_of','macro_uses','expands_to'):
            self.assertIn(relation,build_atlas.RELATIONS)
            self.assertIn(relation,build_atlas.ARROWS)
            self.assertNotEqual(build_atlas.ARROWS[relation],build_atlas.ARROWS['calls'])

    def test_non_member_does_not_get_a_class_access_label(self):
        node={'id':'m','kind':'api','name':'M','path':'include/test.hpp','line':1,'signature':'#define M(n) alignas(n)',
              'entity_kind':'macro_definition','access':'public','access_scope':'not_class_member'}
        text=diagram_source('macro',['m'],[],{'m':node},{},True)
        self.assertNotIn('access: public',text)
        self.assertIn('#define M(n) alignas(n)',text)

    def test_macro_edge_shows_parameter_binding_and_expansion(self):
        nodes={x:{'id':x,'kind':'binding','name':x,'path':'include/test.hpp','line':1}for x in ['use','definition']}
        edge={'id':'macro','source':'use','target':'definition','relation':'expands_to','macro_bindings':{'n':'Alignment'},'expanded_spelling':'alignas(Alignment)'}
        text=diagram_source('macro',['use','definition'],['macro'],nodes,{'macro':edge},True)
        self.assertIn('n := Alignment',text);self.assertIn('展开结果：alignas(Alignment)',text)

    def test_implicit_range_for_calls_have_distinct_derived_site_identities(self):
        path='include/cute/container/array.hpp';edges={}
        for name in ('begin','end'):
            edges[name]={'id':name,'source':'fill','target':name,'relation':'implicit_calls',
                'source_expression':'for (auto& e : *this) ','evaluation':'language_desugaring',
                'synthetic_expression':'__range.'+name+'()','source_expression_is_not_literal_call':True,
                'evidence':[{'path':path,'start_line':174,'end_line':174}]}
        issues=[];validate_evidence(edges,{'files':[{'path':path}]},issues)
        self.assertFalse(issues)
        a,b=(edges[name]['callsite']for name in ('begin','end'))
        self.assertEqual(a['start_byte'],b['start_byte']);self.assertEqual(a['end_byte'],b['end_byte'])
        self.assertNotEqual(a['callsite_id'],b['callsite_id'])
        self.assertEqual(a['kind'],'language_desugaring');self.assertTrue(a['source_anchor_is_not_literal_call'])
        self.assertNotIn('__range',a['raw_source_expression'])
        edges['begin'].pop('synthetic_expression');issues=[]
        validate_evidence({'begin':edges['begin']},{'files':[{'path':path}]},issues)
        self.assertEqual(issues[0]['kind'],'implicit_call_missing_derivation')

    def test_merged_entity_diagram_uses_the_relation_source_occurrence(self):
        first={'declaration_occurrence_id':'o1','signature':'namespace API_NS','qualified_name':'std','path':'include/test.hpp',
               'signature_range':{'path':'include/test.hpp','start_byte':0,'end_byte':16,'start_line':1},
               'declaration_range':{'path':'include/test.hpp','start_byte':0,'end_byte':50}}
        second={'declaration_occurrence_id':'o2','signature':'namespace std','qualified_name':'std','path':'include/test.hpp',
               'signature_range':{'path':'include/test.hpp','start_byte':60,'end_byte':74,'start_line':6},
               'declaration_range':{'path':'include/test.hpp','start_byte':60,'end_byte':100}}
        node={'id':'std','signature':'namespace API_NS','source_declaration_occurrences':[first,second]}
        edge={'source':'std','target':'target','evidence':[{'path':'include/test.hpp','start_byte':80,'end_byte':90}]}
        view=build_atlas.declaration_for_edge(node,edge)
        self.assertEqual(view['declaration_occurrence_id'],'o2');self.assertEqual(view['line'],6)
        self.assertEqual(view['signature'],'namespace std');self.assertEqual(node['signature'],'namespace API_NS')

    def test_source_paths_are_local_and_version_independent(self):
        self.assertEqual(source_href('include/cute/mma.hpp',94),'source/include/cute/mma.hpp.html#L94')
        self.assertIsNone(auxiliary_path('/etc/passwd'))
        self.assertIsNone(auxiliary_path('data/../../../../../../etc/passwd'))

    def test_scope_comparison_ignores_only_template_arguments(self):
        self.assertEqual(without_templates('cutlass::X<A<B>, C>::run'),'cutlass::X::run')
        self.assertNotEqual(without_templates('run'),'cutlass::run')

    def test_callsite_is_not_evidence_range_start(self):
        import json
        scope=json.loads((ROOT/'data/scope.json').read_text())
        edge={'id':'e','relation':'calls','source_expression':'GemmKernel::to_underlying_arguments(args, workspace)',
              'evidence':[{'path':'include/cutlass/gemm/device/gemm_universal_adapter.h','start_line':323,'end_line':328}]}
        issues=[];validate_evidence({'e':edge},scope,issues)
        self.assertEqual(edge['callsite']['start_line'],328)
        self.assertIn('start_byte',edge['callsite'])
        self.assertFalse(issues)

    def callsite_fixture(self,authored=None):
        raw=b'void f() { g(1); g(1); }\n'
        edge={'id':'same_line','relation':'calls','source_expression':'g(1)',
              'evidence':[{'path':'include/test.hpp','start_line':1,'end_line':1}]}
        if authored is not None:edge['callsite']=authored
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);path=root/'snapshot/include/test.hpp';path.parent.mkdir(parents=True);path.write_bytes(raw)
            with patch.object(build_atlas,'ROOT',root):
                issues=[];validate_evidence({'e':edge},{'files':[{'path':'include/test.hpp'}]},issues)
        return edge,issues,raw

    def test_same_line_calls_do_not_merge(self):
        edge,issues,_=self.callsite_fixture()
        self.assertNotIn('callsite',edge)
        self.assertEqual(issues[0]['kind'],'callsite_not_uniquely_located')
        self.assertEqual(len(issues[0]['matching_ranges']),2)

    def test_authored_byte_site_selects_one_and_preserves_context(self):
        edge,issues,raw=self.callsite_fixture({'path':'include/test.hpp','start_byte':17,'end_byte':21,
            'start_line':1,'end_line':1,'context':'second call','caller':'f'})
        self.assertFalse(issues)
        self.assertEqual(edge['callsite']['context'],'second call')
        self.assertEqual(edge['callsite']['caller'],'f')
        self.assertTrue(edge['callsite']['callsite_id'].startswith('site_'))
        self.assertEqual(raw[edge['callsite']['start_byte']:edge['callsite']['end_byte']],b'g(1)')

    def test_wrong_authored_site_is_not_silently_replaced(self):
        _,issues,_=self.callsite_fixture({'path':'include/test.hpp','start_byte':10,'end_byte':14})
        self.assertEqual(issues[0]['kind'],'authored_callsite_not_matched')

    def test_labels_do_not_break_identifiers_or_path_components(self):
        self.assertIn('sm100_mma_warpspecialized.hpp',puml_label('include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp',28))
        self.assertNotIn('includ\\ne',puml_label('include/cutlass',6))
        self.assertIn('GemmUniversalAdapter',puml_label('cutlass::gemm::device::GemmUniversalAdapter',12))

    def test_source_string_quotes_are_not_changed_to_character_quotes(self):
        text=puml_label('check("N must be a power of 2")')
        self.assertIn('<U+0022>N must be a power of 2<U+0022>',text)
        self.assertNotIn("'N must",text)

    def test_line_alias_is_supported_for_call_evidence(self):
        import json
        scope=json.loads((ROOT/'data/scope.json').read_text())
        edge={'id':'alias','relation':'calls','source_expression':'GemmKernel::to_underlying_arguments(args, workspace)',
              'evidence':[{'path':'include/cutlass/gemm/device/gemm_universal_adapter.h','line':328}]}
        issues=[];validate_evidence({'e':edge},scope,issues)
        self.assertFalse(issues);self.assertEqual(edge['callsite']['start_line'],328)

    def test_wrong_authored_expression_is_not_silently_replaced(self):
        _,issues,_=self.callsite_fixture({'path':'include/test.hpp','start_byte':17,'end_byte':21,'source_expression':'g(2)'})
        self.assertEqual(issues[0]['kind'],'authored_callsite_fields_disagree')
        self.assertIn('source_expression',issues[0]['fields'])

    def test_line_only_author_is_still_checked(self):
        import json
        scope=json.loads((ROOT/'data/scope.json').read_text())
        edge={'id':'alias','source':'initialize','relation':'calls','source_expression':'GemmKernel::to_underlying_arguments(args, workspace)',
              'callsite':{'line':328,'caller':'wrong','source_expression':'wrong()'},
              'evidence':[{'path':'include/cutlass/gemm/device/gemm_universal_adapter.h','line':328}]}
        issues=[];validate_evidence({'e':edge},scope,issues)
        self.assertEqual(issues[0]['kind'],'authored_callsite_fields_disagree')
        self.assertEqual(set(issues[0]['fields']),{'caller','source_expression'})

    def test_auxiliary_line_alias_is_normalized_without_source_text(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);path=root/'data/probe file.cpp';path.parent.mkdir();path.write_text('void f() { g(1); }\n')
            edge={'id':'aux','relation':'calls','source_expression':'g(1)','evidence':[{'path':'data/probe file.cpp','line':1}]}
            with patch.object(build_atlas,'ROOT',root):
                issues=[];validate_evidence({'e':edge},{'files':[]},issues)
            self.assertFalse(issues)
            self.assertEqual(edge['callsite']['raw_source_expression'],'g(1)')

    def test_evidence_preserves_newline_and_checks_explicit_bytes(self):
        raw=b'int a;\nint b;\n'
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);path=root/'snapshot/include/test.hpp';path.parent.mkdir(parents=True);path.write_bytes(raw)
            edge={'id':'e','relation':'member_of','evidence':[{'path':'include/test.hpp','start_line':1,'end_line':1,'start_byte':0,'end_byte':7,'quote':'int a;\n'}]}
            with patch.object(build_atlas,'ROOT',root):
                issues=[];validate_evidence({'e':edge},{'files':[{'path':'include/test.hpp'}]},issues)
                self.assertFalse(issues);self.assertEqual(edge['evidence'][0]['source_text'],'int a;\n')
                edge['evidence'][0]['end_byte']=14;issues=[]
                validate_evidence({'e':edge},{'files':[{'path':'include/test.hpp'}]},issues)
                self.assertEqual(issues[0]['kind'],'invalid_evidence_byte_range')

    def test_manual_identity_retires_only_after_exact_clean_canonical_match(self):
        import copy
        span={'path':'include/test.hpp','start_byte':10,'end_byte':30}
        node={'qualified_name':'n::S::f','manual_declaration':{'kind':'method','signature_range':span,'raw_signature':'void f(int value)'}}
        occurrence={'kind':'method','signature_range':span,'raw_signature':'void f(int value)','qualified_name':'n::S::f',
                    'parse_status':'parsed','entity_id':'canonical','declaration_occurrence_id':'occ'}
        good=copy.deepcopy(node);self.assertTrue(retire_resolved_manual_identity(good,[occurrence]))
        self.assertNotIn('manual_declaration',good);self.assertIn('manual_declaration_history',good)
        self.assertEqual(good['identity_correction']['canonical_entity_id'],'canonical')
        for change in [{'scope_review_required':True},{'qualified_name':'S::f'},{'raw_signature':'void f(float value)'},{'parse_status':'scope_closure_mismatch'}]:
            unresolved=copy.deepcopy(node);self.assertFalse(retire_resolved_manual_identity(unresolved,[{**occurrence,**change}]))
            self.assertIn('manual_declaration',unresolved)


if __name__=='__main__':unittest.main()
