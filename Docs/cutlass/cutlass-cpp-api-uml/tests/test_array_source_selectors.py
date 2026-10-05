"""Select real conditionally named array APIs without short-name conflation."""
import copy
import contextlib
import io
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_atlas
from extract_declarations import Extractor

PATH='include/cute/container/array.hpp'


class ArraySourceSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        extractor=Extractor('8f50b052e1099fb982392a622caab69b97b63128')
        extractor.extract(PATH,(ROOT/'snapshot'/PATH).read_bytes())
        cls.records=extractor.occurrences

    def source(self,qualified,line=433):
        return next(o for o in self.records if o['qualified_name']==qualified and o['signature_range']['start_line']==line)

    def node(self,source,manual=False):
        selector={k:source[k]for k in ('path','kind','qualified_name','name','signature_range')}
        selector['signature_sha256']=hashlib.sha256(source['raw_signature'].encode()).hexdigest()
        node={'id':'selected','kind':'type','name':source['name'],'qualified_name':source['qualified_name'],
              'path':PATH,'line':source['signature_range']['start_line'],'source_selector':copy.deepcopy(selector)}
        if manual:node['manual_declaration']={k:source[k]for k in ('name','kind','signature_range','raw_signature')}
        return node

    def enrich(self,node):
        issues=[]
        with patch.object(build_atlas,'top_items',return_value=[('occurrences',o,True)for o in self.records]),contextlib.redirect_stdout(io.StringIO()):
            build_atlas.enrich_nodes({'selected':node},{'files':[{'path':PATH}]},issues)
        return node,issues

    def test_same_physical_source_selects_std_and_cuda_std_independently(self):
        selected=[]
        for qualified in ('std::tuple_size<cute::array<T,N>>','cuda::std::tuple_size<cute::array<T,N>>'):
            source=self.source(qualified);node,issues=self.enrich(self.node(source))
            self.assertFalse(issues);self.assertEqual(node['full_name'],qualified)
            self.assertEqual(node['declaration_occurrence_id'],source['declaration_occurrence_id'])
            self.assertTrue(node['conditions']);selected.append(node['entity_id'])
        self.assertEqual(len(set(selected)),2)

    def test_same_entity_bridge_has_distinct_source_occurrences(self):
        qualified='std::tuple_size<cute::array<T,N>>'
        a,errors_a=self.enrich(self.node(self.source(qualified,433)))
        b,errors_b=self.enrich(self.node(self.source(qualified,464)))
        self.assertFalse(errors_a+errors_b);self.assertEqual(a['entity_id'],b['entity_id'])
        self.assertNotEqual(a['declaration_occurrence_id'],b['declaration_occurrence_id'])
        self.assertNotEqual(a['conditions'],b['conditions'])

    def test_qualified_name_without_byte_selector_also_disambiguates_namespace(self):
        source=self.source('cuda::std::tuple_size<cute::array<T,N>>');node=self.node(source)
        node.pop('source_selector');node,issues=self.enrich(node)
        self.assertFalse(issues);self.assertEqual(node['entity_id'],source['entity_id'])

    def test_exact_selector_mismatch_never_uses_short_name_or_manual_fallback(self):
        source=self.source('std::tuple_size<cute::array<T,N>>')
        for field,value in [('kind','alias'),('qualified_name','wrong::tuple_size<cute::array<T,N>>'),('signature_sha256','0'*64)]:
            for manual in (False,True):
                node=self.node(source,manual);node['source_selector'][field]=value
                node,issues=self.enrich(node)
                self.assertTrue(issues);self.assertEqual(node['declaration_status'],'explicit_source_selector_unresolved')
                self.assertNotIn('entity_id',node)

    def test_whitespace_is_not_a_different_template_argument(self):
        source=self.source('std::tuple_size<cute::array<T,N>>');selector=self.node(source)['source_selector']
        selector['qualified_name']='std :: tuple_size< cute::array<T, N> >'
        self.assertTrue(build_atlas.source_selector_matches(selector,source))
        selector['qualified_name']='std :: tuple_size< cute::array<T, 0> >'
        self.assertFalse(build_atlas.source_selector_matches(selector,source))

    def test_scope_review_cannot_retire_manual_identity(self):
        source=copy.deepcopy(self.source('std::tuple_size<cute::array<T,N>>'));node=self.node(source,True)
        source['scope_review_required']=True
        self.assertFalse(build_atlas.retire_resolved_manual_identity(node,[source]))
        self.assertIn('manual_declaration',node)

    def test_explicit_empty_or_invalid_selector_never_means_no_selector(self):
        source=self.source('std::tuple_size<cute::array<T,N>>')
        for invalid in ({},None,[],False,''):
            for manual in (False,True):
                node=self.node(source,manual);node['source_selector']=invalid
                node,issues=self.enrich(node)
                self.assertTrue(issues);self.assertNotIn('entity_id',node)
                self.assertEqual(node['declaration_status'],'explicit_source_selector_unresolved')
                if manual:self.assertIn('manual_declaration',node)

    def test_plural_selectors_preserve_source_conditions_for_same_entity(self):
        qualified='std::tuple_size<cute::array<T,N>>'
        node=self.node(self.source(qualified,433))
        node['source_selectors']=[node['source_selector'],self.node(self.source(qualified,464))['source_selector']]
        node,issues=self.enrich(node);self.assertFalse(issues)
        occurrences=node['source_declaration_occurrences'];self.assertEqual(len(occurrences),2)
        self.assertEqual(len({o['entity_id']for o in occurrences}),1)
        self.assertNotEqual(occurrences[0]['preprocessor_conditions'],occurrences[1]['preprocessor_conditions'])
        self.assertNotIn('conditions',node)
        self.assertEqual(node['declaration_availability']['operator'],'any_of')
        self.assertEqual(node['declaration_availability']['occurrence_condition_groups'],[o['preprocessor_conditions']for o in occurrences])

    def test_plural_selectors_never_hide_missing_duplicates_or_different_entities(self):
        original=self.node(self.source('std::tuple_size<cute::array<T,N>>'))
        other=self.node(self.source('cuda::std::tuple_size<cute::array<T,N>>'))['source_selector']
        for invalid in ([],{},None,[{}],[original['source_selector']]*2,[original['source_selector'],other]):
            node=copy.deepcopy(original);node['source_selectors']=invalid
            node,issues=self.enrich(node);self.assertTrue(issues)
            self.assertNotIn('entity_id',node)

    def test_invalid_explicit_selector_is_reported_for_non_type_nodes_too(self):
        for kind in ('namespace','binding','resource'):
            node=self.node(self.source('std::tuple_size<cute::array<T,N>>'));node['kind']=kind;node['source_selector']={}
            node,issues=self.enrich(node);self.assertTrue(issues);self.assertNotIn('entity_id',node)

    def test_relation_source_may_include_exact_class_terminating_semicolon(self):
        source=self.source('std::tuple_size<cute::array<T,N>>',433);node=self.node(source)
        second=self.source('std::tuple_size<cute::array<T,N>>',464)
        node['source_selectors']=[node['source_selector'],self.node(second)['source_selector']]
        node,issues=self.enrich(node);self.assertFalse(issues)
        span=next(o['declaration_range']for o in node['source_declaration_occurrences']if o['start_line']==464)
        edge={'source':node['id'],'target':'std','relation':'member_of',
              'evidence':[{'path':PATH,'start_byte':span['start_byte'],'end_byte':span['end_byte']+1}]}
        selected=build_atlas.declaration_for_edge(node,edge)
        self.assertEqual(selected['declaration_occurrence_id'],second['declaration_occurrence_id'])
        self.assertEqual(selected['line'],464)
        edge['evidence'][0]['end_byte']=len((ROOT/'snapshot'/PATH).read_bytes())
        self.assertIs(build_atlas.declaration_for_edge(node,edge),node)


if __name__=='__main__':unittest.main()
