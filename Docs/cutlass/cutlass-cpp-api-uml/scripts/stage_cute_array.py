#!/usr/bin/env python3
"""Join the complete array header inventory without collapsing branch owners."""
from collections import Counter
import copy
import importlib.util
import json
import re
from pathlib import Path
from functools import lru_cache
from tree_sitter import Language,Parser
import tree_sitter_cpp

from build_atlas import ROOT,enrich_nodes,dump,source_selector_matches
from reconcile_candidates import file_sha,token_spelling,condition_form,condition_line

PATH='include/cute/container/array.hpp'


def condition_key(conditions):
    return tuple(sorted((c.get('directive_path',c.get('directive_span',{}).get('path')),
                         condition_line(c),repr(condition_form(c['expression'])))for c in conditions))


def parameter_key(parameter,template=False,index=None):
    fields=(parameter.get('name'),token_spelling(parameter.get('raw','')),parameter.get('default'),
            parameter.get('path'),parameter.get('start_byte'),parameter.get('end_byte'),parameter.get('index',index))
    return fields if template else fields+(token_spelling(parameter.get('type','')),)


@lru_cache(maxsize=128)
def type_key(text):
    """Normalize only grammar-level qualifier order, never resolve type aliases."""
    if not text:return None
    raw=('using ReturnType = '+text+';').encode()
    tree=Parser(Language(tree_sitter_cpp.language())).parse(raw)
    if tree.root_node.has_error:raise ValueError('Type spelling needs explicit review: '+text)
    node=tree.root_node.named_children[0].child_by_field_name('type')
    def key(n):
        children=[c for c in n.children if c.type!='comment']
        if not children:return (n.type,token_spelling(raw[n.start_byte:n.end_byte].decode()))
        qualifiers=tuple(sorted(key(c)for c in children if c.type=='type_qualifier'))
        others=tuple(key(c)for c in children if c.type!='type_qualifier')
        return (n.type,qualifiers,others)
    return key(node)


@lru_cache(maxsize=1)
def source_nodes():
    raw=(ROOT/'snapshot'/PATH).read_bytes()
    parser_source=re.sub(rb'\bCUTE_HOST_DEVICE\b',lambda m:b' '*len(m[0]),raw)
    tree=Parser(Language(tree_sitter_cpp.language())).parse(parser_source)
    if tree.root_node.has_error:raise ValueError('Array source syntax changed; split metadata needs source review')
    nodes={};todo=[tree.root_node]
    while todo:
        node=todo.pop();nodes[(node.start_byte,node.end_byte,node.type)]=node;todo.extend(node.named_children)
    return raw,tree,nodes


def verify_join(authored,nodes,records):
    canonical=records[PATH];seen=[];mapping=[]
    originals={n['id']:n for n in authored['nodes']}
    physical={p['id']:p for p in authored['coverage']['physical_declarations']}
    raw,tree,syntax=source_nodes()
    for binding in authored['coverage']['conditional_declaration_bindings']:
        selected=[o for o in canonical if source_selector_matches(binding['source_selector'],o)]
        if len(selected)!=1:raise ValueError('Each bound declaration needs one clean canonical occurrence: '+binding['id'])
        occurrence=selected[0];original=originals[binding['node_id']];node=nodes[binding['node_id']]
        origin=physical[binding['physical_id']];span=origin['syntax_range']
        ast=syntax[(span['start_byte'],span['end_byte'],origin['syntax_kind'])]
        spelling=lambda n:raw[n.start_byte:n.end_byte].decode()if n else None
        if origin['syntax_kind']=='function_definition':
            expected_cv=[spelling(c)for c in ast.named_children if c.type=='type_qualifier'and spelling(c)in {'const','volatile','__restrict','__restrict__'}]
            if occurrence.get('return_cv_qualifiers')!=expected_cv:raise ValueError('Return cv metadata differs from source: '+node['id'])
        if origin['syntax_kind']=='field_declaration':
            expected_base=spelling(ast.child_by_field_name('type'))
            expected_declarator=spelling(ast.child_by_field_name('declarator'))
            if occurrence.get('declared_type')!=expected_base or occurrence.get('declarator')!=expected_declarator or occurrence.get('declared_type_role')!='declaration_type_specifier':
                raise ValueError('Split member type metadata differs from source: '+node['id'])
        if node.get('declaration_status')!='linked_to_source_ledger'or node.get('manual_declaration'):
            raise ValueError('Array declaration cannot use a source anchor or manual identity: '+node['id'])
        if node['entity_id']!=occurrence['entity_id']:raise ValueError('Different source entities were merged: '+node['id'])
        if condition_key(binding['preprocessor_conditions'])!=condition_key(occurrence.get('preprocessor_conditions',[])):
            raise ValueError('Bound source conditions differ: '+binding['id'])
        actual_parameters=occurrence.get('parameters',[])
        if [parameter_key(p,index=i)for i,p in enumerate(original.get('parameters',[]))]!=[parameter_key(p,index=i)for i,p in enumerate(actual_parameters)]:
            raise ValueError('Function parameter contract differs: '+node['id'])
        actual_templates=[p for h in occurrence.get('template_parameters',[])for p in h['parameters']]
        declaration=binding['source_declaration']
        enclosing_signatures=[p['signature_range']for p in authored['coverage']['physical_declarations']
            if p['source_declaration']['start_byte']<=declaration['start_byte']and declaration['end_byte']<=p['source_declaration']['end_byte']]
        expected_templates=[p for p in original.get('template_parameters',[])
            if any(s['start_byte']<=p['start_byte']<p['end_byte']<=s['end_byte']for s in enclosing_signatures)]
        if [parameter_key(p,True)for p in expected_templates]!=[parameter_key(p,True)for p in actual_templates]:
            raise ValueError('Template environment differs: '+node['id'])
        for field in ('attributes','qualifiers'):
            if field in original and Counter(original[field])!=Counter(occurrence.get(field,[])):
                raise ValueError('Declaration attributes differ: '+node['id']+':'+field)
        for field in ('return_type','target_type','declared_type','initializer'):
            key=type_key if field=='return_type'else lambda value:token_spelling(value or'')
            actual=occurrence.get('declared_type_spelling')if field=='declared_type'else occurrence.get(field)
            if field in original and key(original[field])!=key(actual):
                raise ValueError('Declaration type or initializer differs: '+node['id']+':'+field)
        if original.get('access_scope')=='class_member'and original.get('access')!=occurrence.get('access'):
            raise ValueError('Class member access differs: '+node['id'])
        if 'body_range'in original:
            body=occurrence.get('body_range',{});expected=original['body_range']
            if any(body.get(k)!=expected.get(k)for k in ('path','start_byte','end_byte')):
                raise ValueError('Callable body range differs: '+node['id'])
        seen.append(occurrence['declaration_occurrence_id'])
        mapping.append({'binding_id':binding['id'],'physical_id':binding['physical_id'],'node_id':node['id'],
            'entity_id':occurrence['entity_id'],'declaration_occurrence_id':occurrence['declaration_occurrence_id'],
            'variant_id':occurrence['variant_id'],'signature_range':occurrence['signature_range']})
    if len(seen)!=92 or len(set(seen))!=92 or Counter(seen)!=Counter(o['declaration_occurrence_id']for o in canonical):
        raise ValueError('Canonical and independently inventoried bound-declaration multisets differ')
    if len({m['physical_id']for m in mapping})!=87:raise ValueError('Physical source denominator differs')
    node_entities={}
    for item in mapping:
        previous=node_entities.setdefault(item['entity_id'],item['node_id'])
        if previous!=item['node_id']:raise ValueError('Distinct independently inventoried API nodes share a canonical entity')
    for original in originals.values():
        if not original.get('source_selectors'):continue
        node=nodes[original['id']];occurrences=node.get('source_declaration_occurrences',[])
        expected={m['declaration_occurrence_id']for m in mapping if m['node_id']==node['id']}
        if {o['declaration_occurrence_id']for o in occurrences}!=expected or len(occurrences)!=len(expected):
            raise ValueError('Node lost or duplicated a declaration occurrence: '+node['id'])
        if 'conditions'in node:raise ValueError('Selected occurrence conditions must not replace entity availability')
        availability=node.get('declaration_availability',{})
        if availability.get('operator')!='any_of'or Counter(condition_key(c)for c in availability.get('occurrence_condition_groups',[]))!=Counter(condition_key(c)for c in original['availability']['occurrence_condition_groups']):
            raise ValueError('Entity source availability differs: '+node['id'])
    return mapping


def main():
    draft=ROOT/'data/module-drafts/cute_array/relations.json'
    checker=ROOT/'data/module-drafts/cute_array/check_source.py'
    spec=importlib.util.spec_from_file_location('array_source_checker',checker)
    review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
    authored=json.loads(draft.read_text());source_check=review.verify(authored)
    tracked=[draft,checker,ROOT/'data/module-drafts/cute_array/oracle_results.json',ROOT/'data/declarations.json',
             Path(__file__),ROOT/'scripts/build_atlas.py',ROOT/'scripts/reconcile_candidates.py',ROOT/'scripts/scan_candidates.py']
    inputs={str(p.relative_to(ROOT)):file_sha(p)for p in tracked}
    data=copy.deepcopy(authored);nodes={n['id']:n for n in data['nodes']};issues=[]
    records=enrich_nodes(nodes,json.loads((ROOT/'data/scope.json').read_text()),issues)
    if issues:raise ValueError(issues)
    mapping=verify_join(authored,nodes,records)
    titles={'storage':'数组的内嵌存储与零长度特化','access':'元素访问、指针与引用寿命',
            'iteration':'迭代器与循环展开','mutation':'fill、clear 与 swap 的不同要求',
            'algorithms':'相等比较与反转','tuple':'get、tuple traits 与标准库桥接'}
    for contract in data['contracts']:
        contract['kind']='source_contract';contract['title']=titles[contract['id'].split('.')[-1]]
    for issue in data['issues']:
        if issue['id']=='array.issue.canonical_join':issue.update(status='closed',
            resolution='All 87 physical declarations / 92 conditional bindings joined to the canonical ledger with parameter, qualifier, source-condition and owner checks; no manual identity.')
    data['canonical_integration']={'inputs':inputs,'source_check':source_check,'binding_map':mapping,
        'physical_declarations':87,'bound_declarations':92,'current_manual_overrides':0,
        'dependency_identity_errors_still_open':['array.issue.remove_cv_using'],
        'actual_browser_acceptance':False,'global_completion_claimed':False}
    data['status']='source_and_canonical_join_checked_dependency_gap_and_use_acceptance_pending'
    for path,expected in inputs.items():
        if file_sha(ROOT/path)!=expected:raise ValueError('Stage input changed: '+path)
    dump(ROOT/'data/modules/cute_array/relations.json',data)
    dump(ROOT/'audits/cute-array-canonical-integration.json',data['canonical_integration'])
    print(json.dumps({'physical_declarations':87,'bound_declarations':92,'manual_overrides':0,
        'nodes':len(nodes),'edges':len(data['edges']),'dependency_identity_gap_retained':True,'global_complete':False}))


if __name__=='__main__':main()
