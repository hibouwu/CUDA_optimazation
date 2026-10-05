#!/usr/bin/env python3
"""Independent read-only checks of the bounded array source draft.

Does not import author_draft, global extractor, or global data by default.
--extractor-fixture adds a labelled, single-file current-extractor comparison;
it is not represented as a read of the canonical ledger.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from tree_sitter import Language, Parser
import tree_sitter_cpp

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PATH='include/cute/container/array.hpp'
EXPECTED_SHA='a5a6c8357cf4311bed395db7dfa8fcc5c28b3d295fdfec9ad2b00b75253ad8ee'
AST_KINDS={'function_definition','struct_specifier','alias_declaration','field_declaration','namespace_definition',
 'template_declaration','call_expression','subscript_expression','type_identifier','primitive_type','namespace_identifier',
 'using_declaration','declaration','for_statement','for_range_loop','if_statement','static_assert_declaration','return_statement',
 'binary_expression','assignment_expression','update_expression','pointer_expression','initializer_list','preproc_include','preproc_call'}
DECL_KINDS={'function_definition','struct_specifier','alias_declaration','field_declaration','namespace_definition'}

def walk(n):
    yield n
    for c in n.named_children:yield from walk(c)
def ancestors(n):
    while n.parent:n=n.parent;yield n
def raw(path):
    assert path.startswith('include/') and '..'not in path.split('/')
    return (ROOT/'snapshot'/path).read_bytes()
def checked_span(s,field=None):
    r=raw(s['path']);a,b=s['start_byte'],s['end_byte']
    assert 0<=a<b<=len(r),s
    assert r[:a].count(b'\n')+1==s['start_line'],s
    assert r[:b-1].count(b'\n')+1==s['end_line'],s
    text=r[a:b].decode()
    if field:assert text==s[field],s
    return text
def signature_key(s):
    r=s['signature_range']
    return(s['path'],s['kind'],s['qualified_name'],r['start_byte'],r['end_byte'])

def verify(data,extractor_fixture=False):
    r=raw(PATH)
    assert hashlib.sha256(r).hexdigest()==EXPECTED_SHA
    assert data['scope']['paths']==[PATH]
    assert data['scope']['files']==[{'path':PATH,'sha256':EXPECTED_SHA,'bytes':len(r),'line_count':476}]
    assert not data['global_completion_claimed']and not data['runtime_execution_claimed']and not data['coverage']['api_coverage_passed']
    assert data['provenance']['global_ledger_read']is False
    tree=Parser(Language(tree_sitter_cpp.language())).parse(re.sub(rb'\bCUTE_HOST_DEVICE\b',lambda m:b' '*len(m[0]),r))
    assert not tree.root_node.has_error
    syntax=list(walk(tree.root_node))
    nodes={n['id']:n for n in data['nodes']};edges={e['id']:e for e in data['edges']}
    assert len(nodes)==len(data['nodes'])and len(edges)==len(data['edges'])
    for n in nodes.values():
        if n.get('path')==PATH and n['kind']in('api','type','resource','namespace'):assert n.get('source_selectors'),n['id']
        for key in('source_range','source_declaration','body_range'):
            if key in n:checked_span(n[key],'raw')
        if 'signature_range'in n:assert checked_span(n['signature_range']).rstrip()==n['signature']
        if n.get('source_selector'):
            ss=n['source_selector'];assert ss['qualified_name']==n['qualified_name']
            assert checked_span(ss['signature_range']).rstrip()==n['signature']
        for ss in n.get('source_selectors',[]):
            assert hashlib.sha256(checked_span(ss['signature_range']).rstrip().encode()).hexdigest()==ss['signature_sha256']
            assert ss['qualified_name']==n['qualified_name']
        if n.get('source_selectors'):
            assert n['source_selector']in n['source_selectors']
            assert Counter(signature_key(s)for s in n['source_selectors'])==Counter(signature_key(o['source_selector'])for o in n['source_occurrences'])
            assert len(set(signature_key(s)for s in n['source_selectors']))==len(n['source_selectors'])
            assert n['availability']=={'operator':'any_of','occurrence_condition_groups':[o['preprocessor_conditions']for o in n['source_occurrences']],
                                      'meaning':'at_least_one_physical_declaration_active'}
            if len(n['source_occurrences'])>1:assert'preprocessor_conditions'not in n
    for e in edges.values():
        assert e['source']in nodes and e['target']in nodes,e['id']
        assert e['evidence'],e['id']
        for ev in e['evidence']:
            assert checked_span(ev,'quote')==ev['quote']
        assert any(ev['quote']==e['source_expression']for ev in e['evidence']),e['id']
        if e['relation']=='calls':
            c=e['callsite'];assert checked_span(c,'raw_source_expression')==c['source_expression']==e['source_expression']
            assert c['caller']==e['source']
            assert e['evaluation']=='potentially_evaluated'
    view_edges=set()
    for v in data['views']:
        assert len(v['node_ids'])==len(set(v['node_ids']))and len(v['edge_ids'])==len(set(v['edge_ids']))
        assert set(v['node_ids'])<=nodes.keys()and set(v['edge_ids'])<=edges.keys()
        for eid in v['edge_ids']:assert {edges[eid]['source'],edges[eid]['target']}<=set(v['node_ids'])
        view_edges.update(v['edge_ids'])
    assert view_edges==edges.keys()
    obligations=data['coverage']['obligations'];recorded=set()
    for o in obligations:
        s=o['source_range'];checked_span(s,'raw')
        assert s['path']==PATH and set(o['graph_refs'])<=(nodes.keys()|edges.keys())
        if o['syntax_kind']in AST_KINDS:recorded.add((s['start_byte'],s['end_byte'],o['syntax_kind']))
    actual={(n.start_byte,n.end_byte,n.type)for n in syntax if n.type in AST_KINDS and not(n.type=='declaration'and not any(a.type=='function_definition'for a in ancestors(n)))}
    assert actual==recorded,{'missing':sorted(actual-recorded),'invented':sorted(recorded-actual)}
    assert Counter(o['category']for o in obligations)==data['coverage']['counts']
    assert not data['coverage']['unclassified_source_obligations']
    # Removing a reference edge together with every graph/view reference must
    # still fail: the independent syntax interval itself requires a relation.
    for ast in syntax:
        if ast.type not in('type_identifier','primitive_type'):continue
        parent=ast.parent
        is_decl=(parent.type in('type_parameter_declaration','variadic_type_parameter_declaration')or
                 parent.type in('alias_declaration','struct_specifier')and parent.child_by_field_name('name')==ast or
                 parent.type=='template_type'and parent.child_by_field_name('name')==ast and parent.parent.type=='struct_specifier'and parent.parent.child_by_field_name('name')==parent)
        if is_decl:continue
        matches=[e for e in edges.values()if e['relation']in('type_uses','template_binds')and e['evidence'][0]['path']==PATH and
                 (e['evidence'][0]['start_byte'],e['evidence'][0]['end_byte'])==(ast.start_byte,ast.end_byte)]
        assert matches,('unmapped type/reference token',ast.start_point,r[ast.start_byte:ast.end_byte])
    physical=data['coverage']['physical_declarations'];physical_by_id={p['id']:p for p in physical}
    assert len(physical_by_id)==len(physical)==87
    expected_decl={(n.start_byte,n.end_byte,n.type)for n in syntax if n.type in DECL_KINDS}
    declared={(p['syntax_range']['start_byte'],p['syntax_range']['end_byte'],p['syntax_kind'])for p in physical}
    assert expected_decl==declared
    assert Counter(p['syntax_kind']for p in physical)=={'function_definition':52,'struct_specifier':8,'alias_declaration':22,'field_declaration':1,'namespace_definition':4}
    for p in physical:
        checked_span(p['syntax_range'],'raw');checked_span(p['source_declaration'],'raw')
        assert checked_span(p['signature_range']).rstrip()==p['raw_signature']
    binds=data['coverage']['conditional_declaration_bindings']
    assert len(binds)==92 and len({b['id']for b in binds})==92
    assert set(b['physical_id']for b in binds)==physical_by_id.keys()
    expected_membership=Counter((b['node_id'],b['source_declaration']['start_byte'],b['source_declaration']['end_byte'])for b in binds if b['source_selector']['kind']!='namespace')
    actual_membership=Counter((e['source'],e['evidence'][0]['start_byte'],e['evidence'][0]['end_byte'])for e in edges.values()if e['relation']=='member_of'and nodes[e['source']].get('entity_kind')!='namespace')
    assert expected_membership==actual_membership
    for b in binds:
        if b['source_selector']['kind']!='alias':continue
        decl=physical_by_id[b['physical_id']]['syntax_range'];ast=next(a for a in syntax if a.type=='alias_declaration'and a.start_byte==decl['start_byte'])
        tp=ast.child_by_field_name('type')
        assert len([e for e in edges.values()if e['relation']=='aliases'and e['source']==b['node_id']and
                    (e['evidence'][0]['start_byte'],e['evidence'][0]['end_byte'])==(tp.start_byte,tp.end_byte)])==1
    # Exactly one physical namespace occurrence produces three owner components;
    # two tuple types and the type alias each produce std/cuda::std variants.
    for p in physical:
        line=p['syntax_range']['start_line'];bb=[b for b in binds if b['physical_id']==p['id']]
        assert len(bb)==(3 if line==430 else 2 if line in(434,439,441)else 1),(line,len(bb))
        for b in bb:
            assert b['node_id']in nodes
            assert b['source_selector']in nodes[b['node_id']]['source_selectors']
            assert b['source_selector']['signature_range']==p['signature_range']
        if line in (430,434,439,441):
            names={b['source_selector']['qualified_name']for b in bb}
            if line==430:assert names=={'std','cuda','cuda::std'}
            else:assert len({n.removeprefix('cuda::')for n in names})==1 and any(n.startswith('cuda::std::')for n in names)
            for b in bb:
                cond=b['preprocessor_conditions'];assert len(cond)==1
                assert cond[0]['directive_path']=='include/cute/config.hpp'
                cuda=b['source_selector']['qualified_name'].startswith('cuda')
                assert cond[0]['expression']==('defined(__CUDACC_RTC__)'if cuda else'!defined(__CUDACC_RTC__)')
        if line in (457,460):
            assert {c['directive_line']for c in bb[0]['preprocessor_conditions']}=={446,454,455}
        if line in (447,465,470,472):assert bb[0]['preprocessor_conditions'][0]['directive_line']==446
    expected_join={
        'array.namespace.cute':{38,401},'array.namespace.std':{430,447},
        'array.type.std.tuple_size':{433,464},'array.type.std.tuple_element':{438,469},
        'array.alias.std.tuple_element.type':{441,472}}
    for ident,ls in expected_join.items():
        n=nodes[ident];assert {s['signature_range']['start_line']for s in n['source_selectors']}==ls
    # Independent review regressions: entity availability is an OR of physical
    # declaration conditions, and preprocessor expressions/includes are not C++
    # runtime calls or cute namespace operations.
    expected_includes={396:{395},398:{397},452:{446,450}}
    for line,directives in expected_includes.items():
        ob=next(o for o in obligations if o['category']=='include'and o['source_range']['start_line']==line)
        assert {c['directive_line']for c in ob['preprocessor_conditions']}==directives
    preproc=[e for e in edges.values()if e['relation']=='evaluates'and e['evidence'][0]['start_line']==450]
    assert len(preproc)==1 and preproc[0]['evaluation']=='preprocessing'
    assert preproc[0]['source']=='array.namespace.std'
    assert 'defined(CUTE_STL_NAMESPACE_IS_CUDA_STD)'in preproc[0]['condition']
    for line in (433,438,459,464,469):
        refs=[e for e in edges.values()if e['relation']=='type_uses'and e['source_expression']=='size_t'and e['evidence'][0]['start_line']==line]
        assert refs and all(e['source']!='array.namespace.cute'for e in refs)
        assert all(e['source_occurrence_condition_groups']for e in refs)
    # Written parameter denominator from the fresh physical tree, not node list.
    param_set={(n.start_byte,n.end_byte,'template_parameter'if n.parent.type=='template_parameter_list'else'function_parameter')
        for n in syntax if n.parent and(n.parent.type=='template_parameter_list'and n.type in('parameter_declaration','type_parameter_declaration','variadic_type_parameter_declaration')
            or n.parent.type=='parameter_list'and n.type=='parameter_declaration')}
    params=data['coverage']['parameter_obligations']
    assert len(params)==54 and len({(p['start_byte'],p['end_byte'],p['category'])for p in params})==54
    assert {(p['start_byte'],p['end_byte'],p['category'])for p in params}==param_set
    assert Counter(p['category']for p in params)=={'function_parameter':19,'template_parameter':35}
    for p in params:checked_span(p,'raw');assert p['default']is None and set(p['owners'])<=nodes.keys()
    # Every explicit ordinary call and every overloaded indexing expression is
    # required exactly once as a calls edge, independently of graph counts.
    call_spans=set();ordinary=[];overloaded=[]
    for ast in syntax:
        text=r[ast.start_byte:ast.end_byte].decode();line=ast.start_point[0]+1
        if ast.type=='call_expression'and text not in('T(0)','CUDA_STD_HEADER(tuple)'):ordinary.append(ast);call_spans.add((ast.start_byte,ast.end_byte))
        elif ast.type=='subscript_expression'and text!='operator[]'and line not in(58,64,215,221):overloaded.append(ast);call_spans.add((ast.start_byte,ast.end_byte))
    calls=[e for e in edges.values()if e['relation']=='calls']
    assert len(ordinary)==31 and len(overloaded)==9 and len(calls)==40
    assert Counter((e['callsite']['start_byte'],e['callsite']['end_byte'])for e in calls)==Counter(call_spans)
    for e in calls:
        c=e['callsite'];caller=nodes[e['source']];br=caller['body_range']
        assert br['start_byte']<=c['start_byte']<c['end_byte']<=br['end_byte']
    def call(line,text):
        selected=[e for e in calls if e['callsite']['start_line']==line and e['source_expression']==text]
        assert len(selected)==1,(line,text,selected)
        return selected[0]
    # Overload-sensitive chains, including nonconst cbegin/cend and N=0 back.
    expected={58:106,64:112,70:106,76:112,83:56,90:62,108:94,114:100,120:106,126:112,
              144:130,150:136,156:160,168:160,182:172,189:160,
              215:261,221:267,227:261,233:267,239:261,245:267}
    for ast in ordinary:
        line=ast.start_point[0]+1;text=r[ast.start_byte:ast.end_byte].decode()
        if line in expected:assert call(line,text)['target']==f'array.api.fn_{expected[line]}'
        if line in(132,138):assert call(line,text)['target']==f'array.api.fn_{(94 if line==132 else 100)if text=="data()"else 160}'
    assert call(425,'cute::move(a[I])')['target']=='array.dependency.move'
    assert call(425,'a[I]')['target']=='array.api.fn_56'
    assert call(417,'a[I]')['target']=='array.api.fn_62'
    assert call(409,'a[I]')['target']=='array.api.fn_56'
    for text in('t_r[k]','t[N - k - 1]'):
        ctx=call(382,text)['control_contexts']
        assert any(c['kind']=='if_constexpr_branch'and c['branch']=='alternative'and c['guard_source']['raw']=='N == 0u'for c in ctx)
        assert any(c['kind']=='ordinary_for_body'for c in ctx)
    for text in('lhs[i]','rhs[i]'):
        ctx=call(344,text)['control_contexts']
        assert any(c['kind']=='ordinary_for_body'and c['does_not_discard_body_at_template_instantiation']for c in ctx)
    for line,text,candidates in[(355,'a.fill(T(0))',{172,327}),(362,'a.fill(value)',{172,327}),(369,'a.swap(b)',{186,335})]:
        e=call(line,text);b=nodes[e['target']];assert e['resolution']=='symbolic'
        assert {c['node_id']for c in b['known_candidates']}=={f'array.api.fn_{l}'for l in candidates}
        assert set(b['missing_bindings'])>={'T','N'}
    swap=nodes[call(190,'swap((*this)[i], other[i])')['target']]
    assert checked_span(swap['using_declaration'],'raw').strip()=='using CUTE_STL_NAMESPACE::swap;'
    assert {c.get('qualified_name')for c in swap['known_candidates']if c.get('qualified_name')}=={'std::swap','cuda::std::swap'}
    assert any(c['kind']=='dependent_lookup'for c in swap['known_candidates'])
    implicit=[e for e in edges.values()if e['relation']=='implicit_calls']
    assert len(implicit)==2 and {e['target']for e in implicit}=={'array.api.fn_106','array.api.fn_130'}
    assert all(e['source_expression_is_not_literal_call']and e['evaluation']=='language_desugaring'for e in implicit)
    for e in implicit:
        assert e['source']=='array.api.fn_172'
        assert e['target']=={'__range.begin()':'array.api.fn_106','__range.end()':'array.api.fn_130'}[e['synthetic_expression']]
    # All52 definitions have HOST_DEVICE + constexpr; no invented storage in0.
    functions=[n for n in nodes.values()if n.get('body_range')and n.get('path')==PATH]
    assert len(functions)==52 and len({n['id']for n in functions})==52
    for n in functions:
        assert n['attributes']==['CUTE_HOST_DEVICE'] and'constexpr'in n['qualifiers']
        assert len([e for e in edges.values()if e['relation']=='has_attribute'and e['source']==n['id']])==1
    assert [n['id']for n in nodes.values()if n.get('entity_kind')=='member']==['array.member.decl_194']
    for line in(327,331,335):assert nodes[f'array.api.fn_{line}']['body_range']['raw']=='{}'
    for line in(249,255,261,267,273,279,285,291,297,303):assert 'return nullptr;'in nodes[f'array.api.fn_{line}']['body_range']['raw']
    assert nodes['array.api.fn_422']['return_type']=='T&&'and nodes['array.api.fn_414']['return_type']=='T const&'
    assert len([n for n in functions if n['name']=='get'])==3
    zero_array_ref=[e for e in edges.values()if e['relation']=='type_uses'and e['source_expression']=='array'and e['evidence'][0]['start_line']==335]
    assert len(zero_array_ref)==1 and zero_array_ref[0]['target']=='array.type.decl_199'
    macro188=[e for e in edges.values()if e['relation']=='macro_uses'and e['evidence'][0]['start_line']==188]
    assert len(macro188)==1 and macro188[0]['source']=='array.api.fn_186'
    # No own macro definitions are present; imported using remains a distinct
    # dependency concept until root repairs its canonical QName.
    assert not any(n.type in('preproc_def','preproc_function_def')for n in syntax)
    imp=nodes['array.dependency.remove_cv_import'];assert imp['kind']=='binding'and'source_selector'not in imp
    assert imp['lookup_name']=='cute::remove_cv_t'and imp['target_expression']=='CUTE_STL_NAMESPACE::remove_cv_t'
    for e in edges.values():
        if e['relation']=='inherits':
            line=e['evidence'][0]['start_line'];target=nodes[e['target']]
            assert line in(435,466)
            assert target['qualified_name'].startswith('cuda::std::'if line==466 or 'cuda_std'in e['source']else'std::')
    for line,target in((465,'array.type.decl_457'),(470,'array.type.decl_460')):
        choices=[e for e in edges.values()if e['relation']=='specializes'and e['evidence'][0]['start_line']==line]
        assert len(choices)==2
        local=next(e for e in choices if e['target']==target)
        assert '!(__CUDACC_VER_MAJOR__ >= 13)'in local['condition']and'defined(__CUDACC_RTC__)'in local['condition']
        boundary=next(e for e in choices if nodes[e['target']]['kind']=='external')
        assert '(__CUDACC_VER_MAJOR__ >= 13) || !defined(__CUDACC_RTC__)'in boundary['condition']
    assert set(n['id']for n in functions)<={x for c in data['contracts']for x in c['participants']}
    oracle=json.loads((HERE/'oracle_results.json').read_text())
    assert oracle['source_header_sha256']==EXPECTED_SHA and oracle['compilation_count']==oracle['expectations_met']==14
    assert sum(len(g['variants'])for g in oracle['results'])==14
    result={'source_physical_declarations':87,'bound_declarations':92,'callable_definitions':52,'parameters':54,
            'ordinary_explicit_calls':31,'overloaded_subscript_calls':9,'implicit_range_for_calls':2,
            'nodes':len(nodes),'edges':len(edges),'obligations':len(obligations),'recorded_clang_expectations':14,
            'canonical_ledger_read':False,'runtime_execution':False}
    if extractor_fixture:
        sys.path.insert(0,str(ROOT/'scripts'))
        from extract_declarations import Extractor
        from build_atlas import source_selector_matches
        e=Extractor(data['snapshot_commit']);e.extract(PATH,r)
        assert not e.diagnostics and len(e.occurrences)==92
        mapped=[]
        for b in binds:
            matches=[o for o in e.occurrences if source_selector_matches(b['source_selector'],o)]
            assert len(matches)==1,(b['node_id'],len(matches))
            mapped.append(matches[0]['declaration_occurrence_id'])
        assert len(set(mapped))==92
        for n in nodes.values():
            if n.get('source_selectors'):
                matches=[o for o in e.occurrences if any(source_selector_matches(s,o)for s in n['source_selectors'])]
                assert len({o['entity_id']for o in matches})==1,n['id']
        result['current_single_file_extractor_fixture']={'matches':92,'diagnostics':0,'not_a_global_ledger_read':True}
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extractor-fixture',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(json.loads((HERE/'relations.json').read_text()),args.extractor_fixture),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
