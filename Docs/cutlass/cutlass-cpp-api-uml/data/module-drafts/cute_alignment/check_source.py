#!/usr/bin/env python3
"""Independent read-only two-header accounting checks; no author/core import."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from tree_sitter import Language,Parser
import tree_sitter_cpp

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
A='include/cute/container/alignment.hpp';R='include/cute/container/array_aligned.hpp'
EXPECTED={A,R}
CATEGORIES={'template_declaration':'api_template','function_definition':'function_definition','struct_specifier':'type_declaration',
    'namespace_definition':'namespace','return_statement':'return_statement','static_assert_declaration':'constraint',
    'binary_expression':'operator_expression','call_expression':'call_expression','type_identifier':'type_spelling_token',
    'primitive_type':'type_spelling_token','preproc_include':'include','preproc_call':'pragma','preproc_function_def':'macro_definition'}

def raw(path):return (ROOT/'snapshot'/path).read_bytes()
def walk(node):
    yield node
    for child in node.named_children:yield from walk(child)
def check_range(s,field=None):
    b=raw(s['path']);a,z=s['start_byte'],s['end_byte'];assert 0<=a<z<=len(b),s
    assert b[:a].count(b'\n')+1==s['start_line'];assert b[:z-1].count(b'\n')+1==s['end_line']
    spelling=b[a:z].decode()
    if field:assert spelling==s[field],(s,spelling)
    return spelling

def verify(data):
    assert set(data['scope']['paths'])==EXPECTED
    assert not data['coverage']['api_coverage_passed'] and not data['global_completion_claimed'] and not data['runtime_execution_claimed']
    nodes={n['id']:n for n in data['nodes']};edges={e['id']:e for e in data['edges']}
    assert len(nodes)==len(data['nodes']) and len(edges)==len(data['edges'])
    assert all(not n.get('entity_id') and not n.get('declaration_occurrence_id') for n in nodes.values()),'Must not reuse pre-adaptation identities'
    for f in data['scope']['files']:
        b=raw(f['path']);assert hashlib.sha256(b).hexdigest()==f['sha256'];assert len(b)==f['bytes'];assert len(b.splitlines())==f['line_count']
    for n in nodes.values():
        if 'signature_range' in n:
            assert check_range(n['signature_range']).rstrip()==n['signature']
            assert n['source_selector']['signature_range']==n['signature_range']
            assert hashlib.sha256(n['signature'].encode()).hexdigest()==n['source_selector']['signature_sha256']
            m=n['manual_declaration'];assert check_range(m['signature_range']).rstrip()==m['raw_signature']
        if 'source_range' in n:check_range(n['source_range'],'raw')
        if 'source_declaration' in n:check_range(n['source_declaration'],'raw')
    for e in edges.values():
        assert e['source']in nodes and e['target']in nodes and e['evidence']
        for ev in e['evidence']:assert check_range(ev,'quote')
        if e['relation']=='calls':
            assert e['source_expression']=='has_single_bit(N)'
            assert e['evaluation']=='constant_evaluation_static_assert'
            assert e['target']=='alignment.dep.has_single_bit'
            assert e['template_bindings']=={'T':'int'}
            c=e['callsite'];assert check_range(c,'raw_source_expression')==e['source_expression'];assert c['caller']==e['source']
    assert sum(e['relation']=='calls' for e in edges.values())==1
    for e in edges.values():
        if e['relation']=='type_uses' and e['target']=='alignment.external.size_t':assert e['source_expression']=='size_t'
    assert not any(e['relation']=='calls' and ('reinterpret_cast' in e['source_expression'] or 'CUTE_ALIGNAS' in e['source_expression'])for e in edges.values())
    for v in data['views']:
        assert set(v['node_ids'])<=nodes.keys() and set(v['edge_ids'])<=edges.keys()
        for e in v['edge_ids']:assert {edges[e]['source'],edges[e]['target']}<=set(v['node_ids'])
    assert {e['id'] for e in data['edges']}=={e for v in data['views'] for e in v['edge_ids']}
    obligations=data['coverage']['obligations'];assert len({o['id'] for o in obligations})==len(obligations)
    assert Counter(o['category']for o in obligations)==data['coverage']['counts']
    assert not data['coverage']['unclassified_source_obligations']
    recorded=set()
    for o in obligations:
        check_range(o['source_range'],'raw');assert set(o['graph_refs'])<=nodes.keys()|edges.keys()
        if o['syntax_kind']in CATEGORIES:recorded.add((o['source_range']['path'],o['source_range']['start_byte'],o['source_range']['end_byte'],o['syntax_kind']))
        if o['category']=='include':assert o['target_path']=='include/'+re.search(r'<([^>]+)>',o['source_range']['raw']).group(1)
        if o['category']=='named_cast':assert o['source_range']['raw']=='reinterpret_cast<uintptr_t>(ptr)'
        if o['category']=='type_spelling_token' and o['source_range']['raw'] in ('size_t','uintptr_t'):
            assert o['semantic_role']=='external_integer_typedef_reference'
        if o['category']=='type_spelling_token' and o['source_range']['raw'] in ('N','Alignment'):
            assert o['semantic_role']=='non_type_template_parameter_reference'
    actual=set();params=set();declarations=[];macro_calls=[];directives=[];namespace_signatures=set()
    for path in sorted(EXPECTED):
        b=raw(path);mask=bytearray(b)
        for m in re.finditer(rb'\bCUTE_HOST_DEVICE\b',b):mask[m.start():m.end()]=b' '*len(m[0])
        for m in re.finditer(rb'\bCUTE_ALIGNAS\b',b):
            if b[b.rfind(b'\n',0,m.start())+1:m.start()].lstrip().startswith(b'#'):continue
            # Alignas keyword followed by same-width whitespace preserves argument positions.
            mask[m.start():m.end()]=b'alignas'+b' '*(len(m[0])-7)
        tree=Parser(Language(tree_sitter_cpp.language())).parse(bytes(mask));assert not tree.root_node.has_error
        for n in walk(tree.root_node):
            if n.type in CATEGORIES:actual.add((path,n.start_byte,n.end_byte,n.type))
            if n.type=='namespace_definition':namespace_signatures.add((path,n.start_byte,n.child_by_field_name('body').start_byte))
            if n.type=='template_declaration':
                target=next(c for c in n.named_children if c.type in ('function_definition','struct_specifier'))
                body=target.child_by_field_name('body');declarations.append((path,n.start_byte,body.start_byte))
                for p in n.child_by_field_name('parameters').named_children:params.add((path,p.start_byte,p.end_byte,'template_parameter'))
                if target.type=='function_definition':
                    for p in target.child_by_field_name('declarator').child_by_field_name('parameters').named_children:params.add((path,p.start_byte,p.end_byte,'function_parameter'))
            if n.type=='preproc_function_def':
                for p in n.child_by_field_name('parameters').named_children:params.add((path,p.start_byte,p.end_byte,'macro_parameter'))
        for m in re.finditer(rb'CUTE_ALIGNAS\([^\n)]*\)',b):
            if not b[b.rfind(b'\n',0,m.start())+1:m.start()].lstrip().startswith(b'#'):macro_calls.append((path,m.start(),m.end()))
        for m in re.finditer(rb'^#(?:if defined\(__CUDACC__\)|else|endif)(?!\w)[^\n]*\n?',b,re.M):directives.append((path,m.start(),m.end()))
    assert actual==recorded,{'missing':sorted(actual-recorded),'invented':sorted(recorded-actual)}
    observed_parameters={(p['path'],p['start_byte'],p['end_byte'],p['category']) for p in data['coverage']['parameter_obligations']}
    assert params==observed_parameters
    assert Counter(p['category']for p in data['coverage']['parameter_obligations'])=={'template_parameter':15,'function_parameter':1,'macro_parameter':2}
    for p in data['coverage']['parameter_obligations']:check_range(p,'raw')
    selectors={(n['path'],n['signature_range']['start_byte'],n['signature_range']['end_byte']) for n in nodes.values()
               if n.get('source_selector') and n['path']in EXPECTED and n['source_selector']['kind']!='macro_definition'}
    assert selectors==set(declarations),('API selector mismatch',selectors^set(declarations))
    assert len(declarations)==12
    ns=nodes['alignment.namespace.cute']
    assert {(s['path'],s['signature_range']['start_byte'],s['signature_range']['end_byte'])for s in ns['source_selectors']}==namespace_signatures
    for s in ns['source_occurrences']:check_range(s,'raw')
    # The dependency's opening brace is on its name line; line_span alone must
    # not swallow it into the API signature. Scope-local function/array checks
    # are likewise byte-exact, including the newline before their next-line body.
    for ident,path,first,name_line in [('alignment.api.is_byte_aligned',A,42,45),
        ('alignment.dep.array','include/cute/container/array.hpp',41,42),
        ('alignment.dep.has_single_bit','include/cute/numeric/math.hpp',126,129)]:
        data_bytes=raw(path);lines=data_bytes.splitlines(keepends=True)
        begin=sum(map(len,lines[:first-1]));opening=data_bytes.index(b'{',sum(map(len,lines[:name_line-1])))
        assert nodes[ident]['signature_range']['start_byte']==begin
        assert nodes[ident]['signature_range']['end_byte']==opening
    attrs=[n for n in nodes.values()if n.get('semantic_class')=='requested_alignment_attribute']
    assert len(attrs)==10
    assert {(n['path'],n['source_range']['start_byte'],n['source_range']['end_byte'])for n in attrs}==set(macro_calls)
    assert {(o['source_range']['path'],o['source_range']['start_byte'],o['source_range']['end_byte'])for o in obligations if o['category']=='preprocessor_branch'}==set(directives)
    assert len(directives)==3
    for n in attrs:
        assert n['alignment_unit']=='bytes' and n['owner_ref'] in nodes
        assert check_range(n['expression_range'],'raw')==n['alignment_expression']
        assert len(n['expansion_variants'])==2
        variants={v['condition']:v for v in n['expansion_variants']}
        assert variants['defined(__CUDACC__)']['expanded_spelling']=='__align__('+n['alignment_expression']+')'
        assert variants['!defined(__CUDACC__)']['expanded_spelling']=='alignas('+n['alignment_expression']+')'
        for v in variants.values():assert nodes[v['definition_ref']]['source_selector']['kind']=='macro_definition'
        attached=[e for e in edges.values()if e['relation']=='has_attribute' and e['target']==n['id']]
        assert len(attached)==1 and attached[0]['source']==n['owner_ref']
    special=[n for n in nodes.values()if n.get('specialization')=='partial']
    assert {n['specialization_arguments']['Alignment']for n in special}=={'1','2','4','8','16','32','64','128','256'}
    assert len(special)==9 and sum(e['relation']=='specializes'for e in edges.values())==9
    assert all(n['bases']==[] and n['template_parameters']==[{'name':'Child','kind':'type','default':None,'raw':'class Child'}]for n in special)
    assert nodes['alignment.type.aligned_primary']['bases']==[]
    assert nodes['alignment.type.aligned_primary']['template_parameters'][1]['default']=='void'
    assert nodes['alignment.type.array_aligned']['template_parameters'][2]['default']=='16'
    inherited=[e for e in edges.values()if e['relation']=='inherits'];assert len(inherited)==1 and inherited[0]['source_expression']=='cute::array<T,N>'
    assert not any(e['relation']=='has_attribute' and e['source']=='alignment.type.aligned_primary' for e in edges.values())
    macro_defs=[n for n in nodes.values()if n.get('source_selector',{}).get('kind')=='macro_definition']
    assert len(macro_defs)==2 and all(n['qualified_name']=='CUTE_ALIGNAS'for n in macro_defs)
    return {'source_accounting_check_passed':True,'physical_declaration_selectors':16,'alignment_uses':10,
            'obligations':len(obligations),'parameters':len(params),'api_coverage_passed':False,'reason':'Corrected canonical identity enrichment and independent semantic review remain required.'}

def clang_oracle():
    compiler=shutil.which('clang++')
    if not compiler:return {'status':'not_run','reason':'clang++ unavailable'}
    common=[compiler,'-std=c++17','-I',str(ROOT/'snapshot/include'),'-I','/usr/local/cuda-13.0/targets/x86_64-linux/include',
            '-I','/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl','-fsyntax-only','-x','c++','-']
    prefix='#include "cute/container/array_aligned.hpp"\n'
    positive=prefix+'''struct Child { char x[4096]; };
static_assert(alignof(cute::aligned_struct<128,Child>)==128);
static_assert(!__is_base_of(Child,cute::aligned_struct<128,Child>));
static_assert(alignof(cute::aligned_struct<512>)==1);
static_assert(alignof(cute::array_aligned<char,1>)==16);
static_assert(__is_base_of(cute::array<float,3>,cute::array_aligned<float,3,64>));
bool check(void const* p) { return cute::is_byte_aligned<8>(p); }
'''
    negative=prefix+'bool check(void const* p) { return cute::is_byte_aligned<3>(p); }\n'
    bad_constexpr=prefix+'static_assert(cute::is_byte_aligned<8>(nullptr));\n'
    result=[]
    for label,text,expected in [('host_source_contract',positive,0),('non_power_two_rejected',negative,1),('pointer_cast_not_cpp17_constexpr',bad_constexpr,1)]:
        r=subprocess.run(common,input=text,text=True,capture_output=True)
        assert (r.returncode==0)==(expected==0),(label,r.stderr)
        result.append({'label':label,'returncode':r.returncode,'source':text,'stderr':r.stderr})
    return {'status':'host_oracles_pass','commands_common':common,'results':result,'cuda_branch_compiled':False,'gpu_executed':False}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--clang',action='store_true');args=p.parse_args()
    d=json.loads((HERE/'relations.json').read_text());result=verify(d)
    if args.clang:result['compiler_oracle']=clang_oracle()
    print(json.dumps(result,ensure_ascii=False,indent=2))
