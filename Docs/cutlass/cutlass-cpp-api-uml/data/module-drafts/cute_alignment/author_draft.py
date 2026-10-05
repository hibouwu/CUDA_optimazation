#!/usr/bin/env python3
"""Two complete-header source-selector draft; never imports the faulty ledger."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from tree_sitter import Language,Parser
import tree_sitter_cpp

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
A='include/cute/container/alignment.hpp'
R='include/cute/container/array_aligned.hpp'
M='include/cute/numeric/math.hpp'
C='include/cute/container/array.hpp'
CFG='include/cute/config.hpp'
SCOPE=(A,R)
LANGUAGE=Language(tree_sitter_cpp.language())
RAW={}

def source(path):
    if path not in RAW:RAW[path]=(ROOT/'snapshot'/path).read_bytes()
    return RAW[path]

def span(path,a,b):
    raw=source(path)
    return {'path':path,'start_byte':a,'end_byte':b,'start_line':raw[:a].count(b'\n')+1,
            'end_line':raw[:max(a,b-1)].count(b'\n')+1,'raw':raw[a:b].decode()}

def line_span(path,first,last=None):
    lines=source(path).splitlines(keepends=True);last=first if last is None else last
    return span(path,sum(map(len,lines[:first-1])),sum(map(len,lines[:last])))

def locate(path,text,line,last=None):
    region=line_span(path,line,last);a=source(path).index(text.encode(),region['start_byte'],region['end_byte'])
    return span(path,a,a+len(text.encode()))

def evidence(s):return {k:v for k,v in s.items() if k!='raw'}|{'quote':s['raw']}

def projected_tree(path):
    raw=source(path);masked=bytearray(raw)
    for m in re.finditer(rb'\bCUTE_HOST_DEVICE\b',raw):masked[m.start():m.end()]=b' '*len(m[0])
    for m in re.finditer(rb'\bCUTE_ALIGNAS\b',raw):
        if raw[raw.rfind(b'\n',0,m.start())+1:m.start()].lstrip().startswith(b'#'):continue
        masked[m.start():m.end()]=b'alignas'+b' '*(len(m[0])-7)
    tree=Parser(LANGUAGE).parse(bytes(masked));assert not tree.root_node.has_error,path
    return tree

def walk(node):
    yield node
    for child in node.named_children:yield from walk(child)

def main():
    nodes=[];edges=[];obligations=[];parameters=[];by_id={};api_ranges=[]
    def node(key,kind,name,path,line,role,**fields):
        n={'id':'alignment.'+key,'kind':kind,'name':name,'path':path,'line':line,'role':role,**fields}
        nodes.append(n);by_id[key]=n;return n
    def api(key,path,line,name,kind,signature_range,declaration_range,role,qualified_name=None,**extra):
        qualified_name=qualified_name or 'cute::'+name
        sr={k:v for k,v in signature_range.items()if k!='raw'};signature=signature_range['raw'].rstrip()
        n=node(key,'api' if kind in ('function','macro_definition') else 'type',name,path,line,role,
            qualified_name=qualified_name,signature=signature,signature_range=sr,
            source_selector={'path':path,'line':line,'name':name,'kind':kind,'qualified_name':qualified_name,
                             'signature_range':sr,'signature_sha256':hashlib.sha256(signature.encode()).hexdigest()},
            manual_declaration={'name':name,'kind':kind,'raw_signature':signature,'signature_range':sr},
            declaration_status='source_selector_pending_canonical_enrichment',
            source_declaration=declaration_range,**extra)
        if path in SCOPE:api_ranges.append((declaration_range['start_byte'],declaration_range['end_byte'],path,key))
        return n
    def edge(key,owner,target,relation,s,condition='无额外条件',resolution='source_proven',**extra):
        item={'id':'alignment.edge.'+key,'source':'alignment.'+owner,'target':'alignment.'+target,
              'relation':relation,'source_expression':s['raw'],'condition':condition,'resolution':resolution,'evidence':[evidence(s)],**extra}
        if relation=='calls':item['callsite']={k:v for k,v in s.items()if k!='raw'}|{'source_expression':s['raw'],'raw_source_expression':s['raw'],'caller':item['source']}
        edges.append(item);return item['id']
    def obligation(category,s,refs=(),explanation='',syntax_kind=None,**extra):
        ident='alignment.obligation.'+hashlib.sha256(repr((category,s['path'],s['start_byte'],s['end_byte'])).encode()).hexdigest()[:20]
        obligations.append({'id':ident,'category':category,'syntax_kind':syntax_kind or category,'source_range':s,
            'status':'source_accounted_draft_pending_independent_review','graph_refs':list(refs),'explanation':explanation,**extra})

    # One namespace identity, two independently located source occurrences.
    namespace_occ=[]
    for path,line in ((A,38),(R,36)):
        word=locate(path,'namespace cute',line)
        namespace_occ.append(span(path,word['start_byte'],source(path).index(b'{',word['end_byte'])))
    ns=node('namespace.cute','namespace','namespace cute',A,38,'两文件共同namespace，保留两个物理出现；不复用旧错误类型身份。',qualified_name='cute',
            source_selectors=[{'path':s['path'],'kind':'namespace','name':'cute','signature_range':{k:v for k,v in s.items()if k!='raw'}} for s in namespace_occ],
            source_occurrences=namespace_occ,declaration_status='source_selectors_pending_enrichment')
    for builtin in ('int','bool','void'):
        node('builtin.'+builtin,'external',builtin,None,None,'语言内建类型，不伪造库声明位置。',qualified_name=builtin,external_kind='builtin_type')
    for typedef in ('size_t','uintptr_t'):
        node('external.'+typedef,'external',typedef,None,None,'固定源码使用的外部整数typedef；声明来源不在这两文件范围，不猜库内同名实体。',
             external_kind='integer_typedef',lookup_expression=typedef,lookup_status='external_declaration_provenance_pending')
    fn_sig=line_span(A,42,45);fn_decl=line_span(A,42,49)
    fn=api('api.is_byte_aligned',A,45,'is_byte_aligned','function',fn_sig,fn_decl,
        '检查指针整数表示与N-1掩码，不解引用ptr；static_assert调用与运行返回表达式分开。',
        template_parameters=[{'name':'N','kind':'non_type','type':'int','default':None,'raw':'int N'}],
        parameters=[{'name':'ptr','type':'void const* const','default':None,'raw':'void const* const ptr'}],return_type='bool',qualifiers=['constexpr'],attributes=['CUTE_HOST_DEVICE'])
    fn['manual_declaration'].update(parameters=fn['parameters'],return_type='bool',attributes=['CUTE_HOST_DEVICE'])
    primary=api('type.aligned_primary',A,58,'aligned_struct','struct_specifier',span(A,line_span(A,57)['start_byte'],source(A).index(b'{',line_span(A,58)['start_byte'])),line_span(A,57,58),
        '无alignment属性的空主模板；不继承Child，不对任意Alignment值自动施加对齐。',
        template_parameters=[{'name':'Alignment','kind':'non_type','type':'size_t','default':None,'raw':'size_t Alignment'},
                             {'name':'Child','kind':'type','default':'void','raw':'class Child = void'}],specialization='primary',bases=[])
    for line,value in zip(range(60,69),(1,2,4,8,16,32,64,128,256)):
        whole=line_span(A,line);name=re.search(r'aligned_struct<[^>]+>',whole['raw']).group(0)
        sig=span(A,whole['start_byte'],source(A).index(b'{',whole['start_byte']))
        api(f'type.aligned_{value}',A,line,name,'struct_specifier',sig,whole,
            f'Alignment={value}、Child仍为模板参数的独立偏特化；空body，无Child基类。',
            template_parameters=[{'name':'Child','kind':'type','default':None,'raw':'class Child'}],specialization='partial',bases=[],
            specialization_arguments={'Alignment':str(value),'Child':'Child'})
        edge(f'specializes_{value}',f'type.aligned_{value}','type.aligned_primary','specializes',locate(A,name,line),
             condition=f'Alignment={value}；Child任意且未作为base或数据成员使用。')
    array_whole=line_span(R,39,40);array_sig=span(R,array_whole['start_byte'],source(R).index(b'{',line_span(R,40)['start_byte']))
    api('type.array_aligned',R,40,'array_aligned','struct_specifier',array_sig,array_whole,
        '继承cute::array<T,N>，显式请求Alignment对齐；Alignment默认16，未重新定义array成员函数。',
        template_parameters=[{'name':'T','kind':'type','default':None,'raw':'class T'},
                             {'name':'N','kind':'non_type','type':'size_t','default':None,'raw':'size_t N'},
                             {'name':'Alignment','kind':'non_type','type':'size_t','default':'16','raw':'size_t Alignment = 16'}],
        specialization='primary',bases=['cute::array<T,N>'])
    dep_array=api('dep.array',C,42,'array','struct_specifier',line_span(C,41,42),line_span(C,41,43),'依赖文件中的真实array主模板；该header不纳入完整文件分母。',dependency_only=True)
    has_single_bit_signature=span(M,line_span(M,126)['start_byte'],source(M).index(b'{',line_span(M,129)['start_byte']))
    api('dep.has_single_bit',M,129,'has_single_bit','function',has_single_bit_signature,line_span(M,126,131),
        'static_assert的真实constexpr候选；本调用T由int N绑定为int，N的值仍为模板值。',dependency_only=True)
    node('binding.array_base','binding','cute::array<T,N>',R,40,'array_aligned的依赖基类实例；不是构造函数调用。',qualified_name='cute::array<T,N>',bindings={'T':'array_aligned::T','N':'array_aligned::N'})
    edge('array_inherits','type.array_aligned','binding.array_base','inherits',locate(R,'cute::array<T,N>',40),condition='struct默认public继承；T/N保持模板绑定。')
    edge('array_base_binding','binding.array_base','dep.array','template_binds',locate(R,'cute::array<T,N>',40),condition='已知主模板；不枚举依赖header全部成员或用户额外特化。')
    for key,n in list(by_id.items()):
        if n.get('source_selector') and n['path'] in SCOPE:
            edge('namespace_'+key.replace('.','_'),key,'namespace.cute','member_of',n['source_declaration'],condition='C++声明在namespace cute内；宏定义另按全局预处理名字。')

    defs=[]
    for line,branch,condition,replacement in [(52,'cuda','defined(__CUDACC__)','__align__(n)'),(54,'cpp','!defined(__CUDACC__)','alignas(n)')]:
        s=line_span(A,line)
        n=api('macro.'+branch,A,line,'CUTE_ALIGNAS','macro_definition',s,s,
              '同一预处理宏名字的独立条件定义；不属于cute namespace成员。',qualified_name='CUTE_ALIGNAS',
              preprocessor_conditions=[{'expression':condition,'directive_path':A,'directive_line':51 if branch=='cuda' else 53}],
              macro_parameters=['n'],replacement=replacement,macro_identity_group='CUTE_ALIGNAS')
        defs.append((branch,condition,replacement,n['id']))
    node('macro.host_device','binding','CUTE_HOST_DEVICE',A,43,'注解宏单独保留，两个固定定义分支；不是函数调用。',
        definition_variants=[{'condition':'defined(__CUDACC__) || defined(_NVHPC_CUDA)','path':CFG,'line':34,'replacement':'__forceinline__ __host__ __device__','source_range':line_span(CFG,34)},
                             {'condition':'!(defined(__CUDACC__) || defined(_NVHPC_CUDA))','path':CFG,'line':38,'replacement':'inline','source_range':line_span(CFG,38)}])
    host_edge=edge('host_device_attribute','api.is_byte_aligned','macro.host_device','has_attribute',locate(A,'CUTE_HOST_DEVICE',43),evaluation='declaration_attribute')
    attribute_uses=[]
    for path,line,owner in [(A,l,f'type.aligned_{v}') for l,v in zip(range(60,69),(1,2,4,8,16,32,64,128,256))]+[(R,40,'type.array_aligned')]:
        ls=line_span(path,line);match=re.search(r'CUTE_ALIGNAS\(([^)]*)\)',ls['raw']);assert match
        s=locate(path,match[0],line);expression=match[1];key=f'attribute.{"aligned" if path==A else "array"}_{line}'
        n=node(key,'binding',match[0],path,line,'请求alignment的源属性；两个定义分支共享数值合同但不合并实际展开拼写。',
            semantic_class='requested_alignment_attribute',source_range=s,alignment_expression=expression,
            expression_range=span(path,s['start_byte']+s['raw'].index('(')+1,s['end_byte']-1),alignment_unit='bytes',
            value_evaluation='literal_or_template_expression_preserved_not_instantiated',owner_ref='alignment.'+owner,
            expansion_variants=[{'condition':condition,'definition_ref':ident,'expanded_spelling':replacement.replace('(n)','('+expression+')')} for _,condition,replacement,ident in defs])
        attached=edge('attach_'+key.replace('.','_'),owner,key,'has_attribute',s,evaluation='declaration_attribute')
        refs=[n['id'],attached]
        for branch,condition,replacement,ident in defs:
            refs.append(edge('expand_'+key.replace('.','_')+'_'+branch,key,'macro.'+branch,'expands_to',s,condition,
                evaluation='preprocessing_attribute_expansion_not_call',expanded_spelling=replacement.replace('(n)','('+expression+')'),definition_evidence=[evidence(line_span(A,52 if branch=='cuda' else 54))]))
        attribute_uses.append((s,refs))

    node('param.N','binding','N：int模板值',A,42,'is_byte_aligned的非类型模板参数，参与static_assert和N-1。',source_range=locate(A,'int N',42))
    node('param.ptr','resource','ptr：指针值',A,45,'只读取指针值；本函数没有读取指向内存，也没有分配/同步。',source_range=locate(A,'void const* const ptr',45))
    call=locate(A,'has_single_bit(N)',47)
    call_edge=edge('has_single_bit','api.is_byte_aligned','dep.has_single_bit','calls',call,
        condition='static_assert控制表达式；T=int，由模板参数N的声明类型决定，不是一次运行期调用。',
        evaluation='constant_evaluation_static_assert',execution_phase='template_instantiation',template_bindings={'T':'int'},argument_bindings=[{'actual':'N','formal':'x','formal_type':'int'}])
    edge('predicate_reads_N','api.is_byte_aligned','param.N','reads',locate(A,'N',47),evaluation='constant_evaluation_static_assert')
    expressions=[('assert','static_assert(has_single_bit(N), "N must be a power of 2 in alignment check");',47,'static_assert_constraint'),
                 ('cast','reinterpret_cast<uintptr_t>(ptr)',48,'builtin_named_cast'),
                 ('minus','N-1',48,'builtin_integer_subtraction'),
                 ('and','reinterpret_cast<uintptr_t>(ptr) & (N-1)',48,'builtin_bitwise_and'),
                 ('equal','(reinterpret_cast<uintptr_t>(ptr) & (N-1)) == 0',48,'builtin_equality'),
                 ('return','return (reinterpret_cast<uintptr_t>(ptr) & (N-1)) == 0;',48,'return_statement')]
    expression_refs={}
    for key,text,line,kind in expressions:
        s=locate(A,text,line);node('expr.'+key,'binding',text,A,line,'语言表达式或控制语法；不伪造同名API调用。',semantic_class=kind,source_range=s,
            evaluation='constant_evaluation_static_assert' if key=='assert' else 'potentially_evaluated_function_body')
        ref=edge('evaluate_'+key,'api.is_byte_aligned','expr.'+key,'evaluates',s,
                 evaluation='constant_evaluation_static_assert' if key=='assert' else 'potentially_evaluated_function_body')
        expression_refs[(s['start_byte'],s['end_byte'])]=['alignment.expr.'+key,ref]
    edge('cast_type','expr.cast','external.uintptr_t','type_uses',locate(A,'uintptr_t',48),condition='外部整数typedef的具体声明/位宽依赖目标环境，不能猜库内实体。')
    edge('cast_reads_ptr','expr.cast','param.ptr','reads',locate(A,'ptr',48),evaluation='pointer_value_only_no_pointee_read')
    edge('mask_reads_N','expr.minus','param.N','reads',locate(A,'N',48),evaluation='template_value_in_function_body')
    for owner,target,path,text,line in [('api.is_byte_aligned','builtin.int',A,'int',42),('api.is_byte_aligned','builtin.bool',A,'bool',44),
        ('api.is_byte_aligned','builtin.void',A,'void',45),('type.aligned_primary','external.size_t',A,'size_t',57),
        ('type.aligned_primary','builtin.void',A,'void',57),('type.array_aligned','external.size_t',R,'size_t N',39),
        ('type.array_aligned','external.size_t',R,'size_t Alignment',39)]:
        s=locate(path,text,line)
        if text in ('size_t N','size_t Alignment'):s=span(path,s['start_byte'],s['start_byte']+len('size_t'))
        edge('type_use_'+owner.replace('.','_')+'_'+str(line)+'_'+str(len(edges)),owner,target,'type_uses',s)

    # Every written parameter is separately located, including macro formals.
    for key,n in by_id.items():
        if n.get('source_selector') and n['path'] in SCOPE:
            first=n['source_declaration']['start_line'];last=n['source_declaration']['end_line']
            for index,p in enumerate(n.get('template_parameters',[])):
                s=locate(n['path'],p['raw'],first,last)
                parameters.append({'owner':n['id'],'category':'template_parameter','index':index,**p,**s})
            for index,p in enumerate(n.get('parameters',[])):
                s=locate(n['path'],p['raw'],first,last)
                parameters.append({'owner':n['id'],'category':'function_parameter','index':index,**p,**s})
            for index,name in enumerate(n.get('macro_parameters',[])):
                s=locate(n['path'],'(n)',n['line']);s=span(n['path'],s['start_byte']+1,s['end_byte']-1)
                parameters.append({'owner':n['id'],'category':'macro_parameter','index':index,'name':name,**s})

    categories={'template_declaration':'api_template','function_definition':'function_definition',
        'struct_specifier':'type_declaration','namespace_definition':'namespace','return_statement':'return_statement',
        'static_assert_declaration':'constraint','binary_expression':'operator_expression',
        'call_expression':'call_expression','type_identifier':'type_spelling_token','primitive_type':'type_spelling_token',
        'preproc_include':'include','preproc_call':'pragma','preproc_function_def':'macro_definition'}
    for path in SCOPE:
        for ast in walk(projected_tree(path).root_node):
            if ast.type not in categories:continue
            s=span(path,ast.start_byte,ast.end_byte);category=categories[ast.type]
            containing=sorted((b-a,key) for a,b,p,key in api_ranges if p==path and a<=s['start_byte'] and s['end_byte']<=b)
            owner=containing[0][1] if containing else None;refs=['alignment.'+owner] if owner else []
            explanation='物理源码构造入账；声明与引用角色以对应selector/关系说明，不凭parser类型token生成新API。'
            if category=='namespace':refs=[ns['id']]
            if category=='call_expression':
                if s['raw'].startswith('reinterpret_cast<'):category='named_cast';refs=expression_refs[(s['start_byte'],s['end_byte'])]
                else:assert s['raw']=='has_single_bit(N)';refs=[call_edge]
            if category in ('constraint','operator_expression','return_statement'):refs=expression_refs[(s['start_byte'],s['end_byte'])]
            extra={}
            if category=='type_spelling_token':
                token=s['raw'];parent=ast.parent
                if token in ('size_t','uintptr_t'):
                    role='external_integer_typedef_reference';refs.append('alignment.external.'+token)
                elif ast.type=='primitive_type':
                    role='builtin_type';refs.append('alignment.builtin.'+token)
                elif parent and parent.type in ('type_parameter_declaration','optional_type_parameter_declaration'):
                    role='type_parameter_declaration'
                elif parent and parent.type=='struct_specifier' and parent.child_by_field_name('name')==ast:
                    role='type_declaration_name'
                elif token=='aligned_struct':role='partial_specialization_family_pattern'
                elif token in ('Child','T'):role='type_parameter_reference'
                elif token in ('Alignment','N'):role='non_type_template_parameter_reference'
                elif token=='array':
                    role='base_template_reference';refs.append('alignment.dep.array')
                else:raise AssertionError(('unclassified type-spelling token',path,s,owner))
                extra['semantic_role']=role
            if category=='include':
                target='include/'+re.search(r'<([^>]+)>',s['raw']).group(1);extra={'target_path':target,'relation':'file_dependency'}
                explanation='文件include义务；不是API calls，依赖header不加入完整文件范围。'
            obligation(category,s,refs,explanation,syntax_kind=ast.type,**extra)
        for m in re.finditer(rb'^#(?:if defined\(__CUDACC__\)|else|endif)(?!\w)[^\n]*\n?',source(path),re.M):
            obligation('preprocessor_branch',span(path,m.start(),m.end()),[d[3] for d in defs],
                '完整保留两个宏定义条件，非配置选择；不作为运行时分支。')
    for s,refs in attribute_uses:obligation('alignment_macro_use',s,refs,'两分支请求同一表达式的对齐；属性不是函数调用。')
    obligation('annotation_macro_use',locate(A,'CUTE_HOST_DEVICE',43),[host_edge],'原函数属性宏及两条固定配置定义单独保留。')
    for p in parameters:
        if p.get('default') is not None:
            owner=next(n for n in nodes if n['id']==p['owner']);s=locate(p['path'],p['default'],p['start_line'],p['end_line'])
            obligation('template_default',s,[owner['id']],'模板默认值的独立原始位置；不等同函数默认实参。')

    for n in nodes:
        if n.get('source_selector'):n['identity_policy']='No entity_id copied from pre-adaptation ledger; root must enrich against corrected declaration identity.'
    views=[]
    def view(key,title,selected_edges):
        selected=[e for e in edges if e['id'] in selected_edges];ids=sorted({i for e in selected for i in (e['source'],e['target'])})
        views.append({'id':'alignment.view.'+key,'title':title,'kind':'relationships','node_ids':ids,'edge_ids':[e['id'] for e in selected]})
    view('predicate','is_byte_aligned：实例化约束、地址转换与掩码',[e['id'] for e in edges if any(k in (e['source']+' '+e['target']) for k in ['api.is_byte_aligned','expr.','param.'])])
    view('specializations','aligned_struct：主模板与九个独立偏特化',[e['id'] for e in edges if e['relation']=='specializes' or (e['relation']=='member_of' and 'aligned_' in e['source'])])
    view('array','array_aligned：默认Alignment=16与array基类',[e['id'] for e in edges if 'array' in e['source'] and e['relation']!='expands_to'])
    view('alignment_attributes','十个属性位置与两个CUTE_ALIGNAS定义分支',[e['id'] for e in edges if e['relation'] in ['has_attribute','expands_to']])
    remaining=set(e['id'] for e in edges)-{i for v in views for i in v['edge_ids']}
    if remaining:view('type_boundaries','基础类型引用与namespace归属',remaining)
    data={'schema_version':1,'module_id':'cute_alignment','part':'complete_two_headers',
        'snapshot_commit':'8f50b052e1099fb982392a622caab69b97b63128','status':'draft_pending_canonical_enrichment_and_independent_review',
        'scope':{'kind':'complete_physical_headers_not_configuration_slice','paths':list(SCOPE),
            'files':[{'path':p,'sha256':hashlib.sha256(source(p)).hexdigest(),'bytes':len(source(p)),'line_count':len(source(p).splitlines())} for p in SCOPE],
            'dependency_headers_are_full_scope':False,'all_template_instances_resolved':False},
        'nodes':nodes,'edges':edges,'views':views,
        'contracts':[
            {'id':'alignment.contract.pointer_predicate','resource':'alignment.param.ptr','participants':['alignment.api.is_byte_aligned'],
             'events':[call_edge]+[e['id'] for e in edges if e['relation']=='evaluates'],
             'preconditions':['static_assert requires a well-formed constant evaluation of has_single_bit<int>(N) yielding true.',
                              'uintptr_t must be provided by the target environment as a suitable integer type; exact external declaration remains a boundary.',
                              'constexpr declaration does not make reinterpret_cast pointer conversion a C++17 constant expression.'],
             'release_condition':'No allocation, pointee memory access, barrier or ownership transfer; returns the pointer-integer mask comparison result.',
             'evidence':[evidence(fn_decl)]},
            {'id':'alignment.contract.types','participants':['alignment.type.aligned_primary','alignment.type.array_aligned']+[f'alignment.type.aligned_{v}' for v in (1,2,4,8,16,32,64,128,256)],
             'preconditions':['Only nine listed aligned_struct partial specializations request their fixed alignment; primary template has no alignment attribute.',
                              'Child is a discriminator/defaulted type parameter, not a base or member; no inherits edge to Child exists.',
                              'array_aligned defaults Alignment to16 and inherits array<T,N>; validity of the requested alignment and stronger natural alignment belong to the target compiler.',
                              'Macro spelling is __align__(n) under __CUDACC__, alignas(n) otherwise; both definitions and all ten physical uses retained.'],
             'release_condition':'Type/representation contract, not a runtime allocation or initialization event.',
             'evidence':[evidence(line_span(A,51,68)),evidence(array_whole)]}],
        'issues':[{'id':'alignment.issue.adapter_and_identity','status':'open','message':'Corrected canonical identity enrichment and retirement of temporary manual selectors remain pending; source-selector accounting is not API coverage acceptance.'},
                  {'id':'alignment.issue.external_typedefs','status':'boundary','message':'size_t/uintptr_t exact external declarations/ABI widths are not attributed to invented in-scope entities.'},
                  {'id':'alignment.issue.implicit_members','status':'boundary','message':'Compiler-generated special members and all inherited array implementations are not separately written in these headers; dependency inheritance is recorded, not duplicated.'},
                  {'id':'alignment.issue.arbitrary_instantiations','status':'boundary','message':'All written declarations/branches are targeted; arbitrary Alignment/T/N/Child instantiations and CUDA backend layout are not universally validated.'}],
        'coverage':{'obligations':obligations,'parameter_obligations':parameters,'counts':dict(Counter(o['category']for o in obligations)),
            'parameter_counts':dict(Counter(p['category']for p in parameters)),
            'unclassified_source_obligations':[],'semantic_completion':'source_accounting_draft_not_api_acceptance',
            'expected_physical_declaration_occurrences':16,'expected_alignment_uses':10,'expected_macro_definition_branches':2,
            'non_executable_regions':'Both headers retain copyright/license lines1–30, comments and whitespace in snapshot; includes/pragma/namespace and conditional directives are independently inventoried.',
            'parser_only_masks':['CUTE_HOST_DEVICE equal-length whitespace; semantic macro attribute remains separately accounted.',
                                 'CUTE_ALIGNAS keyword becomes same-width alignas plus spaces only in checker/author parse views; source arguments, macro uses and both real expansions are independently retained. This does not implement the root adapter.'],
            'api_coverage_passed':False},
        'runtime_execution_claimed':False,'global_completion_claimed':False,
        'provenance':{'generator_path':str(Path(__file__).relative_to(ROOT)),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'global_ledger_read':False,'only_source_selectors_used':True}}
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=HERE,prefix='.relations-',delete=False)as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');temporary=f.name
    os.replace(temporary,HERE/'relations.json')
    print(json.dumps({'nodes':len(nodes),'edges':len(edges),'views':len(views),'obligations':len(obligations),'counts':data['coverage']['counts'],'parameters':len(parameters)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
