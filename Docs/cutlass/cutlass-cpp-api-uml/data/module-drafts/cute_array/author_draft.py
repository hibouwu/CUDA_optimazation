#!/usr/bin/env python3
"""Fixed array.hpp source inventory and relation draft, not a C++ resolver.

The denominator comes from the original physical source. No global ledger or
extractor is imported. Namespace macro bindings are the two fixed config.hpp
definitions, not a guessed owner. All writes stay in this draft directory.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from tree_sitter import Language, Parser
import tree_sitter_cpp

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PATH = 'include/cute/container/array.hpp'
CFG = 'include/cute/config.hpp'
TRAITS = 'include/cute/util/type_traits.hpp'
CUTLASS = 'include/cutlass/cutlass.h'
COMMIT = '8f50b052e1099fb982392a622caab69b97b63128'
DECLARATIONS = {'function_definition', 'struct_specifier', 'alias_declaration', 'field_declaration', 'namespace_definition'}
CATEGORIES = {
    'function_definition':'function_definition', 'struct_specifier':'type_declaration',
    'alias_declaration':'alias_declaration', 'field_declaration':'member_declaration',
    'namespace_definition':'namespace', 'template_declaration':'template_declaration',
    'call_expression':'call_expression', 'subscript_expression':'subscript_expression',
    'type_identifier':'type_spelling_token', 'primitive_type':'type_spelling_token',
    'namespace_identifier':'namespace_spelling_token', 'using_declaration':'local_using_declaration',
    'declaration':'local_declaration', 'for_statement':'control_flow', 'for_range_loop':'control_flow',
    'if_statement':'control_flow', 'static_assert_declaration':'constraint', 'return_statement':'return_statement',
    'binary_expression':'operator_expression', 'assignment_expression':'operator_expression',
    'update_expression':'operator_expression', 'pointer_expression':'pointer_expression',
    'initializer_list':'initializer_list', 'preproc_include':'include', 'preproc_call':'pragma',
}

def source(path=PATH): return (ROOT/'snapshot'/path).read_bytes()
RAW = source()

def span(a, b, path=PATH):
    raw=source(path)
    return {'path':path,'start_byte':a,'end_byte':b,'start_line':raw[:a].count(b'\n')+1,
            'end_line':raw[:max(a,b-1)].count(b'\n')+1,'raw':raw[a:b].decode()}

def lines(first,last=None,path=PATH):
    ls=source(path).splitlines(keepends=True);last=first if last is None else last
    return span(sum(map(len,ls[:first-1])),sum(map(len,ls[:last])),path)

def locate(text,line,last=None,path=PATH):
    region=lines(line,last,path);a=source(path).index(text.encode(),region['start_byte'],region['end_byte'])
    return span(a,a+len(text.encode()),path)

def coords(s): return {k:v for k,v in s.items()if k!='raw'}
def evidence(s): return coords(s)|{'quote':s['raw']}
def walk(n):
    yield n
    for c in n.named_children: yield from walk(c)
def spelling(n): return RAW[n.start_byte:n.end_byte].decode() if n else ''
def ancestors(n):
    while n.parent:
        n=n.parent;yield n
def projected_tree():
    masked=re.sub(rb'\bCUTE_HOST_DEVICE\b',lambda m:b' '*len(m[0]),RAW)
    tree=Parser(Language(tree_sitter_cpp.language())).parse(masked)
    assert not tree.root_node.has_error
    return tree

def conditions(line):
    if line==396:return [{'expression':'defined(__CUDACC_RTC__)','directive_path':PATH,'directive_line':395}]
    if line==398:return [{'expression':'!defined(__CUDACC_RTC__)','directive_path':PATH,'directive_line':397}]
    if 447<=line<=475:
        result=[{'expression':'defined(CUTE_STL_NAMESPACE_IS_CUDA_STD)','directive_path':PATH,'directive_line':446}]
        if 451<=line<=453:result.append({'expression':'(__CUDACC_VER_MAJOR__ >= 13)','directive_path':PATH,'directive_line':450})
        if 456<=line<=460:
            result += [{'expression':'!((__CUDACC_VER_MAJOR__ >= 13))','directive_path':PATH,'directive_line':454},
                       {'expression':'defined(__CUDACC_RTC__)','directive_path':PATH,'directive_line':455}]
        return result
    return []

def namespace_variants(line):
    if 430<=line<=444:
        return [('std',[{'expression':'!defined(__CUDACC_RTC__)','directive_path':CFG,'directive_line':109}]),
                ('cuda::std',[{'expression':'defined(__CUDACC_RTC__)','directive_path':CFG,'directive_line':106}])]
    return [('std' if line>=447 else 'cute',conditions(line))]

def function_declarator(ast):
    declarator=ast.child_by_field_name('declarator')
    return next(n for n in walk(declarator)if n.type=='function_declarator')

def start_with_prefix(ast):
    if ast.parent and ast.parent.type=='template_declaration':return ast.parent.start_byte
    a=ast.start_byte
    if ast.type=='function_definition':
        beginning=RAW.rfind(b'\n',0,a)+1
        annotation=RAW.find(b'CUTE_HOST_DEVICE',beginning,a)
        if annotation>=0:a=annotation
    return a

def written_parameters(ast):
    records=[]
    if ast.type=='function_definition':
        ps=function_declarator(ast).child_by_field_name('parameters')
        for i,p in enumerate(ps.named_children):
            d=p.child_by_field_name('declarator')
            names=[n for n in walk(d)if n.type in ('identifier','field_identifier')] if d else []
            name=names[-1] if names else None
            text=spelling(p);name_text=spelling(name)if name else None
            tp=text[:name.start_byte-p.start_byte].rstrip()if name else text
            records.append({'category':'function_parameter','index':i,'name':name_text,'type':tp,'default':None,**span(p.start_byte,p.end_byte)})
    return records

def build():
    tree=projected_tree();ast_nodes=list(walk(tree.root_node));nodes=[];edges=[];obligations=[];parameters=[]
    by_id={};physical=[];bindings=[];range_owners=[];api_by_line={};syntax_refs={}
    def node(key,kind,name,s=None,role='',**extra):
        ident='array.'+key
        if ident in by_id:return by_id[ident]
        n={'id':ident,'kind':kind,'name':name,'path':s['path']if s else None,'line':s['start_line']if s else None,'role':role,**extra}
        if s:n['source_range']=s
        nodes.append(n);by_id[ident]=n;return n
    def edge(owner,target,relation,s,condition='无额外条件',resolution='source_proven',**extra):
        assert owner in by_id and target in by_id,(owner,target)
        active=[o['preprocessor_conditions']for o in by_id[owner].get('source_occurrences',[])
                if o['source_declaration']['path']==s['path'] and o['source_declaration']['start_byte']<=s['start_byte']
                and s['end_byte']<=o['source_declaration']['end_byte']]
        if condition=='无额外条件'and active:
            condition=' || '.join('('+' && '.join(c['expression']for c in g)+')'if g else'true'for g in active)
        contexts=[]
        if s['path']==PATH:
            candidates=[a for a in ast_nodes if a.start_byte<=s['start_byte']and s['end_byte']<=a.end_byte]
            current=min(candidates,key=lambda a:a.end_byte-a.start_byte)if candidates else None
            if current:
                for control in ancestors(current):
                    if control.type in ('for_statement','for_range_loop'):
                        body=control.child_by_field_name('body')
                        if body and body.start_byte<=s['start_byte']and s['end_byte']<=body.end_byte:
                            guard=control.child_by_field_name('condition')or control.child_by_field_name('right')
                            contexts.append({'kind':'range_for_body'if control.type=='for_range_loop'else'ordinary_for_body',
                                             'guard_source':span(guard.start_byte,guard.end_byte),
                                             'does_not_discard_body_at_template_instantiation':True})
                    elif control.type=='if_statement':
                        for branch in('consequence','alternative'):
                            region=control.child_by_field_name(branch)
                            if region and region.start_byte<=s['start_byte']and s['end_byte']<=region.end_byte:
                                guard=control.child_by_field_name('condition');guard=guard.child_by_field_name('value')or guard
                                contexts.append({'kind':'if_constexpr_branch'if spelling(control).startswith('if constexpr')else'ordinary_if_branch',
                                                 'guard_source':span(guard.start_byte,guard.end_byte),'branch':branch,
                                                 'discarded_branch_not_instantiated':spelling(control).startswith('if constexpr')})
        ident=f'array.edge.{len(edges):04d}.{relation}'
        e={'id':ident,'source':owner,'target':target,'relation':relation,'source_expression':s['raw'],
           'condition':condition,'resolution':resolution,'evidence':[evidence(s)],
           'source_occurrence_condition_groups':active,'control_contexts':contexts,**extra}
        if relation=='calls':e['callsite']=coords(s)|{'source_expression':s['raw'],'raw_source_expression':s['raw'],'caller':owner}
        edges.append(e);return ident
    def remember(ast,ref): syntax_refs.setdefault((ast.start_byte,ast.end_byte,ast.type),[]).append(ref)
    file_node=node('file.source','binding',PATH,span(0,len(RAW)),
                   '文件级include/预处理义务端点，不是namespace或C++ API。',semantic_class='source_file')
    # Physical declarations are enumerated before macro namespace variants.
    for ast in ast_nodes:
        if ast.type not in DECLARATIONS:continue
        line=ast.start_point[0]+1;body=ast.child_by_field_name('body')
        start=start_with_prefix(ast);end=body.start_byte if body else ast.end_byte
        sig=span(start,end);decl=span(start,ast.parent.end_byte if ast.parent and ast.parent.type=='template_declaration'else ast.end_byte)
        name_node=ast.child_by_field_name('name');name=spelling(name_node)
        class_parent=next((p for p in ancestors(ast)if p.type=='struct_specifier'),None)
        owner_class=spelling(class_parent.child_by_field_name('name')) if class_parent else None
        kind={'namespace_definition':'namespace','alias_declaration':'alias','field_declaration':'member'}.get(ast.type,ast.type)
        fn=None
        if ast.type=='function_definition':
            fn=function_declarator(ast);name_node=fn.child_by_field_name('declarator');name=spelling(name_node);line=name_node.start_point[0]+1
            kind='operator'if name.startswith('operator')else'method'if class_parent else'function'
        if ast.type=='field_declaration':name='__elems_'
        variants=namespace_variants(line)
        if ast.type=='namespace_definition'and line==430:
            variants=[variants[0],('cuda',variants[1][1]),variants[1]]
        physical_id=f'array.physical.{ast.start_byte}'
        physical.append({'id':physical_id,'syntax_kind':ast.type,'name':name,'syntax_range':span(ast.start_byte,ast.end_byte),
                         'signature_range':coords(sig),'raw_signature':sig['raw'].rstrip(),'source_declaration':decl})
        for ns,conds in variants:
            qualified=ns if kind=='namespace'else ns+'::'+(owner_class+'::'if owner_class else'')+name
            if kind=='namespace':key='namespace.'+ns.replace('::','_')
            elif line in (434,465):key='type.'+ns.replace('::','_')+'.tuple_size'
            elif line in (439,470):key='type.'+ns.replace('::','_')+'.tuple_element'
            elif line in (441,472):key='alias.'+ns.replace('::','_')+'.tuple_element.type'
            elif ast.type=='function_definition':key=f'api.fn_{line}'
            elif ast.type=='struct_specifier':key=f'type.decl_{line}'
            else:key=f'{kind}.decl_{line}'
            selector={'path':PATH,'kind':kind,'name':name if kind!='namespace'else ns.split('::')[-1],
                      'qualified_name':qualified,'signature_range':coords(sig),'signature_sha256':hashlib.sha256(sig['raw'].rstrip().encode()).hexdigest()}
            occurrence={'physical_id':physical_id,'source_selector':selector,'preprocessor_conditions':conds,'source_declaration':decl,
                        'declared_owner':ns+('::'+owner_class if owner_class else'')}
            n=node(key,'namespace'if kind=='namespace'else'type'if kind in ('struct_specifier','alias')else'resource'if kind=='member'else'api',
                   selector['name'],sig,'原文独立声明/重载；相同实体的条件物理出现分别保留。',qualified_name=qualified,
                   signature=sig['raw'].rstrip(),signature_range=coords(sig),source_selector=selector,source_selectors=[],source_occurrences=[],
                   source_declaration=decl,entity_kind=kind,declaration_status='source_selector_pending_canonical_enrichment',
                   access_scope='class_member'if class_parent else'not_class_member',access='public'if class_parent else None,
                   parameters=written_parameters(ast),template_parameters=[],preprocessor_conditions=conds)
            n['source_selectors'].append(selector);n['source_occurrences'].append(occurrence)
            bindings.append({'id':f'array.binding_occurrence.{len(bindings):03d}','node_id':n['id'],**occurrence})
            range_owners.append((start,ast.end_byte,n['id'],ast.type))
            remember(ast,n['id'])
            if fn:
                n.update(return_type=RAW[ast.child_by_field_name('type').start_byte:name_node.start_byte].decode().strip(),
                         qualifiers=['constexpr']+(['const']if any(c.type=='type_qualifier'and spelling(c)=='const'for c in fn.named_children)else[]),
                         attributes=['CUTE_HOST_DEVICE'],body_range=span(body.start_byte,body.end_byte),name_line=line)
                api_by_line[line]=n['id']
            elif kind=='alias':n['target_type']=spelling(ast.child_by_field_name('type'))
            elif kind=='member':n.update(declared_type='element_type[N]',initializer=None)
            elif kind=='struct_specifier':
                base=next((c for c in ast.named_children if c.type=='base_class_clause'),None)
                n.update(specialization='partial'if '<'in name else'primary',bases=[spelling(base)]if base else[],
                         definition=body is not None)
        # Written function formals are physical, not duplicated per owner variant.
        for p in written_parameters(ast):parameters.append({'owners':syntax_refs[(ast.start_byte,ast.end_byte,ast.type)],**p})
    assert len(physical)==87 and len(bindings)==92,(len(physical),len(bindings))
    for n in nodes:
        if not n.get('source_occurrences'):continue
        # Entity presence is an OR of physical occurrence conjunctions, never
        # the first physical declaration's condition alone.
        groups=[o['preprocessor_conditions']for o in n['source_occurrences']]
        n['availability']={'operator':'any_of','occurrence_condition_groups':groups,'meaning':'at_least_one_physical_declaration_active'}
        if len(groups)>1:
            n.pop('preprocessor_conditions',None)
            n['conditions_scope']='per_source_occurrence_only; availability is explicit OR'
    def owners(ast,allow_namespace=False):
        matches=[(b-a,ident,kind)for a,b,ident,kind in range_owners if a<=ast.start_byte and ast.end_byte<=b and (allow_namespace or kind!='namespace_definition')]
        if not matches:
            matches=[(b-a,ident,kind)for a,b,ident,kind in range_owners if a<=ast.start_byte and ast.end_byte<=b and kind=='namespace_definition']
        if not matches:return [file_node['id']]
        shortest=min(n[0]for n in matches);return list(dict.fromkeys(ident for length,ident,_ in matches if length==shortest))
    # Template parameter declarations are recorded once at their physical site;
    # inherited template bindings on members reference these same declarations.
    parameter_nodes={}
    for ast in ast_nodes:
        if ast.parent and ast.parent.type=='template_parameter_list':
            if ast.type not in ('type_parameter_declaration','variadic_type_parameter_declaration','parameter_declaration'):continue
            text=spelling(ast);name=re.findall(r'\w+',text)[-1];s=span(ast.start_byte,ast.end_byte)
            template=ast.parent.parent;subject=next(c for c in template.named_children if c.type!='template_parameter_list')
            own=owners(subject);index=ast.parent.named_children.index(ast)
            n=node(f'parameter.{ast.start_byte}','binding',name,s,'源码模板参数，不是一个具体类型/数值实例。',parameter_kind=ast.type,default=None,owner_refs=own)
            p={'category':'template_parameter','index':index,'name':name,'default':None,'owners':own,**s};parameters.append(p)
            parameter_nodes[ast.start_byte]=n['id']
            for api in nodes:
                for occ in api.get('source_occurrences',[]):
                    d=occ['source_declaration']
                    if template.start_byte<=d['start_byte'] and d['end_byte']<=template.end_byte and p not in api.get('template_parameters',[]):api.setdefault('template_parameters',[]).append(p)
    def template_target(ast,name):
        for p in [ast,*ancestors(ast)]:
            if p.type=='template_declaration':
                params=p.child_by_field_name('parameters')
                if params:
                    for q in params.named_children:
                        if re.findall(r'\w+',spelling(q))[-1]==name:return parameter_nodes[q.start_byte]
        return None
    # Explicit namespace/member ownership remains per source occurrence.
    for b in bindings:
        n=by_id[b['node_id']];q=n['qualified_name'];s=b['source_declaration']
        if n['entity_kind']=='namespace':
            if q=='cuda::std':edge(n['id'],'array.namespace.cuda','member_of',s,condition='defined(__CUDACC_RTC__)；嵌套namespace component')
            continue
        parent_q=b['declared_owner']
        candidates=[x['id']for x in nodes if x.get('qualified_name')==parent_q and x.get('entity_kind')in ('namespace','struct_specifier')]
        assert len(candidates)==1,(q,candidates)
        edge(n['id'],candidates[0],'member_of',s,condition=' && '.join(c['expression']for c in b['preprocessor_conditions'])or'namespace/class scope from physical source')
    primary='array.type.decl_42';zero='array.type.decl_199';storage='array.member.decl_194'
    edge(zero,primary,'specializes',locate('array<T, 0>',199),condition='N=0；T保持模板类型；不是继承或调用。',template_bindings={'T':'T','N':'0'})
    # Shared dependency boundaries are explicit; type_traits using-declaration
    # and its imported lookup name are not conflated with an alias definition.
    external={}
    for name in ('void','bool','size_t','ptrdiff_t'):
        n=node('external.'+name,'external',name,role='语言内建类型。'if name in ('void','bool')else'外部整数typedef；具体声明/ABI位宽不在本文件。',external_kind='builtin_type'if name in ('void','bool')else'integer_typedef')
        external[name]=n['id']
    imported=node('dependency.remove_cv_import','binding','using CUTE_STL_NAMESPACE::remove_cv_t;',lines(92,path=TRAITS),
                  '真实using声明引入既存名字，不定义新的alias template；暂不采用旧canonical的错误QName。',
                  semantic_class='using_declaration',lookup_name='cute::remove_cv_t',target_expression='CUTE_STL_NAMESPACE::remove_cv_t',
                  target_variants=[{'condition':'!defined(__CUDACC_RTC__)','qualified_name':'std::remove_cv_t'},
                                   {'condition':'defined(__CUDACC_RTC__)','qualified_name':'cuda::std::remove_cv_t'}])
    move_sig=span(lines(194,path=TRAITS)['start_byte'],source(TRAITS).index(b'{',lines(196,path=TRAITS)['start_byte']),TRAITS)
    move=node('dependency.move','api','move',move_sig,'CuTe自己定义的move；并非std::move导入。',qualified_name='cute::move',
              signature=move_sig['raw'].rstrip(),signature_range=coords(move_sig),dependency_only=True,
              source_selector={'path':TRAITS,'kind':'function','name':'move','qualified_name':'cute::move','signature_range':coords(move_sig),
                               'signature_sha256':hashlib.sha256(move_sig['raw'].rstrip().encode()).hexdigest()},
              declaration_status='source_selector_pending_canonical_enrichment',return_type='remove_reference_t<T>&&')
    namespaces=node('macro.stl_namespace','binding','CUTE_STL_NAMESPACE',lines(106,111,CFG),'两个真实宏定义分支，所有namespace/type/using位置保留。',
                    definition_variants=[{'condition':'defined(__CUDACC_RTC__)','replacement':'cuda::std','definition':lines(107,path=CFG)},
                                         {'condition':'!defined(__CUDACC_RTC__)','replacement':'std','definition':lines(110,path=CFG)}])
    host=node('macro.host_device','binding','CUTE_HOST_DEVICE',lines(33,40,CFG),'原始声明注解；等长解析视图不会丢失真实定义。',
              definition_variants=[{'condition':'defined(__CUDACC__) || defined(_NVHPC_CUDA)','replacement':'__forceinline__ __host__ __device__','definition':lines(34,path=CFG)},
                                   {'condition':'!(defined(__CUDACC__) || defined(_NVHPC_CUDA))','replacement':'inline','definition':lines(38,path=CFG)}])
    for ast in ast_nodes:
        if ast.type=='function_definition':
            s=locate('CUTE_HOST_DEVICE',ast.start_point[0]+1);edge(owners(ast)[0],host['id'],'has_attribute',s)
    # Every lexical CUTE_STL_NAMESPACE use (including the local using) remains
    # a macro relation, but macro name mentions in comments are not uses.
    for line in (188,430,435,466):
        s=locate('CUTE_STL_NAMESPACE',line)
        ast=min((n for n in ast_nodes if n.start_byte<=s['start_byte']and s['end_byte']<=n.end_byte and n.type in ('using_declaration','namespace_definition','base_class_clause')),key=lambda n:n.end_byte-n.start_byte)
        for own in owners(ast,True):edge(own,namespaces['id'],'macro_uses',s,evaluation='preprocessing_not_cpp_call')
    # Aliases have an exact target-type relation. Each T/alias reference is
    # also individually inventoried below, so no parameter name enters a type span.
    for ast in ast_nodes:
        if ast.type!='alias_declaration':continue
        target=ast.child_by_field_name('type');s=span(target.start_byte,target.end_byte)
        for own in owners(ast):
            binding=node(f'binding.alias.{ast.start_byte}.{own}','binding',s['raw'],s,'别名目标表达式；不是构造或运行调用。',
                         owner_ref=own,missing_bindings=['T']if 'T'in s['raw']else[],lookup_scope=by_id[own]['qualified_name'].rsplit('::',1)[0])
            edge(own,binding['id'],'aliases',s,resolution='dependent_type_expression'if any(x in s['raw']for x in ('T','element_type'))else'source_proven')
    # Struct tuple specializations, bases and the bridge are separate physical
    # relationships. std bridge inherits cuda::std integral_constant under RTC.
    for ast in ast_nodes:
        if ast.type!='struct_specifier' or ast.start_point[0]+1 not in (434,439,465,470):continue
        line=ast.start_point[0]+1;name=spelling(ast.child_by_field_name('name'));family='tuple_size'if name.startswith('tuple_size')else'tuple_element'
        for own in owners(ast):
            ns=by_id[own]['qualified_name'].split('::tuple_')[0]
            ext=node('external.'+ns.replace('::','_')+'.'+family,'external',ns+'::'+family,
                     role='标准库/CCCL primary template；本文件只提供array偏特化和旧RTC条件前置声明。',external_kind='library_template')
            name_span=span(ast.child_by_field_name('name').start_byte,ast.child_by_field_name('name').end_byte)
            if line in (465,470):
                forward='array.type.decl_457'if family=='tuple_size'else'array.type.decl_460'
                edge(own,forward,'specializes',name_span,
                     condition='defined(CUTE_STL_NAMESPACE_IS_CUDA_STD) && !(__CUDACC_VER_MAJOR__ >= 13) && defined(__CUDACC_RTC__)',
                     resolution='visible_conditional_primary_declaration',primary_lookup='This file explicitly forward-declares the primary template on the old RTC branch.')
                external_condition='defined(CUTE_STL_NAMESPACE_IS_CUDA_STD) && ((__CUDACC_VER_MAJOR__ >= 13) || !defined(__CUDACC_RTC__))'
            else:external_condition='defined(__CUDACC_RTC__)'if ns=='cuda::std'else'!defined(__CUDACC_RTC__)'
            edge(own,ext['id'],'specializes',name_span,condition=external_condition,
                 primary_lookup='External tuple/structured_bindings header declaration; its content is not silently treated as a local declaration.')
            base=next((c for c in ast.named_children if c.type=='base_class_clause'),None)
            if base:
                s=span(base.start_byte+1,base.end_byte)
                while s['raw'][:1].isspace():s=span(s['start_byte']+1,s['end_byte'])
                target_ns='cuda::std'if line==465 or ns=='cuda::std'else'std'
                b=node(f'binding.base.{line}.{ns.replace("::","_")}','binding',s['raw'],s,'public基类实例；提供value=N，不是函数调用。',
                       qualified_name=target_ns+'::integral_constant<size_t, N>',bindings={'T':'size_t','v':'N'},missing_bindings=['N'],
                       macro_condition='defined(__CUDACC_RTC__)'if target_ns=='cuda::std'else'!defined(__CUDACC_RTC__)')
                edge(own,b['id'],'inherits',s,condition=b['macro_condition'])
    # Exact callsite targets for non-dependent member overload selection.
    direct={58:106,64:112,70:106,76:112,83:56,90:62,108:94,114:100,120:106,126:112,
            144:130,150:136,156:160,168:160,182:172,189:160,
            215:261,221:267,227:261,233:267,239:261,245:267}
    explicit_calls=[];overloaded_subscripts=[];builtin_subscripts=[];conversions=[];dependent_calls=[]
    def dispatch_binding(ast,name,candidates,known,missing,lookup):
        s=span(ast.start_byte,ast.end_byte)
        b=node(f'binding.dispatch.{ast.start_byte}','binding',name,s,'依赖表达式：保存具体候选、实参关系、查找路径和缺少的模板绑定，不猜用户类型。',
               known_candidates=candidates,known_bindings=known,missing_bindings=missing,lookup_path=lookup)
        for candidate in candidates:
            if candidate.get('node_id'):edge(b['id'],candidate['node_id'],'template_binds',s,condition=candidate['condition'],resolution='dependent_selection_not_call')
        return b
    for ast in ast_nodes:
        if ast.type!='call_expression':continue
        s=span(ast.start_byte,ast.end_byte);line=s['start_line'];text=s['raw'];owner=owners(ast)[0]
        if text=='CUDA_STD_HEADER(tuple)':
            b=node('macro.cuda_std_header','binding',text,s,'头文件名宏展开；不是运行调用。',definition=lines(40,path=CUTLASS),
                   bindings={'header':'tuple'},expanded_spelling='<cuda/std/tuple>',condition='defined(__CUDACC_RTC__)')
            r=edge(file_node['id'],b['id'],'macro_uses',s,condition='defined(__CUDACC_RTC__)',evaluation='include_filename_expansion');remember(ast,r);continue
        if text=='T(0)':
            b=node(f'expression.conversion.{line}','binding',text,s,'函数式类型转换/初始化，是否调用构造函数取决于T；不能一概称T API调用。',
                   semantic_class='functional_type_conversion',missing_bindings=['T'],known_argument_type='int',argument_expression='0')
            r=edge(owner,b['id'],'evaluates',s,resolution='type_dependent_initialization',evaluation='potentially_evaluated')
            conversions.append(r);remember(ast,r);continue
        if line in (132,138):target=api_by_line[(94 if line==132 else 100)if text=='data()'else 160]
        elif line in direct:target=api_by_line[direct[line]]
        elif line in (355,362,369):
            name='swap'if line==369 else'fill';positive=186 if name=='swap'else 172;empty=335 if name=='swap'else 327
            b=dispatch_binding(ast,'array<T,N>::'+name,
                [{'node_id':api_by_line[positive],'condition':'N>0 (fixed primary/zero family; user specializations excluded)'},
                 {'node_id':api_by_line[empty],'condition':'N==0'}],
                {'object':'a: array<T,N>&','T':'T','N':'N','argument':'b'if line==369 else'T(0)'if line==355 else'value'},
                ['T','N','well_formed_instantiation'],'member lookup of the array<T,N> specialization')
            target=b['id'];dependent_calls.append(target)
        elif line==190:
            using=lines(188)
            b=dispatch_binding(ast,'swap(T&, T&)',
                [{'qualified_name':'std::swap','condition':'!defined(__CUDACC_RTC__)','kind':'ordinary_lookup_import'},
                 {'qualified_name':'cuda::std::swap','condition':'defined(__CUDACC_RTC__)','kind':'ordinary_lookup_import'},
                 {'expression':'ADL candidates associated with T','condition':'T has associated namespaces/classes','kind':'dependent_lookup'}],
                {'arg0':'(*this)[i]: T&','arg1':'other[i]: T&'},['T','associated_namespaces_and_classes','overload_viability'],
                'block using CUTE_STL_NAMESPACE::swap followed by unqualified call plus ADL')
            b['using_declaration']=using;target=b['id'];dependent_calls.append(target)
        elif line==425:target=move['id']
        else:raise AssertionError(('unclassified explicit call',s))
        arguments=ast.child_by_field_name('arguments')
        actuals=[spelling(c)for c in arguments.named_children]
        formals=by_id[target].get('parameters',[])
        argument_bindings=[{'index':i,'actual':a,'formal':formals[i]['name']if i<len(formals)else't'if line==425 else'other'if line==369 else'value'if line in (355,362)else None}for i,a in enumerate(actuals)]
        r=edge(owner,target,'calls',s,resolution='symbolic'if target in dependent_calls else'direct',evaluation='potentially_evaluated',
               argument_bindings=argument_bindings,template_bindings={'T':'T& (deduced move parameter)'}if line==425 else{})
        explicit_calls.append(r);remember(ast,r)
    # operator[] syntax consists of 9 real overloaded index expressions, four
    # built-in pointer indexes, and two function designators; these differ.
    for ast in ast_nodes:
        if ast.type!='subscript_expression':continue
        s=span(ast.start_byte,ast.end_byte);line=s['start_line'];owner=owners(ast)[0]
        if s['raw']=='operator[]':remember(ast,owner);continue
        if line in (58,64,215,221):
            b=node(f'expression.pointer_index.{line}','binding',s['raw'],s,'begin()的返回指针做内建下标；不是第二次调用array::operator[]。',
                   semantic_class='builtin_pointer_subscript',precondition='pointer denotes an element at offset pos; N=0 branch cannot supply such element')
            r=edge(owner,b['id'],'evaluates',s);builtin_subscripts.append(r);remember(ast,r);continue
        const=line in (344,417) or(line==382 and s['raw'].startswith('t['))
        target_line=62 if const else 56
        if line==344:
            b=dispatch_binding(ast,'array<T,N> const::operator[]',
                [{'node_id':api_by_line[62],'condition':'N>0'}, {'node_id':api_by_line[219],'condition':'N==0; loop has zero runtime iterations but body must be well formed'}],
                {'object':spelling(ast.child_by_field_name('argument')),'index':'i','result':'const T&'},['T','N'],'const member overload lookup')
            target=b['id'];res='symbolic'
        else:target=api_by_line[target_line];res='direct'
        condition='N>0: non-discarded reverse branch'if line==382 else'I<N required when get body is instantiated; therefore a valid body has N>0'if line in (409,417,425)else'i<N in the primary array swap loop'if line==190 else'ordinary i<N loop; not if constexpr'
        r=edge(owner,target,'calls',s,condition=condition,resolution=res,evaluation='potentially_evaluated',
               call_syntax='overloaded_subscript_expression',argument_bindings=[{'actual':spelling(ast.child_by_field_name('indices'))[1:-1],'formal':'pos'}])
        overloaded_subscripts.append(r);remember(ast,r)
    # Local source objects are not library APIs. Hidden range-for variables are
    # stated as language obligations, not invented source declarations.
    locals_=[]
    for ast in ast_nodes:
        if ast.type=='declaration'and any(p.type=='function_definition'for p in ancestors(ast)) or ast.type=='for_range_loop':
            s=span(ast.start_byte,ast.end_byte);owner=owners(ast)[0]
            if ast.type=='for_range_loop':s=span(ast.start_byte,ast.child_by_field_name('body').start_byte)
            n=node(f'local.{ast.start_byte}','binding',s['raw'],s,'函数局部对象/循环变量；不是库API。',entity_kind='local_object',owner_ref=owner,
                   initialization='array<T,N> aggregate initialization t_r{}'if s['start_line']==380 else'range element reference to existing element'if ast.type=='for_range_loop'else'integer loop counter initialized to zero')
            r=edge(owner,n['id'],'evaluates',s,evaluation='local_initialization_or_binding_not_explicit_constructor_call');locals_.append(n['id']);remember(ast,r)
    loop=next(a for a in ast_nodes if a.type=='for_range_loop');s=span(loop.start_byte,loop.child_by_field_name('body').start_byte)
    for targetline,synthetic in [(106,'__range.begin()'),(130,'__range.end()')]:
        r=edge(api_by_line[172],api_by_line[targetline],'implicit_calls',s,
               condition='C++17 range-for over *this of primary array; lookup finds member begin/end',
               evaluation='language_desugaring',synthetic_expression=synthetic,source_expression_is_not_literal_call=True)
        remember(loop,r)
    # Type spelling roles: no type-declaration identifier is a reference and
    # template value arguments mis-tagged by TS are not turned into types.
    for ast in ast_nodes:
        if ast.type not in ('type_identifier','primitive_type'):continue
        s=span(ast.start_byte,ast.end_byte);token=s['raw'];parent=ast.parent;own=owners(ast)
        declaration_name=(parent.type in ('type_parameter_declaration','variadic_type_parameter_declaration') or
            parent.type in ('alias_declaration','struct_specifier')and parent.child_by_field_name('name')==ast or
            parent.type=='template_type'and parent.child_by_field_name('name')==ast and parent.parent.type=='struct_specifier'and parent.parent.child_by_field_name('name')==parent)
        if declaration_name:remember(ast,own[0]);continue
        target=template_target(ast,token)
        role='template_value_reference'if token in ('N','I','_Ip')else'type_reference'
        if not target:
            if token in external:target=external[token]
            elif token=='remove_cv_t':target=imported['id']
            elif token=='array':
                if parent.type!='template_type'and any(p.type=='struct_specifier'and spelling(p.child_by_field_name('name'))=='array<T, 0>'for p in ancestors(ast)):
                    target=zero;role='injected_class_name_reference_to_zero_specialization'
                else:target=primary;role='template_family_reference_not_instantiation_choice'
            elif token=='integral_constant':
                target=node('external.integral_constant','external','CUTE_STL_NAMESPACE::integral_constant',role='std/cuda::std的外部模板，由config分支决定；不是cute::integral_constant。',
                            candidate_qualified_names=['std::integral_constant','cuda::std::integral_constant'])['id']
            else:
                qprefix=by_id[own[0]].get('qualified_name','').rsplit('::',1)[0]
                found=[n['id']for n in nodes if n.get('qualified_name')==qprefix+'::'+token and n.get('entity_kind')=='alias']
                if len(found)==1:target=found[0]
                else:raise AssertionError(('unclassified type token',s,qprefix,found))
        for o in own:
            r=edge(o,target,'template_binds'if role=='template_value_reference'else'type_uses',s,
                   resolution='dependent_reference'if target.startswith('array.parameter.')else'source_lookup',semantic_role=role)
            remember(ast,r)
    # Remaining computations/control constructs have explicit semantic classes.
    # Operator overloading is possible only for expressions involving T; their
    # lookup obligations are preserved without inventing concrete callees.
    for ast in ast_nodes:
        if ast.type not in ('binary_expression','assignment_expression','update_expression','pointer_expression','return_statement','initializer_list','static_assert_declaration','if_statement','for_statement'):continue
        s=span(ast.start_byte,ast.end_byte);owner=owners(ast)[0];text=s['raw']
        preprocessing=any(p.type.startswith('preproc_')and p.child_by_field_name('condition')and
                          p.child_by_field_name('condition').start_byte<=ast.start_byte and ast.end_byte<=p.child_by_field_name('condition').end_byte for p in ancestors(ast))
        dependent=ast.type=='assignment_expression'or(ast.type=='binary_expression'and'!='in text)
        semantic='preprocessor_condition_expression'if preprocessing else'dependent_element_operation'if dependent else'builtin_pointer_operation'if ast.type=='pointer_expression'else'body_static_assert_not_SFINAE'if ast.type=='static_assert_declaration'else'constexpr_branch'if ast.type=='if_statement'and'if constexpr'in text else ast.type
        n=node(f'expression.{ast.type}.{ast.start_byte}','binding',text,s,'原文语言表达式/控制语法，不自动冒充函数调用。',
               semantic_class=semantic,owner_ref=owner,missing_bindings=['T','selected element operator viability']if dependent else[],
               known_operand_contract='destination T lvalue and source const T lvalue'if ast.type=='assignment_expression'else'const T lvalue != const T lvalue'if dependent else None)
        r=edge(owner,n['id'],'evaluates',s,resolution='symbolic'if dependent else'source_proven',
               condition=' && '.join(c['expression']for c in conditions(s['start_line']))or'无额外条件',
               evaluation='preprocessing'if preprocessing else'template_instantiation_constraint'if ast.type=='static_assert_declaration'else'potentially_evaluated')
        remember(ast,r)
    # Direct primary storage exposure has exactly two source returns. Other
    # accesses are expressed through the callable and builtin-operation paths.
    for line,api in [(96,94),(102,100)]:edge(api_by_line[api],storage,'reads',locate('__elems_',line),evaluation='array_to_pointer_decay_not_element_value_read')
    edge(primary,storage,'has_member',lines(194),condition='Primary definition only; element_type=T, extent=N, no initializer.')
    # Every AST candidate + every physical directive + annotation macro use.
    for ast in ast_nodes:
        if ast.type not in CATEGORIES:continue
        if ast.type=='declaration'and not any(p.type=='function_definition'for p in ancestors(ast)):continue
        s=span(ast.start_byte,ast.end_byte);category=CATEGORIES[ast.type]
        refs=syntax_refs.get((ast.start_byte,ast.end_byte,ast.type),[])
        if not refs:refs=owners(ast,True)
        extras={}
        if ast.type=='call_expression':
            category='include_macro_expression'if s['raw']=='CUDA_STD_HEADER(tuple)'else'functional_type_conversion'if s['raw']=='T(0)'else'explicit_call'
        elif ast.type=='subscript_expression':category='operator_function_designator'if s['raw']=='operator[]'else'builtin_pointer_subscript'if s['start_line']in (58,64,215,221)else'overloaded_subscript_call'
        elif ast.type=='preproc_include':
            extras={'relation':'file_dependency','execution':'preprocessing','included_expression':s['raw'].split('include',1)[1].strip(),
                    'preprocessor_conditions':conditions(s['start_line']),
                    'target_kind':'snapshot'if any(t in s['raw']for t in ('cute/','cutlass/'))else'external_standard_or_CCCL_header'}
        elif ast.type in ('type_identifier','primitive_type'):
            extras={'semantic_role':'declaration_or_reference_role_is_in_graph; primitive_type size_t/ptrdiff_t are external typedefs'}
        obligations.append({'id':f'array.obligation.{len(obligations):04d}','category':category,'syntax_kind':ast.type,
                            'source_range':s,'graph_refs':list(dict.fromkeys(refs)),'status':'source_accounted_draft_pending_review',**extras})
    for match in re.finditer(rb'^\s*#(?:if|ifdef|else|endif)\b[^\n]*',RAW,re.M):
        a=match.start()
        while RAW[a:a+1].isspace():a+=1
        obligations.append({'id':f'array.obligation.{len(obligations):04d}','category':'preprocessor_directive','syntax_kind':'physical_directive',
                            'source_range':span(a,match.end()),'graph_refs':[],'status':'source_accounted_draft_pending_review'})
    for m in re.finditer(rb'\bCUTE_HOST_DEVICE\b',RAW):
        obligations.append({'id':f'array.obligation.{len(obligations):04d}','category':'annotation_macro_use','syntax_kind':'physical_macro_token',
                            'source_range':span(m.start(),m.end()),'graph_refs':[host['id']],'status':'source_accounted_draft_pending_review'})
    contracts=[
        {'id':'array.contract.storage','participants':[primary,zero,storage],
         'claims':['Primary array has one element_type __elems_[N] member; element_type=T preserves cv while value_type=remove_cv_t<T> does not change storage.',
                   'N=0 specialization has no element member; logical length zero is not sizeof(object)==0.',
                   'Primary array requires a valid element array type and valid extent; no universal ABI sizeof/alignment equality is asserted.',
                   'No explicitly written constructor/destructor/copy/move/assignment APIs. Compiler-generated special members and their deletion/triviality depend on T/N and language mode.'],
         'evidence':[evidence(lines(41,53)),evidence(lines(194,210)),evidence(lines(327,337))]},
        {'id':'array.contract.access','participants':[api_by_line[l]for l in (56,62,68,74,80,87,94,100,213,219,225,231,237,243,249,255)],
         'claims':['[] has no runtime bounds check. Primary front/back require an existing element; zero specialization pointer accessors return nullptr and element access is not safe to execute.',
                   'data returns a pointer into existing storage, not a new allocation. Reference/pointer lifetime follows the array and underlying object rules.',
                   'Primary back calls operator[](N-1), while zero back dereferences begin(); commented rbegin is not a call.'],
         'evidence':[evidence(lines(55,102)),evidence(lines(212,257))]},
        {'id':'array.contract.iteration','participants':[api_by_line[l]for l in (106,112,118,124,130,136,142,148,154,160,166,261,267,273,279,285,291,297,303,309,315,321)],
         'claims':['Iterator types are raw pointer aliases, not separate iterator class methods.',
                   'Nonconst cbegin/cend call mutable begin/end and convert to const_pointer on return; const overloads choose const members.',
                   'Primary end=data()+size(); primary size=N; max_size=size(). Zero begin/end/cbegin/cend/data are nullptr, size=max_size=0 and empty=true.',
                   'fill range-for invokes begin/end by C++17 desugaring; hidden iterator operations are builtin pointer operations, not extra authored APIs.'],
         'evidence':[evidence(lines(105,175)),evidence(lines(260,323))]},
        {'id':'array.contract.mutation','participants':[api_by_line[l]for l in (172,180,186,327,331,335,353,360,367)],
         'claims':['Primary fill assigns each existing element from const T&; it does not construct a new array.',
                   'Primary clear calls fill(T(0)); zero member clear/fill/swap have empty bodies.',
                   'Free clear calls a.fill(T(0)), not a.clear(); T(0) must still be well formed for N=0.',
                   'Member swap imports CUTE_STL_NAMESPACE::swap then calls unqualified swap on two T lvalues; ADL and overload viability require T.',
                   'The swap loop calls size() at its condition and indexes both arrays; index and loop operations on size_type are builtin.'],
         'evidence':[evidence(lines(171,190)),evidence(lines(326,336)),evidence(lines(351,369))]},
        {'id':'array.contract.algorithms','participants':[api_by_line[341],api_by_line[375]],
         'claims':['operator== checks lhs[i] != rhs[i], not element ==. Ordinary for with N=0 still requires the dependent body to instantiate well formed in the tested C++17 mode.',
                   'reverse N==0 uses if constexpr and returns t; nonzero branch creates t_r{}, assigns t_r[k]=t[N-k-1], returns t_r.',
                   'Do not replace exact well-formedness of t_r{}, const-element assignment and by-value return with is_default_constructible/is_copy_constructible requirements.',
                   'No exact number of copies/moves/NRVO outcomes is claimed; no GPU/runtime performance execution is performed.'],
         'evidence':[evidence(lines(339,349)),evidence(lines(373,386))]},
        {'id':'array.contract.tuple','participants':[api_by_line[l]for l in (406,414,422)],
         'claims':['Exactly three get overloads: &, const&, &&; const&& arguments bind const& and do not imply a fourth const&& overload.',
                   'get&& uses named a as lvalue, calls mutable operator[], then cute::move; result still references original storage and does not extend temporary lifetime.',
                   'I<N is body static_assert, not signature SFINAE; unevaluated decltype may not instantiate that body.',
                   'tuple_element type=T does not check I<N. tuple_size inherits the configured standard-library integral_constant<size_t,N>.',
                   'Both CUTE_STL_NAMESPACE branches and the extra std bridge retain distinct physical declarations; CUDA13 external structured_bindings include and older RTC forwards are not silently omitted.'],
         'evidence':[evidence(lines(394,476)),evidence(lines(194,199,TRAITS))]},
    ]
    views=[]
    groups=[('primary','主模板：逐成员、别名与访问/迭代链',lambda e:any(by_id[x].get('qualified_name','').startswith('cute::array::')for x in(e['source'],e['target']))),
            ('empty','N=0：独立成员与空体边界',lambda e:any(by_id[x].get('qualified_name','').startswith('cute::array<T, 0>')for x in(e['source'],e['target']))),
            ('algorithms','free clear/fill/swap/比较与reverse',lambda e:e['source']in[api_by_line[x]for x in(341,353,360,367,375)]),
            ('tuple','get/tuple traits/条件std桥接',lambda e:any((by_id[x].get('line')or 0)>=394 and by_id[x].get('path')==PATH for x in(e['source'],e['target'])))]
    assigned=set()
    for key,title,predicate in groups:
        selected=[e for e in edges if predicate(e)];assigned.update(e['id']for e in selected)
        views.append({'id':'array.view.'+key,'title':title,'kind':'relationships','node_ids':sorted({x for e in selected for x in(e['source'],e['target'])}),
                      'edge_ids':[e['id']for e in selected]})
    rest=[e for e in edges if e['id']not in assigned]
    views.append({'id':'array.view.dependencies','title':'依赖类型、宏、局部对象与剩余源码表达式','kind':'relationships',
                  'node_ids':sorted({x for e in rest for x in(e['source'],e['target'])}),'edge_ids':[e['id']for e in rest]})
    return {'schema_version':1,'module_id':'cute_array','part':'complete_array_header','snapshot_commit':COMMIT,
        'status':'draft_pending_independent_review_and_canonical_enrichment',
        'scope':{'kind':'complete_physical_headers_not_configuration_slice','paths':[PATH],
                 'files':[{'path':PATH,'sha256':hashlib.sha256(RAW).hexdigest(),'bytes':len(RAW),'line_count':len(RAW.splitlines())}],
                 'dependency_headers_are_full_scope':False,'all_template_instances_resolved':False},
        'nodes':nodes,'edges':edges,'views':views,'contracts':contracts,
        'issues':[{'id':'array.issue.canonical_join','status':'open','message':'Strict canonical source-selector enrichment and independent review remain pending; no entity ID was guessed.'},
                  {'id':'array.issue.remove_cv_using','status':'open','message':'type_traits.hpp92 canonical currently conflates imported target expression and cute lookup QName. Draft retains true using declaration and lookup binding separately; root repair required.'},
                  {'id':'array.issue.external_boundaries','status':'boundary','message':'Exact external std/CCCL declarations, typedef ABI widths, CUDA13 structured-bindings header content are not part of this complete-file source denominator.'},
                  {'id':'array.issue.element_semantics','status':'boundary','message':'All T/N instances, ADL candidate sets, constructor/assignment deletion and implicit special-member generation cannot be selected without concrete bindings; exact dependent expressions retained.'},
                  {'id':'array.issue.configuration_oracles','status':'open','message':'14 host C++17 Clang syntax-only cases exist; CUDA/RTC/header-version branch compilation and arbitrary instantiations are not yet verified.'}],
        'coverage':{'physical_declarations':physical,'conditional_declaration_bindings':bindings,'obligations':obligations,
                    'parameter_obligations':parameters,'counts':dict(Counter(o['category']for o in obligations)),
                    'parameter_counts':dict(Counter(p['category']for p in parameters)),
                    'expected_physical_declaration_occurrences':87,'expected_bound_declaration_occurrences':92,
                    'expected_callable_definitions':52,'explicit_cpp_call_expressions':len(explicit_calls),
                    'overloaded_subscript_calls':len(overloaded_subscripts),'builtin_pointer_subscripts':len(builtin_subscripts),
                    'functional_type_conversions':len(conversions),'implicit_range_for_calls':2,
                    'unclassified_source_obligations':[],'api_coverage_passed':False,'semantic_completion':'source_draft_not_acceptance',
                    'parser_only_masks':['52 CUTE_HOST_DEVICE tokens replaced by equal-length spaces for parsing only; original signature and attribute definitions retained.'],
                    'non_executable_regions':'Original license/comments/whitespace remain in snapshot; no macros are defined in array.hpp. All7includes and allphysical conditional directives inventoried.'},
        'compile_evidence':[{'path':str((HERE/'oracle_results.json').relative_to(ROOT)),'review_path':str((HERE/'oracle_review.md').relative_to(ROOT)),
                             'kind':'host_clang_cxx17_syntax_only','runtime_execution':False,'cases':14,'expected_success':8,'expected_rejection':6}],
        'runtime_execution_claimed':False,'global_completion_claimed':False,
        'provenance':{'generator_path':str(Path(__file__).relative_to(ROOT)),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'global_ledger_read':False,'denominator':'independent raw-source AST with byte-preserving annotation-only mask'}}

def main():
    data=build()
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=HERE,prefix='.relations-',delete=False)as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');temporary=f.name
    os.replace(temporary,HERE/'relations.json')
    print(json.dumps({'nodes':len(data['nodes']),'edges':len(data['edges']),'physical':len(data['coverage']['physical_declarations']),
                      'bindings':len(data['coverage']['conditional_declaration_bindings']),'counts':data['coverage']['counts'],
                      'parameters':data['coverage']['parameter_counts']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
