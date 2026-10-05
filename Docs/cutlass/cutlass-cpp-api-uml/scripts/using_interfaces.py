"""Source using interfaces are not newly declared target entities.

This layer joins mapped lexical contexts with immutable physical statements.
It does not solve overload sets or infer inherited-constructor accessibility.
"""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from functools import lru_cache
import re
import sys


def _local(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('using_interface_'+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


_scan=_local('scan_candidates')


@lru_cache(maxsize=1)
def target_expander():return _local('using_target_expansion').expand_using_target


def identity(prefix,value):
    return prefix+hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]


def words(text):return tuple(t.text for t in _scan.lex((text or'').encode())[0])


def _base_expressions(clause):
    tokens=list(_scan.lex(clause.encode())[0]);parts=[];start=1 if tokens and tokens[0].text==':'else 0;depth=0
    for i,t in enumerate(tokens):
        if t.text in ('<','(','['):depth+=1
        elif t.text in ('>',')',']'):depth-=1
        elif t.text==','and depth==0:parts.append(tokens[start:i]);start=i+1
    parts.append(tokens[start:])
    return [tuple(t.text for t in part if t.text not in {'public','private','protected','virtual'})for part in parts if part]


def constructor_proof(source,context,raw):
    """Require a matching direct base/visible alias chain, not repeated names."""
    owner=context.get('class_owner')
    if context.get('scope_kind')not in {'class','local_class'}:return None
    segments=source.get('qualifier_segments',[])
    if not segments:return None
    target_range=source['target_name_source_range'];last_separator=source['scope_separator_ranges'][-1]
    prefix=raw[target_range['start_byte']:last_separator['start_byte']].decode().strip()
    original_prefix=prefix;chain=[];seen=set()
    aliases={a['name']:a for a in context.get('visible_class_aliases',[])}
    while prefix in aliases and prefix not in seen:
        seen.add(prefix);alias=aliases[prefix]
        extra=[c for c in alias['preprocessor_conditions']if c not in context['preprocessor_conditions']]
        if extra:return {'status':'pending','reason':'base_alias_visibility_has_unresolved_conditions','alias':alias}
        chain.append(alias);prefix=alias['target_type'].strip()
    if not owner:
        return {'status':'pending','reason':'class_owner_definition_unavailable'}if source.get('constructor_spelling_candidate')else None
    bases=[base for clause in owner.get('bases',[])for base in _base_expressions(clause)]
    matched=words(prefix)in bases
    prefix_head=segments[-1].get('head_name')
    # The resolved base's final class template-id head is needed for aliases
    # whose spelling differs from the injected class name.
    base_tokens=words(prefix);angle=0;heads=[]
    for i,t in enumerate(base_tokens):
        if t=='<':angle+=1
        elif t=='>':angle-=1
        elif angle==0 and re.fullmatch(r'[A-Za-z_]\w*',t):heads.append(t)
    terminal=source.get('terminal_name');constructor_name=bool(terminal in {prefix_head,heads[-1]if heads else None})
    if matched and constructor_name:
        return {'status':'source_base_proven','derived_type_ref':owner['entity_id'],
            'owner_signature_range':owner['signature_range'],'base_expression':prefix,
            'using_qualifier_expression':original_prefix,'base_source_clauses':owner['bases'],'alias_chain':chain,
            'source_using_access':context.get('lexical_access'),
            'constructor_candidates':[],'candidate_resolution':'not_attempted',
            'accessibility_contract':'The using declaration lexical access is not inherited-constructor accessibility; base constructor and use site must be checked'}
    if source.get('constructor_spelling_candidate'):
        return {'status':'pending','reason':'constructor_spelling_without_verified_matching_base','base_expression':prefix,
            'owner_signature_range':owner['signature_range'],'base_source_clauses':owner['bases'],'alias_chain':chain}
    return None


def _lookup(source,context,semantic,ctor):
    if semantic=='namespace_directive':return {'kind':'none'}
    if semantic=='inherited_constructor_import':return {'kind':'constructor_family','derived_type_ref':ctor['derived_type_ref']}
    if semantic=='import_classification_pending':return {'kind':'pending','spelling_candidate':source.get('terminal_name')}
    name=source.get('declared_name')if semantic=='local_alias_declaration'else source.get('terminal_name')
    result={'kind':'ordinary_name','spelling':name,
        'source_name_range':source.get('declared_name_range')if semantic=='local_alias_declaration'else source.get('terminal_name_range')}
    scopes=context.get('scope_chain',[])
    named=all(s['kind']in {'namespace','class_specifier','struct_specifier','union_specifier'}and
        not s.get('anonymous')and s.get('name_resolution',{}).get('status')in (None,'source_name','literal_macro_binding')and '<anonymous_'not in s['name']for s in scopes)
    if context.get('scope_kind')in {'namespace','class'}and named:
        result['qualified_lookup_spelling']='::'.join([s['name']for s in scopes]+[name])
        result['qualified_lookup_is_not_new_entity']=True
    return result


def build_using_interfaces(commit,path,raw,inventory,contexts,occurrences):
    by_range=defaultdict(list)
    for context in contexts:
        span=context['source_range']
        if span['path']==path:by_range[(span['start_byte'],span['end_byte'])].append(context)
    sources=[];interfaces=[];diagnostics=[]
    for source in inventory['sources']:
        source=dict(source);span=source['source_range'];matches=by_range[(span['start_byte'],span['end_byte'])]
        source['alias_entity_refs']=sorted({c['alias_entity_ref']for c in matches if c.get('alias_entity_ref')})
        source['alias_occurrence_refs']=sorted({c['alias_occurrence_ref']for c in matches if c.get('alias_occurrence_ref')})
        source['using_interface_refs']=[]
        if source['kind']=='alias'and source['alias_occurrence_refs']:
            source['source_contract_status']='existing_alias_occurrences_pending'if any(c['syntax_status']!='parsed'or c.get('scope_review_required')for c in matches)else'existing_alias_occurrences_linked'
            sources.append(source);continue
        if not matches:
            source['source_contract_status']='context_extraction_pending'
            diagnostics.append({'category':'using_context_missing',**span,'source_using_id':source['source_using_id'],
                'message':'Written using source has no verified mapped lexical context; source retained, no fake entity introduced'})
        expansion=None
        for context in matches:
            if context.get('alias_entity_ref'):continue
            ctor=constructor_proof(source,context,raw)if source['kind']=='import'else None
            semantic='local_alias_declaration'if source['kind']=='alias'else'namespace_directive'if source['kind']=='directive'else \
                'inherited_constructor_import'if ctor and ctor['status']=='source_base_proven'else \
                'import_classification_pending'if ctor and ctor['status']=='pending'else'dependent_type_import'if source.get('typename_range')else'ordinary_import'
            context_key=[source['source_using_id'],[s['identity']for s in context.get('scope_chain',[])],
                         context.get('lexical_blocks',[]),context.get('preprocessor_conditions',[]),context.get('scope_kind')]
            context_id=identity('using_ctx_',context_key);identifier=identity('using_iface_',[source['source_using_id'],context_id,semantic])
            blockers=[]
            if source['syntax_status']!='source_shape_classified':blockers.append('source_syntax_unclassified')
            if context['syntax_status']!='parsed':blockers.append('parser_source_context_contains_error')
            if context.get('scope_reviews'):blockers.append('enclosing_scope_requires_review')
            if semantic=='import_classification_pending':blockers.append('constructor_category_needs_base_proof')
            if expansion is None and source.get('target_name_source_range'):
                expansion=target_expander()(path,raw,source['target_name_source_range'])
            if expansion and expansion.get('source_contract_status')=='source_contract_pending':blockers.append('target_macro_source_contract_pending')
            lexical={'context_variant_id':context_id,'scope_kind':context.get('scope_kind'),
                'scope_chain':context.get('scope_chain',[]),'lexical_blocks':context.get('lexical_blocks',[]),
                'enclosing_function_ref':context.get('enclosing_function_ref'),'template_environment':context.get('template_environment',[]),
                'lexical_access':context.get('lexical_access'),'source_preprocessor_conditions':context.get('preprocessor_conditions',[]),
                'namespace_choices':context.get('namespace_choices',{}),'scope_reviews':context.get('scope_reviews',[])}
            interface={'using_interface_id':identifier,'source_using_id':source['source_using_id'],
                'source_range':span,'display_name':source['raw_signature'],'semantic_form':semantic,
                'lexical_context':lexical,'introduced_lookup':_lookup(source,context,semantic,ctor),
                'target_expression':{'source_expression':source.get('target_source_expression'),'source_range':source.get('target_source_range'),
                    'has_typename_keyword':bool(source.get('typename_range')),'typename_range':source.get('typename_range'),
                    'root_qualified_in_source':bool(source.get('leading_global_scope_range')),'expansion_status':'not_attempted'},
                'source_contract_status':'pending'if blockers else'source_syntax_and_context_verified',
                'blockers':blockers,'target_resolution':{'status':'not_attempted','targets':[],
                    'membership_completeness':{'complete_for_bound_context':False}},
                'relationship_obligations':[{'id':identity('using_lookup_',[identifier]),'kind':'using_target_lookup',
                    'status':'pending_phase2','lookup_phase':'declaration_or_instantiation_and_use_site'}]}
            if expansion:
                interface['target_expression'].update(expansion=expansion,expansion_status=expansion.get('source_contract_status'),
                    context_condition_policy='Each target alternative remains guarded by its recorded evaluation conditions and the lexical context conditions; no independent Cartesian binding is asserted')
            if semantic=='namespace_directive':interface['nominated_namespace_expression']=source['target_name_source_expression']
            if ctor:interface['constructor_inheritance']=ctor
            if source.get('typename_range'):interface['requires_type_target']=True
            if identifier not in {i['using_interface_id']for i in interfaces}:interfaces.append(interface)
            source['using_interface_refs'].append(identifier)
            if blockers:diagnostics.append({'category':'using_interface_source_contract_pending',**span,
                'source_using_id':source['source_using_id'],'message':'; '.join(blockers)})
        source.setdefault('source_contract_status','source_interfaces_recorded')
        sources.append(source)
    return {'sources':sources,'interfaces':interfaces,'diagnostics':diagnostics}
