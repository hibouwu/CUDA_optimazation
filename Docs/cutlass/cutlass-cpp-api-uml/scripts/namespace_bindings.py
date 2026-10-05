"""Finite, source-grounded namespace bindings for the approved library corpus.

An unbound user namespace suffix is represented as an expression, not guessed.
Literal alternatives are materialized independently by the declaration walker.
"""
import re
import importlib.util
from pathlib import Path
import sys
_spec=importlib.util.spec_from_file_location('namespace_macro_source',Path(__file__).with_name('macro_expansion.py'))
_macros=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_macros;_spec.loader.exec_module(_macros)


def _compact(s):
    return re.sub(r'\s+', '', s)


def _definition(d):
    return {k:v for k,v in vars(d).items() if k!='declaration_candidate'}


def _literal(value, conditions, evidence, spelling):
    return {'components':value.split('::'),'conditions':conditions,
            'resolution':{'status':'literal_macro_binding','spelling':spelling,
                          'expanded_namespace':value,'definition_chain':evidence,
                          'missing_bindings':[]}}


def resolve_namespace(spelling, path, byte, source, local_defs, config_defs, conditions_at=None, seen=()):
    if spelling in seen:return {'status':'pending','reason':'Recursive namespace macro expansion'}
    # Only definitions that precede this namespace can participate. Definitions
    # after it must not retroactively rename a previously declared namespace.
    local=[d for d in local_defs if d.name==spelling and d.start_byte<byte]
    logical,positions=_macros.splice_lines(source[:byte].decode('latin1'))
    visible_source=_macros.mask_comments_literals(logical)
    undefs=[]
    for match in re.finditer(r'^\s*#\s*undef\s+'+re.escape(spelling)+r'\b',visible_source,re.M):
        start,end=_macros.physical_span(positions,match.start(),match.end(),byte)
        conditions=conditions_at(start) if conditions_at else []
        if any(c.get('constant_false')for c in conditions):continue
        if conditions:
            return {'status':'pending','reason':'Conditional namespace undef requires alternate active macro states',
                    'undef':{'path':path,'start_byte':start,'end_byte':end,'conditions':conditions}}
        undefs.append(end)
    # The header that defines an imported configuration macro can itself use
    # that macro later. Reuse the verified model only for those exact source
    # definitions, not for arbitrary same-named local replacements.
    verified_local_config = bool(local) and not undefs and spelling in {'cutlass','CUTE_STL_NAMESPACE'} and all(
        any(_definition(d) == _definition(c) for c in config_defs) for d in local)
    if local and not verified_local_config:
        # Use byte rather than codepoint offsets when comparing against defs.
        if undefs:
            cutoff=undefs[-1]
            local=[d for d in local if d.start_byte>=cutoff]
            if not local:return {'status':'not_macro'}
        unconditional=[d for d in local if not getattr(d,'preprocessor_conditions',[])]
        if unconditional and unconditional[-1] is local[-1]:
            d=unconditional[-1]
            if d.parameters is None and re.fullmatch(r'[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*',d.body.strip()):
                # Another object macro may require rescanning: do not claim a
                # literal owner while the referenced replacement is unresolved.
                other={x.name for x in local_defs if x.start_byte<byte}|{x.name for x in config_defs}
                if any(t in other for t in d.body.strip().split('::')):
                    if '::' not in d.body.strip():
                        nested=resolve_namespace(d.body.strip(),path,byte,source,local_defs,config_defs,conditions_at,seen+(spelling,))
                        if nested['status']=='resolved':
                            for alt in nested['alternatives']:
                                alt['resolution']={**alt['resolution'],'source_alias_spelling':spelling,
                                    'definition_chain':[_definition(d)]+alt['resolution'].get('definition_chain',[])}
                            return nested
                    return {'status':'pending','reason':'Namespace replacement requires further macro rescanning',
                            'definitions':[_definition(x)for x in local]}
                return {'status':'resolved','alternatives':[_literal(d.body.strip(),[],[_definition(d)],spelling)]}
        return {'status':'pending','reason':'Conditional/local namespace macro lifetime requires variant resolution',
                'definitions':[_definition(d)for d in local]}
    # A local undef must override the imported library macro environment.
    if undefs:
        return {'status':'not_macro'}

    defs=[d for d in config_defs if d.name==spelling]
    if not defs:return {'status':'not_macro'}
    if spelling=='CUTE_STL_NAMESPACE':
        by_body={d.body.strip():d for d in defs}
        if set(by_body)!={'std','cuda::std'} or any(d.path!='include/cute/config.hpp' for d in defs):
            return {'status':'pending','reason':'Unexpected CUTE_STL_NAMESPACE definitions','definitions':[_definition(d)for d in defs]}
        out=[]
        for name in ('std','cuda::std'):
            d=by_body[name]
            conditions=getattr(d,'preprocessor_conditions',[])
            if not conditions:return {'status':'pending','reason':'Missing configuration guard evidence'}
            alt=_literal(name,conditions,[_definition(d)],spelling)
            alt['resolution'].update({'binding_key':'include/cute/config.hpp::CUTE_STL_NAMESPACE','choice_key':name})
            out.append(alt)
        return {'status':'resolved','alternatives':out,'environment':'Definitions from include/cute/config.hpp'}
    if spelling=='cutlass':
        helper='include/cutlass/detail/helper_macros.hpp'
        relevant={d.name:d for d in config_defs if d.path==helper and d.name in {'cutlass','mkcutlassnamespace','concat_tok'}}
        required={'cutlass':'mkcutlassnamespace(cutlass_,CUTLASS_NAMESPACE)',
                  'mkcutlassnamespace':'concat_tok(pre,ns)','concat_tok':'a##b'}
        if set(relevant)!=set(required) or any(_compact(relevant[n].body)!=v for n,v in required.items()):
            return {'status':'pending','reason':'Unexpected namespace concatenation macro definitions'}
        evidence=[_definition(relevant[n])for n in ('cutlass','mkcutlassnamespace','concat_tok')]
        conditions=getattr(relevant['cutlass'],'preprocessor_conditions',[])
        if len(conditions)!=1 or _compact(conditions[0]['expression'])!='defined(CUTLASS_NAMESPACE)':
            return {'status':'pending','reason':'Unexpected namespace configuration guard'}
        default_guard=[{**conditions[0],'expression':'!defined(CUTLASS_NAMESPACE)','derived_from':'No cutlass object macro is defined when the guarded block is inactive'}]
        parameter={'components':['<cutlass_namespace(CUTLASS_NAMESPACE)>'],'conditions':conditions,
                   'resolution':{'status':'parameterized_macro_binding','spelling':spelling,
                     'binding_key':helper+'::cutlass','choice_key':'custom_namespace',
                     'scope_expression':'rescan(token_paste("cutlass_", expand(CUTLASS_NAMESPACE)))',
                     'definition_chain':evidence,'missing_bindings':['CUTLASS_NAMESPACE replacement tokens and any macros reached by argument/result rescanning'],
                     'requirements':['Final preprocessing output must be a valid C++ namespace name'],
                     'is_literal_namespace':False}}
        default=_literal('cutlass',default_guard,evidence,spelling)
        default['resolution'].update({'binding_key':helper+'::cutlass','choice_key':'default_namespace'})
        return {'status':'resolved','alternatives':[default,parameter],
                'environment':'Namespace customization defined by include/cutlass/detail/helper_macros.hpp'}
    return {'status':'pending','reason':'No verified binding model for configuration macro','definitions':[_definition(d)for d in defs]}
