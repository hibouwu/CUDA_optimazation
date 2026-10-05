"""Source-mapped expansion of local declaration macros, not a full preprocessor.

Each successful result preserves the original invocation and definition. Any
ambiguous/unsupported case remains a diagnostic; no candidate is silently lost.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import re
from tree_sitter import Language, Parser
import tree_sitter_cpp

_DECLARATION_PARSER = Parser(Language(tree_sitter_cpp.language()))


@dataclass
class MacroDefinition:
    name: str
    parameters: list[str] | None
    body: str
    path: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    declaration_candidate: bool
    replacement_classification: str = 'unclassified'
    declaration_evidence: tuple[str, ...] = ()
    requires_callsite_semicolon: bool = False


def _all_nodes(node):
    stack=[node]
    while stack:
        current=stack.pop();yield current
        stack.extend(reversed(current.children))


@lru_cache(maxsize=4096)
def classify_replacement(body, parameters=()):
    """Classify source form, not declaration meaning or preprocessor validity.

    Primitive/user-type declarators and multiple declarations are recognized by
    the C++ grammar. Clear statement blocks do not become declaration macros
    merely because they contain local variables. Existing template/type/operator
    macros remain candidates even when a dependent macro obstructs parsing.
    """
    visible=mask_comments_literals(body).strip()
    if not visible:return (False,'empty_or_literal',(),False)
    named_type_marker=bool(re.search(r'\b(?:class|struct|union|enum|typedef|using)\b',visible))
    if re.match(r'^(?:do|if|for|while|switch|return|throw|break|continue)\b',visible) or visible.startswith('{'):
        if named_type_marker:return (True,'statement_with_nested_named_declarations',('Named declaration occurs inside a statement; caller scope must be preserved',),False)
        return (False,'statement_or_expression',('Leading control/return/block statement; local variables are not independent APIs',),False)
    if re.match(r'^static_assert\s*\(',visible):
        return (False,'static_assertion',('Compile-time assertion, not a named API declaration',),False)
    # Classification only: CUDA annotations are not type names. The actual
    # expansion retains every original token and its normal source provenance.
    probe=body
    for annotation in re.finditer(r'\b(?:CUTE_HOST_DEVICE|CUTE_HOST|CUTE_DEVICE|CUTLASS_HOST_DEVICE|CUTLASS_HOST|CUTLASS_DEVICE|CUTLASS_GLOBAL|__host__|__device__|__global__|__forceinline__)\b',mask_comments_literals(body)):
        probe=probe[:annotation.start()]+' '*(annotation.end()-annotation.start())+probe[annotation.end():]
    tree=_DECLARATION_PARSER.parse(probe.encode('latin1'))
    nodes=[n for n in tree.root_node.named_children if n.type!='comment']
    declaration_types={'declaration','field_declaration','function_definition','template_declaration','class_specifier','struct_specifier','union_specifier','enum_specifier','alias_declaration','type_definition','namespace_definition'}
    declared=[n for n in nodes if n.type in declaration_types and (n.type!='declaration' or n.child_by_field_name('declarator')is not None)]
    if declared and all(n.type in declaration_types|{'static_assert_declaration'}for n in nodes):
        errors=[n for n in _all_nodes(tree.root_node)if n.type=='ERROR'or n.is_missing]
        suffix=bool(errors) and all(n.is_missing and n.type==';'and n.start_byte==len(probe.encode('latin1'))for n in errors)
        # Annotation/type fragments such as '__device__ inline' must not become
        # object declarations. A missing semicolon is accepted only for an
        # explicit primitive type or a formal type/declarator macro parameter.
        if suffix and not (';'in visible or re.match(r'^(?:const\s+|volatile\s+|static\s+|constexpr\s+)*(?:void|bool|char|short|int|long|float|double|auto|unsigned|signed)\b',visible) or any(re.search(r'\b'+re.escape(p)+r'\b',visible)for p in parameters)):
            return (False,'declaration_fragment',('Incomplete spelling has no proven complete declarator',),False)
        return (True,'declaration_sequence',tuple(n.type for n in declared),suffix)
    # Token paste cannot be parsed before CPP replacement, but the declaration
    # prefix/terminator is still explicit. Actual substitution and parsing are
    # performed downstream; unsupported expansion remains a diagnostic.
    if re.search(r'\b(?:template|class|struct|union|enum|typedef|using|operator)\b|\)\s*(?:const\s*)?\{',visible):
        return (True,'declaration_syntax_candidate',('Explicit type/template/operator/function-body syntax; dependent grammar may remain',),False)
    if re.match(r'^(?:const\s+|volatile\s+|static\s+|constexpr\s+)*(?:void|bool|char|short|int|long|float|double|auto|unsigned|signed)\b',visible) and ';'in visible:
        return (True,'declaration_syntax_candidate',('Explicit primitive declaration prefix and terminator, possibly token-pasted',),False)
    if any(p!='...'and re.match(r'^'+re.escape(p)+r'\s*\(',visible)for p in parameters) or visible in set(parameters)|{'__VA_ARGS__'}:
        return (True,'argument_dependent_syntax',('Formal replacement can be a type declaration or expression; invocation must be checked',),False)
    return (False,'statement_or_expression',('No named declaration production or explicit declaration marker',),False)


def splice_lines(source):
    """CPP phase 2 precedes comment recognition; retain physical byte mapping."""
    chars=[];positions=[];i=0
    while i<len(source):
        if source.startswith('\\\r\n',i):i+=3;continue
        if source.startswith('\\\n',i):i+=2;continue
        chars.append(source[i]);positions.append(i);i+=1
    return ''.join(chars),positions


def physical_span(positions,start,end,source_length):
    a=positions[start] if start<len(positions)else source_length
    b=positions[end-1]+1 if end>start else a
    return a,b


def mask_comments_literals(source):
    # ASCII-transparent latin1 text keeps byte offsets even in UTF-8 comments.
    pattern = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|R"([^ ()\\\t\r\n]{0,16})\([\s\S]*?\)\1"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
    return pattern.sub(lambda m: ''.join('\n' if c=='\n' else ' ' for c in m[0]), source)


def definitions(path, source: bytes):
    physical=source.decode('latin1')
    s,positions=splice_lines(physical)
    visible = mask_comments_literals(s)
    defs = []
    pattern = re.compile(r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\([^\n]*?\))?((?:[^\n]*\\\r?\n)*[^\n]*)', re.M)
    for m in pattern.finditer(s):
        if not visible[m.start():m.end()].lstrip().startswith('#'):
            continue
        # A continuation belongs to the replacement list, not a new source API.
        body = re.sub(r'\\\r?\n', '', m[3]).strip()
        params = None if m[2] is None else [x.strip() for x in m[2][1:-1].split(',') if x.strip()]
        candidate,classification,evidence,suffix=classify_replacement(body,tuple(params or ()))
        if m[1]=='static_assert'and params==['__e','__m']and re.fullmatch(r'typedef\s+int\s+__platform_cat\(AsSeRt,\s*__LINE__\)\s*\[\s*\(__e\)\s*\?\s*1\s*:\s*-1\s*\]',body):
            candidate=False;classification='assertion_emulation'
            evidence=('Fixed platform static_assert fallback uses a line-named typedef solely to reject a false predicate',)
            suffix=False
        start,end=physical_span(positions,m.start(),m.end(),len(physical))
        defs.append(MacroDefinition(m[1],params,body,path,start,end,physical.count('\n',0,start)+1,physical.count('\n',0,end)+1,candidate,classification,evidence,suffix))
    # A wrapper of a known declaration macro is itself a declaration candidate.
    # It is not silently treated as an ordinary type/call if nested expansion is
    # beyond this module's supported direct-expansion boundary.
    changed=True
    while changed:
        changed=False;known={d.name for d in defs if d.declaration_candidate}
        for definition in defs:
            called=set(re.findall(r'\b([A-Za-z_]\w*)\s*\(',mask_comments_literals(definition.body)))
            if not definition.declaration_candidate and called&known:
                definition.declaration_candidate=True
                definition.replacement_classification='nested_declaration_macro'
                definition.declaration_evidence=('Invokes declaration macro(s): '+', '.join(sorted(called&known)),)
                changed=True
    return defs


def arguments(text, opening):
    """Split CPP arguments: only parentheses protect commas, not angle brackets."""
    assert text[opening]=='('
    masked=mask_comments_literals(text)
    depth=1;start=opening+1;args=[]
    for i in range(opening+1,len(text)):
        c=masked[i]
        if c=='(':depth+=1
        elif c==')':
            depth-=1
            if depth==0:
                if text[start:i].strip() or args:args.append(text[start:i].strip())
                return args,i+1
        elif c==',' and depth==1:
            args.append(text[start:i].strip());start=i+1
    raise ValueError('unclosed macro invocation')


def substitute(definition, args):
    params=definition.parameters
    if params is None:raise ValueError('not a function-like macro')
    variadic=bool(params and params[-1]=='...')
    if any('...' in p for p in params[:-1]) or (params and '...' in params[-1] and not variadic):raise ValueError('GNU named variadic macro unsupported')
    required=len(params)-int(variadic)
    if len(args)<required or (not variadic and len(params)!=len(args)):raise ValueError(f'argument count {len(args)} incompatible with {params}')
    bindings=dict(zip(params[:required],args[:required]))
    if variadic:bindings['__VA_ARGS__']=', '.join(args[required:])
    body=definition.body
    if '__VA_OPT__' in body:raise ValueError('__VA_OPT__ requires dedicated expansion')
    token=re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|R"([^ ()\\\t\r\n]{0,16})\([\s\S]*?\)\1"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|##|[A-Za-z_]\w*|\s+|.',re.S)
    toks=[m[0]for m in token.finditer(body)]
    expanded=[];i=0
    while i<len(toks):
        t=toks[i]
        if t.startswith(('//','/*')):expanded.append(' ');i+=1;continue
        if t=='#':
            j=i+1
            while j<len(toks) and toks[j].isspace():j+=1
            if j<len(toks) and toks[j] in bindings:
                value_tokens=[x[0]for x in token.finditer(bindings[toks[j]])]
                pieces=[];space=False
                for value in value_tokens:
                    if value.isspace() or value.startswith(('//','/*')):space=True;continue
                    if space and pieces:pieces.append(' ')
                    pieces.append(value);space=False
                v=''.join(pieces).replace('\\','\\\\').replace('"','\\"')
                expanded.append('"'+v+'"');i=j+1;continue
        expanded.append(bindings.get(t,t));i+=1
    # Paste only preprocessing ## tokens, never text inside a string literal.
    while '##' in expanded:
        pos=expanded.index('##');left=pos-1;right=pos+1
        while left>=0 and expanded[left].isspace():left-=1
        while right<len(expanded) and expanded[right].isspace():right+=1
        if left<0 or right>=len(expanded):raise ValueError('token paste missing operand')
        expanded[left:right+1]=[expanded[left]+expanded[right]]
    result=''.join(expanded)
    return result,bindings


def expand_file(path, source: bytes):
    defs=definitions(path,source);physical=source.decode('latin1');s,positions=splice_lines(physical);masked=mask_comments_literals(s)
    ranges=[(d.start_byte,d.end_byte)for d in defs]
    out=[];diagnostics=[];non_declarations=[]
    known={d.name for d in defs if d.declaration_candidate and d.parameters is not None}
    for m in re.finditer(r'\b([A-Za-z_]\w*)\s*\(',masked):
        name=m[1]
        start_byte,_=physical_span(positions,m.start(),m.end(),len(physical))
        if name not in known or any(a<=start_byte<b for a,b in ranges):continue
        # Ignore all directive lines; their existence/conditions are captured by
        # the independent source-candidate scanner.
        line_start=s.rfind('\n',0,m.start())+1
        if s[line_start:m.start()].lstrip().startswith('#'):continue
        opening=masked.find('(',m.start(),m.end())
        try:args,end=arguments(s,opening)
        except ValueError as err:
            diagnostics.append({'kind':'macro_unclosed','path':path,'start_byte':start_byte,'end_byte':len(physical),'message':str(err)});continue
        _,end_byte=physical_span(positions,m.start(),end,len(physical))
        prior=[d for d in defs if d.name==name and d.start_byte<start_byte]
        if not prior:continue  # A later definition does not retroactively expand this spelling.
        # Multiple definitions (usually conditional) cannot be resolved by
        # choosing the last textual definition. Preserve the uncertainty.
        bodies={(tuple(d.parameters or []),d.body)for d in prior}
        invocation_id='macro-call:'+hashlib.sha256(f'{path}:{start_byte}:{name}'.encode()).hexdigest()[:24]
        if len(bodies)!=1:
            diagnostics.append({'kind':'macro_definition_ambiguous','invocation_id':invocation_id,'path':path,'start_byte':start_byte,'end_byte':end_byte,'name':name,'definitions':[asdict(d)for d in prior]});continue
        d=prior[-1]
        between=splice_lines(physical[d.end_byte:start_byte])[0]
        if re.search(r'^\s*#\s*undef\s+'+re.escape(name)+r'\b',between,re.M):
            diagnostics.append({'kind':'macro_definition_lifetime_unresolved','invocation_id':invocation_id,'path':path,'start_byte':start_byte,'end_byte':end_byte,'name':name});continue
        # Prescan/rescan can change API names. Until fully source-mapped, these
        # cases must remain pending rather than materializing a wrong entity.
        object_names={x.name for x in defs if x.parameters is None and x.start_byte<start_byte}
        parameter_macros=sorted(set(re.findall(r'\b[A-Za-z_]\w*\b',mask_comments_literals(','.join(args)))) & object_names)
        if parameter_macros:
            diagnostics.append({'kind':'macro_argument_prescan_pending','invocation_id':invocation_id,'path':path,'start_byte':start_byte,'end_byte':end_byte,'name':name,'macro_dependencies':parameter_macros});continue
        try:expanded,bindings=substitute(d,args)
        except ValueError as err:
            diagnostics.append({'kind':'macro_expansion_unsupported','invocation_id':invocation_id,'path':path,'start_byte':start_byte,'end_byte':end_byte,'name':name,'message':str(err)});continue
        rescan_macros=sorted(set(re.findall(r'\b[A-Za-z_]\w*\b',mask_comments_literals(expanded))) & object_names)
        if rescan_macros:
            diagnostics.append({'kind':'macro_result_rescan_pending','invocation_id':invocation_id,'path':path,'start_byte':start_byte,'end_byte':end_byte,'name':name,'macro_dependencies':rescan_macros});continue
        nested_names=[]
        for nm in re.finditer(r'\b([A-Za-z_]\w*)\s*\(',mask_comments_literals(expanded)):
            if nm[1] in known:nested_names.append(nm[1])
        if nested_names:
            diagnostics.append({'kind':'nested_declaration_macro_pending','invocation_id':invocation_id,'path':path,'start_byte':start_byte,'end_byte':end_byte,'name':name,'nested_macros':sorted(set(nested_names))});continue
        actual_candidate,actual_kind,actual_evidence,needs_suffix=classify_replacement(expanded)
        if d.replacement_classification=='argument_dependent_syntax'and not actual_candidate:
            diagnostics.append({'kind':'macro_declaration_context_pending','invocation_id':invocation_id,'path':path,'start_byte':start_byte,'end_byte':end_byte,'name':name,
                                'message':'Argument-dependent replacement does not establish declaration-vs-expression without name/type lookup',
                                'expanded_source':expanded.encode('latin1').decode('utf-8'),'classification':actual_kind});continue
        callsite_suffix=None
        macro_invocation_end_byte=end_byte
        if d.requires_callsite_semicolon or needs_suffix:
            tail=re.match(r'\s*;',masked[end:])
            if tail is None:
                diagnostics.append({'kind':'macro_declaration_terminator_pending','invocation_id':invocation_id,'path':path,'start_byte':start_byte,'end_byte':end_byte,'name':name,
                                    'message':'Declaration replacement requires a semicolon supplied by the callsite; none was found'});continue
            semicolon=end+tail.end()-1
            suffix_start,suffix_end=physical_span(positions,semicolon,semicolon+1,len(physical))
            callsite_suffix={'path':path,'start_byte':suffix_start,'end_byte':suffix_end,'spelling':';',
                             'reason':'Physical callsite terminator completes the macro-generated declaration'}
            expanded+=';'
            end_byte=suffix_end
        out.append({'invocation_id':invocation_id,'path':path,'name':name,'start_byte':start_byte,'end_byte':end_byte,
            'start_line':physical.count('\n',0,start_byte)+1,'end_line':physical.count('\n',0,end_byte)+1,
            'definition':asdict(d),'bindings':bindings,'virtual_source':expanded.encode('latin1').decode('utf-8'),
            'replacement_classification':actual_kind,'declaration_evidence':actual_evidence,
            'macro_invocation_end_byte':macro_invocation_end_byte,'callsite_suffix':callsite_suffix,
            'source_mapping':'generated_tokens_reference_definition_and_invocation_not_physical_expanded_offsets'})
    # Object-like declaration macros are not expanded by this direct function-
    # macro engine. Unlike before, each use is now an explicit pending target.
    object_candidates={d.name for d in defs if d.declaration_candidate and d.parameters is None}
    function_statements={d.name for d in defs if not d.declaration_candidate and d.parameters is not None}-known
    for match in re.finditer(r'\b([A-Za-z_]\w*)\b',masked):
        name=match[1]
        if name not in object_candidates|function_statements:continue
        start,end=physical_span(positions,match.start(),match.end(),len(physical))
        if any(a<=start<b for a,b in ranges):continue
        line_start=s.rfind('\n',0,match.start())+1
        if s[line_start:match.start()].lstrip().startswith('#'):continue
        prior=[d for d in defs if d.name==name and d.start_byte<start]
        if not prior:continue
        if name in object_candidates:
            diagnostics.append({'kind':'object_declaration_macro_pending','path':path,'start_byte':start,'end_byte':end,'name':name,
                                'message':'Known object-like declaration macro requires source-mapped expansion; not accepted as a literal type/name',
                                'definitions':[asdict(d)for d in prior]})
            continue
        opening=re.match(r'\s*\(',masked[match.end():])
        if not opening:continue
        try:_,logical_end=arguments(s,match.end()+opening.end()-1)
        except ValueError:continue
        _,end=physical_span(positions,match.start(),logical_end,len(physical))
        non_declarations.append({'path':path,'start_byte':start,'end_byte':end,'name':name,
                                 'start_line':physical.count('\n',0,start)+1,'end_line':physical.count('\n',0,end)+1,
                                 'classification':'no_named_api_declaration_in_any_known_replacement',
                                 'definitions':[asdict(d)for d in prior],
                                 'scope_note':'Statement/assertion expressions remain in original source for later call/state analysis'})
    return {'definitions':[asdict(d)for d in defs],'expansions':out,'diagnostics':diagnostics,'non_declaration_invocations':non_declarations,
            'coverage_claim':'local_single_definition_direct_macros_only; remaining declaration candidates require separate accounting'}


def constraint_expansions(path, source, traits_source):
    """Actual replacements for the two unconditional CuTe SFINAE macros.

    Returns edits; the caller must preserve a source map when parsing their
    expanded projection. These are constraints, not whitespace annotations.
    """
    trait_path='include/cute/util/type_traits.hpp'
    known={d.name:d for d in definitions(trait_path,traits_source) if d.name in ('__CUTE_REQUIRES','__CUTE_REQUIRES_V')}
    s=source.decode('latin1');visible=mask_comments_literals(s)
    def_ranges=[(d.start_byte,d.end_byte)for d in definitions(path,source)]
    edits=[]
    for m in re.finditer(r'\b(__CUTE_REQUIRES_V|__CUTE_REQUIRES)\s*\(',visible):
        if any(a<=m.start()<b for a,b in def_ranges):continue
        args,end=arguments(s,visible.find('(',m.start(),m.end()))
        if m[1] not in known:raise ValueError('Missing verified CuTe constraint definition')
        d=known[m[1]];expanded,bindings=substitute(d,args)
        edits.append({'path':path,'name':m[1],'start_byte':m.start(),'end_byte':end,
            'spelling':s[m.start():end].encode('latin1').decode('utf-8'),
            'expanded':expanded.encode('latin1').decode('utf-8'),'definition':asdict(d),'bindings':bindings,
            'transformation':'source_mapped_constraint_expansion_not_annotation_mask'})
    return edits
