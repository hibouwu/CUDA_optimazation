"""Check recognized namespace/type closing bounds without repairing identities.

The input source and Tree-sitter tree MUST use the same byte coordinate space.
An equal-length annotation mask may be parsed while original bytes are supplied.
For a length-changing projection, call this on that projection and map the
returned ranges back with the caller's existing source map. No C++ configuration
is selected. Only unique literal-brace closure across all considered branches
can produce a blocking mismatch; uncertain condition/macro structure cannot.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import re

SCOPE_TYPES={'namespace_definition','class_specifier','struct_specifier','union_specifier'}
RAW_START=re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')
NUMBER=re.compile(r"[0-9][A-Za-z_0-9'.]*")

def _nodes(root):
    stack=[root]
    while stack:
        node=stack.pop();yield node
        stack.extend(reversed(node.named_children))

def _id(prefix,*values):
    return prefix+hashlib.sha256(repr(values).encode()).hexdigest()[:24]

def _mask_literals(text):
    """Byte-transparent lexical shield, independent of the declaration parser."""
    masked=list(text);issues=[];i=0;n=len(text)
    def hide(a,b):
        for k in range(a,b):
            if masked[k] not in '\r\n':masked[k]=' '
    while i<n:
        start=i
        if text.startswith('//',i):
            end=i+2
            while True:
                newline=text.find('\n',end)
                if newline<0:end=n;break
                before=newline-1 if newline and text[newline-1]=='\r' else newline
                if before>start and text[before-1]=='\\':end=newline+1;continue
                end=newline;break
            hide(i,end);i=end;continue
        if text.startswith('/*',i):
            end=text.find('*/',i+2)
            if end<0:issues.append((i,n,'unterminated_block_comment'));end=n
            else:end+=2
            hide(i,end);i=end;continue
        raw=RAW_START.match(text,i) if i==0 or not(text[i-1].isalnum() or text[i-1]=='_') else None
        if raw:
            terminator=')'+raw[1]+'"';end=text.find(terminator,raw.end())
            if end<0:issues.append((i,n,'unterminated_raw_string'));end=n
            else:end+=len(terminator)
            hide(i,end);i=end;continue
        if text[i].isdigit():
            match=NUMBER.match(text,i)
            if match:i=match.end();continue  # apostrophe digit separators are not char literals
        if text[i] in '\"\'':
            quote=text[i];i+=1;closed=False
            while i<n:
                if text[i]=='\\':
                    i+=3 if text.startswith('\\\r\n',i) else 2
                elif text[i]==quote:i+=1;closed=True;break
                elif text[i] in '\r\n':break
                else:i+=1
            if not closed:issues.append((start,min(i,n),'unterminated_quoted_literal'))
            hide(start,min(i,n));continue
        i+=1
    return ''.join(masked),issues

class _Source:
    def __init__(self,path,source):
        self.path=path;self.bytes=source;self.text=source.decode('latin1')
        self.line_starts=[0]+[m.end() for m in re.finditer('\n',self.text)]
        visible,self.lexical_issues=_mask_literals(self.text)
        self.directives=[];self.scope_macros={};code=list(visible)
        lines=self.text.splitlines(keepends=True);vlines=visible.splitlines(keepends=True)
        offsets=[];cursor=0
        for line in lines:offsets.append(cursor);cursor+=len(line)
        index=0
        while index<len(lines):
            match=re.match(r'[ \t]*#[ \t]*([A-Za-z_]\w*)',vlines[index])
            if not match:index+=1;continue
            first=index;start=offsets[first]
            while index+1<len(lines) and lines[index].rstrip('\r\n').endswith('\\'):index+=1
            end=offsets[index]+len(lines[index]);raw=self.text[start:end]
            expression=re.sub(r'^[ \t]*#[ \t]*[A-Za-z_]\w*','',raw,count=1)
            expression=re.sub(r'\\\r?\n','',expression).strip()
            directive={'command':match[1],'expression':expression,**self.span(start,end)}
            self.directives.append(directive)
            if match[1]=='define':
                logical=re.sub(r'\\\r?\n','',visible[start:end])
                definition=re.match(r'[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(?:\([^\n]*?\))?(.*)',logical,re.S)
                if definition:
                    braces=re.findall(r'[{}]',definition[2]);depth=minimum=0
                    for brace in braces:depth+=1 if brace=='{' else -1;minimum=min(minimum,depth)
                    if depth or minimum<0:self.scope_macros[definition[1]]=directive
            for pos in range(start,end):
                if code[pos] not in '\r\n':code[pos]=' '
            index+=1
        self.code=''.join(code)
        self.structure_issues=[];self.groups=[];self.items=self._structure()

    def span(self,a,b):
        return {'path':self.path,'start_byte':a,'end_byte':b,
                'start_line':bisect_right(self.line_starts,a),'end_line':bisect_right(self.line_starts,max(a,b-1))}

    def _structure(self):
        root=[];current=root;stack=[]
        tokens=[(d['start_byte'],'directive',d) for d in self.directives]
        tokens.extend((m.start(),'brace',m[0]) for m in re.finditer(r'[{}]',self.code));tokens.sort(key=lambda x:x[0])
        for position,kind,value in tokens:
            if kind=='brace':current.append({'kind':'brace','position':position,'token':value});continue
            command=value['command']
            if command in {'if','ifdef','ifndef'}:
                group={'kind':'conditional','id':_id('scope-pp:',self.path,position),
                    'start_byte':position,'end_byte':len(self.bytes),'opening':value,'branches':[],'complete':False}
                branch={'id':group['id']+'.branch0','directive':value,'start_byte':value['end_byte'],
                        'end_byte':len(self.bytes),'items':[],'implicit':False}
                group['branches'].append(branch);current.append(group);self.groups.append(group)
                stack.append((group,current));current=branch['items']
            elif command in {'else','elif'}:
                if not stack:self.structure_issues.append((position,len(self.bytes),'unmatched_'+command));continue
                group,parent=stack[-1]
                if any(b['directive']['command']=='else' for b in group['branches']):
                    self.structure_issues.append((position,len(self.bytes),'branch_after_else'))
                group['branches'][-1]['end_byte']=position
                branch={'id':group['id']+'.branch'+str(len(group['branches'])),'directive':value,
                        'start_byte':value['end_byte'],'end_byte':len(self.bytes),'items':[],'implicit':False}
                group['branches'].append(branch);current=branch['items']
            elif command=='endif':
                if not stack:self.structure_issues.append((position,len(self.bytes),'unmatched_endif'));continue
                group,parent=stack.pop();group['branches'][-1]['end_byte']=position
                if not any(b['directive']['command']=='else' for b in group['branches']):
                    group['branches'].append({'id':group['id']+'.implicit_else','directive':value,
                        'start_byte':position,'end_byte':position,'items':[],'implicit':True})
                group.update(end_byte=value['end_byte'],closing=value,complete=True);current=parent
        for group,_ in stack:self.structure_issues.append((group['start_byte'],len(self.bytes),'unclosed_conditional_group'))
        return root

def _continuation(items,opening,conditions=()):
    """Select only the containing branch when the scope itself starts in #if."""
    for i,item in enumerate(items):
        if item['kind']=='brace' and item['position']==opening:
            return items[i+1:],list(conditions)
        if item['kind']=='conditional' and item['start_byte']<=opening<item['end_byte']:
            for branch in item['branches']:
                if branch['start_byte']<=opening<branch['end_byte']:
                    condition={'group_id':item['id'],'branch_id':branch['id'],'directive':branch['directive']}
                    found=_continuation(branch['items'],opening,conditions+(condition,))
                    if found:return found[0]+items[i+1:],found[1]
    return None

def _closures(items,depths,*,seen_groups,budget):
    remaining=set(depths);closed=set();limited=False
    for item in items:
        if not remaining:break
        budget[0]-=1
        if budget[0]<0:return remaining,closed,True
        if item['kind']=='brace':
            next_depths=set()
            for depth in remaining:
                result=depth+(1 if item['token']=='{' else -1)
                if result==0:closed.add(item['position'])
                elif result>0:next_depths.add(result)
            remaining=next_depths
        else:
            seen_groups.add(item['id']);next_depths=set()
            for branch in item['branches']:
                after,closes,hit_limit=_closures(branch['items'],remaining,seen_groups=seen_groups,budget=budget)
                next_depths|=after;closed|=closes;limited|=hit_limit
            remaining=next_depths
    return remaining,closed,limited

def analyze_scope_integrity(path,source,tree_or_root,*,known_scope_macros=(),max_steps=200000):
    """Return evidence, not repaired scopes. All ranges use input source bytes.

    diagnostics contains only proven literal-closing mismatches and blocks phase1.
    uncertainties records unchecked conditional/macro/lexical structure without
    asserting invalid source. Callers must not treat uncertainty as verified scope.
    Only parser-recognized namespace/type bodies are checked; absent scopes and
    unresolved external macro effects remain separate declaration obligations.
    """
    view=_Source(path,source);root=getattr(tree_or_root,'root_node',tree_or_root)
    scopes=[];diagnostics=[];uncertainties=[]
    groups={g['id']:g for g in view.groups}
    if root.end_byte>len(source):
        return {'schema_version':1,'path':path,'source_sha256':hashlib.sha256(source).hexdigest(),
            'coordinate_space':'input_source_bytes','scopes':[],'diagnostics':[],
            'uncertainties':[{'category':'scope_integrity_input_coordinates_pending','blocks_phase_1':False,
                'requires_scope_review':True,**view.span(0,len(source)),
                'message':'Parse tree extends beyond supplied source. Supply bytes in the actual parse coordinate layer before comparing scope bounds.',
                'tree_end_byte':root.end_byte}],
            'summary':{'recognized_scope_bodies':0,'verified_bounds':0,'blocking_mismatches':0,'uncertain_scopes':0,'input_rejected':True}}
    for node in _nodes(root):
        if node.type not in SCOPE_TYPES:continue
        body=node.child_by_field_name('body')
        if body is None:continue
        name=node.child_by_field_name('name')
        name_text=source[name.start_byte:name.end_byte].decode('utf-8','replace') if name is not None else None
        opening=body.start_byte;parsed_end=body.end_byte
        base={'scope_id':_id('scope-integrity:',path,node.type,opening),'kind':node.type,'name':name_text,
              'declaration_range':view.span(node.start_byte,node.end_byte),
              'parsed_body_range':view.span(opening,parsed_end),'parser_node_has_error':node.has_error}
        closing_tokens=[child for child in body.children if child.type=='}']
        base['parsed_closing_token']=({**view.span(closing_tokens[-1].start_byte,closing_tokens[-1].end_byte),
            'is_missing':closing_tokens[-1].is_missing} if closing_tokens else None)
        reasons=[];expected=None;conditions=[];seen=set();remaining=set();closes=set()
        if root.end_byte>len(source) or not(0<=opening<len(source)) or source[opening:opening+1]!=b'{':
            reasons.append({'reason':'tree_source_coordinate_or_literal_opening_unverified'})
        else:
            found=_continuation(view.items,opening)
            if not found:reasons.append({'reason':'literal_scope_opening_not_found_in_lexical_structure'})
            else:
                items,conditions=found
                remaining,closes,limited=_closures(items,{1},seen_groups=seen,budget=[max_steps])
                if limited:reasons.append({'reason':'conditional_brace_analysis_budget_exceeded','max_steps':max_steps})
                if len(closes)==1 and not remaining and not limited:expected=next(iter(closes))
                else:reasons.append({'reason':'conditional_closure_not_unique' if seen else 'physical_closure_unproven',
                                     'candidate_closing_bytes':sorted(closes),'unclosed_depths':sorted(remaining)})
        limit=max(parsed_end,(expected+1) if expected is not None else len(source))
        for a,b,reason in view.lexical_issues+view.structure_issues:
            if a<limit and b>opening:reasons.append({'reason':reason,'source':view.span(a,b)})
        for macro in set(known_scope_macros)|set(view.scope_macros):
            match=re.search(r'\b'+re.escape(macro)+r'\b',view.code[opening:limit])
            if match:reasons.append({'reason':'unexpanded_scope_changing_macro','name':macro,
                'source':view.span(opening+match.start(),opening+match.end()),'definition':view.scope_macros.get(macro)})
        base.update(opening_brace=view.span(opening,opening+1),enclosing_conditions=conditions)
        base['candidate_closing_braces']=[view.span(p,p+1) for p in sorted(closes)]
        base['conditional_groups_considered']=[{'id':gid,'range':view.span(groups[gid]['start_byte'],groups[gid]['end_byte']),
            'branches':[{'id':b['id'],'directive':b['directive'],'implicit':b['implicit']} for b in groups[gid]['branches']]} for gid in sorted(seen)]
        base['parsed_closing_brace']=view.span(parsed_end-1,parsed_end) if 0<parsed_end<=len(source) and source[parsed_end-1:parsed_end]==b'}' else None
        if reasons:
            base.update(status='unverified_conditional_or_lexical_structure',trust_scope_bounds=False,reasons=reasons)
            uncertainties.append({'category':'scope_integrity_uncertain','blocks_phase_1':False,'requires_scope_review':True,
                'message':'No unique reliable physical scope closure proved; this is not a source-invalidity claim.',
                **view.span(opening,min(limit,len(source))),**base})
        else:
            expected_end=expected+1;base['expected_closing_brace']=view.span(expected,expected_end)
            base['expected_body_range']=view.span(opening,expected_end)
            if parsed_end!=expected_end:
                base.update(status='proven_closing_mismatch',trust_scope_bounds=False,
                    mismatch_direction='premature_close' if parsed_end<expected_end else 'late_close',
                    affected_tail=view.span(min(parsed_end,expected_end),max(parsed_end,expected_end)))
                diagnostics.append({'category':'scope_closure_mismatch','blocks_phase_1':True,
                    'message':'Tree-sitter scope closing differs from the unique literal physical closing across all considered branches; no owner repair is performed.',
                    **view.span(opening,max(parsed_end,expected_end)),**base})
            else:base.update(status='literal_bounds_verified',trust_scope_bounds=True)
        scopes.append(base)
    return {'schema_version':1,'path':path,'source_sha256':hashlib.sha256(source).hexdigest(),
        'coordinate_space':'input_source_bytes','scopes':scopes,'diagnostics':diagnostics,'uncertainties':uncertainties,
        'summary':{'recognized_scope_bodies':len(scopes),'verified_bounds':sum(s['trust_scope_bounds'] for s in scopes),
                   'blocking_mismatches':len(diagnostics),'uncertain_scopes':len(uncertainties)},
        'policy':{'changes_owner_identity':False,'selects_preprocessor_configuration':False,
            'branch_model':'All alternatives considered independently, including implicit else; correlated conditions are conservatively over-approximated.',
            'scope_coverage':'Only parser-recognized namespace/class/struct/union bodies, not all source declarations.',
            'macro_boundary':'Locally defined unbalanced brace macros and caller-supplied known_scope_macros make the affected scope unverified; unknown external macro expansion is not claimed.',
            'uncertainty_is_source_error':False}}
