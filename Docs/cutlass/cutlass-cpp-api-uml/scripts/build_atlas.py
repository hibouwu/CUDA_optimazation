#!/usr/bin/env python3
"""Build the offline modular atlas from reviewed relation data and the fixed ledger.

No CDN, server API, or runtime fetch is required. Compact view diagrams are
navigation only; each relationship also gets a separate full-signature UML.
Unresolved node selectors remain explicit issues, never invented declarations.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from urllib.parse import quote

from reconcile_candidates import top_items, LEXER, file_sha
from protocol_diagrams import render_protocols

ROOT=Path(__file__).resolve().parents[1]
KINDS={'api':{'function','method','operator','constructor','destructor'},
       'type':{'class_specifier','struct_specifier','union_specifier','alias','typedef','enum','namespace_alias'},
       'resource':{'member','member_constant','variable','constant','variable_template'}}
RELATIONS={'calls':'调用','launches':'设备启动','type_uses':'类型依赖','template_binds':'模板绑定',
    'specializes':'特化选择','reads':'读取','writes':'写入','signals':'完成通知','waits_for':'等待',
    'permits_reuse':'允许复用','inherits':'继承','aliases':'别名','precedes':'先于',
    'requires_completed_event':'以前置完成事件为条件','instance_of':'配置实例','issues':'指令发起','submits':'异步提交','evaluates':'表达式求值',
    'macro_uses':'宏使用','expands_to':'条件宏展开','has_attribute':'声明属性','member_of':'作用域归属',
    'implicit_calls':'语言规则展开的调用','has_member':'成员包含'}
ARROWS={'calls':'-->','launches':'-[#9a3e9c,bold]->','template_binds':'..>','specializes':'..>',
        'type_uses':'..>','reads':'-[#237d78]->','writes':'-[#237d78]->','signals':'-[#b27616,bold]->',
        'waits_for':'-[#b27616]->','permits_reuse':'-[#347444]->','inherits':'--|>','aliases':'..>',
        'precedes':'-[#778899,dashed]->','requires_completed_event':'-[#778899,dashed]->','instance_of':'..>',
        'issues':'-[#a26f23,bold]->','submits':'-[#a26f23,bold]->','evaluates':'-[#687aa2,dashed]->',
        'has_attribute':'-[#6c5691,dashed]->','member_of':'-[#728394,dotted]->',
        'macro_uses':'-[#795d44,dashed]->','expands_to':'-[#795d44,dotted]->',
        'implicit_calls':'-[#386fa1,dashed]->','has_member':'*--'}


def h(value):return html.escape(str(value),quote=True)
def digest(value):return hashlib.sha256(value.encode()).hexdigest()
def slug(value):return re.sub(r'[^A-Za-z0-9_.-]+','_',value)[:100]+'_'+digest(value)[:8]
def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def normalized_path(path):return path.removeprefix('snapshot/')if path else None
def source_href(path,line=1):return 'source/'+quote(path,safe='/')+'.html#L'+str(line or 1)
def without_templates(value):
    result=[];depth=0
    for char in value:
        if char=='<':depth+=1
        elif char=='>'and depth:depth-=1
        elif not depth:result.append(char)
    return ''.join(result)


def auxiliary_path(path):
    if not path:return None
    if path.startswith(('data/','audits/')):candidate=ROOT/path;allowed=ROOT
    elif path.startswith('../02_cutlass_and_gemm/exemples/'):
        candidate=ROOT/path;allowed=ROOT.parent/'02_cutlass_and_gemm/exemples'
    elif path.startswith('Docs/cutlass/02_cutlass_and_gemm/exemples/'):
        candidate=ROOT.parents[2]/path;allowed=ROOT.parent/'02_cutlass_and_gemm/exemples'
    else:return None
    candidate=candidate.resolve()
    return candidate if candidate.is_relative_to(allowed.resolve())and candidate.is_file()else None


def render_source_page(site,out,label,raw,commit,notice):
    out.parent.mkdir(parents=True,exist_ok=True);relative=os.path.relpath(site,out.parent)
    upstream='https://github.com/NVIDIA/cutlass/blob/'+commit+'/'+quote(label,safe='/')if label.startswith(('include/cutlass/','include/cute/'))else None
    lines=['<div class="line" id="L'+str(i)+'"><a class="ln" href="#L'+str(i)+'">'+str(i)+'</a><code>'+h(line)+'</code></div>'
           for i,line in enumerate(raw.decode('utf-8','replace').splitlines(),1)]
    if not lines:lines=['<div class="line" id="L1"><span class="ln">—</span><code>（空文件；0字节）</code></div>']
    out.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+h(label)+'</title>'
        '<link rel="stylesheet" href="'+h(relative)+'/assets/atlas.css"><body class="source-page">'
        '<header><a href="'+h(relative)+'/index.html">← 接口图集</a><strong>'+h(label)+'</strong><small>'+h(commit)+'</small>'
        +('<a href="'+h(upstream)+'" target="_blank" rel="noopener">固定提交原文（联网）</a>'if upstream else'')+'</header>'
        '<p class="source-notice">'+h(notice)+'</p><main class="source-code">'+''.join(lines)+'</main></body></html>')


def load_modules():
    modules=[];nodes={};edges={};views={};issues=[];contracts=[];closures=[]
    manifests=(ROOT/'data/modules').glob('*/module.json')
    for manifest in sorted(manifests,key=lambda p:(json.loads(p.read_text()).get('reading_priority',100),str(p))):
        module=json.loads(manifest.read_text());module['manifest']=str(manifest.relative_to(ROOT))
        modules.append(module)
        for name in module.get('parts',[]):
            path=manifest.parent/name
            if not path.exists():
                issues.append({'kind':'module_part_pending','module_id':module['module_id'],'path':str(path.relative_to(ROOT))});continue
            part=json.loads(path.read_text())
            module.setdefault('protocols',[]).extend(part.get('protocols',[]))
            if part.get('coverage'):
                module.setdefault('work_packages',[]).append({'part':part.get('part',path.stem),
                    'contribution_file':str(path.relative_to(ROOT)),'scope':part.get('scope',{}),'coverage':part['coverage']})
            for key,target in [('nodes',nodes),('edges',edges),('views',views)]:
                for item in part.get(key,[]):
                    item=dict(item);item['module_id']=module['module_id'];item['part']=part.get('part',path.stem)
                    item['contribution_file']=str(path.relative_to(ROOT))
                    if item['id']in target:raise ValueError('Duplicate '+key+' ID: '+item['id'])
                    target[item['id']]=item
            for item in part.get('issues',[]):
                issues.append({'module_id':module['module_id'],'part':part.get('part'),**(item if isinstance(item,dict)else{'description':item})})
            contracts.extend({'module_id':module['module_id'],**c}for c in part.get('contracts',[]))
            closures.extend({'part':part.get('part'),**c}for c in part.get('resolves_external_issues',[]))
            module.setdefault('compile_evidence',[]).extend(part.get('compile_evidence',[]))
    for issue in issues:
        closure=next((c for c in closures if c.get('id')==issue.get('id')),None)
        if closure:issue.update(status='closed_for_selected_configuration',closure=closure)
    return modules,nodes,edges,views,issues,contracts


def load_file_ownership(scope):
    registry=json.loads((ROOT/'data/module-registry.json').read_text())
    expected={f['path']:f['sha256']for f in scope['files']}
    actual={f['path']:f['source_sha256']for f in registry['files']}
    if registry['commit']!=scope['commit']or actual!=expected or len(actual)!=len(registry['files']):
        raise ValueError('Module ownership does not match the fixed source scope')
    if registry['scope_sha256']!=hashlib.sha256((ROOT/'data/scope.json').read_bytes()).hexdigest():
        raise ValueError('Module ownership was generated from a different scope manifest')
    return registry


def annotate_work_packages(modules,scope):
    paths={f['path']for f in scope['files']};cache={}
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,dict):
            path=value.get('path')
            if path in paths:
                if path not in cache:cache[path]=(ROOT/'snapshot'/path).read_bytes()
                raw=cache[path];line=value.get('start_line',value.get('line',1))
                if not 1<=line<=len(raw.splitlines()):raise ValueError('Work-package source line invalid: '+path)
                if 'start_byte'in value and 'end_byte'in value:
                    a,b=value['start_byte'],value['end_byte']
                    if not 0<=a<b<=len(raw)or('raw'in value and raw[a:b].decode()!=value['raw']):
                        raise ValueError('Work-package source byte range invalid: '+path)
                value['source_url']=source_href(path,line)
            for key,item in list(value.items()):
                if key!='source_url':visit(item)
    for module in modules:visit(module.get('work_packages',[]))


def source_selector_matches(selector,occurrence):
    """An explicit source selector never falls back to a matching short name."""
    if not isinstance(selector,dict):return False
    if not {'path','kind','qualified_name','signature_range'}<=selector.keys():return False
    if not isinstance(selector['qualified_name'],str)or not selector['qualified_name'].strip():return False
    if occurrence.get('parse_status')!='parsed'or occurrence.get('scope_review_required'):return False
    if selector['path']!=occurrence.get('path')or selector['kind']!=occurrence.get('kind'):return False
    def tokens(value):return tuple(t.text for t in LEXER.lex(str(value).encode())[0])
    if tokens(selector['qualified_name'])!=tokens(occurrence.get('qualified_name','')):return False
    if 'name'in selector and tokens(selector['name'])!=tokens(occurrence.get('name','')):return False
    expected=selector['signature_range'];actual=occurrence.get('signature_range',{})
    if not isinstance(expected,dict):return False
    if not {'path','start_byte','end_byte'}<=expected.keys():return False
    if any(type(expected[k])is not int for k in ('start_byte','end_byte')):return False
    if any(expected[k]!=actual.get(k)for k in ('path','start_byte','end_byte')):return False
    if any(k in expected and expected[k]!=actual.get(k)for k in ('start_line','end_line')):return False
    if selector.get('signature_sha256')and selector['signature_sha256']!=digest(occurrence.get('raw_signature','')):return False
    for field in ('entity_id','declaration_occurrence_id','variant_id'):
        if field in selector and selector[field]!=occurrence.get(field):return False
    return True


def retire_resolved_manual_identity(node,occurrences):
    """Prefer a repaired canonical declaration, retaining the old local proof."""
    manual=node.get('manual_declaration')
    if not manual:return False
    span=manual['signature_range'];matches=[]
    for occurrence in occurrences:
        if 'source_selector'in node and not source_selector_matches(node['source_selector'],occurrence):continue
        candidate=occurrence.get('signature_range',{})
        if occurrence.get('parse_status')!='parsed'or occurrence.get('scope_review_required'):continue
        if (candidate.get('path'),candidate.get('start_byte'),candidate.get('end_byte'))!=(span.get('path'),span.get('start_byte'),span.get('end_byte')):continue
        if occurrence.get('raw_signature')!=manual.get('raw_signature'):continue
        if without_templates(occurrence.get('qualified_name',''))!=without_templates(node.get('qualified_name','')):continue
        if occurrence.get('kind')!=manual.get('kind'):continue
        matches.append(occurrence)
    if len({o['entity_id']for o in matches})!=1:return False
    canonical=matches[0]
    node['manual_declaration_history']=node.pop('manual_declaration')
    correction=node.setdefault('identity_correction',{})
    if correction.get('reason'):correction['historical_reason']=correction.pop('reason')
    correction.update(status='resolved_by_canonical_source_ledger',
        canonical_entity_id=canonical['entity_id'],canonical_occurrence_id=canonical['declaration_occurrence_id'],
        resolution_basis='Exact physical signature range, raw signature, kind and qualified owner; clean scope reviewed by current extractor')
    return True


def resolve_multiple_declarations(node,records):
    """One source entity may have multiple independently selected appearances."""
    if 'source_selectors'not in node:return None
    selectors=node['source_selectors']
    if not isinstance(selectors,list)or not selectors:raise ValueError('Explicit source_selectors must be a nonempty list')
    selected=[]
    for selector in selectors:
        if not isinstance(selector,dict):raise ValueError('Invalid source selector in source_selectors')
        expanded={'qualified_name':node.get('qualified_name'),**selector}
        matches=[o for o in records.get(expanded.get('path'),[])if source_selector_matches(expanded,o)]
        if len(matches)!=1:raise ValueError('Each source selector must match exactly one clean occurrence')
        selected.append(matches[0])
    if len({o['declaration_occurrence_id']for o in selected})!=len(selected):raise ValueError('Duplicate source occurrence in source_selectors')
    if len({o['entity_id']for o in selected})!=1:raise ValueError('source_selectors must not combine different source entities')
    if 'source_selector'in node and not any(source_selector_matches(node['source_selector'],o)for o in selected):
        raise ValueError('Primary source_selector is not a member of source_selectors')
    return selected


def enrich_nodes(nodes,scope,issues):
    source_paths={f['path']for f in scope['files']};wanted=defaultdict(list);needed_paths=set()
    for node in nodes.values():
        node['path']=normalized_path(node.get('path'))
        if node['path']in source_paths:
            node['source_url']=source_href(node['path'],node.get('line'))
            wanted[node['path']].append(node)
            needed_paths.add(node['path'])
            selectors=node.get('source_selectors')
            if isinstance(selectors,list):needed_paths.update(s['path']for s in selectors if isinstance(s,dict)and s.get('path')in source_paths)
        elif node.get('kind')=='external':node['declaration_status']='external_boundary'
        elif node.get('path'):node['declaration_status']='auxiliary_or_unresolved_source'
    records=defaultdict(list)
    ledger=ROOT/'data/declarations.json'
    if wanted:
        print('Indexing selected API declarations from immutable-source ledger…',flush=True)
        for key,value,item in top_items(ledger,arrays=('occurrences',)):
            if item and key=='occurrences'and value['path']in needed_paths:
                records[value['path']].append(value)
    for path,selected in wanted.items():
        for node in selected:
            try:multiple=resolve_multiple_declarations(node,records)
            except ValueError as error:
                node['declaration_status']='explicit_source_selector_unresolved'
                issues.append({'kind':'explicit_source_selector_unresolved','node_id':node['id'],'reason':str(error)})
                continue
            retire_resolved_manual_identity(node,records[path])
            bound=node['id'].endswith('_bound')and bool(node.get('qualified_name'))
            if bound:
                node['representation']='configured_type_instance';node['configured_type']=node['qualified_name']
                node['full_name']=node['qualified_name']
            manual=node.get('manual_declaration')
            if manual:
                if 'source_selector'in node or multiple is not None:
                    node['declaration_status']='explicit_source_selector_unresolved'
                    issues.append({'kind':'explicit_source_selector_unresolved','node_id':node['id'],'selector':node.get('source_selector'),'source_selectors':node.get('source_selectors')})
                    continue
                sig=manual.get('signature_range',{});raw=(ROOT/'snapshot'/path).read_bytes()
                if sig.get('path')!=path or not 0<=sig.get('start_byte',-1)<sig.get('end_byte',-1)<=len(raw)or raw[sig['start_byte']:sig['end_byte']].decode().rstrip()!=manual.get('raw_signature'):
                    raise ValueError('Manual declaration does not match exact fixed source: '+node['id'])
                node.update(entity_id='manual_'+digest(json.dumps([path,sig['start_byte'],sig['end_byte'],manual['kind']]))[:24],
                    declaration_occurrence_id='manual_occ_'+digest(node['id'])[:24],entity_kind=manual['kind'],
                    full_name=node.get('qualified_name')or manual['name']+' [local '+manual['kind']+']',
                    signature=manual['raw_signature'],signature_range=sig,parameters=manual.get('parameters',[]),
                    return_type=manual.get('return_type'),attributes=manual.get('attributes',[]),
                    declaration_status='manual_source_declaration_verified',parse_status='source_range_reviewed')
                node['source_url']=source_href(path,sig['start_line']);continue
            if 'source_selector'not in node and multiple is None and (node.get('kind')in {'hardware_event','binding'} or node.get('entity_kind')=='local_object'):
                node['declaration_status']='source_anchored_concept';continue
            line=node.get('line',0)
            name=without_templates(node.get('qualified_name')or node.get('name','')).split('::')[-1].split('(')[0].strip()
            choices=[]
            allowed=KINDS.get(node.get('kind'),set())
            if node.get('kind')=='api':allowed=KINDS['api']|KINDS['type']|KINDS['resource']|{'macro_definition'}
            if node.get('entity_kind')=='member':allowed=KINDS['resource']
            for o in records[path]:
                if multiple is not None and o['declaration_occurrence_id']not in {x['declaration_occurrence_id']for x in multiple}:continue
                sig=o['signature_range'];base=(o['name']or'').split('<')[0]
                selector=node.get('source_selector')
                explicit='source_selector'in node
                if explicit and not source_selector_matches(selector,o):continue
                if not explicit and '<cutlass_namespace('in o['qualified_name']and '<cutlass_namespace('not in (node.get('qualified_name')or''):continue
                if allowed and o['kind']not in allowed and not explicit:continue
                if sig['start_line']<=line<=sig['end_line']:
                    matches_name=name==o['name']or name==base or (name=='operator'and (o['name']or'').startswith('operator'))
                    if node.get('qualified_name')and not matches_name:continue
                    if node.get('qualified_name')and without_templates(node['qualified_name'])!=without_templates(o['qualified_name']):continue
                    score=(0 if matches_name else 1,sig['end_byte']-sig['start_byte'])
                    choices.append((score,o))
            if choices:
                choices.sort(key=lambda x:x[0]);best=choices[0]
                equivalent=[o for score,o in choices if score==best[0]]
                identities={o['entity_id']for o in equivalent}
                if len(identities)!=1:
                    node['declaration_status']='ambiguous_selector'
                    issues.append({'kind':'ambiguous_api_selector','node_id':node['id'],'path':path,'line':line,'candidates':list(identities)})
                    continue
                o=best[1]
                if bound:
                    node.update(template_entity_id=o['entity_id'],template_declaration_occurrence_id=o['declaration_occurrence_id'],
                                declaration_status='configured_type_with_source_template',source_url=source_href(path,o['signature_range']['start_line']))
                    continue
                if node.get('qualified_name')and node.get('qualified_name','').startswith(('cutlass::','cute::')):
                    if without_templates(node['qualified_name'])!=without_templates(o['qualified_name']):
                        issues.append({'kind':'claimed_namespace_disagrees_with_ledger','node_id':node['id'],
                            'claimed':node['qualified_name'],'ledger':o['qualified_name'],'path':path,'line':line})
                        node['identity_review_required']=True
                node.update(entity_id=o['entity_id'],declaration_occurrence_id=o['declaration_occurrence_id'],
                    full_name=o['qualified_name'],signature=o['raw_signature'],entity_kind=o['kind'],
                    signature_range=o['signature_range'],parameters=o.get('parameters',[]),
                    template_parameters=o.get('template_parameters',[]),return_type=o.get('return_type'),
                    attributes=o.get('attributes',[]),qualifiers=o.get('qualifiers',[]),
                    alignment_specifiers=o.get('alignment_specifiers',[]),
                    conditions=o.get('preprocessor_conditions',[]),access=o.get('access'),
                    access_scope='class_member'if o['kind']!='macro_definition'and any(s['kind']in {'class_specifier','struct_specifier','union_specifier'}for s in o.get('scope_chain',[]))else'not_class_member',
                    parse_status=o.get('parse_status'),declaration_status='linked_to_source_ledger')
                for field in ('target_type','declared_type','declared_type_spelling','declared_type_role','declarator','initializer','bases','specialization','template_arguments','return_cv_qualifiers'):
                    if field in o:node[field]=o[field]
                if multiple is not None:
                    node['selected_declaration_conditions']=node.pop('conditions')
                    node['declaration_availability']={'operator':'any_of',
                        'occurrence_condition_groups':[x.get('preprocessor_conditions',[])for x in multiple],
                        'occurrence_ids':[x['declaration_occurrence_id']for x in multiple],
                        'meaning':'Source preprocessing availability only; template well-formedness and runtime support are separate'}
                    node['source_declaration_occurrences']=[{
                        **{k:x[k]for k in ('declaration_occurrence_id','entity_id','variant_id','path','qualified_name','kind','signature_range')},
                        'signature':x['raw_signature'],'start_line':x['signature_range']['start_line'],
                        'declaration_range':{k:x[k]for k in ('path','start_byte','end_byte','start_line','end_line')},
                        'source_url':source_href(x['path'],x['signature_range']['start_line']),
                        **{k:x[k]for k in ('parameters','template_parameters','return_type','target_type','declared_type','initializer','bases','attributes','qualifiers','preprocessor_conditions')if k in x}}
                        for x in multiple]
                node['source_url']=source_href(path,o['signature_range']['start_line'])
                raw=(ROOT/'snapshot'/path).read_bytes();s=o['signature_range']
                if raw[s['start_byte']:s['end_byte']].decode().rstrip()!=o['raw_signature']:
                    raise ValueError('Selected declaration signature differs from snapshot: '+node['id'])
            else:
                node['declaration_status']='explicit_source_selector_unresolved'if 'source_selector'in node else'source_anchor_only'
                if 'source_selector'in node or multiple is not None or node.get('kind')in {'api','type'}:
                    issues.append({'kind':'explicit_source_selector_unresolved'if 'source_selector'in node else'api_selector_not_resolved','node_id':node['id'],'path':path,'line':line})
    return records


def copy_source_pages(site,scope):
    print('Writing offline source pages for all '+str(scope['file_count'])+' files…',flush=True)
    for entry in scope['files']:
        path=entry['path'];out=site/'source'/(path+'.html');out.parent.mkdir(parents=True,exist_ok=True)
        raw=(ROOT/'snapshot'/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Snapshot drift: '+path)
        render_source_page(site,out,path,raw,scope['commit'],'固定提交原文；版权与许可证按源码保留。行号是浏览标记，不属于原文件。')
    license_file=ROOT/'snapshot/LICENSE.txt'
    if not license_file.exists():license_file=ROOT/'snapshot/LICENSE'
    if license_file.exists():shutil.copyfile(license_file,site/'CUTLASS-LICENSE.txt')


def package_auxiliary(site,modules,nodes,edges,contracts,issues):
    cache={}
    def attach(value):
        if isinstance(value,list):
            for item in value:attach(item)
        elif isinstance(value,dict):
            path=value.get('path');candidate=auxiliary_path(path)if isinstance(path,str)else None
            if candidate:
                raw=candidate.read_bytes();key=digest(str(candidate))[:16]+'-'+candidate.name
                if key not in cache:
                    destination=site/'evidence'/key;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
                    preview=raw if candidate.suffix not in {'.cubin','.o','.a','.so'} else ('Binary artifact\nSHA256: '+hashlib.sha256(raw).hexdigest()+'\nBytes: '+str(len(raw))).encode()
                    render_source_page(site,site/'evidence'/(key+'.html'),path,preview,hashlib.sha256(raw).hexdigest(),
                        '辅助材料或编译证据；不计入824库文件分母。此副本按当前内容散列固定。')
                    cache[key]={'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
                line=value.get('start_line',value.get('line',1))or 1
                value['source_url']='evidence/'+quote(key+'.html')+'#L'+str(line)
                value['download_url']='evidence/'+quote(key);value['auxiliary_sha256']=cache[key]['sha256']
                if value.get('start_line'):
                    lines=raw.decode('utf-8','replace').splitlines();end=value.get('end_line',line)
                    if not 1<=line<=end<=len(lines):issues.append({'kind':'auxiliary_range_invalid','path':path,'start_line':line,'end_line':end})
                    else:value['source_text']='\n'.join(lines[line-1:end])
            for key,item in list(value.items()):
                if key not in {'source_text','source_url','download_url'}:attach(item)
    for value in [modules,list(nodes.values()),list(edges.values()),contracts]:attach(value)
    for module in modules:
        for evidence in module.get('compile_evidence',[]):
            source=auxiliary_path(evidence.get('path'))
            if not source:continue
            artifacts=[]
            for name in evidence.get('artifacts',{}):
                candidate=source.parent/name
                if candidate.is_file()and candidate.suffix in {'.txt','.ptx','.hpp','.cu','.json','.cubin','.log'}:
                    item={'path':str(candidate.relative_to(ROOT))};attach(item);artifacts.append(item)
            evidence['offline_artifacts']=artifacts
    return list(cache.values())


def puml_label(value,width=60):
    # Break at scope/path/argument boundaries, never through an identifier.
    # PlantUML Unicode escapes preserve source string-literal quotes in both
    # quoted node labels and edge labels; changing them to apostrophes changes C++.
    value=str(value).replace('"','<U+0022>')
    if '\n'in value:return r'\n'.join(puml_label(line,width)for line in value.splitlines())
    tokens=re.split(r'(?<=::)|(?<=/)|(?<=,)|(?<=\s)',value)
    lines=[];current=''
    for token in tokens:
        if current and len(current)+len(token)>width:
            lines.append(current.rstrip());current=''
        current+=token
    if current:lines.append(current.rstrip())
    return r'\n'.join(lines)


def diagram_source(title,node_ids,edge_ids,nodes,edges,detail=False):
    lines=['@startuml','skinparam backgroundColor #ffffff','skinparam shadowing false',
           'skinparam defaultFontName Noto Sans CJK SC','skinparam defaultFontSize 14','skinparam ArrowFontSize 12',
           'skinparam classAttributeIconSize 0','skinparam svgLinkTarget _top','hide empty members','hide circle',
           'skinparam noteBackgroundColor #F3F6FA','skinparam noteBorderColor #C5D0DD',
           'skinparam classBackgroundColor #F9FBFE','skinparam classBorderColor #4B6078',
           'top to bottom direction','title '+title.replace('\n',' ')]
    aliases={identifier:'n'+str(i)for i,identifier in enumerate(node_ids)}
    for identifier in node_ids:
        n=nodes[identifier];label=n.get('full_name')if detail and n.get('full_name')and not n.get('configured_type')else n['name']
        if detail and n.get('signature')and n.get('entity_kind')in KINDS['type']:
            # The declaration below retains all template parameters and
            # specialization constraints. Avoid repeating them in the title.
            label=without_templates(n.get('full_name')or label)
        elif detail and n.get('entity_kind')in KINDS['api']:
            operation=without_templates(n.get('full_name')or label).split('::')[-1]
            label=operation+'\n'+label
        label=puml_label(label,72 if detail else 34)
        location=(n.get('path')or'外部边界')+(':'+str(n['line'])if n.get('line')else'')
        stereotype='operation'if n.get('entity_kind')in KINDS['api']else n.get('kind','node')
        lines.append('class "'+label+r'\n'+puml_label(location,82 if detail else 38)+'" as '+aliases[identifier]+' <<'+stereotype+'>> [[../index.html?api='+quote(identifier,safe='')+']]')
        if detail:
            code=n.get('signature')or n.get('configured_type')or n.get('source_expression')or n.get('notes')or n.get('role')or'尚无范围内声明；查看边界说明。'
            wrapped='\n'.join('\n'.join(textwrap.wrap(line,96,replace_whitespace=False,drop_whitespace=False,break_long_words=False,break_on_hyphens=False))or''for line in str(code).splitlines())
            # A column-0 __device__ line is misread as Creole formatting even
            # inside <code> by this PlantUML version. Indent the code block;
            # whitespace changes only layout, not any source token.
            lines+=['note bottom of '+aliases[identifier],'<code>','\n'.join('  '+line for line in wrapped.splitlines()),'</code>']
            if n.get('attributes'):lines.append('attributes: '+', '.join(n['attributes']).replace('_','~_'))
            if n.get('access')and n.get('access_scope')!='not_class_member':lines.append('access: '+n['access'])
            lines.append('end note')
    for identifier in edge_ids:
        e=edges[identifier];relation=e['relation']
        evidence=next((p for p in e.get('evidence',[])if p.get('path')and p.get('start_line')),None)
        callsite=e.get('callsite')
        anchor=(callsite['path']+':'+str(callsite['start_line'])+(' [implicit-call source origin]'if callsite.get('kind')=='language_desugaring'else' [callsite]'))if callsite else (evidence['path']+':'+str(evidence['start_line'])+'-'+str(evidence.get('end_line',evidence['start_line']))+' [evidence range]')if evidence else'recorded evidence'
        label=RELATIONS.get(relation,relation)+' · '+identifier
        if e.get('resolution')=='symbolic':label+='\n参数化表达 / 条件候选，非已确定实例目标'
        if detail:label+='\n'+str(e.get('source_expression',''))+'\n'+str(e.get('condition',''))+'\n'+anchor
        if detail and e.get('dependency_parameters'):label+='\n依赖参数：'+', '.join(e['dependency_parameters'])
        if detail and e.get('evaluation'):label+='\n'+str(e['evaluation'])
        if detail and e.get('macro_bindings'):label+='\n宏参数：'+', '.join(k+' := '+str(v)for k,v in e['macro_bindings'].items())
        if detail and e.get('expanded_spelling'):label+='\n展开结果：'+str(e['expanded_spelling'])
        if detail and e.get('synthetic_expression'):label+='\n语言展开示意（非源码字面调用）：'+str(e['synthetic_expression'])
        arrow='-[#687aa2,dashed]->'if relation=='calls'and e.get('resolution')=='symbolic'else ARROWS.get(relation,'..>')
        lines.append(aliases[e['source']]+' '+arrow+' '+aliases[e['target']]+' : '+puml_label(label,70))
    lines+=['legend bottom','operation为独立API操作节点，不是新增C++类。类型、调用、事件分别标记。',
            '本图为逐关系详细视图；完整字段及条件见接口页。'if detail else'导航视图，不代替完整接口签名；点击节点或选择关系查看详细图。',
            'endlegend','@enduml']
    return '\n'.join(lines)+'\n'


def declaration_for_edge(node,edge):
    """Use this relation's physical declaration, not a merged entity's first one."""
    occurrences=node.get('source_declaration_occurrences',[])
    if len(occurrences)<2:return node
    if edge.get('source')!=node['id']and not(edge.get('relation')=='member_of'and edge.get('target')==node['id']):return node
    markers=[edge['callsite']]if edge.get('callsite')else edge.get('evidence',[])
    def contains_source(span,marker):
        if marker.get('path')!=span['path']or not span['start_byte']<=marker.get('start_byte',-1)<marker.get('end_byte',-1):return False
        if marker['end_byte']<=span['end_byte']:return True
        path=ROOT/'snapshot'/span['path']
        if not path.is_file():return False
        raw=path.read_bytes()
        # C++ class AST ranges end at }, while a source declaration also has
        # its terminating semicolon. Accept only that exact source suffix.
        return marker['start_byte']<span['end_byte']and marker['end_byte']<=len(raw)and raw[span['end_byte']:marker['end_byte']].strip()==b';'
    matched=[]
    for occurrence in occurrences:
        span=occurrence.get('declaration_range',occurrence['signature_range'])
        if any(contains_source(span,m)for m in markers):
            matched.append(occurrence)
    if len(matched)!=1:return node
    occurrence=matched[0]
    return {**node,**occurrence,'id':node['id'],'kind':node.get('kind'),'full_name':occurrence.get('qualified_name',node.get('full_name')),
            'line':occurrence['signature_range']['start_line'],'signature':occurrence['signature'],
            'selected_for_relation_source':True}


def build_diagrams(site,nodes,edges,views,issues):
    directory=site/'diagrams';directory.mkdir(parents=True,exist_ok=True)
    pending=[];outputs=[]
    panels={}
    for view in views.values():
        groups=defaultdict(list)
        for edge_id in view.get('edge_ids',[]):groups[edges[edge_id]['source']].append(edge_id)
        view['panels']=[]
        for source,group in groups.items():
            for offset in range(0,len(group),2):
                edge_ids=group[offset:offset+2]
                identifier=view['id']+'.panel.'+str(len(view['panels'])+1)
                panel={'id':identifier,'title':nodes[source]['name']+' · '+str(len(view['panels'])+1),
                       'node_ids':list(dict.fromkeys(x for eid in edge_ids for x in (edges[eid]['source'],edges[eid]['target']))),'edge_ids':edge_ids}
                panels[identifier]=panel;view['panels'].append(panel)
    for category,items in [('panel',panels),('edge',edges)]:
        for identifier,item in items.items():
            node_ids=item.get('node_ids',[])if category=='panel'else list(dict.fromkeys([item['source'],item['target']]))
            edge_ids=item.get('edge_ids',[])if category=='panel'else[identifier]
            if any(n not in nodes for n in node_ids)or any(e not in edges for e in edge_ids):
                raise ValueError('Unresolved diagram reference: '+identifier)
            # View edges may supply endpoints missing from an authored node list.
            for eid in edge_ids:
                for nid in [edges[eid]['source'],edges[eid]['target']]:
                    if nid not in node_ids:raise ValueError('View omitted edge endpoint: '+identifier+'/'+nid)
            title=item.get('title')or(RELATIONS.get(item.get('relation'),item.get('relation',''))+' · '+identifier)
            display_nodes=nodes
            if category=='edge':
                display_nodes={**nodes,**{nid:declaration_for_edge(nodes[nid],item)for nid in node_ids}}
                item['displayed_declaration_occurrences']={nid:{
                    'declaration_occurrence_id':display_nodes[nid]['declaration_occurrence_id'],
                    'signature':display_nodes[nid].get('signature'),
                    'path':display_nodes[nid].get('path'),'start_line':display_nodes[nid].get('line')}
                    for nid in node_ids if display_nodes[nid].get('declaration_occurrence_id')}
            text=diagram_source(title,node_ids,edge_ids,display_nodes,edges,category=='edge')
            base=category+'-'+slug(identifier);puml=directory/(base+'.puml');svg=directory/(base+'.svg')
            same=puml.exists()and puml.read_text()==text and svg.exists()
            puml.write_text(text)
            if not same:pending.append(puml)
            item['svg']='diagrams/'+svg.name;item['plantuml']='diagrams/'+puml.name
            outputs.append(svg)
    if pending:
        env=dict(os.environ);env['JAVA_TOOL_OPTIONS']='-Djava.awt.headless=true';env['PLANTUML_LIMIT_SIZE']='24000'
        print('Rendering '+str(len(pending))+' PlantUML diagrams…',flush=True)
        result=subprocess.run(['plantuml','-tsvg','-charset','UTF-8','-nometadata',*[str(p)for p in pending]],env=env,text=True,capture_output=True)
        (site/'plantuml-render.log').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:issues.append({'kind':'plantuml_render_failed','returncode':result.returncode,'log':'plantuml-render.log'})
    for svg in outputs:
        if not svg.exists()or'<svg'not in svg.read_text(errors='replace'):
            issues.append({'kind':'missing_svg','path':str(svg.relative_to(site))})
        elif 'Syntax Error' in svg.read_text(errors='replace'):
            issues.append({'kind':'plantuml_syntax_error','path':str(svg.relative_to(site))})
    for item in list(panels.values())+list(edges.values()):
        svg=site/item['svg']
        if svg.exists():
            root=ET.parse(svg).getroot()
            item['svg_dimensions']={k:float(re.sub(r'[^0-9.]','',root.get(k,'0')))for k in ('width','height')}


def validate_evidence(edges,scope,issues):
    files={f['path']:f for f in scope['files']}
    cache={};raw_cache={}
    def source_bytes(path):
        if path not in raw_cache:
            source=ROOT/'snapshot'/path if path in files else auxiliary_path(path)
            raw_cache[path]=source.read_bytes()if source and source.is_file()else None
        return raw_cache[path]
    for edge in edges.values():
        if not edge.get('evidence'):issues.append({'kind':'edge_without_evidence','edge_id':edge['id']})
        for evidence in edge.get('evidence',[]):
            path=normalized_path(evidence.get('path'))
            a=evidence.get('start_line',evidence.get('line'));b=evidence.get('end_line',a)
            if a is not None:evidence['start_line']=a;evidence['end_line']=b
            if path in files or auxiliary_path(path):
                evidence['path']=path
                if path not in cache:cache[path]=source_bytes(path).decode('utf-8','replace').splitlines(keepends=True)
                if not a or not b or not 1<=a<=b<=len(cache[path]):
                    issues.append({'kind':'invalid_evidence_range','edge_id':edge['id'],'evidence':evidence});continue
                if path in files:evidence['source_url']=source_href(path,a)
                evidence['source_text']=''.join(cache[path][a-1:b])
                if 'start_byte'in evidence or 'end_byte'in evidence:
                    raw=source_bytes(path);x=evidence.get('start_byte');y=evidence.get('end_byte')
                    if not isinstance(x,int)or not isinstance(y,int)or not 0<=x<y<=len(raw)or (raw[:x].count(b'\n')+1,raw[:y-1].count(b'\n')+1)!=(a,b):
                        issues.append({'kind':'invalid_evidence_byte_range','edge_id':edge['id'],'evidence':evidence});continue
                    evidence['source_text']=raw[x:y].decode('utf-8','replace')
                if evidence.get('quote')and evidence['quote']not in evidence['source_text']:
                    issues.append({'kind':'evidence_quote_mismatch','edge_id':edge['id'],'evidence':evidence})
            elif evidence.get('url'):evidence['external_link']=True
        if edge.get('relation')in {'calls','launches','implicit_calls'}:
            implicit=edge['relation']=='implicit_calls'
            if implicit and (edge.get('evaluation')!='language_desugaring'or edge.get('source_expression_is_not_literal_call')is not True or not edge.get('synthetic_expression')):
                issues.append({'kind':'implicit_call_missing_derivation','edge_id':edge['id']});continue
            needle=[t.text for t in LEXER.lex(str(edge.get('source_expression','')).encode())[0]]
            matches=[]
            for evidence in edge.get('evidence',[]):
                if not evidence.get('source_text')or not needle:continue
                raw=source_bytes(evidence['path'])
                if raw is None:continue
                source_lines=raw.splitlines(keepends=True)
                first=evidence['start_line']-1;last=evidence.get('end_line',evidence['start_line'])
                base=sum(map(len,source_lines[:first]));fragment=b''.join(source_lines[first:last])
                tokens=LEXER.lex(fragment)[0];words=[t.text for t in tokens]
                for i in range(len(words)-len(needle)+1):
                    if words[i:i+len(needle)]==needle:
                        a,b=tokens[i].start,tokens[i+len(needle)-1].end
                        actual_line=evidence['start_line']+fragment[:a].count(b'\n')
                        source_url=evidence.get('source_url','').split('#')[0]+'#L'+str(actual_line) if evidence.get('source_url')else source_href(evidence['path'],actual_line)
                        matches.append({'path':evidence['path'],'start_line':actual_line,'source_url':source_url,
                                        'end_line':evidence['start_line']+fragment[:b].count(b'\n'),
                                        'start_byte':base+a,'end_byte':base+b,'raw_source_expression':fragment[a:b].decode(),
                                        'source_expression':fragment[a:b].decode()})
            # Two calls on one line are two sites. An authored byte range may
            # select one; a line-only selector must not collapse them.
            unique={(x['path'],x['start_byte'],x['end_byte']):x for x in matches}
            authored=edge.get('callsite',{})
            if 'start_byte'in authored or 'end_byte'in authored:
                key=(authored.get('path'),authored.get('start_byte'),authored.get('end_byte'))
                if key not in unique:
                    issues.append({'kind':'authored_callsite_not_matched','edge_id':edge['id'],'callsite':authored})
                    continue
                unique={key:unique[key]}
            if len(unique)==1:
                actual=next(iter(unique.values()))
                inconsistent=[field for field in ['path','start_line','end_line','raw_source_expression']
                              if field in authored and authored[field]!=actual[field]]
                if 'source_expression'in authored and [t.text for t in LEXER.lex(authored['source_expression'].encode())[0]]!=needle:
                    inconsistent.append('source_expression')
                if authored.get('caller')and edge.get('source')and authored['caller']!=edge['source']:
                    inconsistent.append('caller')
                if inconsistent:
                    issues.append({'kind':'authored_callsite_fields_disagree','edge_id':edge['id'],'fields':inconsistent})
                    continue
                edge['callsite']={**authored,**actual}
                site=edge['callsite'];raw=source_bytes(site['path'])
                identity=[scope.get('commit'),site['path'],hashlib.sha256(raw).hexdigest(),site['start_byte'],site['end_byte']]
                if implicit:
                    site.update(kind='language_desugaring',synthetic_expression=edge['synthetic_expression'],source_anchor_is_not_literal_call=True)
                    identity+=['language_desugaring',edge['synthetic_expression']]
                site['callsite_id']='site_'+digest(json.dumps(identity))[:24]
            else:issues.append({'kind':'callsite_not_uniquely_located','edge_id':edge['id'],'matching_ranges':list(unique)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-sources',action='store_true',help='Reuse previously generated source pages for UI/graph iterations')
    args=parser.parse_args();scope=json.loads((ROOT/'data/scope.json').read_text());site=ROOT/'site';site.mkdir(exist_ok=True)
    tracked=[Path(__file__),ROOT/'scripts/reconcile_candidates.py',ROOT/'scripts/scan_candidates.py',ROOT/'scripts/protocol_diagrams.py',
             ROOT/'data/scope.json',ROOT/'data/declarations.json',ROOT/'data/module-registry.json',
             *(ROOT/'templates/atlas').glob('*'),*(ROOT/'data/modules').glob('*/*.json')]
    inputs={str(p.relative_to(ROOT)):file_sha(p)for p in tracked if p.is_file()}
    modules,nodes,edges,views,issues,contracts=load_modules()
    protocols=[{'module_id':module['module_id'],**p}for module in modules for p in module.pop('protocols',[])]
    annotate_work_packages(modules,scope)
    annotate_work_packages([{'work_packages':protocols}],scope)
    registry=load_file_ownership(scope);owners={f['path']:f for f in registry['files']}
    enrich_nodes(nodes,scope,issues)
    for edge in edges.values():
        if edge['source']not in nodes or edge['target']not in nodes:raise ValueError('Unknown edge endpoint: '+edge['id'])
    auxiliary=package_auxiliary(site,modules,nodes,edges,contracts,issues)
    validate_evidence(edges,scope,issues)
    for node in nodes.values():node['primary_file_module_id']=owners.get(node.get('path'),{}).get('primary_module_id')
    for edge in edges.values():
        edge['source_file_module_id']=nodes[edge['source']].get('primary_file_module_id')
        edge['target_file_module_id']=nodes[edge['target']].get('primary_file_module_id')
        edge['callsite_file_module_id']=owners.get(edge.get('callsite',{}).get('path'),{}).get('primary_module_id')
        edge['evidence_file_module_ids']=sorted({owners[e['path']]['primary_module_id']for e in edge.get('evidence',[])if e.get('path')in owners})
    validate_evidence({c['id']:{'id':c['id'],'relation':'contract','evidence':c.get('evidence',[])}for c in contracts},scope,issues)
    for protocol in protocols:
        validate_evidence({protocol['id']:{'id':protocol['id'],'relation':'protocol','evidence':protocol['evidence']}},scope,issues)
        for event in protocol['events']:
            edge=edges.get(event.get('call_edge_ref'))
            if edge and edge.get('callsite'):
                # Keep the authored selector for validation; attach only the
                # proven offline location, never silently repair the selector.
                event.setdefault('callsite',{})['source_url']=edge['callsite']['source_url']
    if not args.skip_sources:copy_source_pages(site,scope)
    build_diagrams(site,nodes,edges,views,issues)
    protocol_views=render_protocols(ROOT,site,protocols,nodes,edges,issues,puml_label)
    assets=site/'assets';assets.mkdir(exist_ok=True)
    for name in ['atlas.css','atlas.js','index.html']:
        target=site/name if name=='index.html'else assets/name
        shutil.copyfile(ROOT/'templates/atlas'/name,target)
    drift=[p for p,checksum in inputs.items()if file_sha(ROOT/p)!=checksum]
    if drift:issues.append({'kind':'build_inputs_changed_during_run','paths':drift})
    data={'commit':scope['commit'],'library_version':'4.6.0','api_reading_mainline':'3.x',
        'scope_file_count':scope['file_count'],'modules':modules,'nodes':list(nodes.values()),'edges':list(edges.values()),
        'views':list(views.values()),'contracts':contracts,'protocols':protocols,'issues':issues,'auxiliary_files':auxiliary,
        'source_ownership':{key:registry[key]for key in ['modules','rules','counts','policy']},
        'files':[{'path':f['path'],'source_url':source_href(f['path']),'sha256':f['sha256'],
                  'upstream_url':'https://github.com/NVIDIA/cutlass/blob/'+scope['commit']+'/'+quote(f['path'],safe='/'),
                  'primary_module_id':owners[f['path']]['primary_module_id'],'ownership_rule_id':owners[f['path']]['rule_id'],
                  'navigation':owners[f['path']]['navigation'],'coverage':owners[f['path']]['coverage'],
                  'status':'source_indexed_module_review_pending'}for f in scope['files']],
        'global_completion_claimed':False,'generated_by':'scripts/build_atlas.py','input_sha256':inputs,'inputs_changed_during_build':drift}
    (assets/'atlas-data.js').write_text('window.ATLAS_DATA = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    dump(ROOT/'data/atlas.json',data)
    report={'scope_files':scope['file_count'],'modules':len(modules),'nodes':len(nodes),'edges':len(edges),'views':len(views),
        'panels':sum(len(v.get('panels',[]))for v in views.values()),'auxiliary_files':len(auxiliary),
        'protocols':len(protocols),'protocol_views':len(protocol_views),
        'source_linked_nodes':sum(n.get('declaration_status')=='linked_to_source_ledger'for n in nodes.values()),
        'issues':issues,'global_complete':False,'runtime_verified':False}
    dump(ROOT/'data/atlas-build-report.json',report)
    print(json.dumps({k:v for k,v in report.items()if k!='issues'},ensure_ascii=False))
    print('Issue categories: '+json.dumps(dict(Counter(i.get('kind','authored_scope_boundary')for i in issues)),ensure_ascii=False))


if __name__=='__main__':main()
