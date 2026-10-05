#!/usr/bin/env python3
"""Offline artifact/link/graph invariants. Does not replace browser or source review."""
import hashlib
import html.parser
import json
from pathlib import Path
import re
from urllib.parse import urlsplit,parse_qs,unquote
import xml.etree.ElementTree as ET
from reconcile_candidates import file_sha
from protocol_diagrams import verify_protocol
from refresh_presentation import verify_overlay

ROOT=Path(__file__).resolve().parents[1]


def audit():
    site=ROOT/'site';data=json.loads((ROOT/'data/atlas.json').read_text());scope=json.loads((ROOT/'data/scope.json').read_text())
    issues=[];nodes={n['id']:n for n in data['nodes']};edges={e['id']:e for e in data['edges']}
    if len(nodes)!=len(data['nodes'])or len(edges)!=len(data['edges']):issues.append({'kind':'duplicate_node_or_edge_id'})
    if {f['path']for f in data['files']}!={f['path']for f in scope['files']}:issues.append({'kind':'fixed_file_denominator_changed'})
    ownership=data.get('source_ownership',{});owner_ids={m['module_id']for m in ownership.get('modules',[])}
    if len(owner_ids)!=20:issues.append({'kind':'module_registry_incomplete'})
    for f in data['files']:
        if f.get('primary_module_id')not in owner_ids:issues.append({'kind':'file_owner_missing','path':f['path']})
    for module in ownership.get('modules',[]):
        if sum(f.get('primary_module_id')==module['module_id']for f in data['files'])!=module['primary_file_count']:
            issues.append({'kind':'module_file_count_mismatch','module':module['module_id']})
    for module in data['modules']:
        for package in module.get('work_packages',[]):
            if not set(package.get('scope',{}).get('paths',[]))<={f['path']for f in scope['files']}:
                issues.append({'kind':'work_package_scope_outside_library','module':module['module_id']})
            for obligation in package.get('coverage',{}).get('obligations',[]):
                for ref in obligation.get('graph_refs',[]):
                    if ref not in nodes and ref not in edges:issues.append({'kind':'work_package_dangling_reference','id':obligation['id'],'target':ref})
    presentation,presentation_issues=verify_overlay(ROOT)
    issues.extend(presentation_issues)
    presentation_paths=set(presentation.get('updated_template_paths',[]))if presentation and not presentation_issues else set()
    allowed_presentation_paths={'templates/atlas/atlas.js','templates/atlas/atlas.css','templates/atlas/index.html'}
    if not presentation_paths<=allowed_presentation_paths:
        issues.append({'kind':'presentation_override_outside_templates'});presentation_paths=set()
    for path,checksum in data.get('input_sha256',{}).items():
        # Only explicitly rebuilt UI templates may differ from the historical
        # source build. The immutable graph and all source inputs still match.
        expected=presentation['inputs'].get(path)if path in presentation_paths else checksum
        if not(ROOT/path).is_file()or file_sha(ROOT/path)!=expected:issues.append({'kind':'build_input_changed','path':path})
    for edge in edges.values():
        if edge['source']not in nodes or edge['target']not in nodes:issues.append({'kind':'missing_endpoint','edge':edge['id']})
        if not edge.get('evidence'):issues.append({'kind':'missing_edge_evidence','edge':edge['id']})
        if edge.get('relation')in {'calls','launches','implicit_calls'}:
            site_range=edge.get('callsite',{})
            if edge.get('relation')=='implicit_calls'and (site_range.get('kind')!='language_desugaring'or not site_range.get('source_anchor_is_not_literal_call')or site_range.get('synthetic_expression')!=edge.get('synthetic_expression')):
                issues.append({'kind':'implicit_call_derivation_not_preserved','edge':edge['id']})
            if not all(field in site_range for field in ['callsite_id','path','start_byte','end_byte','start_line','end_line','raw_source_expression']):
                issues.append({'kind':'callsite_lacks_byte_identity','edge':edge['id']})
            else:
                from build_atlas import auxiliary_path
                path=ROOT/'snapshot'/site_range['path']
                if not path.is_file():path=auxiliary_path(site_range['path'])
                if path is None or not path.is_file():issues.append({'kind':'callsite_source_missing','edge':edge['id']})
                else:
                    raw=path.read_bytes();a,b=site_range['start_byte'],site_range['end_byte']
                    if not 0<=a<b<=len(raw)or raw[a:b].decode()!=site_range['raw_source_expression']:
                        issues.append({'kind':'callsite_byte_roundtrip_failed','edge':edge['id']})
                    if (raw[:a].count(b'\n')+1,raw[:b].count(b'\n')+1)!=(site_range['start_line'],site_range['end_line']):
                        issues.append({'kind':'callsite_line_roundtrip_failed','edge':edge['id']})
    diagrams=[]
    for view in data['views']:
        recorded=[e for p in view.get('panels',[])for e in p['edge_ids']]
        if sorted(recorded)!=sorted(view['edge_ids']):issues.append({'kind':'panel_lost_or_duplicated_edge','view':view['id']})
        for panel in view.get('panels',[]):
            if len(panel['edge_ids'])>2:issues.append({'kind':'oversized_navigation_panel','panel':panel['id']})
            diagrams.append((panel,False))
    diagrams.extend((e,True)for e in data['edges'])
    if presentation:
        overview_data=json.loads((ROOT/'data/overviews.json').read_text())
        for graph in overview_data['diagrams']:
            if len(graph['edge_ids'])!=len(set(graph['edge_ids'])):
                issues.append({'kind':'duplicate_overview_edge','id':graph['id']})
            if not set(graph['edge_ids'])<=set(edges)or not set(graph['node_ids'])<=set(nodes):
                issues.append({'kind':'unknown_overview_endpoint','id':graph['id']})
            diagrams.append((graph,False))
    protocols={p['id']:p for p in data.get('protocols',[])}
    for protocol in protocols.values():
        try:verify_protocol(protocol,nodes,edges)
        except ValueError as error:issues.append({'kind':'protocol_invariant_failed','id':protocol['id'],'error':str(error)})
        diagrams.extend((view,False)for view in protocol['views'])
    linked_files=set()
    def link(value,base):
        if not value:return
        parsed=urlsplit(value)
        if parsed.scheme in {'http','https','data','mailto'}:return
        path=(base/unquote(parsed.path)).resolve()if parsed.path else base/'index.html'
        if not path.is_relative_to(site.resolve()):issues.append({'kind':'link_escapes_site','href':value});return
        if not path.exists():issues.append({'kind':'missing_link_target','href':value,'from':str(base.relative_to(site))});return
        linked_files.add(str(path.relative_to(site)))
        query=parse_qs(parsed.query)
        for key,registry in [('api',nodes),('edge',edges)]:
            if query.get(key)and query[key][0]not in registry:issues.append({'kind':'unknown_query_target','href':value})
        if query.get('protocol'):
            protocol=protocols.get(query['protocol'][0])
            if protocol is None:issues.append({'kind':'unknown_protocol_link','href':value})
            elif query.get('event')and query['event'][0]not in {e['id']for e in protocol['events']}:
                issues.append({'kind':'unknown_protocol_event_link','href':value})
            elif query.get('order')and query['order'][0]not in {o['id']for o in protocol['partial_order']}:
                issues.append({'kind':'unknown_protocol_order_link','href':value})
        if parsed.fragment.startswith('L')and path.suffix=='.html':
            if ('id="'+parsed.fragment+'"')not in path.read_text():issues.append({'kind':'missing_line_anchor','href':value})
    sizes=[]
    for item,detailed in diagrams:
        for key in ['svg','plantuml']:link(item[key],site)
        path=site/item['svg']
        if not path.exists():continue
        root=ET.parse(path).getroot();text=''.join(root.itertext())
        if 'Syntax Error'in text:issues.append({'kind':'uml_syntax_error','path':item['svg']})
        if item.get('dot'):link(item['dot'],site)
        if item.get('protocol_id'):
            # Within-API constraints may be individual rows, while cross-API
            # constraints have distinct port edges. All authored IDs remain
            # visible and traceable in SVG and editable UML.
            for identifier in item.get('order_ids',item.get('transition_ids',[])):
                if identifier not in text:issues.append({'kind':'protocol_constraint_not_visible','view':item['id'],'id':identifier})
                if identifier not in (site/item['plantuml']).read_text():issues.append({'kind':'protocol_constraint_missing_in_plantuml','view':item['id'],'id':identifier})
            if item.get('dot'):link(item['dot'],site)
        for element in root.iter():
            for key,value in element.attrib.items():
                if key in {'href','{http://www.w3.org/1999/xlink}href'}:link(value,path.parent)
        size=item.get('svg_dimensions')
        if not size or not size.get('width')or not size.get('height'):issues.append({'kind':'missing_svg_dimensions','path':item['svg']})
        else:sizes.append({'id':item['id'],'detail':detailed,**size})
        if detailed:
            for nid in {item['source'],item['target']}:
                signature=nodes[nid].get('signature')or nodes[nid].get('configured_type')
                used=item.get('displayed_declaration_occurrences',{}).get(nid)
                if used:
                    candidates=[{'declaration_occurrence_id':nodes[nid].get('declaration_occurrence_id'),'signature':nodes[nid].get('signature')},*nodes[nid].get('source_declaration_occurrences',[])]
                    if not any(o['declaration_occurrence_id']==used['declaration_occurrence_id']and o.get('signature')==used.get('signature')for o in candidates):
                        issues.append({'kind':'unproven_displayed_declaration','edge':item['id'],'node':nid})
                    signature=used.get('signature')or signature
                if signature and re.sub(r'\s+','',signature)not in re.sub(r'\s+','',text):
                    issues.append({'kind':'full_signature_not_visible_in_svg_text','edge':item['id'],'node':nid})
    def walk(value):
        if isinstance(value,list):
            for x in value:walk(x)
        elif isinstance(value,dict):
            if value.get('source_url'):link(value['source_url'],site)
            if value.get('download_url'):link(value['download_url'],site)
            for x in value.values():walk(x)
    walk(data)
    if presentation:
        walk(overview_data)
        interface_docs=json.loads((ROOT/'data/interface-docs.json').read_text())
        expected_interfaces={n['id']for n in data['nodes']if n['kind']in {'api','type','external'}}
        if set(interface_docs['nodes'])!=expected_interfaces:
            issues.append({'kind':'interface_documentation_scope_mismatch'})
        if interface_docs['base_atlas_sha256']!=file_sha(ROOT/'data/atlas.json'):
            issues.append({'kind':'interface_documentation_base_changed'})
        walk(interface_docs)
        main_path=json.loads((ROOT/'data/main-path.json').read_text())
        from build_main_path import STEP_IDS
        if main_path.get('base_atlas_sha256')!=file_sha(ROOT/'data/atlas.json'):
            issues.append({'kind':'main_path_base_changed'})
        if [step['id']for step in main_path['steps']]!=STEP_IDS:
            issues.append({'kind':'main_path_review_step_missing'})
        for step in main_path['steps']:
            if not set(step['api_refs'])<=set(nodes)or not set(step['edge_refs'])<=set(edges):
                issues.append({'kind':'main_path_unknown_reference','step':step['id']})
        walk(main_path)
    for f in data['files']:link(f['source_url'],site)
    for asset in ['index.html','assets/atlas.css','assets/atlas.js','assets/atlas-data.js',*(['assets/overview-data.js','assets/interface-docs.js','assets/main-path.js']if presentation else [])]:
        if not(site/asset).is_file():issues.append({'kind':'missing_site_asset','path':asset})
    code=(site/'assets/atlas.js').read_text()
    if re.search(r'\bfetch\s*\(|XMLHttpRequest|https?://',code):issues.append({'kind':'unexpected_runtime_network_dependency'})
    if 'contentDocument'in code:issues.append({'kind':'cross_file_dom_access_required'})
    report={'artifact_invariants_passed':not issues,'scope_files':len(data['files']),'nodes':len(nodes),'edges':len(edges),
            'protocols':len(protocols),'protocol_views':sum(len(p['views'])for p in protocols.values()),
            'diagrams':len(diagrams),'linked_local_files':len(linked_files),'issues':issues,
            'diagram_sizes':sizes,'actual_browser_viewports_verified':False,
            'browser_boundary':'file URL and HTTP-preview action denied by current browser/security policy; no workaround used',
            'global_complete':False,'semantic_source_review_is_separate':True}
    (ROOT/'data/atlas-artifact-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in {'issues','diagram_sizes'}},ensure_ascii=False))
    print('issues='+str(len(issues)))
    return report


if __name__=='__main__':raise SystemExit(0 if audit()['artifact_invariants_passed']else 1)
