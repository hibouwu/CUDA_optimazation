#!/usr/bin/env python3
"""Source-anchored protocol views; never infer temporal order from list order.

This renderer accepts explicit event-anchor partial orders and per-resource
state transitions. It does not construct a protocol from a bag of API names.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import unicodedata
from urllib.parse import quote
import xml.etree.ElementTree as ET


ANCHOR_LABELS={'enter':'调用进入','return':'调用返回','effect':'指定效果满足'}
QUANTIFIER_LABELS={'pointwise_same_thread':'同一线程逐实例',
    'all_sources_to_each_target':'收齐源贡献后，每个目标才能通过',
    'operation_to_each_target':'本侧协作操作 → 各目标实例',
    'operation_to_operation':'明确协作操作之间',
    'all_sources_to_operation':'全部源实例 → 本侧协作操作',
    'per_thread_instance':'每线程实例','local_warp_operation':'本侧warp操作',
    'all_instances':'全部相关实例已记录','any_instance':'至少一个实例已记录','single_local_operation':'本侧协作操作已记录'}


def protocol_wrap(value,width=64):
    """Wrap prose/CJK at character boundaries without cutting API identifiers."""
    value=str(value).replace('"',"'")
    if '\n'in value:return r'\n'.join(protocol_wrap(line,width)for line in value.splitlines())
    tokens=re.split(r'(?<=::)|(?<=/)|(?<=,)|(?<=\s)|(?<=[\u3400-\u9fff，。；、：])',value)
    rows=[];row='';columns=0
    for token in tokens:
        size=sum(2 if unicodedata.east_asian_width(c)in {'W','F'}else 1 for c in token)
        if row and columns+size>width:rows.append(row.rstrip());row='';columns=0
        row+=token;columns+=size
    if row:rows.append(row.rstrip())
    return r'\n'.join(rows).replace('__','~_~_')


def subdivide_protocol_views(protocol):
    """Split the actual Dense views, preserving every authored edge endpoint.

    Local constraints use one actor per panel; cross-actor dependencies remain
    explicit and use compact endpoint labels instead of parallel giant boxes.
    Neither selection nor array order creates a temporal edge.
    """
    events={e['id']:e for e in protocol['events']};actors={a['id']:a for a in protocol['actors']}
    orders={o['id']:o for o in protocol['partial_order']};result=[]
    for view in protocol['views']:
        if view['kind']!='partial_order':
            if view['resource_id'].endswith('.allocation_barrier'):
                transitions={t['id']:t for t in protocol['state_transitions']}
                for branch,excluded in [('mma_fact','epi_only'),('epi_fact','mma_only')]:
                    tids=[tid for tid in view['transition_ids']if not any(transitions[tid][side].endswith('.'+excluded)for side in ['from','to'])]
                    result.append({**view,'id':view['id']+'.'+branch,'transition_ids':tids,
                        'title':view['title']+' · '+('MMA贡献事实推导'if branch=='mma_fact'else'Epilogue贡献事实推导'),
                        'parent_view_id':view['id'],'parent_title':view['title']})
            else:result.append(view)
            continue
        groups=defaultdict(list)
        for oid in view['order_ids']:
            order=orders[oid];pair=tuple(sorted({events[order[side]['event_id']]['actor_id']for side in ['before','after']}))
            groups[pair].append(oid)
        for actor_ids,oids in groups.items():
            limit=6 if len(actor_ids)==1 else 3
            for offset in range(0,len(oids),limit):
                selected=oids[offset:offset+limit]
                event_ids=list(dict.fromkeys(orders[oid][side]['event_id']for oid in selected for side in ['before','after']))
                # Include the authored induced subgraph, including lifecycle
                # and between-event edges whose endpoints are both visible.
                # This makes a handshake legible without inventing order.
                for oid,order in orders.items():
                    if {order['before']['event_id'],order['after']['event_id']}<=set(event_ids)and oid not in selected:selected.append(oid)
                number=1+sum(v.get('parent_view_id')==view['id']for v in result)
                actor_label=' / '.join(actors[a].get('label',a)for a in actor_ids)
                result.append({'id':view['id']+'.part'+str(number),'kind':'partial_order','title':view['title']+' · '+actor_label+' · '+str(number),
                    'event_ids':event_ids,'order_ids':selected,'parent_view_id':view['id'],'parent_title':view['title'],
                    'group_by_actor':len(actor_ids)==1,'scope':'Exact subset of authored constraints; endpoints retain global event identity'})
        shown={eid for fragment in result if fragment.get('parent_view_id')==view['id']for eid in fragment['event_ids']}
        for eid in sorted(set(view['event_ids'])-shown):
            result.append({'id':view['id']+'.event.'+eid,'kind':'partial_order','title':view['title']+' · '+events[eid]['label'],
                'event_ids':[eid],'order_ids':[],'additional_anchors':[{'event_id':eid,'anchor':a}for a in events[eid]['anchors']],
                'parent_view_id':view['id'],'parent_title':view['title'],'group_by_actor':True})
    separated=[]
    for view in result:
        if view['kind']!='partial_order':separated.append(view);continue
        adjacent={eid:set()for eid in view['event_ids']}
        for oid in view['order_ids']:
            a=orders[oid]['before']['event_id'];b=orders[oid]['after']['event_id']
            adjacent[a].add(b);adjacent[b].add(a)
        unseen=set(adjacent);components=[]
        while unseen:
            todo=[min(unseen)];component=set()
            while todo:
                key=todo.pop()
                if key in component:continue
                component.add(key);unseen.discard(key);todo.extend(adjacent[key]-component)
            components.append(component)
        if len(components)==1:separated.append(view);continue
        for index,component in enumerate(components,1):
            separated.append({**view,'id':view['id']+'.component'+str(index),'title':view['title']+' · 独立分量'+str(index),
                'event_ids':[eid for eid in view['event_ids']if eid in component],
                'order_ids':[oid for oid in view['order_ids']if orders[oid]['before']['event_id']in component],
                'additional_anchors':[a for a in view.get('additional_anchors',[])if a['event_id']in component]})
    return separated


def unique(items,label):
    result={item['id']:item for item in items}
    if len(result)!=len(items):raise ValueError('Duplicate '+label+' identity')
    return result


def anchor_key(endpoint):return endpoint['event_id']+'.'+endpoint['anchor']


def verify_protocol(protocol,nodes,edges):
    actors=unique(protocol['actors'],'actor');resources=unique(protocol['resources'],'resource')
    events=unique(protocol['events'],'event');orders=unique(protocol['partial_order'],'partial order')
    transitions=unique(protocol['state_transitions'],'state transition');views=unique(protocol['views'],'protocol view')
    evidence=unique(protocol['evidence'],'protocol evidence')
    for record in [*actors.values(),*resources.values(),*events.values(),*orders.values(),*transitions.values()]:
        for reference in record.get('evidence_refs',[]):
            if reference not in evidence:raise ValueError('Unknown evidence: '+reference)
    for event in events.values():
        if event['actor_id']not in actors:raise ValueError('Unknown event actor: '+event['id'])
        anchors=event['anchors']
        if len(anchors)!=len(set(anchors))or not set(anchors)<=ANCHOR_LABELS.keys():raise ValueError('Invalid event anchors: '+event['id'])
        if not event.get('evidence_refs'):raise ValueError('Event without source evidence: '+event['id'])
        if event.get('api_node_ref')and event['api_node_ref']not in nodes:raise ValueError('Unknown protocol API: '+event['id'])
        if event.get('binding_node_ref')and event['binding_node_ref']not in nodes:raise ValueError('Unknown protocol binding: '+event['id'])
        if event.get('call_edge_ref'):
            edge=edges.get(event['call_edge_ref'])
            if edge is None:raise ValueError('Unknown protocol call edge: '+event['id'])
            if event.get('api_node_ref')and edge['target']!=event['api_node_ref']:
                binding=nodes.get(event.get('binding_node_ref'),{})
                if edge['target']!=event.get('binding_node_ref')or binding.get('instance_of')!=event['api_node_ref']:
                    raise ValueError('Protocol API does not match call target or its explicit binding: '+event['id'])
            authored=event.get('callsite',{});actual=edge.get('callsite',{})
            for field in ['path','start_line','end_line','start_byte','end_byte']:
                if field in authored and authored[field]!=actual.get(field):raise ValueError('Protocol callsite drift: '+event['id']+'/'+field)
    def check_anchor(endpoint):
        event=events.get(endpoint['event_id'])
        if event is None or endpoint['anchor']not in event['anchors']:raise ValueError('Unknown event anchor: '+str(endpoint))
    adjacency=defaultdict(set)
    for order in orders.values():
        check_anchor(order['before']);check_anchor(order['after'])
        if not order.get('evidence_refs'):raise ValueError('Order without source evidence: '+order['id'])
        adjacency[anchor_key(order['before'])].add(anchor_key(order['after']))
    # A cycle would claim mutually preceding milestones. Alternatives should
    # have separately guarded event identities, not contradictory fixed edges.
    visiting=set();done=set()
    def visit(key):
        if key in visiting:raise ValueError('Cyclic protocol partial order: '+key)
        if key in done:return
        visiting.add(key)
        for nxt in adjacency.get(key,()):visit(nxt)
        visiting.remove(key);done.add(key)
    for key in list(adjacency):visit(key)
    quantified=bool(protocol.get('semantics',{}).get('composition'))
    if quantified:
        for event in events.values():
            if not event.get('instance_domain'):raise ValueError('Missing event instance domain: '+event['id'])
            if set(event.get('anchor_quantifiers',{}))!=set(event['anchors']):raise ValueError('Missing anchor quantifier: '+event['id'])
        for order in orders.values():
            q=order.get('quantifier',{});source=events[order['before']['event_id']];target=events[order['after']['event_id']]
            if q.get('source_domain_ref')!=source['id']+'.instance_domain'or q.get('target_domain_ref')!=target['id']+'.instance_domain':
                raise ValueError('Quantified order domain drift: '+order['id'])
            if q.get('kind')=='pointwise_same_thread'and source['actor_id']!=target['actor_id']:
                raise ValueError('Pointwise order cannot equate different actors: '+order['id'])
            if order.get('relation')=='program_order'and q.get('kind')!='pointwise_same_thread':
                raise ValueError('Program order was promoted to a family barrier: '+order['id'])
    for transition in transitions.values():
        resource=resources.get(transition['resource_id'])
        if resource is None:raise ValueError('Unknown transition resource: '+transition['id'])
        states=unique(resource['states'],'resource state')
        if transition['from']not in states or transition['to']not in states:raise ValueError('Unknown resource state: '+transition['id'])
        check_anchor(transition['trigger'])
        if not transition.get('evidence_refs'):raise ValueError('Transition without source evidence: '+transition['id'])
        if quantified:
            derivation=transition.get('derivation',{})
            if derivation.get('kind')!='monotone_history_fact_rule'or derivation.get('adds_temporal_order')is not False or derivation.get('consumes_from_fact')is not False:
                raise ValueError('State inference would manufacture time or consume history: '+transition['id'])
            expected={'kind':'state_fact','resource_id':transition['resource_id'],'state_id':transition['to']}
            if derivation.get('conclusion')!=expected:raise ValueError('State conclusion differs from transition: '+transition['id'])
            observed=[c for c in derivation.get('all_of',[])if c.get('kind')=='observed_anchor_history']
            if len(observed)!=1 or observed[0].get('point')!=transition['trigger']or observed[0].get('quantifier')!=transition.get('trigger_quantifier'):
                raise ValueError('State trigger or quantifier lost: '+transition['id'])
    for rule in protocol.get('validity_rules',[]):
        if rule.get('explicitly_not_temporal')is not True or not rule.get('all_of'):raise ValueError('Invalid logical acceptance rule: '+rule['id'])
        for clause in rule['all_of']:
            if clause.get('kind')=='qualified_anchor_observed':check_anchor(clause['point'])
    covered_orders=set();covered_transitions=set()
    for view in views.values():
        if view['kind']=='partial_order':
            if not set(view['event_ids'])<=events.keys()or not set(view['order_ids'])<=orders.keys():raise ValueError('Unknown view reference: '+view['id'])
            for oid in view['order_ids']:
                if not {orders[oid]['before']['event_id'],orders[oid]['after']['event_id']}<=set(view['event_ids']):
                    raise ValueError('View loses event endpoint: '+oid)
            covered_orders.update(view['order_ids'])
        elif view['kind']=='state':
            if view['resource_id']not in resources or not set(view['transition_ids'])<=transitions.keys():raise ValueError('Unknown state view reference: '+view['id'])
            if any(transitions[x]['resource_id']!=view['resource_id']for x in view['transition_ids']):raise ValueError('View mixes independent resource states: '+view['id'])
            covered_transitions.update(view['transition_ids'])
        else:raise ValueError('Unsupported protocol view kind: '+view['kind'])
    if covered_orders!=orders.keys():raise ValueError('Some partial orders are not visible')
    if covered_transitions!=transitions.keys():raise ValueError('Some state transitions are not visible')
    return {'actors':actors,'resources':resources,'events':events,'orders':orders,'transitions':transitions,'evidence':evidence}


def protocol_puml(protocol,view,nodes,edges,wrap):
    registry=verify_protocol(protocol,nodes,edges);events=registry['events'];actors=registry['actors']
    wrap=protocol_wrap
    def link(event_id):return '../index.html?'+str('protocol='+quote(protocol['id'],safe='')+'&event='+quote(event_id,safe=''))
    def event_label(event,anchor):
        api=nodes.get(event.get('api_node_ref'),{});site=event.get('callsite',{})
        location=site.get('path','')+(':'+str(site['start_line'])if site.get('start_line')else'')
        actor=actors[event['actor_id']].get('label',event['actor_id'])if not view.get('group_by_actor',True)else''
        return wrap('\n'.join(x for x in [actor,event.get('label')or api.get('name')or event['id'],ANCHOR_LABELS[anchor],location]if x),48)
    lines=['@startuml','skinparam backgroundColor #ffffff','skinparam shadowing false',
           'skinparam defaultFontName Noto Sans CJK SC','skinparam defaultFontSize 14',
           'skinparam ArrowFontSize 12','skinparam svgLinkTarget _top','top to bottom direction',
           'skinparam stateBackgroundColor #F4F7FB','skinparam stateBorderColor #57718A',
           'title '+wrap(view['title'],90)]
    if view['kind']=='partial_order':
        selected=[registry['orders'][identifier]for identifier in view['order_ids']]
        endpoints={anchor_key(o[side]):o[side]for o in selected for side in ['before','after']}
        for endpoint in view.get('additional_anchors',[]):endpoints[anchor_key(endpoint)]=endpoint
        event_ids=sorted({endpoint['event_id']for endpoint in endpoints.values()})
        aliases={eid:'e'+str(i)for i,eid in enumerate(event_ids)}
        grouped=defaultdict(list)
        for eid in event_ids:grouped[events[eid]['actor_id']].append(eid)
        lines+=['hide circle','skinparam classAttributeIconSize 0','skinparam classBackgroundColor #F4F7FB',
                'skinparam classBorderColor #57718A','skinparam ArrowFontSize 11']
        for index,(actor,items)in enumerate(grouped.items()):
            if view.get('group_by_actor',True):lines.append('package "'+wrap(actors[actor].get('label')or actors[actor].get('name')or actor,45)+'" as actor'+str(index)+' {')
            for eid in items:
                event=events[eid];api=nodes.get(event.get('api_node_ref'),{});site=event.get('callsite',{})
                location=site.get('path','')+(':'+str(site['start_line'])if site.get('start_line')else'')
                label='\n'.join(x for x in [actors[actor].get('label',actor),event.get('label')or api.get('name')or eid,location]if x)
                lines.append('class "'+wrap(label,48)+'" as '+aliases[eid]+' <<event>> [['+link(eid)+']] {')
                for anchor in ['enter','effect','return']:
                    if eid+'.'+anchor in endpoints:lines.append('  '+anchor+' : '+ANCHOR_LABELS[anchor])
                for order in selected:
                    if order['before']['event_id']==eid==order['after']['event_id']:
                        text=order['id']+' : '+order['before']['anchor']+' → '+order['after']['anchor']
                        text+=' · '+QUANTIFIER_LABELS.get(order.get('quantifier',{}).get('kind'),'明确约束')
                        lines.append('  '+wrap(text,70))
                lines.append('}')
            if view.get('group_by_actor',True):lines.append('}')
        for order in selected:
            label=order['id']+'\n'+str(order.get('condition')or order.get('relation',''))
            if order.get('quantifier'):label+='\n'+QUANTIFIER_LABELS.get(order['quantifier']['kind'],order['quantifier']['kind'])
            before=order['before'];after=order['after']
            if before['event_id']==after['event_id']:continue
            lines.append(aliases[before['event_id']]+'::'+before['anchor']+' --> '+aliases[after['event_id']]+'::'+after['anchor']+' : '+wrap(label,54))
        lines+=['legend bottom','箭头只表示记录的偏序约束；图中高度和数组顺序均不构成额外先后关系。',
                'enter / return / effect分别指调用进入、返回和明确定义的效果；不将返回等同异步完成。',
                'event框不是C++类；同一API内部约束按ID列在框内，完整条件见相应记录。',
                '跨API约束逐条连线，不同API和CTA事件不合并。','endlegend']
    else:
        resource=registry['resources'][view['resource_id']];states=unique(resource['states'],'resource state')
        selected=[registry['transitions'][identifier]for identifier in view['transition_ids']]
        used={state for transition in selected for state in [transition['from'],transition['to']]}
        aliases={key:'s'+str(i)for i,key in enumerate(sorted(used))}
        for key in sorted(used):
            state=states[key];lines.append('state "'+wrap(state.get('label')or key,55)+'" as '+aliases[key])
            invariants=state.get('invariants',[])
            if invariants:lines+=['note right of '+aliases[key],wrap('\n'.join(invariants),65).replace(r'\n','\n'),'end note']
        for transition in selected:
            event=events[transition['trigger']['event_id']]
            label=transition['id']+'\n'+(event.get('label')or event['id'])+' · '+ANCHOR_LABELS[transition['trigger']['anchor']]
            if transition.get('trigger_quantifier'):label+='\n'+QUANTIFIER_LABELS.get(transition['trigger_quantifier']['kind'],transition['trigger_quantifier']['kind'])
            # Full premises remain in the linked rule record. State views are
            # navigation over proof facts, not a second flattened timeline.
            lines.append(aliases[transition['from']]+' --> '+aliases[transition['to']]+' : [['+link(event['id'])+' '+wrap(label,64)+']]')
        lines+=['legend bottom','状态是非互斥的历史事实判定；箭头表示推导规则，不添加API时间顺序。',
                '实际先后只由量词化偏序规定；本图不是本轮GPU执行轨迹。','endlegend']
    return '\n'.join(lines+['@enduml'])+'\n'


def protocol_dot(protocol,view,nodes,edges):
    """Exact row ports avoid PlantUML's self-member arrows crossing API text.

    This is a second rendering of the same authored constraints, not a second
    semantic model. PlantUML remains the editable UML export; DOT is retained
    as the reproducible SVG layout input.
    """
    registry=verify_protocol(protocol,nodes,edges);events=registry['events'];actors=registry['actors']
    selected=[registry['orders'][oid]for oid in view['order_ids']]
    endpoints={anchor_key(o[side]):o[side]for o in selected for side in ['before','after']}
    for point in view.get('additional_anchors',[]):endpoints[anchor_key(point)]=point
    event_ids=sorted({p['event_id']for p in endpoints.values()});aliases={eid:'e'+str(i)for i,eid in enumerate(event_ids)}
    def quoted(value):return json.dumps(value.replace(r'\n','\n'),ensure_ascii=False)
    def rich(value,width=48):return html.escape(protocol_wrap(value,width).replace('~_','_')).replace(r'\n','<BR ALIGN="LEFT"/>')
    def url(event=None,order=None):
        result='../index.html?protocol='+quote(protocol['id'],safe='')+'&protocol_view='+quote(view['id'],safe='')
        if event:result+='&event='+quote(event,safe='')
        if order:result+='&order='+quote(order,safe='')
        return result
    lines=['digraph protocol {','graph [rankdir=TB, bgcolor="white", pad="0.18", nodesep="0.5", ranksep="0.7", splines=polyline, fontname="Noto Sans CJK SC", fontsize=15, labelloc=t];',
           'node [shape=plain, fontname="Noto Sans CJK SC", fontsize=14];',
           'edge [fontname="Noto Sans CJK SC", fontsize=11, color="#235F91", arrowsize=0.7, dir=both, arrowtail=dot, arrowhead=normal];',
           'label='+quoted(protocol_wrap(view['title'],90).replace('~_','_'))+';']
    for eid in event_ids:
        event=events[eid];actor=actors[event['actor_id']];site=event.get('callsite',{})
        path=site.get('path','')+(':'+str(site['start_line'])if site.get('start_line')else'')
        header='\n'.join([actor.get('label',event['actor_id']),event.get('label',eid),path])
        effect=event.get('effect')or{}
        if 'predicate_value'in effect:header+='\npredicate='+str(effect['predicate_value']).lower()+'；arrival='+str(effect.get('arrival_family_size','见事件定义'))
        rows=['<TR><TD BGCOLOR="#ECF2F8" ALIGN="LEFT" HREF="'+html.escape(url(event=eid),quote=True)+'" TARGET="_top">'+rich(header)+'</TD></TR>']
        for anchor in ['enter','effect','return']:
            if eid+'.'+anchor not in endpoints:continue
            q=event.get('anchor_quantifiers',{}).get(anchor,{})
            label=anchor+' · '+ANCHOR_LABELS[anchor]
            if q:label+=' / '+QUANTIFIER_LABELS.get(q['kind'],q['kind'])
            rows.append('<TR><TD PORT="'+anchor+'" ALIGN="LEFT" HREF="'+html.escape(url(event=eid),quote=True)+'" TARGET="_top">'+rich(label)+'</TD></TR>')
        # Intra-API constraints retain individual identities inside the event
        # box; drawing them as self-loops crosses text in some UML engines and
        # overwhelms the cross-API handshake. No different APIs are merged.
        for order in selected:
            if order['before']['event_id']==eid==order['after']['event_id']:
                label=order['id']+' · '+order['before']['anchor']+' → '+order['after']['anchor']+'\n'+QUANTIFIER_LABELS.get(order.get('quantifier',{}).get('kind'),'明确约束')
                rows.append('<TR><TD ALIGN="LEFT" BGCOLOR="#F7F9FC" HREF="'+html.escape(url(order=order['id']),quote=True)+'" TARGET="_top"><FONT POINT-SIZE="11">'+rich(label)+'</FONT></TD></TR>')
        lines.append(aliases[eid]+' [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8" COLOR="#57718A">'+''.join(rows)+'</TABLE>>];')
    for order in selected:
        before=order['before'];after=order['after'];same=before['event_id']==after['event_id']
        if same:continue
        label=order['id']+'\n'+QUANTIFIER_LABELS.get(order.get('quantifier',{}).get('kind'),'明确约束')+'\n'+str(order.get('condition')or order.get('relation',''))
        label=protocol_wrap(label,52).replace('~_','_')
        start=aliases[before['event_id']]+':'+before['anchor']+':e'
        end=aliases[after['event_id']]+':'+after['anchor']+':e'
        lines.append(start+' -> '+end+' [id='+quoted(order['id'])+', label='+quoted(label)+', URL='+quoted(url(order=order['id']))+', target="_top"];')
    lines+=['legend [shape=box, color="#BAC8D5", fontsize=12, label="协议导航：蓝色小点为源锚点，箭头指向目标锚点。\n不同API/CTA独立，跨API约束逐条连线。\n同一API内部约束按ID逐条列在框内；点击ID查看完整条件。\n同线程关系不是整族屏障；框内行序不添加时间关系。"];','{rank=sink; legend;}','}']
    return '\n'.join(lines)+'\n'


def render_protocols(root,site,protocols,nodes,edges,issues,wrap):
    directory=site/'diagrams';directory.mkdir(parents=True,exist_ok=True);pending=[];all_views=[]
    for protocol in protocols:
        verify_protocol(protocol,nodes,edges)
        for view in protocol['views']:
            name='protocol-'+re.sub(r'[^A-Za-z0-9_.-]','_',view['id'])[:100]+'_'+hashlib.sha256(view['id'].encode()).hexdigest()[:8]
            puml=directory/(name+'.puml');svg=directory/(name+'.svg');content=protocol_puml(protocol,view,nodes,edges,wrap)
            unchanged=puml.is_file()and svg.is_file()and puml.read_text()==content
            puml.write_text(content)
            if not unchanged:pending.append(puml)
            view.update(svg='diagrams/'+svg.name,plantuml='diagrams/'+puml.name,protocol_id=protocol['id'])
            all_views.append(view)
    if pending:
        env=dict(os.environ);env['JAVA_TOOL_OPTIONS']='-Djava.awt.headless=true';env['PLANTUML_LIMIT_SIZE']='24000'
        result=subprocess.run(['plantuml','-tsvg','-charset','UTF-8','-nometadata',*[str(p)for p in pending]],env=env,text=True,capture_output=True)
        (site/'protocol-render.log').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:issues.append({'kind':'protocol_render_failed','returncode':result.returncode})
    # Validate all PlantUML exports above. For event/member ports, Graphviz's
    # explicit HTML-table ports give non-overlapping SVG endpoints; save DOT
    # alongside PlantUML so both representations can be inspected/regenerated.
    for protocol in protocols:
        for view in protocol['views']:
            if view['kind']!='partial_order':continue
            dot=site/view['plantuml'].replace('.puml','.dot');content=protocol_dot(protocol,view,nodes,edges)
            svg=site/view['svg'];checksum=hashlib.sha256(content.encode()).hexdigest()
            unchanged=dot.is_file()and dot.read_text()==content and svg.is_file()
            if unchanged:unchanged=ET.parse(svg).getroot().get('data-dot-sha256')==checksum
            dot.write_text(content);view['dot']=str(dot.relative_to(site));view['svg_renderer']='graphviz_explicit_row_ports'
            if not unchanged:
                result=subprocess.run(['dot','-Tsvg',str(dot),'-o',str(site/view['svg'])],text=True,capture_output=True)
                if result.returncode:issues.append({'kind':'protocol_dot_render_failed','id':view['id'],'error':result.stderr})
                else:
                    tree=ET.parse(svg);element=tree.getroot();box=element.get('viewBox','').split()
                    if len(box)==4:element.set('width',box[2]+'px');element.set('height',box[3]+'px')
                    element.set('data-dot-sha256',checksum);element.set('data-renderer','graphviz_explicit_row_ports')
                    ET.register_namespace('','http://www.w3.org/2000/svg');ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
                    tree.write(svg,encoding='utf-8',xml_declaration=True)
    for view in all_views:
        svg=site/view['svg']
        if not svg.is_file():issues.append({'kind':'protocol_svg_missing','id':view['id']});continue
        parsed=ET.parse(svg).getroot();text=''.join(parsed.itertext())
        if 'Syntax Error'in text:issues.append({'kind':'protocol_svg_syntax_error','id':view['id']})
        view['svg_dimensions']={key:float(re.sub(r'[^0-9.]','',parsed.get(key,'0')))for key in ['width','height']}
    return all_views
