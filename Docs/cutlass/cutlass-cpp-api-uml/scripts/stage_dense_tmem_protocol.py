#!/usr/bin/env python3
"""Attach the reviewed Dense TMEM protocol to canonical API/call identities.

Only this fixed-source work package is adapted. Event ordering is never added
or changed here. API supplements enter the same declaration enrichment and
callsite checks as the rest of the atlas.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from build_atlas import ROOT,validate_evidence,without_templates
from protocol_diagrams import subdivide_protocol_views


def identifier(prefix,parts):
    return prefix+hashlib.sha256(json.dumps(parts,ensure_ascii=False).encode()).hexdigest()[:20]


def assemble(protocol,base_nodes,base_edges,scope):
    protocol=copy.deepcopy(protocol);nodes=dict(base_nodes);edges=dict(base_edges);added_nodes={};added_edges={}
    known_paths={x['path']for x in scope['files']}
    if protocol['snapshot_commit']!=scope['commit']:raise ValueError('Protocol is not the fixed source revision')
    def node_for(source,preferred=None):
        if preferred:
            if preferred not in nodes:raise ValueError('Unknown explicit API node: '+preferred)
            return preferred
        if not source or source['path']not in known_paths:raise ValueError('API declaration location is required')
        for n in nodes.values():
            if n.get('kind')=='api'and n.get('path')==source['path']and n.get('line')==source['line']:
                if without_templates(n.get('qualified_name',''))==without_templates(source['qualified_name']):return n['id']
        nid=identifier('tmem.api.',[source['path'],source['line'],source['qualified_name']])
        n={'id':nid,'kind':'api','name':source['qualified_name'],'qualified_name':source['qualified_name'],
           'path':source['path'],'line':source['line'],'role':'TMEM协议实际调用路径所需的独立API；签名由固定声明账本关联。'}
        nodes[nid]=n;added_nodes[nid]=n;return nid
    def call_for(source,target,site,condition):
        key=(source,target,site['path'],site['start_line'],site.get('end_line',site['start_line']),site['source_expression'])
        for edge in edges.values():
            c=edge.get('callsite',{})
            if edge.get('relation')=='calls'and edge.get('source')==source and edge.get('target')==target:
                if c.get('path')==site['path']and c.get('start_byte')==site.get('start_byte')and c.get('end_byte')==site.get('end_byte')and site.get('start_byte')is not None:return edge['id']
        eid=identifier('tmem.call.',key)
        callsite={k:v for k,v in site.items()if k in {'path','start_line','end_line','start_byte','end_byte','source_expression','raw_source_expression'}}
        callsite['caller']=source
        edge={'id':eid,'source':source,'target':target,'relation':'calls','resolution':'source_proven',
              'source_expression':site['source_expression'],'condition':condition,'callsite':callsite,
              'evidence':[{'path':site['path'],'start_line':site['start_line'],'end_line':site.get('end_line',site['start_line'])}]}
        errors=[];validate_evidence({eid:edge},scope,errors)
        if errors:raise ValueError('Protocol supplement callsite failed: '+json.dumps(errors,ensure_ascii=False))
        edges[eid]=edge;added_edges[eid]=edge;return eid
    for event in protocol['events']:
        if event.get('kind')=='api_call'and not event.get('api_node_ref'):
            event['api_node_ref']=node_for(event.get('api_source'))
        elif event.get('kind')=='builtin_call'and not event.get('api_node_ref'):
            name=event.get('name')
            if not name:raise ValueError('Builtin name required: '+event['id'])
            nid=identifier('tmem.external.',[name])
            if nid not in nodes:
                n={'id':nid,'name':name,'kind':'external','path':None,'line':None,
                   'role':'CUDA内建接口，声明不属于固定824个CUTLASS库文件；保留实际调用位置，不伪造库内声明。'}
                nodes[nid]=n;added_nodes[nid]=n
            event['api_node_ref']=nid
        if event.get('api_node_ref')and not event.get('call_edge_ref'):
            event['call_edge_ref']=call_for('contract.api.kernel',event['api_node_ref'],event['callsite'],event.get('guard',{}).get('expression',''))
            event['callsite']=copy.deepcopy(edges[event['call_edge_ref']]['callsite'])
    for path in protocol.get('implementation_paths',[]):
        for step in path['steps']:
            if step['kind']=='calls':
                source=node_for(step['source'],step.get('api_node_ref')or step['source'].get('node_ref'))
                target=node_for(step['callee_source'],step.get('callee_node_ref'))
                eid=call_for(source,target,step['callsite'],step.get('guard')or step.get('repeat')or'当前Dense类型绑定下的实际转发')
                step.update(api_node_ref=source,callee_node_ref=target,call_edge_ref=eid)
            elif step['kind']=='hardware_effect':
                source=node_for(step['source']);site=step['instruction_source']
                nid=identifier('tmem.instruction.',[site['path'],site['start_line'],step['opcode']])
                n={'id':nid,'kind':'hardware_event','name':step['opcode'],'path':site['path'],'line':site['start_line'],
                   'source_expression':step['opcode'],'role':'内联PTX指令位置；不是C++函数，也不自动表示异步操作完成。'}
                nodes[nid]=n;added_nodes[nid]=n
                eid=identifier('tmem.issue.',[source,nid])
                edge={'id':eid,'source':source,'target':nid,'relation':'issues','resolution':'source_proven',
                    'source_expression':step['opcode'],'condition':'CUDA_BARRIER_ENABLED; '+step.get('guard',''),
                    'evidence':[site]}
                edges[eid]=edge;added_edges[eid]=edge;step.update(api_node_ref=source,instruction_node_ref=nid,issue_edge_ref=eid)
            else:raise ValueError('Unknown implementation step kind: '+step['kind'])
    eids=list(added_edges);nids=list(dict.fromkeys(nid for eid in eids for nid in [edges[eid]['source'],edges[eid]['target']]))
    protocol['source_views']=copy.deepcopy(protocol['views']);protocol['views']=subdivide_protocol_views(protocol)
    return {'schema_version':1,'part':'tmem_protocol','nodes':list(added_nodes.values()),'edges':list(added_edges.values()),
            'views':[{'id':'tmem.view.api_implementation','title':'TMEM协议：补充的逐API实现路径','node_ids':nids,'edge_ids':eids}],
            'protocols':[protocol],'issues':[],'global_completion_claimed':False,'runtime_execution_claimed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/module-drafts/dense_protocols/tmem_lifetime.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/modules/dense_fp16/tmem_protocol.json')
    args=parser.parse_args();source=args.source.read_bytes();scope=json.loads((ROOT/'data/scope.json').read_text())
    nodes={};edges={};inputs={str(args.source.relative_to(ROOT)):hashlib.sha256(source).hexdigest()}
    for name in ['host.json','type_path.json','contracts.json','gemm_dispatch.json']:
        path=ROOT/'data/modules/dense_fp16'/name;raw=path.read_bytes();part=json.loads(raw)
        nodes.update((n['id'],n)for n in part.get('nodes',[]));edges.update((e['id'],e)for e in part.get('edges',[]))
        inputs[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    result=assemble(json.loads(source),nodes,edges,scope)
    overrides_path=ROOT/'data/modules/dense_fp16/protocol_api_overrides.json'
    overrides_raw=overrides_path.read_bytes();overrides=json.loads(overrides_raw)
    result_nodes={n['id']:n for n in result['nodes']}
    for override in overrides['overrides']:
        node=result_nodes[override['id']]
        node['access']=override['access'];node['manual_declaration']=override['manual_declaration']
        node['identity_correction']={'reason':'Global ledger lost the enclosing cutlass namespace; selected physical scopes verified, global repair remains pending',
            'incorrect_global_entity_id':override['incorrect_global_entity_id'],'evidence':overrides['scope_evidence']}
    inputs[str(overrides_path.relative_to(ROOT))]=hashlib.sha256(overrides_raw).hexdigest()
    result['staging_provenance']={'inputs_sha256':inputs,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy':'API/call references supplemented and display views subdivided; no protocol event, partial order or resource transition removed or added'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'nodes_added':len(result['nodes']),'edges_added':len(result['edges']),'protocol':result['protocols'][0]['id'],'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':main()
