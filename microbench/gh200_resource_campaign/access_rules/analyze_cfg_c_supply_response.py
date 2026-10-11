#!/usr/bin/env python3
"""Closed cfg_c source binding and conditional influence frontiers, without prices."""
import argparse
from collections import deque
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

from supply_model import cfg_c_source_bindings,cfg_c_source_port_graph


def compressed(value):
    return gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0)


def influence(graph,children,event):
    """Descendant support only; reachability is not a guaranteed positive delay."""
    if event not in graph['nodes']:raise ValueError('unknown intervention event')
    parents={event:None};queue=deque([event])
    while queue:
        current=queue.popleft()
        for child in children[current]:
            if child not in parents:parents[child]=current;queue.append(child)
    earliest={}
    for rank in range(2):
        affected=[g['group'] for g in graph['groups'] if f'r{rank}.g{g["group"]}.full' in parents]
        earliest[str(rank)]=min(affected,default=None)
    path=[]
    if earliest['0'] is not None:
        current=f'r0.g{earliest["0"]}.full'
        while current is not None:path.append(current);current=parents[current]
        path.reverse()
    return dict(event=event,descendants=len(parents)-1,earliest_full_group=earliest,
        affected_FIRST=sorted(n for n in graph['markers'] if n.endswith('.FIRST') and n in parents),
        first_remote_full_path=path,
        remote_producer_tail_reachable='r0.producer_tail' in parents,
        remote_D_visible_reachable='r0.D_visible' in parents,
        source_safe_exit_reachable='source_safe_exit' in parents,cuda_complete_reachable='cuda_complete' in parents)


def analyze(root,c_model_path,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        path=path.resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in inputs and inputs[str(path)]!=digest:raise ValueError('input changed: '+str(path))
        inputs[str(path)]=digest
        return digest
    def read(path):
        sha(path);return json.loads(path.read_text())
    base=root/'20261009-R13-R15-R18-composition-job738496/reanalysis'
    previous=base/'B-20261011-cfg-c-source-ports-v1'
    prior=read(previous/'B-cfg-c-source-ports.json')
    rechecked=0
    for name,digest in prior['input_sha256'].items():
        path=Path(name)
        if path.is_relative_to(root):
            if sha(path)!=digest:raise ValueError('prior data input changed')
            rechecked+=1
    bindings_path=previous/'B-cfg-c-source-bindings.json.gz';sha(bindings_path)
    bindings=json.loads(gzip.open(bindings_path,'rt').read());rebuilt=[]
    for item in bindings:
        run=root/item['run']
        row=next(r for r in read(run/'cases.json') if r['id']==item['case'])
        setup=next(v['setup'] for v in read(run/'static_setup.json') if v['case']==item['case'])
        rebuilt.append(dict(run=item['run'],case=item['case'],binding=cfg_c_source_bindings(row,setup)))
    if compressed(rebuilt)!=bindings_path.read_bytes():raise ValueError('physical source bindings changed')
    sha(c_model_path)
    spec=importlib.util.spec_from_file_location('cfg_c_C_closed_rank_input',c_model_path)
    c_model=importlib.util.module_from_spec(spec);spec.loader.exec_module(c_model)
    audits=[];frontiers=[];cases_for_frontier={(4,4,1),(4,16,2),(4,64,2)}
    main=base/'main-cfg-c-rank-dependencies-v2'
    for template in prior['source_templates']:
        dimensions=(template['stages'],template['k_tiles'],template['output_tiles'])
        source=cfg_c_source_port_graph(*dimensions);path=previous/template['artifact'];sha(path)
        if compressed(source)!=path.read_bytes():raise ValueError('source-port template bytes changed')
        graph=c_model.cfg_c_rank_dependencies(source);order,children=c_model.dependency_order(graph)
        preserved=0
        for node,parents in source['nodes'].items():
            if not set(parents)<=set(graph['nodes'][node]):raise ValueError('source predecessor removed')
            preserved+=len(parents)
        if ({n for n,p in graph['nodes'].items() if not p}!={'cluster_initialized'}
                or graph['external_producer_ports'] or graph['external_entry_ports']!=['cluster_initialized']
                or graph['input_requests']!=source['input_requests'] or graph['input_deliveries']!=source['input_deliveries']
                or graph['input_totals']!=source['totals']):raise ValueError('closed lifecycle/source contract changed')
        for g in range(dimensions[1]*dimensions[2]):
            for rank in range(2):
                prefix=f'r{rank}.g{g}.'
                expected=[f'r{rank}.g{g-1}.B_issue'] if g else ['cluster_initialized']
                if g>=4:expected+=[f'r{s}.w{w}.g{g-4}.empty_to_r{rank}' for s in range(2) for w in range(2)]
                if (set(graph['nodes'][prefix+'acquire'])!=set(expected)
                        or graph['nodes'][prefix+'arrive_expect_tx']!=[prefix+'acquire']
                        or graph['nodes'][prefix+'acquire_return']!=[prefix+'arrive_expect_tx']):
                    raise ValueError('source roots overwrote finite-stage predecessors')
        name=f'cfg_c-S{dimensions[0]}-K{dimensions[1]}-T{dimensions[2]}-rank-dependencies.json.gz'
        canonical=main/name;sha(canonical);encoded=compressed(graph)
        if encoded!=canonical.read_bytes():raise ValueError('independent closed graph differs from main canonical bytes')
        (output/name).write_bytes(encoded)
        audits.append(dict(stages=dimensions[0],k_tiles=dimensions[1],output_tiles=dimensions[2],
            nodes=len(order),source_edges_preserved=preserved,artifact=name,main_graph_bytes_identical=True))
        if dimensions in cases_for_frontier:
            frontiers.extend(check_frontiers(graph,children))
    # A short synthetic output pair exposes the prologue exception and initial empty slots.
    for kt,T in ((2,2),(4,2)):
        graph=c_model.cfg_c_rank_dependencies(cfg_c_source_port_graph(4,kt,T))
        _,children=c_model.dependency_order(graph)
        frontiers.extend(check_frontiers(graph,children,synthetic=True))
    for name in ('analyze_cfg_c_supply_response.py','supply_model.py'):
        sha(Path(__file__).with_name(name))
    sha(c_model_path)
    report=dict(scope='Closed cfg_c source composition and conditional causal support; no service fit or timing forecast.',
        closed_templates=audits,physical_binding_cases=len(bindings),binding_bytes_identical=True,
        physical_clusters=sum(len(x['binding']['clusters']) for x in bindings),
        rank_outputs=sum(len(o) for x in bindings for c in x['binding']['clusters'] for o in c['rank_outputs']),
        reused_numeric_replay=dict(processes=prior['archived_processes_replayed'],ctas=prior['archived_ctas_replayed'],
            checked_values=prior['checked_sample_values'],data_inputs_rehashed=rechecked,new_numeric_replay=False),
        frontiers=frontiers,input_sha256=inputs,
        assumptions=['Only the named event completion changes; roots, other edge lags and resource organization are held fixed.',
            'No new global source/server or instruction-scheduler competition edge is added.',
            'Descendants may be delayed; unreachable events cannot change under this restricted model.',
            'Reachability alone does not imply a positive response, a magnitude or critical-path activation.'],
        conclusions=['A multicast source_ready can affect both full tokens in its own group.',
            'A local B source_ready first reaches remote full at g+S in an output prologue, otherwise g+S-1.',
            'A B0 issue delay can affect remote full at g+1 through producer order and next A multicast.',
            'If the first remote group is beyond the invocation, remote D may stay outside the cone while remote producer_tail/source_safe_exit remains inside.',
            'These distinct support predictions do not identify real fill latency, root shifts or shared hardware resources.'],
        limitations=['cluster_initialized remains externally unpriced; closed structure is not a zero-root autonomous timing model.',
            'Actual delay injection can add instruction/control effects outside the named event; it requires separate source/observer qualification.',
            'Prior 150-process numeric/work evidence is reused after data SHA checks; V09 remains static geometry and original failed scores stay unchanged.',
            'No new GPU pair is requested until a quantitative candidate and qualified endpoints provide distinct frozen predictions.'])
    (output/'B-cfg-c-supply-response.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# cfg_c闭合供给图：源事件影响的最早边界','',
        '26份物理绑定与12份源模板逐字节保持；闭合图与主线12份规范gzip逐字节相同。',
        '以下只改变所列事件，固定入口/其它边成本，不加入新共享资源；可达不等于必然变慢。','',
        '| Ktiles/T | 源事件 | g | 最早remote full | remote当前FIRST/MAIN可达 |','|---|---|---:|---:|---|']
    for r in frontiers:
        if r['kind'] not in ('B0_source_ready','B0_issue') or r['k_tiles']!=16:continue
        lines.append(f'| {r["k_tiles"]}/{r["output_tiles"]} | {r["kind"]} | {r["group"]} | '
            f'{r["earliest_full_group"]["0"]} | {r["remote_current_FIRST_reachable"]}/{r["remote_current_MAIN_reachable"]} |')
    lines+=['','B完成迟延通过release/empty影响伙伴：prologue为g+4，普通位置为g+3；不能一律写成四组以后。',
        'B发出迟延另可通过下一producer工作及A多播，在g+1影响伙伴。A完成迟延则本group直接影响两rank。',
        '短K无后续full时，伙伴D不一定受影响，producer尾部/全局退出仍可能等待远端empty；两种完成边界不能合并。',
        '源码图、逐事件最短见证路径及不可达标记保存在JSON；未赋服务价、未锚定实际入口，也未增加观察资格。','']
    (output/'B-cfg-c-supply-response.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),closed_templates=len(audits),frontier_queries=len(frontiers))))


def check_frontiers(graph,children,synthetic=False):
    kt,T=graph['k_tiles'],graph['output_tiles'];total=kt*T
    groups=sorted({g for g in (0,1,3,kt-1,kt) if g<total});results=[]
    for g in groups:
        for kind in ('A_source_ready','B0_source_ready','B0_issue','acquire_return','empty_emit_to_r0','empty_to_r0'):
            event=f'r1.g{g}.{kind}' if not kind.startswith('empty') else f'r1.w0.g{g}.{kind}'
            result=influence(graph,children,event)
            expected=g if kind in ('A_source_ready','acquire_return') else g+1 if kind=='B0_issue' else g+4-(g%kt>0) if kind=='B0_source_ready' else g+4
            expected=expected if expected<total else None
            if result['earliest_full_group']['0']!=expected:raise ValueError('remote influence frontier differs from recurrence')
            tile=g//kt;names=set(result['affected_FIRST'])
            result.update(kind=kind,group=g,k_tiles=kt,output_tiles=T,synthetic_geometry=synthetic,
                remote_current_FIRST_reachable=any(f'r0.w{w}.t{tile}.FIRST' in names for w in range(2)))
            # MAIN is not a FIRST marker, so check its descendant support separately.
            reachable=set();queue=[event]
            while queue:
                name=queue.pop()
                if name not in reachable:reachable.add(name);queue.extend(children[name])
            result['remote_current_MAIN_reachable']=any(f'r0.w{w}.t{tile}.MAIN' in reachable for w in range(2))
            results.append(result)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--c-output-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.c_output_model.resolve(),args.output.resolve())
