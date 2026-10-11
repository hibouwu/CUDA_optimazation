#!/usr/bin/env python3
"""Independent topological propagation checks the saved BFS support/path result."""
from pathlib import Path
import json,sys,hashlib
out=Path(__file__).resolve().parent;repo=next(p for p in out.parents if (p/'microbench').is_dir())
sys.path.insert(0,str(repo/'microbench/gh200_resource_campaign/access_rules'))
from supply_model import cfg_c_source_port_graph
from output_model import cfg_c_rank_dependencies,dependency_order
report=json.loads((out/'B-cfg-c-supply-response.json').read_text());cache={};paths=0
for q in report['frontiers']:
    key=q['k_tiles'],q['output_tiles']
    if key not in cache:
        g=cfg_c_rank_dependencies(cfg_c_source_port_graph(4,*key));order,_=dependency_order(g);cache[key]=(g,order)
    g,order=cache[key];event=q['event'];distance={}
    for n in order:
        if n==event:distance[n]=0
        else:
            parents=[distance[p] for p in g['nodes'][n] if p in distance]
            if parents:distance[n]=min(parents)+1
    assert len(distance)-1==q['descendants']
    for rank in (0,1):
        earliest=min((a['group'] for a in g['groups'] if f'r{rank}.g{a["group"]}.full' in distance),default=None)
        assert earliest==q['earliest_full_group'][str(rank)]
    assert sorted(n for n in g['markers'] if n.endswith('.FIRST') and n in distance)==q['affected_FIRST']
    for field,n in [('remote_producer_tail_reachable','r0.producer_tail'),('remote_D_visible_reachable','r0.D_visible'),('source_safe_exit_reachable','source_safe_exit'),('cuda_complete_reachable','cuda_complete')]:assert q[field]==(n in distance)
    tile=q['group']//q['k_tiles']
    for marker in ('FIRST','MAIN'):assert q['remote_current_'+marker+'_reachable']==any(f'r0.w{wg}.t{tile}.{marker}' in distance for wg in (0,1))
    path=q['first_remote_full_path']
    if path:
        assert path[0]==event and path[-1]==f'r0.g{q["earliest_full_group"]["0"]}.full'
        assert all(a in g['nodes'][b] for a,b in zip(path,path[1:]))
        assert len(path)-1==distance[path[-1]];paths+=1
    else:assert q['earliest_full_group']['0'] is None
result=dict(queries=len(report['frontiers']),shortest_paths_verified=paths,distinct_graphs=len(cache),method='Topological minimum edge-distance propagation, independent of analyzer BFS',
    report_sha256=hashlib.sha256((out/'B-cfg-c-supply-response.json').read_bytes()).hexdigest(),observation_qualified=False)
(out/'path-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
