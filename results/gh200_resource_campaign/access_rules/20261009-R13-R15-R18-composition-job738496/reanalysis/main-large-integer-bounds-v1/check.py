#!/usr/bin/env python3
"""Project the closed DAG onto saved globaltimer markers, without service prices."""
from pathlib import Path
import gzip,hashlib,json,sys,subprocess,types
out=Path(__file__).resolve().parent;run=out.parents[1];root=run.parent
repo=next(p for p in out.parents if (p/'microbench').is_dir());code=repo/'microbench/gh200_resource_campaign/access_rules'
sys.path.insert(0,str(code));from output_model import dependency_order,dependency_bounds
old=types.ModuleType("old_bounds")
old_source=subprocess.check_output(["git","show","696e0b5:microbench/gh200_resource_campaign/access_rules/output_model.py"],cwd=repo,text=True)
exec(old_source,old.__dict__)
sources=run/'reanalysis/main-cfg-c-source-ports-v1';graphs=run/'reanalysis/main-cfg-c-rank-dependencies-v2'
inputs={}
def bind(p):inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):bind(p);return json.loads(p.read_text())
def gz(p):bind(p);return json.loads(gzip.decompress(p.read_bytes()))
prior=read(sources/'B-cfg-c-source-ports.json');bindings=gz(sources/'B-cfg-c-source-bindings.json.gz')
contracts=read(graphs/'C-cfg-c-rank-dependencies.json');cache={}
for template in contracts['source_templates']:
    graph=gz(graphs/template['artifact']);cache[graph['k_tiles'],graph['output_tiles']]=graph
results=[];checked=violations=0
for case in prior['cases']:
    if case['actual_work_processes_replayed']!=10:continue
    data=root/case['run'];binding=next(b['binding'] for b in bindings if (b['run'],b['case'])==(case['run'],case['case']))
    paths=sorted((data/'samples'/case['case']).glob('dual-*.json'));assert len(paths)==10
    for path in paths:
        record=read(path);raw=data/record['raw'];bind(raw)
        assert not record['returncode'] and inputs[str(raw)]==record['raw_sha256']==prior['input_sha256'][str(raw)]
        with gzip.open(raw,'rt') as f:packets=[json.loads(line) for line in f]
        events={p['event']:p for p in packets};assert len(events)==len(packets)
        setup,call=events['setup'],events['call'];trace=call['trace']
        assert setup['trace_tile_words']==10 and len(trace)==setup['grid'][0]*setup['grid'][1]*1296
        cluster_results=[]
        for cluster in binding['clusters'][:1]:
            T=cluster['output_tiles'];anchors={}
            for rank,cta in enumerate(cluster['ctas']):
                b=trace[cta*1296:(cta+1)*1296];assert b[2]==b[11]==b[12] and b[2]>0
                for wg in (0,1):
                    assert b[6+3*wg]==T
                    for j in range(T):
                        t=b[16+wg*640+j*10:26+wg*640+j*10]
                        assert [v-1 for v in t[4:6]]==cluster['rank_outputs'][rank][j]['coord']
                        for event,n in zip(('FIRST','MAIN','PERMIT','DONE'),t[6:10]):
                            assert n>0;anchors[f'r{rank}.w{wg}.t{j}.{event}']=n
                    anchors[f'r{rank}.w{wg}.FINAL']=b[5+3*wg]
            graph=cache[binding['k_tiles'],T];origin=min(anchors.values())
            absolute={n:(v,v) for n,v in anchors.items()};relative={n:(v-origin,v-origin) for n,v in anchors.items()}
            baseline=dependency_bounds(graph,relative);current=dependency_bounds(graph,absolute)
            normalized={n:[v-origin if v is not None else None for v in values] for n,values in current.items()}
            assert normalized==baseline
            try:old.dependency_bounds(graph,absolute);old_rejected=False
            except ValueError:old_rejected=True
            assert old.dependency_bounds(graph,relative)==baseline
            cluster_results.append(dict(ctas=cluster['ctas'],outputs=T,nodes=len(current),old_absolute_rejected=old_rejected,current_translation_exact=True))
            checked+=len(current);violations+=old_rejected
        results.append(dict(run=case['run'],case=case['case'],trial=record['trial'],clusters=cluster_results))
bind(code/'output_model.py');bind(Path(__file__))
report=dict(input_sha256=inputs,old_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),old_revision='696e0b5',
    processes=len(results),selected_clusters=sum(len(r['clusters']) for r in results),complete_node_intervals_compared=checked,
    old_absolute_rejections=violations,current_translation_failures=0,results=results,
    scope='First physical cluster of each of 150 archived processes, chosen before results; zero-lag integer anchor translation regression only.',
    observation_qualified=False)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','results')}))
