#!/usr/bin/env python3
"""Project the closed DAG onto saved globaltimer markers, without service prices."""
from pathlib import Path
import gzip,hashlib,json,sys
out=Path(__file__).resolve().parent;run=out.parents[1];root=run.parent
repo=next(p for p in out.parents if (p/'microbench').is_dir());code=repo/'microbench/gh200_resource_campaign/access_rules'
sys.path.insert(0,str(code));from output_model import dependency_order
sources=run/'reanalysis/main-cfg-c-source-ports-v1';graphs=run/'reanalysis/main-cfg-c-rank-dependencies-v2'
inputs={}
def bind(p):inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):bind(p);return json.loads(p.read_text())
def gz(p):bind(p);return json.loads(gzip.decompress(p.read_bytes()))
prior=read(sources/'B-cfg-c-source-ports.json');bindings=gz(sources/'B-cfg-c-source-bindings.json.gz')
contracts=read(graphs/'C-cfg-c-rank-dependencies.json');cache={}
for template in contracts['source_templates']:
    graph=gz(graphs/template['artifact']);order,_=dependency_order(graph)
    markers=set(graph['markers']);ancestors={}
    for n in order:
        a=set()
        for p in graph['nodes'][n]:
            a.update(ancestors[p])
            if p in markers:a.add(p)
        ancestors[n]=a
    pairs=[(a,b) for b in sorted(markers) for a in sorted(ancestors[b])]
    cache[graph['k_tiles'],graph['output_tiles']]=pairs
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
        for cluster in binding['clusters']:
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
            pairs=cache[binding['k_tiles'],T];bad=[dict(before=a,after=b,violation_ns=anchors[a]-anchors[b]) for a,b in pairs if anchors[a]>anchors[b]]
            maxgap=max((b['violation_ns'] for b in bad),default=0)
            cluster_results.append(dict(ctas=cluster['ctas'],outputs=T,marker_pairs=len(pairs),violations=len(bad),maximum_order_reversal_ns=maxgap,
                minimum_uniform_integer_endpoint_allowance_ns=(maxgap+1)//2,worst=max(bad,key=lambda x:x['violation_ns']) if bad else None))
            checked+=len(pairs);violations+=len(bad)
        results.append(dict(run=case['run'],case=case['case'],trial=record['trial'],clusters=cluster_results))
bind(code/'output_model.py');bind(Path(__file__))
report=dict(projections=[dict(k_tiles=k,output_tiles=t,ordered_marker_pairs=len(pairs),cross_rank_pairs=sum(a.split('.')[0]!=b.split('.')[0] for a,b in pairs)) for (k,t),pairs in sorted(cache.items())],input_sha256=inputs,processes=len(results),clusters=sum(len(r['clusters']) for r in results),ordered_marker_pairs_checked=checked,
    reversed_marker_pairs=violations,clusters_with_reversals=sum(c['violations']>0 for r in results for c in r['clusters']),
    maximum_order_reversal_ns=max(c['maximum_order_reversal_ns'] for r in results for c in r['clusters']),results=results,observation_qualified=False,
    limits=['Zero-lag partial-order consistency using archived globaltimer markers only; no prices or fitted times.',
        'Numerical correctness inherited from the prior bound 150-process replay; raw hashes rechecked here.',
        'A graph-compatible trace does not prove complete or accurate dependencies, physical clock enclosure, observer perturbation or predictive validity.',
        'Uniform endpoint allowance is a mathematical order diagnostic, not timer precision or an acceptance gate.',
        'No native clock origins are compared across ranks or SMs; unobserved internal events and completion times remain unpriced.'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','results','limits')}))
