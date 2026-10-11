#!/usr/bin/env python3
"""Same observed markers admit distinct exits in the unpriced partial order."""
from pathlib import Path
import gzip,hashlib,json,sys
out=Path(__file__).resolve().parent;run=out.parents[1];repo=next(p for p in out.parents if (p/'microbench').is_dir())
sys.path.insert(0,str(repo/'microbench/gh200_resource_campaign/access_rules'))
import output_model as model
raw=run/'samples/cfg_c_r13_comp_k4096_aligned/dual-00.txt.gz'
record=json.loads((raw.parent/'dual-00.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(raw)==record['raw_sha256']
with gzip.open(raw,'rt') as f:events={v['event']:v for v in map(json.loads,f)}
trace=events['call']['trace'];assert events['setup']['grid']==[66,2,1]
graph_path=run/'reanalysis/main-cfg-c-rank-dependencies-v2/cfg_c-S4-K64-T2-rank-dependencies.json.gz'
with gzip.open(graph_path,'rt') as f:graph=json.load(f)
anchors={};entries=[]
for rank,cta in enumerate((0,66)):
    b=trace[cta*1296:(cta+1)*1296];entries.append(b[1]);assert b[6]==b[9]==2
    for wg in (0,1):
        for j in (0,1):
            t=b[16+wg*640+j*10:26+wg*640+j*10]
            for event,n in zip(('FIRST','MAIN','PERMIT','DONE'),t[6:10]):anchors[f'r{rank}.w{wg}.t{j}.{event}']=(n,n)
        anchors[f'r{rank}.w{wg}.FINAL']=(b[5+3*wg],b[5+3*wg])
assert set(anchors)==set(graph['markers'])
origin=min(entries);anchors={n:(lo-origin,hi-origin) for n,(lo,hi) in anchors.items()};anchors['cluster_initialized']=(0,0)
bounds=model.dependency_bounds(graph,anchors)
watched=['r0.producer_tail','r1.producer_tail','r0.D_visible','r1.D_visible','source_safe_exit','cuda_complete']
assert all(bounds[n][1] is None for n in watched)
def witness(extra):
    merged=dict(anchors,**extra);b=model.dependency_bounds(graph,merged);values={n:v[0] for n,v in b.items()}
    assert all(v is not None for v in values.values())
    assert all(values[n]>=values[p] for n,ps in graph['nodes'].items() for p in ps)
    assert all(values[n]==a[0] for n,a in merged.items())
    return values
base=witness({});late=base['cuda_complete']+1000
source=witness({'r0.producer_tail':(late,late)})
visible=witness({'r0.d15.D_visible':(late,late)})
assert source['source_safe_exit']==source['cuda_complete']==late
assert visible['source_safe_exit']==base['source_safe_exit'] and visible['cuda_complete']==late
report=dict(input_sha256={str(p):sha(p) for p in (raw,graph_path,Path(model.__file__),Path(__file__))},ctas=[0,66],
    exact_observed_markers=36,observed_anchors=anchors,entry_boundary_assumption='cluster_initialized fixed at minimum recorded CTA entry only for a conservative mathematical witness',
    bounds={n:bounds[n] for n in watched},
    witnesses={name:{n:v[n] for n in watched} for name,v in [('earliest',base),('producer_tail_delayed',source),('D_visibility_delayed',visible)]},
    synthetic_delay_ns=1000,observation_qualified=False,
    limits=['Schedules satisfy only the unpriced DAG, not a calibrated hardware service model.',
        'All36 observed markers remain exactly fixed; 1000ns is arbitrary for demonstrating an unbounded endpoint, not an inferred latency.',
        'Aggregate CUDA event timing cannot identify each cluster exit or per-rank visibility.',
        'A producer-tail timestamp could constrain source_safe_exit but would not measure global D visibility or full kernel completion.'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(markers=36,unbounded_upper=watched,witnesses=report['witnesses'])))
