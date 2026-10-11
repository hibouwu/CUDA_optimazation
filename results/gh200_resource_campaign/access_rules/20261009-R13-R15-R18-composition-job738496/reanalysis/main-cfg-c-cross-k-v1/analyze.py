#!/usr/bin/env python3
"""Fixed cfg_c cross-K proportional-service diagnostic; no fitted price."""
from pathlib import Path
from collections import defaultdict
import gzip,hashlib,json,statistics,sys
out=Path(__file__).resolve().parent;run=out.parents[1]
repo=next(p for p in out.parents if (p/'microbench').is_dir())
code=repo/'microbench/gh200_resource_campaign/access_rules';sys.path.insert(0,str(code))
from analyze_r18 import replay
inputs={}
def bind(p):
    inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):bind(p);return json.loads(p.read_text())
rows={r['id']:r for r in read(run/'cases.json')}
windows={};processes=[]
for pitch in ('aligned','a16','b16'):
    for k in (1024,4096):
        case=f'cfg_c_r13_comp_k{k}_{pitch}';row=rows[case]
        paths=sorted((run/'samples'/case).glob('dual-*.json'));assert len(paths)==10
        for path in paths:
            record=read(path);raw=run/record['raw'];bind(raw)
            assert not record['returncode'] and inputs[str(raw)]==record['raw_sha256']
            observed=replay(run,record,row,retain_roles=True)
            with gzip.open(raw,'rt') as stream:events={v['event']:v for line in stream for v in [json.loads(line)]}
            setup=events['setup'];assert setup['stages']==4 and setup['cluster']==[1,2]
            processes.append(dict(case=case,trial=record['trial'],gpu_uuid=setup['gpu_uuid'],grid=setup['grid'],checked_values=4096))
            for cta in observed['ctas']:
                for role,data in enumerate(cta['roles'],1):
                    for j,(cycles,ns,coord) in enumerate(zip(data['tiles'],data['tiles_ns'],cta['role_work'][role-1])):
                        key=(pitch,k,record['trial'],cta['cta'],role,j)
                        assert key not in windows
                        windows[key]=dict(cycles=cycles[1]-cycles[0],ns=ns[1]-ns[0],sm=cta['sm'],coord=list(coord),tiles=len(data['tiles']))
assert len({p['gpu_uuid'] for p in processes})==1 and len({tuple(p['grid']) for p in processes})==1
paired=[];groups=defaultdict(list)
for key,a in windows.items():
    pitch,k,trial,cta,role,j=key
    if k!=1024:continue
    b=windows[pitch,4096,trial,cta,role,j]
    assert a['coord']==b['coord'] and a['tiles']==b['tiles'] and min(a['cycles'],a['ns'],b['cycles'],b['ns'])>0
    row=dict(pitch=pitch,trial=trial,cta=cta,role=role,output_ordinal=j,outputs=a['tiles'],coordinate=a['coord'],
        sm_changed=a['sm']!=b['sm'],cycles_1024=a['cycles'],cycles_4096=b['cycles'],ns_1024=a['ns'],ns_4096=b['ns'],
        cycles_proportional_error=b['cycles']/(4*a['cycles'])-1,ns_proportional_error=b['ns']/(4*a['ns'])-1)
    paired.append(row);groups[pitch,role,j,a['tiles'],trial].append(row)
centers=[]
for (pitch,role,j,count,trial),rs in sorted(groups.items()):
    centers.append(dict(pitch=pitch,role=role,output_ordinal=j,outputs=count,trial=trial,windows=len(rs),
        cycles_error=statistics.median(r['cycles_proportional_error'] for r in rs),
        ns_error=statistics.median(r['ns_proportional_error'] for r in rs)))
summary=[]
for pitch,role,j,count in sorted({(c['pitch'],c['role'],c['output_ordinal'],c['outputs']) for c in centers}):
    cs=[c for c in centers if (c['pitch'],c['role'],c['output_ordinal'],c['outputs'])==(pitch,role,j,count)]
    assert len(cs)==10
    summary.append(dict(pitch=pitch,role=role,output_ordinal=j,outputs=count,processes=10,
        cycles_error_median=statistics.median(c['cycles_error'] for c in cs),cycles_error_min=min(c['cycles_error'] for c in cs),cycles_error_max=max(c['cycles_error'] for c in cs),
        ns_error_median=statistics.median(c['ns_error'] for c in cs)))
for name in ('analyze_r18.py','v08_model.py','v06_model.py','v06_run.py'):bind(code/name)
bind(Path(__file__))
report=dict(input_sha256=inputs,processes=processes,paired_windows=len(paired),sm_changes=sum(r['sm_changed'] for r in paired),
    summaries=summary,process_centers=centers,observation_qualified=False,
    limits=['Development diagnostic, not new heldout prediction or acceptance score.',
        'Within-role differences only; raw clock origins are never compared across calls or SMs.',
        'Same trial numbers align descriptive repeats, not causal randomized pairs.',
        'K and the corresponding A pitch/allocation both change; this does not isolate an instruction service law.',
        'No local observer perturbation qualification is established by numerical replay.',
        'Proportional null is C4096=4*C1024; disagreement may contain startup, drain, supply, overlap and observation effects. No price is fitted.'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'windows.json.gz').write_bytes(gzip.compress((json.dumps(paired)+'\n').encode(),mtime=0))
print(json.dumps(dict(processes=len(processes),paired_windows=len(paired),summaries=summary)))
