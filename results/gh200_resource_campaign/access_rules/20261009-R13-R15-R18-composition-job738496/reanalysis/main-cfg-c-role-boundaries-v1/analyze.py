#!/usr/bin/env python3
"""Same-CTA role endpoint decomposition, bound to verified cross-K records."""
from pathlib import Path
from collections import defaultdict
import gzip,hashlib,json,statistics
out=Path(__file__).resolve().parent;run=out.parents[1];prior=run/'reanalysis/main-cfg-c-cross-k-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((prior/'report.json').read_text())
for path,h in base['input_sha256'].items():assert sha(Path(path))==h,path
with gzip.open(prior/'windows.json.gz','rt') as f:old=json.load(f)
expected={(r['pitch'],r['trial'],r['cta'],r['role'],r['output_ordinal']):r for r in old}
rows=[];groups=defaultdict(list)
for process in base['processes']:
    case=process['case'];k=int(case.split('_k')[1].split('_')[0]);pitch=case.rsplit('_',1)[1]
    paths=list((run/'samples'/case).glob('dual-*.json'))
    record=next(json.loads(p.read_text()) for p in paths if json.loads(p.read_text())['trial']==process['trial'])
    raw=run/record['raw'];assert sha(raw)==record['raw_sha256']
    with gzip.open(raw,'rt') as f:packets=[json.loads(line) for line in f]
    events={p['event']:p for p in packets};assert len(events)==len(packets)
    setup,call=events['setup'],events['call'];trace=call['trace']
    assert setup['trace_tile_words']==10 and len(trace)==132*1296
    for cta in range(132):
        b=trace[cta*1296:(cta+1)*1296];count=b[6]
        assert count==b[9] and b[2]==b[11]==b[12] and b[2]>0
        for j in range(count):
            a=b[16+j*10:26+j*10];z=b[656+j*10:666+j*10]
            assert a[4:6]==z[4:6]
            for role,t in ((1,a),(2,z)):
                e=expected[pitch,process['trial'],cta,role,j]
                assert t[1]-t[0]==e[f'cycles_{k}'] and t[7]-t[6]==e[f'ns_{k}']
            row=dict(pitch=pitch,k=k,trial=process['trial'],cta=cta,sm=b[2]-1,output_ordinal=j,outputs=count,
                FIRST_skew_cycles=z[0]-a[0],MAIN_skew_cycles=z[1]-a[1],L_difference_cycles=(z[1]-z[0])-(a[1]-a[0]),
                FIRST_skew_ns=z[6]-a[6],MAIN_skew_ns=z[7]-a[7],L_difference_ns=(z[7]-z[6])-(a[7]-a[6]))
            assert row['L_difference_cycles']==row['MAIN_skew_cycles']-row['FIRST_skew_cycles']
            assert row['L_difference_ns']==row['MAIN_skew_ns']-row['FIRST_skew_ns']
            rows.append(row);groups[pitch,k,j,count,process['trial']].append(row)
fields=['FIRST_skew_cycles','MAIN_skew_cycles','L_difference_cycles','FIRST_skew_ns','MAIN_skew_ns','L_difference_ns']
centers=[dict(pitch=p,k=k,output_ordinal=j,outputs=n,trial=t,windows=len(rs),**{f:statistics.median(r[f] for r in rs) for f in fields}) for (p,k,j,n,t),rs in sorted(groups.items())]
summary=[]
for p,k,j,n in sorted({key[:4] for key in groups}):
    cs=[c for c in centers if (c['pitch'],c['k'],c['output_ordinal'],c['outputs'])==(p,k,j,n)]
    assert len(cs)==10
    summary.append(dict(pitch=p,k=k,output_ordinal=j,outputs=n,**{f:statistics.median(c[f] for c in cs) for f in fields}))
source_files=[run/'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp',run/'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp']
report=dict(source_sha256={str(p):sha(p) for p in source_files},prior_report_sha256=sha(prior/'report.json'),prior_windows_sha256=sha(prior/'windows.json.gz'),
    processes=len(base['processes']),same_CTA_role_pairs=len(rows),summaries=summary,process_centers=centers,observation_qualified=False,
    limits=['Within-SM WG2 minus WG1 at matching output; does not compare raw clock origins across SMs.',
        'DeltaL=DeltaMAIN-DeltaFIRST holds per window; separately aggregated medians need not sum.',
        'Matched original raw and prior numeric replay; no new device observation or causal attribution.',
        'These endpoints do not measure isolated WGMMA service or justify a fitted role-specific price.'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'windows.json.gz').write_bytes(gzip.compress((json.dumps(rows)+'\n').encode(),mtime=0))
print(json.dumps(dict(processes=report['processes'],pairs=len(rows),aligned=[s for s in summary if s['pitch']=='aligned'])))
