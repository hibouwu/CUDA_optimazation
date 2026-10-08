#!/usr/bin/env python3
"""V07 post-hoc diagnosis (CPU only). Does not change frozen predictions or the V07 verdict.

1. Time split: prediction vs plain and vs stamped; fixed term inside stamped processes;
   F re-calibrated from calibration plain processes.
2. Per-CTA: typical CTA error vs max-CTA excess; where the slowest CTA is.
3. Supply S and last epilogue against input footprint / calibration.
4. cfg_a boundary shapes: round-0 mainloop slowdown by tile and SM, also for R18.
"""
import argparse,json,statistics as st,sys
from pathlib import Path

def load(run,cid,row,replay):
    plain,stamped=[],[]
    for path in sorted((run/'samples'/cid).glob('*.json')):
        rec=json.loads(path.read_text())
        if rec['returncode']:continue
        o=replay(run,rec,row);(plain if o['variant']=='plain' else stamped).append(o)
    return plain,stamped

def round0(stamped,i):
    return st.median(o['ctas'][i]['tiles'][0][1]-o['ctas'][i]['tiles'][0][0] for o in stamped)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--r18',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    run=a.run.resolve();sys.path.insert(0,str(run/'source'))
    from analyze_r18 import replay,tile_class
    rows={r['id']:r for r in json.loads((run/'cases.json').read_text())}
    frozen=json.loads((run/'v07-predictions.json').read_text());cal=frozen['calibration']
    f_plain={c:st.median(v['plain_us']-v['window_us'] for v in cal['cases'].values() if v['config']==c and v['perturbation']<=.05)
             for c in ['cfg_a','cfg_b','cfg_c']}
    cases=[];boundary={}
    for cid,p in frozen['predictions'].items():
        row=rows[cid];cfg=row['config'];plain,stamped=load(run,cid,row,replay)
        tp=st.median(o['elapsed_us'] for o in plain);ts=st.median(o['elapsed_us'] for o in stamped)
        n=len(p['work']);spans={};sup=[];epi=[];fixed=[]
        for o in stamped:
            act=[c for c in o['ctas'] if c['tiles']]
            fixed.append(o['elapsed_us']-(max(c['end_ns'] for c in act)-min(c['entry_ns'] for c in o['ctas']))/1e3)
            for c in act:
                spans.setdefault(c['cta'],[]).append(c['end_c']-c['entry_c'])
                sup.append(c['tiles'][0][0]-c['entry_c']);epi.append(c['tiles'][-1][3]-c['tiles'][-1][2])
        span={i:st.median(v) for i,v in spans.items()};T=max(len(w) for w in p['work'])
        full=[i for i in span if len(p['work'][i])==T];slow=max(span,key=span.get)
        typ_meas=st.median(span[i] for i in full);typ_pred=st.median(p['per_cta_cycles'][i] for i in full)
        pred_us=p['predicted_us'];pred_fplain=pred_us-p['fixed_us']+f_plain[cfg]
        cases.append(dict(case=cid,config=cfg,plain_us=tp,stamped_us=ts,predicted_us=pred_us,
            error_vs_plain=pred_us/tp-1,error_vs_stamped=pred_us/ts-1,stamp_overhead=ts/tp-1,
            fixed_in_stamped_us=st.median(fixed),frozen_fixed_us=p['fixed_us'],
            posthoc_fplain_error=pred_fplain/tp-1,
            typical_cta_error=typ_pred/typ_meas-1,max_cta_excess=span[slow]/typ_meas-1,
            frozen_cta=p['critical_cta'],frozen_cta_span=span[p['critical_cta']],slowest_cta=slow,slowest_span=span[slow],
            slowest_tiles=p['work'][slow],slowest_tile_classes=[tile_class(row,*w) for w in p['work'][slow]],
            supply_pred=p['stage_cycles']['supply'],supply_all_cta=st.median(sup),
            last_epi_pred=p['stage_cycles']['last_epilogue'],last_epi_all_cta=st.median(epi),
            footprint_mib=(row['m']*row['k']*2+row['k']*row['n']*2+row['m']*row['n']*4)/2**20))
        if cfg=='cfg_a' and row['kind']!='aligned':
            ml={i:round0(stamped,i) for i in range(n)};base=st.median(ml.values())
            boundary[cid]=[dict(cta=i,sm=stamped[0]['ctas'][i]['sm'],tile=p['work'][i][0],cls=tile_class(row,*p['work'][i][0]),
                                slowdown=ml[i]/base-1) for i in sorted(ml,key=lambda i:-ml[i])]
    r18=a.r18.resolve();r18rows={r['id']:r for r in json.loads((r18/'cases.json').read_text())}
    for cid in ['cfg_a_oob_k8192','cfg_a_partial_odd_k8192']:
        _,stamped=load(r18,cid,r18rows[cid],replay);n=len(stamped[0]['ctas'])
        ml={i:round0(stamped,i) for i in range(n)};base=st.median(ml.values())
        boundary['R18:'+cid]=[dict(cta=i,sm=stamped[0]['ctas'][i]['sm'],tile=stamped[0]['ctas'][i]['work'][0],
            cls=tile_class(r18rows[cid],*stamped[0]['ctas'][i]['work'][0]),slowdown=ml[i]/base-1) for i in sorted(ml,key=lambda i:-ml[i])]
    def summ(k):
        v=[c[k] for c in cases];return dict(abs_median=st.median(map(abs,v)),abs_max=max(map(abs,v)),positive=sum(x>0 for x in v))
    out=dict(note='post-hoc diagnosis; frozen V07 verdict unchanged; F_plain is calibration-only but chosen after seeing V07',
        f_plain_from_calibration=f_plain,f_stamped_frozen=cal['fixed_us'],
        summary={k:summ(k) for k in ['error_vs_plain','error_vs_stamped','posthoc_fplain_error','typical_cta_error','max_cta_excess']},
        cases=cases,cfg_a_round0_slowdown=boundary)
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'posthoc.json').write_text(json.dumps(out,indent=1))
    for k,v in out['summary'].items():print(k,{x:round(y*100,2) if isinstance(y,float) else y for x,y in v.items()})

if __name__=='__main__':main()
