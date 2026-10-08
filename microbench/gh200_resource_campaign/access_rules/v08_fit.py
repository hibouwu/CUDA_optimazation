#!/usr/bin/env python3
"""V08 candidate: V07 model with two calibration-only changes, each chosen by leave-one-out.

1. Mainloop slope l1: V07 fits L = l0 + l1*Kt on later tiles only; cfg_c then has Kt 16/32
   only. Candidate "joint" also uses first-tile windows (L0 = l0 + dL0 + l1*Kt, own intercept).
2. Time conversion: V07 F = stamped event time - stamped window, but scores plain processes.
   Candidates: stamped_F (V07), plain_F (F = plain - stamped window), plain_lin
   (plain = a + b*window). Pick the fewest parameters within 0.2 pp of the best LOO RMS.

Supply, epilogue and the boundary hypotheses stay as in V07: calibration does not separate a
footprint or concurrency rule (see V07 post-hoc diagnosis).

  v08_fit.py --v07 RUN [--evaluate-v07 OUT.json]
--evaluate-v07 scores the candidate on the V07 held-out cases. That is post-hoc only; the
candidate must be frozen and tested on new held-out shapes.
"""
import argparse,copy,json,math,statistics
from pathlib import Path
import v06_fit
import v06_model as base
import v07_model

med=statistics.median


def slope_fit(cs,form):
    later=[(c['ktiles'],c['intervals']['L']) for c in cs if 'L' in c['intervals']]
    first=[(c['ktiles'],c['intervals']['L0']) for c in cs] if form=='joint' else []
    # y = a_later*[later] + a_first*[first] + l1*kt, least squares
    groups=[later,first] if first else [later]
    means=[(sum(x for x,_ in g)/len(g),sum(y for _,y in g)/len(g)) for g in groups]
    sxy=sum((x-mx)*(y-my) for g,(mx,my) in zip(groups,means) for x,y in g)
    sxx=sum((x-mx)**2 for g,(mx,my) in zip(groups,means) for x,_ in g)
    l1=sxy/sxx;l0=means[0][1]-l1*means[0][0]
    dL0=(means[1][1]-l1*means[1][0]-l0) if first else med(c['intervals']['L0']-(l0+l1*c['ktiles']) for c in cs)
    return l0,l1,dL0


def cycle_params(p,cs,rows,cfg,form):
    p=copy.deepcopy(p);p['l0'],p['l1'],p['dL0']=slope_fit(cs,form)
    sched=base.CONFIGS[cfg]['schedule'];p['x0']=p['x1']=p['xk']=0.0;pts=[]
    for c in cs:
        work=base.scheduled_work(cfg,rows[c['id']]['m'],rows[c['id']]['n'],c['grid'])
        pts.append((c['tiles_max'],max(base.cta_cycles(p,sched,len(w),c['ktiles']) for w in work),c['c_max']))
    p['excess_form'],p['excess_fits']=v06_fit.choose_excess(pts);p.update(p['excess_fits'][p['excess_form']]['coef'])
    return p


def c_pred(p,c,row,cfg):
    work=base.scheduled_work(cfg,row['m'],row['n'],c['grid'])
    return max(base.cta_cycles(p,base.CONFIGS[cfg]['schedule'],len(w),c['ktiles']) for w in work)


def choose(fits):
    best=min(f['loo_rms'] for f in fits.values())
    return min([k for k,f in fits.items() if f['loo_rms']<=best+.002],key=lambda k:(fits[k]['nparam'],fits[k]['loo_rms']))


def fit(cal,rows):
    cases={k:dict(v,id=k) for k,v in cal['cases'].items()};params={};slope={}
    for cfg in base.CONFIGS:
        cs=[c for c in cases.values() if c['config']==cfg];fits={}
        for form,nparam in (('later',0),('joint',1)):
            loo=[]
            for c in cs:
                p=cycle_params(cal['params'][cfg],[x for x in cs if x is not c],rows,cfg,form)
                loo.append(c_pred(p,c,rows[c['id']],cfg)/c['c_max']-1)
            fits[form]=dict(nparam=nparam,loo_rms=math.sqrt(sum(e*e for e in loo)/len(loo)),loo_max=max(map(abs,loo)))
        slope[cfg]=dict(fits=fits,chosen=choose(fits))
        params[cfg]=cycle_params(cal['params'][cfg],cs,rows,cfg,slope[cfg]['chosen'])
    # Clock: same card correction procedure as V06/V07, on the new cycle predictions.
    for c in cases.values():
        row=rows[c['id']];cfg=c['config'];work=base.scheduled_work(cfg,row['m'],row['n'],c['grid'])
        sched=base.CONFIGS[cfg]['schedule']
        cyc=[base.cta_cycles(params[cfg],sched,len(w),c['ktiles']) for w in work]
        _,stages=base.cta_cycles(params[cfg],sched,max(map(len,work)),c['ktiles'],detail=True)
        c['c_pred']=max(cyc);c['phi']=sum(cyc)/(base.SMS*c['c_pred']);c['mu']=stages['mainloop']/c['c_pred']
        c['dram_tbs']=base.dram_bytes(cfg,row['m'],row['n'],row['k'],work)/(c['window_us']*1e-6)/1e12
        c['lnw']=math.log(c['window_us']);c['ghz_v03']=base.clock_ghz(dict(base.V03_RULE),c['phi'],c['lnw'],c['dram_tbs'],c['mu'])
    rule,rule_fits=v06_fit.fit_clock(list(cases.values()))
    for c in cases.values():
        c['w_pred']=base.predict_case(rows[c['id']],c['grid'],params[c['config']],rule,0.0)['window_us']
    # Time conversion against plain processes; LOO uses the predicted window of the left-out case.
    def solve(form,cs):
        if form=='stamped_F':return 0.0,med(c['stamped_us']-c['window_us'] for c in cs),1.0
        if form=='plain_F':return 0.0,med(c['plain_us']-c['window_us'] for c in cs),1.0
        xs=[c['window_us'] for c in cs];ys=[c['plain_us'] for c in cs];mx,my=sum(xs)/len(xs),sum(ys)/len(ys)
        b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs);return 0.0,my-b*mx,b
    time={}
    for cfg in base.CONFIGS:
        cs=[c for c in cases.values() if c['config']==cfg];fits={}
        for form,nparam in (('stamped_F',1),('plain_F',1),('plain_lin',2)):
            loo=[]
            for c in cs:
                _,a,b=solve(form,[x for x in cs if x is not c]);loo.append((a+b*c['w_pred'])/c['plain_us']-1)
            _,a,b=solve(form,cs)
            fits[form]=dict(a=a,b=b,nparam=nparam,loo_rms=math.sqrt(sum(e*e for e in loo)/len(loo)),loo_max=max(map(abs,loo)),
                            loo_median_abs=med(map(abs,loo)),loo_positive=sum(e>0 for e in loo))
        time[cfg]=dict(fits=fits,chosen=choose(fits))
    return dict(params=params,slope_selection=slope,clock_rule=rule,clock_fits=rule_fits,time_conversion=time,
                calibration_source='V07 reanalysis/v07-calibration/calibration.json (same processes as V07 freeze)')


def predict(row,grid,cand,boundary):
    cfg=row['config'];t=cand['time_conversion'][cfg];f=t['fits'][t['chosen']]
    r=v07_model.predict(row,grid,cand['params'][cfg],cand['clock_rule'],0.0,boundary)
    r['predicted_us']=f['a']+f['b']*r['window_us'];r['time_form']=t['chosen'];return r


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v07',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--evaluate-v07',action='store_true');a=ap.parse_args();run=a.v07.resolve()
    cal=json.loads((run/'reanalysis/v07-calibration/calibration.json').read_text())
    rows={r['id']:r for r in json.loads((run/'cases.json').read_text())}
    if a.output.exists():raise ValueError('output exists')
    cand=fit(cal,rows)
    for cfg in base.CONFIGS:
        s=cand['slope_selection'][cfg];t=cand['time_conversion'][cfg];p=cand['params'][cfg]
        print(cfg,'slope',s['chosen'],{k:round(v['loo_rms']*100,2) for k,v in s['fits'].items()},'l0 %.0f l1 %.1f dL0 %.0f excess %s'%(p['l0'],p['l1'],p['dL0'],p['excess_form']))
        print('   time',t['chosen'],{k:(round(v['loo_rms']*100,2),round(v['loo_max']*100,2),round(v['a'],2),round(v['b'],4)) for k,v in t['fits'].items()})
    print('clock',cand['clock_rule']['refit'],'da %.4f dc %.4f'%(cand['clock_rule']['da'],cand['clock_rule']['dc']))
    if a.evaluate_v07:
        frozen=json.loads((run/'v07-predictions.json').read_text());val=json.loads((run/'reanalysis/validation-v2/validation.json').read_text())
        meas={c['case']:c['measured_us'] for c in val['cases']};setups={s['case']:s['setup'] for s in json.loads((run/'static_setup.json').read_text())}
        out={}
        for cid in frozen['predictions']:
            r=predict(rows[cid],setups[cid]['grid'],cand,frozen['boundary_hypotheses'])
            out[cid]=dict(predicted_us=r['predicted_us'],measured_plain_us=meas[cid],error=r['predicted_us']/meas[cid]-1,
                          v07_error=frozen['predictions'][cid]['predicted_us']/meas[cid]-1,critical_cycles=r['critical_cycles'],clock_ghz=r['clock_ghz'])
        e=[v['error'] for v in out.values()]
        cand['posthoc_v07']=dict(note='post-hoc on V07 held-out; not a validation',cases=out,
            abs_median=med(map(abs,e)),abs_max=max(map(abs,e)),positive=sum(x>0 for x in e))
        for cid,v in out.items():print(f"  {cid:26s} V07 {v['v07_error']*100:+6.2f}  cand {v['error']*100:+6.2f}")
        print('post-hoc V07: |e| median %.2f max %.2f positive %d/24'%(cand['posthoc_v07']['abs_median']*100,cand['posthoc_v07']['abs_max']*100,cand['posthoc_v07']['positive']))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(cand,indent=1))

if __name__=='__main__':main()
