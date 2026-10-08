#!/usr/bin/env python3
"""V08 calibration fit and prediction freeze (calibration set only; held-out never read).

Selection rule, fixed before the calibration data exist: components are chosen in the order
slope -> S -> Elast -> X -> cfg_a boundary -> time conversion, each by leave-one-out over the
qualified calibration cases; a form with more parameters is taken only when its LOO RMS of the
relative error is lower by more than 0.2 percentage points (time: lower RMS wins).
Boundary cases (cfg_a odd tile rows) are excluded from all but the boundary and time fits.

  v08_fit.py summarize --run RUN          per-case medians -> RUN/derived/summary.json
  v08_fit.py fit --run RUN                -> RUN/derived/calibration.json (report only)
  v08_fit.py freeze --run RUN --output F  fit + held-out predictions, read-only F
"""
import argparse,copy,datetime,json,math,statistics,time
from pathlib import Path
import v06_fit
import v06_model as base
import v06_run as common
import v08_model as model

med=statistics.median


def rms(v):return math.sqrt(sum(x*x for x in v)/len(v))


def summarize(root):
    rows=json.loads((root/'cases.json').read_text());setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    out={}
    for r in rows:
        if (root/'samples'/r['id']).exists() and r['set']!='heldout':out[r['id']]=model.summarize_case(root,r,setups[r['id']])
    (root/'derived').mkdir(exist_ok=True);common.write_json(root/'derived/summary.json',out);return out


def lsq(xs,ys):
    mx,my=sum(xs)/len(xs),sum(ys)/len(ys);sxx=sum((x-mx)**2 for x in xs)
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sxx if sxx else 0.0;return my-b*mx,b


def fit_params(cs,cfg,choice):
    """Interval medians + chosen forms, from cases cs (non-boundary, interval-qualified)."""
    sched=base.CONFIGS[cfg]['schedule'];p={}
    names=['P0','S','w','E','h','Etail']+(['gm','we','Efull'] if sched=='pingpong' else ['E0'])
    for k in names:
        v=[c['intervals'][k] for c in cs if k in c['intervals']];p[k]=med(v) if v else 0.0
    if sched=='pingpong':p['r']=p['E']/p.pop('Efull') if p.get('Efull') else 1.0
    later=[(c['kt'],c['intervals']['L']) for c in cs if 'L' in c['intervals']]
    if choice['slope']=='joint':
        first=[(c['kt'],c['intervals']['L0']) for c in cs]
        groups=[later,first];means=[(sum(x for x,_ in g)/len(g),sum(y for _,y in g)/len(g)) for g in groups]
        p['l1']=sum((x-mx)*(y-my) for g,(mx,my) in zip(groups,means) for x,y in g)/sum((x-mx)**2 for g,(mx,_) in zip(groups,means) for x,_ in g)
        p['l0']=means[0][1]-p['l1']*means[0][0];p['dL0']=means[1][1]-p['l1']*means[1][0]-p['l0']
    else:
        p['l0'],p['l1']=lsq([x for x,_ in later],[y for _,y in later])
        p['dL0']=med(c['intervals']['L0']-(p['l0']+p['l1']*c['kt']) for c in cs)
    p['S_form']=choice['S']
    if choice['S']=='log_fp':p['s0'],p['s1']=lsq([math.log2(c['fp']) for c in cs],[c['intervals']['S'] for c in cs])
    multi=[c for c in cs if c['T']>=2] if sched=='cooperative' else cs
    p['Elast']=med(c['E_last'] for c in multi) if multi else p['E']
    if sched=='pingpong':p['E']=med(c['E_last'] for c in cs)
    p['E_form']=choice['E']
    if choice['E']=='q':p['e0'],p['e1']=lsq([c['q'] for c in multi],[c['E_last'] for c in multi])
    # Padded tiles whose whole cluster is outside the matrix: measured later-tile mainloop ratio.
    p['rho']={}
    for k in ['padM','padN','padMN']:
        v=[c['tile_L'][k]['median']/c['tile_L']['in']['median']-1 for c in cs if k in c['tile_L'] and 'in' in c['tile_L'] and c['tile_L'][k]['n']>=20]
        if v:p['rho'][k]=med(v)
    p['x0']=p['x1']=p['xk']=0.0;p['boundary_form']='none';p['d0']=p['d1']=0.0
    X=choice['X']
    if X!='none':
        pts=[(c['T'],model.critical_cycles({cfg:p},cfg,feat(c),False)[0],c['c_max_stamped']) for c in cs]
        if X=='const':p['x0']=med(m-b for _,b,m in pts)
        elif X=='prop':p['xk']=med(m/b-1 for _,b,m in pts)
        else:p['x0'],p['x1']=lsq([t for t,_,_ in pts],[m-b for _,b,m in pts])
    return p


_FEAT={}
def feat(c):
    if c['id'] not in _FEAT:
        r=ROWS[c['id']];_FEAT[c['id']]=model.features(r,c['grid'])
    return _FEAT[c['id']]


def loo_c(cs,cfg,choice,boundary=False,bcs=()):
    errs=[]
    targets=bcs if boundary else cs
    for c in targets:
        train=[x for x in cs if x is not c]
        p=fit_params(train,cfg,choice)
        if boundary:fit_boundary(p,cfg,[x for x in bcs if x is not c],choice['boundary'])
        errs.append(model.critical_cycles({cfg:p},cfg,feat(c),boundary)[0]/c['c_max_stamped']-1)
    return rms(errs),errs


def fit_boundary(p,cfg,bcs,form):
    p['boundary_form']=form;p['d0']=p['d1']=0.0
    if form=='none' or not bcs:return
    pts=[(c['kt'],c['c_max_stamped']-model.critical_cycles({cfg:p},cfg,feat(c),False)[0]) for c in bcs]
    if form=='prop':p['d1']=sum(d*k for k,d in pts)/sum(k*k for k,_ in pts)
    elif form=='cols':
        p['d0'],p['d1']=lsq([feat(c)['cols'] for c in bcs],[d/k for k,d in pts])
        p['cols_range']=[min(feat(c)['cols'] for c in bcs),max(feat(c)['cols'] for c in bcs)]
    else:p['d0'],p['d1']=lsq([k for k,_ in pts],[d for _,d in pts])


def choose(options,cs,cfg,choice,key,**kw):
    fits={}
    for name,nparam in options:
        trial=dict(choice,**{key:name});fits[name]=dict(nparam=nparam,loo_rms=loo_c(cs,cfg,trial,**kw)[0])
    best=min(f['loo_rms'] for f in fits.values())
    pick=min([k for k,f in fits.items() if f['loo_rms']<=best+.002],key=lambda k:(fits[k]['nparam'],fits[k]['loo_rms']))
    choice[key]=pick;return fits


def clock_cases(cs,P,cfg,variant):
    out=[]
    for c in cs:
        f=feat(c);cmax,cyc,stages,_=model.critical_cycles(P,cfg,f)
        w=c[f'window_{variant}'];r=ROWS[c['id']]
        d=dict(ghz=c[f'ghz_{variant}'],phi=sum(cyc)/(model.SMS*cmax),mu=stages['mainloop']/cmax,
               dram_tbs=base.dram_bytes(cfg,r['m'],r['n'],r['k'],f['work'])/(w*1e-6)/1e12,lnw=math.log(w))
        d['ghz_v03']=base.clock_ghz(dict(base.V03_RULE),d['phi'],d['lnw'],d['dram_tbs'],d['mu']);out.append(d)
    return out


def fit_time(tcs,P,form):
    """Clock rule jointly over configs (card property); kappa and F per config."""
    v='stamped' if form=='stamped' else 'ends'
    rule,fits=v06_fit.fit_clock([d for cfg in base.CONFIGS for d in clock_cases(tcs[cfg],P,cfg,v)])
    out={}
    for cfg,cs in tcs.items():
        kappa=1.0 if form=='stamped' else med(c['c_max_ends']/c['c_max_stamped'] for c in cs)
        F=med((c['stamped_us'] if form=='stamped' else c['plain_us'])-c[f'window_{v}'] for c in cs)
        out[cfg]=dict(form=form,clock_rule=rule,clock_fits=fits,kappa=kappa,F=F)
    return out


def fit(root):
    global ROWS
    ROWS={r['id']:r for r in json.loads((root/'cases.json').read_text())}
    summary=json.loads((root/'derived/summary.json').read_text())
    cal=[c for c in summary.values() if c['set']=='calib' and c['plain_cv']<=.05]
    params={};selection={};report={};tcs={}
    for cfg in base.CONFIGS:
        allc=[c for c in cal if c['config']==cfg]
        interval_ok=[c for c in allc if abs(c['perturbation'])<=.05 and c['stamped_cv']<=.05]
        cs=[c for c in interval_ok if not c['boundary']];bcs=[c for c in interval_ok if c['boundary']]
        choice=dict(slope='later',S='const',E='const',X='none',boundary='none');fits={}
        fits['slope']=choose([('later',0),('joint',1)],cs,cfg,choice,'slope')
        fits['S']=choose([('const',0),('log_fp',1)],cs,cfg,choice,'S')
        fits['E']=choose([('const',0),('q',1)],cs,cfg,choice,'E')
        fits['X']=choose([('none',0),('const',1),('prop',1),('linT',2)],cs,cfg,choice,'X')
        p=fit_params(cs,cfg,choice)
        if bcs:
            fits['boundary']=choose([('none',0),('prop',1),('affine',2),('cols',2)],cs,cfg,choice,'boundary',boundary=True,bcs=bcs)
            fit_boundary(p,cfg,bcs,choice['boundary'])
        params[cfg]=p;selection[cfg]=dict(choice=choice,fits=fits,cases=[c['id'] for c in cs],boundary_cases=[c['id'] for c in bcs])
        tcs[cfg]=[c for c in allc if c['ends_cv']<=.05 and abs(c['ends_perturbation'])<=.05]
        selection[cfg]['time_cases']=[c['id'] for c in tcs[cfg]]
        report[cfg]=dict(choice=choice,**{k:{n:round(f['loo_rms']*100,2) for n,f in v.items()} for k,v in fits.items()})
    # Time conversion: leave one case out of the joint clock fit and of its config's F/kappa.
    tf={}
    for form in ['stamped','ends']:
        errs={cfg:[] for cfg in base.CONFIGS}
        for cfg in base.CONFIGS:
            for c in tcs[cfg]:
                sub=dict(tcs,**{cfg:[x for x in tcs[cfg] if x is not c]});t=fit_time(sub,params,form)
                errs[cfg].append(model.predict(ROWS[c['id']],c['grid'],dict(params=params,time=t))['predicted_us']/c['plain_us']-1)
        allerr=[e for v in errs.values() for e in v]
        tf[form]=dict(loo_rms=rms(allerr),loo_max=max(map(abs,allerr)),loo_median_abs=med(map(abs,allerr)),
                      loo_positive=sum(e>0 for e in allerr),n=len(allerr),per_config={k:dict(rms=rms(v),max=max(map(abs,v))) for k,v in errs.items()})
    form=min(tf,key=lambda k:tf[k]['loo_rms']);timing=fit_time(tcs,params,form)
    cal_doc=dict(params=params,time=timing,time_fits=tf,selection=selection,
                 excluded=[c['id'] for c in summary.values() if c['set']=='calib' and c not in cal])
    common.write_json(root/'derived/calibration.json',cal_doc)
    for cfg,rep in report.items():print(cfg,json.dumps(rep))
    print('time',form,{k:(round(v['loo_rms']*100,2),round(v['loo_max']*100,2),round(v['loo_median_abs']*100,2),v['loo_positive'],v['n']) for k,v in tf.items()})
    for cfg,t in timing.items():print(cfg,'F %.3f kappa %.4f'%(t['F'],t['kappa']))
    r=timing['cfg_a']['clock_rule'];print('clock',r['refit'],'da %.4f dc %.4f'%(r['da'],r['dc']))
    return cal_doc


def freeze(root,out):
    if out.exists():raise ValueError('never overwrite a frozen prediction')
    cal=fit(root);env=json.loads((root/'environment.json').read_text())
    setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    held=[r for r in ROWS.values() if r['set']=='heldout']
    if any((root/'samples'/r['id']).exists() for r in held):raise ValueError('held-out timing already exists')
    preds={r['id']:model.predict(r,setups[r['id']]['grid'],cal) for r in held}
    now=time.time_ns()
    doc=dict(status='frozen',frozen_unix_ns=now,frozen_utc=datetime.datetime.fromtimestamp(now/1e9,datetime.timezone.utc).isoformat(),
             gpu=env['gpu'],environment=env,model=model.__doc__,fit_rule=__doc__,
             code_sha256={n:common.sha(Path(__file__).parent/n) for n in ['v08_model.py','v08_fit.py','v06_model.py','v06_fit.py','run_v08.py']},
             summary_sha256=common.sha(root/'derived/summary.json'),cases_sha256=common.sha(root/'cases.json'),
             static_setup_sha256=common.sha(root/'static_setup.json'),binary_hashes_sha256=common.sha(root/'build/binary_hashes.json'),
             novelty=json.loads((root/'novelty.json').read_text()),calibration=cal,predictions=preds)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as s:json.dump(doc,s,indent=1);s.write('\n')
    out.chmod(0o444);print('frozen',doc['frozen_utc'],common.sha(out))
    for k,v in preds.items():print(f"{k:16s} T{v['T']:3d} C {v['critical_cycles']:10.0f} f {v['clock_ghz']:.3f} us {v['predicted_us']:9.2f} delta {v['boundary_delta']:.0f}")


ROWS={}
def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step',choices=['summarize','fit','freeze']);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.run.resolve()
    if a.step=='summarize':summarize(root)
    elif a.step=='fit':fit(root)
    else:freeze(root,a.output.resolve())

if __name__=='__main__':main()
