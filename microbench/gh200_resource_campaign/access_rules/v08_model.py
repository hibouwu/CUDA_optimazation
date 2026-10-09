#!/usr/bin/env python3
"""V08 model: V06/V07 per-CTA event recursion with case features, verified swizzle scheduler.

Per CTA (cycles): supply P0+S, mainloop L(Kt)=l0+l1*Kt (+dL0 on tile 0), cooperative
ed = me + w + E0 (tile 0) / E (middle) / Elast (last tile, T>=2), fm[j+1] = ed[j] + h;
pingpong as V06 with E the last (alone) epilogue. end = max(ed) + Etail; X excess; C = max CTA.
Case features (chosen in v08_fit by calibration leave-one-out):
  S     : const | s0 + s1*log2(footprint MiB)
  Elast : const | e0 + e1*q,  q = CTAs holding the maximum tile count / 132
  cfg_a boundary (odd tile-row count, cluster 2x1): C += d0 + d1*Kt | Kt*(d0 + d1*columns), columns clamped to the calibrated range
  padded tiles whose whole cluster lies outside the matrix: L *= 1 + rho[config][padM|padN|padMN]
Time: T = F + kappa*C/f; kappa = ends/stamped critical-cycle ratio (or 1), f the V03-form clock.
"""
import gzip,json,math,statistics
from pathlib import Path
import v06_model as base
from analyze_r18 import replay

med=statistics.median
SMS=132


def cdiv(a,b):return -(-a//b)


def log_swizzle(bm,bn,maxsw):
    d=min(bm,bn)
    return 3 if maxsw>=8 and d>=6 else 2 if maxsw>=4 and d>=3 else 1 if maxsw>=2 and d>=2 else 0


def scheduled_work(config,m,n,grid,swizzle=1):
    """CUTLASS 3.9.2 PersistentTileSchedulerSm90 with heuristic raster and max_swizzle_size."""
    tm,tn,_=base.CONFIGS[config]['tile'];cm,cn=base.CONFIGS[config]['cluster'];gx,gy,gz=grid
    if gz!=1:raise ValueError('one batch only')
    bm=cdiv(cdiv(m,tm),cm)*cm;bn=cdiv(cdiv(n,tn),cn)*cn
    lg=log_swizzle(bm,bn,swizzle);S=1<<lg
    nm=cdiv(bm,S*cm)*S*cm;nn=cdiv(bn,S*cn)*S*cn
    along_n=not nn>nm
    minor,major,major_blocks=(cm,cn,nn//cn) if along_n else (cn,cm,nm//cm)
    out=[]
    for by in range(gy):
        for bx in range(gx):
            first=bx+by*gx if along_n else bx*gy+by;off=bx%cm if along_n else by%cn;work=[]
            for linear in range(first,nm*nn,gx*gy):
                cluster,major_off=divmod(linear//minor,major)
                extra=cluster>>lg;mdiv,cmaj=divmod(extra,major_blocks)
                mi=(mdiv*S+(cluster&(S-1)))*minor+off;ma=cmaj*major+major_off
                work.append((mi,ma) if along_n else (ma,mi))
            out.append(work)
    tiles=[t for w in out for t in w]
    if len(tiles)!=nm*nn or len(set(tiles))!=nm*nn:raise ValueError('scheduler mapping does not cover padded tiles')
    return out


def tile_class(config,r,mi,ni):
    """'pad*' = tile and its whole cluster outside the matrix (swizzle/cluster padding)."""
    tm,tn,_=base.CONFIGS[config]['tile'];cm,cn=base.CONFIGS[config]['cluster']
    m0,n0=(mi//cm)*cm,(ni//cn)*cn
    om=all((x*tm>=r['m']) for x in range(m0,m0+cm));on=all((y*tn>=r['n']) for y in range(n0,n0+cn))
    return 'pad'+('M' if om else '')+('N' if on else '') if (om or on) else 'in'


def footprint_mib(r):return (r['m']*r['k']*2+r['k']*r['n']*2+r['m']*r['n']*4)/2**20


def features(r,grid):
    work=scheduled_work(r['config'],r['m'],r['n'],grid,r['swizzle']);counts=[len(w) for w in work];T=max(counts)
    classes=[tuple(tile_class(r['config'],r,mi,ni) for mi,ni in w) for w in work]
    return dict(work=work,classes=classes,T=T,kt=cdiv(r['k'],64),q=sum(c==T for c in counts)/SMS,fp=footprint_mib(r),
                boundary=r['config']=='cfg_a' and r['kind'] in ('oob','partial_odd'),cols=cdiv(r['n'],128),rows=cdiv(r['m'],128))


# ------------------------------------------------------------------ observation
def _ends(root,record,r):
    raw=root/record['raw']
    with gzip.open(raw,'rt') as stream:events={ev['event']:ev for line in stream if line.strip() for ev in [json.loads(line)]}
    setup,call=events['setup'],events['call']
    if setup['trace_version']!='v08-ends':raise ValueError('not an ends trace')
    words=call['trace'];ctas=math.prod(setup['grid']);width=16+2*64*6
    if len(words)!=ctas*width:raise ValueError('wrong ends trace size')
    cooperative=r['config']!='cfg_b';out=[]
    for c in range(ctas):
        h=words[c*width:c*width+16]
        if h[10] or not h[0] or not h[2] or not h[3]:raise ValueError('missing ends stamp')
        r1,r2=h[4:7],h[7:10]
        if cooperative:
            if r1[2]!=r2[2]:raise ValueError('consumer tile counts differ')
            end_c,end_ns,tiles=r2[0],r2[1],r2[2]
        else:
            last=max((r1,r2),key=lambda x:x[0] if x[2] else 0);end_c,end_ns,tiles=last[0],last[1],r1[2]+r2[2]
        if h[11]!=h[2] or (r2[2] and h[12]!=h[2]):raise ValueError('CTA changed SM')
        out.append(dict(cta=c,entry_c=h[0],entry_ns=h[1],sm=h[2]-1,prod_c=h[3],end_c=end_c,end_ns=end_ns,tiles=tiles))
    return out


def observe(root,record,r,setup=None):
    """Numeric check for every variant; per-CTA events for stamped and ends."""
    replay(root,dict(record,variant='plain'),r)
    if record['variant']=='plain':return dict(elapsed_us=record['elapsed_us'])
    if setup is None:
        setup=next(s['setup'] for s in json.loads((root/'static_setup.json').read_text()) if s['case']==r['id'])
    work=scheduled_work(r['config'],r['m'],r['n'],setup['grid'],r['swizzle'])
    if record['variant']=='stamped':
        o=replay(root,record,r)
        if [[tuple(x) for x in c['work']] for c in o['ctas']]!=work:raise ValueError('scheduler model differs from recorded coordinates')
        return o
    ctas=_ends(root,record,r)
    if [c['tiles'] for c in ctas]!=[len(w) for w in work]:raise ValueError('ends tile counts differ from scheduler')
    return dict(elapsed_us=record['elapsed_us'],ctas=ctas)


def summarize_case(root,r,setup):
    """Medians over processes of everything the fit and the scoring use."""
    import v06_fit
    sched=base.CONFIGS[r['config']]['schedule'];res=dict(plain=[],stamped=[],ends=[]);procs=dict(stamped=[],ends=[])
    for path in sorted((root/'samples'/r['id']).glob('*.json')):
        rec=json.loads(path.read_text())
        if rec['returncode']:continue
        o=observe(root,rec,r,setup);res[rec['variant']].append(rec['elapsed_us'])
        if rec['variant']!='plain':procs[rec['variant']].append(o['ctas'])
    if min(len(v) for v in res.values())<10:raise ValueError('missing processes '+r['id'])
    feat=features(r,setup['grid']);T=feat['T'];out=dict(id=r['id'],config=r['config'],set=r['set'],kind=r['kind'],
        plain_us=med(res['plain']),stamped_us=med(res['stamped']),ends_us=med(res['ends']),
        plain_cv=statistics.pstdev(res['plain'])/statistics.fmean(res['plain']),
        stamped_cv=statistics.pstdev(res['stamped'])/statistics.fmean(res['stamped']),
        ends_cv=statistics.pstdev(res['ends'])/statistics.fmean(res['ends']),
        T=T,kt=feat['kt'],q=feat['q'],fp=feat['fp'],boundary=feat['boundary'],grid=setup['grid'])
    out['perturbation']=out['stamped_us']/out['plain_us']-1;out['ends_perturbation']=out['ends_us']/out['plain_us']-1
    def span(ctas):
        act=[c for c in ctas if (c['tiles'] if isinstance(c['tiles'],int) else len(c['tiles']))]
        cmax=max(c['end_c']-c['entry_c'] for c in act)
        full=[c for c in act if (c['tiles'] if isinstance(c['tiles'],int) else len(c['tiles']))==T]
        ghz=med((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns']) for c in full)
        window=(max(c['end_ns'] for c in act)-min(c['entry_ns'] for c in ctas))/1e3
        per_cta=[c['end_c']-c['entry_c'] for c in ctas]
        return cmax,ghz,window,per_cta
    for v in ['stamped','ends']:
        s=[span(c) for c in procs[v]]
        out[f'c_max_{v}']=med(x[0] for x in s);out[f'ghz_{v}']=med(x[1] for x in s);out[f'window_{v}']=med(x[2] for x in s)
        out[f'per_cta_{v}']=[med(x[3][i] for x in s) for i in range(len(s[0][3]))]
    per=[v06_fit.intervals(dict(ctas=c),sched)[0] for c in procs['stamped']]
    keys=set().union(*per);out['intervals']={k:med(p[k] for p in per if k in p) for k in keys}
    # Last-tile epilogue and supply on CTAs with the maximum tile count.
    elast=[];sup=[]
    for ctas in procs['stamped']:
        for c in ctas:
            if len(c['tiles'])==T:
                t=c['tiles'][-1];elast.append(t[3]-t[2]);sup.append(c['tiles'][0][0]-c['prod_c'])
    out['E_last']=med(elast);out['S_full']=med(sup)
    tl={}
    for ctas in procs['stamped']:
        for c in ctas:
            for j,(t,w) in enumerate(zip(c['tiles'],c['work'])):
                if j:tl.setdefault(tile_class(r['config'],r,*w),[]).append(t[1]-t[0])
    out['tile_L']={k:dict(median=med(v),n=len(v)) for k,v in tl.items()}
    return out


# ------------------------------------------------------------------ prediction
def lmain(p,kt,first=False):return p['l0']+p['l1']*kt+(p['dL0'] if first else 0.0)


def cta_cycles(p,schedule,ntiles,kt,detail=False,classes=None,mainloops=None,epilogues=None):
    """CTA recursion; an explicit per-tile L list replaces the aggregate mainloop rule.

    The optional list is for development composition of conditional supply rules.
    Existing callers keep L*(1+rho[class]); cooperative last tile uses Elast.
    Optional cooperative epilogues supply the merged E interval per output tile.
    """
    if ntiles==0:return (p['P0'],None) if detail else p['P0']
    rho=p.get('rho',{});cls=classes or ('in',)*ntiles
    Ls=([lmain(p,kt,j==0)*(1+rho.get(cls[j],0.0)) for j in range(ntiles)]
        if mainloops is None else list(mainloops))
    if len(Ls)!=ntiles:raise ValueError('one mainloop interval per output tile required')
    if epilogues is not None and (schedule!='cooperative' or len(epilogues)!=ntiles):
        raise ValueError('one merged epilogue per cooperative output tile required')
    if schedule=='cooperative':
        fm=p['P0']+p['S'];ed=None
        for j in range(ntiles):
            if j:fm=ed+p['h']
            ep=fm+Ls[j]+p['w']
            E=(p['E0'] if j==0 else (p['Elast'] if j==ntiles-1 else p['E'])) if epilogues is None else epilogues[j]
            ed=ep+E
        last=ed;eds=[ed]
    else:
        fm,me,ep,ed=[],[],[],[]
        for j in range(ntiles):
            if j==0:f=p['P0']+p['S']
            else:
                f=me[j-1]+p['gm']
                if j>=2:f=max(f,ed[j-2]+p['h'])
            fm.append(f);me.append(f+Ls[j])
            e=me[j]+p['w']
            if j>=1:e=max(e,ed[j-1]+p['we'])
            ep.append(e)
            if j+1<ntiles:
                f2=me[j]+p['gm']
                if j>=1:f2=max(f2,ed[j-1]+p['h'])
                ov=(f2,f2+Ls[j+1])
            else:ov=None
            ed.append(base.epilogue_end(p,e,ov))
        eds=ed;E=ed[-1]-ep[-1]
    b=max(eds)+p['Etail'];x=p['x0']+p['x1']*ntiles+p['xk']*b;end=b+x
    if not detail:return end
    return end,dict(supply=p['P0']+p['S'],mainloop=sum(Ls),last_epilogue=E,tail=p['Etail'],max_cta_excess=x)


def case_params(P,cfg,feat):
    """Config parameters with the case-feature forms applied."""
    p=dict(P[cfg]);sf,ef=p['S_form'],p['E_form']
    if sf=='log_fp':p['S']=p['s0']+p['s1']*math.log2(feat['fp'])
    if ef=='q':
        key='Elast' if base.CONFIGS[cfg]['schedule']=='cooperative' else 'E'
        p[key]=p['e0']+p['e1']*feat['q']
    return p


def critical_cycles(P,cfg,feat,with_boundary=True):
    p=case_params(P,cfg,feat);sched=base.CONFIGS[cfg]['schedule']
    memo={}
    for cl in feat['classes']:
        if cl not in memo:memo[cl]=cta_cycles(p,sched,len(cl),feat['kt'],True,cl)
    cyc=[memo[cl][0] for cl in feat['classes']]
    crit=max(range(len(cyc)),key=cyc.__getitem__);cmax=cyc[crit];stages=memo[feat['classes'][crit]][1]
    delta=0.0
    if with_boundary and feat['boundary'] and p.get('boundary_form','none')!='none':
        lo,hi=p.get('cols_range',[0,10**9]);cols=min(max(feat['cols'],lo),hi)  # no extrapolation beyond calibration
        delta=(p['d0']+p['d1']*cols)*feat['kt'] if p['boundary_form']=='cols' else p['d0']+p['d1']*feat['kt']
    return cmax+delta,cyc,stages,delta


def predict(r,grid,cal):
    cfg=r['config'];feat=features(r,grid);P=cal['params']
    cmax,cyc,stages,delta=critical_cycles(P,cfg,feat)
    t=cal['time'][cfg];kappa=t['kappa'];rule=t['clock_rule']
    phi=sum(cyc)/(SMS*cmax);mu=stages['mainloop']/cmax
    dbytes=base.dram_bytes(cfg,r['m'],r['n'],r['k'],feat['work']);c_conv=kappa*cmax;w=c_conv/1600
    for _ in range(200):
        f=base.clock_ghz(rule,phi,math.log(w),dbytes/(w*1e-6)/1e12,mu)
        if f<=0:raise ValueError('clock outside domain')
        w=.5*w+.5*c_conv/(f*1000)
    return dict(predicted_us=t['F']+w,window_us=w,clock_ghz=f,critical_cycles=cmax,converted_cycles=c_conv,kappa=kappa,
                fixed_us=t['F'],boundary_delta=delta,stage_cycles=stages,supply=stages['supply'],T=feat['T'],q=feat['q'],
                fp=feat['fp'],phi=phi,mu=mu,time_form=t['form'],per_cta_cycles=cyc,critical_cta=max(range(len(cyc)),key=cyc.__getitem__))
