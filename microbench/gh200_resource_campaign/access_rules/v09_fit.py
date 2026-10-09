#!/usr/bin/env python3
"""Extract natural-ns event calibration for the V09 development recurrence.

Uses the original event definitions, with middle/last cooperative outputs separate.
No target timing prediction or automatic cross-job pooling is performed here.
"""
import argparse
import json
from pathlib import Path
import statistics as st

from analyze_r13_sm import select_attempts
from analyze_r18 import replay
import v06_fit
import v06_run as common
import v08_model


def event_observations(run):
    rows = [r for r in json.loads((run/'cases.json').read_text()) if r['set'] in ('calib','ctrl')]
    setups = {s['case']:s['setup'] for s in json.loads((run/'static_setup.json').read_text())}
    cases, failed = [], []
    for row in rows:
        setup = setups[row['id']]
        work = v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'])
        tm,tn,_ = v08_model.base.CONFIGS[row['config']]['tile']
        first_bytes = sum(4*max(0,min(tm,row['m']-c[0][0]*tm))*max(0,min(tn,row['n']-c[0][1]*tn)) for c in work if c)
        chosen, bad, _ = select_attempts(run,row['id'],('plain','dual'))
        failed.extend(bad)
        plain = {p['record']['trial']:p['record'] for p in chosen if p['record']['variant']=='plain'}
        dual = [p['record'] for p in chosen if p['record']['variant']=='dual']
        if len(plain)!=10 or len(dual)!=10:
            raise ValueError('ten successful paired plain/dual processes required: '+row['id'])
        schedule = v08_model.base.CONFIGS[row['config']]['schedule']
        processes=[]
        for record in dual:
            observed = replay(run,record,row)
            replay(run,plain[record['trial']],row)
            ctas=observed['ctas']
            if [[tuple(p) for p in c['work']] for c in ctas]!=work:
                raise ValueError('software work differs from observed CTA mapping')
            ns=[dict(c,entry_c=c['entry_ns'],prod_c=c['prod_ns'],end_c=c['end_ns'],tiles=c['tiles_ns']) for c in ctas]
            intervals,counts=v06_fit.intervals(dict(ctas=ns),schedule)
            if schedule=='cooperative':
                for key,values in (
                    ('E0_mean',[c['tiles'][0][3]-c['tiles'][0][2] for c in ns if c['tiles']]),
                    ('E_middle',[t[3]-t[2] for c in ns for t in c['tiles'][1:-1]]),
                    ('Elast',[c['tiles'][-1][3]-c['tiles'][-1][2] for c in ns if len(c['tiles'])>=2])):
                    if values:intervals[key]=st.fmean(values);counts[key]=len(values)
            envelope=max(c['end_ns'] for c in ctas)-min(c['entry_ns'] for c in ctas)
            processes.append(dict(trial=record['trial'],raw=record['raw'],raw_sha256=record['raw_sha256'],intervals_ns=intervals,counts=counts,
                envelope_ns=envelope,dual_event_ns=record['elapsed_us']*1000,
                plain_event_ns=plain[record['trial']]['elapsed_us']*1000,
                same_call_event_extra_ns=record['elapsed_us']*1000-envelope))
        keys=set().union(*(p['intervals_ns'] for p in processes))
        cases.append(dict(row=row,setup=setup,max_tiles=max(map(len,work)),active_ctas=sum(bool(c) for c in work),
            first_output_bytes=first_bytes,processes=processes,
            intervals_ns={k:st.median(p['intervals_ns'][k] for p in processes if k in p['intervals_ns']) for k in sorted(keys)},
            event_extra_ns=st.median(p['same_call_event_extra_ns'] for p in processes)))
    return dict(run=str(run),selected_sets=['calib','ctrl'],cases_sha256=common.sha(run/'cases.json'),setup_sha256=common.sha(run/'static_setup.json'),environment=json.loads((run/'environment.json').read_text()),cases=cases,failed_attempts=failed,
        scope='Calibration observations. Missing interval keys are unobserved, not zero. F uses a same-call dual envelope; plain remains a separate observation.')


def constant_event_calibration(observations):
    """Same-card case medians; no component-error compensation in time transfer."""
    import numpy as np
    result=dict(events_ns={},dual_event_extra_ns={},plain_transfer={})
    for cfg in sorted({c['row']['config'] for c in observations['cases']}):
        cases=[c for c in observations['cases'] if c['row']['config']==cfg]
        def median(key):
            values=[c['intervals_ns'][key] for c in cases if key in c['intervals_ns']]
            if not values:raise ValueError(cfg+': unobserved event '+key)
            return st.median(values)
        keys=['P0','S','w','h','Etail']
        p={k:median(k) for k in keys}
        if cfg=='cfg_b':
            p.update(E=median('E'),gm=median('gm'),we=median('we'))
            p['r']=p['E']/median('Efull')
        else:
            p.update(E0=median('E0_mean'),E=median('E_middle'),Elast=median('Elast'))
        p.update(x0=0.,x1=0.,xk=0.,rho={})
        result['events_ns'][cfg]=p
        result['dual_event_extra_ns'][cfg]=st.median(c['event_extra_ns'] for c in cases)
        # Regress plain event against observed dual envelope, not an imperfect
        # component prediction: this transfer cannot absorb supply-model errors.
        x=np.array([st.median(p['envelope_ns'] for p in c['processes'])/1000 for c in cases])
        y=np.array([st.median(p['plain_event_ns'] for p in c['processes'])/1000 for c in cases])
        design=np.column_stack([np.ones(len(x)),x])
        coefficients,_,rank,_=np.linalg.lstsq(design,y,rcond=None)
        if rank!=2 or min(coefficients)<0:raise ValueError(cfg+': no positive affine observer transfer')
        residual=(design@coefficients)/y-1
        result['plain_transfer'][cfg]=dict(F_us=float(coefficients[0]),kappa=float(coefficients[1]),
            calibration_median_abs_pct=float(np.median(abs(residual))*100),
            calibration_max_abs_pct=float(max(abs(residual))*100),
            scope='Empirical plain-event / dual-envelope transfer on these calibration cases.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=event_observations(args.run.resolve())
    report['extractor_sha256']=common.sha(Path(__file__))
    args.output.mkdir(parents=True,exist_ok=False)
    common.write_json(args.output/'events.json',report)
    print(len(report['cases']),'cases of direct-ns event observations')
