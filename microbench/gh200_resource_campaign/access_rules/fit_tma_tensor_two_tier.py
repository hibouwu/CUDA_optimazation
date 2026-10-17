"""两种源服务率、两种完成延迟与在途共享的开发拟合；不改冻结成绩。"""
import argparse
from functools import partial
import hashlib
import json
import multiprocessing
from pathlib import Path
import statistics as st

import numpy as np
from scipy.optimize import differential_evolution
import tma_tensor_multicta as model


def parameters(x, base):
    return dict(service=float(x[0]),latency=float(x[1]),resident_service=float(x[2]),
        resident_latency=float(x[3]),destination=float(x[4]),issue=0.,
        control=base['control_ns'],consumer=base['consumer_ns'],source_partitions=base['partitions'])


def predict(p, prices):
    # 同时起步、共享同一地址、独立目的服务时各 CTA 对称。
    if p['source_protocol']=='shared' and p['source_phase_stride']==0:
        p=dict(p,ctas=1)
    candidate='banked_resident_fifo' if prices['source_partitions']>1 else 'two_tier_shared_fifo'
    result=model.simulate(p,candidate,**prices)
    return st.median(result['cta_times'])


def objective(x, data, base):
    prices=parameters(x,base)
    errors=[predict(row['program'],prices)/row['observed_ns']-1 for row in data]
    # 同时压低整体误差和最坏条件，不能用正负抵消。
    return float(np.mean(np.square(errors))+max(abs(e) for e in errors)**2)


def fit(development, discrimination, output, workers, iterations, partitions=1, pitch=None):
    prior=json.loads((development/'candidate-screening.json').read_text())
    calibration=json.loads((development/'calibration-endpoints.json').read_text())
    frozen=json.loads((discrimination/'predictions.json').read_text())
    scored=json.loads((discrimination/'score.json').read_text())
    crows={r['id']:r for r in calibration['rows']}
    anchor=crows['n1-q2-g128-s0-t1-d0']['summary']
    base=dict(control_ns=prior['control_ns'],consumer_ns=anchor['consumer_cycles']['median']/prior['frequency_ghz'],partitions=partitions)
    data=[]
    for row in calibration['rows']:
        a=row['arguments']
        if a['trace']==1 and a['groups']==128 and not a.get('release-delay-ns',0):
            p=model.program(a['ctas'],2,'independent' if a['independent'] else 'shared',groups=128)
            data.append(dict(id=row['id'],program=p,observed_ns=row['summary']['release_window_ns']['median']))
    observations={r['id']:r for r in scored['observations']['rows']}
    for row in frozen['predictions']:
        observed=observations[row['id']+'-t1']['summary']['release_window_ns']['median']
        data.append(dict(id=row['id'],program=row['program'],observed_ns=observed))
    if pitch:
        for row in json.loads(pitch.read_text())['rows']:
            a=row['arguments']
            if a['trace']:
                p=model.program(a['ctas'],1,'shared',groups=64,source_ring=32,ldb=a['ldb'])
                data.append(dict(id=row['id'],program=p,observed_ns=row['summary']['release_window_ns']['median']))
    bounds=[(2.5,6.),(100.,1800.),(.25,6.),(0.,600.),(0.,200.)]
    cost=partial(objective,data=data,base=base)
    with multiprocessing.get_context('fork').Pool(workers) as pool:
        result=differential_evolution(cost,bounds,seed=20261014,popsize=8,maxiter=iterations,
            tol=.002,polish=False,workers=pool.map,updating='deferred',disp=True)
    prices=parameters(result.x,base)
    rows=[]
    for row in data:
        value=predict(row['program'],prices)
        rows.append(dict(**row,predicted_ns=value,error=value/row['observed_ns']-1))
    report=dict(qualification='development after both first-batch frozen price families failed',
        hypothesis='resident and streaming source service/latency differ; in-flight same-key sharing retains independent destinations',
        parameters=prices,bounds=bounds,seed=20261014,iterations=iterations,source_partitions=partitions,
        pitch_source=dict(path=str(pitch),sha256=hashlib.sha256(pitch.read_bytes()).hexdigest()) if pitch else None,
        optimizer=dict(success=bool(result.success),message=str(result.message),evaluations=int(result.nfev),loss=float(result.fun)),
        observations=rows,max_absolute_error=max(abs(r['error']) for r in rows),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [development/'candidate-screening.json',development/'calibration-endpoints.json',discrimination/'predictions.json',discrimination/'score.json']},
        model_sha256=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('development max error',report['max_absolute_error'],prices)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('development',type=Path)
    parser.add_argument('discrimination',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--partitions',type=int,default=1)
    parser.add_argument('--pitch',type=Path)
    a=parser.parse_args()
    fit(a.development,a.discrimination,a.output,a.workers,a.iterations,a.partitions,a.pitch)
