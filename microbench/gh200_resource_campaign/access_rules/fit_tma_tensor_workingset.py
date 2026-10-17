"""源工作集状态候选：用同一请求服务引擎拟合开发端点。"""
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import statistics as st
from functools import partial

import numpy as np
from scipy.optimize import differential_evolution
import tma_tensor_multicta as model


def prices(x, program, base):
    working_mib = program['source_union_bytes']/1024**2
    fraction = min(1., max(0., (working_mib-x[5])/x[6]))
    return dict(service=x[2]+fraction*(x[0]-x[2]),
        latency=x[3]+fraction*(x[1]-x[3]), destination=x[4],
        issue=0., consumer=base['consumer'], control=base['control'])


def predict(x, row, base):
    program = row['program']
    parameters = prices(x, program, base)
    if program['source_protocol']=='shared' and program['source_phase_stride']==0:
        program = dict(program, ctas=1)
    report = model.simulate(program, 'shared_source_fifo', **parameters,
                            trace='common_prefix_ns' in row)
    result = dict(release_ns=st.median(report['cta_times']))
    if 'common_prefix_ns' in row:
        by = [[] for _ in range(program['ctas'])]
        for event in report['group_events']:
            if 4 <= event['group'] < 12:
                by[event['cta']].append(event['wait_return']-event['issue'])
        result['common_prefix_ns'] = st.median(st.median(values) for values in by)
    return result


def cost(x, data, base):
    errors=[]
    for row in data:
        predicted=predict(x,row,base)
        errors.extend(predicted[key]/row[key]-1 for key in predicted)
    return float(np.mean(np.square(errors))+max(abs(error) for error in errors)**2)


def load(base):
    files=[base/'20261014-tma-tensor-development/banked-001/fit-p16.json',
           base/'20261014-tma-tensor-heldout-002/score.json',
           base/'20261015-tma-footprint-discrimination-001/score.json']
    prior=json.loads(files[0].read_text())
    data=[dict(id=row['id'],program=row['program'],release_ns=row['observed_ns'])
          for row in prior['observations']]
    for path in files[1:]:
        for row in json.loads(path.read_text())['observations']['rows']:
            a=row['arguments']
            if not a['trace'] or a.get('release-delay-ns',0):
                continue
            p=model.program(a['ctas'],a['active'],'independent' if a['independent'] else 'shared',
                groups=a['groups'],source_ring=a['ring'],source_phase_stride=a.get('source-phase-stride',0),ldb=a.get('ldb'))
            record=dict(id=row['id'],program=p,release_ns=row['summary']['release_window_ns']['median'])
            if 'common_prefix_ns' in row['summary']:
                record['common_prefix_ns']=row['summary']['common_prefix_ns']['median']
            data.append(record)
    return data,prior['parameters'],files


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=40)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    data,base,files=load(Path('results/gh200_resource_campaign/access_rules'))
    bounds=[(2.,6.),(350.,1300.),(.01,1.),(150.,420.),(0.,100.),(28.,34.),(4.,20.)]
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        result=differential_evolution(partial(cost,data=data,base=base),bounds,seed=20261015,
            popsize=6,maxiter=args.iterations,tol=.002,polish=False,workers=pool.map,
            updating='deferred',disp=True)
    rows=[]
    for row in data:
        predicted=predict(result.x,row,base)
        rows.append(dict(**row,predicted=predicted,errors={k:predicted[k]/row[k]-1 for k in predicted}))
    report=dict(qualification='development; working-set state C2 already supported, quantitative rule not held out',
        parameter_names=['stream_service_ns','stream_latency_ns','resident_service_ns','resident_latency_ns',
                         'destination_ns','transition_start_mib','transition_width_mib'],
        parameters=result.x.tolist(),control_ns=base['control'],consumer_ns=base['consumer'],
        bounds=bounds,seed=20261015,iterations=args.iterations,objective=float(result.fun),
        optimizer=dict(success=bool(result.success),message=str(result.message),evaluations=int(result.nfev)),
        observations=rows,max_absolute_error=max(abs(e) for r in rows for e in r['errors'].values()),
        input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(model.__file__)]})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['max_absolute_error'],report['parameters'])
