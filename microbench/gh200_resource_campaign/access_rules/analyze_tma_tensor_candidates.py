"""用 Q2 校准筛选价格组，计算源环相位干预的候选预测。"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics as st

import tma_tensor_multicta as model


def conditions():
    rows = [p for p in model.cases() if not (p['ctas'] == 1 and p['source_protocol'] == 'independent')]
    rows += [model.program(132,1,'shared',groups=128,source_ring=128,source_phase_stride=phase)
             for phase in (0,1)]
    return rows


def name(p):
    return f"n{p['ctas']}-q{p['active_groups']}-g{p['groups']}-r{p['source_ring']}-{p['source_protocol']}-p{p['source_phase_stride']}"


def analyze(paths):
    data = []
    for path in paths:
        data += json.loads(path.read_text())['rows']
    rows = [r for r in data if r['arguments']['trace'] == 1 and not r['arguments'].get('release-delay-ns',0)]
    lookup = {(r['arguments']['ctas'],r['arguments']['groups'],r['arguments']['independent']):r for r in rows}
    anchor = lookup[1,128,0]['summary']
    frequency = anchor['effective_ghz']['median']
    mapping = dict(prefix='prefix',prefill_issue='prefill_issue',refill_issue='issue',
        first_wait='first_wait',issue_to_wait='issue_to_wait',wait_floor='wait',
        release_to_issue='release_to_issue',prefill_gap='prefill_gap',drain_gap='drain_gap')
    control = {k:anchor[v+'_cycles']['median']/frequency for k,v in mapping.items()}
    consumer = anchor['consumer_cycles']['median']/frequency
    long, short = [lookup[64,g,1]['summary']['release_window_ns'] for g in (128,32)]
    # 饱和流的 G128−G32 增量；每组、每 CTA 为两个 16 KiB 权重。
    slope = (long['median']-short['median'])/(96*128)
    services = [slope*x for x in (.95,1.,1.05)]
    points = list(itertools.product(services,range(0,801,100),(0,50,100,150,200)))
    # 先检查便宜的单 CTA，再检查高负载条件，保留全部满足 5% 的离散价格组。
    order = sorted(rows,key=lambda r:(r['arguments']['ctas'] != 1,
        r['arguments']['ctas'] != 64,r['arguments']['groups']))
    selected = {}
    for candidate in ('shared_source_fifo','resident_source_fifo'):
        accepted = []
        for service, latency, destination in points:
            resident_prices = (.5,1.,2.) if candidate == 'resident_source_fifo' else (None,)
            for resident in resident_prices:
                prices = dict(service=service,latency=latency,destination=destination,issue=0.,
                    consumer=consumer,control=control,resident_service=resident)
                errors = []
                for row in order:
                    a = row['arguments']
                    p = model.program(a['ctas'],2,'independent' if a['independent'] else 'shared',groups=a['groups'])
                    result = model.simulate(p,candidate,trace=a['ctas'] == 1,**prices)
                    predicted = st.median(result['cta_times'])
                    observed = row['summary']['release_window_ns']['median']
                    error = predicted/observed-1
                    errors.append(dict(id=row['id'],predicted_ns=predicted,observed_ns=observed,error=error))
                    if abs(error) > .05:
                        break
                    if a['ctas'] == 1 and 'first_wait_return_ns' in row['summary']:
                        observed_first = row['summary']['first_wait_return_ns']['median']
                        first_error = result['group_events'][0]['wait_return']-observed_first
                        errors[-1]['first_wait_return_error_ns'] = first_error
                        if abs(first_error) > 64:
                            break
                else:
                    accepted.append(dict(parameters=prices,calibration=errors))
        selected[candidate] = accepted
        print(candidate, 'retained parameter sets',len(accepted),flush=True)
    predictions = []
    for p in conditions():
        by_candidate = {}
        for candidate, accepted in selected.items():
            values = []
            for index, entry in enumerate(accepted):
                result = model.simulate(p,candidate,**entry['parameters'])
                values.append(dict(parameter_index=index,median_cta_ns=st.median(result['cta_times']),
                    maximum_cta_ns=result['time'],cta_span_ns=result['cta_span'],merges=result['merges']))
            by_candidate[candidate] = values
        predictions.append(dict(id=name(p),program=p,candidates=by_candidate))
    contrasts = {}
    for candidate, accepted in selected.items():
        before = predictions[-2]['candidates'][candidate]
        after = predictions[-1]['candidates'][candidate]
        differences = [b['median_cta_ns']/a['median_cta_ns']-1 for a,b in zip(before,after)]
        contrasts[candidate] = dict(price_sets=len(accepted),response_min=min(differences) if differences else None,
            response_max=max(differences) if differences else None)
    a = lookup[64,128,0]['summary']['release_window_ns']['median']
    b = lookup[64,128,1]['summary']['release_window_ns']['median']
    return dict(schema='tensor-candidate-screening/1',qualification='development finite parameter grid; not a continuous identifiability proof',
        source_summaries={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        frequency_ghz=frequency,control_ns=control,source_service_slope_ns=slope,
        search=dict(service=services,latency=list(range(0,801,100)),destination=[0,50,100,150,200],
            resident_service=[.5,1.,2.],acceptance_max_relative_error=.05,first_wait_return_tolerance_ns=64),
        excluded_uniform_price_models=dict(candidates=['local_fifo','shared_fifo'],
            counterexample='same N64/Q2/G128, shared vs independent source: both candidates predict equal durations',
            observations_ns=[a,b],equal_prediction_minimax_error=(b-a)/(b+a)),
        retained=selected,predictions=predictions,phase_contrasts=contrasts,
        implementation_sha256=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = analyze(args.summary)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['phase_contrasts'],indent=2))
