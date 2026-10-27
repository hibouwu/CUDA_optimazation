#!/usr/bin/env python3
"""One B32 coverage-price development screen in the cfg_b lifecycle graph.

Fit one AlongM later response on GPU-ef8692f3; retain all first/later, direction,
scale and spatial comparisons. No A16/stage4 price, target-clock input or GPU IO.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from scipy.optimize import brentq

from analyze_r18 import replay
from supply_model import cfg_b_source_program, shared_stage_events, static_tma_requests

GPU='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'


def read(path):
    return json.loads(path.read_text())


def span(values):
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values))


def source_costs(row, setup, service_ns, coverage_price):
    """Pure software features plus explicit prices; no observations or clocks."""
    request=static_tma_requests(row,setup)
    result=[]
    for tiles in request['ctas']:
        actor=[]
        for tile in tiles:
            if tile['A']['source_KiB'][1:] != [0,0]:
                raise ValueError('this B32 screen does not price A coverage')
            costs=[service_ns]
            costs.extend(service_ns*.5+coverage_price*box['source_KiB'][2]
                for box in tile['B']['source_boxes'])
            actor.extend([costs]*(row['k']//64))
        result.append(actor)
    return result


def simulate(row, setup, program, parameters, coverage_price):
    if setup['gpu_uuid']!=GPU or setup['stages']!=6:
        raise ValueError('development prices require the declared GPU and stage6')
    kt=row['k']//64
    model=shared_stage_events(kt,6,len(program['output_counts']),
        source_keys=program['source_keys'],output_counts=program['output_counts'],
        share_inflight_sources=True,
        source_service_costs_ns=source_costs(row,setup,parameters['service_ns'],coverage_price),
        **parameters)
    windows=[]
    for cta,actor in enumerate(model['ctas']):
        for j,coord in enumerate(program['cta_output_coords'][cta]):
            windows.append(dict(cta=cta,j=j,T=program['output_counts'][cta],coord=coord,
                L_ns=actor['main_ends'][j]-actor['starts'][j*kt],
                MAIN_ns=actor['main_ends'][j],
                E_ns=actor['outputs'][j]['end']-actor['outputs'][j]['begin']))
    return dict(windows=windows,source_work_16KiB=model['input_source_work'],
        delivery_work_16KiB=model['input_delivery_work'],source_count=model['input_source_count'],
        output_atoms=sum(program['output_counts'])*8,
        last_MAIN_span_ns=max(a['main_ends'][-1] for a in model['ctas'])-min(a['main_ends'][-1] for a in model['ctas']))


def paired_windows(base, variant, selected=lambda w:True):
    b={(w['cta'],w['j']):w for w in base['windows']}
    v={(w['cta'],w['j']):w for w in variant['windows']}
    if b.keys()!=v.keys():raise ValueError('paired model work differs')
    result=[]
    for key,x in b.items():
        y=v[key]
        if (x['coord'],x['T'])!=(y['coord'],y['T']):raise ValueError('paired model position differs')
        if selected(x):
            pair=dict(cta=x['cta'],j=x['j'],T=x['T'],coord=x['coord'],
                L_ns_base=x['L_ns'],L_ns_variant=y['L_ns'],
                L_ns_delta=y['L_ns']-x['L_ns'],L_ns_ratio=y['L_ns']/x['L_ns']-1)
            if 'L_cycles' in x and 'L_cycles' in y:
                pair.update(L_cycles_base=x['L_cycles'],L_cycles_variant=y['L_cycles'],
                    L_cycles_delta=y['L_cycles']-x['L_cycles'],
                    L_cycles_ratio=y['L_cycles']/x['L_cycles']-1)
            result.append(pair)
    return result


def analyze(root, output):
    output.mkdir(parents=True,exist_ok=False)
    dev=root/'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1'
    parameters_path=dev/'B-source-candidate-examples.json'
    parameters=read(parameters_path)['examples']['shared']['parameters']
    observed=read(dev/'R10-raster-pitch.json')
    target=next(r['phases']['later']['L_ns_ratio']['median'] for r in observed['comparisons']
        if r['control']=='r10_raster_along_m_p0')
    inputs={}
    def fingerprint(path):
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for p in (parameters_path,dev/'R10-raster-pitch.json',dev/'R10-source-pressure.json',
            Path(__file__),Path(__file__).with_name('supply_model.py')):fingerprint(p)
    parameter_setup=root/'Q02-tma-request/R13-async-retirement/20261010-R13-second-prefill/static_setup.json'
    fingerprint(parameter_setup)
    if any(r['setup']['gpu_uuid']!=GPU for r in read(parameter_setup)):
        raise ValueError('declared R13 parameter cohort is not on the R10 card')
    kernel=root/'Q02-tma-request/R13-async-retirement/20261010-R13-second-prefill/source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp'
    fingerprint(kernel)
    contexts={}
    for run_name in ('Q03-source-service/R10-layout-cache/20261010-R10-raster-pitch','Q03-source-service/R10-layout-cache/20261010-R10-source-pressure'):
        run=root/run_name
        rows=read(run/'cases.json');setups={s['case']:s['setup'] for s in read(run/'static_setup.json')}
        for p in (run/'cases.json',run/'static_setup.json'):fingerprint(p)
        for row in rows:
            if row.get('set')=='reference':continue
            setup=setups[row['id']]
            contexts[row['id']]=dict(row=row,setup=setup,program=cfg_b_source_program(row,setup),run=run)
    predicted={}
    for case,context in contexts.items():
        if context['row']['ldb']==3072:
            predicted[case]=simulate(context['row'],context['setup'],context['program'],parameters,0.)
    calibration='r10_raster_along_m_p32';base_case='r10_raster_along_m_p0'
    evaluations=[]
    def residual(price):
        c=contexts[calibration]
        prediction=simulate(c['row'],c['setup'],c['program'],parameters,float(price))
        response=statistics.median(w['L_ns_ratio'] for w in paired_windows(predicted[base_case],prediction,lambda w:w['j']>0))
        evaluations.append(dict(price_ns_per_coverage_KiB=float(price),later_response=response,residual=response-target))
        return response-target
    bounds=[0.,4.]
    left,right=(residual(p) for p in bounds)
    if left*right>0:
        raise ValueError('declared B32 price interval does not bracket response; no bound expansion')
    located_price, solver = brentq(residual,*bounds,xtol=1e-8,maxiter=40,full_output=True)
    price=float(located_price)
    for case,c in contexts.items():
        if case not in predicted:
            predicted[case]=simulate(c['row'],c['setup'],c['program'],parameters,price)
    final_response=statistics.median(w['L_ns_ratio'] for w in paired_windows(
        predicted[base_case],predicted[calibration],lambda w:w['j']>0))
    final_residual=final_response-target
    observations={}
    raw_checks=0
    for case,c in contexts.items():
        records=[]
        for trial in range(10):
            p=c['run']/f'samples/{case}/dual-{trial:02d}.json'
            record=read(p);raw=c['run']/record['raw'];fingerprint(p);fingerprint(raw)
            if inputs[str(raw)]!=record['raw_sha256'] or record['returncode']:
                raise ValueError('raw identity or failed process')
            data=replay(c['run'],record,c['row']);windows=[]
            for actor in data['ctas']:
                cta=actor['cta'];coords=c['program']['cta_output_coords'][cta]
                if actor['work']!=[tuple(v) for v in coords]:
                    raise ValueError('static source program and actual CTA work differ')
                for j,coord in enumerate(coords):
                    t=actor['tiles_ns'][j]
                    cycles=actor['tiles'][j]
                    windows.append(dict(cta=cta,j=j,T=len(coords),coord=coord,
                        L_ns=t[1]-t[0],L_cycles=cycles[1]-cycles[0],MAIN_ns=t[1]))
            records.append(dict(trial=trial,windows=windows,
                last_MAIN_span_ns=max(a['tiles_ns'][-1][1] for a in data['ctas'])-
                    min(a['tiles_ns'][-1][1] for a in data['ctas'])))
            raw_checks+=1
        observations[case]=records
    comparisons=[]
    pairs=[('along_m','r10_raster_along_m_p0','r10_raster_along_m_p32',False),
        ('along_n','r10_raster_along_n_p0','r10_raster_along_n_p32',False)]
    pairs.extend((g,f'r10_source_{g}_p0',f'r10_source_{g}_p32',True)
        for g in ('single36','single126','persistent132'))
    for label,base,variant,first36 in pairs:
        selector=lambda w:first36 is False or (w['cta']<36 and w['j']==0)
        model_pair=paired_windows(predicted[base],predicted[variant],selector)
        observed_pairs=[paired_windows(observations[base][t],observations[variant][t],selector) for t in range(10)]
        groups=sorted({(w['j'],w['T']) for w in model_pair})
        for j,T in groups:
            m=[w for w in model_pair if (w['j'],w['T'])==(j,T)]
            trials=[]
            for trial,windows in enumerate(observed_pairs):
                group=[w for w in windows if (w['j'],w['T'])==(j,T)]
                trials.append(dict(trial=trial,**{f:statistics.median(w[f] for w in group)
                    for f in ('L_ns_base','L_ns_variant','L_ns_delta','L_ns_ratio',
                        'L_cycles_base','L_cycles_variant','L_cycles_delta','L_cycles_ratio')}))
            comparisons.append(dict(pair=label,j=j,T=T,windows=len(m),
                predicted={f:statistics.median(w[f] for w in m)
                    for f in ('L_ns_base','L_ns_variant','L_ns_delta','L_ns_ratio')},
                observed={f:span([t[f] for t in trials])
                    for f in ('L_ns_base','L_ns_variant','L_ns_delta','L_ns_ratio',
                        'L_cycles_base','L_cycles_variant','L_cycles_delta','L_cycles_ratio')},trials=trials))
    spatial=[]
    for case,model in predicted.items():
        for j,T in sorted({(w['j'],w['T']) for w in model['windows']}):
            windows=[w for w in model['windows'] if (w['j'],w['T'])==(j,T)]
            obs=[];cycle_spans=[]
            for r in observations[case]:
                group=[w for w in r['windows'] if (w['j'],w['T'])==(j,T)]
                obs.append(max(w['L_ns'] for w in group)-min(w['L_ns'] for w in group))
                cycle_spans.append(max(w['L_cycles'] for w in group)-min(w['L_cycles'] for w in group))
            spatial.append(dict(case=case,j=j,T=T,windows=len(windows),
                predicted_L_span_ns=max(w['L_ns'] for w in windows)-min(w['L_ns'] for w in windows),
                observed_L_span_ns=span(obs),observed_L_span_cycles=span(cycle_spans)))
    report=dict(scope='Post-reveal development screen on one card. No accepted supply rule or new holdout.',
        gpu_uuid=GPU,parameters=parameters,coverage_price_ns_per_KiB=price,
        calibration=dict(case=calibration,target='median matched later-window B32/B0 ns ratio',
            observed_target=target,price_bounds=bounds,evaluations=evaluations,
            located_response=final_response,response_residual=final_residual,
            zero_response_residual=final_residual==0,
            solver_price_tolerance=1e-8,solver_converged=bool(solver.converged),
            solver_iterations=solver.iterations,solver_function_calls=solver.function_calls,
            method='Brent bracket locator, at most 40 iterations; convergence is in price, not proof of a zero response residual. Retain the nonzero residual at this approximate development point. No A16, first, stage4, AlongN or scale target.'),
        source_price='Existing nominal source service + one nonnegative B extra128 coverage price. Coverage is not physical traffic.',
        contexts={k:dict(row=v['row'],setup=v['setup'],counts=v['program']['output_counts'],
            totals=v['program']['totals']) for k,v in contexts.items()},
        predicted=predicted,comparisons=comparisons,spatial=spatial,
        raw_processes_verified=raw_checks,input_sha256=inputs,
        limitations=['Recovered R13 tuple has exploratory selection history and unresolved clock/first mapping; it is fixed here, not independently calibrated on this data.',
            'Observed native cycles are retained separately; no target cycle/ns ratio converts model predictions. This ns-only model does not provide a validated cycle mapping or invocation-history state.',
            'All CTA roots are simultaneous; neither entry offsets nor observed target clocks are prediction inputs.',
            'Source/destination and accumulator prerequisites are explicit, but release emit/visibility remain combined pending the separate minimal observation.',
            'Predicted end is source-read tail, not global D visibility or CUDA-event time.',
            'GPU-43269fbc old/V09 G5 data do not receive GPU-ef8692f3 prices. Raster intervention on the current card is the causal geometry bridge; cross-card numerical migration is still unsupported.'])
    (output/'B-r10-event-transfer.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# R10：同一供给图中的B32开发迁移','',
        '保留cfg_b真实每CTA输出次数、输入环代次、源box身份和两WG accumulator复用；源/目的/D分账不变。固定既有R13探索参数，仅以AlongM later配对的一个量解B extra128覆盖价。','',
        f'覆盖价={price:.9g} ns/KiB；这是条件源服务代理，不是物理请求价。Brent在[0,4]内按价格容差定位近似开发点，最多40次迭代，未另拟A16或stage4。',
        f'定位点later响应={final_response:.12g}，目标={target:.12g}，残差={final_residual:+.12g}；价格收敛不证明响应有精确零点。','',
        '| 配对 j/T | 预测B32响应 | 实测中位〔范围〕 | 预测/实测基线L ns |',
        '|---|---:|---:|---:|']
    for row in comparisons:
        p=row['predicted'];o=row['observed']['L_ns_ratio']
        lines.append(f'| {row["pair"]} {row["j"]}/{row["T"]} | {100*p["L_ns_ratio"]:.3f}% | {100*o["median"]:.3f}% [{100*o["minimum"]:.3f},{100*o["maximum"]:.3f}] | {p["L_ns_base"]:.1f}/{row["observed"]["L_ns_base"]["median"]:.1f} |')
    lines+=['','所有j/T的局部跨度和逐CTA预测在JSON，不能由一个已校准响应替代first/方向/规模/空间检验。V09原冻结及旧R10失败不变，另一张卡的价格不互相移植。','']
    (output/'B-r10-event-transfer.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),price=price,raw_processes=raw_checks,
        comparisons=[dict(pair=r['pair'],j=r['j'],T=r['T'],predicted=r['predicted']['L_ns_ratio'],observed=r['observed']['L_ns_ratio']['median']) for r in comparisons])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve())
