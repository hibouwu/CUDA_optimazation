#!/usr/bin/env python3
"""One prefix-selected shared request-path point in the complete cfg_b graph."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import statistics

from analyze_r10_granule_overlap import source_granules
from analyze_r10_supply_events import GPU,paired_windows
from supply_model import cfg_b_source_program,shared_stage_events


def windows(model,program,kt):
    result=[]
    for cta,actor in enumerate(model['ctas']):
        for j,coord in enumerate(program['cta_output_coords'][cta]):
            result.append(dict(cta=cta,j=j,T=program['output_counts'][cta],coord=coord,
                L_ns=actor['main_ends'][j]-actor['starts'][j*kt],FIRST_ns=actor['starts'][j*kt],
                MAIN_ns=actor['main_ends'][j],E_ns=actor['outputs'][j]['end']-actor['outputs'][j]['begin'],
                first_input_ready_ns=actor['ready'][j*kt],
                post_FIRST_TC_idle_ns=sum(max(0.,actor['starts'][i]-actor['service_frontier'][i-1])
                    for i in range(j*kt+1,(j+1)*kt))))
    return result


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        path=path.resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in inputs and inputs[str(path)]!=digest:raise ValueError('input changed: '+str(path))
        inputs[str(path)]=digest
        return digest
    def read(path):sha(path);return json.loads(path.read_text())
    dev=root/'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1'
    prefix=read(root/'Q03-source-service/R10-layout-cache/20261010-R10-producer-delay/reanalysis/B-20261011-shared-path-prefix-v1/B-shared-path-prefix.json')
    feedback=read(root/'Q03-source-service/R10-layout-cache/20261010-R10-raster-pitch/reanalysis/B-20261011-address-feedback-v3/B-address-feedback.json')
    original=read(root/'Q03-source-service/R10-layout-cache/20261010-R10-raster-pitch/reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json')
    delay=read(dev/'R10-producer-delay.json');prefill=read(dev/'B-second-prefill.json')
    stage_windows=read(dev/'B-source-join-boundary.json')
    rehashed=set()
    for prior in (prefix,feedback,original,delay,prefill,stage_windows):
        for name,digest in prior['input_sha256'].items():
            path=Path(name)
            if path.is_relative_to(root):
                if sha(path)!=digest:raise ValueError('reused data changed: '+name)
                rehashed.add(name)
    params=prefix['parameters']
    if params!=feedback['parameters'] or params!=original['parameters']:
        raise ValueError('fixed prior tuple changed')
    selected=next(r for r in prefix['results'] if r['placement']=='request')
    interval=next(r for r in selected['delta_results'] if r['delta_ns']==512)['joint_price_intervals']
    if len(interval)!=1 or interval[0]['upper'] is None:raise ValueError('one bounded prior prefix interval required')
    lower,upper=F(interval[0]['lower']),F(interval[0]['upper'])
    exact_price=(lower+upper)/2;price=float(exact_price)
    selection=dict(source='Saved aligned initial-prefix necessary interval, prescribed root512, tau=old read latency.',
        rule='Arithmetic midpoint of the single interval; fixed before the following transfer calculations.',
        lower_exact=str(lower),upper_exact=str(upper),selected_exact=str(exact_price),selected_ns=price,
        limits='Developmental representative, not a unique estimate or independent calibration; no B32/space/scale refit.')
    simulations=0;simulated_demands=0
    def simulate(row,setup,stage=None,**treatment):
        nonlocal simulations,simulated_demands
        if setup['gpu_uuid']!=GPU:raise ValueError('no cross-card price transfer')
        program=cfg_b_source_program(row,setup);kt=row['k']//64
        unique={key for groups in program['source_keys'] for group in groups for key in group}
        units={key:tuple((key[0],i) for i in source_granules(key,128)) for key in unique}
        args=dict(source_keys=program['source_keys'],output_counts=program['output_counts'],
            share_inflight_sources=True,source_address_units=units,source_unit_service_ns=params['service_ns']/128,
            input_min_latency_ns=params['read_latency_ns'],input_path_service_ns=price,trace_requests=True,
            **params,**treatment)
        model=shared_stage_events(kt,stage or setup['stages'],len(program['output_counts']),**args)
        if (model['input_demand_work']*16!=program['totals']['source_payload_KiB']
                or model['input_delivery_work']*16!=program['totals']['destination_valid_KiB']
                +program['totals']['destination_fill_KiB']
                or model['input_path_service']['work']!=model['input_demand_work']
                or model['input_path_service']['demands']!=model['input_demand_count']):
            raise ValueError('logical source/destination/request-path ledger changed')
        simulations+=1;simulated_demands+=model['input_demand_count'];source_granules.cache_clear()
        summary=dict(windows=windows(model,program,kt),source_work_16KiB=model['input_source_work'],
            source_generations=model['input_source_count'],address_unit_service=model['address_unit_service'],
            destination_work_16KiB=model['input_delivery_work'],input_path_service=model['input_path_service'],
            output_atoms=sum(program['output_counts'])*8)
        return model,summary,program
    predicted={};zero_checks=[]
    for case,context in original['contexts'].items():
        row,setup=context['row'],context['setup']
        model,summary,program=simulate(row,setup);predicted[case]=summary
        unique={key for groups in program['source_keys'] for group in groups for key in group}
        units={key:tuple((key[0],i) for i in source_granules(key,128)) for key in unique}
        zero=shared_stage_events(row['k']//64,setup['stages'],len(program['output_counts']),
            source_keys=program['source_keys'],output_counts=program['output_counts'],share_inflight_sources=True,
            source_address_units=units,source_unit_service_ns=params['service_ns']/128,**params)
        zero_windows=windows(zero,program,row['k']//64)
        for current,previous in zip(zero_windows,feedback['predicted'][case]['128']['windows']):
            if any(current[k]!=previous[k] for k in ('cta','j','T','coord','L_ns','MAIN_ns','FIRST_ns')):
                raise ValueError('zero-path prior prediction changed')
        zero_checks.append(dict(case=case,windows=len(zero_windows),previous_endpoints_exact=True))
        source_granules.cache_clear()
        print(json.dumps(dict(case=case,complete=True)),flush=True)
    pairs=dict(along_m=('r10_raster_along_m_p0','r10_raster_along_m_p32'),
        along_n=('r10_raster_along_n_p0','r10_raster_along_n_p32'))
    pairs.update({s:(f'r10_source_{s}_p0',f'r10_source_{s}_p32') for s in ('single36','single126','persistent132')})
    comparisons=[]
    for previous in feedback['comparisons']:
        label,j,T=previous['pair'],previous['j'],previous['T'];base,variant=pairs[label]
        matches=paired_windows(predicted[base],predicted[variant],
            lambda w:w['j']==j and w['T']==T and (label in ('along_m','along_n') or w['cta']<36))
        comparisons.append(dict(pair=label,j=j,T=T,observed=previous['observed'],
            predicted={k:statistics.median(w[k] for w in matches)
                for k in ('L_ns_base','L_ns_variant','L_ns_delta','L_ns_ratio')}))
    spatial=[]
    for previous in original['spatial']:
        case,j,T=previous['case'],previous['j'],previous['T']
        values=[w['L_ns'] for w in predicted[case]['windows'] if (w['j'],w['T'])==(j,T)]
        spatial.append(dict(case=case,j=j,T=T,predicted_L_span_ns=max(values)-min(values),
            observed_ns=previous['observed_L_span_ns'],observed_native_cycles=previous['observed_L_span_cycles']))
    interventions=[]
    for pitch in (0,32):
        context=delay['contexts'][f'r10_proddelay_d0_p{pitch}'];row,setup=context['row'],context['setup']
        model,summary,program=simulate(row,setup,diagnostic_producer_offsets_ns=[512. if c==55 else 0. for c in range(132)])
        case=f'r10_raster_along_m_p{pitch}'
        base=next(w for w in predicted[case]['windows'] if (w['cta'],w['j'])==(55,0))
        changed=next(w for w in summary['windows'] if (w['cta'],w['j'])==(55,0))
        delta={k:changed[k]-base[k] for k in ('FIRST_ns','MAIN_ns','L_ns','first_input_ready_ns')}
        delta.update(S_ns=delta['FIRST_ns']-512.,S_plus_L_ns=delta['MAIN_ns']-512.)
        if pitch==0:
            expected_base=max(F(x['slope'])*exact_price+F(x['intercept']) for x in selected['baseline_envelopes']['full0'])
            expected_on=params['read_latency_ns']+1573*price+144.
            if abs(base['FIRST_ns']-float(expected_base))>1e-7 or abs(changed['FIRST_ns']-expected_on)>1e-7:
                raise ValueError('full graph failed the prior independently derived first-prefix expression')
        observed=next(r for r in delay['direct_delay_summaries'] if (r['pitch_bytes'],r['cta'],r['j'])==(pitch,55,0))
        interventions.append(dict(pitch_bytes=pitch,ideal_root_shift_ns=512.,predicted=delta,observed=observed,
            complete_shifted_windows=summary['windows'],input_path_service=summary['input_path_service']))
    stages=[]
    for stage in (4,6):
        context=prefill['contexts'][f'r13_stage_k4096_s{stage}_prefill0'];row,setup=context['row'],context['setup']
        for flag in (False,True):
            model,summary,program=simulate(row,setup,stage,epilogue_defer=2,second_prefill=flag)
            if program['output_counts']!=[2]*132:raise ValueError('R13 two-output domain changed')
            a=model['ctas'][0];entry=a['mma_returns'][0]+params.get('handoff_delay_ns',0.)
            measured=next(r for r in stage_windows['observed'] if r['stage']==stage and r['flag']==flag)
            stages.append(dict(stage=stage,second_prefill=flag,early_ns=a['commits'][80]-a['commits'][65],
                post_ns=a['mma_returns'][1]-a['commits'][80],L_ns=a['main_ends'][1]-a['starts'][64],
                second_entry_to_C16_ns=a['commits'][80]-entry,
                second_entry_to_mma_return_ns=a['mma_returns'][1]-entry,
                prefill=model.get('second_prefill'),observed=measured,
                model_endpoint_ns=model['makespan_ns'],input_path_service=summary['input_path_service']))
    stage_changes=[]
    for stage in (4,6):
        base,changed=(next(r for r in stages if r['stage']==stage and r['second_prefill']==flag) for flag in (False,True))
        observed=next(r['delta'] for r in prefill['on_minus_off'] if r['stages']==stage and r['variant']=='progress')
        stage_changes.append(dict(stage=stage,predicted={k:changed[k]-base[k] for k in
            ('early_ns','post_ns','L_ns','second_entry_to_C16_ns','second_entry_to_mma_return_ns')},
            observed={k:observed[k] for k in ('early_ns','post_ns','prefill_begin_to_commit16_ns','prefill_begin_to_mma_ns')}))
    for name in ('analyze_r10_input_path.py','analyze_r10_shared_path_prefix.py','analyze_r10_granule_overlap.py',
                 'analyze_r10_supply_events.py','supply_model.py','test_r13_input_path.py'):
        sha(Path(__file__).with_name(name))
    report=dict(scope='One complete fixed request-path candidate, selected from old aligned prefix only; post-reveal development.',
        parameters=params,selection=selection,simulated_runs=simulations,simulated_logical_demands=simulated_demands,
        zero_path_previous_window_checks=zero_checks,predicted=predicted,comparisons=comparisons,spatial=spatial,
        root_interventions=interventions,R13_second_prefill=stages,R13_prefill_changes=stage_changes,
        reused_data_inputs_rehashed=len(rehashed),new_numeric_replay=False,input_sha256=inputs,
        limits=['Only one prefix-selected midpoint is tested; failure of this point is not a proof about its whole feasible interval.',
            'Input-path service is common nominal demand work; no B-extra, direction, scale, first or label price is fitted.',
            'Address128 source batches share pending units, while every demand separately pays request path and destination work.',
            'Source issue remains parallel to the request-path FIFO; the queue has no identified hardware attribution.',
            'R13 uses the same tuple/price with actual two-output geometry and no stage-specific calibration.',
            'Predictions remain ns-only; observed native cycles are retained, not used to convert prices.',
            'Ideal root512 is not exact timer-loop/hidden-history reproduction; R10 and R13 observer limits remain.',
            'Old data/observation reports are reused after SHA checks, not newly replayed numerical or GPU evidence.',
            'cfg_a/c timing, old GPU432/V09 numerical migration and independent frozen local holdout remain unsupported.'])
    (output/'B-input-path.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 独立请求路径FIFO：一个前缀选点的完整生命周期检验','',
        f'σ={price:.12f}ns/16KiB为既有前缀必要区间中点；固定后检查全部R10与R13，不回调价格。','',
        '| R10配对 j/T | 预测L响应 | 已有实测中位 |','|---|---:|---:|']
    for r in comparisons:lines.append(f'| {r["pair"]} {r["j"]}/{r["T"]} | '
        f'{100*r["predicted"]["L_ns_ratio"]:.3f}% | {100*r["observed"]["L_ns_ratio"]["median"]:.3f}% |')
    lines+=['','| 起步512ns / B增量 | 预测S / L / S+L变化ns |','|---|---|']
    for r in interventions:
        d=r['predicted'];lines.append(f'| {r["pitch_bytes"]} | {d["S_ns"]:.3f}/{d["L_ns"]:.3f}/{d["S_plus_L_ns"]:.3f} |')
    lines+=['','| R13预等开关 / stage | early / post / entry→C16 / entry→mma变化ns |','|---|---|']
    for r in stage_changes:
        d=r['predicted'];lines.append(f'| {r["stage"]} | {d["early_ns"]:.3f}/{d["post_ns"]:.3f}/'
            f'{d["second_entry_to_C16_ns"]:.3f}/{d["second_entry_to_mma_return_ns"]:.3f} |')
    lines+=['','逐CTA first/later、空间跨度、输入路径账本和原生cycle观察保存在JSON。',
        '这是已揭示数据上的固定开发点；前缀必要可行不是完整规则成立。','']
    (output/'B-input-path.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),price_ns=price,runs=simulations,demands=simulated_demands)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve())
