#!/usr/bin/env python3
"""Unfitted address-unit service coupled to the complete cfg_b stage graph."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from analyze_r10_granule_overlap import source_granules
from analyze_r10_supply_events import GPU,paired_windows,simulate
from supply_model import cfg_b_source_program,shared_stage_events


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False)
    inputs={}
    def sha(path):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[str(path.resolve())]=digest
        return digest
    old_path=root/'20261010-R10-raster-pitch/reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json'
    sha(old_path)
    old=json.loads(old_path.read_text())
    # Reuse the already-replayed observations, rechecking their data identities.
    # The historical model SHA is retained in that immutable report; its code
    # is deliberately different from this new candidate, not a data mutation.
    for name,digest in old['input_sha256'].items():
        path=Path(name)
        if path.is_relative_to(root) and sha(path)!=digest:
            raise ValueError('reused observation input changed: '+name)
    if old['raw_processes_verified']!=100:
        raise ValueError('unexpected prior numeric replay scope')
    params=old['parameters']
    predicted={};baseline_max_delta=[];legacy_unchanged=[]
    for case,context in old['contexts'].items():
        row,setup=context['row'],context['setup']
        if setup['gpu_uuid']!=GPU or setup['stages']!=6:
            raise ValueError('no cross-card or cross-stage price transfer')
        program=cfg_b_source_program(row,setup)
        kt=row['k']//64
        args=dict(source_keys=program['source_keys'],output_counts=program['output_counts'],
            share_inflight_sources=True,**params)
        legacy_prediction=simulate(row,setup,program,params,
            0. if row['ldb']==3072 else old['coverage_price_ns_per_KiB'])
        if legacy_prediction!=old['predicted'][case]:
            raise ValueError('default old model prediction changed: '+case)
        legacy_unchanged.append(case)
        predictions={}
        unique={key for groups in program['source_keys'] for group in groups for key in group}
        for width in (32,128):
            units={key:tuple((key[0],i) for i in source_granules(key,width)) for key in unique}
            model=shared_stage_events(kt,6,len(program['output_counts']),source_address_units=units,
                source_unit_service_ns=params['service_ns']*width/16384,**args)
            totals=program['totals']
            if (model['input_demand_work']*16!=totals['source_payload_KiB']
                    or model['input_delivery_work']*16!=totals['destination_valid_KiB']+totals['destination_fill_KiB']
                    or sum(program['output_counts'])*8!=totals['output_nominal_subtiles']):
                raise ValueError('logical request/destination/D ledger changed')
            windows=[]
            for cta,actor in enumerate(model['ctas']):
                for j,coord in enumerate(program['cta_output_coords'][cta]):
                    later=range(j*kt+1,(j+1)*kt)
                    windows.append(dict(cta=cta,j=j,T=program['output_counts'][cta],coord=coord,
                        L_ns=actor['main_ends'][j]-actor['starts'][j*kt],MAIN_ns=actor['main_ends'][j],
                        FIRST_ns=actor['starts'][j*kt],first_input_ready_ns=actor['ready'][j*kt],
                        post_FIRST_ready_wait_ns=sum(actor['starts'][i]-actor['wait1_return'][i-1]
                            -params.get('loop_advance_ns',0.) for i in later),
                        post_FIRST_TC_idle_ns=sum(max(0.,actor['starts'][i]-actor['service_frontier'][i-1]) for i in later),
                        minimum_TC_frontier_slack_ns=min(actor['service_frontier'][i-1]-actor['ready'][i] for i in later),
                        minimum_later_ready_slack_ns=min(actor['wait1_return'][i-1]
                            +params.get('loop_advance_ns',0.)-actor['ready'][i] for i in later)))
            predictions[str(width)]=dict(windows=windows,address_units=model['address_unit_service'],
                source_busy_ns=model['source_busy_ns'],destination_work=model['input_delivery_work'],
                logical_source_generation_work=model['input_source_work'],
                source_box_generations=model['input_source_count'],output_atoms=sum(program['output_counts'])*8)
        if row['ldb']==3072:
            legacy=shared_stage_events(kt,6,len(program['output_counts']),**args)
            delta=max(abs(a['main_ends'][j]-predictions[str(width)]['windows'][sum(program['output_counts'][:c])+j]['MAIN_ns'])
                for c,a in enumerate(legacy['ctas']) for j in range(program['output_counts'][c]) for width in (32,128))
            if delta>1e-9:raise ValueError('aligned candidate changed legacy box timing')
            baseline_max_delta.append(dict(case=case,maximum_MAIN_delta_ns=delta))
        predicted[case]=predictions
        source_granules.cache_clear()
        print(json.dumps(dict(case=case,finished=True)),flush=True)
    pairs=dict(along_m=('r10_raster_along_m_p0','r10_raster_along_m_p32'),
        along_n=('r10_raster_along_n_p0','r10_raster_along_n_p32'))
    pairs.update({s:(f'r10_source_{s}_p0',f'r10_source_{s}_p32') for s in ('single36','single126','persistent132')})
    comparisons=[]
    for previous in old['comparisons']:
        label,j,T=previous['pair'],previous['j'],previous['T']
        base,variant=pairs[label]
        result=dict(pair=label,j=j,T=T,observed=previous['observed'],models={})
        for width in (32,128):
            windows=paired_windows(predicted[base][str(width)],predicted[variant][str(width)],
                lambda w:w['j']==j and w['T']==T and (label in ('along_m','along_n') or w['cta']<36))
            result['models'][str(width)]={k:statistics.median(w[k] for w in windows)
                for k in ('L_ns_base','L_ns_variant','L_ns_ratio')}
            for arm,case in (('base',base),('variant',variant)):
                selected=[w for w in predicted[case][str(width)]['windows']
                    if (w['j'],w['T'])==(j,T) and (label in ('along_m','along_n') or w['cta']<36)]
                result['models'][str(width)][arm+'_maximum_post_FIRST_ready_wait_ns']=max(w['post_FIRST_ready_wait_ns'] for w in selected)
                result['models'][str(width)][arm+'_minimum_later_ready_slack_ns']=min(w['minimum_later_ready_slack_ns'] for w in selected)
                result['models'][str(width)][arm+'_maximum_post_FIRST_TC_idle_ns']=max(w['post_FIRST_TC_idle_ns'] for w in selected)
                result['models'][str(width)][arm+'_minimum_TC_frontier_slack_ns']=min(w['minimum_TC_frontier_slack_ns'] for w in selected)
        comparisons.append(result)
    spatial=[]
    for previous in old['spatial']:
        case,j,T=previous['case'],previous['j'],previous['T']
        result=dict(case=case,j=j,T=T,observed=previous['observed_L_span_ns'],models={})
        for width in (32,128):
            windows=[w for w in predicted[case][str(width)]['windows'] if (w['j'],w['T'])==(j,T)]
            result['models'][str(width)]=max(w['L_ns'] for w in windows)-min(w['L_ns'] for w in windows)
        spatial.append(result)
    for name in ('analyze_r10_address_feedback.py','analyze_r10_granule_overlap.py',
            'analyze_r10_supply_events.py','supply_model.py'):
        sha(Path(__file__).with_name(name))
    report=dict(scope='Post-reveal unfitted batch address-unit hypothesis; no accepted rule or holdout.',
        parameters=params,unit_prices_ns={str(w):params['service_ns']*w/16384 for w in (32,128)},
        aligned_legacy_checks=baseline_max_delta,exact_old_predictions=legacy_unchanged,
        predicted=predicted,comparisons=comparisons,spatial=spatial,
        reused_observation_report=str(old_path),previous_numeric_replay_processes=100,
        input_sha256=inputs,limits=[
            'No fit, B-extra price, per-direction/scale constant, measured start offset or target clock input.',
            'Every unit keeps its owner batch ready; box ready is the max of own and subscribed units.',
            'Units expire at ready and work reduction feeds back into ready, subscriptions, stages, consumption and output.',
            'Address coverage is hypothetical service work, not physical traffic; whole requests/destination/D remain.',
            'Legacy exploratory ns parameters are fixed development inputs, not independent calibration.',
            'Only the ten same-card stage6 R10 cases are simulated; old/V09 GPU432 numerical transfer remains unsupported.',
            'Prior observation summaries are reused after raw/metadata SHA checks; numeric replay was already completed there.',
            'Batch completion is a specific hypothesis; per-unit streaming completion and resident caching are not modeled.'])
    (output/'B-address-feedback.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 地址单元服务反馈到有限stage：未拟合开发候选','',
        '单元服务价仅由旧对齐基线换算，未沿用失败的B-extra覆盖价。请求、目的投递和D账目不减；ready、在途订阅和槽释放重新求解。','',
        '| 配对 j/T | 32B预测 | 128B预测 | 实测中位[范围] |','|---|---:|---:|---:|']
    for r in comparisons:
        o=r['observed']['L_ns_ratio']
        lines.append(f'| {r["pair"]} {r["j"]}/{r["T"]} | {100*r["models"]["32"]["L_ns_ratio"]:.3f}% | '
            f'{100*r["models"]["128"]["L_ns_ratio"]:.3f}% | {100*o["median"]:.3f}% [{100*o["minimum"]:.3f},{100*o["maximum"]:.3f}] |')
    lines+=['','全部逐CTA窗口、j/T空间跨度、源box代次/单元服务与目的/D账目在JSON。',
        '这是既有数据上的明确批完成候选；不能将任意单个响应接近实测作为独立留出通过。','']
    (output/'B-address-feedback.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),cases=len(predicted),comparison_groups=len(comparisons))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve())
