#!/usr/bin/env python3
"""Audit a source/destination latency relabeling, without rescanning cutoffs."""
import argparse
import hashlib
import json
from pathlib import Path

from supply_model import cfg_b_source_program,shared_stage_events


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False)
    inputs={}
    def read(path):
        inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    dev=root/'20261009-V09-heldout-job739011/reanalysis/candidate-development-v1'
    params=read(dev/'B-source-candidate-examples.json')['examples']['shared']['parameters']
    prior=read(dev/'B-source-join-boundary.json')
    run=root/'20261010-R13-second-prefill'
    rows=read(run/'cases.json');setups={v['case']:v['setup'] for v in read(run/'static_setup.json')}
    row=next(r for r in rows if r.get('set')!='reference');setup=setups[row['id']]
    program=cfg_b_source_program(row,setup)
    if program['output_counts']!=[2]*132 or row['k']!=4096:
        raise ValueError('original cfg_b two-output source geometry changed')
    fields=('ctas','makespan_ns','shared_busy_ns','last_service_end_ns','request_count',
        'input_demand_count','input_source_count','input_merged_count','input_delivery_count',
        'input_demand_work','input_source_work','input_merged_work','input_delivery_work',
        'source_busy_ns','destination_busy_ns','destination_busy_by_cta_ns',
        'local_epilogue_blocked_ns','local_epilogue_intervals','scope')
    results=[]
    for stage in (4,6):
        for name,treatment in (('ordinary',{}),('root512',dict(diagnostic_producer_offsets_ns=[512. if c==55 else 0. for c in range(132)])),
                ('all_E0_deferred',dict(epilogue_defer=2)),('prefill_after_defer',dict(epilogue_defer=2,second_prefill=True))):
            args=dict(source_keys=program['source_keys'],output_counts=program['output_counts'],
                share_inflight_sources=True,trace_requests=True,**treatment)
            old=shared_stage_events(64,stage,132,source_join_boundary='service_end',**params,**args)
            moved_params=dict(params,read_latency_ns=0.,destination_latency_ns=params['read_latency_ns'])
            moved=shared_stage_events(64,stage,132,**moved_params,**args)
            compared={f:old[f] for f in fields}
            for f in fields:
                if old[f]!=moved[f]:raise ValueError('endpoint/actor state differs: '+f)
            if old.get('second_prefill')!=moved.get('second_prefill'):raise ValueError('prefill bounds differ')
            for a,b in zip(old['input_demands'],moved['input_demands']):
                if (any(a[f]!=b[f] for f in ('demand_id','source_id','merged','cta','kind','index','arrival_ns','work'))
                        or a['source_ready_ns']!=b['destination_arrival_ns']):
                    raise ValueError('source generations or destination arrivals differ')
            if len(old['input_demands'])!=len(moved['input_demands']):raise ValueError('demand counts differ')
            for a,b in zip(old['sources'],moved['sources']):
                if (any(a[f]!=b[f] for f in ('source_id','source_key','arrival_ns','start_ns','service_end_ns','join_end_ns','work'))
                        or a['ready_ns']!=b['ready_ns']+params['read_latency_ns']):
                    raise ValueError('source cutoff or completion relabeling differs')
            for a,b in zip(old['deliveries'],moved['deliveries']):
                if any(a[f]!=b[f] for f in ('demand_id','source_id','cta','kind','index','start_ns','service_end_ns','ready_ns','work')):
                    raise ValueError('destination timeline differs')
            a=old['ctas'][0]
            results.append(dict(stages=stage,treatment=name,matched_public_fields=len(fields),
                complete_actor_and_endpoint_sha256=hashlib.sha256(json.dumps(compared,separators=(',',':')).encode()).hexdigest(),
                demands_matched=len(old['input_demands']),source_generations_matched=len(old['sources']),
                destination_timelines_matched=len(old['deliveries']),source_work=old['input_source_work'],
                early_ns=a['commits'][80]-a['commits'][65],post_ns=a['mma_returns'][1]-a['commits'][80],
                model_endpoint_ns=old['makespan_ns']))
    for name in ('analyze_r13_latency_relocation.py','supply_model.py'):
        path=Path(__file__).with_name(name)
        inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(scope='Semantic equivalence audit of an already tested cutoff; no new family, fit, data or validation.',
        parameters=params,results=results,previous_boundary_calibration=prior['calibration'],input_sha256=inputs,
        theorem=dict(assumptions=['Whole-box sharing; a common nonnegative read latency L for all input atoms.',
                'Old source completion is server_end+L, subscription closes strictly at server_end.',
                'New source completion/subscription end is server_end, each demand reaches destination FIFO after L.',
                'Same FIFO/source costs, roots, stages, consumer/output graph and deterministic equal-time order.'],
            induction='Given matching actor states, input arrivals and server generations match because both close at server_end. '
                'Each demand then reaches the destination FIFO at the identical server_end+L. Deliveries/consumer/releases/output '
                'match, preserving subsequent actor states. Expired pending-key cleanup cannot reopen admission.',
            unobservable_from_these_ports='Issue, destination/full, consumer/output and requested root-delay responses remain equal. '
                'The internal source_ready label changes; it is not independently observed by these ports.'),
        decision='Full relocation is an alternative parameterization of the prior service_end cutoff. '
            'Inherit that fixed-family development failure; do not refit, rename it a new mechanism or request a distinguishing GPU pair.',
        limits=['Eight synthetic model scenarios reuse original geometry/tuple; no measured offset is a prediction input.',
            'root512 is an ideal prescribed root shift, not a claim about observed timer overshoot or kernel intervention equivalence.',
            'This theorem does not identify a physical source/destination latency split or cover all caching/partial-unit families.',
            'The old cutoff search was bounded and is not a proof that every possible source family is infeasible.',
            'Default destination latency remains zero; old models, frozen scores and observer-gate failures are unchanged.'])
    (output/'B-latency-relocation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 源/目的延迟重标记：继承已有截止分支的失败','',
        '整box模型中，将共同读延迟L全部移到每个目的请求，与已有service_end截止分支完全等价。',
        '两者都在server_end严格关闭订阅，每个需求都在server_end+L进入同一目的FIFO。',
        '源代次、投递、消费、释放、输出和后续请求按归纳保持；内部source_ready标签不同。','',
        '| stage / 合成条件 | 匹配需求 / 源代次 | early / post ns |','|---|---:|---:|']
    for r in results:
        lines.append(f'| {r["stages"]}/{r["treatment"]} | {r["demands_matched"]}/{r["source_generations_matched"]} | {r["early_ns"]:.3f}/{r["post_ns"]:.3f} |')
    lines+=['','每对模型的19项完整公开字段、全actor状态、源订阅身份和目的时间线精确相等。',
        '根512ns为合成输入，不使用实测起步戳或声称精确模拟计时循环。',
        '没有重扫截止、重拟价格或重采GPU。原service_end候选的开发失败仍适用，不作为新机制重新验收。',
        '目的后程端口仅使依赖位置明确；其价格和物理分配尚未识别，旧观察资格未提高。','']
    (output/'B-latency-relocation.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),scenarios=len(results),matched_demands=sum(r['demands_matched'] for r in results))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve())
