"""cfg_b同版请求级组合：运行前输入、状态化服务、时间线与plain。"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import cfg_b_composition as aggregate
import cfg_b_frequency as frequency
import cfg_b_phase_frequency as phase
import cfg_b_program
import cfg_b_request_model as request
from mbarrier_service import ConditionSyncService
import supply_model as supply
from tma_stateful_source import ProgramGemmSource


def read(path):
    return json.loads(path.read_text())


def load_profile(repo):
    """读取现用服务及运行前频率版本；不读取待预测条件的观测。"""
    base = Path(repo)/'results/gh200_resource_campaign/access_rules'
    paths = dict(
        reference=base/'20261015-tma-gemm-source-development-001/gemm-phase-state-001/replay.json',
        state=base/'20261015-tma-gemm-source-development-001/domain-layout/replay.json',
        group=base/'20261015-tma-source-sharing-development-001/predictions.json',
        clock_reference=base/'20261015-tma-clock-scaling-heldout-001/plan.json',
        frequency=base/'20261015-cfg-b-time-heldout-002/frequency_model.json',
        phase_frequency=base/'20261015-phase-frequency-development-001/fit-001/model.json',
        calibration=base/'20261009-V09-heldout-job739011/frozen/calibration.json')
    contents = {key:read(path) for key,path in paths.items()}
    return dict(other_parameters=contents['reference']['other_parameters'],
        state_prices=contents['state']['prices'],group_prices=contents['group']['prices_ns'],
        source_frequency_reference_ghz=contents['clock_reference']['frequency_ghz']['0'],
        frequency_model=contents['frequency'],phase_frequency_model=contents['phase_frequency'],
        calibration=contents['calibration'],
        producer_prefix_ns=contents['calibration']['events_ns']['cfg_b']['P0'],
        input_sha256={key:dict(path=str(path.relative_to(repo)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                      for key,path in paths.items()})


def prepare(item, profile):
    program = cfg_b_program.build_program(item['case'],item['setup'])
    case = program['case']
    if case['input_mode'] in profile['phase_frequency_model']['input_modes']:
        call_frequency = frequency.predict(profile['frequency_model'],case)
        phases = phase.predict_phases(profile['phase_frequency_model'],case,program['work'])
        mode = '既有调用频率与阶段频率，原参数外推'
    else:
        call_frequency = aggregate.predict(program,profile['calibration'])['frequency_ghz']
        phases = [call_frequency]*max(map(len,program['work']))
        mode = '经验替代：V09调用频率用于所有阶段'
    ratio = profile['source_frequency_reference_ghz']/call_frequency
    prices = [value*ratio for value in profile['group_prices']]
    state = dict(profile['state_prices'])
    state['hit_ns'] *= ratio
    state['home_ns'] += state['hit_ns']-profile['state_prices']['hit_ns']
    source = ProgramGemmSource(program,item['cta_groups'],prices,**state,trace_stores=False,
        addresses=item['addresses'],b_base_bytes=item['addresses']['B'],
        output_base=item['addresses']['D'],record_diagnostics=False)
    for name in ('A','B','D'):
        source.store(0.,None,item['addresses'][name],item['allocation_bytes'][name])
    parameters = dict(profile['other_parameters'],
        compute_phases_ns=[dict(compute_ns=512/f,commit_lead_ns=256/f) for f in phases],
        reuse_delay_ns=220/call_frequency,compute_ns=512/call_frequency,
        commit_lead_ns=256/call_frequency,destination_service_ns=prices[2])
    clocks = dict(frequency_ghz=call_frequency,phase_frequency_ghz=phases,frequency_mode=mode)
    return program,source,parameters,clocks


def summarize(result, ktiles):
    phases = []
    for j in range(max(len(cta['outputs']) for cta in result['ctas'])):
        actors = [cta for cta in result['ctas'] if len(cta['outputs'])>j]
        phases.append(dict(output=j,active_ctas=len(actors),
            L_ns=statistics.median(cta['main_ends'][j]-cta['starts'][j*ktiles] for cta in actors),
            MAIN_span_ns=max(cta['main_ends'][j] for cta in actors)-min(cta['main_ends'][j] for cta in actors)))
    return phases


def summarize_observations(report):
    phases = []
    for j in range(max(len(cta['observations']) for cta in report['ctas'])):
        endpoints = [cta['observations'][j]['times_ns'] for cta in report['ctas']
                     if len(cta['observations'])>j]
        phases.append(dict(output=j,active_ctas=len(endpoints),
            L_ns=statistics.median(times[1]-times[0] for times in endpoints),
            MAIN_span_ns=max(times[1] for times in endpoints)-min(times[1] for times in endpoints)))
    return phases


def predict(item, profile, protocol, *, detailed_ctas=(0,), report_demands=True):
    program,source,parameters,clocks = prepare(item,profile)
    prefix = profile['producer_prefix_ns']
    counts = list(map(len,program['work']))
    base,history = 100000.,[]
    for launch in range(protocol['calls']):
        source.begin(0,base)
        if launch == protocol['calls']-1:
            report = request.simulate(program,parameters,producer_prefix_ns=prefix,
                input_source_service=source.input,output_source_observer=source.output,
                input_sync_service=ConditionSyncService(program['stages'],0.,0.),
                register_quota=report_demands,ordinary_lsu=report_demands,sm_activity=report_demands,
                detailed_ctas=detailed_ctas)
            internal_window = report['internal_window_ns']
        else:
            result = supply.shared_stage_events(program['ktiles'],program['stages'],len(counts),
                output_counts=counts,input_source_service=source.input,
                output_source_observer=source.output,**parameters)
            internal_window = result['makespan_ns']
        history.append(dict(launch=launch,entry_base_ns=base+launch*prefix,
            service_origin_ns=base+(launch+1)*prefix,internal_window_ns=internal_window,
            window_ns=internal_window+prefix,
            phases=summarize(result,program['ktiles']) if launch<protocol['calls']-1 else summarize_observations(report)))
        base += internal_window+protocol['inter_call_gap_ns']
    projection = aggregate.V09Observation(profile['calibration']).project(
        dict(window_us=report['consumer_span_ns']/1000))
    report.update(mode='request_system',qualification='开发组合：现用假设与经验替代；原反例及独立C2/C3缺口保留',
        input=item,protocol=protocol,service_profile_sha256=hashlib.sha256(
            json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        history=history,clocks=clocks,**projection,
        source_state=dict(version=source.version,hbm_lines=source.hbm_lines,local_lines=source.local_lines,
            remote_lines=source.remote_lines,pending_lines=source.pending_lines,output_bytes=source.output_bytes),
        clock='运行前调用与阶段频率换算既有计算／提交服务，其余使用声明的ns价格',
        endpoint='共同入口至最后consumer exit，经V09 F+κW映射为plain')
    report['modules'].update(frequency=dict(qualification='经验替代',**clocks),
        observation=dict(version=aggregate.V09Observation.version,qualification='经验替代'),
        zero_cost_sync='phase/计数检查；返回价格沿原零额外成本假设')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--case')
    selection.add_argument('--input',type=Path,help='保存的input/profile/protocol；独立预测时无需原实验目录')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.input:
        saved = read(args.input)
        inputs,profile,protocol = (saved[key] for key in ('input','profile','protocol'))
    else:
        if args.repo is None:
            parser.error('--case需要--repo')
        repo = args.repo.resolve()
        plan = read(repo/'results/gh200_resource_campaign/access_rules/20261015-cfg-b-full-matrix-001/plan.json')
        item = next(row for row in plan['rows'] if row['case']['id']==args.case)
        inputs = {key:item[key] for key in ('case','setup','allocation_bytes','addresses','cta_groups','placement')}
        profile = load_profile(repo)
        protocol = {key:plan[key] for key in ('calls','inter_call_gap_ns')}
    report = predict(inputs,profile,protocol,detailed_ctas=range(len(inputs['cta_groups'])))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'inputs.json').write_text(json.dumps(dict(input=inputs,profile=profile,protocol=protocol),indent=2)+'\n')
    with gzip.GzipFile(filename='',mode='wb',fileobj=(args.output/'timeline.json.gz').open('wb'),mtime=0) as stream:
        stream.write(json.dumps(report,separators=(',',':'),allow_nan=False).encode())
    summary = {key:report[key] for key in ('case','stages','qualification','plain_us','consumer_span_ns',
        'program_end_ns','clocks','accounting','dependency_edges_checked','source_state','history')}
    summary['timeline_sha256'] = hashlib.sha256((args.output/'timeline.json.gz').read_bytes()).hexdigest()
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f'{inputs["case"].get("id","cfg_b")}: {report["plain_us"]:.9f} us; {report["accounting"]["input_delivery_count"]} requests; GPU=0')


if __name__ == '__main__':
    main()
