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
        reference=base/'Q02-tma-request/gemm-source-reuse/20261015-tma-gemm-source-development-001/gemm-phase-state-001/replay.json',
        state=base/'Q02-tma-request/gemm-source-reuse/20261015-tma-gemm-source-development-001/domain-layout/replay.json',
        group=base/'Q02-tma-request/gemm-source-reuse/20261015-tma-source-sharing-development-001/predictions.json',
        clock_reference=base/'Q09-frequency/frequency-mechanism/20261015-tma-clock-scaling-heldout-001/plan.json',
        frequency=base/'Q15-full-time/V09-component-validation/20261015-cfg-b-time-heldout-002/frequency_model.json',
        phase_frequency=base/'Q09-frequency/frequency-prediction/20261015-phase-frequency-development-001/fit-001/model.json',
        calibration=base/'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/frozen/calibration.json')
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
    source_type, source_parameters = ProgramGemmSource, {}
    if 'input_network' in profile:
        from cfg_b_input_network import CfgBInputNetwork, AddressRoutedInputNetwork, ReadSharedInputNetwork, JointLocalInputNetwork, PortOrderedInputNetwork, FormationFirstInputNetwork
        source_type = {'address_banks': AddressRoutedInputNetwork,
                       'address_banks_inflight_merge': ReadSharedInputNetwork,
                       'address_banks_local_joint': JointLocalInputNetwork,
                       'address_banks_port_lookup': PortOrderedInputNetwork,
                       'address_banks_formation_first': FormationFirstInputNetwork}.get(
                           profile['input_network'].get('routing'), CfgBInputNetwork)
        source_parameters = dict(network=profile['input_network'], phase_frequency_ghz=phases)
    source = source_type(program,item['cta_groups'],prices,**state,**source_parameters,trace_stores=False,
        addresses=item['addresses'],b_base_bytes=item['addresses']['B'],
        output_base=item['addresses']['D'],record_diagnostics=False)
    for name in ('A','B','D'):
        source.store(0.,None,item['addresses'][name],item['allocation_bytes'][name])
    parameters = dict(profile['other_parameters'],
        compute_phases_ns=[dict(compute_ns=512/f,commit_lead_ns=256/f) for f in phases],
        reuse_delay_ns=220/call_frequency,compute_ns=512/call_frequency,
        commit_lead_ns=256/call_frequency,destination_service_ns=prices[2])
    if 'compute_completion_lag_cycles' in profile:
        lag = profile['compute_completion_lag_cycles']
        parameters['completion_lag_ns'] = lag/call_frequency
        for costs,f in zip(parameters['compute_phases_ns'],phases):
            costs['completion_lag_ns'] = lag/f
    if 'local_chain_transfer_cycles' in profile:
        local = profile['local_chain_transfer_cycles']
        parameters.update(compute_ns=local['compute_cycles']/call_frequency,
            commit_lead_ns=local['commit_lead_cycles']/call_frequency,
            completion_lag_ns=local['completion_lag_cycles']/call_frequency,
            reuse_delay_ns=local['reuse_delay_cycles']/call_frequency,
            input_release_delay_ns=local['release_visibility_cycles']/call_frequency,
            loop_advance_ns=local['consumer_advance_cycles']/call_frequency)
        for costs,f in zip(parameters['compute_phases_ns'],phases):
            costs.update(compute_ns=local['compute_cycles']/f,
                commit_lead_ns=local['commit_lead_cycles']/f,
                completion_lag_ns=local['completion_lag_cycles']/f,
                input_release_delay_ns=local['release_visibility_cycles']/f,
                loop_advance_ns=local['consumer_advance_cycles']/f)
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


def predict(item, profile, protocol, *, detailed_ctas=(0,), report_demands=True, critical_path=False):
    program,source,parameters,clocks = prepare(item,profile)
    prefix = profile['producer_prefix_ns']
    counts = list(map(len,program['work']))
    warm_program = program
    if any(key in item['setup'] for key in ('fold_a_source', 'store_target_ctas', 'private_source')):
        warm_setup = dict(item['setup'],
            fold_a_source=item['setup'].get('warmup_fold_a_source', 0),
            private_source=item['setup'].get('warmup_private_source', item['setup'].get('private_source', 0)),
            store_target_ctas=item['setup'].get('warmup_store_ctas', len(counts)))
        warm_program = cfg_b_program.build_program(item['case'], warm_setup)
    base,history = 100000.,[]
    for launch in range(protocol['calls']):
        active_program = program if launch == protocol['calls']-1 else warm_program
        if warm_program is not program:
            source.use_program(active_program)
        active_stages, execution_parameters = request.execution_inputs(active_program, parameters)
        source.begin(0,base)
        callbacks = (source.callbacks() if 'input_network' in profile else
                     dict(input_source_service=source.input, output_source_observer=source.output))
        if launch == protocol['calls']-1:
            report = request.simulate(program,parameters,producer_prefix_ns=prefix,
                **callbacks,
                input_sync_service=ConditionSyncService(active_stages,0.,0.),
                register_quota=report_demands,ordinary_lsu=report_demands,sm_activity=report_demands,
                detailed_ctas=detailed_ctas,critical_path=critical_path)
            internal_window = report['internal_window_ns']
        else:
            result = supply.shared_stage_events(active_program['ktiles'],active_stages,len(counts),
                output_counts=counts,**callbacks,**execution_parameters)
            internal_window = result['makespan_ns']
        history.append(dict(launch=launch,entry_base_ns=base+launch*prefix,
            service_origin_ns=base+(launch+1)*prefix,internal_window_ns=internal_window,
            window_ns=internal_window+prefix,
            phases=summarize(result,program['ktiles']) if launch<protocol['calls']-1 else summarize_observations(report)))
        if warm_program is not program:
            history[-1]['output_tma_count'] = (report['accounting'].get('output_tma_count',
                report['accounting']['request_count']-report['accounting']['input_source_count'])
                if launch == protocol['calls']-1 else result['request_count']-result['input_source_count'])
            history[-1]['output_bytes'] = source.output_bytes
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
    if 'input_network' in profile:
        report['source_state']['request_reply_network'] = source.network_state()
        report['modules']['input'] = ('假设：cache数据就绪→在途同址共享分区读取→GPC独立应答→逐CTA投递'
            if profile['input_network'].get('routing') == 'address_banks_inflight_merge' else
            '假设：cache数据就绪→地址分区读取→GPC共享PS2应答→逐CTA投递'
            if profile['input_network'].get('routing') == 'address_banks' else
            '假设：cache数据就绪→GPC共享PS2应答→逐CTA投递；两输入工作价格共同估计')
        if profile['input_network'].get('routing') == 'address_banks_local_joint':
            report['modules']['input'] = '假设：共享读取／应答后，目的投递与WGMMA操作数共同使用本地服务'
        elif profile['input_network'].get('routing') == 'address_banks_port_lookup':
            report['modules']['input'] = '假设：读取分区受理→cache查询／数据返回→GPC独立应答→逐CTA投递'
        elif profile['input_network'].get('routing') == 'address_banks_formation_first':
            report['modules']['input'] = '假设：每GPC共享请求形成→cache查询／数据返回→共享分区读取→逐CTA投递'
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
        plan = read(repo/'results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-full-matrix-001/plan.json')
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
