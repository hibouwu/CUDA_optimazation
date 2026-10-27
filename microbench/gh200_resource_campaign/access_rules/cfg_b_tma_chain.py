#!/usr/bin/env python3
"""cfg_b 三笔TMA、真实计算与槽释放短链的有限CPU准备。"""
import argparse
import json
import math
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from cfg_b_compute_chain import output_coordinates, SEEDS
from cfg_b_handoff_chain import reference as ring_reference
from l2_saturation import command_record, digest, read_json, require, save_json
from mbarrier_service import BarrierPhase, ConditionSyncService
from supply_model import shared_stage_events

HERE = Path(__file__).resolve().parent
SCHEMA = 'gh200-cfg-b-tma-chain-plan/1'


def cases():
    return [dict(id=f's{stages}_k{ktiles}_{"trace" if trace else "plain"}', stages=stages,
        ktiles=ktiles, trace=trace, split='heldout' if ktiles == 32 else 'dev',
        wgmma_count=8*ktiles, tma_requests=3*ktiles,
        tma_bytes=32768*ktiles, work_flop=2*128*128*64*ktiles)
        for stages, ktiles in ((6,4), (6,13), (6,32), (4,13)) for trace in (False,True)]


def reference(ktiles, stages, seed, profile='ring'):
    if profile == 'ring':
        return ring_reference(ktiles, stages, seed)
    require(profile == 'generation', '输入profile不符')
    row = np.arange(128, dtype=np.int64)[:,None]
    inner = np.arange(64, dtype=np.int64)[None,:]
    a_base = 1+(7*row+13*inner+3*(seed % 8)) % 8
    b_base = 1+(11*row+5*inner+5*(seed % 8)) % 8
    total = np.zeros((128,128), dtype=np.int64)
    for generation_input in range(ktiles):
        total += (a_base+generation_input) @ (b_base+generation_input).T
    rows, columns = output_coordinates()
    return (total[rows,columns]/1024).astype(np.float32)


def expected_descriptor(source_tiles, source_batches=1):
    source_k = 64*source_tiles
    common = dict(rank=3, encoding_type='UINT16', numeric_type='FP16', swizzle_bytes=128,
        element_strides=[1,1,1], interleave='NONE', l2_promotion_bytes=128, oob_fill='NONE')
    return dict(A=dict(**common, global_dim=[source_k,128,source_batches],
                      byte_strides=[2*source_k,256*source_k], box=[64,128,1]),
                B=dict(**common, global_dim=[128,source_k,source_batches],
                      byte_strides=[256,256*source_k], box=[64,64,1]))


def requests(tile, stages, source_tiles, profile='ring', *, source_batch=0, source_batches=1):
    source = tile % stages if profile == 'ring' else tile
    require(profile in ('ring','generation') and 0 <= source < source_tiles, '源tile越界')
    require(0 <= source_batch < source_batches, '源batch越界')
    batch_offset = source_batch * 16384 * source_tiles
    slot = tile % stages
    common = dict(tile=tile, slot=slot, generation=tile//stages,
                  source_k_tile=source, transaction_group=tile)
    return [dict(**common, operand='A', bytes=16384, source_coord=[64*source,0,source_batch],
        source_byte_offset=batch_offset+128*source, source_row_stride_bytes=128*source_tiles,
        destination_byte_offset=16384*slot),
        dict(**common, operand='B0', bytes=8192, source_coord=[0,64*source,source_batch],
        source_byte_offset=batch_offset+16384*source, source_row_stride_bytes=256,
        destination_byte_offset=98304+16384*slot),
        dict(**common, operand='B1', bytes=8192, source_coord=[64,64*source,source_batch],
        source_byte_offset=batch_offset+16384*source+128, source_row_stride_bytes=256,
        destination_byte_offset=98304+16384*slot+8192)]


def source_offsets(request, source_tiles):
    rows, columns = (128,64) if request['operand'] == 'A' else (64,64)
    return (request['source_byte_offset']+
        np.arange(rows, dtype=np.int64)[:,None]*request['source_row_stride_bytes']+
        2*np.arange(columns, dtype=np.int64)[None,:])


def lifecycle(ktiles, stages):
    """三笔事务分别完成；只保存程序前驱，不填写内部服务时间。"""
    full = [BarrierPhase(1) for _ in range(stages)]
    empty = [BarrierPhase(128) for _ in range(stages)]
    nodes, release, completion = [], {}, {}
    producer, consumer = None, None

    def node(kind, tile, predecessors=(), **extra):
        item = dict(id=len(nodes), kind=kind, tile=tile, slot=tile % stages,
            generation=tile//stages, full_wait_phase=(tile//stages) % 2,
            empty_wait_phase=(tile//stages+1) % 2, predecessors=list(predecessors), **extra)
        nodes.append(item)
        return item['id']

    for tile in range(ktiles):
        slot, generation = tile % stages, tile//stages
        require(empty[slot].wait_complete((generation+1) % 2), 'producer过早取得槽')
        predecessors = [] if producer is None else [producer]
        if tile >= stages:
            predecessors.append(release[tile-stages])
        acquire = node('producer_acquire', tile, predecessors)
        expect = node('expect_tx', tile, [acquire], bytes=32768, arrivals=1)
        full[slot].arrive_and_expect_tx(32768)
        issues, completed = [], []
        previous = expect
        for request in requests(tile, stages, stages):
            issued = node('tma_issue', tile, [previous], operand=request['operand'], bytes=request['bytes'],
                          source_k_tile=request['source_k_tile'], in_timed_window=True)
            issues.append(issued)
            completed.append(node('tma_complete', tile, [issued], operand=request['operand'], bytes=request['bytes']))
            full[slot].complete_tx(request['bytes'])
            previous = issued
        producer = issues[-1]
        require(full[slot].wait_complete(generation % 2), '三笔事务未完成')
        ready = node('full_ready', tile, [expect,*completed])
        predecessors = [ready] if consumer is None else [ready,consumer]
        waited = node('consumer_full_wait', tile, predecessors)
        issued = node('eight_wgmma', tile, [waited], instructions=8)
        committed = node('commit', tile, [issued])
        completion[tile] = node('group_complete', tile, [committed])
        consumer = committed
        if tile:
            wait = node('wait1', tile, [committed,completion[tile-1]])
            for _ in range(128):
                empty[(tile-1) % stages].arrive()
            release[tile-1] = node('empty_release', tile-1, [wait], arrivals=128)
            consumer = release[tile-1]
    tail = node('tail_wait0', ktiles-1, [*completion.values(),consumer])
    for _ in range(128):
        empty[(ktiles-1) % stages].arrive()
    release[ktiles-1] = node('empty_release', ktiles-1, [tail], arrivals=128)
    return dict(ktiles=ktiles, stages=stages, nodes=nodes,
        qualification='无硬件时间的程序／事务／槽代次见证',
        full=[item.snapshot() for item in full], empty=[item.snapshot() for item in empty])


def _input_timeline(actor, stages, release_visibility_ns):
    timeline = []
    for tile, window in enumerate(actor['full_wait_windows']):
        timeline.append(dict(tile=tile, slot=tile % stages, generation=tile//stages,
            full_wait_phase=(tile//stages) % 2, empty_wait_phase=(tile//stages+1) % 2,
            request_A_issue_ns=actor['issued_a'][tile], full_ready_ns=actor['ready'][tile],
            full_enter_ns=window['enter_ns'], full_return_ns=window['wait_return_ns'],
            compute_begin_ns=actor['starts'][tile], commit_ns=actor['commits'][tile],
            compute_complete_ns=actor['completed'][tile],
            wait1_return_ns=None if tile == 0 else actor['wait1_return'][tile],
            release_ns=actor['released'][tile],
            reuse_predecessor_ns=None if tile < stages else actor['released'][tile-stages]))
        if release_visibility_ns:
            timeline[-1].update(release_emit_ns=actor['release_emitted'][tile],
                                empty_complete_ns=actor['released'][tile])
        if 'producer_acquired' in actor:
            timeline[-1]['producer_acquire_ns']=actor['producer_acquired'][tile]
    return timeline


def timed_lifecycle(ktiles, stages, *, request_service, compute_ns, commit_lead_ns,
                    completion_lag_ns=0., producer_gap_ns=0., operand_gap_ns=0.,
                    loop_advance_ns=0., fast_return_ns=0., blocked_return_ns=0.,
                    source_tiles=None, source_profile='ring', release_visibility_ns=0.,
                    consumer_entry_delay_ns=0., input_destination_service=None,
                    input_source_async=None, input_source_bind_events=None,
                    source_batch=0, source_batches=1, producer_control_ns=(0.,0.,0.)):
    """D的计时内三请求接到既有引擎；所有服务时间由调用方显式提供。"""
    observed_requests = []
    source_tiles = stages if source_tiles is None else source_tiles

    def source(t, cta, index, atom, weight):
        request = requests(index, stages, source_tiles, source_profile,
                           source_batch=source_batch, source_batches=source_batches)[atom]
        served = request_service(t, request)
        observed_requests.append(dict(request, arrival_ns=t,
            source_ready_ns=served['source_ready'], issue_return_ns=served.get('issue_return_ns')))
        return served

    sync = ConditionSyncService(stages, fast_return_ns, blocked_return_ns)
    result = shared_stage_events(ktiles, stages, 1, service_ns=0., read_latency_ns=0.,
        compute_ns=compute_ns, commit_lead_ns=commit_lead_ns,
        completion_lag_ns=completion_lag_ns, producer_gap_ns=producer_gap_ns,
        operand_gap_ns=operand_gap_ns, loop_advance_ns=loop_advance_ns,
        producer_control_ns=producer_control_ns,
        input_source_service=source if input_source_async is None else None,
        input_source_async=input_source_async, input_source_bind_events=input_source_bind_events,
        input_sync_service=sync, output_counts=[1],
        output_weight=0., trace_requests=True,input_release_delay_ns=release_visibility_ns,
        consumer_entry_delay_ns=consumer_entry_delay_ns,
        input_destination_service=input_destination_service)
    if input_source_async is not None:
        observed_requests = [dict(requests(demand['index'], stages, source_tiles, source_profile,
            source_batch=source_batch, source_batches=source_batches)[
            {'issue_a':0, 'issue_b0':1, 'issue_b1':2}[demand['kind']]],
            arrival_ns=demand['arrival_ns'], source_ready_ns=demand['source_ready_ns'],
            issue_return_ns=demand.get('issue_return_ns')) for demand in result['input_demands']]
    actor = result['ctas'][0]
    timeline = _input_timeline(actor, stages, release_visibility_ns)
    value = dict(unit='ns', qualification='显式给价的D局部连接；硬件服务未校准',
        ktiles=ktiles, stages=stages, prefilled_tiles=0, physical_slots=6,
        source_tiles=source_tiles, descriptors=expected_descriptor(source_tiles, source_batches),
        transaction_bytes=32768, full_arrivals=1, empty_arrivals=128,
        requests=observed_requests, timeline=timeline,
        tail_wait0_and_release_ns=actor['main_ends'][0],
        slots=[sync.slot_snapshot(0, slot) for slot in range(stages)])
    if any(producer_control_ns):value['producer_control_windows']=result['producer_control_windows']
    if release_visibility_ns:
        value.update(tail_wait0_return_ns=actor['main_ends'][0],
                     tail_wait0_and_release_ns=actor['released'][-1])
    if consumer_entry_delay_ns:value['consumer_entry_delay_ns']=consumer_entry_delay_ns
    if source_batches != 1:
        value.update(source_batch=source_batch, source_batches=source_batches)
    if input_destination_service is not None or input_source_async is not None:
        value['deliveries']=result['deliveries']
    if input_source_async is not None:
        value.update(input_demands=result['input_demands'], sources=result['sources'])
    return value


def timed_cta_group(ktiles, stages, source_batches_by_cta, *, source_batches,
                    request_service, compute_ns, commit_lead_ns, source_tiles=None,
                    source_profile='ring', completion_lag_ns=0., loop_advance_ns=0.,
                    producer_gap_ns=0., operand_gap_ns=0.,
                    release_visibility_ns=0., fast_return_ns=0., blocked_return_ns=0.,
                    input_source_async=None, input_source_bind_events=None,
                    input_destination_service=None, producer_control_ns=(0.,0.,0.)):
    """D的多个CTA在同一次引擎中推进；源callback接收(t, cta, request)。"""
    batches = list(source_batches_by_cta)
    source_tiles = stages if source_tiles is None else source_tiles
    observed = []

    def request(cta, tile, atom):
        return requests(tile, stages, source_tiles, source_profile,
            source_batch=batches[cta], source_batches=source_batches)[atom]

    def source(t, cta, tile, atom, weight):
        q = request(cta, tile, atom)
        served = request_service(t, cta, q)
        observed.append(dict(q, cta=cta, arrival_ns=t, source_ready_ns=served['source_ready'],
                             issue_return_ns=served.get('issue_return_ns')))
        return served

    sync = ConditionSyncService(stages, fast_return_ns, blocked_return_ns)
    result = shared_stage_events(ktiles, stages, len(batches), service_ns=0., read_latency_ns=0.,
        compute_ns=compute_ns, commit_lead_ns=commit_lead_ns, completion_lag_ns=completion_lag_ns,
        producer_gap_ns=producer_gap_ns, operand_gap_ns=operand_gap_ns,
        producer_control_ns=producer_control_ns,
        loop_advance_ns=loop_advance_ns, input_release_delay_ns=release_visibility_ns,
        input_source_service=source if input_source_async is None else None,
        input_source_async=input_source_async, input_source_bind_events=input_source_bind_events,
        input_destination_service=input_destination_service, input_sync_service=sync,
        output_counts=[1]*len(batches), output_weight=0., trace_requests=True)
    if input_source_async is not None:
        observed = [dict(request(q['cta'],q['index'],{'issue_a':0,'issue_b0':1,'issue_b1':2}[q['kind']]),
            cta=q['cta'],arrival_ns=q['arrival_ns'],source_ready_ns=q['source_ready_ns'],
            issue_return_ns=q.get('issue_return_ns')) for q in result['input_demands']]
    actors = []
    for cta,actor in enumerate(result['ctas']):
        actors.append(dict(cta=cta,source_batch=batches[cta],
            timeline=_input_timeline(actor,stages,release_visibility_ns),
            tail_wait0_return_ns=actor['main_ends'][0],
            tail_wait0_and_release_ns=actor['released'][-1] if release_visibility_ns else actor['main_ends'][0],
            slots=[sync.slot_snapshot(cta,slot) for slot in range(stages)]))
    value = dict(schema='gh200-d-cta-group/1',unit='ns',
        qualification='显式给价的D多CTA局部连接；硬件服务未校准',
        entry_rule='所有CTA从0开始，相同时刻沿既有引擎顺序',
        ktiles=ktiles,stages=stages,physical_slots=6,cta_count=len(batches),
        source_tiles=source_tiles,source_batches=source_batches,
        descriptors=expected_descriptor(source_tiles,source_batches),
        transaction_bytes=32768,full_arrivals=1,empty_arrivals=128,
        requests=observed,ctas=actors,deliveries=result['deliveries'],
        input_demands=result['input_demands'],sources=result['sources'],
        accounting={key:result[key] for key in ('input_demand_count','input_source_count',
            'input_merged_count','input_delivery_count','source_busy_ns','destination_busy_ns')},
        tail_wait0_and_release_ns=max(a['tail_wait0_and_release_ns'] for a in actors))
    if any(producer_control_ns):value['producer_control_windows']=result['producer_control_windows']
    return value


def verify_output(path, ktiles, stages, seed):
    require(path.stat().st_size == 65536, '完整累加器输出长度不符')
    actual = np.fromfile(path, dtype='<f4')
    require(np.isfinite(actual).all() and np.array_equal(actual,reference(ktiles,stages,seed)),
            '独立整数参考与完整输出不符')
    return dict(checked_elements=16384, sha256=digest(path))


def validate_layout(row):
    require(row['schema']=='gh200-cfg-b-tma-layout/1' and row['cuda_api_calls']==0 and
        row['descriptor_instance'] is None,'CPU布局查询资格不符')
    require(row['elements_checked']==98304 and row['mismatch_count']==0 and
        row['duplicate_half_addresses']==0 and row['atom_coverage_errors']==0 and
        row['pointer_swizzle_applied'] is True,'SMEM完整地址核对失败')
    require(row['tensor_storage_bytes']==196608 and row['transaction_bytes']==32768 and
        row['a_atom_bytes']==16384 and row['b_atom_bytes']==8192,'布局或请求字节不符')
    require(len(row['slots'])==6,'物理槽查询不完整')
    for slot,item in enumerate(row['slots']):
        a,b0,b1=requests(slot,6,6)
        require(item['slot']==slot and item['a_base_bytes']==a['destination_byte_offset'] and
            item['b0_base_bytes']==b0['destination_byte_offset'] and
            item['b1_base_bytes']==b1['destination_byte_offset'] and
            item['a_unique_halves']==8192 and item['b0_unique_halves']==4096 and
            item['b1_unique_halves']==4096 and item['b_atom_overlap_halves']==0,
            '逐请求目的地址或覆盖不符')
    return dict(elements_checked=98304,mismatch_count=0,descriptor_instance=None)


def validate(folder,row,case,seed):
    require(row.get('schema')=='gh200-cfg-b-tma-chain/1' and row.get('status')=='measured',
            'probe状态不符')
    require(all(row.get(key)==case[key] for key in ('stages','ktiles','trace','wgmma_count','work_flop')) and
        row.get('seed')==seed and row.get('samples')==1,'测量条件不符')
    require(row['source_profile']=='ring' and row['source_tiles']==case['stages'] and
            row['source_k']==64*case['stages'],'源环条件不符')
    device,resources=row['device'],row['resources']
    require('GH200' in device['name'] and device['sms']==132 and
            (device['major'],device['minor'])==(9,0),'目标设备不符')
    require(resources['threads']==256 and resources['grid_ctas']==1 and
        resources['dynamic_smem_bytes']==196608 and resources['local_size_bytes']==0 and
        resources['occupancy_ctas_per_sm']==1,'计算资源不符')
    expected=dict(slots=6,active_slots=case['stages'],full_arrivals=1,empty_arrivals=128,
        signal_kind='arrive_and_expect_tx_then_native_tma',transaction_bytes=32768,
        prefilled_tiles=0,tma_requests_per_tile=3,tma_request_bytes=[16384,8192,8192],
        logical_tma_bytes=case['tma_bytes'],smem_swizzle_bytes=128,
        software_tile=[128,128,64],instruction_shape=[64,128,16],commit_per_ktile=1,
        wait_groups=1,tail_wait_groups=0)
    require(all(row['conditions'].get(key)==value for key,value in expected.items()),
            'TMA／barrier／计算条件不符')
    timing=row['timing']
    require(timing['smid_start']==timing['smid_end'],'CTA的SM身份变化')
    for unit,elapsed in (('cycle','elapsed_cycles'),('ns','elapsed_ns')):
        begin,end=timing[f'begin_{unit}'],timing[f'end_{unit}']
        require(type(begin) is int and type(end) is int and end>begin and timing[elapsed]==end-begin,
                '原始计时不能复算')
    def correctness(value,length):
        require(value['checked_elements']==16384 and value['errors']==0 and value['max_abs_error']==0 and
                value['generation_checked']==128*length and value['generation_errors']==0,'完整输出或代次错误')
    correctness(row['correctness'],case['ktiles'])
    require(row['output_file']=='output.f32' and row['output_elements']==16384 and row['output_bytes']==65536,
            '性能输出见证不符')
    performance=verify_output(folder/'output.f32',case['ktiles'],case['stages'],seed)
    witnesses=row['generation_witnesses']
    require([item['ktiles'] for item in witnesses]==[1,2,case['stages']+1,case['ktiles']],
            '代次重新填槽的长度不符')
    for item in witnesses:
        require(item['source_profile']=='generation','代次见证输入不符')
        correctness(item['correctness'],item['ktiles'])
    require(witnesses[-1]['output_file']=='witness_output.f32' and
        witnesses[-1]['output_elements']==16384 and witnesses[-1]['output_bytes']==65536,'代次完整输出见证不符')
    witness_path=folder/'witness_output.f32'
    require(witness_path.stat().st_size==65536 and
        np.array_equal(np.fromfile(witness_path,dtype='<f4'),reference(case['ktiles'],case['stages'],seed,'generation')),
            '独立整数参考与重新填槽输出不符')
    descriptors=row['descriptors']
    require(descriptors['bytes_each']==128 and all(len(descriptors[key])==256 and
        len(bytes.fromhex(descriptors[key]))==128 for key in ('a_hex','b_hex')),'描述符实例长度不符')
    intervals=[]
    if case['trace']:
        trace=row['trace_detail']
        require(len(trace['tiles'])==case['ktiles'] and trace['tail_release_tile']==case['ktiles']-1,
                'trace长度或尾部释放组错误')
        for tile,item in enumerate(trace['tiles']):
            expected=dict(tile=tile,slot=tile % case['stages'],source_tile=tile % case['stages'],generation=tile//case['stages'],
                full_wait_phase=(tile//case['stages']) % 2,empty_wait_phase=(tile//case['stages']+1) % 2,
                prefilled=False)
            require(all(item.get(key)==value for key,value in expected.items()),'trace槽代次不符')
            p,c=item['producer'],item['consumer']
            require(c['release_tile']==(tile-1 if tile else None),'释放的输入组错误')
            bounds={}
            for unit in ('cycle','ns'):
                begin,end=timing[f'begin_{unit}'],timing[f'end_{unit}']
                points=[p[key][unit] for key in ('acquire_before','acquire_after','expect_before','expect_after',
                    'request_A_before','request_A_after','request_B_pair_before','request_B_pair_after')]
                require(begin<=points[0] and points==sorted(points) and points[-1]<=end,
                        'producer发出顺序或窗口错误')
                points=[c[key][unit] for key in ('wait_before','wait_after','issue_before','commit_after')]
                if tile:
                    points += [c[key][unit] for key in ('wait1_after','release_before','release_after')]
                    require(c['release_before'][unit]<=c['release_last_before'][unit]<=
                        c['release_last_after'][unit] and c['release_after'][unit]<=c['release_last_after'][unit],
                        '最后arrival区间错误')
                    prior_control=trace['tiles'][tile-1]['consumer']['release_after' if tile>1 else 'commit_after']
                    require(c['wait_before'][unit]>=prior_control[unit],
                            'consumer逐组顺序错误')
                require(begin<=points[0] and points==sorted(points) and points[-1]<=end,
                        'consumer顺序或窗口错误')
                require(c['wait_after'][unit]>=p['request_B_pair_before'][unit],
                        '等待返回早于必要的B请求对')
                if tile:
                    require(p['acquire_before'][unit]>=trace['tiles'][tile-1]['producer']['request_B_pair_after'][unit],
                            'producer逐组顺序错误')
                if tile>=case['stages']:
                    prior=trace['tiles'][tile-case['stages']+1]['consumer']
                    require(p['acquire_after'][unit]>=prior['release_last_before'][unit],
                            'producer取得槽早于最后必要arrival')
                bounds[unit]=[p['request_B_pair_before'][unit],c['wait_after'][unit]]
            masks=c['first_try_success_warp_masks']
            require(len(masks)==4 and all(type(mask) is int and 0<=mask<2**32 for mask in masks) and
                    bool(masks[0]&1)==c['first_try_success'],'首次try参与者记录错误')
            pending=all(c['wait_before'][unit]<p['request_B_pair_before'][unit] for unit in ('cycle','ns'))
            intervals.append(dict(tile=tile,consumer_thread=0,full_completion_bounds=bounds,
                entry_state='entered_before_B_pair' if pending else 'unresolved_entry_state',
                first_try_success=c['first_try_success']))
        for unit in ('cycle','ns'):
            points=[trace[key][unit] for key in ('tail_wait0_after','tail_release_before','tail_release_after')]
            require(trace['tiles'][-1]['consumer']['commit_after'][unit]<=points[0] and
                points==sorted(points) and points[1]<=trace['tail_release_last_before'][unit]<=
                trace['tail_release_last_after'][unit]<=timing[f'end_{unit}'] and
                points[2]<=trace['tail_release_last_after'][unit],'尾排空／最后arrival区间错误')
        first=trace['tiles'][0]['consumer']
        prefix={unit:dict(first_wait_return=first['wait_after'][unit]-timing[f'begin_{unit}'],
                         first_compute=first['issue_before'][unit]-timing[f'begin_{unit}'])
                for unit in ('cycle','ns')}
    else:
        require(row['trace_detail'] is None,'plain包含trace')
        prefix=None
    warmup=row['warmup']
    windows=warmup['elapsed_ns']
    require(warmup['limit']==60 and 8<=len(windows)<=60 and min(windows)>0 and
            len(warmup['elapsed_cycles'])==len(windows),'暖机记录不符')
    mean=sum(windows[-5:])/5
    cv=math.sqrt(sum((value-mean)**2 for value in windows[-5:])/4)/mean
    require(warmup['converged']==(cv<=.02) and math.isclose(cv,warmup['cv'],rel_tol=1e-7,abs_tol=1e-12),
            '暖机CV不能复算')
    return dict(device_uuid=device['uuid'],timing=timing,resources=resources,performance_output=performance,
        generation_output_sha256=digest(witness_path),intervals=intervals,prefix=prefix,
        performance_eligible=warmup['converged'],tma_bytes=case['tma_bytes'],tma_requests=case['tma_requests'])


def probe_command(binary, case, seed):
    return list(map(str,[binary,'--stages',case['stages'],'--ktiles',case['ktiles'],
                        '--trace',int(case['trace']),'--seed',seed,'--warmup-limit',60]))


def prepare(root, headers):
    root.mkdir(parents=True,exist_ok=False)
    source = root/'source'
    source.mkdir()
    for name in ('cfg_b_tma_chain.py','cfg_b_handoff_chain.py','cfg_b_compute_chain.py',
                 'l2_saturation.py','mbarrier_service.py','supply_model.py',
                 'test_cfg_b_tma_chain.py','test_cfg_b_tma_records.py'):
        shutil.copy2(HERE/name,source/name)
    for name in ('cfg_b_tma_chain.cu','r00_common.hpp'):
        shutil.copy2(HERE/'probes'/name,source/name)
    shutil.copy2(headers,source/'cutlass-3.9.2-include.tar.gz')
    save_json(root/'plan.json',dict(schema=SCHEMA,cases=cases(),seeds=list(SEEDS),
        threads=256,physical_slots=6,dynamic_smem_bytes=196608,full_arrivals=1,
        transaction_bytes=32768,empty_arrivals=128,dev_processes=18,heldout_processes=6,
        cache_protocol='repeated_launch_warmup_8_to_60_CV2pct',GPU_execution=False,
        source_sha256={path.name:digest(path) for path in source.iterdir()}))
    save_json(root/'program-reference.json',dict(qualification='CPU程序与计数见证，未定价',
        conditions={row['id']:lifecycle(row['ktiles'],row['stages']) for row in cases() if not row['trace']}))
    save_json(root/'requests-reference.json',dict(qualification='输入语义与待编码元信息，非实际descriptor bits',
        conditions={row['id']:dict(descriptor=expected_descriptor(row['stages']),
            requests=[request for tile in range(row['ktiles'])
                      for request in requests(tile,row['stages'],row['stages'])])
                    for row in cases() if not row['trace']}))
    save_json(root/'commands.json',dict(executed=False,commands=[dict(case_id=row['id'],seed=seed,
        argv=probe_command(root/'build/probe',row,seed)) for row in cases() for seed in SEEDS]))
    print(f'PREPARED {root}: 8形状／24进程命令；全部TMA在主窗口内；GPU0次')


def compile_probe(root,nvcc):
    plan=read_json(root/'plan.json')
    require(plan['schema']==SCHEMA,'准备包schema不符')
    for name,checksum in plan['source_sha256'].items():
        require(digest(root/'source'/name)==checksum,f'准备包源码变更：{name}')
    compiler=shutil.which(nvcc)
    require(compiler is not None,'nvcc未找到')
    build=root/'build'
    build.mkdir(exist_ok=False)
    command_record(build/'version',[compiler,'--version'],30)
    flags=['-std=c++17','-O3','-DNDEBUG','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
    with tempfile.TemporaryDirectory(prefix='cfg-b-tma-headers-') as temporary:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(temporary,filter='data')
        command_record(build/'compile',[compiler,*flags,'-I'+temporary+'/include',
            root/'source/cfg_b_tma_chain.cu','-o',build/'probe'],180)
    cuobjdump=Path(compiler).resolve().parent/'cuobjdump'
    command_record(build/'disassemble',[cuobjdump,'--dump-sass',build/'probe'],60)
    command_record(build/'layout',[build/'probe','--inspect-layout'],30)
    validate_layout(read_json(build/'layout/stdout'))
    save_json(build/'identity.json',dict(plan_sha256=digest(root/'plan.json'),
        probe_sha256=digest(build/'probe'),sass_sha256=digest(build/'disassemble/stdout'),
        layout_sha256=digest(build/'layout/stdout'),flags=flags,compiler=str(Path(compiler).resolve()),
        static_instruction_review='待核对'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--headers',type=Path,required=True)
    build=sub.add_parser('compile')
    build.add_argument('--output',type=Path,required=True)
    build.add_argument('--nvcc',default='nvcc')
    check=sub.add_parser('check')
    check.add_argument('--folder',type=Path,required=True)
    check.add_argument('--plan',type=Path,required=True)
    check.add_argument('--case-id',required=True)
    check.add_argument('--seed',type=int,choices=SEEDS,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.output.resolve(),args.headers.resolve())
    elif args.command=='compile':
        compile_probe(args.output.resolve(),args.nvcc)
    else:
        case=next(row for row in read_json(args.plan)['cases'] if row['id']==args.case_id)
        print(json.dumps(validate(args.folder,read_json(args.folder/'stdout'),case,args.seed),
                         ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
