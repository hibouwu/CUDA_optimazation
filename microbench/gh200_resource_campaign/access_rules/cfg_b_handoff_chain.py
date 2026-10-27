#!/usr/bin/env python3
"""cfg_b 实际计算与普通 full/empty 交接短链的 CPU 准备和离线核对。"""
import argparse
import json
import math
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from cfg_b_compute_chain import output_coordinates, SEEDS
from l2_saturation import command_record, digest, read_json, require, save_json
from mbarrier_service import BarrierPhase
from supply_model import wgmma_compute_step

HERE = Path(__file__).resolve().parent
SCHEMA = "gh200-cfg-b-handoff-chain-plan/1"


def cases(delay_cycles):
    require(type(delay_cycles) is int and 0 < delay_cycles <= 1000000, "C延迟须为1..1000000周期")
    rows = []
    for stages, ktiles in ((6, 4), (6, 13), (6, 32), (4, 13)):
        for variant in (('B',) if ktiles <= stages else ('B', 'C')):
            for trace in (False, True):
                rows.append(dict(id=f"s{stages}_k{ktiles}_{variant}_{'trace' if trace else 'plain'}",
                    stages=stages, ktiles=ktiles, variant=variant, trace=trace,
                    delay_cycles=0 if variant == 'B' else delay_cycles,
                    split='heldout' if ktiles == 32 else 'dev',
                    wgmma_count=8*ktiles, work_flop=2*128*128*64*ktiles))
    return rows


def reference(ktiles, stages, seed):
    rows = np.arange(128, dtype=np.int64)[:, None]
    inner = np.arange(64, dtype=np.int64)[None, :]
    total = np.zeros((128, 128), dtype=np.int64)
    for slot in range(stages):
        count = (ktiles + stages - 1 - slot)//stages
        a = (7*rows + 13*inner + 3*slot + 3*(seed % 17)) % 17 - 8
        b = (11*rows + 5*inner + 7*slot + 5*(seed % 17)) % 17 - 8
        total += count*(a @ b.T)
    row, column = output_coordinates()
    return (total[row, column]/1024).astype(np.float32)


def lifecycle(ktiles, stages, delay_cycles, *, wait_retain=1):
    """无时间价格的有序见证；empty由128次独立arrival推进。"""
    require(wait_retain in (0, 1), '正常计算等待保留组数须为0或1')
    full = [BarrierPhase(1) for _ in range(stages)]
    empty = [BarrierPhase(128) for _ in range(stages)]
    published = [-1]*stages
    nodes, releases = [], {}
    last_publish = None

    def node(kind, tile, predecessors=(), **extra):
        value = dict(id=len(nodes), kind=kind, tile=tile, slot=tile % stages,
            generation=tile//stages, full_wait_phase=(tile//stages) % 2,
            empty_wait_phase=(tile//stages+1) % 2,
            predecessors=list(predecessors), **extra)
        nodes.append(value)
        return value['id']

    def publish(tile):
        nonlocal last_publish
        slot, generation = tile % stages, tile//stages
        require(empty[slot].wait_complete((generation+1) % 2), "producer过早取得槽")
        predecessors = [] if generation == 0 else [releases[tile-stages]]
        if last_publish is not None:
            predecessors.append(last_publish)
        acquire = node('producer_acquire', tile, predecessors, prefilled=generation == 0)
        if tile == stages and delay_cycles:
            acquire = node('declared_delay', tile, [acquire], delay_cycles=delay_cycles)
        require(published[slot] == generation-1, "发布代次不连续")
        published[slot] = generation
        full[slot].arrive()
        last_publish = node('full_arrive', tile, [acquire])
        return last_publish

    ready = {tile: publish(tile) for tile in range(min(stages, ktiles))}
    complete, commits = {}, {}
    control = None
    for tile in range(ktiles):
        slot, generation = tile % stages, tile//stages
        if tile not in ready:
            ready[tile] = publish(tile)
        require(full[slot].wait_complete(generation % 2) and published[slot] == generation,
                "消费者取得错误代次")
        previous = [] if control is None else [control]
        wait = node('consumer_full_wait', tile, [ready[tile], *previous])
        issue = node('eight_wgmma', tile, [wait], instructions=8)
        commits[tile] = node('commit', tile, [issue])
        complete[tile] = node('group_complete', tile, [commits[tile]])
        control = commits[tile]
        if tile:
            wait1 = node(f'wait{wait_retain}', tile,
                         [commits[tile], complete[tile-wait_retain]])
            released = tile-1
            for _ in range(128):
                empty[released % stages].arrive()
            releases[released] = node('empty_release', released, [wait1], arrivals=128)
            control = releases[released]
    tail = node('tail_wait0', ktiles-1, [*complete.values(), control])
    for _ in range(128):
        empty[(ktiles-1) % stages].arrive()
    releases[ktiles-1] = node('empty_release', ktiles-1, [tail], arrivals=128)
    return dict(ktiles=ktiles, stages=stages, delay_cycles=delay_cycles,
        qualification='程序依赖与计数见证；未赋硬件时间', nodes=nodes,
        full=[value.snapshot() for value in full], empty=[value.snapshot() for value in empty])


def classify_wait(producer, consumer, prefilled):
    if prefilled:
        return 'ready_before_window'
    # 首次try_wait本身可能等待；其成功标志不用于倒推进入瞬间的状态。
    ready = all(producer['full_after'][unit] <= consumer['wait_before'][unit]
                for unit in ('cycle', 'ns'))
    pending = all(consumer['wait_before'][unit] < producer['full_before'][unit]
                  for unit in ('cycle', 'ns'))
    return 'ready_before_enter' if ready else 'entered_before_signal' if pending else 'overlap'


def timed_lifecycle(ktiles, stages, delay_cycles, *, compute_cycles=512.,
                    commit_lead_cycles=256., completion_lag_cycles=0.,
                    consumer_advance_cycles=0., producer_advance_cycles=0.,
                    release_visibility_cycles=0., full_ready_return_cycles=0.,
                    full_blocked_return_cycles=0., wait_retain=1):
    """递增回绕B/C的局部cycle假设；普通arrival、计时前预填、末尾wait0。

    计算服务进度与完成分开；正常等待返回发布前组empty，最后一组由tail释放。
    release_visibility是发布到最后必要arrival的显式假设，消费者本地推进独立。
    前缀、结果消费及最终CTA会合未定价，返回端点为局部链结束。
    """
    program = lifecycle(ktiles, stages, delay_cycles, wait_retain=wait_retain)
    values = (compute_cycles, commit_lead_cycles, completion_lag_cycles,
              consumer_advance_cycles, producer_advance_cycles, release_visibility_cycles,
              full_ready_return_cycles, full_blocked_return_cycles)
    require(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in values) and compute_cycles > 0, '局部服务须为非负有限cycle')
    rows = []
    consumer_free = producer_free = service_frontier = 0.
    for tile in range(ktiles):
        state = dict(tile=tile, slot=tile % stages, generation=tile//stages,
                     full_wait_phase=(tile//stages) % 2,
                     empty_wait_phase=(tile//stages+1) % 2,
                     prefilled=tile < stages)
        if tile < stages:
            state.update(acquire_enter_cycle=None, empty_ready_cycle=None,
                         acquire_return_cycle=None, full_publish_cycle=0.)
        else:
            enter = producer_free + producer_advance_cycles
            empty_ready = rows[tile-stages]['empty_complete_cycle']
            acquired = max(enter, empty_ready)
            published = acquired + (delay_cycles if tile == stages else 0.)
            state.update(acquire_enter_cycle=enter, empty_ready_cycle=empty_ready,
                         acquire_return_cycle=acquired, full_publish_cycle=published)
            producer_free = published
        full_eligibility = max(consumer_free, state['full_publish_cycle'])
        full_path = 'ready' if consumer_free >= state['full_publish_cycle'] else 'blocked'
        full_cost = full_ready_return_cycles if full_path == 'ready' else full_blocked_return_cycles
        full_return = full_eligibility + full_cost
        if full_ready_return_cycles or full_blocked_return_cycles:
            state.update(full_eligibility_cycle=full_eligibility, full_wait_path=full_path,
                         full_return_cost_cycle=full_cost)
        service_frontier, commit, complete, returned = wgmma_compute_step(
            full_return, service_frontier,
            None if tile == 0 else rows[-1]['compute_complete_cycle'],
            compute_duration=compute_cycles, commit_lead=commit_lead_cycles,
            completion_lag=completion_lag_cycles, wait_retain=wait_retain)
        state.update(full_enter_cycle=consumer_free, full_return_cycle=full_return,
                     compute_begin_cycle=full_return, service_end_cycle=service_frontier,
                     commit_cycle=commit, compute_complete_cycle=complete,
                     wait1_return_cycle=None if tile == 0 else returned,
                     consumer_continue_cycle=returned + consumer_advance_cycles)
        if wait_retain == 0:
            state['wait0_return_cycle'] = state.pop('wait1_return_cycle')
        if tile:
            rows[-1].update(release_emit_cycle=returned,
                            empty_complete_cycle=returned + release_visibility_cycles)
        rows.append(state)
        consumer_free = state['consumer_continue_cycle']
    tail = max(returned, complete)
    rows[-1].update(release_emit_cycle=tail,
                    empty_complete_cycle=tail + release_visibility_cycles)
    return dict(unit='SM clock64 cycle', qualification='工作假设：普通arrival局部时间连接',
                ktiles=ktiles, stages=stages, delay_cycles=delay_cycles,
                physical_slots=6, full_arrivals=1, full_transaction_bytes=0,
                empty_arrivals=128, timeline=rows, tail_wait0_return_cycle=tail,
                local_end_cycle=max(tail, rows[-1]['empty_complete_cycle']),
                control_and_return_costs=('显式给定推进／可见性及条件full返回；empty返回额外成本为0'
                    if full_ready_return_cycles or full_blocked_return_cycles else
                    '显式给定推进／可见性；full及empty返回额外成本为0'),
                barrier_final_state=dict(full=program['full'], empty=program['empty']))


def verify_output(path, ktiles, stages, seed):
    require(path.stat().st_size == 65536, '完整累加器输出长度不符')
    actual = np.fromfile(path, dtype='<f4')
    require(np.isfinite(actual).all() and np.array_equal(actual, reference(ktiles, stages, seed)),
            '独立整数参考与完整累加器输出不符')
    return dict(checked_elements=16384, sha256=digest(path))


def validate(folder, row, case, seed):
    require(row.get('schema') == 'gh200-cfg-b-handoff-chain/1' and row.get('status') == 'measured',
            'probe状态不符')
    require(all(row.get(key) == case[key] for key in
        ('stages', 'ktiles', 'delay_cycles', 'trace', 'wgmma_count', 'work_flop')) and
        row.get('seed') == seed and row.get('samples') == 1, '测量条件不符')
    device, resources = row['device'], row['resources']
    require('GH200' in device['name'] and device['sms'] == 132 and
            (device['major'], device['minor']) == (9, 0), '目标设备不符')
    require(resources['threads'] == 256 and resources['grid_ctas'] == 1 and
        resources['dynamic_smem_bytes'] == 196608 and resources['local_size_bytes'] == 0 and
        resources['occupancy_ctas_per_sm'] == 1, '计算资源不符')
    expected = dict(slots=6, active_slots=case['stages'], full_arrivals=1, empty_arrivals=128,
        signal_kind='normal_arrive', prefilled_tiles=min(case['stages'], case['ktiles']))
    require(all(row['conditions'].get(key) == value for key, value in expected.items()),
            'barrier或槽条件不符')
    timing = row['timing']
    require(timing['smid_start'] == timing['smid_end'], 'CTA的SM身份变化')
    for unit, elapsed in (('cycle', 'elapsed_cycles'), ('ns', 'elapsed_ns')):
        begin, end = timing[f'begin_{unit}'], timing[f'end_{unit}']
        require(type(begin) is int and type(end) is int and end > begin and
                timing[elapsed] == end-begin, '原始计时不能复算')
    correctness = row['correctness']
    require(correctness['generation_checked'] == 128*case['ktiles'] and
        correctness['generation_errors'] == 0 and correctness['checked_elements'] == 16384 and
        correctness['errors'] == 0 and correctness['max_abs_error'] == 0, '完整结果或代次错误')
    witness = verify_output(folder/row['output_file'], case['ktiles'], case['stages'], seed)
    require([check['ktiles'] for check in row['short_checks']] == [1, 2, case['stages']+1] and
        all(check['correctness']['errors'] == 0 and check['correctness']['generation_errors'] == 0 and
            check['correctness']['checked_elements'] == 16384 and
            check['correctness']['generation_checked'] == 128*check['ktiles']
            for check in row['short_checks']), '短链完整正确性见证不符')
    intervals = []
    if case['trace']:
        trace = row['trace_detail']
        require(len(trace['tiles']) == case['ktiles'], 'trace不完整')
        require(trace['tail_release_tile'] == case['ktiles']-1, '尾部释放组不符')
        for tile, point in enumerate(trace['tiles']):
            require(all(point[key] == value for key, value in dict(tile=tile,
                slot=tile % case['stages'], generation=tile//case['stages'],
                full_wait_phase=(tile//case['stages']) % 2,
                empty_wait_phase=(tile//case['stages']+1) % 2,
                prefilled=tile < case['stages']).items()), 'trace代次不符')
            producer, consumer = point['producer'], point['consumer']
            require(consumer['release_tile'] == (tile-1 if tile else None), '释放的输入组不符')
            for unit in ('cycle', 'ns'):
                points = [consumer[name][unit] for name in
                    ('wait_before', 'wait_after', 'issue_before', 'commit_after')]
                if tile:
                    points += [consumer[name][unit] for name in
                        ('wait1_after', 'release_before', 'release_after')]
                    last_before = consumer['release_last_before'][unit]
                    last_after = consumer['release_last_after'][unit]
                    require(consumer['release_before'][unit] <= last_before <= last_after and
                            consumer['release_after'][unit] <= last_after, '最后arrival区间错误')
                require(points == sorted(points) and points[0] >= timing[f'begin_{unit}'] and
                        points[-1] <= timing[f'end_{unit}'], '消费者trace顺序错误')
                if tile >= case['stages']:
                    points = [producer[name][unit] for name in
                        ('acquire_before', 'acquire_after', 'full_before', 'full_after')]
                    require(points == sorted(points), '生产者trace顺序错误')
                    require(consumer['wait_after'][unit] >= producer['full_before'][unit],
                            '消费者返回早于必要信号')
                    prior_release = trace['tiles'][tile-case['stages']+1]['consumer']
                    require(producer['acquire_after'][unit] >= prior_release['release_last_before'][unit],
                            'producer取得槽早于最后必要arrival')
            if tile == case['stages']:
                require(producer['full_before']['cycle']-producer['acquire_after']['cycle'] >=
                        case['delay_cycles'], '声明的延迟未执行')
            masks = consumer['first_try_success_warp_masks']
            require(len(masks) == 4 and all(type(mask) is int and 0 <= mask < 2**32 for mask in masks) and
                    bool(masks[0] & 1) == consumer['first_try_success'], '首次try的参与者记录不符')
            intervals.append(dict(tile=tile, state=classify_wait(producer, consumer, point['prefilled']),
                first_try_success=consumer['first_try_success']))
        for unit in ('cycle', 'ns'):
            points = [trace[key][unit] for key in
                ('tail_wait0_after', 'tail_release_before', 'tail_release_after')]
            require(trace['tiles'][-1]['consumer']['commit_after'][unit] <= points[0] and
                    points == sorted(points) and points[-1] <= timing[f'end_{unit}'], '尾部trace顺序错误')
            require(points[1] <= trace['tail_release_last_before'][unit] <=
                    trace['tail_release_last_after'][unit] <= timing[f'end_{unit}'] and
                    points[2] <= trace['tail_release_last_after'][unit], '尾部最后arrival区间错误')
    else:
        require(row['trace_detail'] is None, 'plain包含trace')
    warmup = row['warmup']
    windows = warmup['elapsed_ns']
    require(8 <= len(windows) <= 60 and warmup['limit'] == 60 and min(windows) > 0,
            '暖机记录不符')
    mean = sum(windows[-5:])/5
    cv = math.sqrt(sum((value-mean)**2 for value in windows[-5:])/4)/mean
    require(warmup['converged'] == (cv <= .02) and
            math.isclose(cv, warmup['cv'], rel_tol=1e-7, abs_tol=1e-12), '暖机CV不能复算')
    return dict(device_uuid=device['uuid'], timing=timing, resources=resources,
        output_witness=witness, intervals=intervals, performance_eligible=warmup['converged'])


def probe_command(binary, case, seed):
    return list(map(str, [binary, '--stages', case['stages'], '--ktiles', case['ktiles'],
        '--delay-cycles', case['delay_cycles'], '--trace', int(case['trace']),
        '--seed', seed, '--warmup-limit', 60]))


def prepare(root, headers, delay_cycles):
    rows = cases(delay_cycles)
    root.mkdir(parents=True, exist_ok=False)
    source = root/'source'
    source.mkdir()
    for name in ('cfg_b_handoff_chain.py', 'cfg_b_compute_chain.py', 'l2_saturation.py',
                 'mbarrier_service.py', 'supply_model.py', 'test_cfg_b_handoff_chain.py',
                 'test_cfg_b_handoff_records.py'):
        shutil.copy2(HERE/name, source/name)
    for name in ('cfg_b_handoff_chain.cu', 'r00_common.hpp'):
        shutil.copy2(HERE/'probes'/name, source/name)
    shutil.copy2(headers, source/'cutlass-3.9.2-include.tar.gz')
    save_json(root/'plan.json', dict(schema=SCHEMA, cases=rows, seeds=list(SEEDS),
        delay_qualification='CPU示例；正式延迟与预测待A开发观测后另行冻结',
        proposed_delay_cycles=delay_cycles, GPU_execution=False, threads=256,
        physical_slots=6, dynamic_smem_bytes=196608, full_arrivals=1, empty_arrivals=128,
        dev_processes=30, heldout_processes=12,
        source_sha256={path.name: digest(path) for path in source.iterdir()}))
    save_json(root/'program-reference.json', dict(qualification='无硬件时间的程序与计数见证',
        conditions={row['id']: lifecycle(row['ktiles'], row['stages'], row['delay_cycles'])
                    for row in rows if not row['trace']}))
    save_json(root/'commands.json', dict(executed=False, commands=[dict(case_id=row['id'], seed=seed,
        argv=probe_command(root/'build/probe', row, seed)) for row in rows for seed in SEEDS]))
    print(f'PREPARED {root}: 14形状／42进程命令；CPU延迟示例{delay_cycles}；未运行GPU')


def compile_probe(root, nvcc):
    plan = read_json(root/'plan.json')
    require(plan['schema'] == SCHEMA, '准备包schema不符')
    for name, checksum in plan['source_sha256'].items():
        require(digest(root/'source'/name) == checksum, f'准备包源码变更：{name}')
    build = root/'build'
    build.mkdir(exist_ok=False)
    compiler = shutil.which(nvcc)
    require(compiler is not None, 'nvcc未找到')
    command_record(build/'version', [compiler, '--version'], 30)
    flags = ['-std=c++17', '-O3', '-DNDEBUG', '-lineinfo',
             '-gencode=arch=compute_90a,code=sm_90a', '-Xptxas=-v']
    with tempfile.TemporaryDirectory(prefix='cfg-b-handoff-headers-') as temporary:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(temporary, filter='data')
        command_record(build/'compile', [compiler, *flags, '-I'+temporary+'/include',
            root/'source/cfg_b_handoff_chain.cu', '-o', build/'probe'], 180)
    disassembler = Path(compiler).resolve().parent/'cuobjdump'
    command_record(build/'disassemble', [disassembler, '--dump-sass', build/'probe'], 60)
    save_json(build/'identity.json', dict(plan_sha256=digest(root/'plan.json'),
        probe_sha256=digest(build/'probe'), sass_sha256=digest(build/'disassemble/stdout'),
        flags=flags, compiler=str(Path(compiler).resolve()), static_instruction_review='待核对'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--headers', type=Path, required=True)
    prep.add_argument('--delay-cycles', type=int, required=True)
    build = sub.add_parser('compile')
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--nvcc', default='nvcc')
    check = sub.add_parser('check')
    check.add_argument('--folder', type=Path, required=True)
    check.add_argument('--plan', type=Path, required=True)
    check.add_argument('--case-id', required=True)
    check.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output.resolve(), args.headers.resolve(), args.delay_cycles)
    elif args.command == 'compile':
        compile_probe(args.output.resolve(), args.nvcc)
    else:
        plan = read_json(args.plan)
        case = next(row for row in plan['cases'] if row['id'] == args.case_id)
        print(json.dumps(validate(args.folder, read_json(args.folder/'stdout'), case, args.seed),
                         ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
