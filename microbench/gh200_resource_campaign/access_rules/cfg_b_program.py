"""无时间价格的 cfg_b 程序实例；按 CTA 展开输入槽与 pingpong 依赖。"""
from graphlib import TopologicalSorter
import math

import case_inputs
from register_quota import cfg_b_quota_requirements
import supply_model as supply


TRACE_ENDPOINTS = {
    'ENTRY': 'thread 0 在角色划分和 pipeline 初始化前记录',
    'PRODUCER_FIRST_WORK': '首个输出的生产者工作循环顶部、坐标生成与 load 前',
    'FIRST_MMA': '首个 consumer_wait 和 accumulator operand fence 后、WARPGROUP.ARRIVE/issue 前',
    'MAIN_END': 'mma_tail 返回后，晚于数学组交接',
    'EPI_PERMIT': 'epilogue order wait 返回后、store 前',
    'EPI_DONE': 'store_tail 返回后，包含源读取完成等待',
}


def build_program(row, setup, source_identity=None):
    """复用源请求账本；Ktile 操作按需展开，长 K 不预先复制整张图。"""
    effective, errors = case_inputs.resolve(row, setup)
    if errors:
        raise ValueError('; '.join(errors))
    missing = case_inputs.missing_for(effective, 'program_graph')
    if missing:
        raise ValueError('missing program input: ' + ', '.join(missing))
    if effective['config'] != 'cfg_b' or effective['stages'] not in (4, 6):
        raise ValueError('cfg_b program supports the S4/S6 pingpong builds')
    if effective['interventions']:
        raise ValueError('program does not expand intervention branches: ' + ', '.join(effective['interventions']))
    inputs = {k: v for k, v in effective.items() if k not in ('setup', 'interventions')}
    ledger = supply.supply_work_ledger(inputs, setup)
    active_stages = setup.get('active_stages', ledger['stages'])
    repeats = setup.get('wgmma_repeats', 1)
    if active_stages not in (4, 6) or active_stages > ledger['stages'] or repeats not in (1, 3):
        raise ValueError('cfg_b runtime intervention requires active 4/6 slots and 1/3 WGMMA passes')
    cta_count = math.prod(setup['grid'])
    fold_a = setup.get('fold_a_source', 0)
    private_source = setup.get('private_source', 0)
    store_ctas = setup.get('store_target_ctas', cta_count)
    if fold_a not in (0, 1):
        raise ValueError('fold_a_source must be 0 or 1')
    if not isinstance(store_ctas, int) or not 0 <= store_ctas <= cta_count:
        raise ValueError('store_target_ctas must be an integer within the CTA grid')
    if fold_a:
        if ((inputs['m'], inputs['n'], inputs['k']) != (2816, 1536, 4096)
                or list(setup['grid']) != [1, 132, 1] or inputs['swizzle'] != 1
                or inputs['input_mode'] != 'dyadic' or inputs['zero_m'] != -1
                or ledger['stages'] != 6 or active_stages != 6 or repeats != 1):
            raise ValueError('fold_a_source requires the G3 S6 grid132 dyadic source experiment')
        for atom in ledger['source_atoms']:
            if atom['operand'] == 'A' and atom['origin'][0] >= 17 * 128:
                atom['origin'] = [atom['origin'][0] - 17 * 128, atom['origin'][1]]
    if private_source:
        copy_bytes = dict(A=2 * setup.get('allocated_a_elements', inputs['storage_m'] * inputs['alloc_lda']),
                          B=2 * setup.get('allocated_b_elements', inputs['k'] * inputs['alloc_ldb']))
        for atom in ledger['source_atoms']:
            atom['base_offset_bytes'] += atom['cta'] * copy_bytes[atom['operand']]
    if fold_a or private_source:
        distinct = {(a['operand'], tuple(a['extents']), tuple(a['stride_bytes']),
                     a['base_offset_bytes'], tuple(a['origin']), tuple(a['shape'])): a['valid_KiB']
                    for a in ledger['source_atoms']}
        ledger['totals']['distinct_source_boxes'] = len(distinct) * ledger['ktiles']
        ledger['totals']['distinct_valid_source_KiB'] = sum(distinct.values()) * ledger['ktiles']
    work = [[] for _ in range(cta_count)]
    outputs = [[] for _ in work]
    for output in ledger['outputs']:
        cta, j = output['cta'], output['j']
        work[cta].append(tuple(output['coord']))
        outputs[cta].append(dict(output, consumer_warpgroup=j % 2,
            input_group_range=[j * ledger['ktiles'], (j + 1) * ledger['ktiles']]))
        if active_stages != ledger['stages']:
            outputs[cta][-1].update(
                first_input_state=supply.input_slot_state(j * ledger['ktiles'], active_stages),
                last_input_state=supply.input_slot_state((j + 1) * ledger['ktiles'] - 1, active_stages))
    program = dict(schema='gh200-cfg-b-program/1', case=inputs, setup=dict(setup),
        source_identity=source_identity or {}, schedule='pingpong', tile=[128, 128, 64],
        work=work, outputs=outputs, source_atoms=ledger['source_atoms'],
        ktiles=ledger['ktiles'], stages=ledger['stages'], totals=ledger['totals'],
        trace_endpoints=TRACE_ENDPOINTS,
        descriptor=dict(tma_rank=3, storage_type='UINT16', value_type='FP16',
            smem_swizzle='128B', provenance='frozen source/build reconstruction'),
        input_path=dict(source='device global A/B allocation', destination='CTA-local SMEM'),
        output_path=dict(source='FP32 accumulators', intermediate='epilogue SMEM', destination='global D'))
    if active_stages != ledger['stages'] or repeats != 1 or fold_a or private_source or store_ctas != cta_count:
        program['runtime_intervention'] = dict(active_stages=active_stages,
            wgmma_repeats=repeats, epilogue_alpha=setup.get('epilogue_alpha', 1.))
        if fold_a:
            program['runtime_intervention']['fold_a_source'] = fold_a
        if private_source:
            program['runtime_intervention']['private_source'] = private_source
        if store_ctas != cta_count:
            program['runtime_intervention']['store_target_ctas'] = store_ctas
            program['output_tma_enabled'] = [cta < store_ctas for cta in range(cta_count)]
    return program


def expand_cta(program, cta, *, register_quota=False, epilogue_defer=0,
               separate_empty_completion=False):
    """生成软件必要依赖与服务完成接口；compute_batch 保留为 Ktile 复合节点。

    requests 的 completion 是后续 TMA 服务的输出；compute_complete 是后续计算
    服务的输出。图只规定先后，不赋予它们零成本或具体持续时间。
    """
    if epilogue_defer not in (0, 1, 2):
        raise ValueError('epilogue_defer must be 0, 1 or 2')
    outputs = program['outputs'][cta]
    kt = program['ktiles']
    intervention = program.get('runtime_intervention', {})
    stages = intervention.get('active_stages', program['stages'])
    nodes, groups, requests, observations = {}, [], [], []

    def add(name, *predecessors):
        nodes[name] = list(predecessors)
        return name

    add('entry')
    if register_quota:
        for warp in range(12):
            add(f'register_{"dec" if warp < 4 else "inc"}.w{warp}', 'entry')
    producer_previous = add('producer_first_work', 'register_dec.w0' if register_quota else 'entry')
    for output in outputs:
        j = output['j']
        prefix = f'j{j}.'
        permit = ['entry']
        if register_quota and j < 2:
            permit.extend(f'register_inc.w{warp}' for warp in range(4+4*j,8+4*j))
        if j:
            permit.append(f'j{j-1}.math_handoff')
        if j >= 2:
            permit.append(f'j{j-2}.epilogue_handoff')
        consumer_previous = add(prefix + 'math_permit', *permit)
        for k in range(kt):
            g = j * kt + k
            gp = f'g{g}.'
            state = supply.input_slot_state(g, stages)
            acquire_pre = [producer_previous]
            if state['previous_group'] is not None:
                acquire_pre.append(f'g{state["previous_group"]}.release')
            add(gp + 'acquire', *acquire_pre)
            add(gp + 'expect_tx', gp + 'acquire')
            issue_previous = gp + 'expect_tx'
            completions = []
            atom_ids = output['operands']['A']['source_atoms'] + output['operands']['B']['source_atoms']
            for name, atom_id in zip(('A', 'B0', 'B1'), atom_ids):
                atom = program['source_atoms'][atom_id]
                issue = add(gp + name + '.issue', issue_previous)
                completion = add(gp + name + '.complete', issue)
                origin = [atom['origin'][0], atom['origin'][1] + 64 * k]
                requests.append(dict(atom=name, group=g, output=j, k=k,
                    source=dict(operand=atom['operand'], origin=origin, shape=atom['shape'],
                        extents=atom['extents'], stride_bytes=atom['stride_bytes'],
                        base_offset_bytes=atom['base_offset_bytes']),
                    destination=dict(operand=atom['operand'], slot=state['slot'],
                        generation=state['generation'],
                        offset_in_operand_buffer=state['slot'] * 16384 + (8192 if name == 'B1' else 0)),
                    bytes=int(atom['payload_KiB'] * 1024), valid_bytes=int(atom['valid_KiB'] * 1024),
                    fill_bytes=int(atom['fill_KiB'] * 1024), issue=issue, completion=completion,
                    full_barrier=gp + 'full_ready'))
                completions.append(completion)
                issue_previous = issue
            producer_previous = issue_previous
            add(gp + 'full_ready', gp + 'expect_tx', *completions)
            add(gp + 'full_query', consumer_previous)
            add(gp + 'consumer_wait', gp + 'full_ready', gp + 'full_query')
            add(gp + 'compute_batch', gp + 'consumer_wait')
            add(gp + 'commit', gp + 'compute_batch')
            add(gp + 'compute_complete', gp + 'compute_batch')
            if k:
                add(gp + 'wait1', gp + 'commit', f'g{g-1}.compute_complete')
                if separate_empty_completion:
                    consumer_previous = add(f'g{g-1}.release_emit', gp + 'wait1')
                    add(f'g{g-1}.release', consumer_previous)
                else:
                    consumer_previous = add(f'g{g-1}.release', gp + 'wait1')
            else:
                consumer_previous = gp + 'commit'
            groups.append(dict(**state, output=j, k=k, consumer_warpgroup=j % 2,
                expected_full_bytes=32768, compute_issue=gp + 'compute_batch',
                compute_completion=gp + 'compute_complete', release=gp + 'release'))
            if intervention.get('wgmma_repeats', 1) != 1:
                groups[-1]['wgmma_count'] = 8 * intervention['wgmma_repeats']
        last = (j + 1) * kt - 1
        add(prefix + 'math_handoff', consumer_previous)
        add(prefix + 'wait0', prefix + 'math_handoff', f'g{last}.compute_complete')
        if separate_empty_completion:
            add(f'g{last}.release_emit', prefix + 'wait0')
            add(f'g{last}.release', f'g{last}.release_emit')
            add(prefix + 'main_end', f'g{last}.release_emit')
        else:
            add(f'g{last}.release', prefix + 'wait0')
            add(prefix + 'main_end', f'g{last}.release')
        epi_pre = [prefix + 'main_end']
        if j == 0 and (epilogue_defer == 2 or (epilogue_defer == 1 and cta == 0)):
            epi_pre.append('j1.math_handoff')
        if j:
            epi_pre.append(f'j{j-1}.epilogue_handoff')
        add(prefix + 'epilogue_permit', *epi_pre)
        add(prefix + 'store', prefix + 'epilogue_permit')
        add(prefix + 'store_tail', prefix + 'store')
        add(prefix + 'epilogue_handoff', prefix + 'store_tail')
        observations.append(dict(output=j, coord=output['coord'], consumer_warpgroup=j % 2,
            nodes=[f'g{j*kt}.consumer_wait', prefix + 'main_end',
                   prefix + 'epilogue_permit', prefix + 'store_tail']))
    # trace 的退出戳来自消费者；producer tail 单列为程序结束的必要条件。
    add('consumer_exit', *([f'j{o["j"]}.epilogue_handoff' for o in outputs] or ['entry']))
    last_groups = groups[-stages:]
    add('producer_tail', producer_previous, *(g['release'] for g in last_groups))
    register_nodes = [f'register_{"dec" if warp < 4 else "inc"}.w{warp}' for warp in range(12)] if register_quota else []
    add('program_end', 'consumer_exit', 'producer_tail', *register_nodes)
    missing = {p for predecessors in nodes.values() for p in predecessors} - nodes.keys()
    if missing:
        raise ValueError('missing program predecessor: ' + ', '.join(sorted(missing)))
    order = list(TopologicalSorter(nodes).static_order())
    graph = dict(schema='gh200-cfg-b-cta-graph/1', cta=cta, nodes=nodes, topological_order=order,
        groups=groups, requests=requests, observations=observations,
        trace_exit='consumer_exit', program_end='program_end',
        service_nodes=dict(tma_completion=[r['completion'] for r in requests],
            compute_completion=[g['compute_completion'] for g in groups],
            output=[f'j{o["j"]}.store_tail' for o in outputs]))
    if register_quota:
        graph['service_nodes']['register_quota'] = register_nodes
    if 'output_tma_enabled' in program:
        graph['output_tma_enabled'] = program['output_tma_enabled'][cta]
    return graph


def register_demands(program):
    """冻结S4/S6构建的资源需求；时间与CTA池仲裁由服务模型提供。"""
    accumulators = program['tile'][0]*program['tile'][1]//128
    demand = cfg_b_quota_requirements()
    warps = demand['warps']
    demand.update(warp_count=len(warps),
        ptxas_registers_per_thread=warps[0]['initial_registers_per_thread'],
        cta_reserved_registers=demand['initial_pool_registers'],
        accumulator_fp32_per_consumer_thread=accumulators,
        roles=[dict(role=warps[4*group]['role'],warps=list(range(4*group,4*group+4)),
                    target_registers_per_thread=warps[4*group]['target_registers_per_thread'])
               for group in range(3)],
        stack_bytes=0, spill_store_bytes=0, spill_load_bytes=0,
        transfers_per_cta=dict(producer_release=sum(w['released_registers'] for w in warps),
                              consumer_acquire=sum(w['needed_registers'] for w in warps)),
        coverage=dict(producer_release='producer warp0 dec within ENTRY to FIRST_WORK prefix',
            remaining_producer_release='other producer warps startup; separately unpriced',
            consumer_acquire='startup before each WG first compute; separately unpriced',
            accumulator_update='compute_ns', accumulator_consumer='epilogue_prepare_ns'),
        qualification='frozen cfg_b S4/S6 build and source facts; no physical RF traffic count')
    return demand

def ordinary_lsu_demands(program):
    """冻结r18直接alpha标量条件下的普通访存需求，静态计数来自主kernel。"""
    s6 = program['stages'] == 6
    return dict(schema='gh200-cfg-b-ordinary-lsu-demands/1',
        scope=dict(alpha=1., beta=0., alpha_ptr=None, alpha_batch_stride=0),
        ordinary_global=dict(static_scalar_ldg_sites=2, opcode='LDG.E', bytes_per_lane=4,
            active_loads=0, active_stores=0, active_payload_bytes=0,
            cache_policy='source scalar dereference; no explicit cache modifier',
            derivation='null pointer selects direct scalar; zero batch stride reuses scalar'),
        local=dict(static_ldl=0, static_stl=0, stack_bytes=0, spill_load_bytes=0, spill_store_bytes=0),
        constants=dict(static_opcodes={'LDC':36 if s6 else 6, 'LDC.64':8 if s6 else 3,
            'ULDC':21 if s6 else 51, 'ULDC.64':14 if s6 else 19}, address_space='c[0x0]',
            coverage='prefix, work-coordinate/control and existing composite services'),
        matrix_payload=dict(A_B='TMA input source/destination', D='SMEM output + TMA store'),
        qualification='frozen S4/S6 source/build and branch derivation; no runtime counter count')


def smem_demands(program):
    """每输出的逻辑需求与现有复合服务覆盖关系；价格独立于程序。"""
    outputs = sum(map(len, program['outputs']))
    output_tma_enabled = program.get('output_tma_enabled', [True] * len(program['outputs']))
    tma_outputs = sum(len(work) for cta, work in enumerate(program['outputs'])
                      if output_tma_enabled[cta])
    ktiles = program['ktiles']
    return dict(schema='gh200-cfg-b-smem-demands/1', output_tiles=outputs, ktiles=ktiles,
        paths=[
            dict(path='tma_input_destination', per_ktile_requests_bytes=[16384,8192,8192],
                total_bytes=outputs*ktiles*32768, covered_by='input ready; optional input_destination_service'),
            dict(path='wgmma_operand_references', opcode='HGMMA.64x128x16.F32.tnspB',
                instructions_per_ktile=8*program.get('runtime_intervention', {}).get('wgmma_repeats', 1),
                a_bytes_per_instruction=2048, b_bytes_per_instruction=4096,
                total_operand_reference_bytes=outputs*ktiles*49152*program.get('runtime_intervention', {}).get('wgmma_repeats', 1), covered_by='compute_ns',
                interpretation='logical operand references; physical SMEM reads may be reused'),
            dict(path='ordinary_output_store', opcode='STS.64', threads=128,
                instructions_per_thread_per_subtile=8, bytes_per_lane=8, subtiles_per_output=8,
                bytes_per_subtile=8192, total_bytes=outputs*65536, covered_by='epilogue_write_ns',
                buffers=2, buffer_distance_bytes=8192),
            dict(path='tma_output_source', per_output_requests=8, bytes_per_request=8192,
                total_bytes=tma_outputs*65536, covered_by='output service source_done'),
        ])


def epilogue_smem_addresses(subtile):
    """冻结cfg_b的8条STS.64：按指令→128线程返回起始word地址。

    SASS地址链与冻结CuTe partition_D相互核对；两个源槽相距8192 B。
    """
    rows = [[] for _ in range(8)]
    for thread in range(128):
        a, b, c = thread & 3, (thread >> 2) & 7, thread >> 5
        f = 512*c + 32*b + 4*((a >> 1) ^ b) + 2*(a & 1)
        base = 0x400 + 4*f
        delta = -32 if b & 2 else 32
        high = 0x3c0 if b & 4 else 0x440
        addresses = (base+0x400, base+0x800, base+delta+0x400, base+delta+0x800,
                     base+high, base+high+0x400, base+high+delta, base+high+delta+0x400)
        for row, address in zip(rows, addresses):
            row.append((address + (subtile % 2)*8192)//4)
    return rows
