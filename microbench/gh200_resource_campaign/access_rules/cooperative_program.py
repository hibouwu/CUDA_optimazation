"""将现有 cooperative 源请求与 rank 图绑定为无服务价格的程序实例。"""
import math

import case_inputs
import output_model as output
import supply_model as supply


TRACE_EVENTS = ('FIRST_MMA', 'MAIN_END', 'EPI_PERMIT', 'EPI_DONE')
GRAPH_MARKERS = ('FIRST', 'MAIN', 'PERMIT', 'DONE')


def cfg_a_template(stages, ktiles, output_tiles):
    """复用 cfg_a 依赖，展开源码 producer_tail 的逐槽等待。"""
    graph = supply.cfg_a_atomic_source_graph(output.cfg_a_rank_dependencies(stages, ktiles, output_tiles))
    nodes = graph['nodes']
    nodes['cluster_initialized'] = []
    tail_waits = []
    total = ktiles * output_tiles
    for rank in range(2):
        nodes[f'r{rank}.g0.acquire'].append('cluster_initialized')
        previous = f'r{rank}.g{total-1}.B_issue'
        for step in range(stages):
            count = total + step
            released = count - stages
            event = f'r{rank}.producer_tail_wait{step}'
            arrivals = [route['visibility_token_template'].format(group=released)
                for route in graph['empty_arrival_routes'] if route['destination_rank'] == rank]
            nodes[event] = [previous] + (arrivals if released >= 0 else ['cluster_initialized'])
            tail_waits.append(dict(rank=rank, step=step, slot=count % stages, generation=count // stages,
                empty_wait_phase=1 ^ (count // stages % 2), release_group=released if released >= 0 else None,
                initial_empty=released < 0, event=event))
            previous = event
        nodes[f'r{rank}.producer_tail'] = [previous]
    graph['producer_tail_waits'] = tail_waits
    return graph


def build_program(row, setup, source_identity=None):
    """覆盖 cfg_a S4/S6的64倍数边界、cfg_c S4/K1024完整输出；无空闲 cluster。"""
    effective, errors = case_inputs.resolve(row, setup)
    if errors:
        raise ValueError('; '.join(errors))
    missing = case_inputs.missing_for(effective, 'program_graph')
    if missing:
        raise ValueError('missing program input: ' + ', '.join(missing))
    config, stages, k = effective['config'], effective['stages'], effective['k']
    if ((config, stages, k) != ('cfg_c', 4, 1024)
        and not (config == 'cfg_a' and stages in (4, 6) and k >= 128 and k % 64 == 0)):
        raise ValueError('cooperative program supports cfg_c S4/K1024 or cfg_a S4/S6/whole Ktiles>=2')
    if effective['interventions']:
        raise ValueError('unexpanded intervention: ' + ', '.join(effective['interventions']))
    tile, cluster_shape = ([128, 128, 64], [2, 1]) if config == 'cfg_a' else ([256, 128, 64], [1, 2])
    if config == 'cfg_a' and (effective['m'] % 64 or effective['n'] % 64):
        raise ValueError('cfg_a M/N multiples of 64 required')
    if config == 'cfg_c' and (effective['m'] % (tile[0] * cluster_shape[0]) or effective['n'] % (tile[1] * cluster_shape[1])):
        raise ValueError(f'complete {config} cluster output tiles required')
    inputs = {k: v for k, v in effective.items() if k not in ('setup', 'interventions')}
    binding = (supply.cfg_a_source_bindings if config == 'cfg_a' else supply.cfg_c_source_bindings)(inputs, setup)
    ktiles = binding['k_tiles']
    if binding['idle_clusters']:
        raise ValueError('idle cluster lifecycle has no program binding')
    templates = {}
    work = [[] for _ in range(math.prod(setup['grid']))]
    for cluster in binding['clusters']:
        count = cluster['output_tiles']
        key = f'K{ktiles}-T{count}'
        if key not in templates:
            templates[key] = (cfg_a_template(stages, ktiles, count) if config == 'cfg_a' else
                output.cfg_c_rank_dependencies(supply.cfg_c_source_port_graph(4, 16, count)))
        for rank, cta in enumerate(cluster['ctas']):
            work[cta] = [list(o['coord']) for o in cluster['rank_outputs'][rank]]
    return dict(schema='gh200-cooperative-program/1', case=inputs, setup=dict(setup),
        source_identity=source_identity or {}, schedule='cooperative', tile=tile,
        cluster_shape=cluster_shape, stages=stages, ktiles=ktiles, work=work,
        clusters=binding['clusters'], templates=templates, totals=binding['totals'],
        descriptor=dict(tma_rank=3, value_type='FP16', smem_swizzle='128B'),
        input_path=dict(source='device global A/B allocation', destination='cluster CTA SMEM',
            A='each rank issues local A16KiB' if config == 'cfg_a' else 'each rank issues A16KiB multicast to both ranks',
            B='each rank issues B0/B1,4KiB each multicast to both ranks' if config == 'cfg_a' else 'each rank issues local B0/B1,8KiB each'),
        output_path=dict(source='FP32 accumulators', intermediate='CTA epilogue SMEM', destination='global D'),
        unresolved=['Serialized tensor-map fields and SMEM byte addresses are absent from the saved setup.',
            'The inherited partial order does not bind B0/B1 labels to their two compiled issue PCs.',
            'Cluster entry, request/retirement/visibility service durations and cycle-to-ns mapping remain external.'])


def expand_cluster(program, cluster_index):
    """实例化一个物理 cluster；节点名跨 cluster 唯一，模板前驱保持原样。"""
    cluster = program['clusters'][cluster_index]
    template = program['templates'][f"K{program['ktiles']}-T{cluster['output_tiles']}"]
    prefix = f'c{cluster_index}.'
    name = lambda node: prefix + node
    nodes = {name(node): [name(p) for p in parents] for node, parents in template['nodes'].items()}
    requests, deliveries, observations, subtiles = [], [], [], []
    groups = template['groups']
    for request in template['input_requests']:
        group, rank = request['group'], request['source_rank']
        state = groups[group]
        entry = cluster['rank_outputs'][rank][state['tile']]
        atom = next(a for a in entry['sources'] if a['atom'] == request['atom'])
        origin = [atom['origin'][0], atom['origin'][1] + 64 * state['k']]
        requests.append(dict(request, id=name(request['issue']), source_cta=cluster['ctas'][rank],
            source_atom_id=atom['source_atom_id'], output=state['tile'], k=state['k'],
            logical_source=dict(operand=atom['operand'], origin=origin, shape=atom['shape'],
                extents=atom['extents'], stride_bytes=atom['stride_bytes'], base_offset_bytes=atom['base_offset_bytes']),
            valid_bytes=int(atom['valid_KiB'] * 1024), fill_bytes=int(atom['fill_KiB'] * 1024),
            multicast_mask=atom['multicast_mask'], destination_ctas=[cluster['ctas'][r] for r in atom['destination_ranks']],
            issue=name(request['issue']), source_ready=name(request['source_ready'])))
    for delivery in template['input_deliveries']:
        state = groups[delivery['group']]
        rank = delivery['destination_rank']
        deliveries.append(dict(delivery, source_cta=cluster['ctas'][delivery['source_rank']],
            destination_cta=cluster['ctas'][rank], output=state['tile'], k=state['k'],
            slot=state['slot'], generation=state['generation'], full_wait_phase=state['full_wait_phase'],
            source_ready=name(delivery['source_ready']), completion=name(delivery['completion']),
            full_barrier=name(f"r{rank}.g{delivery['group']}.full")))
    for rank, cta in enumerate(cluster['ctas']):
        for entry in cluster['rank_outputs'][rank]:
            j = entry['j']
            for wg in range(2):
                for event, marker in zip(TRACE_EVENTS, GRAPH_MARKERS):
                    observations.append(dict(cta=cta, consumer_warpgroup=wg, output=j, coord=entry['coord'],
                        event=event, node=name(f'r{rank}.w{wg}.t{j}.{marker}')))
            geometry = output.cooperative_output_valid_subtiles(program['case']['config'], program['case']['m'],
                program['case']['n'], entry['coord'], j)
            for subtile in geometry['subtiles']:
                q = subtile['subtile']
                subtiles.append(dict(subtile, cta=cta, rank=rank, output=j,
                    store_issue=name(f'r{rank}.d{q}.store_issue'),
                    source_read_complete=name(f'r{rank}.d{q}.read_retired'),
                    destination_visible=name(f'r{rank}.d{q}.D_visible')))
        for wg in range(2):
            observations.append(dict(cta=cta, consumer_warpgroup=wg, event='FINAL',
                node=name(f'r{rank}.w{wg}.FINAL')))
    order, _ = output.dependency_order(dict(nodes=nodes))
    return dict(schema='gh200-cooperative-cluster-graph/1', cluster_index=cluster_index,
        cluster=cluster['cluster'], ctas=cluster['ctas'], nodes=nodes, topological_order=order,
        groups=groups, requests=requests, deliveries=deliveries, output_subtiles=subtiles, observations=observations,
        empty_arrival_routes=[dict(route, source_cta=cluster['ctas'][route['source_rank']],
            destination_cta=cluster['ctas'][route['destination_rank']],
            emit_token_template=name(route['emit_token_template']),
            visibility_token_template=name(route['visibility_token_template'])) for route in template['empty_arrival_routes']],
        producer_tail_waits=[dict(wait, event=name(wait['event'])) for wait in template['producer_tail_waits']],
        completion_ports=dict(source_safe_exit=name('source_safe_exit'),
            global_D_visible=[name('r0.D_visible'), name('r1.D_visible')], cuda_complete=name('cuda_complete')),
        service_nodes=dict(source_ready=[r['source_ready'] for r in requests],
            delivery_complete=[d['completion'] for d in deliveries],
            compute_retired=[name(n) for n in template['nodes'] if n.endswith('.retired')],
            empty_arrival=[name(n) for n in template['nodes'] if '.empty_to_r' in n],
            output_source_read=[d['source_read_complete'] for d in subtiles],
            output_visible=[d['destination_visible'] for d in subtiles]))


def bind_trace(program, observed):
    """核对两个消费者的实际工作列表；保留各角色自己的 cycle/ns 端点。"""
    if observed['case'] != program['case'].get('id'):
        raise ValueError('trace case differs from the program')
    ctas = {c['cta']: c for c in observed['ctas']}
    if set(ctas) != set(range(len(program['work']))):
        raise ValueError('trace CTA identities differ from the program')
    for cta, work in enumerate(program['work']):
        expected = [tuple(coord) for coord in work]
        if ctas[cta]['work'] != expected or ctas[cta]['role_work'] != [expected, expected]:
            raise ValueError(f'trace work differs at CTA {cta}')
    bound = []
    for index in range(len(program['clusters'])):
        graph = expand_cluster(program, index)
        values = []
        for endpoint in graph['observations']:
            cta = ctas[endpoint['cta']]
            role = cta['roles'][endpoint['consumer_warpgroup']]
            if endpoint['event'] == 'FINAL':
                cycle, ns = role['final_c'], role['final_ns']
            else:
                j = endpoint['output']
                position = TRACE_EVENTS.index(endpoint['event'])
                cycle, ns = role['tiles'][j][position], role['tiles_ns'][j][position]
            values.append(dict(endpoint, cycle=cycle, ns=ns, sm=cta['sm']))
        bound.append(dict(cluster_index=index, observations=values))
    return bound
