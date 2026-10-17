"""cfg_c1 单 CTA cooperative 程序：源请求、槽代次、逐 warp 到达与观察端点。"""
import math
import case_inputs
import output_model as output
import supply_model as supply

EVENTS = ('FIRST_MMA', 'MAIN_END', 'EPI_PERMIT', 'EPI_DONE')
MARKERS = ('FIRST', 'MAIN', 'PERMIT', 'DONE')


def build_program(row, setup, source_identity=None):
    effective, errors = case_inputs.resolve(row, setup)
    if errors:
        raise ValueError('; '.join(errors))
    missing = case_inputs.missing_for(effective, 'program_graph')
    if missing:
        raise ValueError('missing program input: ' + ', '.join(missing))
    if effective['config'] != 'cfg_c1' or effective['stages'] != 4:
        raise ValueError('cfg_c1 S4 build required')
    if any(effective[k] % 64 for k in ('m', 'n', 'k')) or effective['k'] < 128:
        raise ValueError('M/N/K multiples of 64 and at least two Ktiles required')
    if effective['interventions']:
        raise ValueError('unexpanded intervention: ' + ', '.join(effective['interventions']))
    inputs = {k: v for k, v in effective.items() if k not in ('setup', 'interventions')}
    ledger = supply.supply_work_ledger(inputs, setup)
    work = [[] for _ in range(math.prod(setup['grid']))]
    outputs = [[] for _ in work]
    for item in ledger['outputs']:
        work[item['cta']].append(item['coord'])
        outputs[item['cta']].append(item)
    if any(not x for x in work):
        raise ValueError('idle CTA lifecycle has no binding')
    return dict(schema='gh200-cfg-c1-program/1', case=inputs, setup=dict(setup),
        source_identity=source_identity or {}, tile=[256, 128, 64], cluster=[1, 1],
        stages=4, ktiles=ledger['ktiles'], work=work, outputs=outputs,
        source_atoms=ledger['source_atoms'], totals=ledger['totals'],
        full_transaction_bytes=49152, empty_arrival_count=256,
        input_path=dict(source='device global A/B', destination='local CTA SMEM',
            requests=['A 32KiB', 'B0 8KiB', 'B1 8KiB'], multicast=False),
        output_path=dict(source='FP32 accumulators', intermediate='epilogue SMEM', destination='global D'))


def expand_cta(program, cta):
    """每 warp 的到达计数为32；到达可见服务保留独立端点。"""
    kt, stages = program['ktiles'], program['stages']
    count = len(program['work'][cta])
    total = kt * count
    nodes = {'cta_initialized': []}
    requests, groups, observations, routes, subtiles, tails = [], [], [], [], [], []

    def add(name, *parents):
        nodes[name] = list(dict.fromkeys(nodes.get(name, []) + list(parents)))

    def arrival(g, w, warp):
        return f'r0.w{w}.g{g}.warp{warp}.empty_visible'

    for w in range(2):
        for warp in range(4):
            first = 128 + 128*w + 32*warp
            routes.append(dict(consumer_warpgroup=w, warp_in_wg=warp,
                physical_threads=list(range(first, first+32)), arrival_count=32,
                emit_template=f'r0.w{w}.g{{group}}.warp{warp}.release_emit',
                visibility_template=f'r0.w{w}.g{{group}}.warp{warp}.empty_visible'))
    for g in range(total):
        j, k = divmod(g, kt)
        state = dict(supply.input_slot_state(g, stages), output=j, k=k,
            expected_full_bytes=49152, expected_empty_arrivals=256)
        groups.append(state)
        prefix = f'r0.g{g}.'
        previous = [f'r0.g{g-1}.B1_issue'] if g else ['cta_initialized']
        reuse = [arrival(g-stages,w,warp) for w in range(2) for warp in range(4)] if g >= stages else []
        add(prefix+'acquire', *previous, *reuse)
        add(prefix+'expect_tx', prefix+'acquire')
        entry = program['outputs'][cta][j]
        atom_ids = entry['operands']['A']['source_atoms'] + entry['operands']['B']['source_atoms']
        delivered = []
        for index, (label, atom_id) in enumerate(zip(('A','B0','B1'), atom_ids)):
            atom = program['source_atoms'][atom_id]
            assert atom['cta'] == cta and atom['destinations'] == [dict(cta=cta,j=j,rank=0)]
            prior = prefix+'expect_tx' if index == 0 else prefix+('A','B0')[index-1]+'_issue'
            issue, completion = prefix+label+'_issue', prefix+label+'_delivered'
            add(issue, prior)
            add(completion, issue)
            delivered.append(completion)
            requests.append(dict(atom=label, output=j, k=k, slot=g%stages, generation=g//stages,
                source_atom_id=atom_id, source_cta=cta, destination_cta=cta,
                bytes=int(atom['payload_KiB']*1024), valid_bytes=int(atom['valid_KiB']*1024),
                fill_bytes=int(atom['fill_KiB']*1024), issue=issue, completion=completion,
                origin=[atom['origin'][0],atom['origin'][1]+64*k], shape=atom['shape'],
                stride_bytes=atom['stride_bytes'], extents=atom['extents']))
        add(prefix+'full', prefix+'expect_tx', *delivered)
        for w in range(2):
            cg = f'r0.w{w}.g{g}.'
            waits = []
            for warp in range(4):
                if k == 0:
                    prior = [f'r0.w{w}.t{j-1}.DONE' if warp == 0 else f'r0.w{w}.t{j-1}.warp{warp}.output_return'] if j else []
                elif k == 1:
                    prior = [f'r0.w{w}.g{g-1}.commit']
                else:
                    prior = [f'r0.w{w}.g{g-2}.warp{warp}.release_emit']
                event = cg+f'warp{warp}.wait_return'
                add(event, prefix+'full', *prior)
                waits.append(event)
            if k == 0:
                marker = f'r0.w{w}.t{j}.FIRST'
                add(marker, waits[0])
                waits.append(marker)
            add(cg+'commit', *waits)
            add(cg+'retired', cg+'commit', *([f'r0.w{w}.g{g-1}.retired'] if k else []))
            for warp in range(4):
                ready = ([f'r0.w{w}.g{g+1}.commit',cg+'retired'] if k < kt-1 else
                    [f'r0.w{w}.g{g-1}.warp{warp}.release_emit',cg+'retired'])
                emit = cg+f'warp{warp}.release_emit'
                add(emit, *ready)
                add(arrival(g,w,warp), emit)
            if k == kt-1:
                add(f'r0.w{w}.t{j}.MAIN', cg+'warp0.release_emit')
    local = output.cooperative_output_source_dependencies('cfg_c1', count)
    for name, parents in local['nodes'].items():
        add(name, *parents)
    for j in range(count):
        for w in range(2):
            done = f'r0.w{w}.t{j}.DONE'
            for warp in range(4):
                add(f'r0.w{w}.t{j}.warp{warp}.output_return', *nodes[done])
            for event, marker in zip(EVENTS, MARKERS):
                observations.append(dict(cta=cta, consumer_warpgroup=w, output=j,
                    event=event, node=f'r0.w{w}.t{j}.{marker}'))
        geometry = output.cooperative_output_valid_subtiles('cfg_c1',program['case']['m'],program['case']['n'],program['work'][cta][j],j)
        subtiles.extend(geometry['subtiles'])
    for w in range(2):
        observations.append(dict(cta=cta,consumer_warpgroup=w,event='FINAL',node=f'r0.w{w}.FINAL'))
    previous = f'r0.g{total-1}.B1_issue'
    for step in range(stages):
        g = total + step
        released = g - stages
        event = f'r0.producer_tail_wait{step}'
        parents = [arrival(released,w,warp) for w in range(2) for warp in range(4)] if released >= 0 else ['cta_initialized']
        add(event, previous, *parents)
        tails.append(dict(event=event,slot=g%stages,generation=g//stages,release_group=released if released>=0 else None))
        previous = event
    add('source_safe_exit', previous, 'r0.w0.FINAL', 'r0.w1.FINAL',
        *(f'r0.w{w}.t{count-1}.warp{warp}.output_return' for w in range(2) for warp in range(4)))
    add('cuda_complete', 'source_safe_exit', 'r0.D_visible')
    order, _ = output.dependency_order(dict(nodes=nodes))
    return dict(schema='gh200-cfg-c1-cta-graph/1',cta=cta,nodes=nodes,topological_order=order,
        groups=groups,requests=requests,observations=observations,empty_arrival_routes=routes,
        output_subtiles=subtiles,producer_tail_waits=tails,
        completion_ports=dict(source_safe_exit='source_safe_exit',global_D_visible='r0.D_visible',cuda_complete='cuda_complete'))


def bind_cta_trace(program, graph, observed_cta):
    cta = graph['cta']
    expected = [tuple(coord) for coord in program['work'][cta]]
    if observed_cta['cta'] != cta or observed_cta['work'] != expected or observed_cta['role_work'] != [expected,expected]:
        raise ValueError('CTA or consumer work differs from program')
    values = []
    for endpoint in graph['observations']:
        role = observed_cta['roles'][endpoint['consumer_warpgroup']]
        if endpoint['event'] == 'FINAL':
            cycle, ns = role['final_c'], role['final_ns']
        else:
            j = endpoint['output'];position = EVENTS.index(endpoint['event'])
            cycle, ns = role['tiles'][j][position], role['tiles_ns'][j][position]
        values.append(dict(endpoint,cycle=cycle,ns=ns,sm=observed_cta['sm']))
    return values
