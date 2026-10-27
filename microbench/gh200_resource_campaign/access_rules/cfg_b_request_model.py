"""将已有请求服务候选绑定到共同 cfg_b 程序；全部服务价格由调用方以 ns 给出。"""
import argparse
import json
from pathlib import Path

import cfg_b_composition as aggregate
import cfg_b_program
import cfg_b_phase_frequency as phase_frequency
from register_quota import CfgBRegisterQuota
from sm_activity import activity_windows
import supply_model as supply
import tma_tensor_multicta as tensor


EXAMPLES = aggregate.DEFAULT_ARCHIVE / 'reanalysis/candidate-development-v1/B-source-candidate-examples.json'


def source_key(source):
    return (source['operand'], 'fp16', 'SW128', tuple(source['extents']),
            tuple(source['stride_bytes']), source['base_offset_bytes'],
            tuple(source['origin']), tuple(source['shape']))


def request_inputs(program):
    """从共同实例读取工作及源身份，保持引擎的完整 tile、活跃 CTA 适用域。"""
    counts = [len(outputs) for outputs in program['outputs']]
    if not all(counts) or program['ktiles'] < 2:
        raise ValueError('request candidate requires active CTAs and at least two Ktiles')
    keys = []
    for outputs in program['outputs']:
        groups = []
        for output in outputs:
            if output['output']['valid_KiB'] != output['output']['nominal_KiB']:
                raise ValueError('request candidate requires complete output tiles')
            atom_ids = output['operands']['A']['source_atoms'] + output['operands']['B']['source_atoms']
            atoms = [program['source_atoms'][i] for i in atom_ids]
            if any(a['fill_KiB'] or a['multicast_mask'] != 1 for a in atoms):
                raise ValueError('request candidate requires valid unicast input boxes')
            for k in range(program['ktiles']):
                groups.append(tuple(source_key(dict(a, origin=[a['origin'][0], a['origin'][1] + 64*k]))
                                    for a in atoms))
        keys.append(groups)
    return keys, counts


class CfgBTensorSource:
    """同一源服务接收 GEMM 的实际地址；不带入探针的控制价格。"""

    def __init__(self, program, keys, candidate, parameters):
        self.keys = keys
        self.parameters = parameters
        costs = {name: parameters[name] for name in ('service', 'latency', 'resident_service',
                 'resident_latency', 'source_partitions', 'cache_capacity_bytes') if name in parameters}
        self.service = tensor.TensorSourceService(candidate, len(keys),
            program['totals']['distinct_valid_source_KiB']*1024, **costs)
        self.weights = {}

    def __call__(self, t, cta, group, atom, weight):
        key = self.keys[cta][group][atom]
        if self.service.partitioned and key not in self.weights:
            operand, _, _, _, strides, base, origin, shape = key
            first = base+sum(a*b for a, b in zip(origin, strides))
            row_axis = 0 if operand == 'A' else 1
            partitions = self.service.partitions
            self.weights[key] = tensor.row_partition_weights(first % (128*partitions),
                strides[row_axis] % (128*partitions), shape[row_axis], partitions)
        busy_before = self.service.busy
        served = self.service.submit(t, cta, key, weight, self.weights.get(key))
        return dict(served, source_key=key, source_busy_ns=self.service.busy-busy_before)


class PrefixSyncService:
    """producer 原点换算；未设置首次进入时刻时从共同入口等待。"""

    def __init__(self, service, prefix_ns):
        self.service = service
        self.prefix_ns = prefix_ns

    def full_wait(self, enter_ns, ready_ns, cta, group):
        common_entry = group == 0 and enter_ns == 0.
        entered = 0. if common_entry else enter_ns + self.prefix_ns
        window = self.service.full_wait(entered, ready_ns + self.prefix_ns, cta, group)
        shifted = dict(window)
        for field in ('enter_ns', 'ready_ns', 'eligibility_ns', 'wait_return_ns'):
            shifted[field] -= self.prefix_ns
        if window['return_cost_ns'] == 0:
            # A zero-price hook must retain the engine's original bits. Adding
            # and subtracting the prefix can otherwise change a time by 1 ULP.
            shifted['enter_ns'] = -self.prefix_ns if common_entry else enter_ns
            shifted['ready_ns'] = ready_ns
            shifted['eligibility_ns'] = max(shifted['enter_ns'], ready_ns)
            shifted['wait_return_ns'] = shifted['eligibility_ns']
        return shifted

    def release(self, t, cta, group):
        entered = t + self.prefix_ns
        returned = self.service.release(entered, cta, group)
        return t if returned == entered else returned - self.prefix_ns


def bind_cta(graph, actor, deliveries, *, producer_prefix_ns=0.):
    """服务端点赋值，其余控制节点取前驱最晚时刻；逐边核对软件依赖。"""
    times = {'entry': 0., 'producer_first_work': producer_prefix_ns}
    requests = []
    for request in graph['requests']:
        delivery = dict(deliveries[request['group'], request['atom']])
        for field in ('arrival_ns', 'source_ready_ns', 'issue_return_ns', 'start_ns', 'service_end_ns', 'ready_ns',
                      'input_path_start_ns', 'input_path_end_ns', 'destination_arrival_ns',
                      'input_earliest_destination_ns'):
            if field in delivery:
                delivery[field] += producer_prefix_ns
        times[request['issue']] = delivery['arrival_ns']
        times[request['completion']] = delivery['ready_ns']
        requests.append(dict(request, service=delivery))
    for group in graph['groups']:
        g = group['group']
        if 'producer_acquired' in actor:
            times[f'g{g}.acquire'] = actor['producer_acquired'][g] + producer_prefix_ns
        query = actor['consumer_query'][g]
        times[f'g{g}.full_query'] = 0. if g == 0 and query == 0. else query + producer_prefix_ns
        for node, field in [('consumer_wait', 'starts'), ('compute_batch', 'starts'),
                            ('commit', 'commits'), ('compute_complete', 'completed'),
                            ('release', 'released')]:
            times[f'g{g}.{node}'] = actor[field][g] + producer_prefix_ns
        if 'full_wait_windows' in actor:
            times[f'g{g}.full_query'] = actor['full_wait_windows'][g]['enter_ns'] + producer_prefix_ns
            times[f'g{g}.consumer_wait'] = actor['full_wait_windows'][g]['wait_return_ns'] + producer_prefix_ns
        if group['k']:
            times[f'g{g}.wait1'] = actor['wait1_return'][g] + producer_prefix_ns
        if f'g{g}.release_emit' in graph['nodes']:
            times[f'g{g}.release_emit'] = actor['release_emitted'][g] + producer_prefix_ns
    output_service = []
    for j, output in enumerate(actor['outputs']):
        for node, value in [('math_handoff', actor['mma_returns'][j]),
                            ('wait0', actor['main_ends'][j]), ('main_end', actor['main_ends'][j]),
                            ('epilogue_permit', output['begin']), ('store', output['begin']),
                            ('store_tail', output['end']), ('epilogue_handoff', output['end'])]:
            times[f'j{j}.{node}'] = value + producer_prefix_ns
        shifted = dict(output, begin=output['begin'] + producer_prefix_ns,
                       end=output['end'] + producer_prefix_ns)
        for field in ('prepare_begin', 'prepare_end', 'write_begin', 'write_end',
                      'submitted', 'source_done'):
            shifted[field] = [t + producer_prefix_ns for t in output[field]]
        output_service.append(shifted)
    derived = []
    edges = 0
    for node in graph['topological_order']:
        predecessors = graph['nodes'][node]
        earliest = max((times[p] for p in predecessors), default=0.)
        if node not in times:
            times[node] = earliest
            derived.append(node)
        if times[node] < earliest:
            raise ValueError(f'CTA {graph["cta"]}: service violates {node} prerequisites')
        edges += len(predecessors)
    bound = dict(cta=graph['cta'], node_times_ns=times, control_nodes_at_predecessor_max=derived,
        requests=requests, groups=graph['groups'], output_service=output_service, dependency_edges_checked=edges,
        observations=[dict(o, times_ns=[times[n] for n in o['nodes']]) for o in graph['observations']],
        consumer_exit_ns=times[graph['trace_exit']], program_end_ns=times[graph['program_end']])
    if 'full_wait_windows' in actor:
        bound['full_wait_windows'] = []
        for window in actor['full_wait_windows']:
            shifted = dict(window)
            for field in ('enter_ns', 'ready_ns', 'eligibility_ns', 'wait_return_ns'):
                shifted[field] += producer_prefix_ns
            bound['full_wait_windows'].append(shifted)
    return bound


def critical_path_analysis(graph, bound, actor, parameters, *, producer_prefix_ns=0.,
                           target='consumer_exit', tol=1e-7):
    """MPM时间与路径分解；时差只返回目标及其祖先节点。"""
    times = bound['node_times_ns']
    kt = len(graph['groups']) // max(len(bound['output_service']), 1) if bound['output_service'] else 0
    stages = max((g['slot'] for g in graph['groups']), default=-1) + 1
    producer_gap = parameters.get('producer_gap_ns', 0.)
    reuse_delay = parameters.get('reuse_delay_ns', 0.)
    issue_floor = parameters.get('issue_floor_ns', 0.)
    wait_overhead = parameters.get('wait_overhead_ns', 0.)
    return_overhead = parameters.get('return_overhead_ns', 0.)
    handoff_delay = parameters.get('handoff_delay_ns', 0.)
    entry_delay = parameters.get('consumer_entry_delay_ns', 0.)
    controls = parameters.get('producer_control_ns', (0.,0.,0.))
    control_a = controls[0]
    control_enabled = any(controls)
    phases = parameters.get('compute_phases_ns')

    def phase_param(j, name, default):
        if phases is not None and j < len(phases) and name in phases[j]:
            return phases[j][name]
        return parameters.get(name, default)

    deliveries = {(r['group'], r['atom']): r['service'] for r in bound['requests']}
    incoming = {node: [] for node in graph['topological_order']}

    def add_edge(u, v, weight, category):
        incoming[v].append((u, float(weight), category))

    for node in graph['topological_order']:
        if node in ('register_dec.w0', 'register_dec.w1', 'register_dec.w2', 'register_dec.w3',
                    'register_inc.w4', 'register_inc.w5', 'register_inc.w6', 'register_inc.w7',
                    'register_inc.w8', 'register_inc.w9', 'register_inc.w10', 'register_inc.w11'):
            add_edge('entry', node, 0., 'control')
        elif node == 'producer_first_work':
            pred = graph['nodes'][node][0]
            add_edge(pred, node, producer_prefix_ns, 'prefix')

    group_diagnostics = []
    for group in graph['groups']:
        g = group['group']
        j, k = group['output'], group['k']
        gp = f'g{g}.'
        prev_issue = 'producer_first_work' if g == 0 else f'g{g-1}.B1.issue'
        if g == 0:
            prod_gap = 0.
        else:
            b1 = deliveries[g - 1, 'B1']
            b1_ret = max(b1['arrival_ns'], b1.get('issue_return_ns', b1['arrival_ns']))
            prod_gap = (b1_ret - b1['arrival_ns']) + producer_gap
        prod_free = times[prev_issue] + prod_gap
        for p in graph['nodes'][gp + 'acquire']:
            weight = (prod_gap if p == prev_issue else reuse_delay) if control_enabled else 0.
            category = ('producer_issue' if p == prev_issue else 'slot_reuse') if control_enabled else 'control'
            add_edge(p, gp + 'acquire', weight, category)
        add_edge(gp + 'acquire', gp + 'expect_tx', 0., 'control')
        add_edge(gp + 'expect_tx', gp + 'A.issue', control_a,
                 'producer_issue' if control_enabled else 'control')
        add_edge(prev_issue, gp + 'A.issue', prod_gap + control_a, 'producer_issue')
        slot_ready = None
        if group['previous_group'] is not None:
            rel_node = f'g{group["previous_group"]}.release'
            slot_ready = times[rel_node] + reuse_delay
            add_edge(rel_node, gp + 'A.issue', reuse_delay + control_a, 'slot_reuse')
        if times[gp + 'A.issue'] > max(prod_free, slot_ready if slot_ready is not None else -1.) + control_a + tol:
            add_edge(prev_issue, gp + 'A.issue', times[gp + 'A.issue'] - times[prev_issue], 'epilogue_exclusion')

        add_edge(gp + 'A.issue', gp + 'B0.issue', times[gp + 'B0.issue'] - times[gp + 'A.issue'], 'producer_issue')
        add_edge(gp + 'B0.issue', gp + 'B1.issue', times[gp + 'B1.issue'] - times[gp + 'B0.issue'], 'producer_issue')
        for atom in ('A', 'B0', 'B1'):
            issue_n, comp_n = gp + f'{atom}.issue', gp + f'{atom}.complete'
            add_edge(issue_n, comp_n, times[comp_n] - times[issue_n], 'tma_supply')
        for p in graph['nodes'][gp + 'full_ready']:
            add_edge(p, gp + 'full_ready', 0., 'tma_supply' if p.endswith('.complete') else 'control')

        if k == 0:
            for p in graph['nodes'][f'j{j}.math_permit']:
                add_edge(p, f'j{j}.math_permit', 0., 'control')
            add_edge(f'j{j}.math_permit', gp + 'full_query', 0., 'control')
            if j == 0:
                cons_prev = 'entry'
                cons_delay = entry_delay + producer_prefix_ns if entry_delay else 0.
                cons_cat = 'prefix' if cons_delay else 'control'
                add_edge(cons_prev, gp + 'full_query', cons_delay, cons_cat)
                cons_free = times[cons_prev] + cons_delay
            else:
                mh = f'j{j-1}.math_handoff'
                add_edge(mh, gp + 'full_query', handoff_delay, 'math_handoff')
                cons_free = times[mh] + handoff_delay
                if j >= 2:
                    eh = f'j{j-2}.epilogue_handoff'
                    add_edge(eh, gp + 'full_query', 0., 'epilogue_wait')
                    cons_free = max(cons_free, times[eh])
        else:
            cons_prev = graph['nodes'][gp + 'full_query'][0]
            loop_adv = phase_param(j, 'loop_advance_ns', 0.)
            base_prev = actor['wait1_return'][g - 1] + producer_prefix_ns
            edge_w = (base_prev + loop_adv) - times[cons_prev]
            add_edge(cons_prev, gp + 'full_query', edge_w, 'consumer_loop')
            cons_free = base_prev + loop_adv

        add_edge(gp + 'full_query', gp + 'consumer_wait', 0., 'control')
        full_ret_w = times[gp + 'consumer_wait'] - times[gp + 'full_ready']
        if times[gp + 'full_ready'] >= cons_free - tol and full_ret_w >= -tol:
            add_edge(gp + 'full_ready', gp + 'consumer_wait', max(0., full_ret_w), 'input_wait')
        else:
            add_edge(gp + 'full_ready', gp + 'consumer_wait', 0., 'input_wait')
        if times[gp + 'consumer_wait'] > max(cons_free, times[gp + 'full_ready']) + tol:
            add_edge(gp + 'full_ready', gp + 'consumer_wait',
                     times[gp + 'consumer_wait'] - times[gp + 'full_ready'], 'sync_return')

        add_edge(gp + 'consumer_wait', gp + 'compute_batch',
                 times[gp + 'compute_batch'] - times[gp + 'consumer_wait'], 'consumer_loop')

        c_dur = phase_param(j, 'compute_ns', 0.)
        c_dur += actor.get('compute_extra_ns', [0.] * len(graph['groups']))[g]
        c_lead = phase_param(j, 'commit_lead_ns', 0.)
        c_lag = phase_param(j, 'completion_lag_ns', 0.)
        add_edge(gp + 'compute_batch', gp + 'compute_complete', c_dur + c_lag, 'wgmma_compute')
        add_edge(gp + 'compute_batch', gp + 'commit', max(issue_floor, c_dur - c_lead), 'wgmma_commit')
        if g > 0:
            prev_lag = phase_param((g - 1) // kt, 'completion_lag_ns', 0.)
            add_edge(f'g{g-1}.compute_complete', gp + 'compute_complete',
                     c_dur + c_lag - prev_lag, 'wgmma_compute')
            add_edge(f'g{g-1}.compute_complete', gp + 'commit',
                     c_dur - c_lead - prev_lag, 'wgmma_commit')

        if k > 0:
            add_edge(gp + 'commit', gp + 'wait1', wait_overhead, 'wgmma_wait')
            add_edge(f'g{g-1}.compute_complete', gp + 'wait1', wait_overhead, 'wgmma_wait')
            if f'g{g-1}.release_emit' in graph['nodes']:
                add_edge(gp + 'wait1', f'g{g-1}.release_emit', 0., 'slot_release')
                rel_delay = phase_param((g - 1) // kt, 'input_release_delay_ns', 0.)
                add_edge(f'g{g-1}.release_emit', f'g{g-1}.release', rel_delay, 'slot_release')
            else:
                add_edge(gp + 'wait1', f'g{g-1}.release', 0., 'slot_release')

        group_diagnostics.append(dict(
            group=g, output=j, k=k, slot=group['slot'],
            full_ready_ns=times[gp + 'full_ready'],
            consumer_free_ns=cons_free,
            input_stall_ns=max(0., times[gp + 'full_ready'] - cons_free),
            supply_slack_ns=max(0., cons_free - times[gp + 'full_ready']),
            slot_reuse_stall_ns=max(0., (slot_ready or 0.) - prod_free) if slot_ready is not None else 0.,
            slot_reuse_slack_ns=max(0., prod_free - slot_ready) if slot_ready is not None else 0.,
        ))

    for j in range(len(bound['output_service'])):
        prefix = f'j{j}.'
        last = (j + 1) * kt - 1
        mh_pred = graph['nodes'][prefix + 'math_handoff'][0]
        add_edge(mh_pred, prefix + 'math_handoff',
                 times[prefix + 'math_handoff'] - times[mh_pred], 'math_handoff')
        add_edge(prefix + 'math_handoff', prefix + 'wait0', 0., 'wgmma_wait')
        add_edge(f'g{last}.compute_complete', prefix + 'wait0', 0., 'wgmma_wait')
        if f'g{last}.release_emit' in graph['nodes']:
            add_edge(prefix + 'wait0', f'g{last}.release_emit', 0., 'slot_release')
            rel_delay = phase_param(j, 'input_release_delay_ns', 0.)
            add_edge(f'g{last}.release_emit', f'g{last}.release', rel_delay, 'slot_release')
            add_edge(f'g{last}.release_emit', prefix + 'main_end', 0., 'math_handoff')
        else:
            add_edge(prefix + 'wait0', f'g{last}.release', 0., 'slot_release')
            add_edge(f'g{last}.release', prefix + 'main_end', 0., 'math_handoff')
        for p in graph['nodes'][prefix + 'epilogue_permit']:
            cat = 'epilogue_wait' if p.endswith('.epilogue_handoff') else 'epilogue'
            add_edge(p, prefix + 'epilogue_permit', 0., cat)
        add_edge(prefix + 'epilogue_permit', prefix + 'store', 0., 'epilogue')
        add_edge(prefix + 'store', prefix + 'store_tail',
                 times[prefix + 'store_tail'] - times[prefix + 'store'], 'epilogue')
        add_edge(prefix + 'store_tail', prefix + 'epilogue_handoff', 0., 'epilogue')

    for p in graph['nodes']['consumer_exit']:
        add_edge(p, 'consumer_exit', 0., 'epilogue')
    for p in graph['nodes']['producer_tail']:
        add_edge(p, 'producer_tail', 0., 'slot_release')
    for p in graph['nodes']['program_end']:
        add_edge(p, 'program_end', 0., 'control')

    earliest = {}
    crit_pred = {}
    outgoing = {node: [] for node in graph['topological_order']}
    for v in graph['topological_order']:
        if not incoming[v]:
            earliest[v] = 0.
            crit_pred[v] = (None, 0., 'entry')
        else:
            best_u, best_w, best_cat, best_t = None, 0., None, -1e30
            for u, w, cat in incoming[v]:
                outgoing[u].append((v, w, cat))
                cand = earliest[u] + w
                # Prefer non-control edge on exact ties so attribution stays meaningful
                if cand > best_t + tol or (abs(cand - best_t) <= tol and best_cat == 'control' and cat != 'control'):
                    best_u, best_w, best_cat, best_t = u, w, cat, cand
            earliest[v] = best_t
            crit_pred[v] = (best_u, best_w, best_cat)
        if abs(earliest[v] - times[v]) > 1e-6:
            raise ValueError(f'MPM forward mismatch at CTA {graph["cta"]} {v}: '
                             f'earliest={earliest[v]} != bound={times[v]}')

    target_time = earliest[target]
    latest = {target: target_time}
    for u in reversed(graph['topological_order']):
        if u != target:
            limits = [latest[v] - w for v, w, _ in outgoing[u] if v in latest]
            if limits:
                latest[u] = min(limits)
    slack = {node: max(0., limit - earliest[node]) for node, limit in latest.items()}

    path = []
    cur = target
    breakdown = {}
    tma_sub = dict(tma_source_ns=0., tma_dest_queue_ns=0., tma_dest_service_ns=0.,
                   merged_requests=0, unmerged_requests=0)
    while cur is not None:
        u, w, cat = crit_pred[cur]
        if u is not None:
            path.append(dict(from_node=u, to_node=cur, start_ns=earliest[u],
                             end_ns=earliest[cur], duration_ns=w, category=cat))
            if not (u.endswith('.full_query') and cat == 'control' and w == 0.):
                breakdown[cat] = breakdown.get(cat, 0.) + w
            if cat == 'tma_supply' and cur.endswith('.complete'):
                parts = cur.split('.')
                g_idx, atom_name = int(parts[0][1:]), parts[1]
                srv = deliveries[g_idx, atom_name]
                src_ns = srv['source_ready_ns'] - srv['arrival_ns']
                dest_arr = srv.get('destination_arrival_ns', srv['source_ready_ns'])
                dq_ns = srv['start_ns'] - dest_arr
                ds_ns = (srv['ready_ns'] - srv['start_ns']) + (dest_arr - srv['source_ready_ns'])
                tma_sub['tma_source_ns'] += src_ns
                tma_sub['tma_dest_queue_ns'] += dq_ns
                tma_sub['tma_dest_service_ns'] += ds_ns
                if srv.get('merged'):
                    tma_sub['merged_requests'] += 1
                else:
                    tma_sub['unmerged_requests'] += 1
        cur = u
    path.reverse()

    crit_nodes = {step['to_node'] for step in path} | {'entry'}
    input_wait_groups = [g['group'] for g in graph['groups']
                         if any(s['to_node'] == f'g{g["group"]}.consumer_wait' and s['category'] == 'input_wait'
                                for s in path)]
    slot_reuse_groups = [g['group'] for g in graph['groups']
                         if any(s['to_node'] in (f'g{g["group"]}.A.issue', f'g{g["group"]}.acquire') and s['category'] == 'slot_reuse'
                                for s in path)]
    compute_groups = [g['group'] for g in graph['groups']
                      if f'g{g["group"]}.compute_complete' in crit_nodes or f'g{g["group"]}.commit' in crit_nodes]

    return dict(
        cta=graph['cta'],
        target=target,
        target_ns=target_time,
        path_length=len(path),
        breakdown_ns=breakdown,
        tma_supply_breakdown=tma_sub,
        critical_groups=dict(
            input_wait_groups=input_wait_groups,
            slot_reuse_groups=slot_reuse_groups,
            compute_groups=compute_groups,
            total_input_stall_ns=sum(d['input_stall_ns'] for d in group_diagnostics),
            total_supply_slack_ns=sum(d['supply_slack_ns'] for d in group_diagnostics),
            total_slot_reuse_stall_ns=sum(d['slot_reuse_stall_ns'] for d in group_diagnostics),
        ),
        group_diagnostics=group_diagnostics,
        slack_ns=slack,
        path=path,
    )


def callback_declaration(callback):
    """运行时对象只导出实现身份，价格与状态仍由调用方另行声明。"""
    owner = getattr(callback, '__self__', None)
    implementation = owner if owner is not None else callback
    return dict(class_name=type(implementation).__name__,
        module=getattr(callback, '__module__', type(implementation).__module__),
        callable=getattr(callback, '__name__', type(implementation).__name__),
        version=getattr(implementation, 'version', None),
        qualification='假设：调用方声明的既有回调；本接线未增加校准价格')


def execution_inputs(program, parameters):
    intervention = program.get('runtime_intervention', {})
    selected = dict(parameters)
    if 'output_tma_enabled' in program:
        selected['output_tma_enabled'] = program['output_tma_enabled']
    repeats = intervention.get('wgmma_repeats', 1)
    if repeats != 1:
        selected['compute_ns'] *= repeats
        if selected.get('compute_phases_ns') is not None:
            selected['compute_phases_ns'] = [dict(phase, compute_ns=phase['compute_ns']*repeats)
                                            for phase in selected['compute_phases_ns']]
    return intervention.get('active_stages', program['stages']), selected


def simulate(program, parameters, *, share_inflight_sources=False, detailed_ctas=(), tensor_source=None,
             producer_prefix_ns=0., input_sync_service=None, input_source_async=None,
             input_destination_service=None, input_source_bind_events=None, register_quota=False,
             ordinary_lsu=False, sm_activity=False, input_source_service=None,
             output_source_observer=None, critical_path=False, compute_service=None):
    """共享服务状态只求解一次；绑定默认或延后首输出的请求/计算/槽释放时间线。"""
    for name in ('second_prefill', 'diagnostic_producer_offsets_ns'):
        if parameters.get(name):
            raise ValueError(f'{name} is not expanded in the common program')
    detailed_set = set(detailed_ctas)
    keys, counts = request_inputs(program)
    active_stages, selected = execution_inputs(program, parameters)
    if input_source_service is not None:
        selected['input_source_service'] = input_source_service
    if output_source_observer is not None:
        selected['output_source_observer'] = output_source_observer
    synchronous_callback = selected.get('input_source_service')
    output_callback = selected.get('output_source_observer')
    source = None
    if tensor_source is not None and input_source_async is not None:
        raise ValueError('tensor_source and input_source_async both replace the input source')
    if tensor_source is not None:
        source = CfgBTensorSource(program, keys, tensor_source['candidate'], tensor_source['parameters'])
        selected['input_source_service'] = source
        selected['destination_service_ns'] = tensor_source['parameters']['destination']
    sync = input_sync_service
    if sync is not None and producer_prefix_ns:
        sync = PrefixSyncService(sync, producer_prefix_ns)
    result = supply.shared_stage_events(program['ktiles'], active_stages, len(counts),
        output_counts=counts, source_keys=keys, share_inflight_sources=share_inflight_sources,
        trace_requests=True, input_sync_service=sync, input_source_async=input_source_async,
        input_destination_service=input_destination_service,
        input_source_bind_events=input_source_bind_events, compute_service=compute_service, **selected)
    by_cta = [{} for _ in counts]
    atoms = {'issue_a': 'A', 'issue_b0': 'B0', 'issue_b1': 'B1'}
    for delivery in result['deliveries']:
        by_cta[delivery['cta']][delivery['index'], atoms[delivery['kind']]] = delivery
    summaries, details = [], []
    cta_graphs_bounds = []
    edges = 0
    for cta, actor in enumerate(result['ctas']):
        graph = cfg_b_program.expand_cta(program, cta, register_quota=register_quota,
                                       epilogue_defer=parameters.get('epilogue_defer', 0),
                                       separate_empty_completion=bool(selected.get('input_release_delay_ns', 0.)))
        bound = bind_cta(graph, actor, by_cta[cta], producer_prefix_ns=producer_prefix_ns)
        if register_quota:
            quota = CfgBRegisterQuota(cta)
            for warp in cfg_b_program.register_demands(program)['warps']:
                quota.try_set(warp['warp'],warp['target_registers_per_thread'])
            bound['register_quota'] = quota.snapshot()
        edges += bound['dependency_edges_checked']
        summaries.append(dict(cta=cta, observations=bound['observations'],
            consumer_exit_ns=bound['consumer_exit_ns'], program_end_ns=bound['program_end_ns']))
        include_detail = cta in detailed_set
        if sm_activity:
            activity = activity_windows(actor,by_cta[cta].values(),prefix_ns=producer_prefix_ns,
                                        detailed=include_detail)
            summaries[-1]['sm_activity'] = {key:value for key,value in activity.items() if key != 'windows_ns'}
            bound['sm_activity'] = activity
        if critical_path and include_detail:
            bound['critical_path'] = critical_path_analysis(
                graph, bound, actor, selected, producer_prefix_ns=producer_prefix_ns)
        if critical_path:
            cta_graphs_bounds.append((graph, bound, actor))
        if include_detail:
            if 'compute_extra_ns' in actor:
                bound['compute_extra_ns'] = actor['compute_extra_ns']
            details.append(bound)
    end = max(c['consumer_exit_ns'] for c in summaries)
    critical_candidates = [c['cta'] for c in summaries if c['consumer_exit_ns'] == end]
    report = dict(schema='gh200-cfg-b-request-composition/1', mode='request_candidate',
        qualification='假设：指定 ns 服务参数下的开发时间线',
        case=program['case'], stages=program['stages'],
        parameters={key:value for key,value in parameters.items()
                    if key not in ('input_source_service', 'output_source_observer')},
        producer_prefix_ns=producer_prefix_ns,
        service_time_origin='内部求解以 producer_first_work 为原点；绑定到 entry 时串联一次前缀',
        share_inflight_sources=share_inflight_sources,
        clock='服务参数直接使用 ns；未求解频率',
        endpoint='共同入口到最后 store_tail；未映射为 plain 时间',
        control_timing='未单独定价的控制节点取前驱最晚时刻',
        source_identity=program['source_identity'],
        modules=dict(input='shared_stage_events 跨 CTA 服务候选',
            prefix=dict(ns=producer_prefix_ns, qualification='经验替代' if producer_prefix_ns else '假设'),
            compute='每 Ktile 复合计算服务及 commit/完成/释放候选',
            output='shared_stage_events 八个 D 子块与两源缓冲候选'),
        consumer_span_ns=end, program_end_ns=max(c['program_end_ns'] for c in summaries),
        critical_candidates=critical_candidates,
        dependency_edges_checked=edges,
        accounting={key: result[key] for key in (
            'input_demand_count', 'input_source_count', 'input_merged_count', 'input_delivery_count',
            'input_demand_work', 'input_source_work', 'input_merged_work', 'input_delivery_work',
            'shared_busy_ns', 'source_busy_ns', 'destination_busy_ns', 'request_count')},
        ctas=summaries, details=details)
    if 'producer_control_windows' in result:
        report['producer_control_windows'] = [dict(window,
            start_ns=window['start_ns']+producer_prefix_ns,
            end_ns=window['end_ns']+producer_prefix_ns)
            for window in result['producer_control_windows']]
    if compute_service is not None:
        report['accounting']['compute_extra_ns'] = sum(sum(a['compute_extra_ns']) for a in result['ctas'])
        report['modules']['compute'] = '显式本地联合服务；延迟传至完成、等待返回和槽释放'
    if critical_path and critical_candidates:
        c_graph, c_bound, c_actor = cta_graphs_bounds[critical_candidates[0]]
        report['critical_path'] = c_bound.get('critical_path') or critical_path_analysis(
            c_graph, c_bound, c_actor, selected, producer_prefix_ns=producer_prefix_ns)
    if 'runtime_intervention' in program:
        report['runtime_intervention'] = dict(program['runtime_intervention'],
            physical_stages=program['stages'], compute_ns=selected['compute_ns'],
            compute_phases_ns=selected.get('compute_phases_ns'),
            compute_service='existing per-pass composite cost scaled by WGMMA work; one commit per Ktile')
    if 'output_tma_enabled' in result:
        report['output_tma_enabled'] = result['output_tma_enabled']
        report['accounting']['output_tma_count'] = sum(r['kind']=='store' for r in result['requests'])
    if sm_activity:
        from sm_residency import cfg_b_capacity, GH200_DEVICE_CONDITIONS
        report['modules']['sm_activity'] = dict(capacity=cfg_b_capacity(program,device=GH200_DEVICE_CONDITIONS),
            activity='per-CTA modeled composite compute, local output and input-pending windows',
            joint_service='cfg_b has no matched measured contract; existing separate services retained',
            entry='existing common modeled entry; physical SM assignment and startup dispersion unresolved')
    if ordinary_lsu:
        report['modules']['ordinary_lsu'] = dict(demands=cfg_b_program.ordinary_lsu_demands(program),
            timing='ordinary global/local active demand is zero; constant/control reads retain existing coverage')
    if input_sync_service is not None:
        report['modules']['sync'] = dict(model='ConditionSyncService',
            fast_return_ns=input_sync_service.fast_return_ns,
            blocked_return_ns=input_sync_service.blocked_return_ns,
            qualification=input_sync_service.qualification)
    if register_quota:
        report['modules']['register_quota'] = dict(demands=cfg_b_program.register_demands(program),
            timing='启动配额节点未单独定价；producer warp0释放已含前缀，其余warp释放与consumer申请时长待测',
            ledger_order='零独立价格下按warp0..11完成配额核对',
            qualification='已核对软件需求；时间沿零独立价格工作假设')
    if source is not None:
        report['tensor_source'] = tensor_source
        report['share_inflight_sources'] = source.service.sharing
        report['modules']['input'] = 'tensor 探针与 GEMM 共用 TensorSourceService；GEMM 地址及槽复用由程序输入'
        report['input_output_service_relation'] = '输入与输出服务独立；输出保留现有候选'
        report['tensor_resident_path'] = source.service.resident
    if synchronous_callback is not None or output_callback is not None:
        report['internal_window_ns'] = result['makespan_ns']
        report['source_callbacks'] = dict(
            input_source=callback_declaration(synchronous_callback) if synchronous_callback is not None else None,
            output_source_observer=callback_declaration(output_callback) if output_callback is not None else None,
            qualification='既有源服务替换与输出source-read完成观察；不新增写回或global可见性端点')
    if (synchronous_callback is not None or input_source_async is not None
            or input_destination_service is not None or input_source_bind_events is not None):
        report['input_services'] = dict(
            source=type(input_source_async).__name__ if input_source_async is not None else
                   'TensorSourceService' if source is not None else
                   callback_declaration(synchronous_callback)['class_name'] if synchronous_callback is not None
                   else 'existing_source',
            destination=type(input_destination_service).__name__ if input_destination_service is not None else 'per_cta_fifo',
            source_timers=input_source_bind_events is not None,
            qualification='假设：调用方显式接入的源／目的服务；无新增校准价格')
        report['modules']['input'] = '显式源／目的服务接入既有事件引擎；程序地址保留于请求，外部源身份与合并由调用方服务定义'
        last_destination = {}
        for delivery in result['deliveries']:
            sid = delivery['source_id']
            last_destination[sid] = max(last_destination.get(sid, 0.), delivery['ready_ns'])
        report['input_sources'] = []
        for record in result['sources']:
            shifted = dict(record, last_destination_ready_ns=last_destination[record['source_id']])
            for field in ('arrival_ns', 'start_ns', 'service_end_ns', 'ready_ns',
                          'join_end_ns', 'last_destination_ready_ns', 'batch_ready_ns',
                          'joined_last_ready_ns'):
                if shifted.get(field) is not None:
                    shifted[field] += producer_prefix_ns
            report['input_sources'].append(shifted)
    return report


def predict(program, parameters, frequency_model, calibration, *, compute_cycles=512.,
            commit_lead_cycles=256., **options):
    """静态阶段频率换算计算服务，再由 V09 经验映射得到完整时间。"""
    frequencies = phase_frequency.predict_phases(frequency_model, program['case'], program['work'])
    selected = dict(parameters, compute_phases_ns=[dict(compute_ns=compute_cycles/f,
        commit_lead_ns=commit_lead_cycles/f) for f in frequencies])
    report = simulate(program, selected, **options)
    window_us = report['consumer_span_ns']/1000
    projected = aggregate.V09Observation(calibration).project(dict(window_us=window_us))
    report.update(mode='request_prediction', window_us=window_us, **projected,
        phase_frequency_ghz=frequencies,
        clock='每个输出阶段的静态经验频率换算计算与提交服务；其余服务沿用声明的 ns',
        endpoint='共同入口到最后 store_tail；经 V09 F+κW 映射为 plain 时间',
        qualification='开发组合：服务候选、阶段经验频率与经验时间映射；尚未通过组合 C2/C3',
        compute_cycles=compute_cycles, commit_lead_cycles=commit_lead_cycles)
    report['modules'].update(frequency=dict(version=frequency_model['schema'], qualification='经验替代'),
        observation=dict(version=aggregate.V09Observation.version, qualification='经验替代'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--candidate', choices=('shared', 'unshared'), required=True,
                        help='复用已失败的历史参数作开发回归，不重新拟合')
    parser.add_argument('--tensor-source', type=Path,
                        help='替换输入服务的 tensor 参数 JSON；保留其余程序、计算和输出价格')
    parser.add_argument('--tensor-candidate', choices=tensor.CANDIDATES, default='banked_resident_fifo')
    parser.add_argument('--cta', type=int, default=0)
    parser.add_argument('--phase-frequency', type=Path,
                        help='运行前阶段频率模型 JSON；接入计算周期换算与 V09 plain 映射')
    parser.add_argument('--compute-cycles', type=float, default=512.)
    parser.add_argument('--commit-lead-cycles', type=float, default=256.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = aggregate.DEFAULT_ARCHIVE
    if args.output.resolve().is_relative_to(root.parents[4]):
        parser.error('output must be outside historical results')
    identity = aggregate.archive_identity(root)
    rows = aggregate.load_json(root/'cases.json')
    row = next((r for r in rows if r['id'] == args.case and r['config'] == 'cfg_b'), None)
    if row is None:
        parser.error('no matching cfg_b case')
    setups = {r['case']: r['setup'] for r in aggregate.load_json(root/'static_setup.json')}
    program = cfg_b_program.build_program(row, setups[row['id']], identity)
    if not 0 <= args.cta < len(program['work']):
        parser.error('--cta outside grid')
    parameters = aggregate.load_json(EXAMPLES)['examples'][args.candidate]['parameters']
    tensor_source = None
    if args.tensor_source:
        saved = aggregate.load_json(args.tensor_source)
        tensor_source = dict(candidate=args.tensor_candidate, parameters={name: saved['parameters'][name]
            for name in ('service', 'latency', 'resident_service', 'resident_latency', 'destination',
                         'source_partitions', 'cache_capacity_bytes') if name in saved['parameters']},
            path=str(args.tensor_source), sha256=aggregate.sha256(args.tensor_source),
            qualification='假设：探针候选参数向 GEMM 的开发迁移')
    options = dict(share_inflight_sources=args.candidate == 'shared',
                   detailed_ctas=(args.cta,), tensor_source=tensor_source)
    if args.phase_frequency:
        calibration_path = root/'frozen/calibration.json'
        report = predict(program, parameters, aggregate.load_json(args.phase_frequency),
            aggregate.load_json(calibration_path), compute_cycles=args.compute_cycles,
            commit_lead_cycles=args.commit_lead_cycles, **options)
        report['frequency_source'] = dict(path=str(args.phase_frequency),
            sha256=aggregate.sha256(args.phase_frequency))
        report['observation_source'] = dict(path=str(calibration_path),
            sha256=aggregate.sha256(calibration_path))
    else:
        report = simulate(program, parameters, **options)
    report['parameter_source'] = dict(path=str(EXAMPLES), sha256=aggregate.sha256(EXAMPLES),
        candidate=args.candidate, status=('历史参数已失败；计算与提交价格由阶段频率换算，其余沿用'
            if args.phase_frequency else '历史参数已失败；本次原样用于普通 cfg_b 程序的接线检查'))
    report['implementation_sha256'] = {Path(m.__file__).name: aggregate.sha256(Path(m.__file__))
        for m in (cfg_b_program, cfg_b_program.case_inputs, supply, aggregate, tensor, phase_frequency)}
    report['implementation_sha256'][Path(__file__).name] = aggregate.sha256(Path(__file__))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'PASS {args.case}: {len(report["ctas"])} CTAs, '
          f'{report["accounting"]["input_delivery_count"]} input deliveries, '
          f'{report["dependency_edges_checked"]} dependency edges')
    if args.phase_frequency:
        print(f'development plain prediction {report["plain_us"]:.6f} us; candidate remains unqualified')
    else:
        print('candidate parameters remain unqualified; no frequency/plain prediction')
    print(args.output)


if __name__ == '__main__':
    main()
