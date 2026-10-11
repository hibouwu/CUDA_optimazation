"""Cooperative output intervals from valid-output fractions and calibrated rules."""

import heapq
import math
from numbers import Real


def output_intervals(params, schedule, valid, first_bytes, single, output_rule):
    """Return copied CTA parameters and optional per-tile merged epilogues in ns.

    first_bytes is the total valid first-output byte count across all CTAs.
    single describes the maximum scheduled tile count across the whole grid.
    """
    p = dict(params)
    if not output_rule or not valid:
        return p, None
    if schedule != 'cooperative':
        raise ValueError('these output rules were calibrated for cooperative kernels')
    if single:
        rule = output_rule['E0_single']
        e0 = max(rule['floor_ns'], rule['beta_ns_per_MiB']*first_bytes/2**20) + rule['R_ns']
        p['Etail'] = output_rule['Etail_single_mean_ns']
    else:
        e0 = output_rule['E0_multi']['effective_ns']
        p['Etail'] = ((1-valid[-1])*output_rule['Etail_oob_median_ns']
                      + valid[-1]*output_rule['v09_Etail_multi_median_ns'])
    middle = output_rule['E_middle_ns']
    if middle is None:
        middle = output_rule['middle_padding_proxy']['ns']
    epilogues = [e0]
    for j, fraction in enumerate(valid[1:], 1):
        key = 'Elast' if j == len(valid)-1 else 'E_middle'
        full = output_rule['Elast_ns'] if key == 'Elast' else middle
        epilogues.append((1-fraction)*output_rule[key+'_oob_ns'] + fraction*full)
    return p, epilogues


def cfg_a_rank_dependencies(stages, k_tiles, output_tiles):
    """Source-derived partial order for cluster2x1, two consumer WGs, K_PIPE_MMAS=1.

    No service prices or trace inputs. Release issue and remote arrival are separate.
    Groups advance across output tiles; this is not the cfg_b pingpong graph.
    """
    if stages < 2 or k_tiles < 2 or output_tiles < 1:
        raise ValueError('requires stages>=2, Ktiles>=2 and at least one output tile')
    nodes, markers, groups, transfers, outputs = {}, {}, [], [], []
    def producer(r, g, event):return f'r{r}.g{g}.{event}'
    def consumer(r, w, g, event):return f'r{r}.w{w}.g{g}.{event}'
    def arrival(s, w, r, g):return f'r{s}.w{w}.g{g}.empty_to_r{r}'
    def add(name, predecessors):
        if name in nodes:raise ValueError('duplicate event '+name)
        nodes[name] = list(dict.fromkeys(predecessors))
    total = k_tiles*output_tiles
    for g in range(total):
        tile, local = divmod(g, k_tiles)
        groups.append(dict(group=g,tile=tile,k=local,slot=g%stages,generation=g//stages,
            empty_wait_phase=1^(g//stages%2),full_wait_phase=g//stages%2,
            expected_full_bytes=32768,empty_arrivals=4))
        for r in range(2):
            predecessors=[producer(r,g-1,'B_issue')] if g else []
            if g>=stages:
                predecessors += [arrival(s,w,r,g-stages) for s in range(2) for w in range(2)]
            add(producer(r,g,'acquire'),predecessors)
            add(producer(r,g,'arrive_expect_tx'),[producer(r,g,'acquire')])
            add(producer(r,g,'A_issue'),[producer(r,g,'arrive_expect_tx')])
            add(producer(r,g,'B_issue'),[producer(r,g,'A_issue')])
            add(producer(r,g,'A_visible'),[producer(r,g,'A_issue')])
            transfers.append(dict(group=g,operand='A',source_rank=r,destination_rank=r,
                                  bytes=16384,completion=producer(r,g,'A_visible')))
            for s in range(2):
                add(f'r{s}.g{g}.B_to_r{r}',[producer(s,g,'B_issue')])
                transfers.append(dict(group=g,operand='B',source_rank=s,destination_rank=r,
                                      bytes=8192,completion=f'r{s}.g{g}.B_to_r{r}'))
            add(producer(r,g,'full'),[producer(r,g,'arrive_expect_tx'),producer(r,g,'A_visible')]
                +[f'r{s}.g{g}.B_to_r{r}' for s in range(2)])
            for w in range(2):
                previous=[]
                if local==0 and tile:
                    previous=[f'r{r}.w{w}.t{tile-1}.DONE']
                elif local==1:
                    previous=[consumer(r,w,g-1,'commit')]
                elif local>=2:
                    previous=[consumer(r,w,g-2,'release_issue')]
                add(consumer(r,w,g,'wait_return'),[producer(r,g,'full')]+previous)
                before_commit=consumer(r,w,g,'wait_return')
                if local==0:
                    marker=f'r{r}.w{w}.t{tile}.FIRST'
                    add(marker,[before_commit]);markers[marker]=marker;before_commit=marker
                add(consumer(r,w,g,'commit'),[before_commit])
                add(consumer(r,w,g,'retired'),[consumer(r,w,g,'commit')]
                    +([consumer(r,w,g-1,'retired')] if local else []))
                if local<k_tiles-1:
                    add(consumer(r,w,g,'safe_release'),[consumer(r,w,g+1,'commit'),consumer(r,w,g,'retired')])
                else:
                    add(consumer(r,w,g,'mma_return'),[consumer(r,w,g-1,'release_issue')])
                    add(consumer(r,w,g,'tail_wait0'),[consumer(r,w,g,'mma_return'),consumer(r,w,g,'retired')])
                    add(consumer(r,w,g,'safe_release'),[consumer(r,w,g,'tail_wait0')])
                add(consumer(r,w,g,'release_issue'),[consumer(r,w,g,'safe_release')])
                for d in range(2):add(arrival(r,w,d,g),[consumer(r,w,g,'release_issue')])
                if local==k_tiles-1:
                    prefix=f'r{r}.w{w}.t{tile}.'
                    add(prefix+'MAIN',[consumer(r,w,g,'release_issue')])
                    add(prefix+'PERMIT',[prefix+'MAIN'])
                    add(prefix+'DONE',[f'r{r}.w{v}.t{tile}.PERMIT' for v in range(2)])
                    for event in ('MAIN','PERMIT','DONE'):markers[prefix+event]=prefix+event
    for r in range(2):
        # cfg_a: 128x128 FP32 -> four nominal 128x32 subtiles, two D buffers.
        # DelayTmaStore prepares the next registers before issuing the previous
        # subtile, then acquire/read1 protects reuse. State continues across tiles.
        for q in range(4*output_tiles):
            tile,subtile=divmod(q,4)
            prefix=f'r{r}.d{q}.'
            predecessors=[f'r{r}.w{w}.t{tile}.PERMIT' for w in range(2)]
            if subtile:predecessors.append(f'r{r}.d{q-1}.r2s')
            add(prefix+'register_prepare',predecessors)
            reuse=[f'r{r}.d{q-1}.acquire_after'] if q else []
            add(prefix+'r2s',[prefix+'register_prepare']+reuse)
            issue=[prefix+'r2s']
            if subtile<3:issue.append(f'r{r}.d{q+1}.register_prepare')
            add(prefix+'store_issue',issue)
            add(prefix+'store_commit',[prefix+'store_issue'])
            add(prefix+'source_read',[prefix+'store_issue'])
            add(prefix+'read_retired',[prefix+'source_read']
                +([f'r{r}.d{q-1}.read_retired'] if q else []))
            add(prefix+'acquire_after',[prefix+'store_commit']
                +([f'r{r}.d{q-1}.read_retired'] if q else []))
            add(prefix+'D_visible',[prefix+'source_read'])
            if subtile==3:
                for w in range(2):nodes[f'r{r}.w{w}.t{tile}.DONE'].append(prefix+'acquire_after')
            outputs.append(dict(rank=r,subtile=q,tile=tile,local_subtile=subtile,buffer=q%2,
                generation=q//2,nominal_bytes=16384,
                source_completion=prefix+'source_read',global_completion=prefix+'D_visible'))
        # Last generation of each used slot: producer-tail waits for all empty arrivals.
        last=[g for g in range(max(0,total-stages),total)]
        add(f'r{r}.producer_tail',[producer(r,total-1,'B_issue')]
            +[arrival(s,w,r,g) for g in last for s in range(2) for w in range(2)])
        add(f'r{r}.output_source_read',[f'r{r}.d{4*output_tiles-1}.read_retired'])
        for w in range(2):
            key=f'r{r}.w{w}.FINAL'
            add(key,[f'r{r}.w{w}.t{output_tiles-1}.DONE']
                +([f'r{r}.output_source_read'] if w==1 else []));markers[key]=key
        add(f'r{r}.D_visible',[f'r{r}.d{q}.D_visible' for q in range(4*output_tiles)])
    add('source_safe_exit',[f'r{r}.producer_tail' for r in range(2)]
        +[f'r{r}.w{w}.FINAL' for r in range(2) for w in range(2)])
    add('cuda_complete',['source_safe_exit','r0.D_visible','r1.D_visible'])
    return dict(nodes=nodes,markers=markers,groups=groups,transfers=transfers,outputs=outputs,stages=stages,k_tiles=k_tiles,
                output_tiles=output_tiles,scope='Partial order; no calibrated service or GPU timing prediction')


def dependency_order(graph):
    """Deterministic topological order; reject missing predecessors and dependency cycles."""
    nodes=graph['nodes'];children={n:[] for n in nodes};degree={n:len(p) for n,p in nodes.items()}
    for name,predecessors in nodes.items():
        for p in predecessors:
            if p not in children:raise ValueError('missing predecessor '+p)
            children[p].append(name)
    queue=[n for n,d in degree.items() if not d];heapq.heapify(queue);order=[]
    while queue:
        n=heapq.heappop(queue);order.append(n)
        for c in children[n]:
            degree[c]-=1
            if not degree[c]:heapq.heappush(queue,c)
    if len(order)!=len(nodes):raise ValueError('dependency cycle')
    return order,children


def dependency_bounds(graph, anchors=None, latencies_ns=None):
    """Feasible intervals under nonnegative edge lags; anchors are diagnostic only.

    Each latency is from a node's latest predecessor to that node. Anchors are
    (finite lower, finite or positive-infinite upper) pairs in an arbitrary time
    origin; negative timestamps are allowed. Unbounded endpoints become None.
    Without anchors, roots start at zero. Earliest values in that case are synthetic
    max-plus witnesses, not fitted service times or autonomous GEMM predictions.
    """
    anchors=anchors or {};latencies_ns=latencies_ns or {};order,children=dependency_order(graph)
    if (set(anchors)|set(latencies_ns))-set(order):raise ValueError('unknown event key')
    if any(not isinstance(v,Real) or not math.isfinite(v) or v<0 for v in latencies_ns.values()):
        raise ValueError('latencies must be finite nonnegative ns')
    for name,interval in anchors.items():
        if not isinstance(interval,(tuple,list)) or len(interval)!=2:
            raise ValueError('anchor must contain two endpoints: '+name)
        low,high=interval
        if (not isinstance(low,Real) or not math.isfinite(low)
                or not isinstance(high,Real) or math.isnan(high) or high==-math.inf or low>high):
            raise ValueError('invalid anchor interval: '+name)
    origin=-math.inf if anchors else 0
    lower={n:anchors.get(n,(origin,math.inf))[0] for n in order}
    upper={n:anchors.get(n,(origin,math.inf))[1] for n in order}
    for n in order:
        root_lower=origin if anchors else latencies_ns.get(n,0)
        lower[n]=max(lower[n],max((lower[p]+latencies_ns.get(n,0) for p in graph['nodes'][n]),default=root_lower))
    for n in reversed(order):
        upper[n]=min(upper[n],min((upper[c]-latencies_ns.get(c,0) for c in children[n]),default=math.inf))
    invalid=[n for n in order if lower[n]>upper[n]+1e-9]
    if invalid:raise ValueError('infeasible event bounds: '+','.join(invalid[:4]))
    return {n:[lower[n] if math.isfinite(lower[n]) else None,
               upper[n] if math.isfinite(upper[n]) else None] for n in order}
