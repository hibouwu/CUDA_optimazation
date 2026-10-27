"""cfg_b 三请求组探针的 CPU 程序与候选比较；无 GPU 调用。"""
import argparse
from functools import lru_cache
import hashlib
import heapq
import itertools
import json
from pathlib import Path


ATOMS = (('A', (128, 64), 16384), ('B0', (64, 64), 8192), ('B1', (64, 64), 8192))
CANDIDATES = ('local_fifo', 'shared_fifo', 'shared_source_fifo', 'resident_source_fifo',
              'two_tier_fifo', 'two_tier_shared_fifo', 'banked_resident_fifo')


class TensorSourceService:
    """由请求驱动的源服务；探针控制流、消费者与目的槽由调用方管理。"""

    def __init__(self, candidate, ctas, source_union_bytes, *, service, latency,
                 resident_service=None, resident_latency=None,
                 cache_capacity_bytes=60*1024*1024, source_partitions=1):
        self.tiered = candidate in ('resident_source_fifo', 'two_tier_fifo',
                                   'two_tier_shared_fifo', 'banked_resident_fifo')
        self.resident = self.tiered and source_union_bytes <= cache_capacity_bytes
        separate_latency = candidate.startswith('two_tier') or candidate == 'banked_resident_fifo'
        self.price = resident_service if self.resident else service
        self.latency = resident_latency if self.resident and separate_latency else latency
        self.sharing = candidate in ('shared_source_fifo', 'two_tier_shared_fifo', 'banked_resident_fifo')
        self.partitioned = self.resident and candidate == 'banked_resident_fifo'
        self.local = candidate == 'local_fifo'
        self.partitions = source_partitions
        self.frontiers = [0.] * (ctas if self.local else source_partitions if self.partitioned else 1)
        self.pending = {}
        self.count = 0
        self.work = 0.
        self.busy = 0.

    def submit(self, t, cta, key, weight, partition_demand=None):
        previous = self.pending.get(key)
        merged = self.sharing and previous is not None and t < previous[1]
        if merged:
            source_id, ready = previous
            start = end = None
        else:
            if self.partitioned:
                starts, ends = [], []
                for owner, fraction in enumerate(partition_demand):
                    if fraction:
                        begin = max(t, self.frontiers[owner])
                        done = begin + self.price*self.partitions*fraction
                        starts.append(begin)
                        ends.append(done)
                        self.frontiers[owner] = done
                        self.busy += done-begin
                start, end = min(starts), max(ends)
            else:
                owner = cta if self.local else 0
                start = max(t, self.frontiers[owner])
                end = start+self.price*weight
                self.frontiers[owner] = end
                self.busy += end-start
            ready = end+self.latency
            source_id = self.count
            self.count += 1
            self.work += weight
            if self.sharing:
                self.pending[key] = (source_id, ready)
        return dict(source_id=source_id, merged=merged, service_start=start,
                    service_end=end, source_ready=ready)


@lru_cache(maxsize=4096)
def row_partition_weights(first, stride, rows, partitions):
    """候选映射：每行 128 B，按行首地址的 128 B 序号低位分区。"""
    counts = [0]*partitions
    for row in range(rows):
        address=first+row*stride
        unit,offset=divmod(address,128)
        counts[unit%partitions] += 128-offset
        counts[(unit+1)%partitions] += offset
    return tuple(count/16384 for count in counts)


def partition_weights(p, cta, source_index, atom, partitions):
    tile = cta if p['source_protocol']=='independent' else 0
    strides = p.get('source_strides_elements',dict(A=p['source_ring']*64,
        B=128*(p['ctas'] if p['source_protocol']=='independent' else 1)))
    if atom=='A':
        first=(tile*128*strides['A']+source_index*64)*2
        stride=strides['A']*2
        rows=128
    else:
        first=(source_index*64*strides['B']+tile*128+(64 if atom=='B1' else 0))*2
        stride=strides['B']*2
        rows=64
    return row_partition_weights(first%(partitions*128),stride%(partitions*128),rows,partitions)


def program(ctas, active_groups, source_protocol, groups=64, source_ring=32, source_phase_stride=0, ldb=None):
    """四个物理组槽；先填活动窗口，等待/消费/释放最老组，再补发。"""
    if ctas < 1 or active_groups not in (1, 2, 4) or groups < active_groups or source_ring < 1:
        raise ValueError('positive geometry and active_groups in 1/2/4 required')
    if source_protocol not in ('shared', 'independent'):
        raise ValueError('source protocol must be shared or independent')
    source_indices = {(g+c*source_phase_stride) % source_ring for c in range(ctas) for g in range(groups)}
    union_groups = len(source_indices) if source_protocol == 'shared' else min(groups,source_ring)*ctas
    m = n = 128*(ctas if source_protocol == 'independent' else 1)
    k = source_ring*64
    ldb = n if ldb is None else ldb
    return dict(schema='tma-tensor-probe-program/1', ctas=ctas, active_groups=active_groups,
        groups=groups, source_protocol=source_protocol, source_ring=source_ring,
        source_phase_stride=source_phase_stride,
        source_shape=dict(m=m,n=n,k=k),source_strides_elements=dict(A=k,B=ldb),
        source_allocation_bytes=dict(A=m*k*2,B=ldb*k*2),
        physical_slots=4, payload_bytes_per_slot=32768, physical_payload_bytes=131072,
        expected_full_bytes=32768, atoms=[dict(name=n, shape=s, bytes=b) for n, s, b in ATOMS],
        source='GPU local cudaMalloc A/B allocations', destination='CTA local SMEM',
        descriptor=dict(target_sass='UTMALDG.3D', swizzle='128B', binding='cfg_b Mainloop factory; instance hex in native samples'),
        requested_residency='one CTA per SM; native binary uses 132096 B SMEM',
        start='cooperative grid sync, shared globaltimer deadline, CTA sync',
        control='single issuing/retiring controller; fill Q, retire oldest, refill, final drain',
        release='group ready -> declared consumer/control cost -> slot release',
        source_union_bytes=union_groups*32768,
        input_requests=ctas*groups*3, input_bytes=ctas*groups*32768)


def cases(groups=64):
    return [program(n, q, source, groups) for source in ('shared', 'independent')
            for n in (1, 16, 132) for q in (1, 4)]


def simulate(p, candidate, *, service, latency, issue, destination, consumer, trace=False, control=None,
             resident_service=None, cache_capacity_bytes=60*1024*1024, resident_latency=None,
             source_partitions=1, source_service=None, memory_observer=None, source_async=None):
    """统一抽象时间单位；价格由调用方提供。源服务完成后才进入每 CTA 目的 FIFO。"""
    if candidate not in CANDIDATES:
        raise ValueError('unknown candidate')
    if service <= 0 or min(latency, issue, destination, consumer) < 0:
        raise ValueError('positive service and nonnegative remaining costs required')
    tiered = candidate in ('resident_source_fifo','two_tier_fifo','two_tier_shared_fifo','banked_resident_fifo')
    resident = tiered and p['source_union_bytes'] <= cache_capacity_bytes
    if tiered and (resident_service is None or resident_service <= 0):
        raise ValueError('resident service must be positive')
    separate_latency = candidate.startswith('two_tier') or candidate=='banked_resident_fifo'
    if separate_latency and (resident_latency is None or resident_latency < 0):
        raise ValueError('two-tier service needs a nonnegative resident latency')
    source_price = resident_service if resident else service
    source_latency = resident_latency if resident and separate_latency else latency
    sharing = candidate in ('shared_source_fifo','two_tier_shared_fifo','banked_resident_fifo')
    partitioned = resident and candidate=='banked_resident_fifo'
    queue, sequence = [], itertools.count()
    source = source_async or source_service or TensorSourceService(candidate, p['ctas'], p['source_union_bytes'], service=service,
        latency=latency, resident_service=resident_service, resident_latency=resident_latency,
        cache_capacity_bytes=cache_capacity_bytes, source_partitions=source_partitions)
    destination_frontiers = [0.] * p['ctas']
    actors = [dict(next=0, oldest=0, waiting=False, finished=None,
                   ready_counts=[0]*p['groups'], group_ready=[None]*p['groups'],
                   released=[None]*p['groups'], issued=[None]*p['groups']) for _ in range(p['ctas'])]
    demands, group_events = [], []

    def push(t, kind, cta, index):
        heapq.heappush(queue, (t, next(sequence), kind, cta, index))

    def source_ready(index, t, served):
        demands[index].update(served, source_ready=t)
        push(t, 'deliver', demands[index]['cta'], index)

    if source_async is not None:
        source.bind_scheduler(lambda t, callback: push(t, 'source_callback', 0, callback))

    control = control or {}
    wait_floor = control.get('wait_floor', 0.)

    def advance(t, cta, origin):
        a = actors[cta]
        g, oldest = a['next'], a['oldest']
        if oldest == p['groups']:
            a['finished'] = t
            if memory_observer is not None:
                memory_observer(t, cta, oldest, 'cta_record')
        elif g == p['groups'] or g-oldest == p['active_groups']:
            a['waiting'] = True
            delay = control.get('drain_gap' if origin == 'release' else
                                ('first_wait' if oldest == 0 else 'issue_to_wait'), 0.)
            a['wait_begin'] = t+delay
            if a['group_ready'][oldest] is not None:
                push(max(a['wait_begin']+wait_floor, a['group_ready'][oldest])+consumer, 'release', cta, oldest)
        else:
            key = {'initial':'prefix', 'issue':'prefill_gap', 'release':'release_to_issue'}[origin]
            push(t+control.get(key, 0.), 'issue', cta, g*3)

    offsets = p.get('initial_offsets_ns', [0.]*p['ctas'])
    for cta in range(p['ctas']):
        push(offsets[cta], 'advance', cta, 'initial')
    while queue:
        t, _, kind, cta, index = heapq.heappop(queue)
        if kind == 'source_callback':
            index(t)
            continue
        a = actors[cta]
        if kind == 'advance':
            advance(t, cta, index)
        elif kind == 'issue':
            g, atom = divmod(index, 3)
            name, _, size = ATOMS[atom]
            weight = size/16384
            source_index = (g+cta*p['source_phase_stride']) % p['source_ring']
            key = (cta if p['source_protocol'] == 'independent' else 0, source_index, name)
            weights = partition_weights(p,cta,source_index,name,source_partitions) if partitioned else None
            demand = dict(cta=cta, group=g, atom=name, bytes=size,
                slot=g % 4, generation=g//4, phase=(g//4) % 2,
                destination_offset=(g % 4)*32768+(0, 16384, 24576)[atom],
                source_key=key, issue=t)
            demands.append(demand)
            demand_id = len(demands)-1
            if source_async is None:
                served = source.submit(t, cta, key, weight, weights)
                demand.update(served)
                push(served['source_ready'], 'deliver', cta, demand_id)
            else:
                source.issue(t, cta, g, atom, weight, demand_id, source_ready)
            if atom == 0:
                a['issued'][g] = t
            issue_step = control.get('prefill_issue' if g < p['active_groups'] else 'refill_issue', issue*3)/3
            if atom < 2:
                push(t+issue_step, 'issue', cta, index+1)
            else:
                a['next'] += 1
                if memory_observer is not None:
                    push(t+issue_step, 'record_issue', cta, g)
                push(t+issue_step, 'advance', cta, 'issue')
        elif kind == 'record_issue':
            memory_observer(t, cta, index, 'issue_record')
        elif kind == 'deliver':
            r = demands[index]
            if source_async is None:
                start = max(t, destination_frontiers[cta])
                done = start+destination*r['bytes']/16384
                destination_frontiers[cta] = done
            else:
                delivered = source.destination_service(t, cta, r['group'], ('A', 'B0', 'B1').index(r['atom']), r['bytes']/16384)
                start, done = delivered['start_ns'], delivered['ready_ns']
                push(done, 'destination_done', cta, index)
            r.update(destination_start=start, ready=done)
            push(done, 'ready', cta, r['group'])
        elif kind == 'destination_done':
            source.destination_done(t, index)
        elif kind == 'ready':
            a['ready_counts'][index] += 1
            if a['ready_counts'][index] == 3:
                a['group_ready'][index] = t
                if a['waiting'] and a['oldest'] == index:
                    push(max(t, a['wait_begin']+wait_floor)+consumer, 'release', cta, index)
        elif kind == 'release':
            a['released'][index] = t
            if memory_observer is not None:
                memory_observer(t, cta, index, 'release_record')
            group_events.append(dict(cta=cta, group=index, slot=index % 4, generation=index//4,
                issue=a['issued'][index], ready=a['group_ready'][index], wait_begin=a['wait_begin'],
                wait_return=max(a['wait_begin']+wait_floor, a['group_ready'][index]), release=t))
            a['oldest'] += 1
            a['waiting'] = False
            advance(t, cta, 'release')
    if any(a['finished'] is None for a in actors):
        raise RuntimeError('probe program did not drain')
    durations = [a['finished'] for a in actors]
    result = dict(candidate=source.version if source_service is not None or source_async is not None else candidate,
        endpoint='common origin to final group release' if 'initial_offsets_ns' in p else 'controller start to final group release',
        time=max(durations), cta_times=durations, cta_span=max(durations)-min(durations),
        input_requests=len(demands), input_bytes=sum(r['bytes'] for r in demands),
        source_generations=source.count, source_work_16KiB=source.work,
        merges=sum(r['merged'] for r in demands), deliveries=len(demands))
    if tiered:
        result.update(source_price=source_price,source_latency=source_latency, cache_capacity_bytes=cache_capacity_bytes,
            residency_assumption='warm full working set fits: resident path; otherwise streaming path',inflight_sharing=sharing)
    if partitioned:
        result.update(source_partitions=source_partitions,
            partition_map='128-byte row address modulo partition count; A/B base relative phase assumed zero')
    if trace:
        result.update(request_events=demands, group_events=group_events)
    return result


def screening_report():
    # 人工比值检查实现与条件灵敏度；不是 GH200 参数或预测冻结。
    scenarios = dict(service_visible=dict(service=1., latency=4., issue=.05, destination=.1, consumer=.1, resident_service=.1,resident_latency=2.),
        latency_dominated=dict(service=1., latency=4096., issue=.05, destination=.1, consumer=.1, resident_service=.1,resident_latency=4096.))
    rows = []
    for name, costs in scenarios.items():
        for p in cases():
            values = {c: simulate(p, c, **costs) for c in CANDIDATES}
            rows.append(dict(scenario=name, program=p, candidates=values))
    contrasts = []
    for name in scenarios:
        for a, b in itertools.combinations(CANDIDATES, 2):
            group = [r for r in rows if r['scenario'] == name]
            gaps = [abs(r['candidates'][a]['time']-r['candidates'][b]['time']) /
                    max(r['candidates'][a]['time'], r['candidates'][b]['time']) for r in group]
            contrasts.append(dict(scenario=name, candidates=[a, b], equal_cases=sum(x == 0 for x in gaps),
                max_relative_gap=max(gaps), best_condition=group[gaps.index(max(gaps))]['program']))
    return dict(schema='tma-tensor-condition-screening/1', qualification='人工参数下的 CPU 结构筛选',
        time_unit='normalized service unit; not ns or cycles', scenarios=scenarios,
        frozen_hardware_prediction=False, cuda_probe_implemented=True,
        remaining=['候选参数范围与来源', '测量不确定性及判定阈值',
                   '全参数范围的候选区分检查与正式预测冻结'],
        conditions=12, rows=rows, contrasts=contrasts,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = screening_report()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'CPU screening: 12 programs x {len(CANDIDATES)} candidates x 2 synthetic cost scenarios')
    print('Native probe correctness passed; hardware predictions are not frozen.')
    for item in report['contrasts']:
        print(item['scenario'], '/'.join(item['candidates']), 'equal=', item['equal_cases'],
              'max_relative_gap=', round(item['max_relative_gap'], 6))
    print(args.output)


if __name__ == '__main__':
    main()
