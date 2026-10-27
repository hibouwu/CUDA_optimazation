"""cfg_b输入候选：共享读取、GPC应答处理和逐CTA投递。"""
from collections import Counter, deque

from supply_model import wgmma_compute_step
from tma_stateful_source import ProgramGemmSource


class LocalServiceCalendar:
    """先到先预留；一份请求可使用多个空闲片段，已预留的工作不抢占。"""

    def __init__(self):
        self.intervals = []
        self.work_ns = 0.

    def reserve(self, now, earliest, duration):
        self.intervals = [(a, b) for a, b in self.intervals if b > now]
        cursor, remaining, added = earliest, duration, []
        for a, b in self.intervals:
            if b <= cursor:
                continue
            if a > cursor:
                taken = min(a-cursor, remaining)
                added.append((cursor, cursor+taken))
                cursor += taken
                remaining -= taken
                if remaining <= 1e-10:
                    break
            cursor = max(cursor, b)
        if remaining > 1e-10:
            added.append((cursor, cursor+remaining))
        merged = []
        for a, b in sorted(self.intervals+added):
            if merged and a <= merged[-1][1]+1e-10:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self.intervals = merged
        self.work_ns += duration
        return added[0][0], added[-1][1]


class ReplyGroup:
    """两条活动请求共享总处理速率；其余按数据就绪次序等待。"""

    def __init__(self, schedule, returned):
        self.schedule, self.returned = schedule, returned
        self.now = 0.
        self.active, self.waiting = [], deque()
        self.version = 0
        self.busy_ns = 0.
        self.peak_waiting = 0

    def advance(self, t):
        if self.active:
            used = t-self.now
            for request in self.active:
                request['remaining_ns'] = max(0., request['remaining_ns']-used/len(self.active))
            self.busy_ns += used
        self.now = t

    def admit(self):
        while self.waiting and len(self.active) < 2:
            request = self.waiting.popleft()
            request['processing_start_ns'] = self.now
            self.active.append(request)

    def enqueue(self, t, request):
        self.advance(t)
        self.waiting.append(request)
        self.admit()
        self.peak_waiting = max(self.peak_waiting, len(self.waiting))
        self.arm()

    def arm(self):
        self.version += 1
        version = self.version
        if not self.active:
            return
        selected = min(self.active, key=lambda request: request['remaining_ns'])
        end = self.now+len(self.active)*selected['remaining_ns']

        def finished(t):
            if version != self.version:
                return
            self.advance(t)
            # 选定请求在本定时器完成；同刻完成者一并归还处理槽。
            done = [r for r in self.active if r is selected or r['remaining_ns'] == 0.]
            self.active = [r for r in self.active if all(r is not q for q in done)]
            for request in done:
                self.returned(t, request)
            self.admit()
            self.arm()

        self.schedule(end, finished)


class CfgBInputNetwork(ProgramGemmSource):
    version = 'cfg-b-cache-reply-ps2-destination/1'

    def __init__(self, *args, network, phase_frequency_ghz, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = dict(network)
        self.phase_frequency_ghz = list(phase_frequency_ghz)
        self.geometry = {}

    def begin(self, work_index, base, groups=64):
        super().begin(work_index, base, groups)
        self.reply_groups = {}
        self.pending_destinations = {}
        self.destination_frontiers = [0.]*len(self.coordinates)
        self.processing_work_ns = 0.
        self.returned_count = self.delivered_count = 0

    def bind_scheduler(self, schedule):
        self.schedule = schedule
        self.reply_groups = {g: ReplyGroup(schedule, self.reply_returned) for g in set(self.cta_groups)}

    def request_geometry(self, cta, group, atom):
        output, k = divmod(group, self.ktiles)
        source = self.atom_templates[cta][output][atom]
        origin = [source['origin'][0], source['origin'][1]+64*k]
        axis = 0 if atom == 0 else 1
        identity = (source['operand'], source['base_offset_bytes'], tuple(origin),
                    tuple(source['stride_bytes']), tuple(source['shape']))
        if identity not in self.geometry:
            first = (self.addresses[source['operand']]+source['base_offset_bytes']+
                     sum(a*b for a, b in zip(origin, source['stride_bytes'])))
            row_bytes = source['shape'][1-axis]*2
            lines = []
            for row in range(source['shape'][axis]):
                start = first+row*source['stride_bytes'][axis]
                lines.extend(range(start//128, (start+row_bytes-1)//128+1))
            minimum = source['shape'][axis]*((row_bytes+127)//128)
            self.geometry[identity] = dict(lines=tuple(lines),
                logical_bytes=source['shape'][axis]*row_bytes,
                extra128=len(lines)-minimum)
        return identity, self.geometry[identity], output

    def issue(self, t, cta, index, atom, weight, demand_id, notify_ready):
        identity, geometry, output = self.request_geometry(cta, index, atom)
        data_ready = max(self.read(t+self.base, self.domains[cta], line)
                         for line in geometry['lines'])-self.base
        cycles = (self.network['reply_cycles_per_16KiB']*geometry['logical_bytes']/16384+
                  self.network['split_cycles_per_128_extra_lines']*geometry['extra128']/128)
        work = cycles/self.phase_frequency_ghz[output]
        request = dict(id=demand_id, cta=cta, index=index, atom=atom, identity=identity,
            issue_ns=t, data_ready_ns=data_ready, processing_work_ns=work,
            remaining_ns=work, notify_ready=notify_ready)
        self.pending_destinations[demand_id] = request
        self.count += 1
        self.work += weight
        self.processing_work_ns += work
        self.schedule(data_ready, lambda now: self.reply_groups[self.cta_groups[cta]].enqueue(now, request))

    def reply_returned(self, t, request):
        self.returned_count += 1
        request['processing_end_ns'] = t
        self.busy += request['processing_work_ns']
        ready = t+self.frontend_ns
        request['notify_ready'](request['id'], ready,
            dict(source_id=request['id'], source_key=request['identity'], merged=False,
                source_busy_ns=request['processing_work_ns'],
                service_start=request['processing_start_ns'], service_end=t))

    def destination_service(self, t, cta, index, atom, weight):
        start = max(t, self.destination_frontiers[cta])
        end = start+self.prices[2]*weight
        self.destination_frontiers[cta] = end
        return dict(start_ns=start, service_end_ns=end, ready_ns=end)

    def destination_done(self, t, demand_id):
        self.pending_destinations.pop(demand_id)
        self.delivered_count += 1

    def callbacks(self):
        return dict(input_source_async=self, input_source_bind_events=self.bind_scheduler,
                    input_destination_service=self.destination_service,
                    output_source_observer=self.output)

    def network_state(self):
        return dict(parameters=self.network, processing_work_ns=self.processing_work_ns,
            processing_busy_ns=sum(g.busy_ns for g in self.reply_groups.values()),
            peak_waiting=max(g.peak_waiting for g in self.reply_groups.values()),
            returned=self.returned_count, delivered=self.delivered_count,
            pending_destinations=len(self.pending_destinations),
            source_cache='现用两域经验子模块；在途填充共享，逐接收者应答独立')


def address_read_bank(line):
    """公开H200的IPOLY+MODULO与局部channel折叠，保留原bank选择位。"""
    address = line*128
    higher, channel = divmod(address >> 8, 48)
    index, offset = divmod(channel*2+((address >> 7) & 1), 3)
    remainder = (higher & 0x7fff) << 5
    while remainder.bit_length() > 5:
        remainder ^= 0x25 << (remainder.bit_length()-6)
    hashed_subpartition = (index ^ remainder)*3+offset
    return (hashed_subpartition//2 % 24)*2+((address >> 7) & 1)


class AddressRoutedInputNetwork(CfgBInputNetwork):
    """逐接收者的实际行片段经过分区读取，再进入GPC应答。"""

    version = 'cfg-b-address-read-reply-destination/2'

    def begin(self, *args, **kwargs):
        super().begin(*args, **kwargs)
        self.read_frontiers = [[0.]*48 for _ in range(2)]
        self.read_work_ns = 0.
        self.read_fragment_count = 0

    def request_geometry(self, *args):
        identity, geometry, output = super().request_geometry(*args)
        if 'read_banks' not in geometry:
            geometry['read_banks'] = tuple(sorted(Counter(
                address_read_bank(line) for line in geometry['lines']).items()))
        return identity, geometry, output

    def read_return(self, t, request):
        # 同一分区按data-ready到达顺序处理整份请求的片段；不同分区并行。
        frontiers = self.read_frontiers[request['domain']]
        done = t
        for bank, count in request['read_banks']:
            duration = count*request['read_unit_ns']
            frontiers[bank] = max(t, frontiers[bank])+duration
            done = max(done, frontiers[bank])
            self.read_work_ns += duration
            self.read_fragment_count += count
        request['read_end_ns'] = done
        self.schedule(done, lambda now: self.reply_groups[self.cta_groups[request['cta']]].enqueue(now, request))

    def issue(self, t, cta, index, atom, weight, demand_id, notify_ready):
        identity, geometry, output = self.request_geometry(cta, index, atom)
        data_ready = max(self.read(t+self.base, self.domains[cta], line)
                         for line in geometry['lines'])-self.base
        frequency = self.phase_frequency_ghz[output]
        work = self.network['reply_cycles_per_16KiB']*geometry['logical_bytes']/16384/frequency
        request = dict(id=demand_id, cta=cta, index=index, atom=atom, identity=identity,
            issue_ns=t, data_ready_ns=data_ready, domain=self.domains[cta],
            read_banks=geometry['read_banks'],
            read_unit_ns=self.network['bank_cycles_per_fragment']/frequency,
            processing_work_ns=work, remaining_ns=work, notify_ready=notify_ready)
        self.pending_destinations[demand_id] = request
        self.count += 1
        self.work += weight
        self.processing_work_ns += work
        self.schedule(data_ready, lambda now: self.read_return(now, request))

    def network_state(self):
        state = super().network_state()
        state.update(read_work_ns=self.read_work_ns, read_fragment_count=self.read_fragment_count,
            read_bank_frontiers_ns=self.read_frontiers,
            read_mapping='每域48分区，公开IPOLY+MODULO后折叠channel并恢复原bank位；GH200工作假设')
        return state


class ReadSharedInputNetwork(AddressRoutedInputNetwork):
    """同域在途同址读取共享分区服务；每个接收者仍独立应答。"""

    version = 'cfg-b-inflight-read-share-reply-destination/3'

    def begin(self, *args, **kwargs):
        super().begin(*args, **kwargs)
        self.read_pending = [{}, {}]
        self.read_merged_fragments = 0

    def request_geometry(self, *args):
        identity, geometry, output = super().request_geometry(*args)
        if 'line_banks' not in geometry:
            geometry['line_banks'] = tuple((line, address_read_bank(line)) for line in geometry['lines'])
        return identity, geometry, output

    def read_return(self, t, request):
        domain = request['domain']
        frontiers, pending = self.read_frontiers[domain], self.read_pending[domain]
        done = t
        for line, bank in self.geometry[request['identity']]['line_banks']:
            previous = pending.get(line)
            limit = self.network.get('max_read_subscribers')
            if previous is not None and previous[0] > t and (limit is None or previous[1] < limit):
                ready = previous[0]
                previous[1] += 1
                self.read_merged_fragments += 1
            else:
                ready = max(t, frontiers[bank])+request['read_unit_ns']
                frontiers[bank] = ready
                pending[line] = [ready, 1]
                self.read_work_ns += request['read_unit_ns']
                self.read_fragment_count += 1
            done = max(done, ready)
        request['read_end_ns'] = done
        self.schedule(done, lambda now: self.read_finished(now, request))

    def read_finished(self, t, request):
        self.reply_groups[self.cta_groups[request['cta']]].enqueue(t, request)

    def network_state(self):
        return dict(super().network_state(), read_merged_fragments=self.read_merged_fragments,
            max_read_subscribers=self.network.get('max_read_subscribers'),
            read_sharing='同域同128B源行，仅合并分区读取尚未完成的请求；应答和投递独立')


class JointLocalInputNetwork(ReadSharedInputNetwork):
    """相邻B复用的计算操作数与TMA目的写共享本地服务。"""

    version = 'cfg-b-local-input-compute-shared/1'

    def __init__(self, program, *args, **kwargs):
        super().__init__(program, *args, **kwargs)
        self.compute_repeats = program.get('runtime_intervention', {}).get('wgmma_repeats', 1)

    def begin(self, *args, **kwargs):
        super().begin(*args, **kwargs)
        self.local_services = [LocalServiceCalendar() for _ in self.coordinates]
        self.local_compute_work_ns = self.local_input_work_ns = 0.
        self.local_compute_delay_ns = self.local_input_delay_ns = 0.

    def destination_service(self, t, cta, index, atom, weight):
        frequency = self.phase_frequency_ghz[index//self.ktiles]
        work_bytes = 16384*weight
        if self.network.get('local_input_work') == 'source_fragments128':
            _, geometry, _ = self.request_geometry(cta, index, atom)
            work_bytes = 128*len(geometry['lines'])
        duration = work_bytes/self.network['local_bytes_per_cycle']/frequency
        earliest = max(t, self.destination_frontiers[cta])
        start, end = self.local_services[cta].reserve(t, earliest, duration)
        self.destination_frontiers[cta] = end
        self.local_input_work_ns += duration
        self.local_input_delay_ns += end-earliest-duration
        return dict(start_ns=start, service_end_ns=end, ready_ns=end,
                    busy_ns=duration)

    def compute_service(self, t, cta, index, previous_service_end, previous_complete, **costs):
        frequency = self.phase_frequency_ghz[index//self.ktiles]
        pairs = 4*self.compute_repeats
        period = costs['compute_duration']/pairs
        memory = 8192/self.network['local_bytes_per_cycle']/frequency
        cursor = max(t, previous_service_end)
        nominal_end = cursor+costs['compute_duration']
        for _ in range(pairs):
            _, end = self.local_services[cta].reserve(t, cursor, memory)
            cursor = end+max(0., period-memory)
        self.local_compute_work_ns += pairs*memory
        self.local_compute_delay_ns += cursor-nominal_end
        effective = dict(costs, compute_duration=cursor-max(t, previous_service_end))
        return wgmma_compute_step(t, previous_service_end, previous_complete, **effective)

    def callbacks(self):
        return dict(super().callbacks(), compute_service=self.compute_service)

    def network_state(self):
        return dict(super().network_state(), local_joint_service=dict(
            bytes_per_cycle=self.network['local_bytes_per_cycle'],
            compute_work_ns=self.local_compute_work_ns, input_work_ns=self.local_input_work_ns,
            compute_delay_ns=self.local_compute_delay_ns, input_delay_ns=self.local_input_delay_ns,
            calendar_work_ns=sum(s.work_ns for s in self.local_services),
            input_work=self.network.get('local_input_work', 'payload_bytes'),
            arbitration='逐CTA先到先预留空闲区间；计算每对8KiB与固定非访存余量',
            qualification='相邻B复用与本地共同服务工作假设'))


class PortOrderedInputNetwork(ReadSharedInputNetwork):
    """源行取得读取分区后才查询cache；在途数据返回可被有限订阅。"""

    version = 'cfg-b-port-ordered-cache-reply-destination/1'

    def begin(self, *args, **kwargs):
        super().begin(*args, **kwargs)
        self.lookup_count = 0
        self.lookup_queue_ns = 0.

    def fragment_ready(self, request, ready):
        request['data_ready_ns'] = max(request['data_ready_ns'], ready)
        request['fragments_pending'] -= 1
        if request['fragments_pending'] == 0:
            end = request['data_ready_ns']
            self.schedule(end, lambda now: self.reply_groups[self.cta_groups[request['cta']]].enqueue(now, request))

    def lookup(self, t, domain, line, entry):
        ready = self.read(t+self.base, domain, line)-self.base
        entry['ready_ns'] = ready
        self.lookup_count += 1
        for request in entry.pop('waiting'):
            self.fragment_ready(request, ready)

    def issue(self, t, cta, index, atom, weight, demand_id, notify_ready):
        identity, geometry, output = self.request_geometry(cta, index, atom)
        frequency = self.phase_frequency_ghz[output]
        work = self.network['reply_cycles_per_16KiB']*geometry['logical_bytes']/16384/frequency
        request = dict(id=demand_id, cta=cta, index=index, atom=atom, identity=identity,
            issue_ns=t, data_ready_ns=t, fragments_pending=len(geometry['line_banks']),
            processing_work_ns=work, remaining_ns=work, notify_ready=notify_ready)
        self.pending_destinations[demand_id] = request
        self.count += 1
        self.work += weight
        self.processing_work_ns += work
        domain = self.domains[cta]
        pending, frontiers = self.read_pending[domain], self.read_frontiers[domain]
        duration = self.network['bank_cycles_per_fragment']/frequency
        limit = self.network['max_read_subscribers']
        for line, bank in geometry['line_banks']:
            entry = pending.get(line)
            if (entry is not None and (entry['ready_ns'] is None or entry['ready_ns'] > t)
                    and entry['subscribers'] < limit):
                entry['subscribers'] += 1
                self.read_merged_fragments += 1
                if entry['ready_ns'] is None:
                    entry['waiting'].append(request)
                else:
                    self.fragment_ready(request, entry['ready_ns'])
            else:
                start = max(t, frontiers[bank])
                end = start+duration
                frontiers[bank] = end
                self.lookup_queue_ns += start-t
                self.read_work_ns += duration
                self.read_fragment_count += 1
                entry = dict(ready_ns=None, subscribers=1, waiting=[request])
                pending[line] = entry
                self.schedule(end, lambda now, d=domain, line=line, entry=entry: self.lookup(now, d, line, entry))

    def network_state(self):
        return dict(super().network_state(), lookup_count=self.lookup_count,
            lookup_queue_ns=self.lookup_queue_ns,
            read_sharing='同域同128B读取从端口排队至数据返回最多4份订阅；应答／投递独立',
            source_cache='读取分区服务后查询cache；lookup时更新既有LRU／在途缺失状态')


class FormationFirstInputNetwork(ReadSharedInputNetwork):
    """每接收者形成源tag引用后查cache；同址读取共享，目的投递独立。"""

    version = 'cfg-b-formation-cache-shared-read-destination/1'

    def bind_scheduler(self, schedule):
        self.schedule = schedule
        self.reply_groups = {g: ReplyGroup(schedule, self.formed) for g in set(self.cta_groups)}

    def issue(self, t, cta, index, atom, weight, demand_id, notify_ready):
        identity, geometry, output = self.request_geometry(cta, index, atom)
        frequency = self.phase_frequency_ghz[output]
        work = len(geometry['lines'])*self.network['formation_cycles_per_tag']/frequency
        request = dict(id=demand_id, cta=cta, index=index, atom=atom, identity=identity,
            issue_ns=t, domain=self.domains[cta], read_banks=geometry['read_banks'],
            read_unit_ns=self.network['bank_cycles_per_fragment']/frequency,
            processing_work_ns=work, remaining_ns=work, notify_ready=notify_ready)
        self.pending_destinations[demand_id] = request
        self.count += 1
        self.work += weight
        self.processing_work_ns += work
        self.reply_groups[self.cta_groups[cta]].enqueue(t, request)

    def formed(self, t, request):
        request['formation_end_ns'] = t
        geometry = self.geometry[request['identity']]
        ready = max(self.read(t+self.base, request['domain'], line)
                    for line in geometry['lines'])-self.base
        request['data_ready_ns'] = ready
        self.schedule(ready, lambda now: self.read_return(now, request))

    def read_finished(self, t, request):
        self.reply_returned(t, request)

    def network_state(self):
        return dict(super().network_state(),
            formation='每接收者128B源tag引用；每GPC两活动流共享总速率，形成完成后查询cache',
            source_cache='形成完成时更新既有LRU／在途缺失；返回后共享分区读取，再独立投递')
