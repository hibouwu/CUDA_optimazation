"""单CTA连续请求：两槽共享源速率，按字节和额外行覆盖定价。"""
from collections import deque
import math


class ExcessCoverageSource:
    def __init__(self, keys, base_addresses, prices, *, completion_tail_ns, frequency_ghz=1.,
                 return_at_enqueue=False):
        if len(keys) != 1:
            raise ValueError('ExcessCoverageSource requires one CTA')
        values = [prices[name] for name in ('lambda', 'beta32', 'beta128')]
        if (not math.isfinite(frequency_ghz) or frequency_ghz <= 0 or
                not math.isfinite(completion_tail_ns) or completion_tail_ns < 0 or
                any(not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError('finite nonnegative prices/tail and positive frequency required')
        self.keys = keys
        self.base_addresses = dict(base_addresses)
        self.prices = dict(zip(('lambda', 'beta32', 'beta128'), map(float, values)))
        self.completion_tail_ns = float(completion_tail_ns)
        self.frequency_ghz = float(frequency_ghz)
        self.return_at_enqueue = return_at_enqueue
        self.now = 0.
        self.active = []
        self.waiting = deque()
        self.timer_version = 0
        self.starts = {}
        self.ends = {}
        self.coverage = {}

    def bind_scheduler(self, schedule):
        self.schedule = schedule

    def features(self, key):
        """按实际分配地址逐行计覆盖，单位为byte和逻辑granule数。"""
        if key not in self.coverage:
            operand, _, _, _, strides, base, origin, shape = key
            first = self.base_addresses[operand] + base + sum(a*b for a, b in zip(origin, strides))
            axis = 0 if operand == 'A' else 1
            rows, row_bytes = shape[axis], 2*shape[1-axis]
            counts = [sum((first+r*strides[axis]+row_bytes-1)//grain -
                          (first+r*strides[axis])//grain + 1 for r in range(rows))
                      for grain in (32, 128)]
            self.coverage[key] = dict(bytes=rows*row_bytes, N32=counts[0], N128=counts[1])
        return self.coverage[key]

    def work_ns(self, key):
        f = self.features(key)
        cycles = (self.prices['lambda']*f['bytes']/8192 +
                  self.prices['beta32']*(f['N32']-f['bytes']/32)/256 +
                  self.prices['beta128']*(f['N128']-f['bytes']/128)/64)
        return cycles/self.frequency_ghz

    def advance(self, t):
        if self.active:
            service = (t-self.now)/len(self.active)
            for request in self.active:
                request['remaining'] = max(0., request['remaining']-service)
        self.now = t

    def complete(self, request):
        demand_id = request['id']
        self.ends[demand_id] = self.now
        request['ready'](demand_id, self.now+self.completion_tail_ns,
            dict(source_id=demand_id, source_key=request['key'], merged=False,
                 source_busy_ns=request['work'], service_start=self.starts[demand_id],
                 service_end=self.now))

    def admit(self):
        while self.waiting and len(self.active) < 2:
            request = self.waiting.popleft()
            self.starts[request['id']] = self.now
            if not self.return_at_enqueue:
                request['admitted'](request['id'], self.now)
            if request['work'] == 0:
                self.complete(request)
            else:
                self.active.append(request)

    def schedule_completion(self):
        self.timer_version += 1
        version = self.timer_version
        if not self.active:
            return
        selected = min(self.active, key=lambda r: r['remaining'])
        finish = self.now + len(self.active)*selected['remaining']

        def complete_at(t):
            if version != self.timer_version:
                return
            self.advance(t)
            # 选定请求在该定时器上完成；减法的1ulp残余不延长其生命周期。
            finished = [r for r in self.active if r is selected or r['remaining'] == 0.]
            self.active = [r for r in self.active if all(r is not q for q in finished)]
            for request in finished:
                self.complete(request)
            self.admit()
            self.schedule_completion()

        self.schedule(finish, complete_at)

    def issue_with_admission(self, t, cta, index, atom, weight, demand_id,
                             notify_ready, notify_admitted):
        self.advance(t)
        key = self.keys[cta][index][atom]
        work = self.work_ns(key)
        self.waiting.append(dict(id=demand_id, key=key, work=work, remaining=work,
                                 ready=notify_ready, admitted=notify_admitted))
        if self.return_at_enqueue:
            notify_admitted(demand_id, t)
        self.admit()
        self.schedule_completion()

    def destination_done(self, t, demand_id):
        pass


class BoundedExcessCoverageSource(ExcessCoverageSource):
    """单CTA的PS2源和一个waiting信用；pending尚未获得受理返回。"""
    def __init__(self, keys, base_addresses, prices, *, completion_tail_ns, frequency_ghz=1.,
                 return_at_enqueue=True):
        if not return_at_enqueue:
            raise ValueError('BoundedExcessCoverageSource returns at bounded enqueue')
        super().__init__(keys, base_addresses, prices, completion_tail_ns=completion_tail_ns,
                         frequency_ghz=frequency_ghz, return_at_enqueue=True)
        self.pending = deque()
        self.admissions = {}

    def admit(self):
        while True:
            while self.waiting and len(self.active) < 2:
                request = self.waiting.popleft()
                self.starts[request['id']] = self.now
                if request['work'] == 0:
                    self.complete(request)
                else:
                    self.active.append(request)
            if not self.pending or self.waiting:
                return
            request = self.pending.popleft()
            self.waiting.append(request)
            self.admissions[request['id']] = self.now
            request['admitted'](request['id'], self.now)

    def issue_with_admission(self, t, cta, index, atom, weight, demand_id,
                             notify_ready, notify_admitted):
        self.advance(t)
        key = self.keys[cta][index][atom]
        work = self.work_ns(key)
        self.pending.append(dict(id=demand_id, key=key, work=work, remaining=work,
                                 ready=notify_ready, admitted=notify_admitted))
        self.admit()
        self.schedule_completion()
