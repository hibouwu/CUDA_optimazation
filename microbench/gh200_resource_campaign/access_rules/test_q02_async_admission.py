"""异步准入返回的精确事件夹具；两槽 PS 只用于接线验证。"""
from collections import deque
import json
import unittest

from supply_model import shared_stage_events


class KnownAdmissionSource:
    def __init__(self, first_ready):
        self.first_ready = first_ready

    def issue_with_admission(self, t, cta, index, atom, weight, demand_id,
                             notify_ready, notify_admitted):
        start = 10. if index == atom == 0 else t
        end = self.first_ready if index == atom == 0 else start + 20.
        notify_admitted(demand_id, start)
        notify_ready(demand_id, end, dict(source_id=demand_id,
            source_key=(cta, index, atom), merged=False, source_busy_ns=end-start,
            service_start=start, service_end=end))

    def destination_done(self, t, demand_id):
        pass


class TwoSlotProcessorSharing:
    def __init__(self):
        self.now = 0.
        self.active = []
        self.waiting = deque()
        self.version = 0
        self.starts = {}
        self.ends = {}

    def bind_scheduler(self, schedule):
        self.schedule = schedule

    def advance(self, t):
        if self.active:
            service = (t-self.now) / len(self.active)
            for request in self.active:
                request['remaining'] -= service
        self.now = t
        finished = [request for request in self.active if request['remaining'] <= 0.]
        self.active = [request for request in self.active if request['remaining'] > 0.]
        for request in finished:
            demand_id = request['id']
            self.ends[demand_id] = t
            request['ready'](demand_id, t, dict(source_id=demand_id,
                source_key=request['key'], merged=False,
                source_busy_ns=request['work'], service_start=self.starts[demand_id],
                service_end=t))

    def admit(self):
        while self.waiting and len(self.active) < 2:
            request = self.waiting.popleft()
            self.starts[request['id']] = self.now
            self.active.append(request)
            request['admitted'](request['id'], self.now)

    def schedule_completion(self):
        self.version += 1
        version = self.version
        if not self.active:
            return
        finish = self.now + len(self.active)*min(r['remaining'] for r in self.active)

        def complete(t):
            if version != self.version:
                return
            self.advance(t)
            self.admit()
            self.schedule_completion()

        self.schedule(finish, complete)

    def issue_with_admission(self, t, cta, index, atom, weight, demand_id,
                             notify_ready, notify_admitted):
        self.advance(t)
        work = 20. if atom == 0 else 10.
        self.waiting.append(dict(id=demand_id, key=(cta, index, atom), work=work,
            remaining=work, ready=notify_ready, admitted=notify_admitted))
        self.admit()
        self.schedule_completion()

    def destination_done(self, t, demand_id):
        pass


class LegacySource:
    def issue(self, t, cta, index, atom, weight, demand_id, notify_ready):
        start = t+10. if index == atom == 0 else t
        notify_ready(demand_id, start+20., dict(source_id=demand_id,
            source_key=(cta, index, atom), merged=False, source_busy_ns=20.,
            service_start=start, service_end=start+20., issue_return_ns=start))
        return start

    def destination_done(self, t, demand_id):
        pass


def legacy_reports(engine):
    """修复前后各调用三次，完整报告经 JSON 归一化后逐值比较。"""
    kwargs = dict(service_ns=0., read_latency_ns=0., compute_ns=1.,
        output_counts=[1], operand_gap_ns=1., producer_gap_ns=3., trace_requests=True)
    next_id = 0

    def synchronous(t, cta, index, atom, weight):
        nonlocal next_id
        start = t+10. if index == atom == 0 else t
        served = dict(source_id=next_id, source_key=(cta, index, atom), merged=False,
            source_busy_ns=20., service_start=start, service_end=start+20.,
            source_ready=start+20., issue_return_ns=start)
        next_id += 1
        return served

    reports = dict(
        legacy_async=engine(2, 2, 1, input_source_async=LegacySource(), **kwargs),
        synchronous=engine(2, 2, 1, input_source_service=synchronous, **kwargs),
        zero_default=engine(2, 2, 1, **kwargs))
    return json.loads(json.dumps(reports, allow_nan=False))


class AsyncAdmissionTests(unittest.TestCase):
    def test_next_issue_follows_admission_not_readiness(self):
        for ready in (30., 100.):
            with self.subTest(ready=ready):
                report = shared_stage_events(2, 2, 1, service_ns=0.,
                    read_latency_ns=0., compute_ns=1., output_counts=[1],
                    operand_gap_ns=1., producer_gap_ns=3., trace_requests=True,
                    input_source_async=KnownAdmissionSource(ready))
                actor = report['ctas'][0]
                self.assertEqual([actor[field][0] for field in
                    ('issued_a', 'issued_b0', 'issued_b1')], [0., 11., 12.])
                self.assertEqual(actor['issued_a'][1], 15.)
                first = report['input_demands'][0]
                self.assertEqual(first['arrival_ns'], 0.)
                self.assertEqual(first['issue_return_ns'], 10.)
                self.assertEqual(first['source_ready_ns'], ready)
                self.assertEqual(actor['ready'][0], max(ready, 32.))

    def test_two_slots_delay_admission_and_revise_active_completion(self):
        source = TwoSlotProcessorSharing()
        report = shared_stage_events(2, 2, 1, service_ns=0., read_latency_ns=0.,
            compute_ns=1., output_counts=[1], operand_gap_ns=2., producer_gap_ns=100.,
            destination_service_ns=100., trace_requests=True,
            input_source_async=source, input_source_bind_events=source.bind_scheduler)
        first = report['input_demands'][:3]
        self.assertEqual([d['arrival_ns'] for d in first], [0., 2., 4.])
        self.assertEqual([source.starts[i] for i in range(3)], [0., 2., 22.])
        self.assertEqual([source.ends[i] for i in range(3)], [38., 22., 40.])
        self.assertEqual([d['issue_return_ns'] for d in first], [0., 2., 22.])
        self.assertEqual(report['ctas'][0]['issued_b'][0], 4.)
        self.assertEqual(report['ctas'][0]['issued_a'][1], 122.)
        self.assertEqual([d['demand_id'] for d in report['deliveries'][:3]], [1, 0, 2])
        self.assertEqual(report['deliveries'][0]['ready_ns'], 72.)
        self.assertLess(source.starts[2], report['deliveries'][0]['ready_ns'])


if __name__ == '__main__':
    unittest.main()
