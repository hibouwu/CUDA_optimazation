"""单信用源和共享目的回传的最小接线反例。"""
from collections import deque
import json
import unittest

from supply_model import shared_stage_events
from tma_tensor_multicta import TensorSourceService


class OneCreditSource:
    """测试服务：读取1、延迟10；订阅截止service_end，信用等全部目的完成。"""

    def __init__(self):
        self.current = None
        self.waiting = deque()
        self.by_demand = {}
        self.sources = []
        self.completed = []
        self.released = []

    def notify(self, source, demand_id):
        source['notify'](demand_id, source['service_end']+10., dict(
            source_id=source['id'], source_key=source['key'],
            merged=demand_id != source['owner'], source_busy_ns=1.,
            service_start=source['start'], service_end=source['service_end']))

    def admit(self, t, source):
        self.current = source
        source.update(start=t, service_end=t+1.)
        for demand_id in source['subscribers']:
            self.notify(source, demand_id)

    def issue(self, t, cta, index, atom, weight, demand_id, notify_ready):
        key = 'A' if index == atom == 0 and cta in (0, 2) else (cta, index, atom)
        source = self.current
        if source is not None and source['key'] == key and t < source['service_end']:
            source['subscribers'].append(demand_id)
            source['pending'].add(demand_id)
            self.by_demand[demand_id] = source
            self.notify(source, demand_id)
            return
        source = dict(id=len(self.sources), key=key, owner=demand_id,
            subscribers=[demand_id], pending={demand_id}, notify=notify_ready, issue=t)
        self.sources.append(source)
        self.by_demand[demand_id] = source
        if self.current is None:
            self.admit(t, source)
        else:
            self.waiting.append(source)

    def destination_done(self, t, demand_id):
        self.completed.append((t, demand_id))
        source = self.by_demand.pop(demand_id)
        source['pending'].remove(demand_id)
        if not source['pending']:
            self.released.append((t, source['id']))
            self.current = None
            if self.waiting:
                self.admit(t, self.waiting.popleft())


class SharedDestination:
    def __init__(self, ready_delay=0.):
        self.frontier = 0.
        self.ready_delay = ready_delay
        self.calls = []

    def __call__(self, t, cta, index, atom, weight):
        self.calls.append((t, cta, index, atom, weight))
        start = max(t, self.frontier)
        self.frontier = start+1.
        return dict(start_ns=start, service_end_ns=self.frontier,
                    ready_ns=self.frontier+self.ready_delay)


class ReadyCreditSource(OneCreditSource):
    def bind_scheduler(self, schedule):
        self.schedule = schedule

    def admit(self, t, source):
        super().admit(t, source)
        self.schedule(t+11., self.credit_ready)

    def credit_ready(self, t):
        self.released.append((t, self.current['id']))
        self.current = None
        if self.waiting:
            self.admit(t, self.waiting.popleft())

    def destination_done(self, t, demand_id):
        self.completed.append((t, demand_id))
        self.by_demand.pop(demand_id)['pending'].remove(demand_id)


class DeferredNotifySource:
    """只延后通知的测试夹具：A 晚于 B 回调，逻辑 issue 次序保持不变。"""

    def __init__(self):
        self.notified = []

    def bind_scheduler(self, schedule):
        self.schedule = schedule

    def issue(self, t, cta, index, atom, weight, demand_id, notify_ready):
        def notify(ready):
            self.notified.append(demand_id)
            notify_ready(demand_id,ready,dict(source_id=demand_id,
                source_key=(cta,index,atom),merged=False,source_busy_ns=0.,
                service_start=t,service_end=t))
        self.schedule(t+(20. if atom==0 else 1.),notify)

    def destination_done(self, t, demand_id):
        pass


def run_case(trace=True, shared_destination=True, ready_delay=0., third_issue=.5):
    source = OneCreditSource()
    destination = SharedDestination(ready_delay)
    result = shared_stage_events(2, 2, 3, service_ns=0., read_latency_ns=0., compute_ns=1.,
        output_counts=[1, 1, 1], operand_gap_ns=1000.,
        diagnostic_producer_offsets_ns=[0., .25, third_issue], destination_service_ns=1.,
        input_source_async=source,
        input_destination_service=destination if shared_destination else None,
        trace_requests=trace)
    return result, source, destination


class AsyncSourceTests(unittest.TestCase):
    def test_source_timer_wakes_queue_before_destination_completion(self):
        source = ReadyCreditSource()
        result = shared_stage_events(2, 2, 1, service_ns=0., read_latency_ns=0., compute_ns=1.,
            output_counts=[1], input_source_async=source,
            input_source_bind_events=source.bind_scheduler,
            input_destination_service=SharedDestination(ready_delay=100.), trace_requests=True)
        self.assertEqual(source.sources[1]['start'], 11.)
        self.assertEqual(source.released[:2], [(11., 0), (22., 1)])
        self.assertEqual(source.completed[0][0], 112.)
        self.assertEqual(result['input_delivery_count'], 6)
        self.assertEqual(source.by_demand, {})

    def test_queued_B_starts_after_both_A_destinations_at_13(self):
        result, source, destination = run_case()
        first = [d for d in result['deliveries'] if d['kind'] == 'issue_a' and d['index'] == 0]
        by_cta = {d['cta']:d for d in first}
        self.assertEqual([by_cta[c]['arrival_ns'] for c in range(3)], [0., .25, .5])
        self.assertEqual(by_cta[0]['source_id'], by_cta[2]['source_id'])
        self.assertEqual([by_cta[c]['ready_ns'] for c in (0, 2, 1)], [12., 13., 25.])
        self.assertEqual(source.sources[1]['start'], 13.)
        self.assertEqual(source.released[0], (13., 0))
        self.assertEqual(destination.calls[:3], [(11., 0, 0, 0, 1.),
                                               (11., 2, 0, 0, 1.), (24., 1, 0, 0, 1.)])
        self.assertEqual([t for t, _ in source.completed], sorted(t for t, _ in source.completed))
        self.assertEqual(result['input_source_count'], 17)
        self.assertEqual(result['source_busy_ns'], 17.)
        self.assertEqual(result['input_merged_count'], 1)
        self.assertEqual(result['input_delivery_count'], 18)
        self.assertEqual(result['destination_busy_ns'], 18.)
        self.assertEqual(len(source.completed), 18)

    def test_credit_waits_for_ready_not_destination_service_end(self):
        result, source, _ = run_case(ready_delay=2.)
        first = [d for d in result['deliveries'] if d['source_id'] == 0]
        self.assertEqual([d['service_end_ns'] for d in first], [12., 13.])
        self.assertEqual([d['ready_ns'] for d in first], [14., 15.])
        self.assertEqual(source.released[0], (15., 0))
        self.assertEqual(source.sources[1]['start'], 15.)

    def test_service_end_closes_subscription(self):
        result, source, _ = run_case(third_issue=1.)
        first = {d['cta']:d for d in result['deliveries'] if d['kind'] == 'issue_a' and d['index'] == 0}
        self.assertNotEqual(first[0]['source_id'], first[2]['source_id'])
        self.assertFalse(first[2]['merged'])
        self.assertEqual(source.sources[1]['start'], 12.)

    def test_trace_off_preserves_events_and_completion_callbacks(self):
        traced, traced_source, _ = run_case()
        compact, compact_source, _ = run_case(trace=False)
        for key in ('sources', 'input_demands', 'deliveries', 'requests', 'local_epilogue_intervals'):
            traced[key] = []
        self.assertEqual(compact, traced)
        self.assertEqual(compact_source.completed, traced_source.completed)
        self.assertEqual(compact_source.released, traced_source.released)

    def test_default_destination_stays_per_cta(self):
        result, source, destination = run_case(shared_destination=False)
        first = [d for d in result['deliveries'] if d['source_id'] == 0]
        self.assertEqual([d['ready_ns'] for d in first], [12., 12.])
        self.assertEqual(source.sources[1]['start'], 12.)
        self.assertEqual(destination.calls, [])

    def test_destination_callback_without_async_source(self):
        destination = SharedDestination(ready_delay=2.)
        result = shared_stage_events(2, 2, 2, service_ns=0., read_latency_ns=0., compute_ns=1.,
            output_counts=[1, 1], input_destination_service=destination, trace_requests=True)
        self.assertEqual(len(destination.calls), 12)
        self.assertEqual(result['input_delivery_count'], 12)
        self.assertEqual(result['destination_busy_ns'], 12.)
        self.assertEqual([d['service_end_ns'] for d in result['deliveries']], list(range(1, 13)))
        self.assertEqual([d['ready_ns'] for d in result['deliveries']], list(range(3, 15)))

    def test_every_source_path_respects_independent_destination_admission(self):
        for mode in ('builtin','sync','async'):
            for costs,expected in ((dict(input_min_latency_ns=100.),100.),
                    (dict(input_path_service_ns=20.,input_min_latency_ns=3.),23.),
                    (dict(destination_latency_ns=100.),111.)):
                with self.subTest(source=mode,costs=costs):
                    kwargs={}
                    if mode=='sync':
                        source=TensorSourceService('local_fifo',1,32768,service=1.,latency=10.)
                        def submit(t,cta,index,atom,weight):
                            key=(cta,index,atom)
                            before=source.busy
                            return dict(source.submit(t,cta,key,weight),source_key=key,
                                source_busy_ns=source.busy-before)
                        kwargs['input_source_service']=submit
                    elif mode=='async':
                        kwargs['input_source_async']=OneCreditSource()
                    result=shared_stage_events(2,2,1,service_ns=1.,read_latency_ns=10.,
                        compute_ns=1.,output_counts=[1],trace_requests=True,**kwargs,**costs)
                    first=result['deliveries'][0]
                    self.assertEqual(first['source_ready_ns'],11.)
                    self.assertEqual(first['destination_arrival_ns'],expected)
                    self.assertEqual(first['ready_ns'],expected)
                    self.assertEqual(result['input_delivery_count'],6)
                    for demand in result['deliveries']:
                        path_end=demand.get('input_path_end_ns',demand['arrival_ns'])
                        admission=max(demand['source_ready_ns']+costs.get('destination_latency_ns',0.),
                            path_end+costs.get('input_min_latency_ns',0.))
                        self.assertEqual(demand['destination_arrival_ns'],admission)

    def test_front_end_reservation_uses_issue_order_before_deferred_notifications(self):
        source=DeferredNotifySource()
        result=shared_stage_events(2,2,1,service_ns=0.,read_latency_ns=0.,compute_ns=1.,
            output_counts=[1],input_source_async=source,input_source_bind_events=source.bind_scheduler,
            input_path_service_ns=10.,trace_requests=True)
        demands=sorted(result['input_demands'],key=lambda d:d['demand_id'])
        self.assertEqual(source.notified,[1,2,4,5,0,3])
        self.assertEqual([d['input_path_end_ns'] for d in demands],[10.,15.,20.,30.,35.,40.])
        self.assertEqual(result['input_path_service']['busy_ns'],40.)
        self.assertEqual(result['input_delivery_count'],6)

    def test_explicit_zero_admission_costs_preserve_complete_default_output(self):
        baseline,_,_=run_case()
        source=OneCreditSource()
        zero=shared_stage_events(2,2,3,service_ns=0.,read_latency_ns=0.,compute_ns=1.,
            output_counts=[1,1,1],operand_gap_ns=1000.,
            diagnostic_producer_offsets_ns=[0.,.25,.5],destination_service_ns=1.,
            input_source_async=source,input_destination_service=SharedDestination(),trace_requests=True,
            input_min_latency_ns=0.,input_path_service_ns=0.,destination_latency_ns=0.)
        self.assertEqual(json.dumps(zero,separators=(',',':')),json.dumps(baseline,separators=(',',':')))


if __name__ == '__main__':
    unittest.main()
