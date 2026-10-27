"""真实请求覆盖价格与连续PS源服务的两个边界。"""
import heapq
import unittest

from q02_excess_source import BoundedExcessCoverageSource, ExcessCoverageSource
from supply_model import shared_stage_events


def keys():
    return [[(
        ('A','fp16','SW128',(),(8192,2),0,(0,64*g),(128,64)),
        ('B','fp16','SW128',(),(2,4608),0,(0,64*g),(64,64)),
        ('B','fp16','SW128',(),(2,4608),0,(64,64*g),(64,64)),
    ) for g in range(2)]]


class ExcessSourceTests(unittest.TestCase):
    def test_queue_return_precedes_service_start_without_changing_first_group(self):
        source = ExcessCoverageSource(keys(),dict(A=0,B=1<<25),
            {'lambda':10.,'beta32':0.,'beta128':0.},completion_tail_ns=3.,
            return_at_enqueue=True)
        result = shared_stage_events(2,2,1,service_ns=0.,read_latency_ns=0.,
            compute_ns=1.,output_counts=[1],operand_gap_ns=2.,producer_gap_ns=100.,
            destination_service_ns=4.,input_source_async=source,
            input_source_bind_events=source.bind_scheduler,trace_requests=True)
        self.assertEqual([d['issue_return_ns'] for d in result['input_demands'][:3]],[0.,2.,4.])
        self.assertEqual([source.starts[i] for i in range(3)],[0.,2.,22.])
        self.assertEqual([source.ends[i] for i in range(3)],[38.,22.,40.])
        self.assertEqual([d['ready_ns'] for d in result['deliveries'][:3]],[27.,45.,47.])
        self.assertEqual(result['ctas'][0]['issued_a'][1],104.)

    def test_actual_request_work_and_shared_group_timeline(self):
        source = ExcessCoverageSource(keys(),dict(A=0,B=1<<25),
            {'lambda':10.,'beta32':0.,'beta128':0.},
            completion_tail_ns=3.)
        result = shared_stage_events(2,2,1,service_ns=0.,read_latency_ns=0.,
            compute_ns=1.,output_counts=[1],operand_gap_ns=2.,producer_gap_ns=100.,
            destination_service_ns=4.,input_source_async=source,
            input_source_bind_events=source.bind_scheduler,trace_requests=True)
        self.assertEqual([source.starts[i] for i in range(3)],[0.,2.,22.])
        self.assertEqual([source.ends[i] for i in range(3)],[38.,22.,40.])
        self.assertEqual([d['demand_id'] for d in result['deliveries'][:3]],[1,0,2])
        self.assertEqual([d['ready_ns'] for d in result['deliveries'][:3]],[27.,45.,47.])
        self.assertEqual(result['ctas'][0]['issued_b'][0],4.)
        self.assertEqual(result['ctas'][0]['issued_a'][1],122.)

    def test_zero_work_keeps_default_actor_and_completion(self):
        source = ExcessCoverageSource(keys(),dict(A=0,B=1<<25),
            {'lambda':0.,'beta32':0.,'beta128':0.},completion_tail_ns=0.)
        parameters = dict(service_ns=0.,read_latency_ns=0.,compute_ns=1.,output_counts=[1])
        baseline = shared_stage_events(2,2,1,**parameters)
        result = shared_stage_events(2,2,1,input_source_async=source,
            input_source_bind_events=source.bind_scheduler,**parameters)
        self.assertEqual(result['ctas'],baseline['ctas'])
        self.assertEqual(result['makespan_ns'],baseline['makespan_ns'])
        self.assertEqual(result['input_delivery_count'],6)


class BoundedSourceTests(unittest.TestCase):
    def direct_source(self, price=10.):
        source = BoundedExcessCoverageSource(keys(), dict(A=0, B=1<<25),
            {'lambda':price, 'beta32':0., 'beta128':0.}, completion_tail_ns=0.)
        events, admissions, ready = [], [], []
        def schedule(t, callback):
            heapq.heappush(events, (t, len(events), callback))
        source.bind_scheduler(schedule)
        def admitted(demand_id, t):
            admissions.append((demand_id, t))
            self.assertLessEqual(len(source.active), 2)
            self.assertLessEqual(len(source.waiting), 1)
        def completed(demand_id, t, served):
            ready.append((demand_id, t))
        def issue(demand_id, t=0.):
            # All direct requests use an8KiB B atom, so their work is price ns.
            source.issue_with_admission(t, 0, 0, 1, 1., demand_id, completed, admitted)
            self.assertLessEqual(len(source.active), 2)
            self.assertLessEqual(len(source.waiting), 1)
        def drain():
            count = 0
            while events:
                t, _, callback = heapq.heappop(events)
                callback(t)
                self.assertLessEqual(len(source.active), 2)
                self.assertLessEqual(len(source.waiting), 1)
                count += 1
                self.assertLess(count, 100)
        return source, admissions, ready, issue, drain

    def test_waiting_credit_delays_fourth_admission_and_keeps_fifo(self):
        source, admissions, ready, issue, drain = self.direct_source()
        for demand_id in range(6):
            issue(demand_id)
        self.assertEqual(admissions, [(0,0.), (1,0.), (2,0.)])
        self.assertEqual([request['id'] for request in source.waiting], [2])
        self.assertEqual([request['id'] for request in source.pending], [3,4,5])
        drain()
        self.assertEqual(admissions, [(0,0.), (1,0.), (2,0.), (3,20.), (4,20.), (5,40.)])
        self.assertEqual(list(source.starts), list(range(6)))
        self.assertEqual([source.starts[i] for i in range(6)], [0.,0.,20.,20.,40.,40.])
        self.assertEqual([t for _,t in ready], [20.,20.,40.,40.,60.,60.])
        self.assertEqual(source.admissions[4], 20.)
        self.assertEqual(source.starts[4], 40.)
        self.assertFalse(source.active or source.waiting or source.pending)

    def test_zero_work_and_completion_time_issue_do_not_deadlock(self):
        source, admissions, ready, issue, drain = self.direct_source(0.)
        for demand_id in range(6):
            issue(demand_id)
        drain()
        self.assertEqual(admissions, [(i,0.) for i in range(6)])
        self.assertEqual(ready, [(i,0.) for i in range(6)])
        self.assertFalse(source.active or source.waiting or source.pending)
        source, admissions, ready, issue, drain = self.direct_source()
        for demand_id in range(3):
            issue(demand_id)
        # Issue before the already scheduled completion at the same timestamp.
        issue(3, 20.)
        drain()
        self.assertEqual([demand_id for demand_id,_ in admissions], [0,1,2,3])
        self.assertEqual(source.admissions[3], 20.)
        self.assertEqual([t for _,t in ready], [20.,20.,40.,40.])
        self.assertFalse(source.active or source.waiting or source.pending)

    def test_engine_uses_bounded_enqueue_return_before_source_start(self):
        source = BoundedExcessCoverageSource(keys(), dict(A=0,B=1<<25),
            {'lambda':10.,'beta32':0.,'beta128':0.}, completion_tail_ns=0.)
        result = shared_stage_events(2,2,1,service_ns=0.,read_latency_ns=0.,
            compute_ns=1.,output_counts=[1],input_source_async=source,
            input_source_bind_events=source.bind_scheduler,trace_requests=True)
        self.assertEqual(source.admissions[3], 20.)
        self.assertEqual(source.starts[3], 40.)
        self.assertEqual(result['input_demands'][3]['issue_return_ns'], 20.)
        self.assertEqual(result['input_delivery_count'], 6)
        self.assertFalse(source.active or source.waiting or source.pending)


if __name__=='__main__':
    unittest.main()
