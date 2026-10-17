"""多 CTA tensor 探针的手算、槽复用、共享与区分能力检查。"""
import unittest

from tma_tensor_multicta import CANDIDATES, cases, program, simulate, partition_weights
from supply_model import shared_stage_events


class TensorProbeTests(unittest.TestCase):
    def costs(self, **changes):
        result = dict(service=10., latency=20., issue=2., destination=0., consumer=3., resident_service=1.,resident_latency=20.)
        result.update(changes)
        return result

    def test_first_group_known_answer(self):
        p = program(1, 1, 'independent', groups=1)
        r = simulate(p, 'shared_fifo', trace=True, **self.costs())
        self.assertEqual([x['service_end'] for x in r['request_events']], [10., 15., 20.])
        self.assertEqual([x['ready'] for x in r['request_events']], [30., 35., 40.])
        self.assertEqual(r['time'], 43.)
        self.assertEqual(r['input_bytes'], 32768)

    def test_measured_program_costs_do_not_replace_service_wait(self):
        control = dict(prefix=5., prefill_issue=6., refill_issue=9., first_wait=3.,
                       issue_to_wait=6., wait_floor=4., release_to_issue=7.)
        r = simulate(program(1,1,'independent',groups=2), 'shared_fifo',
                     trace=True, control=control, **self.costs(consumer=2.))
        self.assertEqual([g['issue'] for g in r['group_events']], [5.,54.])
        self.assertEqual([g['wait_begin'] for g in r['group_events']], [14.,69.])
        self.assertEqual([g['ready'] for g in r['group_events']], [45.,94.])
        self.assertEqual([g['release'] for g in r['group_events']], [47.,96.])
        self.assertEqual(r['time'],96.)

    def test_wait_floor_and_drain_apply_after_prefill(self):
        control = dict(prefill_issue=6., prefill_gap=3., first_wait=5., wait_floor=4., drain_gap=7.)
        r = simulate(program(1,2,'independent',groups=2),'local_fifo',trace=True,control=control,
                     **self.costs(service=.1,latency=0.,consumer=2.))
        self.assertEqual([g['issue'] for g in r['group_events']],[0.,9.])
        self.assertEqual([g['wait_begin'] for g in r['group_events']],[20.,33.])
        self.assertEqual([g['release'] for g in r['group_events']],[26.,39.])

    def test_two_cta_source_sharing_keeps_both_deliveries(self):
        p = program(2, 1, 'shared', groups=1)
        local = simulate(p, 'local_fifo', **self.costs())
        shared = simulate(p, 'shared_fifo', **self.costs())
        merged = simulate(p, 'shared_source_fifo', **self.costs())
        self.assertEqual(local['time'], 43.)
        self.assertEqual(shared['time'], 63.)
        self.assertEqual(merged['time'], 43.)
        self.assertEqual(merged['source_generations'], 3)
        self.assertEqual(merged['merges'], 3)
        self.assertEqual(merged['deliveries'], 6)

    def test_every_group_waits_three_atoms_and_reuses_released_slot(self):
        for candidate in CANDIDATES:
            for active in (1, 4):
                r = simulate(program(3, active, 'shared', groups=11), candidate,
                             trace=True, **self.costs(destination=7.))
                groups = {(g['cta'], g['group']): g for g in r['group_events']}
                self.assertEqual(len(groups), 33)
                self.assertEqual(r['input_requests'], 99)
                self.assertEqual(r['input_bytes'], 33*32768)
                for key, g in groups.items():
                    reqs = [q for q in r['request_events'] if (q['cta'], q['group']) == key]
                    self.assertEqual([q['atom'] for q in reqs], ['A', 'B0', 'B1'])
                    self.assertEqual(g['ready'], max(q['ready'] for q in reqs))
                    self.assertGreaterEqual(g['release'], g['ready'])
                    self.assertEqual(g['generation'], g['group']//4)
                    if g['group'] >= active:
                        self.assertGreaterEqual(g['issue'], groups[g['cta'], g['group']-active]['release'])
                    if g['group'] >= 4:
                        previous = groups[g['cta'], g['group']-4]
                        self.assertEqual(g['slot'], previous['slot'])
                        self.assertGreaterEqual(g['issue'], previous['release'])

    def test_source_keys_expire_at_source_ready(self):
        r = simulate(program(1, 4, 'shared', groups=12, source_ring=1), 'shared_source_fifo',
                     trace=True, **self.costs(latency=100., consumer=25.))
        owners = {}
        for q in r['request_events']:
            if q['merged']:
                self.assertLess(q['issue'], owners[q['source_id']]['source_ready'])
            else:
                owners[q['source_id']] = q
        self.assertGreater(r['source_generations'], 3)
        self.assertGreater(r['merges'], 0)

    def test_candidate_equivalences_and_source_protocol(self):
        for active in (1, 4):
            one = simulate(program(1, active, 'independent'), 'local_fifo', **self.costs())
            many = simulate(program(16, active, 'independent'), 'local_fifo', **self.costs())
            self.assertEqual(one['time'], many['time'])
            self.assertEqual(many['cta_span'], 0.)
            p = program(16, active, 'independent')
            a = simulate(p, 'shared_fifo', **self.costs())
            b = simulate(p, 'shared_source_fifo', **self.costs())
            self.assertEqual(a['time'], b['time'])
            self.assertEqual(b['merges'], 0)

    def test_resident_service_and_phase_have_different_predictions(self):
        base = program(8,1,'shared',groups=32)
        shifted = program(8,1,'shared',groups=32,source_phase_stride=1)
        self.assertEqual(base['source_union_bytes'], shifted['source_union_bytes'])
        a = simulate(base,'resident_source_fifo',**self.costs())
        b = simulate(shifted,'resident_source_fifo',**self.costs())
        self.assertEqual(a['time'],b['time'])
        a = simulate(base,'shared_source_fifo',**self.costs())
        b = simulate(shifted,'shared_source_fifo',**self.costs())
        self.assertGreater(b['time'],a['time'])

    def test_resident_and_streaming_latency_are_separate(self):
        p=program(2,1,'independent',groups=1)
        resident=simulate(p,'two_tier_fifo',cache_capacity_bytes=65536,**self.costs(resident_latency=5.))
        streaming=simulate(p,'two_tier_fifo',cache_capacity_bytes=32768,**self.costs(resident_latency=5.))
        self.assertEqual(resident['source_price'],1.)
        self.assertEqual(resident['source_latency'],5.)
        self.assertEqual(streaming['source_price'],10.)
        self.assertEqual(streaming['source_latency'],20.)
        self.assertGreater(streaming['time'],resident['time'])

    def test_partition_demand_and_one_partition_equivalence(self):
        p=program(3,2,'independent',groups=8)
        for atom,weight in [('A',1.),('B0',.5),('B1',.5)]:
            self.assertEqual(sum(partition_weights(p,1,3,atom,16)),weight)
        aligned=program(16,1,'independent',groups=8)
        self.assertEqual(partition_weights(aligned,1,3,'A',16)[3],1.)
        self.assertEqual(partition_weights(aligned,1,3,'B0',16)[2],.5)
        self.assertEqual(partition_weights(aligned,1,3,'B1',16)[3],.5)
        p['source_strides_elements']['B']+=8
        self.assertEqual(sum(partition_weights(p,1,3,'B0',16)),.5)
        a=simulate(p,'two_tier_shared_fifo',**self.costs())
        b=simulate(p,'banked_resident_fifo',source_partitions=1,**self.costs())
        self.assertEqual(a['cta_times'],b['cta_times'])

    def test_twelve_conditions_hold_work_and_physical_slots_fixed(self):
        programs = cases()
        self.assertEqual(len(programs), 12)
        for p in programs:
            self.assertEqual(p['input_requests']//p['ctas'], 64*3)
            self.assertEqual(p['physical_payload_bytes'], 4*32768)
            self.assertEqual(p['source_ring'], 32)
            self.assertEqual(p['source_union_bytes'], 32*32768*
                             (p['ctas'] if p['source_protocol'] == 'independent' else 1))

    def test_large_latency_can_hide_the_shared_service_contrast(self):
        p = program(16, 1, 'independent', groups=8)
        def relative_gap(latency):
            a = simulate(p, 'local_fifo', **self.costs(latency=latency))['time']
            b = simulate(p, 'shared_fifo', **self.costs(latency=latency))['time']
            return (b-a)/b
        self.assertGreater(relative_gap(0.), .5)
        self.assertLess(relative_gap(1e8), 1e-5)

    def test_prefill_services_match_existing_gemm_backend(self):
        # 四组全部预填，不触及两种程序不同的消费者/复用顺序；只核对共享服务。
        p = program(3, 4, 'shared', groups=4)
        keys = [[[(0, g, atom) for atom in ('A', 'B0', 'B1')] for g in range(4)] for _ in range(3)]
        costs = self.costs(destination=7.)
        for candidate, sharing in [('shared_fifo', False), ('shared_source_fifo', True)]:
            probe = simulate(p, candidate, trace=True, **costs)
            gemm = shared_stage_events(4, 4, 3, service_ns=costs['service'],
                read_latency_ns=costs['latency'], compute_ns=.01,
                producer_gap_ns=costs['issue'], operand_gap_ns=costs['issue'],
                destination_service_ns=costs['destination'], output_counts=[1]*3,
                source_keys=keys, share_inflight_sources=sharing, trace_requests=True)
            names = {'issue_a': 'A', 'issue_b0': 'B0', 'issue_b1': 'B1'}
            expected = {(r['cta'], r['index'], names[r['kind']]): r for r in gemm['deliveries']}
            for r in probe['request_events']:
                reference = expected[r['cta'], r['group'], r['atom']]
                self.assertEqual(r['issue'], reference['arrival_ns'])
                self.assertEqual(r['source_ready'], reference['source_ready_ns'])
                self.assertEqual(r['ready'], reference['ready_ns'])


class GroupedServiceTests(unittest.TestCase):
    def test_capacity_and_groups_have_separate_frontiers(self):
        from tma_gpc_service import GroupedTensorSource
        service = GroupedTensorSource([0, 0, 1], 2, 10., 100.)
        actual = [service.submit(0., cta, index, 1.)['source_ready']
                  for index, cta in enumerate((0, 1, 0, 2))]
        self.assertEqual(actual, [110., 110., 120., 110.])

    def test_one_group_one_slot_reproduces_fifo_and_start_delay(self):
        from tma_gpc_service import GroupedTensorSource
        p = program(2, 1, 'independent', groups=4)
        costs = dict(service=10., latency=100., issue=2., destination=3., consumer=5.)
        original = simulate(p, 'shared_fifo', trace=True, **costs)
        grouped = simulate(p, 'shared_fifo', trace=True,
            source_service=GroupedTensorSource([0, 0], 1, 10., 100.), **costs)
        self.assertEqual(grouped['request_events'], original['request_events'])
        self.assertEqual(grouped['group_events'], original['group_events'])
        p['initial_offsets_ns'] = [1000., 1000.]
        delayed = simulate(p, 'shared_fifo', trace=True,
            source_service=GroupedTensorSource([0, 0], 1, 10., 100.), **costs)
        self.assertEqual(delayed['cta_times'], [v+1000. for v in original['cta_times']])


if __name__ == '__main__':
    unittest.main()
