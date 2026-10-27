"""应答共享速率、缓存共享与独立投递的CPU反例。"""
import heapq
import itertools
import unittest

import cfg_b_program
import cfg_b_request_model
from cfg_b_input_network import (CfgBInputNetwork, AddressRoutedInputNetwork,
    ReadSharedInputNetwork, JointLocalInputNetwork, PortOrderedInputNetwork,
    FormationFirstInputNetwork, LocalServiceCalendar, ReplyGroup, address_read_bank)


class Events:
    def __init__(self):
        self.events, self.order = [], itertools.count()

    def schedule(self, t, callback):
        heapq.heappush(self.events, (t, next(self.order), callback))

    def run(self):
        while self.events:
            t, _, callback = heapq.heappop(self.events)
            callback(t)


def request(index, work):
    return dict(id=index, remaining_ns=work)


def program_and_source(source_type=CfgBInputNetwork, network=None, lda=256):
    program = cfg_b_program.build_program(
        dict(config='cfg_b', m=128, n=256, k=256, stages=4, lda=lda), dict(grid=[1, 2, 1], stages=4))
    source = source_type(program, [0, 0], [0., 20., 4.],
        output_base=32*1024*1024, addresses=dict(A=0, B=16*1024*1024),
        network=network if network is not None else dict(reply_cycles_per_16KiB=12., split_cycles_per_128_extra_lines=30.),
        phase_frequency_ghz=[1.], cache_bytes=2*1024*1024, hit_ns=2.,
        home_ns=10., link_ns=3., hbm_service_ns=128., record_diagnostics=False)
    source.begin(0, 100000.)
    return program, source


class InputNetworkTests(unittest.TestCase):
    def test_formation_controls_cache_query_time_and_keeps_receiver_work(self):
        _, source = program_and_source(FormationFirstInputNetwork,
            dict(formation_cycles_per_tag=1., bank_cycles_per_fragment=2., max_read_subscribers=4))
        events, queries, completed = Events(), [], []
        source.bind_scheduler(events.schedule)
        source.read = lambda t, d, line: queries.append(t-source.base) or t+2.
        for i in range(2):
            source.issue(0., i, 0, 0, 1., i, lambda i, t, served: completed.append((i, t)))
        self.assertFalse(queries)
        events.run()
        self.assertEqual(set(queries), {256.})
        self.assertEqual(source.processing_work_ns, 256.)
        self.assertEqual(source.read_fragment_count, 128)
        self.assertEqual(source.read_merged_fragments, 128)
        self.assertEqual(source.returned_count, 2)
        self.assertTrue(all(t >= 258. for i, t in completed))

    def test_formation_request_lifecycle_completes_without_post_return_service(self):
        program, source = program_and_source(FormationFirstInputNetwork,
            dict(formation_cycles_per_tag=.125, bank_cycles_per_fragment=2., max_read_subscribers=4))
        report = cfg_b_request_model.simulate(program,
            dict(service_ns=0., read_latency_ns=0., compute_ns=512., commit_lead_ns=256.,
                 epilogue_prepare_ns=1., epilogue_write_ns=1.),
            **source.callbacks(), detailed_ctas=(0,), critical_path=True)
        self.assertEqual(source.delivered_count, 24)
        self.assertFalse(source.pending_destinations)
        self.assertGreater(report['dependency_edges_checked'], 0)
        self.assertAlmostEqual(source.processing_work_ns,
            source.network_state()['processing_busy_ns'])

    def test_port_lookup_changes_cache_only_when_service_executes(self):
        _, source = program_and_source(PortOrderedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3., max_read_subscribers=4))
        events, lookups = Events(), []
        source.bind_scheduler(events.schedule)
        source.read_frontiers[0][address_read_bank(0)] = 1000.
        source.read = lambda t, d, line: lookups.append((t-source.base, line)) or t+2.
        source.issue(0., 0, 0, 0, 1., 0, lambda *_: None)
        self.assertFalse(lookups)
        self.assertEqual(source.lookup_count, 0)
        events.run()
        self.assertEqual(len(lookups), 128)
        self.assertEqual(sorted(lookups), lookups)
        self.assertNotEqual(lookups[0][1], 0)
        self.assertEqual(next(t for t, line in lookups if line == 0), 1001.)

    def test_port_lookup_finite_subscribers_receive_all_independent_replies(self):
        _, source = program_and_source(PortOrderedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3., max_read_subscribers=4))
        events, completed = Events(), []
        source.bind_scheduler(events.schedule)
        for index in range(5):
            source.issue(0., index % 2, 0, 0, 1., index,
                lambda demand_id, t, served: completed.append((demand_id, t)))
        events.run()
        self.assertEqual(source.read_fragment_count, 256)
        self.assertEqual(source.lookup_count, 256)
        self.assertEqual(source.read_merged_fragments, 384)
        self.assertEqual(source.returned_count, 5)
        self.assertCountEqual([i for i, t in completed], range(5))
        self.assertGreater(source.hbm_lines, 0)

    def test_port_lookup_late_subscriber_reuses_data_return_without_early_cache_access(self):
        _, source = program_and_source(PortOrderedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3., max_read_subscribers=4))
        events, lookups, completed = Events(), [], []
        source.bind_scheduler(events.schedule)
        source.read = lambda t, d, line: lookups.append(t-source.base) or t+100.
        source.issue(0., 0, 0, 0, 1., 0, lambda i, t, s: completed.append((i, t)))
        events.schedule(20., lambda now: source.issue(now, 1, 0, 0, 1., 1,
            lambda i, t, s: completed.append((i, t))))
        events.run()
        self.assertEqual(source.lookup_count, 128)
        self.assertEqual(source.read_merged_fragments, 128)
        self.assertEqual(len(completed), 2)
        self.assertTrue(all(t < 20. for t in lookups))

    def test_port_lookup_completes_program_and_preserves_request_lifetimes(self):
        program, source = program_and_source(PortOrderedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3., max_read_subscribers=4))
        report = cfg_b_request_model.simulate(program,
            dict(service_ns=0., read_latency_ns=0., compute_ns=512., commit_lead_ns=256.,
                 epilogue_prepare_ns=1., epilogue_write_ns=1.),
            **source.callbacks(), detailed_ctas=(0,), critical_path=True)
        self.assertEqual(source.delivered_count, 24)
        self.assertFalse(source.pending_destinations)
        self.assertEqual(source.lookup_count, source.read_fragment_count)
        self.assertGreater(report['dependency_edges_checked'], 0)

    def test_joint_calendar_uses_future_gaps_without_overbooking(self):
        calendar = LocalServiceCalendar()
        self.assertEqual(calendar.reserve(0., 10., 10.), (10., 20.))
        self.assertEqual(calendar.reserve(1., 1., 12.), (1., 23.))
        self.assertEqual(calendar.intervals, [(1., 23.)])
        self.assertEqual(calendar.work_ns, 22.)

    def test_joint_compute_alone_preserves_service_and_commit(self):
        _, source = program_and_source(JointLocalInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3., local_bytes_per_cycle=128.))
        step = source.compute_service(0., 0, 0, 0., None,
            compute_duration=512., commit_lead=256., completion_lag=0.)
        self.assertEqual(step, (512., 256., 512., 256.))
        self.assertEqual(source.local_compute_delay_ns, 0.)
        self.assertEqual(source.local_compute_work_ns, 256.)

    def test_gather_joint_prices_each_destination_source_fragment(self):
        profile = dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3.,
                       local_bytes_per_cycle=128., local_input_work='source_fragments128')
        program, source = program_and_source(JointLocalInputNetwork, profile)
        template = source.atom_templates[0][0][0]
        template['stride_bytes'] = [8208, 2]
        delivered = source.destination_service(0., 0, 0, 0, 1.)
        self.assertEqual(delivered['busy_ns'], 240.)
        _, aligned = program_and_source(JointLocalInputNetwork, profile)
        aligned.atom_templates[0][0][0]['stride_bytes'] = [8192, 2]
        self.assertEqual(aligned.destination_service(0., 0, 0, 0, 1.)['busy_ns'], 128.)
        self.assertEqual(aligned.destination_service(0., 0, 0, 1, .5)['busy_ns'], 64.)

    def test_joint_compute_reservation_delays_input(self):
        _, source = program_and_source(JointLocalInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3., local_bytes_per_cycle=128.))
        source.compute_service(0., 0, 0, 0., None,
            compute_duration=512., commit_lead=256., completion_lag=0.)
        delivered = source.destination_service(0., 0, 1, 0, 1.)
        self.assertEqual((delivered['start_ns'], delivered['ready_ns']), (64., 256.))
        self.assertEqual(delivered['busy_ns'], 128.)
        self.assertEqual(source.local_input_delay_ns, 128.)
        other = source.destination_service(0., 1, 1, 0, 1.)
        self.assertEqual(other['ready_ns'], 128.)

    def test_joint_input_reservation_delays_compute_and_tail(self):
        _, source = program_and_source(JointLocalInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3., local_bytes_per_cycle=128.))
        source.bind_scheduler(Events().schedule)
        source.destination_service(0., 0, 0, 0, 1.)
        E, C, D, W = source.compute_service(0., 0, 1, 0., 1.,
            compute_duration=512., commit_lead=256., completion_lag=20.)
        self.assertEqual((E, C, D, W), (640., 384., 660., 384.))
        self.assertEqual(source.local_compute_delay_ns, 128.)
        state = source.network_state()['local_joint_service']
        self.assertEqual(state['calendar_work_ns'], state['compute_work_ns']+state['input_work_ns'])

    def test_joint_program_reconstructs_delayed_compute_graph(self):
        program, source = program_and_source(JointLocalInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3., local_bytes_per_cycle=128.))
        report = cfg_b_request_model.simulate(program,
            dict(service_ns=0., read_latency_ns=0., compute_ns=512., commit_lead_ns=256.,
                 destination_service_ns=999., epilogue_prepare_ns=1., epilogue_write_ns=1.),
            **source.callbacks(), detailed_ctas=(0,), critical_path=True)
        self.assertGreater(report['dependency_edges_checked'], 0)
        self.assertGreater(report['accounting']['compute_extra_ns'], 0.)
        state = source.network_state()['local_joint_service']
        self.assertEqual(state['compute_work_ns'], 2*4*256.)
        self.assertEqual(state['input_work_ns'], 2*4*256.)
        self.assertEqual(report['accounting']['destination_busy_ns'], state['input_work_ns'])

    def test_public_fold_restores_raw_bank_selector(self):
        # addrdec.cc fold_to_chiplet按channel折叠，再由原bk字段重建sub_partition。
        vectors = {0: 0, 1: 1, 95: 47, 96: 14, 97: 17, 98: 16, 127: 47, 255: 45, 1024: 6}
        for line, bank in vectors.items():
            self.assertEqual(address_read_bank(line), bank)
        for line in range(4096):
            self.assertEqual(address_read_bank(line) & 1, line & 1)
            self.assertLess(address_read_bank(line), 48)

    def test_two_replies_share_rate_and_release_in_completion_order(self):
        events, completed = Events(), []
        group = ReplyGroup(events.schedule, lambda t, r: completed.append((r['id'], t)))
        group.enqueue(0., request(0, 20.))
        group.enqueue(0., request(1, 10.))
        events.run()
        self.assertEqual(completed, [(1, 20.), (0, 30.)])
        self.assertEqual(group.busy_ns, 30.)
        self.assertFalse(group.active or group.waiting)

    def test_later_arrival_changes_running_completion(self):
        events, completed = Events(), []
        group = ReplyGroup(events.schedule, lambda t, r: completed.append((r['id'], t)))
        group.enqueue(0., request(0, 20.))
        events.schedule(5., lambda t: group.enqueue(t, request(1, 15.)))
        events.run()
        self.assertEqual(sorted(completed), [(0, 35.), (1, 35.)])
        self.assertEqual(group.busy_ns, 35.)

    def test_cache_fill_shared_but_each_receiver_processes_and_delivers(self):
        _, source = program_and_source()
        events, returned, before_delivery = Events(), [], []
        source.bind_scheduler(events.schedule)

        def ready(demand_id, t, served):
            returned.append((demand_id, t, served))
            before_delivery.append(len(source.pending_destinations))
            cta = demand_id
            delivery = source.destination_service(t, cta, 0, 0, 1.)
            events.schedule(delivery['ready_ns'], lambda now: source.destination_done(now, demand_id))

        source.issue(0., 0, 0, 0, 1., 0, ready)
        fetched = source.hbm_lines
        source.issue(0., 1, 0, 0, 1., 1, ready)
        self.assertEqual(source.hbm_lines, fetched)
        self.assertEqual(fetched, 128)
        self.assertEqual(len(source.pending_destinations), 2)
        events.run()
        self.assertEqual(source.returned_count, 2)
        self.assertEqual(source.delivered_count, 2)
        self.assertEqual(before_delivery, [2, 2])
        self.assertFalse(source.pending_destinations)
        self.assertEqual(source.processing_work_ns, 24.)
        self.assertAlmostEqual(source.reply_groups[0].busy_ns, 24.)
        self.assertEqual({r[2]['merged'] for r in returned}, {False})

    def test_real_program_binding_prices_delivery_once(self):
        program, source = program_and_source()
        report = cfg_b_request_model.simulate(program,
            dict(service_ns=0., read_latency_ns=0., compute_ns=64., commit_lead_ns=32.,
                 destination_service_ns=999., epilogue_prepare_ns=1., epilogue_write_ns=1.),
            **source.callbacks(), detailed_ctas=(0,))
        self.assertGreater(report['dependency_edges_checked'], 0)
        self.assertEqual(report['accounting']['input_delivery_count'], 24)
        self.assertEqual(source.delivered_count, 24)
        self.assertFalse(source.pending_destinations)
        for request in report['details'][0]['requests']:
            event = request['service']
            self.assertLess(event['ready_ns']-event['start_ns'], 5.)

    def test_system_default_matches_frozen_and_network_is_opt_in(self):
        import importlib.util
        from pathlib import Path
        import cfg_b_system as system
        repo = Path(__file__).resolve().parents[3]
        old_path = repo/'results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-program-intervention-001/source/cfg_b_system.py'
        spec = importlib.util.spec_from_file_location('prior_system', old_path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        profile = system.load_profile(repo)
        item = dict(case=dict(config='cfg_b', m=128, n=256, k=256, stages=4),
            setup=dict(grid=[1, 2, 1], stages=4), cta_groups=[0, 0],
            addresses=dict(A=0, B=2*1024*1024, D=4*1024*1024),
            allocation_bytes=dict(A=128*256*2, B=256*256*2, D=128*256*4))
        protocol = dict(calls=2, inter_call_gap_ns=100000)
        before = old.predict(item, profile, protocol, detailed_ctas=(0,), report_demands=False)
        after = system.predict(item, profile, protocol, detailed_ctas=(0,), report_demands=False)
        self.assertEqual(before, after)
        enabled = system.predict(item, dict(profile, input_network=dict(
            reply_cycles_per_16KiB=13., split_cycles_per_128_extra_lines=20.)),
            protocol, detailed_ctas=(0,), report_demands=False)
        state = enabled['source_state']['request_reply_network']
        self.assertEqual(state['returned'], 24)
        self.assertEqual(state['delivered'], 24)
        self.assertEqual(state['pending_destinations'], 0)

    def test_read_banks_independent_and_shared_bank_fifo(self):
        _, source = program_and_source(AddressRoutedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=0.))
        events, completed = Events(), []
        source.bind_scheduler(events.schedule)
        for i, banks in enumerate((((0, 3), (1, 7)), ((0, 2), (2, 4)))):
            row = dict(id=i, cta=0, domain=0, read_banks=banks, read_unit_ns=1.,
                processing_work_ns=0., remaining_ns=0., identity=i,
                notify_ready=lambda demand, t, served: completed.append((demand, t)))
            source.read_return(0., row)
        events.run()
        self.assertEqual(source.read_frontiers[0][:3], [5., 7., 4.])
        self.assertEqual(source.read_work_ns, 16.)
        self.assertEqual(source.read_fragment_count, 16)
        self.assertEqual(completed, [(1, 23.), (0, 25.)])

    def test_same_bank_two_domains_do_not_share_frontier(self):
        _, source = program_and_source(AddressRoutedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=0.))
        events, completed = Events(), []
        source.bind_scheduler(events.schedule)
        for i in range(2):
            source.read_return(0., dict(id=i, cta=0, domain=i, read_banks=((0, 7),),
                read_unit_ns=1., processing_work_ns=0., remaining_ns=0., identity=i,
                notify_ready=lambda demand, t, served: completed.append(t)))
        events.run()
        self.assertEqual(completed, [25., 25.])
        self.assertEqual([b[0] for b in source.read_frontiers], [7., 7.])

    def test_actual_stride_routes_all_row_fragments(self):
        network = dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=1.)
        geometries = []
        for lda in (256, 264):
            _, source = program_and_source(AddressRoutedInputNetwork, network, lda=lda)
            _, geometry, _ = source.request_geometry(0, 0, 0)
            self.assertEqual(sum(count for _, count in geometry['read_banks']), len(geometry['lines']))
            geometries.append(geometry)
        self.assertEqual([len(g['lines']) for g in geometries], [128, 240])
        self.assertNotEqual(geometries[0]['read_banks'], geometries[1]['read_banks'])

    def test_routed_program_delivers_every_recipient(self):
        program, source = program_and_source(AddressRoutedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3.))
        report = cfg_b_request_model.simulate(program,
            dict(service_ns=0., read_latency_ns=0., compute_ns=64., commit_lead_ns=32.,
                 destination_service_ns=999., epilogue_prepare_ns=1., epilogue_write_ns=1.),
            **source.callbacks(), detailed_ctas=(0,))
        self.assertEqual(report['accounting']['input_delivery_count'], 24)
        self.assertEqual(source.delivered_count, 24)
        self.assertFalse(source.pending_destinations)
        self.assertGreater(source.read_work_ns, 0.)

    def test_shared_read_keeps_independent_replies_and_rereads_after_completion(self):
        _, source = program_and_source(ReadSharedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3.))
        events, completed = Events(), []
        source.bind_scheduler(events.schedule)

        def ready(index, t, served):
            completed.append(index)
            row = source.destination_service(t, index % 2, 0, 0, 1.)
            events.schedule(row['ready_ns'], lambda now: source.destination_done(now, index))

        source.issue(0., 0, 0, 0, 1., 0, ready)
        source.issue(0., 1, 0, 0, 1., 1, ready)
        events.run()
        self.assertEqual(source.read_fragment_count, 128)
        self.assertEqual(source.read_merged_fragments, 128)
        self.assertEqual(source.returned_count, 2)
        self.assertEqual(source.delivered_count, 2)
        source.issue(10000., 0, 0, 0, 1., 2, ready)
        events.run()
        self.assertEqual(source.read_fragment_count, 256)
        self.assertEqual(source.read_merged_fragments, 128)
        self.assertCountEqual(completed, [0, 1, 2])
        self.assertFalse(source.pending_destinations)

    def test_tensor_probe_uses_same_async_read_reply_and_delivery(self):
        import tma_tensor_multicta as tensor
        _, source = program_and_source(ReadSharedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3.))
        probe = tensor.program(2, 4, 'independent', groups=4, source_ring=4)
        result = tensor.simulate(probe, 'shared_source_fifo', source_async=source,
            service=1., latency=0., issue=1., destination=999., consumer=2., trace=True)
        self.assertEqual(result['candidate'], source.version)
        self.assertEqual(result['input_requests'], 24)
        self.assertEqual(source.delivered_count, 24)
        self.assertFalse(source.pending_destinations)
        for row in result['request_events']:
            self.assertEqual(row['ready']-row['destination_start'], 4.*row['bytes']/16384)

    def test_fifth_subscriber_requires_second_read_and_all_receive_replies(self):
        _, source = program_and_source(ReadSharedInputNetwork,
            dict(bank_cycles_per_fragment=1., reply_cycles_per_16KiB=3., max_read_subscribers=4))
        events, completed = Events(), []
        source.bind_scheduler(events.schedule)

        def ready(index, t, served):
            completed.append(index)
            row = source.destination_service(t, index % 2, 0, 0, 1.)
            events.schedule(row['ready_ns'], lambda now: source.destination_done(now, index))

        for i in range(5):
            source.issue(0., i % 2, 0, 0, 1., i, ready)
        events.run()
        self.assertEqual(source.read_fragment_count, 256)
        self.assertEqual(source.read_merged_fragments, 384)
        self.assertEqual(source.returned_count, 5)
        self.assertEqual(source.delivered_count, 5)
        self.assertEqual(source.processing_work_ns, 15.)
        self.assertCountEqual(completed, range(5))
        self.assertFalse(source.pending_destinations)


if __name__ == '__main__':
    unittest.main()
