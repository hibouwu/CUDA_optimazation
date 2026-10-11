"""CPU checks of conditional arithmetic, binding rejection and unknown exits."""
import copy
import gzip
import json
from pathlib import Path
import unittest

from lifecycle_brackets import (MARKER, canonical_sha256, last_candidate_certificate,
                                maximum_interval, project_first_main)
from model_snapshot.output_model import (cfg_a_rank_dependencies, cfg_c_rank_dependencies,
                                        dependency_order)


HERE = Path(__file__).parent


def cfg_c_graph(output_tiles):
    path = HERE/'model_snapshot'/f'cfg_c-S4-K64-T{output_tiles}-source-ports.json.gz'
    return cfg_c_rank_dependencies(json.loads(gzip.decompress(path.read_bytes())))


def synthetic_packet(graph, family, shift=0):
    # Explicit topological witness only. These 10ns lags are invented test data,
    # not a service model, calibration, fit or output prediction.
    order, _ = dependency_order(graph)
    witness = {}
    for node in order:
        witness[node] = max([witness[p] for p in graph['nodes'][node]] or [0]) + 10
    bindings = {}
    records = []
    contract = dict(schema='C-gt-lifecycle-contract/1', graph_family=family,
        graph_sha256=canonical_sha256(graph), invocation='synthetic-one-invocation',
        device_uuid='synthetic-GPU', binary_sha256='1'*64, ordering_review_sha256='2'*64,
        actual_capture_order_assumed=True, monotone_gt_reference_assumed=True,
        rank_ctas=[4,70], sm_count=132, anchor_bindings=bindings)
    for node in graph['nodes']:
        match = MARKER.fullmatch(node)
        if not match:
            continue
        rank, wg, ordinal, event = match.groups()
        rank, wg, ordinal = int(rank), int(wg), int(ordinal)
        binding = dict(cta=contract['rank_ctas'][rank], physical_thread=128*(wg+1), coord=[ordinal,rank])
        bindings[node] = binding
        stamp = 1791725328316000001 + shift + witness[node]
        records.append(dict(node=node, invocation=contract['invocation'], device_uuid=contract['device_uuid'],
            binary_sha256=contract['binary_sha256'], **copy.deepcopy(binding), sm=130-6*rank,
            gt_before=stamp-1, cycles=10**6+2*witness[node], gt_after=stamp+1, overflow=False))
    packet = dict(schema='C-gt-lifecycle-packet/1', contract_sha256=canonical_sha256(contract),
        invocation=contract['invocation'], device_uuid=contract['device_uuid'],
        binary_sha256=contract['binary_sha256'], data_origin='synthetic_test', records=records)
    return contract, packet, witness


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.graph = cfg_a_rank_dependencies(4,3,2)
        self.contract, self.packet, self.witness = synthetic_packet(self.graph, 'cfg_a')

    def project(self):
        return project_first_main(self.graph, self.contract, self.packet)

    def test_both_real_source_families_and_output_counts_keep_unknown_completions(self):
        for family in ('cfg_a','cfg_c'):
            for tiles in (1,2):
                graph = cfg_a_rank_dependencies(4,64,tiles) if family == 'cfg_a' else cfg_c_graph(tiles)
                contract, packet, witness = synthetic_packet(graph, family)
                result = project_first_main(graph, contract, packet)
                self.assertEqual(result['anchor_count'], 8*tiles)
                self.assertEqual(result['max_bracket_width_gt_counter_ns'], 2)
                complete = result['completion_intervals_gt_counter_ns']
                self.assertTrue(all(interval[1] is None for interval in complete['consumer_FINAL_by_CTA'].values()))
                self.assertIsNone(complete['cluster_source_safe_exit'][1])
                self.assertIsNone(complete['cluster_cuda_complete'][1])
                self.assertFalse(any(result['qualification'].values()))
                origin_delta = result['shared_origin_gt_counter_ns'] - 1791725328316000001
                for node, interval in result['bounds_gt_counter_ns'].items():
                    expected = witness[node]-origin_delta
                    if interval[0] is not None:
                        self.assertLessEqual(interval[0], expected)
                    if interval[1] is not None:
                        self.assertGreaterEqual(interval[1], expected)

    def test_gt_integer_translation_preserves_single_tick_precision(self):
        first = self.project()
        contract, packet, _ = synthetic_packet(self.graph, 'cfg_a', shift=257)
        second = project_first_main(self.graph, contract, packet)
        self.assertEqual(second['shared_origin_gt_counter_ns']-first['shared_origin_gt_counter_ns'],257)
        for field in ('anchors_gt_counter_ns','bounds_gt_counter_ns','completion_intervals_gt_counter_ns'):
            self.assertEqual(first[field],second[field])
        self.assertTrue(all(type(x) is int for interval in first['anchors_gt_counter_ns'].values() for x in interval))
        self.assertTrue(all(interval[1]-interval[0] == 2 for interval in first['anchors_gt_counter_ns'].values()))

    def test_cross_sm_native_values_are_never_subtracted(self):
        expected = self.project()['bounds_gt_counter_ns']
        for record in self.packet['records']:
            if record['cta'] == 70:
                record['cycles'] -= 900000
        self.assertEqual(self.project()['bounds_gt_counter_ns'],expected)

    def test_explicit_same_call_origin_and_invalid_future_origin(self):
        local = self.project()
        shared = local['shared_origin_gt_counter_ns']-257
        result = project_first_main(self.graph,self.contract,self.packet,shared)
        self.assertEqual(result['shared_origin_gt_counter_ns'],shared)
        for node,interval in local['anchors_gt_counter_ns'].items():
            self.assertEqual(result['anchors_gt_counter_ns'][node],[x+257 for x in interval])
        with self.assertRaisesRegex(ValueError,'must not exceed'):
            project_first_main(self.graph,self.contract,self.packet,local['shared_origin_gt_counter_ns']+1)

    def test_missing_duplicate_and_legacy_records_rejected(self):
        original = copy.deepcopy(self.packet)
        variants = [original['records'][:-1], original['records'][:-1]+[original['records'][0]]]
        legacy = copy.deepcopy(original['records'])
        del legacy[0]['gt_before']
        variants.append(legacy)
        for records in variants:
            self.packet = dict(original, records=records)
            with self.assertRaises(ValueError):
                self.project()

    def test_node_thread_coord_and_call_id_binding_rejections(self):
        original = copy.deepcopy(self.packet)
        mutations = [('node','r0.w0.t0.DONE'),('physical_thread',168),('coord',[1,1]),
                     ('cta',True),('invocation','another-call'),('device_uuid','another-GPU'),
                     ('binary_sha256','3'*64)]
        for name,value in mutations:
            self.packet = copy.deepcopy(original)
            self.packet['records'][0][name] = value
            with self.subTest(name=name),self.assertRaises(ValueError):
                self.project()

    def test_zero_uint64_overflow_and_reversed_gt_rejections(self):
        original = copy.deepcopy(self.packet)
        for name,value in [('gt_before',0),('cycles',0),('gt_after',2**64),
                           ('cycles',True),('overflow',True),('sm',-1),('sm',132)]:
            self.packet = copy.deepcopy(original)
            self.packet['records'][0][name] = value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                self.project()
        self.packet = copy.deepcopy(original)
        self.packet['records'][0]['gt_before'] = self.packet['records'][0]['gt_after']+1
        with self.assertRaisesRegex(ValueError,'reversed'):
            self.project()

    def test_same_cta_sm_and_same_thread_cycle_regression_rejected(self):
        self.packet['records'][1]['sm'] += 1
        with self.assertRaisesRegex(ValueError,'SM identity'):
            self.project()
        self.contract,self.packet,_ = synthetic_packet(self.graph,'cfg_a')
        main = next(x for x in self.packet['records'] if x['node']=='r0.w0.t0.MAIN')
        main['cycles'] = 1
        with self.assertRaisesRegex(ValueError,'cycle order'):
            self.project()

    def test_source_infeasible_brackets_rejected(self):
        first = next(x for x in self.packet['records'] if x['node']=='r0.w0.t0.FIRST')
        main = next(x for x in self.packet['records'] if x['node']=='r0.w0.t0.MAIN')
        main['gt_before'],main['gt_after'] = first['gt_before']-20,first['gt_before']-10
        with self.assertRaisesRegex(ValueError,'infeasible'):
            self.project()

    def test_order_assumption_and_frozen_identity_required(self):
        original = copy.deepcopy(self.contract)
        for name,value in [('actual_capture_order_assumed',False),('monotone_gt_reference_assumed',False),
                           ('ordering_review_sha256',''),('graph_sha256','4'*64)]:
            self.contract = dict(original, **{name:value})
            self.packet['contract_sha256'] = canonical_sha256(self.contract)
            with self.subTest(name=name),self.assertRaises(ValueError):
                self.project()

    def test_caller_cannot_promote_qualification(self):
        self.contract['hardware_sampling_enclosure_qualified'] = True
        self.contract['observation_qualified'] = True
        self.packet['contract_sha256'] = canonical_sha256(self.contract)
        result = self.project()
        self.assertFalse(any(result['qualification'].values()))
        self.assertIn('caller_assumption',result['sampling_order_status'])


class LastCandidateTests(unittest.TestCase):
    def certify(self, bounds, candidates, scope, contexts=None):
        if contexts is None:
            context = dict(invocation='synthetic-one-invocation',device_uuid='synthetic-GPU',
                           binary_sha256='1'*64,shared_origin_gt_counter_ns=1791725328316000001)
            contexts = {cta:dict(context) for cta in bounds}
        return last_candidate_certificate(bounds,candidates,scope,contexts)

    def test_ties_remain_possible_and_do_not_certify_one_candidate(self):
        result = self.certify({0:[90,100],1:[100,110],2:[0,99]},[1],'consumer_FINAL')
        self.assertEqual(result['possible_by_marginal_bounds'],[0,1])
        self.assertFalse(result['candidate_coverage_certified'])
        self.assertIsNone(result['unique_last_certified'])

    def test_strict_separation_certifies_coverage_and_unique_last(self):
        candidates = [1]
        result = self.certify({0:[90,99],1:[100,110],2:[0,99]},candidates,'consumer_FINAL')
        self.assertTrue(result['candidate_coverage_certified'])
        self.assertEqual(result['unique_last_certified'],1)
        self.assertEqual(candidates,[1])
        self.assertEqual(result['frozen_candidates'],[1])
        self.assertFalse(any(result['qualification'].values()))

    def test_unknown_upper_prevents_candidate_coverage(self):
        result = self.certify({0:[90,None],1:[100,110]},[1],'source_safe_exit')
        self.assertEqual(result['possible_by_marginal_bounds'],[0,1])
        self.assertFalse(result['candidate_coverage_certified'])
        self.assertIsNone(result['max_endpoint_interval_gt_counter_ns'][1])
        self.assertIsNone(result['unique_last_certified'])
        result = self.certify({0:[None,None],1:[None,None]},[1],'cuda_complete')
        self.assertEqual(result['possible_by_marginal_bounds'],[0,1])

    def test_guaranteed_candidate_miss_is_only_a_lower_bound(self):
        result = self.certify({0:[0,2],1:[3,4]},[0],'consumer_FINAL')
        self.assertEqual(result['guaranteed_candidate_miss_gap_gt_counter_ns'],1)
        self.assertFalse(result['candidate_coverage_certified'])

    def test_maximum_retains_partial_knowledge(self):
        self.assertEqual(maximum_interval([[None,5],[2,3]]),[2,5])
        self.assertEqual(maximum_interval([[None,None],[2,3]]),[2,None])

    def test_invalid_scope_candidates_and_bounds_rejected(self):
        variants = [({},[0],'consumer_FINAL'),({0:[0,1]},[0,0],'consumer_FINAL'),
                    ({0:[0,1]},[1],'consumer_FINAL'),({0:[0,1]},[0],'whole_GEMM'),
                    ({0:[2,1]},[0],'consumer_FINAL'),({0:[0,float('inf')]},[0],'consumer_FINAL'),
                    ({0:[0,True]},[0],'consumer_FINAL')]
        for bounds,candidates,scope in variants:
            with self.subTest(bounds=bounds,candidates=candidates,scope=scope),self.assertRaises(ValueError):
                self.certify(bounds,candidates,scope)

    def test_cross_call_and_different_origin_contexts_rejected(self):
        bounds = {0:[0,2],1:[3,4]}
        context = dict(invocation='synthetic-one-invocation',device_uuid='synthetic-GPU',
                       binary_sha256='1'*64,shared_origin_gt_counter_ns=1791725328316000001)
        for name,value in [('invocation','second-call'),('device_uuid','second-GPU'),
                           ('binary_sha256','3'*64),('shared_origin_gt_counter_ns',1791725328316000002)]:
            contexts = {0:dict(context),1:dict(context,**{name:value})}
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'domains differ'):
                self.certify(bounds,[1],'consumer_FINAL',contexts)


if __name__ == '__main__':
    unittest.main()
