#!/usr/bin/env python3
"""Bind the archived R10 cfg_b D protocol. No durations or timing observations."""
import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import json
from pathlib import Path

KERNEL = 'cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp'
EPILOGUE = 'cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp'
PIPELINE = 'cutlass/include/cutlass/pipeline/sm90_pipeline.hpp'
TMA = 'cutlass/include/cute/arch/copy_sm90_tma.hpp'
BUILDER = 'cutlass/include/cutlass/epilogue/collective/builders/sm90_builder.inl'
MAINLOOP = 'cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp'
SOURCE_SITES = {
    'probes/r10.cu': ['using KernelTile = Shape<_128, _128, _64>', 'using EpilogueTile = Shape<_64, _32>',
                      'using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong', 'float, float, void'],
    KERNEL: ['NumMmaWarpGroups = 2', 'StagesPerMathWarpGroup = 2',
             'epi_store_pipeline_params.always_wait = true',
             'params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup',
             'epi_store_pipe_producer_state.advance(d_tile_count)',
             'math_wg_order_barrier.wait()', 'math_wg_order_barrier.arrive()',
             'collective_mainloop.mma_tail(', 'collective_epilogue.store(',
             'collective_epilogue.store_tail(', 'collective_mainloop.load_tail(',
             'scheduler.advance_to_next_work(NumMmaWarpGroups)'],
    BUILDER: ['constexpr bool ReuseSmem =', 'constexpr bool DelayTmaStore =',
              'constexpr int StagesD =', 'constexpr int FragmentSize ='],
    EPILOGUE: ['using StorePipeline =', 'get_store_pipe_increment(',
               'NamedBarrier::sync(size(TiledMma{})',
               'bool issue_tma_store = (thread_idx / NumThreadsPerWarp) == 0',
               'cutlass::arch::fence_view_async_shared(); // ensure smem writes',
               'copy(params.tma_store_d,', 'store_pipeline.producer_commit(',
               'store_pipeline.producer_acquire(', 'for (int epi_n = 0;', 'for (int epi_m = 0;',
               'tma_store_fn(epi_m_prev, epi_n_prev)', 'copy(tiled_r2s,',
               'constexpr bool issue_smem_store = true', 'store_pipeline.producer_tail('],
    PIPELINE: ['static constexpr uint32_t UnacquiredStages =', 'tma_store_wait<0>()',
               'if (params_.always_wait || count > UnacquiredStages)',
               'tma_store_wait<UnacquiredStages>()', 'tma_store_arrive()',
               'stage_({0, params.group_id == 0, 0})',
               'barrier_ptr_[d * Length + l].init(params.group_size)',
               'get_barrier_for_current_stage(params_.group_id).wait(stage_.phase())',
               'int signalling_id = (params_.group_id + 1) % Length',
               'get_barrier_for_current_stage(signalling_id).arrive()',
               'for (int count = 0; count < Stages; ++count)',
               'empty_barrier_ptr_[stage].wait(phase)'],
    TMA: ['struct SM90_TMA_STORE_2D', 'cp.async.bulk.tensor.2d.global.shared::cta.bulk_group',
          'cp.async.bulk.commit_group;', 'cp.async.bulk.wait_group.read %0;'],
    MAINLOOP: ['pipeline.producer_tail(smem_pipe_write)', 'warpgroup_wait<0>()',
               'pipeline.consumer_release(smem_pipe_release)'],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_prefix(cta, output):
    return f'cta{cta}.j{output}'


def bind_output(output, shape):
    """Reorder B's coordinate-sorted boxes to the source's n-outer/m-inner loop."""
    if shape != [2600, 3000, 2000]:
        raise ValueError('Only the archived R10 M2600 N3000 K2000 binding is accepted')
    cta = output['cta']
    ordinal = output['output_ordinal']
    wg = output['consumer_warp_group']
    if wg != ordinal % 2 or output['global_group_binding'] != {'first': ordinal * 32, 'last': ordinal * 32 + 31}:
        raise ValueError('Mismatched persistent ordinal / WG / K group binding')
    supplied = {tuple(box['origin']): box for box in output['D_boxes']}
    if len(supplied) != 8 or len(output['D_boxes']) != 8:
        raise ValueError('Every output must contain eight distinct 64x32 D boxes')
    tile_m, tile_n = output['coord']
    boxes = []
    for epi_n in range(4):
        for epi_m in range(2):
            q = epi_n * 2 + epi_m
            origin = (tile_m * 128 + epi_m * 64, tile_n * 128 + epi_n * 32)
            valid = [max(0, min(64, shape[0] - origin[0])), max(0, min(32, shape[1] - origin[1]))]
            valid_bytes = valid[0] * valid[1] * 4
            expected = {'origin': list(origin), 'shape': [64, 32], 'valid_shape': valid,
                        'nominal_source_bytes': 8192, 'global_valid_bytes': valid_bytes,
                        'global_oob_bytes': 8192 - valid_bytes}
            if supplied.get(origin) != expected:
                raise ValueError(f'D box mismatch at {cta=} {ordinal=} {origin=}')
            box = dict(expected)
            box.update({'subtile_ordinal': q, 'epi_m': epi_m, 'epi_n': epi_n,
                        'class': 'full' if valid_bytes == 8192 else 'zero_valid' if valid_bytes == 0 else 'partial',
                        'store_state_count_before_commit': ordinal * 8 + q,
                        'store_state_slot': q % 2,
                        'store_state_generation': (ordinal * 8 + q) // 2,
                        'store_state_phase_unused_by_store_wait': 1 ^ (((ordinal * 8 + q) // 2) % 2),
                        'read1_required_prior_subtiles': list(range(q)),
                        'same_slot_previous_subtile': q - 2 if q >= 2 else None,
                        'register_and_smem_preparation_required': True,
                        'fence_and_sync_and_store_path_required': True,
                        'read_completion_obligation_required': True,
                        'global_valid_write_obligation_required': valid_bytes > 0,
                        'physical_smem_read_bytes': None, 'time_price': None})
            boxes.append(box)
    phase = int(wg == 0) ^ ((ordinal // 2) % 2)
    prefix = output_prefix(cta, ordinal)
    threads = []
    for local_thread in range(128):
        threads.append({'wg_thread': local_thread, 'cta_thread': 128 + 128 * wg + local_thread,
                        'issuing_warp_path': local_thread < 32,
                        'mma_tail_return': f'{prefix}.mma_tail.return.t{local_thread}',
                        'epilogue_wait_return': f'{prefix}.epilogue_order.wait.return.t{local_thread}',
                        'store_tail_return': f'{prefix}.store_tail.return.t{local_thread}',
                        'epilogue_order_arrive': f'{prefix}.epilogue_order.arrive.t{local_thread}'})
    return {'cta': cta, 'output_ordinal': ordinal, 'consumer_warp_group': wg,
            'coord': output['coord'], 'global_group_binding': output['global_group_binding'],
            'order_barrier': {'sequence_depth': 2, 'sequence_length': 2, 'participants': 128,
                              'math_stage_index': 0, 'epilogue_stage_index': 1, 'wait_phase': phase,
                              'wait_group_id': wg, 'arrival_destination_group_id': 1 - wg,
                              'previous_output_ordinal': ordinal - 1 if ordinal else None,
                              'initial_wait_satisfied': ordinal == 0},
            'store_state': {'initial_count': ordinal * 8, 'after_store_count': ordinal * 8 + 8,
                            'after_pingpong_extra_advance_count': ordinal * 8 + 16,
                            'next_same_wg_output_ordinal': ordinal + 2,
                            'stages_D': 2, 'unacquired_stages': 1, 'always_wait': True,
                            'delay_tma_store': True, 'reuse_smem_C': False},
            'aggregate_ports': {'entry': f'{prefix}.D.entry_all128',
                                'store_return': f'{prefix}.store.return_all128',
                                'store_tail_return': f'{prefix}.store_tail.return_all128',
                                'epilogue_order_arrive': f'{prefix}.epilogue_order.arrive_all128',
                                'exit': f'{prefix}.D.exit_all128'},
            'thread_ports': threads, 'D_boxes_source_order': boxes}


def make_cta_graph(outputs):
    """Necessary unpriced source constraints; aggregate ports are max envelopes."""
    outputs = sorted(outputs, key=lambda row: row['output_ordinal'])
    if not outputs or [row['output_ordinal'] for row in outputs] != list(range(len(outputs))):
        raise ValueError('A complete, consecutive CTA output sequence is required')
    cta = outputs[0]['cta']
    if any(row['cta'] != cta for row in outputs):
        raise ValueError('One CTA per graph')
    nodes = {}
    edges = []

    def add(name, kind='source_event', scope='aggregate', external=False):
        record = {'id': name, 'kind': kind, 'scope': scope, 'external': external, 'time_price': None}
        if name in nodes and nodes[name] != record:
            raise ValueError(f'Conflicting node {name}')
        nodes[name] = record
        return name

    def edge(before, after, rule):
        if before not in nodes or after not in nodes:
            raise ValueError('Unknown port')
        edges.append([before, after, rule])

    async_completions = []
    for row in outputs:
        j = row['output_ordinal']
        prefix = output_prefix(cta, j)
        math_arrive = add(f'{prefix}.math_order.arrive_all128', 'external_input_port', external=True)
        tail_all = add(f'{prefix}.mma_tail.return_all128', 'max_envelope')
        epi_wait_all = add(f'{prefix}.epilogue_order.wait.return_all128', 'max_envelope')
        entry_all = add(row['aggregate_ports']['entry'], 'max_envelope')
        for port in row['thread_ports']:
            t = port['wg_thread']
            own_tail = add(port['mma_tail_return'], 'external_input_port', f'wg_thread{t}', True)
            wait = add(port['epilogue_wait_return'], scope=f'wg_thread{t}')
            edge(own_tail, wait, 'same_thread_mma_tail_returns_before_epilogue_order_wait_call')
            if j:
                prior = nodes[f'{output_prefix(cta, j-1)}.epilogue_order.arrive_all128']['id']
                edge(prior, wait, 'previous_WG_all128_arrivals_unlock_stage1')
            else:
                init = add(f'cta{cta}.epilogue_order.initial_phase_ready', 'initial_condition')
                edge(init, wait, 'WG0_initial_opposite_phase')
            edge(own_tail, tail_all, 'max_over128_threads')
            edge(wait, epi_wait_all, 'max_over128_threads')
        edge(math_arrive, tail_all, 'own_math_arrivals_precede_own_tail_per_thread_then_max')
        edge(tail_all, entry_all, 'own_mma_tail_required')
        edge(epi_wait_all, entry_all, 'epilogue_order_wait_required')

        # NamedBarrier syncs require all128 arrivals, not all128 return instants.
        prior_post_sync = None
        source_reads = {}
        for box in row['D_boxes_source_order']:
            q = box['subtile_ordinal']
            box_prefix = f'{prefix}.q{q}'
            r2s = add(f'{box_prefix}.r2s_ready_all128', 'max_envelope')
            edge(entry_all if q == 0 else prior_post_sync, r2s,
                 'first_store_entry' if q == 0 else 'previous_acquire_sync_before_next_R2S')
            pre_sync = add(f'{box_prefix}.fence_pre_sync.arrivals_all128', 'named_barrier_arrival_completion')
            edge(r2s, pre_sync, 'all128_R2S_then_proxy_fence_then_pre_sync_arrival')
            post_sync = add(f'{box_prefix}.post_acquire_sync.arrivals_all128', 'named_barrier_arrival_completion')
            read_all = add(f'{box_prefix}.source_read_complete_all_issuing_paths', 'max_envelope')
            issue_all = add(f'{box_prefix}.issue_all_issuing_paths', 'max_envelope')
            for t in range(32):
                issue = add(f'{box_prefix}.issue.t{t}', 'issuing_warp_source_path', f'wg_thread{t}')
                read = add(f'{box_prefix}.source_read_complete.t{t}', 'latent_completion', f'wg_thread{t}')
                commit = add(f'{box_prefix}.commit.t{t}', scope=f'wg_thread{t}')
                wait1 = add(f'{box_prefix}.read1.return.t{t}', scope=f'wg_thread{t}')
                edge(pre_sync, issue, 'all128_fences_before_TMA_source_path')
                edge(issue, issue_all, 'max_over_issuing_warp_source_paths')
                edge(issue, read, 'bulk_source_read_completion_no_price')
                edge(issue, commit, 'issue_then_per_thread_bulk_group_commit')
                edge(commit, wait1, 'always_wait_read1_after_each_commit')
                for previous in range(q):
                    edge(source_reads[(previous, t)], wait1, 'read1_drains_all_older_than_latest_group')
                edge(wait1, post_sync, 'issuing_thread_post_sync_arrival_after_its_read1')
                edge(read, read_all, 'max_over_issuing_warp_paths')
                source_reads[(q, t)] = read
            # Non-issuing threads also reach post_sync; this port is their arrival envelope.
            nonissuer = add(f'{box_prefix}.post_sync.nonissuing_arrivals96', 'max_envelope')
            edge(pre_sync, nonissuer, 'nonissuing_threads_skip_issue_commit_acquire')
            edge(nonissuer, post_sync, 'all128_named_barrier_includes_other96')
            completion = add(f'{box_prefix}.async_operation_complete', 'external_completion_requirement', external=True)
            edge(read_all, completion, 'entire_async_completion_includes_source_reads')
            if box['global_valid_write_obligation_required']:
                visible = add(f'{box_prefix}.global_valid_writes_visible', 'external_completion_requirement', external=True)
                edge(issue_all, visible, 'valid_destination_visibility_follows_issuing_source_paths')
                edge(visible, completion, 'valid_D_visibility_separate_from_read0')
            async_completions.append(completion)
            prior_post_sync = post_sync

        store_all = add(row['aggregate_ports']['store_return'], 'max_envelope')
        store_tail_all = add(row['aggregate_ports']['store_tail_return'], 'max_envelope')
        arrive_all = add(row['aggregate_ports']['epilogue_order_arrive'], 'arrival_completion_128')
        output_exit = add(row['aggregate_ports']['exit'], 'max_envelope')
        for port in row['thread_ports']:
            t = port['wg_thread']
            store_return = add(f'{prefix}.store.return.t{t}', scope=f'wg_thread{t}')
            tail_return = add(port['store_tail_return'], scope=f'wg_thread{t}')
            arrive = add(port['epilogue_order_arrive'], scope=f'wg_thread{t}')
            edge(prior_post_sync, store_return, 'last_named_barrier_then_store_return')
            edge(store_return, tail_return, 'per_thread_store_then_store_tail_read0')
            if t < 32:
                for q in range(8):
                    edge(source_reads[(q, t)], tail_return, 'read0_drains_own_issuing_path_groups')
            edge(store_return, store_all, 'max_over128_threads')
            edge(tail_return, store_tail_all, 'max_over128_threads')
            edge(tail_return, arrive, 'per_thread_tail_return_then_state_advance_then_order_arrive')
            edge(arrive, arrive_all, 'all128_order_arrivals_required')
        edge(store_tail_all, arrive_all, 'per_thread_precedence_implies_max_precedence_not_a_barrier')
        edge(store_tail_all, output_exit, 'aggregate_exit_includes_tail_return')
        edge(arrive_all, output_exit, 'aggregate_exit_includes_all128_order_arrivals')

        math_wait_all = add(f'{prefix}.math_order.wait.return_all128', 'max_envelope')
        permission_all = None
        if j >= 2:
            permission_all = add(f'{prefix}.same_wg_next_mma_permission_all128', 'max_envelope')
        for t in range(128):
            permission = add(f'{prefix}.math_order.wait.return.t{t}', 'input_composition_port', f'wg_thread{t}', True)
            if j:
                previous_math = nodes[f'{output_prefix(cta, j-1)}.math_order.arrive_all128']['id']
                edge(previous_math, permission, 'previous_other_WG_all128_math_arrivals_unlock_stage0')
            else:
                math_init = add(f'cta{cta}.math_order.initial_phase_ready', 'initial_condition')
                edge(math_init, permission, 'WG0_initial_opposite_phase')
            if j >= 2:
                previous_same_wg = output_prefix(cta, j - 2)
                edge(nodes[f'{previous_same_wg}.epilogue_order.arrive.t{t}']['id'], permission,
                     'same_thread_previous_own_epilogue_arrive_then_next_math_wait')
                edge(permission, permission_all, 'max_over128_threads')
            edge(permission, math_wait_all, 'max_over128_threads')
            edge(permission, nodes[f'{prefix}.mma_tail.return.t{t}']['id'],
                 'own_MMA_and_tail_follow_math_wait')
        edge(math_wait_all, math_arrive, 'own_MMA_return_order_arrive_follows_math_wait_per_thread_then_max')

    producer_tail = add(f'cta{cta}.input.producer_tail_return', 'external_input_port', external=True)
    sources_closed = add(f'cta{cta}.source_lifetimes_closed', 'completion_requirement')
    edge(producer_tail, sources_closed, 'input_lifetime_closes_separately')
    for wg in range(2):
        owned = [row for row in outputs if row['consumer_warp_group'] == wg]
        if owned:
            edge(owned[-1]['aggregate_ports']['exit'], sources_closed, 'last_own_output_closes_D_source_lifetime')
    threads_return = add(f'cta{cta}.threads_return_all384', 'external_completion_requirement', external=True)
    edge(sources_closed, threads_return, 'all_threads_return_requires_closed_input_and_D_source_lifetimes')
    cuda_complete = add(f'cta{cta}.cuda_kernel_completion_requirement', 'external_completion_requirement', external=True)
    edge(threads_return, cuda_complete, 'CUDA_completion_covers_all_threads')
    for completion in async_completions:
        edge(completion, cuda_complete, 'CUDA_completion_also_covers_async_D_operations')
    return {'schema': 'cfg-b-legacy-D-source-constraints/1', 'cta': cta,
            'semantics': 'Necessary precedence only. Aggregate ports are maxima or arrival-completion requirements; no timestamp, physical clock, price, or new synchronization is supplied.',
            'nodes': list(nodes.values()), 'edges': edges}


def check_dag(graph):
    names = {node['id'] for node in graph['nodes']}
    if len(names) != len(graph['nodes']):
        raise ValueError('Duplicate node')
    incoming = dict.fromkeys(names, 0)
    successors = defaultdict(list)
    for before, after, rule in graph['edges']:
        if before not in names or after not in names:
            raise ValueError('Dangling edge')
        incoming[after] += 1
        successors[before].append(after)
    queue = deque(name for name in names if incoming[name] == 0)
    count = 0
    while queue:
        name = queue.popleft()
        count += 1
        for after in successors[name]:
            incoming[after] -= 1
            if incoming[after] == 0:
                queue.append(after)
    if count != len(names):
        raise ValueError('Source constraint cycle')
    return {'nodes': count, 'edges': len(graph['edges'])}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-doc', type=Path, default=Path(__file__).parent / 'source/R15-output-service-at-e6eda283.md')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / 'C-legacy-D-lifecycle.json'
    if report_path.exists():
        raise FileExistsError('Use a fresh output directory; old reports are immutable')
    inputs = {}
    manifest_file = args.run / 'source_hashes.json'
    manifest = json.loads(manifest_file.read_text())
    inputs[str(manifest_file)] = sha256(manifest_file)
    if inputs[str(manifest_file)] != 'bb54fd59a527fda4a318e4c9ba5f0f6537b1ef0e1163968fb0a7b15dd5f8bf1f':
        raise ValueError('Wrong original R10 manifest')
    source_sites = {}
    for relative, patterns in SOURCE_SITES.items():
        path = args.run / 'source' / relative
        actual = sha256(path)
        if manifest.get(relative) != actual:
            raise ValueError(f'Archived source SHA mismatch: {relative}')
        inputs[str(path)] = actual
        lines = path.read_text().splitlines()
        sites = []
        for pattern in patterns:
            matches = [i for i, line in enumerate(lines, 1) if pattern in line]
            if not matches:
                raise ValueError(f'Expected source site missing: {relative}: {pattern}')
            sites.append({'pattern': pattern, 'lines': matches})
        source_sites[relative] = {'sha256': actual, 'sites': sites}
    for relative in ['reanalysis/B-20261011-boundary-binding-v1/B-boundary-binding.json',
                     'reanalysis/B-20261011-boundary-lifecycle-v2/B-boundary-lifecycle.json']:
        inputs[str(args.run / relative)] = sha256(args.run / relative)
    b_request = json.loads((args.run / 'reanalysis/B-20261011-boundary-lifecycle-v2/B-boundary-lifecycle.json').read_text())['C_boundary_request']
    inputs[str(args.reuse_doc)] = sha256(args.reuse_doc)
    inputs[str(Path(__file__).resolve())] = sha256(__file__)
    cases = []
    payloads = []
    for ldb in [3000, 3008, 3072]:
        source = args.run / f'reanalysis/B-20261011-boundary-binding-v1/cfg_b_ldb{ldb}-persistent-ledger.json.gz'
        inputs[str(source)] = sha256(source)
        ledger = json.load(gzip.open(source, 'rt'))
        bound = [bind_output(output, ledger['shape']) for output in ledger['outputs']]
        bound.sort(key=lambda row: (row['cta'], row['output_ordinal']))
        by_cta = defaultdict(list)
        counts = Counter()
        for row in bound:
            by_cta[row['cta']].append(row)
            for box in row['D_boxes_source_order']:
                counts[box['class']] += 1
                counts['nominal_source_bytes'] += box['nominal_source_bytes']
                counts['global_valid_bytes'] += box['global_valid_bytes']
        if sorted(by_cta) != list(range(132)):
            raise ValueError('Missing CTA binding')
        dag_counts = Counter()
        for cta in sorted(by_cta):
            rows = by_cta[cta]
            if len(rows) != ledger['scheduler']['output_counts'][cta]:
                raise ValueError('Scheduler T mismatch')
            dag_counts.update(check_dag(make_cta_graph(rows)))
        if counts['global_valid_bytes'] != 2600 * 3000 * 4 or counts['nominal_source_bytes'] != ledger['totals']['D_nominal_bytes']:
            raise ValueError('D area does not match original work ledger')
        if counts['zero_valid'] != ledger['totals']['D_zero_valid_boxes']:
            raise ValueError('Zero-valid box count mismatch')
        payload = args.out / f'cfg_b_ldb{ldb}-D-lifecycle-ledger.json.gz'
        # Deterministic gzip: no filename or current timestamp embedded.
        raw = (json.dumps({'schema': 'cfg-b-legacy-D-lifecycle-ledger/1', 'ldb': ldb,
                           'shape': ledger['shape'], 'outputs': bound}, sort_keys=True, separators=(',', ':')) + '\n').encode()
        with payload.open('wb') as stream:
            with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
                zipped.write(raw)
        payloads.append({'file': payload.name, 'sha256': sha256(payload)})
        cases.append({'ldb': ldb, 'CTAs': len(by_cta), 'outputs': len(bound),
                      'T_distribution': dict(sorted(Counter(map(len, by_cta.values())).items())),
                      'D_boxes': len(bound) * 8, 'box_classes': {k: counts[k] for k in ['full', 'partial', 'zero_valid']},
                      'D_nominal_source_bytes': counts['nominal_source_bytes'],
                      'D_global_valid_bytes': counts['global_valid_bytes'],
                      'explicit_thread_tail_and_order_pairs': len(bound) * 128,
                      'source_subtile_store_sequences': len(bound) * 8,
                      'source_output_store_tail_invocations_all128': len(bound),
                      'source_thread_read1_invocations_in_issuing_warp_paths': len(bound) * 8 * 32,
                      'source_thread_read0_invocations': len(bound) * 128,
                      'expanded_graph_counts': dict(dag_counts), 'payload': payload.name})
    report = {'schema': 'C-cfg-b-legacy-D-static-lifecycle/1',
              'scope': 'Original R10 cfg_b pingpong D lifecycle only; M2600 N3000 K2000, T3/T4, legacy C=void, float D, 64x32 epilogue. No prices or qualification.',
              'C_boundary_request': b_request,
              'C_boundary_response': {
                  'entry': 'Per-thread own mma_tail return AND epilogue stage1 order wait return; aggregate entry is their all128 max. Own MMA notification precedes own mma_tail in the original source.',
                  'store_tail': 'Every output calls read0. Bulk groups are per issuing thread; other threads may return/arrive before the issuing warp source reads finish.',
                  'handoff': 'All128 epilogue-order arrivals target the other WG stage1 barrier. Their completion includes the issuing warp read0 returns, hence all output D source buffers are safe for the next epilogue.',
                  'same_WG_j_plus_2': 'Each thread follows its own previous epilogue arrival then stage0 math wait. This wait also needs all128 math-order arrivals from output j+1. The all128 permission port is a max, not a new WG barrier.',
                  'global_D': 'Valid global D writes have a separate latent completion/visibility obligation. read0 and order arrivals do not certify it. Zero-valid boxes have no valid-global-write obligation but retain the source-level store protocol and source-read-completion obligation.',
                  'exit': 'producer_tail closes input slots only. Source-safe CTA lifetime requires producer_tail plus both WGs final D source/order closures. CUDA kernel completion additionally covers all384 thread returns and asynchronous D operations.',
              },
              'reuse': {'R15_doc_sha256': sha256(args.reuse_doc),
                        'used': ['64x32 / eight8KiB subtiles / StagesD2 / DelayTmaStore / ReuseSmemCfalse', 'read0 source-read rather than global-write completion'],
                        'new_binding': ['original pingpong per-output read0 + 128 arrivals', 'two-stage ordered sequence / j+2 continuation', 'original persistent output coordinates and D state counts'],
                        'not_transferred': ['cooperative topology', 'scalar STS or cooperative prices', 'R14 different output geometry', 'timestamps or GT boxes']},
              'graph_API': {'function': 'make_cta_graph(bound_outputs_for_one_CTA)',
                            'aggregate_definition': 'max of participating thread events or all-required-arrivals completion; never a single-thread sample',
                            'edges': 'necessary source precedence with symbolic completion obligations; all time_price fields null',
                            'issuer_scope': 'C++ first-warp paths t0..31. No SASS/lane lowering or physical issue multiplicity is inferred.',
                            'unmodeled_interleaving': 'Current register computation may overlap delayed prior TMA store; aggregate constraints provide no duration or total order for that preparation.'},
              'source_manifest_matches': len(source_sites), 'archived_source_sites': source_sites,
              'cases': cases, 'payloads': payloads, 'input_sha256': inputs,
              'external_sources': [
                  {'url': 'https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group',
                   'use': 'Per-thread bulk groups; read wait completes source/tensormap reads and excludes a destination-visibility guarantee.'},
                  {'url': 'https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma',
                   'use': 'Only initiating threads wait for shared-to-global bulk copies; partial tensor store boxes may cross positive global bounds. Valid output area is a source-bound geometry inference from the destination tensor; physical SMEM read amount is not identified.'},
                  {'url': 'https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__EVENT.html',
                   'use': 'A recorded event captures preceding stream work. A successful completion wait bounds the completed kernel, not an internal CTA read0 timestamp.'}],
              'qualification': {'source_binding_checked': True, 'DAG_checked': True,
                                'GPU_executed': False, 'time_prices_identified': False,
                                'global_D_visibility_observed': False, 'observer_qualified': False,
                                'physical_clock_anchor_used': False, 'independent_local_holdout_passed': False,
                                'V09_candidates_changed': False, 'qualified': False}}
    write_json(report_path, report)
    print(json.dumps({'report': str(report_path), 'source_manifest_matches': len(source_sites),
                      'cases': cases, 'qualified': False}, indent=2))


if __name__ == '__main__':
    main()
