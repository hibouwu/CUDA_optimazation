#!/usr/bin/env python3
"""Project the new smoke's raw FIRST/MAIN slots onto pinned C source graphs.

Reads archived observations only. Outputs remain explicitly unqualified.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from lifecycle_brackets import canonical_sha256, project_first_main
from model_snapshot.output_model import cfg_c_rank_dependencies


HERE = Path(__file__).parent


def review(run, ordering_review, output):
    if output.exists():
        raise ValueError('output must be a new C-owned directory')
    inputs = {}

    def read(path):
        data = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data

    def load(path):
        return json.loads(read(path))

    freeze = load(run/'freeze.json')['files']
    for name, expected in freeze.items():
        path = run/name
        if not path.exists():
            path = run/'returned'/name
        assert hashlib.sha256(read(path)).hexdigest() == expected, name
    assert load(run/'returned/freeze.json') == {'files':freeze}
    manifest = load(run/'manifest.json')
    assert manifest['binary_sha256'] == freeze['build/cfg_c_s4_dual']
    ordering = load(ordering_review)
    ordering_sha = inputs[str(ordering_review.resolve())]
    assert ordering['source_and_sass_dependency_review_passed'] is True
    assert ordering['binary_sha256'] == manifest['binary_sha256']
    assert not any(ordering['qualification'].values())
    assert manifest['qualification']['observer_perturbation'] is False
    assert manifest['qualification']['null_control'] is False

    snapshot_manifest = load(HERE/'model_snapshot/manifest.json')
    for name, expected in snapshot_manifest['input_sha256'].items():
        assert hashlib.sha256(read(Path(name))).hexdigest() == expected, name
    source_model_path = next(name for name in snapshot_manifest['input_sha256'] if name.endswith('/output_model.py'))
    assert read(HERE/'model_snapshot/output_model.py') == read(Path(source_model_path))
    static = load(HERE/'model_snapshot/cfg_c-K64-aligned-bindings.json')
    binding_source = next(name for name in snapshot_manifest['input_sha256'] if name.endswith('/B-cfg-c-source-bindings.json.gz'))
    original_rows = json.loads(gzip.decompress(read(Path(binding_source))))
    source_cases = [row for row in original_rows if row['run'] == static['source_run'] and row['case'] == static['source_case']]
    assert len(source_cases) == 1
    for original, reduced in zip(source_cases[0]['binding']['clusters'],static['clusters']):
        expected = dict(cluster=original['cluster'],ctas=original['ctas'],output_tiles=original['output_tiles'],
                        rank_outputs=[[dict(j=o['j'],coord=o['coord']) for o in rank] for rank in original['rank_outputs']])
        assert expected == reduced
    assert len(static['clusters']) == len(source_cases[0]['binding']['clusters']) == 66
    graphs = {}
    for tiles in (1,2):
        path = HERE/'model_snapshot'/f'cfg_c-S4-K64-T{tiles}-source-ports.json.gz'
        source_bytes = read(path)
        source_path = next(name for name in snapshot_manifest['input_sha256'] if name.endswith(path.name))
        assert source_bytes == read(Path(source_path))
        graphs[tiles] = cfg_c_rank_dependencies(json.loads(gzip.decompress(source_bytes)))

    run_records = load(run/'returned/records.json')
    raw_record = next(row for row in run_records if row['id'] == 'flag1')
    raw_path = run/'returned'/raw_record['raw']
    raw_bytes = read(raw_path)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    assert raw_sha == raw_record['sha256'] and raw_record['returncode'] == 0
    packets = [json.loads(line) for line in gzip.decompress(raw_bytes).decode().splitlines()]
    assert [p['event'] for p in packets] == ['setup','call','check']
    setup, call, check = packets
    assert setup['gpu_uuid'] == manifest['gpu_uuid']
    expected = dict(config='cfg_c',m=2304,n=3072,k=4096,grid=[66,2,1],tile=[256,128,64],
                    cluster=[1,2],stages=4,threads=384,sm_count=132,trace_words=1296,trace_tile_words=10,
                    trace_version='cfg-c-gt-bracket-v1',gt_bracket_observer=1,producer_tail_observer=0)
    for name,value in expected.items():
        assert setup[name] == value, name
    assert call['gt_bracket_observer'] == 1 and call['producer_tail_observer'] == 0
    assert call['target_calls'] == 1 and call['preceding_target_calls'] == 0
    assert call['gt_bracket_prefix_flag'] == 0 and call['warmup_calls'] == 16
    trace, shadow = call['trace'],call['gt_shadow']
    assert len(trace) == 132*1296 and len(shadow) == 132*9
    assert all(type(x) is int and 0 <= x < 2**64 for x in trace+shadow)
    # The check packet is retained as evidence, not used to filter records.
    assert check['status'] == 'ok' and check['samples'] == 4096

    invocation = raw_sha+':warm-call-0'
    flat_stamps = []
    cluster_packets = []
    for cluster in static['clusters']:
        tiles = cluster['output_tiles']
        graph = graphs[tiles]
        bindings = {}
        records = []
        for rank, cta in enumerate(cluster['ctas']):
            base = cta*1296
            assert trace[base+10] == 0 and shadow[cta*9+8] == 0
            sm = trace[base+2]-1
            assert 0 <= sm < 132
            assert trace[base+11] == trace[base+12] == sm+1
            for wg in (0,1):
                assert trace[base+6+3*wg] == tiles
                for ordinal, static_output in enumerate(cluster['rank_outputs'][rank]):
                    assert ordinal == static_output['j']
                    tile_base = base+16+(wg*64+ordinal)*10
                    coord = [trace[tile_base+4]-1,trace[tile_base+5]-1]
                    assert coord == static_output['coord']
                    for event_index,event in enumerate(('FIRST','MAIN')):
                        node = f'r{rank}.w{wg}.t{ordinal}.{event}'
                        binding = dict(cta=cta,physical_thread=128*(wg+1),coord=coord)
                        bindings[node] = binding
                        before = shadow[cta*9+wg*4+ordinal*2+event_index]
                        cycles = trace[tile_base+event_index]
                        after = trace[tile_base+6+event_index]
                        records.append(dict(node=node,invocation=invocation,device_uuid=manifest['gpu_uuid'],
                            binary_sha256=manifest['binary_sha256'],**binding,sm=sm,
                            gt_before=before,cycles=cycles,gt_after=after,overflow=False))
                        flat_stamps.append(dict(cta=cta,cycles=cycles,
                            event=('FIRST_MMA','MAIN_END')[event_index],gt_after=after,gt_before=before,
                            output_ordinal=ordinal,sm=sm,thread=128*(wg+1),width=after-before,work=coord))
                # No GT_pre is invented for unused ordinals or for other events.
                assert all(x == 0 for x in shadow[cta*9+wg*4+tiles*2:cta*9+(wg+1)*4])
                assert all(x == 0 for x in trace[base+16+(wg*64+tiles)*10:base+16+(wg+1)*64*10])
        contract = dict(schema='C-gt-lifecycle-contract/1',graph_family='cfg_c',
            graph_sha256=canonical_sha256(graph),invocation=invocation,device_uuid=manifest['gpu_uuid'],
            binary_sha256=manifest['binary_sha256'],ordering_review_sha256=ordering_sha,
            actual_capture_order_assumed=True,monotone_gt_reference_assumed=True,
            rank_ctas=cluster['ctas'],sm_count=setup['sm_count'],anchor_bindings=bindings)
        packet = dict(schema='C-gt-lifecycle-packet/1',contract_sha256=canonical_sha256(contract),
            invocation=invocation,device_uuid=manifest['gpu_uuid'],binary_sha256=manifest['binary_sha256'],
            data_origin='measured_unqualified',records=records)
        cluster_packets.append(dict(cluster=cluster['cluster'],graph_tiles=tiles,contract=contract,packet=packet))
    key = lambda stamp:(stamp['cta'],stamp['thread'],stamp['output_ordinal'],stamp['event'])
    main_stamps = json.loads(gzip.decompress(read(run/'stamps.json.gz')))
    assert sorted(flat_stamps,key=key) == sorted(main_stamps,key=key)
    assert len(flat_stamps) == 864
    origin = min(stamp['gt_before'] for stamp in flat_stamps)
    projections = []
    endpoints = []
    for item in cluster_packets:
        projection = project_first_main(graphs[item['graph_tiles']],item['contract'],item['packet'],origin)
        complete = projection['completion_intervals_gt_counter_ns']
        assert not any(projection['qualification'].values())
        assert complete['cluster_source_safe_exit'][1] is None
        assert complete['cluster_cuda_complete'][1] is None
        for cta, interval in complete['consumer_FINAL_by_CTA'].items():
            assert interval[1] is None
            endpoints.append(dict(cta=int(cta),consumer_FINAL_gt_counter_ns=interval))
        projections.append(dict(cluster=item['cluster'],projection=projection))
    assert sorted(row['cta'] for row in endpoints) == list(range(132))
    for name in ('lifecycle_brackets.py','review_smoke.py','test_lifecycle_brackets.py','tests-final.txt'):
        read(HERE/name)
    result = dict(schema='C-smoke-lifecycle-review/1',input_sha256=inputs,
        invocation=invocation,run=str(run),data_origin='measured_unqualified',
        source_binding_case=static['source_case'],clusters=66,CTAs=132,
        output_tiles_by_cluster=dict(Counter(item['graph_tiles'] for item in cluster_packets)),
        FIRST_MAIN_stamps=864,WG_output_windows=432,raw_and_main_stamp_records_equal=True,
        GT_width_counts=dict(sorted(Counter(stamp['width'] for stamp in flat_stamps).items())),
        shared_origin_gt_counter_ns=origin,all_cluster_source_constraints_feasible=True,
        consumer_FINAL_upper_unbounded_CTAs=132,source_safe_exit_upper_unbounded_clusters=66,
        cuda_complete_upper_unbounded_clusters=66,
        conservative_possible_last_consumer_CTAs=list(range(132)),
        possible_set_is_only_a_superset=True,V09_candidates_loaded=False,V09_candidates_changed=False,
        counter_origin_constructed_after_smoke=True,prospective_numeric_prediction=False,
        current_interface_has_no_FINAL_producer_tail_or_D_visible_brackets=True,
        qualification=dict(hardware_sampling_enclosure=False,physical_ns_accuracy=False,
            observation=False,observer_perturbation=False,null_control=False,
            lifecycle_prediction=False,critical_CTA=False,whole_gemm=False),
        C_new_gpu_processes=0,public_probe_or_reader_changed_by_C=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name, value in [('packets.json.gz',cluster_packets),('projections.json.gz',projections)]:
        raw = json.dumps(value,separators=(',',':'),allow_nan=False).encode()
        (output/name).write_bytes(gzip.compress(raw,mtime=0))
    (output/'consumer-endpoints.json').write_text(json.dumps(sorted(endpoints,key=lambda row:row['cta']),indent=2)+'\n')
    print(json.dumps({name:result[name] for name in ('clusters','CTAs','FIRST_MAIN_stamps','GT_width_counts',
        'consumer_FINAL_upper_unbounded_CTAs','source_safe_exit_upper_unbounded_clusters','cuda_complete_upper_unbounded_clusters')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--ordering-review',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    review(args.run.resolve(),args.ordering_review.resolve(),args.output.resolve())
