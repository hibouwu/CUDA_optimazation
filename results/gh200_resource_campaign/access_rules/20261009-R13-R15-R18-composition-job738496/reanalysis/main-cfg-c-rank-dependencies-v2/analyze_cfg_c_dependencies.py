#!/usr/bin/env python3
"""Bind cfg_c rank closure to archived CUTLASS source and B's saved source ports."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from output_model import cfg_c_rank_dependencies, dependency_order


def analyze(run,source_ports,output):
    if output.exists():raise ValueError('output must be a new C-only directory')
    inputs={}
    def bind(path):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=digest
        return path.read_text()
    paths=dict(probe=run/'source/probes/r18.cu',
        collective=run/'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
        kernel=run/'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp',
        pipeline=run/'source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp',
        epilogue=run/'source/cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp',
        builder=run/'source/cutlass/include/cutlass/epilogue/collective/builders/sm90_builder.inl',
        tma=run/'source/cutlass/include/cute/arch/copy_sm90_tma.hpp')
    text={key:bind(path) for key,path in paths.items()}
    contracts=dict(
        probe=['using KernelTile = Shape<_256, _128, _64>;', 'using KernelCluster = Shape<_1, _2, _1>;',
            'using EpilogueTile = Shape<_128, _32>;', 'StageCount<4>;',
            'float, float, void, cutlass::layout::RowMajor, 4', 'Kernel::NumMMAThreads == 256'],
        collective=['static constexpr int K_PIPE_MMAS = 1;',
            'get_slice(cluster_local_block_id.y)', 'get_slice(cluster_local_block_id.x)',
            'pipeline.producer_acquire(smem_pipe_write);', 'warpgroup_wait<K_PIPE_MMAS>();',
            'pipeline.consumer_release(smem_pipe_release);', 'warpgroup_wait<0>();'],
        pipeline=['Swizzle<2,0,-2>{}', 'uint32_t thread_row = warp_idx % 4;',
            'uint32_t thread_col = (thread_idx_in_warpgroup / 8) % 4;',
            '(cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1)',
            'empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_ & (!skip));',
            'full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);',
            'for (int count = 0; count < Stages; ++count)', 'constexpr uint32_t InitialProducerPhase = 1;'],
        kernel=['mainloop_pipeline_params.num_consumers = NumMMAThreads;',
            'mainloop_pipe_producer_state.advance(work_k_tile_count);',
            'mainloop_pipe_consumer_state.advance(work_k_tile_count);',
            'collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);'],
        builder=['constexpr bool DelayTmaStore = is_void_v<ElementC>', 'constexpr int StagesD = cute::min(EpiTiles, 2);'],
        epilogue=['for (int epi_n = 0; epi_n < size<3>(gD_epi); ++epi_n)',
            'for (int epi_m = 0; epi_m < size<2>(gD_epi); ++epi_m)',
            'store_pipeline.producer_acquire(store_pipe_producer_state);',
            'store_pipeline.producer_tail(store_pipe_producer_state);'],
        tma=['cp.async.bulk.wait_group.read %0;'])
    ports=[]
    for key,needles in contracts.items():
        for needle in needles:
            if needle not in text[key]:raise ValueError('source contract changed: '+key+' '+needle)
            ports.append(dict(source=key,line=text[key].count('\n',0,text[key].index(needle))+1,text=needle))
    # Verify the relevant source-order cuts, not merely the presence of symbols.
    mma=text['collective'][text['collective'].index('  mma(MainloopPipeline pipeline,'):]
    steady=mma[mma.index('    // Mainloop GMMAs'):mma.index('  /// Perform a Consumer Epilogue')]
    assert steady.index('warpgroup_commit_batch();')<steady.index('warpgroup_wait<K_PIPE_MMAS>();')<steady.index('pipeline.consumer_release(smem_pipe_release);')
    kernel=text['kernel'];mma_call=kernel.index('          collective_mainloop.mma(\n')
    cuts=['          collective_mainloop.mma_tail(', '          v06_stamp(V06_MAIN_END',
          '          v06_stamp(V06_EPI_PERMIT', '          collective_epilogue.store(',
          '          v06_stamp(V06_EPI_DONE', '        collective_epilogue.store_tail(', '      v06_final(']
    indices=[kernel.index(needle,mma_call) for needle in cuts]
    assert indices==sorted(indices)
    report=json.loads(bind(source_ports/'B-cfg-c-source-ports.json'))
    verified=0
    for name,digest in report['input_sha256'].items():
        path=Path(name)
        if run.parent in path.parents:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path
            inputs[str(path)]=digest;verified+=1
    output.mkdir(parents=True,exist_ok=False)
    templates=[]
    for template in report['source_templates']:
        path=source_ports/template['artifact']
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path,'rt') as stream:source=json.load(stream)
        graph=cfg_c_rank_dependencies(source);order,_=dependency_order(graph)
        assert all(set(parents)<=set(graph['nodes'][name]) for name,parents in source['nodes'].items())
        filename=template['artifact'].replace('source-ports','rank-dependencies')
        raw=json.dumps(graph,separators=(',',':'),allow_nan=False).encode()
        (output/filename).write_bytes(gzip.compress(raw,mtime=0))
        templates.append(dict(stages=4,k_tiles=graph['k_tiles'],output_tiles=graph['output_tiles'],
            nodes=len(order),artifact=filename,external_entry_ports=graph['external_entry_ports'],
            closed_producer_ports=len(source['external_producer_ports']),
            source_subtiles_per_rank=8*graph['output_tiles']))
    sizes={(x['k_tiles'],x['output_tiles']) for x in templates}
    binding_path=source_ports/'B-cfg-c-source-bindings.json.gz'
    inputs[str(binding_path)]=hashlib.sha256(binding_path.read_bytes()).hexdigest()
    with gzip.open(binding_path,'rt') as stream:bindings=json.load(stream)
    clusters=outputs=0
    for case in bindings:
        binding=case['binding']
        assert binding['stages']==4 and binding['schema']=='gh200-cfg-c-source-bindings/1'
        for cluster in binding['clusters']:
            assert (binding['k_tiles'],cluster['output_tiles']) in sizes
            clusters+=1;outputs+=2*cluster['output_tiles']
    for name in ('output_model.py','analyze_cfg_c_dependencies.py','test_cfg_c_dependencies.py'):
        path=Path(__file__).with_name(name);inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(source_contract=ports,input_sha256=inputs,verified_archived_B_inputs=verified,
        source_templates=templates,static_cases=len(bindings),static_clusters=clusters,static_rank_outputs=outputs,
        unresolved=['cluster_initialized is an entry boundary with no calibrated startup time.',
            'Logical dependency closure supplies no service prices, raw-event precision or autonomous time prediction.',
            'Source retirement and D visibility are distinct. No new cfg_c fine-stage measurements or local holdout.'],
        new_gpu_processes=0,service_prices_changed=False,observation_qualified=False)
    (output/'C-cfg-c-rank-dependencies.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(templates=len(templates),static_cases=len(bindings),static_clusters=clusters,
        static_rank_outputs=outputs,verified_archived_B_inputs=verified)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--source-ports',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();analyze(args.run.resolve(),args.source_ports.resolve(),args.output.resolve())
