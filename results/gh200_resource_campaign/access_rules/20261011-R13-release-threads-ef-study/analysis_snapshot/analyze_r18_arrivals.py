#!/usr/bin/env python3
"""CPU-only verification of cfg_a empty-arrival emitters from frozen CUTE source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from output_model import cfg_a_empty_arrival_routes, cfg_a_rank_dependencies


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_layout_source(pipeline_source):
    """Keep the frozen layout function body unchanged; remove only device annotation."""
    signature='cute::tuple<bool, uint32_t> spread_arrivals_to_warpgroup('
    if pipeline_source.count(signature)!=1:raise ValueError('arrival layout function is not unique')
    start=pipeline_source.index(signature)
    brace=pipeline_source.index('{',start);depth=1;end=brace+1
    while depth:
        if end>=len(pipeline_source):raise ValueError('unclosed arrival layout function')
        depth+=(pipeline_source[end]=='{')-(pipeline_source[end]=='}');end+=1
    function=pipeline_source[start:end]
    return ('#include <cute/layout.hpp>\n#include <cute/layout_composed.hpp>\n'
            '#include <cute/swizzle.hpp>\n#include <cute/swizzle_layout.hpp>\n#include <iostream>\n'
            'using namespace cute;\nconstexpr int NumThreadsPerWarpGroup=128;\n'+function+'''
int main() {
  for (int physical=128; physical<384; ++physical) {
    auto [signal,destination]=spread_arrivals_to_warpgroup(physical%128,physical/32);
    // cluster2x1: both destinations share N=0, hence row/column filtering is true.
    if (signal && destination<2) std::cout << physical << " " << destination << "\\n";
  }
}
''')


def analyze(run,output,cuda_include,previous_graph_dir,compiler='c++'):
    run,output,cuda_include=Path(run),Path(output),Path(cuda_include)
    if output.exists():raise ValueError('output must be a new directory')
    pipeline=run/'source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp'
    code=host_layout_source(pipeline.read_text())
    output.mkdir(parents=True,exist_ok=False)
    source=output/'arrival_layout.cpp';binary=output/'arrival_layout'
    source.write_text(code)
    command=[compiler,'-std=c++17','-I'+str(run/'source/cutlass/include'),
             '-I'+str(cuda_include),'-I'+str(cuda_include/'cccl'),str(source),'-o',str(binary)]
    build=subprocess.run(command,capture_output=True,text=True)
    (output/'build.log').write_text(build.stdout+build.stderr)
    (output/'build.json').write_text(json.dumps(dict(command=command,returncode=build.returncode),indent=2)+'\n')
    if build.returncode:raise ValueError('host layout compilation failed; preserved build.log')
    execution=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    (output/'layout.stdout').write_text(execution.stdout)
    observed=[tuple(map(int,line.split())) for line in execution.stdout.splitlines()]
    routes=cfg_a_empty_arrival_routes()
    expected=[(r['physical_thread'],r['destination_rank']) for r in routes if r['source_rank']==0]
    if observed!=expected:raise ValueError('Python route metadata differs from frozen CUTE execution')
    source_paths=[pipeline,run/'source/cutlass/include/cute/layout.hpp',
        run/'source/cutlass/include/cute/layout_composed.hpp',run/'source/cutlass/include/cute/swizzle.hpp',
        run/'source/cutlass/include/cute/swizzle_layout.hpp',run/'source/cutlass/include/cutlass/cutlass.h',
        run/'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp',
        run/'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
        run/'source/cutlass/include/cutlass/arch/barrier.h',run/'build/cfg_a_dual.sass']
    graph=cfg_a_rank_dependencies(6,64,2)
    for route in routes:
        token=route['visibility_token_template'].format(group=0)
        if token not in graph['nodes'][f"r{route['destination_rank']}.g6.acquire"]:
            raise ValueError('emitter route is absent from the reuse prerequisite graph')
    sass=(run/'build/cfg_a_dual.sass').read_text()
    kernels=[block for block in sass.split('Function : ')[1:] if block.startswith('_ZN7cutlass13device_kernel')]
    if len(kernels)!=1:raise ValueError('unique archived GEMM SASS required')
    sites={}
    expected_sites={'1fe0':('WARPGROUP.DEPBAR.LE','0x1'),
        '1ff0':('SYNCS.ARRIVE.TRANS64.RED.A1T0',),'2050':('WARPGROUP.DEPBAR.LE','0x0'),
        '2120':('SYNCS.ARRIVE.TRANS64.RED.A1T0',),'2140':('CS2R','SR_CLOCKLO')}
    for offset,required in expected_sites.items():
        matching=[line for line in kernels[0].splitlines() if '/*'+offset+'*/' in line]
        if len(matching)!=1 or any(token not in matching[0] for token in required):
            raise ValueError('archived SASS site changed: '+offset)
        sites[offset]=matching[0].strip().split(';',1)[0]+';'
    projections=[]
    for path in sorted(Path(previous_graph_dir).glob('cfg_a-S*-K*-T*.json')):
        old=json.loads(path.read_text())
        current=cfg_a_rank_dependencies(old['stages'],old['k_tiles'],old['output_tiles'])
        def expand(event):
            if event in old['nodes']:return [event]
            return [prior for parent in current['nodes'][event] for prior in expand(parent)]
        for event,parents in old['nodes'].items():
            projected={prior for parent in current['nodes'][event] for prior in expand(parent)}
            if projected!=set(parents):raise ValueError('old dependency order changed: '+event)
        projections.append(dict(previous=str(path),previous_sha256=sha(path),old_nodes=len(old['nodes']),
            new_nodes=len(current['nodes']),old_predecessors_preserved_after_subdivision=True))
    if not projections:raise ValueError('previous source-guarantee graphs required')
    report=dict(scope='Static emitters for cfg_a cluster2x1; no GPU launch or timing/visibility inference.',
        routes=routes,cute_host_output=observed,host_matches_prediction_metadata=True,
        layout_function='Unchanged frozen spread_arrivals_to_warpgroup body compiled for CPU.',
        compiler_version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        cuda_include=str(cuda_include),cuda_runtime_header_sha256=sha(cuda_include/'cuda_runtime_api.h'),
        source_sha256={str(path):sha(path) for path in source_paths},
        code_sha256={name:sha(Path(__file__).parent/name) for name in ('analyze_r18_arrivals.py','output_model.py')},
        host_source_sha256=sha(source),host_binary_sha256=sha(binary),
        sass_sites=sites,dependency_refinement=projections,
        limits=['SASS sites confirm source order for this archived binary, not cross-warp ordering or remote visibility.',
                'WG leader threads128/256 directly emit only to destination0; destination1 requires threads168/296.',
                'cfg_b cluster1 needs128 consumer arrivals and must keep its separate four-warp observation.',
                'No point service price or new local heldout follows from these static routes.'])
    (output/'C-empty-arrival-routes.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    contract=dict(status='Conditional second-stage development interface; no GPU job or runnable runner config.',
        preceding_stage='First-output g0/g16/g32/g63 commit and mma-return localization over all five static boundary clusters and interior0/1.',
        unchanged_scope='Second-stage group is selected only after first-stage localization; no per-K sweep or posthoc frozen CTA expansion.',
        routes=routes,
        sender_fields=['invocation_id','cluster_id','source_rank','consumer_wg','physical_thread','destination_rank','release_group',
            'slot','generation','consumer_phase','release_pre_ns','release_post_ns'],
        acquire_fields=['invocation_id','cluster_id','destination_rank','acquire_group','stages','slot','generation','empty_wait_phase','acquire_end_ns'],
        identity='One raw invocation and physical cluster; acquire_group=release_group+S, same slot, next generation, producer empty-wait phase equals release consumer phase.',
        units='Use globaltimer ns to compare ranks. Preserve local clock64 separately; never compare native clock64 across SMs.',
        endpoints='Own sender pre/post enclose its original arrival emit, not remote visibility. acquire_end is after the complete producer_acquire call, including expect_tx.',
        graph_ports=graph['producer_observation_ports'],
        result='last_emit in [max(pre), min(max(post), acquire_end)]; remaining path is mixed visibility/polling/scheduling/expect_tx/clock overhead.',
        validity='Same group/slot/phase and all four actual senders per destination required; source/SASS brackets and matched observer-on/off validity must pass before interpretation.',
        preparation='Same binary and trace capacity/clearing in each paired arm; extra observer off for all warmups. History identities are declared and not pooled across protocols.',
        gates='Specific ns resolution/noise and observer validity gates remain unfrozen; freeze them before target sampling after independent observer controls. Do not borrow cfg_b cycle gates.',
        limits='No added arrival, named barrier, warpgroup synchronization, polling warp or reset of pipeline state. Main integration owns public implementation and scheduling.')
    (output/'observation-contract.json').write_text(json.dumps(contract,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cuda-include',required=True,type=Path)
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--previous-graph-dir',required=True,type=Path)
    args=parser.parse_args()
    report=analyze(args.input,args.output,args.cuda_include,args.previous_graph_dir,args.compiler)
    print(json.dumps(dict(host_layout_matches=report['host_matches_prediction_metadata'],routes=len(report['routes']))))
