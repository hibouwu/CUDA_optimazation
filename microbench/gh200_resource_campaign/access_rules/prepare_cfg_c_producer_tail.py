#!/usr/bin/env python3
"""Prepare a fixed cfg_c producer load_tail-return observer; no device execution."""
import argparse,json,shutil
from pathlib import Path
import run_r18
import v06_run as common
from prepare_r18_first_output import replace_once
VERSION='cfg-c-producer-tail-v1'
HOOK='        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);'
HELPER='''
#ifdef R18_PRODUCER_TAIL
__device__ __constant__ unsigned r18_producer_tail_observer;
__device__ __forceinline__ void r18_producer_tail_stamp() {
  if (threadIdx.x != 0 || !r18_producer_tail_observer) return;
  uint64_t c=v06_clock(), n=v06_ns();
  auto b=v06_base(); b[14]=c; b[15]=n;
}
#endif
'''

def patch_source(source):
    probe=source/'probes/r18.cu';text=probe.read_text()
    text=replace_once(text,'#if defined(R13_RELEASE_WARPS) && defined(R13_STAGE_SPARSE)',
        '#if defined(R18_PRODUCER_TAIL)\nconstexpr const char* kTraceVersion="'+VERSION+'";\n#elif defined(R13_RELEASE_WARPS) && defined(R13_STAGE_SPARSE)')
    text=replace_once(text,'  unsigned release_warp_observer = 0;',
        '  unsigned producer_tail_observer = 0;\n  unsigned release_warp_observer = 0;')
    text=replace_once(text,'      else if(key=="--release-warp-observer")',
        '      else if(key=="--producer-tail-observer")producer_tail_observer=std::stoul(v);\n      else if(key=="--release-warp-observer")')
    text=replace_once(text,'    int ctas = int(grid.x * grid.y * grid.z);','''    int ctas = int(grid.x * grid.y * grid.z);
    if ((o.mode!="setup" && o.mode!="warm") || o.producer_tail_observer>1 ||
        V06_CFG!=2 || Mainloop::DispatchPolicy::Stages!=4 ||
        o.m!=2304 || o.n!=3072 || o.k!=4096 || grid.x!=66 || grid.y!=2 || grid.z!=1 ||
        storage_m!=2304 || storage_n!=3072 || input_map_m!=2304 || input_map_n!=3072 ||
        lda!=4096 || alloc_lda!=4096 || ldb!=3072 || alloc_ldb!=3072 || ldd!=3072 ||
        o.zero_m!=-1 || o.zero_n!=-1 || o.swizzle!=1 || o.input_mode!="dyadic" || o.seed!=17 ||
        o.history_mode!="off" || mirror_sources || o.evict || o.b_offset_bytes ||
        o.b_allocation_extra_bytes || o.work_swap55_91 || o.producer_delay_ns ||
        o.second_prefill || o.epilogue_defer || o.a_row_rotation || o.first_supply_events ||
        o.ready_observer || o.release_warp_observer || o.warmup_policy!="common-sw1-16")
      throw std::runtime_error("producer-tail observer requires fixed cfg_c geometry and history");
    unsigned producer_tail_off=0;
    CUDA_CHECK(cudaMemcpyToSymbol(r18_producer_tail_observer,&producer_tail_off,sizeof(unsigned)));''')
    text=replace_once(text,'    auto set_target_epilogue = [&]() {',
        '    auto set_target_epilogue = [&]() {\n      CUDA_CHECK(cudaMemcpyToSymbol(r18_producer_tail_observer,&o.producer_tail_observer,sizeof(unsigned)));')
    line='                << ",\\\"release_warp_observer\\\":" << o.release_warp_observer'
    if text.count(line)!=2:raise ValueError('two setup/call flag sites required')
    text=text.replace(line,'                << ",\\\"producer_tail_observer\\\":" << o.producer_tail_observer\n                << ",\\\"producer_tail_prefix_flag\\\":0"\n'+line)
    text=replace_once(text,'#if defined(R10_WORK_SWAP) || defined(R10_PRODUCER_DELAY) || defined(R13_EPILOGUE_ORDER) || defined(R13_A_ROW_ROTATION) || defined(R13_RELEASE_WARPS)',
        '#if defined(R18_PRODUCER_TAIL) || defined(R10_WORK_SWAP) || defined(R10_PRODUCER_DELAY) || defined(R13_EPILOGUE_ORDER) || defined(R13_A_ROW_ROTATION) || defined(R13_RELEASE_WARPS)')
    probe.write_text(text)
    header=source/'overlay/v06_trace.hpp';original=header.read_text()
    (source/'probes/v06_trace.hpp').write_text(original+HELPER)
    header.write_text('#pragma once\n#include "../probes/v06_trace.hpp"\n')
    kernel=source/'overlay'/common.COOP
    kernel.write_text(replace_once(kernel.read_text(),HOOK,HOOK+'\n        r18_producer_tail_stamp();'))

def prepare(root,cutlass):
    rows=[dict(id='cfg_c_producer_tail',config='cfg_c',m=2304,n=3072,k=4096,
        lda=4096,alloc_lda=4096,ldb=3072,alloc_ldb=3072,ldd=3072,
        storage_m=2304,storage_n=3072,zero_m=-1,zero_n=-1,swizzle=1,stages=4,
        input_mode='dyadic',seed=17,warmup_policy='common-sw1-16',variants=['dual'])]
    run_r18.prepare(root,cutlass,'r18',rows=rows,dual_clock=True)
    patch_source(root/'source')
    commands=json.loads((root/'build/commands.json').read_text())
    for command in commands.values():command.insert(1,'-DR18_PRODUCER_TAIL')
    common.write_json(root/'build/commands.json',commands)
    common.write_json(root/'observer_contract.json',dict(version=VERSION,writer_thread=0,fields=dict(cycles=14,ns=15),
        cta_words=1296,grid=[66,2,1],prefix_calls=16,prefix_flag=0,target_flags=[0,1],
        endpoint='Mainloop producer load_tail return; does not measure D visibility or kernel completion',
        status='Source preparation only. Target compile/SASS, decoder, full-D correctness and perturbation qualification pending.'))
    shutil.copy2(Path(__file__),root/'source/prepare_cfg_c_producer_tail.py')
    common.write_json(root/'source_hashes.json',{str(p.relative_to(root)):common.sha(p) for p in (root/'source').rglob('*') if p.is_file()})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cutlass',type=Path,required=True)
    args=parser.parse_args();prepare(args.output,args.cutlass)
