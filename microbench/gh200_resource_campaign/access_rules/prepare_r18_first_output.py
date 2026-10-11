#!/usr/bin/env python3
"""Prepare the fixed cfg_a first-output probe; no compilation or sampling here."""
import argparse
import json
from pathlib import Path
import shutil
import run_r18
import v06_run as common

ROOT=Path(__file__).resolve().parent
VERSION='r18-first-output-v1'


def replace_once(text, old, new):
    if text.count(old)!=1:raise ValueError('source port changed: '+old[:100])
    return text.replace(old,new,1)


def patch_source(source):
    probe=source/'probes/r18.cu';text=probe.read_text()
    text=replace_once(text,'#include <cute/tensor.hpp>',
        '__device__ __constant__ unsigned r18_first_output_observer;\n#include <cute/tensor.hpp>')
    text=replace_once(text,'#if defined(R13_RELEASE_WARPS) && defined(R13_STAGE_SPARSE)',
        '#if defined(R18_FIRST_OUTPUT)\nconstexpr const char* kTraceVersion="'+VERSION+'";\n#elif defined(R13_RELEASE_WARPS) && defined(R13_STAGE_SPARSE)')
    text=replace_once(text,'constexpr int kTraceWords = 16 + 2 * 64 * kTraceTileWords;',
        'constexpr int kTraceWords = 16 + 2 * 64 * kTraceTileWords + 22;')
    text=replace_once(text,'  unsigned release_warp_observer = 0;',
        '  unsigned first_output_observer = 0;\n  unsigned release_warp_observer = 0;')
    text=replace_once(text,'      else if(key=="--release-warp-observer")',
        '      else if(key=="--first-output-observer")first_output_observer=std::stoul(v);\n      else if(key=="--release-warp-observer")')
    text=replace_once(text,'    int ctas = int(grid.x * grid.y * grid.z);',
        '''    int ctas = int(grid.x * grid.y * grid.z);
    if ((o.mode!="setup" && o.mode!="warm" && o.mode!="check-first-output") ||
        o.first_output_observer>1 || V06_CFG!=0 || Mainloop::DispatchPolicy::Stages!=6 ||
        o.m!=960 || o.n!=2816 || o.k!=4096 || grid.x!=132 || grid.y!=1 || grid.z!=1 ||
        o.swizzle!=1 || storage_m!=1024 || storage_n!=2816 || o.zero_m!=960 || o.zero_n!=-1 ||
        (input_map_m!=960 && input_map_m!=1024) || input_map_n!=2816 ||
        lda!=4096 || ldb!=2816 || ldd!=2816 || alloc_lda!=4096 || alloc_ldb!=2816 ||
        o.input_mode!="dyadic" || o.seed!=17 || o.history_mode!="off" || mirror_sources ||
        o.evict || o.b_offset_bytes || o.b_allocation_extra_bytes || o.work_swap55_91 ||
        o.producer_delay_ns || o.second_prefill || o.epilogue_defer || o.a_row_rotation ||
        o.first_supply_events || o.ready_observer || o.release_warp_observer ||
        o.warmup_policy!="common-sw1-16")
      throw std::runtime_error("first-output observer requires fixed cfg_a control geometry/history");
    unsigned first_output_off=0;
    CUDA_CHECK(cudaMemcpyToSymbol(r18_first_output_observer,&first_output_off,sizeof(unsigned)));''')
    text=replace_once(text,'    auto set_target_epilogue = [&]() {',
        '    auto set_target_epilogue = [&]() {\n      CUDA_CHECK(cudaMemcpyToSymbol(r18_first_output_observer,&o.first_output_observer,sizeof(unsigned)));')
    # Both setup and call identity carry the same flag; no silent reuse of R13 fields.
    line='                << ",\\\"release_warp_observer\\\":" << o.release_warp_observer'
    if text.count(line)!=2:raise ValueError('setup/call identity sites changed')
    text=text.replace(line,'                << ",\\\"first_output_observer\\\":" << o.first_output_observer\n                << ",\\\"first_output_warmup\\\":0"\n'+line)
    text=replace_once(text,'#if defined(R10_WORK_SWAP) || defined(R10_PRODUCER_DELAY) || defined(R13_EPILOGUE_ORDER) || defined(R13_A_ROW_ROTATION) || defined(R13_RELEASE_WARPS)',
        '#if defined(R18_FIRST_OUTPUT) || defined(R10_WORK_SWAP) || defined(R10_PRODUCER_DELAY) || defined(R13_EPILOGUE_ORDER) || defined(R13_A_ROW_ROTATION) || defined(R13_RELEASE_WARPS)')
    text=replace_once(text,'if (o.mode == "check-work-swap" ||',
        'if (o.mode == "check-first-output" || o.mode == "check-work-swap" ||')
    text=replace_once(text,'      auto set_check_flag = [&](unsigned value) {',
        '      auto set_check_flag = [&](unsigned value) {\n        if (o.mode == "check-first-output") {\n          CUDA_CHECK(cudaMemcpyToSymbol(r18_first_output_observer,&value,sizeof(unsigned)));\n          return;\n        }')
    text=replace_once(text,'(o.mode == "check-release-warps" ? "release_warps_check"',
        '(o.mode == "check-first-output" ? "first_output_check" : o.mode == "check-release-warps" ? "release_warps_check"')
    probe.write_text(text)
    header=source/'overlay/v06_trace.hpp';text=header.read_text()
    text=replace_once(text,'V06Head + 2 * V06Tiles * V06TileWords;',
        'V06Head + 2 * V06Tiles * V06TileWords + 22;')
    # Helpers need v06_clock/ns/base, and precede the collective using them.
    text=replace_once(text,'#ifdef R13_FIRST_SUPPLY',
        '#include "r18_first_output.hpp"\n\n#ifdef R13_FIRST_SUPPLY')
    (source/'probes/v06_trace.hpp').write_text(text)
    header.write_text('#pragma once\n#include "../probes/v06_trace.hpp"\n')
    shutil.copy2(ROOT/'probes/r18_first_output.hpp',source/'probes/r18_first_output.hpp')
    path=source/'overlay'/common.MAINLOOP;text=path.read_text()
    text=replace_once(text,'Params const& mainloop_params, int v06_tile_seq = 0) {',
        'Params const& mainloop_params, int v06_tile_seq = 0, R18FirstOutput* first_out = nullptr) {')
    start=text.index('  mma(MainloopPipeline pipeline,');end=text.index('  /// Perform a Consumer Epilogue',start)
    body=text[start:end];needle='      warpgroup_commit_batch();'
    if body.count(needle)!=3:raise ValueError('three commit source sites required')
    chunks=body.split(needle)
    body=chunks[0]+needle+'\n      r18_first_stamp<0>(first_out,v06_tile_seq);'+chunks[1]+needle+chunks[2]+needle+'''
      if (k_tile_count==48) r18_first_stamp<1>(first_out,v06_tile_seq);
      if (k_tile_count==32) r18_first_stamp<2>(first_out,v06_tile_seq);
      if (k_tile_count==1) r18_first_stamp<3>(first_out,v06_tile_seq);'''+chunks[3]
    text=text[:start]+body+text[end:];path.write_text(text)
    path=source/'overlay'/common.COOP;text=path.read_text()
    text=replace_once(text,'    int v06_tile_seq = -1;',
        '    int v06_tile_seq = -1;\n    R18FirstOutput first_output;')
    text=replace_once(text,'            params.mainloop, v06_tile_seq\n          );',
        '            params.mainloop, v06_tile_seq, &first_output\n          );\n          r18_first_stamp<4>(&first_output,v06_tile_seq);')
    text=replace_once(text,'      v06_final(v06_tile_seq + 1);',
        '      v06_final(v06_tile_seq + 1);\n      r18_first_flush(first_output);')
    path.write_text(text)


def prepare(root, cutlass):
    rows=[]
    for extent in (960,1024):
        rows.append(dict(id=f'r18_first_map{extent}',config='cfg_a',m=960,n=2816,k=4096,
            lda=4096,ldb=2816,ldd=2816,storage_m=1024,storage_n=2816,zero_m=960,zero_n=-1,
            input_map_m=extent,input_map_n=2816,swizzle=1,stages=6,input_mode='dyadic',seed=17,
            warmup_policy='common-sw1-16',variants=['dual']))
    run_r18.prepare(root,cutlass,'r18',rows=rows,dual_clock=True)
    patch_source(root/'source')
    commands=json.loads((root/'build/commands.json').read_text())
    for command in commands.values():command.insert(1,'-DR18_FIRST_OUTPUT')
    common.write_json(root/'build/commands.json',commands)
    common.write_json(root/'run_config.json',dict(trace_version=VERSION,role_words=11,extra_cta_words=22,
        cta_words=1318,grid=[132,1,1],timestamp_payload_bytes=1920,extra_allocation_bytes=23232,
        allocation_bytes=1391808,history='same binary, observer off for 16 prefix calls',
        status='Prepared source only; build, numerical checks, SASS and observer qualification pending'))
    shutil.copy2(Path(__file__),root/'source/prepare_r18_first_output.py')
    common.write_json(root/'source_hashes.json',{str(p.relative_to(root)):common.sha(p)
        for p in (root/'source').rglob('*') if p.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cutlass',required=True,type=Path)
    args=parser.parse_args();prepare(args.output,args.cutlass)
