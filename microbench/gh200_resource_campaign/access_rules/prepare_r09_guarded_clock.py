#!/usr/bin/env python3
"""Prepare actual cfg_b CUTLASS clock observer builds. Never samples or qualifies."""
import argparse,hashlib,json,shutil
from pathlib import Path
import run_r18
import v06_run as common
ROOT=Path(__file__).resolve().parent

def once(text,old,new):
    if text.count(old)!=1:raise ValueError('source binding changed: '+old[:80])
    return text.replace(old,new,1)

def prepare(root,cutlass):
    request=json.loads((ROOT/'r09_clock_stamp_integration.json').read_text())
    for name,h in request['public_base']['read_only_source_sha256'].items():
        path=ROOT.parents[2]/name
        if common.sha(path)!=h:raise ValueError('public source changed: '+name)
    if common.sha(ROOT/'r09_clock_stamp.cuh')!=request['candidate']['header_sha256']:
        raise ValueError('candidate header changed')
    row=dict(id='r09_clock_compile_only',config='cfg_b',m=2816,n=1536,k=4096,
        lda=4096,ldb=1536,ldd=1536,alloc_lda=4104,alloc_ldb=1536,storage_m=2816,storage_n=1536,
        zero_m=-1,zero_n=-1,swizzle=1,stages=6,input_mode='dyadic',seed=17,
        warmup_policy='common-sw1-16',variants=['plain','dual'])
    run_r18.prepare(root,cutlass,'r18',rows=[row],dual_clock=True)
    source=root/'source'
    for name in ('r09_plain_prefix.cu','r09_plain_prefix.h','r09_plain_prefix_host.hpp'):
        shutil.copy2(ROOT/'probes'/name,source/'probes'/name)
    shutil.copy2(ROOT/'r09_clock_stamp.cuh',source/'probes/r09_clock_stamp.cuh')
    probe=source/'probes/r18.cu';text=probe.read_text()
    text=once(text,'constexpr int kTraceTileWords = 10;','constexpr int kTraceTileWords = 12;')
    text=once(text,'#if defined(R13_RELEASE_WARPS) && defined(R13_STAGE_SPARSE)',
        '''#if defined(R09_CLOCK_GUARDED)
constexpr const char* kTraceVersion="r09-guarded-dual12/v1";
#elif defined(R09_CLOCK_MATCHED)
constexpr const char* kTraceVersion="r09-matched-dual12/v1";
#elif defined(R09_CLOCK_PLAIN)
constexpr const char* kTraceVersion="none";
#elif defined(R13_RELEASE_WARPS) && defined(R13_STAGE_SPARSE)''')
    site='                << ",\\\"release_warp_observer\\\":" << o.release_warp_observer'
    if text.count(site)!=2:raise ValueError('setup/call metadata changed')
    text=text.replace(site,'''                << ",\\\"r09_clock_observer_protocol\\\":\\\"r09-clock-observer/v1\\\""
#ifdef R09_CLOCK_GUARDED
                << ",\\\"r09_clock_observer_arm\\\":\\\"guarded_bracket\\\""
#elif defined(R09_CLOCK_MATCHED)
                << ",\\\"r09_clock_observer_arm\\\":\\\"matched_dual\\\""
#else
                << ",\\\"r09_clock_observer_arm\\\":\\\"plain\\\""
#endif
'''+site)
    probe.write_text(text)
    header=source/'probes/v06_trace.hpp';text=header.read_text()
    text=once(text,'constexpr int V06TileWords = 10;','constexpr int V06TileWords = 12;')
    text=once(text,'#include <cstdint>','#include <cstdint>\n#include "r09_clock_stamp.cuh"')
    start=text.index('__device__ __forceinline__ void v06_stamp(int event, int tile) {')
    end=text.index('__device__ __forceinline__ void v06_final',start)
    body=text[start:end]
    body=once(body,'''  uint64_t c = v06_clock();
#ifdef R18_DUAL_CLOCK
  uint64_t ns = v06_ns();
#endif''','''  uint64_t c, ns, after=0;
#ifdef R09_CLOCK_GUARDED
  if (event==V06_FIRST_MMA || event==V06_MAIN_END) {
    auto stamp=r09_guarded_clock_stamp();
    c=stamp.clock_before; ns=stamp.globaltimer_ns; after=stamp.clock_after;
  } else { c=v06_clock(); ns=v06_ns(); }
#else
  c=v06_clock(); ns=v06_ns();
#endif''')
    body=once(body,'  out[6 + event] = ns;','  out[6 + event] = ns;\n  if (event==V06_FIRST_MMA || event==V06_MAIN_END) out[10+event]=after;')
    text=text[:start]+body+text[end:];header.write_text(text)
    (source/'overlay/v06_trace.hpp').write_text('#pragma once\n#include "../probes/v06_trace.hpp"\n')
    old=json.loads((root/'build/commands.json').read_text());commands={}
    for arm in ('plain','matched_dual','guarded_bracket'):
        command=list(old['cfg_b_s6_plain' if arm=='plain' else 'cfg_b_s6_dual'])
        command[1:1]=['-DR09_PLAIN_PREFIX_HOST','-DR09_CLOCK_'+{'plain':'PLAIN','matched_dual':'MATCHED','guarded_bracket':'GUARDED'}[arm]]
        if arm=='plain':command.insert(1,'-DR18_MATCH_DUAL_LAYOUT')
        command[-1]='build/cfg_b_'+arm;command.append('-ldl');commands['cfg_b_'+arm]=command
    commands['plain_prefix']=['nvcc','-std=c++17','-O3','-DNDEBUG','-shared','-Xcompiler','-fPIC,-fvisibility=hidden','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','--ptxas-options=-v','-DV06_CFG=1','-DR18_STAGE_COUNT=6','-Isource/probes','-Isource/cutlass/include','-Isource/cutlass/tools/util/include','source/probes/r09_plain_prefix.cu','-o','build/libplain_prefix.so']
    common.write_json(root/'build/commands.json',commands)
    common.write_json(root/'interface.json',request)
    shutil.copy2(Path(__file__),source/'prepare_r09_guarded_clock.py')
    common.write_json(root/'source_hashes.json',{str(p.relative_to(root)):common.sha(p) for p in source.rglob('*') if p.is_file()})
    common.write_json(root/'run_config.json',dict(scope='CPU compile only; reader and runtime qualification incomplete',arms=list(commands),tile_words=12,cta_words=1552,scratch_bytes=1638912))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cutlass',type=Path,required=True)
    a=p.parse_args();prepare(a.output,a.cutlass)
