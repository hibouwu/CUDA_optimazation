#!/usr/bin/env python3
"""Derive the bounded R13 cumulative-full-wait observer from its archived progress source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
BASE_SHA = '2bc91a493dcbbbbfa7f08504d6864f4038025bc20bac3254624b6f2cf57fe6e2'
VERSION = 'r13-early-wait-sum-v1'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source binding changed: ' + old[:100])
    return text.replace(old,new,1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(base, output):
    manifest = base/'source_hashes.json'
    if sha(manifest) != BASE_SHA:
        raise ValueError('wrong archived progress source')
    hashes = json.loads(manifest.read_text())
    for name, expected in hashes.items():
        if sha(base/name) != expected:
            raise ValueError('archived source changed: ' + name)
    if output.exists():
        raise ValueError('new output directory required')
    output.mkdir(parents=True)
    shutil.copytree(base/'source',output/'source')
    (output/'build').mkdir()
    source = output/'source'
    shutil.copyfile(HERE/'r13_wait_sum.cuh',source/'probes/r13_wait_sum.cuh')
    header = source/'overlay/v06_trace.hpp'
    if (header.read_bytes() != (source/'probes/r18_trace.hpp').read_bytes()
            or header.read_bytes() != (source/'probes/v06_trace.hpp').read_bytes()):
        raise ValueError('archived active trace and reference header differ')
    text = header.read_text()
    text = replace_once(text,'#include <cstdint>', '''#include <cstdint>
#include "r13_wait_sum.cuh"
#if !defined(R13_STAGE_PROGRESS) || !defined(R13_WAIT_SUM)
#error "This prepared trace requires the fixed progress wait-sum configuration"
#endif
__constant__ unsigned r13_wait_sum_observer;''')
    text = replace_once(text,'#ifdef R13_STAGE_PROGRESS\nstruct R13SparseConsumer {\n  uint64_t wait_begin',
                        '#ifdef R13_STAGE_PROGRESS\nstruct R13SparseConsumer {\n  R13WaitSum wait_sum;\n  uint64_t wait_begin')
    text = replace_once(text,'b[32]=s.mma_return; b[31]=0x52313302;', '''b[32]=s.mma_return; b[31]=0x52313309;
  b[34]=s.wait_sum.cycles; b[35]=s.wait_sum.count; b[36]=s.wait_sum.mask;
  b[37]=s.wait_sum.first_begin; b[38]=s.wait_sum.last_end;
  b[39]=s.wait_sum.invalid; b[40]=r13_wait_sum_observer;''')
    # Both the host and CUTLASS include this header. A single canonical file
    # prevents duplicate definitions from the host-local and overlay paths.
    (source/'probes/v06_trace.hpp').write_text(text)
    header.write_text('#pragma once\n#include "../probes/v06_trace.hpp"\n')
    (source/'probes/r18_trace.hpp').write_text(text)
    collective = source/'overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp'
    text = collective.read_text()
    text = replace_once(text,'    const bool r13_target = r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0;',
                        '    const bool r13_target = r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0;\n    const bool r13_sum_enabled = r13_target && r13_wait_sum_observer;')
    old = '''      if (r13_target && r13_k==17) r13_s->next_wait_begin=v06_clock();
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);'''
    text = replace_once(text,old,'''      if (r13_target && r13_k==17) r13_s->next_wait_begin=v06_clock();
      const bool r13_sum_selected = r13_sum_enabled && r13_k>=2 && r13_k<=16;
      uint64_t r13_wait_begin = 0;
      if (r13_sum_selected) r13_wait_begin = v06_clock();
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      if (r13_sum_selected) {
        uint64_t r13_wait_end = v06_clock();
        r13_s->wait_sum.record(r13_k, r13_wait_begin, r13_wait_end);
      }''')
    collective.write_text(text)
    probe = source/'probes/r18.cu'
    text = probe.read_text()
    text = replace_once(text,'kTraceVersion="r13-stage-progress-v1"',f'kTraceVersion="{VERSION}"')
    text = replace_once(text,'  std::string warmup_policy = "adaptive";',
                        '  unsigned wait_sum_observer = 0;\n  std::string warmup_policy = "common-sw1-16";')
    text = replace_once(text,'      else if(key=="--warmup-policy")warmup_policy=v;',
                        '      else if(key=="--wait-sum-observer")wait_sum_observer=std::stoul(v);\n      else if(key=="--warmup-policy")warmup_policy=v;')
    old = '    int ctas = int(grid.x * grid.y * grid.z);'
    text = replace_once(text,old,old+'''
    if (V06_CFG != 1 || (Mainloop::DispatchPolicy::Stages != 4 && Mainloop::DispatchPolicy::Stages != 6) ||
        o.m!=2816 || o.n!=1536 || o.k!=4096 || grid.x!=1 || grid.y!=132 || grid.z!=1 ||
        o.swizzle!=1 || std::string(resolved_raster)!="along_n" || o.raster_order!="heuristic" ||
        lda!=4096 || ldb!=1536 || ldd!=1536 || alloc_lda!=4104 || alloc_ldb!=1536 ||
        storage_m!=2816 || storage_n!=1536 || input_map_m!=2816 || input_map_n!=1536 ||
        o.zero_m!=-1 || o.zero_n!=-1 || o.input_mode!="dyadic" || o.seed!=17 || o.evict ||
        o.warmup_policy!="common-sw1-16" || o.wait_sum_observer>1 ||
        (o.mode!="warm" && o.mode!="setup"))
      throw std::runtime_error("wait-sum observer requires the declared fixed cfg_b pair");
    unsigned wait_sum_off=0;
    CUDA_CHECK(cudaMemcpyToSymbol(r13_wait_sum_observer,&wait_sum_off,sizeof(unsigned)));''')
    old = '        for (int i = 0; i < 16; ++i) warm.push_back(timed(warmup_launch_params));\n        float ms = timed(launch_params);'
    text = replace_once(text,old,'''        for (int i = 0; i < 16; ++i) warm.push_back(timed(warmup_launch_params));
        CUDA_CHECK(cudaMemcpyToSymbol(r13_wait_sum_observer,&o.wait_sum_observer,sizeof(unsigned)));
        float ms = timed(launch_params);''')
    # Both setup and target call declare the exact observer and history contract.
    needle = '<< ",\\\"warmup_fixed_calls\\\":" << (common_history ? 16 : 0)'
    if text.count(needle) != 2:
        raise ValueError('setup/call history fields changed')
    extra = '''
                << ",\\\"wait_sum_protocol\\\":\\\"r13-early-wait-sum-v1\\\""
                << ",\\\"wait_sum_observer\\\":" << o.wait_sum_observer
                << ",\\\"wait_sum_prefix_flag\\\":0,\\\"wait_sum_first_group\\\":2,\\\"wait_sum_last_group\\\":16"'''
    text = text.replace(needle,needle+extra)
    probe.write_text(text)
    commands = json.loads((base/'build/commands.json').read_text())
    selected = {}
    for name, command in commands.items():
        if 'stamped' in name:
            command.insert(command.index('-DR13_STAGE_PROGRESS'),'-DR13_WAIT_SUM')
            selected[name] = command
    if len(selected) != 2:
        raise ValueError('expected stage4 and stage6 build commands')
    (output/'build/commands.json').write_text(json.dumps(selected,indent=2)+'\n')
    shutil.copyfile(Path(__file__),source/'prepare_r13_wait_sum.py')
    (output/'source_hashes.json').write_text(json.dumps({str(p.relative_to(output)):sha(p)
        for p in sorted(source.rglob('*')) if p.is_file()},indent=2)+'\n')
    contract = dict(version=VERSION,base_source_manifest_sha256=BASE_SHA,
        source_files_verified=len(hashes),config='cfg_b',stages=[4,6],shape=[2816,1536,4096],
        observer_threads=[256],cta=0,cta_output_ordinal=1,WG_local_output_ordinal=0,
        groups=list(range(2,17)),expected_count=15,expected_mask=sum(1<<i for i in range(2,17)),
        prefix_calls=16,prefix_flag=0,source_buffer_relation='same pointers within each process',
        prefix_warning='stage4 and stage6 use their respective kernels; matching off/on history does not equate histories across stages.',
        target_flags=[0,1],trace_words_per_CTA=784,original_fields_preserved=True,
        new_fields={34:'sum_full_wait_cycles',35:'sample_count',36:'group_mask',37:'first_H2',
                    38:'last_B16',39:'invalid_bits',40:'target_observer_flag'},
        interval='H immediately before consumer_try_wait; B after consumer_wait returns; sum of 15 disjoint same-thread intervals.',
        writeback='Only after consumer work loop; collector arithmetic runs after each B and changes the observed path.',
        mandatory_before_gpu=['target CUDA12.9 build and SASS order/resource audit','strict decoder and numerical validation plan','local perturbation/null decision and estimand frozen before acquisition'],
        compiled=False,hardware_correctness=False,observation_qualified=False,prediction_qualified=False)
    (output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.base.resolve(),args.output.resolve())
