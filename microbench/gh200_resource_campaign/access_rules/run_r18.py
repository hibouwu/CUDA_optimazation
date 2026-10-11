#!/usr/bin/env python3
"""R18 matched physical grids; R19 reuses this actual-work-coordinate probe."""
import argparse,fcntl,gzip,json,os,random,shutil,statistics,subprocess,time
from pathlib import Path
import v06_run as common
import v06_model as model

ROOT=Path(__file__).resolve().parent
CONFIGS={'cfg_a':0,'cfg_b':1,'cfg_c':2,'cfg_a1':3,'cfg_c1':4}
common.EXPECTED_HGMMA.update(cfg_a1=8,cfg_c1=16)
model.CONFIGS.update(cfg_a1=dict(index=3,tile=(128,128,64),cluster=(1,1),schedule='cooperative'),
                     cfg_c1=dict(index=4,tile=(256,128,64),cluster=(1,1),schedule='cooperative'))


def cases(family='r18'):
    rows=[]
    if family=='b01_cache':
        return [dict(id=f'gemm_sw{sw}_'+('evict_prepared' if ev else 'repeat'),config='cfg_a',group='cache',kind='ordinary',
                     m=4096,n=4096,k=2048,lda=2048,ldb=4096,ldd=4096,storage_m=4096,storage_n=4096,zero_m=-1,zero_n=-1,swizzle=sw,evict=ev)
                for sw in [1,8] for ev in [0,1]]
    if family=='r19':
        for label,m,n,k in [('near_short',1408,1408,1024),('near_long',1408,1408,8192),
                            ('multi_short',4096,4096,1024),('multi_long',4096,4096,8192)]:
            for swizzle in [1,8]:
                rows.append(dict(id=f'cfg_b_{label}_sw{swizzle}',config='cfg_b',group=label,kind='ordinary',
                    m=m,n=n,k=k,lda=k,ldb=n,ldd=n,storage_m=m,storage_n=n,zero_m=-1,zero_n=-1,swizzle=swizzle))
        return rows
    for config in ['cfg_a','cfg_c']:
        for k in [1024,8192]:
            for kind,extent in [('ordinary',1536),('oob',1408),('explicit_zero',1536),('partial_even',1504),('partial_odd',1376)]:
                m,n=(extent,2048) if config=='cfg_a' else (3072,extent)
                pm,pn=(1536,2048) if config=='cfg_a' else (3072,1536)
                rows.append(dict(id=f'{config}_{kind}_k{k}',config=config,group=f'{config}_k{k}',kind=kind,
                  m=m,n=n,k=k,lda=k,ldb=pn,ldd=pn,storage_m=pm,storage_n=pn,
                  zero_m=1408 if kind=='explicit_zero' and config=='cfg_a' else -1,
                  zero_n=1408 if kind=='explicit_zero' and config=='cfg_c' else -1,swizzle=1))
        for kind,extent in [('oob',1408),('partial_odd',1376)]:
            m,n=(extent,2048) if config=='cfg_a' else (3072,extent)
            pm,pn=(1536,2048) if config=='cfg_a' else (3072,1536)
            rows.append(dict(id=f'{config}1_{kind}_k8192',config=config+'1',group=f'{config}_k8192',kind=kind,
                 m=m,n=n,k=8192,lda=8192,ldb=pn,ldd=pn,storage_m=pm,storage_n=pn,zero_m=-1,zero_n=-1,swizzle=1))
    if family=='r18_lite':
        rows=[r for r in rows if r['config'] in ['cfg_a','cfg_c'] and (r['kind']=='ordinary' or (r['config']=='cfg_a' and r['kind']=='partial_even') or (r['config']=='cfg_c' and r['kind']=='partial_odd'))]
    return rows



def make_stage_sparse_overlay(source, progress=False, handoff=False):
    """Only cfg_b: CTA0, output 1, Ktile16 and its next same-slot producer use."""
    path=source/'overlay'/common.MAINLOOP
    text=path.read_text()
    if not progress:
        begin=text.index('  /// Producer Perspective')
        end=text.index('  /// Perform a Producer Epilogue',begin)
        load=text[begin:end]
        load=load.replace('TensorStorage& shared_tensors) {',
            'TensorStorage& shared_tensors, int r13_output_seq = -1) {',1)
        site='    if (lane_predicate) {\n'
        load=common.insert_once(load,site,
            '      const bool r13_target = r13_sparse_cta() && r13_output_seq == 1;\n'
            '      const int r13_initial_k = k_tile_count;\n'
            '      R13SparseProducer r13_p;\n',after=True)
        site='        pipeline.producer_acquire(smem_pipe_write);\n'
        load=common.insert_once(load,site,
            '        const int r13_k = r13_initial_k-k_tile_count;\n'
            '        if (r13_target && r13_k==16) { r13_p.state=smem_pipe_write.index() | (smem_pipe_write.phase()<<3); r13_p.acquire_begin=v06_clock(); }\n'
            '        if (r13_target && r13_k==16+DispatchPolicy::Stages) { r13_p.state |= (smem_pipe_write.index()<<4) | (smem_pipe_write.phase()<<7); r13_p.refill_begin=v06_clock(); }\n')
        load=common.insert_once(load,site,
            '        if (r13_target && r13_k==16) r13_p.acquire_end=v06_clock();\n'
            '        if (r13_target && r13_k==16+DispatchPolicy::Stages) r13_p.refill_end=v06_clock();\n',after=True)
        site='        ++k_tile_iter;\n'
        load=common.insert_once(load,site,
            '        if (r13_target && r13_k==16) r13_p.issue_end=v06_clock();\n'
            '        if (r13_target && r13_k==16+DispatchPolicy::Stages) r13_p.refill_issue_end=v06_clock();\n')
        # Deferred writes occur after the producer has issued this entire output.
        site='      }\n    }\n  }\n'
        if load.count(site)!=1:raise ValueError('sparse producer closing site mismatch')
        load=load.replace(site,'      }\n      if (r13_target) r13_sparse_producer_flush(r13_p);\n    }\n  }\n')
        text=text[:begin]+load+text[end:]
    signature='      Params const& mainloop_params, int v06_tile_seq = 0) {\n'
    text=common.insert_once(text,signature,
        '    const int r13_initial_k = k_tile_count;\n'
        '    const bool r13_target = r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0;\n',after=True)
    text=text.replace(signature,'      Params const& mainloop_params, int v06_tile_seq = 0, R13SparseConsumer* r13_s = nullptr) {\n',1)
    text=text.replace('      v06_stamp(V06_FIRST_MMA, v06_tile_seq);\n',
        '      if (r13_target) r13_s->first_mma=v06_clock();\n',1)
    if handoff:
        begin=text.index('    // Prologue GMMAs');end=text.index('    // Mainloop GMMAs',begin)
        prologue=text[begin:end].replace('      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);\n',
            '#ifdef R13_STAGE_HANDOFF\n'
            '      if (r13_target) r13_s->first_wait_begin=v06_clock();\n'
            '#endif\n      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);\n',1)
        text=text[:begin]+prologue+text[end:]
    begin=text.index('    // Mainloop GMMAs')
    end=text.index('  /// Perform a Consumer Epilogue',begin)
    mma=text[begin:end]
    site='      pipeline.consumer_wait(smem_pipe_read, barrier_token);\n'
    # Include try_wait in the interval: it can already return WaitDone.
    before='      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);\n'
    if progress:
        mma=common.insert_once(mma,before,
            '      const int r13_k = r13_initial_k-k_tile_count;\n'
            '      if (r13_target && r13_k==16) r13_s->wait_begin=v06_clock();\n'
            '      if (r13_target && r13_k==17) r13_s->next_wait_begin=v06_clock();\n')
        mma=common.insert_once(mma,'      warpgroup_commit_batch();\n',
            '      if (r13_target && r13_k==1) r13_s->commit1=v06_clock();\n'
            '#ifdef R13_OUTPUT_BACKGROUND\n'
            '      if (r13_target && r13_k==1) r13_s->commit1_ns=v06_ns();\n'
            '#endif\n'
            '      if (r13_target && r13_k==16) r13_s->commit16=v06_clock();\n'
            '#ifdef R13_OUTPUT_BACKGROUND\n'
            '      if (r13_target && r13_k==16) r13_s->commit16_ns=v06_ns();\n'
            '#endif\n'
            '      if (r13_target && r13_k==17) r13_s->commit17=v06_clock();\n'
            '      if (r13_target && r13_k==63) r13_s->commit63=v06_clock();\n',after=True)
        mma=common.insert_once(mma,'      warpgroup_wait<K_PIPE_MMAS>();\n',
            '      if (r13_target && r13_k==16) r13_s->retire_end=v06_clock();\n',after=True)
    else:
        mma=common.insert_once(mma,before,
            '      const int r13_k = r13_initial_k-k_tile_count;\n'
            '      if (r13_target && r13_k==16) { r13_s->read_slot=smem_pipe_read.index(); r13_s->wait_begin=v06_clock(); }\n')
        mma=common.insert_once(mma,site,'      if (r13_target && r13_k==16) r13_s->wait_end=v06_clock();\n',after=True)
        mma=common.insert_once(mma,'      warpgroup_commit_batch();\n',
            '      if (r13_target && r13_k==16) r13_s->commit_end=v06_clock();\n',after=True)
        mma=common.insert_once(mma,'      warpgroup_wait<K_PIPE_MMAS>();\n',
            '      if (r13_target && r13_k==16) r13_s->retire_end=v06_clock();\n',after=True)
        site='      pipeline.consumer_release(smem_pipe_release);\n'
        mma=common.insert_once(mma,site,
            '      if (r13_target && r13_k==17) { r13_s->release_slot=smem_pipe_release.index(); r13_s->release_phase=smem_pipe_release.phase(); r13_s->release_begin=v06_clock(); }\n')
        mma=common.insert_once(mma,site,'      if (r13_target && r13_k==17) r13_s->release_end=v06_clock();\n',after=True)
    text=text[:begin]+mma+text[end:]
    path.write_text(text)
    path=source/'overlay'/common.PING;text=path.read_text()
    if not progress:
        begin=text.index('      // Mainloop Producer Warp')
        end=text.index('else if (producer_warp_role == ProducerWarpRole::MainloopAux)',begin)
        producer=text[begin:end]
        site='            shared_storage.tensors.mainloop\n          );'
        if producer.count(site)!=1:raise ValueError('sparse producer call mismatch')
        producer=producer.replace(site,'            shared_storage.tensors.mainloop, v06_tile_seq\n          );')
        text=text[:begin]+producer+text[end:]
    begin=text.index('    else if (warp_group_role == WarpGroupRole::Consumer0')
    consumer=text[begin:]
    consumer=common.insert_once(consumer,'      while (work_tile_info.is_valid()) {\n',
        '      R13SparseConsumer r13_s;\n')
    consumer=consumer.replace('          params.mainloop, v06_tile_seq\n',
        '          params.mainloop, v06_tile_seq, &r13_s\n',1)
    if handoff:
        site='        math_wg_order_barrier.wait();\n'
        consumer=consumer.replace(site,
            '#ifdef R13_STAGE_HANDOFF\n'
            '        if (r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0) r13_s.math_wait_begin=v06_clock();\n'
            '#endif\n'+site+
            '#ifdef R13_STAGE_HANDOFF\n'
            '        if (r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0) r13_s.math_wait_end=v06_clock();\n'
            '#endif\n',1)
        consumer=common.insert_once(consumer,'        // Cue for next Math WG\'s MMA to start',
            '#ifdef R13_STAGE_HANDOFF\n'
            '        if (r13_sparse_cta() && threadIdx.x==128 && v06_tile_seq==0) r13_s.sender_mma_return=v06_clock();\n'
            '#endif\n')
        consumer=common.insert_once(consumer,'      v06_final(v06_tile_seq + 1);\n',
            '#ifdef R13_STAGE_HANDOFF\n'
            '      if (r13_sparse_cta() && threadIdx.x==128) {\n'
            '        unsigned sm; asm volatile("mov.u32 %0, %%smid;":"=r"(sm));\n'
            '        v06_base()[34]=r13_s.sender_mma_return; v06_base()[38]=uint64_t(sm)+1;\n'
            '      }\n#endif\n')
    consumer=common.insert_once(consumer,'        // Cue for next Math WG\'s MMA to start',
        '        if (r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0) r13_s.mma_return=v06_clock();\n'
        '#ifdef R13_OUTPUT_BACKGROUND\n'
        '        if (threadIdx.x==256 && v06_tile_seq==0) r13_s.receiver_return_ns=v06_ns();\n'
        '#endif\n')
    if not progress:
        consumer=common.insert_once(consumer,'        collective_mainloop.mma_tail(\n',
            '        if (r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0) r13_s.tail_begin=v06_clock();\n')
    consumer=consumer.replace('          v06_stamp(V06_MAIN_END, v06_tile_seq);\n',
        '        if (r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0) r13_s.main_end=v06_clock();\n',1)
    consumer=common.insert_once(consumer,'      v06_final(v06_tile_seq + 1);\n',
        '      if (r13_sparse_cta() && threadIdx.x==256) r13_sparse_consumer_flush(r13_s);\n')
    # Both normal/deferred progress binaries collect the same two E0 markers.
    consumer=consumer.replace('        v06_stamp(V06_EPI_PERMIT, v06_tile_seq);\n',
        '#ifdef R13_EPILOGUE_ORDER\n'
        '        if (r13_sparse_cta() && threadIdx.x==128 && v06_tile_seq==0) r13_s.sender_epi_begin=v06_clock();\n'
        '#endif\n'
        '#ifdef R13_OUTPUT_BACKGROUND\n'
        '        if (threadIdx.x==128 && v06_tile_seq==0) r13_s.sender_epi_begin_ns=v06_ns();\n'
        '#endif\n        v06_stamp(V06_EPI_PERMIT, v06_tile_seq);\n',1)
    consumer=consumer.replace('        v06_stamp(V06_EPI_DONE, v06_tile_seq);\n',
        '#ifdef R13_EPILOGUE_ORDER\n'
        '        if (r13_sparse_cta() && threadIdx.x==128 && v06_tile_seq==0) r13_s.sender_epi_end=v06_clock();\n'
        '#endif\n'
        '#ifdef R13_OUTPUT_BACKGROUND\n'
        '        if (threadIdx.x==128 && v06_tile_seq==0) r13_s.sender_epi_end_ns=v06_ns();\n'
        '#endif\n        v06_stamp(V06_EPI_DONE, v06_tile_seq);\n',1)
    consumer=common.insert_once(consumer,'      v06_final(v06_tile_seq + 1);\n',
        '#ifdef R13_EPILOGUE_ORDER\n'
        '      if (r13_sparse_cta() && threadIdx.x==128) {\n'
        '        unsigned sm; asm volatile("mov.u32 %0, %%smid;":"=r"(sm));\n'
        '        auto b=v06_base(); b[41]=r13_s.sender_epi_begin; b[42]=r13_s.sender_epi_end; b[43]=uint64_t(sm)+1;\n'
        '#ifdef R13_SECOND_PREFILL\n'
        '        r13_first_tail_flush(r13_s);\n'
        '#endif\n'
        '      }\n#endif\n')
    consumer=common.insert_once(consumer,'      v06_final(v06_tile_seq + 1);\n',
        '#ifdef R13_OUTPUT_BACKGROUND\n'
        '      if (threadIdx.x==128) { v06_base()[46]=r13_s.sender_epi_begin_ns; v06_base()[47]=r13_s.sender_epi_end_ns; }\n'
        '      if (threadIdx.x==256) v06_base()[48]=r13_s.receiver_return_ns;\n'
        '#endif\n')
    path.write_text(text[:begin]+consumer)


def make_first_supply_overlay(source):
    """CTA55 first output: issued A/B pair and entry to the first consumer wait."""
    path=source/'overlay'/common.MAINLOOP;text=path.read_text()
    start=text.index('  /// Producer Perspective');end=text.index('  /// Perform a Producer Epilogue',start)
    load=text[start:end].replace('TensorStorage& shared_tensors) {',
        'TensorStorage& shared_tensors, int r13_output_seq = -1) {',1)
    load=common.insert_once(load,'    if (lane_predicate) {\n',
        '      const int r13_first_k = k_tile_count;\n'
        '      const bool r13_target = r13_first_target(r13_output_seq);\n',after=True)
    load=common.insert_once(load,'        ++k_tile_iter;\n',
        '        if (r13_target && k_tile_count==r13_first_k) r13_first_issue_stamp();\n')
    text=text[:start]+load+text[end:]
    start=text.index('    // Prologue GMMAs');end=text.index('    // Mainloop GMMAs',start)
    prologue=text[start:end]
    site='      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);\n'
    # Only the initial block, not the later k_tile_prologue loop.
    pos=prologue.index(site)
    prologue=prologue[:pos]+('      const bool r13_target = threadIdx.x==128 && r13_first_target(v06_tile_seq);\n'
        '      uint64_t r13_wait_c=0, r13_wait_ns=0;\n'
        '      if (r13_target) { r13_wait_c=v06_clock(); r13_wait_ns=v06_ns(); }\n')+prologue[pos:]
    prologue=common.insert_once(prologue,'      v06_stamp(V06_FIRST_MMA, v06_tile_seq);\n',
        '      if (r13_target) { r10_producer_delay_trace[6]=r13_wait_c; r10_producer_delay_trace[7]=r13_wait_ns; }\n',after=True)
    path.write_text(text[:start]+prologue+text[end:])
    path=source/'overlay'/common.PING;text=path.read_text()
    start=text.index('      // Mainloop Producer Warp');end=text.index('else if (producer_warp_role == ProducerWarpRole::MainloopAux)',start)
    producer=text[start:end]
    site='            shared_storage.tensors.mainloop\n          );'
    if producer.count(site)!=1:raise ValueError('first-supply producer call site changed')
    producer=producer.replace(site,'            shared_storage.tensors.mainloop, v06_tile_seq\n          );')
    path.write_text(text[:start]+producer+text[end:])


def make_second_prefill_overlay(source):
    """Change both plain and progress; only progress reads diagnostic endpoints."""
    path=source/'cutlass/include'/common.MAINLOOP
    text=path.read_text()
    end=text.index('  /// Perform a Consumer Epilogue')
    text=common.insert_once(text[:end],'    // Prologue GMMAs\n',
        '#ifdef R13_SECOND_PREFILL\n'
        '    if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 &&\n'
        '        threadIdx.x>=256 && threadIdx.x<384 &&\n'
        '        smem_pipe_read.count()==64 && k_tile_count==64) {\n'
        '#ifdef R13_STAGE_PROGRESS\n'
        '      if (threadIdx.x==256) {\n'
        '        r13_s->prefill_begin_c=v06_clock(); r13_s->prefill_begin_ns=v06_ns();\n'
        '        r13_s->prefill_waited=0;\n'
        '      }\n'
        '#endif\n'
        '      if (::r13_second_prefill) {\n'
        '        auto prefill_state=smem_pipe_read;\n'
        '        for (int slot=0; slot<DispatchPolicy::Stages; ++slot) {\n'
        '          pipeline.consumer_wait(prefill_state);\n'
        '          ++prefill_state;\n'
        '        }\n'
        '#ifdef R13_STAGE_PROGRESS\n'
        '        if (threadIdx.x==256) r13_s->prefill_waited=DispatchPolicy::Stages;\n'
        '#endif\n'
        '      }\n'
        '#ifdef R13_STAGE_PROGRESS\n'
        '      if (threadIdx.x==256) {\n'
        '        r13_s->prefill_end_c=v06_clock(); r13_s->prefill_end_ns=v06_ns();\n'
        '      }\n'
        '#endif\n'
        '    }\n'
        '#endif\n')+text[end:]
    path.write_text(text)


def make_a_row_rotation(source):
    """Change only A's TMA source coordinate; scheduler/B/D retain their coordinates."""
    path=source/'cutlass/include'/common.MAINLOOP
    text=path.read_text()
    site='      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);'
    if text.count(site)!=1:raise ValueError('A rotation source partition site changed')
    text=text.replace(site,
        '#ifdef R13_A_ROW_ROTATION\n'
        '      int r13_m=int(m_coord);\n'
        '      int r13_source_m=(r13_m/11)*11+(r13_m%11+int(::r13_a_row_rotation))%11;\n'
        '      Tensor gA = gA_mkl(_,_,r13_source_m,_,l_coord);\n'
        '#else\n'+site+'\n#endif',1)
    path.write_text(text)


def prepare(root,cutlass,family,rows=None,dual_clock=False,stage_sparse=False,stage_progress=False,stage_handoff=False):
    stage_progress=stage_progress or stage_handoff
    rows=cases(family) if rows is None else rows
    root.mkdir(parents=True,exist_ok=False);source=root/'source';(source/'probes').mkdir(parents=True);(root/'build').mkdir()
    for name in ['run_r18.py','run_r18_lite.py','run_r19.py','run_b01_cache.py','analyze_r18.py','v06_run.py','v06_model.py','v08_model.py']:
        shutil.copy2(ROOT/name,source/name)
    for name in ['r18.cu','r18_trace.hpp','gaps_common.hpp','r00_common.hpp']:
        shutil.copy2(ROOT/'probes'/name,source/'probes'/name)
    shutil.copy2(ROOT/'probes/r18_trace.hpp',source/'probes/v06_trace.hpp')
    for part in ['include','tools/util/include']:shutil.copytree(cutlass/part,source/'cutlass'/part)
    a_rotation=any('a_row_rotation' in row for row in rows)
    if a_rotation:
        if not dual_clock or stage_sparse or stage_progress:
            raise ValueError('A rotation uses plain/dual full-CTA observation')
        for row in rows:
            if ((row['config'],row['m'],row['n'],row['k'],row.get('stages'),row.get('a_row_rotation')) not in
                    [('cfg_b',2816,1536,4096,4,0),('cfg_b',2816,1536,4096,4,1)] or
                    row.get('epilogue_defer')!=2 or row.get('input_mode')!='dyadic' or
                    row.get('raster_order')!='along_n' or row.get('variants')!=['plain','dual']):
                raise ValueError('A rotation is only the declared stage4 paired control')
        make_a_row_rotation(source)
    work_swap = any('work_swap55_91' in row for row in rows)
    if work_swap:
        # Patch the shared scheduler source, so plain and dual use the same map.
        path = source/'cutlass/include/cutlass/gemm/kernel/static_tile_scheduler.hpp'
        text = path.read_text()
        site = '    total_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);\n'
        text = common.insert_once(text, site,
            '#ifdef R10_WORK_SWAP\n'
            '    if (::r10_work_swap55_91) {\n'
            '      if (current_work_linear_idx_ == 55) current_work_linear_idx_ = 91;\n'
            '      else if (current_work_linear_idx_ == 91) current_work_linear_idx_ = 55;\n'
            '    }\n'
            '#endif\n')
        path.write_text(text)
    producer_delay = any('producer_delay_ns' in row for row in rows)
    if producer_delay:
        path = source/'cutlass/include'/common.PING
        text = path.read_text()
        start = text.index('      // Mainloop Producer Warp')
        end = text.index('else if (producer_warp_role == ProducerWarpRole::MainloopAux)',start)
        producer = common.insert_once(text[start:end], '        while (work_tile_info.is_valid()) {\n',
            '#ifdef R10_PRODUCER_DELAY\n'
            '        ::r10_delay_first_producer();\n'
            '#endif\n')
        path.write_text(text[:start]+producer+text[end:])
    epilogue_order = any('epilogue_defer' in row for row in rows)
    output_background = any(row.get('epilogue_defer')==2 for row in rows)
    second_prefill = any('second_prefill' in row for row in rows)
    if second_prefill:
        if not stage_progress or stage_handoff or stage_sparse:
            raise ValueError('second prefill uses the original progress observer')
        if any(row['second_prefill'] not in (0,1) or row.get('epilogue_defer')!=2
               for row in rows if 'second_prefill' in row):
            raise ValueError('second prefill requires switch 0/1 and all-CTA epilogue deferral')
    if epilogue_order:
        if (not stage_progress and not a_rotation) or stage_handoff or stage_sparse:
            raise ValueError('epilogue order uses progress or the declared A-rotation dual observer')
        path=source/'cutlass/include'/common.PING;text=path.read_text()
        site='        // Update starting mainloop pipeline state for the next tile\n'
        text=common.insert_once(text,site,
            '#ifdef R13_EPILOGUE_ORDER\n'
            '#if defined(R13_SECOND_PREFILL) && defined(R13_STAGE_PROGRESS)\n'
            '        if (blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 &&\n'
            '            threadIdx.x==128 && mainloop_pipe_consumer_state.count()==0) {\n'
            '          r13_s.first_tail_c=v06_clock(); r13_s.first_tail_ns=v06_ns();\n'
            '        }\n'
            '#endif\n'
            '        // The first WG has already released its last input slot.\n'
            '        // Read the next WG math completion without advancing its barrier.\n'
            '        if (warp_group_role==WarpGroupRole::Consumer0 &&\n'
            '            mainloop_pipe_consumer_state.count()==0 &&\n'
            '            (::r13_epilogue_defer==2 || (::r13_epilogue_defer==1 &&\n'
            '             blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)))\n'
            '          shared_storage.pipelines.math_wg_order.barrier_[0][0].wait(0);\n'
            '#endif\n')
        path.write_text(text)
    if second_prefill:make_second_prefill_overlay(source)
    common.make_overlay(source/'cutlass',source/'overlay')
    if a_rotation:
        # The generic overlay stamps MAIN at the next-state update. Our added
        # output deferral precedes that site, so move MAIN to the true tail end.
        path=source/'overlay'/common.PING;text=path.read_text()
        stamp='          v06_stamp(V06_MAIN_END, v06_tile_seq);\n'
        if text.count(stamp)!=1:raise ValueError('A rotation MAIN stamp site changed')
        text=text.replace(stamp,'',1)
        site='#ifdef R13_EPILOGUE_ORDER\n#if defined(R13_SECOND_PREFILL) && defined(R13_STAGE_PROGRESS)\n'
        if text.count(site)!=1:raise ValueError('A rotation deferral site changed')
        text=text.replace(site,stamp+site,1);path.write_text(text)
    shutil.copy2(ROOT/'probes/r18_trace.hpp',source/'overlay/v06_trace.hpp')
    for header in [common.COOP,common.PING]:
        path=source/'overlay'/header;text=path.read_text()
        site='        ++v06_tile_seq;\n'
        if text.count(site)!=1:raise ValueError('consumer coordinate site changed')
        text=text.replace(site,site+'        v06_begin(work_tile_info, v06_tile_seq);\n')
        path.write_text(text)
    first_supply = any('first_supply_events' in row for row in rows)
    if first_supply:
        if not producer_delay or not dual_clock:raise ValueError('first supply uses the declared delay/dual experiment')
        make_first_supply_overlay(source)
    ready_observer = any('ready_observer' in row for row in rows)
    if ready_observer:
        if not first_supply:raise ValueError('ready observer requires first supply markers')
        path=source/'overlay'/common.PING;text=path.read_text()
        site='        } \n      } // Scheduler Producer Warp End'
        text=common.insert_once(text,site,
            '        } else {\n'
            '          ::r13_observe_ready(cute::cast_smem_ptr_to_uint(&shared_storage.pipelines.mainloop.full_barrier_[0]));\n',after=False)
        path.write_text(text)
    ready_release = any(row.get('ready_generation_bound')=='first_release' for row in rows)
    if ready_release:
        if not ready_observer:raise ValueError('release-bound observation requires the ready observer')
        path=source/'overlay'/common.MAINLOOP;text=path.read_text()
        text=common.insert_once(text,'      pipeline.consumer_release(smem_pipe_release);\n',
            '      if (threadIdx.x==128 && r13_first_target(v06_tile_seq) && r13_ready_observer && smem_pipe_release.count()==0) r13_first_release_stamp();\n')
        path.write_text(text)
    if stage_sparse and stage_progress:raise ValueError("choose one stage observation protocol")
    if stage_sparse or stage_progress:
        if dual_clock:raise ValueError('stage-sparse and dual-clock are separate protocols')
        if any((r['config'],r['m'],r['n'],r['k'],r.get('stages',6)) not in [('cfg_b',2816,1536,4096,4),('cfg_b',2816,1536,4096,6)] and not (r['config']=='cfg_b' and r.get('set')=='reference' and r.get('variants')==['plain']) for r in rows):raise ValueError('stage-sparse is the declared cfg_b pair only')
        stage_variants= ['plain','progress'] if epilogue_order else ['plain','progress','stamped'] if stage_handoff else ['plain','stamped']
        if any(r.get('variants',stage_variants)!=stage_variants and not (r['config']=='cfg_b' and r.get('set')=='reference' and r.get('variants')==['plain']) for r in rows):raise ValueError('stage observation variants differ from declared protocol')
        make_stage_sparse_overlay(source, progress=stage_progress, handoff=stage_handoff)
    commands={}
    variants=['plain','dual'] if a_rotation else ['plain','progress'] if epilogue_order else ['plain','progress','stamped'] if stage_handoff else ['plain','wide','stamped','dual'] if dual_clock else ['plain'] if family=='b01_cache' else ['plain','stamped']
    configurations={common.binary_prefix(r):(r['config'],r.get('stages')) for r in rows}
    for prefix,(config,stages) in sorted(configurations.items()):
        if stages is not None and stages<=0:raise ValueError('explicit stage count must be positive')
        selected_variants=sorted({v for r in rows if common.binary_prefix(r)==prefix for v in r.get('variants',variants)})
        if selected_variants==['ends']:
            selected_variants=['ends','plain']  # Plain is used only for static setup.
        for variant in selected_variants:
            if variant not in ['plain','wide','stamped','dual','ends'] and not (stage_progress and variant=='progress'):raise ValueError('unknown observation variant')
            if variant=='dual' and not dual_clock:raise ValueError('dual observation requires --dual-clock')
            cmd=['nvcc','-std=c++17','-O3','-DNDEBUG','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','--ptxas-options=-v',f'-DV06_CFG={CONFIGS[config]}']
            if epilogue_order:cmd.append('-DR13_EPILOGUE_ORDER')
            if output_background:cmd.append('-DR13_OUTPUT_BACKGROUND')
            if second_prefill:cmd.append('-DR13_SECOND_PREFILL')
            if work_swap:cmd.append('-DR10_WORK_SWAP')
            if a_rotation:cmd.append('-DR13_A_ROW_ROTATION')
            if producer_delay:cmd.append('-DR10_PRODUCER_DELAY')
            if first_supply:cmd.append('-DR13_FIRST_SUPPLY')
            if ready_observer:cmd.append('-DR13_READY_OBSERVER')
            if ready_release:cmd.append('-DR13_READY_RELEASE')
            if stages is not None:cmd.append(f'-DR18_STAGE_COUNT={stages}')
            if stage_sparse and variant=='stamped':cmd.append('-DR13_STAGE_SPARSE')
            if stage_progress and variant in ['stamped','progress']:cmd.append('-DR13_STAGE_PROGRESS')
            if stage_handoff and variant=='stamped':cmd.append('-DR13_STAGE_HANDOFF')
            if dual_clock and variant not in ['plain','ends']:cmd+=['-DR18_DUAL_CLOCK' if variant=='dual' else '-DR18_MATCH_DUAL_LAYOUT']
            if variant in ['stamped','dual','ends','progress']:
                cmd+=['-DV06_TRACE','-Isource/overlay']
                if variant=='ends':cmd+=['-DV08_ENDS']
                if family=='r18_lite':cmd+=['-DR18_LIGHT']
            cmd+=['-Isource/probes','-Isource/cutlass/include','-Isource/cutlass/tools/util/include','source/probes/r18.cu','-o',f'build/{prefix}_{variant}']
            commands[f'{prefix}_{variant}']=cmd
    common.write_json(root/'build/commands.json',commands);common.write_json(root/'cases.json',rows)
    common.write_json(root/'run_config.json',dict(family=family,cases_sha256=common.sha(root/'cases.json'),variants=variants,trace_version='r18-dual-clock-events' if a_rotation else 'r13-second-prefill-v1' if second_prefill else 'r13-output-background-v1' if output_background else 'r13-epilogue-order-v1' if epilogue_order else 'r13-stage-handoff-v1' if stage_handoff else 'r13-stage-progress-v1' if stage_progress else 'r13-stage-sparse-v1' if stage_sparse else None))
    common.write_json(root/'source_hashes.json',{str(p.relative_to(root)):common.sha(p) for p in source.rglob('*') if p.is_file()})


def case_args(row):
    keys=['m','n','k','lda','ldb','ldd','storage_m','storage_n','zero_m','zero_n','swizzle']
    if 'evict' in row:keys.append('evict')
    for key in ['sm_count','input_mode','seed','alloc_lda','alloc_ldb','input_map_m','input_map_n','warmup_policy','raster_order','history_mode','history_source','mirror_target_slot','b_offset_bytes','b_allocation_extra_bytes','work_swap55_91','producer_delay_ns','first_supply_events','ready_observer','epilogue_defer','second_prefill','a_row_rotation']:
        if key in row:keys.append(key)
    return [str(x) for key in keys for x in ('--'+key.replace('_','-'),row[key])]


def setup(root):
    common.verify(root);records=[]
    for row in json.loads((root/'cases.json').read_text()):
        raw=subprocess.check_output([str(root/'build'/f"{common.binary_prefix(row)}_plain"),'--mode','setup',*case_args(row)],text=True)
        event=next(json.loads(l) for l in raw.splitlines() if '"setup"' in l)
        expected_stages=row.get('stages',4 if row['config'].startswith('cfg_c') else 6)
        if event['stages']!=expected_stages:raise ValueError('stage count changed')
        records.append(dict(case=row['id'],setup=event))
    for group in {r['group'] for r in json.loads((root/'cases.json').read_text())}:
        ids={r['id'] for r in json.loads((root/'cases.json').read_text()) if r['group']==group and r['kind'] in ['ordinary','oob','explicit_zero'] and r['config'] in ['cfg_a','cfg_c']}
        grids={tuple(r['setup']['grid']) for r in records if r['case'] in ids}
        if len(grids)>1:raise ValueError('matched boundary grid changed')
    common.write_json(root/'static_setup.json',records);common.write_json(root/'environment.json',common.identity())
    (root/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    print('setup verified',len(records),'cases')


def run_one(root,row,variant,trial,attempt=0):
    folder=root/'samples'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    suffix=f'{variant}-{trial:02}'+(f'-retry{attempt}' if attempt else '')
    path=folder/(suffix+'.json')
    if path.exists():
        record=json.loads(path.read_text())
        if row.get('warmup_policy','adaptive')=='adaptive' and record['returncode'] and 'last five warmups CV exceeds 2%' in record['stderr'] and attempt<2:
            return run_one(root,row,variant,trial,attempt+1)
        if record['returncode'] or common.sha(root/record['raw'])!=record['raw_sha256']:raise ValueError('invalid prior process')
        return record
    cmd=[str(root/'build'/f"{common.binary_prefix(row)}_{variant}"),*case_args(row)]
    start=time.time_ns();proc=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
    raw=folder/(suffix+'.txt.gz')
    with gzip.open(raw,'wt') as stream:stream.write(proc.stdout)
    record=dict(case=row['id'],variant=variant,trial=trial,attempt=attempt,command=cmd,returncode=proc.returncode,
       host_start_ns=start,host_stop_ns=time.time_ns(),stderr=proc.stderr,raw=str(raw.relative_to(root)),raw_sha256=common.sha(raw))
    for line in proc.stdout.splitlines():
        ev=json.loads(line)
        if ev.get('event')=='call':record.update(elapsed_us=ev['elapsed_us'],warmup=ev['warmup_us'])
    common.write_json(path,record)
    if row.get('warmup_policy','adaptive')=='adaptive' and proc.returncode and 'last five warmups CV exceeds 2%' in proc.stderr and attempt<2:
        return run_one(root,row,variant,trial,attempt+1)
    if proc.returncode:raise ValueError('failed GEMM process: '+str(path))
    if variant=='ends' and row.get('history_mode')!='series':
        from v08_model import observe
        observe(root,record,row)
    else:
        from analyze_r18 import replay
        replay(root,record,row)
    return record


def trial_rows(subset,trial):
    subset=subset[:]
    if subset and all(r.get('sampling_order')=='alternate_swizzle' for r in subset):
        return sorted(subset,key=lambda r:r['swizzle'],reverse=bool(trial%2))
    if subset and all(r.get('sampling_order')=='alternate_history' for r in subset):
        return sorted(subset,key=lambda r:r['history_mode']=='series',reverse=bool(trial%2))
    random.Random(20261008+trial).shuffle(subset)
    return subset


def main(family='r18'):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('step',choices=['prepare','build','setup','sample','list'])
    p.add_argument('--output',type=Path);p.add_argument('--cutlass-root',type=Path);p.add_argument('--procs',type=int,choices=[1,10],default=10)
    p.add_argument('--cases-file',type=Path,help='prepare: use a listed extension of an existing experiment')
    p.add_argument('--dual-clock',action='store_true',help='prepare: add same-call globaltimer events; wide/stamped/dual share scratch layout, ends keeps its original layout')
    p.add_argument('--stage-sparse',action='store_true',help='prepare: cfg_b stage4/6 same-slot sparse clock events, plain already matches scratch')
    p.add_argument('--stage-progress',action='store_true',help='prepare: consumer commit/loop progress only, same cfg_b stage pair')
    p.add_argument('--stage-handoff',action='store_true',help='prepare: extend progress with same-call sender/receiver handoff endpoints')
    p.add_argument('--set',help='sample: select the named case set, e.g. reference before ctrl')
    a=p.parse_args()
    if a.step=='list':print(json.dumps(cases(family),indent=2));return
    root=a.output.resolve()
    if a.step=='prepare':
        rows=json.loads(a.cases_file.read_text()) if a.cases_file else None
        prepare(root,a.cutlass_root.resolve(),family,rows,a.dual_clock,a.stage_sparse,a.stage_progress,a.stage_handoff);return
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Slurm allocation required')
    if a.step=='build':
        version=subprocess.check_output(['nvcc','--version'],text=True)
        if 'release 12.9,' not in version:raise ValueError('CUDA12.9 required')
        (root/'build/nvcc-version.txt').write_text(version);common.build(root)
        common.write_json(root/'build/sass_hashes.json',{p.name:common.sha(p) for p in (root/'build').glob('*.sass')});return
    env=common.identity();gpu=env['gpu'].split(',')[0].strip()
    locks=[open('/tmp/gh200-measurement-'+gpu+'.lock','a'),(root/'.run.lock').open('a')]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.step=='setup':setup(root);return
    common.verify(root)
    if env!=json.loads((root/'environment.json').read_text()):raise ValueError('allocation/device changed')
    config=json.loads((root/'run_config.json').read_text())
    if config['family']!=family or common.sha(root/'cases.json')!=config['cases_sha256']:raise ValueError('matrix changed')
    rows=json.loads((root/'cases.json').read_text())
    if a.set is not None:
        rows=[r for r in rows if r.get('set')==a.set]
        if not rows:raise ValueError('empty requested case set')
    # Matching physical-grid controls stay adjacent; group and variant order randomized.
    for trial in range(a.procs):
        groups=sorted({r['group'] for r in rows});random.Random(20261008+trial).shuffle(groups)
        for group in groups:
            subset=trial_rows([r for r in rows if r['group']==group],trial)
            for row in subset:
                variants=row.get('variants',config.get('variants',['plain'] if family=='b01_cache' else ['plain','stamped']))[:]
                random.Random(20261008+trial).shuffle(variants)
                for variant in variants:run_one(root,row,variant,trial)
        print(family,'trial',trial,'complete',flush=True)
    (root/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))

if __name__=='__main__':main()
