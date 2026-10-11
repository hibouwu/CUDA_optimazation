#!/usr/bin/env python3
"""Bind the original R10 boundary ledger to archived static scheduler defaults."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from analyze_r10_granule_overlap import source_granules
from supply_model import cfg_b_persistent_boundary_ledger


def analyze(run,prior,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        value=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=value
        return value
    def read(path):sha(path);return json.loads(path.read_text())
    manifest=read(run/'source_hashes.json');protocol=read(run/'protocol-formal.json')
    previous=read(prior/'B-boundary-ledger.json');review=read(run/'reviews/ROOT-independent.json')
    contracts={
        'probes/r10.cu':['using KernelCluster = Shape<_1, _1, _1>',
            'KernelTmaWarpSpecializedPingpong','Shape<int, int, int, int>, Mainloop, Epilogue>',
            '8, cutlass::half_t, cutlass::layout::RowMajor, 8',
            '{{1.f, 0.f}, nullptr, sd, d.pointer, sd}}'],
        'cutlass/include/cutlass/gemm/kernel/gemm_universal_decl.h':['class TileScheduler_ = void'],
        'cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp':[
            'using Scheduler = PersistentTileSchedulerSm90','Default (void) for Sm90'],
        'cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp':[
            'StaticPersistentTileScheduler<PersistentTileSchedulerSm90>',
            'IsDynamicPersistent = false','return {major_work_idx, minor_work_idx}'],
        'cutlass/include/cutlass/gemm/kernel/static_tile_scheduler.hpp':[
            'int max_swizzle_size = 1','raster_order = RasterOrderOptions::Heuristic',
            'current_work_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y)',
            'current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count)'],
        'cutlass/include/cutlass/gemm/kernel/tile_scheduler_params.h':[
            'if (tiles_n > tiles_m)','return RasterOrder::AlongM',
            'launch_grid = dim3(1, cluster_shape.n(), 1)',
            'launch_grid.x = possibly_truncate(sm_count, problem_blocks_total)'],
        'cutlass/include/cutlass/gemm/collective/builders/sm90_common.inl':[
            'return (sizeof(ElementA) == 2 && sizeof(ElementB) == 2)',
            'IsUseRmemA = (!IsInputSizeTwoBytes && !IsLayoutAkBk) || IsABDifferentWidth || HasScales'],
        'cutlass/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl':[
            'not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()',
            'MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>'],
        'cutlass/include/cutlass/gemm/dispatch_policy.hpp':[
            'struct Has_SwapAB { static constexpr bool value = false; }',
            '{ static constexpr bool value = T::SwapAB; }'],
        'cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp':[
            'MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>'],
        'cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp':[
            'TileSchedulerArguments scheduler{}','if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>)',
            'mainloop_pipe_consumer_state.advance(k_tile_count)',
            'mainloop_pipe_producer_state.advance(k_tile_count)',
            'mainloop_pipe_consumer_state.advance(k_tile_count * NumMmaWarpGroups)',
            'scheduler.advance_to_next_work(NumMmaWarpGroups)'],
        'cutlass/include/cutlass/pipeline/sm90_pipeline.hpp':[
            'PipelineState(): index_{}, phase_{}, count_{}','phase_ ^= 1',
            'index_ = (index_ + num_iterations) % Stages','InitialProducerPhase = 1'],
        'overlay/r10_trace.hpp':['r10_block() >= 4 || slot > 1','r10_block() >= 4 || group != 0',
            'r10_trace_ptr[r10_block() * 8 + slot] = cycle',
            'r10_trace_ptr[r10_block() * 8 + 4 + group] += 1'],
        'overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp':[
            'r10_stamp(2 * (int(threadIdx.x / 128) - 1))'],
        'overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp':[
            'r10_tail(int(threadIdx.x / 128) - 1)']}
    sites={}
    for relative,needles in contracts.items():
        path=run/'source'/relative
        if sha(path)!=manifest[relative]:raise ValueError('source differs from archived manifest: '+relative)
        text=path.read_text();lines=text.splitlines();sites[relative]=[]
        if relative.endswith('sm90_mma_tma_gmma_ss_warpspecialized.hpp') and 'SwapAB' in text:
            raise ValueError('SS collective unexpectedly declares SwapAB')
        for needle in needles:
            matches=[i+1 for i,line in enumerate(lines) if needle in line]
            if not matches:raise ValueError('archived source contract changed: '+needle)
            sites[relative].append(dict(pattern=needle,lines=matches))
    cases=[];metadata=0;trace_checks=0;ring_checks=0
    for row in protocol['cases']:
        setup=dict(grid=[132,1,1],stages=6,tile=[128,128,64],cluster=[1,1,1])
        bound=cfg_b_persistent_boundary_ledger(dict(row,swizzle=1,raster_order='heuristic'),setup)
        old_case=next(c for c in previous['cases'] if c['case']==row['id'])
        old_path=prior/old_case['ledger_file']
        if sha(old_path)!=old_case['ledger_sha256']:raise ValueError('prior logical ledger SHA differs')
        logical=json.loads(gzip.decompress(old_path.read_bytes()));old={tuple(o['coord']):o for o in logical['outputs']}
        if bound['totals']!=logical['totals']:raise ValueError('binding changed aggregate work')
        counts=bound['scheduler']['output_counts'];per_cta=defaultdict(list);cohorts=defaultdict(lambda:dict(A=set(),B=set()))
        class_counts=Counter();windows=[]
        for out in bound['outputs']:
            c,j=out['cta'],out['output_ordinal'];mi,ni=out['coord'];linear=c+132*j
            if out['coord']!=(linear%21,linear//21):raise ValueError('direct static scheduler formula differs')
            saved=old[out['coord']];per_cta[c].append(out)
            if out['D_boxes']!=[dict(d,origin=tuple(d['origin']),shape=tuple(d['shape']),valid_shape=tuple(d['valid_shape']))
                    for d in saved['D_boxes']]:raise ValueError('binding changed D bounds')
            class_counts[j,mi==20,ni==23]+=1
            for q,(group,prior_group) in enumerate(zip(out['groups'],saved['groups'])):
                if json.loads(json.dumps(group['atoms']))!=prior_group['atoms']:
                    raise ValueError('binding changed logical source/fill atoms')
                state=group['slot_state'];g=j*32+q
                # Independently increment the archived C++ index/phase state from zero.
                index,phase=0,0
                for _ in range(g):
                    index+=1
                    if index==6:index=0;phase^=1
                if (state['group'],state['slot'],state['full_wait_phase'],state['empty_wait_phase'])!=(g,index,phase,phase^1):
                    raise ValueError('ring differs from repeated archived PipelineState increment')
                if state['previous_group']!=(g-6 if g>=6 else None):raise ValueError('slot generation predecessor differs')
                ring_checks+=1
                for atom in group['atoms']:
                    key=atom['source_key'];cohorts[g][key[0]].update(source_granules(key,128))
            if c<4 and j==2:
                windows.append(dict(cta=c,derived_output_ordinal=j,derived_coord=[mi,ni],consumer_warp_group=0,
                    source_valid_bytes=sum(a['source_valid_bytes'] for g in out['groups'] for a in g['atoms']),
                    destination_fill_bytes=sum(a['destination_fill_bytes'] for g in out['groups'] for a in g['atoms']),
                    first_ring_state=out['groups'][0]['slot_state'],last_ring_state=out['groups'][-1]['slot_state'],
                    coordinate_status='Derived from archived source, not observed coordinate payload.'))
        records=[]
        for mode in ('plain','trace'):
            for trial in range(protocol['processes_per_mode']):
                path=run/f'samples/formal/{row["id"]}/{mode}/{trial:02d}/result.json';record=read(path)
                if inputs[str(path)]!=review['evidence'][str(path)]:raise ValueError('metadata differs from prior review')
                if any(record[name]!=row[name] for name in ('config','m','n','k','lda','ldb','ldd')):
                    raise ValueError('recorded geometry differs')
                if record['grid_ctas']!=132 or record['stages']!=6:raise ValueError('recorded grid/stages differ')
                if mode=='trace':
                    if (record['trace_cta_limit'],record['trace_consumer_groups'],record['trace_tile'])!=(4,[0],'last'):
                        raise ValueError('trace endpoint scope differs')
                    words=record['trace']
                    if len(words)!=132*8 or any(words[32:]):raise ValueError('trace scope differs')
                    for c in range(4):
                        h=words[8*c:8*c+8];wg0=[o for o in per_cta[c] if o['consumer_warp_group']==0]
                        if not (h[0]>0 and h[1]>=h[0] and h[2:4]==[0,0] and h[4]==len(wg0)==2 and h[5]==0 and h[7]==4):
                            raise ValueError('saved last-WG0 count/slots incompatible with source binding')
                        trace_checks+=1
                records.append(dict(mode=mode,trial=trial,binary_sha256=record['binary_sha256']))
                metadata+=1
        compressed=gzip.compress(json.dumps(bound,separators=(',',':')).encode(),mtime=0)
        filename=row['id']+'-persistent-ledger.json.gz';(output/filename).write_bytes(compressed)
        cases.append(dict(case=row['id'],scheduler=bound['scheduler'],totals=bound['totals'],trace_windows=windows,
            output_classes=[dict(output_ordinal=j,M_tail=mt,N_tail=nt,count=count)
                for (j,mt,nt),count in sorted(class_counts.items())],
            global_K_cohort_coverage=[dict(global_group=g,operand=op,units128=len(cohorts[g][op]),
                next_group_intersection128=len(cohorts[g][op]&cohorts[g+1][op]) if g+1 in cohorts else None)
                for g in sorted(cohorts) for op in ('A','B')],
            records=records,ledger_file=filename,ledger_sha256=hashlib.sha256(compressed).hexdigest()))
        source_granules.cache_clear()
    for name in ('analyze_r10_boundary_binding.py','supply_model.py','v08_model.py','v06_model.py',
            'analyze_r10_granule_overlap.py','test_r13_boundary_binding.py'):
        sha(Path(__file__).with_name(name))
    report=dict(scope='Static archived scheduler/input-ring binding only; original timing graph remains unsupported.',
        source_derivation=['FP16/FP16 ordinary inputs select SS; SS has no SwapAB and Has_SwapAB defaults false.',
            'Omitted scheduler template is void -> PersistentTileSchedulerSm90; omitted Arguments use swizzle1/heuristic.',
            '21 M tiles,24 N tiles -> AlongM, no padding. Cluster1 and recorded grid product132 imply grid(132,1,1).',
            'linear=c+j*132; coord=(linear%21,linear//21). WG0 owns even j, WG1 odd j.',
            'Producer advances32 per output; WG consumers start at0/32 and advance64. g=j*32+q persists across outputs.',
            'Six-stage index=g%6, generation=g//6; full-wait phase=generation%2, empty-wait phase=(generation+1)%2.'],
        archived_source_sites=sites,archived_source_manifest_matches=len(sites),cases=cases,
        metadata_records_matching_prior_review=metadata,saved_trace_CTA_count_checks=trace_checks,
        ring_state_checks=ring_checks,input_sha256=inputs,
        limits=['CTA/output coordinates are source-derived, not explicitly sampled in the old trace.',
            'The trace WG0 count2 alone cannot distinguish total CTA output count3 from4; source mapping resolves it.',
            'Trace begins after the first full-barrier wait; input first readiness is outside the measured window.',
            'SMID is not used to infer coordinates, residency, overlap or hardware concurrency.',
            'Ring phases are software wait arguments; readiness, actual barrier phase, release time and safe exit remain unknown.',
            'Coverage unions/intersections are geometry only, not in-flight merging, cache residence or physical traffic.',
            'D bounds are attached to output coordinates; D pipeline source lifetime and completion remain unbound.',
            'No price, timing candidate, GPU/numerical replay, new observation qualification or independent frozen holdout.'])
    (output/'B-boundary-binding.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 原始R10的静态CTA、输出与ring绑定','',
        '归档默认参数推出AlongM/swizzle1/grid(132,1,1)。linear=c+j×132，坐标=(linear%21,linear//21)。',
        'CTA0–107各4个输出，108–131各3个输出；WG0取偶数j，WG1取奇数j。','',
        '| 保存trace的CTA | 源码推导j | 源码推导M/N tile | global K组 | 首/末slot |',
        '|---:|---:|---|---|---|']
    for w in cases[0]['trace_windows']:
        a,b=w['first_ring_state'],w['last_ring_state']
        lines.append(f'| {w["cta"]} | 2 | {w["derived_coord"]} | {a["group"]}–{b["group"]} | {a["slot"]}/{b["slot"]} |')
    lines+=['','每个保存窗都是完整M/N输出，只含最后K组的填充；坐标来自源码，旧trace没有坐标字段。',
        '60个元数据SHA复用旧独立审查，120个保存CTA的WG0计数/范围与绑定兼容。',
        '逻辑源、fill、D账本逐盒不变；ring只绑定软件slot/phase，不提供就绪、释放、退出或时间预测。','']
    (output/'B-boundary-binding.md').write_text('\n'.join(lines))
    print(json.dumps(dict(cases=len(cases),source_manifest_matches=len(sites),metadata=metadata,
        trace_CTA_checks=trace_checks,ring_checks=ring_checks),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    analyze(args.run.resolve(),args.prior.resolve(),args.output.resolve())
