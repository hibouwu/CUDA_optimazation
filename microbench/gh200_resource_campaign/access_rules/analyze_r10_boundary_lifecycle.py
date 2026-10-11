#!/usr/bin/env python3
"""Audit original cfg_b tail MMA work and the existing stage graph's source contract."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from supply_model import cfg_b_source_program,input_slot_state


PTX='https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html'


def mma_atom_work(shape,coord,q):
    """Eight source-level m64n128k16 atom calls, with valid math kept separate."""
    m,n,k=shape;mi,ni=coord;atoms=[]
    for dk in range(0,64,16):
        for dm in (0,64):
            origin=(128*mi+dm,128*ni,64*q+dk)
            valid=tuple(max(0,min(tile,extent-start))
                for tile,extent,start in zip((64,128,16),(m,n,k),origin))
            atoms.append(dict(origin=origin,shape=(64,128,16),valid_shape=valid,
                nominal_FLOP=2*64*128*16,valid_math_FLOP=2*valid[0]*valid[1]*valid[2]))
    return atoms


def stage_lifecycle(kt,stages,outputs):
    """Source-only release obligations; no time, transport or barrier-phase observation."""
    if any(type(v) is not int for v in (kt,stages,outputs)) or kt<1 or stages<2 or outputs<1:
        raise ValueError('positive K/output counts and at least two stages required')
    groups=[];tiles=[];boundaries=[]
    for g in range(kt*outputs):
        j,q=divmod(g,kt)
        groups.append(dict(**input_slot_state(g,stages),output_ordinal=j,logical_k_group=q,consumer_wg=j%2,
            transaction_expected_bytes=32768,empty_arrivals_required=128,
            release_path='normal_wait1' if q<kt-1 else 'mma_tail_wait0',
            release_after_commit_group=g+1 if q<kt-1 else g,
            retired_group_required=g,tail_output=j if q==kt-1 else None))
    for j in range(outputs):
        last=(j+1)*kt-1
        tiles.append(dict(output_ordinal=j,consumer_wg=j%2,first_group=j*kt,last_group=last,
            full_wait_API_calls_per_consumer_thread=kt,commit_groups_per_WG=kt,
            normal_wait1_calls_per_consumer_thread=kt-1,tail_wait0_calls_per_consumer_thread=1,
            release_calls_per_consumer_thread=kt,
            mma_return_requires_commit_group=last,mma_return_requires_normal_release_group=last-1 if kt>1 else None,
            MMA_order_arrival='after mma return, before mma_tail',
            epilogue_entry_requires_own_mma_tail=True,epilogue_order_predecessor=j-1 if j else None,
            next_same_WG_output=j+2 if j+2<outputs else None,
            same_WG_next_output_requires_own_store_tail_and_epilogue_order_arrival=True))
        if j+1<outputs:
            predecessors=[]
            for q in range(min(stages,kt)):
                g=(j+1)*kt+q;previous=g-stages
                predecessors.append(dict(next_group=g,next_logical_k_group=q,release_group=previous if previous>=0 else None,
                    release_path=groups[previous]['release_path'] if previous>=0 else 'initial_empty'))
            boundaries.append(dict(from_output=j,to_output=j+1,first_next_output_acquires=predecessors,
                next_MMA_order_wait_requires_prior_MMA_order_arrival=True,
                prior_tail_wait0_is_not_an_MMA_order_barrier_predecessor=True))
    tail=[]
    for step in range(stages):
        count=kt*outputs+step;state=input_slot_state(count,stages)
        tail.append(dict(step=step,**state,release_group=count-stages if count>=stages else None,
            initial_empty=count<stages))
    reacquired=list(range(max(0,kt*outputs-stages)))
    final=[x['release_group'] for x in tail if x['release_group'] is not None]
    if sorted(reacquired+final)!=list(range(kt*outputs)):
        raise ValueError('input producer exit omitted a released generation')
    return dict(groups=groups,outputs=tiles,output_boundaries=boundaries,producer_tail_waits=tail,
        input_exit_certificate=dict(releases_required_by_later_acquire=reacquired,releases_required_by_tail=final,
            every_used_group_covered_once=True),
        scope='Software ordering and full/empty obligations only. Empty completion needs all128 arrivals; input producer tail is not kernel/D completion.')


def analyze(run,binding,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        value=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=value;return value
    def read(path):sha(path);return json.loads(path.read_text())
    manifest=read(run/'source_hashes.json');previous=read(binding/'B-boundary-binding.json')
    contracts={
        'probes/r10.cu':['using KernelTile = Shape<_128, _128, _64>','float, float, void',
            '8, cutlass::half_t, cutlass::layout::RowMajor, 8, float'],
        'cutlass/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl':[
            'Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>',
            'cute::make_tiled_mma(cute::GMMA::ss_op_selector<'],
        'cutlass/include/cute/arch/mma_sm90.hpp':['else if constexpr (is_same_v<ElementC, float>)',
            'return SM90::GMMA::MMA_64x128x16_F32F16F16_SS<MajorA, MajorB, Args...>{}'],
        'cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp':[
            'struct MMA_Traits<SM90_64x128x16_F32F16F16_SS','using Shape_MNK = Shape<_64,_128,_16>'],
        'cutlass/include/cute/atom/mma_atom.hpp':['auto a_tensor = zipped_divide(t_tensor, a_tile)',
            'size<I>(AtomShape_MNK{}) * size<I+1>(get_thr_layout_vmnk())'],
        'cutlass/include/cute/algorithm/gemm.hpp':['for (int m = 0; m < M; ++m)',
            'for (int n = 0; n < N; ++n)','for (int k = 0; k < K; ++k)','mma.call(D, A, B, C)'],
        'cutlass/include/cute/arch/mma_sm90_gmma.hpp':['struct MMA_64x128x16_F32F16F16_SS',
            'wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16','wgmma.wait_group.sync.aligned %0',
            'wgmma.commit_group.sync.aligned'],
        'cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp':[
            'static constexpr int K_PIPE_MMAS = 1','Tensor gA_mkl = local_tile(mA_mkl, TileShape{}',
            'pipeline.producer_acquire(smem_pipe_write)','warpgroup_commit_batch()',
            'for ( ; k_tile_count > 0; --k_tile_count)','warpgroup_wait<K_PIPE_MMAS>()',
            'pipeline.consumer_release(smem_pipe_release)','warpgroup_wait<0>()',
            'smem_pipe_release.advance(k_tile_count)','pipeline.producer_tail(smem_pipe_write)'],
        'cutlass/include/cutlass/pipeline/sm90_pipeline.hpp':[
            'uint32_t multicast_consumer_arrival_count = params.num_consumers',
            'is_signaling_thread_ = true','empty_barrier_ptr_[stage].wait(phase)',
            'full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes)',
            'empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_ & (!skip))',
            'for (int count = 0; count < Stages; ++count)','int signalling_id = (params_.group_id + 1) % Length'],
        'cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp':[
            'auto k_tile_count = size<3>(gA_mkl)','mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup',
            'StagesPerMathWarpGroup = 2','math_wg_order_barrier.arrive()',
            'collective_mainloop.mma_tail(','collective_epilogue.store_tail(']}
    sites={}
    for relative,needles in contracts.items():
        path=run/'source'/relative
        if sha(path)!=manifest[relative]:raise ValueError('archived source hash differs: '+relative)
        lines=path.read_text().splitlines();sites[relative]=[]
        for needle in needles:
            matches=[i+1 for i,line in enumerate(lines) if needle in line]
            if not matches:raise ValueError('archived source contract changed: '+needle)
            sites[relative].append(dict(pattern=needle,lines=matches))
    model_path=Path(__file__).with_name('supply_model.py');model=model_path.read_text();sha(model_path)
    model_contracts=['if i>=stages:','a[\'released\'][i-stages]',"a['ready_count'][i]!=3",
        "previous=a['outputs'][i//kt-2]['end']",'W=C if i%kt==0 else max(C,a[\'completed\'][i-1])+wait_overhead_ns',
        "if i%kt:push(W,'release',cta,i-1)","push(M,'mma_return',cta,out);push(max(M,D),'main_end',cta,out)",
        "a['consumer_free']=t+handoff_delay_ns","a['released'][(i+1)*kt-1]=t",
        "not isinstance(units,tuple) or not units"]
    model_sites={}
    for needle in model_contracts:
        matches=[i+1 for i,line in enumerate(model.splitlines()) if needle in line]
        if not matches:raise ValueError('existing stage contract changed: '+needle)
        model_sites[needle]=matches
    cases=[]
    for old in previous['cases']:
        path=binding/old['ledger_file']
        if sha(path)!=old['ledger_sha256']:raise ValueError('bound ledger hash differs')
        ledger=json.loads(gzip.decompress(path.read_bytes()));kt=ledger['ktiles'];counts=ledger['scheduler']['output_counts']
        setup=dict(tile=[128,128,64],cluster=[1,1,1],grid=[132,1,1],stages=6)
        m,n,k=ledger['shape'];row=dict(config='cfg_b',m=m,n=n,k=k,lda=k,ldb=int(old['case'].split('ldb')[1]),ldd=n,swizzle=1)
        try:cfg_b_source_program(row,setup)
        except ValueError as error:guard=str(error)
        else:raise ValueError('old full-tile guard was relaxed')
        lifecycles={t:stage_lifecycle(kt,6,t) for t in sorted(set(counts))}
        histogram=Counter();valid_math=0;nominal=0;atom_count=0;zero_atoms=0;examples=[];release_paths=Counter()
        for out in ledger['outputs']:
            lifecycle=lifecycles[counts[out['cta']]];j=out['output_ordinal']
            for q,group in enumerate(out['groups']):
                atoms=mma_atom_work(ledger['shape'],out['coord'],q)
                valid=sum(a['valid_math_FLOP'] for a in atoms);full=sum(a['nominal_FLOP'] for a in atoms)
                valid_math+=valid;nominal+=full;atom_count+=len(atoms)
                zero=sum(a['valid_math_FLOP']==0 for a in atoms);zero_atoms+=zero
                histogram[tuple(out['coord'])[0]==20,tuple(out['coord'])[1]==23,group['valid_k'],zero]+=1
                binding_group=lifecycle['groups'][j*kt+q]
                if any(binding_group[name]!=group['slot_state'][name] for name in group['slot_state']):
                    raise ValueError('lifecycle ring binding differs')
                if group['transaction_expected_bytes']!=32768 or len(atoms)!=8 or full!=2097152:
                    raise ValueError('partial K changed full software service work')
                release_paths[binding_group['release_path']]+=1
                if out['coord'] in ([12,12],[20,23]) and q==31:
                    examples.append(dict(cta=out['cta'],output_ordinal=j,coord=out['coord'],logical_K_group=q,
                        valid_K=group['valid_k'],atoms=atoms,source_and_fill=group['atoms'],lifecycle=binding_group))
        if valid_math!=2*m*n*k or nominal!=len(ledger['outputs'])*kt*2097152:
            raise ValueError('independent MMA volume conservation differs')
        cases.append(dict(case=old['case'],shape=ledger['shape'],existing_full_tile_guard=guard,
            source_level_MMA_atom_calls=atom_count,nominal_MMA_FLOP=nominal,valid_math_FLOP=valid_math,
            zero_valid_math_atom_calls=zero_atoms,commit_groups=len(ledger['outputs'])*kt,
            release_groups_by_path=dict(release_paths),full_wait_calls_per_consumer_thread=len(ledger['outputs'])*kt,
            software_empty_arrivals=len(ledger['outputs'])*kt*128,tail_examples=examples,
            MMA_group_classes=[dict(M_tail=mt,N_tail=nt,valid_K=vk,zero_valid_atoms=z,K_groups=count)
                for (mt,nt,vk,z),count in sorted(histogram.items())],
            lifecycle_by_CTA_output_count={str(t):graph for t,graph in lifecycles.items()}))
    for name in ('analyze_r10_boundary_lifecycle.py','test_r13_boundary_lifecycle.py'):sha(Path(__file__).with_name(name))
    report=dict(scope='Archived cfg_b K16/M/N tail work and existing stage-graph structure, without timing replay or new prices.',
        archived_source_sites=sites,source_manifest_matches=len(sites),existing_model_sites=model_sites,cases=cases,
        source_required_structure=[
            'Use ceil(K/64), all full groups and persistent g=j*Kt+q with the archived per-CTA T and WG alternation.',
            'Every K group retains one full 128x128x64 MMA group:8 m64n128k16 atom calls/one commit, including K16.',
            'Each full wait covers32768B destination valid+fill; zero-source B1 still needs its destination obligation.',
            'For local q>0: commit(q) then wait1 then release(q-1); prologue q0 never releases a prior output group.',
            'mma return signals the next WG before tail wait0; last group is released only after that wait0.',
            'Reacquire g waits empty completion of g-S, one owning WG/all128 arrivals, not both WGs/256 arrivals.',
            'Same WG output j+2 follows its own store_tail; epilogues additionally obey their separate sequence stage.',
            'Producer tail traverses S slots at the final persistent state. It closes input stage lifetime, not global D completion.'],
        existing_graph_reusable=[
            dict(source='Same-slot reacquire',fields=['released','producer_index'],
                condition='Topology only; empty arrival completion must remain distinct from optimistic release time.'),
            dict(source='Full wait and one full MMA group',fields=['ready_count','ready','starts','commits','completed'],
                condition='Keep all3 destination obligations and full8 atom calls; source/fill completion and compute time unpriced.'),
            dict(source='Normal wait1/release',fields=['wait1_return','released'],
                condition='q>0 releases q-1; q0 is a prologue and releases no prior output group.'),
            dict(source='MMA handoff versus tail release',fields=['mma_returns','main_ends'],
                condition='Next WG follows mma return/order arrival; last group release follows tail wait0.'),
            dict(source='Same WG j+2 and separate epilogue sequence',fields=['outputs.end','consumer_free'],
                condition='Needs the matching legacy D/store_tail completion interface; its price and OOB path are not transferred.')],
        missing_or_unidentified=[
            'Existing source program/full-tile guards reject old geometry; kt floor division cannot replace the explicit boundary ledger.',
            'Address-unit source path requires nonempty units;672 zero-source B1 boxes per case are rejected and cannot be dropped or treated as free full completion.',
            'Source/fill/destination completion, physical source reuse and arbitration remain unpriced; identical work does not identify timing.',
            'Existing release time is optimistic at wait1/tail;128 arrival completion and acquire-return time are not independently identified.',
            'Existing timing graph has no explicit producer_tail empty-wait traversal/input-safe-exit node. Its returned end time does not prove input or kernel safe exit.',
            'MMA atom/commit counts fix work, not compute_ns, commit lead, retire lag, wait/handoff cost or cross-card clock conversion.',
            'D64x32 bounds are known, but legacy OOB store source/read completion and endpoint composition remain unbound.',
            'No predicted old/new pitch response, full CTA distribution or independent frozen local holdout is qualified by this audit.'],
        C_boundary_request=dict(scope='cfg_b legacy D only; do not import cooperative graphs or prices.',
            entry='Own mma_tail return plus epilogue ordered-barrier wait return.',
            outputs_needed=['Per-output store_tail return and all128 epilogue order arrivals for the next WG.',
                'D SMEM source read completion versus global D visibility for each valid/partial/zero-valid64x32 box.',
                'Same-WG j+2 permission and a kernel-completion boundary separate from input producer_tail.'],
            status='Concrete unbound interface for main/C; output_model unchanged.'),
        external_sources=[dict(url=PTX,section='9.7.15.7.2-3',use='Commit forms a WGMMA group; wait N completes groups older than the N most recent; wait0 drains all.')],
        input_sha256=inputs)
    (output/'B-boundary-lifecycle.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 原始R10最后K16的工作与释放语义','',
        '每个K组仍调用8个m64n128k16 atom并commit一次。最后K16只有2个atom包含有效K，另6个仍执行零填充区域。',
        '每输出32次full等待、32个commit组、31次普通wait1/前组释放、1次tail wait0/末组释放；每组期待32768B。',
        '全调用16128组、129024次atom调用；有效数学量31200000000FLOP，名义MMA量33822867456FLOP。','',
        'S6/Kt32输出切换：后输出q0–4的acquire等待前输出q26–30普通释放，q5等待前输出q31的tail释放。',
        '下一WG的MMA顺序通知位于上一mma返回之后、mma_tail之前；不能给首组额外添加前输出tail依赖。',
        'empty完成需要拥有该组的一个WG共128次到达，不是两个WG共256次；到达延迟未定。','',
        '既有图可复用持续ring、普通/末组释放、MMA与epilogue两级顺序。旧full-tile guard仍拒绝该几何，零源fill/D生命周期与服务价格未补齐。','']
    (output/'B-boundary-lifecycle.md').write_text('\n'.join(lines))
    print(json.dumps(dict(cases=len(cases),source_manifest_matches=len(sites),
        per_case={key:cases[0][key] for key in ('source_level_MMA_atom_calls','nominal_MMA_FLOP','valid_math_FLOP',
            'zero_valid_math_atom_calls','commit_groups','release_groups_by_path','existing_full_tile_guard')}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    analyze(args.run.resolve(),args.binding.resolve(),args.output.resolve())
