#!/usr/bin/env python3
"""Build a source-auditable TMEM partial-order draft, not a runtime trace."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
K='include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp'
A='include/cute/arch/tmem_allocator_sm100.hpp'
B='include/cutlass/arch/barrier.h'
P='include/cutlass/pipeline/sm100_pipeline.hpp'
PI='include/cutlass/pipeline/sm90_pipeline.hpp'
E='include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp'
M='include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp'
ISA='https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    contracts_path=ROOT/'data/modules/dense_fp16/contracts.json'
    contract_bytes=contracts_path.read_bytes();contracts=json.loads(contract_bytes)
    nodes={n['id']:n for n in contracts['nodes']};edges={e['id']:e for e in contracts['edges']}
    evidence=[];evmap={}
    def ev(path,start,end=None,claim=''):
        end=start if end is None else end
        key=(path,start,end,claim)
        if key in evmap:return evmap[key]
        raw=(ROOT/'snapshot'/path).read_bytes();lines=raw.splitlines(keepends=True)
        ident=f'tmem.evidence.{len(evidence)+1:03d}'
        evidence.append({'id':ident,'path':path,'start_line':start,'end_line':end,
            'start_byte':sum(map(len,lines[:start-1])),'end_byte':sum(map(len,lines[:end])),
            'quote':b''.join(lines[start-1:end]).decode().rstrip('\n'),'source_sha256':hashlib.sha256(raw).hexdigest(),'claim':claim})
        evmap[key]=ident;return ident
    def external(ident,claim):
        evidence.append({'id':ident,'url':ISA,'section':'PTX ISA 9.0 / CUDA 13.0.0, 9.7.16.7.1',
            'claim':claim,'retrieved':'2026-09-08','quotation_policy':'brief paraphrase; no document copy'})
        return ident
    isa_alloc=external('tmem.evidence.isa_alloc','alloc may block when capacity is unavailable and stores the allocated address in shared memory; the store is weak. The source comment does not establish non-blocking behavior.')
    isa_permit=external('tmem.evidence.isa_permit','After relinquish_alloc_permit, that CTA may not allocate again. This operation is distinct from deallocation.')
    isa_pair=external('tmem.evidence.isa_pair','For cta_group::2, one warp from each peer CTA must participate while the peer is active. sync describes warp convergence, not a replacement for the caller peer-lifetime handshake.')
    cfg=[ev(K,426,429),ev(K,138,142),ev(K,234,240),ev(E,127),ev('include/cutlass/cutlass.h',96),ev('include/cutlass/gemm/dispatch_policy.hpp',1029,1034)]
    ev_count=ev(K,527,535);ev_init=ev(K,553,575);ev_mem=ev(K,725,735);ev_tail=ev(K,782,803)
    actors=[];resources=[];events=[];orders=[];transitions=[];views=[]
    def res(ident,ref,cta,label,states,**extra):
        resources.append({'id':ident,'contract_resource_ref':ref,'cta_rank':cta,'label':label,
            'states':[{'id':ident+'.'+s,'label':label_, 'invariants':inv,'evidence_refs':proof} for s,label_,inv,proof in states],**extra})
    def event(cta,role,key,label,line,ref=None,edge=None,kind='api_call',path=K,guard=None,effect=None,proof=(),anchors=None,**extra):
        ident=f'tmem.cta{cta}.{key}'
        if ref is not None:assert ref in nodes,ref
        if edge is not None:assert edge in edges,edge
        callsite=edges[edge].get('callsite') if edge else None
        if callsite:
            assert callsite['path']==path and callsite['start_line']==line,(edge,line)
            callsite=dict(callsite)
        else:
            callsite={'path':path,'start_line':line,'end_line':line,'source_expression':(ROOT/'snapshot'/path).read_text().splitlines()[line-1].strip()}
        events.append({'id':ident,'label':label,'kind':kind,'actor_id':f'tmem.actor.cta{cta}.{role}',
            'api_node_ref':ref,'call_edge_ref':edge,'callsite':callsite,
            'guard':guard or {'expression':'is_participant.'+('mma' if role=='mma' else 'epilogue'),'configured':True},
            'anchors':anchors or (['enter','effect','return'] if effect else ['enter','return']),
            'effect':effect,'evidence_refs':[ev(path,line)]+list(proof),**extra})
        return ident
    def point(event_id,anchor):return {'event_id':event_id,'anchor':anchor}
    def order(before,ba,after,aa,relation,condition,proof,group,**extra):
        ident=f'tmem.order.{len(orders)+1:03d}'
        orders.append({'id':ident,'before':point(before,ba),'after':point(after,aa),'relation':relation,
            'condition':condition,'evidence_refs':list(proof),'view_group':group,**extra})
    def local_chain(ids,group,proof):
        for a,b in zip(ids,ids[1:]):order(a,'return',b,'enter','program_order',
            'Only corresponding actor/thread invocations that execute this path; not a global total order.',proof,group,quantifier='per corresponding actor thread')
    def transition(resource,fr,to,event_,anchor,condition,proof,pre=()):
        ident=f'tmem.transition.{len(transitions)+1:03d}'
        transitions.append({'id':ident,'resource_id':resource,'from':resource+'.'+fr,'to':resource+'.'+to,
            'trigger':point(event_,anchor),'condition':condition,'preconditions':list(pre),'evidence_refs':list(proof)})
    for c in range(4):
        leader=(c%2==0);peer=c^1;pair=f'pair{c//2}'
        for role,count,selector in [('mma',32,'warp_category == WarpCategory::MMA (warp 0)'),('epilogue',128,'warp_category == WarpCategory::Epilogue (warps 4..7)')]:
            actors.append({'id':f'tmem.actor.cta{c}.{role}','label':f'CTA {c} '+('leader' if leader else 'peer')+' '+role,
                'cta_rank':c,'pair_id':pair,'peer_cta_rank':peer,'role':role,'thread_count':count,'selector':selector,
                'is_mma_leader_cta':leader,'evidence_refs':cfg,
                'event_instance_semantics':'A per-thread event family within this one CTA role; actors never merge different CTAs. Cross-thread joins state required arrival counts explicitly.'})
        alloc=f'tmem.cta{c}.allocation';ptr=f'tmem.cta{c}.base_pointer';bar=f'tmem.cta{c}.dealloc_barrier';named=f'tmem.cta{c}.allocation_barrier'
        res(alloc,'contract.res.tmem_alloc',c,f'CTA {c} TMEM allocation / 512-column request',[
            ('absent','尚未分配',['no live allocation yet'],[ev(A,124,141)]),
            ('live_permit_held','已分配；分配许可仍持有',['allocation live; per-stage read/write protocol separate'],[ev_mem]),
            ('live_permit_relinquished','已放弃再分配许可；内存仍存活',['no subsequent alloc by this CTA; no deallocation implied'],[isa_permit,ev(A,172,179)]),
            ('caller_safe','该CTA释放前置条件成立',['local handshake path completed; leader drain premise established locally or inherited through peer acknowledgement'],[ev_tail]),
            ('dealloc_invoked','本CTA已进入free',['paired operation required; own enter alone does not prove deallocation'],[isa_pair]),
            ('deallocated','对应TMEM已由dealloc释放',['not an output-D completion or Host-visibility event'],[ev(A,147,168)])],
            pair_id=pair,num_columns=512,cooperative_group='cta_group::2',
            notes='Per-CTA resource instance participating in a paired allocation. Do not sum these records into an inferred single physical address space.')
        res(ptr,'contract.res.tmem_ptr',c,f'CTA {c} shared_storage.tmem_base_ptr',[
            ('unset','尚未取得分配结果',[],[ev(K,201)]),
            ('written','alloc写入本CTA的SMEM结果槽',['weak store; not yet a cross-warp publication claim'],[isa_alloc]),
            ('published','MMA已同步并向NamedBarrier到达',['epilogue still needs its own arrive_and_wait completion'],[ev(K,727,729)]),
            ('observed_by_epi','epilogue同步完成后读取base pointer',['TMEM Tensor binding remains a separate API call'],[ev(K,868,872)])],
            named_barrier_ref='contract.res.alloc_barrier',named_barrier_thread_count=160,
            notes='NamedBarrier is CTA-local. 32 MMA + 128 epilogue arrivals, not two CTA arrivals.')
        res(bar,'contract.res.dealloc_barrier',c,f'CTA {c} local tmem_dealloc phase 0',[
            ('uninitialized','未初始化',[],[ev_init]),('phase0_pending','阶段0等待32次到达',['arrive count=NumMMAThreads=32; only incoming peer MMA arrivals count'],[ev(K,560,565)]),
            ('phase0_satisfied','阶段0已满足',['remote predicate-true peer MMA arrival family contributed 32; wait is on this CTA local barrier'],[ev(B,370,384),ev(B,486,499)])],
            initial_phase=0,expected_arrivals=32,incoming_cta_rank=peer)
        res(named,'contract.res.alloc_barrier',c,f'CTA {c} allocation-result NamedBarrier / 160 threads',[
            ('neither','两类参与者均尚未收齐',['flags represent complete contribution families, not zero partial arrivals'],[ev(K,557)]),
            ('mma_only','MMA到达已收齐；epilogue尚未收齐',[],[ev(K,729)]),
            ('epi_only','epilogue已进入；MMA到达尚未收齐',[],[ev(K,870)]),
            ('both','32+128到达条件已满足',[],[ev(K,557),ev(B,285,308)]),
            ('observed','epilogue wait已成功返回',['publication observed in same CTA; no cross-CTA barrier identity'],[ev(K,870,872)])],
            expected_arrivals=160,initial_state=named+'.neither',state_model='Two unordered contribution-family flags followed by observation; no imposed MMA-first order.')
        group_init='init';group_alloc='allocation_'+pair;group_free='release_'+pair
        init=event(c,'mma','init_dealloc','初始化本CTA dealloc barrier',564,kind='api_call',
            guard={'expression':'(WarpCategory::MMA == warp_category) && !IsOverlappingAccum && has_mma_peer_cta && lane_predicate','configured':True},
            proof=[ev(B,355,358),ev(B,390,398),ev_init],
            effect={'kind':'barrier_init','resource_id':bar,'arrive_count':32,'phase':0,'thread_selector':'one elected lane of MMA warp'})
        iw=event(c,'mma','mma_init_wait','MMA等待cluster初始化完成',614,'contract.api.init_wait','contract.edge.kernel_init_wait',proof=[ev(K,577,614)])
        eiw=event(c,'epilogue','epi_init_wait','Epilogue等待cluster初始化完成',614,'contract.api.init_wait','contract.edge.kernel_init_wait',proof=[ev(K,577,614)])
        local_chain([init,iw],group_init,[ev(K,560,614)])
        alloc_ev=event(c,'mma','allocate','Allocator2Sm::allocate(512, &base)',727,'contract.api.allocate','contract.edge.kernel_allocate',
            effect={'kind':'tmem_allocation_and_result_store','resource_id':alloc,'pointer_resource_id':ptr,'may_block':True},proof=[ev(A,124,145),isa_alloc,isa_pair],
            preconditions=['paired CTA active','same logical full warp (warp 0) in both CTAs','same shared dst offset','uniform nCols=512'])
        sync=event(c,'mma','syncwarp','__syncwarp()',728,kind='builtin_call',proof=[ev_mem],name='__syncwarp',
            notes='Distinct warp synchronization call. It is not the allocation-result NamedBarrier.')
        pub=event(c,'mma','alloc_arrive','NamedBarrier::arrive()',729,'contract.api.alloc_arrive','contract.edge.kernel_alloc_arrive',
            effect={'kind':'named_barrier_arrival','resource_id':named,'cta_rank':c,'arrival_family_size':32},proof=[ev(B,222,224),ev(B,305,308)])
        mr=event(c,'mma','mma_read_base','MMA读取SMEM base pointer',730,kind='source_read',proof=[ev_mem])
        mb=event(c,'mma','mma_set_tmem','MMA绑定TMEM Tensor地址',731,'contract.api.set_tmem','contract.edge.kernel_set_tmem',proof=[ev(M,482,487)])
        mi=event(c,'mma','mma_init','CollectiveMma::mma_init',733,kind='api_call',proof=[ev(K,733,735),ev(M,550,562)])
        ew=event(c,'epilogue','alloc_wait','NamedBarrier::arrive_and_wait()',870,'contract.api.alloc_wait','contract.edge.kernel_alloc_wait',
            effect={'kind':'named_barrier_epilogue_arrival','resource_id':named,'arrival_family_size':128},
            proof=[ev(K,557),ev(B,210,212),ev(B,285,288)],
            postconditions=['same CTA allocation publication observed; 32+128 named-barrier participants satisfied'])
        er=event(c,'epilogue','epi_read_base','Epilogue读取SMEM base pointer',871,kind='source_read',proof=[ev(K,868,872)])
        eb=event(c,'epilogue','epi_set_tmem','Epilogue绑定TMEM Tensor地址',872,'contract.api.set_tmem','contract.edge.kernel_set_tmem_epilogue',proof=[ev(M,482,487)])
        local_chain([iw,alloc_ev,sync,pub,mr,mb,mi],group_alloc,[ev_mem])
        local_chain([eiw,ew,er,eb],group_alloc,[ev(K,868,872)])
        order(pub,'effect',ew,'return','named_barrier_completion_dependency',
            'All 32 MMA arrivals plus the 128 epilogue participants in this same CTA; no required ordering between arrive_and_wait.enter and MMA arrival.',
            [ev(K,557),ev(B,285,308)],group_alloc,join={'all_contributors_required':True,'mma_threads':32,'epilogue_threads':128})
        order(ew,'enter',ew,'effect','local_contribution_issue','Epilogue arrive_and_wait contributes its own arrivals before waiting.',[ev(B,210,212),ev(B,285,288)],group_alloc)
        order(ew,'effect',ew,'return','named_barrier_completion_dependency','Same CTA epilogue contribution family required as well as MMA publication.',[ev(K,557),ev(B,285,288)],group_alloc,join={'all_contributors_required':True,'epilogue_threads':128})
        done=event(c,'mma','work_loop_exit','MMA工作循环已退出',775,kind='control_boundary',proof=[ev(K,737,775)],
            anchors=['enter','return'],notes='A source control boundary, not a fabricated composite API. Work scheduling and MMA/commit calls remain in existing Dense contracts; not expanded by this lifetime draft.')
        gdc=event(c,'mma','launch_dependent_grids','launch_dependent_grids()',780,kind='api_call',proof=[ev(K,777,780)],
            notes='Separate source API; no TMEM free/drain effect assigned.')
        permit=event(c,'mma','release_permit','release_allocation_lock()',783,'contract.api.release_permit','contract.edge.kernel_release_permit',
            effect={'kind':'relinquish_allocation_permit','resource_id':alloc,'deallocates_memory':False},proof=[ev(A,172,179),isa_permit])
        head=[mi,done,gdc,permit]
        tail=None
        if leader:
            tail=event(c,'mma','acc_tail','PipelineUmmaAsync::producer_tail',788,'contract.api.acc_tail','contract.edge.kernel_acc_tail',
                guard={'expression':'is_participant.mma && !IsOverlappingAccum && is_mma_leader_cta','configured':True},
                proof=[ev(P,216,221),ev(PI,1126,1134),ev(PI,1184,1189),ev_count],
                wait_condition={'resource_ref':'contract.res.acc_empty','resource_id':f'tmem.pair{c//2}.acc_empty','owner_cta_rank':c,'stages':4,'state':'terminal accumulator_pipe_producer_state; locally copied and advanced 4 times',
                    'contributors':'128 epilogue threads from each of the two CTAs per released stage generation','expected_arrivals_per_stage':256},
                notes='Only leader calls this API. Initial/already-empty stages do not require invented consumer events. Pipeline-tail return is not equivalent to an arbitrary earlier consumer_release call return.')
            head.append(tail)
        ar1=event(c,'mma','arrive793','dealloc arrive(peer, !leader)',793,'contract.api.dealloc_arrive','contract.edge.kernel_dealloc_arrive',
            guard={'expression':'is_participant.mma && !IsOverlappingAccum && has_mma_peer_cta','configured':True},
            effect={'kind':'remote_barrier_arrival' if not leader else 'no_barrier_arrival','predicate':'!is_mma_leader_cta','predicate_value':not leader,
                    'target_resource_id':f'tmem.cta{peer}.dealloc_barrier','arrival_family_size':32 if not leader else 0},
            proof=[ev(B,381,385),ev(B,486,501)],binding_node_ref='contract.binding.dealloc_arrive_follower')
        wait=event(c,'mma','wait794','wait(local dealloc barrier, phase=0)',794,'contract.api.cluster_wait_member','contract.edge.kernel_dealloc_wait',
            guard={'expression':'is_participant.mma && !IsOverlappingAccum && has_mma_peer_cta','configured':True},
            proof=[ev(B,370,373),ev(B,410,427)],binding_node_ref='contract.api.dealloc_wait',
            wait_condition={'resource_id':bar,'phase':0,'required_arrivals':32,'source_actor_id':f'tmem.actor.cta{peer}.mma'},
            notes='Wait target is the local CTA barrier; it is not a remote-address wait or a direct wait for epilogue execution.')
        ar2=event(c,'mma','arrive795','dealloc arrive(peer, leader)',795,'contract.api.dealloc_arrive','contract.edge.kernel_dealloc_arrive_leader',
            guard={'expression':'is_participant.mma && !IsOverlappingAccum && has_mma_peer_cta','configured':True},
            effect={'kind':'remote_barrier_arrival' if leader else 'no_barrier_arrival','predicate':'is_mma_leader_cta','predicate_value':leader,
                    'target_resource_id':f'tmem.cta{peer}.dealloc_barrier','arrival_family_size':32 if leader else 0},
            proof=[ev(B,381,385),ev(B,486,501)],binding_node_ref='contract.binding.dealloc_arrive_leader')
        free=event(c,'mma','free','Allocator2Sm::free(base,512)',803,'contract.api.free','contract.edge.kernel_free',
            effect={'kind':'tmem_deallocation','resource_id':alloc,'requires_cooperative_peer_call':f'tmem.cta{peer}.free','does_not_prove':'output store completion or peer synchronization by the free API itself'},
            proof=[ev(A,147,169),isa_pair],preconditions=['owns the requested allocation','nCols=512 uniform across the full MMA warp','peer active and participating in same logical warp','local caller handshake completed; accumulator drain premise established'])
        local_chain(head+[ar1,wait,ar2,free],group_free,[ev_tail,ev(K,775,780)])
        fence=event(c,'epilogue','read_fence','fence_view_async_tmem_load()',887,'contract.api.tmem_read_fence','contract.edge.epi_wait_tmem_reads',path=E,
            guard={'expression':'is_participant.epilogue && scheduler.compute_epilogue(work_tile_info) && do_acc_release','configured':'guarded_runtime_family',
                   'context':'Composition of caller and callee guards; ReuseTmem=false makes do_acc_release select the last subtile.'},
            proof=[ev(E,824,832),ev(E,880,889),ev(B,923,930)],
            repeat={'unit':'each relevant accumulator stage generation','dynamic_count':'scheduler-dependent'},postconditions=['this thread prior TMEM reads completed before its stage release'])
        rel=event(c,'epilogue','acc_release','PipelineUmmaAsync::consumer_release',888,'contract.api.acc_release','contract.edge.epi_release_acc',path=E,
            guard={'expression':'is_participant.epilogue && scheduler.compute_epilogue(work_tile_info) && do_acc_release','configured':'guarded_runtime_family',
                   'context':'Same caller/callee guard instance as the immediately preceding read_fence call.'},
            effect={'kind':'accumulator_empty_arrival','contract_resource_ref':'contract.res.acc_empty','target_resource_id':f'tmem.pair{c//2}.acc_empty','target_cta_rank':c&~1,'arrival_family_size':128,'stage':'acc_pipe_consumer_state.index()','phase':'matching current consumer generation'},
            proof=[ev(P,241,276),ev(B,905,916)],repeat={'unit':'each relevant accumulator stage generation','dynamic_count':'scheduler-dependent'})
        outtail=event(c,'epilogue','store_tail','CollectiveEpilogue::store_tail',949,'contract.api.ep_store_tail','contract.edge.kernel_epi_tail',
            guard={'expression':'is_participant.epilogue && do_tail_store','configured':'guarded_runtime_family'},
            proof=[ev(K,944,952)],notes='Included only to show the absence of a TMEM-free versus output-tail ordering claim.')
        local_chain([eb,fence,rel,outtail],group_free,[ev(E,880,889),ev(K,910,952)])
        # Anchors are explicit. effect is a semantic effect, not a synonymous API
        # return or a promise of hardware completion. No implicit array ordering.
        for x in [init,alloc_ev,pub,permit,ar1,ar2,free,rel]:
            order(x,'enter',x,'effect','event_effect','Only if the invocation guard is active; predicate-false effect is explicitly no arrival.',
                  next(e['evidence_refs'] for e in events if e['id']==x), group_init if x==init else group_alloc if x in (alloc_ev,pub) else group_free)
        transition(alloc,'absent','live_permit_held',alloc_ev,'effect','allocation succeeds', [ev(A,135,145),isa_alloc],['valid paired warp participation'])
        transition(alloc,'live_permit_held','live_permit_relinquished',permit,'effect','release permission executes; current allocation remains live',[isa_permit,ev(A,174,176)])
        transition(alloc,'live_permit_relinquished','caller_safe',ar2,'return','793→794→795 path completed locally; only predicate-true arrival sends a signal',[ev_tail],['leader tail has returned directly (leader) or its premise is inherited from leader795 (peer)'])
        transition(alloc,'caller_safe','dealloc_invoked',free,'enter','source free call reached',[ev(K,803)],['proper peer warp participation remains required'])
        transition(alloc,'dealloc_invoked','deallocated',free,'effect','paired deallocation operation takes effect',[ev(A,159,166),isa_pair],['not inferred from free.enter alone','does not imply relative API-return order between peer CTAs'])
        transition(ptr,'unset','written',alloc_ev,'effect','alloc writes local shared result slot',[isa_alloc,ev(A,135,141)])
        transition(ptr,'written','published',pub,'effect','MMA __syncwarp precedes barrier arrival',[ev(K,727,729)])
        transition(ptr,'published','observed_by_epi',er,'return','same-CTA arrive_and_wait returned, then pointer read completed',[ev(K,868,872)])
        transition(bar,'uninitialized','phase0_pending',init,'effect','one elected MMA lane initializes count32 phase0',[ev_init])
        for fr,to,event_,condition in [('neither','mma_only',pub,'MMA contribution family completes before epilogue family'),
                                       ('epi_only','both',pub,'epilogue contribution family already complete'),
                                       ('neither','epi_only',ew,'epilogue contribution family completes before MMA family'),
                                       ('mma_only','both',ew,'MMA contribution family already complete')]:
            transition(named,fr,to,event_,'effect',condition,[ev(K,557),ev(K,729),ev(K,870),ev(B,285,308)])
        transition(named,'both','observed',ew,'return','bar.sync completion observed after both required participant families',[ev(B,285,288)])
    for lead in (0,2):
        peer=lead+1;group='release_pair'+str(lead//2)
        empty=f'tmem.pair{lead//2}.acc_empty'
        res(empty,'contract.res.acc_empty',lead,f'CTA{lead}/{peer} accumulator.empty[4] at leader CTA{lead}',[
            ('live_ring','4槽环；各槽可能在用或已空',['matching index/phase and prior full/empty protocol required'],[ev_count]),
            ('draining','leader按state副本依次检查4槽',['not only the last slot; caller state not advanced by this by-value tail'],[ev(P,219,220),ev(PI,1129,1133)]),
            ('drained','4个对应阶段均已观测为空',['all required outstanding reader releases accounted; no output-D completion inferred'],[ev(PI,1185,1188),ev(K,785,788)])],
            initial_state=empty+'.live_ring',stage_count=4,arrivals_per_stage=256,pair_id=f'pair{lead//2}',
            notes='Projection of contract.protocol.accumulator into lifetime termination, not a replacement for its per-generation full/empty state machine.')
        transition(empty,'live_ring','draining',f'tmem.cta{lead}.acc_tail','enter','leader invokes producer_tail with terminal producer state',[ev(K,788),ev(P,219,220)])
        transition(empty,'draining','drained',f'tmem.cta{lead}.acc_tail','return','all four producer_acquire waits successfully returned',[ev(PI,1129,1133),ev(PI,1185,1188)],['initially empty stages require no fictional release event'])
        first=f'tmem.cta{peer}.arrive793';ack=f'tmem.cta{lead}.arrive795'
        order(first,'effect',f'tmem.cta{lead}.wait794','return','barrier_completion_dependency',
            'Leader local phase0 wait completes after all 32 predicate-true peer MMA arrivals.',[ev(K,791,795),ev(B,410,427),ev(B,486,499)],group,
            join={'all_contributors_required':True,'required_arrivals':32,'target_resource_id':f'tmem.cta{lead}.dealloc_barrier'})
        order(ack,'effect',f'tmem.cta{peer}.wait794','return','barrier_completion_dependency',
            'Peer local phase0 wait completes after all 32 leader MMA acknowledgement arrivals; leader tail premise is thereby propagated.',[ev_tail,ev(B,410,427),ev(B,486,499)],group,
            join={'all_contributors_required':True,'required_arrivals':32,'target_resource_id':f'tmem.cta{peer}.dealloc_barrier'})
        for c in (lead,peer):
            order(f'tmem.cta{c}.acc_release','effect',f'tmem.cta{lead}.acc_tail','return','accumulator_drain_dependency',
                'For each still-live stage generation encountered by the tail, the required leader+peer epilogue arrivals must have reached leader empty barrier. Already-empty stages need no invented release.',
                [ev_count,ev(P,216,220),ev(P,271,276),ev(PI,1129,1133),ev(PI,1185,1188)],group,
                join={'all_contributors_required':True,'per_cta_threads':128,'pair_total_threads':256,'stage_count':4})
        transition(f'tmem.cta{lead}.dealloc_barrier','phase0_pending','phase0_satisfied',first,'effect','all32 peer contributions applied to leader local barrier',[ev(K,793),ev(B,486,499)])
        transition(f'tmem.cta{peer}.dealloc_barrier','phase0_pending','phase0_satisfied',ack,'effect','all32 leader contributions applied to peer local barrier',[ev(K,795),ev(B,486,499)])
    # No renderer-implied event ordering: even enter -> return is explicit.
    # The edge means "if this invocation returns, its encounter precedes that
    # return", not progress or a hardware-memory-completion promise.
    for item in events:
        group=next(o['view_group'] for o in orders if item['id'] in (o['before']['event_id'],o['after']['event_id']))
        order(item['id'],'enter',item['id'],'return','local_invocation_order',
              'For the same executing thread and invocation, if it returns. This edge does not assert bounded progress or that an asynchronous external effect has completed.',
              item['evidence_refs'],group,quantifier='per corresponding actor thread; control boundary for non-API checkpoints')
        if item.get('effect',{}):
            effect_kind=item['effect']['kind']
            if effect_kind in ('relinquish_allocation_permit','barrier_init','named_barrier_arrival','no_barrier_arrival'):
                order(item['id'],'effect',item['id'],'return','local_effect_before_return',
                      'Only the stated local effect / contribution issue, not a remote barrier completion or a deallocation effect.',
                      item['evidence_refs'],group,quantifier='per corresponding actor thread')
    # Source events absent from contracts still identify one exact API; no fake
    # global entity IDs are manufactured merely for the draft.
    for item in events:
        if item['id'].endswith('.init_dealloc'):
            item['api_source']={'path':B,'line':356,'qualified_name':'cutlass::arch::ClusterBarrier::init',
                                'signature':'void init(uint32_t arrive_count) const'}
        if item['id'].endswith('.mma_init'):
            item['api_source']={'path':M,'line':552,'qualified_name':'cutlass::gemm::collective::CollectiveMma::mma_init',
                                'signature':'template<class TmemStorage> CUTLASS_DEVICE auto mma_init([[maybe_unused]] TmemStorage tmem_storage, TensorStorage& shared_tensors) const'}
            raw=(ROOT/'snapshot'/K).read_bytes()
            expr=b'collective_mainloop.mma_init(\n        tmem_storage,\n        shared_storage.tensors.mainloop)'
            start=raw.index(expr)
            item['callsite'].update({'end_line':735,'start_byte':start,'end_byte':start+len(expr),
                                    'source_expression':expr.decode(),'raw_source_expression':expr.decode(),'caller':'contract.api.kernel'})
        if item['id'].endswith('.launch_dependent_grids'):
            item['api_source']={'path':'include/cutlass/arch/grid_dependency_control.h','line':85,
                                'qualified_name':'cutlass::arch::launch_dependent_grids','signature':'CUTLASS_DEVICE void launch_dependent_grids()'}
            item['evidence_refs'].append(ev('include/cutlass/arch/grid_dependency_control.h',84,91))
    for group,title in [('init','TMEM：初始化与全局初始化前提'),('allocation_pair0','CTA0/1：分配结果发布'),('release_pair0','CTA0/1：排空、握手与释放'),('allocation_pair1','CTA2/3：分配结果发布'),('release_pair1','CTA2/3：排空、握手与释放')]:
        relevant=[o for o in orders if o['view_group']==group]
        ids=sorted({p['event_id'] for o in relevant for p in (o['before'],o['after'])})
        views.append({'id':'tmem.view.'+group,'kind':'partial_order','title':title,'event_ids':ids,'order_ids':[o['id'] for o in relevant]})
    for resource in resources:
        ids=[t['id'] for t in transitions if t['resource_id']==resource['id']]
        views.append({'id':'tmem.view.state.'+resource['id'],'kind':'state','title':resource['label'],'resource_id':resource['id'],'transition_ids':ids})
    by_event={e['id']:e for e in events};by_res={r['id']:r for r in resources}
    for o in orders:
        for p in (o['before'],o['after']):assert p['anchor'] in by_event[p['event_id']]['anchors']
    for t in transitions:
        assert t['from'] in {s['id'] for s in by_res[t['resource_id']]['states']}
        assert t['to'] in {s['id'] for s in by_res[t['resource_id']]['states']}
        assert t['trigger']['anchor'] in by_event[t['trigger']['event_id']]['anchors']
    evidence_ids={e['id'] for e in evidence}
    for collection in (events,orders,transitions):
        assert all(set(x['evidence_refs'])<=evidence_ids for x in collection)
    assert {o['id'] for o in orders}=={i for v in views for i in v.get('order_ids',[])}
    assert {t['id'] for t in transitions}=={i for v in views for i in v.get('transition_ids',[])}
    paths={e['path'] for e in evidence if 'path' in e}
    scope=json.loads((ROOT/'data/scope.json').read_text());hashes={x['path']:x['sha256'] for x in scope['files']}
    assert all(sha(ROOT/'snapshot'/p)==hashes[p] for p in paths)
    data={'schema_version':1,'id':'dense.protocol_draft.tmem_lifetime','status':'source_semantics_draft_not_runtime_trace',
        'title':'Dense FP16 2SM：TMEM生命周期的actor化偏序与状态',
        'contract_protocol_ref':'contract.protocol.tmem_lifetime','snapshot_commit':scope['commit'],
        'configuration':{'recipe_arch_tag':'Sm100','binary_target':'sm_110a','cluster_shape':[2,2,1],
            'cta_pairs':[[0,1],[2,3]],'pair_mapping':'leader=(rank % 2)==0; peer=rank^1','accumulator_stages':4,'IsOverlappingAccum':False,
            'NumMMAThreads':32,'NumEpilogueThreads':128,'allocation_named_barrier_count':160,'deallocation_barrier_count':32,'accumulator_empty_count':256},
        'semantics':{'array_order_is_not_execution_order':True,'implicit_partial_order':False,
            'event_anchors':'enter: API/control encounter; return: successful local completion of that invocation; effect: stated semantic effect only, not automatically API return. Events are thread families, not a collapse of different CTA actors.',
            'barrier_join':'A dependency into wait.return requires all contributions named by join; a single thread arrival is insufficient.',
            'liveness':'No bounded completion, scheduling fairness or observed GPU execution claimed.',
            'pair_independence':'No order between the two pair lifetimes is asserted. Cluster initialization remains a shared prerequisite, not an invented pair0-before-pair1 order.'},
        'actors':actors,'resources':resources,'events':events,'partial_order':orders,'state_transitions':transitions,'views':views,'evidence':evidence,
        'preconditions':[
            {'id':'tmem.precondition.init','claim':'Pipeline/deallocation barrier initialization is cluster-visible after pipeline_init_arrive_relaxed579 and pipeline_init_wait614; all participating roles must honor it. The draft shows relevant MMA/epilogue calls but does not claim to enumerate scheduler/load actor initialization.', 'evidence_refs':[ev(K,577,614)]},
            {'id':'tmem.precondition.stage_protocol','claim':'Accumulator pipeline initialized with4 stages and matching phase/index; leader MMA commit/full wait and both epilogues correctly discharge every used stage. This lifetime draft references rather than re-proves the full accumulator protocol.', 'contract_protocol_ref':'contract.protocol.accumulator','evidence_refs':[ev_count,ev(K,751,775),ev(E,824,889)]},
            {'id':'tmem.precondition.warp','claim':'One fully active warp per peer CTA, same logical warp id, uniform512-column request and corresponding SMEM result slot; peer remains active for paired alloc/dealloc.', 'evidence_refs':[ev(A,124,166),isa_pair]}],
        'implementation_paths':[
            {'id':'tmem.impl.named_arrive','steps':[{'api_node_ref':'contract.api.alloc_arrive','callsite':{'path':B,'line':224},'callee':'NamedBarrier::arrive_internal(uint32_t,uint32_t)','callee_line':305,'effect_line':308}], 'evidence_refs':[ev(B,222,224),ev(B,305,308)]},
            {'id':'tmem.impl.named_wait','steps':[{'api_node_ref':'contract.api.alloc_wait','callsite':{'path':B,'line':212},'callee':'NamedBarrier::arrive_and_wait_internal(uint32_t,uint32_t)','callee_line':285,'effect_line':287}], 'evidence_refs':[ev(B,210,212),ev(B,285,288)]},
            {'id':'tmem.impl.cluster_wait','steps':[{'api_node_ref':'contract.api.cluster_wait_member','callsite':{'path':B,'line':372},'callee_node_ref':'contract.api.cluster_wait_static','callee_line':410,'effect_line':420}], 'evidence_refs':[ev(B,370,373),ev(B,410,427)]},
            {'id':'tmem.impl.remote_arrive','steps':[{'api_node_ref':'contract.api.dealloc_arrive','callsite':{'path':B,'line':384},'callee':'ClusterBarrier::arrive(ValueType const*,uint32_t,uint32_t)','callee_line':486,'predicate_line':489,'remote_map_line':493,'effect_line':494}], 'evidence_refs':[ev(B,381,385),ev(B,486,501)]},
            {'id':'tmem.impl.acc_tail','steps':[{'api_node_ref':'contract.api.acc_tail','callsite':{'path':P,'line':220},'callee':'PipelineAsync<4>::producer_tail(PipelineState)','callee_path':PI,'callee_line':1129},
                {'callsite':{'path':PI,'line':1131},'callee':'PipelineAsync<4>::producer_acquire(PipelineState,ProducerToken)','callee_line':1110,'repeat':'count=0..3, ++state each iteration'},
                {'callsite':{'path':PI,'line':1111},'callee':'PipelineAsync<4>::producer_acquire(uint32_t,uint32_t,ProducerToken)','callee_line':1185},
                {'callsite':{'path':PI,'line':1188},'callee_node_ref':'contract.api.cluster_wait_member','guard':'barrier_token==WaitAgain (default token on this tail path)'}], 'evidence_refs':[ev(P,216,220),ev(PI,1109,1112),ev(PI,1129,1133),ev(PI,1185,1188)]},
            {'id':'tmem.impl.acc_release','steps':[{'api_node_ref':'contract.api.acc_release','callsite':{'path':P,'line':245},'callee_node_ref':'contract.api.acc_release_stage'},
                {'api_node_ref':'contract.api.acc_release_stage','callsite':{'path':P,'line':275},'callee_node_ref':'contract.api.acc_release_sm0','effect_path':B,'effect_line':912}], 'evidence_refs':[ev(P,241,276),ev(B,905,916)]}],
        'inactive_branches':[{'id':'tmem.inactive.overlap_wait799','call_edge_ref':'contract.edge.kernel_dealloc_wait_overlapping','binding_node_ref':'contract.binding.dealloc_wait_overlapping','guard':'IsOverlappingAccum','configured':False,'evidence_refs':[ev(K,798,800)]},
            {'id':'tmem.inactive.epilogue_dealloc_arrivals','guard':'IsOverlappingAccum','configured':False,'claim':'939 remote arrive and941 local arrive are not the active non-overlap handshake and must not supply its32 arrivals.', 'evidence_refs':[ev(K,936,942)]}],
        'not_proven_orders':[
            {'id':'tmem.not_ordered.free_calls','event_pairs':[[f'tmem.cta{x}.free',f'tmem.cta{x+1}.free'] for x in (0,2)],'anchors':['enter','return'],'claim':'No total order of leader and peer free API encounters or returns. Both must satisfy their caller path and cooperative instruction requirements.'},
            {'id':'tmem.not_ordered.wait_enter','claim':'peer793 can signal before peer enters wait794; leader wait return does not prove peer wait enter has happened.'},
            {'id':'tmem.not_ordered.alloc_wait_enter','claim':'Epilogue may enter its allocation wait before MMA alloc/arrive. Only its successful return is constrained by publication.'},
            {'id':'tmem.not_ordered.output_tail','claim':'TMEM free is not ordered after both output store_tail completions here; TMEM source-read release can precede fusion/SMEM/TMA output work.'},
            {'id':'tmem.not_ordered.pairs','claim':'No lifetime order between CTA pair0/1 and pair2/3 is inferred from array placement or CTA rank.'}],
        'issues':[{'id':'tmem.issue.nonblocking_comment','kind':'source_comment_spec_conflict','status':'resolved_for_draft_by_normative_semantics','claim':'Allocator2Sm allocate comment125 calls it non-blocking; pinned PTX9.0 explicitly allows capacity blocking. No runtime guarantee is based on the comment.','evidence_refs':[ev(A,124,128),isa_alloc]},
            {'id':'tmem.issue.runtime','status':'not_verified','claim':'No GPU execution, race test, progress/fairness proof or output numerical validation performed.'}],
        'provenance':{'contracts_path':'data/modules/dense_fp16/contracts.json','contracts_sha256':hashlib.sha256(contract_bytes).hexdigest(),
            'scope_sha256':sha(ROOT/'data/scope.json'),'generator_path':str(Path(__file__).relative_to(ROOT)),'generator_sha256':sha(Path(__file__)),
            'reuse_policy':'Existing nodes/edges are references only; contracts.json and renderer are not edited.'}}
    source_specs={
        'tmem.impl.named_arrive':[(B,222,'cutlass::arch::NamedBarrier::arrive')],
        'tmem.impl.named_wait':[(B,210,'cutlass::arch::NamedBarrier::arrive_and_wait')],
        'tmem.impl.cluster_wait':[(B,371,'cutlass::arch::ClusterBarrier::wait')],
        'tmem.impl.remote_arrive':[(B,383,'cutlass::arch::ClusterBarrier::arrive')],
        'tmem.impl.acc_tail':[(P,219,'cutlass::PipelineUmmaAsync::producer_tail'),
            (PI,1129,'cutlass::PipelineAsync::producer_tail'),(PI,1110,'cutlass::PipelineAsync::producer_acquire'),
            (PI,1185,'cutlass::PipelineAsync::producer_acquire')],
        'tmem.impl.acc_release':[(P,242,'cutlass::PipelineUmmaAsync::consumer_release'),
            (P,272,'cutlass::PipelineUmmaAsync::consumer_release_2x1SM')],
    }
    callee_names={(B,305):'cutlass::arch::NamedBarrier::arrive_internal',
        (B,285):'cutlass::arch::NamedBarrier::arrive_and_wait_internal',
        (B,410):'cutlass::arch::ClusterBarrier::wait',(B,371):'cutlass::arch::ClusterBarrier::wait',
        (B,486):'cutlass::arch::ClusterBarrier::arrive',
        (PI,1129):'cutlass::PipelineAsync::producer_tail',
        (PI,1110):'cutlass::PipelineAsync::producer_acquire',(PI,1185):'cutlass::PipelineAsync::producer_acquire',
        (P,272):'cutlass::PipelineUmmaAsync::consumer_release_2x1SM',(B,907):'cutlass::arch::umma_arrive_2x1SM_sm0'}
    for implementation in data['implementation_paths']:
        for index,step in enumerate(implementation['steps']):
            step['id']=implementation['id']+f'.call{index+1}'
            step['kind']='calls'
            sp,sl,sq=source_specs[implementation['id']][index]
            step['source']={'path':sp,'line':sl,'qualified_name':sq}
            if step.get('api_node_ref'):step['source']['node_ref']=step['api_node_ref']
            if step.get('callee_node_ref'):
                callee=nodes[step['callee_node_ref']]
                step['callee_source']={'path':callee['path'],'line':callee['line'],
                    'qualified_name':callee.get('qualified_name',callee['name'])}
            else:
                path=step.get('callee_path',step['callsite']['path'])
                step['callee_source']={'path':path,'line':step['callee_line'],'qualified_name':step['callee']}
            cs=step['callee_source'];cs['qualified_name']=callee_names[(cs['path'],cs['line'])]
            call=step['callsite'];line=call['line'];raw=(ROOT/'snapshot'/call['path']).read_bytes();physical=raw.splitlines(keepends=True)
            expression=physical[line-1].strip();start=sum(map(len,physical[:line-1]))+physical[line-1].index(expression)
            call.update({'start_line':line,'end_line':line,'start_byte':start,'end_byte':start+len(expression),'source_expression':expression.decode()})
            # These were implementation-location hints, not extra C++ calls.
            for key in ('effect_line','effect_path','remote_map_line','predicate_line'):step.pop(key,None)
    terminal_effects={
        'tmem.impl.named_arrive':(B,305,'cutlass::arch::NamedBarrier::arrive_internal',308,'bar.arrive'),
        'tmem.impl.named_wait':(B,285,'cutlass::arch::NamedBarrier::arrive_and_wait_internal',287,'bar.sync'),
        'tmem.impl.cluster_wait':(B,410,'cutlass::arch::ClusterBarrier::wait',420,'mbarrier.try_wait.parity.shared::cta.b64'),
        'tmem.impl.remote_arrive':(B,486,'cutlass::arch::ClusterBarrier::arrive',494,'mbarrier.arrive.shared::cluster.b64'),
        'tmem.impl.acc_release':(B,907,'cutlass::arch::umma_arrive_2x1SM_sm0',912,'mbarrier.arrive.shared::cluster.b64'),
    }
    for implementation in data['implementation_paths']:
        if implementation['id'] in terminal_effects:
            path,line,name,opline,opcode=terminal_effects[implementation['id']]
            implementation['steps'].append({'id':implementation['id']+'.instruction','kind':'hardware_effect',
                'source':{'path':path,'line':line,'qualified_name':name},'opcode':opcode,
                'instruction_source':{'path':path,'start_line':opline,'end_line':opline},
                'guard':'pred' if implementation['id']=='tmem.impl.remote_arrive' else 'the corresponding architecture-enabled implementation branch',
                'evidence_refs':[ev(path,opline)],
                'notes':'Inline PTX effect, not a C++ callable. Remote arrival uses mapa.shared::cluster at493 under predicate489.' if implementation['id']=='tmem.impl.remote_arrive' else 'Inline PTX effect, not a C++ callable.'})
    def acyclic(extra=()):
        links=[(tuple(o['before'].values()),tuple(o['after'].values())) for o in orders]+list(extra)
        successors={};degree={}
        for a,b in links:
            successors.setdefault(a,set()).add(b);degree.setdefault(a,0);degree.setdefault(b,0)
        for bs in successors.values():
            for b in bs:degree[b]+=1
        todo=[a for a,n in degree.items() if n==0];visited=0
        while todo:
            a=todo.pop();visited+=1
            for b in successors.get(a,()):
                degree[b]-=1
                if degree[b]==0:todo.append(b)
        return visited==len(degree)
    assert acyclic(),'Partial order has a cycle'
    counterexamples=[]
    for lead in (0,2):
        peer=lead+1
        a=(f'tmem.cta{lead}.free','enter');b=(f'tmem.cta{peer}.free','enter')
        assert acyclic([(a,b)]) and acyclic([(b,a)]),'A total free-call order was accidentally imposed'
        counterexamples.append({'id':f'tmem.counterexample.pair{lead//2}.free_arrival_order',
            'first_alternative':[point(a[0],a[1]),point(b[0],b[1])],
            'second_alternative':[point(b[0],b[1]),point(a[0],a[1])],
            'both_partial_order_extensions_are_acyclic':True,
            'claim':'These are possible API encounter orders under the recorded source partial order, not independently completed single-CTA deallocations.'})
    data['counterexample_checks']=counterexamples
    data['validation']={'partial_order_acyclic':True,'opposite_free_enter_orders_not_forbidden':True,
        'all_orders_in_views':True,'all_transitions_in_views':True,'existing_node_edge_refs_checked':True,
        'source_hashes_checked':True,'does_not_validate':'Actual GPU progress, memory safety under invalid inputs, ISA implementation or numerical output'}
    # ev() calls in supplementary sections append to the same evidence list.
    assert contracts_path.read_bytes()==contract_bytes,'Contracts changed during generation; retry after freeze'
    (HERE/'tmem_lifetime.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'events':len(events),'actors':len(actors),'orders':len(orders),'resources':len(resources),'transitions':len(transitions),'views':len(views),'output':str(HERE/'tmem_lifetime.json')},ensure_ascii=False))

if __name__=='__main__':main()
