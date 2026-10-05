"""Explicit event-instance domains and logical validity for TMEM draft v2."""
from collections import defaultdict

OPERATION_EFFECTS={'tmem_allocation_and_result_store','local_dealloc_instruction_issued'}

def apply_quantified_semantics(data,isa_pair_evidence):
    actors={a['id']:a for a in data['actors']}
    events={e['id']:e for e in data['events']}
    for e in events.values():
        actor=actors[e['actor_id']];elected=e['id'].endswith('.init_dealloc')
        domain={'kind':'actor_thread_invocations','actor_id':actor['id'],
                'selection':'elected_lane' if elected else 'all_role_threads',
                'cardinality':1 if elected else actor['thread_count'],
                'iteration_keys':['executed_stage_generation'] if e.get('repeat') else ['once'],
                'elected_lane_binding':'lane_predicate from elect_one_sync; no fixed lane number asserted' if elected else None}
        e['instance_domain']=domain
        e['anchor_quantifiers']={a:{'kind':'per_thread_instance','domain_ref':e['id']+'.instance_domain'} for a in e['anchors']}
        if (e.get('effect') or {}).get('kind') in OPERATION_EFFECTS:
            e['anchor_quantifiers']['effect']={'kind':'local_warp_operation','operation_id':e['id']+'.operation',
                'participant_domain_ref':e['id']+'.instance_domain','required_local_threads':actor['thread_count'],
                'does_not_merge_peer_cta':True,'does_not_define_peer_return_order':True}
    for o in data['partial_order']:
        before=events[o['before']['event_id']];after=events[o['after']['event_id']]
        bkind=before['anchor_quantifiers'][o['before']['anchor']]['kind']
        akind=after['anchor_quantifiers'][o['after']['anchor']]['kind']
        relation=o['relation']
        if relation in ('barrier_completion_dependency','named_barrier_completion_dependency','accumulator_drain_dependency'):
            kind='all_sources_to_each_target'
        elif bkind=='local_warp_operation':
            kind='operation_to_each_target' if akind=='per_thread_instance' else 'operation_to_operation'
        elif akind=='local_warp_operation':
            kind='all_sources_to_operation'
        else:
            assert before['actor_id']==after['actor_id'],o['id']
            kind='pointwise_same_thread'
        o['quantifier']={'kind':kind,'source_domain_ref':before['id']+'.instance_domain',
            'target_domain_ref':after['id']+'.instance_domain',
            'source_cardinality':before['instance_domain']['cardinality'],
            'target_cardinality':after['instance_domain']['cardinality'],
            'instance_matching':'same thread_id; executed source/target control-path instances' if kind=='pointwise_same_thread' else 'only the explicit join or local-operation mapping, not a default family barrier',
            'iteration_matching': 'same executed_stage_generation for fence/release; once-to-each or each-to-terminal only where source caller order states so'}
        if relation=='accumulator_drain_dependency':
            o['quantifier']['iteration_matching']='all actually outstanding stage generations observed by this terminal tail; no fabricated release for initially empty slots'
    for resource in data['resources']:
        resource.setdefault('initial_state',resource['states'][0]['id'])
        resource['state_semantics']={'kind':'logical_obligation_projection','temporal_state_machine':False,
            'history_facts_allowed':True,'adds_happens_before_edges':False,
            'inference':'least fixed point of state_transitions.derivation all_of rules over observed anchor history',
            'state_facts_are_nonexclusive':True,'initial_fact':resource['initial_state'],
            'meaning':'States are nonexclusive proof facts, not current physical snapshots. Historical from-facts are not consumed. A trigger may have been observed earlier than another prerequisite; deriving its conclusion introduces no timing edge.'}
        for state in resource['states']:
            suffix=state['id'].rsplit('.',1)[-1]
            labels={'absent':'本次CTA调用尚无allocation记录','live_permit_held':'已记录本侧allocation结果',
                'live_permit_relinquished':'已记录CTA放弃再分配许可','caller_safe':'已核查本侧各线程释放前置路径',
                'dealloc_invoked':'已记录本侧各线程进入free','local_dealloc_recorded':'已记录本侧warp dealloc操作（非配对完成时刻）',
                'unset':'尚无分配结果写记录','written':'已记录分配结果写','published':'已记录MMA发布贡献',
                'observed_by_epi':'已记录epilogue同步后同址读取','uninitialized':'尚无dealloc barrier初始化记录',
                'phase0_pending':'已记录本地phase0初始化','phase0_satisfied':'已记录所需peer到达贡献',
                'neither':'尚未登记完整贡献族','mma_only':'已记录MMA完整贡献（不判断谁先）',
                'epi_only':'已记录epilogue完整贡献（不判断谁先）','both':'双方完整贡献均已记录',
                'observed':'已记录epilogue等待成功返回','live_ring':'已记录4槽pipeline前置',
                'draining':'已记录leader尾部排空调用','drained':'已记录leader排空返回/4槽观察结果'}
            state['label']=labels[suffix]
            state['fact_interpretation']='Monotone historical proof fact; this label does not assert a physical time or absence of subsequently recorded facts.'
            if suffix in ('mma_only','epi_only','neither'):
                state['invariants']=['only the named contribution fact is added here; no negative fact about other participants is inferred']
    for t in data['state_transitions']:
        e=events[t['trigger']['event_id']];anchor=t['trigger']['anchor']
        q=e['anchor_quantifiers'][anchor]
        if q['kind']=='local_warp_operation':kind='single_local_operation'
        elif (e.get('effect') or {}).get('kind')=='relinquish_allocation_permit' and anchor=='effect':kind='any_instance'
        else:kind='all_instances'
        t['trigger_quantifier']={'kind':kind,'domain_ref':e['id']+'.instance_domain',
            'cardinality':1 if kind=='single_local_operation' else e['instance_domain']['cardinality'],
            'iteration_scope':'all relevant executed stage generations' if e.get('repeat') else 'this once-only invocation family',
            'evaluation':'logical history facts; not a physical transition timestamp'}
        t['derivation']={'id':t['id']+'.derive','kind':'monotone_history_fact_rule',
            'all_of':[{'kind':'state_fact','resource_id':t['resource_id'],'state_id':t['from']},
                      {'kind':'observed_anchor_history','point':dict(t['trigger']),'quantifier':dict(t['trigger_quantifier'])}],
            'conclusion':{'kind':'state_fact','resource_id':t['resource_id'],'state_id':t['to']},
            'consumes_from_fact':False,'adds_temporal_order':False}
        if e['id'].endswith(('.alloc_wait','.alloc_arrive')):
            t['condition']='The from-proof and the qualified contribution/wait anchor are both recorded in history; no physical order between MMA and epilogue family arrivals is inferred.'
    rules=[]
    for lead in (0,2):
        pair=lead//2
        rules.append({'id':f'tmem.validity.pair{pair}.deallocation_participation',
            'kind':'logical_protocol_acceptance','explicitly_not_temporal':True,
            'acceptance_state':'paired_deallocation_participation_validated',
            'all_of':[
                {'kind':'qualified_anchor_observed','point':{'event_id':f'tmem.cta{c}.free','anchor':'effect'},
                 'quantifier':{'kind':'single_local_operation','participant_domain_ref':f'tmem.cta{c}.free.instance_domain','required_local_threads':32}}
                for c in (lead,lead+1)]+[
                {'kind':'qualified_anchor_observed','point':{'event_id':f'tmem.cta{c}.free','anchor':'enter'},
                 'quantifier':{'kind':'all_instances','domain_ref':f'tmem.cta{c}.free.instance_domain','cardinality':32}}
                for c in (lead,lead+1)]+[
                {'kind':'source_precondition','id':'uniform_512_columns_same_logical_warp','required':True,
                 'claim':'Both local operation records bind512 columns, logical warp0, the correct allocation and peer-active precondition.'}],
            'resource_ids':[f'tmem.cta{c}.allocation' for c in (lead,lead+1)],
            'evidence_refs':[isa_pair_evidence],
            'not_claimed':['physical paired-deallocation completion timestamp','peer.free.enter before local.free.return','order between peer free returns','output-D completion']})
    data['validity_rules']=rules
    data['semantics']['composition']={'temporal_constraints':'Only quantified partial_order. Flattening a pointwise edge to all-to-all is invalid.',
        'state_projection':'Resource states are logical summaries derived from quantified historical anchor facts; they do not add event order.',
        'acceptance':'A completed lifetime claim additionally requires every applicable validity_rule all_of. Mere DAG acyclicity or a local free operation is insufficient.',
        'operation_mapping':'alloc/free effect is one local warp operation with32 participating thread invocations, not32 independent resource stores/frees and not one merged peer-pair event.'}

def expand_graph(data):
    """Concrete one-generation witness model; elected lane0 is alpha-renaming only.

    Does not claim lane0 was elected in the real kernel. A runtime generation
    family is represented by one corresponding generation for bounded tests.
    """
    events={e['id']:e for e in data['events']}
    def domain(e,anchor):
        if e['anchor_quantifiers'][anchor]['kind']=='local_warp_operation':return ['operation']
        return list(range(e['instance_domain']['cardinality']))
    graph=defaultdict(set)
    for o in data['partial_order']:
        a=o['before'];b=o['after'];ea=events[a['event_id']];eb=events[b['event_id']]
        ad=domain(ea,a['anchor']);bd=domain(eb,b['anchor']);kind=o['quantifier']['kind']
        pairs=[(i,i) for i in set(ad)&set(bd)] if kind=='pointwise_same_thread' else [(i,j) for i in ad for j in bd]
        for i,j in pairs:
            aa=(a['event_id'],a['anchor'],i);bb=(b['event_id'],b['anchor'],j)
            graph[aa].add(bb);graph.setdefault(bb,set())
    return graph

def reaches(graph,source,target):
    todo=[source];seen=set()
    while todo:
        x=todo.pop()
        if x==target:return True
        if x not in seen:seen.add(x);todo.extend(graph.get(x,()))
    return False

def acyclic(graph,extras=()):
    successors={x:set(ys) for x,ys in graph.items()}
    for a,b in extras:successors.setdefault(a,set()).add(b);successors.setdefault(b,set())
    degree={x:0 for x in successors}
    for ys in successors.values():
        for y in ys:degree[y]=degree.get(y,0)+1
    todo=[x for x,n in degree.items() if n==0];seen=0
    while todo:
        x=todo.pop();seen+=1
        for y in successors.get(x,()):
            degree[y]-=1
            if degree[y]==0:todo.append(y)
    return seen==len(degree)

def rule_satisfied(rule,anchor_facts,precondition_facts):
    for clause in rule['all_of']:
        if clause['kind']=='qualified_anchor_observed':
            if not qualified_anchor_observed(clause['point'],clause['quantifier'],anchor_facts):return False
        elif clause['kind']=='source_precondition':
            if not precondition_facts.get(clause['id'],False):return False
        else:raise ValueError('Unknown validity clause')
    return True

def qualified_anchor_observed(point,quantifier,anchor_facts):
    """Evaluate a bounded once-only / one-generation observation set.

    Thread indices are identifiers in the event domain, not a chosen GPU lane.
    Runtime generation-family enumeration is intentionally not inferred here.
    """
    eid=point['event_id'];anchor=point['anchor'];kind=quantifier['kind']
    if kind=='single_local_operation':return (eid,anchor,'operation') in anchor_facts
    expected={(eid,anchor,t) for t in range(quantifier['cardinality'])}
    if kind=='all_instances':return expected<=anchor_facts
    if kind=='any_instance':return bool(expected&anchor_facts)
    raise ValueError('Unknown history quantifier: '+kind)

def derive_state_facts(data,anchor_facts):
    """Least fixed point; old anchor facts remain usable after new premises arrive.

    Does not validate temporal prefix closure or declare the observation history
    to be a hardware trace. Time is checked separately using partial_order.
    """
    facts={r['initial_state'] for r in data['resources']}
    changed=True
    while changed:
        changed=False
        for t in data['state_transitions']:
            derivation=t['derivation'];ok=True
            for clause in derivation['all_of']:
                if clause['kind']=='state_fact':ok=ok and clause['state_id'] in facts
                elif clause['kind']=='observed_anchor_history':
                    ok=ok and qualified_anchor_observed(clause['point'],clause['quantifier'],anchor_facts)
                else:raise ValueError('Unknown derivation clause')
            if ok and t['to'] not in facts:facts.add(t['to']);changed=True
    return facts

def validate_semantics(data):
    events={e['id']:e for e in data['events']}
    for o in data['partial_order']:
        assert isinstance(o['quantifier'],dict)
        if o['relation']=='program_order':assert o['quantifier']['kind']=='pointwise_same_thread'
    assert all(isinstance(t['trigger_quantifier'],dict) for t in data['state_transitions'])
    graph=expand_graph(data);assert acyclic(graph)
    for c in range(4):
        write=(f'tmem.cta{c}.allocate','effect','operation')
        for read in ('mma_read_base','epi_read_base'):
            assert reaches(graph,write,(f'tmem.cta{c}.{read}','return',0)),(c,read)
        assert not reaches(graph,write,(f'tmem.cta{c}.allocate','return',0))
        assert events[f'tmem.cta{c}.free']['effect']['kind']=='local_dealloc_instruction_issued'
        assert not any(s['id'].endswith('.deallocated') for r in data['resources'] for s in r['states'])
    for lead in (0,2):
        peer=lead+1
        a=(f'tmem.cta{lead}.free','enter',0);b=(f'tmem.cta{peer}.free','enter',0)
        assert acyclic(graph,[(a,b)]) and acyclic(graph,[(b,a)])
        assert not reaches(graph,(f'tmem.cta{peer}.free','enter',0),(f'tmem.cta{lead}.free','return',0))
        rule=next(r for r in data['validity_rules'] if r['id']==f'tmem.validity.pair{lead//2}.deallocation_participation')
        facts={(f'tmem.cta{lead}.free','effect','operation')};pre={'uniform_512_columns_same_logical_warp':True}
        assert not rule_satisfied(rule,facts,pre)
        facts.add((f'tmem.cta{peer}.free','effect','operation'));assert not rule_satisfied(rule,facts,pre)
        facts|={(f'tmem.cta{c}.free','enter',t) for c in (lead,peer) for t in range(32)}
        assert rule_satisfied(rule,facts,pre)
        assert not rule_satisfied(rule,facts,{})
    assert not reaches(graph,('tmem.cta0.arrive795','return',1),('tmem.cta0.free','enter',0))
    return {'expanded_one_generation_thread_graph_acyclic':True,'allocation_read_before_write_rejected':True,
        'local_free_not_paired_completion':True,'single_sided_pair_acceptance_rejected':True,
        'program_order_not_all_to_all':True,'no_peer_enter_to_local_return_added':True,
        'scope':'bounded semantic-model counterexamples; no hardware/runtime proof'}
