"""Cooperative output intervals from valid-output fractions and calibrated rules."""

import heapq
import math
from numbers import Real


def output_intervals(params, schedule, valid, first_bytes, single, output_rule):
    """Return copied CTA parameters and optional per-tile merged epilogues in ns.

    first_bytes is the total valid first-output byte count across all CTAs.
    single describes the maximum scheduled tile count across the whole grid.
    """
    p = dict(params)
    if not output_rule or not valid:
        return p, None
    if schedule != 'cooperative':
        raise ValueError('these output rules were calibrated for cooperative kernels')
    if single:
        rule = output_rule['E0_single']
        e0 = max(rule['floor_ns'], rule['beta_ns_per_MiB']*first_bytes/2**20) + rule['R_ns']
        p['Etail'] = output_rule['Etail_single_mean_ns']
    else:
        e0 = output_rule['E0_multi']['effective_ns']
        p['Etail'] = ((1-valid[-1])*output_rule['Etail_oob_median_ns']
                      + valid[-1]*output_rule['v09_Etail_multi_median_ns'])
    middle = output_rule['E_middle_ns']
    if middle is None:
        middle = output_rule['middle_padding_proxy']['ns']
    epilogues = [e0]
    for j, fraction in enumerate(valid[1:], 1):
        key = 'Elast' if j == len(valid)-1 else 'E_middle'
        full = output_rule['Elast_ns'] if key == 'Elast' else middle
        epilogues.append((1-fraction)*output_rule[key+'_oob_ns'] + fraction*full)
    return p, epilogues


def cfg_a_empty_arrival_routes():
    """Actual signaling threads for the fixed cluster2x1, two-WG consumer layout.

    spread_arrivals_to_warpgroup uses Swizzle<2,0,-2> over (4*row+col).
    Both cluster ranks have the same N coordinate, so both destinations survive
    the same-row-or-column mask. These are emitters, not visibility timestamps.
    """
    routes=[]
    for source in range(2):
        for wg in range(2):
            for local in range(128):
                row=local//32
                col=(local//8)%4
                offset=4*row+col
                destination=offset^((offset&3)<<2)
                if local%8 or destination>=2:continue
                routes.append(dict(source_rank=source,destination_rank=destination,consumer_wg=wg,
                    physical_thread=128+128*wg+local,thread_in_wg=local,
                    warp_in_wg=local//32,lane=local%32,arrival_count=1,
                    emit_token_template=f'r{source}.w{wg}.g{{group}}.empty_emit_to_r{destination}',
                    visibility_token_template=f'r{source}.w{wg}.g{{group}}.empty_to_r{destination}'))
    return routes


def cfg_a_empty_reuse_interval(emitters, acquire):
    """Conservative last-emit bounds from four actual senders, in globaltimer ns.

    Raw stamps must be positive; zero is the unobserved trace sentinel.
    Brackets must enclose each sender's own original arrival; source/SASS and
    observer validity remain the caller's obligation. Cross-SM clock64 values
    must not be supplied. The post-emit path includes remote visibility, polling,
    scheduling, expect_tx and the acquire-return clock path, not a bare latency.
    """
    fields=('cluster_id','destination_rank','acquire_group','stages','slot','generation','empty_wait_phase','acquire_end_ns')
    if any(type(acquire.get(field)) is not int for field in fields):
        raise ValueError('integer producer identity and globaltimer ns required')
    destination=acquire['destination_rank'];stages=acquire['stages'];group=acquire['acquire_group']
    invocation=acquire.get('invocation_id')
    if not isinstance(invocation,str) or not invocation or acquire['cluster_id']<0:
        raise ValueError('invocation and cluster identity required')
    if destination not in (0,1) or stages<2 or group<stages:
        raise ValueError('invalid reused cfg_a slot')
    released=group-stages
    if (acquire['slot'],acquire['generation'],acquire['empty_wait_phase'])!=(group%stages,group//stages,1^(group//stages%2)):
        raise ValueError('producer slot/generation/phase mismatch')
    expected={(r['source_rank'],r['consumer_wg']):r for r in cfg_a_empty_arrival_routes()
              if r['destination_rank']==destination}
    if len(emitters)!=4:raise ValueError('four distinct required sender brackets required')
    before=[];after=[];seen=set()
    sender_fields=('cluster_id','source_rank','consumer_wg','physical_thread','destination_rank','release_group',
                   'slot','generation','consumer_phase','release_pre_ns','release_post_ns')
    for record in emitters:
        if any(type(record.get(field)) is not int for field in sender_fields):
            raise ValueError('integer sender identity and globaltimer ns required')
        if record.get('invocation_id')!=invocation or record['cluster_id']!=acquire['cluster_id']:
            raise ValueError('sender belongs to another invocation or cluster')
        key=record['source_rank'],record['consumer_wg']
        if key not in expected or key in seen:raise ValueError('missing or duplicate required sender')
        seen.add(key)
        if (record['physical_thread'],record['destination_rank'])!=(expected[key]['physical_thread'],destination):
            raise ValueError('wrong physical arrival sender or destination')
        if (record['release_group'],record['slot'],record['generation'],record['consumer_phase'])!=(released,released%stages,released//stages,released//stages%2):
            raise ValueError('sender slot/generation/phase mismatch')
        low,high=record['release_pre_ns'],record['release_post_ns']
        if not 0<low<=high<2**64:raise ValueError('invalid or unobserved globaltimer sender bracket')
        before.append(low);after.append(high)
    end=acquire['acquire_end_ns']
    if not 0<end<2**64:raise ValueError('invalid or unobserved globaltimer acquire-return stamp')
    low=max(before);high=min(max(after),end)
    if low>high:raise ValueError('producer returned before a required sender could emit')
    return dict(destination_rank=destination,release_group=released,acquire_group=group,
        last_emit_ns=[low,high],post_last_emit_to_acquire_end_ns=[end-high,end-low],
        maximum_sender_bracket_ns=max(e-b for b,e in zip(before,after)),required_sender_count=4)


def cfg_a_output_source_guarantees(output_tiles):
    """Bind each output's last source buffer to the acquire/read0 guarantee.

    R14 records after the wait returns: its timestamp is an upper bound on the
    guarantee and source completion, never an exact physical read-completion time.
    """
    if not isinstance(output_tiles,int) or output_tiles<1:
        raise ValueError('requires a positive integer output tile count')
    guarantees=[]
    for rank in range(2):
        for tile in range(output_tiles):
            last=4*tile+3
            terminal=tile==output_tiles-1
            acquire=None if terminal else last+1
            guarantees.append(dict(rank=rank,tile=tile,issuer_wg=1,terminal=terminal,
                last_subtile=last,protected_buffer=last%2,source_generation=last//2,
                read_through_count=last+1,committed_count=last+1 if terminal else acquire+1,
                pending_batches=0 if terminal else 1,
                acquire_subtile=acquire,reuse_subtile=None if terminal else last+2,
                source_completion=f'r{rank}.d{last}.read_retired',
                guarantee_token=f'r{rank}.store_tail_read0' if terminal else f'r{rank}.d{acquire}.acquire_after',
                observation='r14_source_tail' if terminal else 'r14_source_acquire',
                following_output=None if terminal else tile+1))
    return guarantees


def cooperative_output_source_layout(config, output_tiles):
    """Local output generations for the fixed a/c and legacy R14 baseline tiles.

    cfg_c is 256x128, not R14 baseline's 128x256. Their eight subtiles have
    different coordinates despite identical two-buffer source-guarantee counts.
    No full/empty input dependencies or service prices are transferred here.
    """
    if type(output_tiles) is not int or output_tiles<1:
        raise ValueError('positive integer output tile count required')
    dimensions={'cfg_a':(128,128),'cfg_c':(256,128),'cfg_c1':(256,128),'r14_baseline':(128,256)}
    if config not in dimensions:raise ValueError('unsupported cooperative output layout')
    tile_m,tile_n=dimensions[config];epi_m_count=tile_m//128;epi_n_count=tile_n//32
    chunks=epi_m_count*epi_n_count
    subtiles=[];guarantees=[]
    for tile in range(output_tiles):
        for epi_n in range(epi_n_count):
            for epi_m in range(epi_m_count):
                local=epi_n*epi_m_count+epi_m;q=tile*chunks+local
                subtiles.append(dict(tile=tile,local_subtile=local,subtile=q,epi_m=epi_m,epi_n=epi_n,
                    origin=[128*epi_m,32*epi_n],shape=[128,32],nominal_bytes=16384,
                    buffer=q%2,generation=q//2))
        last=(tile+1)*chunks-1;terminal=tile==output_tiles-1
        acquire=None if terminal else last+1
        guarantees.append(dict(rank=0,tile=tile,issuer_wg=1,terminal=terminal,
            last_subtile=last,protected_buffer=last%2,source_generation=last//2,
            read_through_count=last+1,committed_count=last+1 if terminal else acquire+1,
            pending_batches=0 if terminal else 1,acquire_subtile=acquire,
            reuse_subtile=None if terminal else last+2,
            source_completion=f'r0.d{last}.read_retired',
            guarantee_token='r0.store_tail_read0' if terminal else f'r0.d{acquire}.acquire_after',
            observation='r14_source_tail' if terminal else 'r14_source_acquire',
            following_output=None if terminal else tile+1))
    return dict(config=config,tile=[tile_m,tile_n],epilogue_tile=[128,32],subtiles_per_output=chunks,
        store_buffers=2,unacquired_stages=1,delay_tma_store=True,reuse_smem_c=False,
        subtiles=subtiles,source_guarantees=guarantees,
        scope='Local source lifecycle only; cfg_c and R14 baseline timing and input clusters remain distinct.')


def cooperative_output_source_dependencies(config, output_tiles):
    """One local CTA's cooperative output order, without its input/mainloop graph."""
    layout=cooperative_output_source_layout(config,output_tiles)
    nodes={};markers={};chunks=layout['subtiles_per_output']
    def add(event,parents):nodes[event]=list(dict.fromkeys(parents))
    for tile in range(output_tiles):
        for wg in range(2):
            prefix=f'r0.w{wg}.t{tile}.'
            add(prefix+'MAIN',[f'r0.w{wg}.t{tile-1}.DONE'] if tile else [])
            add(prefix+'PERMIT',[prefix+'MAIN'])
            add(prefix+'DONE',[f'r0.w{other}.t{tile}.PERMIT' for other in range(2)]
                +[f'r0.d{(tile+1)*chunks-1}.acquire_after'])
            for event in ('MAIN','PERMIT','DONE'):markers[prefix+event]=prefix+event
    for item in layout['subtiles']:
        q=item['subtile'];tile=item['tile'];local=item['local_subtile'];prefix=f'r0.d{q}.'
        parents=[f'r0.w{wg}.t{tile}.PERMIT' for wg in range(2)]
        if local:parents.append(f'r0.d{q-1}.r2s')
        add(prefix+'register_prepare',parents)
        add(prefix+'r2s',[prefix+'register_prepare']+([f'r0.d{q-1}.acquire_after'] if q else []))
        add(prefix+'store_issue',[prefix+'r2s']+([f'r0.d{q+1}.register_prepare'] if local<chunks-1 else []))
        add(prefix+'store_commit',[prefix+'store_issue'])
        add(prefix+'source_read',[prefix+'store_issue'])
        add(prefix+'read_retired',[prefix+'source_read']+([f'r0.d{q-1}.read_retired'] if q else []))
        add(prefix+'acquire_after',[prefix+'store_commit']+([f'r0.d{q-1}.read_retired'] if q else []))
        add(prefix+'D_visible',[prefix+'source_read'])
    add('r0.store_tail_read0',[f'r0.w1.t{output_tiles-1}.DONE',f'r0.d{output_tiles*chunks-1}.read_retired'])
    for wg in range(2):
        event=f'r0.w{wg}.FINAL'
        add(event,[f'r0.w{wg}.t{output_tiles-1}.DONE']+(['r0.store_tail_read0'] if wg==1 else []))
        markers[event]=event
    add('r0.D_visible',[f'r0.d{q}.D_visible' for q in range(output_tiles*chunks)])
    return dict(nodes=nodes,markers=markers,layout=layout,
        scope='Local cooperative output only; no input cluster, producer tail, kernel-completion or service prediction.')


def cooperative_output_valid_subtiles(config, logical_m, logical_n, tile_coord, output_index=0):
    """Logical FP32 output coverage in source-buffer order, not physical traffic."""
    if config not in ('cfg_a','cfg_c','cfg_c1'):raise ValueError('fixed cooperative a/c/c1 output geometry required')
    if (type(logical_m) is not int or type(logical_n) is not int or min(logical_m,logical_n)<=0
        or type(output_index) is not int or output_index<0):
        raise ValueError('positive integer logical shape and nonnegative output index required')
    if len(tile_coord)!=2 or any(type(value) is not int or value<0 for value in tile_coord):
        raise ValueError('nonnegative integer output tile coordinate required')
    layout=cooperative_output_source_layout(config,output_index+1)
    chunks=layout['subtiles_per_output'];tile_m,tile_n=layout['tile']
    items=[];buffer_bytes=[0,0];last_valid=None
    for original in layout['subtiles'][-chunks:]:
        item=dict(original)
        m=tile_coord[0]*tile_m+item['origin'][0];n=tile_coord[1]*tile_n+item['origin'][1]
        rows=max(0,min(128,logical_m-m));columns=max(0,min(32,logical_n-n))
        item.update(global_origin=[m,n],valid_rows=rows,valid_columns=columns,valid_bytes=rows*columns*4)
        items.append(item);buffer_bytes[item['buffer']]+=item['valid_bytes']
        if item['valid_bytes']:last_valid=item['subtile']
    return dict(config=config,coord=list(tile_coord),output_index=output_index,subtiles=items,
        valid_bytes=sum(item['valid_bytes'] for item in items),buffer_valid_bytes=buffer_bytes,
        last_valid_subtile=last_valid,
        scope='Logical destination coverage only; empty descriptor boxes retain source pipeline groups and no service price is inferred.')


def cfg_a_rank_dependencies(stages, k_tiles, output_tiles):
    """Source-derived partial order for cluster2x1, two consumer WGs, K_PIPE_MMAS=1.

    No service prices or trace inputs. Release issue and remote arrival are separate.
    Groups advance across output tiles; this is not the cfg_b pingpong graph.
    """
    if stages < 2 or k_tiles < 2 or output_tiles < 1:
        raise ValueError('requires stages>=2, Ktiles>=2 and at least one output tile')
    nodes, markers, groups, transfers, outputs = {}, {}, [], [], []
    def producer(r, g, event):return f'r{r}.g{g}.{event}'
    def consumer(r, w, g, event):return f'r{r}.w{w}.g{g}.{event}'
    def arrival(s, w, r, g):return f'r{s}.w{w}.g{g}.empty_to_r{r}'
    def add(name, predecessors):
        if name in nodes:raise ValueError('duplicate event '+name)
        nodes[name] = list(dict.fromkeys(predecessors))
    total = k_tiles*output_tiles
    for g in range(total):
        tile, local = divmod(g, k_tiles)
        groups.append(dict(group=g,tile=tile,k=local,slot=g%stages,generation=g//stages,
            empty_wait_phase=1^(g//stages%2),full_wait_phase=g//stages%2,
            expected_full_bytes=32768,empty_arrivals=4))
        for r in range(2):
            predecessors=[producer(r,g-1,'B_issue')] if g else []
            if g>=stages:
                predecessors += [arrival(s,w,r,g-stages) for s in range(2) for w in range(2)]
            add(producer(r,g,'acquire'),predecessors)
            add(producer(r,g,'arrive_expect_tx'),[producer(r,g,'acquire')])
            add(producer(r,g,'acquire_return'),[producer(r,g,'arrive_expect_tx')])
            add(producer(r,g,'A_issue'),[producer(r,g,'acquire_return')])
            add(producer(r,g,'B_issue'),[producer(r,g,'A_issue')])
            add(producer(r,g,'A_visible'),[producer(r,g,'A_issue')])
            transfers.append(dict(group=g,operand='A',source_rank=r,destination_rank=r,
                                  bytes=16384,completion=producer(r,g,'A_visible')))
            for s in range(2):
                add(f'r{s}.g{g}.B_to_r{r}',[producer(s,g,'B_issue')])
                transfers.append(dict(group=g,operand='B',source_rank=s,destination_rank=r,
                                      bytes=8192,completion=f'r{s}.g{g}.B_to_r{r}'))
            add(producer(r,g,'full'),[producer(r,g,'arrive_expect_tx'),producer(r,g,'A_visible')]
                +[f'r{s}.g{g}.B_to_r{r}' for s in range(2)])
            for w in range(2):
                previous=[]
                if local==0 and tile:
                    previous=[f'r{r}.w{w}.t{tile-1}.DONE']
                elif local==1:
                    previous=[consumer(r,w,g-1,'commit')]
                elif local>=2:
                    previous=[consumer(r,w,g-2,'release_issue')]
                add(consumer(r,w,g,'wait_return'),[producer(r,g,'full')]+previous)
                before_commit=consumer(r,w,g,'wait_return')
                if local==0:
                    marker=f'r{r}.w{w}.t{tile}.FIRST'
                    add(marker,[before_commit]);markers[marker]=marker;before_commit=marker
                add(consumer(r,w,g,'commit'),[before_commit])
                add(consumer(r,w,g,'retired'),[consumer(r,w,g,'commit')]
                    +([consumer(r,w,g-1,'retired')] if local else []))
                if local<k_tiles-1:
                    add(consumer(r,w,g,'safe_release'),[consumer(r,w,g+1,'commit'),consumer(r,w,g,'retired')])
                else:
                    add(consumer(r,w,g,'mma_return'),[consumer(r,w,g-1,'release_issue')])
                    add(consumer(r,w,g,'tail_wait0'),[consumer(r,w,g,'mma_return'),consumer(r,w,g,'retired')])
                    add(consumer(r,w,g,'safe_release'),[consumer(r,w,g,'tail_wait0')])
                add(consumer(r,w,g,'release_issue'),[consumer(r,w,g,'safe_release')])
                for d in range(2):
                    emit=f'r{r}.w{w}.g{g}.empty_emit_to_r{d}'
                    add(emit,[consumer(r,w,g,'release_issue')])
                    add(arrival(r,w,d,g),[emit])
                if local==k_tiles-1:
                    prefix=f'r{r}.w{w}.t{tile}.'
                    add(prefix+'MAIN',[consumer(r,w,g,'release_issue')])
                    add(prefix+'PERMIT',[prefix+'MAIN'])
                    add(prefix+'DONE',[f'r{r}.w{v}.t{tile}.PERMIT' for v in range(2)])
                    for event in ('MAIN','PERMIT','DONE'):markers[prefix+event]=prefix+event
    for r in range(2):
        # cfg_a: 128x128 FP32 -> four nominal 128x32 subtiles, two D buffers.
        # DelayTmaStore prepares the next registers before issuing the previous
        # subtile, then acquire/read1 protects reuse. State continues across tiles.
        for q in range(4*output_tiles):
            tile,subtile=divmod(q,4)
            prefix=f'r{r}.d{q}.'
            predecessors=[f'r{r}.w{w}.t{tile}.PERMIT' for w in range(2)]
            if subtile:predecessors.append(f'r{r}.d{q-1}.r2s')
            add(prefix+'register_prepare',predecessors)
            reuse=[f'r{r}.d{q-1}.acquire_after'] if q else []
            add(prefix+'r2s',[prefix+'register_prepare']+reuse)
            issue=[prefix+'r2s']
            if subtile<3:issue.append(f'r{r}.d{q+1}.register_prepare')
            add(prefix+'store_issue',issue)
            add(prefix+'store_commit',[prefix+'store_issue'])
            add(prefix+'source_read',[prefix+'store_issue'])
            add(prefix+'read_retired',[prefix+'source_read']
                +([f'r{r}.d{q-1}.read_retired'] if q else []))
            add(prefix+'acquire_after',[prefix+'store_commit']
                +([f'r{r}.d{q-1}.read_retired'] if q else []))
            add(prefix+'D_visible',[prefix+'source_read'])
            if subtile==3:
                for w in range(2):nodes[f'r{r}.w{w}.t{tile}.DONE'].append(prefix+'acquire_after')
            outputs.append(dict(rank=r,subtile=q,tile=tile,local_subtile=subtile,buffer=q%2,
                generation=q//2,nominal_bytes=16384,
                source_completion=prefix+'source_read',global_completion=prefix+'D_visible'))
        # Last generation of each used slot: producer-tail waits for all empty arrivals.
        last=[g for g in range(max(0,total-stages),total)]
        add(f'r{r}.producer_tail',[producer(r,total-1,'B_issue')]
            +[arrival(s,w,r,g) for g in last for s in range(2) for w in range(2)])
        add(f'r{r}.output_source_read',[f'r{r}.d{4*output_tiles-1}.read_retired'])
        add(f'r{r}.store_tail_read0',[f'r{r}.w1.t{output_tiles-1}.DONE',f'r{r}.output_source_read'])
        for w in range(2):
            key=f'r{r}.w{w}.FINAL'
            add(key,[f'r{r}.w{w}.t{output_tiles-1}.DONE']
                +([f'r{r}.store_tail_read0'] if w==1 else []));markers[key]=key
        add(f'r{r}.D_visible',[f'r{r}.d{q}.D_visible' for q in range(4*output_tiles)])
    add('source_safe_exit',[f'r{r}.producer_tail' for r in range(2)]
        +[f'r{r}.w{w}.FINAL' for r in range(2) for w in range(2)])
    add('cuda_complete',['source_safe_exit','r0.D_visible','r1.D_visible'])
    return dict(nodes=nodes,markers=markers,groups=groups,transfers=transfers,outputs=outputs,
                empty_arrival_routes=cfg_a_empty_arrival_routes(),
                producer_observation_ports=dict(empty_wait_return='r{rank}.g{group}.acquire',
                    local_full_arrival='r{rank}.g{group}.arrive_expect_tx',
                    producer_acquire_call_return='r{rank}.g{group}.acquire_return'),
                source_guarantees=cfg_a_output_source_guarantees(output_tiles),stages=stages,k_tiles=k_tiles,
                output_tiles=output_tiles,scope='Partial order; no calibrated service or GPU timing prediction')


def cfg_c_empty_arrival_routes():
    """Enumerate signaling threads from cfg_c's actual 1x2 cluster mask."""
    routes=[]
    for source in range(2):
        source_coord=(0,source)
        for physical in range(128,384):
            local=physical%128;row=(physical//32)%4;col=(local//8)%4
            offset=4*row+col
            destination=offset^((offset&3)<<2)
            if local%8 or destination>=2:continue
            destination_coord=(destination%1,destination//1)
            same_row_or_col=(destination_coord[0]==source_coord[0]
                             or destination_coord[1]==source_coord[1])
            if not same_row_or_col:continue
            wg=(physical-128)//128
            routes.append(dict(source_rank=source,destination_rank=destination,consumer_wg=wg,
                source_coord=list(source_coord),destination_coord=list(destination_coord),
                physical_thread=physical,thread_in_wg=local,warp_in_wg=local//32,lane=local%32,
                arrival_count=1,
                emit_token_template=f'r{source}.w{wg}.g{{group}}.empty_emit_to_r{destination}',
                visibility_token_template=f'r{source}.w{wg}.g{{group}}.empty_to_r{destination}'))
    return routes


def cfg_c_rank_dependencies(source_ports):
    """Close B's cfg_c source ports using the fixed cooperative S4 lifecycle.

    The 256x128/cluster1x2 graph is constructed from its own source contract.
    Source-only empty predecessor lists never replace lifecycle predecessors.
    cluster_initialized is an external entry boundary, not a measured zero time.
    This partial order contains no service prices, trace anchors or GPU claims.
    """
    if not isinstance(source_ports,dict) or source_ports.get('schema')!='gh200-cfg-c-source-ports/1':
        raise ValueError('cfg_c source-port schema/1 required')
    stages=source_ports.get('stages');k_tiles=source_ports.get('k_tiles');output_tiles=source_ports.get('output_tiles')
    if (any(type(v) is not int for v in (stages,k_tiles,output_tiles))
        or stages!=4 or k_tiles<2 or output_tiles<1):
        raise ValueError('fixed cfg_c stage4, integer Ktiles>=2 and outputs>=1 required')
    total=k_tiles*output_tiles;supplied=source_ports.get('nodes')
    source_groups=source_ports.get('groups',[])
    if len(source_groups)!=total:raise ValueError('all source group states required')
    for g,item in enumerate(source_groups):
        expected=dict(group=g,slot=g%4,generation=g//4,full_wait_phase=g//4%2,
            empty_wait_phase=1^(g//4%2),previous_group=g-4 if g>=4 else None,
            tile=g//k_tiles,k=g%k_tiles,expected_full_bytes=49152)
        if any(item.get(key)!=value for key,value in expected.items()):
            raise ValueError('cfg_c source slot/generation/full-byte state differs')
    if not isinstance(supplied,dict) or any(not isinstance(p,list) for p in supplied.values()):
        raise ValueError('source node predecessor lists required')
    expected_ports={f'r{r}.g{g}.{port}' for g in range(total) for r in range(2)
                    for port in ('arrive_expect_tx','acquire_return')}
    declared=source_ports.get('external_producer_ports',[])
    if len(declared)!=len(expected_ports) or set(declared)!=expected_ports:
        raise ValueError('exact external producer-port identities required')
    expected_requests={(g,r,a,size) for g in range(total) for r in range(2)
                       for a,size in (('A',16384),('B0',8192),('B1',8192))}
    requests=source_ports.get('input_requests',[])
    identities={(x['group'],x['source_rank'],x['atom'],x['bytes']) for x in requests}
    if len(requests)!=len(expected_requests) or identities!=expected_requests:
        raise ValueError('cfg_c requires six source atoms and 64KiB per cluster/group')
    expected_deliveries={(g,s,d,a,size) for g in range(total) for s in range(2)
                         for a,size,targets in (('A',16384,(0,1)),('B0',8192,(s,)),('B1',8192,(s,)))
                         for d in targets}
    deliveries=source_ports.get('input_deliveries',[])
    identities={(x['group'],x['source_rank'],x['destination_rank'],x['atom'],x['bytes']) for x in deliveries}
    if len(deliveries)!=len(expected_deliveries) or identities!=expected_deliveries:
        raise ValueError('cfg_c requires two multicast A16 and two local B8 atoms per destination')
    def require(name,parents):
        if name not in supplied or not set(parents)<=set(supplied[name]):
            raise ValueError('source dependency missing: '+name)
    for row in requests:
        prefix=f"r{row['source_rank']}.g{row['group']}.";atom=row['atom']
        if (row['issue'],row['source_ready'])!=(prefix+atom+'_issue',prefix+atom+'_source_ready'):
            raise ValueError('source atom port identity differs')
        require(row['issue'],[prefix+'acquire_return' if atom=='A' else prefix+'A_issue'])
        require(row['source_ready'],[row['issue']])
    for row in deliveries:
        prefix=f"r{row['source_rank']}.g{row['group']}.";atom=row['atom']
        if (row['source_ready'],row['completion'])!=(prefix+atom+'_source_ready',prefix+atom+f"_delivered_r{row['destination_rank']}"):
            raise ValueError('destination atom port identity differs')
        require(row['completion'],[row['source_ready']])
    full_ports={f'r{r}.g{g}.full' for g in range(total) for r in range(2)}
    if len(source_ports.get('full_output_ports',[]))!=len(full_ports) or set(source_ports['full_output_ports'])!=full_ports:
        raise ValueError('all cfg_c full output ports required')
    for g in range(total):
        for r in range(2):
            prefix=f'r{r}.g{g}.'
            require(prefix+'B_issue',[prefix+'B0_issue',prefix+'B1_issue'])
            require(prefix+'full',[prefix+'arrive_expect_tx']+
                [f'r{s}.g{g}.A_delivered_r{r}' for s in range(2)]+
                [prefix+a+f'_delivered_r{r}' for a in ('B0','B1')])
    # Build lifecycle first. Later source-node union cannot erase its predecessors.
    nodes={'cluster_initialized':[]};markers={};groups=[];tail_waits=[]
    routes=cfg_c_empty_arrival_routes()
    def add(name,parents):nodes[name]=list(dict.fromkeys(nodes.get(name,[])+parents))
    def arrivals(rank,group):
        return [route['visibility_token_template'].format(group=group)
                for route in routes if route['destination_rank']==rank]
    for g in range(total):
        tile,local=divmod(g,k_tiles)
        groups.append(dict(group=g,tile=tile,k=local,slot=g%4,generation=g//4,
            empty_wait_phase=1^(g//4%2),full_wait_phase=g//4%2,
            expected_full_bytes=49152,empty_arrivals=4))
        for r in range(2):
            prefix=f'r{r}.g{g}.'
            add(prefix+'acquire',([f'r{r}.g{g-1}.B_issue'] if g else ['cluster_initialized'])
                +(arrivals(r,g-4) if g>=4 else []))
            add(prefix+'arrive_expect_tx',[prefix+'acquire'])
            add(prefix+'acquire_return',[prefix+'arrive_expect_tx'])
            for w in range(2):
                c=f'r{r}.w{w}.g{g}.'
                previous=[]
                if local==0 and tile:previous=[f'r{r}.w{w}.t{tile-1}.DONE']
                elif local==1:previous=[f'r{r}.w{w}.g{g-1}.commit']
                elif local>=2:previous=[f'r{r}.w{w}.g{g-2}.release_issue']
                add(c+'wait_return',[prefix+'full']+previous)
                before=c+'wait_return'
                if local==0:
                    before=f'r{r}.w{w}.t{tile}.FIRST';add(before,[c+'wait_return']);markers[before]=before
                add(c+'commit',[before])
                add(c+'retired',[c+'commit']+([f'r{r}.w{w}.g{g-1}.retired'] if local else []))
                if local<k_tiles-1:
                    add(c+'safe_release',[f'r{r}.w{w}.g{g+1}.commit',c+'retired'])
                else:
                    add(c+'mma_return',[f'r{r}.w{w}.g{g-1}.release_issue'])
                    add(c+'tail_wait0',[c+'mma_return',c+'retired'])
                    add(c+'safe_release',[c+'tail_wait0'])
                    add(f'r{r}.w{w}.t{tile}.MAIN',[c+'release_issue'])
                add(c+'release_issue',[c+'safe_release'])
                for route in routes:
                    if (route['source_rank'],route['consumer_wg'])!=(r,w):continue
                    emit=route['emit_token_template'].format(group=g)
                    add(emit,[c+'release_issue'])
                    add(route['visibility_token_template'].format(group=g),[emit])
    local_output=cooperative_output_source_dependencies('cfg_c',output_tiles)
    for r in range(2):
        for name,parents in local_output['nodes'].items():
            add(name.replace('r0.',f'r{r}.'),[p.replace('r0.',f'r{r}.') for p in parents])
        for name in local_output['markers']:
            key=name.replace('r0.',f'r{r}.');markers[key]=key
        previous=f'r{r}.g{total-1}.B_issue'
        # producer_tail waits all four slots in its actual cyclic state order.
        for step in range(4):
            count=total+step;released=count-4;event=f'r{r}.producer_tail_wait{step}'
            add(event,[previous]+(arrivals(r,released) if released>=0 else ['cluster_initialized']))
            tail_waits.append(dict(rank=r,step=step,slot=count%4,generation=count//4,
                empty_wait_phase=1^(count//4%2),release_group=released if released>=0 else None,
                initial_empty=released<0,event=event))
            previous=event
        add(f'r{r}.producer_tail',[previous])
    for name,parents in supplied.items():add(name,parents)
    add('source_safe_exit',[f'r{r}.producer_tail' for r in range(2)]+
        [f'r{r}.w{w}.FINAL' for r in range(2) for w in range(2)])
    add('cuda_complete',['source_safe_exit','r0.D_visible','r1.D_visible'])
    if {name for name,parents in nodes.items() if not parents}!={'cluster_initialized'}:
        raise ValueError('unresolved source roots remain after lifecycle composition')
    graph=dict(schema='gh200-cfg-c-rank-dependencies/1',nodes=nodes,markers=markers,groups=groups,
        stages=4,k_tiles=k_tiles,output_tiles=output_tiles,tile=[256,128,64],cluster=[1,2],
        empty_arrival_routes=routes,producer_tail_waits=tail_waits,
        input_requests=[dict(x) for x in requests],input_deliveries=[dict(x) for x in deliveries],
        input_totals=dict(source_requests=len(requests),source_payload_bytes=sum(x['bytes'] for x in requests),
            destination_atoms=len(deliveries),destination_payload_bytes=sum(x['bytes'] for x in deliveries)),
        local_output_layout=local_output['layout'],
        source_guarantees=[dict(item,rank=r,source_completion=item['source_completion'].replace('r0.',f'r{r}.'),
            guarantee_token=item['guarantee_token'].replace('r0.',f'r{r}.'))
            for r in range(2) for item in local_output['layout']['source_guarantees']],
        external_entry_ports=['cluster_initialized'],external_producer_ports=[],
        completion_ports=dict(source_safe_exit='source_safe_exit',global_D_visible=['r0.D_visible','r1.D_visible'],cuda_complete='cuda_complete'),
        scope='Fixed cfg_c static cooperative cluster lifecycle. Entry time and every service remain unpriced; no autonomous timing prediction or qualification.')
    dependency_order(graph)
    return graph


def dependency_order(graph):
    """Deterministic topological order; reject missing predecessors and dependency cycles."""
    nodes=graph['nodes'];children={n:[] for n in nodes};degree={n:len(p) for n,p in nodes.items()}
    for name,predecessors in nodes.items():
        for p in predecessors:
            if p not in children:raise ValueError('missing predecessor '+p)
            children[p].append(name)
    queue=[n for n,d in degree.items() if not d];heapq.heapify(queue);order=[]
    while queue:
        n=heapq.heappop(queue);order.append(n)
        for c in children[n]:
            degree[c]-=1
            if not degree[c]:heapq.heappush(queue,c)
    if len(order)!=len(nodes):raise ValueError('dependency cycle')
    return order,children


def dependency_bounds(graph, anchors=None, latencies_ns=None):
    """Feasible intervals under nonnegative edge lags; anchors are diagnostic only.

    Each latency is from a node's latest predecessor to that node. Anchors are
    (finite lower, finite or positive-infinite upper) pairs in an arbitrary time
    origin; negative timestamps are allowed. Unbounded endpoints become None.
    Without anchors, roots start at zero. Earliest values in that case are synthetic
    max-plus witnesses, not fitted service times or autonomous GEMM predictions.
    """
    anchors=anchors or {};latencies_ns=latencies_ns or {};order,children=dependency_order(graph)
    if (set(anchors)|set(latencies_ns))-set(order):raise ValueError('unknown event key')
    if any(not isinstance(v,Real) or not math.isfinite(v) or v<0 for v in latencies_ns.values()):
        raise ValueError('latencies must be finite nonnegative ns')
    for name,interval in anchors.items():
        if not isinstance(interval,(tuple,list)) or len(interval)!=2:
            raise ValueError('anchor must contain two endpoints: '+name)
        low,high=interval
        if (not isinstance(low,Real) or not math.isfinite(low)
                or not isinstance(high,Real) or math.isnan(high) or high==-math.inf or low>high):
            raise ValueError('invalid anchor interval: '+name)
    origin=-math.inf if anchors else 0
    lower={n:anchors.get(n,(origin,math.inf))[0] for n in order}
    upper={n:anchors.get(n,(origin,math.inf))[1] for n in order}
    for n in order:
        root_lower=origin if anchors else latencies_ns.get(n,0)
        lower[n]=max(lower[n],max((lower[p]+latencies_ns.get(n,0) for p in graph['nodes'][n]),default=root_lower))
    for n in reversed(order):
        upper[n]=min(upper[n],min((upper[c]-latencies_ns.get(c,0) for c in children[n]),default=math.inf))
    # Subtract raw integer endpoints before the float tolerance: adding it to
    # an absolute globaltimer stamp would round away low bits near 10**18.
    invalid=[n for n in order if lower[n]-upper[n]>1e-9]
    if invalid:raise ValueError('infeasible event bounds: '+','.join(invalid[:4]))
    return {n:[lower[n] if math.isfinite(lower[n]) else None,
               upper[n] if math.isfinite(upper[n]) else None] for n in order}
