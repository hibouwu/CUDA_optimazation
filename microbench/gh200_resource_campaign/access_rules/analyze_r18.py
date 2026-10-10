#!/usr/bin/env python3
"""Independent numeric replay and actual CTA tile/event analysis for R18/R19."""
import argparse,gzip,json,math,statistics
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import v06_model as model
import v06_run as common


@lru_cache(None)
def reference(row,col,k,seed=17):
    total=sum(((row*7+t*13+seed*3)%17-8)*((t*5+col*11+seed*5)%17-8) for t in range(17))
    remainder=sum(((row*7+t*13+seed*3)%17-8)*((t*5+col*11+seed*5)%17-8) for t in range(k%17))
    return (total*(k//17)+remainder)/1024


def random_references(indices,m,n,k,seed):
    """Regenerate logical FP16 inputs independently of the device-side output checker."""
    import numpy as np

    def values(rows,cols,is_a):
        x=np.uint32(seed)^np.uint32(0xa511e9b3 if is_a else 0x63d83595)
        x=x^(rows*np.uint32(0x9e3779b9))^(cols*np.uint32(0x85ebca6b))
        x=(x^(x>>16))*np.uint32(0x7feb352d)
        x=(x^(x>>15))*np.uint32(0x846ca68b)
        x=x^(x>>16)
        return ((x>>8).astype(np.float32)*np.float32(2**-23)-np.float32(1)).astype(np.float16).astype(np.float64)

    positions=np.asarray(indices,dtype=np.int64)
    rows=(positions//n).astype(np.uint32);cols=(positions%n).astype(np.uint32)
    inner=np.arange(k,dtype=np.uint32)[None,:]
    expected=[];tolerances=[]
    for start in range(0,len(indices),32):
        a=values(rows[start:start+32,None],inner,True)
        b=values(inner,cols[start:start+32,None],False)
        products=a*b
        expected.extend(products.sum(axis=1).tolist())
        tolerances.extend((2**-20+2**-21*np.abs(products).sum(axis=1)).tolist())
    return expected,tolerances



def decode_stage_sparse(setup, call):
    """CTA0 cycle windows; background protocol adds direct globaltimer endpoints."""
    words=call['trace'];width=setup['trace_words']
    background=setup.get('trace_version')=='r13-output-background-v1'
    if len(words)!=math.prod(setup['grid'])*width or (not background and any(words[width:])):
        raise ValueError('sparse trace size or non-target CTA writes')
    if background:
        for offset in range(width,len(words),width):
            head=words[offset:offset+width]
            if any(head[:46]) or any(head[49:]):
                raise ValueError('unexpected non-target CTA background fields')
    b=words[:width]
    if setup.get('trace_version') in ('r13-stage-progress-v1','r13-stage-handoff-v1','r13-epilogue-order-v1','r13-output-background-v1'):
        handoff=setup['trace_version']=='r13-stage-handoff-v1'
        epilogue_order=background or setup['trace_version']=='r13-epilogue-order-v1'
        mapping={'first_mma':21,'commit1':20,'full_wait_begin16':15,'commit16':17,
            'wait1_end16':18,'full_wait_begin17':16,'commit17':19,'commit63':24,
            'mma_return':32,'main_end':22}
        c={name:b[index] for name,index in mapping.items()}
        sequence=[c[name] for name in mapping]
        marker=0x52313305 if background else 0x52313304 if epilogue_order else 0x52313303 if handoff else 0x52313302
        if b[31]!=marker or b[10] or not all(sequence) or any(a>z for a,z in zip(sequence,sequence[1:])):
            raise ValueError('invalid consumer progress event order')
        if not b[1] or b[1]!=b[23] or min(b[13],b[14])<=0:
            raise ValueError('progress consumer SM/work mismatch')
        result=dict(cta=0,sm=b[1]-1,work=[b[13]-1,b[14]-1],output=1,clocks=c,
            intervals={name:c[end]-c[start] for name,start,end in [
                ('commit16_to17','commit16','commit17'),
                ('loop_advance_16_to17','wait1_end16','full_wait_begin17'),
                ('full_wait_to_commit16','full_wait_begin16','commit16'),
                ('full_wait_to_commit17','full_wait_begin17','commit17'),
                ('commit_to_wait1_16','commit16','wait1_end16'),
                ('first_to_commit1','first_mma','commit1'),
                ('first_to_commit16','first_mma','commit16'),
                ('first_to_commit17','first_mma','commit17'),
                ('first_to_commit63','first_mma','commit63'),
                ('last_commit_to_mma_return','commit63','mma_return'),
                ('mma_return_to_main_end','mma_return','main_end'),
                ('last_commit_to_main_end','commit63','main_end'),
                ('L','first_mma','main_end')]})
        if handoff:
            h={name:b[i] for name,i in dict(sender_mma_return=34,math_wait_begin=35,
                math_wait_end=36,first_wait_begin=37).items()}
            if b[38]!=b[1] or min(b[39],b[40])<=0:raise ValueError('handoff sender SM/work mismatch')
            if not all(h.values()) or not (h['math_wait_begin']<=h['math_wait_end']<=h['first_wait_begin']<=c['first_mma']) or h['sender_mma_return']>h['math_wait_end']:
                raise ValueError('invalid handoff event order')
            result['sender_work']=[b[39]-1,b[40]-1]
            c.update(h)
            result['intervals'].update({name:c[end]-c[start] for name,start,end in [
                ('sender_to_math_return','sender_mma_return','math_wait_end'),
                ('math_wait_total','math_wait_begin','math_wait_end'),
                ('math_return_to_full_begin','math_wait_end','first_wait_begin'),
                ('first_full_wait','first_wait_begin','first_mma'),
                ('sender_to_first','sender_mma_return','first_mma')]})
            result['intervals']['receiver_late_entry']=max(0,h['math_wait_begin']-h['sender_mma_return'])
            result['intervals']['after_sender_and_wait_entry']=h['math_wait_end']-max(h['math_wait_begin'],h['sender_mma_return'])
        if epilogue_order:
            if b[43]!=b[1] or min(b[44],b[45])<=0 or (b[44],b[45])==(b[13],b[14]):
                raise ValueError('epilogue sender SM/work mismatch')
            begin,end=b[41],b[42]
            if not 0<begin<=end:
                raise ValueError('invalid sender epilogue event order')
            if setup.get('epilogue_defer') not in ((0,1,2) if background else (0,1)):
                raise ValueError('missing epilogue defer identity')
            if setup['epilogue_defer']>0 and begin<c['mma_return']:
                raise ValueError('deferred epilogue precedes receiver mma return')
            result['sender_work']=[b[44]-1,b[45]-1]
            c.update(sender_epi_begin=begin,sender_epi_end=end)
            result['intervals'].update(
                sender_epilogue=end-begin,
                epilogue_overlap_L=max(0,min(end,c['main_end'])-max(begin,c['first_mma'])),
                epilogue_overlap_early=max(0,min(end,c['commit16'])-max(begin,c['commit1'])))
        if background:
            direct=[]
            for cta,offset in enumerate(range(0,len(words),width)):
                begin,end,mma=words[offset+46:offset+49]
                if not (0<begin<=end and mma>0):
                    raise ValueError('invalid direct background endpoints')
                if (setup['epilogue_defer']==2 or (cta==0 and setup['epilogue_defer']>0)) and begin<mma:
                    raise ValueError('deferred background epilogue precedes receiver mma return')
                direct.append(dict(cta=cta,epi_begin_ns=begin,epi_end_ns=end,mma_return_ns=mma))
            if not 0<b[49]<=b[50]<=direct[0]['mma_return_ns']:
                raise ValueError('invalid direct early-window endpoints')
            result.update(direct_ns_background=direct,early_ns=[b[49],b[50]])
        return result
    clocks={name:b[index] for name,index in {
        'entry':0,'acquire_begin':6,'acquire_end':7,'issue_end':8,
        'refill_begin':9,'refill_end':11,'refill_issue_end':12,
        'full_wait_begin':15,'full_wait_end':16,'commit_end':17,'wait1_end':18,
        'release_begin':19,'release_end':20,'first_mma':21,'main_end':22,
        'mma_return':32,'tail_begin':33}.items()}
    if not all(clocks.values()) or b[31]!=0x52313301 or b[10]:
        raise ValueError('incomplete sparse events')
    if not b[1] or b[1]!=b[23] or (b[2],b[3])!=(b[13],b[14]) or not min(b[2],b[3]):
        raise ValueError('sparse role SM/work mismatch')
    global_k=setup['k']//64+16
    slot=global_k%setup['stages']
    consumer_phase=(global_k//setup['stages'])%2
    if (b[29],b[30],b[4],b[26])!=(1,16,slot+1,consumer_phase+1):
        raise ValueError('sparse output/Ktile/slot sequence mismatch')
    if not (b[4]==b[24]==b[25]==b[27] and b[5]==3-b[26] and b[28]==b[26]):
        raise ValueError('sparse producer/consumer slot or wrap phase mismatch')
    for seq in [('acquire_begin','acquire_end','issue_end'),('refill_begin','refill_end','refill_issue_end'),
                ('first_mma','full_wait_begin','full_wait_end','commit_end','wait1_end','release_begin','release_end','mma_return','tail_begin','main_end')]:
        if any(clocks[a]>clocks[z] for a,z in zip(seq,seq[1:])):
            raise ValueError('sparse within-role event order')
    # Release may become visible to producer before the releasing lane's post-arrive clock.
    if clocks['refill_end']<clocks['release_begin']:
        raise ValueError('same slot reacquired before consumer release')
    return dict(cta=0,sm=b[1]-1,work=[b[2]-1,b[3]-1],output=1,ktile=16,
        slot=b[4]-1,phase=b[5]-1,clocks=clocks,
        intervals={name:clocks[end]-clocks[start] for name,start,end in [
            ('producer_empty_wait','acquire_begin','acquire_end'),
            ('producer_issue','acquire_end','issue_end'),
            ('refill_empty_wait','refill_begin','refill_end'),
            ('consumer_full_wait','full_wait_begin','full_wait_end'),
            ('consumer_issue_to_commit','full_wait_end','commit_end'),
            ('consumer_wait1','commit_end','wait1_end'),
            ('consumer_release','release_begin','release_end'),
            ('mainloop_before_tail','first_mma','mma_return'),
            ('handoff_before_tail','mma_return','tail_begin'),
            ('mma_tail','tail_begin','main_end'),('L','first_mma','main_end')]})



def decode_ready_observer(values, enabled, cta, release_values=None):
    """Read-only phase observer; late successes and polling limits are retained."""
    if len(values)!=8 or (not enabled and any(values)):
        raise ValueError('ready observation record length or disabled contents')
    release_bound=release_values is not None
    protocol='r13-ready-release-v1' if release_bound else 'r13-ready-observer-v1'
    result=dict(cta=55,enabled=bool(enabled),protocol=protocol)
    if release_bound and (len(release_values)!=2 or (not enabled and any(release_values))):
        raise ValueError('first-release observation record length or disabled contents')
    if not enabled:return result
    sc,sn,lc,ln,ec,en,tests,flags=values
    success=bool(flags&1);sm=(flags>>1)-1
    if sm!=cta['sm'] or not 1<=tests<=4096 or not all((sc,sn,ec,en)):
        raise ValueError('ready observer SM, count or missing endpoints')
    if not (cta['entry_c']<=sc<=ec and cta['entry_ns']<=sn<=en):
        raise ValueError('ready observer endpoints precede this invocation')
    if success and tests==1:
        if lc or ln:raise ValueError('first-test success has a false endpoint')
    elif not (sc<=lc<=ec and sn<=ln<=en):
        raise ValueError('missing or unordered false-test endpoint')
    if not success and tests!=4096:raise ValueError('polling stopped before its limit without success')
    bound_c=cta['tiles'][0][0]
    if release_bound:
        rc,rn=release_values
        if not (cta['tiles'][0][0]<=rc<=cta['tiles'][0][1] and cta['tiles_ns'][0][0]<=rn<=cta['tiles_ns'][0][1]):
            raise ValueError('first release marker outside the current first consumer mainloop')
        result.update(release_pre_c=rc,release_pre_ns=rn,generation_bound='first_release')
        bound_c=rc
    first_generation=success and ec<=bound_c
    late='after_release' if release_bound else 'after_first'
    status=('upper_only' if tests==1 else 'interval') if first_generation else (late if success else 'poll_limit')
    result.update(start_c=sc,start_ns=sn,last_false_pre_c=lc,last_false_pre_ns=ln,
        exit_c=ec,exit_ns=en,tests=tests,success=success,sm=sm,
        first_generation=first_generation,status=status)
    return result


def validate_warmup(setup, call, row):
    """Legacy adaptive convergence, or an explicitly fixed common history."""
    if any(event.get('work_swap55_91',0) != row.get('work_swap55_91',0) for event in (setup,call)):
        raise ValueError('work permutation identity changed')
    if 'epilogue_defer' in row:
        if any(event.get('epilogue_defer')!=row['epilogue_defer'] for event in (setup,call)):
            raise ValueError('epilogue order identity changed')
        if any(event.get('epilogue_warmup_defer')!=0 for event in (setup,call)):
            raise ValueError('epilogue treatment entered the warmup prefix')
    if 'producer_delay_ns' in row:
        for event in (setup,call):
            if event.get('producer_delay_ns')!=row['producer_delay_ns'] or event.get('producer_delay_protocol')!='cta55-pre-first-load-v1':
                raise ValueError('producer delay identity changed')
    if 'first_supply_events' in row:
        for event in (setup,call):
            if event.get('first_supply_events')!=row['first_supply_events'] or event.get('first_supply_protocol')!='r13-first-supply-v1' or event.get('first_supply_scratch_bytes')!=(144 if row.get('ready_generation_bound')=='first_release' else 128 if 'ready_observer' in row else 64):
                raise ValueError('first supply observation identity changed')
    if 'ready_observer' in row:
        for event in (setup,call):
            if event.get('ready_observer')!=row['ready_observer'] or event.get('ready_observer_protocol')!=('r13-ready-release-v1' if row.get('ready_generation_bound')=='first_release' else 'r13-ready-observer-v1'):
                raise ValueError('ready observation identity changed')
    policy = row.get('warmup_policy','adaptive')
    if policy not in ('adaptive','common-sw1-16'):
        raise ValueError('unknown warmup policy')
    expected_swizzle = 1 if policy=='common-sw1-16' else row['swizzle']
    fixed_calls = 16 if policy=='common-sw1-16' else 0
    for event in (setup,call):
        if event.get('warmup_policy','adaptive') != policy:
            raise ValueError('warmup policy identity changed')
        if policy=='common-sw1-16' or 'warmup_swizzle' in event:
            if event.get('warmup_swizzle') != expected_swizzle or event.get('warmup_fixed_calls') != fixed_calls:
                raise ValueError('warmup prefix identity changed')
    if call.get('target_swizzle',row['swizzle']) != row['swizzle']:
        raise ValueError('target swizzle identity changed')
    history = row.get('history_mode','off')
    protocol = {'off':'none','single':'single-common16-ends-v1','series':'prefix16-target19-ends-v1'}.get(history)
    if protocol is None or setup.get('history_mode','off')!=history:
        raise ValueError('history mode identity changed')
    if any(e.get('history_protocol','none')!=protocol for e in (setup,call)):
        raise ValueError('history protocol identity changed')
    if history!='off':
        if policy!='common-sw1-16' or setup.get('history_capacity_bytes')!=35*math.prod(setup['grid'])*16*8:
            raise ValueError('history prefix or capacity changed')
        if call.get('target_calls')!=(19 if history=='series' else 1) or call.get('preceding_target_calls')!=(18 if history=='series' else 0):
            raise ValueError('history target-call identity changed')
    if 'b_offset_bytes' in row or 'b_allocation_extra_bytes' in row:
        offset=row.get('b_offset_bytes',0);extra=row.get('b_allocation_extra_bytes',0)
        if offset<0 or extra<offset or offset%16 or extra%16:
            raise ValueError('invalid B base-offset capacity')
        for event in (setup,call):
            if event.get('b_offset_bytes')!=offset or event.get('b_allocation_extra_bytes')!=extra:
                raise ValueError('B base-offset identity changed')
            if event.get('b_payload_elements')!=row['k']*row.get('alloc_ldb',row.get('ldb',row['n'])):
                raise ValueError('B initialized payload extent changed')
            if event.get('target_b_ptr',0)-event.get('b_buffer_ptr',0)!=offset or not event.get('b_buffer_ptr'):
                raise ValueError('B target pointer does not match declared offset')
        for key in ('target_b_ptr','b_buffer_ptr'):
            if setup[key]!=call[key]:raise ValueError('B pointer changed within process')
        if setup.get('allocated_b_elements')!=setup['b_payload_elements']+extra//2:
            raise ValueError('B allocation does not include the common extra capacity')
    source=row.get('history_source','original')
    slot=row.get('mirror_target_slot',0)
    if source not in ('original','same','mirror') or slot not in (0,1):
        raise ValueError('unknown history source or target slot')
    for event in (setup,call):
        if event.get('history_source','original')!=source or event.get('mirror_target_slot',0)!=slot:
            raise ValueError('history source identity changed')
        if source!='original':
            if history!='series' or row['swizzle']!=1 or event.get('source_copies')!=2:
                raise ValueError('mirror source comparison changed its history or capacity')
            for op in ('a','b'):
                target,prefix=event.get('target_'+op+'_ptr'),event.get('prefix_'+op+'_ptr')
                if not isinstance(target,int) or not isinstance(prefix,int) or min(target,prefix)<=0:
                    raise ValueError('missing source pointer identity')
                if (target==prefix)!=(source=='same'):
                    raise ValueError('prefix does not use declared source identity')
    # Raw setup and call belong to one process; static_setup.json is separate.
    if source!='original':
        for key in ('target_a_ptr','target_b_ptr','prefix_a_ptr','prefix_b_ptr'):
            if setup[key]!=call[key]:raise ValueError('source pointer changed within process')
    warm = call['warmup_us']
    if not warm or any(not math.isfinite(x) or x<=0 for x in warm):
        raise ValueError('invalid warmup durations')
    if call.get('warmup_calls',len(warm)) != len(warm):
        raise ValueError('warmup call count changed')
    if policy=='common-sw1-16':
        if len(warm)!=16:
            raise ValueError('common-sw1-16 must execute exactly sixteen prefix calls')
    elif not 8<=len(warm)<=30 or statistics.pstdev(warm[-5:])/statistics.mean(warm[-5:])>.02:
        raise ValueError('warmup did not converge')
    if history!='off':
        calls=35 if history=='series' else 17
        gaps=call.get('history_inter_call_event_us',[])
        if call.get('history_recorded_calls')!=calls or len(gaps)!=calls-1 or any(not math.isfinite(x) or x<0 for x in gaps):
            raise ValueError('history event intervals incomplete')
        if abs(call['history_prefix_event_us']-sum(warm)-sum(gaps[:15]))>1.:
            raise ValueError('history prefix event timeline does not close')
        if history=='single' and abs(call['history_event_us']-sum(warm)-call['elapsed_us']-sum(gaps))>1.:
            raise ValueError('single event timeline does not close')
    return statistics.pstdev(warm[-5:])/statistics.mean(warm[-5:])


def decode_history_series(setup, call, event, row):
    """Explicit 16-prefix/19-target ends history; never a single common16 call."""
    import v08_model
    count=math.prod(setup['grid'])
    if (event.get('protocol'),event.get('prefix_calls'),event.get('target_calls'),event.get('prefix_swizzle'),event.get('target_swizzle'),event.get('ctas'),event.get('head_words')) != ('prefix16-target19-ends-v1',16,19,1,row['swizzle'],count,16):
        raise ValueError('history series contract changed')
    if len(event['heads'])!=35*count*16 or len(event['call_elapsed_us'])!=35 or len(event['inter_call_event_us'])!=34:
        raise ValueError('history series is incomplete')
    if any(not math.isfinite(x) or x<=0 for x in event['call_elapsed_us']) or any(not math.isfinite(x) or x<0 for x in event['inter_call_event_us']):
        raise ValueError('invalid history event timing')
    for name,calls,gaps in [('prefix_elapsed_us',16,15),('series_elapsed_us',35,34)]:
        if abs(event[name]-sum(event['call_elapsed_us'][:calls])-sum(event['inter_call_event_us'][:gaps]))>1.:
            raise ValueError('history event timeline does not close')
    if any(abs(a-b)>1e-3 for a,b in zip(call['warmup_us'],event['call_elapsed_us'][:16])) or abs(call['elapsed_us']-event['call_elapsed_us'][-1])>1e-3:
        raise ValueError('history summary does not identify its prefix and last target')
    if event['prefix_elapsed_us']!=call['history_prefix_event_us'] or event['series_elapsed_us']!=call['history_event_us'] or event['inter_call_event_us']!=call['history_inter_call_event_us']:
        raise ValueError('series event timeline differs from its call summary')
    decoded=[]
    for j in range(35):
        sw=1 if j<16 else row['swizzle']
        work=v08_model.scheduled_work(row['config'],row['m'],row['n'],setup['grid'],sw,row.get('raster_order','heuristic'))
        ctas=[]
        for c in range(count):
            h=event['heads'][(j*count+c)*16:(j*count+c+1)*16]
            if h[10] or not all(h[k] for k in (0,1,2,3)):
                raise ValueError('missing history CTA head')
            r1,r2=h[4:7],h[7:10]
            if row['config']!='cfg_b':
                if r1[2]!=r2[2]:raise ValueError('history consumers disagree')
                end_c,end_ns,tiles=r2
            else:
                last=max((r1,r2),key=lambda r:r[0] if r[2] else 0)
                end_c,end_ns=last[:2];tiles=r1[2]+r2[2]
            if tiles!=len(work[c]) or h[11]!=h[2] or (r2[2] and h[12]!=h[2]):
                raise ValueError('history work/SM identity changed')
            if tiles and (end_c<=h[0] or end_ns<=h[1]):
                raise ValueError('history CTA interval invalid')
            ctas.append(dict(cta=c,sm=h[2]-1,tiles=tiles,entry_c=h[0],entry_ns=h[1],end_c=end_c,end_ns=end_ns,
                roles=[dict(final_c=r[0],final_ns=r[1],tiles=r[2]) for r in (r1,r2)]))
        active=[c for c in ctas if c['tiles']];largest=max(c['tiles'] for c in active)
        first=min(c['entry_ns'] for c in ctas);last=max(c['end_ns'] for c in active)
        if decoded and first<decoded[-1]['last_final_ns']:
            raise ValueError('history calls overlap or were reordered')
        decoded.append(dict(index=j,phase='prefix' if j<16 else 'target',swizzle=sw,
            event_us=event['call_elapsed_us'][j],first_entry_ns=first,last_final_ns=last,
            window_us=(last-first)/1000,max_cta_cycles=max(c['end_c']-c['entry_c'] for c in active),
            frequency_ghz=statistics.median((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns']) for c in active if c['tiles']==largest),
            ctas=ctas))
    for c in range(count):
        saved=event['heads'][(34*count+c)*16:(34*count+c+1)*16]
        if call['trace'][c*setup['trace_words']:c*setup['trace_words']+16]!=saved:
            raise ValueError('last-target trace is not the last series slot')
    return dict(protocol=event['protocol'],calls=decoded,prefix_elapsed_us=event['prefix_elapsed_us'],
        series_elapsed_us=event['series_elapsed_us'],inter_call_event_us=event['inter_call_event_us'],
        inter_call_globaltimer_us=[(b['first_entry_ns']-a['last_final_ns'])/1000 for a,b in zip(decoded,decoded[1:])],
        numeric_check_scope='Only the final target has the process numeric output check; every call has complete ends/work/SM records.',
        scope='Recorded history includes 16 prefix and 19 target calls; the call summary is the last target, after 18 preceding target calls.')


def replay(root,record,row,retain_roles=False):
    raw=root/record['raw']
    if common.sha(raw)!=record['raw_sha256']:raise ValueError('raw process identity changed')
    with gzip.open(raw,'rt') as stream:events={ev['event']:ev for line in stream if line.strip() for ev in [json.loads(line)]}
    setup,call,check=(events[k] for k in ['setup','call','check'])
    for key in ['m','n','k','lda','ldb','ldd','zero_m','zero_n','storage_m','storage_n','swizzle']:
        if setup[key]!=row[key]:raise ValueError('setup coordinate changed: '+key)
    mode=row.get('input_mode','dyadic');seed=row.get('seed',17)
    if setup.get('input_mode','dyadic')!=mode or setup.get('seed',17)!=seed:
        raise ValueError('input mode or seed changed')
    if setup.get('requested_sm_count',0)!=row.get('sm_count',0):raise ValueError('requested SM count changed')
    if setup.get('raster_order','heuristic')!=row.get('raster_order','heuristic'):
        raise ValueError('requested raster order changed')
    if row.get('raster_order','heuristic')!='heuristic' and setup.get('resolved_raster_order')!=row['raster_order']:
        raise ValueError('resolved raster order differs from explicit request')
    for key in ['alloc_lda','alloc_ldb','input_map_m','input_map_n']:
        if key in row and setup.get(key)!=row[key]:raise ValueError('input allocation changed: '+key)
    if 'input_map_m' in row or 'input_map_n' in row:
        if (setup['initialized_input_rows'],setup['initialized_input_columns'])!=(row['storage_m'],row['storage_n']):
            raise ValueError('matched input initialization extent changed')
        expected_zero=(max(0,row['storage_m']-row['m'])*row['k'] if row['zero_m']==row['m'] else 0)
        expected_zero+=(row['k']*max(0,row['storage_n']-row['n']) if row['zero_n']==row['n'] else 0)
        witness=events['input_zero_storage']
        if witness['checked']!=expected_zero or witness['nonzero']:
            raise ValueError('explicit zero storage witness failed')
    if check['status']!='ok' or check['padding_errors'] or len(check['checked_values'])!=4096:
        raise ValueError('incorrect or incomplete GEMM check')
    if len(set(check['checked_indices']))!=4096:raise ValueError('duplicate sampled values')
    if mode=='random':
        if setup['check_atol']!=2**-20 or setup['check_sum_abs_rtol']!=2**-21:
            raise ValueError('random input tolerance changed')
        expected_values,tolerances=random_references(check['checked_indices'],row['m'],row['n'],row['k'],seed)
    elif mode not in ['dyadic','zero']:
        raise ValueError('unknown input mode')
    zero_count=0
    for q,(index,value) in enumerate(zip(check['checked_indices'],check['checked_values'])):
        i,j=divmod(index,row['n'])
        if not 0<=i<row['m']:raise ValueError('invalid check position')
        zero=mode=='zero' or (row['zero_m']>=0 and i>=row['zero_m']) or (row['zero_n']>=0 and j>=row['zero_n'])
        zero_count+=zero
        expected=0 if zero else expected_values[q] if mode=='random' else reference(i%17,j%17,row['k'],seed)
        tolerance=(2**-20 if zero else tolerances[q]) if mode=='random' else 0
        if value is None or not math.isfinite(value) or abs(value-expected)>tolerance:
            raise ValueError('wrong GEMM output at '+str((i,j)))
    if row['kind']=='explicit_zero' and zero_count<1:raise ValueError('zero panel unchecked')
    validate_warmup(setup,call,row)
    result=dict(case=row['id'],variant=record['variant'],trial=record['trial'],elapsed_us=call['elapsed_us'],checked_values=4096,zero_values=zero_count,ctas=[])
    if 'b_offset_bytes' in row or 'b_allocation_extra_bytes' in row:
        result['b_address']={key:setup[key] for key in ('b_buffer_ptr','target_b_ptr','b_offset_bytes','b_allocation_extra_bytes','b_payload_elements','allocated_b_elements')}
    if row.get('history_mode')=='series':
        if setup['trace_version']!='v08-ends' or 'history_series' not in events:
            raise ValueError('series requires its explicit ends history record')
        result['history_series']=decode_history_series(setup,call,events['history_series'],row)
        result['history_protocol']='prefix16-target19-ends-v1'
        result['preceding_target_calls']=18
        return result
    if record['variant'] in ['plain','wide']:return result
    if setup.get('trace_version') in ('r13-stage-sparse-v1','r13-stage-progress-v1','r13-stage-handoff-v1','r13-epilogue-order-v1','r13-output-background-v1'):
        key='sparse' if setup['trace_version']=='r13-stage-sparse-v1' else 'progress'
        result[key]=decode_stage_sparse(setup,call)
        return result
    tile_words=setup.get('trace_tile_words',6)
    words=call['trace'];ctas=math.prod(setup['grid']);width=16+2*64*tile_words
    if len(words)!=ctas*width:raise ValueError('wrong trace size')
    dual=setup['trace_version']=='r18-dual-clock-events'
    if tile_words not in [6,10] or (dual and tile_words!=10):raise ValueError('wrong tile layout')
    schedule='pingpong' if row['config']=='cfg_b' else 'cooperative'
    seen=[]
    light=setup['trace_version']=='r18-light-events'
    inferred_work=None
    if light:
        from v08_model import scheduled_work
        inferred_work=scheduled_work(row['config'],row['m'],row['n'],setup['grid'],row['swizzle'],row.get('raster_order','heuristic'))
    for c in range(ctas):
        head=words[c*width:(c+1)*width]
        if head[10] or not head[0] or not head[2]:raise ValueError('missing/overflow CTA trace')
        roles=[];coords=[]
        for role in range(2):
            end,ns,count=head[4+3*role:7+3*role]
            if count>64 or head[11+role]!=head[2]:raise ValueError('invalid count/SM-local clock boundary')
            tiles=[];tiles_ns=[];work=[]
            for j in range(count):
                x=16+(role*64+j)*tile_words;tile=head[x:x+4];mi,ni=head[x+4:x+6]
                if light:
                    if schedule!='cooperative' or count!=len(inferred_work[c]):raise ValueError('light work-count mismatch')
                    mi,ni=(q+1 for q in inferred_work[c][j])
                if not all(tile) or min(mi,ni)<=0 or not tile[0]<=tile[1]<=tile[2]<=tile[3]<=end:
                    raise ValueError('invalid tile events/coordinates')
                tiles.append(tile);work.append((mi-1,ni-1))
                if dual:
                    times=head[x+6:x+10]
                    if not head[1]<=head[13]<=times[0]<=times[1]<=times[2]<=times[3]<=ns:
                        raise ValueError('invalid direct globaltimer tile events')
                    tiles_ns.append(times)
            roles.append(dict(final_c=end,final_ns=ns,tiles=tiles,tiles_ns=tiles_ns));coords.append(work)
        if schedule=='cooperative':
            if coords[0]!=coords[1]:raise ValueError('consumer work mismatch')
            work=coords[0]
        else:
            work=[coords[j%2][j//2] for j in range(len(coords[0])+len(coords[1]))]
        entry=dict(entry_c=head[0],entry_ns=head[1],smid=head[2]-1,prod_c=head[3],roles=roles)
        tiles,end_c,end_ns=model.cta_timeline(entry,schedule)
        cta=dict(cta=c,entry_c=head[0],entry_ns=head[1],sm=head[2]-1,prod_c=head[3],
                 end_c=end_c,end_ns=end_ns,tiles=tiles,work=work)
        if dual:
            # Reuse the same role merge with ns-valued endpoints; no clock interpolation.
            ns_roles=[dict(tiles=r['tiles_ns'],final_c=r['final_ns'],final_ns=r['final_ns']) for r in roles]
            cta['tiles_ns']=model.cta_timeline(dict(roles=ns_roles),schedule)[0]
            cta['prod_ns']=head[13]
        if retain_roles:
            cta['roles']=roles
            cta['role_work']=coords
        result['ctas'].append(cta)
        seen+=work
    if 'producer_delay_ns' in row:
        values=call.get('producer_delay_trace',[])
        if len(values)!=4 or not all(values):raise ValueError('missing producer delay endpoints')
        bc,bn,ec,en=values;c=result['ctas'][55]
        if not (c['entry_c']<=bc<=ec<=c['prod_c'] and c['entry_ns']<=bn<=en<=c['prod_ns']):
            raise ValueError('producer delay endpoints do not precede first work')
        if en-bn<row['producer_delay_ns']:raise ValueError('requested producer delay not observed')
        result['producer_delay']=dict(cta=55,requested_ns=row['producer_delay_ns'],begin_c=bc,begin_ns=bn,
            end_c=ec,end_ns=en,duration_cycles=ec-bc,duration_ns=en-bn)
    if 'first_supply_events' in row:
        values=call.get('first_supply_trace',[]);enabled=bool(row['first_supply_events'])
        if len(values)!=4 or (enabled and not all(values)) or (not enabled and any(values)):
            raise ValueError('first supply observation record is incomplete or enabled unexpectedly')
        first=dict(cta=55,enabled=enabled,protocol='r13-first-supply-v1')
        if enabled:
            ic,ins,wc,wns=values;c=result['ctas'][55]
            if not (c['prod_c']<=ic<=c['end_c'] and c['prod_ns']<=ins<=c['end_ns']):
                raise ValueError('first TMA issue marker is outside its producer call')
            if not (c['entry_c']<=wc<=c['tiles'][0][0] and c['entry_ns']<=wns<=c['tiles_ns'][0][0]):
                raise ValueError('first consumer wait marker is outside its local interval')
            first.update(issue_end_c=ic,issue_end_ns=ins,wait_begin_c=wc,wait_begin_ns=wns)
        result['first_supply']=first
    if 'ready_observer' in row:
        release_values=call.get('first_release_trace',[]) if row.get('ready_generation_bound')=='first_release' else None
        result['ready_observer']=decode_ready_observer(call.get('ready_observer_trace',[]),row['ready_observer'],result['ctas'][55],release_values)
    if len(seen)!=len(set(seen)):raise ValueError('duplicated logical work tiles')
    tm=256 if row['config'].startswith('cfg_c') else 128;tn=128
    required={(i,j) for i in range(model.cdiv(row['m'],tm)) for j in range(model.cdiv(row['n'],tn))}
    if not required<=set(seen):raise ValueError('actual scheduler omitted real tile')
    return result


def tile_class(row,mi,ni):
    axis=0 if row['config'].startswith('cfg_a') else 1
    q=(mi,ni)[axis];extent=row['m'] if axis==0 else row['n'];tile=128
    if q*tile>=extent:return 'oob'
    zero=row['zero_m'] if axis==0 else row['zero_n']
    if zero>=0 and q*tile>=zero:return 'explicit_zero'
    if not row['config'].endswith('1') and (q^1)*tile>=extent:return 'partner'
    if zero>=0 and (q^1)*tile>=zero:return 'zero_partner'
    if q==extent//tile and extent%tile:return 'partial'
    return 'other'


def analyze(root,output=None):
    common.verify(root)
    for name,h in json.loads((root/'build/sass_hashes.json').read_text()).items():
        if common.sha(root/'build'/name)!=h:raise ValueError('SASS changed')
    dest=output or root/'analysis';dest.mkdir(parents=True,exist_ok=True)
    rows=json.loads((root/'cases.json').read_text());results=[];classes=[];tails=[];paired=[];all_records={};increments=[];rejected=[]
    for row in rows:
        recs=[]
        for path in sorted((root/'samples'/row['id']).glob('*.json')):
            record=json.loads(path.read_text())
            if record['returncode']:
                if 'last five warmups CV exceeds 2%' in record['stderr']:
                    rejected.append(dict(path=str(path.relative_to(root)),reason='warmup_not_converged_no_output_check',record=record));continue
                raise ValueError('failed process remains unqualified')
            recs.append(replay(root,record,row))
        plain=[r['elapsed_us'] for r in recs if r['variant']=='plain'];trace=[r['elapsed_us'] for r in recs if r['variant']=='stamped']
        if not plain or len(plain)!=len(trace):raise ValueError('missing paired plain/trace')
        all_records[row['id']]=recs
        disturbance=statistics.median(trace)/statistics.median(plain)-1
        trace_cv=statistics.pstdev(trace)/statistics.mean(trace)
        cv=statistics.pstdev(plain)/statistics.mean(plain)
        qualified=len(plain)>=10 and cv<=.05
        results.append(dict(**row,plain_us=statistics.median(plain),plain_cv=cv,processes=len(plain),
                            perturbation=disturbance,trace_cv=trace_cv,plain_qualified=qualified,trace_qualified=qualified and abs(disturbance)<=.05 and trace_cv<=.05))
        per_class=defaultdict(list)
        for rec in recs:
            if rec['variant']!='stamped':continue
            active=[c for c in rec['ctas'] if c['tiles']]
            start=min(c['entry_ns'] for c in rec['ctas']);end=max(c['end_ns'] for c in active)
            critical=max(active,key=lambda c:c['end_ns'])
            durations=[c['end_c']-c['entry_c'] for c in active]
            tails.append(dict(case=row['id'],trial=rec['trial'],critical_cta=critical['cta'],critical_sm=critical['sm'],
                critical_work=critical['work'],sm_distribution=dict(Counter(c['sm'] for c in rec['ctas'])),critical_cycles=critical['end_c']-critical['entry_c'],median_cta_cycles=statistics.median(durations),
                max_cta_cycles=max(durations),critical_entry_skew_ns=critical['entry_ns']-start,
                entry_spread_ns=max(c['entry_ns'] for c in rec['ctas'])-start,envelope_ns=end-start,
                tiles_count_histogram=dict(Counter(len(c['tiles']) for c in rec['ctas'])),
                critical_mainloop_cycles=[t[1]-t[0] for t in critical['tiles']],
                critical_epilogue_cycles=[t[3]-t[2] for t in critical['tiles']],
                critical_permit_cycles=[t[2]-t[1] for t in critical['tiles']],
                critical_handoff_cycles=[b[0]-a[1] for a,b in zip(critical['tiles'],critical['tiles'][1:])]))
            groups=defaultdict(list)
            for c in active:
                for j,((mi,ni),t) in enumerate(zip(c['work'],c['tiles'])):
                    cls=tile_class(row,mi,ni) if row['config']!='cfg_b' else 'all'
                    groups[(j,cls)].append(t[1]-t[0])
            for key,vals in groups.items():per_class[key].append(statistics.median(vals))
        for (j,cls),vals in sorted(per_class.items()):
            classes.append(dict(case=row['id'],round=j,cls=cls,mainloop_cycles=statistics.median(vals),
                                stddev=statistics.pstdev(vals),processes=len(vals),qualified=qualified and abs(disturbance)<=.05 and trace_cv<=.05))
    for row in results:
        if row['kind'] not in ['oob','explicit_zero'] or row['config'].endswith('1'):continue
        base=next(x for x in results if x['group']==row['group'] and x['kind']=='ordinary' and x['config']==row['config'])
        paired.append(dict(case=row['id'],baseline=base['id'],delta_plain_us=row['plain_us']-base['plain_us'],
                           relative_plain=row['plain_us']/base['plain_us']-1,
                           qualified=row['plain_qualified'] and base['plain_qualified']))
    for row in results:
        if row['kind'] not in ['oob','explicit_zero','partial_even','partial_odd'] or row['config'].endswith('1'):continue
        base=next(x for x in results if x['group']==row['group'] and x['kind']=='ordinary' and x['config']==row['config'])
        controls={r['trial']:r for r in all_records[base['id']] if r['variant']=='stamped'}
        deltas=defaultdict(list)
        for rec in all_records[row['id']]:
            if rec['variant']!='stamped':continue
            control=controls[rec['trial']]
            if [c['work'] for c in rec['ctas']]!=[c['work'] for c in control['ctas']]:
                raise ValueError('boundary comparison changed physical tile list/CTA assignment')
            current=defaultdict(list)
            # Boundary columns/rows reached in each round; distinguish affected
            # neighbors from other CTAs, instead of pooling every valid tile.
            affected=defaultdict(set)
            for c in rec['ctas']:
                for j,(mi,ni) in enumerate(c['work']):
                    if tile_class(row,mi,ni)!='other':affected[j].add(ni if row['config']=='cfg_a' else mi)
            for c,b in zip(rec['ctas'],control['ctas']):
                for j,((mi,ni),t,bt) in enumerate(zip(c['work'],c['tiles'],b['tiles'])):
                    cls=tile_class(row,mi,ni)
                    transverse=ni if row['config']=='cfg_a' else mi
                    if cls=='other' and transverse in affected[j]:cls='same_column' if row['config']=='cfg_a' else 'same_row'
                    current[j,cls].append((t[1]-t[0])-(bt[1]-bt[0]))
            for key,values in current.items():deltas[key].append(statistics.median(values))
        for (j,cls),values in sorted(deltas.items()):
            increments.append(dict(case=row['id'],baseline=base['id'],round=j,cls=cls,k=row['k'],
             delta_cycles=statistics.median(values),stddev=statistics.pstdev(values),process_deltas=values,
             qualified=row['trace_qualified'] and base['trace_qualified'],
             scope='same physical coordinates/CTA assignment; per-process paired difference'))
    result=dict(conditions=results,rejected_processes=rejected,checked_values=sum(r['processes']*2*4096 for r in results),
                plain_qualified=sum(r['plain_qualified'] for r in results),trace_qualified=sum(r['trace_qualified'] for r in results),
                paired_complete_time=paired,analyzer_sha256=common.sha(Path(__file__)),scope='full traces record actual CUTLASS coordinates; light R18 traces infer unchanged static mapping checked against full cohort; traces failing 5% perturbation do not yield cycle parameters')
    for name,data in [('rules.json',result),('classes.json',classes),('tails.json',tails),('paired-tile-increments.json',increments)]:common.write_json(dest/name,data)
    print('conditions',len(results),'plain qualified',result['plain_qualified'],'trace qualified',result['trace_qualified'])

def analyze_v08_padding(root, output):
    """Separate V08F padding observations by round and total CTA work; no causal fit."""
    import v08_model
    med = statistics.median
    rows = json.loads((root / 'cases.json').read_text())
    cases = []
    for row in rows:
        if row['swizzle'] != 8 or not any(label in row['id'] for label in ('_padM_', '_padN_', '_padMN_')):
            continue
        pooled, groups, differences = defaultdict(list), defaultdict(list), defaultdict(list)
        for path in sorted((root / 'samples' / row['id']).glob('stamped-*.json')):
            record = json.loads(path.read_text())
            if record['returncode']:
                continue
            observed = replay(root, record, row)
            current = defaultdict(list)
            for cta in observed['ctas']:
                for j, (tile, coord) in enumerate(zip(cta['tiles'], cta['work'])):
                    cls = v08_model.tile_class(row['config'], row, *coord)
                    cycles = tile[1] - tile[0]
                    current[j, len(cta['tiles']), cls].append(cycles)
                    if j:
                        pooled[cls].append(cycles)
            for (j, count, cls), values in current.items():
                groups[j, count, cls].append(dict(trial=record['trial'], windows=len(values), cycles=med(values)))
                if cls != 'in' and (j, count, 'in') in current:
                    differences[j, count, cls].append(dict(trial=record['trial'],
                        cycles=med(values) - med(current[j, count, 'in'])))
        kt = v08_model.cdiv(row['k'], 64)
        strata = []
        for (j, count, cls), values in sorted(groups.items()):
            deltas = differences[j, count, cls]
            strata.append(dict(round=j, cta_tiles=count, cls=cls, processes=values,
                cycles_per_ktile=med(v['cycles'] for v in values) / kt,
                delta_processes=deltas,
                delta_per_ktile=med(v['cycles'] for v in deltas) / kt if deltas else None,
                delta_stddev_per_ktile=statistics.pstdev(v['cycles'] / kt for v in deltas) if deltas else None))
        timings = defaultdict(list)
        for path in (root / 'samples' / row['id']).glob('*.json'):
            record = json.loads(path.read_text())
            if not record['returncode']:
                timings[record['variant']].append(record['elapsed_us'])
        perturbation = med(timings['stamped']) / med(timings['plain']) - 1
        cases.append(dict(**row, kt=kt, perturbation=perturbation,
            trace_qualified=(min(len(timings[v]) for v in ('plain', 'stamped')) >= 10
                and abs(perturbation) <= .05
                and max(statistics.pstdev(timings[v]) / statistics.mean(timings[v]) for v in ('plain', 'stamped')) <= .05),
            pooled_later_tiles={cls: dict(windows=len(v), cycles=med(v),
                delta_per_ktile=(med(v) - med(pooled['in'])) / kt) for cls, v in sorted(pooled.items())},
            strata=strata))
    output.mkdir(parents=True, exist_ok=False)
    common.write_json(output / 'padding.json', dict(cases=cases, analyzer_sha256=common.sha(Path(__file__)),
        scope='Posthoc V08F. Groups share round and CTA work count, not coordinates or absolute event time. No physical service attribution or frozen prediction.'))
    print('padding cases', len(cases), '->', output)


def analyze_input_map_pairs(root, output):
    """Matched logical output/work lists; only the input descriptor bounds differ."""
    from analyze_r13_sm import select_attempts
    from v08_model import tile_class as logical_tile_class
    med = statistics.median
    common.verify(root)
    rows = json.loads((root / 'cases.json').read_text())
    setups = {s['case']: s['setup'] for s in json.loads((root / 'static_setup.json').read_text())}
    env = json.loads((root / 'environment.json').read_text())
    observations, summaries, all_hashes = {}, [], {}
    variants = ('plain', 'wide', 'stamped', 'dual')
    for row in rows:
        selected, failed, ignored = select_attempts(root, row['id'], variants)
        if ignored or any(sum(x['record']['variant'] == v for x in selected) != 10 for v in variants):
            raise ValueError('ten successful processes per variant required')
        if setups[row['id']]['gpu_uuid'] != env['gpu'].split(',')[0]:
            raise ValueError('input-map cases are not on the recorded GPU')
        by_variant = {v: {} for v in variants}
        for item in selected:
            rec = item['record']; obs = replay(root, rec, row)
            by_variant[rec['variant']][rec['trial']] = obs
            all_hashes[rec['raw']] = rec['raw_sha256']
        observations[row['id']] = by_variant
        times = {v: [by_variant[v][t]['elapsed_us'] for t in range(10)] for v in variants}
        summaries.append(dict(condition=row, median_us={v: med(t) for v,t in times.items()},
            cv={v: statistics.pstdev(t)/statistics.mean(t) for v,t in times.items()},
            wide_over_plain=med(times['wide'])/med(times['plain'])-1,
            dual_over_wide=med(times['dual'])/med(times['wide'])-1,
            dual_over_wide_pairs=[b/a-1 for a,b in zip(times['wide'],times['dual'])],
            failed_attempts=failed))
    pairs = []
    fixed = ('config','m','n','k','lda','ldb','ldd','storage_m','storage_n','alloc_lda','alloc_ldb',
             'zero_m','zero_n','swizzle','sm_count','input_mode','seed')
    for group in sorted({r['group'] for r in rows}):
        control = next(r for r in rows if r['group']==group and r['input_path']=='oob')
        mapped = next(r for r in rows if r['group']==group and r['input_path']=='address_zero')
        if any(control[k] != mapped[k] for k in fixed):
            raise ValueError('paired source-path conditions changed another variable')
        if setups[control['id']]['grid'] != setups[mapped['id']]['grid']:
            raise ValueError('input descriptor changed the output grid')
        a,b = observations[control['id']], observations[mapped['id']]
        time_pairs = {v: [b[v][t]['elapsed_us']/a[v][t]['elapsed_us']-1 for t in range(10)] for v in variants}
        strata = defaultdict(list)
        kt = (control['k']+63)//64
        for trial in range(10):
            ac,bc = a['dual'][trial]['ctas'], b['dual'][trial]['ctas']
            if [c['work'] for c in ac] != [c['work'] for c in bc]:
                raise ValueError('paired descriptor path changed CTA work lists')
            groups = defaultdict(list)
            for x,y in zip(ac,bc):
                for j,coord in enumerate(x['work']):
                    cls = logical_tile_class(control['config'],control,*coord)
                    xc,yc = x['tiles'][j],y['tiles'][j]
                    xn,yn = x['tiles_ns'][j],y['tiles_ns'][j]
                    groups[j,len(x['work']),cls].append(((xc[1]-xc[0])/kt,(yc[1]-yc[0])/kt,
                                                        (xn[1]-xn[0])/kt,(yn[1]-yn[0])/kt))
            for key,values in groups.items():
                means = [statistics.fmean(v[i] for v in values) for i in range(4)]
                strata[key].append(dict(trial=trial,windows=len(values),oob_cycle_per_kt=means[0],
                    address_zero_cycle_per_kt=means[1],oob_ns_per_kt=means[2],address_zero_ns_per_kt=means[3],
                    delta_cycle_per_kt=means[1]-means[0],delta_ns_per_kt=means[3]-means[2]))
        groups = []
        for (j,T,cls),values in sorted(strata.items()):
            groups.append(dict(j=j,T=T,logical_class=cls,processes=values,
                median_delta_cycle_per_kt=med(x['delta_cycle_per_kt'] for x in values),
                median_delta_ns_per_kt=med(x['delta_ns_per_kt'] for x in values)))
        pairs.append(dict(group=group,control=control['id'],address_zero=mapped['id'],
            complete_time_relative_pairs=time_pairs,
            complete_time_relative_median={v:med(x) for v,x in time_pairs.items()},groups=groups))
    output.mkdir(parents=True,exist_ok=False)
    common.write_json(output/'input-map-pairs.json',dict(environment=env,cases=summaries,pairs=pairs,
        raw_sha256=all_hashes,analyzer_sha256=common.sha(Path(__file__)),
        scope='Address-zero minus TMA OOB with the same logical output, storage, grid and CTA coordinates. '
              'Stage windows are dual-observer measurements; logical pad classes do not change when the input map expands. '
              'These are paired process comparisons, not physical traffic counts or a frozen prediction.'))
    print('input-map pairs',len(pairs),'successful processes',len(all_hashes),'->',output)


def boundary_geometry(row,coord):
    """Logical output footprint, independent of input descriptor padding."""
    tm,tn,_=model.CONFIGS[row['config']]['tile']
    mi,ni=coord
    mr=max(0,min(tm,row['m']-mi*tm));nr=max(0,min(tn,row['n']-ni*tn))
    return 'oob' if not mr or not nr else 'partial' if mr<tm or nr<tn else 'real'


def maximum_set(values):
    top=max(values)
    return [i for i,v in enumerate(values) if abs(v-top)<1e-7]


def critical_replacements(ctas,prediction):
    """Diagnostic counterfactuals, never model input or a revised candidate set."""
    entry0=min(c['entry_ns'] for c in ctas)
    entries=[c['entry_ns']-entry0 for c in ctas]
    durations=[c['end_ns']-c['entry_ns'] for c in ctas]
    cycles=[c['end_c']-c['entry_c'] for c in ctas]
    ends=[e+d for e,d in zip(entries,durations)]
    pred=prediction['cta_duration_ns'];frozen=prediction['critical_candidates']
    if len(pred)!=len(ctas):raise ValueError('prediction and observed CTA counts differ')
    with_entry=[e+d for e,d in zip(entries,pred)]
    last=maximum_set(ends);local=maximum_set(durations)
    representative=prediction['critical_cta']
    return dict(observed_last=last,observed_longest_ns=local,observed_longest_cycles=maximum_set(cycles),
        frozen_candidates=frozen,frozen_representative=representative,
        frozen_hit=bool(set(last)&set(frozen)),candidate_count=len(frozen),
        entry_only_candidates=maximum_set(with_entry),
        entry_only_hit=bool(set(last)&set(maximum_set(with_entry))),
        internal_only_candidates=local,internal_only_hit=bool(set(last)&set(local)),
        entry_spread_ns=max(entries),observed_envelope_ns=max(ends),
        frozen_envelope_ns=max(pred),entry_only_envelope_ns=max(with_entry),
        internal_only_envelope_ns=max(durations),
        representative_duration_deficit=1-durations[representative]/max(durations[i] for i in last),
        representative_finish_deficit_ns=max(ends)-ends[representative],
        best_frozen_finish_deficit_ns=max(ends)-max(ends[i] for i in frozen),
        ctas=[dict(cta=c['cta'],sm=c['sm'],entry_offset_ns=e,duration_ns=d,duration_cycles=cy,
                   end_offset_ns=en,prediction_ns=p,work=c['work'])
              for c,e,d,cy,en,p in zip(ctas,entries,durations,cycles,ends,pred)])


def boundary_stage_rows(row,observed,prediction=None,source=''):
    """Same-role paired endpoints; merged CTA residuals are labeled separately."""
    rows=[]
    for c in observed['ctas']:
        for role,(r,work) in enumerate(zip(c['roles'],c['role_work'])):
            for j,(cy,ns,coord) in enumerate(zip(r['tiles'],r['tiles_ns'],work)):
                for stage,i,k in [('L',0,1),('E',2,3)]:
                    rows.append(dict(case=row['id'],trial=observed['trial'],source=source,cta=c['cta'],role=role,
                        j=j,T=len(work),phase='first' if j==0 else 'later',geometry=boundary_geometry(row,coord),
                        mi=coord[0],ni=coord[1],stage=stage,cycles=cy[k]-cy[i],ns=ns[k]-ns[i],
                        prediction_ns=None,measurement='same_role'))
            if work:
                rows.append(dict(case=row['id'],trial=observed['trial'],source=source,cta=c['cta'],role=role,
                    j=len(work)-1,T=len(work),phase='tail',geometry=boundary_geometry(row,work[-1]),
                    mi=work[-1][0],ni=work[-1][1],stage='tail',cycles=r['final_c']-r['tiles'][-1][3],
                    ns=r['final_ns']-r['tiles_ns'][-1][3],prediction_ns=None,measurement='same_role'))
        if prediction is not None:
            pe=prediction['cta_events'][c['cta']]['events']
            for j,(ns,coord,p) in enumerate(zip(c['tiles_ns'],c['work'],pe)):
                for stage,i,k in [('L',0,1),('E',2,3)]:
                    rows.append(dict(case=row['id'],trial=observed['trial'],source=source,cta=c['cta'],role=-1,
                        j=j,T=len(c['work']),phase='first' if j==0 else 'later',geometry=boundary_geometry(row,coord),
                        mi=coord[0],ni=coord[1],stage=stage,cycles=None,ns=ns[k]-ns[i],
                        prediction_ns=p[k]-p[i],measurement='merged_cta'))
            if c['work']:
                coord=c['work'][-1]
                rows.append(dict(case=row['id'],trial=observed['trial'],source=source,cta=c['cta'],role=-1,
                    j=len(c['work'])-1,T=len(c['work']),phase='tail',geometry=boundary_geometry(row,coord),
                    mi=coord[0],ni=coord[1],stage='tail',cycles=None,
                    ns=c['end_ns']-max(t[3] for t in c['tiles_ns']),
                    prediction_ns=prediction['cta_events'][c['cta']]['tail'],measurement='merged_cta'))
    return rows


def stage_process_summary(rows):
    """Reduce correlated CTA windows inside each process before case statistics."""
    grouped=defaultdict(list)
    keys=('case','trial','measurement','role','phase','geometry','stage')
    for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
    processes=[]
    for key,group in sorted(grouped.items()):
        value=dict(zip(keys,key),windows=len(group),ns=statistics.median(r['ns'] for r in group))
        cycles=[r['cycles'] for r in group if r['cycles'] is not None]
        if cycles:value['cycles']=statistics.median(cycles)
        predicted=[r for r in group if r['prediction_ns'] is not None]
        if predicted:
            value['prediction_ns']=statistics.median(r['prediction_ns'] for r in predicted)
            value['error_ns']=statistics.median(r['prediction_ns']-r['ns'] for r in predicted)
            rel=[r['prediction_ns']/r['ns']-1 for r in predicted if r['ns']>0]
            if rel:value['relative_error']=statistics.median(rel)
        processes.append(value)
    return processes


def old_boundary_pairs(base):
    """Reuse existing fixed-output input-map interventions, with role endpoints."""
    from analyze_r13_sm import select_attempts
    summaries=[];provenance={}
    fixed=('config','m','n','k','lda','ldb','ldd','storage_m','storage_n','alloc_lda','alloc_ldb',
           'zero_m','zero_n','swizzle','sm_count','input_mode','seed')
    for name in ('20261009-R18-first-fill-job738707','20261009-R18-first-fill-job738865'):
        root=base/name;rows=json.loads((root/'cases.json').read_text());cache={}
        for row in rows:
            selected,_,_=select_attempts(root,row['id'],('dual','plain'))
            calls={};times=defaultdict(list)
            for item in selected:
                rec=item['record'];obs=replay(root,rec,row,retain_roles=rec['variant']=='dual')
                times[rec['variant']].append(obs['elapsed_us'])
                provenance[str(root/rec['raw'])]=rec['raw_sha256']
                if rec['variant']=='dual':
                    calls[rec['trial']]=(obs,stage_process_summary(boundary_stage_rows(row,obs,source=str(root/rec['raw']))))
            cache[row['id']]=(calls,{v:statistics.median(t) for v,t in times.items()})
        for group in sorted({r['group'] for r in rows}):
            a=next(r for r in rows if r['group']==group and r['input_path']=='oob')
            b=next(r for r in rows if r['group']==group and r['input_path']=='address_zero')
            if any(a[k]!=b[k] for k in fixed):raise ValueError('input-map pair changed another factor')
            ca,ta=cache[a['id']];cb,tb=cache[b['id']];deltas=[]
            for trial in sorted(ca):
                oa,sa=ca[trial];ob,sb=cb[trial]
                if [c['work'] for c in oa['ctas']]!=[c['work'] for c in ob['ctas']]:raise ValueError('work changed')
                lookup={(r['role'],r['phase'],r['geometry'],r['stage']):r for r in sb}
                for r in sa:
                    key=r['role'],r['phase'],r['geometry'],r['stage'];s=lookup[key]
                    deltas.append(dict(trial=trial,role=key[0],phase=key[1],geometry=key[2],stage=key[3],
                        oob_ns=r['ns'],address_zero_ns=s['ns'],delta_ns=s['ns']-r['ns'],
                        oob_cycles=r['cycles'],address_zero_cycles=s['cycles'],delta_cycles=s['cycles']-r['cycles']))
            summaries.append(dict(root=str(root),group=group,oob=a,address_zero=b,
                plain_relative=tb['plain']/ta['plain']-1,dual_relative=tb['dual']/ta['dual']-1,
                oob_plain_us=ta['plain'],address_zero_plain_us=tb['plain'],processes=deltas))
    return dict(pairs=summaries,provenance=provenance,
        scope='Same output and allocation, changed input descriptor; independent processes grouped by trial, not one shared invocation. Positive delta means address-zero slower. No physical write-completion claim.')


def failure_diagnostics(base,output,calls_factory):
    """V09 post-hoc endpoint/geometry analysis. Does not modify predictions."""
    import csv
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    processes=[];stages=[];provenance={}
    selected=('v09d_cfg_a_P1_partial_first_and_later','v09d_cfg_c_P2_partial_later_sw8',
              'v09d_cfg_a_G3_A16','v09d_cfg_a_G3_base')
    for call in calls_factory(base):
        row=call['row'];obs=call['observed'];pred=call['prediction'];rec=call['record']
        if not pred.get('cta_duration_ns'):continue
        source=call.get('source',str(Path(base)/'20261009-V09-heldout-job739011'/rec['raw']))
        value=critical_replacements(obs['ctas'],pred)
        value.update(case=row['id'],trial=call['trial'],source=source,raw_sha256=rec['raw_sha256'])
        if row['id'] not in selected:value.pop('ctas')
        elif row['config']!='cfg_b':
            for c,detail,pp in zip(obs['ctas'],value['ctas'],pred['cta_events']):
                ns=c['tiles_ns'];pe=pp['events']
                actual=dict(prefix=ns[0][0]-c['entry_ns'],L=sum(t[1]-t[0] for t in ns),
                    E=sum(t[3]-t[2] for t in ns),tail=c['end_ns']-ns[-1][3])
                actual['gaps']=detail['duration_ns']-sum(actual.values())
                predicted=dict(prefix=pe[0][0],L=sum(t[1]-t[0] for t in pe),
                    E=sum(t[3]-t[2] for t in pe),tail=pp['tail'])
                predicted['gaps']=detail['prediction_ns']-sum(predicted.values())
                detail['observed_components_ns']=actual;detail['predicted_components_ns']=predicted
        processes.append(value);provenance[source]=rec['raw_sha256']
        if row['id'] in selected:stages.extend(boundary_stage_rows(row,obs,pred,source))
    stage_summary=stage_process_summary(stages)
    grouped=defaultdict(list)
    for p in processes:grouped[p['case']].append(p)
    cases=[]
    for case,group in sorted(grouped.items()):
        value=dict(case=case,processes=len(group),candidate_count=group[0]['candidate_count'])
        for k in ('frozen_hit','entry_only_hit','internal_only_hit'):value[k]=sum(p[k] for p in group)
        for k in ('entry_spread_ns','observed_envelope_ns','frozen_envelope_ns','entry_only_envelope_ns',
                  'internal_only_envelope_ns','representative_duration_deficit','representative_finish_deficit_ns','best_frozen_finish_deficit_ns'):
            value[k]=statistics.median(p[k] for p in group)
        cases.append(value)
    common.write_json(output/'C-processes.json',dict(processes=processes,provenance=provenance))
    common.write_json(output/'C-summary.json',dict(cases=cases,stages=stage_summary,
        scope='Development diagnostics: fixed frozen candidates; entry-only replacement adds measured entry offsets to predicted duration; internal-only removes entry offsets from measured duration. These are observed counterfactuals, not forecasts. Same-role stage endpoints; merged CTA model residuals kept separate.'))
    with gzip.open(output/'C-stages.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stages[0]));writer.writeheader();writer.writerows(stages)
    common.write_json(output/'C-old-input-pairs.json',old_boundary_pairs(Path(base)))
    return dict(cases=len(cases),processes=len(processes),stage_rows=len(stages))


def paired_boundary_windows(row,control,mapped,control_source='',mapped_source=''):
    """Match CTA/role/work within two independent processes of one trial."""
    if control['trial']!=mapped['trial']:raise ValueError('unmatched trial')
    if [[c['work'],c['role_work']] for c in control['ctas']]!=[[c['work'],c['role_work']] for c in mapped['ctas']]:
        raise ValueError('descriptor pair changed work coordinates')
    left=boundary_stage_rows(row,control,source=control_source)
    right=boundary_stage_rows(row,mapped,source=mapped_source)
    keys=('cta','role','j','stage')
    lookup={tuple(r[k] for k in keys):r for r in right}
    if len(lookup)!=len(left):raise ValueError('descriptor pair has different role windows')
    paired=[]
    for a in left:
        b=lookup[tuple(a[k] for k in keys)]
        cm,cn=model.CONFIGS[row['config']]['cluster']
        m0,n0=(a['mi']//cm)*cm,(a['ni']//cn)*cn
        cluster_classes={boundary_geometry(row,(mi,ni)) for mi in range(m0,m0+cm) for ni in range(n0,n0+cn)}
        cluster_boundary='partial' if 'partial' in cluster_classes else 'oob' if 'oob' in cluster_classes else 'interior'
        paired.append(dict(trial=a['trial'],cta=a['cta'],role=a['role'],j=a['j'],T=a['T'],cluster_boundary=cluster_boundary,
            mi=a['mi'],ni=a['ni'],phase=a['phase'],geometry=a['geometry'],stage=a['stage'],
            oob_ns=a['ns'],address_zero_ns=b['ns'],delta_ns=b['ns']-a['ns'],
            oob_cycles=a['cycles'],address_zero_cycles=b['cycles'],delta_cycles=b['cycles']-a['cycles'],
            oob_source=control_source,address_zero_source=mapped_source))
    return paired


def control_diagnostics(root,output):
    """Same-card P1 input-map intervention; no old-card frozen parameters."""
    import csv
    from analyze_r13_sm import select_attempts
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    common.verify(root)
    rows=[r for r in json.loads((root/'cases.json').read_text()) if r['group']=='r18_partial_map']
    a=next(r for r in rows if r['input_map_m']==960);b=next(r for r in rows if r['input_map_m']==1024)
    if {k:v for k,v in a.items() if k not in ('id','input_map_m')}!={k:v for k,v in b.items() if k not in ('id','input_map_m')}:
        raise ValueError('P1 descriptor intervention changed another field')
    setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    fixed=('grid','allocated_a_elements','allocated_b_elements','storage_m','storage_n',
           'initialized_input_rows','initialized_input_columns','gpu_uuid','stages','smem')
    if any(setups[a['id']][k]!=setups[b['id']][k] for k in fixed):raise ValueError('P1 setup mismatch')
    cache={};cases=[];provenance={};last_ctas=[]
    for row in rows:
        selected,failed,_=select_attempts(root,row['id'],('plain','dual'))
        variants={v:{} for v in ('plain','dual')};records={}
        for item in selected:
            rec=item['record'];obs=replay(root,rec,row,retain_roles=rec['variant']=='dual')
            variants[rec['variant']][rec['trial']]=obs
            source=str(root/rec['raw']);provenance[source]=rec['raw_sha256'];records[rec['variant'],rec['trial']]=source
            if rec['variant']=='dual':
                ctas=obs['ctas'];earliest=min(c['entry_ns'] for c in ctas);last=maximum_set([c['end_ns'] for c in ctas])
                details=[]
                for i in last:
                    c=ctas[i];ns=c['tiles_ns']
                    details.append(dict(cta=i,sm=c['sm'],work=c['work'],geometry=[boundary_geometry(row,w) for w in c['work']],
                        entry_offset_ns=c['entry_ns']-earliest,duration_ns=c['end_ns']-c['entry_ns'],
                        prefix_ns=ns[0][0]-c['entry_ns'],L_ns=[t[1]-t[0] for t in ns],E_ns=[t[3]-t[2] for t in ns],
                        tail_ns=c['end_ns']-ns[-1][3]))
                last_ctas.append(dict(case=row['id'],trial=rec['trial'],source=source,
                    envelope_ns=max(c['end_ns'] for c in ctas)-earliest,
                    entry_spread_ns=max(c['entry_ns'] for c in ctas)-earliest,last=details))
        if any(set(variants[v])!=set(range(10)) for v in variants):raise ValueError('P1 needs ten successful processes per variant')
        times={v:[variants[v][t]['elapsed_us'] for t in range(10)] for v in variants}
        med={v:statistics.median(t) for v,t in times.items()}
        cases.append(dict(case=row['id'],condition=row,median_us=med,
            cv={v:statistics.pstdev(t)/statistics.mean(t) for v,t in times.items()},
            dual_over_plain=med['dual']/med['plain']-1,times_us=times,failed_attempts=failed))
        cache[row['id']]=(variants,records)
    ac,ar=cache[a['id']];bc,br=cache[b['id']];windows=[]
    for trial in range(10):
        windows.extend(paired_boundary_windows(a,ac['dual'][trial],bc['dual'][trial],ar['dual',trial],br['dual',trial]))
    groupkeys=('trial','role','phase','geometry','cluster_boundary','stage','j','T')
    grouped=defaultdict(list)
    for w in windows:grouped[tuple(w[k] for k in groupkeys)].append(w)
    processes=[]
    for key,group in sorted(grouped.items()):
        value=dict(zip(groupkeys,key),windows=len(group))
        for metric in ('oob_ns','address_zero_ns','delta_ns','oob_cycles','address_zero_cycles','delta_cycles'):
            value[metric]=statistics.median(r[metric] for r in group)
        processes.append(value)
    groups=defaultdict(list)
    for p in processes:groups[tuple(p[k] for k in groupkeys[1:])].append(p)
    summary=[]
    for key,group in sorted(groups.items()):
        value=dict(zip(groupkeys[1:],key),processes=len(group))
        for metric in ('oob_ns','address_zero_ns','delta_ns','oob_cycles','address_zero_cycles','delta_cycles'):
            value[metric]=statistics.median(r[metric] for r in group)
        value['delta_ns_min']=min(r['delta_ns'] for r in group);value['delta_ns_max']=max(r['delta_ns'] for r in group)
        summary.append(value)
    contrasts={v:dict(oob_us=statistics.median(ac[v][t]['elapsed_us'] for t in range(10)),
        address_zero_us=statistics.median(bc[v][t]['elapsed_us'] for t in range(10)),
        paired_relative=[bc[v][t]['elapsed_us']/ac[v][t]['elapsed_us']-1 for t in range(10)]) for v in ('plain','dual')}
    for v in contrasts:contrasts[v]['relative']=contrasts[v]['address_zero_us']/contrasts[v]['oob_us']-1
    result=dict(root=str(root),environment=json.loads((root/'environment.json').read_text()),cases=cases,
        contrasts=contrasts,stage_summary=summary,stage_processes=processes,critical_ctas=last_ctas,
        raw_sha256=provenance,analyzer_sha256=common.sha(Path(__file__)),
        scope='Same-card, same-output P1 descriptor intervention. Positive delta means valid-address zero slower than descriptor OOB. Independent processes paired by trial and coordinate; statistics reduce correlated CTA windows within each process first. Same-role cycles/ns and merged last-CTA durations are separate. No old-card model parameter or physical write completion inferred.')
    common.write_json(output/'C-controls.json',result)
    with gzip.open(output/'C-controls-windows.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    return dict(cases=2,processes=40,paired_windows=len(windows),contrasts=contrasts,stage_summary=summary)


def static_boundary_features(row,setup):
    """Compatibility entry; static construction lives with prediction inputs."""
    from supply_model import boundary_fill_features
    return boundary_fill_features(row,setup)


def boundary_candidate_analysis(base,output):
    """Identify which fill coupling features old fixed-map pairs can determine."""
    import supply_model as feature_model
    import csv
    import numpy as np
    from analyze_r13_sm import select_attempts
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    names=['20261009-R18-first-fill-job738707','20261009-R18-first-fill-job738865',
           '20261009-R18-input-map-job738296','20261009-R18-input-map-n-job738307',
           '20261010-R09-R13-R18-failure-controls']
    kinds=('own','mask_max','mask_sum','cluster_max','cluster_sum','cluster_any')
    windows=[];static_cases={};provenance={};call_cache={}
    def read(path):return json.loads(path.read_text())
    for name in names:
        root=base/name;setups={s['case']:s['setup'] for s in read(root/'static_setup.json')}
        rows=read(root/'cases.json')
        if '739095' in name or 'failure-controls' in name:rows=[r for r in rows if r['group']=='r18_partial_map']
        for row in rows:
            fs=static_boundary_features(row,setups[row['id']]);static_cases[row['id']]=dict(row=row,setup=setups[row['id']],features=fs)
            selected,_,_=select_attempts(root,row['id'],('dual',))
            if len(selected)!=10:raise ValueError('boundary pair requires ten processes')
            calls={}
            for item in selected:
                rec=item['record'];obs=replay(root,rec,row,retain_roles=True)
                source=str(root/rec['raw']);provenance[source]=rec['raw_sha256']
                calls[rec['trial']]=obs
            call_cache[row['id']]=calls
        for group in sorted({r['group'] for r in rows}):
            pair=[r for r in rows if r['group']==group]
            op='A' if pair[0]['input_map_m']!=pair[1]['input_map_m'] else 'B'
            axis='input_map_m' if op=='A' else 'input_map_n';a,b=sorted(pair,key=lambda r:r[axis])
            features={(f['cta'],f['j']):f for f in static_cases[a['id']]['features']}
            mapped={(f['cta'],f['j']):f for f in static_cases[b['id']]['features']}
            for trial in range(10):
                ac=call_cache[a['id']][trial]['ctas'];bc=call_cache[b['id']][trial]['ctas'];local=[]
                if [c['work'] for c in ac]!=[c['work'] for c in bc]:raise ValueError('measured map work changed')
                for ca,cb in zip(ac,bc):
                    # Cooperative roles kept separate; role2 matches the issuing-role convention.
                    aa,bb=(ca['roles'][1],cb['roles'][1]) if a['config']!='cfg_b' else (ca,cb)
                    for j,(t,u,tn,un,coord) in enumerate(zip(aa['tiles'],bb['tiles'],aa['tiles_ns'],bb['tiles_ns'],ca['work'])):
                        f=features[ca['cta'],j];fm=mapped[ca['cta'],j]
                        if tuple(f['coord'])!=tuple(coord):raise ValueError('static scheduler and observation differ')
                        row=dict(case=a['id'],root=name,split='diagnostic' if 'failure-controls' in name else 'calibration',
                            config=a['config'],operand=op,k=a['k'],kt=a['k']//64,trial=trial,cta=ca['cta'],j=j,T=len(ca['work']),
                            phase=f['phase'],mi=coord[0],ni=coord[1],geometry=boundary_geometry(a,coord),
                            cycle_delta_per_kt=((t[1]-t[0])-(u[1]-u[0]))/(a['k']//64),
                            ns_delta_per_kt=((tn[1]-tn[0])-(un[1]-un[0]))/(a['k']//64),
                            mapped_L_ns_per_kt=(un[1]-un[0])/(a['k']//64))
                        for kind in kinds:row[kind]=f['operands'][op][kind]-fm['operands'][op][kind]
                        local.append(row)
                for r in local:
                    baseline=[v for v in local if v['j']==r['j'] and v['T']==r['T'] and v['cluster_max']==0 and v['geometry']=='real']
                    r['baseline_windows']=len(baseline)
                    for unit in ('ns','cycle'):
                        offset=statistics.median(v[unit+'_delta_per_kt'] for v in baseline) if baseline else None
                        r[unit+'_baseline_per_kt']=offset
                        r[unit+'_excess_per_kt']=r[unit+'_delta_per_kt']-offset if offset is not None else None
                    r['ns_excess']=r['ns_excess_per_kt']*r['kt'] if r['ns_excess_per_kt'] is not None else None
                    r['cycle_excess']=r['cycle_excess_per_kt']*r['kt'] if r['cycle_excess_per_kt'] is not None else None
                    r['relative_excess']=(r['ns_excess_per_kt']/statistics.median(v['mapped_L_ns_per_kt'] for v in baseline)) if baseline else None
                    windows.append(r)
    # Each trial contributes one median for a static stratum; CTA rows are not independent observations.
    keynames=('case','root','split','config','operand','k','kt','phase','j','T','geometry',*kinds)
    grouped=defaultdict(list)
    for w in windows:grouped[tuple(w[k] for k in (*keynames,'trial'))].append(w)
    processes=[]
    metrics=('cycle_delta_per_kt','ns_delta_per_kt','cycle_excess_per_kt','ns_excess_per_kt','cycle_excess','ns_excess','relative_excess')
    for key,rows in sorted(grouped.items()):
        r=dict(zip((*keynames,'trial'),key),windows=len(rows),baseline_windows=rows[0]['baseline_windows'])
        for metric in metrics:
            values=[x[metric] for x in rows if x[metric] is not None]
            r[metric]=statistics.median(values) if values else None
        processes.append(r)
    grouped=defaultdict(list)
    for r in processes:grouped[tuple(r[k] for k in keynames)].append(r)
    strata=[]
    for key,rows in sorted(grouped.items()):
        r=dict(zip(keynames,key),processes=len(rows))
        for metric in metrics:
            values=[x[metric] for x in rows if x[metric] is not None]
            r[metric]=statistics.median(values) if values else None
        strata.append(r)
    fits=[]
    for cfg,op,phase in sorted({(r['config'],r['operand'],r['phase']) for r in strata}):
        rows=[r for r in strata if r['config']==cfg and r['operand']==op and r['phase']==phase and r['split']=='calibration' and r['ns_excess_per_kt'] is not None]
        if not rows:continue
        design=np.array([[r[k] for k in kinds] for r in rows]);nonzero=design[:,3]>0
        counts=Counter(r['case'] for r in rows);weights=np.array([1/counts[r['case']] for r in rows])
        group=dict(config=cfg,operand=op,phase=phase,features=list(kinds),rank=int(np.linalg.matrix_rank(design)),
            feature_patterns=sorted({tuple(r[k] for k in kinds) for r in rows}),calibration_cases=sorted(counts),
            cluster_max_sum_rank=int(np.linalg.matrix_rank(design[:,3:5])),cluster_coupling_rank=int(np.linalg.matrix_rank(design[:,3:])),fits=[])
        for target in ('cycle_excess_per_kt','ns_excess_per_kt','cycle_excess','ns_excess','relative_excess'):
            y=np.array([r[target] for r in rows])
            for i,kind in enumerate(kinds):
                x=design[:,i];den=float(np.sum(weights*x*x));coef=float(np.sum(weights*x*y)/den) if den else None
                if coef is None:continue
                errors=x*coef-y
                by_k={}
                for k in sorted({r['k'] for r in rows}):
                    selected=np.array([r['k']==k for r in rows]);xk=x[selected];yk=y[selected];wk=weights[selected]
                    if np.any(xk):by_k[str(k)]=float(np.sum(wk*xk*yk)/np.sum(wk*xk*xk))
                group['fits'].append(dict(target=target,feature=kind,coefficient=coef,
                    weighted_rms=float(np.sqrt(np.sum(weights*errors**2)/np.sum(weights))),
                    affected_median_abs_residual=float(np.median(np.abs(errors[nonzero]))) if np.any(nonzero) else None,
                    coefficients_by_k=by_k))
        fits.append(group)
    # Static feature ranges of V09 are diagnostics only; no timing response enters the fit.
    root=base/'20261009-V09-freeze-job738972-r2';setups={s['case']:s['setup'] for s in read(root/'static_setup.json')}
    for row in read(root/'cases.json'):
        if row['id'] in ('v09d_cfg_a_P1_partial_first_and_later','v09d_cfg_c_P2_partial_later_sw8'):
            static_cases[row['id']]=dict(row=row,setup=setups[row['id']],features=static_boundary_features(row,setups[row['id']]))
    diagnostics=[]
    for case,value in static_cases.items():
        if not (case.startswith('v09d_') or case.startswith('r18_partial_')):continue
        for op in ('A','B'):
            for phase in ('first','later'):
                vals=[f['operands'][op] for f in value['features'] if f['phase']==phase]
                patterns=sorted({tuple(v[k] for k in kinds) for v in vals})
                if not any(any(p) for p in patterns):continue
                fit=next((f for f in fits if f['config']==value['row']['config'] and f['operand']==op and f['phase']==phase),None)
                span=[]
                for p in patterns:
                    train=np.array(fit['feature_patterns'],dtype=float) if fit else np.empty((0,len(kinds)))
                    rank=int(np.linalg.matrix_rank(train)) if len(train) else 0
                    rank_after=int(np.linalg.matrix_rank(np.vstack([train,np.array(p)])))
                    span.append(dict(pattern=p,in_joint_calibration_span=rank_after==rank))
                diagnostics.append(dict(case=case,operand=op,phase=phase,patterns=span))
    structural_ranking=[]
    for case,value in static_cases.items():
        if not (case.startswith('v09d_') or case=='r18_partial_map960'):continue
        row=value['row'];features=value['features'];maximum_work=max(f['T'] for f in features)
        op='A' if row['config']=='cfg_a' else 'B'
        first=[f for f in features if f['j']==0 and f['T']==maximum_work and f['operands'][op]['cluster_any']]
        complete=[f['cta'] for f in first if f['operands'][op]['own']==0]
        signature=[f['cta'] for f in first]
        if case in call_cache:calls=call_cache[case]
        else:
            data=base/'20261009-V09-heldout-job739011'
            selected,_,_=select_attempts(data,case,('dual',))
            calls={}
            for item in selected:
                rec=item['record'];calls[rec['trial']]=replay(data,rec,row)
                provenance[str(data/rec['raw'])]=rec['raw_sha256']
        hits=[]
        for trial,obs in sorted(calls.items()):
            last=maximum_set([c['end_ns'] for c in obs['ctas']])
            hits.append(dict(trial=trial,last=last,first_cluster_signature_hit=bool(set(last)&set(signature)),
                complete_partner_signature_hit=bool(set(last)&set(complete))))
        structural_ranking.append(dict(case=case,first_cluster_max_work_signature=signature,
            complete_partner_signature=complete,processes=hits,
            scope='Static class membership only; not a new timing prediction or expanded frozen candidate set.'))
    common.write_json(output/'C-candidate.json',dict(fits=fits,strata=strata,processes=processes,
        analyzer_sha256=common.sha(Path(__file__)),feature_model_sha256=common.sha(Path(feature_model.__file__)),
        structural_ranking=structural_ranking,
        feature_units={k:('indicator' if k=='cluster_any' else 'KiB/Ktile') for k in kinds},
        target_units=dict(cycle_excess_per_kt='cycles/Ktile',ns_excess_per_kt='ns/Ktile',cycle_excess='cycles/window',ns_excess='ns/window',relative_excess='dimensionless'),
        diagnostic_patterns=diagnostics,raw_sha256=provenance,
        scope='Candidate static features only. Fit descriptive old-card paired excess over interior-real j/T controls, separately by configuration/operand/phase. Scalar through-origin slopes are diagnostic, not production timing prices. New-card controls and V09 do not enter fitting. Joint feature span identifies ambiguities between candidates.'))
    common.write_json(output/'C-static-features.json',static_cases)
    with gzip.open(output/'C-candidate-windows.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    return dict(calibration_conditions=16,diagnostic_conditions=2,windows=len(windows),fits=fits,diagnostic_patterns=diagnostics)


def fit_fill_hypotheses(rows):
    """Fit descriptive map contrasts; rows are process-stratum medians, not CTAs."""
    import numpy as np
    fields=('cluster_max','cluster_sum','cluster_any')
    x=np.array([[r[k] for k in fields] for r in rows],dtype=float)
    y=np.array([r['ns_excess'] for r in rows],dtype=float)
    result=[]
    for i,name in enumerate(fields):
        d=float(x[:,i]@x[:,i]);coef=float(x[:,i]@y/d) if d else 0.
        predictions=x[:,i]*coef
        result.append(dict(hypothesis=name,coefficients={name:coef},rms_ns=float(np.sqrt(np.mean((predictions-y)**2))),
            predictions_ns=predictions.tolist(),residuals_ns=(predictions-y).tolist()))
    coef=np.linalg.lstsq(x,y,rcond=None)[0];predictions=x@coef
    result.append(dict(hypothesis='mixed',coefficients=dict(zip(fields,coef.tolist())),rank=int(np.linalg.matrix_rank(x)),
        rms_ns=float(np.sqrt(np.mean((predictions-y)**2))),predictions_ns=predictions.tolist(),residuals_ns=(predictions-y).tolist(),
        scope='Three nonzero map patterns can interpolate three map means. Replication residual is not new-shape validation; signed coefficients are descriptive, not physical prices.'))
    return result


def fill_price_diagnostics(fit_report,root):
    """Conditional use of four-map prices; observed baselines are not forecasts."""
    from analyze_r13_sm import select_attempts
    root=Path(root)
    environment=json.loads((root/'environment.json').read_text())
    if environment['gpu'].split(',')[0]!=fit_report['environment']['gpu'].split(',')[0]:
        raise ValueError('conditional price transfer requires the same GPU')
    rows=[r for r in json.loads((root/'cases.json').read_text()) if r['id'].startswith(('r18_cluster_fill_map','r18_partial_map'))]
    rows.sort(key=lambda r:r['input_map_m']);baseline=rows[-1]
    setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    coeff={(f['role'],f['T'],h['hypothesis']):h['coefficients'][h['hypothesis']]
           for f in fit_report['hypotheses'] if f['phase']=='first' for h in f['hypotheses'] if h['hypothesis']!='mixed'}
    names=('cluster_max','cluster_sum','cluster_any');cache={};sources={};provenance={}
    static={r['input_map_m']:{(f['cta'],f['j']):f for f in static_boundary_features(r,setups[r['id']])} for r in rows}
    for row in rows:
        extent=row['input_map_m'];cache[extent]={}
        selected,_,_=select_attempts(root,row['id'],('dual',))
        for item in selected:
            rec=item['record'];cache[extent][rec['trial']]=replay(root,rec,row,retain_roles=True)
            source=str(root/rec['raw']);sources[extent,rec['trial']]=source;provenance[source]=rec['raw_sha256']
    stage=[];ranking=[]
    for row in rows[:-1]:
        extent=row['input_map_m']
        for trial in range(10):
            target=cache[extent][trial]['ctas'];base=cache[1024][trial]['ctas']
            if [c['work'] for c in target]!=[c['work'] for c in base]:raise ValueError('diagnostic baseline changed work')
            offsets=defaultdict(list)
            for a,b in zip(target,base):
                f=static[extent][a['cta'],0]
                if f['operands']['A']['cluster_any'] or boundary_geometry(row,f['coord'])!='real':continue
                for role in range(2):
                    at=a['roles'][role]['tiles_ns'][0];bt=b['roles'][role]['tiles_ns'][0]
                    offsets[role,len(a['work'])].append((at[1]-at[0])-(bt[1]-bt[0]))
            offset={k:statistics.median(v) for k,v in offsets.items()}
            earliest=min(c['entry_ns'] for c in target);last=maximum_set([c['end_ns'] for c in target])
            for name in names:
                for adjust in (False,True):
                    groups=defaultdict(list);predicted_end=[]
                    for a,b in zip(target,base):
                        f=static[extent][a['cta'],0];fill=f['operands']['A'];T=len(a['work'])
                        context='input_fill' if fill['own'] else 'clear_partner' if fill['cluster_any'] else 'interior'
                        for role in range(2):
                            at=a['roles'][role]['tiles_ns'][0];bt=b['roles'][role]['tiles_ns'][0]
                            actual=at[1]-at[0];base_L=bt[1]-bt[0];price=coeff[role,T,name]*fill[name]
                            predicted=base_L+price+(offset[role,T] if adjust else 0)
                            groups[role,T,context].append(dict(actual_ns=actual,predicted_ns=predicted,
                                signed_relative=predicted/actual-1,absolute_relative=abs(predicted/actual-1),
                                observed_excess_ns=actual-base_L-offset[role,T],price_ns=price))
                        delta=coeff[1,T,name]*fill[name]+(offset[1,T] if adjust else 0)
                        predicted_end.append(a['entry_ns']-earliest+(b['end_ns']-b['entry_ns'])+delta)
                    for (role,T,context),values in groups.items():
                        r=dict(input_map_m=extent,trial=trial,hypothesis=name,observed_interior_adjustment=adjust,role=role,T=T,context=context,
                            windows=len(values),source=sources[extent,trial],baseline_source=sources[1024,trial])
                        for key in values[0]:r[key]=statistics.median(v[key] for v in values)
                        r['maximum_absolute_relative']=max(v['absolute_relative'] for v in values);stage.append(r)
                    predicted=maximum_set(predicted_end)
                    ranking.append(dict(input_map_m=extent,trial=trial,hypothesis=name,observed_interior_adjustment=adjust,
                        predicted_candidates=predicted,observed_last=last,hit=bool(set(predicted)&set(last)),
                        candidate_count=len(predicted),best_candidate_finish_gap_ns=max(c['end_ns'] for c in target)-max(target[i]['end_ns'] for i in predicted)))
    summary=[]
    keys=('input_map_m','hypothesis','observed_interior_adjustment','role','T','context')
    groups=defaultdict(list)
    for r in stage:groups[tuple(r[k] for k in keys)].append(r)
    for key,group in sorted(groups.items()):
        value=dict(zip(keys,key),processes=len(group))
        for metric in ('actual_ns','predicted_ns','signed_relative','absolute_relative','maximum_absolute_relative','observed_excess_ns','price_ns'):
            value[metric]=statistics.median(r[metric] for r in group)
        value['largest_window_absolute_relative']=max(r['maximum_absolute_relative'] for r in group)
        summary.append(value)
    rank_summary=[];groups=defaultdict(list)
    for r in ranking:groups[r['input_map_m'],r['hypothesis'],r['observed_interior_adjustment']].append(r)
    for key,group in sorted(groups.items()):
        rank_summary.append(dict(input_map_m=key[0],hypothesis=key[1],observed_interior_adjustment=key[2],
            hits=sum(r['hit'] for r in group),processes=len(group),
            median_candidate_count=statistics.median(r['candidate_count'] for r in group),
            median_finish_gap_ns=statistics.median(r['best_candidate_finish_gap_ns'] for r in group),
            maximum_finish_gap_ns=max(r['best_candidate_finish_gap_ns'] for r in group)))
    return dict(root=str(root),stage_summary=summary,stage_processes=stage,ranking=ranking,ranking_summary=rank_summary,raw_sha256=provenance,
        scope='Prices fitted only on job739529. Full first L = same-trial map1024 observed L + price, optionally plus target observed interior shift. CTA ordering uses target observed entry + map1024 observed internal duration + role2 first increment; other components remain baseline. All uses are conditional diagnostics with observed inputs, not deployable or heldout predictions.')


def fill_discrimination(root,output):
    """Four-map fixed-output contrast of fill aggregation, using static cohorts."""
    import supply_model as feature_model
    import csv
    from analyze_r13_sm import select_attempts
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);common.verify(root)
    rows=sorted([r for r in json.loads((root/'cases.json').read_text()) if r['id'].startswith('r18_cluster_fill_map')],key=lambda r:r['input_map_m'])
    if [r['input_map_m'] for r in rows]!=[832,896,960,1024]:raise ValueError('four map extents required')
    reference=rows[0];baseline=rows[-1]
    fixed={k:v for k,v in reference.items() if k not in ('id','input_map_m')}
    if any({k:v for k,v in r.items() if k not in ('id','input_map_m')}!=fixed for r in rows):raise ValueError('map discrimination changed another field')
    setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    setup_keys=('grid','allocated_a_elements','allocated_b_elements','storage_m','storage_n',
                'initialized_input_rows','initialized_input_columns','gpu_uuid','stages','smem')
    if any(any(setups[r['id']][k]!=setups[reference['id']][k] for k in setup_keys) for r in rows):raise ValueError('different paired setup')
    static={r['input_map_m']:{(f['cta'],f['j']):f for f in static_boundary_features(r,setups[r['id']])} for r in rows}
    calls={};sources={};provenance={};cases=[]
    for row in rows:
        extent=row['input_map_m'];calls[extent]={v:{} for v in ('plain','dual')}
        selected,failed,_=select_attempts(root,row['id'],('plain','dual'))
        for item in selected:
            rec=item['record'];obs=replay(root,rec,row,retain_roles=rec['variant']=='dual')
            calls[extent][rec['variant']][rec['trial']]=obs
            source=str(root/rec['raw']);sources[extent,rec['variant'],rec['trial']]=source;provenance[source]=rec['raw_sha256']
        if any(set(calls[extent][v])!=set(range(10)) for v in ('plain','dual')):raise ValueError('incomplete map processes')
        times={v:[calls[extent][v][t]['elapsed_us'] for t in range(10)] for v in ('plain','dual')}
        med={v:statistics.median(t) for v,t in times.items()}
        cases.append(dict(case=row['id'],input_map_m=extent,median_us=med,times_us=times,
            cv={v:statistics.pstdev(t)/statistics.mean(t) for v,t in times.items()},
            dual_over_plain=med['dual']/med['plain']-1,failed_attempts=failed))
    windows=[];critical=[]
    for row in rows:
        extent=row['input_map_m']
        for trial in range(10):
            observed=calls[extent]['dual'][trial];base=calls[1024]['dual'][trial]
            if [c['role_work'] for c in observed['ctas']]!=[c['role_work'] for c in base['ctas']]:raise ValueError('map changed work coordinates')
            observed_rows=boundary_stage_rows(row,observed);base_rows=boundary_stage_rows(baseline,base)
            by_key={(r['cta'],r['role'],r['j'],r['stage']):r for r in base_rows}
            local=[]
            for r in observed_rows:
                if r['stage'] not in ('L','E'):continue
                b=by_key[r['cta'],r['role'],r['j'],r['stage']]
                sf=static[extent][r['cta'],r['j']];ref=static[832][r['cta'],r['j']]
                if sf['coord']!=[r['mi'],r['ni']]:raise ValueError('static scheduler differs from observed coordinates')
                own=sf['operands']['A'];fixed_fill=ref['operands']['A']['own'];affected=bool(ref['operands']['A']['cluster_any'])
                state='input_fill' if own['own'] else 'clear_partner' if own['cluster_any'] else 'cleared_reference_cluster' if affected else 'interior'
                local.append(dict(input_map_m=extent,trial=trial,cta=r['cta'],role=r['role'],j=r['j'],T=r['T'],
                    phase=r['phase'],stage=r['stage'],mi=r['mi'],ni=r['ni'],logical_geometry=r['geometry'],
                    reference_cluster_affected=affected,reference_own_fill_kib=fixed_fill,input_state=state,
                    own_fill_kib=own['own'],cluster_max=own['cluster_max'],cluster_sum=own['cluster_sum'],cluster_any=own['cluster_any'],
                    ns=r['ns'],baseline_ns=b['ns'],ns_delta=r['ns']-b['ns'],cycles=r['cycles'],baseline_cycles=b['cycles'],cycle_delta=r['cycles']-b['cycles'],
                    source=sources[extent,'dual',trial],baseline_source=sources[1024,'dual',trial]))
            for r in local:
                interior=[a for a in local if not a['reference_cluster_affected'] and a['logical_geometry']=='real' and
                          (a['role'],a['j'],a['T'],a['stage'])==(r['role'],r['j'],r['T'],r['stage'])]
                for unit in ('ns','cycle'):
                    offset=statistics.median(a[unit+'_delta'] for a in interior) if interior else None
                    r[unit+'_interior_offset']=offset;r[unit+'_excess']=r[unit+'_delta']-offset if offset is not None else None
                windows.append(r)
            ctas=observed['ctas'];earliest=min(c['entry_ns'] for c in ctas);last=maximum_set([c['end_ns'] for c in ctas])
            critical.append(dict(input_map_m=extent,trial=trial,last=last,envelope_ns=max(c['end_ns'] for c in ctas)-earliest,
                work=[ctas[i]['work'] for i in last],first_reference_affected=[bool(static[832][i,0]['operands']['A']['cluster_any']) for i in last]))
    keynames=('input_map_m','role','phase','j','T','stage','logical_geometry','reference_cluster_affected','reference_own_fill_kib',
              'input_state','own_fill_kib','cluster_max','cluster_sum','cluster_any')
    metrics=('ns','baseline_ns','ns_delta','cycles','baseline_cycles','cycle_delta','ns_excess','cycle_excess','ns_interior_offset')
    grouped=defaultdict(list)
    for r in windows:grouped[tuple(r[k] for k in (*keynames,'trial'))].append(r)
    processes=[]
    for key,group in sorted(grouped.items()):
        value=dict(zip((*keynames,'trial'),key),windows=len(group))
        for metric in metrics:
            vals=[r[metric] for r in group if r[metric] is not None];value[metric]=statistics.median(vals) if vals else None
        processes.append(value)
    grouped=defaultdict(list)
    for r in processes:grouped[tuple(r[k] for k in keynames)].append(r)
    strata=[]
    for key,group in sorted(grouped.items()):
        value=dict(zip(keynames,key),processes=len(group))
        for metric in metrics:
            vals=[r[metric] for r in group if r[metric] is not None];value[metric]=statistics.median(vals) if vals else None
        value['ns_excess_range']=[min(r['ns_excess'] for r in group),max(r['ns_excess'] for r in group)] if value['ns_excess'] is not None else None
        strata.append(value)
    fits=[]
    for role,phase,j,T in sorted({(r['role'],r['phase'],r['j'],r['T']) for r in processes if r['stage']=='L'}):
        selected=[r for r in processes if r['stage']=='L' and r['reference_cluster_affected'] and
                  (r['role'],r['phase'],r['j'],r['T'])==(role,phase,j,T) and r['ns_excess'] is not None]
        if selected:fits.append(dict(role=role,phase=phase,j=j,T=T,rows=selected,hypotheses=fit_fill_hypotheses(selected)))
    # Adjacent direct interventions, not differences between independent median tables.
    direct=[]
    for lo,hi in ((832,896),(896,960),(960,1024)):
        lookup={(r['trial'],r['cta'],r['role'],r['j'],r['stage']):r for r in windows if r['input_map_m']==hi}
        interior=defaultdict(list)
        for r in windows:
            if r['input_map_m']!=lo or r['reference_cluster_affected'] or r['logical_geometry']!='real':continue
            v=lookup[r['trial'],r['cta'],r['role'],r['j'],r['stage']]
            interior[r['trial'],r['role'],r['j'],r['T'],r['stage']].append(v['ns']-r['ns'])
        pairs=defaultdict(list)
        for r in windows:
            if r['input_map_m']!=lo:continue
            v=lookup[r['trial'],r['cta'],r['role'],r['j'],r['stage']]
            control=interior[r['trial'],r['role'],r['j'],r['T'],r['stage']]
            offset=statistics.median(control) if control else None
            key=tuple(r[k] for k in ('trial','role','phase','j','T','stage','logical_geometry','reference_own_fill_kib'))
            pairs[key].append(dict(ns=v['ns']-r['ns'],cycle=v['cycles']-r['cycles'],
                excess_ns=v['ns']-r['ns']-offset if offset is not None else None,interior_ns=offset))
        direct.append(dict(from_map=lo,to_map=hi,processes=[dict(zip(('trial','role','phase','j','T','stage','logical_geometry','reference_own_fill_kib'),key),
            delta_ns=statistics.median(v['ns'] for v in group),delta_cycles=statistics.median(v['cycle'] for v in group),
            excess_delta_ns=statistics.median(v['excess_ns'] for v in group) if group[0]['excess_ns'] is not None else None,
            interior_delta_ns=group[0]['interior_ns']) for key,group in sorted(pairs.items())]))
    role_agreement=[]
    by_role={(r['input_map_m'],r['trial'],r['cta'],r['j'],r['stage'],r['role']):r for r in windows}
    grouped=defaultdict(list)
    for r in windows:
        if r['role']!=0 or r['stage']!='L' or not r['reference_cluster_affected']:continue
        other=by_role[r['input_map_m'],r['trial'],r['cta'],r['j'],r['stage'],1]
        grouped[r['input_map_m'],r['trial'],r['phase'],r['j'],r['T']].append(abs(r['ns_delta']-other['ns_delta']))
    for key,values in sorted(grouped.items()):
        role_agreement.append(dict(zip(('input_map_m','trial','phase','j','T'),key),
            median_abs_delta_difference_ns=statistics.median(values),max_abs_delta_difference_ns=max(values)))
    result=dict(root=str(root),environment=json.loads((root/'environment.json').read_text()),cases=cases,stage_strata=strata,
        role_agreement=role_agreement,
        stage_processes=processes,hypotheses=fits,direct=direct,critical=critical,raw_sha256=provenance,
        analyzer_sha256=common.sha(Path(__file__)),feature_model_sha256=common.sha(Path(feature_model.__file__)),
        scope='Fixed map832 static affected cohort across all maps, map1024 paired baseline. Logical output geometry is separate from input fill. Per-process strata precede fitting. Interior subtraction is a diagnostic baseline assumption; raw deltas retained. Four map points identify conditional contrasts, not a universal price. No old-card coefficient used.')
    result['conditional_approximations']=[fill_price_diagnostics(result,root),
        fill_price_diagnostics(result,root.parent/'20261010-R09-R13-R18-failure-controls')]
    common.write_json(output/'C-fill-discrimination.json',result)
    with gzip.open(output/'C-fill-discrimination-windows.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    return dict(cases=cases,fits=[{k:v for k,v in f.items() if k!='rows'} for f in fits],windows=len(windows))


def first_cluster_development(base,output):
    """Refit only frozen cfg_a first rows; compare deployable-input compositions."""
    import copy
    from contextlib import ExitStack
    from unittest.mock import patch
    import analyze_r13_supply as fitting
    import supply_model as supply
    import v09_model as predictor
    from analyze_r13_sm import select_attempts
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=base/'20261009-V09-freeze-job738972-r2';target=base/'20261009-V09-heldout-job739011'
    read=lambda path:json.loads(path.read_text())
    calibration=read(frozen/'frozen/calibration.json')
    with gzip.open(frozen/'inputs/supply-training.json.gz','rt') as stream:training=json.load(stream)
    own=training['first']['cfg_a'];cluster=copy.deepcopy(own);source_cases={};source_hashes={}
    if len(own)!=2028:raise ValueError('unexpected frozen cfg_a first training set')
    for record in cluster:
        run=Path(record['run']);key=str(run),record['case']
        if key not in source_cases:
            rows={r['id']:r for r in read(run/'cases.json')};setups={r['case']:r['setup'] for r in read(run/'static_setup.json')}
            source_cases[key]={(f['cta'],f['j']):f for f in static_boundary_features(rows[record['case']],setups[record['case']])}
            for name in ('cases.json','static_setup.json'):source_hashes[str(run/name)]=common.sha(run/name)
        f=source_cases[key][record['cta'],record['j']]
        if f['coord']!=record['coord']:raise ValueError('calibration static coordinate mismatch')
        record['X'][5:7]=[f['operands'][op]['cluster_max'] for op in ('A','B')]
    if any({k:v for k,v in a.items() if k!='X'}!={k:v for k,v in b.items() if k!='X'} or a['X'][:5]!=b['X'][:5] for a,b in zip(own,cluster)):
        raise ValueError('cluster transform changed a calibration target or another request field')
    models={'frozen_r2':copy.deepcopy(calibration['first_supply']['cfg_a'])}
    models['refit_own']=fitting.fit_supply(own,unit='ns',calibration_clock='observed',assumed_ghz=1.6,phase='first')
    models['cluster_max']=fitting.fit_supply(cluster,unit='ns',calibration_clock='observed',assumed_ghz=1.6,phase='first')
    for name,m in models.items():m['request_scope']='cluster_max' if name=='cluster_max' else 'own_destination_fill'
    identification={name:fitting.supply_identification_bounds(cluster if name=='cluster_max' else own,m) for name,m in models.items()}
    later_id=fitting.supply_identification_bounds(training['later']['cfg_a'],calibration['supply']['cfg_a'])
    original_requests=supply.supply_request_rows
    def cluster_requests(row,setup):
        requests=original_requests(row,setup)
        if row['config']=='cfg_a':
            features={(f['cta'],f['j']):f for f in static_boundary_features(row,setup)}
            for r in requests:
                if r['j']==0:r['X'][5:7]=[features[r['cta'],0]['operands'][op]['cluster_max'] for op in ('A','B')]
        return requests
    def predict(row,setup,cal,transform,f=None):
        with ExitStack() as stack:
            if transform:stack.enter_context(patch.object(supply,'supply_request_rows',side_effect=cluster_requests))
            if f is not None:
                stack.enter_context(patch.object(predictor.clock,'solve_source_envelope',side_effect=lambda point,clock,envelope:dict(frequency_ghz=f,window_us=envelope(f))))
            return predictor.predict_components(row,setup,cal)
    def support(requests,first,first_id,pred):
        groups={};identified=True
        for phase,model,identity in [('first',first,first_id),('later',calibration['supply']['cfg_a'],later_id)]:
            selected=[r for r in requests if (r['j']==0)==(phase=='first')]
            bounds=supply.supply_predict_bounds(selected,model,pred['frequency_ghz'],identity)
            for r,b in zip(selected,bounds):
                if b['status']!='identified prediction':identified=False
                key=(phase,b['status'],b.get('minimum'),b.get('maximum'))
                if key not in groups:groups[key]=dict(phase=phase,**b,window_count=0,example=dict(cta=r['cta'],j=r['j'],X=r['X'],kt=r['kt']))
                groups[key]['window_count']+=1
                if b['status']!='identified prediction':groups[key].setdefault('windows',[]).append([r['cta'],r['j']])
        return dict(status='identified' if identified else 'unsupported_or_nonunique',bounds=list(groups.values()))
    def score(row,obs,pred):
        components=defaultdict(list);ctas=obs['ctas']
        for c,p in zip(ctas,pred['cta_events']):
            actual=c['tiles_ns'];events=p['events']
            for key,a,b in [('prefix',actual[0][0]-c['entry_ns'],p['supply']),
                            ('prefill',actual[0][0]-c['prod_ns'],p['prefill']),
                            ('first_L',actual[0][1]-actual[0][0],events[0][1]-events[0][0]),
                            ('first_E',actual[0][3]-actual[0][2],events[0][3]-events[0][2]),
                            ('last_E',actual[-1][3]-actual[-1][2],events[-1][3]-events[-1][2]),
                            ('tail',c['end_ns']-max(t[3] for t in actual),p['tail'])]:
                if a>0:components[key].append(b/a-1)
            components['later_L'] += [(p[1]-p[0])/(a[1]-a[0])-1 for a,p in zip(actual[1:],events[1:]) if a[1]>a[0]]
        last=maximum_set([c['end_ns'] for c in ctas]);candidates=pred['critical_candidates']
        return dict(component_signed={k:statistics.median(v) for k,v in components.items() if v},
            component_absolute={k:statistics.median(abs(x) for x in v) for k,v in components.items() if v},
            critical_hit=bool(set(last)&set(candidates)),actual_last=last,
            best_candidate_finish_gap_ns=max(c['end_ns'] for c in ctas)-max(ctas[i]['end_ns'] for i in candidates),
            envelope_error=pred['window_us']/((max(c['end_ns'] for c in ctas)-min(c['entry_ns'] for c in ctas))/1000)-1)
    setups={s['case']:s['setup'] for s in read(frozen/'static_setup.json')}
    rows=[r for r in read(frozen/'cases.json') if r['set']=='heldout' and r['config']=='cfg_a']
    frozen_predictions=read(frozen/'frozen/predictions.json')['predictions'];cases=[];provenance={}
    for row in rows:
        setup=setups[row['id']];selected,failed,_=select_attempts(target,row['id'],('plain','dual'))
        plain={};observed={}
        for item in selected:
            rec=item['record'];provenance[str(target/rec['raw'])]=rec['raw_sha256']
            if rec['variant']=='plain':plain[rec['trial']]=rec['elapsed_us']
            else:observed[rec['trial']]=replay(target,rec,row)
        if len(plain)!=10 or len(observed)!=10:raise ValueError('incomplete cfg_a heldout calls')
        current=dict(case=row['id'],condition=row,plain_us=statistics.median(plain.values()),failed_attempts=failed,versions={})
        features={(f['cta'],f['j']):f for f in static_boundary_features(row,setup)}
        for name,m in models.items():
            cal=copy.deepcopy(calibration);cal['first_supply']['cfg_a']=m
            free=predict(row,setup,cal,name=='cluster_max')
            requests=(cluster_requests if name=='cluster_max' else original_requests)(row,setup)
            bounds=support(requests,m,identification[name],free)
            candidates=[]
            for cta in free['critical_candidates']:
                f=features[cta,0];candidates.append(dict(cta=cta,coord=f['coord'],T=f['T'],logical_geometry=boundary_geometry(row,f['coord']),
                    own_fill=[f['operands'][op]['own'] for op in ('A','B')],cluster_fill=[f['operands'][op]['cluster_max'] for op in ('A','B')]))
            import numpy as np
            branches={}
            for r in requests:
                if r['j']!=0:continue
                service=float(np.asarray(r['X'])@np.asarray(m['projection'])@np.asarray(m['parameters'][1:]))
                floor=r['compute_cycles']/free['frequency_ghz'];key=tuple(r['X'])
                if key not in branches:branches[key]=dict(X=r['X'],service_ns_per_ktile=service,floor_ns_per_ktile=floor,
                    branch='supply' if service>floor else 'compute_floor',window_count=0,example_cta=r['cta'])
                branches[key]['window_count']+=1
            version=dict(request_scope=m['request_scope'],free_prediction=free,free_support=bounds,first_branch_witness=list(branches.values()),
                free_plain_error=free['plain_us']/current['plain_us']-1,critical_categories=candidates,processes=[])
            for trial,obs in sorted(observed.items()):
                ctas=obs['ctas'];max_tiles=max(len(c['tiles']) for c in ctas)
                f=statistics.median((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns']) for c in ctas if len(c['tiles'])==max_tiles)
                conditional=predict(row,setup,cal,name=='cluster_max',f)
                conditional_support=support(requests,m,identification[name],conditional)
                version['processes'].append(dict(trial=trial,observed_f_ghz=f,free=score(row,obs,free),
                    observed_f=score(row,obs,conditional),observed_f_plain_us=conditional['plain_us'],
                    observed_f_plain_error=conditional['plain_us']/plain[trial]-1,
                    observed_f_support=conditional_support,observed_f_critical_candidates=conditional['critical_candidates']))
            def aggregate(mode):
                group=[p[mode] for p in version['processes']]
                return dict(components={k:statistics.median(p['component_absolute'][k] for p in group if k in p['component_absolute'])
                    for k in set().union(*(p['component_absolute'] for p in group))},
                    critical_hits=sum(p['critical_hit'] for p in group),
                    median_candidate_finish_gap_ns=statistics.median(p['best_candidate_finish_gap_ns'] for p in group),
                    median_envelope_error=statistics.median(p['envelope_error'] for p in group))
            version['free_summary']=aggregate('free');version['observed_f_summary']=aggregate('observed_f')
            version['observed_f_summary']['median_plain_error']=statistics.median(p['observed_f_plain_error'] for p in version['processes'])
            if name=='frozen_r2':
                version['frozen_plain_us']=frozen_predictions[row['id']]['plain_us']
                version['free_replay_plain_delta_us']=free['plain_us']-frozen_predictions[row['id']]['plain_us']
            # Timelines are regenerated by the pure predictor; do not duplicate every tile in the report.
            version['free_prediction']={k:v for k,v in free.items() if k not in ('cta_events','cta_duration_ns')}
            current['versions'][name]=version
        cases.append(current)
    metrics={name:dict(all_numeric_plain=fitting.metrics([c['versions'][name]['free_plain_error'] for c in cases]),
        lp_identified_cases=sum(c['versions'][name]['free_support']['status']=='identified' for c in cases)) for name in models}
    result=dict(models=models,identification=identification,training_rows=2028,
        transformed_columns=[5,6],unchanged_fields='All other X entries, L_ns, L_cycle, calibration_floor_ns and record metadata retained',
        calibration_hash=common.sha(frozen/'frozen/calibration.json'),training_hash=common.sha(frozen/'inputs/supply-training.json.gz'),
        static_source_hashes=source_hashes,cases=cases,metrics=metrics,raw_sha256=provenance,
        analyzer_sha256=common.sha(Path(__file__)),feature_model_sha256=common.sha(Path(supply.__file__)),
        scope='Only frozen cfg_a first 2028 rows refit on this CPU. Other phases/configurations, clock, output and plain transfer unchanged. Free predictions use static inputs; target data only scores them. Observed-frequency diagnostics are separate. LP equivalent-parameter bounds use each corresponding transformed calibration set. All V09 data is development, not new heldout validation; unsupported numbers are diagnostics.')
    common.write_json(output/'C-first-cluster-model.json',result)
    return dict(metrics=metrics,cases=len(cases))


def boundary_endpoint_distribution(base,output):
    """Locate map-pair excess using saved endpoints; mma-tail start was not stamped."""
    from analyze_r13_sm import select_attempts
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summaries=[];process_rows=[];provenance={};sources={}
    for name in ('20261010-R18-cluster-fill-discrimination','20261010-R09-R13-R18-failure-controls'):
        root=base/name;rows=[r for r in json.loads((root/'cases.json').read_text()) if r['id'].startswith(('r18_cluster_fill_map','r18_partial_map'))]
        rows.sort(key=lambda r:r['input_map_m']);baseline=rows[-1];reference=rows[0]
        setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
        features={r['input_map_m']:{f['cta']:f for f in static_boundary_features(r,setups[r['id']]) if f['j']==0} for r in rows}
        calls={}
        for row in rows:
            selected,_,_=select_attempts(root,row['id'],('dual',));calls[row['input_map_m']]={}
            for item in selected:
                rec=item['record'];calls[row['input_map_m']][rec['trial']]=replay(root,rec,row,retain_roles=True)
                provenance[str(root/rec['raw'])]=rec['raw_sha256']
        for row in rows[:-1]:
            extent=row['input_map_m']
            for trial in range(10):
                target=calls[extent][trial]['ctas'];control=calls[1024][trial]['ctas'];windows=[]
                for a,b in zip(target,control):
                    if a['role_work']!=b['role_work']:raise ValueError('endpoint comparison changed work')
                    first=features[extent][a['cta']];ref=features[reference['input_map_m']][a['cta']]
                    for role in range(2):
                        at=a['roles'][role];bt=b['roles'][role]
                        def intervals(cta,r):
                            t=r['tiles_ns'][0];q=r['tiles'][0]
                            return dict(producer_prefix_ns=cta['prod_ns']-cta['entry_ns'],S_ns=t[0]-cta['prod_ns'],
                                first_prefix_ns=t[0]-cta['entry_ns'],L_ns=t[1]-t[0],main_to_permit_ns=t[2]-t[1],E_ns=t[3]-t[2],
                                first_tile_span_ns=t[3]-cta['entry_ns'],exit_tail_ns=r['final_ns']-r['tiles_ns'][-1][3],
                                S_cycles=q[0]-cta['prod_c'],L_cycles=q[1]-q[0],E_cycles=q[3]-q[2])
                        actual=intervals(a,at);old=intervals(b,bt)
                        if actual['first_tile_span_ns']!=actual['first_prefix_ns']+actual['L_ns']+actual['main_to_permit_ns']+actual['E_ns']:
                            raise ValueError('same-role first-tile accounting failed')
                        windows.append(dict(cta=a['cta'],role=role,T=len(a['work']),geometry=boundary_geometry(row,first['coord']),
                            reference_affected=bool(ref['operands']['A']['cluster_any']),reference_own_fill=ref['operands']['A']['own'],
                            own_fill=first['operands']['A']['own'],actual=actual,baseline=old,delta={k:actual[k]-old[k] for k in actual}))
                groups=defaultdict(list)
                for w in windows:groups[w['role'],w['T'],w['geometry'],w['reference_affected'],w['reference_own_fill'],w['own_fill']].append(w)
                for key,group in groups.items():
                    interior=[w for w in windows if w['role']==key[0] and w['T']==key[1] and not w['reference_affected'] and w['geometry']=='real']
                    if not interior:raise ValueError('no matched interior endpoint reference')
                    p=dict(run=name,input_map_m=extent,trial=trial,role=key[0],T=key[1],geometry=key[2],reference_affected=key[3],reference_own_fill=key[4],own_fill=key[5],windows=len(group))
                    p['actual']={k:statistics.median(w['actual'][k] for w in group) for k in group[0]['actual']}
                    p['delta']={k:statistics.median(w['delta'][k] for w in group) for k in group[0]['delta']}
                    p['interior_delta']={k:statistics.median(w['delta'][k] for w in interior) for k in group[0]['delta']}
                    p['excess']={k:statistics.median(w['delta'][k]-p['interior_delta'][k] for w in group) for k in group[0]['delta']}
                    process_rows.append(p)
        for rel in ('cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp','cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp'):
            path=root/'source/overlay'/rel;lines=path.read_text().splitlines()
            sources[str(path)]=dict(sha256=common.sha(path),sites={site:[i+1 for i,line in enumerate(lines) if site in line] for site in
                ('V06_FIRST_MMA','V06_MAIN_END','consumer_wait(smem_pipe_read','warpgroup_commit_batch','warpgroup_wait<','collective_mainloop.mma_tail')})
    keys=('run','input_map_m','role','T','geometry','reference_affected','reference_own_fill','own_fill');groups=defaultdict(list)
    for p in process_rows:groups[tuple(p[k] for k in keys)].append(p)
    for key,group in sorted(groups.items()):
        row=dict(zip(keys,key),processes=len(group))
        for metric in ('actual','delta','interior_delta','excess'):row[metric]={k:statistics.median(p[metric][k] for p in group) for k in group[0][metric]}
        summaries.append(row)
    result=dict(summary=summaries,processes=process_rows,raw_sha256=provenance,source_sites=sources,analyzer_sha256=common.sha(Path(__file__)),
        limits=['S uses producer-first-work to consumer-FIRST_MMA in one CTA: cross-role preparation interval, not a pure consumer-wait measurement.',
            'FIRST_MMA is after the initial consumer full wait, before warpgroup_arrive and first GMMA issuance.',
            'MAIN_END is after mma_tail, which contains warpgroup_wait<0> and final consumer releases. No mma-return stamp separates drain from K loop.',
            'exit_tail is final role stamp minus last EPI_DONE; it is not GMMA drain or physical global-write completion.',
            'Current two jobs both use K4096: first versus later output tile is not first versus later K iteration.',
            'Per-window accounting is exact; medians of separately grouped terms are not additive.'])
    common.write_json(output/'C-endpoint-localization.json',result)
    return result


def boundary_cluster_coupling(base,output):
    """Within-call rank timing versus static nonpartner controls; no stall inference."""
    import csv
    from analyze_r13_sm import select_attempts
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    roots=('20261010-R18-cluster-fill-discrimination','20261010-R09-R13-R18-failure-controls','20261009-V09-heldout-job739011')
    all_pairs=[];provenance={}
    for name in roots:
        root=base/name;all_rows=json.loads((root/'cases.json').read_text())
        rows=[r for r in all_rows if r['id'].startswith(('r18_cluster_fill_map','r18_partial_map')) or r['id']=='v09d_cfg_c_P2_partial_later_sw8']
        setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
        data={};statics={}
        for row in rows:
            data[row['id']]={};statics[row['id']]=static_boundary_features(row,setups[row['id']])
            selected,_,_=select_attempts(root,row['id'],('dual',))
            for item in selected:
                rec=item['record'];data[row['id']][rec['trial']]=replay(root,rec,row,retain_roles=True)
                provenance[str(root/rec['raw'])]=rec['raw_sha256']
        baseline=next((r for r in rows if r.get('input_map_m')==1024),None)
        for row in rows:
            op='A' if row['config']=='cfg_a' else 'B';groups=defaultdict(list)
            for f in statics[row['id']]:groups[tuple(f['cluster']),f['j']].append(f)
            # Static rank classes and rotation are chosen without any observed timing.
            strata=defaultdict(list)
            for (cluster,j),members in sorted(groups.items()):
                members.sort(key=lambda f:f['rank'])
                if len(members)!=2 or len({f['T'] for f in members})!=1:raise ValueError('cooperative cluster pair mismatch')
                classes=tuple(boundary_geometry(row,f['coord']) for f in members)
                fills=tuple(f['operands'][op]['own'] for f in members)
                strata[j,members[0]['T'],classes,fills].append(members)
            for trial,obs in sorted(data[row['id']].items()):
                ct=obs['ctas'];earliest=min(c['entry_ns'] for c in ct);last=set(maximum_set([c['end_ns'] for c in ct]))
                if any(tuple(ct[f['cta']]['work'][f['j']])!=tuple(f['coord']) for f in statics[row['id']]):
                    raise ValueError('cluster analysis static scheduler differs from observed work')
                bc=data[baseline['id']][trial]['ctas'] if baseline else None
                for (j,T,classes,fills),clusters in strata.items():
                    for pos,members in enumerate(clusters):
                        controls=[('partner',members[0],members[1])]
                        if len(clusters)>1:controls.append(('rotated_nonpartner',members[0],clusters[(pos+1)%len(clusters)][1]))
                        for relation,fa,fb in controls:
                            a,b=ct[fa['cta']],ct[fb['cta']]
                            for role in range(2):
                                ta=a['roles'][role]['tiles_ns'][j];tb=b['roles'][role]['tiles_ns'][j]
                                La,Lb=ta[1]-ta[0],tb[1]-tb[0]
                                p=dict(run=name,case=row['id'],trial=trial,relation=relation,role=role,j=j,T=T,
                                    phase='first' if j==0 else 'later',classes='+'.join(classes),fill_pattern='+'.join(str(v) for v in fills),
                                    cta_a=fa['cta'],cta_b=fb['cta'],mi_a=fa['coord'][0],ni_a=fa['coord'][1],mi_b=fb['coord'][0],ni_b=fb['coord'][1],
                                    cluster_m=fa['cluster'][0],cluster_n=fa['cluster'][1],
                                    start_a_ns=ta[0]-earliest,start_b_ns=tb[0]-earliest,end_a_ns=ta[1]-earliest,end_b_ns=tb[1]-earliest,
                                    L_a_ns=La,L_b_ns=Lb,start_signed_gap_ns=tb[0]-ta[0],end_signed_gap_ns=tb[1]-ta[1],
                                    start_gap_ns=abs(tb[0]-ta[0]),end_gap_ns=abs(tb[1]-ta[1]),L_gap_ns=abs(Lb-La),
                                    contains_actual_last=bool(last&{fa['cta'],fb['cta']}),delta_L_a_ns=None,delta_L_b_ns=None,delta_L_gap_ns=None)
                                if bc:
                                    ba=bc[fa['cta']]['roles'][role]['tiles_ns'][j];bb=bc[fb['cta']]['roles'][role]['tiles_ns'][j]
                                    da=La-(ba[1]-ba[0]);db=Lb-(bb[1]-bb[0])
                                    p.update(delta_L_a_ns=da,delta_L_b_ns=db,delta_L_gap_ns=abs(da-db))
                                all_pairs.append(p)
    keynames=('run','case','relation','role','phase','j','T','classes','fill_pattern')
    grouped=defaultdict(list)
    for p in all_pairs:grouped[tuple(p[k] for k in (*keynames,'trial'))].append(p)
    metrics=('start_gap_ns','end_gap_ns','L_gap_ns','delta_L_a_ns','delta_L_b_ns','delta_L_gap_ns')
    processes=[]
    for key,group in sorted(grouped.items()):
        r=dict(zip((*keynames,'trial'),key),pairs=len(group))
        for metric in metrics:
            values=[p[metric] for p in group if p[metric] is not None];r[metric]=statistics.median(values) if values else None
        r['end_within_32ns_fraction']=sum(p['end_gap_ns']<=32 for p in group)/len(group)
        r['end_within_128ns_fraction']=sum(p['end_gap_ns']<=128 for p in group)/len(group)
        r['end_gap_smaller_than_start_fraction']=sum(p['end_gap_ns']<p['start_gap_ns'] for p in group)/len(group)
        processes.append(r)
    groups=defaultdict(list)
    for p in processes:groups[tuple(p[k] for k in keynames)].append(p)
    summary=[]
    for key,group in sorted(groups.items()):
        r=dict(zip(keynames,key),processes=len(group))
        for metric in (*metrics,'end_within_32ns_fraction','end_within_128ns_fraction','end_gap_smaller_than_start_fraction'):
            values=[p[metric] for p in group if p[metric] is not None];r[metric]=statistics.median(values) if values else None
        summary.append(r)
    # Retain actual-last partner rows separately for P2 without constructing new prediction candidates.
    critical=[p for p in all_pairs if p['case']=='v09d_cfg_c_P2_partial_later_sw8' and p['relation']=='partner' and p['contains_actual_last']]
    frozen_path=base/'20261009-V09-heldout-job739011/frozen/predictions.json'
    predicted=json.loads(frozen_path.read_text())['predictions']['v09d_cfg_c_P2_partial_later_sw8']['critical_candidates']
    last_clusters=[]
    for trial in range(10):
        pairs=sorted({(p['cta_a'],p['cta_b']) for p in critical if p['trial']==trial and p['role']==1 and p['j']==0})
        last_clusters.append(dict(trial=trial,actual_last_clusters=pairs,
            intersects_frozen_candidates=any(set(pair)&set(predicted) for pair in pairs)))
    source_root=base/'20261010-R18-cluster-fill-discrimination/source'
    evidence={}
    for relative,markers in [
        ('cutlass/include/cutlass/gemm/collective/builders/sm90_common.inl',
         ('sm90_cluster_shape_to_tma_atom','return cute::SM90_TMA_LOAD','return cute::SM90_TMA_LOAD_MULTICAST')),
        ('cutlass/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl',
         ('using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom',
          'using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom')),
        ('cutlass/include/cutlass/pipeline/sm90_pipeline.hpp',
         ('multicast_consumer_arrival_count =','is_same_row_or_col','empty_barrier_ptr_[stage].wait',
          'full_barrier_ptr_[stage].arrive_and_expect_tx','empty_barrier_ptr_[stage].arrive','void producer_tail')),
        ('overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
         ('NumProducerThreadEvents =','K_PIPE_MMAS =','TmaTransactionBytes =','mcast_mask_b |=',
          'get_slice(cluster_local_block_id.x)','pipeline.producer_acquire','tma_load_b.with','warpgroup_wait<','pipeline.consumer_release')),
        ('overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp',
         ('NumMMAThreads == 256','mainloop_pipeline_params.num_consumers','mainloop_pipe_producer_state.advance',
          'mainloop_pipe_consumer_state.advance','collective_mainloop.load_tail'))]:
        path=source_root/relative;lines=path.read_text().splitlines()
        evidence[str(path)]=dict(sha256=common.sha(path),sites={m:[i+1 for i,line in enumerate(lines) if m in line] for m in markers})
    common.write_json(output/'C-cluster-coupling.json',dict(summary=summary,processes=processes,p2_actual_last_pairs=critical,
        p2_frozen_candidates=predicted,p2_last_cluster_overlap=last_clusters,frozen_prediction_sha256=common.sha(frozen_path),
        source_evidence=evidence,cfg_a_pipeline=dict(stages=6,consumer_warpgroups_per_cta=2,empty_arrival_count=4,
            empty_scope='same row OR column; both ranks for cluster 2x1',full_producer_arrival_count=1,
            full_expected_bytes=32768,A_local_bytes=16384,B_slice_bytes_per_rank=8192,B_multicast_mask=3,
            constraints=['Stage reuse waits for prior-generation releases by both consumer warpgroups in both ranks.',
                'Each local full barrier requires local producer arrival and completion of local A plus B slices issued by both ranks.',
                'GMMA issuance waits for local full readiness and local issue/queue dependencies; safe release follows warpgroup_wait<1>.',
                'Pipeline generation advances across output tiles; startup free stages, mma_tail wait<0>, and producer-tail empty waits remain distinct.'],
            scope='Source dependencies only, not a measured attribution of the microsecond excess or a new timing coefficient.'),
        raw_sha256=provenance,analyzer_sha256=common.sha(Path(__file__)),
        limits=['Start/end differences are same role in two ranks of one invocation, using globaltimer.',
            'Map increments compare two independent processes of the same trial; they are not within-call treatment effects.',
            'Static rotated nonpartners match j/T/logical geometry/fill pattern; no nearest-time selection.',
            'Whole-L endpoints cannot identify shared stalls, which peer waited, or the wait-full/GMMA/drain interval.',
            'End gap versus start gap is descriptive; duration differences algebraically depend on both and are not independent causal evidence.',
            'P2 has no matched map intervention here. Actual-last rows are diagnostics, not an enlarged frozen candidate set.']))
    with gzip.open(output/'C-cluster-coupling-pairs.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_pairs[0]));writer.writeheader();writer.writerows(all_pairs)
    return dict(pairs=len(all_pairs),process_strata=len(processes),summary_rows=len(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--v08-padding',action='store_true')
    p.add_argument('--input-map-pairs',action='store_true')
    a=p.parse_args()
    if a.input_map_pairs:
        if a.output is None:p.error('--input-map-pairs requires a new --output directory')
        analyze_input_map_pairs(a.input.resolve(),a.output)
    elif a.v08_padding:
        if a.output is None:p.error('--v08-padding requires a new --output directory')
        analyze_v08_padding(a.input.resolve(),a.output)
    else:analyze(a.input.resolve(),a.output)
