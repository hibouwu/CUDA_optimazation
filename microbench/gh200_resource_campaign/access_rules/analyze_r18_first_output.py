#!/usr/bin/env python3
"""Source ports and same-CTA windows for the proposed cfg_a first-output observer.

This is a prospective local interface. It does not decode a public trace, certify
an observer, fit a service price, or contain new GPU observations.
"""
import argparse
import hashlib
import json
from pathlib import Path


EVENTS=('FIRST','COMMIT_G0','COMMIT_G16','COMMIT_G32','COMMIT_G63','MMA_RETURN','MAIN')
SELECTED_CTAS=(0,1,6,7,14,15,22,23,30,31,38,39)


def first_output_windows(record):
    """Partition one bound observer record using native integer timestamps.

    The caller must validate raw-source binding, GPU/setup/history identities and
    local observer controls. Existing entry/final SM IDs must match. This helper
    never compares native clock64 between CTAs or converts cycles to ns.
    """
    integer_fields=('cta','role','physical_thread','tile_seq','sm_start','sm_end')
    if any(type(record.get(field)) is not int for field in integer_fields):
        raise ValueError('integer CTA, role, tile and SM identities required')
    if not isinstance(record.get('invocation_id'),str) or not record['invocation_id']:
        raise ValueError('one invocation identity required')
    if record.get('config')!='cfg_a' or record['tile_seq']!=0 or record['cta'] not in SELECTED_CTAS:
        raise ValueError('record is outside the fixed first-output observation scope')
    if record['role'] not in (1,2) or record['physical_thread']!=128*record['role']:
        raise ValueError('consumer WG leader required')
    if record['sm_start']<0 or record['sm_start']!=record['sm_end']:
        raise ValueError('same-SM record required for native clock64 windows')
    stamps=record.get('stamps')
    if not isinstance(stamps,dict) or set(stamps)!=set(EVENTS):
        raise ValueError('all seven bound endpoints required exactly once')
    for event in EVENTS:
        stamp=stamps[event]
        if not isinstance(stamp,dict) or set(stamp)!=set(('clock64','globaltimer_ns')):
            raise ValueError('both native clock64 and globaltimer ns required')
        if any(type(value) is not int or not 0<value<2**64 for value in stamp.values()):
            raise ValueError('positive raw uint64 stamps required; zero is unobserved')
    windows=[]
    for start,end in zip(EVENTS,EVENTS[1:]):
        cycles=stamps[end]['clock64']-stamps[start]['clock64']
        ns=stamps[end]['globaltimer_ns']-stamps[start]['globaltimer_ns']
        if cycles<0 or ns<0:raise ValueError('endpoints violate the bound source order')
        windows.append(dict(start=start,end=end,cycles=cycles,ns=ns))
    duration=dict(cycles=stamps['MAIN']['clock64']-stamps['FIRST']['clock64'],
                  ns=stamps['MAIN']['globaltimer_ns']-stamps['FIRST']['globaltimer_ns'])
    if any(sum(window[unit] for window in windows)!=duration[unit] for unit in ('cycles','ns')):
        raise ValueError('window decomposition is not exact')
    return dict(invocation_id=record['invocation_id'],cta=record['cta'],role=record['role'],sm=record['sm_start'],
                physical_thread=record['physical_thread'],tile_seq=record['tile_seq'],
                windows=windows,L=duration,
                scope='Endpoint windows only; source binding and observer validity are caller obligations.')


def first_output_call_windows(records):
    """Require every selected CTA/WG in one invocation before local comparison."""
    results=[];seen=set();invocation=None;sm_by_cta={}
    for record in records:
        result=first_output_windows(record)
        key=result['cta'],result['role']
        if key in seen:raise ValueError('duplicate selected CTA/role')
        seen.add(key)
        if invocation is None:invocation=result['invocation_id']
        if result['invocation_id']!=invocation:raise ValueError('mixed invocations in one call')
        cta=result['cta'];sm=record['sm_start']
        if cta in sm_by_cta and sm_by_cta[cta]!=sm:
            raise ValueError('two roles of one CTA disagree on SM identity')
        sm_by_cta[cta]=sm;results.append(result)
    expected={(cta,role) for cta in SELECTED_CTAS for role in (1,2)}
    if seen!=expected:raise ValueError('complete fixed 12-CTA / 24-role coverage required')
    return dict(invocation_id=invocation,records=sorted(results,key=lambda item:(item['cta'],item['role'])),
        scope='Complete selected first-output call; source, setup, preparation and observer validity remain caller obligations.')


def paired_first_output_windows(map960_records, map1024_records):
    """Signed differences of within-call windows, with all 24 observers retained.

    Caller binds maps, trials, GPU, binary, preparation/history, raw SHA and valid
    controls before calling. This function does not qualify or pair raw samples.
    Native clock origins may differ between CTAs and between invocations.
    """
    target=first_output_call_windows(map960_records)
    reference=first_output_call_windows(map1024_records)
    if target['invocation_id']==reference['invocation_id']:
        raise ValueError('two distinct map invocations required')
    records=[]
    for left,right in zip(target['records'],reference['records']):
        windows=[]
        for a,b in zip(left['windows'],right['windows']):
            windows.append(dict(start=a['start'],end=a['end'],
                map960={unit:a[unit] for unit in ('cycles','ns')},
                map1024={unit:b[unit] for unit in ('cycles','ns')},
                delta={unit:a[unit]-b[unit] for unit in ('cycles','ns')}))
        delta={unit:left['L'][unit]-right['L'][unit] for unit in ('cycles','ns')}
        if any(sum(window['delta'][unit] for window in windows)!=delta[unit] for unit in ('cycles','ns')):
            raise ValueError('paired signed windows do not sum to the L difference')
        records.append(dict(cta=left['cta'],role=left['role'],windows=windows,
            map960_sm=left['sm'],map1024_sm=right['sm'],sm_changed=left['sm']!=right['sm'],
            map960_L=left['L'],map1024_L=right['L'],delta_L=delta))
    return dict(map960_invocation=target['invocation_id'],map1024_invocation=reference['invocation_id'],
        records=records,
        scope='Paired within-CTA/role duration differences only; no cross-call raw-clock subtraction or service inference.')


def source_ports(run,output):
    """Bind the five proposed ports to the archived cooperative mainloop source."""
    run,output=Path(run),Path(output)
    if output.exists():raise ValueError('output must be a new directory')
    paths={
        'collective':run/'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
        'kernel':run/'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp',
        'pipeline':run/'source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp'}
    source={name:path.read_text() for name,path in paths.items()}
    if source['collective'].count('static constexpr int K_PIPE_MMAS = 1;')!=1:
        raise ValueError('the source is not the fixed K_PIPE_MMAS=1 mainloop')
    collective=source['collective']
    start=collective.index('  mma(MainloopPipeline pipeline,')
    end=collective.index('  /// Perform a Consumer Epilogue',start)
    mma=collective[start:end]
    commits=[i for i in range(len(mma)) if mma.startswith('warpgroup_commit_batch();',i)]
    if len(commits)!=3:raise ValueError('expected prologue, unused extra prologue, and steady commit sites')
    steady=commits[2]
    if not steady<mma.index('warpgroup_wait<K_PIPE_MMAS>();',steady)<mma.index('pipeline.consumer_release(smem_pipe_release);',steady):
        raise ValueError('steady commit/wait1/release order changed')
    def line_at(text,index):return text.count('\n',0,index)+1
    kernel=source['kernel'];call=kernel.index('          collective_mainloop.mma(\n')
    if ('int v06_tile_seq = -1;' not in kernel
        or 'PipelineState(): index_{}, phase_{}, count_{} {}' not in source['pipeline']):
        raise ValueError('first-output counter or initial consumer state changed')
    if kernel.count('          collective_mainloop.mma(\n')!=1:
        raise ValueError('unique cooperative mma call required')
    returned=kernel.index('          );',call)
    tail=kernel.index('          collective_mainloop.mma_tail(',returned)
    main=kernel.index('          v06_stamp(V06_MAIN_END, v06_tile_seq);',tail)
    if not returned<tail<main:raise ValueError('mma return / tail / MAIN order changed')
    points=[]
    for group in (0,16,32,63):
        points.append(dict(event=f'COMMIT_G{group}',group=group,slot=group%6,
            generation=group//6,full_wait_phase=(group//6)%2,
            source='collective',after_line=line_at(collective,start+commits[0 if group==0 else 2]),
            selector='prologue' if group==0 else f'64-k_tile_count=={group}',
            before='read-state increment' if group==0 else 'wait<1> and release(group-1)'))
    points.append(dict(event='MMA_RETURN',source='kernel',after_line=line_at(kernel,returned),
        before_line=line_at(kernel,tail),completed_source_path='last wait<1> and release62; final fence and return',
        pending_source_path='mma_tail wait<0> and release63'))
    from v08_model import scheduled_work
    case_path=run/'cases.json';setup_path=run/'static_setup.json'
    cases={row['id']:row for row in json.loads(case_path.read_text())}
    setups={row['case']:row['setup'] for row in json.loads(setup_path.read_text())}
    static=[]
    for extent in (960,1024):
        case=f'r18_partial_map{extent}';row=cases[case];setup=setups[case]
        if any(row[field]!=expected for field,expected in dict(config='cfg_a',m=960,n=2816,k=4096,
            storage_m=1024,zero_m=960,input_map_m=extent,swizzle=1).items()):
            raise ValueError('the fixed two-map conditions changed')
        if any(setup[field]!=expected for field,expected in dict(grid=[132,1,1],stages=6,cluster=[2,1],tile=[128,128,64]).items()):
            raise ValueError('the fixed grid/cluster/stages changed')
        work=scheduled_work('cfg_a',row['m'],row['n'],setup['grid'],row['swizzle'])
        maximum=max(map(len,work));affected=[]
        for cta in range(0,len(work),2):
            members=work[cta:cta+2]
            if len(members)!=2 or members[0][0][0]//2!=members[1][0][0]//2 or members[0][0][1]!=members[1][0][1]:
                raise ValueError('physical CTA pair differs from the static cluster')
            # Use the map960 reference geometry for both maps, never timing or last-CTA data.
            fill=any(128*member[0][0]+128>960 for member in members)
            if fill and all(len(member)==maximum for member in members):affected.extend((cta,cta+1))
        if tuple(sorted([0,1]+affected))!=SELECTED_CTAS:
            raise ValueError('static affected first/Tmax clusters changed')
        static.append(dict(case=case,maximum_output_tiles=maximum,
            selected=[dict(cta=cta,work=[list(coord) for coord in work[cta]]) for cta in SELECTED_CTAS]))
    report=dict(status='Prospective first-stage source contract; no GPU observations or accepted service prices.',
        source={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()},
        source_ports=points,existing_endpoints=['FIRST','MAIN'],
        static_selection=static,
        condition_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (case_path,setup_path)},
        fixed_scope=dict(config='cfg_a',logical=[960,2816,4096],storage_m=1024,zero_m=960,
            input_map_m=[960,1024],stages=6,k_tiles=64,output_tile_seq=0,ctas=list(SELECTED_CTAS),
            roles=[1,2],physical_threads=[128,256]),
        payload=dict(new_points=5,writers=24,timestamp_words_per_point=2,uint64_bytes=8,new_timestamp_bytes=1920,
            meaning='Timestamp payload only; total allocation, metadata and clearing capacity are separately declared.'),
        windows=[dict(start=start,end=end) for start,end in zip(EVENTS,EVENTS[1:])],
        comparison='Subtract endpoints inside each invocation/CTA/role first; then map960 minus map1024 durations. Preserve signed segment differences.',
        aggregation='Per-record windows and paired signed deltas telescope exactly. Independently aggregated medians need not add.',
        implementation='Main integration owns probe/runner and actual source/SASS binding. Do not reuse cfg_b commit1 naming or its trace decoder.',
        gates='Matched on/off preparation and an independent control pilot must freeze concrete ns validity gates before target sampling.',
        limits=['COMMIT_G63 to MMA_RETURN includes last wait<1>/release62 and return overhead.',
                'MMA_RETURN to MAIN includes final wait<0>/release63 and endpoint overhead, not bare GMMA retirement.',
                'WG leaders identify source control-flow cuts, not completion of every remote empty arrival.',
                'No added waits, arrivals or synchronization; no pipeline-state reset between outputs.',
                'clock64 stays within one CTA on one SM; cross-rank comparison uses globaltimer windows.'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'first-output-source-ports.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=source_ports(args.input,args.output)
    print(json.dumps(dict(points=len(report['source_ports']),timestamp_payload_bytes=report['payload']['new_timestamp_bytes'])))
