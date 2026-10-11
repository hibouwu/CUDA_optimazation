#!/usr/bin/env python3
"""Bind archived R14 source guarantees to actual output generations, in cycles only."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from analyze_r14 import decode
from output_model import (cfg_a_rank_dependencies, cfg_a_output_source_guarantees, dependency_bounds,
                          cooperative_output_source_layout, cooperative_output_source_dependencies,
                          cooperative_output_valid_subtiles)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cfg_c_layout_source_evidence(archive, target):
    """Match output dispatch source, while retaining distinct compiled geometries."""
    archive,target=Path(archive),Path(target)
    manifest=json.loads((target/'source_hashes.json').read_text())
    names=['cutlass/include/cutlass/pipeline/sm90_pipeline.hpp',
        'cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp',
        'cutlass/include/cutlass/epilogue/collective/builders/sm90_builder.inl',
        'cutlass/include/cute/arch/copy_sm90_tma.hpp']
    matched={}
    for name in names:
        path=target/'source'/name
        if sha(path)!=manifest['source/'+name] or sha(path)!=sha(archive/'source'/name):
            raise ValueError('cfg_c output dispatch source differs from R14: '+name)
        matched[name]=sha(path)
    probe=target/'source/probes/r18.cu'
    if sha(probe)!=manifest['source/probes/r18.cu']:raise ValueError('cfg_c probe source changed')
    text=probe.read_text();start=text.index('constexpr const char* kConfigName = "cfg_c";')
    branch=text[start:text.index('#define V06_STAGES_AUTO 0',start)]
    for token in ('using KernelTile = Shape<_256, _128, _64>;',
                  'using KernelCluster = Shape<_1, _2, _1>;',
                  'using EpilogueTile = Shape<_128, _32>;',
                  'TmaWarpSpecializedCooperative;'):
        if token not in branch:raise ValueError('cfg_c output geometry or schedule changed')
    if 'float, float, void' not in text:raise ValueError('expected void source C for delayed stores')
    builder=(target/'source'/names[2]).read_text()
    for token in ('constexpr int StagesD = cute::min(EpiTiles, 2);',
                  'constexpr bool DelayTmaStore = is_void_v<ElementC>',
                  '(sizeof_bits_v<ElementC> == sizeof_bits_v<ElementD>)'):
        if token not in builder:raise ValueError('output dispatch policy changed')
    setups=json.loads((target/'static_setup.json').read_text())
    selected=[row['setup'] for row in setups if row['case']=='v09d_cfg_c_P2_partial_later_sw8']
    if len(selected)!=1:raise ValueError('exact cfg_c P2 static setup required')
    setup=selected[0]
    if any(setup[key]!=value for key,value in dict(config='cfg_c',tile=[256,128,64],cluster=[1,2],
        epilogue_tile=[128,32],stages=4,threads=384).items()):
        raise ValueError('compiled cfg_c P2 geometry changed')
    binaries=json.loads((target/'build/binary_hashes.json').read_text())
    for name in ('cfg_c_plain','cfg_c_dual'):
        if sha(target/'build'/name)!=binaries['build/'+name]:raise ValueError('cfg_c binary changed')
    from v08_model import scheduled_work
    cases=json.loads((target/'cases.json').read_text())
    rows=[row for row in cases if row['id']=='v09d_cfg_c_P2_partial_later_sw8']
    if len(rows)!=1:raise ValueError('exact cfg_c P2 case required')
    row=rows[0]
    work=scheduled_work('cfg_c',row['m'],row['n'],setup['grid'],row['swizzle'])
    output_work=[];patterns={};valid_total=0
    for cta,coords in enumerate(work):
        tiles=[]
        for tile,coord in enumerate(coords):
            coverage=cooperative_output_valid_subtiles('cfg_c',row['m'],row['n'],coord,tile)
            values=tuple(item['valid_bytes'] for item in coverage['subtiles'])
            patterns[values]=patterns.get(values,0)+1;valid_total+=coverage['valid_bytes']
            tiles.append(dict(tile=tile,coord=list(coord),subtile_valid_bytes=list(values),
                buffer_valid_bytes=coverage['buffer_valid_bytes'],valid_bytes=coverage['valid_bytes'],
                first_subtile=coverage['subtiles'][0]['subtile'],
                first_source_generation=coverage['subtiles'][0]['generation'],
                last_valid_subtile=coverage['last_valid_subtile']))
        output_work.append(dict(cta=cta,tiles=tiles))
    if valid_total!=row['m']*row['n']*4:raise ValueError('static output coverage does not conserve logical FP32 bytes')
    return dict(target=str(target),matched_dispatch_source_sha256=matched,
        target_probe_sha256=sha(probe),target_static_setup_sha256=sha(target/'static_setup.json'),
        target_binary_sha256={name:binaries['build/'+name] for name in ('cfg_c_plain','cfg_c_dual')},
        target_setup=setup,target_case=row,target_case_sha256=sha(target/'cases.json'),
        static_scheduler_code_sha256=sha(Path(__file__).parent/'v08_model.py'),
        layout=cooperative_output_source_layout('cfg_c',max(map(len,work))),
        output_work=output_work,valid_output_bytes=valid_total,
        coverage_patterns=[dict(subtile_valid_bytes=list(values),tiles=count) for values,count in sorted(patterns.items())],
        limits='Matching dispatch policy supports source-generation counts only; no cfg_c R14 fine trace or transferred timing price.')


def output_source_windows(rows, output_tiles, config='cfg_a'):
    """Exact same-role accounting; middle release is a next-output guarantee."""
    issuer=sorted((row for row in rows if row['role']==2),key=lambda row:row['seq'])
    if [row['seq'] for row in issuer]!=list(range(output_tiles)):
        raise ValueError('missing or repeated issuer tile')
    if config=='cfg_a':
        guarantees=[entry for entry in cfg_a_output_source_guarantees(output_tiles) if entry['rank']==0]
    elif config=='baseline':
        guarantees=cooperative_output_source_layout('r14_baseline',output_tiles)['source_guarantees']
    else:raise ValueError('R14 lifecycle observations support cfg_a or legacy baseline, not cfg_c')
    result=[]
    for tile,(row,guarantee) in enumerate(zip(issuer,guarantees)):
        permit=row['epi_permit_cycle'];release=row['source_release_cycle']
        if not row['main_end_cycle']<=permit<=release:
            raise ValueError('invalid source guarantee program order')
        following=None if guarantee['terminal'] else issuer[tile+1]
        lower=permit if following is None else following['epi_permit_cycle']
        if following is not None and not following['main_end_cycle']<=lower<=release:
            raise ValueError('middle guarantee precedes next output epilogue')
        total=release-permit
        carried=lower-permit
        after=release-lower
        if total!=carried+after:raise ValueError('source interval accounting mismatch')
        result.append(dict(tile=tile,terminal=guarantee['terminal'],guarantee_token=guarantee['guarantee_token'],
            source_completion=guarantee['source_completion'],protected_buffer=guarantee['protected_buffer'],
            source_generation=guarantee['source_generation'],read_through_count=guarantee['read_through_count'],
            committed_count=guarantee['committed_count'],pending_batches=guarantee['pending_batches'],
            observed_permit_cycle=permit,observed_release_cycle=release,
            guarantee_lower_cycle=lower,guarantee_upper_cycle=release,
            permit_to_release_cycles=total,carried_to_next_permit_cycles=carried,
            after_next_permit_cycles=None if following is None else after,
            terminal_service_upper_cycles=after if following is None else None))
    return result


def output_cycle_bounds(setup, rows, windows, config='cfg_a'):
    """Zero-lag topology in clock64 units; no ns prices or frequency conversion."""
    if config=='cfg_a':graph=cfg_a_rank_dependencies(setup['stages'],setup['k']//64,setup['tiles_per_cta'])
    elif config=='baseline':graph=cooperative_output_source_dependencies('r14_baseline',setup['tiles_per_cta'])
    else:raise ValueError('no R14 cfg_c timing evidence')
    origin=min(row['main_end_cycle'] for row in rows if row['role'])
    anchors={}
    for row in rows:
        if not row['role']:continue
        prefix=f"r0.w{row['role']-1}.t{row['seq']}."
        for event,field in [('MAIN','main_end_cycle'),('PERMIT','epi_permit_cycle')]:
            value=row[field]-origin
            anchors[prefix+event]=(value,value)
    for window in windows:
        anchors[window['guarantee_token']]=(window['guarantee_lower_cycle']-origin,
                                            window['guarantee_upper_cycle']-origin)
    # dependency_bounds is numeric. With zero lags, these bounds use the anchor
    # unit (cycles); latencies_ns is deliberately not supplied.
    bounds=dependency_bounds(graph,anchors)
    result=[]
    for window in windows:
        low,high=bounds[window['source_completion']]
        result.append(dict(tile=window['tile'],source_completion=window['source_completion'],
            lower_cycles=low,upper_cycles=high,width_cycles=None if low is None or high is None else high-low,
            observed_guarantee_upper_cycles=window['observed_release_cycle']-origin))
    return result


def analyze(archive, source_reference, output, config='cfg_a', cfg_c_reference=None):
    import numpy as np
    archive,source_reference,output=Path(archive),Path(source_reference),Path(output)
    if output.exists():raise ValueError('output must be a new directory')
    if config not in ('cfg_a','baseline'):raise ValueError('unsupported R14 observed configuration')
    if config=='baseline' and cfg_c_reference is None:raise ValueError('explicit cfg_c source reference required')
    target_evidence=cfg_c_layout_source_evidence(archive,cfg_c_reference) if config=='baseline' else None
    cases=tuple(f'{config}_k4096_t{tiles}' for tiles in (1,8))
    tile_n=128 if config=='cfg_a' else 256
    stages=6 if config=='cfg_a' else 4
    manifest=json.loads((archive/'source_hashes.json').read_text())
    for relative,digest in manifest.items():
        if sha(archive/relative)!=digest:raise ValueError('archived compiled input changed: '+relative)
    originals=['cutlass/include/cutlass/pipeline/sm90_pipeline.hpp',
        'cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp',
        'cutlass/include/cutlass/epilogue/collective/builders/sm90_builder.inl',
        'cutlass/include/cute/arch/copy_sm90_tma.hpp']
    source_hashes={}
    for name in originals+['probes/r14.cu','probes/r14_trace.hpp','run_r14.py']:
        path=archive/'source'/name;digest=sha(path)
        if digest!=manifest['source/'+name]:raise ValueError('archived source changed: '+name)
        source_hashes[name]=digest
        if name in originals and digest!=sha(source_reference/'source'/name):
            raise ValueError('output lifecycle source differs from rank dependency source: '+name)
    binary_hashes=json.loads((archive/'build/binary_hashes.json').read_text())
    for name in (config+'_plain',config+'_trace'):
        if sha(archive/'build'/name)!=binary_hashes[name]:raise ValueError('archived binary changed')
    grid=np.arange(17,dtype=np.int64)
    r,c,k=grid[:,None,None],grid[None,:,None],grid[None,None,:]
    products=((7*r+13*k)%17-8)*((5*k+11*c)%17-8)
    reference=(products.sum(axis=2)*(4096//17)+products[:,:,:4096%17].sum(axis=2))/1024
    provenance={str(archive/name):sha(archive/name) for name in
                ('source_hashes.json','environment.json','protocol-main.json','build/binary_hashes.json')}
    processes=[];windows=[];intervals=[];failures=[];calls={};checked_values=0
    for case in cases:
        for variant in ('plain','trace'):
            for path in sorted((archive/'samples'/(variant+'_'+case)).glob('main-*/result.json')):
                record=json.loads(path.read_text());raw=path.parent/'stdout.txt.gz'
                provenance[str(path)]=sha(path)
                if (record['case_id']!=variant+'_'+case or record['set']!='main'
                        or path.parent.name!='main-%02d'%record['trial'] or record.get('timed_out',False)
                        or record['raw']['path']!=raw.name or record['raw']['bytes']!=raw.stat().st_size
                        or record['raw']['sha256']!=sha(raw)):
                    raise ValueError('process or raw identity changed')
                provenance[str(raw)]=record['raw']['sha256']
                if record['status']!='measured':
                    failures.append(dict(case=case,variant=variant,trial=record['trial'],record=record));continue
                if record['returncode']!=0:raise ValueError('measured process did not succeed')
                with gzip.open(raw,'rt') as stream:events={event['event']:event for line in stream for event in [json.loads(line)]}
                setup,call,check=(events[key] for key in ('setup','call','check'))
                tiles=1 if case.endswith('t1') else 8
                if (setup!=record['setup'] or setup['config']!=config or setup['tile']!=[128,tile_n,64]
                        or setup['epilogue_tile']!=[128,32] or setup['cluster']!=[2,1] or setup['stages']!=stages
                        or setup['m']!=256 or setup['n']!=2*tile_n*tiles or setup['k']!=4096
                        or setup['lda']!=4096 or setup['ldb']!=setup['n'] or setup['ldd']!=setup['n']
                        or math.prod(setup['grid'])!=4
                        or setup['tiles_per_cta']!=tiles or setup['trace_pair']!='output'
                        or setup['traced']!=(variant=='trace')):
                    raise ValueError('R14 output geometry or record identity changed')
                indices=np.asarray(check['checked_indices'],dtype=np.int64)
                values=np.asarray(check['checked_values'],dtype=np.float64)
                if (check['status']!='ok' or check['padding_errors'] or check['nonfinite']
                        or check['samples']!=setup['m']*setup['n'] or len(values)!=check['samples']
                        or not np.array_equal(indices,np.arange(check['samples']))):
                    raise ValueError('incomplete output check')
                expected=reference[(indices//setup['n'])%17,(indices%setup['n'])%17]
                if not np.array_equal(values,expected):raise ValueError('independent numeric replay failed')
                checked_values+=len(values)
                warm=call['warmup_us']
                if not 8<=len(warm)<=30 or statistics.pstdev(warm[-5:])/statistics.fmean(warm[-5:])>.02:
                    raise ValueError('warmup not qualified')
                key=case,variant,record['trial']
                if key in calls:raise ValueError('duplicate successful process')
                calls[key]=call['elapsed_us']
                if variant=='plain':continue
                rows=decode(setup,call['trace'])
                current=output_source_windows(rows,tiles,config)
                bounds=output_cycle_bounds(setup,rows,current,config)
                meta=dict(case=case,trial=record['trial'],source=str(raw))
                windows.extend(dict(meta,**window) for window in current)
                intervals.extend(dict(meta,**bound) for bound in bounds)
                processes.append(dict(meta,source_guarantee_windows=len(current),zero_lag_feasible=True))
    qualification=[]
    for case in cases:
        plain=[calls.get((case,'plain',trial)) for trial in range(10)]
        trace=[calls.get((case,'trace',trial)) for trial in range(10)]
        if None in plain+trace or any(key[2] not in range(10) for key in calls if key[0]==case):
            raise ValueError('requires original ten paired processes, without substitutions')
        delta=[b/a-1 for a,b in zip(plain,trace)]
        cvs=[statistics.pstdev(values)/statistics.fmean(values) for values in (plain,trace)]
        qualified=max(cvs)<=.05 and abs(statistics.median(delta))<=.05 and not failures
        if not qualified:raise ValueError('archived condition not qualified by original policy')
        current=[w for w in windows if w['case']==case]
        middle=[w for w in current if not w['terminal']];terminal=[w for w in current if w['terminal']]
        middle_processes=[]
        for trial in range(10):
            sample=[w for w in middle if w['trial']==trial]
            if sample:
                middle_processes.append(dict(trial=trial,windows=len(sample),
                    permit_to_release_cycles=statistics.median(w['permit_to_release_cycles'] for w in sample),
                    carried_to_next_permit_cycles=statistics.median(w['carried_to_next_permit_cycles'] for w in sample),
                    after_next_permit_cycles=statistics.median(w['after_next_permit_cycles'] for w in sample)))
        qualification.append(dict(case=case,plain_cv=cvs[0],trace_cv=cvs[1],paired_perturbation=statistics.median(delta),
            terminal_E_cycles=statistics.median(w['terminal_service_upper_cycles'] for w in terminal),
            middle_windows=len(middle),middle_processes=middle_processes,
            middle_permit_to_release_cycles=None if not middle else statistics.median(w['permit_to_release_cycles'] for w in middle_processes),
            middle_after_next_permit_cycles=None if not middle else statistics.median(w['after_next_permit_cycles'] for w in middle_processes)))
    output.mkdir(parents=True,exist_ok=False)
    for name,records in [('source-guarantees',windows),('source-intervals',intervals)]:
        with (output/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=dict(scope='Existing R14 endpoint binding in cycles; no new heldout or transferred service price.',
        archive=str(archive),source_reference=str(source_reference),
        archived_source_files_verified=len(manifest),
        qualification=qualification,processes=processes,failures=failures,checked_values=checked_values,
        source_sha256=source_hashes,binary_sha256={name:binary_hashes[name] for name in (config+'_plain',config+'_trace')},
        raw_sha256=provenance,
        code_sha256={name:sha(Path(__file__).parent/name) for name in ('analyze_r15_lifecycle.py','output_model.py','analyze_r14.py')},
        limits=['Middle source guarantee belongs to the next output acquire; not a standalone epilogue duration.',
                'Actual source read completion may precede the guarantee timestamp.',
                'Fine-stage ns slots are unobserved; no frequency conversion is used.',
                'Four-CTA historical output observations do not calibrate full-GPU or new-card service.'])
    if config=='baseline':
        result['observed_configuration']=dict(config='baseline',tile=[128,256,64],cluster=[2,1],stages=4,
            target_cfg_c=dict(tile=[256,128,64],cluster=[1,2],stages=4),
            transferable='Source-guarantee counts only after matching dispatch policy; not timing, traversal coordinates or input dependencies.')
        result['output_layout']=cooperative_output_source_layout('r14_baseline',8)
        result['cfg_c_source_evidence']=target_evidence
    (output/'C-output-lifecycle.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--source-reference',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',choices=('cfg_a','baseline'),default='cfg_a')
    parser.add_argument('--cfg-c-reference',type=Path)
    args=parser.parse_args()
    result=analyze(args.archive,args.source_reference,args.output,args.config,args.cfg_c_reference)
    print(json.dumps(dict(trace_processes=len(result['processes']),checked_values=result['checked_values'],qualification=result['qualification'])))
