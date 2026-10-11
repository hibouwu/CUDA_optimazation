#!/usr/bin/env python3
"""Reuse frozen cfg_b boundary interfaces on four full-tile raster/pitch cases."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from analyze_r10_boundary_completion import completion_fragment,work_identity_projection
from analyze_r10_boundary_lifecycle import stage_lifecycle
from supply_model import cfg_b_persistent_boundary_ledger,cfg_b_source_program


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        value=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=value;return value
    def read(path):sha(path);return json.loads(path.read_text())
    old=root/'20261007-R10-job735876-v4';new=root/'20261010-R10-raster-pitch'
    accepted_path=new/'reanalysis/B-20261011-request-work-audit-v3/B-request-work-audit.json'
    accepted=read(accepted_path)
    if inputs[str(accepted_path)]!='63d701d1f4b12d33a1389ad76257d58748bdf3917bb9169d84daacfa6f704842':
        raise ValueError('accepted full-work audit changed')
    context_path=new/'reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json'
    contexts=read(context_path)['contexts']
    if inputs[str(context_path)]!=accepted['input_sha256'][str(context_path)]:
        raise ValueError('accepted full-work context changed')
    legacy=read(old/'reanalysis/B-20261011-boundary-completion-v2/B-boundary-completion.json')
    lifecycle=read(old/'reanalysis/B-20261011-boundary-lifecycle-v2/B-boundary-lifecycle.json')
    old_manifest=read(old/'source_hashes.json');new_manifest=read(new/'source_hashes.json')
    common_sources=[]
    for relative in (
        'cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
        'cutlass/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl',
        'cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp',
        'cutlass/include/cutlass/pipeline/sm90_pipeline.hpp',
        'cutlass/include/cutlass/gemm/kernel/static_tile_scheduler.hpp',
        'cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp',
        'cutlass/include/cutlass/gemm/kernel/tile_scheduler_params.h',
        'cutlass/include/cute/arch/mma_sm90.hpp','cutlass/include/cute/arch/mma_sm90_gmma.hpp',
        'cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp','cutlass/include/cute/atom/mma_atom.hpp',
        'cutlass/include/cute/algorithm/gemm.hpp'):
        a=old/'source'/relative;b=new/'source'/relative
        if sha(a)!=old_manifest[relative] or sha(b)!=new_manifest['source/'+relative]:
            raise ValueError('old/new archived source differs from its manifest: '+relative)
        if inputs[str(a)]!=inputs[str(b)]:raise ValueError('old/new core rule source differs: '+relative)
        common_sources.append(dict(relative=relative,sha256=inputs[str(a)]))
    probe=new/'source/probes/r18.cu'
    if sha(probe)!=new_manifest['source/probes/r18.cu']:raise ValueError('new probe SHA differs')
    sites={}
    for needle in ('using KernelTile = Shape<_128, _128, _64>','using KernelCluster = Shape<_1, _1, _1>',
            'KernelTmaWarpSpecializedPingpong','arguments.scheduler.max_swizzle_size = o.swizzle',
            'arguments.scheduler.raster_order = o.raster_order'):
        sites[needle]=[i+1 for i,line in enumerate(probe.read_text().splitlines()) if needle in line]
        if not sites[needle]:raise ValueError('new probe contract changed: '+needle)
    cases=[];programs={}
    for axis in ('m','n'):
        for pitch in (0,32):
            name=f'r10_raster_along_{axis}_p{pitch}';context=contexts[name]
            row,setup=context['row'],context['setup']
            program=cfg_b_source_program(row,setup)
            previous=next(c for c in accepted['cases'] if c['case']==name)
            source_digest=hashlib.sha256(json.dumps(program['source_keys'],separators=(',',':')).encode()).hexdigest()
            if source_digest!=previous['source_identity_sha256']:raise ValueError('accepted source program changed')
            if setup['cluster']!=[1,1]:raise ValueError('new ordinary cfg_b cluster differs')
            ledger=cfg_b_persistent_boundary_ledger(row,dict(setup,cluster=[1,1,1]));kt=ledger['ktiles']
            if ledger['scheduler']['output_counts']!=program['output_counts']:raise ValueError('CTA work counts differ')
            identities={};work_hash=hashlib.sha256();identity_hash=hashlib.sha256();counts=Counter();witness=[]
            for out in ledger['outputs']:
                c,j=out['cta'],out['output_ordinal']
                for q,group in enumerate(out['groups']):
                    keys=tuple(a['source_key'] for a in group['atoms'])
                    if keys!=program['source_keys'][c][j*kt+q]:raise ValueError('full and boundary source API keys differ')
                    fragment=completion_fragment(out,group,setup['stages'])
                    projected=work_identity_projection(fragment,identities)
                    identity_hash.update(json.dumps(projected,sort_keys=True,separators=(',',':')).encode()+b'\n')
                    work=dict(fragment,atoms=[{key:value for key,value in a.items() if key!='source_key'} for a in fragment['atoms']])
                    work_hash.update(json.dumps(work,sort_keys=True,separators=(',',':')).encode()+b'\n')
                    if any(a['destination_fill_bytes'] for a in fragment['atoms']):raise ValueError('new full-tile case acquired fill')
                    if any(a['valid_available'] is None or a['fill_completion'] is not None for a in fragment['atoms']):
                        raise ValueError('new full-tile completion shape differs')
                    counts.update(K_groups=1,source_atoms=3,source_valid_bytes=32768,
                        destination_expected_bytes=32768,nodes=len(fragment['nodes']),edges=sum(map(len,fragment['nodes'].values())))
                    if c<2 and j==0 and q==0:witness.append(dict(cta=c,coord=out['coord'],source_keys=keys))
            if counts['source_valid_bytes']/1024!=program['totals']['source_valid_KiB']:
                raise ValueError('existing source work total differs')
            compressed=gzip.compress(json.dumps(ledger,separators=(',',':')).encode(),mtime=0)
            filename=name+'-bound-ledger.json.gz';(output/filename).write_bytes(compressed)
            programs[name]=program
            cases.append(dict(case=name,row=row,gpu_uuid=setup['gpu_uuid'],scheduler=ledger['scheduler'],
                shape=ledger['shape'],totals=ledger['totals'],completion_counts=dict(counts),
                source_API_identity_sha256=source_digest,work_topology_sha256=work_hash.hexdigest(),
                work_identity_topology_sha256=identity_hash.hexdigest(),source_identity_classes=len(identities),
                source_reuse_witness=witness,B_allocation_bytes=row['k']*row['alloc_ldb']*2,
                lifecycle_by_T={str(t):stage_lifecycle(kt,setup['stages'],t) for t in sorted(set(program['output_counts']))},
                ledger_file=filename,ledger_sha256=hashlib.sha256(compressed).hexdigest()))
    pitch_pairs=[];raster_pairs=[]
    for axis in ('m','n'):
        a,b=(next(c for c in cases if c['case']==f'r10_raster_along_{axis}_p{p}') for p in (0,32))
        if a['work_identity_topology_sha256']!=b['work_identity_topology_sha256']:
            raise ValueError('within-raster pitch changed work/source equality projection')
        same=programs[a['case']]['source_keys']==programs[b['case']]['source_keys']
        if same:raise ValueError('pitch did not change actual source keys')
        pitch_pairs.append(dict(raster=axis,work_and_source_equality_same=True,actual_source_keys_same=same,
            projection_sha256=a['work_identity_topology_sha256']))
    for pitch in (0,32):
        a,b=(next(c for c in cases if c['case']==f'r10_raster_along_{axis}_p{pitch}') for axis in ('m','n'))
        if a['work_topology_sha256']!=b['work_topology_sha256']:raise ValueError('raster changed pure completion work')
        if a['work_identity_topology_sha256']==b['work_identity_topology_sha256']:
            raise ValueError('raster source equality witness unexpectedly vanished')
        witness=[]
        for case in (a,b):
            first,second=case['source_reuse_witness']
            equal=[x==y for x,y in zip(first['source_keys'],second['source_keys'])]
            expected=[False,True,True] if case['row']['raster_order']=='along_m' else [True,False,False]
            if equal!=expected:raise ValueError('adjacent CTA source reuse witness differs')
            witness.append(dict(case=case['case'],CTA0_coord=first['coord'],CTA1_coord=second['coord'],
                same_source_A_B0_B1=equal))
        raster_pairs.append(dict(pitch_bytes=pitch,pure_work_and_completion_topology_same=True,
            source_equality_partition_same=False,witness=witness))
    for name in ('analyze_r10_boundary_transfer.py','analyze_r10_boundary_completion.py','analyze_r10_boundary_lifecycle.py',
            'supply_model.py','v08_model.py','v06_model.py'):sha(Path(__file__).with_name(name))
    report=dict(scope='Same unpriced work/state interface on old boundary and new full-tile cases; no composed timing or new source server.',
        cases=cases,pitch_pairs=pitch_pairs,raster_pairs=raster_pairs,old_new_common_sources=common_sources,new_probe_sites=sites,
        legacy_summary=dict(shape=[2600,3000,2000],K_groups_per_output=32,output_counts_histogram={'4':108,'3':24},
            per_case=legacy['cases'][0]['totals'],lifecycle_by_T=lifecycle['cases'][0]['lifecycle_by_CTA_output_count']),
        common_rules=['g=j*ceil(K/64)+q; same-slot predecessor g-S, owning WG=j%2.',
            'A/B0/B1 destination work32768B and full MMA group remain whole; valid source/fill are separate.',
            'Normal wait1 releases previous group; last group tail wait0; next MMA and epilogue orders are distinct.',
            'With S6 and Kt>=6, first five next-output acquires need normal release; sixth needs prior final tail release.'],
        conclusions=['Old/new core rule sources are identical, but Kt32 versus16 changes global groups, slots, phases, per-CTA work counts and final generations.',
            'Pitch pairs keep completion work and logical source-equality partition. Their actual addresses/strides still differ.',
            'Raster changes the source-equality partition even when pure completion work is identical; adjacent CTA first outputs reuse B alongM and A alongN.',
            'Within the new raster/pitch quartet, pure completion work is identical. Source-equality-aware rules receive different inputs under raster, but equality metadata alone still cannot distinguish either pitch pair.',
            'No source/fill/D price, per-card timing transfer, observed-window normalization, old/new response prediction or frozen qualification follows from this audit.'],
        input_sha256=inputs)
    (output/'B-boundary-transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'B-boundary-transfer.md').write_text('\n'.join(['# 旧／新R10的共同工作接口','',
        '旧：504输出×32K组，T4=108/T3=24；新：432输出×16K组，T4=36/T3=96。',
        '共同生命周期源码逐字节相同，持续ring公式相同；同一j的slot/phase不能直接复制。',
        '同raster改变pitch：完整工作、完成拓扑和源身份等价类不变，实际地址改变。',
        '同pitch改变raster：完整工作/完成拓扑不变，源身份等价关系改变；相邻CTA首输出沿M复用B、沿N复用A。',
        '旧／新shape、stage状态和输出计数保留各自定义；没有时间价格或资格转移。','']))
    print(json.dumps(dict(cases=len(cases),common_sources=len(common_sources),pitch_pairs=pitch_pairs,raster_pairs=raster_pairs),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.root.resolve(),args.output.resolve())
