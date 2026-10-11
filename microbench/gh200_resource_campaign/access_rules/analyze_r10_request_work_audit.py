#!/usr/bin/env python3
"""Static full request-work collisions; no new timing model or candidate score."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from analyze_r10_granule_overlap import source_granules,static_wave
from supply_model import cfg_b_source_program


def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def independent_count(key,width):
    """Sum full row spans; assert rows cannot overlap an address unit."""
    op,_,_,extents,strides,offset,origin,shape=key
    inner=1 if op=='A' else 0;outer=1-inner
    if any(origin[d]+shape[d]>extents[d] for d in range(2)):
        raise ValueError('full-tile audit cannot silently remove source/fill work')
    row_bytes=shape[inner]*2
    if strides[outer]<row_bytes+width-1:raise ValueError('independent row spans may overlap')
    base=offset+sum(origin[d]*strides[d] for d in range(2))
    return sum(((base+r*strides[outer])%width+row_bytes+width-1)//width for r in range(shape[outer]))


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        value=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=value
        return value
    def read(path):sha(path);return json.loads(path.read_text())
    current=read(root/'20261010-R10-raster-pitch/reanalysis/B-20261011-input-path-v1/B-input-path.json')
    contexts=read(root/'20261010-R10-raster-pitch/reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json')['contexts']
    clock=read(root/'20261010-R10-raster-pitch/reanalysis/B-20261011-clock-consistency-v1/B-r10-clock-consistency.json')
    old=read(root/'20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/B-summary.json')
    prior=read(root/'20261010-R10-raster-pitch/reanalysis/B-20261011-granule-overlap-v1/B-source-granule-overlap.json')
    legacy=read(root/'20261007-R10-job735876-v4/protocol-formal.json')
    rehashed=set()
    for previous in (current,clock,prior):
        for name,value in previous['input_sha256'].items():
            path=Path(name)
            if path.is_relative_to(root):
                if sha(path)!=value:raise ValueError('reused evidence changed: '+name)
                rehashed.add(name)
    for name,value in old['additional_input_sha256'].items():
        path=Path(name)
        if path.is_relative_to(root):
            if sha(path)!=value:raise ValueError('old evidence changed: '+name)
            rehashed.add(name)
    for run,names in (
        ('20261009-R10-b-coverage-job738203',('cfg_b_bcurve_p0','cfg_b_bcurve_p32')),
        ('20261009-V09-heldout-job739011',('v09d_cfg_b_G5_base','v09d_cfg_b_G5_B32'))):
        rows={row['id']:row for row in read(root/run/'cases.json')}
        setups={row['case']:row['setup'] for row in read(root/run/'static_setup.json')}
        contexts.update({name:dict(row=rows[name],setup=setups[name]) for name in names})
    programs={};tensors={};cases=[];atom_count=0;descriptor_checks=0
    for name,context in sorted(contexts.items()):
        row,setup=context['row'],context['setup'];program=cfg_b_source_program(row,setup)
        programs[name]=program;kt=row['k']//64
        keys={key for groups in program['source_keys'] for group in groups for key in group}
        counts={key:{str(width):len(source_granules(key,width)) for width in (32,128)} for key in keys}
        for key in keys:
            for width in (32,128):
                if counts[key][str(width)]!=independent_count(key,width):raise ValueError('independent row-span count differs')
                descriptor_checks+=1
        tensor={str(width):[[[counts[key][str(width)] for key in group] for group in groups]
            for groups in program['source_keys']] for width in (32,128)}
        tensors[name]=dict(ktiles=kt,output_counts=program['output_counts'],counts=tensor)
        histogram=Counter(tuple(counts[key]['128'] for key in group)
            for groups in program['source_keys'] for group in groups)
        width32=Counter(tuple(counts[key]['32'] for key in group)
            for groups in program['source_keys'] for group in groups)
        if set(width32)!={(512,256,256)}:
            raise ValueError('32B work word differs from the stated full-tile result')
        expected128=(128,64,64) if row['ldb']*2%128==0 else (128,112,112)
        if set(histogram)!={expected128}:
            raise ValueError('128B work word differs from the stated aligned/B32 result')
        demands=sum(len(groups)*3 for groups in program['source_keys']);atom_count+=demands
        totals=program['totals']
        if totals['source_box_fill_KiB'] or totals['destination_fill_KiB']:
            raise ValueError('full-tile destination ledger changed')
        if (totals['source_requests']!=demands or totals['source_payload_KiB']!=demands/3*32
                or totals['destination_valid_KiB']!=totals['source_payload_KiB']
                or totals['output_nominal_subtiles']!=8*sum(program['output_counts'])):
            raise ValueError('source/destination/D accounting changed')
        waves=[dict(j=j,active_ctas=sum(n>j for n in program['output_counts']),
            coverage=static_wave(program['source_keys'],j,kt,128)) for j in range(max(program['output_counts']))]
        # Existing wave results are reused as an independent archive comparison.
        archived=next((case for case in prior['cases'] if case['case']==name),None)
        if archived:
            for a,b in zip(waves,archived['software_waves']):
                if a['coverage']!=[v for v in b['coverage'] if v['width_bytes']==128]:
                    raise ValueError('previous static wave changed')
        cases.append(dict(case=name,gpu_uuid=setup['gpu_uuid'],shape=[row[k] for k in ('m','n','k')],
            lda=row['lda'],ldb=row['ldb'],ldd=row['ldd'],ktiles=kt,output_counts=program['output_counts'],
            logical_demands=demands,distinct_descriptors=len(keys),source_identity_sha256=digest(program['source_keys']),
            full_count_tensor_sha256=digest(tensors[name]),
            per_K_128_unit_words=[dict(A=a,B0=b,B1=c,groups=n) for (a,b,c),n in sorted(histogram.items())],
            per_K_32_unit_words=[dict(A=a,B0=b,B1=c,groups=n) for (a,b,c),n in sorted(width32.items())],
            source_destination_D_ledger=totals,software_ordinal_waves=waves,
            timing_support='Static only here; same-card responses below are reused separately, no price or time predicted.'))
        source_granules.cache_clear()
    raster=[]
    for pitch in (0,32):
        m,n=(f'r10_raster_along_{axis}_p{pitch}' for axis in ('m','n'))
        if tensors[m]!=tensors[n]:raise ValueError('raster count tensors differ; no feature-collision claim')
        raster.append(dict(pitch_bytes=pitch,full_count_tensor_equal=True,tensor_sha256=digest(tensors[m]),
            source_identity_equal=programs[m]['source_keys']==programs[n]['source_keys'],
            scope='Includes every CTA, output, K group and atom, output counts and K count; excludes source identities and measured issue times.'))
    local=[]
    for pitch in (0,32):
        a,b=(f'r10_source_{size}_p{pitch}' for size in ('single36','single126'))
        pa,pb=programs[a],programs[b]
        local.append(dict(pitch_bytes=pitch,ctas=36,
            first_output_source_keys_equal=pa['source_keys']==pb['source_keys'][:36],
            current_count_tensors_equal=tensors[a]['counts']=={w:x[:36] for w,x in tensors[b]['counts'].items()},
            full_grid_counts=[len(pa['output_counts']),len(pb['output_counts'])],
            scope='Same first 36 local programs; other CTA requests and global source state are different.'))
    if not all(r['first_output_source_keys_equal'] and r['current_count_tensors_equal'] for r in local):
        raise ValueError('first36 collision not proven')
    historical_work=[]
    for pitch in (0,32):
        a,b=f'cfg_b_bcurve_p{pitch}',f'r10_source_persistent132_p{pitch}'
        if tensors[a]!=tensors[b]:raise ValueError('old/current work tensor collision not proven')
        historical_work.append(dict(old_case=a,current_case=b,full_count_tensor_equal=True,
            tensor_sha256=digest(tensors[a]),source_identity_equal=programs[a]['source_keys']==programs[b]['source_keys'],
            scope='Static work-definition match only; GPU/card and initial-state evidence are not pooled.'))
    observed=[]
    for group in clock['groups']:
        if (group['pair'] in ('along_m','along_n') and group['j']==0) or group['pair'] in ('single36','single126'):
            observed.append(dict(pair=group['pair'],j=group['j'],T=group['T'],
                ns=group['observed']['L_ns_ratio'],native_cycles=group['observed']['L_cycles_ratio']))
    historical=[dict(cohort=c['cohort'],control=c['control'],variant=c['variant'],
        phases={phase:{name:values[name] for name in ('L_ns_ratio','L_cycles_ratio')}
            for phase,values in c['phases'].items()}) for c in old['comparisons']
        if c['variant'] in ('cfg_b_bcurve_p32','v09d_cfg_b_G5_B32')]
    unsupported=[dict(case=row['id'],shape=[row[k] for k in ('m','n','k')],
        remainders=[row['m']%128,row['n']%128,row['k']%64],
        status='Unsupported by cfg_b_source_program: boundary/OOB output and partial K source/fill need a separate graph.') for row in legacy['cases']]
    data=gzip.compress(json.dumps(tensors,separators=(',',':')).encode(),mtime=0)
    (output/'B-request-work-tensors.json.gz').write_bytes(data)
    for name in ('analyze_r10_request_work_audit.py','analyze_r10_granule_overlap.py','supply_model.py'):
        sha(Path(__file__).with_name(name))
    report=dict(scope='Static request-work identifiability audit; no new candidate, API, service, fit, score or GPU.',
        cases=cases,logical_demands_checked=atom_count,independent_descriptor_width_checks=descriptor_checks,
        tensor_sha256=hashlib.sha256(data).hexdigest(),raster_full_program_collisions=raster,
        first36_local_program_collisions=local,historical_full_work_collisions=historical_work,
        current_observed_responses=observed,historical_observed_responses=historical,
        unsupported_original_R10=unsupported,reused_data_inputs_rehashed=len(rehashed),input_sha256=inputs,
        conclusions=['For these full tiles, aligned per-K 128B count word is (128,64,64), B32 is (128,112,112); 32B word remains (512,256,256).',
            'Nominal source/destination payload remains 32KiB per group; per-demand 128B coverage is 32/44KiB, not source-union traffic.',
            'The full work-only tensors are identical between raster directions at each pitch. A deterministic model using only those tensors and equal initial state cannot distinguish their responses.',
            'This does not exclude a full graph using source identity, issue order, shared state or arbitration; those are explicitly absent from the work-only collision.',
            'The first36 local source programs match between single36 and single126 at each pitch; a local-address-only rule misses the differing global demand.',
            'Old Bcurve and G5 share the same per-group words, but cards, K, output counts and source relations differ; timing cannot be pooled.',
            'The original tail geometry remains unsupported; coverage-only accounting must not silently remove zero-fill or partial output work.'],
        next_mechanism_boundary='Before adding a candidate, specify how source identities and the other CTA requests change request formation/queue state, '
            'give one discriminating prediction using existing controls, and retain source, destination, D and observation boundaries. '
            'Static coverage alone does not identify a hardware request count or its service location.')
    (output/'B-request-work-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 请求工作向量：静态可辨识性检查','',
        '只复核静态全程序工作与既有观察；不增加候选、不评分、不选价。','',
        '| 配对 / 首输出T | ns响应 | native cycle响应 |','|---|---:|---:|']
    for row in observed:lines.append(f'| {row["pair"]} / {row["T"]} | {100*row["ns"]["median"]:.3f}% | {100*row["native_cycles"]["median"]:.3f}% |')
    lines+=['','Aligned每K组128B计数为(128,64,64)，B32为(128,112,112)；两方向全程序计数张量分别相同。',
        'single36与single126前36个CTA的完整局部源程序相同，整体请求群不同。',
        '这排除仅以相同工作张量/相同初态生成不同方向响应的规则，不排除带源身份和共享状态的完整图。',
        '原始2600×3000×2000的边界/OOB与部分K仍不在当前完整tile源程序适用域。','']
    (output/'B-request-work-audit.md').write_text('\n'.join(lines))
    print(json.dumps(dict(cases=len(cases),logical_demands=atom_count,descriptor_width_checks=descriptor_checks,
        raster_collisions=raster,local_collisions=local,rehashed=len(rehashed)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.data_root.resolve(),args.output.resolve())
