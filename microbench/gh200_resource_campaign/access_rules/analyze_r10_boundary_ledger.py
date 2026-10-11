#!/usr/bin/env python3
"""Archive-bound logical source/fill/D accounting for the original R10 tails."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from analyze_r10_granule_overlap import source_granules
from supply_model import cfg_b_logical_boundary_ledger,input_slot_state


PTX_ROOT='https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html'


def brute_units(key,width):
    op,_,_,extents,strides,offset,origin,shape=key
    units=set();elements=0
    for a in range(shape[0]):
        for b in range(shape[1]):
            x,y=origin[0]+a,origin[1]+b
            if x<extents[0] and y<extents[1]:
                address=offset+x*strides[0]+y*strides[1]
                units.add(address//width);units.add((address+1)//width);elements+=1
    return units,elements


def analyze(run,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        value=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=value
        return value
    def read(path):sha(path);return json.loads(path.read_text())
    protocol=read(run/'protocol-formal.json');review=read(run/'reviews/ROOT-independent.json')
    snapshots={}
    archived=('probes/r10.cu','cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
        'cutlass/include/cutlass/pipeline/sm90_pipeline.hpp','cutlass/include/cute/atom/copy_traits_sm90_tma.hpp',
        'cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp')
    required=(('KernelTile = Shape<_128, _128, _64>','EpilogueTile = Shape<_64, _32>','KernelTmaWarpSpecializedPingpong',
         'DeviceBuffer<__half> a(size_t(o.m) * o.lda), b(size_t(o.k) * o.ldb)'),
        ('TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK',
         'pipeline.producer_acquire(smem_pipe_write)','copy(mainloop_params.tma_load_a.with','copy(mainloop_params.tma_load_b.with'),
        ('arrive_and_expect_tx(params_.transaction_bytes)',),('CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE',),
        ('auto k_tile_count = size<3>(gA_mkl)','mainloop_pipe_producer_state.advance(k_tile_count)',
         'mainloop_pipe_consumer_state.advance(k_tile_count * NumMmaWarpGroups)',
         'mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes'))
    for relative,needles in zip(archived,required):
        path=run/'source'/relative;sha(path);text=path.read_text();lines=text.splitlines()
        sites=[]
        for needle in needles:
            matches=[i+1 for i,line in enumerate(lines) if needle in line]
            if not matches:raise ValueError('archived source contract changed: '+needle)
            sites.append(dict(pattern=needle,lines=matches))
        snapshots[relative]=sites
    cases=[];record_checks=0;matched_review=0;brute_checks=[]
    for row in protocol['cases']:
        if row['config']!='cfg_b':raise ValueError('archive protocol no longer cfg_b only')
        ledger=cfg_b_logical_boundary_ledger(row);total=ledger['totals'];m,n,k=ledger['shape']
        mt,nt,kt=(m+127)//128,(n+127)//128,(k+63)//64
        expected=dict(source_valid_bytes=m*k*2*nt+n*k*2*mt,
            transaction_expected_bytes=mt*nt*kt*32768,distinct_valid_source_bytes=m*k*2+n*k*2,
            D_global_valid_bytes=m*n*4,D_nominal_bytes=mt*nt*65536)
        if any(total[name]!=number for name,number in expected.items()):raise ValueError('closed-form conservation differs')
        records=[]
        for mode in ('plain','trace'):
            for trial in range(protocol['processes_per_mode']):
                path=run/f'samples/formal/{row["id"]}/{mode}/{trial:02d}/result.json';record=read(path)
                if any(record[name]!=row[name] for name in ('config','m','n','k','lda','ldb','ldd')):
                    raise ValueError('recorded geometry differs')
                if (record['stages']!=6 or record['grid_ctas']!=132 or record['threads']!=384
                        or record['smem_bytes']!=215040 or record['check']['status']!='ok'):
                    raise ValueError('recorded cfg_b layout or previous numerical status changed')
                if str(path) in review['evidence']:
                    if inputs[str(path.resolve())]!=review['evidence'][str(path)]:raise ValueError('prior review hash differs')
                    matched_review+=1
                records.append(dict(mode=mode,trial=trial,stages=record['stages'],grid_ctas=record['grid_ctas'],
                    binary_sha256=record['binary_sha256'],previous_check_status=record['check']['status']))
                record_checks+=1
        keys={a['source_key'] for o in ledger['outputs'] for g in o['groups'] for a in g['atoms']}
        coverage={key:{w:len(source_granules(key,w))*w for w in (32,128)} for key in keys}
        cohort_units={(op,q):set() for op in ('A','B') for q in range(kt)}
        for key in keys:cohort_units[key[0],key[6][1]//64].update(source_granules(key,128))
        neighboring=[]
        for op in ('A','B'):
            for a,b in [(q,q+1) for q in range(kt-1)]+[(kt-1,0)]:
                neighboring.append(dict(operand=op,first_K_group=a,second_K_group=b,
                    shared_128_units=len(cohort_units[op,a]&cohort_units[op,b])))
        if m==2600 and row['lda']==2000:
            unaligned_rows=sum(2*r*row['lda']%128!=0 for r in range(m))
            if any(x['shared_128_units']!=(unaligned_rows if x['operand']=='A' else 0) for x in neighboring):
                raise ValueError('independent row-phase overlap count differs')
        histogram=Counter();coverage_totals={32:0,128:0};zero_atoms=0
        for out in ledger['outputs']:
            for group in out['groups']:
                if group['transaction_expected_bytes']!=32768:raise ValueError('partial K shrank the destination group')
                for a in group['atoms']:
                    key=a['source_key'];valid=a['source_valid_bytes'];fill=a['destination_fill_bytes']
                    if valid+fill!=a['transaction_bytes']:raise ValueError('atom valid/fill changed')
                    if not valid:
                        if coverage[key][32] or coverage[key][128] or not a['destination_required']:
                            raise ValueError('zero-source destination obligation was removed')
                        zero_atoms+=1
                    histogram[(a['atom'],tuple(a['valid_shape']),valid,fill,coverage[key][32],coverage[key][128])]+=1
                    for width in (32,128):coverage_totals[width]+=coverage[key][width]
        if zero_atoms!=total['zero_source_atoms']:raise ValueError('zero-source count differs')
        # Tiny independent element enumeration at one ordinary and one corner K tile.
        for mi,ni,q in ((0,0,0),(mt-1,nt-1,kt-1)):
            out=next(o for o in ledger['outputs'] if o['coord']==(mi,ni))
            for a in out['groups'][q]['atoms']:
                for width in (32,128):
                    direct,elements=brute_units(a['source_key'],width)
                    if direct!=set(source_granules(a['source_key'],width)) or elements*2!=a['source_valid_bytes']:
                        raise ValueError('independent clipped element enumeration differs')
                    brute_checks.append(dict(case=row['id'],coord=[mi,ni],q=q,atom=a['atom'],width=width,elements=elements))
        compressed=gzip.compress(json.dumps(ledger,separators=(',',':')).encode(),mtime=0)
        filename=row['id']+'-logical-ledger.json.gz';(output/filename).write_bytes(compressed)
        cases.append(dict(case=row['id'],row=row,totals=total,closed_form=expected,
            atom_classes=[dict(atom=a,valid_shape=shape,source_valid_bytes=v,destination_fill_bytes=f,
                coverage32_bytes=c32,coverage128_bytes=c128,logical_atoms=count)
                for (a,shape,v,f,c32,c128),count in sorted(histogram.items())],
            per_demand_coverage_bytes={str(w):v for w,v in coverage_totals.items()},
            allocation_bytes=dict(A=m*row['lda']*2,B=k*row['ldb']*2,D=m*row['ldd']*4),
            ordinary_full_K_128_unit_word=[coverage[a['source_key']][128]//128
                for a in ledger['outputs'][0]['groups'][0]['atoms']],
            neighboring_K_coverage_intersections=neighboring,
            last_corner=dict(coord=[mt-1,nt-1],logical_k_group=kt-1,
                atoms=ledger['outputs'][-1]['groups'][-1]['atoms'],D_boxes=ledger['outputs'][-1]['D_boxes']),
            records=records,ledger_file=filename,ledger_sha256=hashlib.sha256(compressed).hexdigest()))
        source_granules.cache_clear()
    for name in ('analyze_r10_boundary_ledger.py','supply_model.py','analyze_r10_granule_overlap.py','test_r13_boundary_ledger.py'):
        sha(Path(__file__).with_name(name))
    report=dict(scope='Original R10 cfg_b boundary work only; no persistent CTA binding, service or performance prediction.',
        cases=cases,archived_source_sites=snapshots,metadata_records_checked=record_checks,
        metadata_hashes_matching_prior_review=matched_review,independent_element_checks=brute_checks,
        prior_numerical_evidence=dict(records=review['records'],checked_values=review['checked_values'],
            use='Saved review status after SHA checks; no new matrix numerical replay.'),
        stage_binding_template=dict(global_group='output_ordinal*ktiles+logical_k_group',
            example_last_of_first_output=input_slot_state(31,6),
            example_first_of_next_output=input_slot_state(32,6),
            use='Unbound illustrative ordinal only; archived producer/consumer states advance and do not reset per output.'),
        external_sources=[dict(url=PTX_ROOT+'#tensor-tiled-mode-oob-access',
            use='PTX 8.8 zero-fill tensor bounding-box mode; archived CUTLASS selects ordinary fill.'),
            dict(url=PTX_ROOT+'#data-movement-and-conversion-instructions-cp-async-bulk-tensor',
                use='Tensor copy mbarrier completion; software expected bytes are separately pinned to archived SMEM layout.')],
        input_sha256=inputs,
        limits=['A/B0/B1 are logical box accounting; atom count does not assert the old binary hardware TMA instruction count.',
            'Every input group keeps 32768 expected destination bytes, including all-zero source atoms and the partial K group.',
            'D nominal source-box bytes do not prove that all OOB store boxes issue or read that work; global valid/OOB bytes are separate.',
            'Logical K index does not reset a real ring. CTA, output ordinal and global stage binding remain explicitly None.',
            'Source identities/validity/fill are static; in-flight merging, ownership, cache traffic, readiness and safe release are unpriced.',
            'Neighboring K coverage overlap is geometric; it does not prove in-flight sharing or cache residency at actual issue times.',
            'The old full-tile API retains its guards; this new boundary ledger does not make the old geometry supported by its timing graph.',
            'The archived probe allocates B as K*ldb; allocation capacity changes with pitch and is not a fixed-capacity new-card control.',
            'No timing fit, new candidate, GPU/numerical replay, observation qualification or independent frozen holdout.'])
    (output/'B-boundary-ledger.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 原始R10尾部：有效源、目的填充与D范围','',
        '| case | 源有效B | 目的fill B | 目的事务B | D有效B | 全零源盒 |','|---|---:|---:|---:|---:|---:|']
    for c in cases:
        t=c['totals'];lines.append(f'| {c["case"]} | {t["source_valid_bytes"]} | {t["destination_fill_bytes"]} | '
            f'{t["transaction_expected_bytes"]} | {t["D_global_valid_bytes"]} | {t["zero_source_atoms"]} |')
    lines+=['','每条件504个逻辑输出×32K组，每组目的事务32768B；末K组只有16个有效K元素。',
        '逻辑坐标没有被绑定到历史CTA/输出序号/全局stage；时间图仍不支持这组旧尾部。','']
    (output/'B-boundary-ledger.md').write_text('\n'.join(lines))
    print(json.dumps(dict(cases=len(cases),metadata_records=record_checks,prior_review_hash_matches=matched_review,
        per_case_totals=cases[0]['totals'],element_checks=len(brute_checks)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.run.resolve(),args.output.resolve())
