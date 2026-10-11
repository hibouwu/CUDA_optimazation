#!/usr/bin/env python3
"""Unpriced valid/fill completion ports for bound cfg_b logical destination boxes."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from supply_model import input_slot_state


PTX='https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html'


def completion_fragment(output,group,stages):
    """Logical completion conjunction, without prices or inferred physical queues.

    Valid-source availability and valid/fill destination completion are unpriced
    obligations, not independently sampled hardware events. A zero-source atom
    omits valid-data obligations but keeps submission/fill/destination completion.
    Equal source keys do not merge nodes or choose physical source ownership.
    """
    c,j=output['cta'],output['output_ordinal'];q=group['logical_k_group']
    if any(type(v) is not int or v<0 for v in (c,j,q)):
        raise ValueError('bound CTA/output/K-group required')
    g=output['global_group_binding']['first']+q;state=input_slot_state(g,stages)
    if group['slot_state']!=state or output['consumer_warp_group']!=j%2:
        raise ValueError('persistent ring/WG binding differs')
    if [a['atom'] for a in group['atoms']]!=['A','B0','B1']:
        raise ValueError('A/B0/B1 logical destination boxes required')
    prefix=f'c{c}.g{g}.';nodes={prefix+'empty_acquired':[]}
    nodes[prefix+'arrive_expect_tx']=[prefix+'empty_acquired']
    nodes[prefix+'acquire_return']=[prefix+'arrive_expect_tx']
    atoms=[];unpriced=[];full=[]
    for a,size in zip(group['atoms'],(16384,8192,8192)):
        name=a['atom'];valid=a['source_valid_bytes'];fill=a['destination_fill_bytes']
        if (type(valid) is not int or type(fill) is not int or min(valid,fill)<0
                or valid+fill!=size or a['transaction_bytes']!=size
                or a['destination_valid_bytes']!=valid or not a['destination_required']):
            raise ValueError('valid/fill/full destination obligation differs')
        issue=prefix+name+'.submit';dest=prefix+name+'.destination_complete'
        nodes[issue]=[prefix+'acquire_return' if name=='A' else prefix+'A.submit']
        prerequisites=[];source_ready=None;valid_done=None;fill_done=None
        if valid:
            source_ready=prefix+name+'.valid_available';valid_done=prefix+name+'.valid_delivered'
            nodes[source_ready]=[issue];nodes[valid_done]=[source_ready]
            prerequisites.append(valid_done);unpriced.extend((source_ready,valid_done))
        if fill:
            fill_done=prefix+name+'.fill_delivered';nodes[fill_done]=[issue]
            prerequisites.append(fill_done);unpriced.append(fill_done)
        nodes[dest]=prerequisites;full.append(dest)
        atoms.append(dict(atom=name,source_key=a['source_key'],submit=issue,
            source_valid_bytes=valid,destination_valid_bytes=valid,destination_fill_bytes=fill,
            transaction_bytes=size,valid_available=source_ready,valid_completion=valid_done,
            fill_completion=fill_done,destination_completion=dest))
    if group['transaction_expected_bytes']!=32768:
        raise ValueError('full software transaction budget differs')
    full_port=prefix+'full';nodes[full_port]=[prefix+'arrive_expect_tx']+full
    return dict(cta=c,output_ordinal=j,consumer_wg=j%2,logical_k_group=q,ring_state=state,
        transaction_expected_bytes=32768,nodes=nodes,atoms=atoms,
        external_empty_acquired_port=prefix+'empty_acquired',full_output_port=full_port,
        unpriced_completion_ports=unpriced,completion_values=None,
        scope='Open logical completion obligations only; no physical request count, queue, source ownership or time prediction.')


def work_identity_projection(fragment,identities):
    """Keep source equality classes but remove actual addresses/strides, without merging."""
    atoms=[]
    for atom in fragment['atoms']:
        key=json.dumps(atom['source_key'],separators=(',',':'))
        label=identities.setdefault(key,len(identities))
        projected=dict(atom);projected.pop('source_key')
        projected['source_identity_class']=label;atoms.append(projected)
    return dict(fragment,atoms=atoms)


def analyze(run,binding,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        value=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=value;return value
    def read(path):sha(path);return json.loads(path.read_text())
    manifest=read(run/'source_hashes.json');bound_report=read(binding/'B-boundary-binding.json')
    contracts={
        'cutlass/include/cute/atom/copy_traits_sm90_tma.hpp':['CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE'],
        'cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp':[
            'TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK',
            'pipeline.producer_acquire(smem_pipe_write)',
            'copy(mainloop_params.tma_load_a.with','copy(mainloop_params.tma_load_b.with',
            'pipeline.consumer_wait(smem_pipe_read, barrier_token)'],
        'cutlass/include/cutlass/pipeline/sm90_pipeline.hpp':[
            'empty_barrier_ptr_[stage].wait(phase)',
            'full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes)',
            'full_barrier_ptr_[stage].wait(phase)'],
        'cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp':[
            'mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes']}
    sites={}
    for relative,needles in contracts.items():
        path=run/'source'/relative
        if sha(path)!=manifest[relative]:raise ValueError('archived source SHA differs: '+relative)
        lines=path.read_text().splitlines();sites[relative]=[]
        for needle in needles:
            matches=[i+1 for i,line in enumerate(lines) if needle in line]
            if not matches:raise ValueError('archived source contract changed: '+needle)
            sites[relative].append(dict(pattern=needle,lines=matches))
    cases=[]
    for old in bound_report['cases']:
        path=binding/old['ledger_file']
        if sha(path)!=old['ledger_sha256']:raise ValueError('bound ledger SHA differs')
        bound=json.loads(gzip.decompress(path.read_bytes()));fragments=[];histogram=Counter();examples=[]
        counts=Counter();valid_only_deficit=0;drop_zero_deficit=0;identities={};projection=hashlib.sha256()
        for out in bound['outputs']:
            for group in out['groups']:
                fragment=completion_fragment(out,group,bound['scheduler']['stages'])
                projected=work_identity_projection(fragment,identities)
                projection.update(json.dumps(projected,sort_keys=True,separators=(',',':')).encode()+b'\n')
                # All edges are internal except the one explicit empty-acquired root.
                nodes=fragment['nodes'];seen=set()
                for name,predecessors in nodes.items():
                    if any(p not in seen for p in predecessors):raise ValueError('fragment order is not a DAG')
                    seen.add(name)
                atoms=fragment['atoms'];valid=sum(a['source_valid_bytes'] for a in atoms)
                fill=sum(a['destination_fill_bytes'] for a in atoms)
                if valid+fill!=32768 or len(nodes[fragment['full_output_port']])!=4:
                    raise ValueError('full conjunction lost an atom or transaction bytes')
                zero=[a for a in atoms if a['source_valid_bytes']==0]
                for a in zero:
                    if (a['valid_available'] is not None or a['valid_completion'] is not None
                            or nodes[a['destination_completion']]!=[a['fill_completion']]
                            or nodes[a['fill_completion']]!=[a['submit']]):
                        raise ValueError('zero-source transfer lost fill or was made immediately ready')
                histogram[tuple((a['source_valid_bytes'],a['destination_fill_bytes']) for a in atoms)]+=1
                counts.update(groups=1,logical_destination_atoms=3,
                    valid_source_obligations=sum(a['source_valid_bytes']>0 for a in atoms),
                    fill_obligations=sum(a['destination_fill_bytes']>0 for a in atoms),zero_source_atoms=len(zero),
                    source_valid_bytes=valid,destination_fill_bytes=fill,destination_transaction_bytes=32768,
                    valid_only_underbudget_groups=fill>0,drop_zero_underbudget_groups=bool(zero),
                    nodes=len(nodes),edges=sum(map(len,nodes.values())))
                valid_only_deficit+=fill;drop_zero_deficit+=sum(a['transaction_bytes'] for a in zero)
                if out['coord'] in ([12,12],[20,23]) and group['logical_k_group']==31:
                    examples.append(fragment)
                fragments.append(fragment)
        total=bound['totals']
        for field,previous in (('source_valid_bytes','source_valid_bytes'),('destination_fill_bytes','destination_fill_bytes'),
                ('destination_transaction_bytes','transaction_expected_bytes'),('zero_source_atoms','zero_source_atoms')):
            if counts[field]!=total[previous]:raise ValueError('completion port accounting differs from bound ledger')
        artifact=dict(schema='gh200-cfg-b-boundary-completion-ports/1',shape=bound['shape'],scheduler=bound['scheduler'],
            fragments=fragments,outputs=[dict(cta=o['cta'],output_ordinal=o['output_ordinal'],coord=o['coord'],
                D_boxes=o['D_boxes'],D_completion_bound=False) for o in bound['outputs']],totals=dict(counts),
            limits=['All completion ports are unpriced; missing values are not zero latency.',
                'Valid and fill ports are logical conjunction components, not distinct hardware queues or independently observed events.',
                'Each demand retains its own nodes even when source identities are equal; no implicit coalescing or owner selection.',
                'Logical A submit precedes B submits; B0/B1 relative order is not asserted. Atom counts are not hardware instruction counts.',
                'External empty-acquired ports still need the owning WG128-arrival and producer-order graph.',
                'Full output ports still need consumer retirement/release, producer tail and C/D lifecycle composition.'])
        data=gzip.compress(json.dumps(artifact,separators=(',',':')).encode(),mtime=0)
        name=old['case']+'-completion-ports.json.gz';(output/name).write_bytes(data)
        cases.append(dict(case=old['case'],totals=dict(counts),port_file=name,port_sha256=hashlib.sha256(data).hexdigest(),
            work_and_identity_topology_sha256=projection.hexdigest(),distinct_logical_source_identities=len(identities),
            atom_valid_fill_classes=[dict(atom_valid_fill_bytes=pattern,K_groups=count) for pattern,count in sorted(histogram.items())],
            zero_source_examples=examples,
            rejected_translations=dict(valid_bytes_as_full_budget=dict(groups=counts['valid_only_underbudget_groups'],deficit_bytes=valid_only_deficit),
                dropping_zero_source_destination=dict(groups=counts['drop_zero_underbudget_groups'],deficit_bytes=drop_zero_deficit))))
    for name in ('analyze_r10_boundary_completion.py','test_r13_boundary_completion.py','supply_model.py'):sha(Path(__file__).with_name(name))
    report=dict(scope='Unpriced cfg_b logical completion interface, preserving source/valid/fill/full obligations; no timing replay.',
        cases=cases,archived_source_sites=sites,source_manifest_matches=len(sites),input_sha256=inputs,
        address_erased_equivalence=dict(
            equal_work_identity_topology=len({c['work_and_identity_topology_sha256'] for c in cases})==1,
            projection='All CTA/output/K/ring/full nodes, valid/fill obligations and source-equality classes; actual source addresses/strides removed.',
            implication='With equal parameters and initial state, a model reading only this projection cannot distinguish these pitches.',
            boundary='Address-derived work/state, source ownership, sharing, arbitration and additional mechanisms remain possible; no timing family was fitted or tested.'),
        unresolved=['The fragments do not close empty arrivals, input safe exit, consumer order or D completion.',
            'No physical source queue, fill queue, ownership, sharing, arbitration or service time is selected.',
            'Old full-tile guards remain intact and old geometry remains unsupported by the timing API.',
            'This is not a new timing candidate, calibration, GPU observation or independent frozen holdout.'],
        external_sources=[dict(url=PTX,sections=['5.5.3.3','9.7.9.25.1.2','9.7.9.25.5.2'],
            use='Tensor OOB zero fill; full mbarrier completion observes the asynchronous tensor operation. Software expected bytes pinned separately to archived CUTLASS.')])
    (output/'B-boundary-completion.json').write_text(json.dumps(report,indent=2)+'\n')
    t=cases[0]['totals'];lines=['# 原始R10零源与目的完成端口','',
        '每组full合流保留arrive_expect_tx与A/B0/B1三个逻辑目的完成义务。valid/fill只划分逻辑完成条件，不假设独立硬件队列。',
        f'每条件{t["logical_destination_atoms"]}个目的盒，{t["valid_source_obligations"]}个非空有效源义务，{t["fill_obligations"]}个fill义务。',
        '672个零源B1没有valid分支，但保留submit→8192B fill完成→destination完成→full合流，完成时间未定。',
        '按有效源量缩减full预算会在1868组少计26882304B；删除零源目的盒会在672组少计5505024B。',
        '抹去地址/stride后，完整工作、ring/完成拓扑与源身份等价类在三种行距间完全一致；相同参数/初态且只读取这些量的模型不能区分行距。',
        '接口未闭合empty、consumer释放、producer tail或D生命周期；原时间图与guard不变。','']
    (output/'B-boundary-completion.md').write_text('\n'.join(lines))
    print(json.dumps(dict(cases=len(cases),source_manifest_matches=len(sites),per_case_totals=t),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    analyze(args.run.resolve(),args.binding.resolve(),args.output.resolve())
