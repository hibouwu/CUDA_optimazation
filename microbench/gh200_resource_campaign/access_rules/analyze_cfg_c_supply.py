#!/usr/bin/env python3
"""cfg_c physical source/full ports and C's local output, without rank timing."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

from analyze_r18 import replay
from supply_model import cfg_a_source_bindings,cfg_c_source_bindings,cfg_c_source_port_graph


RUNS=('20261009-R13-R15-R18-composition-job738496',
    '20261009-R18-first-fill-job738865','20261009-V09-heldout-job739011')


def analyze(root,c_model_path,output):
    output.mkdir(parents=True,exist_ok=False)
    inputs={}
    def sha(path):
        path=path.resolve();value=hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in inputs and inputs[str(path)]!=value:raise ValueError('input changed: '+str(path))
        inputs[str(path)]=value
        return value
    def read(path):
        sha(path)
        return json.loads(path.read_text())
    def save_gzip(name,value):
        raw=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
        (output/name).write_bytes(gzip.compress(raw,mtime=0))
    sha(c_model_path)
    spec=importlib.util.spec_from_file_location('cfg_c_C_local_output',c_model_path)
    c_model=importlib.util.module_from_spec(spec);spec.loader.exec_module(c_model)
    old_binding_path=root/'20261010-R18-cluster-fill-discrimination/reanalysis/B-20261011-cfg-a-source-composition-v3/B-cfg-a-source-bindings.json.gz'
    sha(old_binding_path)
    previous=json.loads(gzip.open(old_binding_path,'rt').read());current=[]
    for old in previous:
        run=root/old['run']
        row=next(r for r in read(run/'cases.json') if r['id']==old['case'])
        setup=next(r['setup'] for r in read(run/'static_setup.json') if r['case']==old['case'])
        current.append(dict(run=old['run'],case=old['case'],binding=cfg_a_source_bindings(row,setup)))
    encoded=gzip.compress(json.dumps(current,separators=(',',':'),allow_nan=False).encode(),mtime=0)
    if encoded!=old_binding_path.read_bytes():raise ValueError('cfg_a canonical binding bytes changed')
    templates={};output_templates={};bindings=[];cases=[];processes=ctas=0
    for run_name in RUNS:
        run=root/run_name
        rows=read(run/'cases.json');setups={v['case']:v['setup'] for v in read(run/'static_setup.json')}
        for name in ('probes/r18.cu','overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
                'overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp'):
            sha(run/'source'/name)
        for row in rows:
            if row['config']!='cfg_c':continue
            binding=cfg_c_source_bindings(row,setups[row['id']]);kt=binding['k_tiles']
            atoms={};expected_work={};valid_output=0;coverage=[]
            source_count=source_payload=destination_count=destination_payload=0
            for cluster in binding['clusters']:
                T=cluster['output_tiles'];key=(binding['stages'],kt,T)
                if key not in templates:
                    graph=cfg_c_source_port_graph(*key);order,_=c_model.dependency_order(graph)
                    total=kt*T
                    if (graph['totals']['source_requests']!=6*total
                            or graph['totals']['destination_atoms']!=8*total
                            or graph['totals']['source_payload_bytes']!=65536*total
                            or graph['totals']['destination_payload_bytes']!=98304*total
                            or len(graph['external_producer_ports'])!=4*total):
                        raise ValueError('cfg_c source-port work/roots changed')
                    for g in range(total):
                        for rank in range(2):
                            incoming=[d for d in graph['input_deliveries'] if d['group']==g and d['destination_rank']==rank]
                            if (len(incoming)!=4 or sum(d['bytes'] for d in incoming)!=49152
                                    or set(graph['nodes'][f'r{rank}.g{g}.full'][1:])!={d['completion'] for d in incoming}):
                                raise ValueError('full does not join A16+A16+B8+B8')
                    name=f'cfg_c-S{key[0]}-K{kt}-T{T}-source-ports.json.gz'
                    save_gzip(name,graph)
                    templates[key]=dict(graph=graph,artifact=name,nodes=len(order))
                if T not in output_templates:
                    local=c_model.cooperative_output_source_dependencies('cfg_c',T)
                    c_model.dependency_order(local)
                    if local['layout']['subtiles_per_output']!=8:raise ValueError('cfg_c output requires eight nominal subtiles')
                    name=f'cfg_c-T{T}-local-output.json.gz';save_gzip(name,local)
                    output_templates[T]=dict(graph=local,artifact=name)
                graph=templates[key]['graph']
                source_count+=graph['totals']['source_requests'];source_payload+=graph['totals']['source_payload_bytes']/1024
                destination_count+=graph['totals']['destination_atoms'];destination_payload+=graph['totals']['destination_payload_bytes']/1024
                for rank,outputs in enumerate(cluster['rank_outputs']):
                    expected_work[cluster['ctas'][rank]]=[tuple(o['coord']) for o in outputs]
                    for out in outputs:
                        for atom in out['sources']:
                            if atom['source_atom_id'] in atoms:raise ValueError('source atom issued twice')
                            atoms[atom['source_atom_id']]=(cluster['cluster'],rank,out['j'],atom)
                            for q in (0,kt-1):
                                origin=[atom['origin'][0],atom['origin'][1]+64*q]
                                extent=atom['extents'];shape=atom['shape']
                                valid=max(0,min(shape[0],extent[0]-origin[0]))*max(0,min(shape[1],extent[1]-origin[1]))*2/1024
                                if valid!=atom['valid_KiB'] or atom['payload_KiB']!=valid+atom['fill_KiB']:
                                    raise ValueError('source descriptor valid/fill differs')
                        local=c_model.cooperative_output_valid_subtiles('cfg_c',row['m'],row['n'],out['coord'],out['j'])
                        if (len(local['subtiles'])!=8 or local['valid_bytes']/1024!=out['output']['valid_KiB']
                                or out['output']['nominal_subtiles']!=8):raise ValueError('C output geometry differs from supply ledger')
                        guarantee=output_templates[T]['graph']['layout']['source_guarantees'][out['j']]
                        if guarantee['last_subtile']!=8*(out['j']+1)-1:raise ValueError('last nominal source group was skipped')
                        for name in ('guarantee_token','source_completion'):
                            if guarantee[name] not in output_templates[T]['graph']['nodes']:raise ValueError('C source guarantee node missing')
                        coverage.append(dict(cta=cluster['ctas'][rank],j=out['j'],coord=out['coord'],
                            valid_bytes=local['valid_bytes'],buffer_valid_bytes=local['buffer_valid_bytes'],
                            last_valid_subtile=local['last_valid_subtile'],last_nominal_subtile=guarantee['last_subtile'],
                            empty_descriptor_subtiles=sum(s['valid_bytes']==0 for s in local['subtiles'])))
                        valid_output+=out['output']['valid_KiB']
            for cluster in binding['clusters']:
                for rank,outputs in enumerate(cluster['rank_outputs']):
                    for out in outputs:
                        for operand,target in out['destination'].items():
                            incoming=[atoms[i] for i in target['source_atoms']]
                            if len(incoming)!=2 or sum(a[3]['payload_KiB'] for a in incoming)!=(32 if operand=='A' else 16):
                                raise ValueError('cfg_c destination requires two A16 or two local B8')
                            for coord,source,j,atom in incoming:
                                if (coord!=cluster['cluster'] or j!=out['j'] or rank not in atom['destination_ranks']
                                        or atom['operand']!=operand or (operand=='B' and source!=rank)):
                                    raise ValueError('source atom references another physical destination group')
                            if any(sum(a[3][field+'_KiB'] for a in incoming)!=target[field+'_KiB'] for field in ('valid','fill')):
                                raise ValueError('destination valid/fill does not conserve source slices')
            for cluster in binding['idle_clusters']:expected_work.update({cta:[] for cta in cluster['ctas']})
            totals=binding['totals']
            if (source_count!=totals['source_requests'] or source_payload!=totals['source_payload_KiB']
                    or destination_count!=totals['destination_atoms'] or valid_output!=totals['output_valid_KiB']
                    or destination_payload!=totals['destination_valid_KiB']+totals['destination_fill_KiB']):
                raise ValueError('source-port/ledger work differs')
            replayed=0
            if run_name!=RUNS[-1]:
                records=sorted((run/'samples'/row['id']).glob('dual-*.json'))
                if len(records)!=10:raise ValueError('expected ten original dual processes')
                for path in records:
                    record=read(path)
                    if record['returncode'] or sha(run/record['raw'])!=record['raw_sha256']:raise ValueError('raw identity/failure')
                    observed=replay(run,record,row,retain_roles=True)
                    if ({a['cta']:a['work'] for a in observed['ctas']}!=expected_work
                            or any(a['role_work']!=[expected_work[a['cta']]]*2 for a in observed['ctas'])):
                        raise ValueError('actual cooperative work differs from physical source partners')
                    processes+=1;ctas+=len(observed['ctas']);replayed+=1
            cases.append(dict(run=run_name,case=row['id'],physical_clusters=len(binding['clusters']),
                idle_clusters=len(binding['idle_clusters']),output_counts=dict(Counter(c['output_tiles'] for c in binding['clusters'])),
                actual_work_processes_replayed=replayed,source_requests=source_count,source_payload_KiB=source_payload,
                destination_atoms=destination_count,destination_payload_KiB=destination_payload,output_coverage=coverage))
            bindings.append(dict(run=run_name,case=row['id'],binding=binding))
    for name in ('analyze_cfg_c_supply.py','supply_model.py','analyze_r18.py','v08_model.py','v06_model.py','v06_run.py'):
        sha(Path(__file__).with_name(name))
    sha(c_model_path)
    save_gzip('B-cfg-c-source-bindings.json.gz',bindings)
    report=dict(scope='cfg_c static physical source/full ports and local C output audit; closed rank lifecycle remains missing.',
        cases=cases,archived_processes_replayed=processes,archived_ctas_replayed=ctas,
        checked_sample_values=processes*4096,cfg_a_binding_regression_cases=len(previous),cfg_a_binding_bytes_identical=True,
        source_templates=[dict(stages=k[0],k_tiles=k[1],output_tiles=k[2],artifact=v['artifact'],nodes=v['nodes']) for k,v in templates.items()],
        local_output_templates=[dict(output_tiles=k,artifact=v['artifact']) for k,v in output_templates.items()],
        input_sha256=inputs,unresolved=['C currently supplies cfg_a rank graph and cfg_c local output only.',
            'cfg_c empty routes, acquire predecessors, consumer safe release, finite-stage reuse and closed completion remain uncomposed.',
            'External producer ports are unresolved prerequisites, not zero-cost events or measured ready times.'],
        limits=['No source/destination/output service prices or observed clock inputs.',
            'V09 revealed cfg_c cases are static geometry only; old prediction, unsupported cases and failed score remain unchanged.',
            'Idle physical clusters receive no fabricated lifecycle; output-OOB ranks retain partner source obligations.',
            'Nominal output subtiles and source buffer generations remain even when destination valid bytes are zero.',
            'cfg_a empty routes/32KiB full tokens and cfg_b pingpong graph are not transferred to cfg_c.'])
    (output/'B-cfg-c-source-ports.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# cfg_c：物理源绑定与尚未闭合的full接口','',
        f'{len(cases)}静态条件；其中{processes}原双时钟进程、{ctas}个CTA和{processes*4096}个D抽样值重新复核。V09仅静态几何。',
        'cfg_a的22份规范化压缩绑定逐字节不变。','',
        '每rank A16KiB投给两rank；B0/B1各8KiB投给本地。每cluster每K-group六源atom/64KiB、八目的atom/96KiB。',
        '每rank full取本地producer arrival、两rank的A16及本地B0/B1；名义48KiB，valid/fill分开。',
        '输出OOB rank仍可能提供伙伴必需的A；C本地输出图保留每tile八个名义子tile和源buffer代次。','',
        '源接口图在arrive_expect_tx/acquire_return开放；未接入cfg_c的empty、消费安全释放或有限stage约束。',
        '这些根节点表示尚未给出的前驱，不可当作零成本完成或正式时间预测。C的cfg_a rank图未迁入。',
        '所有source/full模板、本地输出模板、静态绑定与逐输出有效面积已保存；没有新价格或GPU运行。','']
    (output/'B-cfg-c-source-ports.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),cases=len(cases),processes=processes,ctas=ctas,
        source_templates=len(templates),local_output_templates=len(output_templates))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--c-output-model',type=Path,required=True,help='read-only C model source')
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.c_output_model.resolve(),args.output.resolve())
