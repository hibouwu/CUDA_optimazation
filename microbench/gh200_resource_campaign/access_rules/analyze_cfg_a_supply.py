#!/usr/bin/env python3
"""Offline B source bindings plus C's cfg_a rank/output dependency graph.

Verify actual archived work, atom bytes and graph composition. No service fit,
target timestamp input, GPU operation or new independent validation claim.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

from analyze_r18 import replay
from supply_model import cfg_a_atomic_source_graph, cfg_a_source_bindings


RUNS = ('20261010-R18-cluster-fill-discrimination',
    '20261009-R13-R15-R18-composition-job738496',
    '20261009-R18-first-fill-job738707')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def save_compressed(path, value):
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0))


def audit_graph(original, graph, dependency_order):
    order, _ = dependency_order(graph)
    preserved = 0
    for name, predecessors in original['nodes'].items():
        if not set(predecessors).issubset(graph['nodes'][name]):
            raise ValueError('C dependency removed: '+name)
        preserved += len(predecessors)
    for field in ('groups', 'outputs', 'markers', 'transfers'):
        if graph[field] != original[field]:
            raise ValueError('C metadata changed: '+field)
    total = graph['k_tiles']*graph['output_tiles']
    for g in range(total):
        requests = [r for r in graph['input_requests'] if r['group']==g]
        deliveries = [d for d in graph['input_deliveries'] if d['group']==g]
        if len(requests)!=6 or sum(r['bytes'] for r in requests)!=48*1024:
            raise ValueError('source group is not six atoms/48KiB')
        for r in range(2):
            incoming = [d for d in deliveries if d['destination_rank']==r]
            if len(incoming)!=5 or sum(d['bytes'] for d in incoming)!=32768:
                raise ValueError('full destination is not A16 plus four B4 atoms')
            for d in incoming:
                aggregate = (f'r{r}.g{g}.A_visible' if d['atom']=='A'
                    else f"r{d['source_rank']}.g{g}.B_to_r{r}")
                if d['completion'] not in graph['nodes'][aggregate]:
                    raise ValueError('missing destination completion in aggregate join')
            expected = [f'r{r}.g{g-1}.B_issue'] if g else []
            if g>=graph['stages']:
                expected += [f'r{s}.w{w}.g{g-graph["stages"]}.empty_to_r{r}'
                    for s in range(2) for w in range(2)]
            if graph['nodes'][f'r{r}.g{g}.acquire']!=expected:
                raise ValueError('acquire lost a producer-order or remote-empty prerequisite')
    # A topology check, without treating synthetic event times as prices.
    ancestors, pending = set(), ['r0.g0.B0_delivered_r1']
    while pending:
        name = pending.pop()
        if name not in ancestors:
            ancestors.add(name)
            pending.extend(graph['nodes'][name])
    if 'r0.g0.B1_issue' in ancestors:
        raise ValueError('B0 source/delivery incorrectly waits for aggregate B issue')
    return dict(nodes=len(order), original_nodes=len(original['nodes']),
        original_edges_preserved=preserved, **graph['totals'],
        B0_progress_independent_of_B1_issue=True,
        release_arrival_and_output_nodes_preserved=True)


def audit_binding(binding, graphs):
    atoms, output_valid, output_nominal, subtile_count = {}, 0, 0, 0
    source_count, source_payload, destination_count, destination_payload = 0, 0, 0, 0
    kt = binding['k_tiles']
    for cluster in binding['clusters']:
        graph = graphs[binding['stages'], kt, cluster['output_tiles']]
        source_count += graph['totals']['source_requests']
        source_payload += graph['totals']['source_payload_bytes']/1024
        destination_count += graph['totals']['destination_atoms']
        destination_payload += graph['totals']['destination_payload_bytes']/1024
        for r, outputs in enumerate(cluster['rank_outputs']):
            for output in outputs:
                for atom in output['sources']:
                    key = atom['source_atom_id']
                    if key in atoms:
                        raise ValueError('source issuer atom occurs twice')
                    atoms[key] = dict(atom, physical_cluster=cluster['cluster'], rank=r,
                        j=output['j'])
                    # Check rectangle intersection independently at first/last Ktile.
                    for q in (0, kt-1):
                        origin = [atom['origin'][0], atom['origin'][1]+64*q]
                        rows = max(0, min(atom['shape'][0], atom['extents'][0]-origin[0]))
                        cols = max(0, min(atom['shape'][1], atom['extents'][1]-origin[1]))
                        if rows*cols*2/1024 != atom['valid_KiB']:
                            raise ValueError('descriptor intersection differs from valid source')
                    if atom['payload_KiB']!=atom['valid_KiB']+atom['fill_KiB']:
                        raise ValueError('source payload does not conserve valid/fill')
                if output['output']['nominal_subtiles']!=4 or output['output']['subtile_shape']!=[128,32]:
                    raise ValueError('cfg_a output lattice differs from C four-subtile graph')
                output_valid += output['output']['valid_KiB']
                output_nominal += output['output']['nominal_KiB']
                subtile_count += output['output']['nominal_subtiles']
    for cluster in binding['clusters']:
        for r, outputs in enumerate(cluster['rank_outputs']):
            for output in outputs:
                for op, target in output['destination'].items():
                    incoming = [atoms[i] for i in target['source_atoms']]
                    expected_count = 1 if op=='A' else 4
                    if len(incoming)!=expected_count or sum(a['payload_KiB'] for a in incoming)!=16:
                        raise ValueError('destination operand atom count/payload differs')
                    for a in incoming:
                        if (a['physical_cluster']!=cluster['cluster'] or a['j']!=output['j']
                                or r not in a['destination_ranks'] or a['operand']!=op):
                            raise ValueError('destination atom references another physical group')
                    if (sum(a['valid_KiB'] for a in incoming)!=target['valid_KiB']
                            or sum(a['fill_KiB'] for a in incoming)!=target['fill_KiB']):
                        raise ValueError('destination valid/fill changed in graph binding')
    totals = binding['totals']
    actual = dict(source_requests=source_count, source_payload_KiB=source_payload,
        destination_atoms=destination_count, output_valid_KiB=output_valid,
        output_nominal_KiB=output_nominal, output_nominal_subtiles=subtile_count)
    for name, value in actual.items():
        if value!=totals[name]:
            raise ValueError('graph/ledger total differs: '+name)
    if destination_payload!=totals['destination_valid_KiB']+totals['destination_fill_KiB']:
        raise ValueError('graph nominal destination differs from valid plus fill')
    return dict(source_atom_templates=len(atoms), source_requests=source_count,
        source_payload_KiB=source_payload, destination_atoms=destination_count,
        destination_payload_KiB=destination_payload, output_nominal_subtiles=subtile_count,
        physical_clusters=len(binding['clusters']), idle_clusters=len(binding['idle_clusters']),
        output_counts=dict(sorted(Counter(c['output_tiles'] for c in binding['clusters']).items())))


def analyze(root, c_model_path, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    def fingerprint(path):
        path = path.resolve()
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in inputs and inputs[str(path)]!=value:
            raise ValueError('input changed during audit: '+str(path))
        inputs[str(path)] = value
        return value
    fingerprint(c_model_path)
    spec = importlib.util.spec_from_file_location('cfg_a_C_dependency_input', c_model_path)
    c_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c_model)
    for name in ('supply_model.py', 'analyze_cfg_a_supply.py', 'analyze_r18.py',
            'v08_model.py', 'v06_model.py', 'v06_run.py'):
        fingerprint(Path(__file__).with_name(name))
    graphs, graph_checks, cases, bindings = {}, [], [], []
    processes, ctas = 0, 0
    for run_name in RUNS:
        run = root/run_name
        fingerprint(run/'cases.json')
        fingerprint(run/'static_setup.json')
        for source in ('probes/r18.cu',
                'overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
                'overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp'):
            fingerprint(run/'source'/source)
        rows = read(run/'cases.json')
        setups = {v['case']:v['setup'] for v in read(run/'static_setup.json')}
        for row in rows:
            if row['config']!='cfg_a':
                continue
            binding = cfg_a_source_bindings(row, setups[row['id']])
            for cluster in binding['clusters']:
                key = binding['stages'], binding['k_tiles'], cluster['output_tiles']
                if key not in graphs:
                    original = c_model.cfg_a_rank_dependencies(*key)
                    graph = cfg_a_atomic_source_graph(original)
                    check = audit_graph(original, graph, c_model.dependency_order)
                    name = f'cfg_a-S{key[0]}-K{key[1]}-T{key[2]}.json.gz'
                    save_compressed(output/name, graph)
                    graph_checks.append(dict(stages=key[0],k_tiles=key[1],output_tiles=key[2],
                        artifact=name, **check))
                    graphs[key] = graph
            check = audit_binding(binding, graphs)
            expected_work = {}
            for cluster in binding['clusters']:
                for r, outputs in enumerate(cluster['rank_outputs']):
                    expected_work[cluster['ctas'][r]] = [tuple(o['coord']) for o in outputs]
            for cluster in binding['idle_clusters']:
                expected_work.update({cta:[] for cta in cluster['ctas']})
            records = sorted((run/'samples'/row['id']).glob('dual-*.json'))
            if len(records)!=10:
                raise ValueError('expected ten archived dual records: '+row['id'])
            for path in records:
                record = read(path)
                fingerprint(path)
                raw_sha = fingerprint(run/record['raw'])
                if (record['case']!=row['id'] or record['variant']!='dual'
                        or record['returncode'] or raw_sha!=record['raw_sha256']):
                    raise ValueError('archived process identity differs')
                observed = replay(run, record, row, retain_roles=True)
                if {a['cta']:a['work'] for a in observed['ctas']}!=expected_work:
                    raise ValueError('actual CTA coordinates differ from physical source bindings')
                for actor in observed['ctas']:
                    if actor['role_work']!=[expected_work[actor['cta']]]*2:
                        raise ValueError('both cooperative WGs do not share output work')
                processes += 1
                ctas += len(observed['ctas'])
            cases.append(dict(run=run_name, case=row['id'], stages=binding['stages'],
                k_tiles=binding['k_tiles'], archived_processes=len(records),
                gpu_uuid=setups[row['id']]['gpu_uuid'], **check))
            bindings.append(dict(run=run_name, case=row['id'], binding=binding))
    fingerprint(c_model_path)
    save_compressed(output/'B-cfg-a-source-bindings.json.gz', bindings)
    report = dict(schema='gh200-cfg-a-source-composition-audit/1', cases=cases,
        graphs=graph_checks, static_cases=len(cases), graph_templates=len(graphs),
        archived_processes=processes, archived_CTAs=ctas,
        status='Structure and archived work binding passed; no timing price or new independent holdout.',
        unresolved=['source-service and destination-visible prices/arbitration',
            'compiled relative issue order of B0 and B1; refinement is a partial order',
            'thread issue to all required empty arrivals',
            'first/later local windows, interventions and spatial timing',
            'cfg_c cooperative topology; cfg_b graph remains separate'],
        limits=['Source/destination payload and coverage are logical work, not physical traffic.',
            'C B_issue/B_to_r are joins after two 4KiB atoms and must not charge transport again.',
            'No physical cluster/CTA/SM label price or target timestamp is a prediction input.',
            'Arbitrary scheduler mappings crossing physical cluster/rank/ordinal are rejected.',
            'Idle clusters are listed without assigning lifecycle time.'],
        input_sha256=inputs)
    save(output/'B-cfg-a-source-composition.json', report)
    rows = ['# cfg_a 源切片与跨 rank 依赖组合复核', '',
        f'离线复核 {len(cases)} 个已有条件、{processes} 个归档 dual 进程、{ctas} 个 CTA，',
        f'覆盖 {len(graphs)} 种完整图模板。结构与实际工作绑定通过，尚无时间价格或新独立留出。', '',
        '每个 K-group 两 rank 共发 A16×2、B4×4，共六个源请求/48 KiB；',
        '每目的 full 依赖本地 producer arrival、本地 A16、两 issuer 各两个 B4，',
        '共五个目的完成事件/32 KiB。两 rank 共十次目的投递/64 KiB。', '',
        '`B_issue` 及 `B_to_r` 保留为汇合节点；B0 可在 B1 发出前推进。',
        '本次没有以 SASS 确认 B0/B1 的相对发起次序，两个节点只保留 A 之后的偏序。',
        'C 的全部旧边、四个 empty arrival 前驱、release/output 节点及标记原样保留。',
        '物理 cluster 绑定使用真实 block 坐标；每 group 对模板 K 坐标加 64×local_K，',
        '全局 group=输出序号×Ktiles+local_K，stage/phase 不在输出边界重置。', '',
        '| 归档 | 条件数 | dual 进程 |', '|---|---:|---:|']
    for run_name in RUNS:
        selected = [r for r in cases if r['run']==run_name]
        rows.append(f'| {run_name} | {len(selected)} | {sum(r["archived_processes"] for r in selected)} |')
    rows += ['', '部分有效输入、OOB fill、显式零地址图分别保留，D 有效面积与四个名义子 tile 守恒。',
        '本复核不读取目标时间用于生成图，不解释物理带宽；先前 V09 与 R10 失败结论不变。',
        '下一步仍需识别源/目的服务与释放可见性的局部规则，再冻结新的局部留出。', '']
    (output/'B-cfg-a-source-composition.md').write_text('\n'.join(rows))
    save(output/'B-artifact-sha256.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(dict(output=str(output), cases=len(cases), graphs=len(graphs),
        archived_processes=processes, archived_CTAs=ctas)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--c-model', type=Path, required=True, help='explicit read-only C output_model.py input')
    parser.add_argument('--output', type=Path, required=True, help='new exclusive B reanalysis directory')
    args = parser.parse_args()
    analyze(args.data_root.resolve(), args.c_model.resolve(), args.output.resolve())
