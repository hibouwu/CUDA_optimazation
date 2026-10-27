#!/usr/bin/env python3
"""R11两组完整数值回放；R12绑定阶段6快照；导出原条件联合窗口合同。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

from sm_joint_service import joint_service

R11 = Path('gh200_resource_campaign/access_rules/Q07-smem/R04-R11-mixed-issue/20261008-R11-job736989-v6')
R12 = Path('gh200_resource_campaign/access_rules/Q07-smem/R12-smem-path-contention/20261008-R12-job736989-v3')
SMEM = Path('gh200_resource_campaign/access_rules/Q07-smem/SMEM-01-replay/20261017-SMEM-01-replay-001/protocol-replay')
R11_FILES = {
    'source_hashes.json': '4c87d7628aec0db08911fad2928789970720311e0fec4d1b3e4e94d806318cb9',
    'cases.json': 'ef0b3c991fedf86e208957a40328a5829388a0f855e846b5aa2a5edd536a6b46',
    'analysis/rules.json': 'b52f0fbe177d7ad13babf0adf235f6ca4074b4a2559f19f2727fcb42959031a6',
    'analysis/checks.json': 'f95682470640e031e58cadcef71d1f548d5e6c551f2d40394a0abf32e3c7b663',
    'analysis/sass-check.json': 'c972329a91ea47b960578dd9a7b1f4b63cc0429048469ef8fcc1afc32a066cc4',
    'build/binary.json': 'bccb58fcb048f15ff7e927f24733aaa0436fc7f71171caa00acbff8383cac570',
}
SMEM_SUMMARY_SHA = '5a3d54b4a71e5c69c417772ece2862357faab1a0d1afb3fe761096a41c0ecbd6'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def replay_r11(root, output):
    output.mkdir()
    for name, checksum in R11_FILES.items():
        require(digest(root/name) == checksum, 'R11固定身份不符：'+name)
    sources = read(root/'source_hashes.json')
    for name, checksum in sources.items():
        require(digest(root/name) == checksum, 'R11冻结源码不符：'+name)
    binary = read(root/'build/binary.json')
    require(digest(root/'build/r11') == binary['sha256'] and
            digest(root/'build/sass.txt') == binary['sass_sha256'], 'R11二进制/SASS身份不符')
    spec = importlib.util.spec_from_file_location('frozen_joint_r11', root/'source/analyze_r11.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    facts = helper.sass_check(root)
    require(facts == read(root/'analysis/sass-check.json'), 'R11原SASS审查不一致')
    records, samples, checked = [], [], 0
    for path in sorted(root.glob('samples/*/*/result.json')):
        row = read(path)
        if row['pair'] not in (0, 2):
            continue
        require(row['returncode'] in (0, 3) and row['event_errors'] == 0 and row['local_bytes'] == 0,
                'R11旧进程或资源状态不符')
        checked += helper.replay(path.parent, row)
        records.append(row)
        samples.append(dict(path=path.relative_to(root).as_posix(), result_sha256=digest(path),
            case_id=row['case_id'], label=row['label'], pair=row['pair'], order=row['order'],
            b_role=row['b_role'], units=row['units'], elapsed_cycles=row['elapsed_cycles'],
            files=row['files']))
    require(len(records) == 72 and checked == 1486080, 'R11必须为72记录/1486080见证字')
    cases = [case for case in read(root/'cases.json') if case['pair'] in (0, 2)]
    conditions = []
    for case in cases:
        rows = [row for row in records if row['case_id'] == case['id']]
        formal = [row for row in rows if row['label'].startswith('formal-')]
        smoke = [row for row in rows if row['label'].startswith('smoke-')]
        require({row['label'] for row in formal} == {'formal-00','formal-01','formal-02'} and
                {row['units'] for row in smoke} == {1, 3} and len(smoke) == 2 and
                all(row['units'] == 256 and row['order'] == case['order'] and row['b_role'] == 0
                    for row in formal), 'R11正式/奇数短见证不齐')
        times = [row['elapsed_cycles'] for row in formal]
        conditions.append(dict(case=case['id'], pair=case['pair'], order=case['order'],
            cycles=helper.med(times), samples=len(formal), cv=helper.cv(times), registers=formal[0]['registers'],
            qualified=helper.cv(times)<=.05 and all(row['warmup_converged'] for row in formal),
            work_a=32768 if case['order'] in (0,2,3,4,5) else 0,
            work_b=32768 if case['order'] in (1,2,3,4,5) else 0))
    controls, joint = [], []
    for pair in (0, 2):
        points = {row['order']:row for row in conditions if row['pair'] == pair}
        values = {}
        for prefix, order, role in (('empty',6,0), ('matched-b',1,1)):
            selected = [row for row in records if row['pair'] == pair and row['label'].startswith(prefix+'-')]
            require(len(selected) == 3 and all(row['units'] == 256 and row['order'] == order
                    and row['b_role'] == role for row in selected), 'R11空单元/角色控制不齐')
            values[prefix] = helper.med([row['elapsed_cycles'] for row in selected])
        controls.append(dict(pair=pair, **values))
        ta, tb, t0 = points[0]['cycles'], points[1]['cycles'], values['empty']
        for order in (2, 3, 4, 5):
            reference = values['matched-b'] if order == 5 else tb
            elapsed = points[order]['cycles']
            joint.append(dict(pair=pair, order=order, cycles=elapsed, a_only=ta, b_only=reference,
                empty=t0, raw_sum=ta+reference, common_cost_corrected_sum=ta+reference-t0,
                max=max(ta,reference), relative_to_max=elapsed/max(ta,reference)-1,
                relative_to_corrected_sum=elapsed/(ta+reference-t0)-1))
    original = read(root/'analysis/rules.json')
    require(conditions == [row for row in original['conditions'] if row['pair'] in (0,2)],
            'R11原条件统计/资格不一致')
    require(joint == [row for row in original['joint_service'] if row['pair'] in (0,2)],
            'R11原匹配基线及联合比较不一致')
    save(output/'profiles.json', dict(conditions=conditions, joint_service=joint, controls=controls,
        unit=original['unit'], qualification=original['qualification']))
    save(output/'samples.json', samples)
    save(output/'sass-check.json', facts)
    identity = dict(evidence_root=R11.as_posix(), files=R11_FILES, binary_sha256=binary['sha256'],
        sass_sha256=binary['sass_sha256'], source_manifest_sha256=digest(root/'source_hashes.json'),
        probe_source_sha256=sources['source/probes/r11.cu'],
        profile_sha256=digest(output/'profiles.json'), samples_sha256=digest(output/'samples.json'),
        gpu_uuid=read(root/'environment.json')['gpu_uuid'])
    summary = dict(status='verified', formal_conditions=12, records=72, formal_records=36,
        short_records=24, control_records=12, checked_u64_words=checked,
        original_conditions_equal=True, original_joint_comparisons_equal=True,
        original_sass_check_equal=True, identity=identity,
        original_qualification='CV<=5%且所有正式进程暖机收敛；没有新预测门槛或独立C2',
        qualification='原完整条件窗口；不迁移到cfg_b，不拆opcode价格')
    save(output/'summary.json', summary)
    return summary, conditions, records, facts


def bind_r12(results, output):
    output.mkdir()
    snapshot = results/SMEM
    require(digest(snapshot/'summary.json') == SMEM_SUMMARY_SHA, '阶段6协议回放总身份不符')
    stage6 = read(snapshot/'summary.json')
    files = {name:sha for name,sha in stage6['artifacts'].items() if name.startswith('r12/')}
    for name, checksum in files.items():
        require(digest(snapshot/name) == checksum, '阶段6 R12快照变更：'+name)
    summary = read(snapshot/'r12/summary.json')
    require(summary['status'] == 'verified' and summary['formal_conditions'] == 24 and
            summary['control_conditions'] == 18 and summary['records'] == 174 and
            summary['checked_values'] == 14966784, 'R12已回放覆盖不符')
    comparison = read(snapshot/'r12/comparison.json')
    require(summary['original_rules_equal'] is False and summary['original_rules_equal_except_adjacent_cv'] is True
            and comparison['all_other_fields_equal'] is True and
            comparison['cv_differences'] == summary['cv_differences'], 'R12已知CV比较状态不符')
    differences = summary['cv_differences']
    require(len(differences) == 1 and differences[0]['case'] == 'tma_wgmma_SW128_serial' and
            differences[0]['original_hex'] == '0x1.1a3c0fd7d9660p-10' and
            differences[0]['replayed_hex'] == '0x1.1a3c0fd7d965fp-10' and
            differences[0]['adjacent_ULPs'] == 1 and differences[0]['qualified'] is True,
            'R12仅复用原已登记的相邻CV差异')
    for name, checksum in summary['input_identities']['files'].items():
        require(digest(results/R12/name) == checksum, 'R12原归档身份变更：'+name)
    profiles = read(snapshot/'r12/rules.json')
    phases = read(snapshot/'r12/phase-controls.json')
    facts = read(snapshot/'r12/sass-check.json')
    require(len(profiles['conditions']) == 24 and len(phases) == 6 and len(facts) == 42,
            'R12条件/相位/机器码覆盖不符')
    save(output/'profiles.json', profiles)
    save(output/'phase-controls.json', phases)
    identity = dict(evidence_root=R12.as_posix(), reused_snapshot=SMEM.as_posix(),
        stage6_summary_sha256=SMEM_SUMMARY_SHA, snapshot_files=files,
        binary_sha256=summary['input_identities']['binary_sha256'],
        sass_sha256=summary['input_identities']['files']['build/sass.txt'],
        source_manifest_sha256=summary['input_identities']['files']['source_hashes.json'],
        probe_source_sha256=summary['input_identities']['source_files_sha256']['source/probes/r12.cu'],
        profile_sha256=digest(output/'profiles.json'), gpu_uuid=summary['gpu_uuid'])
    bound = dict(status='bound_existing_replay', formal_conditions=24, control_conditions=18,
        records=174, checked_values=14966784, decompressed_records_this_run=0,
        original_rules_equal=False, original_rules_equal_except_adjacent_cv=True,
        cv_differences=differences, identity=identity, qualification=summary['qualification'])
    save(output/'summary.json', bound)
    return bound, profiles['conditions'], facts


def contracts_r11(root, summary, profiles, records, facts):
    config = read(root/'source/configs/r11.json')
    contracts = []
    for profile in profiles:
        pair, order = profile['pair'], profile['order']
        name, org = config['pairs'][pair], config['organizations'][order]
        image = next(fact for fact in facts if fact['pair'] == pair and
            (pair == 0 or f'r11_tensorILi{order}E' in fact['kernel']))
        row = next(row for row in records if row['case_id'] == profile['case'] and row['label'] == 'formal-00')
        active = [path for path in ('a','b') if profile['work_'+path]]
        work = dict(units=256, actions_per_path_per_unit=128, a_actions=profile['work_a'],
            b_actions=profile['work_b'], participating_lanes_per_path=32 if pair == 0 else 128)
        if pair == 2:
            work.update(mma_shape='m64n128k16 FP16/FP32 SS K/K', matrix_flop=profile['work_a']*262144)
        contract = dict(id='R11.'+profile['case'], pair=name, org=org, cycles=profile['cycles'], work=work,
            roles=dict(threads=row['threads'], group_threads=32 if pair == 0 else 128,
                a_role=0, b_role=1 if order == 5 else 0, registers_per_thread=row['registers'],
                dynamic_smem_bytes=row['dynamic_shared_bytes'], static_smem_bytes=row['static_shared_bytes']),
            consumers=dict(per_unit=2, stage_seed='2*unit+1 and 2*unit+2',
                method='retained FP32 values and nonlinear integer digest; peer shared-ready read'),
            barriers=dict(per_unit=2, participants=row['threads'], named_barrier=1,
                wgmma='8 instructions/commit, wait1; phase end wait0' if pair == 2 and 'a' in active else 'inactive'),
            sass=dict(kernel=image['kernel'], facts=image, sha256=summary['identity']['sass_sha256']),
            identity=dict(summary['identity'], case=profile['case']),
            covers=['R11.'+name+'.'+path+'_actions' for path in active]+['R11.loop_control','R11.consumers','R11.phase_barriers','R11.required_drain'],
            boundary=config['boundary'], qualification=dict(kind='conditional_measured_full_window',
                qualified=profile['qualified'], cv=profile['cv'], samples=profile['samples'],
                cfg_b_match=False, independent_C2=False))
        contracts.append(contract)
    return contracts


def contracts_r12(summary, profiles, facts):
    pairs = ('tma_wgmma', 'wgmma_sts128', 'tma_sts128')
    organizations = ('a_only', 'b_only', 'serial', 'concurrent')
    contracts = []
    for profile in profiles:
        pair, swizzle, order = (profile[key] for key in ('pair','swizzle','order'))
        image = next(fact for fact in facts if (fact['pair'],fact['swizzle'],fact['order']) == (pair,swizzle,order))
        paths = [path for field,path in (('transport','tma_input'),('matrix_flop','wgmma'),('stores','sts128')) if profile[field]]
        contracts.append(dict(id='R12.'+profile['case'], pair=pairs[pair], org=organizations[order],
            cycles=profile['cycles'], work=dict(units=32, transport_bytes=profile['transport'],
                store_bytes=profile['stores'], matrix_flop=profile['matrix_flop'],
                mma_shape='m64n128k16 FP16/FP32 SS K/MN', layout='SW128' if swizzle else 'INTER'),
            roles=dict(threads=384, tma_issuer=0, wgmma_warpgroup=1, sts_warpgroup=2,
                registers_per_thread=profile['registers'], dynamic_smem_bytes=155648, independent_address_regions=True),
            consumers=dict(per_unit=2, method='64 FP32 accumulators with distinct epoch seeds; peer shared-ready read',
                sts_and_tma_value_check='full physical SMEM snapshot after timed window'),
            barriers=dict(per_unit=2, participants=384, named_barrier=1,
                tma='one arrival, expected_tx=65536, four rank2 requests, acquire parity wait' if profile['transport'] else 'inactive',
                wgmma='8 instructions/commit, 8 batches/unit, wait1; phase end wait0' if profile['matrix_flop'] else 'inactive'),
            sass=dict(kernel=image['kernel'], facts=image, sha256=summary['identity']['sass_sha256']),
            identity=dict(summary['identity'], case=profile['case']),
            covers=['R12.'+path for path in paths]+['R12.loop_control','R12.consumers','R12.phase_barriers','R12.required_drain'],
            boundary='initial operands ready to complete same-SM CTA envelope after retained consumption, peer-ready reads and required drain',
            qualification=dict(kind='conditional_measured_full_window', qualified=profile['qualified'],
                cv=profile['cv'], samples=profile['samples'], cfg_b_match=False, independent_C2=False)))
    return contracts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    results, output = args.results_root.resolve(strict=True), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    r11, p11, records, f11 = replay_r11(results/R11, output/'r11')
    r12, p12, f12 = bind_r12(results, output/'r12')
    contracts = contracts_r11(results/R11, r11, p11, records, f11)+contracts_r12(r12, p12, f12)
    save(output/'contracts.json', contracts)
    examples = [dict(contract_id=contract['id'], result=joint_service(contract, [100.,125.], 2.)) for contract in contracts]
    save(output/'service-examples.json', examples)
    files = {str(path.relative_to(output)):digest(path) for path in output.rglob('*.json')}
    save(output/'summary.json', dict(schema='gh200-sm-joint-replay/1', status='verified',
        source_sha256={Path(__file__).name:digest(Path(__file__)), 'sm_joint_service.py':digest(Path(__file__).with_name('sm_joint_service.py'))},
        r11=r11, r12=r12, contracts=len(contracts), service_examples=len(examples), artifacts=files,
        new_GPU_executions=0, new_fits=0, cfg_b_matched_contracts=0,
        qualification='R11原数值/比较回放及R12阶段6复用；完整合同外部窗口，不新增独立C2或cfg_b价格'))
    print(f'VERIFIED {output}: R11 72 records / 1486080 words; R12 reused 174 records; 36 full-window contracts; GPU=0')


if __name__ == '__main__':
    main()
