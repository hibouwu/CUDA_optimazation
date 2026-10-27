#!/usr/bin/env python3
"""回放固定R03／R12的数值、完整窗口与阶段对照，不运行GPU或拟合。"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys


R03 = Path('gh200_resource_campaign/access_rules/Q07-smem/R03-access-demand/20261006-c-job735059/r03-formal-v1')
R12 = Path('gh200_resource_campaign/access_rules/Q07-smem/R12-smem-path-contention/20261008-R12-job736989-v3')
R03_IDENTITIES = {
    'source_hashes.json': 'a5bed452e623fd422d96b4eacea395864ddee28e484953336a0c9fac25dc7317',
    'cases.json': '935b9d9273ae7e0499c3bd41dc36d32a60dce560c75f18d7f228f8b222ecf60e',
    'protocol.json': '4dddf4e65602e5559b676745acc8bdd163639a87ee41dae31806e28f633a0044',
    'build/binary_hash.json': '2c64721823813b6891bc2310d89944236b064cad58e05b51771153bd28615aad',
    'build/r03.sass': '8269e72bc80940071641bbc36415fe260115cff3d13595c483e4bc90deab4725',
    'build/sass-audit.json': '4ddaf2bd829b91253c0d2df9fcea332fdf8b276289024f90ce0c1b3d8bc8858a',
    'rules.json': '5e9138b68f5a9ff3bba33f40987522b58facc22126a07529624bbb38340acc94',
    'cases.csv': '2576afb1ddb5708bc6654be8c6b817d2b19dc880b0b325c6ff508b8ff443038b',
}
R12_IDENTITIES = {
    'source_hashes.json': 'ec9a2cc152b87a2c5dd707bf9406c3d4faf6d72d5868ef2b4995d9a44cdb87b9',
    'cases.json': '9ebfe9e6f086b7e6c7b38f5852c77464fb53f0b165bb4d8b01c95dadd7889f52',
    'build/binary.json': '38344f68d8c4f1de9a4c3e181d2e7b523806bfd162bf26c376c73ebac8976e5a',
    'build/sass.txt': '52e17e8e655be8fa4df3b5f4375f0c74f4205dcee68301ff09b1dbbc4875f3e5',
    'analysis/rules.json': '88075cf4bb25099ed60a7559f48555c26acc555b6504073e86ce8ab653675b53',
    'analysis/checks.json': '75890c4835658a2ca6974141be8a452b9b7a6ca539ef4cf0de730986b385f67e',
    'analysis/sass-check.json': 'e6c503057c0c80161c5020d102002d876e80cd695c42ea63ccce40be2be45c64',
    'analysis/phase-controls.json': '3cd28729662e58321bad5eaa57d97919e12349db3d12aeba83e3addda7517161',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def bind_sources(root, identities, source_prefix):
    for name, expected in identities.items():
        require(digest(root / name) == expected, '固定归档身份不符: ' + str(root / name))
    sources = read(root / 'source_hashes.json')
    for name, expected in sources.items():
        require(digest(root / source_prefix / name) == expected, '冻结源码不符: ' + name)
    return dict(files=identities, source_files=len(sources), source_files_sha256=sources)


def load_helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_r03(root, output):
    identities = bind_sources(root, R03_IDENTITIES, Path('source'))
    binary = digest(root / 'build/r03')
    require(binary == read(root / 'build/binary_hash.json'), 'R03 binary不符')
    identities['binary_sha256'] = binary
    helper = load_helper(root / 'source/analyze_r03.py', 'frozen_r03_replay')
    cases = read(root / 'cases.json')
    require(len(cases) == 18, 'R03需18条件')
    protocol = read(root / 'protocol.json')
    require(protocol['iterations'] == 44693, 'R03正式循环数不符')
    groups = {case['id']: [] for case in cases}
    records, samples, checked = [], [], 0
    for path in sorted((root / 'samples').glob('*/*/result.json')):
        record = read(path)
        case_id = record['case_id']
        require(case_id in groups and record['configuration'] == next(
            case for case in cases if case['id'] == case_id), 'R03条件不符')
        require(record['errors'] == 0 and record['warmup_converged'] and
                record['local_bytes_per_thread'] == 0 and record['elapsed'] > 0,
                'R03原状态或窗口不符')
        require(read(path.parent / 'process.json')['returncode'] == 0, 'R03旧进程失败')
        count = helper.check(record, path.parent / 'output.u32')
        require(count == record['checked_elements'], 'R03声明检查量不符')
        checked += count
        if record['role'].startswith('trial-'):
            require(record['iterations'] == protocol['iterations'], 'R03正式轮数不符')
            groups[case_id].append(record)
        else:
            require(record['role'] == 'pilot' and record['iterations'] == 128, 'R03 pilot不符')
        records.append(record)
        samples.append(dict(path=path.relative_to(root).as_posix(), result_sha256=digest(path),
            output_sha256=record['output_sha256'], case_id=case_id, role=record['role'],
            elapsed_cycles=record['elapsed'], work_bytes=record['work_bytes'],
            requested_bytes_per_cycle=record['work_bytes'] / record['elapsed'], checked_elements=count))
    require(len(records) == 72 and sum(r['role'] == 'pilot' for r in records) == 18 and
            all({r['role'] for r in rows} == {'trial-00', 'trial-01', 'trial-02'} for rows in groups.values()),
            'R03需18 pilot及54正式记录')
    old_rules = read(root / 'rules.json')
    require([r['configuration'] for r in old_rules] == cases, 'R03原rules条件不符')
    rules = []
    environment = read(root / 'environment.json')
    for old in old_rules:
        rows = groups[old['configuration']['id']]
        rates = [r['work_bytes'] / r['elapsed'] for r in rows]
        mean = statistics.mean(rates)
        cv = statistics.stdev(rates) / mean
        # 原rules后增的资格字段原样保存；只独立重算旧分析器的数值与状态字段。
        rule = dict(old, processes=len(rows), mean_requested_bytes_per_cycle=mean, cv=cv,
            status='observed' if cv <= .05 and all(r['warmup_converged'] for r in rows) else 'unstable',
            end_event=rows[0]['end_event'], gpu_uuid=environment['gpu_uuid'])
        require(rule == old, 'R03原rules逐对象不符: ' + old['configuration']['id'])
        rules.append(rule)
    with (root / 'cases.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 18, 'R03原CSV覆盖不符')
    for row, rule in zip(csv_rows, rules):
        require(row['id'] == rule['configuration']['id'] and int(row['processes']) == rule['processes'] and
                float(row['mean_requested_bytes_per_cycle']) == rule['mean_requested_bytes_per_cycle'] and
                float(row['cv']) == rule['cv'] and row['status'] == rule['status'], 'R03原CSV数值不符')
    sass = (root / 'build/r03.sass').read_text()
    sass_facts = []
    for old in read(root / 'build/sass-audit.json'):
        match = re.search(r'r03_probeILi([0-3])ELi(4|16)EE', old['symbol'])
        require(match is not None, 'R03 SASS完整symbol不符')
        op, width = map(int, match.groups())
        instruction = ('LDS' if width == 4 else 'LDS.128',
                       'STS' if width == 4 else 'STS.128', 'LDSM.16.M88.4', 'STSM.16.M88.4')[op]
        body = next(body for body in sass.split('Function : ')[1:] if body.splitlines()[0].strip() == old['symbol'])
        clocks = [int(pc, 16) for pc in re.findall(r'/\*([0-9a-f]+)\*/[^\n]*CS2R[^\n]*SR_CLOCKLO', body)]
        require(len(clocks) == 2, 'R03需两个clock64端点')
        loops = [(int(target, 16), int(pc, 16)) for pc, target in re.findall(
            r'/\*([0-9a-f]+)\*/[^\n]*\bBRA\s+0x([0-9a-f]+)', body)
            if clocks[0] < int(target, 16) < int(pc, 16) < clocks[1]]
        require(len(loops) == 1 and old['transport_instructions_per_round'] == 8,
                'R03 SASS计时主循环不符')
        lo, hi = loops[0]
        target_lines = []
        for line in old['static_matches']:
            require(line in body, 'R03旧SASS静态匹配不符')
            pc = int(re.search(r'/\*([0-9a-f]+)\*/', line)[1], 16)
            if lo <= pc <= hi:
                require(re.search(r'\*/\s+' + re.escape(instruction) + r'\s', line), 'R03实际SASS目标不符')
                target_lines.append(line)
        require(len(target_lines) == 8, 'R03计时循环需八条目标访问')
        sass_facts.append(dict(symbol=old['symbol'], actual_width_bytes=width, instruction=instruction,
            timed_target_count=8, loop_start_pc=lo, loop_end_pc=hi,
            original_static_matches=len(old['static_matches']), original_audit_width=old['width']))
    require(len(sass_facts) == 6, 'R03需六个目标kernel')
    output.mkdir()
    save(output / 'rules.json', rules)
    save(output / 'samples.json', samples)
    save(output / 'sass-check.json', sass_facts)
    summary = dict(status='verified', conditions=18, records=72, pilot_records=18, formal_records=54,
        checked_elements=checked, original_rules_equal=True, original_csv_equal=True,
        gpu_uuid=environment['gpu_uuid'], input_identities=identities,
        qualification='原条件完整窗口；保留原rules资格字段，不新增预测适用性或C2成绩',
        sass_note='旧audit将Li16的width解析为1，STS静态匹配含初始化及sink；保留旧文件，按完整symbol及clock64间主循环核对八条4/16B访问',
        timing='clock64 CTA窗口，含八次访问/轮、地址、消费/sink与CTA会合；不是裸SMEM率')
    save(output / 'summary.json', summary)
    return summary


def replay_r12(root, output):
    identities = bind_sources(root, R12_IDENTITIES, Path())
    binary = read(root / 'build/binary.json')
    require(digest(root / 'build/r12') == binary['sha256'] and
            digest(root / 'build/sass.txt') == binary['sass_sha256'], 'R12 binary/SASS不符')
    identities['binary_sha256'] = binary['sha256']
    helper = load_helper(root / 'source/analyze_r12.py', 'frozen_r12_replay')
    # 冻结analyze只在显式output写入；不导入会在原目录执行的plot_r12.py。
    helper.analyze(root, output)
    rules = read(output / 'rules.json')
    original_rules = read(root / 'analysis/rules.json')
    cv_differences = []
    comparable = dict(original_rules, conditions=[dict(row) for row in original_rules['conditions']])
    require(len(rules['conditions']) == len(comparable['conditions']), 'R12原条件数不符')
    for index, (old, new) in enumerate(zip(original_rules['conditions'], rules['conditions'])):
        if old['cv'] != new['cv']:
            require(math.nextafter(old['cv'], new['cv']) == new['cv'] and
                    (old['cv'] <= .05) == (new['cv'] <= .05) and old['qualified'] == new['qualified'],
                    'R12仅允许cv字段相邻1 ULP且原资格判断不变')
            cv_differences.append(dict(field=f'conditions[{index}].cv', case=old['case'],
                original=old['cv'], replayed=new['cv'], original_hex=old['cv'].hex(),
                replayed_hex=new['cv'].hex(), adjacent_ULPs=1,
                original_threshold_passed=old['cv'] <= .05, replayed_threshold_passed=new['cv'] <= .05,
                qualified=old['qualified']))
            comparable['conditions'][index]['cv'] = new['cv']
    require(rules == comparable, 'R12原rules存在非相邻CV差异')
    for name in ('checks.json', 'sass-check.json'):
        require(read(output / name) == read(root / 'analysis' / name), 'R12原分析逐对象不符: ' + name)
    save(output / 'comparison.json', dict(original_rules_equal=rules == original_rules,
        all_other_fields_equal=True, cv_differences=cv_differences,
        criterion='仅cv字段相邻1 ULP且原qualified及5%门槛判断一致；其他字段严格相等',
        prior_relocation_note_sha256=digest(root / 'analysis/replay-verification.json')))
    records, samples = [], []
    for path in sorted((root / 'samples').glob('*/*/result.json')):
        record = read(path)
        require(record['returncode'] == 0 and record['event_errors'] == 0 and record['local_bytes'] == 0,
                'R12原进程/资源状态不符')
        records.append(record)
        samples.append(dict(path=path.relative_to(root).as_posix(), result_sha256=digest(path),
            case_id=record['case_id'], label=record['label'], units=record['units'],
            elapsed_cycles=record['elapsed_cycles'], payload_files=record['files']))
    require(len(records) == 174 and len(rules['conditions']) == 24 and
            len(read(output / 'sass-check.json')) == 42, 'R12原覆盖不符')
    require(sum(r['label'].startswith('formal-') for r in records) == 72 and
            sum(r['label'].startswith('smoke-') for r in records) == 48, 'R12正式/短记录数不符')
    controls = []
    for group in rules['joint_service']:
        selected = [r for r in records if r['pair'] == group['pair'] and r['swizzle'] == group['swizzle']]
        phase = {}
        for label in ('b-first', 'a-second', 'empty'):
            rows = [r for r in selected if r['label'].startswith(label + '-')]
            require(len(rows) == 3 and all(r['units'] == 32 for r in rows), 'R12阶段控制覆盖不符')
            phase[label] = statistics.median(r['elapsed_cycles'] for r in rows)
        matched_max = max(group['a'], phase['b-first'])
        phase.update(pair=group['pair'], swizzle=group['swizzle'], a_first=group['a'],
            b_second=group['b'], concurrent=group['concurrent'], phase_matched_max=matched_max,
            concurrent_excess=group['concurrent'] / matched_max - 1)
        controls.append(phase)
    require(controls == read(root / 'analysis/phase-controls.json'), 'R12原阶段对比逐对象不符')
    save(output / 'phase-controls.json', controls)
    save(output / 'samples.json', samples)
    summary = dict(status='verified', formal_conditions=24, control_conditions=18, records=174,
        formal_records=72, short_records=48, control_records=54,
        checked_values=rules['checked_values'], original_rules_equal=rules == original_rules,
        original_rules_equal_except_adjacent_cv=True, cv_differences=cv_differences, original_checks_equal=True,
        original_sass_check_equal=True, original_phase_controls_equal=True,
        gpu_uuid=read(root / 'environment.json')['gpu_uuid'], input_identities=identities,
        qualification='独立区域与固定角色的原联合完整窗口；不辨识物理SMEM端口或迁移到cfg_b',
        timing='SM clock64 CTA包络；32单元；匹配第一阶段B-only，保留消费与同步')
    save(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    results = args.results_root.resolve(strict=True)
    output = args.output.resolve()
    for relative in (R03, R12):
        old = results / relative
        require(output != old and old not in output.parents, '输出不能位于旧证据目录内')
    output.mkdir(parents=True, exist_ok=False)
    try:
        r03 = replay_r03(results / R03, output / 'r03')
        r12 = replay_r12(results / R12, output / 'r12')
        artifacts = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
        summary = dict(schema='gh200-smem-protocol-replay/1', status='verified',
            results_root=str(results), replay_source_sha256=digest(Path(__file__)),
            r03=dict(conditions=18, records=72, checked_elements=r03['checked_elements']),
            r12=dict(formal_conditions=24, control_conditions=18, records=174, checked_values=r12['checked_values']),
            new_GPU_executions=0, new_fits=0, artifacts=artifacts,
            scope='仅R03/R12旧完整窗口回放；不扩量EXP09/10，不重开m11，不新增独立C2')
        save(output / 'summary.json', summary)
    except Exception as error:
        save(output / 'failure.json', dict(status='failed', error=str(error)))
        raise
    print(json.dumps(dict(status='verified', output=str(output), r03_records=72, r12_records=174), ensure_ascii=False))


if __name__ == '__main__':
    main()
