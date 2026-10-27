#!/usr/bin/env python3
"""固定分配内的六条件 TMA 行距定位：准备、目标执行与独立重算。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import tarfile
import tempfile
import time

from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json

HERE = Path(__file__).resolve().parent
SCHEMA = 'gh200-q02-tma-stride-execution/1'
INPUT_FILES = ('cases.json', 'commands.json', 'correctness_commands.json',
               'hypotheses.json', 'program.json', 'worklist.txt')
ADDRESSES = ('source_a_address', 'source_b_address', 'records_address', 'ctas_address',
             'worklist_address', 'gate_address', 'pollution_address')


def prepare(root, baseline, headers):
    manifest = read_json(baseline/'delivery-manifest.json')
    selected = [*INPUT_FILES, 'plan.json', 'binding.json',
                'source/tma_gemm_sources.cu', 'source/r00_common.hpp',
                'source-before/tma_gemm_sources.cu']
    for name in selected:
        require(digest(baseline/name) == manifest['files'][name], '001 冻结输入变更：'+name)
    root.mkdir(parents=True, exist_ok=False)
    source = root/'source'
    source.mkdir()
    (root/'source-before').mkdir()
    for name in INPUT_FILES:
        shutil.copy2(baseline/name, root/name)
    for original, target in (('plan.json', 'input-plan.json'), ('binding.json', 'input-binding.json'),
                             ('delivery-manifest.json', 'input-manifest.json'),
                             ('source/tma_gemm_sources.cu', 'source/tma_gemm_sources.cu'),
                             ('source/r00_common.hpp', 'source/r00_common.hpp'),
                             ('source-before/tma_gemm_sources.cu', 'source-before/tma_gemm_sources.cu')):
        shutil.copy2(baseline/original, root/target)
    for name in ('q02_tma_stride.py', 'l2_saturation.py'):
        shutil.copy2(HERE/name, source/name)
    shutil.copy2(headers, source/'cutlass-3.9.2-include.tar.gz')
    inputs = [*INPUT_FILES, 'input-plan.json', 'input-binding.json', 'input-manifest.json']
    snapshots = [p for p in source.iterdir() if p.is_file()]
    snapshots.append(root/'source-before/tma_gemm_sources.cu')
    plan = dict(schema=SCHEMA, qualification='development_localization', independent_C2=False,
        gpu_uuid=read_json(root/'program.json')['case']['gpu_uuid'],
        seeds=[3, 22, 41], development_conditions=6, correctness_processes=6, formal_processes=6,
        correctness_arm_checks=18, formal_arm_observations=18, heldout_enabled=False,
        allocation_budget_seconds=600, maximum_gpu_seconds=3600, maximum_repairs=2,
        automatic_retries=False, source_rounds=[0, 1], omitted_third_round_ctas=60,
        numerical_response_prediction=None, frozen_at_ns=time.time_ns(),
        input_origin=baseline.name, input_manifest_sha256=digest(root/'input-manifest.json'),
        inputs_sha256={name:digest(root/name) for name in inputs},
        source_sha256={str(p.relative_to(root)):digest(p) for p in snapshots})
    save_json(root/'plan.json', plan)
    (root/'README.md').write_text('''# Q02：B 行距执行包

从冻结 001 复用 6 个开发条件、seed=3/22/41 和配对次序。先执行 6 条 verify=1 数值资格命令，再执行 6 条 verify=0 正式命令，分别为 18 个逐臂结果。两个留出保持未启用。

```sh
python source/q02_tma_stride.py execute --root "$PWD"
python source/q02_tma_stride.py score --root "$PWD" --output /tmp/q02-stride-independent-score.json
```

目标执行要求一个已分配 GH200 与 CUDA 12.9.0；原生编译新旧源码，核对 tensor_probe 的 SASS 与资源相同后采样。CUTLASS header 已打包，未带本机 host-compat。分配预算 10 分钟，绝对上限 1 小时，修复上限两次；runner 不自动重试。

stdout 按进程保存为 NDJSON，每行对应 pair_index。score 从原始 stdout、命令凭据和冻结输入独立复算每 CTA 的 cycles/ns 窗口、ns 包络、同 seed 行距响应与 Q1/Q4 响应。数值资格按对应 verify=1 臂检查完整加载值；正式时间来自 verify=0。执行只含前两轮源坐标，第三轮 60 个 CTA 未包含；没有冻结数值响应幅度，本轮结论资格为开发定位。

plan.json 保存输入和代码身份。source/q02_tma_stride.py 与 source/l2_saturation.py 是离线执行所需 Python 模块；source/r00_common.hpp 和 CUTLASS archive 是探针所需本地 header。
''')
    print('PREPARED: 6 conditions, 3 seeds; 6 correctness + 6 formal paired processes; no GPU')


def load_plan(root):
    plan = read_json(root/'plan.json')
    require(plan['schema'] == SCHEMA and plan['heldout_enabled'] is False, '执行包范围不符')
    for name, checksum in {**plan['inputs_sha256'], **plan['source_sha256']}.items():
        require(digest(root/name) == checksum, '冻结身份变更：'+name)
    require(digest(Path(__file__)) == plan['source_sha256']['source/q02_tma_stride.py'], 'runner 身份不符')
    return plan


def remaining(start, maximum):
    seconds = int(600-(time.monotonic()-start)-15)
    require(seconds > 0, '10 分钟分配预算耗尽')
    return min(maximum, seconds)


def successful_receipt(folder):
    receipt = read_json(folder/'receipt.json')
    require(receipt['returncode'] == 0 and receipt['timed_out'] is False, '命令未完成')
    require(receipt['stdout_sha256'] == digest(folder/'stdout') and
            receipt['stderr_sha256'] == digest(folder/'stderr'), '原 stdout/stderr 身份变更')
    return receipt


def build_review(root):
    folder = root/'build'
    successful_receipt(folder/'version')
    version = (folder/'version/stdout').read_text()
    rows = []
    for label, executable in (('before', 'probe-before'), ('after', 'probe')):
        for step in ('compile', 'disassemble', 'resources'):
            successful_receipt(folder/label/step)
        sass = (folder/label/'disassemble/stdout').read_text()
        sections = [section for section in re.split(r'Function\s*:\s*', sass)[1:]
                    if section.splitlines()[0].startswith('_Z12tensor_probe')]
        require(len(sections) == 1, 'tensor_probe SASS 唯一入口缺失')
        section = sections[0]
        symbol = section.splitlines()[0].strip()
        words = [line.strip() for line in section.splitlines()[1:]
                 if re.match(r'\s*/\*[0-9a-f]+\*/', line) or re.search(r'/\* 0x[0-9a-f]+ \*/', line)]
        instructions = [line for line in words if re.match(r'/\*[0-9a-f]+\*/', line)]
        resources = (folder/label/'resources/stdout').read_text()
        match = re.search(r'Function '+re.escape(symbol)+r':\s*\n\s*([^\n]+)', resources)
        require(match is not None, 'tensor_probe 资源缺失')
        resource = match[1].strip()
        log = (folder/label/'compile/stdout').read_text()+(folder/label/'compile/stderr').read_text()
        rows.append(dict(label=label, symbol=symbol, instruction_count=len(instructions),
            instruction_words_sha256=hashlib.sha256('\n'.join(words).encode()).hexdigest(),
            utma_count=sum('UTMALDG' in line for line in instructions), resources=resource,
            zero_local_stack=bool(re.search(r'\bSTACK:0\b', resource) and re.search(r'\bLOCAL:0\b', resource)),
            compile_warning_c7515='C7515' in log, probe_sha256=digest(folder/executable)))
    passed = bool(re.search(r'release 12\.9(?:,|\s)', version))
    passed = passed and rows[0]['instruction_words_sha256'] == rows[1]['instruction_words_sha256']
    passed = passed and rows[0]['resources'] == rows[1]['resources']
    passed = passed and all(row['utma_count'] == 18 and row['zero_local_stack'] and
                            not row['compile_warning_c7515'] for row in rows)
    return dict(passed=passed, compiler_version=version, rows=rows, plan_sha256=digest(root/'plan.json'))


def build(root, start):
    folder = root/'build'
    folder.mkdir(exist_ok=False)
    compiler = shutil.which('nvcc')
    require(compiler is not None, 'nvcc 缺失')
    command_record(folder/'version', [compiler, '--version'], remaining(start, 20))
    require(re.search(r'release 12\.9(?:,|\s)', (folder/'version/stdout').read_text()), '需要 CUDA 12.9')
    flags = ['-std=c++17', '-O3', '-DNDEBUG', '-lineinfo',
             '-gencode=arch=compute_90a,code=sm_90a', '-Xptxas=-v']
    disassembler = Path(compiler).resolve().parent/'cuobjdump'
    with tempfile.TemporaryDirectory(prefix='q02-cutlass-') as directory:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(directory, filter='data')
        for label, source, executable in (('before', 'source-before/tma_gemm_sources.cu', 'probe-before'),
                                           ('after', 'source/tma_gemm_sources.cu', 'probe')):
            command_record(folder/label/'compile', [compiler, *flags, '-I'+directory+'/include',
                '-I'+str(root/'source'), root/source, '-o', folder/executable], remaining(start, 180))
            command_record(folder/label/'disassemble', [disassembler, '--dump-sass', folder/executable], remaining(start, 40))
            command_record(folder/label/'resources', [disassembler, '--dump-resource-usage', folder/executable], remaining(start, 30))
    report = build_review(root)
    save_json(folder/'target-review.json', report)
    require(report['passed'], '原生 SASS/资源审核未通过，停止采样')


def probe_command(root, task):
    command = task['argv'][:]
    command[0] = str(root/'build/probe')
    command[command.index('--worklist')+1] = str(root/'worklist.txt')
    return command


def validate_process(root, folder, task, uuid):
    receipt = successful_receipt(folder)
    command = receipt['argv'][:]
    probe = Path(command[0])
    require(probe.name == 'probe' and probe.parent.name == 'build', '执行的二进制路径不符')
    require(command[command.index('--worklist')+1] == str(probe.parent.parent/'worklist.txt'), '工作列表路径不符')
    command[0] = task['argv'][0]
    command[command.index('--worklist')+1] = 'worklist.txt'
    require(command == task['argv'], '执行参数与冻结命令不符')
    raw = [json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.strip()]
    require(len(raw) == len(task['paired_ldbs']), '配对臂数不符')
    arguments = dict(zip(task['argv'][1::2], task['argv'][2::2]))
    group_flag = '--measure-groups' if int(arguments.get('--measure-groups', '0')) > 0 else '--groups'
    active_flag = '--measure-active' if int(arguments.get('--measure-active', '0')) > 0 else '--active'
    fields = {'ctas':'--ctas', 'active_groups':active_flag, 'groups':group_flag, 'ring':'--ring',
        'verify':'--verify', 'trace':'--trace', 'seed':'--seed', 'warmup':'--warmup',
        'source_phase_stride':'--source-phase-stride', 'source_tile_shift':'--source-tile-shift',
        'work_index':'--work-index', 'warmup_work_offset':'--warmup-work-offset',
        'release_delay_ns':'--release-delay-ns', 'pollute_kib':'--pollute-kib',
        'launch_skew_ns':'--launch-skew-ns', 'warmup_skew_ns':'--warmup-skew-ns', 'alloc_ldb':'--alloc-ldb'}
    lines = (root/'worklist.txt').read_text().splitlines()
    cta_count = int(arguments['--ctas'])
    header = list(map(int, lines[0].split()))
    require(header == [2304, 2304, 4096, cta_count] and len(lines) == 1+2*cta_count, '工作列表形状或 CTA 数不符')
    first = 1+int(arguments['--work-index'])*cta_count
    coordinates = [list(map(int, line.split())) for line in lines[first:first+cta_count]]
    checked = []
    for index, row in enumerate(raw):
        require(row['schema'] == 'cfg-b-gemm-source-probe/1' and row['gpu_uuid'] == uuid, 'schema/卡身份不符')
        require(all(row[name] == int(arguments[flag]) for name, flag in fields.items()), '逐臂参数不符')
        if active_flag == '--measure-active':
            require(row.get('warmup_active_groups') == int(arguments['--active']), '暖机活动组数不符')
        require(row['paired_ldbs'] == task['paired_ldbs'] and row['pair_index'] == index and
                row['ldb'] == task['paired_ldbs'][index], '逐臂行距或次序不符')
        require(all(row[name] == raw[0][name] for name in ADDRESSES), '同进程配对的分配地址变化')
        require(row['source_a_address'] > 0 and row['source_b_address'] > 0, '源地址缺失')
        require(row['m'] == row['n'] == 2304 and row['lda'] == row['alloc_lda'] == 4096 and
                row['allocated_a_elements'] == 9437184 and row['allocated_b_elements'] == 9666560,
                '问题形状或固定分配不符')
        require(row['warmup_groups'] == int(arguments['--groups']) and row['warmup_source_phase_stride'] == 0 and
                row['record_warmup'] == 0 and not row['warmup_ctas'] and not row['group_events'], '暖机或 trace 范围不符')
        require(row['errors'] == row['pollution_errors'] == 0 and row['pollution_checked'] == 0, '加载或污染检查错误')
        expected = cta_count*int(arguments[group_flag])*16384 if row['verify'] else 0
        require(row['checked'] == expected, '完整加载检查量不符')
        require(row['observed_sms'] == cta_count and row['max_active_ctas_per_sm'] == 1 and
                row['smem'] == 132096 and row['local_bytes'] == 0, '驻留或资源不符')
        resource = read_json(root/'build/target-review.json')['rows'][1]['resources']
        require(row['registers'] == int(re.search(r'\bREG:(\d+)\b', resource)[1]), '运行寄存器与目标构建不符')
        ctas = row['ctas_data']
        require([cta['cta'] for cta in ctas] == list(range(cta_count)), 'CTA 清单不符')
        require([cta['tile'] for cta in ctas] == coordinates and len({cta['smid'] for cta in ctas}) == cta_count,
                '工作坐标或唯一 SM 不符')
        windows = []
        for cta in ctas:
            require(type(cta['smid']) is int and cta['smid'] >= 0, 'SM id 缺失')
            for unit in (0, 1):
                require(type(cta['begin'][unit]) is int and type(cta['end'][unit]) is int and
                        0 < cta['begin'][unit] < cta['end'][unit], 'CTA 完成戳无效')
            windows.append(dict(cta=cta['cta'], smid=cta['smid'], cycles=cta['end'][0]-cta['begin'][0],
                                ns=cta['end'][1]-cta['begin'][1]))
        metrics = dict(ns_envelope=max(c['end'][1] for c in ctas)-min(c['begin'][1] for c in ctas),
            **{unit+'_median':statistics.median(c[unit] for c in windows) for unit in ('cycles','ns')},
            **{unit+'_spread':max(c[unit] for c in windows)-min(c[unit] for c in windows) for unit in ('cycles','ns')})
        checked.append(dict(seed=row['seed'], active_groups=row['active_groups'], ldb=row['ldb'],
            pair_index=index, checked_values=row['checked'], numerical_qualified=bool(row['verify']),
            stdout_sha256=receipt['stdout_sha256'], source=str(folder.relative_to(root)),
            device_uuid=uuid, ctas=windows, metrics=metrics))
    return checked


def summarize(root):
    plan = load_plan(root)
    review = build_review(root)
    require(review == read_json(root/'build/target-review.json') and review['passed'], '构建证据重算不符')
    environment = read_json(root/'allocation/identity.json')
    require(environment['uuid'] == plan['gpu_uuid'], '分配卡身份不符')
    numeric, formal = [], []
    for filename, split, rows in (('correctness_commands.json', 'correctness', numeric),
                                 ('commands.json', 'formal', formal)):
        for task in read_json(root/filename):
            rows.extend(validate_process(root, root/'runs'/split/task['id'], task, plan['gpu_uuid']))
    by_numeric = {(r['seed'], r['active_groups'], r['ldb']):r for r in numeric}
    by_formal = {(r['seed'], r['active_groups'], r['ldb']):r for r in formal}
    require(len(by_numeric) == len(by_formal) == 18 and by_numeric.keys() == by_formal.keys(), '数值资格与正式臂不对应')
    for key, row in by_formal.items():
        row['numeric_source'] = by_numeric[key]['source']
    pairs = []
    for seed in plan['seeds']:
        comparisons = [('stride', (4,2304), (4,ldb)) for ldb in (2320,2344,2360)]
        comparisons += [('stride', (1,2304), (1,2344)), ('Q', (1,2304), (4,2304)), ('Q', (1,2344), (4,2344))]
        for kind, left, right in comparisons:
            a, b = [by_formal[(seed,*key)] for key in (left,right)]
            pairs.append(dict(seed=seed, kind=kind, base=list(left), changed=list(right),
                smid_mapping_equal=[c['smid'] for c in a['ctas']] == [c['smid'] for c in b['ctas']],
                metrics={name:dict(difference=b['metrics'][name]-value, response=b['metrics'][name]/value-1)
                         for name,value in a['metrics'].items() if name not in ('cycles_spread','ns_spread')},
                ctas=[dict(cta=x['cta'], **{unit:dict(difference=y[unit]-x[unit], response=y[unit]/x[unit]-1)
                    for unit in ('cycles','ns')}) for x,y in zip(a['ctas'],b['ctas'])]))
    return dict(schema='gh200-q02-tma-stride-score/1', qualification=plan['qualification'], independent_C2=False,
        numerical_response_prediction=None, numeric_arm_checks=len(numeric), formal_arm_observations=len(formal),
        numeric=[{key:value for key,value in row.items() if key not in ('metrics','ctas')} for row in numeric],
        formal=formal, pairs=pairs, heldout_enabled=False, plan_sha256=digest(root/'plan.json'),
        build_review_sha256=digest(root/'build/target-review.json'))


def execute(root):
    plan = load_plan(root)
    start = time.monotonic()
    environment = inspect_allocated_gpu(root/'allocation')
    require(environment['uuid'] == plan['gpu_uuid'], '分配卡与冻结研究对象不符')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        build(root, start)
        for filename, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
            for task in read_json(root/filename):
                folder = root/'runs'/split/task['id']
                command_record(folder, probe_command(root,task), remaining(start,60))
                checked = validate_process(root,folder,task,environment['uuid'])
                save_json(folder/'validated.json',checked)
                print(json.dumps(dict(split=split,process=task['id'],arms=len(checked))),flush=True)
        save_json(root/'score.json',summarize(root))
    print('CLOSED: 18 numeric checks + 18 formal arms; development localization',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','execute','score'):
        item = sub.add_parser(name)
        item.add_argument('--root',type=Path,required=True)
        if name == 'prepare':
            item.add_argument('--baseline',type=Path,required=True)
            item.add_argument('--headers',type=Path,required=True)
        if name == 'score':
            item.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'prepare':prepare(root,args.baseline.resolve(),args.headers.resolve())
    elif args.command == 'execute':execute(root)
    else:
        report = summarize(root)
        save_json(args.output,report)
        print('VERIFIED: 18 numeric checks, 18 formal arms and same-seed stride/Q responses')


if __name__ == '__main__':main()
