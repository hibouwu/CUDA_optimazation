#!/usr/bin/env python3
"""四条件plain索引干预：准备、目标执行与独立重算。"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import tarfile
import tempfile
import time

import cfg_b_handoff_chain as witness
from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json

HERE = Path(__file__).resolve().parent
SEEDS = (3, 22, 41)


def cases(round_name):
    if round_name == 'localize':
        return [dict(id=f's6_k{k}_{mode}_d{delay}', index_mode=mode, stages=6, ktiles=k,
                     delay_cycles=delay, trace=False, wgmma_count=8*k, work_flop=2*128*128*64*k)
                for mode,k,delay in (('incremental',4,0), ('incremental',13,0),
                                     ('consumer-only',13,0), ('consumer-only',13,4096))]
    return [dict(id=f's6_k13_{mode}_d{delay}', index_mode=mode, stages=6, ktiles=13,
                 delay_cycles=delay, trace=False, wgmma_count=104, work_flop=2*128*128*64*13)
            for mode in ('runtime', 'incremental') for delay in (0, 4096)]


def prepare(root, headers, round_name):
    source = root/'source'
    source.mkdir(exist_ok=False)
    for name in ('q04_index_control.py', 'cfg_b_handoff_chain.py', 'cfg_b_compute_chain.py',
                 'l2_saturation.py', 'mbarrier_service.py', 'supply_model.py'):
        shutil.copy2(HERE/name, source/name)
    for name in ('cfg_b_handoff_chain.cu', 'r00_common.hpp'):
        shutil.copy2(HERE/'probes'/name, source/name)
    shutil.copy2(headers, source/'cutlass-3.9.2-include.tar.gz')
    order = []
    rows = cases(round_name)
    for index, seed in enumerate(SEEDS):
        sequence = rows[index:]+rows[:index]
        order += [dict(case_id=row['id'], seed=seed) for row in sequence]
    save_json(root/'plan.json', dict(schema='gh200-q04-index-control/1', cases=rows,
        round=round_name, seeds=list(SEEDS), order=order, processes=12, allocation_budget_seconds=600,
        maximum_gpu_seconds=3600, maximum_repairs=2, qualification='development_discrimination',
        prediction=('索引控制改变循环推进；幅度未辨识，原零控制形式作为已失败基线保留' if round_name=='first' else
                    'consumer-only隔离full之前控制；增量K4/K13检验循环量，独立数值幅度未辨识，本轮开发定位'),
        source_sha256={path.name:digest(path) for path in source.iterdir()}))
    print('PREPARED: 4 conditions, 12 processes, plain only')


def build(root):
    folder = root/'build'
    folder.mkdir(exist_ok=False)
    compiler = shutil.which('nvcc')
    require(compiler is not None, 'nvcc缺失')
    command_record(folder/'version', [compiler, '--version'], 20)
    flags = ['-std=c++17', '-O3', '-DNDEBUG', '-lineinfo',
             '-gencode=arch=compute_90a,code=sm_90a', '-Xptxas=-v']
    with tempfile.TemporaryDirectory(prefix='q04-cutlass-') as directory:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(directory, filter='data')
        command_record(folder/'compile', [compiler, *flags, '-I'+directory+'/include',
            root/'source/cfg_b_handoff_chain.cu', '-o', folder/'probe'], 180)
    command_record(folder/'disassemble', [Path(compiler).resolve().parent/'cuobjdump',
                                        '--dump-sass', folder/'probe'], 40)
    version = (folder/'version/stdout').read_text()
    log = (folder/'compile/stdout').read_text()+(folder/'compile/stderr').read_text()
    functions = []
    for section in re.split(r'Function\s*:\s*', (folder/'disassemble/stdout').read_text())[1:]:
        symbol = section.splitlines()[0].strip()
        if 'handoff_chain' not in symbol:
            continue
        instructions = [m[1].strip() for line in section.splitlines()[1:]
                        if (m := re.match(r'\s*/\*[0-9a-f]+\*/\s+([^;]+);', line))]
        properties = re.search(r'Function properties for '+re.escape(symbol)+
            r'\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
        functions.append(dict(symbol=symbol,
            mma=[x for x in instructions if 'HGMMA.' in x],
            waits=[x for x in instructions if 'WARPGROUP.DEPBAR' in x],
            barrier_instructions=[x for x in instructions if 'SYNCS.' in x],
            instruction_count=len(instructions),
            no_local=not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)', x) for x in instructions),
            zero_spill=bool(properties and all(int(x)==0 for x in properties.groups()))))
    passed = bool(re.search(r'release 12\.9(?:,|\s)', version)) and 'C7515' not in log and len(functions)==6
    passed = passed and all(len(x['mma'])==8 and all('.tnspB' in m for m in x['mma'])
        and len(x['waits'])==2 and any(re.search(r',\s*0x1$', w) for w in x['waits'])
        and any(re.search(r',\s*0x0$', w) for w in x['waits'])
        and x['barrier_instructions'] and x['no_local'] and x['zero_spill'] for x in functions)
    save_json(folder/'target-review.json', dict(passed=passed, functions=functions,
        probe_sha256=digest(folder/'probe'), sass_sha256=digest(folder/'disassemble/stdout'),
        source_sha256=digest(root/'source/cfg_b_handoff_chain.cu'), flags=flags,
        plan_sha256=digest(root/'plan.json'), compiler_version=version))
    require(passed, '目标SASS/资源审核未通过，停止采样')


def summarize(root):
    plan = read_json(root/'plan.json')
    rows = []
    for task in plan['order']:
        case = next(row for row in plan['cases'] if row['id']==task['case_id'])
        folder = root/'runs'/case['id']/f"seed{task['seed']}"
        record = read_json(folder/'stdout')
        checked = witness.validate(folder, record, case, task['seed'])
        require(record['index_mode']==case['index_mode'], '索引变体不符')
        receipt = read_json(folder/'receipt.json')
        require(receipt['returncode']==0 and receipt['timed_out'] is False, '命令未成功')
        rows.append(dict(case_id=case['id'], seed=task['seed'], index_mode=case['index_mode'],
            delay_cycles=case['delay_cycles'], cycles=record['timing']['elapsed_cycles'],
            ns=record['timing']['elapsed_ns'], resources=record['resources'],
            device_uuid=record['device']['uuid'], performance_eligible=checked['performance_eligible'],
            output_witness=checked['output_witness']))
    require(len({row['device_uuid'] for row in rows})==1, '配对跨卡')
    conditions = []
    for case in plan['cases']:
        group = [row for row in rows if row['case_id']==case['id']]
        conditions.append(dict(case_id=case['id'], **{unit:dict(
            median=statistics.median(row[unit] for row in group),
            minimum=min(row[unit] for row in group), maximum=max(row[unit] for row in group))
            for unit in ('cycles','ns')}))
    pairs = []
    for seed in SEEDS:
        selected = {(r['index_mode'],r['delay_cycles'],next(c['ktiles'] for c in plan['cases'] if c['id']==r['case_id'])):r
                    for r in rows if r['seed']==seed}
        if plan.get('round','first') == 'first':
            values = {unit:dict(
                runtime_response=selected['runtime',4096,13][unit]-selected['runtime',0,13][unit],
                incremental_response=selected['incremental',4096,13][unit]-selected['incremental',0,13][unit],
                baseline_change=selected['incremental',0,13][unit]-selected['runtime',0,13][unit],
                response_change=(selected['incremental',4096,13][unit]-selected['incremental',0,13][unit])-
                                (selected['runtime',4096,13][unit]-selected['runtime',0,13][unit]))
                for unit in ('cycles','ns')}
        else:
            values = {unit:dict(
                incremental_length_growth=selected['incremental',0,13][unit]-selected['incremental',0,4][unit],
                incremental_per_extra_tile=(selected['incremental',0,13][unit]-selected['incremental',0,4][unit])/9,
                consumer_only_response=selected['consumer-only',4096,13][unit]-selected['consumer-only',0,13][unit],
                consumer_only_baseline_change=selected['consumer-only',0,13][unit]-selected['incremental',0,13][unit])
                for unit in ('cycles','ns')}
        pairs.append(dict(seed=seed, **values))
    return dict(schema='gh200-q04-index-control-score/1', rows=rows, conditions=conditions, pairs=pairs,
        all_performance_eligible=all(row['performance_eligible'] for row in rows),
        qualification='development_discrimination', independent_C2=False,
        plan_sha256=digest(root/'plan.json'), build_sha256=digest(root/'build/target-review.json'))


def execute(root):
    plan = read_json(root/'plan.json')
    for name, checksum in plan['source_sha256'].items():
        require(digest(root/'source'/name)==checksum, '准备包源码变更：'+name)
    require(digest(Path(__file__))==plan['source_sha256']['q04_index_control.py'], 'runner身份不符')
    start = time.time()
    environment = inspect_allocated_gpu(root/'allocation')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        build(root)
        for task in plan['order']:
            require(time.time()-start < plan['allocation_budget_seconds']-45, '本轮时间预算耗尽')
            case = next(row for row in plan['cases'] if row['id']==task['case_id'])
            folder = root/'runs'/case['id']/f"seed{task['seed']}"
            command = witness.probe_command(root/'build/probe', case, task['seed'])
            command += ['--index-mode', case['index_mode']]
            command_record(folder, command, 60)
            record = read_json(folder/'stdout')
            require(record['device']['uuid']==environment['uuid'], '分配与probe卡身份不符')
            checked = witness.validate(folder, record, case, task['seed'])
            require(record['index_mode']==case['index_mode'], '索引变体不符')
            print(json.dumps(dict(case_id=case['id'],seed=task['seed'],
                cycles=record['timing']['elapsed_cycles'],eligible=checked['performance_eligible'])),flush=True)
        save_json(root/'score.json', summarize(root))
    print('CLOSED: 12 plain processes; development evidence',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'execute', 'score'):
        item = sub.add_parser(name)
        item.add_argument('--root', type=Path, required=True)
        if name=='prepare':
            item.add_argument('--headers', type=Path, required=True)
            item.add_argument('--round', choices=('first','localize'), default='first')
        if name=='score':
            item.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command=='prepare': prepare(root,args.headers.resolve(),args.round)
    elif args.command=='execute': execute(root)
    else:
        save_json(args.output,summarize(root))
        print('VERIFIED: 12 outputs, receipts, identities, conditions and paired responses')

if __name__=='__main__': main()
