#!/usr/bin/env python3
"""Check the returned fixed CUDA12.9 observer's fast/slow SASS paths."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules')
RUN = ROOT / '20261011-R13-early-wait-sum-build'
PREPARE = RUN / 'attempts/prepare3'
RETURNED = PREPARE / 'returned'
OUTPUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_kernel(path):
    active = False
    instructions = {}
    function = None
    for line in path.read_text().splitlines():
        if 'Function :' in line:
            if active:
                break
            if '_ZN7cutlass13device_kernel' in line:
                active = True
                function = line.split('Function :', 1)[1].strip()
        if active and (match := re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*', line)):
            address = int(match[1], 16)
            assert address not in instructions
            instructions[address] = match[2]
    assert function and instructions
    return function, instructions


def branch_destination(instruction):
    match = re.search(r'\bBRA (0x[0-9a-f]+)$', instruction)
    assert match, instruction
    return int(match[1], 16)


def main():
    source_manifest = PREPARE / 'source_hashes.json'
    assert source_manifest.read_bytes() == (RETURNED / 'source_hashes.json').read_bytes()
    hashes = json.loads(source_manifest.read_text())
    for name, expected in hashes.items():
        assert sha(PREPARE / name) == expected, name
    assert (PREPARE / 'build/commands.json').read_bytes() == (RETURNED / 'build/commands.json').read_bytes()
    assert (RETURNED / 'exit_code.txt').read_text().strip() == '0'
    main_receipt = json.loads((PREPARE / 'receipt.json').read_text())
    assert sha(PREPARE / 'output.tar.gz') == main_receipt['sha256']
    input_receipt = json.loads((PREPARE / 'input-receipt.json').read_text())
    assert sha(PREPARE / 'input.tar.gz') == input_receipt['sha256']
    allocation = (RETURNED / 'allocation.txt').read_text()
    assert 'AllocTRES=cpu=4,mem=16G,node=1,billing=4' in allocation
    assert 'JOB_GRES=(null)' in allocation
    assert 'release 12.9, V12.9.41' in (RETURNED / 'compiler.txt').read_text()
    input_sha = {str(path):sha(path) for path in [source_manifest,
        PREPARE / 'input.tar.gz', PREPARE / 'output.tar.gz', PREPARE / 'input-receipt.json',
        PREPARE / 'receipt.json', RETURNED / 'exit_code.txt', RETURNED / 'allocation.txt',
        RETURNED / 'compiler.txt', RETURNED / 'build/commands.json', RUN / 'build-audit.json']}
    records = []
    archived_files_checked = 0
    with tarfile.open(PREPARE / 'output.tar.gz') as archive:
        for name, stages in [('cfg_b_s4_stamped', 4), ('cfg_b_stamped', 6)]:
            paths = [RETURNED / 'build' / (name + suffix) for suffix in ['', '.sass', '.resources', '.log']]
            for path in paths:
                relative = str(path.relative_to(RETURNED))
                members = [member for member in archive.getmembers() if member.name.lstrip('./') == relative]
                assert len(members) == 1, relative
                assert hashlib.sha256(archive.extractfile(members[0]).read()).hexdigest() == sha(path)
                input_sha[str(path)] = sha(path)
                archived_files_checked += 1
            function, instructions = read_kernel(paths[1])
            assert f'ILi{stages}E' in function
            starts = [address for address, instruction in instructions.items()
                      if instruction == '@P2 CS2R R176, SR_CLOCKLO']
            assert len(starts) == 1
            H = starts[0]
            assert instructions[H-0x20] == 'ISETP.LT.U32.AND P2, PT, R176, 0xf, P2'
            assert instructions[H-0x40] == 'ISETP.NE.AND P2, PT, RZ, UR29, P1'
            first = H+0x30
            second = H+0x50
            assert instructions[first].startswith('SYNCS.PHASECHK.TRANS64.TRYWAIT P3,')
            assert instructions[second] == instructions[first]
            assert instructions[first+0x10].startswith('@P3 BRA ')
            join = branch_destination(instructions[first+0x10])
            assert join == H+0x70
            assert instructions[second+0x10].startswith('@!P3 BRA ')
            slow = branch_destination(instructions[second+0x10])
            assert instructions[join].startswith('BSSY B1, ')
            assert instructions[join+0x10].startswith('@!P2 BRA ')
            collector_end = branch_destination(instructions[join+0x10])
            B = join+0x20
            assert instructions[B] == 'CS2R R178, SR_CLOCKLO'
            assert instructions[slow+0x20].startswith('SYNCS.PHASECHK.TRANS64.TRYWAIT P3,')
            assert instructions[slow+0x30].startswith('@!P3 NANOSLEEP.SYNCS ')
            assert instructions[slow+0x40].startswith('@!P3 SYNCS.PHASECHK.TRANS64 P3,')
            assert branch_destination(instructions[slow+0x50]) == slow+0x10
            assert instructions[slow+0x50].startswith('@!P3 BRA ')
            assert branch_destination(instructions[slow+0x60]) == join
            assert instructions[collector_end] == 'BSYNC B1'
            assert instructions[collector_end+0x20] == 'WARPGROUP.ARRIVE'
            collector = [(address, instruction) for address, instruction in instructions.items()
                         if B <= address <= collector_end]
            assert len(collector) == 38
            assert not any(re.search(r'\b(?:STG|STL|STS|LDL)(?:\.|\s)', instruction)
                           for _, instruction in collector)
            assert not any('SR_CLOCKLO' in instruction or 'HGMMA' in instruction or
                           'SYNCS.PHASECHK' in instruction
                           for address, instruction in collector if address > B)
            selected_control = [(address, instruction) for address, instruction in instructions.items()
                                if H <= address <= B or slow <= address <= slow+0x60]
            assert not any(re.search(r'\b(?:STG|STL|STS|HGMMA)(?:\.|\s)', instruction)
                           for _, instruction in selected_control)
            resources = paths[2].read_text()
            assert 'REG:168 STACK:0 SHARED:1024 LOCAL:0' in resources
            log = paths[3].read_text()
            spill_records = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
            assert spill_records and all(row == ('0','0','0') for row in spill_records)
            assert sum('HGMMA.' in instruction for instruction in instructions.values()) == 16
            excerpts = [(address, instruction) for address, instruction in instructions.items()
                        if H-0x50 <= address <= collector_end+0x20 or slow <= address <= slow+0x60]
            (OUTPUT / f'stage{stages}-instructions.txt').write_text(
                '\n'.join(f'{address:#06x} {instruction}' for address, instruction in excerpts)+'\n')
            records.append(dict(name=name, stages=stages, H=hex(H), first_try=hex(first),
                second_try=hex(second), fast_join=hex(join), slow_entry=hex(slow),
                slow_try=hex(slow+0x20), slow_wait=hex(slow+0x40),
                slow_retry=hex(slow+0x10), slow_return=hex(slow+0x60), B=hex(B),
                collector_end=hex(collector_end), next_warpgroup_arrive=hex(collector_end+0x20),
                static_B_through_collector_end_instructions=38,
                selected_uses_uint32_group_minus2_less15=True,
                both_success_paths_reach_B_after_try_wait=True,
                retry_path_loops_until_success_then_returns_before_B=True,
                collector_after_B_has_no_global_local_shared_store=True,
                GEMM_registers=168, static_HGMMA_instructions=16,
                zero_stack_and_local_resource=True, all_logged_kernels_zero_spill=True))
    audit = json.loads((RUN / 'build-audit.json').read_text())
    for record, previous in zip(records, audit['binaries']):
        assert record['name'] == previous['name']
        assert record['H'] == previous['clock_start'] and record['B'] == previous['clock_end']
        assert record['first_try'] == previous['barrier_fast_test']
        assert record['second_try'] == previous['barrier_wait_test']
        assert sha(RETURNED / 'build' / record['name']) == previous['binary_sha256']
    report = dict(source=str(PREPARE), input_sha256=input_sha,
        source_files_verified=len(hashes), returned_archive_files_verified=archived_files_checked,
        CUDA='12.9.41', job=740240, recorded_allocation_TRES='cpu=4,mem=16G,node=1,billing=4',
        records=records, findings=[], main_SASS_address_audit_independently_matched=True,
        interpretation=[
            'Compiled selected H precedes the first full try/test. Fast success bypasses the second try; slow failure retries until success. Both paths join before selected B.',
            'Q includes after-H barrier address/phase preparation, try/wait control, and before-B branch/join work, plus observer clock effects. It is not physical ready or isolated TMA latency.',
            'The B-to-collector-end listing has 38 static instructions including B. Predicated checks and overflow branches mean this is not a fixed executed count or cycle cost.',
            'Collector branches reconverge before WARPGROUP.ARRIVE. Arithmetic/reconvergence lies in the residual and can change later waiting and warpgroup progress.',
            'REG168 and zero spill do not establish equal occupancy, scheduling, overhead, or equivalence to the old binary.'
        ],
        compiler_order_review_passed=True, SASS_reviewed=True, B_GPU_SSH_Slurm_executed=False,
        hardware_latency_or_ordering_certificate=False, hardware_correctness=False,
        observation_qualified=False, service_identified=False, prediction_qualified=False)
    path = OUTPUT / 'B-q-early-sass-review.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    receipt = dict(report_sha256=sha(path), script_sha256=sha(Path(__file__)),
        source_files_verified=len(hashes), returned_archive_files_verified=archived_files_checked,
        reviewed_binary_stages=[4,6], findings=[])
    (OUTPUT / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (OUTPUT / 'README.md').write_text('''固定 prepare3 的 CUDA12.9.41 返回物已核对：857 个 source 文件与返回 manifest 一致，两个 binary/SASS/resources/log 均与返回归档逐字匹配。job740240 的记录分配为 4 CPU、16 GiB、无 GRES，exit code 0。B 只读本地返回物，没有 GPU、SSH/Slurm 操作。

| stages | H | 首次try | 再次try | 慢路径入口/返回 | B | collector末尾 |
|---|---|---|---|---|---|---|
| 4 | 0x1e10 | 0x1e40 | 0x1e60 | 0x5830 / 0x5890→0x1e80 | 0x1ea0 | 0x20f0 |
| 6 | 0x1d10 | 0x1d40 | 0x1d60 | 0x5770 / 0x57d0→0x1d80 | 0x1da0 | 0x1ff0 |

H 后才计算 barrier 地址/phase 并 try；首次成功直接跳到汇合点，再次失败进入慢路径。慢路径执行 try、必要时 NANOSLEEP/PHASECHK，失败重试，成功返回汇合点；选中线程随后读取 B。H 的谓词包含目标线程/输出、开关以及 unsigned(group−2)<15，即 local2～16。两条路径都未把 barrier 检查提前到 H 之前，累计算术均在 B 之后。原始指令见 [stage4](stage4-instructions.txt)、[stage6](stage6-instructions.txt)。

B 至 collector 末尾有 38 条静态指令，包含 B；校验谓词和溢出分支会改变实际执行条数，不能把 38 当固定周期成本。该区域没有 global/local/shared store，在下一次 WARPGROUP.ARRIVE 前汇合；Q 仍含地址/phase 准备、try/wait 控制、汇合和时钟影响。record 与汇合开销归到残余，但会改变后续等待和整个 WG 推进。

两组 GEMM 均 REG168、STACK0、LOCAL0；构建日志中的所有 kernel spill 均为零，静态 HGMMA 各16条。资源一致和 SASS 顺序只能支持编译布局与源码分段，不能证明实际延迟、无扰动、旧 binary 等价或模型资格。主线的指令地址审查已独立匹配；详见 [报告](B-q-early-sass-review.json)。
''')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
