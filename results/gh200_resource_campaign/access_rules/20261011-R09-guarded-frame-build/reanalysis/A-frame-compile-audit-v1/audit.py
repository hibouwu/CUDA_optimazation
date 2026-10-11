#!/usr/bin/env python3
"""Read-only checks of the archived 740070 cfg_b frame; no binary execution."""
import hashlib
import json
from pathlib import Path
import re

out = Path(__file__).resolve().parent
run = out.parents[1]
old = run.parent / '20261010-R09-R13-R18-failure-controls'
inputs = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(data).hexdigest()
    return data


receipt = json.loads(read(run / 'receipt.json'))
returned_archive = read(run / 'output.tar.gz')
assert not receipt['returncode'] and len(returned_archive) == receipt['bytes']
assert hashlib.sha256(returned_archive).hexdigest() == receipt['sha256']
freeze = json.loads(read(run / 'freeze.json'))
for name, digest in freeze['files'].items():
    assert hashlib.sha256(read(run / name)).hexdigest() == digest
    assert hashlib.sha256(read(run / 'returned' / name)).hexdigest() == digest
hashes = json.loads(read(run / 'source_hashes.json'))
for name, digest in hashes.items():
    assert hashlib.sha256(read(run / name)).hexdigest() == digest
interface = json.loads(read(run / 'interface.json'))
assert inputs[str(run / 'source/probes/r09_clock_stamp.cuh')] == interface['candidate']['header_sha256']
commands = json.loads(read(run / 'build/commands.json'))
main = json.loads(read(run / 'build-audit.json'))
for name in ('compiler.txt', 'allocation.txt', 'exit_code.txt'):
    read(run / 'returned' / name)
compiler = (run / 'returned/compiler.txt').read_text()
assert 'release 12.9' in compiler
assert (run / 'returned/exit_code.txt').read_text().strip() == '0'
source_ports = json.loads(read(run / 'source-port-audit.json'))
for name, port in source_ports.items():
    assert read(run / name) == read(old / name)
    assert inputs[str(run / name)] == port['sha256'] and port['exact_same']

artifacts = {}
instructions = {}
symbols = []
for arm in ('cfg_b_plain', 'cfg_b_matched_dual', 'cfg_b_guarded_bracket', 'plain_prefix'):
    binary = 'libplain_prefix.so' if arm == 'plain_prefix' else arm
    binary_hash = hashlib.sha256(read(run / 'returned/build' / binary)).hexdigest()
    sass = read(run / 'returned/build' / (arm + '.sass')).decode()
    sass_hash = hashlib.sha256(sass.encode()).hexdigest()
    assert binary_hash == main['artifacts'][arm]['binary_sha256']
    assert sass_hash == main['artifacts'][arm]['sass_sha256']
    functions = [part for part in sass.split('Function : ')[1:] if part.startswith('_ZN7cutlass13device_kernel')]
    assert len(functions) == 1
    symbol = functions[0].splitlines()[0].strip()
    symbols.append(symbol)
    insns = {}
    for line in functions[0].splitlines():
        match = re.search(r'/\*([0-9a-f]+)\*/\s+([^;]+);', line)
        if match:
            pc = int(match[1], 16)
            assert pc not in insns
            insns[pc] = match[2].strip()
    instructions[arm] = insns
    resources = read(run / 'returned/build' / (arm + '.resources')).decode()
    kernel_resource = resources.split(' Function ' + symbol + ':', 1)[1].splitlines()[1].strip()
    assert 'REG:168 STACK:0 SHARED:1024 LOCAL:0' in kernel_resource
    log = read(run / 'returned/build' / (arm + '.log')).decode()
    log_tail = log.split("Function properties for " + symbol, 1)[1][:400]
    assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in log_tail
    assert 'Used 168 registers' in log_tail and 'used 2 barriers' in log_tail
    ops = {token: sum(token in instruction for instruction in insns.values()) for token in
           ('HGMMA.', 'UTMALDG.', 'UTMASTG.', 'SYNCS.ARRIVE.', 'WARPGROUP.DEPBAR.')}
    assert ops == main['artifacts'][arm]['ops']
    pool = [instruction for instruction in insns.values() if 'SETMAXREG' in instruction]
    assert pool == ['USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xe8', 'USETMAXREG.DEALLOC.CTAPOOL 0x28']
    assert '-DV06_CFG=1' in commands[arm] and '-DR18_STAGE_COUNT=6' in commands[arm]
    artifacts[arm] = dict(binary_sha256=binary_hash, sass_sha256=sass_hash, resource_line=kernel_resource,
                          barriers=2, stack_bytes=0, spill_bytes=0, ops=ops, register_pool=pool)
assert len(set(symbols)) == 1

insns = instructions['cfg_b_guarded_bracket']


def check_site(name, before, timer, after, registers, counter_init, stores):
    cb, gt, ca = registers
    counter = re.search(r'R\d+', insns[before + 0x40]).group()
    # The increment's destination is the retry counter; both init forms occur.
    assert insns[counter_init] == f'IMAD.MOV.U32 {counter}, RZ, RZ, RZ'
    assert insns[before] == f'CS2R {cb}, SR_CLOCKLO'
    for pc, reg in ((before, cb), (timer, gt)):
        assert insns[pc + 0x10] == f'ISETP.NE.U32.AND P1, PT, {reg}, RZ, PT'
        high = 'R' + str(int(reg[1:]) + 1)
        assert insns[pc + 0x20] == f'ISETP.NE.U32.AND.EX P1, PT, {high}, RZ, PT, P1'
        count = re.search(r'R\d+', insns[pc + 0x40]).group()
        assert insns[pc + 0x40] in (f'IADD3 {count}, {count}, 0x1, RZ', f'VIADD {count}, {count}, 0x1')
        assert insns[pc + 0x50] == f'ISETP.GE.U32.AND P1, PT, {count}, 0x2, PT'
        assert insns[pc + 0x60] == f'@!P1 BRA {hex(pc)}'
    assert insns[before + 0x30] == f'@P1 BRA {hex(timer - 0x10)}'
    assert insns[before + 0x70] == f'BRA {hex(timer + 0x70)}'
    assert insns[timer - 0x10] in (f'MOV {ca}, RZ', f'IMAD.MOV.U32 {ca}, RZ, RZ, RZ')
    assert insns[timer] == f'CS2R {gt}, SR_GLOBALTIMERLO'
    assert insns[timer + 0x30] == f'@P1 BRA {hex(after)}'
    for offset, reg in zip((0x70, 0x80, 0x90), registers):
        assert insns[timer + offset] == f'CS2R {reg}, SRZ'
    assert insns[timer + 0xa0] == f'BRA {hex(after + 0x10)}'
    assert insns[after] == f'CS2R {ca}, SR_CLOCKLO'
    assert not any('SEL' in insns[pc] for pc in range(before, after + 0x10, 0x10))
    for pc, expected in stores.items():
        assert insns[pc] == expected
    return dict(event=name, read_pcs=[hex(pc) for pc in (before, timer, after)],
                instructions={hex(pc): insns[pc] for pc in range(before, after + 0x10, 0x10)},
                store_instructions={hex(pc): insns[pc] for pc in stores},
                clock_and_timer_maximum_reads_each=2, exhaustion_outputs='all three uint64 zero',
                final_clock_zero='unretried native value; must be rejected by the reader')


sites = [check_site('FIRST', 0x1a60, 0x1af0, 0x1ba0, ('R18', 'R20', 'R24'), 0x1a40,
                   {0x1c40: 'STG.E.64 desc[UR30][R16.64], R18',
                    0x1c50: 'STG.E.64 desc[UR30][R16.64+0x30], R20',
                    0x1c60: 'STG.E.64 desc[UR30][R16.64+0x50], R24'}),
         check_site('MAIN', 0x24e0, 0x2570, 0x2620, ('R16', 'R18', 'R20'), 0x24d0,
                   {0x26f0: 'STG.E.64 desc[UR30][R154.64+0x88], R16',
                    0x2700: 'STG.E.64 desc[UR30][R154.64+0xb8], R18',
                    0x2710: 'STG.E.64 desc[UR30][R154.64+0xd8], R20'})]
assert insns[0x10] == 'S2R R5, SR_TID.X'
assert insns[0x16b0] == 'LOP3.LUT P0, RZ, R5.reuse, 0x7f, RZ, 0xc0, !PT'
assert insns[0x16d0] == 'ISETP.LT.U32.OR P0, PT, R5, 0x80, P0'
assert insns[0x19f0] == '@P0 BRA 0x1cd0' and insns[0x24c0] == '@P0 BRA 0x2780'
assert not any(re.match(r'\S+\s+P0,', instruction) for pc, instruction in insns.items() if 0x16d0 < pc < 0x24c0)
assert not any(re.match(r'\S+\s+R5,', instruction) for pc, instruction in insns.items() if 0x10 < pc < 0x24c0)

overlay = run / 'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp'
base = run / 'source/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp'
storage = lambda p: p.read_text().split('  // Kernel level shared memory storage', 1)[1].split('  // Device side arguments', 1)[0]
assert storage(overlay) == storage(base)
assert 'static_assert(MaxThreadsPerBlock == 384' in overlay.read_text()
adapter = run / 'source/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h'
text = adapter.read_text()
assert 'int smem_size = GemmKernel::SharedStorageSize;' in text
assert 'GemmKernel::MaxThreadsPerBlock,\n        smem_size' in text
trace = (run / 'source/probes/v06_trace.hpp').read_text()
assert 'constexpr int V06TileWords = 12;' in trace
assert 'if (threadIdx.x % 128 != 0 || threadIdx.x < 128) return;' in trace
assert 'if (event==V06_FIRST_MMA || event==V06_MAIN_END)' in trace
assert 'out[10+event]=after;' in trace and 'uint64_t c, ns, after=0;' in trace
read(out / 'audit.py')
result = dict(schema='gh200-r09-actual-frame-cpu-audit/1', input_sha256=inputs,
              source_files_verified=len(hashes), compiler=compiler, artifacts=artifacts, sites=sites,
              same_kernel_template_symbol=True, same_original_overlay_bytes=True,
              same_shared_storage_definition=True, launch_threads_from_source=384,
              dynamic_smem_input='GemmKernel::SharedStorageSize; same SharedStorage definition and same instantiated types',
              dynamic_smem_numeric_bytes_verified=False, runtime_occupancy_queried=False,
              selector='SASS R5=TID.X; P0=(R5&127)!=0 or R5<128; both read sites skip on P0. With384 threads only128/256 write.',
              compile_interface_checks_passed=True, runtime_qualified=False, observation_qualified=False,
              limits=['Exact checks are specific to these frozen PCs and binaries, not a general SASS validator.',
                      'No binary execution or CUDA driver/runtime calls. Static shared1024 is not the dynamic launch byte count.',
                      'Identical resource inputs and no spill do not prove runtime occupancy or absence of timing perturbation.',
                      'Control flow does not prove hardware sampling enclosure or numerical precision.',
                      'Full reader/source/binary/module/raw identity binding remains a separate obligation; no runtime or thirty-process authorization.'])
target = out / 'audit.json'
if target.exists():
    assert json.loads(target.read_text()) == result
else:
    target.write_text(json.dumps(result, indent=2) + '\n')
print('verified860source files;4kernel resources;2actual guarded control chains;no binary execution')
