#!/usr/bin/env python3
"""Read-only replay of the two compiled result-dependent GT/clock/GT paths."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


def review(run, base, prior, output):
    if output.exists():
        raise ValueError('output must be a new C-owned file')
    inputs = {}

    def read(path):
        data = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data

    def load(path):
        return json.loads(read(path))

    freeze = load(run / 'freeze.json')
    for name, expected in freeze.items():
        assert hashlib.sha256(read(run / name)).hexdigest() == expected, name
    manifest = load(run / 'source_hashes.json')
    old = load(base / 'source_hashes.json')
    assert len(old) == 856 and len(manifest) == 858
    changed = sorted(name for name in old if old[name] != manifest[name])
    added = sorted(set(manifest) - set(old))
    assert changed == ['source/probes/r18.cu', 'source/probes/v06_trace.hpp']
    assert added == ['source/prepare_cfg_c_gt_bracket.py', 'source/probes/cfg_c_gt_bracket.cuh']
    with tarfile.open(run / 'input.tar.gz') as archive:
        files = {m.name.removeprefix('./'): m for m in archive.getmembers() if m.isfile()}
        for name, expected in manifest.items():
            data = read(run / name)
            assert hashlib.sha256(data).hexdigest() == expected, name
            assert archive.extractfile(files[name]).read() == data, name
        for name in ('run.sh', 'source_hashes.json', 'contract.json', 'build/commands.json'):
            assert archive.extractfile(files[name]).read() == read(run / name), name

    receipt = load(run / 'receipt.json')
    assert receipt['returncode'] == 0
    assert hashlib.sha256(read(run / 'output.tar.gz')).hexdigest() == receipt['sha256']
    with tarfile.open(run / 'output.tar.gz') as archive:
        output_files = {m.name.removeprefix('./'): m for m in archive.getmembers() if m.isfile()}
        for name, member in output_files.items():
            assert archive.extractfile(member).read() == read(run / 'returned' / name), name
    assert (run / 'returned/exit_code.txt').read_text().strip() == '0'
    assert read(run / 'returned/source_hashes.json') == read(run / 'source_hashes.json')
    allocation = read(run / 'returned/allocation.txt').decode()
    assert 'JobId=740156 ' in allocation and 'JOB_GRES=(null)' in allocation
    assert 'AllocTRES=cpu=4,mem=16G,node=1,billing=4' in allocation
    compiler = read(run / 'returned/compiler.txt').decode()
    assert 'release 12.9, V12.9.41' in compiler

    contract = load(run / 'contract.json')
    assert contract['base_manifest_sha256'] == inputs[str((base/'source_hashes.json').resolve())]
    assert contract['shadow_bytes'] == 132*9*8 == 9504
    assert contract['prefix_calls'] == 16 and contract['prefix_flag'] == 0
    assert contract['events'] == ['FIRST_MMA', 'MAIN_END']
    header = read(run / 'source/probes/v06_trace.hpp').decode()
    assert read(run/'source/overlay/v06_trace.hpp').decode() == '#pragma once\n#include "../probes/v06_trace.hpp"\n'
    for needle in ('out[event] = c;', 'out[6 + event] = ns;',
                   'shadow[((role-1)*2+tile)*2+event]=before;',
                   'if (threadIdx.x % 128 != 0 || threadIdx.x < 128) return;'):
        assert needle in header, needle
    slots = [((r-1)*2+t)*2+e for r in (1,2) for t in (0,1) for e in (0,1)]
    assert slots == list(range(8))
    helper = read(run / 'source/probes/cfg_c_gt_bracket.cuh').decode()
    cuts = ['mov.u64 before,%%globaltimer;',
            'setp.eq.u64 valid,before,0; @valid bra CFG_C_GT_INVALID;',
            'mov.u64 cycles,%%clock64;',
            'setp.eq.u64 valid,cycles,0; @valid bra CFG_C_GT_INVALID;',
            'mov.u64 after,%%globaltimer; bra CFG_C_GT_DONE;',
            'CFG_C_GT_INVALID: mov.u64 before,0; mov.u64 cycles,0; mov.u64 after,0;',
            'CFG_C_GT_DONE: mov.u64 %0,before; mov.u64 %1,cycles; mov.u64 %2,after;']
    indices = [helper.index(needle) for needle in cuts]
    assert indices == sorted(indices)
    # All local PTX outputs are assigned on both paths before the final moves.
    assert ': "=l"(stamp.before), "=l"(stamp.cycles), "=l"(stamp.after)' in helper

    sass = read(run / 'returned/build/cfg_c_s4_dual.sass').decode()
    section = sass.split('Function : _ZN7cutlass13device_kernel', 1)[1].split('Function :', 1)[0]
    instructions = {int(m[1],16):m[2].strip() for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;', section)}
    chains = []
    for event, start, invalid, done in [('FIRST',0x1790,0x1820,0x1850), ('MAIN',0x2450,0x24e0,0x2510)]:
        expected = ['CS2R R12, SR_GLOBALTIMERLO',
                    'ISETP.NE.U32.AND P3, PT, R12, RZ, PT',
                    'ISETP.NE.U32.AND.EX P3, PT, R13, RZ, PT, P3',
                    f'@!P3 BRA {hex(invalid)}', 'CS2R R14, SR_CLOCKLO',
                    'ISETP.NE.U32.AND P3, PT, R14, RZ, PT',
                    'ISETP.NE.U32.AND.EX P3, PT, R15, RZ, PT, P3',
                    '@P3 CS2R R16, SR_GLOBALTIMERLO', f'@P3 BRA {hex(done)}',
                    'CS2R R12, SRZ', 'CS2R R14, SRZ', 'CS2R R16, SRZ']
        observed = [instructions[start+16*i] for i in range(len(expected))]
        assert observed == expected, (event, observed)
        chains.append(dict(event=event, instructions=[dict(pc=hex(start+16*i), text=x) for i,x in enumerate(observed)]))
    log = read(run/'returned/build/cfg_c_s4_dual.log').decode()
    assert 'Used 168 registers, used 2 barriers' in log
    assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in log

    old_protocol = read(prior/'protocol.txt')
    assert hashlib.sha256(old_protocol).hexdigest() == '29d3dd5384118bb3a872b88a6ec10d0965bd2cc01224c5ef1ca9dfcfa818227e'
    read(prior/'report.json')
    read(Path(__file__))
    result = dict(schema='C-compiled-GT-ordering-review/1', input_sha256=inputs,
        base_source_files=856, unchanged_base_source_files=854, changed=changed, added=added,
        input_source_files_verified=len(manifest), returned_archive_files_verified=len(output_files),
        build_job=740156, build_has_gpu_allocation=False,
        binary_sha256=inputs[str((run/'returned/build/cfg_c_s4_dual').resolve())],
        sass_sha256=inputs[str((run/'returned/build/cfg_c_s4_dual.sass').resolve())],
        compiled_dependency_chains=chains, shadow_slots=slots, shadow_bytes=9504,
        source_and_sass_dependency_review_passed=True,
        conditional_output_constraint_issue=False,
        assumptions_required=['Actual special-register capture honors the reviewed dependency/control path.',
                              'globaltimer is a monotone target-specific reference counter in the same invocation.'],
        qualification=dict(hardware_sampling_enclosure=False, physical_ns_accuracy=False,
                           observation=False, observer_perturbation=False, null_control=False,
                           lifecycle_prediction=False, whole_gemm=False),
        C_new_gpu_processes=0, public_probe_modified_by_C=False, prior_v1_overwritten=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(inputs=len(inputs), sources=len(manifest), chains=len(chains),
                         binary_sha256=result['binary_sha256'], sass_sha256=result['sass_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--prior',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    review(args.run.resolve(),args.base.resolve(),args.prior.resolve(),args.output.resolve())
