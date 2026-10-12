"""CPU preflight or execute the exact 48-condition m11 lane panel."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

from r03_lane_contrast_model import validate_plan


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(package):
    for manifest, folder in [('source-sha256.json', 'source'),
                             ('cutlass-closure-sha256.json', 'cutlass-closure')]:
        for name, expected in json.loads((package / manifest).read_text()).items():
            if sha(package / folder / name) != expected:
                raise ValueError('historical source mismatch: ' + name)
    if (package / 'execution-sha256.json').exists():
        for name, expected in json.loads((package / 'execution-sha256.json').read_text()).items():
            if sha(package / 'execution-source' / name) != expected:
                raise ValueError('execution source mismatch: ' + name)
    identity = json.loads((package / 'identity.json').read_text())
    if sha(package / 'historical-build/smem') != identity['binary_sha256']:
        raise ValueError('historical binary mismatch')
    if sha(package / 'historical-build/smem.sass') != identity['historical_sass_sha256']:
        raise ValueError('historical SASS does not match reviewed manifest')
    if sha(package / 'protocol.json') != identity['protocol_sha256']:
        raise ValueError('historical protocol mismatch')
    if sha(package / 'minimal-measurement-cases.json') != identity['cases_sha256']:
        raise ValueError('prelisted case identity mismatch')
    cases = json.loads((package / 'minimal-measurement-cases.json').read_text())
    return cases, validate_plan(cases), identity


def allocated_gpu_uuid(output, expected):
    # Inspect Slurm's effective step allocation before changing visibility.
    visible_before = os.environ.get('CUDA_VISIBLE_DEVICES')
    allocated = os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS')
    if not allocated:
        raise ValueError('cannot prove effective GPU allocation')
    inventory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    index_to_uuid = {}
    for line in inventory.splitlines():
        index, uuid = line.split(',')
        index_to_uuid[index.strip()] = uuid.strip()
    selected = []
    for token in allocated.split(','):
        token = token.strip()
        if token.startswith('GPU-'):
            selected.append(token)
        elif token.isdigit():
            selected.append(index_to_uuid[token])
        else:
            raise ValueError('unrecognized Slurm GPU allocation: ' + token)
    proof = dict(SLURM_JOB_GPUS=os.environ.get('SLURM_JOB_GPUS'),
                 SLURM_STEP_GPUS=os.environ.get('SLURM_STEP_GPUS'),
                 CUDA_VISIBLE_DEVICES_before=visible_before, inventory=inventory,
                 effective_allocated_uuids=selected, required_uuid=expected,
                 required_uuid_allocated=expected in selected)
    write(output / 'allocation-proof.json', proof)
    if expected not in selected:
        raise ValueError('required historical GPU is not allocated')
    os.environ['CUDA_VISIBLE_DEVICES'] = expected


def synchronization_audit(path):
    rows = []
    for block in path.read_text().split('Function : ')[1:]:
        name = block.splitlines()[0].strip()
        match = re.match(r'_Z10smem_probeILi([012])', name)
        if not match:
            continue
        op = int(match[1])
        instructions = []
        for line in block.splitlines():
            parsed = re.search(r'/\*([0-9a-f]+)\*/\s+(.*?);', line)
            if parsed:
                instructions.append((int(parsed[1], 16), parsed[2]))
        loops = []
        for pc, text in instructions:
            branch = re.search(r'\bBRA\s+(?:0x)?([0-9a-f]+)', text)
            if not branch or int(branch[1], 16) >= pc:
                continue
            begin = int(branch[1], 16)
            body = [text for at, text in instructions if begin <= at <= pc]
            accesses = sum(bool(re.search(r'\b(?:LDS|STS)(?:\.|\s)', text)) for text in body)
            if accesses == (0 if op == 2 else 8):
                loops.append(dict(begin_pc=begin, end_pc=pc,
                                  CTA_barriers=sum('BAR.SYNC' in text for text in body),
                                  warp_sync=sum('WARPSYNC' in text or 'BAR.WARP' in text for text in body)))
        barriers = [text for _, text in instructions if 'BAR.SYNC' in text]
        if len(loops) != 1 or len(barriers) != 3 or any(not text.startswith('BAR.SYNC') for text in barriers):
            raise ValueError('unqualified repeated loop or CTA boundary synchronization: ' + name)
        if loops[0]['CTA_barriers'] or loops[0]['warp_sync']:
            raise ValueError('actual target/control synchronization structure changed: ' + name)
        rows.append(dict(kernel=name, op=op, total_CTA_barriers=3,
                         source_warp_sync_not_materialized=True, **loops[0]))
    if len(rows) != 14:
        raise ValueError('fourteen target/control synchronization structures required')
    return rows


def main(package, output, cpu_only, audit_existing=False):
    cases, pairs, identity = validate_inputs(package)
    sys.path.insert(0, str(package / 'source'))
    from run_r03_lds_sts import build, command, run_case, sass_audit
    if not audit_existing:
        output.mkdir(parents=True, exist_ok=False)
    write(output / 'address-invariants.json', pairs)
    if cpu_only:
        version = subprocess.check_output(['nvcc', '--version'], text=True)
        if 'V12.9.41' not in version:
            raise ValueError('exact compiler V12.9.41 required')
        if audit_existing:
            binaries = output / 'build'
            if 'V12.9.41' not in (binaries / 'nvcc-version.txt').read_text():
                raise ValueError('existing build compiler identity mismatch')
            for name in ('smem', 'exp09', 'r03'):
                if json.loads((binaries / ('compile-' + name + '.process.json')).read_text())['returncode']:
                    raise ValueError('existing full-source compilation was not successful')
        else:
            binaries = build(package, output, package / 'cutlass-closure')
        audit = json.loads((binaries / 'sass-audit.json').read_text())
        if len(audit) != 11 or any(row['status'] != 'verified_target_loop' for row in audit):
            raise ValueError('all ordinary and native SASS windows must pass')
        source_log = (binaries / 'compile-smem.stderr').read_text()
        properties = re.findall(r'Function properties for (_Z10smem_probe[^\n]+)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', source_log)
        if len(properties) != 14 or any(int(stack) or int(stores) or int(loads) for _, stack, stores, loads in properties):
            raise ValueError('target/control stack or spill gate failed')
        actual_sass = command(['cuobjdump', '--dump-sass', str(package / 'historical-build/smem')],
                              output, 'disassemble-historical-ARM')
        (output / 'historical-ARM.sass').write_text(actual_sass)
        historical_targets = sass_audit(output / 'historical-ARM.sass')
        if any(row['status'] != 'verified_target_loop' for row in historical_targets):
            raise ValueError('actual reused ARM binary target loop failed')
        cpu_sync = synchronization_audit(binaries / 'smem.sass')
        historical_sync = synchronization_audit(output / 'historical-ARM.sass')
        write(output / 'synchronization-coverage.json', dict(
            CPU=cpu_sync, historical_ARM=historical_sync,
            target_and_control_boundaries_match=True,
            source_per_round_warp_sync_not_materialized=True,
            control_is_loop_consumer_and_CTA_boundary_combination=True,
            isolated_warp_sync_cost_not_measured=True))
        control_windows = [row['kernel'] for row in cpu_sync if row['op'] == 2]
        write(output / 'result.json', dict(status='pass', target_kernels=11,
              target_and_control_no_spill=14, combined_control_kernels=control_windows,
              combined_control_not_isolated_sync_cost=True, address_pairs=len(pairs),
              gpu_execution=False, compiler=(binaries / 'nvcc-version.txt').read_text(),
              historical_binary_sha256=identity['binary_sha256'],
              historical_binary_directly_disassembled=True,
              target_and_control_actual_sync_structure_match=True,
              source_per_round_warp_sync_not_materialized=True,
              protocol_sha256=identity['protocol_sha256']))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm allocation required')
    preflight = json.loads((package / 'cpu-preflight/result.json').read_text())
    if preflight['status'] != 'pass':
        raise ValueError('CPU preflight is not qualified')
    try:
        allocated_gpu_uuid(output, identity['gpu_uuid'])
        binary = package / 'historical-build/smem'
        audit = sass_audit(package / 'historical-build/smem.sass')
        if any(row['status'] != 'verified_target_loop' for row in audit):
            raise ValueError('historical SASS window gate failed')
        device = json.loads(command([str(binary), 'device'], output, 'device'))
        if device['gpu_uuid'] != identity['gpu_uuid']:
            raise ValueError('CUDA probe UUID mismatch')
        write(output / 'environment.json', dict(**device, host=os.uname().nodename,
              job=os.environ['SLURM_JOB_ID'], binary_sha256=sha(binary),
              protocol_sha256=sha(package / 'protocol.json'),
              CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
        with open('/tmp/gh200-access-' + device['gpu_uuid'] + '.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            failures = []
            recovery_used = False
            rng = random.Random(2026101130)
            for trial in range(3):
                ordered = cases.copy()
                rng.shuffle(ordered)
                for case in ordered:
                    folder = output / 'samples' / case['id'] / f'trial-{trial:02}'
                    folder.mkdir(parents=True)
                    settled = None
                    for attempt in range(2):
                        target = folder / f'attempt-{attempt:02}'
                        try:
                            run_case(binary, case, target)
                            settled = dict(status='measured', attempt=attempt,
                                           result=f'attempt-{attempt:02}/result.json')
                            break
                        except Exception as error:
                            failure = dict(case=case['id'], trial=trial, attempt=attempt,
                                           error=str(error), type=type(error).__name__)
                            failures.append(failure)
                            if target.exists():
                                write(target / 'failure.json', failure)
                            if isinstance(error, subprocess.TimeoutExpired) and not recovery_used:
                                recovery_used = True
                                continue
                            settled = dict(status='failed', failure=failure)
                            break
                    write(folder / 'completion.json', settled)
                    print(case['id'], trial, settled['status'], flush=True)
            write(output / 'failures.json', failures)
            write(output / 'completion.json', dict(prelisted_conditions=48,
                  primary_processes=144, actual_process_attempts=144 + int(recovery_used),
                  recovery_used=recovery_used, failures_retained=len(failures)))
    except Exception as error:
        write(output / 'fatal.json', dict(error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--audit-existing', action='store_true')
    args = parser.parse_args()
    main(args.package.resolve(), args.output.resolve(), args.cpu_only, args.audit_existing)
