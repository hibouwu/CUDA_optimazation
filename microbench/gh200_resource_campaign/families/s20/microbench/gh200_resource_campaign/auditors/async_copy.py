"""S12 finite-case and exact reviewed machine-instruction audit, without I/O.

The opcode/operand digest pins the independently reviewed CUDA 12.9 lowering.
A changed lowering needs fresh independent B2 review; token presence alone does
not prove wait/barrier dataflow or result drain.
"""
from __future__ import annotations
import hashlib
import re
from common.suite_io import require
from auditors.async_copy_sass_baseline import BASELINE


def kernel_symbol(case):
    p = case['parameters']
    return (f'_Z11ac_pipelineILi{p["request_bytes_per_thread"]}ELi{p["stages"]}'
            f'ELb{int(p["ptx"] == "cp.async.cg.shared.global")}EEvPKjiPN2gh5StampEPjS5_S5_')


def case_identity(case):
    require(isinstance(case, dict) and isinstance(case.get('parameters'), dict), 'case/parameters')
    p = case['parameters']
    width, stages = p.get('request_bytes_per_thread'), p.get('stages')
    require(type(width) is int and width in (4, 8, 16) and type(stages) is int and stages in (1, 2, 4), 'finite width/stages')
    ptx = p.get('ptx')
    require(ptx in ('cp.async.ca.shared.global', 'cp.async.cg.shared.global') and (width == 16 or '.ca.' in ptx), 'cache/width')
    scope = case.get('scope')
    require(scope in ('one_cta', 'all_gpu') and case.get('id') == f'{"cg" if ".cg." in ptx else "ca"}_w{width}_stages{stages}_{scope}', 'case identity')
    require(type(case.get('threads')) is int and case['threads'] == 128 and type(case.get('iterations')) is int and case['iterations'] == 8192, 'original participants/iterations')
    require(case.get('launch') == ({'kind': 'one_cta'} if scope == 'one_cta' else {'kind': 'sms_capped_occupancy', 'maximum_ctas_per_sm': 4}), 'launch geometry')
    require(p == {'ptx': ptx, 'request_bytes_per_thread': width, 'stages': stages,
                  'copies_per_thread_iteration': 8, 'requests_per_commit_group': 1,
                  'global_array_bytes': 8388608, 'consumer_thread': '(thread+1)%128',
                  'shared_payload_bytes': 128*width*stages,
                  'source_alignment_bytes': width, 'destination_alignment_bytes': width,
                  'input_word': 'uint32(17*word_index+seed)',
                  'consumer': 'read every uint32 word copied by neighbor and accumulate modulo2^32'}, 'frozen S12 input/lifecycle')
    require(all(type(p[k]) is int for k in ('copies_per_thread_iteration','requests_per_commit_group','global_array_bytes','shared_payload_bytes','source_alignment_bytes','destination_alignment_bytes')), 'parameter integer types')
    require(case.get('capabilities') == {'cc': '9.0', 'device_name_contains': 'GH200'}, 'target capabilities')
    return p


def validate_device(device):
    require(isinstance(device, dict), 'device object')
    require(type(device.get('schema_version')) is int and device['schema_version'] == 2 and device.get('type') == 'device', 'device schema')
    require(device.get('cc') == '9.0' and isinstance(device.get('name'), str) and 'GH200' in device['name'], 'GH200 SM90')
    require(isinstance(device.get('uuid'), str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', device['uuid']), 'GPU UUID')
    for name in ('sms', 'driver_version', 'registers_per_sm', 'smem_per_sm_bytes', 'smem_per_cta_optin_bytes'):
        require(type(device.get(name)) is int and device[name] > 0, 'device integer capacity: '+name)
    require(device['sms'] <= 512 and device.get('runtime_version') == 12090, 'bounded GH200/CUDA12.9')
    return device


def instructions(body):
    result = []
    for line in body.splitlines():
        m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;\s*/\*', line)
        if m:
            result.append((int(m[1],16), ' '.join(m[2].split())))
    require(result and len({pc for pc,_ in result}) == len(result), 'nonempty unique instruction PCs')
    return result


def audit_sass(text, contract):
    require(isinstance(text, str) and isinstance(contract, dict) and contract.get('family') == 'async_copy', 'S12 SASS/contract')
    functions = re.split(r'Function\s*:\s*', text)[1:]
    out = []
    for case in contract['cases']:
        p = case_identity(case)
        symbol = kernel_symbol(case)
        matches = [part for part in functions if part.splitlines()[0].strip() == symbol]
        require(len(matches) == 1, 'missing/ambiguous S12 symbol '+symbol)
        ops = instructions(matches[0])
        normalized = '\n'.join(f'{pc:04x}:{op}' for pc,op in ops)
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        require(digest == BASELINE[symbol]['instructions_sha256'], 'S12 instruction stream differs from independently reviewed B2 baseline: '+symbol)
        # Human-readable service boundaries supplement, not replace, exact identity.
        timers = [pc for pc,op in ops if 'SR_GLOBALTIMERLO' in op]
        copies = [(pc,op) for pc,op in ops if re.search(r'\bLDGSTS(?:\.|\s)', op)]
        commits = [pc for pc,op in ops if op.startswith('LDGDEPBAR')]
        waits = [(pc,op) for pc,op in ops if op.startswith('DEPBAR.LE')]
        barriers = [pc for pc,op in ops if op.startswith('BAR.SYNC')]
        require(len(timers) == 2 and len(copies) == p['stages']+1 and len(commits) == len(copies), 'copy/commit/timer count')
        require(all(timers[0] < pc < timers[1] for pc,_ in copies), 'timed copy instructions')
        require(all(('.BYPASS' in op) == ('.cg.' in p['ptx']) for _,op in copies), 'cache policy lowering')
        require(waits and barriers, 'wait/CTA synchronization')
        out.append({'case_id':case['id'], 'symbol':symbol, 'instructions_sha256':digest,
                    'instruction_count':len(ops), 'copy_pcs':[pc for pc,_ in copies],
                    'commit_pcs':commits, 'wait_pcs':[pc for pc,_ in waits],
                    'barrier_pcs':barriers, 'timer_pcs':timers,
                    'baseline_review':'reviews/S12-source-pre-review-r3.json'})
    require(out, 'empty S12 case set')
    return out
