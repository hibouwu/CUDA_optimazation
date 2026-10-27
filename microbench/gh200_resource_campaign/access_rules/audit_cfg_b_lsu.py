#!/usr/bin/env python3
"""Audit ordinary LSU demand of the two frozen cfg_b plain builds, CPU only.

This is a deliberately fixed-build audit. Known PC fragments are checked rather
than interpreted by a general SASS emulator. No binary or GPU program is run.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


FROZEN = Path('results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011')
MANIFEST_SHA = '09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab'
LOG_SHA = {
    'cfg_b_plain': 'cb7d5bca3d5e98814f141149deb93a13da83596ae1e90701cc7545118eb0b1fe',
    'cfg_b_s4_plain': 'ee3356def500cad763d8687322bb9e36f82420406d3f87f2feb491b6ba8a2fcb',
}
SOURCE_PATHS = {
    'probe': 'source/probes/r18.cu',
    'callbacks': 'source/cutlass/include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp',
    'visitor': 'source/cutlass/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp',
}
EXPECTED_COUNTS = {
    'cfg_b_plain': {'LDC': 36, 'LDC.64': 8, 'ULDC': 21, 'ULDC.64': 14},
    'cfg_b_s4_plain': {'LDC': 6, 'LDC.64': 3, 'ULDC': 51, 'ULDC.64': 19},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path, line, **fields):
    return dict(path=path, line=line, **fields)


def fragment(root, path, pattern, start=0):
    text = (root / path).read_text()
    match = re.search(pattern, text[start:], re.S)
    require(match is not None, 'missing source fragment: ' + path + ': ' + pattern)
    begin, end = start + match.start(), start + match.end()
    return reference(path, text.count('\n', 0, begin) + 1,
                     end_line=text.count('\n', 0, end) + 1, text=text[begin:end]), match


def identity(root):
    manifest_path = root / 'frozen/manifest.json'
    require(sha(manifest_path) == MANIFEST_SHA, 'changed frozen manifest')
    manifest = json.loads(manifest_path.read_text())
    verified = {}

    def check(path, digest, basis):
        actual = sha(root / path)
        require(actual == digest, 'changed frozen artifact: ' + path)
        verified[path] = dict(sha256=actual, basis=basis)

    for path in ('source_hashes.json', 'build/binary_hashes.json', 'build/sass_hashes.json',
                 'build/commands.json', 'build/resources.json', 'build/nvcc-version.txt'):
        check(path, manifest['files'][path], 'frozen/manifest.json')
    sources = json.loads((root / 'source_hashes.json').read_text())
    binaries = json.loads((root / 'build/binary_hashes.json').read_text())
    sass = json.loads((root / 'build/sass_hashes.json').read_text())
    for path in SOURCE_PATHS.values():
        check(path, sources[path], 'source_hashes.json')
    for name in EXPECTED_COUNTS:
        check('build/' + name, binaries['build/' + name], 'build/binary_hashes.json')
        check('build/' + name + '.sass', sass[name + '.sass'], 'build/sass_hashes.json')
        # These logs were not members of the original frozen manifest.
        check('build/' + name + '.log', LOG_SHA[name], 'explicit digest in audit_cfg_b_lsu.py')
    return dict(manifest_sha256=MANIFEST_SHA, source_commit=manifest['source_commit'],
                verified_files=verified,
                scope='selected actual plain-build inputs; no trace overlay; no binaries executed')


def source_facts(root):
    probe = SOURCE_PATHS['probe']
    config, _ = fragment(root, probe, r'#elif V06_CFG == 1\s+constexpr.*?using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;')
    builder, _ = fragment(root, probe, r'using Epilogue = typename.*?float, float, void,.*?EpilogueSchedule>::CollectiveOp;')
    args, match = fragment(root, probe, r'typename Gemm::Arguments arguments\{.*?\{\{(1\.f),\s*(0\.f)\}, nullptr, sd, d\.pointer, sd\}\};')
    callbacks = SOURCE_PATHS['callbacks']
    text = (root / callbacks).read_text()
    linear = text.index('fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,')
    defaults, _ = fragment(root, callbacks, r'struct Arguments \{\s+ElementScalar alpha = ElementScalar\(1\);\s+ElementScalar beta = ElementScalar\(0\);\s+ElementScalar const\* alpha_ptr = nullptr;\s+ElementScalar const\* beta_ptr = nullptr;.*?StrideAlpha dAlpha = \{_0\{\}, _0\{\}, 0\};\s+StrideBeta\s+dBeta\s+= \{_0\{\}, _0\{\}, 0\};', linear)
    conversion, _ = fragment(root, callbacks, r'operator typename Impl::Arguments\(\) const \{.*?\{\{alpha\}, \{alpha_ptr\}, \{dAlpha\}\}', linear)
    visitor = SOURCE_PATHS['visitor']
    constructor, _ = fragment(root, visitor, r'Sm90ScalarBroadcast\(Params const& params, SharedStorage const& shared_storage\).*?if \(size<2>\(params_ptr->dScalar\[0\]\) == 0\) \{\s+update_scalar\(\);\s+\}')
    update, _ = fragment(root, visitor, r'update_scalar\(int l_coord = 0\) \{.*?if \(params_ptr->scalar_ptrs\[0\] != nullptr\) \{\s+scalar = params_ptr->scalar_ptrs\[0\]\[l_offset\];\s+\}\s+else \{.*?scalar = params_ptr->scalars\[0\];\s+\}')
    return dict(alpha=float(match[1][:-1]), beta=float(match[2][:-1]),
                alpha_ptr=None, beta_ptr=None, alpha_batch_stride=0, beta_batch_stride=0,
                c_element_type='void',
                evidence=dict(configuration=config, epilogue_builder=builder, arguments=args,
                              linear_combination_defaults=defaults, callback_conversion=conversion,
                              scalar_constructor=constructor, scalar_update=update))


def read_kernel(root, name):
    path = 'build/' + name + '.sass'
    kernel = None
    instructions = []
    for line_number, line in enumerate((root / path).read_text().splitlines(), 1):
        if 'Function :' in line:
            if kernel is not None:
                break
            kernel = line.split('Function :', 1)[1].strip()
        if kernel is None:
            continue
        match = re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;', line)
        if match:
            instruction = match[2].strip()
            opcode = re.sub(r'^@\S+\s+', '', instruction).split()[0]
            instructions.append(reference(path, line_number, pc='0x' + match[1],
                                          opcode=opcode, instruction=instruction))
    require(kernel is not None and 'cutlass13device_kernel' in kernel and
            'KernelTmaWarpSpecializedPingpong' in kernel and 'LinearCombinationIffvf' in kernel,
            'first function is not the expected cfg_b GEMM')
    require(instructions and len({v['pc'] for v in instructions}) == len(instructions),
            'missing/duplicate instruction PCs')
    return kernel, instructions


def check_fragments(instructions, expected):
    by_pc = {int(v['pc'], 16): v for v in instructions}
    result = []
    for pc, instruction in expected:
        require(pc in by_pc and by_pc[pc]['instruction'] == instruction,
                'changed branch fragment at ' + hex(pc) + ': ' + instruction)
        result.append(by_pc[pc])
    return result


def alpha_branches(name, instructions):
    if name == 'cfg_b_plain':
        startup = [(0x7f0, 'ULDC.64 UR4, c[0x0][0x468]'),
                   (0x800, 'ISETP.NE.U32.AND P0, PT, RZ, UR4, PT'),
                   (0x810, 'ISETP.NE.AND.EX P0, PT, RZ, UR5, PT, P0'),
                   (0x820, '@!P0 BRA 0x860'), (0x830, 'LDC.64 R2, c[0x0][0x468]'),
                   (0x840, 'LDG.E R7, desc[UR22][R2.64]'),
                   (0x850, 'BRA 0x870'), (0x860, 'LDC R7, c[0x0][0x460]')]
        epilogue = [(0x1d50, 'ULDC.64 UR20, c[0x0][0x470]'),
                    (0x1d70, 'ISETP.NE.U32.AND P0, PT, RZ, UR20, PT'),
                    (0x1d90, 'ISETP.NE.AND.EX P0, PT, RZ, UR21, PT, P0'),
                    (0x1dc0, '@!P0 BRA 0x1e70'),
                    (0x1dd0, 'ULDC.64 UR24, c[0x0][0x468]'),
                    (0x1de0, 'ISETP.NE.U32.AND P0, PT, RZ, UR24, PT'),
                    (0x1df0, 'ISETP.NE.AND.EX P0, PT, RZ, UR25, PT, P0'),
                    (0x1e00, '@!P0 BRA 0x1e60'),
                    (0x1e10, 'LDC.64 R6, c[0x0][0x468]'),
                    (0x1e20, 'IMAD R5, R4, UR20, RZ'),
                    (0x1e30, 'IMAD.WIDE R6, R5, 0x4, R6'),
                    (0x1e40, 'LDG.E R7, desc[UR22][R6.64]'),
                    (0x1e50, 'BRA 0x1e70'), (0x1e60, 'LDC R7, c[0x0][0x460]')]
    else:
        startup = [(0x7b0, 'ULDC.64 UR4, c[0x0][0x468]'),
                   (0x7c0, 'ISETP.NE.U32.AND P0, PT, RZ, UR4, PT'),
                   (0x7d0, 'ISETP.NE.AND.EX P0, PT, RZ, UR5, PT, P0'),
                   (0x7e0, '@!P0 BRA 0x820'), (0x7f0, 'LDC.64 R6, c[0x0][0x468]'),
                   (0x800, 'LDG.E R6, desc[UR16][R6.64]'),
                   (0x810, 'BRA 0x830'), (0x820, 'LDC R6, c[0x0][0x460]')]
        epilogue = [(0x1d40, 'ULDC.64 UR12, c[0x0][0x470]'),
                    (0x1d60, 'ISETP.NE.U32.AND P0, PT, RZ, UR12, PT'),
                    (0x1d80, 'ISETP.NE.AND.EX P0, PT, RZ, UR13, PT, P0'),
                    (0x1db0, '@!P0 BRA 0x1e60'),
                    (0x1dc0, 'ULDC.64 UR10, c[0x0][0x468]'),
                    (0x1dd0, 'ISETP.NE.U32.AND P0, PT, RZ, UR10, PT'),
                    (0x1de0, 'ISETP.NE.AND.EX P0, PT, RZ, UR11, PT, P0'),
                    (0x1df0, '@!P0 BRA 0x1e50'),
                    (0x1e00, 'LDC.64 R4, c[0x0][0x468]'),
                    (0x1e10, 'IMAD R13, R0, UR12, RZ'),
                    (0x1e20, 'IMAD.WIDE R4, R13, 0x4, R4'),
                    (0x1e30, 'LDG.E R6, desc[UR16][R4.64]'),
                    (0x1e40, 'BRA 0x1e60'), (0x1e50, 'LDC R6, c[0x0][0x460]')]
    result = []
    for role, expected, load_index, branch_index in (
            ('startup_scalar', startup, 5, 3), ('epilogue_batch_update', epilogue, 11, 3)):
        checked = check_fragments(instructions, expected)
        branch_target = int(re.search(r'BRA 0x([0-9a-f]+)', checked[branch_index]['instruction'])[1], 16)
        load = checked[load_index]
        require(expected[branch_index][0] < int(load['pc'], 16) < branch_target,
                'branch does not skip scalar load')
        result.append(dict(role=role, load=load, width_bits=32,
                           controlling_skip=checked[branch_index], branch_target=hex(branch_target),
                           checked_fragment=checked, active=False,
                           source_condition='alpha_ptr == nullptr' if role == 'startup_scalar'
                           else 'alpha batch stride == 0; null pointer also selects direct scalar fallback'))
    require({v['load']['pc'] for v in result} ==
            {v['pc'] for v in instructions if v['opcode'].split('.')[0] == 'LDG'},
            'unaccounted ordinary global load')
    return result


def stage(name, pc):
    if name == 'cfg_b_plain':
        ranges = [(0, 0x1410, 'startup'), (0x1820, 0x1ba0, 'consumer_ktile_loop'),
                  (0x1bb0, 0x1bb0, 'consumer_math_tail'), (0x1d50, 0x1e60, 'epilogue_scalar'),
                  (0x3350, 0x37b0, 'consumer_next_work'), (0x3a20, 0x3af0, 'producer_role_setup'),
                  (0x3b50, 0x3b50, 'producer_output_control'),
                  (0x3c20, 0x3eb0, 'producer_ktile_loop'), (0x3f00, 0x4570, 'producer_next_work')]
    else:
        ranges = [(0, 0x14c0, 'startup'), (0x18d0, 0x1c50, 'consumer_ktile_loop'),
                  (0x1c60, 0x1c60, 'consumer_math_tail'), (0x1d40, 0x1e50, 'epilogue_scalar'),
                  (0x3340, 0x3980, 'consumer_next_work'), (0x3b50, 0x3c20, 'producer_role_setup'),
                  (0x3c80, 0x3c80, 'producer_output_control'),
                  (0x3d50, 0x3fe0, 'producer_ktile_loop'), (0x4010, 0x46c0, 'producer_next_work')]
    return next((label for low, high, label in ranges if low <= pc <= high), 'other')


def build_facts(root, name):
    kernel, instructions = read_kernel(root, name)
    observed = Counter(v['opcode'] for v in instructions)
    counts = {op: observed[op] for op in ('LDG.E', 'STG', 'LDL', 'STL', 'LDC', 'LDC.64',
                                        'ULDC', 'ULDC.64', 'STS.64', 'LDS', 'LDSM', 'STSM',
                                        'UTMALDG.3D', 'UTMASTG.3D', 'UTMACCTL.PF')}
    for op, expected in dict(EXPECTED_COUNTS[name], **{'LDG.E': 2, 'STS.64': 64,
                           'UTMALDG.3D': 3, 'UTMASTG.3D': 8, 'UTMACCTL.PF': 3}).items():
        require(observed[op] == expected, 'changed static count: ' + name + ': ' + op)
    for prefix in ('STG', 'LDL', 'STL', 'LDS', 'LDSM', 'STSM'):
        require(not any(op.split('.')[0] == prefix for op in observed), 'unexpected ordinary opcode: ' + prefix)
    constants = []
    for value in instructions:
        if value['opcode'].split('.')[0] in ('LDC', 'ULDC'):
            operand = re.search(r'c\[0x0\]\[0x[0-9a-f]+\]', value['instruction'])
            require(operand is not None, 'unexpected constant address space')
            label = stage(name, int(value['pc'], 16))
            require(label != 'other', 'unclassified parameter load: ' + value['pc'])
            constants.append(reference(value['path'], value['line'], pc=value['pc'],
                                       opcode=value['opcode'], width_bits=64 if '.64' in value['opcode'] else 32,
                                       operand=operand[0], stage=label))
    loops = ((0x1ba0, 0x1820, 0x3eb0, 0x3c20) if name == 'cfg_b_plain'
             else (0x1c50, 0x18d0, 0x3fe0, 0x3d50))
    loop_facts = {}
    for role, end, begin in (('consumer', *loops[:2]), ('producer', *loops[2:])):
        backedge = check_fragments(instructions, [(end, '@P0 BRA ' + hex(begin) if role == 'consumer'
                                                   else '@P1 BRA ' + hex(begin))])[0]
        body = [v for v in instructions if begin <= int(v['pc'], 16) <= end]
        require(not any(v['opcode'].split('.')[0] in ('LDG', 'STG', 'LDL', 'STL') for v in body),
                'ordinary global/local instruction in mainloop')
        loop_facts[role] = dict(begin_pc=hex(begin), backedge=backedge,
                               ordinary_global_local_instructions=0,
                               constant_instructions=[v for v in constants if begin <= int(v['pc'], 16) <= end])
    log_path = 'build/' + name + '.log'
    log = (root / log_path).read_text()
    match = re.search(r'Function properties for ' + re.escape(kernel) +
                      r'\n\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\n'
                      r'ptxas info\s+: Used (\d+) registers, used (\d+) barriers', log)
    require(match is not None, 'missing ptxas record for actual GEMM')
    stack, spill_stores, spill_loads, registers, barriers = map(int, match.groups())
    require((stack, spill_stores, spill_loads, registers) == (0, 0, 0, 168), 'changed main-kernel resources')
    command = json.loads((root / 'build/commands.json').read_text())[name]
    require('-DV06_CFG=1' in command and 'source/probes/r18.cu' in command and
            '-Isource/cutlass/include' in command and not any('overlay' in x for x in command),
            'plain build command changed')
    require(('-DR18_STAGE_COUNT=4' in command) == (name == 'cfg_b_s4_plain'), 'stage override changed')
    resources = json.loads((root / 'build/resources.json').read_text())['facts'][name]
    require(resources['gemm_registers'] == [registers] and resources['spills'] == [], 'resource summary mismatch')
    return dict(stages=6 if name == 'cfg_b_plain' else 4, kernel=kernel, counts=counts,
                alpha_load_branches=alpha_branches(name, instructions),
                constant_instructions=constants, mainloop=loop_facts,
                tma_instructions=[v for v in instructions if v['opcode'] in ('UTMALDG.3D', 'UTMASTG.3D', 'UTMACCTL.PF')],
                command=command, static_resources=reference(log_path, log.count('\n', 0, match.start()) + 1,
                    registers_per_thread=registers, stack_bytes=stack, spill_store_bytes=spill_stores,
                    spill_load_bytes=spill_loads, barriers=barriers),
                cache_policy=dict(source='unqualified scalar C++ dereference', sass='LDG.E',
                                  ptx_modifier='not bound by these saved artifacts',
                                  qualification='no physical L1 hit, traffic, or bypass inference'))


def audit(root):
    root = Path(root).resolve()
    proven_identity = identity(root)
    sources = source_facts(root)
    builds = {name: build_facts(root, name) for name in EXPECTED_COUNTS}
    return dict(schema='gh200-cfg-b-lsu-build-audit/1', frozen_root=str(root),
                auditor_sha256=sha(Path(__file__)), identity=proven_identity,
                source_facts=sources, builds=builds,
                dynamic_ordinary_global_demand=dict(read_instructions=0, write_instructions=0,
                    read_bytes=0, write_bytes=0, local_load_instructions=0, local_store_instructions=0,
                    qualification='source_and_sass_branch_derivation_not_counters',
                    scope='fixed r18 cfg_b alpha=1 beta=0, null scalar pointers and zero scalar batch strides',
                    derivation=['r18 directly initializes alpha=1 and beta=0; omitted scalar pointer/stride members use checked defaults',
                                'constructor null-pointer branch selects parameter scalar; epilogue zero-batch-stride branch skips update',
                                'all two static ordinary LDG instructions are inside those checked scalar branches',
                                'no ordinary STG or local LDL/STL exists in the audited GEMM function']),
                coverage=dict(matrix_inputs_and_output='TMA; excluded from ordinary LSU zero demand',
                              constants='actual LDC/ULDC parameter instructions; retain existing composite service coverage',
                              timing='no service price, counter, cache-hit measurement or GPU execution'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--input', type=Path, help='explicit frozen directory; overrides --repo')
    parser.add_argument('--output', type=Path, default=Path('/tmp/l1-lsu-build-audit'))
    args = parser.parse_args()
    facts = audit(args.input or args.repo / FROZEN)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'build-facts.json'
    target.write_text(json.dumps(facts, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status='passed', output=str(target.resolve()),
                          counts={name: value['counts'] for name, value in facts['builds'].items()})))


if __name__ == '__main__':
    main()
