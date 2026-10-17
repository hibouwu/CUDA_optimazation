#!/usr/bin/env python3
"""Fixed cfg_b S4/S6 capacity accounting and replay of two existing SMID traces.

No GPU calls, placement rule or timing price. Allocation rounding explains the
saved API result; it does not reconstruct the old runtime's internal factors.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import audit_cfg_b_lsu as build_audit
import cfg_b_program


GH200_DEVICE_CONDITIONS = dict(
    cc='9.0', warp_size=32, registers_per_sm=65536, smem_per_sm_bytes=233472,
    smem_per_cta_optin_bytes=232448, max_threads_per_sm=2048, max_blocks_per_sm=32,
    reserved_smem_bytes_per_cta=1024, register_allocation_granularity=256,
    smem_allocation_granularity=128, register_subpartitions=4)

DEVICE_RECORD = Path('results/gh200_resource_campaign/20261001-memory-c/device.jsonl')
HEADERS = {
    'adapter': 'source/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h',
    'entry': 'source/cutlass/include/cutlass/device_kernel.h',
    'kernel': 'source/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp',
    'scheduler': 'source/cutlass/include/cutlass/gemm/kernel/static_tile_scheduler.hpp',
    'scheduler_params': 'source/cutlass/include/cutlass/gemm/kernel/tile_scheduler_params.h',
}


def round_up(value, granularity):
    return (value + granularity - 1) // granularity * granularity


def cfg_b_capacity(program, *, device):
    """Use explicit device conditions and the fixed build's initial CTA quota.

    setup.smem already includes epilogue/pipeline storage. Historical occupancy
    is a separate observation, never an input used to force a calculated result.
    """
    stages = program['stages']
    if stages not in (4, 6) or list(program['tile']) != [128, 128, 64]:
        raise ValueError('capacity accounting only covers cfg_b S4/S6 128x128x64')
    if device['cc'] != '9.0':
        raise ValueError('capacity accounting only covers the explicit SM90 conditions')
    registers = cfg_b_program.register_demands(program)
    setup = program.get('setup', {})
    threads = registers['threads']
    if 'threads' in setup and setup['threads'] != threads:
        raise ValueError('setup thread count differs from cfg_b register demand')
    dynamic = setup.get('smem', 32768 * stages + 18432)
    smem_origin = 'setup.smem; complete Kernel::SharedStorage' if 'smem' in setup else 'frozen_build_requirement: 32768*stages+18432'
    warp_size = device['warp_size']
    warps = (threads + warp_size - 1) // warp_size
    per_warp = round_up(registers['ptxas_registers_per_thread'] * warp_size,
                        device['register_allocation_granularity'])
    per_cta = per_warp * warps
    subpartitions = device['register_subpartitions']
    per_partition = device['registers_per_sm'] // subpartitions
    warps_per_partition = per_partition // per_warp
    assumed_registers = per_warp * round_up(warps, subpartitions)
    # The fixed device kernel uses extern shared storage and has no static SMEM
    # allocation in its build record. The old func attribute was not saved.
    static = 0
    allocated_smem = round_up(static + dynamic + device['reserved_smem_bytes_per_cta'],
                              device['smem_allocation_granularity'])
    limits = dict(registers=(warps_per_partition * subpartitions) // warps,
                  smem=device['smem_per_sm_bytes'] // allocated_smem,
                  threads=(device['max_threads_per_sm'] // warp_size) // warps,
                  blocks=device['max_blocks_per_sm'])
    capacity = min(limits.values())
    historical = setup.get('max_active_ctas_per_sm')
    return dict(schema='gh200-cfg-b-sm-capacity/1', stages=stages, device_conditions=dict(device),
        demands=dict(threads=threads, warps=warps,
            registers_per_thread=registers['ptxas_registers_per_thread'],
            cta_initial_pool_registers=registers['cta_reserved_registers'],
            dynamic_smem_bytes=dynamic, static_smem_bytes=static,
            smem_origin=smem_origin, smem_epilogue_counted_again=False,
            static_smem_qualification='frozen extern-shared kernel/build interpretation; old func attribute not separately saved'),
        allocations=dict(registers_per_warp=per_warp, registers_per_cta=per_cta,
            launch_check_registers_per_cta=assumed_registers,
            registers_per_subpartition=per_partition, warps_per_subpartition=warps_per_partition,
            smem_per_cta_bytes=allocated_smem),
        capacity_limits=limits, capacity_ctas_per_sm=capacity,
        limiting_resources=[name for name, value in limits.items() if value == capacity],
        historical_api=dict(ctas_per_sm=historical,
            matches_calculation=historical == capacity if historical is not None else None,
            method='cudaOccupancyMaxActiveBlocksPerMultiprocessor(actual kernel, 384, complete dynamic SMEM)',
            qualification='saved total upper bound; reserved value and individual limiting factors were not saved'),
        qualification=dict(allocation='explicit SM90 hardware conditions and local CUDA 13 SDK rounding interpretation',
            capacity='selected resource bounds for the fixed S4/S6 builds; saved API is the complete historical query',
            scope='no CTA placement, active-warp efficiency, launch distribution or timing prediction',
            register_release='setmaxnreg redistributes the CTA-reserved pool; no second-CTA admission implied'))


def audit(repo):
    """Bind the two actual builds/setup records and decode their saved activity."""
    from analyze_r18 import replay

    repo = Path(repo).resolve()
    root = repo / build_audit.FROZEN
    facts = build_audit.audit(root)
    manifest = json.loads((root / 'frozen/manifest.json').read_text())
    source_hashes = json.loads((root / 'source_hashes.json').read_text())
    verified = {}
    for path in HEADERS.values():
        digest = build_audit.sha(root / path)
        build_audit.require(digest == source_hashes[path], 'changed capacity source: ' + path)
        verified[path] = digest
    for path in ('cases.json', 'freeze-setup.json', 'freeze-environment.json'):
        digest = build_audit.sha(root / path)
        build_audit.require(digest == manifest['files'][path], 'changed frozen setup input: ' + path)
        verified[path] = digest
    for kind, suffix in (('binary', ''), ('sass', '.sass')):
        hashes = json.loads((root / ('build/' + kind + '_hashes.json')).read_text())
        for name in ('cfg_b_dual', 'cfg_b_s4_dual'):
            path = 'build/' + name + suffix
            expected = hashes[path if kind == 'binary' else name + suffix]
            digest = build_audit.sha(root / path)
            build_audit.require(digest == expected, 'changed activity build: ' + path)
            verified[path] = digest
    device_path = repo / DEVICE_RECORD
    records = [json.loads(line) for line in device_path.read_text().splitlines() if line.strip()]
    build_audit.require(len(records) == 1, 'expected one historical GH200 device record')
    device_record = records[0]
    device = dict(GH200_DEVICE_CONDITIONS)
    for key in ('cc', 'warp_size', 'registers_per_sm', 'smem_per_sm_bytes',
                'smem_per_cta_optin_bytes', 'max_threads_per_sm', 'max_blocks_per_sm'):
        build_audit.require(device_record[key] == device[key], 'device condition changed: ' + key)
    setup_rows = json.loads((root / 'static_setup.json').read_text())
    setups = {value['case']: value['setup'] for value in setup_rows}
    frozen_setups = {value['case']: value['setup'] for value in json.loads((root / 'freeze-setup.json').read_text())}
    cases = {value['id']: value for value in json.loads((root / 'cases.json').read_text())}
    reports = {}
    for name, build in (('v09d_cfg_b_G3_base', 'cfg_b_plain'), ('v09d_cfg_b_G3_stage4', 'cfg_b_s4_plain')):
        setup = setups[name]
        build_audit.require(setup == frozen_setups[name], 'saved setup differs from frozen setup')
        expected_uuid = 'GPU-' + '-'.join(device_record['uuid_hex'][a:b] for a, b in ((0, 8), (8, 12), (12, 16), (16, 20), (20, 32)))
        build_audit.require(setup['gpu_uuid'] == expected_uuid, 'device query and setup refer to different GPUs')
        program = cfg_b_program.build_program(cases[name], setup)
        capacity = cfg_b_capacity(program, device=device)
        build_audit.require(capacity['demands']['registers_per_thread'] == facts['builds'][build]['static_resources']['registers_per_thread'], 'program/build register mismatch')
        build_audit.require(capacity['historical_api']['matches_calculation'], 'capacity differs from saved occupancy')
        record_path = root / 'samples' / name / 'dual-00.json'
        record = json.loads(record_path.read_text())
        build_audit.require(record['returncode'] == 0 and record['variant'] == 'dual', 'activity record is not successful dual trace')
        actual = replay(root, record, cases[name], retain_roles=True)
        ctas = actual['ctas']
        build_audit.require([cta['work'] for cta in ctas] == program['work'], 'activity work mapping differs from program')
        smids = Counter(cta['sm'] for cta in ctas)
        build_audit.require(len(ctas) == math.prod(setup['grid']) == len(smids) == setup['sm_count'], 'G3 activity does not show one CTA on each SM')
        blocks = [dict(cta=cta['cta'], smid=cta['sm'], entry_ns=cta['entry_ns'],
                       consumer_exit_ns=cta['end_ns'], output_coordinates=cta['work']) for cta in ctas]
        reports[name] = dict(build=build, setup=setup, capacity=capacity,
            activity=dict(record=str(record_path.relative_to(root)), record_sha256=build_audit.sha(record_path),
                raw=record['raw'], raw_sha256=record['raw_sha256'],
                grid_ctas=len(ctas), distinct_smids=len(smids), ctas_per_smid=dict(sorted(smids.items())),
                output_counts=dict(sorted(Counter(len(cta['work']) for cta in ctas).items())),
                shared_observed_interval_ns=min(cta['end_ns'] for cta in ctas)-max(cta['entry_ns'] for cta in ctas),
                blocks_detail=blocks,
                qualification='one existing traced invocation; ENTRY to consumer exit is not full CTA residency or plain-build activity'))
    sources = {
        'occupancy_call': build_audit.fragment(root, HEADERS['adapter'], r'static int maximum_active_blocks\(.*?return max_active_blocks;')[0],
        'dynamic_entry': build_audit.fragment(root, HEADERS['entry'], r'void device_kernel\(CUTLASS_GRID_CONSTANT.*?op\(params, smem\);')[0],
        'launch_bounds': build_audit.fragment(root, HEADERS['entry'], r'__launch_bounds__\(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor\)')[0],
        'work_advance': build_audit.fragment(root, HEADERS['scheduler'], r'advance_to_next_work\(uint32_t advance_count = 1\) \{.*?\}')[0],
        'single_cluster_grid': build_audit.fragment(root, HEADERS['scheduler_params'], r'if \(cluster_size == 1\) \{.*?else if \(max_active_clusters')[0],
    }
    return dict(schema='gh200-cfg-b-sm-residency-audit/1', auditor_sha256=build_audit.sha(Path(__file__)),
        identity=facts['identity'], additional_verified_sha256=verified,
        static_setup_sha256=build_audit.sha(root / 'static_setup.json'),
        device_record=dict(path=str(DEVICE_RECORD), sha256=build_audit.sha(device_path), record=device_record,
            interpretation_only=['reserved_smem_bytes_per_cta', 'register_allocation_granularity',
                                 'smem_allocation_granularity', 'register_subpartitions']),
        allocation_rule_sources=dict(sdk='/usr/local/cuda-13.0/include/cuda_occupancy.h',
            functions=['cudaOccRegAllocationGranularity', 'cudaOccSubPartitionsPerMultiprocessor',
                       'cudaOccSMemAllocationGranularity', 'cudaOccMaxBlocksPerSMRegsLimit',
                       'cudaOccMaxBlocksPerSMSmemLimit'],
            qualification='local SDK explanation; not an execution or internal factor dump of historical CUDA 12.9'),
        source_fragments=sources, cases=reports, gpu_runs=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repo)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'capacity-audit.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status='passed', output=str(target.resolve()),
        capacities={name: value['capacity']['capacity_ctas_per_sm'] for name, value in report['cases'].items()})))


if __name__ == '__main__':
    main()
