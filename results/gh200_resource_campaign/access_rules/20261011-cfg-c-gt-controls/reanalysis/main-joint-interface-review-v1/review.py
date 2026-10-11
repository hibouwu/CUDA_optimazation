#!/usr/bin/env python3
"""Bounded C review of frozen v3 call identity and completion-marker scope."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


TARGET = {
    'extract_joint.py': '01878be55d69e57e416747bb5d0eeb077cebd58c887513bc20c859848b1bf94c',
    'joint_records.py': '0e8ab31b2dfc295ce50aed13c1323a140d355cc1f67fa5405b0528e806293455',
}
SOURCE = {
    'source/probes/r18.cu': '7103d5df2628046a43772322f36d89a730d268ae18453f836383573108aaf6e4',
    'source/probes/v06_trace.hpp': 'fa599ddaeaea9e074ccca8eaef7cb6b2305081b078a29a0c3f9d2a2ef4bd9061',
    'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp': '03e4fcdd2911cb78454cd818cd27bdbbbaa83dfa3dd15f12851d44c42e93c7a4',
    'source/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h': '711c0ebdc9cf5cfde7bdaec100a2d9dada6a25bb6ab5d316779c5e881108e62e',
    'source/cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp': '5400755ecaaf74acc4d726b4f9f626336c47c5abd929018efbe81c06c26cb70d',
    'source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp': '7c61d847933cb12a14fcdc68cec902b44c802c9eb34273218c29e601f9c53dee',
    'source/cutlass/include/cute/arch/copy_sm90_tma.hpp': 'd31756a7105dc875751fdef875be1b9f2bc008111d0c45734fed39465c12c636',
}
BINARY = 'c5476d9b00679859eef64bff34e87532e4936ea73f7f432c476597b8c36a6480'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh C output directory required')
    run, build = args.run.resolve(), args.build.resolve()
    package = run / 'reanalysis/main-median-protocol-v1'
    replay = package / 'joint-replay-v3'
    inputs = {}

    def read(path, expected=None):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('frozen identity changed: ' + str(path))
        inputs[str(path)] = digest
        return payload

    manifest = json.loads(read(run / 'manifest.json'))
    source_manifest_bytes = read(build / 'source_hashes.json')
    source_manifest = json.loads(source_manifest_bytes)
    build_audit = json.loads(read(build / 'build-audit.json', manifest['build_audit_sha256']))
    if (manifest['binary_sha256'] != BINARY or build_audit['binary_sha256'] != BINARY
            or build_audit['source_manifest_sha256'] != hashlib.sha256(source_manifest_bytes).hexdigest()
            or Path(manifest['build_run']).resolve() != build):
        raise ValueError('source/build/control identity mismatch')
    for name, digest in SOURCE.items():
        if source_manifest[name] != digest:
            raise ValueError('reviewed source differs: ' + name)
        read(build / name, digest)
    # Main may advance the working files concurrently. Review the requested v3
    # source bytes preserved from commit 510d401b, never a moving working tree.
    for name, digest in TARGET.items():
        read(Path(__file__).with_name('source_snapshot') / name, digest)
    receipt = json.loads(read(replay / 'receipt.json'))
    if any(receipt['source_sha256'][name] != digest for name, digest in TARGET.items()):
        raise ValueError('replay used a different extractor')
    joint_bytes = read(replay / 'joint-process-records.json.gz',
                       receipt['output_sha256']['joint-process-records.json.gz'])
    joint = json.loads(gzip.decompress(joint_bytes))
    returned_bytes = read(run / 'returned/records.json', receipt['input_sha256']['returned/records.json'])
    returned = json.loads(returned_bytes)
    if len(joint) != 40 or len(returned) != 40:
        raise ValueError('fixed archive process count changed')
    rows = []
    final_count = 0
    enabled_windows = 0
    for row in joint:
        original = returned[row['archive_record_index']]
        expected = (original['id'], original['comparison'], original['pair'], original['arm'], original['flag'])
        actual = (row['id'], row['cohort'], row['block'], row['arm'], row['flag'])
        if actual != expected:
            raise ValueError('archive invocation binding changed')
        raw_path = run / 'returned' / original['raw']
        if Path(row['raw_path']).resolve() != raw_path.resolve():
            raise ValueError('raw path identity changed')
        raw_bytes = read(raw_path, original['sha256'])
        if row['raw_sha256'] != original['sha256']:
            raise ValueError('joint raw identity changed')
        events = [json.loads(line) for line in gzip.decompress(raw_bytes).decode().splitlines()]
        packets = {event['event']: event for event in events}
        if len(packets) != len(events):
            raise ValueError('duplicate event packet')
        setup, call = packets['setup'], packets['call']
        if row['setup'] != setup or row['same_call_event_us'] != call['elapsed_us']:
            raise ValueError('setup/target event differs from own raw call')
        if (row['completed_prefix_event_us'] != call['warmup_us']
                or len(row['completed_prefix_event_us']) != 16
                or row['call_protocol'] != {key: value for key, value in call.items()
                                           if 'warmup' in key or 'prefix' in key}):
            raise ValueError('prefix identity differs from own raw call')
        for packet in (setup, call):
            if (packet['gt_bracket_prefix_flag'] != 0 or packet['producer_tail_observer'] != 0
                    or packet['history_protocol'] != 'none'
                    or packet['prefix_a_ptr'] != packet['target_a_ptr']
                    or packet['prefix_b_ptr'] != packet['target_b_ptr']):
                raise ValueError('unsupported prefix or producer-tail condition')
        if (call['target_calls'] != 1 or call['preceding_target_calls'] != 0
                or call['warmup_calls'] != 16 or call['label'] != 'warm'):
            raise ValueError('target invocation is ambiguous')
        trace = call['trace']
        if len(trace) != 132 * 1296 or len(call['gt_shadow']) != 132 * 9:
            raise ValueError('raw trace layout changed')
        origin = min(trace[cta * 1296 + 1] for cta in range(132))
        finals = []
        for cta in range(132):
            base = cta * 1296
            for wg in range(2):
                offset = base + 4 + wg * 3
                final_cycles, final_gt = trace[offset:offset + 2]
                if row['series'][f'{cta}/{wg}/GLOBAL_FINAL'] != [final_gt - origin, None]:
                    raise ValueError('recorded FINAL/reference origin differs')
                if row['series'][f'{cta}/{wg}/FINAL'] != [final_gt - trace[base + 1],
                                                        final_cycles - trace[base]]:
                    raise ValueError('local FINAL/reference origin differs')
                finals.append(final_gt - origin)
                final_count += 1
        if row['composites']['consumer_FINAL_envelope'] != [max(finals), None]:
            raise ValueError('FINAL envelope does not use own GT vector')
        if 'brackets' in row or 'shared_origin_gt_counter_ns' in row:
            raise ValueError('review target schema changed')
        windows = sum(name.endswith('/L') for name in row['series'])
        if windows != 432:
            raise ValueError('fixed window count changed')
        if row['flag']:
            for name in row['series']:
                if not name.endswith('/L'):
                    continue
                cta, wg, ordinal, coord, _ = name.split('/')
                cta, wg, ordinal = int(cta), int(wg), int(ordinal)
                start = cta * 1296 + 16 + (wg * 64 + ordinal) * 10
                before_start = cta * 9 + wg * 4 + ordinal * 2
                for event in (0, 1):
                    before = call['gt_shadow'][before_start + event]
                    cycles = trace[start + event]
                    after = trace[start + 6 + event]
                    if not (0 < before <= after and cycles > 0):
                        raise ValueError('cannot recover complete enabled raw bracket')
                enabled_windows += 1
        elif any(call['gt_shadow']):
            raise ValueError('disabled observer wrote GT-before shadow')
        rows.append(dict(id=row['id'], archive_record_index=row['archive_record_index'],
                         raw_sha256=row['raw_sha256'], target_calls=1, prefix_calls=16,
                         gt_observer_flag=row['flag'],
                         target_event_exactly_bound=True, setup_and_prefix_exactly_bound=True))
    report = dict(
        schema='C-joint-record-interface-review/1',
        reviewed_target_sha256=TARGET, binary_sha256=BINARY,
        input_sha256=inputs, processes=40, final_markers_verified=final_count,
        recoverable_enabled_bracket_windows=enabled_windows,
        recoverable_enabled_first_main_stamps=2 * enabled_windows,
        records=rows, implementation_findings_within_C_endpoint_scope=[],
        accepted_scope='Given the supplied v3 archive mapping, target-event/prefix/FINAL values match their own raw packets and reviewed source boundaries. Upstream acquisition mapping authenticity is outside this C review.',
        source_witnesses=[
            dict(file='source/probes/r18.cu', lines=[736, 759],
                 meaning='Clear trace/shadow and synchronize before event begin; one Gemm::run between events; synchronize event end before returning elapsed ms.'),
            dict(file='source/probes/r18.cu', lines=[941, 953],
                 meaning='Exactly 16 completed warmup calls, set target observer flag, one target timed call, then report its duration and current trace.'),
            dict(file='source/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h', lines=[365, 370, 464, 471],
                 meaning='Gemm::run default stream is nullptr; cfg_c SM90 cluster launch forwards this stream.'),
            dict(file='source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp', lines=[866, 872, 888, 896, 908, 927, 946],
                 meaning='MAIN after own mma_tail; PERMIT after epilogue order wait; DONE after store/store_tail but before final state advance and own order arrival; FINAL after scheduler loop.'),
            dict(file='source/probes/v06_trace.hpp', lines=[267, 269, 295, 303],
                 meaning='Only physical threads 128/256 write consumer markers and FINAL; no all-384-thread completion stamp.'),
            dict(file='source/cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp', lines=[920, 926],
                 meaning='store_tail calls PipelineTmaStore::producer_tail.'),
            dict(file='source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp', lines=[686, 687],
                 meaning='Store producer tail calls tma_store_wait<0>.'),
            dict(file='source/cutlass/include/cute/arch/copy_sm90_tma.hpp', lines=[1248, 1254],
                 meaning='The actual wait instruction is cp.async.bulk.wait_group.read 0: source/tensormap reads, not destination global visibility.')],
        event_contract=dict(same_invocation_as_trace=True,
            boundary='CUDA event begin -> target kernel launch on default stream -> CUDA event end',
            outside_timed_region=['trace/shadow clear and preceding device synchronization',
                                  '16 prefix invocations', 'post-end host trace copy and numerical check'],
            absolute_event_timestamps_saved=False, event_to_gt_origin_calibrated=False,
            may_subtract_event_and_final_envelope_as_exit_tail=False),
        completion_contract=dict(
            consumer_FINAL_envelope='max of 264 recorded leader FINAL GT reads minus minimum CTA ENTRY GT read in this invocation',
            source_safe_exit='Not fully observed: producer input-tail completion, all relevant D source/order and rank exit constraints remain separate.',
            D_global_visible='No per-CTA timestamp; read0 does not observe destination visibility.',
            cuda_complete='Successful end-event synchronization establishes completion of captured target work at host observation; no per-CTA completion timestamp or last-returning-CTA identity is supplied.'),
        interval_adapter_contract=dict(
            direct_joint_record_input_supported=False,
            recover_from_bound_raw=['absolute FIRST/MAIN GT-before/clock/GT-after and common integer origin'],
            additionally_bind=['exact invocation/device/binary/source-graph hash',
                               'rank/CTA, physical thread, work coordinate and SM',
                               'explicit sampling-order and monotone-GT assumptions plus reviewed error/enclosure scope'],
            disabled_arm_has_gt_before=False,
            full_joint_physical_enclosure_proved=False),
        normative_sources=[
            'https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html',
            'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group'],
        preserved_limits=['This endpoint review does not independently authenticate the v3 ID/block/arm-to-raw mapping or certify the later v4 acquisition-archive fix.',
                         'No change to 6372-series coverage family, old failures, V09 candidates or GPU collection.',
                         'No cycle maximum across SMs, no event-minus-FINAL tail estimate, no median-as-DAG-cost substitution.'],
        observation_qualified=False, prediction_qualified=False,
        critical_CTA_qualified=False, independent_holdout_passed=False)
    args.output.mkdir()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('processes', 'final_markers_verified',
          'recoverable_enabled_first_main_stamps', 'implementation_findings_within_C_endpoint_scope', 'observation_qualified')}, indent=2))


if __name__ == '__main__':
    main()
