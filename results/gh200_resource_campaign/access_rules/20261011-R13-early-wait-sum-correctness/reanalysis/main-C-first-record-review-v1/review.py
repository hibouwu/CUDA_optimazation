#!/usr/bin/env python3
"""C offline review of the sole returned target in failed four-target job740259."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new C review output directory required')
    run, build = args.run.resolve(), args.build.resolve()
    inputs = {}

    def read(path, expected=None):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('frozen bytes changed: ' + str(path))
        inputs[str(path)] = digest
        return data

    freeze = json.loads(read(run / 'freeze.json'))
    for name, digest in freeze['files'].items():
        read(run / name, digest)
    manifest = json.loads(read(run / 'manifest.json'))
    receipt = json.loads(read(run / 'receipt.json'))
    archive_bytes = read(run / 'output.tar.gz', receipt['sha256'])
    if receipt['returncode'] != 1 or receipt['sha256'] != '313d844177db6d8757eb5315e122a3c1d6b0c4dfccdce247daf8c829c3053e84':
        raise ValueError('review is scoped to original failed job740259')
    records_bytes = read(run / 'returned/records.json')
    records = json.loads(records_bytes)
    if len(records) != 1 or records[0]['id'] != 'stage4_flag0':
        raise ValueError('not the original one-target failed batch')
    record = records[0]
    declared = manifest['samples'][0]
    if any(record[key] != value for key, value in declared.items()) or record['returncode'] != 0:
        raise ValueError('first target metadata or process failed')
    raw_bytes = read(run / 'returned' / record['raw'], record['sha256'])
    with tarfile.open(run / 'output.tar.gz', 'r:gz') as archive:
        if archive.extractfile('./records.json').read() != records_bytes:
            raise ValueError('returned process map differs from original archive')
        if archive.extractfile('./' + record['raw']).read() != raw_bytes:
            raise ValueError('returned raw differs from original archive')
        if archive.extractfile('./manifest.json').read() != (run / 'manifest.json').read_bytes():
            raise ValueError('acquisition manifest differs from freeze')
        if archive.extractfile('./exit_code.txt').read().strip() != b'1':
            raise ValueError('original failed exit must be retained')
        pipeline_log = archive.extractfile('./run.log').read().decode()
        if 'expected exactly setup, target call and numerical check' not in pipeline_log:
            raise ValueError('different batch failure')
        if any(member.name.endswith('complete.json') for member in archive.getmembers()):
            raise ValueError('failed original unexpectedly contains completion certificate')

    source = build / 'attempts/prepare3'
    source_manifest = json.loads(read(source / 'source_hashes.json', manifest['source_manifest_sha256']))
    for name in ('source/probes/r00_common.hpp', 'source/probes/gaps_common.hpp', 'source/probes/r18.cu'):
        read(source / name, source_manifest[name])
    read(build / 'build-audit.json', manifest['build_audit_sha256'])
    if 'return value / 32.0f;' not in (source / 'source/probes/r00_common.hpp').read_text():
        raise ValueError('input witness divisor changed')
    snapshot = run / 'analysis_snapshot'
    for name, digest in manifest['analysis_sha256'].items():
        read(snapshot / name, digest)
    sys.path.insert(0, str(snapshot))
    spec = importlib.util.spec_from_file_location('C_original_wait_sum_decoder', snapshot / 'decode_r13_wait_sum.py')
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    events = [json.loads(line) for line in gzip.decompress(raw_bytes).decode().splitlines()]
    packets = {event['event']: event for event in events}
    if len(packets) != len(events) or set(packets) != {'setup', 'call', 'input_zero_storage', 'check'}:
        raise ValueError('unexpected or duplicated original packet')
    extra = packets['input_zero_storage']
    if extra != {'event': 'input_zero_storage', 'checked': 0, 'nonzero': 0}:
        raise ValueError('unexpected input-region check; do not discard it')
    try:
        decoder.decode(events, manifest['gpu_uuid'])
    except ValueError as error:
        if str(error) != 'expected exactly setup, target call and numerical check':
            raise
    else:
        raise ValueError('original frozen decoder no longer rejects four packets')
    setup, call, check = packets['setup'], packets['call'], packets['check']
    if setup['gpu_uuid'] != manifest['gpu_uuid']:
        raise ValueError('target GPU differs')
    try:
        decoder.decode_trace(setup, call)
    except ValueError as error:
        if str(error) != 'fixed setup mismatch: cluster':
            raise
    else:
        raise ValueError('original cluster-layout mismatch no longer reproduced')
    # Inspect the raw two-axis cluster representation independently. Do not
    # alter a packet or discard one to manufacture a frozen-decoder success.
    geometry = dict(config='cfg_b', stages=4, m=2816, n=1536, k=4096,
        tile=[128,128,64], cluster=[1,1], grid=[1,132,1],
        trace_words=784, trace_tile_words=6, trace_version='r13-early-wait-sum-v1')
    if any(setup.get(key) != value for key, value in geometry.items()):
        raise ValueError('unexpected raw geometry')
    for packet in (setup, call):
        if packet['wait_sum_observer'] != 0 or packet['wait_sum_prefix_flag'] != 0:
            raise ValueError('first target or prefix flag differs')
    if call['warmup_calls'] != 16 or len(call['warmup_us']) != 16:
        raise ValueError('incomplete declared prefix')
    words = call['trace']
    if len(words) != 132 * 784 or any(type(value) is not int or not 0 <= value < 2**64 for value in words):
        raise ValueError('invalid raw integer trace')
    head = words[:784]
    allowed = {1,13,14,15,16,17,18,19,20,21,22,23,24,31,32,34,35,36,37,38,39,40}
    if any(words[784:]) or any(value for index, value in enumerate(head) if index not in allowed):
        raise ValueError('unexpected raw writer or overflow field')
    if head[31] != 0x52313309 or head[13:15] != [12,1] or not 1 <= head[1] <= 132 or head[1] != head[23]:
        raise ValueError('marker/work/SM mismatch')
    clock_slots = dict(FIRST=21, C1=20, H16=15, C16=17, WAIT1_16=18,
        H17=16, C17=19, C63=24, MMA_RETURN=32, MAIN=22)
    clocks = {name: head[slot] for name, slot in clock_slots.items()}
    ordered = list(clocks.values())
    if not all(ordered) or any(before > after for before, after in zip(ordered, ordered[1:])):
        raise ValueError('same-thread recorded progress order mismatch')
    if any(head[34:41]):
        raise ValueError('off target wrote cumulative or flag fields')
    trace_result = dict(cta=0, physical_thread=256, cta_output_ordinal=1,
        WG_local_output_ordinal=0, work=[11,0], sm=head[1]-1, stages=4, observer=0,
        early_cycles=head[17]-head[20], full_wait_cycles=None, outside_wait_cycles=None,
        full_wait_ns=None, outside_wait_ns=None, clocks=clocks,
        inspection='Independent raw progress layout/order check only; original frozen decoder still rejects the two-axis cluster.',
        observation_qualified=False, prediction_qualified=False)

    indices, values = check['checked_indices'], check['checked_values']
    if (check['status'] != 'ok' or check['samples'] != 4096 or len(indices) != 4096
            or len(values) != 4096 or len(set(indices)) != 4096
            or any(check[key] != 0 for key in ('nonfinite', 'padding_errors', 'max_storage_reference_error'))):
        raise ValueError('original sampled numeric check incomplete or failed')
    # Regenerate each distinct coordinate class by all 4096 integer products.
    # This does not call either the GPU checker or its period-sum Python reference.
    numerators = {}
    for index, value in zip(indices, values):
        if type(index) is not int or not 0 <= index < 2816 * 1536:
            raise ValueError('invalid sampled coordinate')
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('invalid sampled output')
        row, column = divmod(index, 1536)
        key = row % 17, column % 17
        if key not in numerators:
            total = 0
            for inner in range(4096):
                a_numerator = (key[0] * 7 + inner * 13 + 17 * 3) % 17 - 8
                b_numerator = (inner * 5 + key[1] * 11 + 17 * 5) % 17 - 8
                total += a_numerator * b_numerator
            numerators[key] = total
        if value * 1024 != numerators[key]:
            raise ValueError('direct integer-dot reference mismatch')

    report = dict(schema='C-wait-sum-first-record-review/1', job=740259,
        input_sha256=inputs, original_archive_sha256=receipt['sha256'],
        original_batch_returncode=1, original_first_process_returncode=0,
        original_frozen_decoder_packet_count_rejection_reproduced=True,
        original_frozen_decoder_cluster_layout_rejection_reproduced=True,
        raw_cluster=[1,1], frozen_decoder_expected_cluster=[1,1,1],
        original_failed_batch_preserved=True, batch_complete=False,
        targets_declared=4, targets_returned=1,
        missing_targets=[sample['id'] for sample in manifest['samples'][1:]],
        first_target_id=record['id'], raw_sha256=record['sha256'],
        input_zero_storage_packet=extra,
        input_zero_storage_interpretation='No extra logical rectangle was checked; checked=0 is not a positive-size zero-storage validation.',
        independently_verified_sampled_values=4096, unique_indices=4096,
        direct_dot_coordinate_classes=len(numerators),
        direct_integer_products=len(numerators) * 4096, exact_value_mismatches=0,
        trace_scope=trace_result, full_D_verified=False, hardware_ready_identified=False,
        limits=['Only stage4 observer-off ran. No enabled Q sample and no stage6 result.',
                'A successful target process and offline checks do not repair the frozen four-target collector failure.',
                'No whole-D dump, observer perturbation, service estimate, critical CTA or holdout score.',
                'Prefix duration fields are retained; this trace version does not record A/B pointer addresses.'],
        observation_qualified=False, prediction_qualified=False, independent_holdout_passed=False)
    args.output.mkdir()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('job', 'targets_returned', 'batch_complete',
        'independently_verified_sampled_values', 'direct_dot_coordinate_classes', 'exact_value_mismatches')}, indent=2))


if __name__ == '__main__':
    main()
