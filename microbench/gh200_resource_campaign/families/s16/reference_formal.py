"""Independent full UINT32 final-state reference for the S16 host path."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import sys

if sys.flags.optimize != 0:
    raise RuntimeError('S16 requires active assertions; Python optimization forbidden')
import re
import struct


@lru_cache(maxsize=8192)
def words(base, step):
    return struct.pack('<4096I', *[((base + step * i) & 0xffffffff) for i in range(4096)])


def verify_values(directory, row, case_id):
    directory = Path(directory)
    assert type(row['errors']) is int and row['errors'] == 0
    match = re.fullmatch(r'(gmem_to_smem|smem_to_gmem)_16kib_s([124])_r([124])_(one_cta|all_gpu)', case_id)
    assert match and row['case_id'] == case_id
    direction, stages, requests, scope = match.groups()
    stages, requests = int(stages), int(requests)
    g2s = direction == 'gmem_to_smem'
    blocks, iterations, seed, invocation = (row[k] for k in ('blocks', 'iterations', 'seed', 'final_invocation'))
    assert all(type(x) is int for x in (blocks, iterations, seed, invocation))
    assert all(type(row[k]) is int for k in ('threads', 'software_stages', 'requests_per_item',
        'payload_bytes', 'global_slots_per_cta', 'work_count', 'read_payload_bytes',
        'write_payload_bytes', 'start_ns', 'stop_ns', 'post_timing_final_export_bytes'))
    assert blocks > 0 and 128 <= iterations <= 65536 and 0 <= seed <= 0xffffffff
    assert scope != 'one_cta' or blocks == 1
    assert row['scope'] == scope and row['threads'] == 128 and row['capture'] is False
    assert row['software_stages'] == stages and row['requests_per_item'] == requests
    assert row['payload_bytes'] == 16384 and row['global_slots_per_cta'] == 32
    assert row['phase'] in ('pilot', 'formal')
    if row['phase'] == 'pilot':
        assert iterations == 128 and invocation == 1 and row['warmup_executed'] is False
    else:
        assert row['warmup_executed'] is True and 8 <= len(row['warmup_samples_ns']) <= 30
        assert invocation == len(row['warmup_samples_ns']) + 1
    ring_words = blocks * 32 * requests * 4096
    final_words = blocks * stages * requests * 4096
    shapes = {'ring_guards': [ring_words + 8], 'final_slots': [blocks, stages, requests, 4096],
              'counts': [blocks, 12, 2], 'stamps': [blocks, 5, 2]}
    artifacts = row['full_output_artifacts']
    assert len(artifacts) == 4
    for item, (leaf, shape) in zip(artifacts, shapes.items()):
        assert item['path'] == f'stage_{invocation}_{leaf}.u32le'
        assert item['dtype'] == 'uint32' and item['shape'] == shape
        assert all(type(size) is int and size > 0 for size in item['shape'])
        path = directory / item['path']
        assert path.is_file() and not path.is_symlink()
        count = 1
        for size in shape:
            count *= size
        assert path.stat().st_size == count * 4
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        assert h.hexdigest() == item['sha256']
    def path(leaf):
        return directory / f'stage_{invocation}_{leaf}.u32le'
    with path('ring_guards').open('rb') as stream:
        assert stream.read(16) == struct.pack('<4I', *range(0xd15ea5e0, 0xd15ea5e4))
        for block in range(blocks):
            for slot in range(32):
                for request in range(requests):
                    # N>=128 visits every one of the 32 global slots. S divides
                    # 32, so the last global-slot write uses software slot slot%S.
                    index = ((block * 32 + slot) * requests + request) * 4096 if g2s else (
                        (block * stages + slot % stages) * requests + request) * 4096
                    step = 17 if g2s else 29
                    assert stream.read(16384) == words((seed + step * index) & 0xffffffff, step), 'full global ring mismatch'
        assert stream.read(16) == struct.pack('<4I', *range(0xd15ea5e4, 0xd15ea5e8))
        assert stream.read(1) == b''
    with path('final_slots').open('rb') as stream:
        for block in range(blocks):
            for slot in range(stages):
                last_item = slot + ((iterations - 1 - slot) // stages) * stages
                for request in range(requests):
                    index = ((block * 32 + last_item % 32) * requests + request) * 4096 if g2s else (
                        (block * stages + slot) * requests + request) * 4096
                    step = 17 if g2s else 29
                    assert stream.read(16384) == words((seed + step * index) & 0xffffffff, step), 'full SMEM final slots mismatch'
        assert stream.read(1) == b''
    counts = list(struct.iter_unpack('<12Q', path('counts').read_bytes()))
    stamps = list(struct.iter_unpack('<5Q', path('stamps').read_bytes()))
    assert len(row['blocks_detail']) == len(counts) == len(stamps) == blocks
    for block, (counter, stamp, detail) in enumerate(zip(counts, stamps, row['blocks_detail'])):
        expected = [iterations * requests, 0 if g2s else iterations, iterations if g2s else 0,
                    iterations, 0, iterations, iterations - stages, 0 if g2s else 1, 0, None if g2s else 0,
                    stages if g2s else 0, stages if g2s else 0]
        assert all((iterations <= actual < 2**64 - 1) if wanted is None else actual == wanted
                   for actual, wanted in zip(counter, expected))
        assert all(x < 2**64 - 1 for x in stamp[:4]) and stamp[4] < 2**32 - 1
        assert stamp[1] > stamp[0] and stamp[3] > stamp[2]
        assert detail['block_id'] == block
        assert [detail[k] for k in ('start_ns', 'stop_ns', 'start_cycle', 'stop_cycle', 'smid')] == list(stamp)
    assert row['start_ns'] == min(t[0] for t in stamps) and row['stop_ns'] == max(t[1] for t in stamps)
    checked = ring_words + 8 + final_words + blocks * 34
    assert row['correctness']['checked_elements'] == checked
    work = blocks * iterations * requests * 16384
    assert row['work_unit'] == 'byte' and row['work_count'] == work
    assert row['read_payload_bytes'] == (work if g2s else 0) and row['write_payload_bytes'] == (0 if g2s else work)
    assert row['post_timing_final_export_bytes'] == final_words * 4
    identity_path = path('identity').with_suffix('.json')
    assert identity_path.is_file() and not identity_path.is_symlink()
    identity = json.loads(identity_path.read_text())
    assert identity['capture'] is False and identity['validation_failed'] is False
    assert all(type(identity[k]) is int for k in ('iterations', 'seed', 'invocation', 'errors',
        'checked_elements', 'blocks', 'software_stages', 'requests_per_item'))
    assert identity == {'case_id': case_id, 'phase': row['phase'], 'iterations': iterations, 'seed': seed,
                        'invocation': invocation, 'capture': False, 'errors': 0, 'checked_elements': checked,
                        'blocks': blocks, 'software_stages': stages, 'requests_per_item': requests,
                        'validation_failed': False, 'diagnostic': ''}
    return {'complete_final_values_verified': True, 'checked_uint32_elements': checked,
            'work_count': work, 'qualified': False}
