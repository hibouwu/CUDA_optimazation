#!/usr/bin/env python3
"""Replay R16 quota witnesses and instruction boundaries, including the legacy pilot."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cv(values):
    return statistics.pstdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def identity(root):
    legacy = (root / 'identity.json').exists()
    manifest = root / ('source_binary_hashes.json' if legacy else 'source_hashes.json')
    hashes = json.loads(manifest.read_text())
    for name, digest in hashes.items():
        if sha(root / name) != digest:
            raise ValueError('changed frozen artifact: ' + name)
    if legacy:
        meta = json.loads((root / 'identity.json').read_text())
        dependency = (root / meta['cutlass_source']).resolve()
        dependency_root = dependency.parent.parent
        dependency_hashes = json.loads((dependency_root / 'source_hashes.json').read_text())
        headers = {n: h for n, h in dependency_hashes.items() if n.startswith('source/cutlass/')}
        if not headers:
            raise ValueError('missing legacy dependency hashes')
        for name, digest in headers.items():
            if sha(dependency_root / name) != digest:
                raise ValueError('changed CUTLASS dependency: ' + name)
        return dict(legacy=True, artifacts=hashes, cutlass_headers=headers,
                    dependency_path=str(dependency), environment=meta)
    binary = json.loads((root / 'build/binary.json').read_text())
    for name, key in [('r16', 'sha256'), ('sass.txt', 'sass_sha256')]:
        if sha(root / 'build' / name) != binary[key]:
            raise ValueError('changed binary/SASS: ' + name)
    config = json.loads((root / 'run_config.json').read_text())
    if config['subset'] != 'quota' or config['cases_sha256'] != sha(root / 'cases.json'):
        raise ValueError('changed quota matrix')
    return dict(legacy=False, source_manifest_sha256=sha(manifest), binary=binary,
                environment=json.loads((root / 'environment.json').read_text()))


def sass(root):
    path = root / 'build/sass.txt'
    if not path.exists():
        path = root / 'sass.txt'
    functions = [b for b in path.read_text().split('Function : ')[1:] if 'quota16' in b.splitlines()[0]]
    if len(functions) != 1:
        raise ValueError('expected one quota kernel')
    body = functions[0]
    if re.search(r'\b(?:LDL|STL)\b', body):
        raise ValueError('quota spill/local instruction')
    instructions = []
    for line in body.splitlines():
        match = re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;', line)
        if match:
            instructions.append((int(match[1], 16), match[2]))
    allocations = [i for i, (_, op) in enumerate(instructions) if 'USETMAXREG.TRY_ALLOC' in op]
    releases = [i for i, (_, op) in enumerate(instructions) if 'USETMAXREG.DEALLOC' in op]
    if len(allocations) != 1 or len(releases) != 1:
        raise ValueError('expected single inc retry loop and single dec')
    facts = []
    for index, limit, operation in [(allocations[0], '0xe8', 'inc'), (releases[0], '0x28', 'dec')]:
        if limit not in instructions[index][1]:
            raise ValueError('wrong quota limit')
        before = max(i for i in range(index) if 'SR_CLOCKLO' in instructions[i][1])
        after = next(i for i in range(index + 1, len(instructions)) if 'SR_CLOCKLO' in instructions[i][1])
        between = instructions[before + 1:after]
        if any('LDG' in op or 'STG' in op for _, op in between):
            raise ValueError('memory work moved inside quota window')
        if operation == 'inc':
            branches = [int(m[1], 16) for _, op in between if (m := re.search(r'BRA 0x([0-9a-f]+)', op))]
            if len(branches) != 1 or not instructions[before][0] < branches[0] <= instructions[index][0]:
                raise ValueError('inc retry must exclude initial timestamp and include allocation')
        facts.append(dict(operation=operation, begin=hex(instructions[before][0]),
                          end=hex(instructions[after][0]), instructions=[op for _, op in between]))
    inc_end = next(i for i in range(allocations[0] + 1, len(instructions)) if 'SR_CLOCKLO' in instructions[i][1])
    consumer = instructions[inc_end + 1:releases[0]]
    adds = [i for i, (_, op) in enumerate(consumer) if re.search(r'\bFADD\b', op)]
    loads = [i for i, (_, op) in enumerate(consumer) if 'LDG.' in op]
    if len(adds) != 192 or len(loads) != 192:
        raise ValueError('expected 192 value loads and 192 dependent additions')
    clocks = [i for i, (_, op) in enumerate(consumer) if 'SR_CLOCKLO' in op]
    compute = [i for i in clocks if max(loads) < i < min(adds)]
    finish = [i for i in clocks if max(adds) < i < next(i for i, (_, op) in enumerate(consumer) if 'STG.' in op)]
    # The old binary has only producer clocks after consumer stores.
    compute_boundary = bool(compute and finish and max(loads) < min(adds))
    compute_facts = None
    if compute_boundary:
        compute_facts = dict(begin=hex(consumer[compute[-1]][0]), end=hex(consumer[finish[0]][0]),
                             first_add=hex(consumer[min(adds)][0]), last_add=hex(consumer[max(adds)][0]),
                             endpoint_sequence=[op for _, op in consumer[max(adds):finish[0]+1]])
    return dict(kernel=body.splitlines()[0], quota_boundaries=facts, loads=192, dependent_fadd=192,
                loads_interleaved_with_reduction=max(loads)>min(adds),
                compute_boundary_present=compute_boundary, compute_boundary=compute_facts,
                note='SASS ordering checked; no isolated instruction latency or physical pool arbitration inferred')


def replay(path, record):
    def raw(name, kind, size):
        data = gzip.open(path / (name + '.gz'), 'rb').read()
        meta = record['files'][name]
        if len(data) != size or len(data) != meta['bytes'] or hashlib.sha256(data).hexdigest() != meta['sha256']:
            raise ValueError('quota witness identity mismatch: ' + str(path))
        return struct.unpack('<' + kind * (len(data) // 4), data)
    values = raw('values.f32', 'f', 256 * 192 * 4)
    integers = raw('integers.u32', 'I', 384 * 4)
    for t in range(128):
        integer = t + 17
        for _ in range(record['delay']):
            integer = (integer * 1664525 + 1013904223) % (1 << 32)
        if integers[t] != integer:
            raise ValueError('wrong dependent integer result')
    for t in range(256):
        total = 0.0
        for j in range(192):
            expected = (t + 1) / 1024 + j / 32
            if values[t * 192 + j] != expected:
                raise ValueError('wrong retained register value')
            total = struct.unpack('<f', struct.pack('<f', total + expected))[0]
        bits = struct.unpack('<I', struct.pack('<f', total))[0]
        if integers[t + 128] != bits:
            raise ValueError('wrong sequential FP32 reduction')
    if record['local_bytes'] or record['threads'] != 384 or record['registers'] % 8:
        raise ValueError('wrong quota resources')
    initial = record['registers']
    if not 40 < initial < 232 or initial * 384 > 65536 or 128 * 40 + 256 * 232 > 65536:
        raise ValueError('illegal register pool budget')
    stamps = record['stamps']
    if len(stamps) != 12 or len({s[5] for s in stamps}) != 1:
        raise ValueError('missing warp/SM stamp')
    modern = record.get('schema_version', 1) >= 2
    for i, s in enumerate(stamps):
        if len(s) != (9 if modern else 6) or not 0 < s[0] <= s[1] <= s[2]:
            raise ValueError('invalid quota clock order')
        if modern and s[5] != s[8]:
            raise ValueError('clock interval crossed SM')
        if i < 4:
            if not 0 < s[3] <= s[4]:
                raise ValueError('invalid producer delay')
            if record['before'] and not s[0] <= s[3] <= s[4] <= s[1]:
                raise ValueError('producer delay not before release')
            if not record['before'] and not s[2] <= s[3] <= s[4]:
                raise ValueError('producer delay not after release')
        elif modern and not s[2] <= s[6] < s[7]:
            raise ValueError('invalid consumer computation interval')
    warm = record['warmup']
    converged = 8 <= len(warm) <= 30 and cv(warm[-5:]) <= .02
    if converged != record['warmup_converged']:
        raise ValueError('incorrect warmup convergence flag')
    return dict(checked_values=len(values) + len(integers),
                inc_cycles=statistics.median(s[2] - s[1] for s in stamps[4:]),
                dec_cycles=statistics.median(s[2] - s[1] for s in stamps[:4]),
                delay_cycles=statistics.median(s[4] - s[3] for s in stamps[:4]),
                compute_cycles=statistics.median(s[7] - s[6] for s in stamps[4:]) if modern else None,
                modern=modern)


def analyze(root, output=None, sass_only=False, check_only=False):
    legacy = (root / 'identity.json').exists()
    if legacy and output is None:
        raise ValueError('legacy archive requires explicit --output under reanalysis; never replace snapshots')
    destination = output or root / 'analysis'
    destination.mkdir(parents=True, exist_ok=True)
    provenance = identity(root)
    facts = sass(root)
    (destination / 'identity-check.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (destination / 'sass-check.json').write_text(json.dumps(facts, indent=2) + '\n')
    if sass_only:
        print('R16 quota SASS/identity passed'); return
    records = []
    for path in sorted((root / 'samples').glob('*/*/result.json')):
        record = json.loads(path.read_text())
        if record.get('returncode', 0):
            raise ValueError('failed process cannot qualify')
        record['replay'] = replay(path.parent, record)
        record['sample'] = str(path.relative_to(root))
        if not legacy:
            expected_id = f"quota_delay{record['delay']}_{'before_dec' if record['before'] else 'after_dec'}"
            if record['case_id'] != expected_id or path.parent.parent.name != expected_id:
                raise ValueError('quota coordinate mismatch')
        records.append(record)
    if not records:
        raise ValueError('no quota witnesses')
    checks = dict(processes=len(records), checked_values=sum(r['replay']['checked_values'] for r in records))
    (destination / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    if check_only:
        print('R16 quota exact replay', checks); return
    rows = []
    for delay in [0, 64, 256]:
        for before in [0, 1]:
            rr = [r for r in records if r['delay'] == delay and r['before'] == before and
                  (legacy or r['label'].startswith('formal-'))]
            if len(rr) < 3:
                raise ValueError('missing independent quota processes')
            waits = [r['replay']['inc_cycles'] for r in rr]
            computations = [r['replay']['compute_cycles'] for r in rr if r['replay']['modern']]
            reasons = []
            if legacy:
                reasons += ['pilot_was_not_adjacent_randomized_pair_sampling', 'warmup_monitored_producer_delay',
                            'consumer_compute_endpoint_missing', 'end_SM_not_recorded']
            if cv(waits) > .01 and len(rr) < 10:
                reasons.append('requires_ten_processes')
            if cv(waits) > .05:
                reasons.append('CV_exceeds_5_percent')
            if any(not r['warmup_converged'] for r in rr):
                reasons.append('warmup_not_converged')
            if not legacy and (not facts['compute_boundary_present'] or len(computations) != len(rr)):
                reasons.append('compute_boundary_missing')
            rows.append(dict(delay=delay, before=before, processes=len(rr), inc_cycles=statistics.median(waits),
                             inc_cv=cv(waits), delay_cycles=statistics.median(r['replay']['delay_cycles'] for r in rr),
                             dec_cycles=statistics.median(r['replay']['dec_cycles'] for r in rr),
                             compute_cycles=statistics.median(computations) if computations else None,
                             compute_cv=cv(computations) if computations else None,
                             registers=sorted({r['registers'] for r in rr}), qualified=not reasons, limitations=reasons))
    result = dict(subset='quota', conditions=rows, **checks, unit='same_SM_clock64_cycles',
                  numeric_passed=True, sass_quota_order_passed=True,
                  qualified_conditions=sum(r['qualified'] for r in rows),
                  initial_pool=sorted({384*r['registers'] for r in records}), steady_pool=64512,
                  work=dict(producer_imad='128*delay', consumer_fadd=256*192,
                            retained_fp32=256*192, logical_input_bytes=256*192*4),
                  scope='single CTA, one producer and two consumer warpgroups; conditional waiting and reduction service')
    (destination / 'rules.json').write_text(json.dumps(result, indent=2) + '\n')
    (destination / 'processes.json').write_text(json.dumps([dict(sample=r['sample'], **r['replay']) for r in records], indent=2)+'\n')
    with (destination / 'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('R16 quota', checks, 'qualified', result['qualified_conditions'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sass-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    analyze(args.input.resolve(), args.output, args.sass_only, args.check_only)
