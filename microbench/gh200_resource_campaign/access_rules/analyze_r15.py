#!/usr/bin/env python3
"""CPU-only R15 fragment mapping, source lifetime, output/background and pair replay."""
import argparse
import array
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

MODES = ('reg_smem', 'tma_only', 'full_output', 'background', 'serial', 'concurrent')


def value(row, col):
    return (((row % 16) * 7 + (col % 128) * 11) % 31 - 15) / 32


def coordinate(lane, register, group):
    row = (lane // 32) * 16 + (lane % 32) // 4 + ((register // 2) % 2) * 8
    col = (lane % 4) * 2 + register % 2 + (register // 4) * 8
    return row + group * 64, col


def packed(row, col, n):
    return (row % 32) * 128 + col % 128


def lifetime(chunks, buffers):
    pending = []
    events = []
    for chunk in range(chunks):
        if len(pending) == buffers:
            old = pending.pop(0)
            events.append(('read_done', old))
            events.append(('released', old))
        events.append(('prepare', chunk))
        events.append(('issued', chunk))
        pending.append(chunk)
    for old in pending:
        events.append(('read_done', old))
    events.append(('full_done_all', chunks - 1))
    for old in pending:
        events.append(('released', old))
    return events


def cpu_check():
    proof = []
    for n in (128, 256):
        chunks = 128 * n // 4096
        seen = set()
        fragment_slots = [set() for _ in range(chunks)]
        for group in (0, 1):
            for lane in range(128):
                for register in range(n // 2):
                    row, col = coordinate(lane, register, group)
                    assert 0 <= row < 128 and 0 <= col < n
                    assert (row, col) not in seen
                    seen.add((row, col))
                    chunk = (col // 128) * 4 + row // 32
                    index = packed(row, col, n)
                    assert index not in fragment_slots[chunk]
                    fragment_slots[chunk].add(index)
                    # v3: same mapping with one base and64 immediate offsets.
                    chunk_m, chunk_n = chunk % 4, chunk // 4
                    assert group == chunk_m // 2 and lane // 64 == chunk_m % 2
                    local_register = register % 64
                    assert register // 64 == chunk_n
                    row_base = ((lane // 32) % 2) * 16 + (lane % 32) // 4
                    col_base = (lane % 4) * 2
                    immediate_words = ((local_register // 2) % 2) * 8 * 128
                    immediate_words += (local_register // 4) * 8 + local_register % 2
                    assert row_base * 128 + col_base + immediate_words == index
                    # Immutable TMA-only boxes match every full-output chunk.
                    assert value(row, col) == value(row % 32, col % 128)
        assert len(seen) == 128 * n
        assert all(indices == set(range(4096)) for indices in fragment_slots)
        for buffers in (1, 2):
            events = lifetime(chunks, buffers)
            issued = {chunk: i for i, (event, chunk) in enumerate(events) if event == 'issued'}
            read = {chunk: i for i, (event, chunk) in enumerate(events) if event == 'read_done'}
            release = {chunk: i for i, (event, chunk) in enumerate(events) if event == 'released'}
            prepare = {chunk: i for i, (event, chunk) in enumerate(events) if event == 'prepare'}
            for chunk in range(chunks):
                assert prepare[chunk] < issued[chunk] < read[chunk] <= release[chunk]
                if chunk + buffers < chunks:
                    assert release[chunk] < prepare[chunk + buffers]
            proof.append(dict(n=n, fragments=len(seen), chunks=chunks, buffers=buffers,
                              bytes_per_chunk=16384, boxes_per_chunk=1, tma_box_shape=[128,32],
                              tma_box_bytes=16384,
                              background_mma=16 * chunks, lifetime_events=events))
    assert len(proof) == 4
    return dict(status='CPU mapping/lifetime passed', conditions=proof,
                note='No GPU, tensor-map encoding, compilation or timing qualification claimed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_raw(folder, name, digest):
    with gzip.open(folder / (name + '.gz'), 'rb') as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('raw identity mismatch ' + str(folder / name))
    return raw


def replay(record, folder):
    n = 128 if record['kib'] == 64 else 256
    ldd = n + 32
    if record['ldd'] != ldd or record['padding_errors']:
        raise ValueError('output stride or padding mismatch')
    raw = read_raw(folder, 'output.f32', record['raw_hashes']['output.f32'])
    output = array.array('f'); output.frombytes(raw)
    words = array.array('I'); words.frombytes(raw)
    if len(output) != 128 * ldd or len(output) != record['output_elements']:
        raise ValueError('full output length')
    stores = record['mode'] in (1, 2, 4, 5)
    for row in range(128):
        for col in range(ldd):
            index = row * ldd + col
            if stores and col < n:
                if not math.isfinite(output[index]) or output[index] != value(row, col):
                    raise ValueError('fragment or TMA output mismatch')
            elif words[index] != 0x7fc12345:
                raise ValueError('inactive logical output or padding modified')
    background = array.array('f')
    background.frombytes(read_raw(folder, 'background.f32', record['raw_hashes']['background.f32']))
    expected = record['repeats'] * record['chunks'] if record['mode'] >= 3 else 0
    if len(background) != 8192 or any(x != expected for x in background):
        raise ValueError('complete BG mismatch')
    smem = array.array('f')
    smem.frombytes(read_raw(folder, 'smem.f32', record['raw_hashes']['smem.f32']))
    if len(smem) != 8192:
        raise ValueError('SMEM witness length')
    for q, x in enumerate(smem):
        local = q % 4096
        if x != value(local // 128, local % 128):
            raise ValueError('packed SMEM witness mismatch')
    trace = array.array('Q')
    trace.frombytes(read_raw(folder, 'trace.u64', record['raw_hashes']['trace.u64']))
    if len(trace) != record['trace_records'] * 6:
        raise ValueError('trace length')
    if not record['trace']:
        if any(trace):
            raise ValueError('plain unexpectedly traced')
    else:
        selected = record.get('trace_selected_record')
        for q in range(record['trace_records']):
            prepare, issue, read, release, full, bg = trace[q * 6:q * 6 + 6]
            if selected is not None and q != selected:
                if any((prepare, issue, read, release, full, bg)):
                    raise ValueError('unselected trace record was written')
                continue
            if selected is not None and selected != record['repeats']*record['chunks']-1:
                raise ValueError('trace selector mismatch')
            pair_only = record.get('trace_profile') == 'read_full_pair'
            if pair_only:
                if record['mode'] != 1 or not (0 < read <= full):
                    raise ValueError('read/full pair order')
                if prepare or issue or release or bg:
                    raise ValueError('unobserved event fabricated in read/full-only trace')
            elif stores and not (0 < prepare <= issue <= read <= release and issue <= full):
                raise ValueError('source read/release/full ordering')
            if record['mode'] >= 3 and bg == 0:
                raise ValueError('selected background completion missing')
            if record['mode'] == 4 and not full <= bg:
                raise ValueError('serial background precedes full output')
    return dict(output_values=len(output), background_values=len(background),
                smem_values=len(smem), max_error=0)


def audit_sass(path):
    """Conservative machine-code evidence; refuses unknown serialization and spills.

    One BG collective16 site batch must appear in a runtime chunk/repeat loop.
    This is a representative admission gate, not a complete instruction simulator.
    Review the archived body and branch bounds before assigning physical service rules.
    """
    text = Path(path).read_text()
    functions = []
    for part in text.split('Function : '):
        match = re.match(r'_Z12output_probeILi(128|256)ELi([12])ELi([0-5])ELb([01])', part)
        if not match:
            continue
        n, buffers, mode, traced = map(int, match.groups())
        ops = []
        for line in part.splitlines():
            found = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*/\*', line)
            if found:
                ops.append((int(found[1], 16), found[2]))
        loops = []
        for pc, op in ops:
            found = re.search(r'\bBRA\s+0x([0-9a-f]+)', op)
            if found and int(found[1], 16) < pc:
                loops.append((int(found[1], 16), pc))
        mma = [(pc, op) for pc, op in ops if 'HGMMA.64x128x16.F32' in op]
        if mode >= 3:
            if len(mma) != 16 or any(not any(a <= pc <= b for a, b in loops) for pc, _ in mma):
                raise ValueError('BG dynamic16 batch not established; inspect CFG')
        elif mma:
            raise ValueError('baseline has unexpected BG work')
        if any(' STL' in op or ' LDL' in op for pc, op in ops):
            raise ValueError('local spilling is not an R15 service parameter')
        functions.append(dict(n=n, buffers=buffers, mode=mode, traced=bool(traced),
             background_sites=len(mma), backwards_edges=[(hex(a), hex(b)) for a, b in loops],
             tma_store_sites=sum('UTMASTG' in op for pc, op in ops),
             tma_command_groups=sum('UTMACMDFLUSH' in op for pc, op in ops),
             depbar_sites=[dict(pc=hex(pc), op=op) for pc, op in ops if 'DEPBAR' in op],
             note='Need independent source-wait/full-wait and guarded clock CFG review'))
    if len(functions) != 48:
        raise ValueError(f'expected 48 compiled template variants, got {len(functions)}')
    return functions


def analyze(folder, destination):
    folder, destination = Path(folder), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in json.loads((folder / 'source_hashes.json').read_text()).items():
        if sha(folder / 'source' / name) != digest:
            raise ValueError('source identity changed')
    if sha(folder / 'build/r15') != json.loads((folder / 'build/binary_hashes.json').read_text())['r15']:
        raise ValueError('binary identity changed')
    machine = audit_sass(folder / 'build/r15.sass')
    log = (folder / 'build/compile.log').read_text()
    serialization = any(code in log for code in ('C7510', 'C7514', 'C7520'))
    rows, groups, pairs = [], {}, {}
    sample_files = sorted((folder / 'samples').glob('*/*/result.json'))
    for file in sample_files:
        r = json.loads(file.read_text())
        proof = replay(r, file.parent)
        if json.loads(file.with_name('process.json').read_text())['returncode']:
            raise ValueError('nonzero process result')
        key = (r['configuration']['id'], r['role'], r['trace'])
        groups.setdefault(key, []).append(r)
        ordinal = file.parent.name.rsplit('-', 1)[0]
        pairs.setdefault((r['configuration']['id'], ordinal), {})[r['trace']] = r
        rows.append(dict(case_id=key[0], role=key[1], trace=r['trace'],
                         elapsed_cycles=r['elapsed_cycles'], **proof))
    summaries = []
    for (name, role, traced), records in groups.items():
        values = [r['elapsed_cycles'] for r in records]
        r = records[0]
        stores = r['mode'] in (1, 2, 4, 5)
        bg = r['mode'] >= 3
        summaries.append(dict(case_id=name, role=role, traced=traced, processes=len(records),
            median=statistics.median(values), minimum=min(values), maximum=max(values),
            cv=statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None,
            all_warmup_converged=all(r['warmup_converged'] for r in records),
            registers_per_thread=r['registers_per_thread'],
            background_flop=262144 * 16 * r['chunks'] * r['repeats'] if bg else 0,
            tma_bytes=r['kib'] * 1024 * r['repeats'] if stores else 0,
            register_smem_bytes=r['kib'] * 1024 * r['repeats'] if r['mode'] in (0, 2, 4, 5) else 0))
    perturbations = []
    for (name, ordinal), variants in pairs.items():
        if set(variants) != {False, True}:
            raise ValueError('plain/trace pair incomplete')
        plain, trace = variants[False]['elapsed_cycles'], variants[True]['elapsed_cycles']
        perturbations.append(dict(case_id=name, ordinal=ordinal,
                                 relative=(trace - plain) / plain))
    max_perturbation = max((abs(p['relative']) for p in perturbations), default=math.inf)
    representative = [s for s in summaries if s['role'] == 'representative' and not s['traced']]
    admit = (len(representative) == 2 and all(s['processes'] >= 3 for s in representative)
             and max_perturbation <= .05 and not serialization
             and all(s['all_warmup_converged'] for s in representative))
    recheck_ok = ((folder/'protocol-trace-recheck.json').exists()
                  and len({p['case_id'] for p in perturbations})==2
                  and all(s['role']=='tracecheck' and s['processes']>=3 for s in summaries)
                  and max_perturbation<=.05 and not serialization
                  and all(s['all_warmup_converged'] for s in summaries))
    qualified = dict(admit_matrix=admit, trace_recheck_passed=recheck_ok,
                     max_pair_trace_perturbation=max_perturbation,
                     compiler_serialization=serialization,
                     reason='SASS/CPU/order gates + trace perturbation<=5%; independent CFG review remains')
    (destination / 'qualification.json').write_text(json.dumps(qualified, indent=2) + '\n')
    summary = dict(status='conditional observation; GPU qualification pending' if serialization
                   else 'CPU/SASS preliminary replay passed', cases=summaries,
                   machine=machine, trace_pairs=perturbations, qualification=qualified)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if summaries:
        with (destination / 'cases.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)
    text = ['# R15 输出服务', '', '数值、padding、完整背景输出和源生命周期已CPU重算。', '',
            '时间是相同384线程/72KiB动态SMEM预留的单CTA完成窗口；各模式实际寄存器需另核对。', '',
            '源read等待不算写完成；full_done只在非.read的bulk wait0后记录。双缓冲最多两组输出在途。', '',
            'trace扰动超过5%不得进入正式矩阵；先缩小标记范围并保留原失败。', '',
            '| 配置 | role | trace | 进程 | 中位cycle | CV |', '|---|---|---|---:|---:|---:|']
    for s in summaries:
        text.append(f"| {s['case_id']} | {s['role']} | {s['traced']} | {s['processes']} | "
                    f"{s['median']} | {s['cv']} |")
    (destination / 'report.md').write_text('\n'.join(text) + '\n')
    (destination / 'analyzer_identity.json').write_text(json.dumps(dict(
        script_sha256=sha(__file__), python=sys.version), indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
    else:
        if args.input is None or args.output is None:
            parser.error('--input and a NEW --output directory required')
        analyze(args.input, args.output)


if __name__ == '__main__':
    main()
