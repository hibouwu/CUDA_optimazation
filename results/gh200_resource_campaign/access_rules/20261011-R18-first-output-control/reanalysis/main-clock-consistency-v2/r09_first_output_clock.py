#!/usr/bin/env python3
"""A-only clock consistency audit of the frozen 740068 first-output control.

Replays existing raw data. It neither repairs observer gates nor estimates a
physical frequency. Shared-curve certificates concern an explicit hypothesis.
"""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile

import r09_clock_brackets
import r09_shared_clock


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def replay(run):
    """Bind frozen decoder, binary and every original CTA/role/work endpoint."""
    manifest = json.loads((run / 'manifest.json').read_text())
    frozen = json.loads((run / 'freeze.json').read_text())
    inputs = {}
    for name, digest in frozen['files'].items():
        for path in (run / name, run / 'returned' / name):
            if sha(path) != digest:
                raise ValueError('frozen identity changed: ' + str(path))
            inputs[str(path)] = digest
    for name, digest in manifest['analysis_sha256'].items():
        path = run / 'analysis_snapshot' / name
        if sha(path) != digest:
            raise ValueError('frozen analysis changed: ' + name)
        inputs[str(path)] = digest
    with tarfile.open(run / 'input.tar.gz') as archive:
        members = [member for member in archive.getmembers() if member.name == 'build/cfg_a_s6_dual']
        if len(members) != 1 or hashlib.sha256(archive.extractfile(members[0]).read()).hexdigest() != manifest['binary_sha256']:
            raise ValueError('archived binary differs from frozen manifest')
    for name in ('input.tar.gz', 'freeze.json', 'returned/records.json', 'returned/complete.json',
                 'returned/assigned-devices.json', 'analysis.json', 'common-mode.py',
                 'reanalysis/main-common-mode-v1/common-mode.json.gz'):
        path = run / name
        inputs[str(path)] = sha(path)
    sys.path.insert(0, str(run / 'analysis_snapshot'))
    from decode_r18_first_output import decode
    from analyze_r18_first_output import SELECTED_CTAS
    from analyze_r18 import reference
    from v08_model import scheduled_work

    records = json.loads((run / 'returned/records.json').read_text())
    if [{k: row[k] for k in ('id', 'trial', 'flag')} for row in records] != manifest['samples'] or len(records) != 20:
        raise ValueError('all twenty frozen samples required in original order')
    if json.loads((run / 'returned/complete.json').read_text()) != dict(manifest_sha256=sha(run / 'manifest.json'), records=20):
        raise ValueError('completion does not bind the frozen manifest')
    assigned = json.loads((run / 'returned/assigned-devices.json').read_text())
    if manifest['gpu_uuid'] not in [row['gpu_uuid'] for row in assigned['devices']]:
        raise ValueError('target GPU was not in original allocation')
    work = scheduled_work('cfg_a', 960, 2816, [132, 1, 1], 1)
    processes = []
    for row in records:
        path = run / 'returned' / row['raw']
        if row['returncode'] or sha(path) != row['sha256']:
            raise ValueError('failed or changed raw record: ' + row['id'])
        inputs[str(path)] = row['sha256']
        with gzip.open(path, 'rt') as stream:
            packets = [json.loads(line) for line in stream]
        events = {packet['event']: packet for packet in packets}
        if len(events) != len(packets):
            raise ValueError('duplicate raw event')
        setup, call, check = (events[key] for key in ('setup', 'call', 'check'))
        if setup['gpu_uuid'] != manifest['gpu_uuid'] or setup['input_map_m'] != 1024 or setup['first_output_observer'] != row['flag']:
            raise ValueError('card, map or arm changed')
        decoded = decode(setup, call, row['id'])
        if decoded['observation_qualified'] or len(decoded['records']) != 24 * row['flag']:
            raise ValueError('decode incorrectly qualified or changed coverage')
        if setup['scratch_bytes'] != 1391808 or setup['trace_tile_capacity'] != 64:
            raise ValueError('preparation capacity changed')
        for packet in (setup, call):
            if packet['target_a_ptr'] != packet['prefix_a_ptr'] or packet['target_b_ptr'] != packet['prefix_b_ptr']:
                raise ValueError('prefix/target buffers differ within a process')
        if len(call['warmup_us']) != 16 or any(not math.isfinite(x) or x <= 0 for x in call['warmup_us']):
            raise ValueError('incomplete prefix event history')
        if not math.isfinite(call['elapsed_us']) or call['elapsed_us'] <= 0:
            raise ValueError('invalid complete-call event duration')
        if check['status'] != 'ok' or check['samples'] != 4096 or check['nonfinite'] or check['padding_errors']:
            raise ValueError('failed numeric checks')
        indices, values = check['checked_indices'], check['checked_values']
        if len(indices) != 4096 or len(set(indices)) != 4096 or len(values) != 4096:
            raise ValueError('incomplete distinct numeric sample set')
        for index, value in zip(indices, values):
            if type(index) is not int or not 0 <= index < 960 * 2816 or not math.isfinite(value) or value != reference(index // 2816, index % 2816, 4096, 17):
                raise ValueError('independent numeric replay failed')
        zero = events['input_zero_storage']
        if zero['checked'] != 64 * 4096 or zero['nonzero']:
            raise ValueError('zero storage witness changed')
        trace = call['trace']
        first_points, full_points, local, ctas = {}, {}, [], []
        for cta in range(132):
            block = trace[cta * 1318:(cta + 1) * 1318]
            sm = block[2] - 1
            if block[10] or any(block[14:16]) or not 0 <= sm < 132 or sm in full_points:
                raise ValueError('missing/duplicate SM, overflow or reserved head write')
            if not 0 < block[0] <= block[3] or not 0 < block[1] <= block[13]:
                raise ValueError('invalid entry/producer pair')
            first_points[sm] = []
            full_points[sm] = [(block[0], block[1]), (block[3], block[13])]
            labels = ['entry', 'producer']
            for role in (1, 2):
                head = 16 + (role - 1) * 640
                end_c, end_ns, count = block[4 + 3 * (role - 1):7 + 3 * (role - 1)]
                if count != len(work[cta]) or block[10 + role] != block[2]:
                    raise ValueError('original role work/SM identity changed')
                previous_c, previous_ns = block[0], block[1]
                for tile in range(count):
                    offset = head + 10 * tile
                    words = block[offset:offset + 10]
                    cycles, times = words[:4], words[6:10]
                    if words[4:6] != [value + 1 for value in work[cta][tile]] or not all(cycles + times):
                        raise ValueError('original tile coordinate or endpoint missing')
                    if not previous_c <= cycles[0] <= cycles[1] <= cycles[2] <= cycles[3] <= end_c:
                        raise ValueError('original role native-cycle order changed')
                    if not previous_ns <= times[0] <= times[1] <= times[2] <= times[3] <= end_ns:
                        raise ValueError('original role timer order changed')
                    previous_c, previous_ns = cycles[3], times[3]
                    full_points[sm].extend(zip(cycles, times))
                    labels.extend(f'role{role}_tile{tile}_{name}' for name in ('FIRST', 'MAIN', 'PERMIT', 'DONE'))
                    if tile == 0:
                        first_points[sm].extend(zip(cycles[:2], times[:2]))
                        local.append(dict(cta=cta, role=role, sm=sm, selected=cta in SELECTED_CTAS,
                                          cycles=cycles[1] - cycles[0], ns=times[1] - times[0]))
                        if local[-1]['cycles'] <= 0 or local[-1]['ns'] <= 0:
                            raise ValueError('nonpositive original FIRST-to-MAIN window')
                if any(block[head + 10 * count:head + 640]) or end_c <= 0 or end_ns <= 0:
                    raise ValueError('unused tile writes or missing final endpoint')
                full_points[sm].append((end_c, end_ns))
                labels.append(f'role{role}_final')
            witness = r09_clock_brackets.local_clock_witness(full_points[sm])
            ctas.append(dict(cta=cta, sm=sm, labels=labels, points=full_points[sm],
                             local_variable_rate_witness=witness))
        processes.append(dict(id=row['id'], trial=row['trial'], flag=row['flag'], elapsed_us=call['elapsed_us'],
                              prefix_event_us=call['warmup_us'], local=local, ctas=ctas,
                              first_points=first_points, full_points=full_points))
    with gzip.open(run / 'reanalysis/main-common-mode-v1/common-mode.json.gz', 'rt') as stream:
        common = json.load(stream)
    for ours, theirs in zip(processes, common['processes']):
        if any(ours[key] != theirs[key] for key in ('id', 'trial', 'flag', 'local')):
            raise ValueError('independent raw extraction differs from main common-mode report')
    if len(common['processes']) != 20 or common['common_L_control_passed'] or common['internal_endpoints_qualified']:
        raise ValueError('main process coverage or failed qualification changed')
    original = json.loads((run / 'analysis.json').read_text())
    if original['errors'] or any(original[key] for key in ('common_L_control_passed', 'internal_endpoints_qualified', 'observation_qualified')):
        raise ValueError('the original frozen failure must remain a failure')
    return manifest, processes, original, inputs


def paired_decomposition(processes, manifest, original):
    """Exact off-anchored arithmetic, not a causal or hardware-rate partition."""
    pairs = []
    for trial in range(10):
        off, on = [next(p for p in processes if p['trial'] == trial and p['flag'] == flag) for flag in (0, 1)]
        rows = []
        for a, b in zip(off['local'], on['local']):
            if (a['cta'], a['role']) != (b['cta'], b['role']):
                raise ValueError('paired original role changed')
            dc, dn = b['cycles'] - a['cycles'], b['ns'] - a['ns']
            cycle_term = Fraction(dc * a['ns'], a['cycles'])
            conversion_term = Fraction(dn) - cycle_term
            cycle_ratio = Fraction(b['cycles'], a['cycles'])
            rate_ratio = Fraction(b['cycles'] * a['ns'], b['ns'] * a['cycles'])
            if cycle_ratio / rate_ratio != Fraction(b['ns'], a['ns']) or cycle_term + conversion_term != dn:
                raise ValueError('exact paired decomposition did not close')
            rows.append(dict(cta=a['cta'], role=a['role'], selected=a['selected'], sm_changed=a['sm'] != b['sm'],
                             off_cycles=a['cycles'], on_cycles=b['cycles'], off_ns=a['ns'], on_ns=b['ns'],
                             delta_cycles=dc, delta_ns=dn, off_anchored_cycle_term_ns=float(cycle_term),
                             conversion_remainder_ns=float(conversion_term),
                             cycle_term_fraction=[cycle_term.numerator, cycle_term.denominator],
                             conversion_remainder_fraction=[conversion_term.numerator, conversion_term.denominator],
                             apparent_rate_ratio_minus_one=float(rate_ratio - 1)))
        groups = []
        for selected in (True, False):
            group = [row for row in rows if row['selected'] == selected]
            groups.append(dict(selected=selected, roles=len(group),
                               **{key: summary([row[key] for row in group]) for key in
                                  ('delta_cycles', 'delta_ns', 'off_anchored_cycle_term_ns',
                                   'conversion_remainder_ns', 'apparent_rate_ratio_minus_one')}))
        pairs.append(dict(trial=trial, on_was_first=bool(trial % 2), sm_changes=sum(row['sm_changed'] for row in rows),
                          groups=groups, roles=rows))
    local_gates = []
    gates = manifest['gates']
    for row in (row for row in pairs[0]['roles'] if row['selected']):
        values = [next(x['delta_ns'] for x in pair['roles'] if (x['cta'], x['role']) == (row['cta'], row['role'])) for pair in pairs]
        absolute = list(map(abs, values))
        local_gates.append(dict(cta=row['cta'], role=row['role'], signed=values,
                                median_absolute=statistics.median(absolute), maximum_absolute=max(absolute),
                                passed=statistics.median(absolute) <= gates['per_role_L_median_absolute_ns'] and max(absolute) <= gates['per_role_L_maximum_absolute_ns'],
                                sm_changes=[False] * 10))
    if any(pair['sm_changes'] for pair in pairs) or local_gates != original['local']:
        raise ValueError('original gate replay or paired SM preservation failed')
    return pairs, local_gates


def verify_saved(report):
    """Rebuild graphs and check certificates/inversions without solving them."""
    certificates = points_verified = 0
    for process in report['processes']:
        for scope in ('first_L', 'all_original_endpoints'):
            for branch in ('capped', 'uncapped'):
                result = process['shared'][scope][branch]
                points = {int(sm): [tuple(point) for point in values] for sm, values in process['points'][scope].items()}
                cap = 1.98 if branch == 'capped' else None
                upper = result['minimum_integer_endpoint_slack_ns']
                for epsilon, graph_key, certificate_key in ((upper, 'feasible_graph', 'feasible_certificate'),
                                                            (upper - 1, 'infeasible_graph', 'infeasible_certificate')):
                    if epsilon < 0:
                        continue
                    graph = r09_shared_clock.clock_graph(points, epsilon, cap)
                    saved = result[graph_key]
                    saved['clock_origins'] = {int(sm): value for sm, value in saved['clock_origins'].items()}
                    saved['sm_nodes'] = {int(sm): value for sm, value in saved['sm_nodes'].items()}
                    saved['edges'] = [tuple(edge) for edge in saved['edges']]
                    if graph != saved:
                        raise ValueError('saved graph does not match original integer points')
                    r09_shared_clock.verify_certificate(graph, result[certificate_key])
                    certificates += 1
                verification = r09_shared_clock.verify_shared_witness(points, result['feasible_graph'], result['feasible_certificate'])
                if verification != result['original_sample_verification']:
                    raise ValueError('saved original-point inversion differs')
                points_verified += verification['points_verified']
    return dict(boundary_certificates_verified=certificates, original_sample_inversions_verified=points_verified,
                solver_called=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--cap-evidence', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        with gzip.open(args.verify, 'rt') as stream:
            report = json.load(stream)
        for name, digest in report['input_sha256'].items():
            if sha(name) != digest:
                raise ValueError('saved input changed: ' + name)
        print(json.dumps(verify_saved(report)))
        return
    if any(value is None for value in (args.run, args.cap_evidence, args.output)):
        parser.error('--run, --cap-evidence and --output are required for analysis')
    if args.output.exists():
        raise ValueError('output must be a new exclusive A directory')
    cap_evidence = json.loads(args.cap_evidence.read_text())
    if cap_evidence['clock_intercept_cap_ghz'] != 1.98 or sha(cap_evidence['source']) != cap_evidence['sha256']:
        raise ValueError('the original declared cap evidence changed')
    manifest, processes, original, inputs = replay(args.run.resolve())
    inputs[str(args.cap_evidence.resolve())] = sha(args.cap_evidence)
    inputs[cap_evidence['source']] = cap_evidence['sha256']
    modules = [Path(__file__).resolve(), Path(r09_clock_brackets.__file__).resolve(), Path(r09_shared_clock.__file__).resolve()]
    inputs.update({str(path): sha(path) for path in modules})
    pairs, local_gates = paired_decomposition(processes, manifest, original)
    results = []
    for process in processes:
        shared = {}
        for name, key in (('first_L', 'first_points'), ('all_original_endpoints', 'full_points')):
            shared[name] = dict(capped=r09_shared_clock.minimum_shared_slack(process[key], 1.98),
                                uncapped=r09_shared_clock.minimum_shared_slack(process[key], None))
        results.append(dict(id=process['id'], trial=process['trial'], flag=process['flag'], elapsed_us=process['elapsed_us'],
                            prefix_event_us=process['prefix_event_us'], ctas=process['ctas'],
                            points=dict(first_L=process['first_points'], all_original_endpoints=process['full_points']), shared=shared))
        print(process['id'], {scope: {branch: value['minimum_integer_endpoint_slack_ns'] for branch, value in branches.items()} for scope, branches in shared.items()}, flush=True)
    report = dict(schema='gh200-r09-first-output-clock/1', input_sha256=inputs, binary_sha256=manifest['binary_sha256'],
                  cap_evidence=cap_evidence, processes=results, pairs=pairs, original_local_gates=local_gates,
                  common_L_control_passed=False, internal_endpoints_qualified=False, observation_qualified=False,
                  decomposition='DeltaN = (Con-Coff)*Noff/Coff + [DeltaN-(Con-Coff)*Noff/Coff]. Exact per-role off-anchored identity; group medians need not add.',
                  shared_model='C_sm(t)=offset_sm+F(t), nonnegative common F prime; endpoint time in [N-epsilon,N+epsilon]. Capped branch additionally uses the old declared 1.98 bound.',
                  limits=['All twenty original samples retained; no new measurements, fit, changed gate, compensation or outlier deletion.',
                          'Apparent cycle/ns and the conversion remainder contain endpoint effects; neither identifies physical frequency or a causal contribution.',
                          'Unselected CTAs share the invocation and device; they are not an isolated untreated control.',
                          'Shared-curve slack is a hypothesis-compatibility minimum, not measured timer error or clock-read enclosure.',
                          'Only existing entry/producer/original role endpoints enter clock constraints. The five new internal cuts are decoded but excluded from this matched-scope audit.',
                          'No comparison of raw clock64 across SMs or processes. Independent SM offsets remain free.',
                          'The 64/256 ns frozen paired-duration gates are different quantities from a per-endpoint consistency allowance.',
                          'All records use map1024 on the ef GPU; no map960 target or cross-card qualification follows.'])
    args.output.mkdir(parents=True, exist_ok=False)
    encoded = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
    target = args.output / 'clock-consistency.json.gz'
    target.write_bytes(gzip.compress(encoded, mtime=0))
    with gzip.open(target, 'rt') as stream:
        verification = verify_saved(json.load(stream))
    (args.output / 'certificate-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    for path in modules:
        (args.output / path.name).write_bytes(path.read_bytes())
    (args.output / 'input-sha256.json').write_text(json.dumps(inputs, indent=2) + '\n')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
