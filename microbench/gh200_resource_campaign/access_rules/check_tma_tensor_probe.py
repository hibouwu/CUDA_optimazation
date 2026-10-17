"""核对 tensor 探针的原生正确性包；不评分性能或授予机制资格。"""
import argparse
import hashlib
import json
from pathlib import Path


REFERENCE_GPU = 'GPU-43269fbc-449d-3e0f-908a-9c81229546d3'


def check_sample(sample):
    n, q, groups = sample['ctas'], sample['active_groups'], sample['groups']
    assert sample['gpu_uuid'] == REFERENCE_GPU, 'reference GPU changed'
    assert sample['verify'] == 1 and sample['trace'] == 1, 'expected correctness with events'
    assert sample['errors'] == 0, 'data mismatch'
    assert sample['checked'] == n*groups*16384, 'incomplete element check'
    assert sample['smem'] == 132096 and sample['max_active_ctas_per_sm'] == 1, 'resource mismatch'
    assert sample['observed_sms'] == n, 'CTA placement mismatch'
    assert len(sample['ctas_data']) == n and len(sample['group_events']) == n*groups, 'missing records'
    assert len({c['smid'] for c in sample['ctas_data']}) == n, 'duplicate SM placement'
    for name in ('descriptor_a', 'descriptor_b'):
        assert len(bytes.fromhex(sample[name])) == 128, 'missing encoded descriptor'
    ctas = {c['cta']: c for c in sample['ctas_data']}
    events = {(e['cta'], e['group']): e['points'] for e in sample['group_events']}
    assert set(ctas) == set(range(n)), 'CTA identity mismatch'
    assert set(events) == {(c, g) for c in range(n) for g in range(groups)}, 'group identity mismatch'
    for c in range(n):
        for g in range(groups):
            points = events[c, g]
            assert len(points) == 5 and all(len(p) == 2 for p in points), 'timestamp ABI mismatch'
            for unit in (0, 1):
                values = [p[unit] for p in points]
                assert values == sorted(values), 'local event order violated'
                assert ctas[c]['begin'][unit] <= values[0] <= values[-1] <= ctas[c]['end'][unit]
                if g >= q:
                    assert points[0][unit] >= events[c, g-q][4][unit], 'active window reused before release'
                if g >= 4:
                    assert points[0][unit] >= events[c, g-4][4][unit], 'physical slot reused before release'
    return dict(ctas=n, active_groups=q, groups=groups, independent=sample['independent'],
                checked=sample['checked'], events=len(events), errors=sample['errors'])


def check_run(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest() == expected, name
    assert (root/'exit_code.txt').read_text().strip() == '0', 'native job failed'
    rows = []
    for n in (1, 16, 132):
        for q in (1, 4):
            for independent in (0, 1):
                for groups in (8, 33):
                    path = root/f'samples/check-n{n}-q{q}-s{independent}-g{groups}.json'
                    sample = json.loads(path.read_text())
                    assert (sample['ctas'], sample['active_groups'], sample['independent'], sample['groups']) == (n,q,independent,groups)
                    rows.append(dict(file=str(path.relative_to(root)), **check_sample(sample)))
    for path in sorted((root/'preflight').glob('*.json')):
        if 'sanitizer' in path.name:
            output = path.read_text() + path.with_suffix('.log').read_text()
            assert 'ERROR SUMMARY: 0 errors' in output, str(path)
            samples = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
            assert len(samples) == 1
            check_sample(samples[0])
        else:
            rows.append(dict(file=str(path.relative_to(root)), **check_sample(json.loads(path.read_text()))))
    sanitizer_stdout = (root/'samples/sanitizer.json').read_text()
    sanitizer = sanitizer_stdout + (root/'samples/sanitizer.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in sanitizer, 'memcheck did not pass'
    records = [json.loads(line) for line in sanitizer_stdout.splitlines() if line.startswith('{')]
    assert len(records) == 1, 'expected one sanitizer probe result'
    check_sample(records[0])
    sass = (root/'build/probe.sass').read_text()
    return dict(qualification='native correctness and event-order checks only',
        samples=len(rows), checked_elements=sum(r['checked'] for r in rows),
        group_events=sum(r['events'] for r in rows), source_files=len(manifest['source_sha256']),
        tma_3d_static_instructions=sass.count('UTMALDG.3D'), memcheck='0 errors', rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check_run(args.run)
    encoded = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(encoded)
        print(f'PASS {report["samples"]} native checks, {report["checked_elements"]} elements, '
              f'{report["group_events"]} group records; memcheck 0 errors')
    else:
        print(encoded, end='')
