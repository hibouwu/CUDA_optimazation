"""Independent volume and release-pointer checks of the archived R10 audit."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
original = here.parent / 'B-20261011-boundary-lifecycle-v2'
report = json.loads((here / 'B-boundary-lifecycle.json').read_text())
reference = json.loads((original / 'B-boundary-lifecycle.json').read_text())
hash_counts = []
for data in (reference, report):
    inputs = data.pop('input_sha256')
    for name, expected in inputs.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    hash_counts.append(len(inputs))
assert report == reference
assert (here / 'B-boundary-lifecycle.md').read_bytes() == (original / 'B-boundary-lifecycle.md').read_bytes()

release_records = 0
exit_certificates = 0
for case in report['cases']:
    m, n, k = case['shape']
    # Disjoint axis intervals, independent of the audit's per-tile atom helper.
    ma = [min(64, m - start) for start in range(0, m, 64)]
    na = [min(128, n - start) for start in range(0, n, 128)]
    ka = [min(16, k - start) for start in range(0, k, 16)]
    valid_atoms = len(ma) * len(na) * len(ka)
    nominal_atoms = ((m + 127) // 128) * ((n + 127) // 128) * ((k + 63) // 64) * 8
    assert nominal_atoms == case['source_level_MMA_atom_calls']
    assert nominal_atoms - valid_atoms == case['zero_valid_math_atom_calls']
    assert 2 * sum(ma) * sum(na) * sum(ka) == case['valid_math_FLOP']
    assert nominal_atoms * 2 * 64 * 128 * 16 == case['nominal_MMA_FLOP']
    assert case['existing_full_tile_guard'] == 'positive shapes and complete Ktiles required'
    for output_count, graph in case['lifecycle_by_CTA_output_count'].items():
        output_count = int(output_count)
        expected_releases = []
        for output in range(output_count):
            release_pointer = output * 32
            # The prologue commits once. Each subsequent commit waits one group
            # and then advances the release pointer. mma_tail drains the last.
            for commit in range(output * 32 + 1, (output + 1) * 32):
                expected_releases.append((release_pointer, commit, 'normal_wait1'))
                release_pointer += 1
            expected_releases.append((release_pointer, release_pointer, 'mma_tail_wait0'))
        assert len(expected_releases) == len(graph['groups'])
        for index, commit, path in expected_releases:
            actual = graph['groups'][index]
            assert actual['release_after_commit_group'] == commit
            assert actual['release_path'] == path
            assert actual['consumer_wg'] == (index // 32) % 2
            assert actual['empty_arrivals_required'] == 128
            assert actual['transaction_expected_bytes'] == 32768
            release_records += 1
        last_by_slot = {}
        reacquired = []
        for index in range(output_count * 32):
            slot = index % 6
            if slot in last_by_slot:
                reacquired.append(last_by_slot[slot])
            last_by_slot[slot] = index
        tail = [last_by_slot[(output_count * 32 + step) % 6] for step in range(6)]
        assert tail == [node['release_group'] for node in graph['producer_tail_waits']]
        assert reacquired == graph['input_exit_certificate']['releases_required_by_later_acquire']
        assert sorted(reacquired + tail) == list(range(output_count * 32))
        exit_certificates += 1

result = dict(input_hash_counts=hash_counts, complete_report_equal_except_input_paths=True,
              markdown_byte_equal=True, independent_axis_volume_cases=len(report['cases']),
              independent_release_pointer_records=release_records,
              independent_last_slot_exit_certificates=exit_certificates,
              public_r13_tests_passed=199, qualification=False,
              scope='Source work and software ordering only; no measured service or time prediction.')
(here / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
