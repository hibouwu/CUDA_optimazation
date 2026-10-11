"""Verify complete main replay without rerunning unchanged helper tests."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
original = here.parent / 'B-20261011-boundary-transfer-v1'
reference = json.loads((original / 'B-boundary-transfer.json').read_text())
replay = json.loads((here / 'B-boundary-transfer.json').read_text())
hashes = []
for report in (reference, replay):
    inputs = report.pop('input_sha256')
    for name, expected in inputs.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    hashes.append(inputs)
assert reference == replay
common = set(hashes[0]) & set(hashes[1])
assert sorted(hashes[0][key] for key in set(hashes[0]) - common) == sorted(hashes[1][key] for key in set(hashes[1]) - common)
for case in reference['cases']:
    assert (original / case['ledger_file']).read_bytes() == (here / case['ledger_file']).read_bytes()
assert (original / 'B-boundary-transfer.md').read_bytes() == (here / 'B-boundary-transfer.md').read_bytes()
result = dict(input_hashes_per_report=list(map(len, hashes)), common_input_paths=len(common),
    relocated_modules_identical=len(set(hashes[0]) - common), complete_report_equal_except_paths=True,
    complete_ledgers_byte_equal=len(reference['cases']), markdown_byte_equal=True,
    helper_tests_reused_unchanged=True, no_timing_or_calibration_transfer=True, qualification=False)
(here / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
