"""Verify the complete replay and the original C delivery hashes."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
original = here.parent / 'C-20261011-record-partitions-v1'
delivery = json.loads((original / 'delivery.json').read_text())
for name, expected in delivery['artifact_sha256'].items():
    assert hashlib.sha256((original / name).read_bytes()).hexdigest() == expected, name
reference = json.loads((original / 'data/report.json').read_text())
replay = json.loads((here / 'data/report.json').read_text())
inputs = []
for data in (reference, replay):
    hashes = data.pop('input_sha256')
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    inputs.append(hashes)
assert reference == replay
common = set(inputs[0]) & set(inputs[1])
assert len(common) == 36
assert {inputs[0][key] for key in set(inputs[0]) - common} == {inputs[1][key] for key in set(inputs[1]) - common}
payloads = ['call-identities.json', 'records.json.gz', 'record-partitions.csv', 'signed-record-differences.csv']
for name in payloads:
    assert (original / 'data' / name).read_bytes() == (here / 'data' / name).read_bytes(), name
result = dict(delivery_hashes_verified=len(delivery['artifact_sha256']),
              input_hashes_per_report=[len(value) for value in inputs], common_input_paths=len(common),
              relocated_analyzer_byte_identical=True, complete_report_equal_except_input_paths=True,
              complete_payloads_byte_equal=payloads, qualification=False,
              scope='Exact record arithmetic, not causal map comparison or qualified timing.')
(here / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
