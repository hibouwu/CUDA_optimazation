#!/usr/bin/env python3
"""Scope the parser revision difference to the actual forty R10 invocations."""
import ast
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

worktree = Path('/home/jianyeshi/.codex/worktrees/0785/CUDA_optimazation/microbench/gh200_resource_campaign/access_rules')
primary = Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation/microbench/gh200_resource_campaign/access_rules')
results = Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules')
output = Path(__file__).resolve().parent
report_path = results / '20261010-R10-producer-delay/reanalysis/B-20261011-paired-estimands-v1/B-paired-estimands.json'
report = json.loads(report_path.read_text())
inputs = {str(report_path): hashlib.sha256(report_path.read_bytes()).hexdigest()}


def digest(path):
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs[str(path)] = value
    return value


texts = [(directory / 'analyze_r18.py').read_text() for directory in (worktree, primary)]


def function(text, name):
    return next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name)


def canonical(node):
    return ast.dump(node, include_attributes=False)


for name in ('v06_run.py', 'v06_model.py', 'analyze_r10_paired_estimands.py'):
    assert digest(worktree / name) == digest(primary / name)
assert digest(worktree / 'analyze_r18.py') != digest(primary / 'analyze_r18.py')
references = [function(text, 'reference') for text in texts]
assert canonical(references[0]) == canonical(references[1])
replays = [function(text, 'replay') for text in texts]
excluded_versions = []
for replay in replays:
    body = []
    removed = 0
    for node in replay.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            test = node.test
            if (ast.unparse(test.left) == "setup.get('trace_version')"
                    and isinstance(test.ops[0], ast.In) and isinstance(test.comparators[0], ast.Tuple)):
                versions = ast.literal_eval(test.comparators[0])
                if 'r13-stage-sparse-v1' in versions:
                    assert 'r18-dual-clock-events' not in versions and not node.orelse
                    excluded_versions.append(list(versions))
                    removed += 1
                    continue
        body.append(node)
    assert removed == 1
    replay.body = body
assert canonical(replays[0]) == canonical(replays[1])
warmups = [function(text, 'validate_warmup') for text in texts]
assert ast.unparse(warmups[1].body[0]) == 'validate_plain_prefix(setup, call, row)'
assert isinstance(warmups[1].body[1], ast.If)
warmups[1].body = warmups[1].body[2:]
assert canonical(warmups[0]) == canonical(warmups[1])

# The referenced dependency modules above are byte-identical. Only the two
# analyze_r18 modules are loaded under separate names in this comparison.
sys.path.insert(0, str(worktree))
modules = []
for index, directory in enumerate((worktree, primary)):
    spec = importlib.util.spec_from_file_location('decoder_review_' + str(index), directory / 'analyze_r18.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modules.append(module)
cases_path = results / '20261010-R10-producer-delay/cases.json'
digest(cases_path)
rows = {row['id']: row for row in json.loads(cases_path.read_text())}
compared_ctas = 0
compared_tiles = 0
for saved in report['records']:
    record_path, raw_path = Path(saved['metadata']), Path(saved['raw'])
    assert digest(record_path) == report['input_sha256'][str(record_path)]
    assert digest(raw_path) == report['input_sha256'][str(raw_path)]
    record = json.loads(record_path.read_text())
    row = rows[saved['case']]
    events = {event['event']: event for line in gzip.decompress(raw_path.read_bytes()).decode().splitlines()
        for event in [json.loads(line)]}
    setup, call = events['setup'], events['call']
    assert setup['trace_version'] == 'r18-dual-clock-events'
    assert 'release_warp_observer' not in row
    assert not any(event.get('release_warp_observer', 0) for event in (setup, call))
    assert row.get('plain_prefix_protocol') is None
    assert not any(key.startswith('plain_prefix_') for event in (setup, call) for key in event)
    before, after = (module.replay(record_path.parents[2], record, row) for module in modules)
    assert before == after
    assert len(before['ctas']) == 132
    compared_ctas += len(before['ctas'])
    compared_tiles += sum(len(cta['tiles']) for cta in before['ctas'])

main_path = report_path.parent.parent / 'main-paired-estimands-v1/B-paired-estimands.json'
digest(main_path)
main = json.loads(main_path.read_text())
assert {key: value for key, value in report.items() if key != 'input_sha256'} == {
    key: value for key, value in main.items() if key != 'input_sha256'}
for filename in report['output_sha256']:
    assert (report_path.parent / filename).read_bytes() == (main_path.parent / filename).read_bytes()
verification = dict(
    scope='The forty archived R10 dual-clock processes only; not whole-parser equivalence.',
    specialized_replay_AST_equal=True, inactive_sparse_branch_versions=excluded_versions,
    warmup_remainder_AST_equal=True, plain_prefix_added_check='No protocol and no undeclared prefix fields in every actual process.',
    release_added_checks='No declaration, no release trace version, no enabled observed flag in every actual process.',
    reference_AST_equal=True, dependency_modules_byte_equal=True,
    complete_replay_dicts_equal=40, CTA_dicts_equal=compared_ctas, output_tile_records_equal=compared_tiles,
    complete_statistical_report_except_input_identity_equal=True, two_compressed_ledgers_byte_equal=True,
    decoder_module_byte_equal=False, qualification=False, input_sha256=inputs,
    verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(output / 'B-decoder-review.json').write_text(json.dumps(verification, indent=2) + '\n')
print(json.dumps({key: value for key, value in verification.items() if key != 'input_sha256'}, indent=2))
