#!/usr/bin/env bash
# Validate the library plus the explicit delivery contracts, including axiom closure.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="${HOME}/.elan/bin:${PATH}"
# Freeze the delivery contract, not implementation proof bodies. A deliberate
# contract revision must update this baseline explicitly and undergo semantic review.
python3 - <<'BASELINE'
import hashlib
import pathlib
import subprocess

expected = "2bd15febd6c044fb56accf5b8a5ce6bd0ddd047504d74645a34fc925bf7e199e"
contract = pathlib.Path('Checks/Axioms.lean')
def verify(data, source):
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f'FAIL: delivery contract baseline changed ({source}); review the specification change')
if not contract.is_file():
    raise SystemExit('FAIL: delivery contract is missing')
verify(contract.read_bytes(), 'working tree')
# Inspect the index only when this path has staged changes. Standalone source
# copies without Git still validate their actual working-tree contents.
root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
if root.returncode == 0:
    path = contract.resolve().relative_to(pathlib.Path(root.stdout.strip())).as_posix()
    staged = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', path], cwd=root.stdout.strip())
    if staged.returncode == 1:
        blob = subprocess.run(['git', 'show', ':' + path], cwd=root.stdout.strip(), capture_output=True)
        if blob.returncode != 0:
            raise SystemExit('FAIL: staged delivery contract is missing')
        verify(blob.stdout, 'index')
    elif staged.returncode != 0:
        raise SystemExit('FAIL: cannot inspect staged delivery contract')
BASELINE
lake build GemmProofs
check_output=$(mktemp)
trap 'rm -f "$check_output"' EXIT
lake env lean -DwarningAsError=true Checks/Axioms.lean | tee "$check_output"
python3 - "$check_output" <<'PY'
import pathlib
import re
import sys

contracts = pathlib.Path('Checks/Axioms.lean').read_text()
expected = re.findall(r'^#print axioms ([A-Za-z0-9_.]+)\s*$', contracts, re.M)
if not expected or len(set(expected)) != len(expected):
    raise SystemExit('FAIL: empty or duplicate axiom audit inventory')
output = pathlib.Path(sys.argv[1]).read_text()
reported = {}
for name, axioms in re.findall(r"'([^']+)' depends on axioms:\s*\[([^\]]*)\]", output):
    if name in reported:
        raise SystemExit(f'FAIL: duplicate audit output for {name}')
    reported[name] = {x.strip() for x in axioms.split(',') if x.strip()}
for name in re.findall(r"'([^']+)' does not depend on any axioms", output):
    if name in reported:
        raise SystemExit(f'FAIL: duplicate audit output for {name}')
    reported[name] = set()
if set(reported) != set(expected):
    raise SystemExit(f'FAIL: audit inventory mismatch: expected {expected}, got {list(reported)}')
allowed = {'propext', 'Classical.choice', 'Quot.sound'}
for name, axioms in reported.items():
    if extra := axioms - allowed:
        raise SystemExit(f'FAIL: {name} uses forbidden axioms {sorted(extra)}')
print(f'PASS: build, declared type contracts, and {len(expected)} axiom audits. '
      'Stage completion still requires the PLAN.md review.')
PY
