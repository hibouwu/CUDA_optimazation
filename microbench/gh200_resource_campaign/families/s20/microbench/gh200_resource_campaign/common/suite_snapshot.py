"""Immutable project dependency snapshots; external toolchain files are inventoried."""
from __future__ import annotations
import ast
from pathlib import Path
import re
import shutil
from common.suite_io import atomic_json, gate_mapping, read_json, relative, require, sha, validate_gate, verify_files

CAMPAIGN = 'microbench/gh200_resource_campaign'
DOCS = 'Docs/ModelEvaluation/gemm/experiments/gh200_sm90'
CORE_B_REVIEWS = [('S02', 'B', 'reviews/S02-formal-b3-v2-B-review.json'),
                  ('S03', 'B', 'reviews/S03-formal-b3-v2-B-review.json')]


def framework_files(repo):
    root = Path(repo) / CAMPAIGN
    names = {CAMPAIGN + '/run_suite.py', CAMPAIGN + '/launch_romeo.sh'}
    for folder in ('common', 'runners', 'auditors', 'contracts', 'tests', 'probes'):
        for p in (root / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.cuh', '.h', '.hpp', '.cu', '.md', '.sh'):
                names.add(p.relative_to(repo).as_posix())
    for p in (Path(repo) / DOCS).glob('*.md'):
        names.add(p.relative_to(repo).as_posix())
    return names


def local_cpp_closure(repo, names):
    pending = list(names)
    while pending:
        name = pending.pop()
        p = relative(repo, name)
        if p.suffix not in ('.cu', '.cuh', '.h', '.hpp', '.cpp'):
            continue
        for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', p.read_text(), re.M):
            target = (p.parent / include).resolve()
            require(target.is_relative_to(Path(repo).resolve()), 'local include outside repository')
            dep = target.relative_to(Path(repo).resolve()).as_posix()
            relative(repo, dep)
            if dep not in names:
                names.add(dep)
                pending.append(dep)
    return names


def python_closure_check(repo, names):
    """Every campaign-local import must already belong to the copied framework tree."""
    local_packages = ('common', 'runners', 'auditors')
    for name in list(names):
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse(relative(repo, name).read_text(), filename=name)):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module + '.' + alias.name for alias in node.names] if node.module in local_packages else [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for mod in modules:
                if mod.split('.')[0] in local_packages:
                    dep = CAMPAIGN + '/' + mod.replace('.', '/') + '.py'
                    require(dep in names, 'unfrozen Python import: ' + dep)


def required_reviews(contract, preflight=False, *, case_diagnostic=False):
    from common.family_b3 import enabled, policy
    b3 = enabled(contract)
    selected = policy(contract) if b3 else None
    require(not case_diagnostic or (preflight and not b3), 'case diagnostic requires a non-formal contract')
    result = [('S02', 'A', 'reviews/S02-A-review.json'), ('S02', 'preflight-A', 'reviews/S02-preflight-A-review-r2.json')]
    if any(c.get('iteration_policy', {}).get('kind') == 'calibrated' for c in contract['cases']):
        result.append(('S02', 'calibration-A', 'reviews/S02-calibration-A-review-r2.json'))
    if 'S01' in contract.get('foundation_gates', []):
        result.append(('S01', 'review', 'reviews/S01-review.json'))
    if not case_diagnostic:
        result += ([('S02', 'B', 'reviews/S02-formal-b3-s14-B-review.json'),
                    ('S03', 'B', 'reviews/S03-formal-b3-s14-B-review.json')]
                   if b3 and selected['stage'] == 'S14' else CORE_B_REVIEWS)
    if b3:
        if selected['stage'] == 'S12':
            result += [('S02', 'formal-b3-A', 'reviews/S02-formal-b3-A-review.json')]
        review_path = Path(contract['family_b3']['review']['path'])
        require(review_path.parent.name == 'reviews', 'B3 review must be in suite reviews')
        result.append((selected['stage'], selected.get('b3_review_phase', 'early-validation-B3'), 'reviews/' + review_path.name))
        if selected['stage'] == 'S14':
            original = Path(contract['family_b3']['original_transport']['path'])
            require(original.parent.name == 'reviews', 'original S14 bridge must be in suite reviews')
            result.append(('S14', 'packed-evidence-bridge-B', 'reviews/' + original.name))
    for dep in contract['review_dependencies']:
        if dep['phase'] in ('A', 'review') or dep['phase'].endswith('-A') or (dep['phase'] == 'B' and not preflight) or (b3 and dep['phase'] == 'formal-b3-source-B'):
            path = dep['path']
            # New S12 runs need an explicit current-core compatibility bridge.
            # The old contract/reviews and old frozen dispatch stay unchanged.
            if b3 and selected['stage'] == 'S12':
                bridges = {
                    ('S12','formal-b3-source-B','reviews/S12-formal-b3-source-B-review.json'):
                        'reviews/S12-formal-b3-source-B-review-r2.json',
                    ('S12','B','reviews/S12-B-review.json'):'reviews/S12-B-review-r2.json'}
                path = bridges.get((dep['stage'],dep['phase'],path),path)
            result.append((dep['stage'], dep['phase'], path))
    if b3:
        require(any(s == selected['stage'] and p == 'formal-b3-source-B' for s, p, _ in result), 'family integration source B required')
    require(any(s == contract['stage'] and p == 'A' for s, p, _ in result), 'family A dependency required')
    if not preflight:
        require(any(s == contract['stage'] and p == 'B' for s, p, _ in result), 'family B dependency required')
    return list(dict.fromkeys(result))


def freeze(repo, suite, run, contract_path, *, preflight=False, case_diagnostic=False):
    repo, suite, run = Path(repo).resolve(), Path(suite).resolve(), Path(run).resolve()
    contract_name = Path(contract_path).resolve().relative_to(repo).as_posix()
    contract = read_json(relative(repo, contract_name))
    names = framework_files(repo) | {contract_name, contract['source']} | set(contract['dependencies'])
    reviews = []
    for stage, phase, name in required_reviews(contract, preflight, case_diagnostic=case_diagnostic):
        path = relative(suite, name)
        review = validate_gate(path, repo, stage, phase)
        names.update(gate_mapping(review))
        reviews.append({'stage': stage, 'phase': phase, 'path': name, 'sha256': sha(path)})
    from common.family_b3 import validate_bundle
    bundle = validate_bundle(repo, contract)
    if bundle is not None:
        names.update(bundle['names'])
    names = local_cpp_closure(repo, names)
    python_closure_check(repo, names)
    snapshot = run / 'snapshot'
    snapshot.mkdir()
    mapping = {}
    for name in sorted(names):
        source = relative(repo, name)
        dest = snapshot / 'repo' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        mapping['repo/' + name] = sha(dest)
    for review in reviews:
        dest = snapshot / 'suite' / review['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative(suite, review['path']), dest)
        mapping['suite/' + review['path']] = sha(dest)
    atomic_json(snapshot / 'manifest.json', mapping)
    return {'snapshot_manifest_sha256': sha(snapshot / 'manifest.json'), 'contract_path': 'snapshot/repo/' + contract_name,
            'reviews': reviews, 'source_path': 'snapshot/repo/' + contract['source']}


def check_snapshot(run, spec, *, enforce_current_policy=True):
    run = Path(run)
    require(sha(relative(run, 'snapshot/manifest.json')) == spec['snapshot_manifest_sha256'], 'snapshot manifest changed')
    mapping = read_json(run / 'snapshot/manifest.json')
    verify_files(run / 'snapshot', mapping)
    actual = {p.relative_to(run / 'snapshot').as_posix() for p in (run / 'snapshot').rglob('*') if p.is_file() and p.relative_to(run / 'snapshot').as_posix() != 'manifest.json'}
    require(actual == set(mapping), 'snapshot dependency set changed')
    for review in spec['reviews']:
        path = relative(run / 'snapshot/suite', review['path'])
        require(sha(path) == review['sha256'], 'review snapshot changed')
        validate_gate(path, run / 'snapshot/repo', review['stage'], review['phase'])
    contract = read_json(relative(run, spec['contract_path']))
    if enforce_current_policy and spec.get('kind') in ('formal', 'preflight'):
        expected = set(required_reviews(contract, preflight=spec['kind'] == 'preflight'))
        observed = [(r['stage'], r['phase'], r['path']) for r in spec['reviews']]
        require(len(observed) == len(expected) and set(observed) == expected, 'run review dependency set changed')
        from common.family_b3 import validate_bundle
        validate_bundle(run / 'snapshot/repo', contract)
    return contract


def inventory_depfile(run, depfile, *, toolchain_roots=()):
    """nvcc -MD includes system headers; local paths must resolve in snapshot."""
    import shlex
    run = Path(run).resolve()
    snapshot = run / 'snapshot/repo'
    text = Path(depfile).read_text().replace('\\\n', ' ')
    require(': ' in text or ':\n' in text, 'compiler dependency file syntax')
    text = text.split(':', 1)[1]
    deps = shlex.split(text)
    require(deps, 'empty compiler dependencies')
    frozen = read_json(run / 'snapshot/manifest.json')
    external = {}
    local = []
    # Resolve installation aliases, but never allow the shared filesystem wholesale.
    trusted_roots = {Path(prefix).resolve() for prefix in ('/usr', '/lib', '/lib64')}
    for prefix in toolchain_roots:
        resolved = Path(prefix).resolve()
        require(resolved.is_dir() and resolved not in (Path('/'), Path('/gpfs'), Path('/home'), Path('/tmp'), Path('/shared')), 'overbroad toolchain root')
        trusted_roots.add(resolved)
    for name in deps:
        path = Path(name)
        if not path.is_absolute():
            path = snapshot / path
        path = path.resolve()
        require(path.is_file(), 'missing compiler dependency ' + str(path))
        if path.is_relative_to(snapshot):
            key = 'repo/' + path.relative_to(snapshot).as_posix()
            require(key in frozen and sha(path) == frozen[key], 'unfrozen compiler local dependency: ' + key)
            local.append(key)
        else:
            # Toolchain includes can only be system/CUDA installation roots, never another checkout.
            require(any(path.is_relative_to(prefix) for prefix in trusted_roots), 'non-toolchain dependency outside snapshot: ' + str(path))
            external[str(path)] = sha(path)
    atomic_json(run / 'build/dependencies.json', {'local': sorted(set(local)), 'external_toolchain': external,
                'trusted_toolchain_roots': sorted(str(p) for p in trusted_roots), 'external_files_copied': False, 'offline_rebuild_claim': False})
    return external
