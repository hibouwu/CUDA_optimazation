#!/usr/bin/env python3
"""GH200 v2 suite: immutable run dependencies and independent phase gates."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
from pathlib import Path
import subprocess
from common.suite_io import read_json, relative, require, sha, verify_files
from common.suite_snapshot import CAMPAIGN, check_snapshot, required_reviews
from auditors.suite import adapter, audit, load_run, report

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--contract', type=Path, default=HERE / 'contracts/memory_baseline.json')
    run = sub.add_parser('run')
    run.add_argument('--suite', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--contract', type=Path, default=HERE / 'contracts/memory_baseline.json')
    run.add_argument('--preflight', action='store_true')
    run.add_argument('--resolved-cases', type=Path, help='B-bound calibration origin resolved_cases.json; reuses its exact binary')
    resume = sub.add_parser('resume')
    resume.add_argument('run_dir', type=Path)
    resume.add_argument('--preflight', action='store_true')
    for name in ('status', 'audit', 'report'):
        command = sub.add_parser(name)
        command.add_argument('run_dir', type=Path)
        if name == 'audit':
            command.add_argument('--require-complete', action='store_true')
            command.add_argument('--legacy-v1', action='store_true')
    finish = sub.add_parser('finalize')
    finish.add_argument('run_dir', type=Path)
    finish.add_argument('--review', type=Path, required=True)
    return ap


def frozen_dispatch(root, argv):
    root = Path(root).resolve()
    spec = read_json(relative(root, 'run_spec.json'))
    if spec.get('schema_version') != 2:
        return
    script = relative(root, 'snapshot/repo/' + CAMPAIGN + '/run_suite.py')
    # Dispatch verifies immutable files and declared gates first. The archive's
    # own runner enforces its revision; current policy must not rewrite old runs.
    check_snapshot(root, spec, enforce_current_policy=script == Path(__file__).resolve())
    if script != Path(__file__).resolve():
        os.execv(sys.executable, [sys.executable, '-B', str(script), *argv])


def main(argv=None):
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(raw_argv)
    if args.command == 'plan':
        contract = read_json(args.contract)
        adapter(contract).validate_contract(contract)
        print(json.dumps({'schema_version': 2, 'family': contract['family'], 'cases': len(contract['cases']), 'case_ids': [c['id'] for c in contract['cases']], 'adapter': contract['adapter_id'], 'formal_required_reviews': required_reviews(contract), 'preflight_required_reviews': required_reviews(contract, True), 'GPU_execution': False}, indent=2))
        return 0
    if args.command == 'run':
        from runners.suite_runner import initialize
        script = initialize(REPO, args.suite, args.output, args.contract, preflight=args.preflight, resolved_cases=args.resolved_cases)
        resume_args = ['resume', str(args.output.resolve())] + (['--preflight'] if args.preflight else [])
        os.execv(sys.executable, [sys.executable, '-B', str(script), *resume_args])
    root = args.run_dir.resolve()
    if args.command == 'audit' and args.legacy_v1:
        spec = read_json(relative(root, 'run_spec.json'))
        require(spec.get('schema_version') == 1, '--legacy-v1 requires explicit schema_version=1')
        legacy = relative(root, 'source/audit_campaign.py')
        require(sha(legacy) == spec['source_sha256']['audit_campaign.py'], 'legacy frozen auditor hash')
        return subprocess.run([sys.executable, '-B', str(legacy), str(root), '--require-complete'], env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, check=False).returncode
    frozen_dispatch(root, raw_argv)
    if args.command == 'resume':
        from runners.suite_runner import execute
        return execute(root, root.parent.parent, preflight=args.preflight)
    if args.command == 'status':
        spec = read_json(root / 'run_spec.json')
        value = {'kind': spec['kind'], 'state': read_json(root / 'campaign_status.json'), 'snapshot_integrity': 'pass', 'completion_present': (root / 'COMPLETE').exists()}
        if (root / 'COMPLETE').exists():
            audit(root, complete=True)
            value['completion_integrity'] = 'pass'
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    if args.command in ('audit', 'report'):
        spec = read_json(root / 'run_spec.json')
        if spec['kind'] == 'preflight':
            require(args.command == 'audit', 'preflight is diagnostics, not a formal report')
            from runners.suite_runner import audit_preflight
            value = audit_preflight(root)
        else:
            value = audit(root, complete=args.command == 'audit' and args.require_complete)
        print(report(value) if args.command == 'report' else json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    from runners.suite_runner import finalize
    print(json.dumps(finalize(root, root.parent.parent, args.review), indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        print('SUITE_REJECTED: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
