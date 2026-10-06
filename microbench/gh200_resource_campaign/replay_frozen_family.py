"""Replay an extracted GH200 family or S15 cohort with its frozen auditor and CPython 3.9 statistics.

Only CPU evidence verification runs. Acquisition source and archived statistics
remain unchanged; the receipt names the actual interpreter used for this replay.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.util
import json
import platform
import runpy
import sys
import time

STATISTICS_SHA256 = '8dd0406ee8988d42bcb41577e4e45c61bf78423d5158738ce765df96b99b3c23'
AUDITOR_SHA256 = {'S08': 'da25ebe2ba0e1526b578f8be8270f3a68dbd425bde85ba8380a5dcba81fd84a3', 'S15': ('789cf7fbd0f96ec9905f1946eb8e043031614a9d6f571fe4a231b92374343834', 'f02200b4a502b669ef61f7c8ae3daa01450efc0105ef55bcebf4fdbeb254c169'), 'S16': '50cc02c20a599f81691f3c3b5dfeaf5af781ce74a951a771f9e654410a92f5aa', 'S17': 'b2784f3c700d77d6977ab6388081f9c3d2a953650a39c0d4d40ebbadedf82ffc', 'S19': 'b4681ff4caf755005eee0bb668adad4b1e62c48ea0f4b0a6bc2c87c25fa6758a', 'S20': '5934c8df9f90abe00d9947ea293ceb6ae1d38a91259280e8d37127f88a576957'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(stage, run, statistics_source, output):
    if sys.flags.optimize:
        raise RuntimeError('Replay requires active assertions; do not use -O or -OO')
    sys.dont_write_bytecode = True
    run = run.resolve(strict=True)
    statistics_source = statistics_source.resolve(strict=True)
    protected_root = run.parent if stage == 'S15' else run
    auditor = (protected_root / 'repo' / 'audit_formal.py' if stage == 'S15'
               else run / 'repo' / ('audit_' + stage.lower() + '.py'))
    if not auditor.is_file():
        raise FileNotFoundError('Extract the complete original run including repo: ' + str(auditor))
    allowed = AUDITOR_SHA256[stage]
    allowed = (allowed,) if isinstance(allowed, str) else allowed
    if sha(auditor) not in allowed:
        raise ValueError('Frozen auditor hash differs from the accepted source version')
    if sha(statistics_source) != STATISTICS_SHA256:
        raise ValueError('Archived CPython 3.9.21 statistics source hash mismatch')
    # The frozen auditor performs its own complete manifest and source gate checks.
    spec = importlib.util.spec_from_file_location('statistics', statistics_source)
    archived_statistics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived_statistics)
    output = output.resolve()
    if output == protected_root or protected_root in output.parents:
        raise ValueError('Replay output must be outside the immutable original run')
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        'stage': stage, 'run': str(run), 'auditor_sha256': sha(auditor),
        'statistics_source_sha256': sha(statistics_source),
        'statistics_algorithm': 'unmodified archived CPython 3.9.21 module',
        'actual_interpreter': sys.version, 'actual_platform': platform.platform(),
        'original_interpreter_reproduced': False, 'GPU_executed': False,
        'qualified': False, 'independent_review_required': True,
        'started_unix': time.time(),
    }
    original_statistics = sys.modules.get('statistics')
    original_argv = sys.argv[:]
    original_sys_path = sys.path[:]
    sys.dont_write_bytecode = True
    try:
        sys.path.insert(0, str(auditor.parent))
        sys.modules['statistics'] = archived_statistics
        sys.argv = [str(auditor), '--run', str(run)]
        with (output / 'replay.json').open('x') as stdout, (output / 'stderr').open('x') as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                runpy.run_path(str(auditor), run_name='__main__')
        result = json.loads((output / 'replay.json').read_text())
        if result.get('CPU_fixture_only') is not False:
            raise ValueError('CPU fixture cannot establish real-data recomputation')
        if stage == 'S08':
            valid = (result.get('complete_values_recomputed') is True
                     and result.get('qualified_parameters') is False
                     and result.get('case_count') == 32
                     and result.get('formal_process_count') == 320)
        elif stage == 'S15':
            # S15 has four separately sealed cohorts. Its frozen auditor checks
            # complete values and exact manifests before returning this schema.
            valid = (result.get('kind') == 'offline_cohort_replay_pending_independent_C'
                     and result.get('performance_parameters_qualified') is False
                     and result.get('formal_C_review_required') is True
                     and result.get('selected_case_count', 0) > 0
                     and len(result.get('cases', {})) == result['selected_case_count']
                     and result.get('formal_sample_count', 0) > 0)
        else:
            valid = result.get('all_saved_values_recomputed') is True
        if not valid:
            raise ValueError('Frozen auditor result does not match the stage contract')
        receipt['status'] = 'complete_CPU_replay_pending_independent_review'
        receipt['replay_sha256'] = sha(output / 'replay.json')
    except BaseException as error:
        receipt['status'] = 'failed'
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        receipt['stopped_unix'] = time.time()
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        sys.argv = original_argv
        sys.path[:] = original_sys_path
        if original_statistics is None:
            sys.modules.pop('statistics', None)
        else:
            sys.modules['statistics'] = original_statistics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('S08', 'S15', 'S16', 'S17', 'S19', 'S20'), required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--statistics-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.stage, args.run, args.statistics_source, args.output)
