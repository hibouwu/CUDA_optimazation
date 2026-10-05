"""S15 complete byte verification only; legacy storage/runtime stays frozen."""
from pathlib import Path
import argparse
import json
import os
import socket
import tempfile

from common.packed_evidence import PackedEvidence, logical_path
from common.suite_io import read_json, require, sha, validate_gate

SUITE = 'results/gh200_resource_campaign/20261001-resource-suite-v2'
PACK_NAMESPACE = SUITE + '/s15-packs'
RUN_NAMESPACE = SUITE + '/tma_tensor_2d'
REVIEW = SUITE + '/reviews/S15-verify-v2-source-B-review.json'


def deployment_root(repo):
    repo = Path(repo).absolute()
    require('..' not in repo.parts, 'deployment parent traversal forbidden')
    cursor = Path(repo.anchor)
    for part in repo.parts[1:]:
        cursor = cursor / part
        require(not cursor.is_symlink(), 'deployment ancestor symlink forbidden')
        require(cursor.is_dir(), 'deployment ancestor must be directory')
    return repo


def pack_file(repo, name, *, new=False):
    logical_path(name)
    relative = Path(name)
    require(relative.is_relative_to(Path(PACK_NAMESPACE)) and
            relative != Path(PACK_NAMESPACE), 'fixed S15 pack namespace required')
    destination = repo / relative
    cursor = repo
    for part in relative.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), 'pack ancestor/member symlink forbidden')
        if cursor != destination:
            require(cursor.is_dir(), 'existing pack parent directory required')
    if new:
        require(not destination.exists(), 'never overwrite pack evidence')
    else:
        require(destination.is_file(), 'regular existing pack member required')
    return destination


def publish_new_json(path, value):
    """Publish complete bytes exclusively; never replace a concurrent winner."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                         prefix='.verify-v2-', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_pack(repo, entry, location, output):
    require(location in ('remote', 'offhost'), 'finite verifier location required')
    repo = deployment_root(repo)
    output = Path(output)
    output = (output if output.is_absolute() else repo / output).absolute()
    origin = output.with_suffix('.origin.json')
    require(output != origin and output.is_relative_to(repo) and
            origin.is_relative_to(repo), 'distinct in-deployment outputs required')
    output_names = [str(path.relative_to(repo)) for path in (output, origin)]
    # Both destinations must be new and safe before reading/decompressing inputs.
    for name in output_names:
        pack_file(repo, name, new=True)

    for key in ('archive', 'index', 'closure'):
        reference = entry[key]
        require(set(reference) == {'path', 'sha256'}, 'exact sealed reference required')
        path = pack_file(repo, reference['path'])
        require(sha(path) == reference['sha256'], 'sealed input identity changed: ' + key)
    index = read_json(repo / entry['index']['path'])
    closure = read_json(repo / entry['closure']['path'])
    require(set(closure) == {'schema_version', 'run_path', 'manifest_sha256', 'members'} and
            type(closure['schema_version']) is int and closure['schema_version'] == 1,
            'original closure schema required')
    run_path = logical_path(entry['run_path'])
    require(Path(run_path).is_relative_to(Path(RUN_NAMESPACE)) and
            Path(run_path) != Path(RUN_NAMESPACE), 'original S15 run namespace required')
    manifest = run_path + '/validation_manifest.json'
    require(closure['run_path'] == run_path and
            closure['manifest_sha256'] == entry['manifest_sha256'] and
            closure['members'] == index['members'] and
            index['archive_sha256'] == entry['archive']['sha256'] and
            index['roots'].get('original_manifest') == manifest and
            manifest in closure['members'] and
            closure['members'][manifest]['sha256'] == entry['manifest_sha256'] and
            all(name.startswith(run_path + '/') for name in closure['members']),
            'sealed original run/manifest/member closure required')

    bundle = PackedEvidence(repo / entry['archive']['path'], repo / entry['index']['path'],
                            expected_index_sha256=entry['index']['sha256'],
                            expected_archive_sha256=entry['archive']['sha256'],
                            expected_closure=closure['members'])
    result = bundle.verify_all()
    # Recheck the complete input and output identity set after full decoding.
    for key in ('archive', 'index', 'closure'):
        reference = entry[key]
        require(sha(pack_file(repo, reference['path'])) == reference['sha256'],
                'sealed input drift during decoding: ' + key)
    for name in output_names:
        pack_file(repo, name, new=True)
    receipt = {'schema_version': 1, 'status': 'full_members_verified', 'location': location,
               'archive_sha256': entry['archive']['sha256'],
               'index_sha256': entry['index']['sha256'],
               'closure_sha256': entry['closure']['sha256'], 'run_path': run_path,
               'manifest_sha256': entry['manifest_sha256'], 'members': closure['members'],
               'member_count': result['members'], 'uncompressed_bytes': result['uncompressed_bytes'],
               'verification': 'complete_decode_all_members_sha256'}
    publish_new_json(output, receipt)
    attestation = {'schema_version': 1, 'receipt_sha256': sha(output),
                   'archive_sha256': entry['archive']['sha256'],
                   'verifier': {'host': socket.gethostname(),
                                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                'uid': os.geteuid(),
                                'archive_path': str((repo / entry['archive']['path']).resolve())}}
    publish_new_json(origin, attestation)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify',))
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--entry', type=Path, required=True)
    parser.add_argument('--location', choices=('remote', 'offhost'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = deployment_root(args.repo)
    # No production CLI IO/publication before its own independent source B.
    validate_gate(repo / REVIEW, repo, 'S15', 'verify-v2-source-B')
    entry_path = args.entry if args.entry.is_absolute() else repo / args.entry
    entry_name = str(entry_path.absolute().relative_to(repo))
    entry = read_json(pack_file(repo, entry_name))
    verify_pack(repo, entry, args.location, args.output)


if __name__ == '__main__':
    main()
