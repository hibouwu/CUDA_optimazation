"""Invented CPU byte packs; no numerical/GPU or canonical archive mutation."""
from pathlib import Path
import copy
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

from common.suite_io import atomic_json, sha
from runners import s15_verify_v2 as verify

CODE = Path(__file__).resolve().parents[1]


def cpu_pack(repo):
    folder = repo / verify.PACK_NAMESPACE / 'CPU-fixture'
    folder.mkdir(parents=True)
    run = verify.RUN_NAMESPACE + '/CPU-invented-byte-fixture'
    source = repo / run
    source.mkdir(parents=True)
    (source / 'payload.bin').write_bytes(bytes(range(32)))
    atomic_json(source / 'validation_manifest.json', {'payload.bin': sha(source / 'payload.bin')})
    members = {run + '/' + name: {'bytes': (source / name).stat().st_size,
                                 'sha256': sha(source / name)}
               for name in ('payload.bin', 'validation_manifest.json')}
    archive = folder / 'run.tar.xz'
    with tarfile.open(archive, 'w:xz') as stream:
        for name in members:
            stream.add(repo / name, arcname=name, recursive=False)
    manifest_sha = sha(source / 'validation_manifest.json')
    atomic_json(folder / 'index.json', {
        'schema_version': 1, 'namespace': 'repository_relative_v1',
        'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
        'members': members, 'roots': {'original_manifest': run + '/validation_manifest.json'}})
    atomic_json(folder / 'closure.json', {
        'schema_version': 1, 'run_path': run, 'manifest_sha256': manifest_sha, 'members': members})
    entry = {'run_path': run, 'manifest_sha256': manifest_sha, 'CPU_fixture': True}
    for key, path in [('archive', archive), ('index', folder / 'index.json'),
                      ('closure', folder / 'closure.json')]:
        entry[key] = {'path': str(path.relative_to(repo)), 'sha256': sha(path)}
    atomic_json(folder / 'entry.json', entry)
    return entry, folder


class VerifyV2Tests(unittest.TestCase):
    def test_valid_complete_CPU_pack_compatible_receipt_and_origin(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td); entry, folder = cpu_pack(repo)
            before = {p.name: sha(p) for p in folder.iterdir()}
            output = folder / 'verified.json'
            result = verify.verify_pack(repo, entry, 'offhost', output)
            self.assertEqual(result['member_count'], 2)
            self.assertEqual(result['uncompressed_bytes'], sum(x['bytes'] for x in result['members'].values()))
            self.assertEqual(result['verification'], 'complete_decode_all_members_sha256')
            origin = json.loads(output.with_suffix('.origin.json').read_text())
            self.assertEqual(origin['receipt_sha256'], sha(output))
            self.assertEqual(origin['archive_sha256'], entry['archive']['sha256'])
            self.assertEqual({name: sha(folder / name) for name in before}, before)
            self.assertFalse(list(folder.glob('.verify-v2-*')))

    def test_closure_SHA_drift_rejected_before_decode_without_publication(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td); entry, folder = cpu_pack(repo)
            closure = folder / 'closure.json'
            value = json.loads(closure.read_text()); value['run_path'] = 'unbound-run'
            atomic_json(closure, value)
            with patch.object(verify.PackedEvidence, 'verify_all') as decoder:
                with self.assertRaises(ValueError): verify.verify_pack(repo, entry, 'remote', folder / 'new.json')
                decoder.assert_not_called()
            self.assertFalse((folder / 'new.json').exists())
            self.assertFalse((folder / 'new.origin.json').exists())

    def test_selfconsistent_wrong_run_manifest_and_member_closure_rejected(self):
        for field in ('run_path', 'manifest_sha256'):
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td); entry, folder = cpu_pack(repo)
                closure = folder / 'closure.json'; value = json.loads(closure.read_text())
                value[field] = 'wrong-run' if field == 'run_path' else '0' * 64
                atomic_json(closure, value); entry['closure']['sha256'] = sha(closure)
                with self.assertRaises(ValueError): verify.verify_pack(repo, entry, 'remote', folder / 'new.json')
                self.assertFalse((folder / 'new.json').exists())

    def test_outside_parent_symlink_wrong_namespace_and_input_collisions_preIO(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td); repo = parent / 'repo'; repo.mkdir(); entry, folder = cpu_pack(repo)
            outside = parent / 'outside'; outside.mkdir(); (folder / 'link').symlink_to(outside, target_is_directory=True)
            before = {name: sha(repo / entry[name]['path']) for name in ('archive', 'index', 'closure')}
            outputs = [outside / 'x.json', repo / '..' / 'outside' / 'parent.json',
                       folder / 'link' / 'x.json', repo / 'wrong-family' / 'x.json',
                       folder / 'entry.json'] + [repo / entry[name]['path'] for name in before]
            with patch.object(verify.PackedEvidence, 'verify_all') as decoder:
                for output in outputs:
                    with self.assertRaises(ValueError): verify.verify_pack(repo, entry, 'remote', output)
                decoder.assert_not_called()
            self.assertEqual(before, {name: sha(repo / entry[name]['path']) for name in before})
            self.assertFalse(list(outside.iterdir()))

    def test_existing_receipt_or_origin_and_special_output_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td); entry, folder = cpu_pack(repo)
            receipt = folder / 'existing.json'; receipt.write_bytes(b'old receipt')
            origin_only = folder / 'only.json'; origin_only.with_suffix('.origin.json').write_bytes(b'old origin')
            fifo = folder / 'fifo.json'; os.mkfifo(fifo)
            for output in (receipt, origin_only, fifo):
                with self.assertRaises(ValueError): verify.verify_pack(repo, entry, 'remote', output)
            self.assertEqual(receipt.read_bytes(), b'old receipt')
            self.assertEqual(origin_only.with_suffix('.origin.json').read_bytes(), b'old origin')
            self.assertFalse(origin_only.exists())

    def test_input_namespace_symlink_and_hash_change_refused(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td); entry, folder = cpu_pack(repo)
            for key in ('archive', 'index', 'closure'):
                changed = copy.deepcopy(entry); changed[key]['sha256'] = '0' * 64
                with self.assertRaises(ValueError): verify.verify_pack(repo, changed, 'remote', folder / 'new.json')
            link = folder / 'input-link.json'; link.symlink_to(folder / 'index.json')
            changed = copy.deepcopy(entry); changed['index']['path'] = str(link.relative_to(repo))
            with self.assertRaises(ValueError): verify.verify_pack(repo, changed, 'remote', folder / 'new.json')
            self.assertFalse((folder / 'new.json').exists())

    def test_input_drift_during_decode_cannot_publish_success(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td); entry, folder = cpu_pack(repo); actual = verify.PackedEvidence.verify_all
            def change_after_decode(bundle):
                result = actual(bundle)
                with (folder / 'closure.json').open('ab') as stream: stream.write(b'\n')
                return result
            with patch.object(verify.PackedEvidence, 'verify_all', new=change_after_decode):
                with self.assertRaises(ValueError): verify.verify_pack(repo, entry, 'remote', folder / 'new.json')
            self.assertFalse((folder / 'new.json').exists())

    def test_corrupt_archive_ref_cannot_publish_success(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td); entry, folder = cpu_pack(repo)
            with (folder / 'run.tar.xz').open('r+b') as stream: stream.write(b'broken archive')
            with self.assertRaises(ValueError): verify.verify_pack(repo, entry, 'remote', folder / 'new.json')
            self.assertFalse((folder / 'new.json').exists())

    def test_two_concurrent_publishers_preserve_one_complete_winner(self):
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / 'new.json'; barrier = threading.Barrier(2); results = []
            def publish(number):
                barrier.wait()
                try: verify.publish_new_json(destination, {'writer': number}); results.append(('published', number))
                except FileExistsError: results.append(('refused', number))
            workers = [threading.Thread(target=publish, args=(n,)) for n in (1, 2)]
            for worker in workers: worker.start()
            for worker in workers: worker.join()
            self.assertEqual(sorted(x[0] for x in results), ['published', 'refused'])
            self.assertIn(('published', json.loads(destination.read_text())['writer']), results)
            self.assertFalse(list(Path(td).glob('.verify-v2-*')))

    def test_origin_publication_race_keeps_evidence_and_refuses_completion(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td); entry, folder = cpu_pack(repo); output = folder / 'new.json'
            origin = output.with_suffix('.origin.json'); actual = verify.publish_new_json
            archive_sha = sha(folder / 'run.tar.xz')
            def race(path, value):
                if path == output: origin.write_bytes(b'competing origin')
                actual(path, value)
            with patch.object(verify, 'publish_new_json', side_effect=race):
                with self.assertRaises(FileExistsError): verify.verify_pack(repo, entry, 'remote', output)
            self.assertEqual(origin.read_bytes(), b'competing origin')
            self.assertEqual(sha(folder / 'run.tar.xz'), archive_sha)
            self.assertTrue(output.is_file())  # retained incomplete pair, never overwritten/retried

    def test_CLI_requires_own_B_before_any_verification(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td); entry, folder = cpu_pack(repo)
            argv = [sys.executable, '-B', str(CODE / 'runners/s15_verify_v2.py'), 'verify',
                    '--repo', str(repo), '--entry', str(folder / 'entry.json'),
                    '--location', 'remote', '--output', str(folder / 'new.json')]
            environment = dict(os.environ, PYTHONPATH=str(CODE))
            result = subprocess.run(argv, capture_output=True, env=environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'S15-verify-v2-source-B-review.json', result.stderr)
            self.assertFalse((folder / 'new.json').exists())


if __name__ == '__main__':
    unittest.main()
