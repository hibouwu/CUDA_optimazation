import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from acquisition_identity import verify_acquisition


class AcquisitionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        (self.run / 'returned/raw').mkdir(parents=True)
        self.manifest = b'{"samples": ["first", "second"]}'
        self.records = [dict(id=name, raw='raw/' + name, sha256=hashlib.sha256(data).hexdigest())
                        for name, data in [('first', b'first trace'), ('second', b'second trace')]]
        self.files = {'manifest.json': self.manifest,
                      'records.json': json.dumps(self.records).encode(),
                      'complete.json': json.dumps(dict(
                          manifest_sha256=hashlib.sha256(self.manifest).hexdigest(), records=2)).encode(),
                      'raw/first': b'first trace', 'raw/second': b'second trace'}
        for name, data in self.files.items():
            path = self.run / name if name == 'manifest.json' else self.run / 'returned' / name
            path.write_bytes(data)
        self.write_archive()
        self.pinned = hashlib.sha256((self.run / 'output.tar.gz').read_bytes()).hexdigest()

    def write_archive(self):
        with tarfile.open(self.run / 'output.tar.gz', 'w:gz') as bundle:
            for name, data in self.files.items():
                info = tarfile.TarInfo('./' + name)
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))

    def test_original_mapping_accepted(self):
        _, records, hashes = verify_acquisition(self.run, self.pinned)
        self.assertEqual(records, self.records)
        self.assertEqual(hashes['output.tar.gz'], self.pinned)

    def test_swapped_paths_and_self_reported_hashes_rejected(self):
        first, second = self.records
        for field in ('raw', 'sha256'):
            first[field], second[field] = second[field], first[field]
        (self.run / 'returned/records.json').write_text(json.dumps(self.records))
        with self.assertRaisesRegex(ValueError, 'pinned acquisition member: records.json'):
            verify_acquisition(self.run, self.pinned)

    def test_swapped_bytes_with_original_names_and_updated_hashes_rejected(self):
        (self.run / 'returned/raw/first').write_bytes(b'second trace')
        (self.run / 'returned/raw/second').write_bytes(b'first trace')
        for record in self.records:
            record['sha256'] = hashlib.sha256((self.run / 'returned' / record['raw']).read_bytes()).hexdigest()
        (self.run / 'returned/records.json').write_text(json.dumps(self.records))
        with self.assertRaisesRegex(ValueError, 'pinned acquisition member: records.json'):
            verify_acquisition(self.run, self.pinned)

    def test_raw_only_mutation_rejected(self):
        (self.run / 'returned/raw/first').write_bytes(b'second trace')
        with self.assertRaisesRegex(ValueError, 'pinned acquisition member: raw/first'):
            verify_acquisition(self.run, self.pinned)

    def test_rebuilt_archive_cannot_replace_pinned_acquisition(self):
        self.files['raw/first'] = b'changed trace'
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'caller-pinned identity'):
            verify_acquisition(self.run, self.pinned)

    def test_local_manifest_and_completion_mutations_rejected(self):
        for name in ('manifest.json', 'complete.json'):
            path = self.run / name if name == 'manifest.json' else self.run / 'returned' / name
            original = path.read_bytes()
            path.write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'pinned acquisition member: ' + name):
                verify_acquisition(self.run, self.pinned)
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
