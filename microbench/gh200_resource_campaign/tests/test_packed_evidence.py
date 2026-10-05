"""Adversarial evidence-pack tests; all fixtures are synthetic CPU files."""
import copy
import hashlib
import io
import json
import lzma
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from common.packed_evidence import PackedEvidence, SMALL_OBJECT_LIMIT, json_object, logical_path
from common.suite_io import sha


class PackedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, rows, *, closure=None, index_change=None):
        archive, index = self.root / 'evidence.tar.xz', self.root / 'index.json'
        members = {}
        with tarfile.open(archive, 'w:xz') as stream:
            for name, data, kind in rows:
                info = tarfile.TarInfo(name)
                if kind == 'symlink':
                    info.type = tarfile.SYMTYPE
                    info.linkname = '/etc/passwd'
                    stream.addfile(info)
                else:
                    info.size = len(data)
                    stream.addfile(info, io.BytesIO(data))
                members[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        value = {'schema_version': 1, 'namespace': 'repository_relative_v1',
                 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
                 'members': members, 'roots': {'review': rows[0][0]}}
        expected = copy.deepcopy(members) if closure is None else closure
        if index_change:
            index_change(value)
        index.write_text(json.dumps(value))
        return PackedEvidence(archive, index, expected_index_sha256=sha(index),
                              expected_archive_sha256=value['archive_sha256'],
                              expected_closure=expected)

    def test_complete_stream_read_and_relocation(self):
        bundle = self.fixture([('reviews/one.json', b'{"answer":42}', 'file'),
                               ('results/words.bin', bytes(range(256)) * 13, 'file')])
        self.assertEqual(bundle.verify_all()['members'], 2)
        self.assertEqual(bundle.read_json('reviews/one.json'), {'answer': 42})
        self.assertEqual(b''.join(bundle.iter_bytes('results/words.bin', 3)), bytes(range(256)) * 13)
        moved = self.root / 'moved'
        moved.mkdir()
        bundle.archive.rename(moved / bundle.archive.name)
        bundle.index_path.rename(moved / bundle.index_path.name)
        relocated = PackedEvidence(moved / 'evidence.tar.xz', moved / 'index.json',
                                   expected_index_sha256=bundle.index_sha256,
                                   expected_archive_sha256=bundle.archive_sha256,
                                   expected_closure=bundle.members)
        relocated.verify_all()
        self.assertEqual(relocated.read_json('reviews/one.json'), {'answer': 42})

    def test_preverification_unknown_and_chunk_bounds(self):
        bundle = self.fixture([('review.json', b'{}', 'file')])
        with self.assertRaises(ValueError):
            bundle.read_bytes('review.json')
        bundle.verify_all()
        for name in ('missing', '../review.json', '/review.json'):
            with self.assertRaises(ValueError):
                bundle.read_bytes(name)
        for chunk in (0, -1, True, 1048577):
            with self.assertRaises(ValueError):
                list(bundle.iter_bytes('review.json', chunk))

    def test_duplicate_and_link_members(self):
        for rows in ([('review.json', b'{}', 'file'), ('review.json', b'{}', 'file')],
                     [('review.json', b'', 'symlink')]):
            bundle = self.fixture(rows)
            with self.assertRaises(ValueError):
                bundle.verify_all()

    def test_independent_closure_not_self_reported(self):
        with self.assertRaises(ValueError):
            self.fixture([('review.json', b'{}', 'file')],
                         closure={'different': {'bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}})
        with self.assertRaises(ValueError):
            self.fixture([('review.json', b'{}', 'file')],
                         index_change=lambda value: value['members']['review.json'].update(bytes=True))

    def test_content_corruption_even_after_rehashing_outer_pack(self):
        bundle = self.fixture([('review.json', b'{}', 'file')])
        with tarfile.open(bundle.archive, 'w:xz') as stream:
            info = tarfile.TarInfo('review.json')
            info.size = 2
            stream.addfile(info, io.BytesIO(b'[]'))
        value = json.loads(bundle.index_path.read_text())
        value['archive_sha256'] = sha(bundle.archive)
        value['archive_bytes'] = bundle.archive.stat().st_size
        bundle.index_path.write_text(json.dumps(value))
        altered = PackedEvidence(bundle.archive, bundle.index_path,
                                 expected_index_sha256=sha(bundle.index_path),
                                 expected_archive_sha256=sha(bundle.archive),
                                 expected_closure=bundle.members)
        with self.assertRaises(ValueError):
            altered.verify_all()

    def test_unsafe_paths_and_json_keys(self):
        for name in ('../a', '/a', 'a/../b', 'a\\b', '.', 'a//b', './a', 'a/./b'):
            with self.assertRaises(ValueError):
                logical_path(name)
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                json_object(data)

    def test_small_object_bound_and_drift(self):
        bundle = self.fixture([('review.json', b'{}', 'file'),
                               ('large.bin', b'x' * (SMALL_OBJECT_LIMIT + 1), 'file')])
        bundle.verify_all()
        with self.assertRaises(ValueError):
            bundle.read_bytes('large.bin')
        bundle.archive.write_bytes(bundle.archive.read_bytes() + b'changed')
        with self.assertRaises(ValueError):
            bundle.read_bytes('review.json')
        with self.assertRaises(ValueError):
            bundle.file_sha256('review.json')
        with self.assertRaises(ValueError):
            bundle.inventory()

    def test_hidden_member_after_first_tar_end(self):
        bundle = self.fixture([('review.json', b'{}', 'file')])
        expanded = lzma.decompress(bundle.archive.read_bytes())
        extra = tarfile.TarInfo('hidden/extra.bin')
        extra.size = 1
        # Keep the initial two zero blocks, then add a second valid tar member.
        expanded = expanded[:2048] + extra.tobuf() + b'x' + b'\0' * 511 + b'\0' * 1024
        bundle.archive.write_bytes(lzma.compress(expanded))
        value = json.loads(bundle.index_path.read_text())
        value['archive_sha256'] = sha(bundle.archive)
        value['archive_bytes'] = bundle.archive.stat().st_size
        bundle.index_path.write_text(json.dumps(value))
        altered = PackedEvidence(bundle.archive, bundle.index_path,
                                 expected_index_sha256=sha(bundle.index_path),
                                 expected_archive_sha256=sha(bundle.archive),
                                 expected_closure=bundle.members)
        with self.assertRaises(ValueError):
            altered.verify_all()

    def test_compression_footer_damage_is_not_hidden_by_tar_end(self):
        bundle = self.fixture([('review.json', b'{}', 'file')])
        content = bytearray(bundle.archive.read_bytes())
        content[-10] ^= 1
        bundle.archive.write_bytes(content)
        value = json.loads(bundle.index_path.read_text())
        value['archive_sha256'] = sha(bundle.archive)
        bundle.index_path.write_text(json.dumps(value))
        altered = PackedEvidence(bundle.archive, bundle.index_path,
                                 expected_index_sha256=sha(bundle.index_path),
                                 expected_archive_sha256=sha(bundle.archive),
                                 expected_closure=bundle.members)
        with self.assertRaises((ValueError, lzma.LZMAError)):
            altered.verify_all()

    def test_verified_small_cache_avoids_tar_rescan_and_rejects_drift(self):
        bundle = self.fixture([('review.json', b'{"value":7}', 'file')])
        receipt = bundle.verify_all(small_objects=['review.json'])
        self.assertEqual(receipt['cached_objects'], 1)
        self.assertEqual(receipt['decoder_passes'], 2)
        with patch('common.packed_evidence.tarfile.open', side_effect=AssertionError('rescan')):
            self.assertEqual(bundle.read_json('review.json'), {'value': 7})
        bundle.index_path.write_bytes(bundle.index_path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            bundle.read_json('review.json')
        self.assertFalse(bundle.verified)
        self.assertEqual(bundle._small_objects, {})

    def test_whitelist_limits_and_observer_failure_do_not_publish_cache(self):
        bundle = self.fixture([('review.json', b'{}', 'file')])
        for names in (['missing'], ['review.json', 'review.json']):
            with self.assertRaises(ValueError):
                bundle.verify_all(small_objects=names)
        with patch('common.packed_evidence.CACHE_BYTES_LIMIT', 1):
            with self.assertRaises(ValueError):
                bundle.verify_all(small_objects=['review.json'])
        def reject(name, offset, chunk):
            raise ValueError('independent full-word checker rejected data')
        with self.assertRaises(ValueError):
            bundle.verify_all(small_objects=['review.json'], observer=reject)
        self.assertFalse(bundle.verified)
        self.assertEqual(bundle._small_objects, {})

    def test_observer_contiguous_offsets_and_cross_chunk_words(self):
        expected = [17, 12345, 4294967295]
        words = b''.join(value.to_bytes(4, 'little') for value in expected)
        bundle = self.fixture([('review.json', b'{}', 'file'), ('words.bin', words, 'file')])
        actual, remaining = [], bytearray()
        position = 0
        def check(name, offset, chunk):
            nonlocal position
            if name != 'words.bin':
                return
            self.assertEqual(offset, position)
            position += len(chunk)
            remaining.extend(chunk)
            while len(remaining) >= 4:
                actual.append(int.from_bytes(remaining[:4], 'little'))
                del remaining[:4]
        with patch('common.packed_evidence.CHUNK_BYTES', 3):
            bundle.verify_all(small_objects=['review.json'], observer=check)
        self.assertEqual(actual, expected)
        self.assertEqual(position, len(words))
        self.assertEqual(remaining, b'')


if __name__ == '__main__':
    unittest.main()
