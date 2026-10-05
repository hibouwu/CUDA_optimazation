"""Proof-boundary failures must remain failures even when callers catch errors."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from common.packed_evidence import PackedEvidence
from common.suite_io import sha


class PackedTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.number = 0

    def tearDown(self):
        self.temp.cleanup()

    def bundle(self, verified=True):
        self.number += 1
        archive = self.root / (str(self.number) + '.tar.xz')
        index = self.root / (str(self.number) + '.json')
        values = {'review.json': b'{"value":42}', 'payload.bin': b'\x01\x02\x03\x04'}
        members = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                   for name, data in values.items()}
        with tarfile.open(archive, 'w:xz') as stream:
            for name, data in values.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                stream.addfile(info, io.BytesIO(data))
        index.write_text(json.dumps({'schema_version': 1, 'namespace': 'repository_relative_v1',
                                     'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
                                     'members': members, 'roots': {'review': 'review.json'}}))
        result = PackedEvidence(archive, index, expected_index_sha256=sha(index),
                                expected_archive_sha256=sha(archive), expected_closure=members)
        if verified:
            result.verify_all(small_objects=['review.json'])
        return result

    def test_success_uses_two_identity_checks_and_invalidates_view(self):
        bundle = self.bundle()
        with patch.object(bundle, '_identity', wraps=bundle._identity) as identity:
            with bundle.verified_transaction(['review.json']) as view:
                for _ in range(20):
                    self.assertEqual(view.read_json('review.json'), {'value': 42})
                    self.assertEqual(view.file_sha256('payload.bin'), bundle.members['payload.bin']['sha256'])
                listing = view.inventory()
                listing['payload.bin']['bytes'] = 999
                self.assertEqual(view.inventory()['payload.bin']['bytes'], 4)
            self.assertEqual(identity.call_count, 2)
        self.assertTrue(bundle.transaction_receipt['closed'])
        self.assertFalse(bundle.transaction_receipt['qualification_granted'])
        with self.assertRaises(ValueError):
            view.read_bytes('review.json')

    def test_unverified_missing_cache_and_duplicate_whitelist_rejected(self):
        for bundle, names in ((self.bundle(False), ['review.json']),
                              (self.bundle(), ['payload.bin']),
                              (self.bundle(), ['review.json', 'review.json'])):
            with self.assertRaises(ValueError):
                with bundle.verified_transaction(names):
                    self.fail('invalid transaction yielded a view')
            with self.assertRaises(ValueError):
                _ = bundle.transaction_receipt

    def test_caught_view_failure_is_sticky(self):
        for operation in ('read', 'hash', 'json'):
            bundle = self.bundle()
            with self.assertRaises(ValueError):
                with bundle.verified_transaction(['review.json']) as view:
                    try:
                        if operation == 'read':
                            view.read_bytes('payload.bin')
                        elif operation == 'hash':
                            view.file_sha256('missing')
                        else:
                            view.read_json('missing')
                    except ValueError:
                        pass
            self.assertFalse(bundle.verified)
            with self.assertRaises(ValueError):
                _ = bundle.transaction_receipt

    def test_nested_active_verify_and_early_receipt_fail_sticky(self):
        for operation in ('nested', 'verify', 'receipt', 'direct_read'):
            bundle = self.bundle()
            with self.assertRaises(ValueError):
                with bundle.verified_transaction(['review.json']):
                    try:
                        if operation == 'nested':
                            with bundle.verified_transaction(['review.json']):
                                self.fail('nested view yielded')
                        elif operation == 'verify':
                            bundle.verify_all()
                        elif operation == 'receipt':
                            _ = bundle.transaction_receipt
                        else:
                            bundle.read_bytes('review.json')
                    except ValueError:
                        pass
            with self.assertRaises(ValueError):
                _ = bundle.transaction_receipt

    def test_actual_pack_or_index_drift_prevents_commit(self):
        for name in ('archive', 'index_path'):
            bundle = self.bundle()
            with self.assertRaises(ValueError):
                with bundle.verified_transaction(['review.json']) as view:
                    self.assertEqual(view.read_json('review.json'), {'value': 42})
                    path = getattr(bundle, name)
                    path.write_bytes(path.read_bytes() + b'changed')
            self.assertTrue(view.closed)
            self.assertEqual(bundle._small_objects, {})
            with self.assertRaises(ValueError):
                _ = bundle.transaction_receipt

    def test_cached_content_drift_rejected_before_yield(self):
        bundle = self.bundle()
        bundle._small_objects['review.json'] = b'{"value":43}'
        with self.assertRaises(ValueError):
            with bundle.verified_transaction(['review.json']):
                self.fail('corrupt cache yielded view')

    def test_checker_exception_closes_view_without_receipt(self):
        bundle = self.bundle()
        with self.assertRaisesRegex(RuntimeError, 'checker failed'):
            with bundle.verified_transaction(['review.json']) as view:
                raise RuntimeError('checker failed')
        self.assertTrue(view.closed)
        self.assertFalse(bundle.verified)
        with self.assertRaises(ValueError):
            _ = bundle.transaction_receipt


if __name__ == '__main__':
    unittest.main()
