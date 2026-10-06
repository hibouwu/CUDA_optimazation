"""Bounded read-only views between two actual evidence-pack identity checks."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path

from common.suite_io import require, sha
from common.packed_evidence import json_object, logical_path


class EvidenceView:
    def __init__(self, bundle, names):
        self.bundle = bundle
        self.names = tuple(names)
        self.failed = False
        self.closed = False
        self._cache = {name: bundle._small_objects[name] for name in names}
        self._members = deepcopy(bundle.members)

    def _check(self):
        require(not self.closed and self.bundle._active_view is self and not self.failed,
                'packed transaction view is closed or failed')

    def _operation(self, function):
        try:
            self._check()
            return function()
        except BaseException:
            self.failed = True
            raise

    def read_bytes(self, name):
        def read():
            require(logical_path(name) in self._cache, 'transaction read outside whitelist')
            return self._cache[name]
        return self._operation(read)

    def read_json(self, name):
        return self._operation(lambda: json_object(self.read_bytes(name)))

    def file_sha256(self, name):
        def digest():
            require(logical_path(name) in self._members, 'unknown transaction member')
            return self._members[name]['sha256']
        return self._operation(digest)

    def inventory(self, prefix=''):
        def listing():
            require(isinstance(prefix, str), 'transaction prefix type')
            return {name: dict(record) for name, record in self._members.items()
                    if name.startswith(prefix)}
        return self._operation(listing)

    def close(self):
        self.closed = True
        self._cache = {}
        self._members = {}


@contextmanager
def verified_transaction(bundle, required_small_objects):
    if bundle._active_view is not None:
        bundle._active_view.failed = True
        raise ValueError('nested packed transaction forbidden')
    bundle._transaction_receipt = None
    require(bundle.verified, 'complete packed verification required before transaction')
    require(isinstance(required_small_objects, (list, tuple, set)),
            'explicit transaction whitelist required')
    names = list(required_small_objects)
    require(all(isinstance(name, str) for name in names) and len(names) == len(set(names)),
            'transaction whitelist type/duplicates')
    bundle._identity()
    for name in names:
        require(logical_path(name) in bundle._small_objects,
                'transaction requires previously verified cached member')
        data = bundle._small_objects[name]
        record = bundle.members[name]
        require(len(data) == record['bytes']
                and hashlib.sha256(data).hexdigest() == record['sha256'],
                'transaction cached content drift')
    view = EvidenceView(bundle, names)
    bundle._active_view = view
    try:
        yield view
        require(not view.failed, 'packed transaction has a caught failure')
        bundle._identity()
    except BaseException:
        view.failed = True
        bundle._transaction_receipt = None
        bundle._small_objects = {}
        bundle.verified = False
        raise
    else:
        # Invalidate the view before making a successful receipt observable.
        view.close()
        bundle._active_view = None
        bundle._transaction_receipt = {
            'closed': True, 'archive_sha256': bundle.archive_sha256,
            'index_sha256': bundle.index_sha256,
            'small_objects': sorted(names),
            'transaction_source_sha256': sha(Path(__file__)),
            'resolver_source_sha256': sha(Path(__file__).with_name('packed_evidence.py')),
            'qualification_granted': False,
        }
    finally:
        view.close()
        bundle._active_view = None


def transaction_receipt(bundle):
    if bundle._active_view is not None:
        bundle._active_view.failed = True
        raise ValueError('packed transaction receipt is not closed')
    require(bundle._transaction_receipt is not None,
            'successful closed packed transaction receipt missing')
    return deepcopy(bundle._transaction_receipt)
