"""Read-only repository-relative evidence packs; no GPU or executable dispatch."""
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import tarfile

from common.suite_io import require, sha

SMALL_OBJECT_LIMIT = 16 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
CACHE_BYTES_LIMIT = 64 * 1024 * 1024


def json_object(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate packed JSON key: ' + key)
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError('nonfinite packed JSON: ' + value)))


def logical_path(name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
            'invalid packed logical path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts
            and path.as_posix() == name and name != '.', 'unsafe/noncanonical packed path')
    return name


def digest_value(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value),
            'packed SHA256 format')
    return value


class PackedEvidence:
    def __init__(self, archive, index, *, expected_index_sha256,
                 expected_archive_sha256, expected_closure):
        self.archive, self.index_path = Path(archive), Path(index)
        require(self.archive.is_file() and not self.archive.is_symlink()
                and self.index_path.is_file() and not self.index_path.is_symlink(),
                'packed inputs must be regular files')
        self.index_sha256 = digest_value(expected_index_sha256)
        self.archive_sha256 = digest_value(expected_archive_sha256)
        require(sha(self.index_path) == self.index_sha256, 'packed index identity drift')
        require(self.index_path.stat().st_size <= SMALL_OBJECT_LIMIT, 'packed index too large')
        self.index = json_object(self.index_path.read_bytes())
        require(set(self.index) == {'schema_version', 'namespace', 'archive_sha256',
                                   'archive_bytes', 'members', 'roots'}, 'packed index fields')
        require(type(self.index['schema_version']) is int and self.index['schema_version'] == 1
                and self.index['namespace'] == 'repository_relative_v1', 'packed index revision')
        require(self.index['archive_sha256'] == self.archive_sha256
                and type(self.index['archive_bytes']) is int and self.index['archive_bytes'] > 0,
                'packed archive identity/size')
        require(isinstance(self.index['members'], dict) and self.index['members'], 'packed members')
        require(isinstance(expected_closure, dict) and expected_closure,
                'independently bound expected closure required')
        self.members = self.index['members']
        for name, record in self.members.items():
            logical_path(name)
            require(isinstance(record, dict) and set(record) == {'bytes', 'sha256'},
                    'packed member fields')
            require(type(record['bytes']) is int and record['bytes'] >= 0, 'packed member length')
            digest_value(record['sha256'])
        require(self.members == expected_closure, 'packed index differs from independent closure')
        roots = self.index['roots']
        require(isinstance(roots, dict) and roots, 'packed logical roots')
        for value in roots.values():
            names = value if isinstance(value, list) else [value]
            require(names, 'empty packed root list')
            for name in names:
                require(logical_path(name) in self.members, 'packed root missing')
        self.verified = False
        self._small_objects = {}
        self._active_view = None
        self._transaction_receipt = None

    def _no_active_operation(self):
        if self._active_view is not None:
            self._active_view.failed = True
            raise ValueError('use the bounded view during a packed transaction')

    def verified_transaction(self, required_small_objects):
        from common.packed_transaction import verified_transaction
        return verified_transaction(self, required_small_objects)

    @property
    def transaction_receipt(self):
        from common.packed_transaction import transaction_receipt
        return transaction_receipt(self)

    def _identity(self):
        require(self.archive.stat().st_size == self.index['archive_bytes']
                and sha(self.archive) == self.archive_sha256, 'packed archive changed')
        require(sha(self.index_path) == self.index_sha256, 'packed index changed')

    def verify_all(self, small_objects=(), observer=None):
        self._no_active_operation()
        self._transaction_receipt = None
        self.verified = False
        self._small_objects = {}
        require(isinstance(small_objects, (list, tuple, set)), 'explicit packed small-object whitelist')
        names = list(small_objects)
        require(len(names) == len(set(names)), 'duplicate packed small-object request')
        for name in names:
            require(logical_path(name) in self.members, 'unknown cached packed member')
            require(self.members[name]['bytes'] <= SMALL_OBJECT_LIMIT, 'cached object too large')
        require(sum(self.members[name]['bytes'] for name in names) <= CACHE_BYTES_LIMIT,
                'packed cache total limit')
        require(observer is None or callable(observer), 'code-owned packed chunk observer required')
        cached = {name: bytearray() for name in names}
        self._identity()
        # Read the XZ stream to EOF so tar's end markers cannot hide a damaged
        # compression footer. Bound even padding/header expansion in this pass.
        limit = sum(((r['bytes'] + 511) // 512) * 512 + 4096
                    for r in self.members.values()) + 10240
        expanded = 0
        with lzma.open(self.archive, 'rb') as compressed:
            while True:
                chunk = compressed.read(CHUNK_BYTES)
                if not chunk:
                    break
                expanded += len(chunk)
                require(expanded <= limit, 'packed decompression exceeds declared bound')
        seen = set()
        total = 0
        with tarfile.open(self.archive, mode='r|*', ignore_zeros=True) as stream:
            for member in stream:
                name = logical_path(member.name)
                require(member.isfile() and not member.issym() and not member.islnk(),
                        'packed member must be a regular file')
                require(name not in seen and name in self.members, 'duplicate/extra packed member')
                record = self.members[name]
                require(member.size == record['bytes'], 'packed member declared size drift')
                source = stream.extractfile(member)
                require(source is not None, 'packed member stream missing')
                digest = hashlib.sha256()
                length = 0
                while True:
                    chunk = source.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    offset = length
                    length += len(chunk)
                    require(length <= record['bytes'], 'packed member exceeds length')
                    digest.update(chunk)
                    if name in cached:
                        cached[name].extend(chunk)
                    if observer is not None:
                        observer(name, offset, chunk)
                require(length == record['bytes'] and digest.hexdigest() == record['sha256'],
                        'packed member content drift: ' + name)
                seen.add(name)
                total += length
            tar_decoded_bytes = stream.fileobj.tell()
        require(seen == set(self.members), 'missing packed member')
        self._identity()
        self._small_objects = {name: bytes(data) for name, data in cached.items()}
        self.verified = True
        return {'members': len(seen), 'uncompressed_bytes': total,
                'archive_sha256': self.archive_sha256, 'index_sha256': self.index_sha256,
                'decoder_passes': 2, 'xz_EOF_decoded_bytes': expanded,
                'tar_decoded_bytes': tar_decoded_bytes,
                'cached_objects': len(cached),
                'cached_bytes': sum(len(data) for data in cached.values())}

    def inventory(self, prefix=''):
        self._no_active_operation()
        require(self.verified, 'complete packed verification required')
        require(isinstance(prefix, str), 'packed prefix type')
        self._identity()
        return {name: dict(record) for name, record in self.members.items()
                if name.startswith(prefix)}

    def file_sha256(self, name):
        self._no_active_operation()
        require(self.verified, 'complete packed verification required')
        require(logical_path(name) in self.members, 'unknown packed member')
        self._identity()
        return self.members[name]['sha256']

    def iter_bytes(self, name, chunk_bytes=CHUNK_BYTES):
        self._no_active_operation()
        require(self.verified, 'complete packed verification required')
        require(logical_path(name) in self.members, 'unknown packed member')
        require(type(chunk_bytes) is int and 0 < chunk_bytes <= CHUNK_BYTES, 'packed chunk limit')
        self._identity()
        with tarfile.open(self.archive, mode='r|*') as stream:
            for member in stream:
                if member.name != name:
                    continue
                require(member.isfile() and member.size == self.members[name]['bytes'],
                        'packed read member drift')
                source = stream.extractfile(member)
                require(source is not None, 'packed member stream missing')
                length = 0
                digest = hashlib.sha256()
                while True:
                    chunk = source.read(chunk_bytes)
                    if not chunk:
                        break
                    length += len(chunk)
                    digest.update(chunk)
                    yield chunk
                require(length == self.members[name]['bytes']
                        and digest.hexdigest() == self.members[name]['sha256'],
                        'packed streamed content drift')
                self._identity()
                return
        raise ValueError('packed member disappeared')

    def read_bytes(self, name):
        self._no_active_operation()
        require(logical_path(name) in self.members, 'unknown packed member')
        require(self.members[name]['bytes'] <= SMALL_OBJECT_LIMIT, 'packed small-object limit')
        require(self.verified, 'complete packed verification required')
        try:
            self._identity()
        except (ValueError, OSError):
            self._small_objects = {}
            self.verified = False
            raise
        if name in self._small_objects:
            data = self._small_objects[name]
            require(len(data) == self.members[name]['bytes']
                    and hashlib.sha256(data).hexdigest() == self.members[name]['sha256'],
                    'packed small-object cache drift')
            return data
        return b''.join(self.iter_bytes(name))

    def read_json(self, name):
        return json_object(self.read_bytes(name))


def validate_packed_gate(bundle, path, stage, phase):
    """Historical gates resolve entirely within the verified evidence namespace."""
    from common.suite_io import DIMENSIONS, gate_mapping
    review = bundle.read_json(path)
    require(review.get('schema_version') == 2 and review.get('stage') == stage
            and review.get('phase') == phase and review.get('status') == 'pass',
            'packed historical gate identity/status')
    authors = set(re.findall(r'/root(?:/[a-zA-Z0-9_]+)?', review.get('implementer', '')))
    require(review.get('reviewer') and review.get('implementer')
            and review['reviewer'] != review['implementer'] and review['reviewer'] not in authors,
            'packed historical gate not independent')
    require(all(review.get('checks', {}).get(key, {}).get('status') == 'pass'
                for key in DIMENSIONS), 'packed historical gate dimensions')
    require(not any(f.get('status') not in ('closed', 'resolved', 'fixed', 'verified')
                    for f in review.get('findings', [])), 'packed historical finding open')
    inventory = bundle.inventory()
    for name, expected in gate_mapping(review).items():
        require(name in inventory and inventory[name]['sha256'] == expected,
                'packed historical gate file drift')
    bundle._identity()
    return review
