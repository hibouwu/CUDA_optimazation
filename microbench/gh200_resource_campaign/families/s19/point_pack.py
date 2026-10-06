"""单个新短检查的完整、无损压缩归档；仅在发布核验后释放自己的临时副本。"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys

if sys.flags.optimize != 0:
    raise RuntimeError('S16 requires active assertions; Python optimization forbidden')
import os
import shutil
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(b)
    return digest.hexdigest()


def inventory(root):
    root = Path(root)
    assert root.is_dir() and not root.is_symlink() and root.stat().st_uid == os.geteuid()
    result = {}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink()
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                                                        'mode': path.stat().st_mode & 0o777}
        else:
            assert path.is_dir()
    assert result
    return result


def verify_pack(archive, members):
    seen = set()
    with tarfile.open(archive, 'r:xz') as packet:
        for item in packet:
            name = item.name
            relative = PurePosixPath(name)
            assert item.isfile() and name in members and name not in seen
            assert not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == name
            expected = members[name]
            assert item.size == expected['bytes'] and item.mode == expected['mode']
            h = hashlib.sha256()
            count = 0
            f = packet.extractfile(item)
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
                count += len(chunk)
            assert count == expected['bytes'] and h.hexdigest() == expected['sha256']
            seen.add(name)
    assert seen == set(members)
    # Decode through EOF as well: tar end blocks alone do not verify trailing XZ integrity.
    import lzma
    with lzma.open(archive, 'rb') as f:
        while f.read(1024 * 1024):
            pass
    return {'member_count': len(seen), 'uncompressed_file_bytes': sum(v['bytes'] for v in members.values()),
            'all_members_SHA_size_mode_verified': True, 'xz_EOF_verified': True}


class ArchiveLimitExceeded(RuntimeError):
    pass


class LimitedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.written = stream, limit, 0

    def write(self, data):
        if self.written + len(data) > self.limit:
            raise ArchiveLimitExceeded('compressed packet reached its frozen byte ceiling; raw retained')
        count = self.stream.write(data)
        self.written += count
        return count

    def __getattr__(self, name):
        return getattr(self.stream, name)


def make_pack(point, archive, max_archive_bytes=None):
    point, archive = Path(point), Path(archive)
    assert not archive.exists() and not archive.is_relative_to(point)
    members = inventory(point)
    import lzma
    filters = [{'id': lzma.FILTER_DELTA, 'dist': 4}, {'id': lzma.FILTER_LZMA2, 'preset': 1}]
    with archive.open('xb') as stream:
        sink = stream if max_archive_bytes is None else LimitedWriter(stream, max_archive_bytes)
        with lzma.LZMAFile(sink, 'wb', filters=filters) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as packet:
                for name in members:
                    packet.add(point / name, arcname=name, recursive=False)
    assert inventory(point) == members
    proof = verify_pack(archive, members)
    return {'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
            'members': members, 'verification': proof}


def publish_pack(archive, destination, identity):
    archive, destination = Path(archive), Path(destination)
    assert destination.parent.is_dir() and not destination.exists() and not destination.parent.is_symlink()
    partial = destination.with_suffix(destination.suffix + '.partial')
    assert not partial.exists() and sha(archive) == identity['archive_sha256']
    with archive.open('rb') as source, partial.open('xb') as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    assert partial.stat().st_size == identity['archive_bytes'] and sha(partial) == identity['archive_sha256']
    proof = verify_pack(partial, identity['members'])
    # Exclusive publication, followed by parent-directory fsync.
    os.link(partial, destination)
    partial.unlink()
    fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return proof


def release_generated_point(point, archive, destination, identity, owned_stage):
    point, archive, owned_stage = Path(point), Path(archive), Path(owned_stage)
    assert owned_stage.is_dir() and not owned_stage.is_symlink() and owned_stage.stat().st_uid == os.geteuid()
    assert owned_stage.stat().st_mode & 0o077 == 0 and owned_stage.parent == Path('/tmp')
    assert owned_stage.name.startswith('codex-s15-allGPU-')
    assert point.parent == owned_stage and archive.parent == owned_stage
    assert point.name.startswith('index') and archive.name == point.name + '.tar.xz'
    assert inventory(point) == identity['members']
    assert destination.is_file() and not destination.is_symlink()
    assert destination.stat().st_size == identity['archive_bytes'] and sha(destination) == identity['archive_sha256']
    # The original node-local files are new temporary products; no historical directory is touched.
    shutil.rmtree(point)
    archive.unlink()
    return {'node_local_generated_point_released': True, 'persistent_archive_retained': str(destination),
            'persistent_archive_sha256': identity['archive_sha256'], 'historical_files_removed': False}
