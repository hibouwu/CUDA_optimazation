"""Bind local records to a caller-pinned acquisition archive without extracting it."""
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile


def verify_acquisition(run, expected_output_sha256):
    archive = (run / 'output.tar.gz').read_bytes()
    if hashlib.sha256(archive).hexdigest() != expected_output_sha256:
        raise ValueError('acquisition archive SHA differs from caller-pinned identity')
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
        members = {}
        for member in bundle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('unsafe acquisition archive member')
            key = str(name)
            if key in members:
                raise ValueError('duplicate acquisition archive member')
            members[key] = member

        def bound_bytes(name, local):
            member = members.get(name)
            if member is None or not member.isfile():
                raise ValueError('missing regular acquisition member: ' + name)
            data = bundle.extractfile(member).read()
            if local.read_bytes() != data:
                raise ValueError('local file differs from pinned acquisition member: ' + name)
            return data

        manifest_bytes = bound_bytes('manifest.json', run / 'manifest.json')
        records_bytes = bound_bytes('records.json', run / 'returned/records.json')
        complete_bytes = bound_bytes('complete.json', run / 'returned/complete.json')
        manifest = json.loads(manifest_bytes)
        records = json.loads(records_bytes)
        complete = json.loads(complete_bytes)
        if complete != dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                            records=len(records)):
            raise ValueError('acquisition completion record mismatch')
        hashes = {'output.tar.gz': expected_output_sha256,
                  'manifest.json': hashlib.sha256(manifest_bytes).hexdigest(),
                  'returned/records.json': hashlib.sha256(records_bytes).hexdigest(),
                  'returned/complete.json': hashlib.sha256(complete_bytes).hexdigest()}
        for record in records:
            name = record['raw']
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or str(path) != name:
                raise ValueError('unsafe or noncanonical raw member name')
            data = bound_bytes(name, run / 'returned' / name)
            digest = hashlib.sha256(data).hexdigest()
            if digest != record['sha256']:
                raise ValueError('archived raw differs from archived record SHA')
            hashes['returned/' + name] = digest
    return manifest, records, hashes
