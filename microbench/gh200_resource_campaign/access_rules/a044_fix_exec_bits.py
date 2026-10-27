"""Copy a run package with the owner/group/other execute bits added to AArch64 ELF members that lack them.

Some recovered binaries entered packages as 0644 and failed on a044 with "Permission denied". Member order,
content, names and all other metadata are copied unchanged; package-identity.json binds content SHA256 only,
so it stays valid. The changed members are listed in <output>.exec-bits.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def aarch64_executable(head):
    return head[:4] == b'\x7fELF' and head[18:20] == b'\xb7\x00'


def fix(source, target):
    changed = []
    with tarfile.open(source, 'r:gz') as old, tarfile.open(target, 'x:gz') as new:
        for member in old:
            data = old.extractfile(member) if member.isfile() else None
            if data is not None and not member.mode & 0o100:
                head = data.read(20)
                data.seek(0)
                if aarch64_executable(head):
                    changed.append(dict(name=member.name, old_mode=oct(member.mode)))
                    member.mode |= 0o111
            new.addfile(member, data)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = dict(source=str(source), source_sha256=digest(source), target=str(target),
                  target_sha256=digest(target), changed=changed)
    Path(str(target) + '.exec-bits.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    print(json.dumps(fix(args.source, args.target)))
