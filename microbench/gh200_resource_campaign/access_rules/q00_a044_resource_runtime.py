#!/usr/bin/env python3
"""从原 receipt 的进程区间估计重放用时；只读取 CPU 文件。"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tarfile


def measure(task):
    origin = task['origins'][0]
    if origin.get('restored_prior_short'):
        return dict(id=task['id'], family=task['stages'][0], seconds=None,
                    source=origin['receipt'], reason='原短校验复用；本包无独立 receipt')
    source = origin['receipt']
    if '::' in source:
        filename, membername = source.split('::', 1)
        with tarfile.open(filename, 'r|*') as archive:
            for member in archive:
                if member.name.lstrip('./') == membername:
                    data = archive.extractfile(member).read()
                    break
            else:
                raise ValueError(source)
    else:
        data = Path(source).read_bytes()
    assert hashlib.sha256(data).hexdigest() == origin['source_receipt_sha256'], source
    receipt = json.loads(data)
    start, stop = receipt['host_start_ns'], receipt['host_stop_ns']
    assert stop > start
    return dict(id=task['id'], family=task['stages'][0], seconds=(stop-start)/1e9,
                source=source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    tasks = json.loads(args.manifest.read_text())['tasks']
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(measure, tasks), 1):
            rows.append(row)
            if index % 250 == 0:
                print(index, '/', len(tasks), flush=True)
    families = []
    for name in dict.fromkeys(row['family'] for row in rows):
        subset = [row for row in rows if row['family'] == name]
        families.append(dict(family=name, calls=len(subset),
            known_calls=sum(row['seconds'] is not None for row in subset),
            seconds=sum(row['seconds'] or 0 for row in subset),
            missing_calls=sum(row['seconds'] is None for row in subset)))
    result = dict(source=str(args.manifest.resolve()),
        definition='sum(host_stop_ns-host_start_ns)，不包括进程之间的间隔和本轮压缩',
        families=families, calls=rows)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(families, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
