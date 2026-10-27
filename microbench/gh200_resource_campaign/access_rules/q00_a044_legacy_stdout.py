#!/usr/bin/env python3
"""离线逐条件回放 legacy 原 stdout，核对重测 runner 的结果解析。"""
import argparse
import collections
from functools import lru_cache
import gzip
import importlib.util
import json
from pathlib import Path

from q00_a044_remeasure import digest, read, save


@lru_cache(maxsize=8)
def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def original_stdout(origin):
    record = Path(origin['record'])
    if record.suffix == '.jsonl':
        value = next(row for row in jsonl(record) if all(row.get(key) == origin.get(key)
                     for key in ('case', 'variant', 'trial')))
    else:
        value = read(record)
    if isinstance(value, dict) and value.get('raw'):
        return Path(origin['run']) / value['raw']
    if record.name.endswith('.process.json'):
        return record.with_name(record.name.removesuffix('.process.json') + '.stdout')
    candidates = [record.parent / name for name in
                  ('stdout.json', 'stdout.txt', 'stdout.txt.gz', 'stdout')]
    found = [path for path in candidates if path.exists()]
    if len(found) != 1:
        raise ValueError('原 stdout 定位不唯一：' + str(record))
    return found[0]


class Text:
    def __init__(self, path):
        self.path = path

    def read_text(self):
        if self.path.suffix == '.gz':
            with gzip.open(self.path, 'rt') as stream:
                return stream.read()
        return self.path.read_text()


def main(args):
    rows, packages = [], []
    for package in args.package:
        manifest = read(package / 'manifest.json')
        runner_path = args.runner or package / 'q00_a044_remeasure.py'
        spec = importlib.util.spec_from_file_location('audited_runner', runner_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        packages.append(dict(package=str(package), manifest_sha256=digest(package / 'manifest.json'),
                             runner=str(runner_path), runner_sha256=digest(runner_path)))
        for task in manifest['tasks']:
            origin = task['origins'][0]
            row = dict(task=task['id'], stages=task['stages'], record=origin['record'])
            try:
                source = original_stdout(origin)
                row.update(stdout=str(source), stdout_sha256=digest(source),
                           result=runner.inspect_stdout(Text(source), task['protocol']))
            except Exception as error:
                row['error'] = type(error).__name__ + ': ' + str(error)
            rows.append(row)
            if len(rows) % 200 == 0:
                print(len(rows), flush=True)
    counts = collections.Counter('exception' if 'error' in row else row['result']['status'] for row in rows)
    invalid = [row for row in rows if 'error' in row or not row['result']['valid']]
    save(args.output, dict(packages=packages,
        scope='每个唯一 binary+argv 条件回放一份原 stdout，保留原协议分类',
        counts=dict(counts), tasks=len(rows), invalid_tasks=len(invalid), invalid=invalid, rows=rows))
    print(json.dumps(dict(counts=dict(counts), tasks=len(rows), invalid_tasks=len(invalid))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True, action='append')
    parser.add_argument('--runner', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
