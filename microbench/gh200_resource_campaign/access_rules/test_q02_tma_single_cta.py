#!/usr/bin/env python3
"""用旧原始记录及单 CTA 合成记录检查验证器；不执行探针。"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import q02_tma_single_cta as single
import q02_tma_stride as stride
from l2_saturation import digest, read_json, save_json


def write_rows(folder, task, root, rows):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'stdout').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (folder/'stderr').write_text('')
    (folder/'receipt.json').write_text(json.dumps(dict(argv=stride.probe_command(root, task), returncode=0,
        timed_out=False, stdout_sha256=digest(folder/'stdout'), stderr_sha256=digest(folder/'stderr')))+'\n')


def check(root):
    spec = importlib.util.spec_from_file_location('q02_original_validator', root/'baseline/q02_tma_stride.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    uuid = read_json(root/'plan.json')['gpu_uuid']
    replay = root/'baseline/replay'
    old_processes = 0
    for name, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
        for task in read_json(replay/name):
            folder = replay/'runs'/split/task['id']
            before = old.validate_process(replay, folder, task, uuid)
            after = stride.validate_process(replay, folder, task, uuid)
            assert before == after == read_json(folder/'validated.json')
            old_processes += 1
    checks = ['旧132 CTA的12进程／36臂逐项等于原验证结果']
    with tempfile.TemporaryDirectory(prefix='q02-single-cta-cpu-') as temporary:
        fixture = Path(temporary)/'fixture'
        shutil.copytree(root, fixture)
        (fixture/'allocation').mkdir()
        save_json(fixture/'allocation/identity.json', dict(uuid=uuid))
        for name, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
            for task in read_json(fixture/name):
                original = [json.loads(line) for line in (replay/'runs'/split/task['id']/'stdout').read_text().splitlines()]
                by_ldb = {row['ldb']:row for row in original}
                rows = []
                for index, ldb in enumerate(task['paired_ldbs']):
                    row = copy.deepcopy(by_ldb[ldb])
                    row.update(ctas=1, observed_sms=1, paired_ldbs=task['paired_ldbs'], pair_index=index,
                               checked=64*16384 if row['verify'] else 0)
                    row['ctas_data'] = [row['ctas_data'][0]]
                    row['ctas_data'][0]['smid'] = 42
                    rows.append(row)
                folder = fixture/'runs'/split/task['id']
                write_rows(folder, task, fixture, rows)
                assert len(single.validate_pair(fixture, folder, task, uuid)) == 2
        report = single.summarize(fixture)
        assert report['numeric_arm_checks'] == report['formal_arm_observations'] == 6
        assert len(report['pairs']) == 3 and report['independent_C2'] is False
        checks.append('合法1 CTA的3 verify／3 plain合成进程及配对评分通过')
        task = read_json(fixture/'correctness_commands.json')[0]
        folder = fixture/'runs/correctness'/task['id']
        original = [json.loads(line) for line in (folder/'stdout').read_text().splitlines()]
        mutations = [
            ('错误CTA数', lambda rows:rows[0].update(ctas=132)),
            ('错误工作坐标', lambda rows:rows[0]['ctas_data'][0].update(tile=[7,7])),
            ('配对落在不同SM', lambda rows:rows[1]['ctas_data'][0].update(smid=43))]
        for label, mutate in mutations:
            rows = copy.deepcopy(original)
            mutate(rows)
            write_rows(folder, task, fixture, rows)
            try:
                single.validate_pair(fixture, folder, task, uuid)
            except ValueError:
                checks.append('拒绝'+label)
            else:
                raise AssertionError('接受了'+label)
        write_rows(folder, task, fixture, original)
        (fixture/'worklist.txt').write_text('2304 2304 4096 132\n0 0\n7 6\n')
        try:
            single.validate_pair(fixture, folder, task, uuid)
        except ValueError:
            checks.append('拒绝工作列表header CTA数错误')
        else:
            raise AssertionError('接受了错误header')
    return dict(schema='gh200-q02-single-cta-cpu-check/1', checks=checks,
                old_processes=old_processes, old_arms=36, synthetic_processes=6,
                synthetic_arms=12, gpu_programs_executed=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = check(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
