"""执行已登记的 tensor 探针条件，保存逐进程原始结果与命令。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root):
    plan = json.loads((root/'plan.json').read_text())
    if (root/'binding.json').exists():
        binding = json.loads((root/'binding.json').read_text())
        for name, expected in binding['files'].items():
            assert digest(root/name) == expected, name
    probe = root/'build/probe'
    assert digest(probe) == plan['probe_sha256']
    output = root/'samples'
    output.mkdir()
    invocations = []
    for repeat in range(plan['processes']):
        conditions = plan['conditions'][:]
        random.Random(plan['order_seed']+repeat).shuffle(conditions)
        for case in conditions:
            command = [str(probe)]
            for key, value in case['arguments'].items():
                command += ['--'+key, str(value)]
            name = f"r{repeat:02d}-{case['id']}"
            started = time.time_ns()
            with (output/(name+'.json')).open('w') as out, (output/(name+'.stderr')).open('w') as err:
                result = subprocess.run(command, stdout=out, stderr=err,
                                        timeout=plan['process_timeout_seconds'])
            record = dict(id=case['id'], repeat=repeat, command=command, started_ns=started,
                ended_ns=time.time_ns(), returncode=result.returncode,
                sha256=digest(output/(name+'.json')))
            invocations.append(record)
            (root/'invocations.json').write_text(json.dumps(invocations, indent=2)+'\n')
            if result.returncode:
                raise RuntimeError(f"{name} failed: {result.returncode}")
            sample = json.loads((output/(name+'.json')).read_text())
            assert sample['gpu_uuid'] == plan['gpu_uuid']
            assert sample['errors'] == 0 and sample['observed_sms'] == sample['ctas']
            assert sample['max_active_ctas_per_sm'] == 1
            print(name, 'ok', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    run(args.root.resolve())
