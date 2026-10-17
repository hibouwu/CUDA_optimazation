"""核对冻结身份与时间顺序，评分运行前有效频率预测。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from analyze_frequency_development import analyze


def score(root):
    binding = json.loads((root/'binding.json').read_text())
    for name, digest in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    observations = analyze(root)
    frozen_ns = int(datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    assert min(process['host_start_ns'] for row in observations['rows']
               for process in row['processes']) > frozen_ns
    predictions = {row['id']: row for row in json.loads((root/'predictions.json').read_text())['rows']}
    config = json.loads((root/'run_config.json').read_text())
    threshold = config['criterion']['maximum_absolute_relative_error']
    rows = []
    for row in observations['rows']:
        predicted = predictions[row['case']['id']]['predicted_ghz']
        observed = row['metrics']['frequency_ghz']['median']
        error = predicted/observed-1
        rows.append(dict(id=row['case']['id'], predicted_ghz=predicted,
            observed_ghz=observed, error=error, passed=abs(error) <= threshold))
    return dict(schema='frequency-heldout-score/1', frozen_at=binding['frozen_at'],
        threshold=threshold, passed=all(row['passed'] for row in rows),
        maximum_absolute_error=max(abs(row['error']) for row in rows), rows=rows,
        observations=observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = score(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print('passed', result['passed'], 'maximum absolute error', result['maximum_absolute_error'])
    for row in result['rows']:
        print(row)
