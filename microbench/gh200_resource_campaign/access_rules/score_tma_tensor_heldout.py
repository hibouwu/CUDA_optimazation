"""按第二轮冻结判据分别评分地址相位响应与八个局部时间。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from analyze_tma_tensor_probe import analyze


def score(root):
    binding = json.loads((root / 'binding.json').read_text())
    for name, expected in binding['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    plan = json.loads((root / 'plan.json').read_text())
    predictions = json.loads((root / 'predictions.json').read_text())
    invocations = json.loads((root / 'invocations.json').read_text())
    frozen_ns = int(datetime.fromisoformat(binding['frozen_at']).timestamp() * 1e9)
    assert min(row['started_ns'] for row in invocations) > frozen_ns
    observations = analyze(root)
    rows = {row['id']: row for row in observations['rows']}
    errors = []
    for prediction in predictions['rows']:
        observed = rows[prediction['id'] + '-t1']['summary']['release_window_ns']['median']
        predicted = prediction['predicted_median_ns']
        errors.append(dict(id=prediction['id'], predicted_ns=predicted, observed_ns=observed,
                           error=predicted / observed - 1))
    ids = ['n48-q1-independent-p1-d0', 'n48-q1-independent-p2-d0']
    before, after = [rows[name + '-t1'] for name in ids]
    response = after['summary']['release_window_ns']['median'] / before['summary']['release_window_ns']['median'] - 1
    plain_before, plain_after = [rows[name + '-t0'] for name in ids]
    bridge = plain_after['summary']['median_cta_ns']['median'] / plain_before['summary']['median_cta_ns']['median'] - 1
    paired = []
    for repeat in range(plan['processes']):
        left = next(row for row in before['processes'] if row['repeat'] == repeat)
        right = next(row for row in after['processes'] if row['repeat'] == repeat)
        paired.append(right['release_window_ns'] / left['release_window_ns'] - 1)
    c2 = plan['C2']
    candidate_pass = abs(response - c2['candidate_response']) <= c2['tolerance']
    null_pass = abs(response - c2['null_response']) <= c2['tolerance']
    bridge_pass = bridge >= .03
    conclusion = ('通过' if candidate_pass and not null_pass and bridge_pass else
                  '被反证' if not candidate_pass else '信息不足')
    maximum = max(abs(row['error']) for row in errors)
    return dict(schema='tensor-heldout-score/1', frozen_at=binding['frozen_at'],
        C2=dict(predicted_response=c2['candidate_response'], observed_response=response,
                null_response=c2['null_response'], tolerance=c2['tolerance'],
                candidate_pass=candidate_pass, null_pass=null_pass, untraced_response=bridge,
                trace_bridge_pass=bridge_pass, paired_process_responses=paired, conclusion=conclusion),
        local_timing=dict(max_absolute_relative_error=maximum,
            threshold=plan['local_timing']['max_absolute_relative_error'],
            passed=maximum <= plan['local_timing']['max_absolute_relative_error'], rows=errors),
        observations=observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = score(args.root)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report[key] for key in ('C2', 'local_timing')}, indent=2, ensure_ascii=False))
