"""cfg_b 聚合组合的冻结完整时间评分，频率与旧版基线分列。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_frequency_development import analyze


def score(root):
    binding=json.loads((root/'binding.json').read_text())
    for name,digest in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    observations=analyze(root)
    frozen_ns=int(datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    assert min(p['host_start_ns'] for row in observations['rows'] for p in row['processes'])>frozen_ns
    predictions={r['id']:r for r in json.loads((root/'predictions.json').read_text())['predictions']}
    criteria=json.loads((root/'run_config.json').read_text())['criterion']
    rows=[]
    for row in observations['rows']:
        frozen=predictions[row['case']['id']]
        prediction,baseline=frozen['prediction'],frozen['baseline_v09']
        actual=row['metrics']['plain_us']['median'];frequency=row['metrics']['frequency_ghz']['median']
        error=prediction['plain_us']/actual-1;fe=prediction['frequency_ghz']/frequency-1
        rows.append(dict(id=row['case']['id'],predicted_plain_us=prediction['plain_us'],observed_plain_us=actual,
            plain_error=error,baseline_plain_us=baseline['plain_us'],baseline_plain_error=baseline['plain_us']/actual-1,
            predicted_ghz=prediction['frequency_ghz'],observed_ghz=frequency,frequency_error=fe,
            time_pass=abs(error)<=criteria['maximum_absolute_relative_error'],frequency_pass=abs(fe)<=criteria['frequency_maximum_absolute_relative_error']))
    return dict(schema='cfg-b-time-heldout-score/1',frozen_at=binding['frozen_at'],criteria=criteria,
        C3_passed=all(r['time_pass'] for r in rows),frequency_passed=all(r['frequency_pass'] for r in rows),
        median_absolute_error=st.median(abs(r['plain_error']) for r in rows),
        maximum_absolute_error=max(abs(r['plain_error']) for r in rows),
        baseline_median_absolute_error=st.median(abs(r['baseline_plain_error']) for r in rows),
        baseline_maximum_absolute_error=max(abs(r['baseline_plain_error']) for r in rows),
        rows=rows,observations=observations)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=score(args.root);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ('observations','rows')})
    for row in report['rows']:print(row)
