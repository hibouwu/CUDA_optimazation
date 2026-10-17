"""按冻结的相关性、页对同号率与冷/热窗口门槛核对局部域。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from analyze_numa_affinity import analyze


def score(root):
    plan=json.loads((root/'plan.json').read_text());binding=json.loads((root/'binding.json').read_text())
    for name,h in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    calls=json.loads((root/'returned/invocations.json').read_text())
    assert len(calls)==len(plan['conditions'])
    assert {c['seed'] for c in calls}=={c['seed'] for c in plan['conditions']}
    earliest=int(datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    development=json.loads((root/'evidence/development.json').read_text());used=set(development['pairs']);rows=[]
    owner={g:i for i,groups in enumerate(plan['groups']) for g in groups}
    for call in calls:
        path=root/'returned/samples'/f"s{call['seed']}.json"
        assert call['returncode']==0 and call['started_ns']>earliest
        assert hashlib.sha256(path.read_bytes()).hexdigest()==call['sha256']
        expected=next(c['pairs'] for c in plan['conditions'] if c['seed']==call['seed'])
        data=analyze(path)
        assert data['gpu_uuid']==plan['gpu_uuid'] and data['pairs']==expected
        assert not used.intersection(expected)
        used.update(expected)
        signed=[data['group_correlation'][a][b]*(1 if owner[a]==owner[b] else -1)
                for a in range(8) for b in range(a+1,8)]
        ratio=data['hot_quantiles'][3]/data['cold_quantiles'][3]
        checks=dict(correlation=min(signed)>=plan['minimum_signed_correlation'],
            same_gpc=min(data['pair_sign_agreement'])>=plan['minimum_same_gpc_sign_agreement'],
            hot_ratio=ratio<=plan['maximum_hot_cold_median_ratio'])
        rows.append(dict(seed=call['seed'],minimum_signed_correlation=min(signed),
            minimum_same_gpc_sign_agreement=min(data['pair_sign_agreement']),hot_cold_median_ratio=ratio,
            checks=checks,passed=all(checks.values()),analysis=data))
    return dict(conclusion='获得支持' if all(r['passed'] for r in rows) else '被反证',groups=plan['groups'],rows=rows,
        qualification=plan['qualification'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=score(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['conclusion'])
    for row in r['rows']:
        print(row['seed'],row['minimum_signed_correlation'],row['minimum_same_gpc_sign_agreement'],row['hot_cold_median_ratio'],row['checks'])
