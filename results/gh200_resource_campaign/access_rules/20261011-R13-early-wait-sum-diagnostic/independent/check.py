"""Compare all block contrasts with B's independent numeric implementation."""
from pathlib import Path
import hashlib
import importlib.util
import json

root=Path(__file__).resolve().parent
payload=(root.parent/'verification.json').read_bytes()
verified=json.loads(payload)
assert verified['archive_sha256']=='838834ab144f15e460f173c64370126e57dade55d77a58ddb109f215ea5183a2'
assert hashlib.sha256((root/'proposal.json').read_bytes()).hexdigest()=='132cb4ffd7f62d6108a100c7d1a15cda27a39262f47e76126568f2ce1bfc4fa0'
spec=importlib.util.spec_from_file_location('independent_B',root/'plan.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=[dict(block=r['block'],arm=f's{r["stages"]}_{"on" if r["observer"] else "off"}',decoded=r)
      for r in verified['records']]
other=module.contrasts(rows)
ours=['stage_D_on','stage_Q_on','stage_R_on','wait_minus_outside','stage_D_off',
      'stage_effect_change','observer_D_s4','observer_D_s6']
for a,b in zip(ours,module.SERIES,strict=True):
    assert [r[a] for r in verified['blocks']]==[r[b] for r in other['rows']]
    for field in ('median','conditional_median_interval'):
        assert verified['summary'][a][field]==other['series'][b][field]
report=dict(verified_report_sha256=hashlib.sha256(payload).hexdigest(),
            independent_code_sha256=hashlib.sha256((root/'plan.py').read_bytes()).hexdigest(),
            block_contrasts_checked=16*8,summary_values_checked=8*3,
            every_block_wait_difference_positive=all(r['stage_Q_on']>0 for r in verified['blocks']),
            every_block_outside_difference_negative=all(r['stage_R_on']<0 for r in verified['blocks']),
            stage6_observer_positive_blocks=sum(r['observer_D_s6']>0 for r in verified['blocks']),
            observation_qualified=False,prediction_qualified=False)
with (root/'verification.json').open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
