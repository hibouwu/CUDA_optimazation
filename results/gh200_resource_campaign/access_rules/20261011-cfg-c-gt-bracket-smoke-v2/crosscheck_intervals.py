"""Cross-check actual decoded GT-coordinate intervals with A's exact module."""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
MODULE=REPO/'microbench/gh200_resource_campaign/access_rules/r09_gt_brackets.py'
sys.path.insert(0,str(MODULE.parent))
from r09_gt_brackets import GtClockStamp,window_interval
v=json.loads((ROOT/'verification.json').read_text());windows=[]
for cta in v['results'][2]['decoded']['records']:
 for role in cta['roles']:
  for ordinal,tile in enumerate(role['tiles']):
   stamps=[GtClockStamp(v['gpu_uuid'],cta['sm'],s['gt_before'],s['cycles'],s['gt_after']) for s in tile['gt_brackets']]
   interval=window_interval(*stamps,gt_error_ns=0)
   assert [interval['elapsed_lower_ns'],interval['elapsed_upper_ns']]==tile['conditional_elapsed_gt']
   windows.append(dict(cta=cta['cta'],sm=cta['sm'],thread=role['physical_thread'],ordinal=ordinal,conditional_interval=[str(interval['elapsed_lower_ns']),str(interval['elapsed_upper_ns'])]))
with gzip.open(ROOT/'stamps.json.gz','rt') as stream:stamps=json.load(stream)
report=dict(windows_verified=len(windows),A_module_sha256=hashlib.sha256(MODULE.read_bytes()).hexdigest(),width_histogram=dict(sorted(Counter(s['width'] for s in stamps).items())),conditional_extra_GT_coordinate_error=0,interpretation='Exact GT read-coordinate arithmetic with explicit zero extra error budget only; not a physical-ns accuracy estimate.',qualification=False,windows=windows)
(ROOT/'A-interval-crosscheck.json').write_text(json.dumps(report,indent=2)+'\n')
print('verified conditional GT-coordinate windows:',len(windows))
