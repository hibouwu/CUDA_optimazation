"""Element-address verification of static 128B unions; no timing fit."""
from fractions import Fraction as F
import hashlib,itertools,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;repo=out.parents[5]
code=repo/'microbench/gh200_resource_campaign/access_rules'
sys.path.insert(0,str(code))
from supply_model import cfg_b_source_program
from analyze_r10_granule_overlap import source_granules
inputs={}
def read(path):
    inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
contexts=read(out.parent/'B-20261011-source-event-transfer-v3/B-r10-event-transfer.json')['contexts']
fixed=read(out.parent/'main-input-path-v1/B-input-path.json')
results=[]
for case in ('r10_source_single126_p0','r10_source_single126_p32'):
    context=contexts[case];program=cfg_b_source_program(context['row'],context['setup'])
    assert program['output_counts']==[1]*126
    keys={key for c in program['source_keys'] for group in c for key in group}
    independent={};elements=0
    for key in keys:
        op,dtype,layout,extents,strides,offset,origin,shape=key
        assert dtype=='fp16'
        covered=set()
        for i in range(shape[0]):
            x=origin[0]+i
            if not 0<=x<extents[0]:continue
            for j in range(shape[1]):
                y=origin[1]+j
                if not 0<=y<extents[1]:continue
                address=offset+x*strides[0]+y*strides[1]
                covered.add(address//128);covered.add((address+1)//128);elements+=1
        assert covered==set(source_granules(key,128))
        independent[key]={(op,unit) for unit in covered}
    groups=[];unions=[]
    for g in range(16):
        per_op={op:set() for op in ('A','B')}
        for c in program['source_keys']:
            for key in c[g]:per_op[key[0]].update(independent[key])
        union=per_op['A']|per_op['B'];unions.append(union)
        groups.append(dict(group=g,A_units=len(per_op['A']),B_units=len(per_op['B']),
            combined_units=len(union),service_once_per_unit_ns=str(
                F.from_float(fixed['parameters']['service_ns'])*len(union)/128)))
    overlaps=[dict(first=a,second=b,units=len(unions[a]&unions[b]))
        for a,b in itertools.combinations(range(16),2)]
    assert all(r['units']==0 for r in overlaps)
    results.append(dict(case=case,unique_descriptors=len(keys),valid_elements_checked=elements,
        groups=groups,cross_group_overlap_checks=overlaps,
        no_intergroup_address_unit_sharing=True))
    source_granules.cache_clear()
for name in ('supply_model.py','analyze_r10_granule_overlap.py'):
    path=code/name;inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
report=dict(input_sha256=inputs,results=results,
    scope='Static address unions in the existing model. Once-per-unit service requires a separate no-expiry proof; unions are not physical traffic.',
    source_subscription_lifetime_proven=False,whole_price_interval_proven=False)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for r in results:print(r['case'],r['unique_descriptors'],r['valid_elements_checked'],r['groups'][0])
