"""Independent rational sign and equal-time heap audit of B's source snapshot."""
from fractions import Fraction as F
import hashlib,heapq,importlib.util,itertools,json,operator,sys
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
repo=root.parents[5]
sys.path.insert(0,str(repo/'microbench/gh200_resource_campaign/access_rules'))
path=root/'interval_snapshot.py'
spec=importlib.util.spec_from_file_location('interval_snapshot',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checks=0;singletons=0;interior_checks=0
for slope,intercept,point,op in itertools.product(
        (-3,-1,0,2,4),(-2,0,1,3),(F(1,3),F(1),F(3,2)),
        (operator.lt,operator.le,operator.eq,operator.ge,operator.gt)):
    cell=module.PriceCell(point,F(0),F(2))
    expression=module.Affine(cell,slope,intercept)
    actual=op(expression,0)
    assert actual==op(slope*point+intercept,0)
    checks+=1
    if cell.lower==cell.upper:
        assert slope*point+intercept==0 and slope!=0
        singletons+=1
    else:
        for k in range(1,10):
            p=cell.lower+(cell.upper-cell.lower)*F(k,10)
            assert op(slope*p+intercept,0)==actual
            interior_checks+=1

# Equal event times must use the event serial, even when the two expressions
# differ away from that price. Both input orders are tested.
heap_checks=0
for point in (F(1,3),F(1),F(3,2)):
    for reverse in (False,True):
        cell=module.PriceCell(point,F(0),F(2))
        price=module.Affine(cell,1)
        events=[(price,7,'late_serial'),(F(point),2,'early_serial')]
        if reverse:events.reverse()
        heapq.heapify(events)
        assert heapq.heappop(events)[2]=='early_serial'
        assert cell.lower==cell.upper==point
        heap_checks+=1
result=dict(snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    rational_comparisons=checks,strict_interior_checks=interior_checks,
    equality_singletons=singletons,equal_time_heap_checks=heap_checks,
    scope='Primitive sign/heap review of an uncommitted snapshot only; not full graph coverage or an interval certificate.',
    whole_price_interval_proven=False)
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
