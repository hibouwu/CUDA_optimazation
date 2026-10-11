"""Independent envelope-extrema and explicit cohort upper-bound recursion."""
from fractions import Fraction as F
import hashlib,itertools,json
from pathlib import Path
root=Path(__file__).resolve().parent
source=root/'B-report-snapshot.json'
r=json.loads(source.read_text());p=r['parameters']
q=lambda x:F.from_float(x) if isinstance(x,float) else F(x)
lower,upper=map(F,(r['price_interval']['lower'],r['price_interval']['upper']))
env={(e['cta'],e['group']):[(F(l['slope']),F(l['intercept'])) for l in e['lines']]
    for e in r['initial_envelopes']}
def value(lines,x):return max(a*x+b for a,b in lines)
def crossings(lines):
    points={lower,upper}
    for (a,b),(c,d) in itertools.combinations(lines,2):
        if a!=c:
            x=(d-b)/(a-c)
            if lower<=x<=upper:points.add(x)
    return points
checks=0;margins=[]
for row in r['initial_margins']:
    c,g=row['cta'],row['group'];first=env[c,0];later=env[c,g]
    # The difference of upper affine envelopes is affine between every
    # breakpoint of either envelope. Its extrema lie at these endpoints.
    points=crossings(first)|crossings(later)
    worst=max(value(later,x)-value(first,x) for x in points)
    margin=g*q(p['compute_ns'])-worst
    assert margin==F(row['minimum_TC_slack']['exact']) and margin>=0
    checks+=len(points);margins.append(margin)
assert all(a>=0 for lines in env.values() for a,b in lines)
fmin=min(value(env[c,0],lower) for c in range(126))
fmax=max(value(env[c,0],upper) for c in range(126))
assert fmin==F(r['first_min']['exact']) and fmax==F(r['first_max']['exact'])
compute=q(p['compute_ns']);lead=q(p['commit_lead_ns']);operand=q(p['operand_gap_ns'])
producer=q(p['producer_gap_ns']);latency=q(p['read_latency_ns']);destination=2*q(p['destination_service_ns'])
static=json.loads((root.parent/'main-single126-source-union-v1/report.json').read_text())
unions=next(x for x in static['results'] if x['case']==r['case'])
source_work=max(g['combined_units'] for g in unions['groups'])*q(p['service_ns']/128)
path_work=252*upper
assert source_work==F(r['source_cohort_service_upper']['exact'])
span=fmax-fmin+2*operand
assert span<latency and span<compute and 2*operand+producer<compute
first_issue_min=fmin+2*compute-lead
# All six initial cohorts issue before any source can expire. The A batch
# alone keeps each server busy until B1 arrives, and each entire cohort keeps
# it busy until the next A batch. Thus both prefix ends are total service,
# starting at zero; no reported prefix end is used to initialize recursion.
assert 5*(2*operand+producer)+2*operand<latency
assert len({g['combined_units'] for g in unions['groups']})==1
source_A=unions['groups'][0]['A_units']*q(p['service_ns']/128)
assert source_A>2*operand and source_work>2*operand+producer
assert 126*lower>2*operand and 252*lower>2*operand+producer
source_end=6*source_work
path_end=6*path_work
assert first_issue_min-source_end==F(r['bounds']['initial_source_cleared_ns']['exact'])
assert first_issue_min-path_end==F(r['bounds']['initial_request_cleared_ns']['exact'])
assert source_end<first_issue_min and path_end<first_issue_min
dest_end=max(value(env[c,5],upper) for c in range(126))
assert first_issue_min>5*(2*operand+producer)+2*operand+producer
rows=[]
for group in range(6,16):
    issue_min=fmin+(group-6+2)*compute-lead
    issue_max=fmax+(group-6+2)*compute-lead+2*operand
    source_end=max(source_end,issue_max)+source_work
    path_end=max(path_end,issue_max)+path_work
    admission=max(source_end,path_end)+latency
    dest_end=max(dest_end,admission)+destination
    deadline=fmin+group*compute
    assert dest_end<deadline
    rows.append(dict(group=group,latest_issue=str(issue_max),
        source_complete_upper=str(source_end),path_complete_upper=str(path_end),
        destination_complete_upper=str(dest_end),earliest_compute=str(deadline),
        conservative_margin_ns=str(deadline-dest_end)))
assert fmin+16*compute>F(rows[-1]['latest_issue'])
report=dict(B_snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    initial_margin_checks=len(margins),envelope_breakpoint_checks=checks,
    initial_minimum_margin_ns=float(min(margins)),cohorts=rows,
    minimum_refill_margin_ns=float(min(F(x['conservative_margin_ns']) for x in rows)),
    source_union_static_reference_sha256=hashlib.sha256((root.parent/'main-single126-source-union-v1/report.json').read_bytes()).hexdigest(),
    partial_audit_passed=True,full_certificate_accepted=False,
    initial_source_and_request_ends_independently_derived=True,
    remaining='Bind final delivered code and complete initial affine-envelope construction; extrema and later-cohort recursion are independently checked here.',
    scope='Exact algebra and explicit later-cohort recursion for the supplied rational envelopes, not hardware validation.')
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cohorts'},indent=2))
