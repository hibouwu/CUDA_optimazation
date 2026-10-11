#!/usr/bin/env python3
"""Bound endpoint slack for one varying rate shared across SMs.

Each SM has its own unknown clock epoch. No clock64 difference is taken across
SMs. The common-rate assumption is tested, not asserted. An observation-based
existence witness is never a free predictor or a physical clock diagnosis.
"""
import argparse
from bisect import bisect_left
from collections import deque
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics


def clock_graph(sm_points, endpoint_slack_ns, declared_rate_cap_ghz=1.98):
    """Exact difference constraints for C_sm(t)=offset_sm+F(t), 0<=F'<=cap.

    For a sample (C,N), t is in [N-epsilon,N+epsilon]. Continuity and
    monotonicity give F(N-epsilon)-Y_sm<=C and Y_sm-F(N+epsilon)<=-C,
    where Y_sm is minus the unknown counter offset. Adjacent time nodes impose
    0<=F(t_next)-F(t)<=cap*(t_next-t). Multiplying cycles by cap.denominator
    makes every bound an integer. Origins are subtracted before conversion.
    A None cap removes the upper-rate edges and uses unscaled integer cycles.
    """
    if not isinstance(endpoint_slack_ns,int) or isinstance(endpoint_slack_ns,bool) or endpoint_slack_ns<0:
        raise ValueError('nonnegative integer endpoint allowance required')
    if not sm_points or any(not points for points in sm_points.values()):
        raise ValueError('nonempty per-SM point sets required')
    for sm,points in sm_points.items():
        if not isinstance(sm,int) or isinstance(sm,bool):
            raise ValueError('integer SM identities required')
        if any(len(p)!=2 or any(not isinstance(v,int) or isinstance(v,bool) for v in p) for p in points):
            raise ValueError('integer clock/timer samples required')
    cap=None if declared_rate_cap_ghz is None else Fraction(str(declared_rate_cap_ghz))
    if cap is not None and cap<=0:
        raise ValueError('positive declared rate cap required')
    scale=1 if cap is None else cap.denominator
    timer_origin=min(n for points in sm_points.values() for c,n in points)
    clock_origins={sm:points[0][0] for sm,points in sm_points.items()}
    relative={sm:[(c-clock_origins[sm],n-timer_origin) for c,n in sm_points[sm]] for sm in sorted(sm_points)}
    times=sorted({n+delta for points in relative.values() for c,n in points
                  for delta in (-endpoint_slack_ns,endpoint_slack_ns)})
    time_nodes={n:i for i,n in enumerate(times)}
    sm_nodes={sm:len(times)+i for i,sm in enumerate(relative)}
    edges=[]
    for i in range(len(times)-1):
        if cap is not None:
            edges.append((i,i+1,cap.numerator*(times[i+1]-times[i])))
        edges.append((i+1,i,0))
    for sm,points in relative.items():
        for c,n in points:
            edges.append((sm_nodes[sm],time_nodes[n-endpoint_slack_ns],c*scale))
            edges.append((time_nodes[n+endpoint_slack_ns],sm_nodes[sm],-c*scale))
    return dict(endpoint_slack_ns=endpoint_slack_ns,cap_fraction=[cap.numerator,cap.denominator] if cap is not None else None,
                timer_origin_ns=timer_origin,clock_origins=clock_origins,time_nodes_ns=times,
                sm_nodes=sm_nodes,node_count=len(times)+len(relative),edges=edges)


def verify_certificate(graph, certificate):
    """Check an integer feasible potential or a strictly negative closed cycle."""
    edges=graph['edges']
    if certificate['feasible']:
        potential=certificate['potential_scaled_cycles']
        if len(potential)!=graph['node_count'] or any(not isinstance(v,int) for v in potential):
            raise ValueError('complete integer potential required')
        if any(potential[v]>potential[u]+bound for u,v,bound in edges):
            raise ValueError('potential violates a difference constraint')
    else:
        indices=certificate['negative_cycle_edge_ids']
        if not indices or any(not isinstance(i,int) or i<0 or i>=len(edges) for i in indices):
            raise ValueError('valid negative-cycle edges required')
        cycle=[edges[i] for i in indices]
        if any(a[1]!=b[0] for a,b in zip(cycle,cycle[1:]+cycle[:1])):
            raise ValueError('negative-cycle path is not closed')
        if sum(edge[2] for edge in cycle)>=0:
            raise ValueError('cycle is not strictly negative')
    return True


def _negative_cycle(parent, edges, vertex):
    visited={};path=[]
    while vertex not in visited:
        visited[vertex]=len(path)
        edge_id=parent[vertex]
        if edge_id<0:
            return None
        path.append(edge_id)
        vertex=edges[edge_id][0]
    indices=list(reversed(path[visited[vertex]:]))
    return indices if sum(edges[i][2] for i in indices)<0 else None


def solve_clock_graph(graph):
    """Queue relaxation; every return carries an independently checkable proof."""
    node_count=graph['node_count'];edges=graph['edges']
    adjacency=[[] for i in range(node_count)]
    for edge_id,(u,v,bound) in enumerate(edges):
        adjacency[u].append(edge_id)
    potential=[0]*node_count;parent=[-1]*node_count;path_length=[0]*node_count
    queue=deque(range(node_count));queued=[True]*node_count
    while queue:
        u=queue.popleft();queued[u]=False
        for edge_id in adjacency[u]:
            source,v,bound=edges[edge_id]
            if potential[v]<=potential[u]+bound:
                continue
            potential[v]=potential[u]+bound;parent[v]=edge_id
            path_length[v]=path_length[u]+1
            if path_length[v]>=node_count:
                cycle=_negative_cycle(parent,edges,v)
                if cycle is not None:
                    result=dict(feasible=False,negative_cycle_edge_ids=cycle,
                                negative_cycle_bound_scaled_cycles=sum(edges[i][2] for i in cycle))
                    verify_certificate(graph,result)
                    return result
            if not queued[v]:
                queue.append(v);queued[v]=True
    origin=potential[0]
    result=dict(feasible=True,potential_scaled_cycles=[value-origin for value in potential])
    verify_certificate(graph,result)
    return result


def verify_shared_witness(sm_points,graph,certificate):
    """Verify original sample brackets by exact inversion of the shared curve."""
    verify_certificate(graph,certificate)
    if not certificate['feasible']:
        raise ValueError('a feasible common curve is required')
    times=graph['time_nodes_ns'];potential=certificate['potential_scaled_cycles']
    curve=potential[:len(times)]
    cap_num,scale=graph['cap_fraction'] if graph['cap_fraction'] is not None else (None,1)
    sm_nodes={int(sm):node for sm,node in graph['sm_nodes'].items()}
    origins={int(sm):clock for sm,clock in graph['clock_origins'].items()}
    epsilon=graph['endpoint_slack_ns'];timer_origin=graph['timer_origin_ns']
    time_index={n:i for i,n in enumerate(times)}
    maximum_error=Fraction(0)
    for sm,points in sm_points.items():
        for c,n in points:
            relative_ns=n-timer_origin
            left=time_index[relative_ns-epsilon];right=time_index[relative_ns+epsilon]
            target=(c-origins[sm])*scale+potential[sm_nodes[sm]]
            if not curve[left]<=target<=curve[right]:
                raise ValueError('original sample is outside its curve bracket')
            index=bisect_left(curve,target,left,right+1)
            if curve[index]==target:
                adjusted=Fraction(times[index])
            else:
                adjusted=Fraction(times[index-1])+Fraction(target-curve[index-1],curve[index]-curve[index-1])*(times[index]-times[index-1])
            error=abs(adjusted-relative_ns)
            if error>epsilon:
                raise ValueError('common curve inversion exceeds endpoint allowance')
            maximum_error=max(maximum_error,error)
    for i in range(len(times)-1):
        difference=curve[i+1]-curve[i]
        if difference<0 or (cap_num is not None and difference>cap_num*(times[i+1]-times[i])):
            raise ValueError('common curve violates nonnegative capped rate')
    return dict(points_verified=sum(len(points) for points in sm_points.values()),
                maximum_constructed_endpoint_error_ns=float(maximum_error))


def minimum_shared_slack(sm_points, declared_rate_cap_ghz=1.98):
    """Smallest INTEGER epsilon; the real infimum is in (epsilon-1,epsilon]."""
    def solve(epsilon):
        graph=clock_graph(sm_points,epsilon,declared_rate_cap_ghz)
        return graph,solve_clock_graph(graph)
    zero_graph,zero=solve(0)
    if zero['feasible']:
        upper=0;upper_graph=zero_graph;upper_proof=zero
        lower_graph=None;lower_proof=None
    else:
        lower=0;lower_graph=zero_graph;lower_proof=zero;upper=1
        while True:
            upper_graph,upper_proof=solve(upper)
            if upper_proof['feasible']:
                break
            lower=upper;lower_graph=upper_graph;lower_proof=upper_proof;upper*=2
        while upper-lower>1:
            middle=(upper+lower)//2
            graph,proof=solve(middle)
            if proof['feasible']:
                upper=middle;upper_graph=graph;upper_proof=proof
            else:
                lower=middle;lower_graph=graph;lower_proof=proof
    verification=verify_shared_witness(sm_points,upper_graph,upper_proof)
    return dict(minimum_integer_endpoint_slack_ns=upper,original_sample_verification=verification,
                real_minimum_bracket_ns=[max(0,upper-1),upper],
                real_lower_bound_open=(upper>0),
                feasible_graph=upper_graph,feasible_certificate=upper_proof,
                infeasible_graph=lower_graph,infeasible_certificate=lower_proof)


def process_points(process):
    """Reconstruct integer samples from the previously verified local report."""
    sm_points={}
    for cta in process['ctas']:
        witness=cta['local_variable_rate_witness']
        cycles=witness['relative_cycles'];times=witness['original_relative_ns']
        if len(cycles)!=12 or len(times)!=12 or len(witness['point_order'])!=12:
            raise ValueError('all twelve original endpoints required')
        points=[(c+witness['clock_origin'],n+witness['timer_origin_ns']) for c,n in zip(cycles,times)]
        sm_points.setdefault(cta['sm'],[]).extend(points)
    if len(process['ctas'])!=132 or len(sm_points)!=132:
        raise ValueError('this audit requires all 132 one-CTA-per-SM records')
    return sm_points


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve()
    previous=base/'20261010-R09-R13-R18-failure-controls/reanalysis/A-local-clock-consistency-v1'
    source=previous/'local-clock-consistency.json';data=json.loads(source.read_text())
    cap_path=previous/'cap-evidence.json';cap_evidence=json.loads(cap_path.read_text())
    inputs={str(source):hashlib.sha256(source.read_bytes()).hexdigest(),
            str(cap_path):hashlib.sha256(cap_path.read_bytes()).hexdigest(),**data['input_sha256'],
            cap_evidence['source']:cap_evidence['sha256']}
    for path,digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('input identity changed: '+path)
    cap=cap_evidence['clock_intercept_cap_ghz']
    if cap!=data['declared_rate_cap_ghz'] or len(data['processes'])!=21:
        raise ValueError('declared cap or complete process set changed')
    processes=[]
    for process in data['processes']:
        points=process_points(process)
        proof=minimum_shared_slack(points,cap)
        processes.append(dict(case=process['case'],trial=process['trial'],stage=process['stage'],
            evidence=process['evidence'],independent_local_slack_maximum_ns=process['variable_rate_endpoint_slack_maximum_ns'],
            uncapped_shared=minimum_shared_slack(points,None),**proof))
    result=dict(schema='gh200-r09-shared-clock-consistency/1',
        scope='Observation-conditioned common-rate compatibility only; no fitted/predictive or physical clock rule.',
        model='C_sm(t)=offset_sm+F(t), 0<=F prime(t)<=declared cap, sample time in [N-epsilon,N+epsilon].',
        declared_rate_cap_ghz=cap,cap_scope=cap_evidence['use'],input_sha256=inputs,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),processes=processes,
        limits=['Clock origins are removed within each SM; offsets remain free. No cross-SM clock64 subtraction or shared counter epoch is assumed.',
                'A feasible potential defines a continuous piecewise-linear F through time nodes; sample inequalities guarantee a suitable time in each endpoint bracket.',
                'Negative closed cycles with strictly negative integer bound sums prove infeasibility. Both adjacent integer bounds are retained.',
                'Integer-ns bisection brackets a real-valued minimum; the lower bound is open unless zero is feasible.',
                'Endpoint allowance is a hypothetical per-endpoint budget, not a measured timer precision or a proved read enclosure.',
                'Arbitrary shared rate trajectories can change rapidly. This does not impose or identify physical DVFS dynamics.',
                'uncapped_shared removes the rate upper bound; do not attribute a capped-only rejection to common nonnegative rate alone.',
                'The common-rate assumption may fail because SM-local clock behavior or read-pairing effects differ. These causes remain unresolved.',
                'All inputs are development repeats or one correctness smoke. No new independent holdout, public model change, GPU work or qualification follows.'])
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'shared-clock-consistency.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for stage in (4,6):
        group=[p for p in processes if p['stage']==stage and p['evidence'].startswith('development')]
        values=[p['minimum_integer_endpoint_slack_ns'] for p in group]
        print('stage',stage,'minimum integer endpoint allowance median/range',statistics.median(values),min(values),max(values))
        uncapped=[p['uncapped_shared']['minimum_integer_endpoint_slack_ns'] for p in group]
        print('stage',stage,'uncapped common-rate median/range',statistics.median(uncapped),min(uncapped),max(uncapped))
    print('smoke',processes[-1]['minimum_integer_endpoint_slack_ns'])
    print('smoke uncapped',processes[-1]['uncapped_shared']['minimum_integer_endpoint_slack_ns'])


if __name__=='__main__':
    main()
