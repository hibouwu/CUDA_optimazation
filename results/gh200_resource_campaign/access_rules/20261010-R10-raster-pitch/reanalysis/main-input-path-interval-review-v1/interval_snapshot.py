#!/usr/bin/env python3
"""Certify first-output windows on open price cells of the existing supply graph."""
import argparse
import ast
from fractions import Fraction as F
import hashlib
import inspect
import json
from pathlib import Path

import supply_model
from analyze_r10_granule_overlap import source_granules


def rational(value):
    return F.from_float(value) if isinstance(value,float) else F(value)


class PriceCell:
    """Intersection of the exact comparison signs taken at one rational price."""
    def __init__(self,point,lower,upper):
        self.point=point;self.lower=lower;self.upper=upper;self.comparisons=0
        self.point_constraints=0;self.limiting_comparisons={}

    def sign(self,slope,intercept):
        value=slope*self.point+intercept
        if not slope:return (value>0)-(value<0)
        self.comparisons+=1
        root=-intercept/slope
        if not value:
            self.lower=max(self.lower,root);self.upper=min(self.upper,root)
            self.point_constraints+=1
        elif (value>0)==(slope>0):
            if root>self.lower:
                self.lower=root;self.record_limit('lower',slope,intercept)
        elif root<self.upper:
            self.upper=root;self.record_limit('upper',slope,intercept)
        if not self.lower<=self.point<=self.upper:raise ValueError('invalid affine branch cell')
        return (value>0)-(value<0)

    def record_limit(self,bound,slope,intercept):
        frame=inspect.currentframe()
        while frame and frame.f_code.co_filename!='<exact existing first-output graph>':frame=frame.f_back
        self.limiting_comparisons[bound]=dict(slope=str(slope),intercept=str(intercept),
            graph_line=frame.f_lineno if frame else None)


class Affine:
    def __init__(self,cell,slope=0,intercept=0):
        self.cell=cell;self.slope=rational(slope);self.intercept=rational(intercept)

    def pair(self,other):
        if isinstance(other,Affine):
            if self.cell is not other.cell:raise ValueError('different price cells')
            return other.slope,other.intercept
        return F(0),rational(other)

    def __add__(self,other):
        slope,intercept=self.pair(other)
        return Affine(self.cell,self.slope+slope,self.intercept+intercept)
    __radd__=__add__

    def __neg__(self):return Affine(self.cell,-self.slope,-self.intercept)
    def __sub__(self,other):return self+-self.convert(other)
    def __rsub__(self,other):return self.convert(other)+-self

    def convert(self,other):
        slope,intercept=self.pair(other)
        return Affine(self.cell,slope,intercept)

    def __mul__(self,other):
        slope,intercept=self.pair(other)
        if self.slope and slope:raise ValueError('non-affine timing expression')
        return Affine(self.cell,self.slope*intercept+slope*self.intercept,self.intercept*intercept)
    __rmul__=__mul__

    def __truediv__(self,other):
        slope,intercept=self.pair(other)
        if slope:raise ValueError('nonconstant timing divisor')
        return Affine(self.cell,self.slope/intercept,self.intercept/intercept)

    def compare(self,other):
        slope,intercept=self.pair(other)
        return self.cell.sign(self.slope-slope,self.intercept-intercept)
    def __lt__(self,other):return self.compare(other)<0
    def __le__(self,other):return self.compare(other)<=0
    def __gt__(self,other):return self.compare(other)>0
    def __ge__(self,other):return self.compare(other)>=0
    def __eq__(self,other):
        if other is None:return False
        return self.compare(other)==0
    def __bool__(self):return self.compare(0)!=0
    def __float__(self):return float(self.slope*self.cell.point+self.intercept)


class ExactFirstOutput(ast.NodeTransformer):
    """Keep the existing graph; change arithmetic and stop at the last first MAIN."""
    def __init__(self):self.type_guards=0;self.stop_guards=0;self.float_constants=0

    def visit_Constant(self,node):
        if type(node.value) is float:
            self.float_constants+=1
            numerator,denominator=node.value.as_integer_ratio()
            return ast.copy_location(ast.Call(func=ast.Name(id='F',ctx=ast.Load()),
                args=[ast.Constant(numerator),ast.Constant(denominator)],keywords=[]),node)
        return node

    def visit_Tuple(self,node):
        self.generic_visit(node)
        if len(node.elts)==2 and all(isinstance(x,ast.Name) for x in node.elts) and [x.id for x in node.elts]==['int','float']:
            node.elts += [ast.Name(id='F',ctx=ast.Load()),ast.Name(id='Affine',ctx=ast.Load())]
            self.type_guards+=1
        return node

    def visit_If(self,node):
        self.generic_visit(node)
        if ast.dump(node.test)==ast.dump(ast.parse("kind=='main_end'",mode='eval').body):
            node.body += ast.parse("if all(actor['main_ends'][0] is not None for actor in actors):\n    return actors").body
            self.stop_guards+=1
        return node


def exact_first_graph():
    source=inspect.getsource(supply_model.shared_stage_events)
    tree=ast.parse(source);transform=ExactFirstOutput();tree=transform.visit(tree)
    if transform.type_guards!=5 or transform.stop_guards!=1:
        raise ValueError('existing graph validation/first-MAIN structure changed')
    ast.fix_missing_locations(tree)
    namespace=dict(vars(supply_model),F=F,Affine=Affine)
    exec(compile(tree,'<exact existing first-output graph>','exec'),namespace)
    return namespace['shared_stage_events'],dict(type_guards=transform.type_guards,
        stop_guards=transform.stop_guards,float_constants=transform.float_constants,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        scope='Same Python graph and branches; binary float constants become exact rationals, timing becomes affine; stop after all first MAIN events.')


def coefficients(value):
    return (value.slope,value.intercept) if isinstance(value,Affine) else (F(0),rational(value))


def first_certificate(graph,point,lower,upper,kt,stages,count,args):
    cell=PriceCell(point,lower,upper)
    exact_args={name:rational(value) if type(value) is float else value for name,value in args.items()}
    exact_args['input_path_service_ns']=Affine(cell,1)
    actors=graph(kt,stages,count,**exact_args)
    windows=[];slacks=[]
    for cta,actor in enumerate(actors):
        slope,intercept=coefficients(actor['main_ends'][0]-actor['starts'][0])
        windows.append(dict(cta=cta,slope=str(slope),intercept=str(intercept)))
        for i in range(1,kt):
            a,b=coefficients(actor['service_frontier'][i-1]-actor['starts'][i])
            slacks.append(min(a*cell.lower+b,a*cell.upper+b))
    return dict(point=str(point),lower=str(cell.lower),upper=str(cell.upper),
        comparisons=cell.comparisons,point_constraints=cell.point_constraints,
        limiting_comparisons=cell.limiting_comparisons,
        first_windows=windows,minimum_TC_slack_ns=float(min(slacks))),actors


def analyze(root,output,maximum_cells):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def read(path):
        inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    previous=read(root/'20261010-R10-raster-pitch/reanalysis/B-20261011-input-path-v1/B-input-path.json')
    prefix=read(root/'20261010-R10-producer-delay/reanalysis/B-20261011-shared-path-prefix-v1/B-shared-path-prefix.json')
    original=read(root/'20261010-R10-raster-pitch/reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json')
    selection=previous['selection'];lower=F(selection['lower_exact']);upper=F(selection['upper_exact'])
    context=original['contexts']['r10_source_single126_p32'];row,setup=context['row'],context['setup']
    program=supply_model.cfg_b_source_program(row,setup);kt=row['k']//64
    if program['output_counts']!=[1]*126 or setup['stages']!=6:raise ValueError('single126 domain changed')
    keys={key for groups in program['source_keys'] for group in groups for key in group}
    units={key:tuple((key[0],i) for i in source_granules(key,128)) for key in keys}
    params=previous['parameters']
    if params!=prefix['parameters']:raise ValueError('prefix tuple changed')
    args=dict(source_keys=program['source_keys'],output_counts=program['output_counts'],
        share_inflight_sources=True,source_address_units=units,source_unit_service_ns=params['service_ns']/128,
        input_min_latency_ns=params['read_latency_ns'],**params)
    graph,transformation=exact_first_graph()
    remaining=[(lower,upper)];cells=[];float_checks=[]
    while remaining and len(cells)<maximum_cells:
        left,right=remaining.pop();point=(left+right)/2
        certificate,actors=first_certificate(graph,point,lower,upper,kt,6,126,args)
        a,b=F(certificate['lower']),F(certificate['upper'])
        if a==b:raise ValueError('sample price lies on a branch boundary; singleton requires a separate evaluation')
        if a>left:remaining.append((left,a))
        if b<right:remaining.append((b,right))
        # Numeric parity checks are separate evidence, not the interval proof.
        numeric=supply_model.shared_stage_events(kt,6,126,input_path_service_ns=float(point),**args)
        maximum_error=max(abs(float(actors[c][field][index])-numeric['ctas'][c][field][index])
            for c in range(126) for field,index in (('starts',0),('main_ends',0)))
        if maximum_error>1e-7:raise ValueError('exact/float first endpoints differ')
        float_checks.append(dict(point=str(point),maximum_first_endpoint_error_ns=maximum_error))
        cells.append(certificate)
        print(json.dumps(dict(cell=len(cells),lower=float(a),upper=float(b),remaining=len(remaining),
            first_L_ns=certificate['first_windows'][0],minimum_TC_slack_ns=certificate['minimum_TC_slack_ns'])),flush=True)
    for name in ('supply_model.py','analyze_r10_input_path_interval.py','analyze_r10_granule_overlap.py'):
        path=Path(__file__).with_name(name);inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(scope='Exact-arithmetic branch cells for single126 B32 first outputs; no fitted service or new GPU evidence.',
        price_interval=dict(lower=str(lower),upper=str(upper)),transformation=transformation,
        cells=cells,remaining_open_intervals=[dict(lower=str(a),upper=str(b)) for a,b in remaining],
        floating_point_checks=float_checks,open_interval_covered=not remaining,
        boundary_points_require_separate_check=True,input_sha256=inputs,
        limits=['A branch cell records every nonconstant affine comparison sign, including heap time ordering and source subscription.',
            'Cells have open endpoints; all endpoints require explicit execution and are not covered by neighboring sign certificates.',
            'The proof uses exact arithmetic on binary input constants; numeric parity is checked separately for sampled original float graphs.',
            'Only the existing single126 B32 first-output domain and the earlier necessary interval are analyzed.'])
    (output/'B-input-path-interval.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-cells',type=int,default=32)
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve(),args.maximum_cells)
