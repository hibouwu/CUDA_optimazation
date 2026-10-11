#!/usr/bin/env python3
"""Bind the frozen clock functions to the current main clock module read-only."""
import ast,hashlib,json
from pathlib import Path

base=Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation')
current=base/'microbench/gh200_resource_campaign/access_rules/clock_model.py'
frozen=base/'results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/source/r09_input_source_clock.py'
duration=frozen.parent/'r09_v08_clock.py'
def function(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
class QualifiedDuration(ast.NodeTransformer):
    def visit_Attribute(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='clock' and node.attr=='duration_term':
            return ast.Name(id='duration_term',ctx=ast.Load())
        return node
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
checks=dict(solver_AST_identical=same(function(current,'solve_source_envelope'),function(frozen,'solve_source_envelope')),
    clock_mapping_AST_after_import_qualification=same(function(current,'source_frequency'),QualifiedDuration().visit(function(frozen,'source_frequency'))),
    duration_AST_identical=same(function(current,'duration_term'),function(duration,'duration_term')))
assert all(checks.values())
result=dict(checks=checks,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (current,frozen,duration)},
    scope='Read-only current main clock binding; only clock.duration_term import qualification normalized. Does not bind later changes to a caller envelope.')
(Path(__file__).resolve().parent/'current-clock-binding.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(checks,indent=2))
