from pathlib import Path
import importlib.util,json,subprocess,sys,hashlib
out=Path(__file__).resolve().parent
repo=out.parents[5]
modules=repo/'microbench/gh200_resource_campaign/access_rules'
sys.path.insert(0,str(modules))
import analyze_r10_input_latency as analysis
old_source=subprocess.check_output(['git','show','6dce665:microbench/gh200_resource_campaign/access_rules/supply_model.py'],cwd=repo)
(out/'previous-supply_model.py').write_bytes(old_source)
spec=importlib.util.spec_from_file_location('previous_supply',out/'previous-supply_model.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
current=analysis.shared_stage_events
checks=[]
def checked(*args,**kwargs):
 default=dict(kwargs);default.pop('input_min_latency_ns')
 before=old.shared_stage_events(*args,**default)
 after=current(*args,**default)
 assert before==after
 checks.append(dict(demands=after['input_demand_count'],whole_dictionary_equal=True))
 return current(*args,**kwargs)
analysis.shared_stage_events=checked
analysis.analyze(out.parents[2],out/'rerun')
(out/'default-regression.json').write_text(json.dumps(dict(previous_commit='6dce665',previous_source_sha256=hashlib.sha256(old_source).hexdigest(),checks=checks,total_demands=sum(c['demands'] for c in checks)),indent=2)+'\n')
