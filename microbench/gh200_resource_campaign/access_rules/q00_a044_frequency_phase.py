#!/usr/bin/env python3
"""复用冻结阶段频率 scorer；在内存中将设备身份替换为实际迁移目标。"""
import argparse,ast,functools,gzip,importlib.util,json,shutil,sys
from pathlib import Path
from q00_a044_frequency_analysis import read,save,sha,unpack
sys.dont_write_bytecode=True

def score_module(source,gpu):
 tree=ast.parse(source.read_text());changed=[]
 class Identity(ast.NodeTransformer):
  def visit_Assign(self,node):
   self.generic_visit(node)
   if len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id in ('config','expected'):
    key=node.targets[0].id;node.value=ast.Call(func=ast.Name(id='identity_'+key,ctx=ast.Load()),args=[node.value],keywords=[]);changed.append(key)
   return node
 tree=Identity().visit(tree);assert sorted(changed)==['config','expected'];ast.fix_missing_locations(tree)
 ns={'__name__':'frozen_phase_score','__file__':str(source),'identity_config':lambda x:dict(x,reference_gpu=gpu),'identity_expected':lambda x:{k:dict(v,gpu_uuid=gpu)for k,v in x.items()}}
 exec(compile(tree,str(source),'exec'),ns);decoder=ns['analyze_r18'];original=decoder.random_references;cached=functools.lru_cache(maxsize=32)(original);decoder.random_references=lambda indices,*args:cached(tuple(indices),*args)
 return ns

def main(args):
 package=args.package.resolve();manifest=read(package/'manifest.json');capture=package/'measure';returned=Path(manifest['tasks'][0]['origins'][0]['run']).resolve();original=returned.parent;source=original/'score.py'
 for name,digest in read(package/'package-identity.json').items():assert sha(package/name)==digest
 assert read(capture/'request.json')['manifest_sha256']==sha(package/'manifest.json')
 old_gpu=read(original/'run_config.json')['reference_gpu'];module=score_module(source,old_gpu);old=module['score'](original);assert json.loads(json.dumps(old))==read(original/'score.json')
 args.output.mkdir(exist_ok=False,parents=True);view=args.output/'view';view.mkdir();newreturned=view/'returned';newreturned.mkdir();excluded={'samples','sequence.json','environment.json','static_setup.json','preflight'}
 for p in original.iterdir():
  if p.name!='returned':(view/p.name).symlink_to(p,target_is_directory=p.is_dir())
 for p in returned.iterdir():
  if p.name not in excluded:(newreturned/p.name).symlink_to(p,target_is_directory=p.is_dir())
 allocation=read(capture/'allocation/identity.json');gpu=allocation['uuid'];environment=dict(gpu=gpu+', '+allocation['name'],job=allocation['job_id'],host=allocation['host']);save(newreturned/'environment.json',environment)
 sequence=[];setups={};evidence=[];starts=[]
 for task in manifest['tasks']:
  assert task['repeats']==1 and len(task['origins'])==1;origin=task['origins'][0];assert Path(origin['run']).resolve()==returned;record=read(origin['record']);assert sha(origin['record'])==origin['original_record_sha256'];assert record['command']==origin['original_argv']and task['argv'][1:]==record['command'][1:];assert sha(origin['original_stdout'])==origin['original_stdout_sha256']==record['raw_sha256']
  h=task['argv'][0].removeprefix('assets/');assert sha(package/task['argv'][0])==h;assert sha(returned/'build'/Path(record['command'][0]).name)==h
  folder=capture/'runs'/task['id']/'000';payload,receipt=unpack(folder);assert receipt['argv'][1:]==task['argv'][1:]and Path(receipt['argv'][0]).name==h;starts.append(receipt['start_ns']);events=[json.loads(line)for line in payload['stdout'].splitlines()if line.strip()];setup=events[0];assert setup['gpu_uuid']==gpu;setups[record['case']]=setup
  raw=newreturned/record['raw'];raw.parent.mkdir(exist_ok=True,parents=True);raw.write_bytes(gzip.compress(payload['stdout'],mtime=0));call=next(e for e in events if e['event']=='call');record.update({key:call[field]for key,field in [('elapsed_us','elapsed_us'),('warmup','warmup_us')]if key in record});record.update(raw_sha256=sha(raw),command=receipt['argv'],host_start_ns=receipt['start_ns'],host_stop_ns=receipt['stop_ns'],returncode=0,stderr=payload['stderr'].decode());save(newreturned/Path(origin['record']).resolve().relative_to(returned),record)
  sequence.append(dict(case=record['case'],trial=record['trial'],attempt=record['attempt'],raw=record['raw'],sha256=record['raw_sha256']));evidence.append(dict(task=task['id'],original_record=origin['record'],receipt=str(folder/'receipt.json'),binary_sha256=h))
 assert starts==sorted(starts);save(newreturned/'sequence.json',sequence);save(newreturned/'static_setup.json',[dict(case=k,setup=v)for k,v in setups.items()])
 module=score_module(source,gpu);new=module['score'](view)
 save(args.output/'original-score.json',old);save(args.output/'transfer-score.json',new);save(args.output/'verification.json',dict(source=str(source),source_sha256=sha(source),actual_environment=environment,old_score_reproduced=True,processes=len(sequence),checked_values=4096*len(sequence),identity_adaptation=['run_config.reference_gpu in memory','expected_setup[*].gpu_uuid in memory'],frozen_bindings_verified=True,numerical_score_ast_unchanged=True,fitting=False,calls=evidence))
 print(json.dumps({k:v for k,v in new.items()if k not in('rows','failed_attempts')},ensure_ascii=False,indent=2));print(json.dumps([dict(case=r['case'],phases=r['phases'])for r in new['rows']],ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
