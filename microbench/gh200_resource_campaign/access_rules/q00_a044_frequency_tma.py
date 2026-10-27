#!/usr/bin/env python3
"""复用 TMA 频率研究原统计与冻结判据；GPU 身份保留为实际迁移目标。"""
import argparse,ast,json,sys
from pathlib import Path
from q00_a044_frequency_analysis import read,save,sha,unpack
sys.dont_write_bytecode=True

def original_score(source,root,group,gpu):
 tree=ast.parse(source.read_text());changes=[]
 if group=='08':
  class Identity(ast.NodeTransformer):
   def visit_Assign(self,node):
    self.generic_visit(node)
    if len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='plan':node.value=ast.Call(func=ast.Name(id='identity_plan',ctx=ast.Load()),args=[node.value],keywords=[]);changes.append('plan.gpu_uuid')
    return node
  tree=Identity().visit(tree);assert changes==['plan.gpu_uuid']
 else:
  for node in tree.body:
   if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='root':node.value=ast.Name(id='SOURCE_ROOT',ctx=ast.Load());changes.append('root')
  original=len(tree.body);tree.body=[n for n in tree.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='write_text')];assert changes==['root']and len(tree.body)==original-1
 ast.fix_missing_locations(tree);ns={'__name__':'original_tma_score','__file__':str(source),'SOURCE_ROOT':root,'identity_plan':lambda x:dict(x,gpu_uuid=gpu)};exec(compile(tree,str(source),'exec'),ns)
 return ns['score'](root)if group=='08'else ns['report']

def main(args):
 package=args.package.resolve();manifest=read(package/'manifest.json');returned=Path(manifest['tasks'][0]['origins'][0]['run']).resolve();original=returned.parent;group=args.group;source=original/('score-replay.py'if group=='08'else'analyze.py');baseline=original/('score.json'if group=='08'else'analysis.json');oldgpu=read(manifest['tasks'][0]['origins'][0]['original_stdout'])['gpu_uuid']
 old=original_score(source,original,group,oldgpu);assert json.loads(json.dumps(old))==read(baseline)
 if args.verify_original:print(json.dumps(dict(group=group,old_score_reproduced=True,processes=len(manifest['tasks']),source_sha256=sha(source))));return
 capture=package/'measure'
 for name,digest in read(package/'package-identity.json').items():assert sha(package/name)==digest
 assert read(capture/'request.json')['manifest_sha256']==sha(package/'manifest.json');args.output.mkdir(exist_ok=False,parents=True);view=args.output/'view';view.mkdir();newreturned=view/'returned';newreturned.mkdir();(newreturned/'samples').mkdir();excluded={'samples','invocations.json','preflight'}
 for p in original.iterdir():
  if p.name!='returned':(view/p.name).symlink_to(p,target_is_directory=p.is_dir())
 for p in returned.iterdir():
  if p.name not in excluded:(newreturned/p.name).symlink_to(p,target_is_directory=p.is_dir())
 allocation=read(capture/'allocation/identity.json');gpu=allocation['uuid'];calls=[];starts=[];evidence=[]
 for task in manifest['tasks']:
  assert task['repeats']==1 and len(task['origins'])==1;origin=task['origins'][0];assert Path(origin['run']).resolve()==returned;assert sha(origin['record'])==origin['original_record_sha256'];record=read(origin['record'])[origin['record_index']];assert record['command']==origin['original_argv']and task['argv'][1:]==record['command'][1:];assert sha(origin['original_stdout'])==origin['original_stdout_sha256']==record['sha256']
  h=task['argv'][0].removeprefix('assets/');assert sha(package/task['argv'][0])==h;assert sha(returned/'build'/Path(record['command'][0]).name)==h
  folder=capture/'runs'/task['id']/'000';payload,receipt=unpack(folder);assert receipt['argv'][1:]==task['argv'][1:]and Path(receipt['argv'][0]).name==h;starts.append(receipt['start_ns']);row=json.loads(payload['stdout']);assert row['gpu_uuid']==gpu
  oldrow=read(origin['original_stdout'])
  for key in ['ctas','tma_ctas','active_groups','groups','ring','background','trace','checked']:assert row[key]==oldrow[key],key
  raw=newreturned/'samples'/Path(origin['original_stdout']).name;raw.write_bytes(payload['stdout']);record.update(sha256=sha(raw),command=receipt['argv'],started_ns=receipt['start_ns'],ended_ns=receipt['stop_ns'],returncode=0);calls.append(record);evidence.append(dict(task=task['id'],original_record=origin['record'],record_index=origin['record_index'],receipt=str(folder/'receipt.json'),binary_sha256=h))
 assert starts==sorted(starts);save(newreturned/'invocations.json',calls);new=original_score(source,view,group,gpu)
 verification=dict(source=str(source),source_sha256=sha(source),actual_environment=allocation,old_score_reproduced=True,processes=len(calls),identity_adaptation=['plan.gpu_uuid in memory']if group=='08'else[],frozen_bindings_verified=group=='08',numerical_score_ast_unchanged=True,fitting=False,native_validation='sampling verify=0; new preflight checks are scored separately',calls=evidence)
 save(args.output/'original-score.json',old);save(args.output/'transfer-score.json',new);save(args.output/'verification.json',verification);print(json.dumps({k:v for k,v in new.items()if k not in('summaries','processes','rows')},ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--group',choices=['08','09'],required=True);p.add_argument('--verify-original',action='store_true');main(p.parse_args())
