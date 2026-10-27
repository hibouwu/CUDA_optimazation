#!/usr/bin/env python3
"""用原匹配窗口与冻结预测检查 a044 迁移；只读取已回传数据，不拟合。"""
import argparse, functools, gzip, hashlib, importlib.util, json, shutil, sys, tarfile
from pathlib import Path
sys.dont_write_bytecode = True

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def original_modules(root, expected_decoder_sha256):
 candidates=[root/'source/analyze_r18.py',Path('results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-combination-regression-001/source/analyze_r18.py')]
 decoder_source=next(p for p in candidates if sha(p)==expected_decoder_sha256)
 sys.path.insert(0,str(decoder_source.parent))
 import analyze_r18
 # 缓存同一输入的原参考值，函数和数值运算保持原实现。
 original=analyze_r18.random_references
 cached=functools.lru_cache(maxsize=64)(original)
 analyze_r18.random_references=lambda indices,*args:cached(tuple(indices),*args)
 helper=load('analyze_frequency_development',Path(__file__).with_name('analyze_frequency_development.py'))
 return helper,analyze_r18

def unpack(folder):
 index=read(folder/'output-archive.json');archive=folder/index['archive'];assert sha(archive)==index['sha256'];payload={}
 with tarfile.open(archive)as tar:
  for member in tar:
   data=tar.extractfile(member).read();expected=index['members'][member.name];assert len(data)==expected['size']and hashlib.sha256(data).hexdigest()==expected['sha256'];payload[member.name]=data
 assert set(payload)==set(index['members']);receipt=read(folder/'receipt.json')
 assert receipt['returncode']==0 and not receipt['timed_out']
 for name in ['stdout','stderr']:assert hashlib.sha256(payload[name]).hexdigest()==receipt[name+'_sha256']
 return payload,receipt

def main(args):
 package=args.package.resolve();manifest=read(package/'manifest.json');capture=package/'measure';original=Path(manifest['tasks'][0]['origins'][0]['run']).resolve();outer=original.parent
 assert read(capture/'request.json')['manifest_sha256']==sha(package/'manifest.json')
 for name,digest in read(package/'package-identity.json').items():assert sha(package/name)==digest
 baseline_file=outer/('summary.json'if(outer/'summary.json').exists()else'analysis.json'if(outer/'analysis.json').exists()else'score.json');baseline=read(baseline_file);baseline=baseline.get('observations',baseline)
 analysis,decoder=original_modules(original,baseline['decoder_sha256'])
 old=analysis.analyze(original)
 assert old['rows']==baseline['rows'],str(baseline_file)
 args.output.mkdir(parents=True,exist_ok=False);view=args.output/'view';view.mkdir()
 for name in ['cases.json','source_hashes.json']:shutil.copy2(original/name,view/name)
 for name in ['source','build']:(view/name).symlink_to(original/name,target_is_directory=True)
 allocation=read(capture/'allocation/identity.json');gpu=allocation['uuid'];config=read(original/'run_config.json');config['reference_gpu']=gpu;config['remeasurement_source']=str(original);save(view/'run_config.json',config)
 environment=dict(gpu=gpu+', '+allocation['name'],job=allocation['job_id'],host=allocation['host'],measurement_kind='a044 transfer of original protocol');save(view/'environment.json',environment)
 oldsetups={r['case']:r['setup']for r in read(original/'static_setup.json')};setups={};starts=[];evidence=[]
 for task in manifest['tasks']:
  assert task['repeats']==1 and len(task['origins'])==1;origin=task['origins'][0];assert Path(origin['run']).resolve()==original;record=read(origin['record']);assert sha(Path(origin['record']))==origin['original_record_sha256'];assert record['command']==origin['original_argv'];assert task['argv'][1:]==record['command'][1:]
  h=task['argv'][0].removeprefix('assets/');assert sha(package/task['argv'][0])==h;assert record.get('binary_sha256',h)==h
  oldraw=Path(origin['original_stdout']);assert sha(oldraw)==origin['original_stdout_sha256']==record['raw_sha256']
  folder=capture/'runs'/task['id']/'000';payload,receipt=unpack(folder);assert receipt['argv'][1:]==task['argv'][1:];assert Path(receipt['argv'][0]).name==h;starts.append(receipt['start_ns'])
  events=[json.loads(line)for line in payload['stdout'].decode().splitlines()if line.strip()];setup=next(e for e in events if e['event']=='setup');assert setup['gpu_uuid']==gpu
  key=record['case'];oldsetup=oldsetups[key]
  for field in ['grid','tile','cluster','stages','threads','smem','smem_epilogue','max_active_ctas_per_sm','sm_count','requested_sm_count','scheduler_sm_count']:
   assert setup[field]==oldsetup[field],(key,field)
  if key not in setups or record['variant']=='plain':setups[key]=setup
  raw=view/record['raw'];raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(gzip.compress(payload['stdout'],mtime=0))
  call=next(e for e in events if e['event']=='call');record.update({key:call[field] for key,field in [('elapsed_us','elapsed_us'),('warmup','warmup_us')]if key in record})
  record.update(raw_sha256=sha(raw),host_start_ns=receipt['start_ns'],host_stop_ns=receipt['stop_ns'],command=receipt['argv'],returncode=receipt['returncode'],stderr=payload['stderr'].decode())
  dest=view/Path(origin['record']).resolve().relative_to(original);save(dest,record)
  evidence.append(dict(task=task['id'],original_record=origin['record'],new_receipt=str(folder/'receipt.json'),binary_sha256=h,new_stdout_sha256=receipt['stdout_sha256']))
 assert starts==sorted(starts);save(view/'static_setup.json',[dict(case=c['id'],setup=setups[c['id']])for c in read(view/'cases.json')])
 observations=analysis.analyze(view);save(args.output/'observations.json',observations)
 score=None;oldscore=None
 if(outer/'score.json').exists()and'predictions.json'in {p.name for p in original.iterdir()}:
  score_name='score_cfg_b_time.py'if'cfg-b-time'in outer.name else'score_frequency_heldout.py';scorer=load('frozen_score',Path(__file__).with_name(score_name));scorer.analyze=lambda root:old;oldscore=scorer.score(original)
  expected=read(outer/'score.json');assert {k:v for k,v in oldscore.items()if k!='observations'}=={k:v for k,v in expected.items()if k!='observations'}
  # 原 scorer 校验原 binding/预测文件，观测依赖替换为已通过原 oracle 的新卡观测。
  scorer.analyze=lambda root:observations;score=scorer.score(original);save(args.output/'frozen-transfer-score.json',score)
 report=dict(original_root=str(original),actual_environment=environment,original_environment=old['environment'],original_observations_reproduced=True,original_score_reproduced=oldscore is not None,matching_window_analyzer=str(Path(analysis.__file__)),matching_window_analyzer_sha256=sha(Path(analysis.__file__)),frozen_decoder=str(Path(decoder.__file__)),frozen_decoder_sha256=sha(Path(decoder.__file__)),processes=len(evidence),checked_values=4096*len(evidence),comparison_kind='new GPU transfer; original development/heldout roles retained; no refitting',calls=evidence)
 save(args.output/'verification.json',report);save(args.output/'original-observations.json',old)
 if oldscore:save(args.output/'original-score.json',oldscore)
 print(json.dumps(dict(processes=len(evidence),cases=len(observations['rows']),old_fields_reproduced=True,score={k:v for k,v in (score or {}).items()if k not in['observations','rows']}),ensure_ascii=False,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
