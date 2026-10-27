#!/usr/bin/env python3
"""原时钟包围和空闲干预评分的 a044 观测适配；冻结判据不变。"""
import argparse,gzip,hashlib,importlib.util,json,sys
from datetime import datetime
from pathlib import Path
from q00_a044_frequency_analysis import read,save,sha,unpack
sys.dont_write_bytecode=True

def main(args):
 package=args.package.resolve();manifest=read(package/'manifest.json');capture=package/'measure';original=Path(manifest['tasks'][0]['origins'][0]['run']).resolve();outer=original.parent
 for name,digest in read(package/'package-identity.json').items():assert sha(package/name)==digest
 assert read(capture/'request.json')['manifest_sha256']==sha(package/'manifest.json')
 sys.path.insert(0,str(original));spec=importlib.util.spec_from_file_location('original_score',original/'score.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 old=module.score(original);assert json.loads(json.dumps(old))==read(outer/'score.json')
 args.output.mkdir(parents=True,exist_ok=False);view=args.output/'view';view.mkdir();allocation=read(capture/'allocation/identity.json');frozen=int(datetime.fromisoformat(read(original/'binding.json')['frozen_at']).timestamp()*1e9);records={};evidence=[];starts=[]
 for task in manifest['tasks']:
  assert task['repeats']==1 and len(task['origins'])==1;origin=task['origins'][0];assert Path(origin['run']).resolve()==original;record=read(origin['record']);assert sha(Path(origin['record']))==origin['original_record_sha256'];assert task['argv'][1:]==record['command'][1:]==origin['original_argv'][1:];h=task['argv'][0].removeprefix('assets/');assert sha(package/task['argv'][0])==record['binary_sha256']==h;assert sha(Path(origin['original_stdout']))==record['raw_sha256']==origin['original_stdout_sha256']
  folder=capture/'runs'/task['id']/'000';payload,receipt=unpack(folder);assert receipt['argv'][1:]==task['argv'][1:]and Path(receipt['argv'][0]).name==h;assert receipt['start_ns']>frozen;starts.append(receipt['start_ns']);events=[json.loads(s)for s in payload['stdout'].decode().splitlines()if s.strip()];setup=next(e for e in events if e['event']=='setup');assert setup['gpu_uuid']==allocation['uuid']
  raw=view/record['raw'];raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(gzip.compress(payload['stdout'],mtime=0));call=next(e for e in events if e['event']=='call');record.update({key:call[field]for key,field in [('elapsed_us','elapsed_us'),('warmup','warmup_us')]if key in record});record.update(raw_sha256=sha(raw),host_start_ns=receipt['start_ns'],host_stop_ns=receipt['stop_ns'],command=receipt['argv'],returncode=0,stderr=payload['stderr'].decode());assert record['raw']not in records;records[record['raw']]=record;save(view/Path(origin['record']).resolve().relative_to(original),record);evidence.append(dict(task=task['id'],original_record=origin['record'],new_receipt=str(folder/'receipt.json'),binary_sha256=h,raw_sha256=record['raw_sha256']))
 assert starts==sorted(starts)
 decoder=module.measure if hasattr(module,'measure')else module.decode;original_replay=decoder.replay;used=[]
 def replay(root,record,row,plan):
  assert root==original;actual=records[record['raw']];used.append(record['raw']);return original_replay(view,actual,row,dict(plan,reference_gpu=allocation['uuid']))
 decoder.replay=replay
 new=module.score(original);assert len(used)==len(set(used))==len(records);assert new['successful_processes']==len(records)
 save(args.output/'original-score.json',old);save(args.output/'transfer-score.json',new);save(args.output/'verification.json',dict(source=str(original),actual_environment=allocation,old_score_reproduced=True,processes=len(records),original_score_sha256=sha(original/'score.py'),original_decode_sha256=sha(original/'decode.py'),original_plan_sha256=sha(original/'plan.json'),criterion_source='original prepared/binding files verified by original score; replay supplied with a044 observations and actual UUID',fitting=False,calls=evidence))
 print(json.dumps({k:v for k,v in new.items()if k not in['records','controls','contrasts']},ensure_ascii=False,indent=2));print(json.dumps([{k:v for k,v in r.items()if k not in ['low_phases','bounded_low_phases','pairs']}for r in new.get('controls',new.get('contrasts'))],ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
