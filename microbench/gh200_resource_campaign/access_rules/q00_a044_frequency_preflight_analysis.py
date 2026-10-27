#!/usr/bin/env python3
"""检查本轮频率原生预检与 Compute Sanitizer 输出；和采样评分分开归档。"""
import argparse,importlib.util,json,sys
from pathlib import Path
from q00_a044_frequency_analysis import read,save,sha,unpack
sys.dont_write_bytecode=True

def main(args):
 package=args.package.resolve();capture=package/'measure';manifest=read(package/'manifest.json');assert read(capture/'request.json')['manifest_sha256']==sha(package/'manifest.json')
 for name,digest in read(package/'package-identity.json').items():assert sha(package/name)==digest
 sys.path.insert(0,str(package));spec=importlib.util.spec_from_file_location('preflight_runner',package/'q00_a044_remeasure.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 allocation=read(capture/'allocation/identity.json');rows=[];starts=[]
 for task in manifest['tasks']:
  assert task['repeats']==1 and len(task['origins'])==1;origin=task['origins'][0];assert sha(origin['original_source'])==origin['source_sha256'];assert sha(origin['original_stdout'])==origin['original_stdout_sha256'];assert task['timeout_seconds']==origin['original_timeout_seconds']
  expected=[('assets/'+origin['source_binary_sha256']if x=='build/probe'else x)for x in origin['original_argv']];assert expected==task['argv']
  folder=capture/'runs'/task['id']/'000';payload,receipt=unpack(folder);actual=[('assets/'+Path(x).name if Path(x).name==origin['source_binary_sha256']else x)for x in receipt['argv']];assert actual==expected
  assert sha(package/'assets'/origin['source_binary_sha256'])==origin['source_binary_sha256'];assert task['protocol']['expected_gpu_uuid']==allocation['uuid'];result=module.inspect_preflight(module.Text(payload['stdout'].decode()),task['protocol']);assert result['valid'];starts.append(receipt['start_ns']);rows.append(dict(task=task['id'],source=origin['run'],label=origin['label'],gpu_uuid=allocation['uuid'],native_checked=result['native_checked'],native_errors=0,sanitizer=result['sanitizer'],sanitizer_errors=result['sanitizer_errors'],receipt=str(folder/'receipt.json'),stdout_sha256=receipt['stdout_sha256']))
 assert starts==sorted(starts);report=dict(actual_environment=allocation,processes=len(rows),native_checks_without_sanitizer=sum(r['native_checked']for r in rows if not r['sanitizer']),native_checks_under_sanitizer=sum(r['native_checked']for r in rows if r['sanitizer']),native_errors=0,tool_results={r['sanitizer']:r['sanitizer_errors']for r in rows if r['sanitizer']},calls=rows,qualification='本轮后补原预检；数值检查与采样阶段的性能评分分别保留')
 save(args.output,report);print(json.dumps({k:v for k,v in report.items()if k!='calls'},ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
