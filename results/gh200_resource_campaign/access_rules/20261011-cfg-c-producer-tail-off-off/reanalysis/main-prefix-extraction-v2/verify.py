"""Check the extraction/prediction boundary on known data and poisoned targets."""
import copy,gzip,hashlib,json,subprocess,sys,tempfile,types
from pathlib import Path
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;repo=out.parents[5]
code=repo/'microbench/gh200_resource_campaign/access_rules'
sys.path.insert(0,str(code))
from r09_extract_prefix_input import extract_prefix_input, CALL_PROTOCOL_FIELDS, PREFIX_ZERO_FIELDS
from r09_prefix_clock_rule import predict_from_prefix
from r09_prefix_ordinal_clock import SETUP_FIELDS
run=out.parents[1]
rule_root=out.parent/'main-prefix-clock-local-rule-v1'
rule=json.loads((rule_root/'rule.json').read_text())
manifest=json.loads((run/'manifest.json').read_text())
records=json.loads((run/'returned/records.json').read_text())
allowed={'warmup_calls','prefix_a_ptr','target_a_ptr','prefix_b_ptr','target_b_ptr',
    'target_calls','preceding_target_calls','warmup_us','target_swizzle'} | set(CALL_PROTOCOL_FIELDS) | set(PREFIX_ZERO_FIELDS)
old_source=subprocess.check_output(['git','show','255e93a:microbench/gh200_resource_campaign/access_rules/r09_extract_prefix_input.py'],cwd=repo)
old=types.ModuleType('original_extractor');exec(compile(old_source,'original_extractor','exec'),old.__dict__)
class Restricted(dict):
    def __getitem__(self,key):
        if key not in allowed:raise AssertionError('target field read: '+key)
        return super().__getitem__(key)
raw_hashes={};checked=0;protocol_rejections=0;missing_rejections=0
for record in records:
    path=run/'returned'/record['raw'];raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==record['sha256'];raw_hashes[str(path)]=record['sha256']
    setup,call,check=map(json.loads,gzip.decompress(raw).splitlines())
    prefix=extract_prefix_input(setup,Restricted(call),manifest['binary_sha256'])
    assert prefix==old.extract_prefix_input(setup,Restricted(call),manifest['binary_sha256'])
    assert set(prefix)=={'setup','prefix_event_us'} and set(prefix['setup'])==set(SETUP_FIELDS)
    prediction=predict_from_prefix(rule,prefix)
    poisoned=copy.deepcopy(call)
    for name in set(poisoned)-allowed:poisoned[name]=object()
    assert extract_prefix_input(setup,Restricted(poisoned),manifest['binary_sha256'])==prefix
    for name in (*CALL_PROTOCOL_FIELDS, 'target_swizzle', *PREFIX_ZERO_FIELDS):
        wrong=copy.deepcopy(call)
        wrong[name] = wrong[name] + 1 if type(wrong[name]) is int else 'different-protocol'
        try:extract_prefix_input(setup,Restricted(wrong),manifest['binary_sha256'])
        except ValueError:protocol_rejections+=1
        else:raise AssertionError('contradictory protocol accepted: '+name)
        wrong=copy.deepcopy(call);del wrong[name]
        try:extract_prefix_input(setup,Restricted(wrong),manifest['binary_sha256'])
        except KeyError:missing_rejections+=1
        else:raise AssertionError('missing protocol accepted: '+name)
    wrong=copy.deepcopy(call);wrong['prefix_a_ptr']+=16
    try:predict_from_prefix(rule,extract_prefix_input(setup,Restricted(wrong),manifest['binary_sha256']))
    except ValueError:pass
    else:raise AssertionError('different prefix buffer accepted')
    wrong=copy.deepcopy(call);wrong['prefix_a_ptr']=wrong['target_a_ptr']=None
    try:extract_prefix_input(setup,Restricted(wrong),manifest['binary_sha256'])
    except ValueError:pass
    else:raise AssertionError('invalid buffer identities accepted')
    wrong=copy.deepcopy(call);wrong['warmup_us']=wrong['warmup_us'][:-1]
    try:predict_from_prefix(rule,extract_prefix_input(setup,Restricted(wrong),manifest['binary_sha256']))
    except ValueError:pass
    else:raise AssertionError('incomplete history accepted')
    checked+=1

with tempfile.TemporaryDirectory(prefix='r09-extract-') as temp:
    directory=Path(temp);target=directory/'prefix.json'
    command=[sys.executable,str(code/'r09_extract_prefix_input.py'),'--raw',str(path),
        '--raw-sha256',record['sha256'],'--binary-sha256',manifest['binary_sha256'],
        '--output',str(target)]
    receipt=json.loads(subprocess.check_output(command,text=True))
    assert json.loads(target.read_text())==prefix
    assert receipt['prefix_input_sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
    assert predict_from_prefix(rule,json.loads(target.read_text()))==prediction
    assert subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    command[command.index('--raw-sha256')+1]='0'*64
    command[-1]=str(directory/'bad.json')
    assert subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    assert not (directory/'bad.json').exists()
    (out/'example-prefix.json').write_bytes(target.read_bytes())
    (out/'example-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
report=dict(records=checked,poisoned_target_noninterference_checks=checked,
    original_valid_prefixes_unchanged=checked,previous_source_sha256=hashlib.sha256(old_source).hexdigest(),
    contradictory_protocol_rejections=protocol_rejections,missing_protocol_rejections=missing_rejections,
    different_buffer_rejections=checked,invalid_buffer_identity_rejections=checked,
    incomplete_history_rejections=checked,
    raw_hash_mismatch_rejected=True,output_overwrite_rejected=True,
    raw_sha256=raw_hashes,extractor_sha256=hashlib.sha256((code/'r09_extract_prefix_input.py').read_bytes()).hexdigest(),
    rule_sha256=hashlib.sha256((rule_root/'rule.json').read_bytes()).hexdigest(),
    target_validity_checked=False,prospective_holdout=False,observation_qualified=False)
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='raw_sha256'},indent=2))
