"""Check the fixed packaging update against old clock code and the pinned first raw."""
import argparse
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    report = json.loads((root/'review.json').read_text())
    for name,digest in report['snapshot_sha256'].items():
        assert sha(root/'snapshot'/name)==digest
    for path,digest in report['input_sha256'].items():
        assert sha(Path(path))==digest
    stock=json.loads((root/'stock-tests.json').read_text())
    assert stock['returncode']==0 and 'Ran 7 tests' in stock['stderr']
    prior=root.parent/'fixed-snapshot/decode_r13_wait_sum.py'
    assert sha(prior)==report['previous_decoder_sha256']
    old=ast.parse(prior.read_text())
    new=ast.parse((root/'snapshot/decode_r13_wait_sum.py').read_text())
    oldfn=copy.deepcopy(next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name=='decode_trace'))
    newfn=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=='decode_trace')
    changed=0
    for node in ast.walk(oldfn):
        if isinstance(node,ast.keyword) and node.arg=='cluster':
            assert isinstance(node.value,ast.List) and [x.value for x in node.value.elts]==[1,1,1]
            node.value=ast.List(elts=[ast.Constant(value=1),ast.Constant(value=1)],ctx=ast.Load())
            changed+=1
    assert changed==1 and ast.dump(oldfn)==ast.dump(newfn)
    sys.path.insert(0,str(root/'snapshot'))
    from test_r13_wait_sum import WaitSumTests
    from decode_r13_wait_sum import decode,decode_trace
    fixture=WaitSumTests()
    fields=(15,16,17,18,19,20,21,22,24,32,37,38)
    for row in report['translated_clock_cases']:
        setup,call=fixture.fixture(row['flag'])
        setup['stages']=row['stage']
        for index in fields:
            if call['trace'][index]:call['trace'][index]+=row['clock_offset']
        result=decode_trace(setup,call)
        for name in ('early_cycles','full_wait_cycles','outside_wait_cycles'):
            assert result[name]==row[name]
        assert result['full_wait_ns'] is None and result['outside_wait_ns'] is None
    for cluster in ([1,1,1],[1,2]):
        setup,call=fixture.fixture();setup['cluster']=cluster
        try:decode_trace(setup,call)
        except ValueError as error:assert str(error)=='fixed setup mismatch: cluster'
        else:raise AssertionError('unexpected cluster accepted')
    acquisition=Path(next(p for p in report['input_sha256'] if p.endswith('/receipt.json'))).parent
    actual=report['actual_first_record']
    assert sha(acquisition/'output.tar.gz')==actual['archive_sha256']
    assert json.loads((acquisition/'receipt.json').read_text())['returncode']==1
    with tarfile.open(acquisition/'output.tar.gz') as bundle:
        payload=bundle.extractfile('./raw/stage4_flag0.jsonl.gz').read()
        assert bundle.extractfile('./records.json').read()==(acquisition/'returned/records.json').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==actual['raw_sha256']
    assert payload==(acquisition/'returned/raw/stage4_flag0.jsonl.gz').read_bytes()
    events=[json.loads(line) for line in gzip.decompress(payload).decode().splitlines()]
    decoded=decode(events,'GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114')
    assert decoded['early_cycles']==11500 and len(decoded['completed_prefix_event_us'])==16
    assert decoded['independently_checked_values']==4096
    assert all(decoded[name] is None for name in ('full_wait_cycles','outside_wait_cycles','full_wait_ns','outside_wait_ns'))
    assert not decoded['observation_qualified'] and not decoded['prediction_qualified']
    verification=dict(review_sha256=sha(root/'review.json'),verifier_sha256=sha(Path(__file__)),
        clock_history_AST_equal_after_cluster_correction=True,translated_clock_cases=8,
        strict_cluster_rejections=2,pinned_actual_raw_values_checked=4096,
        failed_collection_preserved=True,qualification=False)
    with args.output.open('x') as stream:
        stream.write(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()
