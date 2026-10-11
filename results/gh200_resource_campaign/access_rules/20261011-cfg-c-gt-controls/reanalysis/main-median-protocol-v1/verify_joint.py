"""Check joint composite endpoints directly from archived trace words."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint',type=Path,required=True)
    parser.add_argument('--old-replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    receipt=json.loads((args.joint/'receipt.json').read_text())
    for name,digest in receipt['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest
    for name,digest in receipt['output_sha256'].items():
        assert hashlib.sha256((args.joint/name).read_bytes()).hexdigest()==digest
    records=json.loads(gzip.decompress((args.joint/'joint-process-records.json.gz').read_bytes()))
    windows=0;scalars=0
    for record in records:
        raw=Path(record['raw_path']);assert hashlib.sha256(raw.read_bytes()).hexdigest()==record['raw_sha256']
        with gzip.open(raw,'rt') as stream:packets={r['event']:r for line in stream for r in [json.loads(line)]}
        assert record['setup']==packets['setup']
        assert record['completed_prefix_event_us']==packets['call']['warmup_us']
        assert len(record['completed_prefix_event_us'])==16
        assert record['call_protocol']=={key:value for key,value in packets['call'].items() if 'warmup' in key or 'prefix' in key}
        words=packets['call']['trace'];assert len(words)==132*1296
        origin=min(words[cta*1296+1] for cta in range(132));finals=[]
        expected={}
        for cta in range(132):
            head=cta*1296
            assert record['sm_by_cta'][str(cta)]==words[head+2]-1
            for wg in (0,1):
                finals.append(words[head+5+wg*3]-origin)
                for ordinal in range(words[head+6+wg*3]):
                    i=head+16+wg*640+ordinal*10
                    prefix=f'{cta}/{wg}/{ordinal}/{words[i+4]-1},{words[i+5]-1}/'
                    for j,name in enumerate(('FIRST','MAIN','PERMIT','DONE')):
                        assert record['series'][prefix+name]==[words[i+6+j]-words[head+1],words[i+j]-words[head]]
                    expected[prefix+'FIRST_to_DONE']=[words[i+9]-words[i+6],words[i+3]-words[i]]
                    expected[prefix+'MAIN_to_DONE']=[words[i+9]-words[i+7],words[i+3]-words[i+1]]
                    windows+=1;scalars+=4
        expected['consumer_FINAL_envelope']=[max(finals),None]
        assert record['composites']==expected
        assert record['same_call_event_us']==packets['call']['elapsed_us']
    new=json.loads(gzip.decompress((args.joint/'median-diagnostic.json.gz').read_bytes()))
    old=json.loads(gzip.decompress(args.old_replay.read_bytes()))
    assert new==old
    result=dict(processes=len(records),raw_window_vectors=windows,
        directly_checked_composite_duration_scalars=scalars,consumer_envelopes=len(records),
        complete_median_diagnostic_equal_to_old_replay=True,
        joint_receipt_sha256=hashlib.sha256((args.joint/'receipt.json').read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification=False)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
