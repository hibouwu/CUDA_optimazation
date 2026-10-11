"""Verified raw control records -> joint vectors -> marginal numerical inputs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

from evaluate import evaluate
from joint_records import assemble
from acquisition_identity import verify_acquisition


def validate_pair_identities(ordered):
    for i in range(0,len(ordered),2):
        a,b=ordered[i:i+2]
        if (a['cohort'],a['block'],a['arm'])!=(b['cohort'],b['block'],0) or b['arm']!=1:
            raise ValueError('invalid paired process identity')
        if a['sm_by_cta']!=b['sm_by_cta']:
            raise ValueError('paired CTA-to-SM mapping changed')
        for record in (a,b):
            expected_flag=record['arm'] if record['cohort']=='off_on' else 0
            if record['flag']!=expected_flag:
                raise ValueError('observer flag differs from cohort/arm meaning')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--expected-protocol-sha256',required=True)
    parser.add_argument('--expected-output-sha256',required=True,
                        help='Previously pinned acquisition output.tar.gz SHA; never derive from the input being checked')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if hashlib.sha256(args.protocol.read_bytes()).hexdigest()!=args.expected_protocol_sha256:
        raise ValueError('protocol identity mismatch')
    if args.output.exists():raise ValueError('new output directory required')
    protocol=json.loads(args.protocol.read_text());run=args.run.resolve()
    manifest,returned,hashes=verify_acquisition(run,args.expected_output_sha256)
    if manifest['gpu_uuid']!=protocol['gpu_uuid'] or manifest['binary_sha256']!=protocol['binary_sha256']:
        raise ValueError('acquisition identity differs from protocol')
    # Only the existing, reviewed frozen decoder is supported by this adapter.
    trusted=Path(__file__).resolve().parents[2]/'analysis_snapshot'
    for filename,digest in manifest['analysis_sha256'].items():
        path=run/'analysis_snapshot'/filename
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest or path.read_bytes()!=(trusted/filename).read_bytes():
            raise ValueError('decoder identity differs from reviewed snapshot')
        hashes['analysis_snapshot/'+filename]=digest
    sys.path.insert(0,str(trusted))
    from analyze_cfg_c_gt_control import process
    declared={s['id']:s for s in manifest['samples']}
    if len(declared)!=len(manifest['samples']):raise ValueError('duplicate declared process')
    joint=[]
    for index,record in enumerate(returned):
        if record['id'] not in declared or record['returncode']!=0:
            raise ValueError('undeclared or failed process')
        if any(record[k]!=v for k,v in declared[record['id']].items()):
            raise ValueError('process metadata differs from declaration')
        path=run/'returned'/record['raw'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=record['sha256']:raise ValueError('raw SHA mismatch')
        hashes['returned/'+record['raw']]=digest
        with gzip.open(path,'rt') as stream:events=[json.loads(line) for line in stream]
        decoded=process(events,record,manifest['gpu_uuid'])
        packets={event['event']:event for event in events}
        values=decoded['series'];composites={}
        for name,pair in values.items():
            if not name.endswith('/L'):continue
            prefix=name[:-1]
            for label,begin,end,parts in [('FIRST_to_DONE','FIRST','DONE',('L','W','E')),
                                           ('MAIN_to_DONE','MAIN','DONE',('W','E'))]:
                direct=[values[prefix+end][i]-values[prefix+begin][i] for i in (0,1)]
                if any(direct[i]!=sum(values[prefix+part][i] for part in parts) for i in (0,1)):
                    raise ValueError('same-process endpoint closure failed')
                composites[prefix+label]=direct
        finals=[pair[0] for name,pair in values.items() if name.endswith('/GLOBAL_FINAL')]
        if len(finals)!=264 or len(composites)!=864:raise ValueError('joint endpoint count mismatch')
        composites['consumer_FINAL_envelope']=[max(finals),None]
        joint.append(dict(id=record['id'],block=record['pair'],cohort=record['comparison'],arm=record['arm'],
            flag=record['flag'],archive_record_index=index,raw_path=str(path),raw_sha256=digest,
            gpu_uuid=manifest['gpu_uuid'],binary_sha256=manifest['binary_sha256'],
            setup=packets['setup'],completed_prefix_event_us=packets['call']['warmup_us'],
            call_protocol={key:value for key,value in packets['call'].items()
                           if 'warmup' in key or 'prefix' in key},
            sm_by_cta=decoded['sms'],series=values,composites=composites,
            same_call_event_us=decoded['elapsed_us']))
    if {r['id'] for r in joint}!=set(declared):raise ValueError('missing declared process')
    samples,ordered=assemble(protocol,joint)
    validate_pair_identities(ordered)
    result=evaluate(protocol,samples)
    args.output.mkdir()
    for name,value in [('joint-process-records',ordered),('median-diagnostic',result)]:
        (args.output/(name+'.json.gz')).write_bytes(gzip.compress(json.dumps(value,separators=(',',':')).encode(),mtime=0))
    receipt=dict(protocol_sha256=args.expected_protocol_sha256,input_sha256=hashes,
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('evaluate.py','joint_records.py','extract_joint.py','acquisition_identity.py')},
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()},
        processes=len(ordered),series_per_process=len(protocol['series']),
        composite_windows_per_process=864,consumer_FINAL_envelopes=len(ordered),
        median_counts=result['counts'],original_point_failures=result['original_point_failures'],
        raw_identity_and_vector_binding_verified=True,
        acquisition_archive_sha256=args.expected_output_sha256,
        acquisition_record_mapping_verified=True,
        scope='Recorded domains only. Composite samples are preserved separately, not added to the6372-series equivalence family. No cross-SM raw-cycle maximum.',
        limits=['Block is the archive pair label; old cohorts were sequential, not the new randomized proposal.',
                'Archive identity is caller-pinned evidence of the original mapping, not independent proof of collector correctness or physical execution.',
                'Raw hash binding does not establish sampling stationarity, timestamp accuracy, causal equivalence or last-CTA accuracy.',
                'Consumer FINAL envelope is not a certified CUDA completion or global-D-visible endpoint.'],
        observation_qualified=False,prediction_qualified=False)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('processes','series_per_process','median_counts','original_point_failures')},indent=2))


if __name__=='__main__':
    main()
