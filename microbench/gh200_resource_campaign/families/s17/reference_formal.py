"""Independent S17 final-state and demand reference; no CUDA calls."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import math
import re
import struct
import sys

if sys.flags.optimize:
    raise RuntimeError('S17 requires active assertions; Python optimization forbidden')

FORMS={'local_read':0,'dsm_read':1,'local_write':2,'dsm_write':3,'cluster_sync':4,
       'bulk_single_target':5,'bulk_all_targets':6}


@lru_cache(maxsize=8192)
def packed(base,step,count,inverse=False):
    return struct.pack('<'+str(count)+'I', *[(((base+step*w)&0xffffffff)^(0xffffffff if inverse else 0)) for w in range(count)])


def sample(mode,C,block,item,word,seed):
    rank=block%C;base=block//C*C
    source=(rank+1)%C if mode==1 else (rank+C-1)%C if mode==3 else rank
    return ((29 if mode>=2 else 17)*((base+source)*1024+word)+seed+(31*item if mode>=2 else 0))&0xffffffff


def verify_values(directory,row,case_id):
    directory=Path(directory)
    match=re.fullmatch(r'(local_read|dsm_read|local_write|dsm_write|cluster_sync|bulk_single_target|bulk_all_targets)_c([248])_(one_cluster|cluster_grid)',case_id)
    assert match and row['case_id']==case_id
    form,C,scope=match.groups();C=int(C);mode=FORMS[form]
    for key in ('iterations','seed','blocks','clusters','cluster_size','threads','errors','final_invocation',
                'work_count','read_payload_bytes','write_payload_bytes','source_request_bytes',
                'total_receiver_bytes','required_write_readback_bytes','post_timing_export_bytes','start_ns','stop_ns','mask'):
        assert type(row[key]) is int and row[key]>=0
    I,seed,B,G,inv=(row[k] for k in ('iterations','seed','blocks','clusters','final_invocation'))
    assert 128<=I<=65536 and 0<=seed<=0xffffffff and G>0 and B==G*C and row['cluster_size']==C
    assert row['scope']==scope and row['form']==form and row['threads']==128 and row['errors']==0 and row['capture'] is False
    assert scope!='one_cluster' or G==1
    assert row['phase'] in ('pilot','formal')
    if row['phase']=='pilot':
        assert I==128 and inv==1 and row['warmup_executed'] is False and row['warmup_samples_ns']==[]
    else:
        assert row['warmup_executed'] is True and 8<=len(row['warmup_samples_ns'])<=30
        assert inv==len(row['warmup_samples_ns'])+1
    W=4096 if mode>=5 else 1024 if mode<4 else 0
    shapes={}
    if W:shapes.update(final_tiles=[B,W],guards=[B,8])
    shapes.update(completion=[B,6,2],stamps=[B,5,2])
    if mode<4:shapes['sums']=[B,128]
    if mode>=5:shapes['source_ring']=[G,32,4096]
    artifacts=row['full_output_artifacts'];assert len(artifacts)==len(shapes)
    for artifact,(name,shape) in zip(artifacts,shapes.items()):
        assert artifact['path']==f'cluster_{inv}_{name}.u32le' and artifact['dtype']=='uint32'
        assert artifact['shape']==shape and all(type(n) is int and n>0 for n in artifact['shape'])
        path=directory/artifact['path'];assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size==math.prod(shape)*4
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
        assert h.hexdigest()==artifact['sha256']
    def path(name):return directory/f'cluster_{inv}_{name}.u32le'
    mask=(1<<C)-1 if mode==6 else 1<<(C-1) if mode==5 else 0
    assert row['mask']==mask
    if W:
        with path('final_tiles').open('rb') as stream:
            for block in range(B):
                if mode>=5:
                    receiver=bool(mask&(1<<(block%C)));slot=(I-1)%32 if receiver else 0
                    base=(17*((block//C*32+slot)*4096)+seed)&0xffffffff
                    expected=packed(base,17,W,not receiver)
                elif mode<2:
                    expected=packed((17*block*1024+seed)&0xffffffff,17,W)
                else:expected=packed(sample(mode,C,block,I-1,0,seed),29,W)
                assert stream.read(W*4)==expected,'complete final tile mismatch'
            assert stream.read(1)==b''
        guards=struct.pack('<8I',*range(0xd15ea5e0,0xd15ea5e8))
        with path('guards').open('rb') as stream:
            for block in range(B):assert stream.read(32)==guards
            assert stream.read(1)==b''
    if mode<4:
        with path('sums').open('rb') as stream:
            for block in range(B):
                expected=[]
                for thread in range(128):
                    base=sum(sample(mode,C,block,0,thread+128*j,seed) for j in range(8))
                    expected.append((base*I+(31*8*I*(I-1)//2 if mode>=2 else 0))&0xffffffff)
                assert stream.read(512)==struct.pack('<128I',*expected),'complete checksum mismatch'
            assert stream.read(1)==b''
    if mode>=5:
        with path('source_ring').open('rb') as stream:
            for cluster in range(G):
                for slot in range(32):
                    assert stream.read(16384)==packed((17*((cluster*32+slot)*4096)+seed)&0xffffffff,17,4096)
            assert stream.read(1)==b''
    counts=list(struct.iter_unpack('<6Q',path('completion').read_bytes()))
    stamps=list(struct.iter_unpack('<5Q',path('stamps').read_bytes()))
    assert len(counts)==len(stamps)==len(row['blocks_detail'])==B
    for block,(count,stamp,detail) in enumerate(zip(counts,stamps,row['blocks_detail'])):
        assert count==(I,block%C,C,1,1,0)
        assert all(v<2**64-1 for v in stamp[:4]) and stamp[4]<2**32-1
        assert stamp[1]>stamp[0] and stamp[3]>stamp[2] and detail['block_id']==block
        assert list(stamp)==[detail[k] for k in ('start_ns','stop_ns','start_cycle','stop_cycle','smid')]
    assert row['start_ns']==min(s[0] for s in stamps) and row['stop_ns']==max(s[1] for s in stamps)
    assert type(row['event_ms']) in (int,float) and math.isfinite(row['event_ms']) and row['event_ms']>0
    assert row['stop_ns']-row['start_ns']<=row['event_ms']*1.05e6
    requested=B*I*4096 if mode<4 else G*I*16384 if mode>=5 else 0
    work=G*I if mode==4 else requested
    assert row['work_count']==work and row['work_unit']==('cluster_phase' if mode==4 else 'byte')
    assert row['read_payload_bytes']==(requested if mode!=4 else 0)
    assert row['write_payload_bytes']==(requested if mode in (2,3) else 0)
    assert row['source_request_bytes']==(requested if mode>=5 else 0)
    assert row['total_receiver_bytes']==(requested*(C if mode==6 else 1) if mode>=5 else 0)
    assert row['required_write_readback_bytes']==(requested if mode in (2,3) else 0)
    assert row['post_timing_export_bytes']==B*W*4
    checked=B*22+(B*(W+8) if W else 0)+(B*128 if mode<4 else 0)+(G*32*4096 if mode>=5 else 0)
    assert row['correctness']['checked_elements']==checked
    identity_path=directory/f'cluster_{inv}_identity.json';assert identity_path.is_file() and not identity_path.is_symlink()
    identity=json.loads(identity_path.read_text());assert identity['capture'] is False and type(identity['errors']) is int
    assert all(type(identity[k]) is int for k in ('iterations','seed','invocation','blocks','clusters','cluster_size','errors'))
    assert type(identity['case_id']) is str and type(identity['diagnostic']) is str
    assert identity=={'case_id':case_id,'iterations':I,'seed':seed,'invocation':inv,'blocks':B,'clusters':G,
                      'cluster_size':C,'capture':False,'errors':0,'diagnostic':''}
    return {'all_final_values_verified':True,'checked_uint32_elements':checked,'requested_bytes':requested,
            'work_count':work,'primary_value':work/(row['stop_ns']-row['start_ns']),'qualified':False}
