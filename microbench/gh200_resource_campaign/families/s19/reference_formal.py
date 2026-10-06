"""S19 full final-value and demand replay; no CUDA or host helper imports."""
from pathlib import Path
import hashlib,json,math,re,struct,sys
if sys.flags.optimize:raise RuntimeError('S19 requires active assertions; Python optimization forbidden')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'microbench/gh200_resource_campaign'))
from auditors import s19_reference as reference


def coordinate(case_id,iterations,seed):
    match=re.fullmatch(r'(compute|transport|serial|overlap|output)_s([124])_k(1|2|4|8|16|32|64)',case_id)
    assert match and type(iterations) is int and 128<=iterations<=65536
    assert type(seed) is int and 0<=seed<=0xffffffff
    mode,S,K=match.groups();return mode,int(S),int(K)


def expected_final(case_id,iterations,seed):
    mode,S,K=coordinate(case_id,iterations,seed)
    # C resets on each repetition. Only transport digest accumulates repetitions.
    value=reference.expected(reference.Case(mode,S,K,1,seed,'periodic'))
    return {'input':value['input_bits'],'output':value['output_bits'],
        'digest':[(v*iterations)&0xffffffff for v in value['digest']], 'slots':value['shared_bits'],
        'guards':([0xd1900000+i for i in range(8)]+[0xd1910000+i for i in range(8)])*6}


def verify_values(directory,row):
    directory=Path(directory);mode,S,K=coordinate(row['case_id'],row['iterations'],row['seed']);N=row['iterations'];seed=row['seed']
    for key in ('threads','blocks','errors','final_invocation','mode','stages','tiles','work_count','read_payload_bytes','write_payload_bytes','fma_flop','epilogue_flop','post_timing_output_export_bytes','post_timing_diagnostic_export_bytes','start_ns','stop_ns'):
        assert type(row[key]) is int and row[key]>=0
    assert row['threads']==128 and row['blocks']==1 and row['scope']=='one_cta' and row['errors']==0
    assert row['mode']==reference.MODES.index(mode) and row['stages']==S and row['tiles']==K and row['trace_enabled'] is False and row['input_profile']=='periodic'
    pilot=row['phase']=='pilot';assert pilot or row['phase']=='formal'
    assert row['performance_eligible'] is (not pilot) and row['warmup_executed'] is (not pilot)
    inv=row['final_invocation']
    if pilot:assert N==128 and inv==1 and row['warmup_samples_ns']==[] and row['warmup_converged'] is False
    else:assert 8<=len(row['warmup_samples_ns'])<=30 and inv==len(row['warmup_samples_ns'])+1
    expected=expected_final(row['case_id'],N,seed);artifacts=row['full_output_artifacts'];assert len(artifacts)==6
    for item,(name,words) in zip(artifacts,list(expected.items())+[('stamp',None)]):
        assert item['path']==f's19_{inv}_{name}.u32le' and item['dtype']=='uint32'
        count=10 if words is None else len(words);assert item['shape']==[count] and type(item['shape'][0]) is int
        path=directory/item['path'];assert path.is_file() and not path.is_symlink();raw=path.read_bytes()
        assert len(raw)==count*4 and hashlib.sha256(raw).hexdigest()==item['sha256']
        if words is not None:assert raw==struct.pack('<'+str(count)+'I',*words),'complete '+name+' mismatch'
        else:
            ns0,ns1,cy0,cy1,smid,event_bits=struct.unpack('<4Q2I',raw)
            detail=row['blocks_detail'];assert len(detail)==1
            d=detail[0];assert all(type(d[k]) is int for k in ('block_id','smid','start_ns','stop_ns','start_cycle','stop_cycle'))
            assert d=={'block_id':0,'smid':smid,'start_ns':ns0,'stop_ns':ns1,'start_cycle':cy0,'stop_cycle':cy1}
            assert 0<=smid<0xffffffff and 0<=ns0<ns1<2**64-1 and 0<=cy0<cy1<2**64-1
            assert row['start_ns']==ns0 and row['stop_ns']==ns1
            assert type(row['event_ms']) in (int,float) and math.isfinite(row['event_ms']) and row['event_ms']>0
            assert row['event_ms']==struct.unpack('<f',struct.pack('<I',event_bits))[0] and ns1-ns0<=row['event_ms']*1.05e6
    identity_path=directory/f's19_{inv}_identity.json';assert identity_path.is_file() and not identity_path.is_symlink()
    identity=json.loads(identity_path.read_text());assert all(type(identity[k]) is int for k in ('iterations','seed','invocation','errors'))
    assert identity=={'case_id':row['case_id'],'iterations':N,'seed':seed,'invocation':inv,'errors':0,'diagnostic':'','trace_enabled':False} and identity['trace_enabled'] is False
    fma=0 if mode=='transport' else 65536*K*N;epilogue=2048*N if mode=='output' else 0
    read=0 if mode=='compute' else 8192*K*N;write=4096*N if mode=='output' else 0;work=read if mode=='transport' else fma+epilogue
    assert row['fma_flop']==fma and row['epilogue_flop']==epilogue and row['work_count']==work
    assert row['work_unit']==('byte' if mode=='transport' else 'FLOP') and row['read_payload_bytes']==read and row['write_payload_bytes']==write
    assert row['post_timing_output_export_bytes']==(0 if mode=='output' else 4096) and row['post_timing_diagnostic_export_bytes']==33280
    checked=K*2048+1024+128+8192+96+10;assert row['correctness']['checked_elements']==checked
    return {'all_final_values_verified':True,'checked_uint32_elements':checked,'work_count':work,
        'primary_value':work/(row['blocks_detail'][0]['stop_cycle']-row['blocks_detail'][0]['start_cycle']),'qualified':False}
