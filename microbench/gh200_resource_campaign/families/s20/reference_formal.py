"""S20 full final-value and metadata replay, reusing the reviewed short mathematics."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,math,re,struct,sys
if sys.flags.optimize:raise RuntimeError('S20 requires active assertions; Python optimization forbidden')
ROOT=Path(__file__).resolve().parent
LEGACY=ROOT/'results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/combinations/s20-v1'
spec=importlib.util.spec_from_file_location('s20_legacy_raw_audit',LEGACY/'host-v2/raw_audit.py');legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
reference=legacy.reference


def coordinate(case_id,iterations,seed):
    match=re.fullmatch(r'(compute|transport|serial|overlap|output)_s([124])_k(1|2|4|8|16|32|64)',case_id)
    assert match and type(iterations) is int and 128<=iterations<=65536
    assert type(seed) is int and 0<=seed<=0xffffffff
    mode,S,K=match.groups();return SimpleNamespace(mode=mode,stages=int(S),k_tiles=int(K),iterations=iterations,seed=seed,profile='periodic')


def expected_final(case_id,iterations,seed):
    c=coordinate(case_id,iterations,seed);value=reference.expected(reference.Case(c.mode,c.stages,c.k_tiles,1,seed,'periodic'))
    return {'input':('uint16',value['input_bf16']),'output':('uint32',value['output_bits']),
        'digest':('uint32',[(v*iterations)&0xffffffff for v in value['digest']]),'slots':('uint16',value['shared_bf16'])}


def verify_values(directory,row):
    directory=Path(directory);c=coordinate(row['case_id'],row['iterations'],row['seed']);N=c.iterations;inv=row['final_invocation']
    for k in ('threads','blocks','errors','final_invocation','mode','stages','tiles','work_count','read_payload_bytes','write_payload_bytes','MMA_FLOP','epilogue_FLOP','post_timing_output_export_bytes','post_timing_diagnostic_export_bytes','start_ns','stop_ns'):
        assert type(row[k]) is int and row[k]>=0
    assert row['threads']==128 and row['blocks']==1 and row['scope']=='one_cta' and row['errors']==0
    assert row['mode']==reference.MODES.index(c.mode) and row['stages']==c.stages and row['tiles']==c.k_tiles
    assert row['trace_enabled'] is False and row['input_profile']=='periodic'
    pilot=row['phase']=='pilot';assert pilot or row['phase']=='formal'
    assert row['performance_eligible'] is (not pilot) and row['warmup_executed'] is (not pilot)
    if pilot:assert N==128 and inv==1 and row['warmup_samples_ns']==[] and row['warmup_converged'] is False
    else:assert 8<=len(row['warmup_samples_ns'])<=30 and inv==len(row['warmup_samples_ns'])+1
    expected=expected_final(row['case_id'],N,c.seed);expected.update(metadata=('uint32',None),guards=('uint32',[0xd2000000+b*0x100+s*0x80+i for b in range(7) for s in range(2) for i in range(8)]),event=('uint32',None))
    artifacts=row['full_output_artifacts'];assert len(artifacts)==7;decoded={}
    for item,(name,(dtype,words)) in zip(artifacts,expected.items()):
        count=len(words) if words is not None else 116 if name=='metadata' else 1
        suffix='u16le' if dtype=='uint16' else 'u32le';assert item['path']==f's20_{inv}_{name}.{suffix}' and item['dtype']==dtype
        assert item['shape']==[count] and type(item['shape'][0]) is int and type(item['bytes']) is int
        width=2 if dtype=='uint16' else 4;path=directory/item['path'];assert path.is_file() and not path.is_symlink();raw=path.read_bytes()
        assert len(raw)==item['bytes']==count*width and hashlib.sha256(raw).hexdigest()==item['sha256']
        kind='H' if width==2 else 'I'
        if words is not None:assert raw==struct.pack('<'+str(count)+kind,*words),'complete '+name+' mismatch'
        else:decoded[name]=list(struct.unpack('<'+str(count)+kind,raw))
    metadata=legacy.decode_metadata(decoded['metadata']);event_ms=struct.unpack('<f',struct.pack('<I',decoded['event'][0]))[0]
    assert type(row['event_ms']) in (int,float) and math.isfinite(row['event_ms']) and row['event_ms']==event_ms and event_ms>0
    legacy.check_metadata(metadata,c,event_ms)
    detail=row['blocks_detail'];assert len(detail)==1;d=detail[0]
    assert all(type(d[k]) is int for k in ('block_id','smid','start_ns','stop_ns','start_cycle','stop_cycle'))
    assert d=={'block_id':0,'smid':metadata['smid'],'start_ns':metadata['begin_ns'],'stop_ns':metadata['end_ns'],'start_cycle':metadata['begin_cycle'],'stop_cycle':metadata['end_cycle']}
    assert 0<=row['start_ns']==metadata['begin_ns']<row['stop_ns']==metadata['end_ns']<2**64-1
    assert 0<=metadata['begin_cycle']<metadata['end_cycle']<2**64-1
    identity_path=directory/f's20_{inv}_identity.json';assert identity_path.is_file() and not identity_path.is_symlink();identity=json.loads(identity_path.read_text())
    assert all(type(identity[k]) is int for k in ('iterations','seed','invocation','errors'))
    assert identity=={'case_id':row['case_id'],'iterations':N,'seed':c.seed,'invocation':inv,'errors':0,'diagnostic':'','trace_enabled':False} and identity['trace_enabled'] is False
    mma=0 if c.mode=='transport' else 524288*c.k_tiles*N;epi=8192*N if c.mode=='output' else 0
    read=0 if c.mode=='compute' else 16384*c.k_tiles*N;write=16384*N if c.mode=='output' else 0;work=read if c.mode=='transport' else mma+epi
    assert row['MMA_FLOP']==mma and row['epilogue_FLOP']==epi and row['read_payload_bytes']==read and row['write_payload_bytes']==write
    assert row['work_count']==work and row['work_unit']==('byte' if c.mode=='transport' else 'FLOP')
    assert row['post_timing_output_export_bytes']==(0 if c.mode=='output' else 16384) and row['post_timing_diagnostic_export_bytes']==66048
    checked=c.k_tiles*8192+4096+128+32768+116+112+1;assert row['correctness']['checked_elements']==checked
    return {'all_final_values_verified':True,'checked_elements':checked,'work_count':work,'primary_value':work/(metadata['end_cycle']-metadata['begin_cycle']),'qualified':False}
