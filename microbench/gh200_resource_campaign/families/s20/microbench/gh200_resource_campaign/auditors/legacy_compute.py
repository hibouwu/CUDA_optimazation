"""S05--S07 audit: independently derived arithmetic and bounded CPU oracles."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
from auditors.memory_baseline import validate_device, timed_loops
from auditors.observation import validate_observation, validate_timer_contract, TIMER_FLAG
from common.suite_io import require

ADAPTER_IDS = {"legacy_fma_v2": ("legacy_fma", "S05", 93),
               "legacy_mma_v2": ("legacy_mma", "S06", 55),
               "legacy_wgmma_v2": ("legacy_wgmma", "S07", 85)}
TIMING = "v2_cta_start_gate_result_drain_v1"


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False)


def kernel_symbol(case):
    p=case['parameters']
    signature=(p['ptx'],case['threads'],p['chains'],p['batch'],p['wait'],
               p['drain']['post_loop_wgmma_wait'],p['operand_source_form'])
    return 'lc_'+hashlib.sha256(json.dumps(signature,separators=(',', ':')).encode()).hexdigest()[:20]


def validate_contract(contract):
    revised=validate_timer_contract(contract)
    identity=contract.get('adapter_id');require(identity in ADAPTER_IDS,'unknown compute adapter')
    family,stage,count=ADAPTER_IDS[identity]
    require(contract.get('schema_version')==2 and contract.get('family')==family
            and contract.get('stage')==stage,'compute family/schema/stage')
    mapping_path=Path(__file__).resolve().parents[1]/'contracts/legacy_mapping.json'
    require(hashlib.sha256(mapping_path.read_bytes()).hexdigest()==contract['legacy_mapping_sha256'],
            'frozen legacy mapping hash')
    source=json.loads(mapping_path.read_text())
    legacy={r['mapping_id']:r for r in source['mappings'] if r['destination']['stage']==stage}
    seen=[];new=set();ids=set()
    require(len(contract['cases'])==count,'compute finite case count')
    for case in contract['cases']:
        require(case['id'] not in ids,'duplicate case');ids.add(case['id']);p=case['parameters']
        width=1 if family=='legacy_fma' else 32 if family=='legacy_mma' else 128
        require(p['collective_width_threads']==width and case['threads'] in (32,128,256)
                and case['threads']%width==0,'collective participation')
        require(p['chains'] in ((1,2) if width==128 else (1,8)) and
                p['batch'] in ((1,4,16) if width==128 else (16,)),'chains/batch')
        old=case.get('legacy')
        if old is not None:
            mid=old['mapping_id'];require(mid in legacy,'unknown legacy mapping');seen.append(mid)
            key=legacy[mid]['key'];require(old['key']==key and case['id']=='legacy-'+mid,'legacy key/identity')
            for field in ('ptx','instruction_template','shape','chains','batch','wait','input','drain','sources','layout'):
                require(p[field]==key[field],'changed legacy compute field: '+field)
            require(p['input_type']==key['types']['input'] and p['accumulator_type']==key['types']['accumulator'],
                    'legacy numeric types')
            require(case['threads']==key['threads'] and case['scope']==
                    {'single_cta':'one_cta','full_gpu':'all_gpu'}[key['scope']],'legacy launch')
            require(p['phase']==key['phase'],'legacy phase')
        else:
            require(width==128,'unexpected new representative')
            form=p['operand_source_form'];require(form in ('SS','RS'),'WGMMA source form')
            expected_id=f'bf16_m64n64k16_{form.lower()}_t128_c2_q16_p0_{case["scope"]}'
            require(case['id']==expected_id and case['scope'] in ('one_cta','all_gpu'),'representative identity')
            require(p['input_type']=='bf16' and p['shape']==[64,64,16] and case['threads']==128
                    and p['chains']==2 and p['batch']==16 and p['wait']==0
                    and p['drain']['post_loop_wgmma_wait']==0 and case['iterations']==8192,'representative parameters')
            require(p['sources']=={'A':'registers_4xb32' if form=='RS' else 'shared_memory_descriptor',
                                   'B':'shared_memory_descriptor','accumulator':'registers'},'source operands')
            new.add((form,case['scope']))
        empty=p['phase']=='empty_control';ip=case['iteration_policy']
        if old and old['key']['iteration_policy']['kind']=='dynamic_cuda_event_pilot':
            expected={'kind':'calibrated','pilot_iterations':8192,'target_ns':100000000,
                      'min_iterations':8192,'max_iterations':1048576 if width==1 else 65536,
                      'rounding_model':'legacy_event_ms_truncate_then_clamp_v1','calibration_timer':'cuda_event_ms',
                      'minimum_pilot_ns':10000,'requires_B_bound_resolved_cases':True}
            require(ip==expected and type(case['iterations']) is int and
                    8192<=case['iterations']<=expected['max_iterations'],'calibration policy/domain')
        else:
            it=old['key']['iteration_policy']['iterations'] if old else 8192
            require(ip=={'kind':'fixed','value':it} and case['iterations']==it,'fixed iterations')
        require((case['iterations']==0)==empty,'zero iterations only empty')
        one=case['scope']=='one_cta'
        require(case['scope'] in ('one_cta','all_gpu') and case['launch']==({'kind':'one_cta'} if one else
                {'kind':'sms_capped_occupancy','maximum_ctas_per_sm':4,'formula':'sms*min(4,actual_kernel_occupancy_ctas_per_sm)'}),'grid policy')
        expected_model='compute_empty_window_v1' if empty else {1:'fma_dense_issue_v1',32:'mma_dense_issue_v1',128:'wgmma_dense_issue_v1'}[width]
        require(case['work_model']==expected_model and case['work_unit']=='FLOP','work model/unit')
        require(case['metric']==({'numerator':'cta_clock64_cycles' if revised and one else 'elapsed_ns','denominator':'one','scale':1,'unit':'cycles/window' if revised and one else 'ns/window'} if empty else
                {'numerator':'work_count','denominator':'cta_clock64_cycles' if one else 'elapsed_ns','scale':1,
                 'unit':'FLOP/clock64_cycle/CTA' if one else 'GFLOP/s/GPU'}),'metric quantity/scale/unit')
        require(case['coverage_policy']==('one_sm' if one else 'exact_sms') and case['exportable'] is (not empty),'coverage/export')
        require(case['capabilities']=={'cc':'9.0','device_name_contains':'GH200'} and p['timing_model']==TIMING,'target/timing')
        derived=arithmetic(case,1)
        require(derived['checked']==p['output_elements_per_cta'],'output count from shape/lanes')
    require(len(seen)==len(legacy) and set(seen)==set(legacy),'legacy coverage missing/duplicated')
    require(new==({(a,s) for a in ('SS','RS') for s in ('one_cta','all_gpu')} if family=='legacy_wgmma' else set()),'representative coverage')
    require(contract['build']=={'compiler':'nvcc','flags':['-std=c++17','-O3','-lineinfo','-gencode','arch=compute_90a,code=sm_90a','-Xptxas=-v']+([TIMER_FLAG] if revised else []),'include_dirs':[]},'build contract')
    return contract


def arithmetic(case,blocks):
    """Infer dimensions and lanes from PTX; do not consume recorded FLOP totals."""
    p=case['parameters'];ptx=p['ptx'];threads=case['threads'];chains=p['chains'];batch=p['batch']
    if ptx.startswith('fma.'):
        dtype=ptx.split('.')[-1];lanes=2 if dtype.endswith('x2') else 1
        require(dtype==p['input_type'] and p['shape'] is None and p['packed_lanes']==lanes,'FMA type/lanes')
        per_collective=2*threads*lanes;outputs=threads*chains*lanes;width=1
    else:
        shape=re.search(r'\.m(\d+)n(\d+)k(\d+)\.',ptx);require(shape is not None,'matrix shape')
        m,n,k=map(int,shape.groups());require(p['shape']==[m,n,k],'matrix shape metadata')
        width=128 if ptx.startswith('wgmma.') else 32
        require(threads%width==0,'partial collective')
        per_collective=2*m*n*k*(threads//width);outputs=m*n*(threads//width)*chains
    work=blocks*case['iterations']*batch*chains*per_collective
    return {'blocks':blocks,'read':0,'write':0,'work':work,
            'operations':blocks*case['iterations']*batch*chains*(threads//width),'checked':blocks*outputs}


def validate_trial(row,case,device,seed,protocol):
    occupancy=row.get('occupancy_limit_ctas_per_sm')
    require(type(occupancy) is int and 1<=occupancy<=32,'occupancy result')
    registers=row.get('registers_per_thread');smem=row.get('static_smem_bytes')
    require(type(registers) is int and 1<=registers<=255 and type(smem) is int and smem>=0,'kernel resources')
    require(row.get('local_size_bytes')==0,'local/spill forbidden')
    blocks=1 if case['scope']=='one_cta' else device['sms']*min(4,occupancy)
    expected=arithmetic(case,blocks)
    expected['correctness']={'method':'uniform_and_nonuniform_full_output_v1','checked_elements':expected['checked'],
        'input_conditions':'legacy uniform constants; separate seeded nonuniform validation; D reset each launch'}
    require(row.get('output_elements_checked')==expected['checked'] and row.get('max_abs_error')==0,'full uniform output validation')
    require(row.get('timing_model')==TIMING and row.get('phase')==case['parameters']['phase'],'timing/phase')
    reference='integer_dyadic_rne_fma_v1' if case['parameters']['ptx'].startswith('fma.') else 'integer_dyadic_logical_matrix_v1'
    require(row.get('nonuniform_validation')=={'iterations':[1,2],'checked_elements':expected['checked']*2,
            'input_seed':seed,'errors':0,'reference_model':reference},'nonuniform correctness evidence')
    return validate_observation(row,case,device,seed,protocol,expected)


def resolve_iterations(pilot_raw,case,device):
    validate_device(device);p=case['iteration_policy']
    require(p['kind']=='calibrated' and p['rounding_model']=='legacy_event_ms_truncate_then_clamp_v1','unknown calibration')
    require(pilot_raw.get('case_id')==case['id'] and pilot_raw.get('iterations')==p['pilot_iterations'],'pilot identity/length')
    ms=pilot_raw.get('event_ms')
    require(type(ms) in (int,float) and math.isfinite(ms) and ms>0,'pilot event duration')
    result=int(p['pilot_iterations']*(p['target_ns']/1e6)/max(ms,p['minimum_pilot_ns']/1e6))
    result=max(p['min_iterations'],min(p['max_iterations'],result))
    return {'resolved_iterations':result,'rounding_model':p['rounding_model'],
            'pilot_iterations':p['pilot_iterations'],'pilot_event_ms':ms,
            'legacy_observed_iterations':case['legacy']['actual_iterations'],
            'formula_parameters':{k:p[k] for k in ('target_ns','min_iterations','max_iterations','minimum_pilot_ns')}}


def fma_reference(kind,thread,chain,lane,seed,iterations,batch,nonuniform):
    """Integer oracle independent of the C++ implementation and GPU encodings."""
    precision={'f16':11,'f16x2':11,'bf16':8,'bf16x2':8,'f32':24,'f64':53}[kind]
    value=((thread+chain+lane+seed)%7)<<64 if nonuniform else 0
    add=((1+(thread+3*chain+lane+seed)%7)<<63) if nonuniform else 1<<66
    # Here the common denominator is 2^68 (C++ uses 2^64).
    for _ in range(iterations*batch):
        next_value=(value//2)+add
        discarded=max(0,next_value.bit_length()-precision)
        quantum=1<<discarded;quotient,remainder=divmod(next_value,quantum)
        rounded=(quotient+int(2*remainder>quantum or (2*remainder==quantum and quotient%2)))*quantum
        if rounded==value:break
        value=rounded
    return value/(1<<68)


def audit_sass(text,contract):
    functions=re.split(r'Function\s*:\s*',text)[1:];out=[];seen=set()
    for case in contract['cases']:
        symbol=kernel_symbol(case)
        if symbol in seen:continue
        seen.add(symbol);matches=[f for f in functions if symbol in f.splitlines()[0]]
        require(len(matches)==1,'missing/ambiguous compute SASS: '+symbol)
        body=matches[0];require(not re.search(r'\b(?:LDL|STL)(?:\.|\s)',body),'local/spill SASS')
        token=('FFMA' if case['parameters']['input_type']=='f32' else 'DFMA' if case['parameters']['input_type']=='f64' else 'HFMA2') if case['parameters']['ptx'].startswith('fma.') else 'HGMMA' if case['parameters']['ptx'].startswith('wgmma.') else 'DMMA' if case['parameters']['input_type']=='f64' else 'HMMA'
        instructions=[]
        for line in body.splitlines():
            match=re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?)(?:/\*|$)',line)
            if match:instructions.append((int(match[1],16),match[2].strip()))
        timers=[pc for pc,op in instructions if 'SR_GLOBALTIMERLO' in op]
        require(len(timers)==2,'two globaltimer reads per specialization')
        timed=[(pc,op) for pc,op in instructions if timers[0]<pc<timers[1]]
        computes=[pc for pc,op in timed if re.search(r'\b'+token+r'(?:\.|\s)',op)]
        require(computes,'timed compute missing')
        barriers=[pc for pc,op in timed if re.search(r'\bBAR\.',op) and not op.startswith('@')]
        require(any(pc<min(computes) for pc in barriers),'post-start unconditional CTA barrier missing')
        require(any(pc>max(computes) for pc in barriers),'post-compute drain CTA barrier missing')
        for lo,hi,loop in timed_loops(body):
            count=sum(bool(re.search(r'\b'+token+r'(?:\.|\s)',op)) for _,op in loop)
            require(count>0,'arithmetic missing from timed back-edge')
            out.append({'function':symbol,'start':lo,'stop':hi,'token':token,'static_count':count})
        if token=='HGMMA':
            waits=[int(v,16) for v in re.findall(r'WARPGROUP\.DEPBAR\.LE\s+[^,]+,\s*0x([0-9a-fA-F]+)',body)]
            require(case['parameters']['wait'] in waits,'WGMMA pending wait immediate missing')
            if case['parameters']['drain']['post_loop_wgmma_wait']==0:require(0 in waits,'WGMMA final wait0 missing')
    return out
