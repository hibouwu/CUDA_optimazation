"""Pure checks for one long FP8 observation; never qualify numerical accuracy."""
import hashlib
import json
import math
import re
import struct
from collections import Counter
from common.suite_io import require


def target_instructions(text,symbol):
    names={'lp_wgmma_e4m3_g1':'_Z16lp_wgmma_e4m3_g1ijbPN2gh5StampEPd'}
    require(symbol in names,'unsupported logical diagnostic target')
    functions=re.split(r'Function\s*:\s*',text)[1:]
    bodies=[f for f in functions if f.splitlines()[0].strip()==names[symbol]]
    require(len(bodies)==1,'exactly one diagnostic target symbol required')
    result=[]
    for line in bodies[0].splitlines():
        match=re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;',line)
        if match:result.append((int(match[1],16),' '.join(match[2].split())))
    require(len(result)>0,'empty target instruction stream')
    return result


def compare_target_sass(text,baseline_text,symbol):
    current=target_instructions(text,symbol);baseline=target_instructions(baseline_text,symbol)
    require(current==baseline,'target SASS differs from reviewed short diagnostic')
    digest=hashlib.sha256(json.dumps(current,separators=(',',':')).encode()).hexdigest()
    return {'status':'identical_target_instructions','kernel_symbol':symbol,
            'instruction_count':len(current),'normalized_sha256':digest,'GPU_numerical_qualification':False}


def audit_values(device,row,contract,baseline_device,baseline_resource):
    require(json.dumps(device,sort_keys=True,allow_nan=False)==json.dumps(baseline_device,sort_keys=True,allow_nan=False),'diagnostic device identity changed')
    require(contract['kind']=='long_accumulation_diagnostic' and contract['iterations']==8192
            and contract['target_launches']==1 and contract['outputs']['count']==8192,
            'fixed long diagnostic contract')
    fixed={'schema_version':1,'type':'long_accumulation_diagnostic','profile_id':contract['profile_id'],
           'case_id':contract['case_id'],'target_symbol':contract['target_symbol'],
           'iterations':8192,'seed':3,'threads':128,'blocks':1,'target_launches':1,
           'explicit_auxiliary_launches':0,'mathematical_expected':16384}
    flags=('warmup_executed','pilot_executed','performance_eligible','family_B3_eligible','numerical_kernel_qualification')
    require(set(row)==set(fixed)|set(flags)|{'resource_identity','stamp','event_ms','outputs'},'diagnostic row schema')
    for key,value in fixed.items():
        require(type(row[key]) is type(value) and row[key]==value,'fixed diagnostic identity: '+key)
    require(all(row[flag] is False for flag in flags),'diagnostic cannot grant qualification or additional execution')
    require(json.dumps(row['resource_identity'],sort_keys=True,allow_nan=False)==json.dumps(baseline_resource,sort_keys=True,allow_nan=False),'target resources changed')
    stamp=row['stamp'];require(set(stamp)=={'begin_ns','end_ns','begin_cycle','end_cycle','smid'},'complete stamp fields')
    for field,value in stamp.items():
        maximum=(1<<32)-1 if field=='smid' else (1<<64)-1
        require(type(value) is int and 0<=value<maximum,'unwritten or invalid stamp: '+field)
    require(stamp['end_ns']>=stamp['begin_ns'] and stamp['end_cycle']>=stamp['begin_cycle'],'reversed stamp')
    require(type(row['event_ms']) in (int,float) and math.isfinite(row['event_ms']) and row['event_ms']>=0,'completion event')
    outputs=row['outputs'];require(isinstance(outputs,list) and len(outputs)==8192,'all8192 outputs required')
    values=[];patterns=[]
    for index,item in enumerate(outputs):
        require(isinstance(item,dict) and set(item)=={'index','value','f32_bits'},'output record fields')
        require(type(item['index']) is int and item['index']==index,'output indices must cover0..8191 exactly once in order')
        value,bits=item['value'],item['f32_bits']
        require(type(value) in (int,float) and math.isfinite(value),'finite output required')
        require(type(bits) is int and 0<=bits<1<<32,'uint32 FP32 bits')
        decoded=struct.unpack('<f',struct.pack('<I',bits))[0]
        require(math.isfinite(decoded) and decoded==value,'decimal/FP32 bits mismatch')
        require(math.copysign(1,decoded)==math.copysign(1,value),'signed zero mismatch')
        values.append(value);patterns.append(bits)
    differences=sum(value!=16384 for value in values)
    histogram=Counter(patterns)
    return {'status':'collected_finite_difference' if differences else 'collected_exact_mathematical_match',
            'case_id':contract['case_id'],'output_count':len(values),'finite_count':len(values),
            'difference_count':differences,'mathematical_expected':16384,'minimum':min(values),'maximum':max(values),
            'distinct_values':[{'f32_bits':bits,'value':struct.unpack('<f',struct.pack('<I',bits))[0],'count':histogram[bits]} for bits in sorted(histogram)],
            'performance_eligible':False,'family_B3_eligible':False,'numerical_kernel_qualification':False,
            'historical_failure_phase_proven':False}
