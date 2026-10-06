"""Finite observations for four lengths; no correctness/performance promotion."""
import json
import math
import struct
from collections import Counter
from common.suite_io import require
from auditors.low_precision_accumulation import compare_target_sass


def audit_values(device,row,contract,baseline_device,baseline_resource,profile_id):
    require(json.dumps(device,sort_keys=True,allow_nan=False)==json.dumps(baseline_device,sort_keys=True,allow_nan=False),'diagnostic device identity changed')
    require(contract['kind']=='accumulation_boundary_diagnostic_design', 'boundary contract kind')
    profiles=contract['profiles']
    require([(p['id'],p['iterations'],p['mathematical_expected']) for p in profiles] ==
            [('original_uniform_'+str(n),n,2*n) for n in (31,32,33,64)], 'fixed boundary profiles')
    matches=[p for p in profiles if p['id']==profile_id]
    require(len(matches)==1, 'unknown boundary profile')
    profile=matches[0]
    require(profile['target_launches']==1 and profile['explicit_auxiliary_launches']==0
            and profile['output_count']==8192, 'one complete target observation')
    expected=2*profile['iterations']
    fixed={'schema_version':1,'type':'accumulation_boundary_diagnostic','profile_id':profile_id,
           'case_id':'wgmma_e4m3_g1_one_cta','target_symbol':'lp_wgmma_e4m3_g1',
           'iterations':profile['iterations'],'seed':3,'threads':128,'blocks':1,'target_launches':1,
           'explicit_auxiliary_launches':0,'mathematical_expected':expected}
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
    differences=sum(value!=expected for value in values)
    histogram=Counter(patterns)
    return {'status':'collected_finite_difference' if differences else 'collected_exact_mathematical_match',
            'case_id':fixed['case_id'],'profile_id':profile_id,'iterations':profile['iterations'],'output_count':len(values),'finite_count':len(values),
            'difference_count':differences,'mathematical_expected':expected,'minimum':min(values),'maximum':max(values),
            'distinct_values':[{'f32_bits':bits,'value':struct.unpack('<f',struct.pack('<I',bits))[0],'count':histogram[bits]} for bits in sorted(histogram)],
            'performance_eligible':False,'family_B3_eligible':False,'numerical_kernel_qualification':False,
            'historical_failure_phase_proven':False}


def aggregate_profiles(rows):
    """Retain every physical output position and exact adjacent-length deltas.

    Output layout comes from the unchanged target store index (thread*2+c)*32+j.
    These are physical thread/fragment positions, not an inferred matrix layout.
    """
    from fractions import Fraction
    lengths=(31,32,33,64)
    names=['original_uniform_'+str(n) for n in lengths]
    require(set(rows)==set(names), 'four boundary rows required for aggregation')
    for n,name in zip(lengths,names):
        row=rows[name]
        require(row['profile_id']==name and row['iterations']==n and len(row['outputs'])==8192,
                'aggregate profile/length/output identity')
        require([item['index'] for item in row['outputs']]==list(range(8192)), 'aggregate ordered indices')
    positions=[]
    delta_histograms=[Counter() for _ in range(3)]
    per_chain={name:[] for name in names}
    for name in names:
        for chain in range(2):
            values=[rows[name]['outputs'][(thread*2+chain)*32+j]['value']
                    for thread in range(128) for j in range(32)]
            per_chain[name].append({'chain':chain,'count':len(values),'minimum':min(values),'maximum':max(values)})
    chain_comparisons={}
    for name in names:
        mismatches=[]
        for thread in range(128):
            for fragment in range(32):
                a=rows[name]['outputs'][(thread*2)*32+fragment]
                b=rows[name]['outputs'][(thread*2+1)*32+fragment]
                if a['f32_bits']!=b['f32_bits']:
                    mismatches.append({'thread':thread,'fragment':fragment,'chain0_index':a['index'],'chain1_index':b['index']})
        chain_comparisons[name]={'bitwise_mismatch_count':len(mismatches),'mismatches':mismatches}
    for index in range(8192):
        values=[rows[name]['outputs'][index]['value'] for name in names]
        deltas=[Fraction(values[i+1])-Fraction(values[i]) for i in range(3)]
        for histogram,delta in zip(delta_histograms,deltas):histogram[delta]+=1
        thread,within=divmod(index,64);chain,fragment=divmod(within,32)
        positions.append({'index':index,'thread':thread,'warp':thread//32,'lane':thread%32,
            'chain':chain,'fragment':fragment,'values':values,
            'f32_bits':[rows[name]['outputs'][index]['f32_bits'] for name in names],
            'adjacent_differences':[{'numerator':d.numerator,'denominator':d.denominator} for d in deltas]})
    return {'profile_order':names,'layout':'index=(thread*2+chain)*32+fragment',
        'position_count':8192,'positions':positions,'per_chain':per_chain,'chain_comparisons':chain_comparisons,
        'adjacent_difference_histograms':[{'from_iterations':lengths[i],'to_iterations':lengths[i+1],
          'differences':[{'numerator':d.numerator,'denominator':d.denominator,'count':hist[d]} for d in sorted(hist)]}
          for i,hist in enumerate(delta_histograms)],'performance_eligible':False,
        'internal_precision_inferred':False,'historical_failure_phase_proven':False}
