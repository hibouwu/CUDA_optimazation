"""cfg_b 输出阶段的经验频率推理；参数来源为 phase-frequency-development-001。"""
import math
import numpy as np

FEATURES=['log2_Ktiles','log2_total_nominal_cycles_per_132SM',
    'initial_active_fraction','last_phase_fraction','output_progress',
    'log2_output_index_plus_1','phase_active_fraction','input_zero']

def static_features(case,work,output):
    counts=[len(w) for w in work]
    active=sum(n>0 for n in counts);longest=max(counts)
    kt=math.ceil(case['k']/64)
    if output<0 or output>=longest:raise ValueError('output is outside static work')
    total=sum(counts)*kt*512/132
    return [math.log2(kt),math.log2(total),active/132,
        sum(n==longest for n in counts)/132,
        output/max(1,longest-1),math.log2(output+1),
        sum(n>output for n in counts)/132,
        float(case.get('input_mode','dyadic')=='zero')]

def call_features(case):
    tiles=math.ceil(case['m']/128)*math.ceil(case['n']/128)
    active=min(132,tiles,case.get('sm_count') or 132)
    maximum=math.ceil(tiles/active)
    return [tiles*math.ceil(case['k']/64)*512/132/1e5,
        tiles/(132*maximum),float(case.get('input_mode','dyadic')=='zero'),1.]

def call_formula(parameters,x):
    x=np.atleast_2d(np.asarray(x,dtype=float));p=parameters
    q,phi,z,dual=x.T
    return (p[0]*(1-z)+p[2]*z)-(p[1]*(1-z)+p[3]*z)*phi**p[4]*(1-np.exp(-np.maximum(0,q-p[5])/p[6]))+p[7]*dual

def _linear_basis(x):
    """阶段进度、首段、活动比例、K尺度及零输入的少量交互项。"""
    x=np.asarray(x);k=x[:,0]-6;progress=x[:,4];jlog=x[:,5];active=x[:,2];current=x[:,6];zero=x[:,7]
    first=(jlog==0).astype(float)
    return np.column_stack([first,progress,jlog,current-active,k*first,k*progress,
        active*first,zero*first,zero*progress])

def correction(model,x):
    x=np.atleast_2d(np.asarray(x))
    design=_linear_basis(x) if model['method']=='linear' else x
    a=(design-model['center'])/model['scale']
    if model['method']=='rbf':
        a=np.exp(-np.sum((a[:,None,:]-np.asarray(model['training_features'])[None,:,:])**2,axis=2)/(2*model['length']**2))
    return model['mean']+a@model['weights']

def predict(model,case,work,output):
    if case.get('input_mode','dyadic') not in model['input_modes']:
        raise ValueError('phase frequency supports dyadic and zero inputs')
    baseline=float(call_formula(model['baseline_parameters'],call_features(case))[0])
    return baseline+float(correction(model['correction'],[static_features(case,work,output)])[0])

def predict_phases(model,case,work):
    return [predict(model,case,work,j) for j in range(max(map(len,work)))]
