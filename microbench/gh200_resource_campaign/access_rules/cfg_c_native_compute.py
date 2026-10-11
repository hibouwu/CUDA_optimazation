"""Experimental static compute envelope reference; not a complete GEMM predictor."""
from fractions import Fraction


def compute_reference(setup, reference):
    if setup.get('gpu_uuid') != reference['gpu_uuid']:
        raise ValueError('same-device reference required; cross-card price transfer unsupported')
    required = dict(config='cfg_c',tile=[256,128,64],cluster=[1,2],stages=4)
    for key,value in required.items():
        if type(setup.get(key)) is not type(value) or setup.get(key)!=value:
            raise ValueError('unsupported cfg_c compute reference organization: '+key)
    k=setup.get('k')
    if type(k) is not int or k<=0 or k%64:
        raise ValueError('positive complete K64 tiles required')
    if (reference['protocol']!='cfg-c-mma-reference-v1' or reference['iterations']!=1024
            or reference['mma_shape']!=[64,128,16] or reference['warpgroups']!=2
            or reference['chains_per_wg']!=2 or reference['per_chain_batch']!=4
            or reference['b_major']!='MN' or reference['wait_group']!=1):
        raise ValueError('incompatible independent instruction reference')
    samples=reference['cycle_samples']
    if len(samples)!=3 or any(type(c) is not int or c<=0 for c in samples):
        raise ValueError('three positive native cycle samples required')
    median=sorted(samples)[1]
    reference_work=2*64*128*16*2*4*2*1024
    work=2*256*128*k
    cycles=Fraction(work*median,reference_work)
    return dict(work_flop=work,aggregate_mma_instructions=4*(k//16),
                reference_cycles_fraction=[cycles.numerator,cycles.denominator],
                reference_cycles=float(cycles),gpu_uuid=setup['gpu_uuid'],
                status='experimental_compute_only',observation_qualified=False,prediction_qualified=False)
