"""Preserve complete process vectors before projecting any median input arrays."""
import math
import re


def assemble(protocol, records):
    names=protocol['series'];n=protocol['repeats_per_arm']
    if len(names)!=len(set(names)) or not names or type(n) is not int or n<1:
        raise ValueError('invalid fixed series family or repetition count')
    expected={(cohort,arm,block) for cohort in ('off_on','off_off') for arm in (0,1) for block in range(n)}
    if len(records)!=len(expected):
        raise ValueError('incomplete or extra process records')
    units={};identities=set();raw_hashes=set()
    for record in records:
        if type(record['arm']) is not int or type(record['block']) is not int:
            raise ValueError('integer block and arm required')
        key=(record['cohort'],record['arm'],record['block'])
        if key not in expected or key in units:
            raise ValueError('unknown or duplicate process unit')
        identity=record['id'];digest=record['raw_sha256']
        if not isinstance(identity,str) or not identity or identity in identities:
            raise ValueError('unique nonempty process ID required')
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest) or digest in raw_hashes:
            raise ValueError('unique raw process SHA required; copied traces are not repetitions')
        if record['gpu_uuid']!=protocol['gpu_uuid'] or record['binary_sha256']!=protocol['binary_sha256']:
            raise ValueError('device/binary identity mismatch')
        values=record['series']
        if set(values)!=set(names):
            raise ValueError('complete fixed coordinate vector required')
        for value in values.values():
            if not isinstance(value,list) or len(value)!=2:
                raise ValueError('recorded [ns, native-cycle-or-null] required')
            if type(value[0]) not in (int,float) or not math.isfinite(value[0]):
                raise ValueError('invalid ns value')
            if value[1] is not None and (type(value[1]) not in (int,float) or not math.isfinite(value[1])):
                raise ValueError('invalid cycle value')
        identities.add(identity);raw_hashes.add(digest);units[key]=record
    samples={cohort:{name:{'arm0':[],'arm1':[]} for name in names} for cohort in ('off_on','off_off')}
    ordered=[]
    for cohort in ('off_on','off_off'):
        for block in range(n):
            for arm in (0,1):
                record=units[cohort,arm,block];ordered.append(record)
                for name in names:samples[cohort][name]['arm'+str(arm)].append(record['series'][name][0])
    return samples,ordered
