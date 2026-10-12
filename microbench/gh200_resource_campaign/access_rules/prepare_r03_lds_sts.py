#!/usr/bin/env python3
"""Create the prelisted development and independent address holdout, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

from r03_lds_sts_model import demand, validate_map

BASE = Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign')
OLD_EXP = BASE/'20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/repo/microbench/gh200_resource_campaign'
OLD_R03 = BASE/'access_rules/20261006-c-job735059/r03-formal-v1/source'
CUTLASS = BASE/'access_rules/20261009-V09-freeze-job738972-r2/source/cutlass'


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def make_case(op,width,warps,dependent,layout,seed,iterations,native=False):
    words=width//4
    rng=random.Random(seed)
    addresses=[]
    if layout.startswith('stride'):
        stride=int(layout.removeprefix('stride'))
        offsets=[lane*stride*words for lane in range(32)]
    elif layout=='broadcast':
        offsets=[0]*32
    elif layout=='random_scatter':
        offsets=rng.sample(range(4096//words),32)
        offsets=[x*words for x in offsets]
    elif layout=='random_balanced':
        # Random row/bank combinations, not just a lane permutation of stride.
        offsets=[words*(lane%(32//words)+rng.randrange(1,100)*(32//words)) for lane in range(32)]
        if len(set(offsets))!=32:
            return make_case(op,width,warps,dependent,layout,seed+1,iterations,native)
    elif layout=='cutlass_iterator':
        offsets=[(lane//4)*72+(lane%4)*2 for lane in range(32)]
    else:
        raise ValueError(layout)
    for q in range(8):
        addresses.append([warp*4096+offset+(q*8 if layout=='cutlass_iterator' else 0)
            for warp in range(warps) for offset in offsets])
    case=dict(id=f"o{op}_b{width}_w{warps}_d{int(dependent)}_{layout}_s{seed}_n{iterations}"+('_native' if native else ''),
        op=op,width=width,warps=warps,dependent=dependent,native=native,layout=layout,
        seed=seed,iterations=iterations,addresses_words=addresses)
    validate_map(case)
    case['demand']=demand(case)
    return case


def main(output):
    output.mkdir(exist_ok=False,parents=True)
    root=Path(__file__).resolve().parent
    source=output/'source';(source/'probes').mkdir(parents=True)
    for name in ('r03_lds_sts_model.py','prepare_r03_lds_sts.py','run_r03_lds_sts.py','analyze_r03_lds_sts.py','probes/r03_lds_sts.cu','probes/r00_common.hpp'):
        shutil.copy2(root/name,source/name)
    # Exact bound old source windows and their immutable include closure.
    for name in ('probes/shared_memory.cu','common/probe_runtime.cuh'):
        target=source/'exp09'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(OLD_EXP/name,target)
    for name in ('probes/r03.cu','probes/r00_common.hpp'):
        target=source/'old-r03'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(OLD_R03/name,target)
    # Copy only the actual compiler include closure for the bound epilogue.
    # nvcc can use the original readonly snapshot locally; remote packaging of
    # required headers is done from compiler dependency output before submission.
    write(output/'cutlass-source.json',dict(readonly_source=str(CUTLASS),
        iterator='include/cutlass/epilogue/warp/tile_iterator_tensor_op.h',
        specialization='GemmShape<32,64,16>, GemmShape<16,8,8>, uint32, RowMajor(stride=72)',
        source_mapping='word=warp*4096+(lane/4)*72+(lane%4)*2+q*8',
        target_instructions_per_warp_round=8,target_width_bytes=8,
        full_epilogue_claimed=False,source_version='CUTLASS 3.9.2 snapshot'))
    dev=[];hold=[]
    for op in (0,1):
        for width in (4,8,16):
            for warps in (1,4,8):
                for dependent in ((False,True) if op==0 else (False,)):
                    layouts=[f'stride{s}' for s in (1,2,4,8,16,32)]
                    if op==0:layouts.append('broadcast')
                    for layout in layouts:
                        for n in (256,2048,8192):
                            dev.append(make_case(op,width,warps,dependent,layout,17,n))
                    for seed,layout in ((2026101121,'random_scatter'),(2026101122,'random_balanced')):
                        hold.append(make_case(op,width,warps,dependent,layout,seed,4096))
    for op in (0,1):
        for warps in (1,4,8):
            hold.append(make_case(op,8,warps,False,'cutlass_iterator',2026101123,4096))
            hold.append(make_case(op,8,warps,False,'cutlass_iterator',2026101123,4096,native=True))
    controls=[make_case(2,4,w,False,'stride1',17,n) for w in (1,4,8) for n in (0,256,2048,8192)]
    write(output/'development-cases.json',dev)
    write(output/'holdout-addresses-prelisted.json',hold)
    write(output/'control-cases.json',controls)
    write(output/'protocol.json',dict(schema='r03-ordinary-smem-window/1',
        measured_window='thread0 clock64 after initialization, then CTA join; eight operations/round; full-lane consumers to volatile sink and CTA join before thread0 end',
        iteration_boundary='one explicit bar.warp.sync per round in every organization; ordinary accesses use PTX volatile to preserve dynamic counts; genuine CUTLASS iterator keeps its original C++ loads/stores',
        dependent_read='returned first word feeds next address via returned-address; real RAW chain, zero numerical displacement',
        independent_read='eight independent per-instruction checksum streams',
        store='address/iteration/seed values, exclusive word owner across the sequence; full region checked after timer',
        initialization='load[word]=shared_address(word); store[word]=poison',
        short_checks=[1,2,9],warmup=dict(minimum=8,maximum=30,tail_count=5,CV_limit=.02,basis='same-CTA clock64 cycles'),
        processes_per_case=3,process_order_seed=2026101130,
        performance_unit='clock64 cycles / single CTA',raw_SMID_retained=True,
        acceptance=dict(median_absolute_relative_error=.05,maximum_absolute_relative_error=.10,
            absolute_cycle_tolerance='declare from development resolution/dispersion before holdout',
            failed_prelisted_conditions_retained=True),
        model_form='fixed_g + N*max(floor_g,offset_g+price_g*hypothesized_batches_per_round)',
        batches='sum max distinct 32-bit words per bank within contiguous 128B lane groups; hypothesis only',
        two_explanations=dict(address_demand='at fixed SASS count/organization, conflict/broadcast maps change conditional cycle demand',
            issue_dependency='at identical maps, independent versus actual dependent streams change time'),
        qualification='single GPU UUID and identical frozen binary/protocol only; full warps; no physical port/batch counter claim'))
    write(output/'source-sha256.json',{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()})
    print(json.dumps(dict(development=len(dev),holdout=len(hold),controls=len(controls),output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
