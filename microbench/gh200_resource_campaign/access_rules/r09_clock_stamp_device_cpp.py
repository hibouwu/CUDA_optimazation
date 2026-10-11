#!/usr/bin/env python3
"""Compile the isolated R09 device C++ fixture to PTX with NVRTC only.

No CUDA driver/runtime, device allocation, kernel launch or GPU observation.
The caller explicitly chooses the local NVRTC library and CCCL headers. This
is a device-frontend check, not target CUDA12.9/CUTLASS or runtime qualification.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path


def fixture_cpp(sites):
    if sites not in (1,2):
        raise ValueError('one/two-site compile fixture only')
    text='''#include "r09_clock_stamp.cuh"
static_assert(sizeof(uint64_t)==8,"64-bit timestamps required");
static_assert(sizeof(R09ClockStamp)==24,"three-word stamp required");
static_assert(alignof(R09ClockStamp)==8,"eight-byte alignment required");
extern "C" __global__ void r09_clock_stamp_ptx_fixture(uint64_t* output) {
'''
    for site in range(sites):
        text+=f'  R09ClockStamp stamp_{site}=r09_guarded_clock_stamp();\n'
        for offset,field in enumerate(('clock_before','globaltimer_ns','clock_after')):
            text+=f'  output[{site*3+offset}]=stamp_{site}.{field};\n'
    return text+'}\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header',type=Path,required=True)
    parser.add_argument('--nvrtc-library',type=Path,required=True)
    parser.add_argument('--cccl-include',type=Path,required=True)
    parser.add_argument('--sites',type=int,choices=(1,2),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    header=args.header.resolve();library=args.nvrtc_library.resolve();cccl=args.cccl_include.resolve()
    if header.name!='r09_clock_stamp.cuh' or not (cccl/'cuda/std/cstdint').is_file():
        parser.error('explicit candidate header and CUDA CCCL cstdint required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source=fixture_cpp(args.sites);(output/'fixture.cu').write_text(source)
    compiler=ct.CDLL(str(library));program=ct.c_void_p()
    compiler.nvrtcVersion.argtypes=[ct.POINTER(ct.c_int),ct.POINTER(ct.c_int)]
    compiler.nvrtcCreateProgram.argtypes=[ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_char_p,ct.c_int,ct.POINTER(ct.c_char_p),ct.POINTER(ct.c_char_p)]
    compiler.nvrtcCompileProgram.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_char_p)]
    compiler.nvrtcGetProgramLogSize.argtypes=[ct.c_void_p,ct.POINTER(ct.c_size_t)]
    compiler.nvrtcGetProgramLog.argtypes=[ct.c_void_p,ct.c_char_p]
    compiler.nvrtcGetPTXSize.argtypes=[ct.c_void_p,ct.POINTER(ct.c_size_t)]
    compiler.nvrtcGetPTX.argtypes=[ct.c_void_p,ct.c_char_p]
    compiler.nvrtcDestroyProgram.argtypes=[ct.POINTER(ct.c_void_p)]
    major,minor=ct.c_int(),ct.c_int()
    if compiler.nvrtcVersion(ct.byref(major),ct.byref(minor)):
        raise RuntimeError('NVRTC version query failed')
    options=['--gpu-architecture=compute_90a','--std=c++17','--include-path='+str(header.parent),
             '--include-path='+str(cccl/'cuda/std'),'--include-path='+str(cccl)]
    # Resolve cstdint to CUDA's NVRTC-aware implementation, not an injected
    # typedef or a modified candidate header. The selected path is recorded.
    created=compiler.nvrtcCreateProgram(ct.byref(program),source.encode(),b'r09_clock_stamp_fixture.cu',0,None,None)
    if created:
        raise RuntimeError('NVRTC program creation failed: '+str(created))
    try:
        encoded=(ct.c_char_p*len(options))(*[value.encode() for value in options])
        result=compiler.nvrtcCompileProgram(program,len(options),encoded)
        size=ct.c_size_t()
        if compiler.nvrtcGetProgramLogSize(program,ct.byref(size)):
            raise RuntimeError('NVRTC log-size query failed')
        log=ct.create_string_buffer(size.value)
        if compiler.nvrtcGetProgramLog(program,log):
            raise RuntimeError('NVRTC log retrieval failed')
        (output/'nvrtc.log').write_bytes(log.value)
        report=dict(scope='Isolated device C++ frontend check; no CUDA driver/runtime or GPU execution.',
            compile_result=result,sites=args.sites,nvrtc_version=[major.value,minor.value],options=options,
            nvrtc_library=str(library),nvrtc_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            header=str(header),header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
            cstdint_resolution=str(cccl/'cuda/std/cstdint'),
            cstdint_sha256=hashlib.sha256((cccl/'cuda/std/cstdint').read_bytes()).hexdigest(),
            analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_qualified=False)
        (output/'compile-result.json').write_text(json.dumps(report,indent=2)+'\n')
        if result:
            raise RuntimeError('NVRTC compile failed; preserved nvrtc.log, status '+str(result))
        if compiler.nvrtcGetPTXSize(program,ct.byref(size)):
            raise RuntimeError('NVRTC PTX-size query failed')
        ptx=ct.create_string_buffer(size.value)
        if compiler.nvrtcGetPTX(program,ptx):
            raise RuntimeError('NVRTC PTX retrieval failed')
        (output/'fixture.ptx').write_bytes(ptx.value)
        print(json.dumps(dict(compile_result=result,sites=args.sites,nvrtc_version=[major.value,minor.value],
                              header_sha256=report['header_sha256'],runtime_qualified=False),indent=2))
    finally:
        if compiler.nvrtcDestroyProgram(ct.byref(program)):
            raise RuntimeError('NVRTC program cleanup failed')


if __name__=='__main__':
    main()
