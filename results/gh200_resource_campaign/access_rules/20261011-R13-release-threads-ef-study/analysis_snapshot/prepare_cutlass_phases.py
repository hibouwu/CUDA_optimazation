#!/usr/bin/env python3
"""Create an isolated CUTLASS header overlay recording consumer phase boundaries.

No original CUTLASS headers are modified. Trace writes are outside the collective mainloop.
The diagnostic must be compared with the original binary before using its service parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r00-archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    archive = args.r00_archive.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    header_name = "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
    original = archive / "source/cutlass/include" / header_name
    text = original.read_text()

    def replace_once(before, after):
        nonlocal text
        if text.count(before) != 1:
            raise ValueError("unexpected CUTLASS source: " + before[:80])
        text = text.replace(before, after)

    replace_once("          collective_mainloop.mma(\n", "          gh200_phase_stamp(0);\n          collective_mainloop.mma(\n")
    replace_once("          // Update starting mainloop pipeline state for the next tile\n",
                 "          gh200_phase_stamp(1);\n          // Update starting mainloop pipeline state for the next tile\n")
    replace_once("          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;\n",
                 "          gh200_phase_stamp(2);\n          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;\n")
    replace_once("    } // Consumer Warp Groups End\n", "      gh200_phase_stamp(3);\n    } // Consumer Warp Groups End\n")
    overlay = source / "overlay" / header_name
    overlay.parent.mkdir(parents=True)
    overlay.write_text(text)
    (source / "diag_trace.hpp").write_text('''#pragma once
#include <cuda_runtime.h>
#include <cstdint>
__device__ uint64_t* gh200_diag_trace_pointer;
__device__ __forceinline__ void gh200_phase_stamp(int phase) {
  if (threadIdx.x == 128 || threadIdx.x == 256) {
    uint64_t cycle, nanos;
    unsigned sm;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycle));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanos));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    unsigned consumer = threadIdx.x / 128 - 1;
    uint64_t* dst = gh200_diag_trace_pointer + (block * 8 + consumer * 4 + phase) * 3;
    dst[0] = cycle;
    dst[1] = nanos;
    dst[2] = sm;
  }
}
''')
    cuda = (archive / "source/probes/r00_cutlass.cu").read_text()
    cuda = '#include "diag_trace.hpp"\n#include <fstream>\n' + cuda
    before = "CUDA_CHECK(cudaGetDeviceProperties(&properties,0));"
    if cuda.count(before) != 1:
        raise ValueError("device setup site")
    cuda = cuda.replace(before, before + '''    DeviceBuffer<uint64_t> trace(size_t(properties.multiProcessorCount) * 24);
    CUDA_CHECK(cudaMemset(trace.pointer, 0, trace.count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(gh200_diag_trace_pointer, &trace.pointer, sizeof(trace.pointer)));
''')
    cuda = cuda.replace("return measure_gemm(o,", "int status = measure_gemm(o,")
    before = "},metadata);\n"
    if cuda.count(before) != 1:
        raise ValueError("measurement end site")
    cuda = cuda.replace(before, before + '''    std::vector<uint64_t> host_trace(trace.count);
    CUDA_CHECK(cudaMemcpy(host_trace.data(), trace.pointer, trace.count * 8, cudaMemcpyDeviceToHost));
    std::ofstream log("diag.json");
    log << "[";
    for (size_t i = 0; i < host_trace.size(); ++i) {
      if (i) log << ",";
      log << host_trace[i];
    }
    log << "]\\n";
    std::vector<float> host_output(d.count);
    CUDA_CHECK(cudaMemcpy(host_output.data(), d.pointer, d.count * 4, cudaMemcpyDeviceToHost));
    std::ofstream full("diag_output.f32", std::ios::binary);
    full.write(reinterpret_cast<const char*>(host_output.data()), host_output.size() * 4);
    return status;
''')
    (source / "r00_cutlass.cu").write_text(cuda)
    (source / "r00_common.hpp").write_bytes((archive / "source/probes/r00_common.hpp").read_bytes())
    (source / "prepare_cutlass_phases.py").write_bytes(Path(__file__).read_bytes())
    manifest = dict(original_header_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    cutlass_source=str(archive / "source/cutlass"),
                    source_hashes={str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in sorted(source.rglob("*")) if f.is_file()},
                    phases=["before collective mma", "after mma tail", "after epilogue store", "after store tail (source reuse)"],
                    boundary="phase3 is source reuse, not proof of global-write completion; CUDA event is full completion")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
