#!/usr/bin/env python3
"""Generate and run a resumable all-precision Thor tcgen05 compute campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sm110_gemm_model.tcgen05_descriptors import (  # noqa: E402
    DescriptorRecord,
    encode_block_scaled_fp4,
    encode_unscaled,
)


SCHEMA_VERSION = 1
PTX_SOURCE_URL = (
    "https://docs.nvidia.com/cuda/archive/13.0.0/"
    "parallel-thread-execution/index.html#tcgen05-instruction-descriptor"
)
SHAPES = ((128, 64), (128, 128), (128, 256))
LAUNCHES = {
    "single_warp_block": (32, 1, "1"),
    "full_sm_4warp_block": (128, 4, "sm_count"),
}


@dataclass(frozen=True)
class PrecisionCase:
    precision_id: str
    label: str
    kind: str
    element_type: str
    k: int
    instruction: str
    expected_sass: str
    work_unit: str
    input_bits: int
    scale_type: str | None = None
    scale_vector: str | None = None

    @property
    def block_scaled(self) -> bool:
        return self.kind in {"mxf4", "mxf4nvf4"}


PRECISIONS = (
    PrecisionCase("fp16_f32", "FP16", "f16", "f16", 16,
                  "tcgen05.mma.cta_group::1.kind::f16", "UTCHMMA", "flop", 16),
    PrecisionCase("bf16_f32", "BF16", "f16", "bf16", 16,
                  "tcgen05.mma.cta_group::1.kind::f16", "UTCHMMA", "flop", 16),
    PrecisionCase("tf32_f32", "TF32", "tf32", "tf32", 8,
                  "tcgen05.mma.cta_group::1.kind::tf32", "UTCHMMA", "flop", 32),
    PrecisionCase("e4m3_f32", "E4M3", "f8f6f4", "e4m3", 32,
                  "tcgen05.mma.cta_group::1.kind::f8f6f4", "UTCQMMA", "flop", 8),
    PrecisionCase("e5m2_f32", "E5M2", "f8f6f4", "e5m2", 32,
                  "tcgen05.mma.cta_group::1.kind::f8f6f4", "UTCQMMA", "flop", 8),
    PrecisionCase("e3m2_f32", "E3M2", "f8f6f4", "e3m2", 32,
                  "tcgen05.mma.cta_group::1.kind::f8f6f4", "UTCQMMA", "flop", 6),
    PrecisionCase("e2m3_f32", "E2M3", "f8f6f4", "e2m3", 32,
                  "tcgen05.mma.cta_group::1.kind::f8f6f4", "UTCQMMA", "flop", 6),
    PrecisionCase("e2m1_f32", "E2M1", "f8f6f4", "e2m1", 32,
                  "tcgen05.mma.cta_group::1.kind::f8f6f4", "UTCQMMA", "flop", 4),
    PrecisionCase("mxfp4_f32", "MXFP4", "mxf4", "e2m1", 64,
                  "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32",
                  "UTCOMMA", "flop", 4, "ue8m0", "block32"),
    PrecisionCase("nvfp4_f32", "NVFP4", "mxf4nvf4", "e2m1", 64,
                  "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16",
                  "UTCOMMA", "flop", 4, "ue4m3", "block16"),
    PrecisionCase("s8_s32", "S8", "i8", "s8", 32,
                  "tcgen05.mma.cta_group::1.kind::i8", "UTCIMMA", "operation", 8),
    PrecisionCase("u8_s32", "U8", "i8", "u8", 32,
                  "tcgen05.mma.cta_group::1.kind::i8", "UTCIMMA", "operation", 8),
)


CUDA_TEMPLATE = r'''#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); \
  std::exit(1); } } while(0)

static constexpr unsigned kIdesc = @IDESC@u;
static constexpr long long kWorkPerInstruction = @WORK_PER_INST@LL;
static constexpr int kWarpsPerBlock = @WARPS@;
static constexpr int kThreads = @THREADS@;
static constexpr char kCaseId[] = "@CASE_ID@";
static constexpr char kPrecision[] = "@PRECISION@";
static constexpr char kWorkUnit[] = "@WORK_UNIT@";
static constexpr int kLogicalBits = @LOGICAL_BITS@;
static constexpr int kDescriptorBits = @DESCRIPTOR_BITS@;

__device__ __forceinline__ unsigned smem_u32(void const* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ unsigned long long smem_desc(
    void const* p, unsigned leading, unsigned stride) {
  unsigned long long d=0; unsigned a=smem_u32(p);
  d|=static_cast<unsigned long long>((a>>4)&0x3fff);
  d|=static_cast<unsigned long long>(leading&0x3fff)<<16;
  d|=static_cast<unsigned long long>(stride&0x3fff)<<32;
  d|=1ull<<46; return d;
}

__device__ __forceinline__ unsigned long long global_nanoseconds() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void wait_barrier(unsigned long long* b) {
  unsigned a=smem_u32(b); unsigned ticks=0x989680;
  asm volatile("{ .reg .pred p; L_wait: mbarrier.try_wait.parity.shared::cta.b64 "
               "p,[%0],0,%1; @p bra L_done; bra L_wait; L_done: }"::"r"(a),"r"(ticks));
}

__global__ __launch_bounds__(kThreads,1) void benchmark(
    int iters, unsigned long long* cycles_out, unsigned long long* start_ns_out,
    unsigned long long* stop_ns_out, unsigned* smid_out) {
  __shared__ alignas(16) unsigned char a[32768];
  __shared__ alignas(16) unsigned char b[32768];
  __shared__ alignas(8) unsigned long long done;
  __shared__ unsigned tmem;
  for(int i=threadIdx.x;i<32768;i+=blockDim.x){a[i]=(i*13+1)&255;b[i]=(i*17+3)&255;}
  if(threadIdx.x==0) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::
      "r"(smem_u32(&done)),"r"(kWarpsPerBlock));
  __syncthreads();
  if(threadIdx.x<32) asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::
      "r"(smem_u32(&tmem)),"r"(512));
  __syncthreads();
  unsigned long long ad=smem_desc(a,@LEADING@,@STRIDE@);
  unsigned long long bd=smem_desc(b,@LEADING@,@STRIDE@);
  unsigned long long start_ns=global_nanoseconds();
  unsigned long long start=clock64();
  for(int i=0;i<iters;++i){
    unsigned enable=i!=0; unsigned dst=tmem;
    unsigned tsa=tmem+256, tsb=tmem+384;
    if((threadIdx.x&31)==0){
      @MMA@
    }
  }
  if((threadIdx.x&31)==0) asm volatile(
    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::
    "r"(smem_u32(&done)));
  if(threadIdx.x==0) wait_barrier(&done);
  __syncthreads(); unsigned long long stop=clock64();
  unsigned long long stop_ns=global_nanoseconds();
  if(threadIdx.x==0){cycles_out[blockIdx.x]=stop-start;
                    start_ns_out[blockIdx.x]=start_ns;
                    stop_ns_out[blockIdx.x]=stop_ns;
                    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_out[blockIdx.x]));}
  __syncthreads();
  if(threadIdx.x<32){
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(tmem),"r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"::);
  }
}

int main(int argc,char** argv){
  int iters=argc>1?std::atoi(argv[1]):10000;
  double freq=argc>2?std::atof(argv[2]):1575000000.0;
  cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,0));
  int blocks=@BLOCKS@;
  unsigned long long *dc=nullptr,*db=nullptr,*de=nullptr; unsigned *ds=nullptr;
  CUDA_CHECK(cudaMalloc(&dc,blocks*sizeof(*dc)));
  CUDA_CHECK(cudaMalloc(&db,blocks*sizeof(*db)));
  CUDA_CHECK(cudaMalloc(&de,blocks*sizeof(*de)));
  CUDA_CHECK(cudaMalloc(&ds,blocks*sizeof(*ds)));
  cudaEvent_t begin{},end{}; CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
  benchmark<<<blocks,kThreads>>>(1,dc,db,de,ds); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(begin));
  benchmark<<<blocks,kThreads>>>(iters,dc,db,de,ds); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed_ms=0.0f; CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,begin,end));
  auto* hc=new unsigned long long[blocks]; auto* hb=new unsigned long long[blocks];
  auto* he=new unsigned long long[blocks];
  auto* hs=new unsigned[blocks];
  CUDA_CHECK(cudaMemcpy(hc,dc,blocks*sizeof(*dc),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb,db,blocks*sizeof(*db),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(he,de,blocks*sizeof(*de),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hs,ds,blocks*sizeof(*ds),cudaMemcpyDeviceToHost));
  unsigned long long cycles=0,start_min=~0ull,stop_max=0;
  for(int i=0;i<blocks;++i){if(hc[i]>cycles) cycles=hc[i];
                            if(hb[i]<start_min) start_min=hb[i];
                            if(he[i]>stop_max) stop_max=he[i];}
  unsigned long long nanoseconds=stop_max-start_min;
  int unique_smids=0;
  for(int i=0;i<blocks;++i){bool first=true; for(int j=0;j<i;++j) if(hs[j]==hs[i]) first=false;
                            if(first) ++unique_smids;}
  double issued=double(blocks)*kWarpsPerBlock*iters*kWorkPerInstruction;
  double elapsed_seconds=double(nanoseconds)*1.0e-9;
  double rate=issued/elapsed_seconds;
  std::printf("case_id=%s\nprecision_id=%s\nwork_unit=%s\nsm_count=%d\n"
              "logical_input_bits=%d\ndescriptor_storage_bits=%d\nclock_hz=%.0f\n"
              "blocks=%d\nunique_smid_count=%d\nwarps_per_block=%d\niters=%d\n"
              "cycles=%llu\nglobaltimer_start_min=%llu\nglobaltimer_stop_max=%llu\n"
              "globaltimer_nanoseconds=%llu\nelapsed_seconds=%.9e\n"
              "host_kernel_elapsed_seconds=%.9e\nissued_work=%.0f\nrate_per_second=%.9e\n",
              kCaseId,kPrecision,kWorkUnit,p.multiProcessorCount,kLogicalBits,
              kDescriptorBits,freq,blocks,unique_smids,
              kWarpsPerBlock,iters,cycles,start_min,stop_max,nanoseconds,elapsed_seconds,
              double(elapsed_ms)*1.0e-3,issued,rate);
  CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(dc)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(de));
  CUDA_CHECK(cudaFree(ds)); delete[] hc; delete[] hb; delete[] he; delete[] hs; return 0;
}
'''


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(run_dir: Path, status: str, **extra: object) -> None:
    payload = {
        "status": status,
        "pid": os.getpid(),
        "hostname": platform.node(),
        "updated_at_utc": utc_now(),
        **extra,
    }
    (run_dir / "campaign_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (run_dir / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run(command: list[str], *, cwd: Path = REPO_ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if check and proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}\n{proc.stdout}")
    return proc


def tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"required tool not found in PATH: {name}")
    return found


def descriptor(case: PrecisionCase, m: int, n: int) -> DescriptorRecord:
    if case.block_scaled:
        return encode_block_scaled_fp4(case.kind, m=m, n=n,
                                       scale_type=str(case.scale_type), k=case.k)
    return encode_unscaled(case.kind, m=m, n=n, a_type=case.element_type)


def source_for(case: PrecisionCase, m: int, n: int, launch: str, case_id: str) -> tuple[str, DescriptorRecord]:
    record = descriptor(case, m, n)
    threads, warps, blocks_expr = LAUNCHES[launch]
    if blocks_expr == "sm_count":
        blocks_expr = "p.multiProcessorCount"
    # The matrix descriptor encodes offsets in 16-byte units, not element-bit
    # widths. The canonical no-swizzle layouts use 16/8 for 16-bit inputs,
    # 8/4 for the f8f6f4 family (FP6/FP4 values are held in b8 containers on
    # this direct-SMEM path), and 4/2 for packed block-scaled FP4.
    descriptor_bits = 4 if case.block_scaled else 16 if case.input_bits == 16 else 8
    leading = descriptor_bits
    stride = descriptor_bits // 2
    if case.block_scaled:
        mma = (
            'asm volatile("{ .reg .pred p; setp.ne.b32 p,%4,0; '
            f'{case.instruction} [%0],%1,%2,%3,[%5],[%6],p; }}"::'
            '"r"(dst),"l"(ad),"l"(bd),"r"(kIdesc),"r"(enable),"r"(tsa),"r"(tsb));'
        )
    else:
        mma = (
            'asm volatile("{ .reg .pred p; setp.ne.b32 p,%4,0; '
            f'{case.instruction} [%0],%1,%2,%3,{{%5,%6,%7,%8}},p; }}"::'
            '"r"(dst),"l"(ad),"l"(bd),"r"(kIdesc),"r"(enable),'
            '"r"(0),"r"(0),"r"(0),"r"(0));'
        )
    values = {
        "IDESC": str(record.value_u32),
        "WORK_PER_INST": str(2 * m * n * case.k),
        "WARPS": str(warps),
        "THREADS": str(threads),
        "CASE_ID": case_id,
        "PRECISION": case.precision_id,
        "WORK_UNIT": case.work_unit,
        "LOGICAL_BITS": str(case.input_bits),
        "DESCRIPTOR_BITS": str(descriptor_bits),
        "LEADING": str(leading),
        "STRIDE": str(stride),
        "BLOCKS": blocks_expr,
        "MMA": mma,
    }
    source = CUDA_TEMPLATE
    for key, value in values.items():
        source = source.replace(f"@{key}@", value)
    return source, record


def parse_kv(output: str) -> dict[str, str]:
    rows = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key.strip()] = value.strip()
    return rows


def environment() -> dict[str, object]:
    queries = {
        "nvidia_smi": ["nvidia-smi", "-q"],
        "nvidia_smi_identity_csv": [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader",
        ],
        "nvidia_smi_state_csv": [
            "nvidia-smi",
            "--query-gpu=pstate,clocks.current.graphics,power.limit",
            "--format=csv,noheader",
        ],
        "nvcc_version": ["nvcc", "--version"],
        "ncu_version": ["ncu", "--version"],
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
    }
    result: dict[str, object] = {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "power_mode": {
            "source": "nvpmodel -q",
            "returncode": None,
            "output": None,
        },
    }
    for key, command in queries.items():
        proc = run(command, check=False)
        result[key] = {"returncode": proc.returncode, "output": proc.stdout}
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel:
        proc = run([nvpmodel, "-q"], check=False)
        result["power_mode"] = {"returncode": proc.returncode, "output": proc.stdout}
    freq = Path("/sys/class/devfreq/gpu-gpc-0/cur_freq")
    result["gpu_gpc_frequency_hz"] = freq.read_text().strip() if freq.is_file() else None
    return result


def make_manifest() -> list[dict[str, object]]:
    manifest = []
    for case in PRECISIONS:
        for m, n in SHAPES:
            for launch in LAUNCHES:
                case_id = f"{case.precision_id}_m{m}n{n}k{case.k}_{launch}"
                _, record = source_for(case, m, n, launch, case_id)
                manifest.append({
                    "case_id": case_id,
                    "precision": asdict(case),
                    "m": m,
                    "n": n,
                    "launch": launch,
                    "descriptor": record.to_dict(),
                })
    return manifest


def complete_result(
    path: Path, fingerprint: str, trials: int, *, require_ncu: bool = False
) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    base_ok = (
        data.get("status") == "ok"
        and data.get("fingerprint") == fingerprint
        and data.get("trial_count") == trials
    )
    if not base_ok or not require_ncu:
        return base_ok
    ncu = data.get("ncu", {})
    return (
        ncu.get("selected") is True
        and ncu.get("returncode") == 0
        and ncu.get("permission_denied") is False
        and (path.parent / "ncu" / "profile.ncu-rep").is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--host-compiler",
        help="optional nvcc -ccbin path; omit on Thor unless its default host compiler is unsupported",
    )
    parser.add_argument(
        "--nvcc-host-undef-gnu-source",
        action="store_true",
        help="local glibc/CUDA header workaround; normally omit on Thor",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("run-id may contain only letters, digits, dot, underscore, and hyphen")
    if args.trials < 10 or args.iters <= 0:
        parser.error("trials must be >=10 and iters must be positive")
    nvcc, cuobjdump = tool("nvcc"), tool("cuobjdump")
    if not args.static_only:
        tool("nvidia-smi")
    if args.ncu:
        tool("ncu")

    output_root = (
        args.output_root.resolve()
        if args.output_root
        else REPO_ROOT / "results" / "sm110_gemm_campaign"
    )
    run_dir = output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    global_lock = (REPO_ROOT / "results" / ".sm110_gpu_campaign.lock").open("w")
    try:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another SM110 GPU campaign holds the global lock") from error
    prior_status_path = run_dir / "campaign_status.json"
    if prior_status_path.is_file():
        prior = json.loads(prior_status_path.read_text())
        prior_pid = int(prior.get("pid", -1))
        if (prior.get("status") == "running"
                and prior.get("hostname") == platform.node()
                and prior_pid > 0 and prior_pid != os.getpid()):
            try:
                os.kill(prior_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(
                    f"campaign already has a live process on this host: PID {prior_pid}"
                )
    manifest = make_manifest()
    spec = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "campaign": "sm110_dense_tcgen05_compute_only",
        "expected_sm_count": 20,
        "timed_scope": "device_globaltimer_mma_issue_to_completion_barrier",
        "residency": "compute_oracle_smem_operands",
        "trials": args.trials,
        "iters": args.iters,
        "static_only": args.static_only,
        "host_compiler": args.host_compiler,
        "nvcc_host_undef_gnu_source": args.nvcc_host_undef_gnu_source,
        "ncu_policy": "one full-SM M128N256 case per precision",
        "ptx_primary_source": PTX_SOURCE_URL,
        "generator_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "generator_sha256": sha256_path(Path(__file__).resolve()),
        "manifest": manifest,
    }
    spec_text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    spec_path = run_dir / "run_spec.json"
    if spec_path.exists() and spec_path.read_text() != spec_text:
        raise RuntimeError(f"run-id already exists with a different run spec: {run_dir}")
    complete_marker = run_dir / "COMPLETE"
    if complete_marker.is_file():
        complete_marker.unlink()
    write_status(run_dir, "running", current_case=None, completed_cases=0)
    spec_path.write_text(spec_text)
    if not args.static_only:
        environment_snapshot = environment()
        environment_path = run_dir / "environment.json"
        if not environment_path.is_file():
            environment_path.write_text(
                json.dumps(environment_snapshot, indent=2, sort_keys=True) + "\n"
            )
        with (run_dir / "environment_snapshots.jsonl").open("a") as handle:
            handle.write(json.dumps(environment_snapshot, sort_keys=True) + "\n")
    freq_path = Path("/sys/class/devfreq/gpu-gpc-0/cur_freq")
    freq = int(freq_path.read_text().strip()) if freq_path.is_file() else 1575000000

    summaries = []
    for entry in manifest:
        case_id = str(entry["case_id"])
        write_status(
            run_dir,
            "running",
            current_case=case_id,
            completed_cases=len(summaries),
            total_cases=len(manifest),
        )
        case = next(item for item in PRECISIONS if item.precision_id == entry["precision"]["precision_id"])
        source, record = source_for(case, int(entry["m"]), int(entry["n"]), str(entry["launch"]), case_id)
        case_dir = run_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        source_path = case_dir / "source.cu"
        source_path.write_text(source)
        (case_dir / "descriptor.json").write_text(
            json.dumps({**record.to_dict(), "primary_source": PTX_SOURCE_URL}, indent=2, sort_keys=True) + "\n"
        )
        fingerprint = sha256_bytes((source + spec["generator_sha256"] + str(args.iters)).encode())
        result_path = case_dir / "result.json"
        require_ncu = (
            args.ncu
            and entry["launch"] == "full_sm_4warp_block"
            and int(entry["m"]) == 128
            and int(entry["n"]) == 256
        )
        if complete_result(
            result_path, fingerprint, args.trials, require_ncu=require_ncu
        ):
            summaries.append(json.loads(result_path.read_text()))
            print(f"SKIP {case_id}: complete fingerprint match", flush=True)
            continue

        binary = case_dir / "benchmark"
        compile_command = [nvcc, "-O3", "-std=c++17"]
        if args.host_compiler:
            compile_command.extend(["-ccbin", args.host_compiler])
        if args.nvcc_host_undef_gnu_source:
            compile_command.append("-Xcompiler=-U_GNU_SOURCE")
        compile_command.extend(["-gencode", "arch=compute_110a,code=sm_110a",
                                str(source_path), "-o", str(binary)])
        compile_proc = run(compile_command, check=False)
        (case_dir / "compile_command.json").write_text(
            json.dumps(compile_command, indent=2) + "\n"
        )
        (case_dir / "compile.log").write_text(compile_proc.stdout)
        if compile_proc.returncode:
            raise RuntimeError(f"compile failed for {case_id}; see {case_dir / 'compile.log'}")
        (case_dir / "binary.sha256").write_text(sha256_path(binary) + "  benchmark\n")
        sass_proc = run([cuobjdump, "--dump-sass", str(binary)], check=False)
        (case_dir / "sass.txt").write_text(sass_proc.stdout)
        source_dense = "tcgen05.mma.sp" not in source
        sass_match = case.expected_sass in sass_proc.stdout
        if sass_proc.returncode or not source_dense or not sass_match:
            raise RuntimeError(f"static instruction audit failed for {case_id}")

        if args.static_only:
            result = {
                "case_id": case_id,
                "status": "static_ok",
                "fingerprint": fingerprint,
                "precision_id": case.precision_id,
                "work_unit": case.work_unit,
                "timed_scope": spec["timed_scope"],
                "residency": spec["residency"],
                "descriptor_u32": record.value_u32,
                "source_dense": source_dense,
                "expected_sass": case.expected_sass,
                "expected_sass_found": sass_match,
                "source_sha256": sha256_path(source_path),
                "binary_sha256": sha256_path(binary),
                "sass_sha256": sha256_path(case_dir / "sass.txt"),
                "completed_at_utc": utc_now(),
            }
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            summaries.append(result)
            print(f"STATIC PASS {case_id}", flush=True)
            continue

        trial_rows = []
        trials_path = case_dir / "trials.jsonl"
        with trials_path.open("w") as handle:
            for trial in range(1, args.trials + 1):
                proc = run([str(binary), str(args.iters), str(freq)], check=False)
                fields = parse_kv(proc.stdout)
                row = {"trial": trial, "returncode": proc.returncode,
                       "captured_at_utc": utc_now(), "raw_stdout": proc.stdout, "fields": fields}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                if proc.returncode or fields.get("case_id") != case_id:
                    raise RuntimeError(f"trial {trial} failed for {case_id}")
                expected_unique = (
                    20 if entry["launch"] == "full_sm_4warp_block" else 1
                )
                expected_descriptor_bits = (
                    4 if case.block_scaled else 16 if case.input_bits == 16 else 8
                )
                if (
                    int(fields.get("sm_count", "0")) != spec["expected_sm_count"]
                    or int(fields.get("unique_smid_count", "0")) != expected_unique
                    or int(fields.get("logical_input_bits", "0")) != case.input_bits
                    or int(fields.get("descriptor_storage_bits", "0"))
                    != expected_descriptor_bits
                ):
                    raise RuntimeError(
                        f"trial {trial} has unexpected SM residency for {case_id}: "
                        f"sm_count={fields.get('sm_count')} unique={fields.get('unique_smid_count')}"
                    )
                trial_rows.append(row)

        rates = [float(row["fields"]["rate_per_second"]) for row in trial_rows]
        result = {
            "case_id": case_id,
            "status": "ok",
            "fingerprint": fingerprint,
            "precision_id": case.precision_id,
            "work_unit": case.work_unit,
            "timed_scope": spec["timed_scope"],
            "residency": spec["residency"],
            "trial_count": len(rates),
            "rate_per_second_median": statistics.median(rates),
            "rate_per_second_min": min(rates),
            "rate_per_second_max": max(rates),
            "rate_per_second_mean": statistics.fmean(rates),
            "descriptor_u32": record.value_u32,
            "source_dense": source_dense,
            "expected_sass": case.expected_sass,
            "expected_sass_found": sass_match,
            "source_sha256": sha256_path(source_path),
            "binary_sha256": sha256_path(binary),
            "sass_sha256": sha256_path(case_dir / "sass.txt"),
            "completed_at_utc": utc_now(),
        }
        ncu_selected = (
            args.ncu
            and entry["launch"] == "full_sm_4warp_block"
            and int(entry["m"]) == 128
            and int(entry["n"]) == 256
        )
        if ncu_selected:
            ncu_dir = case_dir / "ncu"
            ncu_dir.mkdir(exist_ok=True)
            report = ncu_dir / "profile"
            ncu_iters = min(args.iters, 1000)
            metrics = (
                "gpu__time_duration.avg,"
                "sm__cycles_elapsed.avg,"
                "sm__cycles_elapsed.avg.per_second,"
                "sm__inst_executed_pipe_tensor.sum,"
                "sm__inst_executed_pipe_tensor.sum.per_cycle_active,"
                "sm__throughput.avg.pct_of_peak_sustained_active"
            )
            proc = run(["ncu", "--metrics", metrics, "--target-processes", "all",
                        "--force-overwrite", "--export", str(report), str(binary),
                        str(ncu_iters), str(freq)], check=False)
            (ncu_dir / "profile.log").write_text(proc.stdout)
            report_path = ncu_dir / "profile.ncu-rep"
            ncu_ok = (
                proc.returncode == 0
                and "ERR_NVGPUCTRPERM" not in proc.stdout
                and report_path.is_file()
            )
            result["ncu"] = {
                "selected": True,
                "policy": spec["ncu_policy"],
                "metrics": metrics.split(","),
                "iters": ncu_iters,
                "returncode": proc.returncode,
                "permission_denied": "ERR_NVGPUCTRPERM" in proc.stdout,
                "report_path": "ncu/profile.ncu-rep",
                "log_sha256": sha256_path(ncu_dir / "profile.log"),
                "report_sha256": sha256_path(report_path) if report_path.is_file() else None,
            }
            if not ncu_ok:
                result["status"] = "ncu_failed"
        else:
            result["ncu"] = {
                "selected": False,
                "policy": spec["ncu_policy"],
            }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        binary.unlink()
        summaries.append(result)
        print(f"PASS {case_id}: median={statistics.median(rates):.9e} {case.work_unit}/s", flush=True)

    expected_status = "static_ok" if args.static_only else "ok"
    ok = len(summaries) == len(manifest) and all(
        row["status"] == expected_status for row in summaries
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": ("static_complete" if args.static_only and ok else
                   "complete" if ok else "incomplete"),
        "case_count": len(manifest),
        "passed_count": sum(row["status"] == expected_status for row in summaries),
        "results": summaries,
        "updated_at_utc": utc_now(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if ok and not args.static_only:
        (run_dir / "COMPLETE").write_text(
            f"run_id={args.run_id}\nsummary_sha256={sha256_path(run_dir / 'summary.json')}\n"
        )
    write_status(
        run_dir,
        "static_complete" if args.static_only and ok else "complete" if ok else "incomplete",
        current_case=None,
        completed_cases=len(summaries),
        total_cases=len(manifest),
    )
    return 0 if ok else 1


def mark_failed_from_argv(message: str) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    args, _ = parser.parse_known_args()
    if not args.run_id or not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        return
    root = (
        args.output_root.resolve()
        if args.output_root
        else REPO_ROOT / "results" / "sm110_gemm_campaign"
    )
    run_dir = root / args.run_id
    if run_dir.is_dir():
        write_status(run_dir, "failed", error=message)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        mark_failed_from_argv(str(exc))
        raise
