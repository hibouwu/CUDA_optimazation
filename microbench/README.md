# Microbenchmarks for SM110 Thor

This directory contains small, standalone component probes for NVIDIA Thor /
SM110 / `sm_110a`.  The goal is to isolate runtime setup, TCGen05/TMEM bring-up,
CLC-style persistent work scheduling, and TCGen05 MMA behavior before turning
them into a full GEMM.

This is intentionally **not** an SM120, Hopper, Ampere, CUTLASS, or
`mma.sync.aligned.kind::f8f6f4` path.

## Layout

| Folder | Focus |
| --- | --- |
| `00_runtime_sanity` | Minimal CUDA Runtime validation: version, device count, `cudaFree(0)`, properties, 4-byte `cudaMalloc`. |
| `01_tcgen05_tmem_probe` | Minimal TCGen05/TMEM path: allocate TMEM, store one FP32 bit pattern, load it back, write to global memory. |
| `02_clc_persistent_tmem_probe` | Persistent CTA worker probe with static and dynamic CLC-style work-tile assignment, reusing one TMEM allocation per worker. |
| `mma_compute_only` | TCGen05 dense MMA completion-throughput benchmark generation, SASS checks, NCU collection, plotting, and report docs. |
| `mma_with_cp` | Placeholder for TCGen05 MMA benchmarks that include copy pipeline / input-feed behavior in the measured workflow. |
| `sm110_tma_payload_campaign` | Five TMA payload sizes under isolated hot-L2/per-SM and cold-DRAM/full-GPU contracts, mandatory NCU, audit, and automatic SVG. |
| `sm110_memory_duplex_campaign` | GEMM-derived HBM/L2 simultaneous read/write ratios, mandatory NCU, audit, and automatic SVG. |
| `sm110_gemm_campaign` | Twelve-precision compute-only campaign with three full-SM MMA atom shapes per precision. |
| `sm110_gemm_component_campaign` | TMA, HBM/L2, TMEM ingress/readback, and NVFP4 epilogue component contracts. |
| `sm110_full_gemm_campaign` | Numerically checked candidate/reference full-GEMM campaign. |
| `common` | Shared SM110-only helpers and kernels used by the demos. |

## Thor Parameter Supplement: TMA Payload Plus Memory Duplex

`run_sm110_parameter_supplement.sh` is the supported sequential entry point
for the TMA payload and simultaneous HBM/L2 read/write campaigns. It launches
one detached runner at a time, waits for completion, runs the independent
auditor, and starts the second campaign only after the first passes. Both
campaigns share the global SM110 GPU lock and require NCU.

Do not enable `set -euo pipefail` directly in an interactive login shell. An
undefined status variable or a transient read of a changing JSON file can then
exit the entire terminal. Strict mode belongs inside the versioned scripts or
the isolated `bash` block below.

### 1. Freeze the checkout and verify Thor

From the repository root:

```bash
git fetch origin
git switch codex/sm110-closure-plots
git pull --ff-only

EXPECTED_COMMIT=$(git rev-parse HEAD)

test "$(git branch --show-current)" = "codex/sm110-closure-plots"
test -z "$(git status --short --untracked-files=no)"

nvcc --version
ncu --version
nvidia-smi \
  --query-gpu=name,uuid,compute_cap,driver_version,pstate,clocks.current.graphics \
  --format=csv,noheader
nvpmodel -q
```

The hardware run accepts exactly one Thor/SM110 GPU, a clean tracked worktree,
the exact recorded commit, and a power-mode report containing `MAXN`. The
runner does not choose an `nvpmodel` mode ID because that ID is platform-local.

Every new code commit requires a new `RUN_ID`. Never continue writing into an
older directory whose `run_spec.json` freezes another commit or generator
SHA-256. Preserve failed directories as diagnostics instead of relabeling them
as evidence.

### 2. Start a persistent run

Choose a unique ID, for example:

```bash
RUN_ID=thor-t5000-parameter-plots-maxn-YYYYMMDD-b
EXPECTED_COMMIT=$(git rev-parse HEAD)
```

Launch inside an isolated strict shell. The outer interactive terminal remains
alive if a preflight check fails:

```bash
bash <<BASH
set -euo pipefail

RUN_ID="$RUN_ID"
EXPECTED_COMMIT="$EXPECTED_COMMIT"
LOG_DIR="results/sm110_parameter_supplement/$RUN_ID"

test "\$(git branch --show-current)" = "codex/sm110-closure-plots"
test "\$(git rev-parse HEAD)" = "\$EXPECTED_COMMIT"
test -z "\$(git status --short --untracked-files=no)"
nvpmodel -q | grep -qi MAXN

mkdir -p "\$LOG_DIR"
nohup bash microbench/run_sm110_parameter_supplement.sh \
  "\$RUN_ID" "\$EXPECTED_COMMIT" \
  >>"\$LOG_DIR/orchestrator.log" 2>&1 &

orchestrator_pid=\$!
echo "\$orchestrator_pid" >"\$LOG_DIR/orchestrator.pid"
echo "run_id=\$RUN_ID"
echo "pid=\$orchestrator_pid"
echo "log=\$LOG_DIR/orchestrator.log"
BASH
```

The orchestrator derives these result directories:

```text
results/sm110_tma_payload_campaign/<run-id>-tma-payload
results/sm110_memory_duplex_campaign/<run-id>-memory-duplex
```

### 3. Recover variables in every new terminal

Shell variables do not survive a terminal close. Recreate them before any
status or `tail` command:

```bash
set +e
set +u
set +o pipefail 2>/dev/null || true

RUN_ID=thor-t5000-parameter-plots-maxn-YYYYMMDD-b
LOG_DIR="results/sm110_parameter_supplement/$RUN_ID"
TMA_DIR="results/sm110_tma_payload_campaign/${RUN_ID}-tma-payload"
DUPLEX_DIR="results/sm110_memory_duplex_campaign/${RUN_ID}-memory-duplex"
```

Follow the persistent orchestrator log:

```bash
tail -F "$LOG_DIR/orchestrator.log"
```

Use this status reader instead of `python3 -m json.tool` on a file that may be
mid-write. It reports missing or temporarily incomplete JSON and still exits
successfully:

```bash
python3 - "$TMA_DIR" "$DUPLEX_DIR" <<'PY'
import json
import sys
from pathlib import Path

for directory_text in sys.argv[1:]:
    status_path = Path(directory_text) / "campaign_status.json"
    print(f"===== {status_path} =====")
    if not status_path.is_file():
        print("pending")
        continue
    try:
        value = json.loads(status_path.read_text())
    except Exception as error:
        print(f"temporarily unreadable: {error}")
        continue
    print(json.dumps(value, indent=2, sort_keys=True))
PY
```

Check all associated processes before any restart:

```bash
pgrep -af "$RUN_ID" || true
```

Do not launch another orchestrator while an existing runner for the same ID is
alive. If every process is gone and the run is incomplete, first inspect the
logs and failure contract. A `termination_failed=true` record requires a Thor
reboot before more GPU work.

### 4. Completion, audits, and plots

The terminal condition is the literal marker in the orchestrator log:

```bash
grep -F PARAMETER_SUPPLEMENT_COMPLETE "$LOG_DIR/orchestrator.log"
```

Then re-run both independent auditors:

```bash
python3 microbench/sm110_tma_payload_campaign/audit_campaign.py "$TMA_DIR"
python3 microbench/sm110_memory_duplex_campaign/audit_campaign.py "$DUPLEX_DIR"
```

The automatic figures and their source bindings are:

```bash
find "$TMA_DIR/plots" "$DUPLEX_DIR/plots" \
  -maxdepth 1 -type f -print
python3 -m json.tool "$TMA_DIR/plots/manifest.json"
python3 -m json.tool "$DUPLEX_DIR/plots/manifest.json"
```

Expected runtime chart counts are one TMA payload curve and one duplex service
curve. `summary.json`, raw trials, `.ncu-rep`, and raw NCU CSV remain the audit
truth; SVG files are reproducible derived views.

### 5. NCU 2025.3.1 raw CSV diagnostics

NCU 2025.3.1 may emit quoted base-unit values with decimal grouping, such as
`"8,299,136"` ns and `"83,886,080"` byte. The runner and auditor accept only a
strict grouping grammar and require the exact `ns`/`byte`/`sector` unit row.
They reject malformed grouping and scaled `ms`/`Mbyte` imports.

If NCU parsing fails, preserve the complete case directory and inspect without
overwriting `raw.csv`:

```bash
CASE_ID=tma_l2_hit_4k_slots2_single_sm
NCU_DIR="$TMA_DIR/cases/$CASE_ID/ncu"

wc -l "$NCU_DIR/raw.csv"
sed -n '1,50p' "$NCU_DIR/raw.csv"
sed -n '1,160p' "$NCU_DIR/profile.stderr.log"
ncu --version

ncu --import "$NCU_DIR/profile.ncu-rep" --page raw --csv \
  >"$NCU_DIR/raw.import.csv" \
  2>"$NCU_DIR/import.stderr.log"
```

The import is diagnostic only: NCU may rescale the imported page to `ms` and
`Mbyte`. Do not replace the base-unit `raw.csv` with `raw.import.csv`.

### 6. Commit a complete result bundle

Only after both auditors pass:

```bash
RESULT_BRANCH="thor-results/$RUN_ID"
git switch -c "$RESULT_BRANCH"
git add -f "$TMA_DIR" "$DUPLEX_DIR"
git diff --cached --check
git commit -m "results: Thor SM110 parameter plots $RUN_ID"
git push -u origin "$RESULT_BRANCH"
```

Return the result branch, result commit, auditor output, final orchestrator log,
and these hashes:

```bash
sha256sum \
  "$TMA_DIR/summary.json" \
  "$TMA_DIR/plots/manifest.json" \
  "$DUPLEX_DIR/summary.json" \
  "$DUPLEX_DIR/plots/manifest.json"
```

## Host-Side Regression Tests

These tests validate schemas, parsers, auditors, plotting, and evidence gates;
they do not replace a Thor run:

```bash
python3 -m unittest -q \
  scripts.sm110_gemm_model.test_campaign_plots \
  scripts.sm110_gemm_model.test_closure_import \
  scripts.sm110_gemm_model.test_model \
  scripts.sm110_gemm_model.test_runner_coverage

env PYTHONPATH="$PWD" python3 -W error::ResourceWarning \
  microbench/sm110_tma_payload_campaign/test_campaign.py -q

env PYTHONPATH="$PWD" python3 -W error::ResourceWarning \
  microbench/sm110_memory_duplex_campaign/test_campaign.py -q

env PYTHONPATH="$PWD" python3 \
  microbench/sm110_full_gemm_campaign/test_campaign.py -q

git diff --check
```

## Build And Run

Run each component from its own subdirectory entrypoint:

```bash
00_runtime_sanity/build_and_run.sh run
01_tcgen05_tmem_probe/build_and_run.sh run
02_clc_persistent_tmem_probe/build_and_run.sh run 128 1
mma_compute_only/build_and_run.sh run --iters 10000
mma_compute_only/build_and_run.sh ncu
mma_compute_only/build_and_run.sh plot
```

Build only:

```bash
00_runtime_sanity/build_and_run.sh build-only
01_tcgen05_tmem_probe/build_and_run.sh build-only
02_clc_persistent_tmem_probe/build_and_run.sh build-only
```

Clean:

```bash
00_runtime_sanity/build_and_run.sh clean
01_tcgen05_tmem_probe/build_and_run.sh clean
02_clc_persistent_tmem_probe/build_and_run.sh clean
```

The build always uses:

```bash
-DTC3_SM110_HOST_HAS_TCGEN05=1
-gencode arch=compute_110a,code=sm_110a
```

No demo links cuBLAS.  No demo explicitly links the CUDA Driver API.

## Expected Behavior

On a non-SM110 GPU, the TCGen05/TMEM demos should print a clear skip message:

```text
Not SM110-class device. Skip TCGen05/TMEM probe.
```

On a broken CUDA runtime/container setup, `00_runtime_sanity` should fail before
any TCGen05 kernel launch and print the exact CUDA Runtime error code/name/msg.
If `cudaFree(0)` or `cudaMalloc(4)` returns `cudaErrorNotSupported`, the problem
is the CUDA Runtime / driver / container / `libcuda` resolution, not the
TCGen05/TMEM kernel.

## Component Boundary

These demos are probes, not full GEMM kernels:

- No A/B matrix allocation.
- No cuBLAS reference.
- No benchmark input generation.
- No SM120 FP8 MMA path.
- No fake fallback kernel that pretends TCGen05 passed.

`01_tcgen05_tmem_probe`, `mma_compute_only`, and `mma_with_cp` all use
TCGen05/TMEM allocation, but their roles are different:

- `01_tcgen05_tmem_probe` is a minimal bring-up probe. It validates that a
  TCGen05 TMEM allocation plus `tcgen05.st`/`tcgen05.ld` can round-trip one
  FP32 value.
- `mma_compute_only` is a dense TCGen05 MMA completion-throughput benchmark.
  It uses shared-memory descriptors, instruction descriptors, `tcgen05.mma`,
  commit/wait barriers, SASS checks, timing, reporting, and optional NCU/plot
  tooling. Its timed window excludes copy pipeline, TMA, epilogue, TMEM
  readback, and global stores.
- `mma_with_cp` is reserved for TCGen05 MMA benchmarks where copy pipeline or
  input-feed behavior is part of the measured workflow.
