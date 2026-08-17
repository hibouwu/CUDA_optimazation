# SM110 full-GEMM closure campaign

This directory freezes the distinction between a working Tensor Core
instruction, a complete GEMM, a numerical reference, and a same-precision
performance denominator. `support_manifest.json` is the machine-readable source
of truth. It intentionally reports gaps instead of substituting a nearby data
type.

The hardware batch contains 15 square `NN`, `beta=0`, no-epilogue,
accumulator-output cases: FP16→FP32, BF16→FP32, TF32→FP32, E4M3→FP32,
and signed INT8→INT32 at
`N=1024,2048,4096`. `N=1024,2048` are calibration points; `N=4096` is frozen as
a holdout point. Each external trial contains the benchmark's own warmups and
timed repetitions, and the campaign stores ten independent external trials.
The TF32 inputs are explicitly rounded round-to-nearest-even to the TF32
mantissa contract before both candidate and reference, with exact halfway cases
checked by a host-only bit-level self-test. The compute-only campaign proves an
SM110 U8 instruction path, but U8 is excluded from this full-GEMM batch: the
official cuBLAS GemmEx support table only admits signed `CUDA_R_8I` inputs for
`CUBLAS_COMPUTE_32I`, so an attempted U8 library call is not a valid closure
reference.

Correctness fails closed on two distinct checks:

- the cuBLAS/cuBLASLt reference is checked against 64 deterministic CPU samples
  computed from the actual quantized input values;
- the candidate's complete output matrix is compared with that checked
  reference (toleranced FP accumulator comparison or exact S8→S32 comparison).

Performance is recomputed by the runner from `2*N^3 / time`, rather than trusting
the legacy `GFLOPS` CSV label. Signed integer work is reported as OP/s, not
FLOP/s. The candidate and library reference are measured in every trial using
the same inputs and precision contract. The SASS proof is function-scoped: the
auditor requires the selected kernel's function block to contain its expected
Tensor Core and store instructions.

Every external custom/reference trial has a 120-second host timeout. Each NCU
holdout collection has a separate 300-second timeout. Timeout handling targets
the complete process group, escalates from `SIGTERM` to `SIGKILL` after five
seconds, records `timeout.json`, and fails the campaign. A recorded
`termination_failed=true` means the process group survived both bounded waits;
reboot the machine before running more GPU work. Successful trial and NCU
records retain the timeout contract so the independent auditor can reject
unbounded or contract-changing evidence.

Audit the current implementation coverage:

```bash
python3 microbench/sm110_full_gemm_campaign/audit_support_manifest.py
```

Current closure-ready paths are FP16→FP32, BF16→FP32, TF32→FP32, FP8
E4M3→FP32, and signed INT8→INT32. E5M2×E5M2 has a native candidate but no
supported cuBLASLt same-contract reference; the official FP8 type table does not
list that A/B pair. Both FP6 encodings, raw unscaled E2M1, and unsigned INT8
still need complete references. The current MXFP4 and
NVFP4 CUTLASS example paths are marked partial because their output contracts
do not match the model's FP32 output contract, their generated external source
was not captured in the historical result bundle, and the historical runner
divides by an FP16 cuBLAS result.

## Thor Git round trip

Do not launch this concurrently with the compute-only or component campaign.
All three runners take the same non-blocking global GPU lock and will reject a
concurrent launch, so sequential execution is enforced mechanically.
To collect all three campaigns with one foreground command, prefer
`microbench/run_sm110_closure_suite.sh`; it waits for and audits each detached
runner before launching the next one.

From the repository root on Thor:

```bash
RUN_ID=thor-t5000-full-gemm-maxn-20260812-a
git fetch origin
git switch codex/thor-sm110-gemm-bounds
git pull --ff-only
bash microbench/sm110_full_gemm_campaign/launch_full_gemm_campaign.sh "$RUN_ID"
```

Before launch, switch the machine to `MAXN`; the runner refuses results unless
`nvpmodel -q` proves `MAXN` and `nvidia-smi` proves Thor/compute capability 11.0.
The launch returns immediately. Monitor it with:

```bash
tail -f "results/sm110_full_gemm_campaign/$RUN_ID/launcher.log"
cat "results/sm110_full_gemm_campaign/$RUN_ID/campaign_status.json"
```

If interrupted, repeat the same launch command with the same run ID. Completed
cases are reused only when their binary and SASS hashes still match. For one NCU
report per precision at the frozen `N=4096` holdout, add `--ncu` to the first
launch. NCU failure fails closed; omit `--ncu` when counter access is unavailable.

After the background runner reaches `complete`:

```bash
python3 microbench/sm110_full_gemm_campaign/audit_campaign.py \
  "results/sm110_full_gemm_campaign/$RUN_ID"
git switch -c "thor-results/$RUN_ID"
git add -f "results/sm110_full_gemm_campaign/$RUN_ID"
git commit -m "results: Thor SM110 full GEMM $RUN_ID"
git push -u origin "thor-results/$RUN_ID"
```

Return the pushed branch name. The result bundle includes the immutable run
specification; environment snapshots; exact compile commands and logs; source,
binary, and SASS hashes; full disassembly; raw stdout and CSV for all 150 trials;
per-case aggregates; optional NCU reports; and a hash-bound `COMPLETE` marker.
The binaries themselves are removed after hashing to avoid committing large
rebuildable executables.

The completed runner automatically writes two figures under `plots/`: absolute
candidate/reference throughput versus `N`, and candidate/reference percentage
versus `N`. Precisions are faceted so FLOP/s and integer OP/s are never mixed
on one axis. `plots/manifest.json` binds the derived SVGs to `summary.json`.

## Local static preflight

This proves compilation and function-scoped SASS only; it does not prove runtime
correctness or performance:

```bash
python3 microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py \
  --run-id local-static --static-only
```
