# SM110 exact GEMM input-resource campaign

This campaign measures schedule-matched TMA input transport for the GEMM
performance model.  It does **not** remeasure the shared L2 architectural
ceilings (`1024 B/cycle` read and `512 B/cycle` write), and it does not promote
any measured rate into a physical upper bound.  Its outputs are empirical
schedule capacities that remain intersected with the independently supplied
shared-L2 and HBM hard ceilings.

## Frozen case matrix

`contract_manifest.json` defines nine transport families.  Crossing those
families with row strides `K=1024, 2048, 4096` and two residency scopes creates
exactly 54 cases:

- generic `M128N64/N128/N256 K64`, stage 2, 8-bit transport;
- generic `M128N64/N128/N256 K64`, stage 2, 32-bit TF32 transport;
- packed 4-bit `M128N256K64`, stage 2, with block-16 or block-32 scale
  transport;
- the existing FP16/BF16 `tc5a M128N256K64`, stage 4 anchor.

The manifest covers exactly the model's twelve precision IDs.  Precision IDs
are metadata on a transport family, not permission to reuse a numerically
similar family.  Raw E2M1/FP6 direct-SMEM schedules use an 8-bit transport
container.  MXFP4 and NVFP4 use packed 4-bit values plus separate physical
scale atoms.  The block-32 scale case retains the four-group tensor-core scale
atom even when only two groups are logically used by `K=64`.

Each case records the tile, value width, scale block, request count, swizzle,
thread/controller assignment, stages, SMEM bytes, row stride, working set,
allocation bytes, grid scope, warmup, iteration count, and rate numerator.
Every backing allocation is zeroed with `cudaMemset` before the kernel starts;
the initialization is outside the device-timed interval and prevents sparse
row-stride first-touch page establishment from being counted as TMA service.
The binary's `--contract-only` mode recomputes all 54 contracts without a GPU;
the runner rejects any disagreement with the independently generated Python
matrix before hardware collection begins.

## What hot and cold mean

The two scopes are deliberately not interchangeable:

- `hot_l2`: one CTA total, exactly one observed SM ID, 16 MiB requested
  payload working set, and a warmup that traverses the complete working set.
  The resulting resource is a **per-SM TMA-to-SMEM ingress** capacity.
- `cold_dram`: one CTA on each of exactly 20 observed SM IDs, 64 MiB requested
  payload working set (twice Thor's measured 32 MiB L2 capacity), and at least
  512 MiB requested payload per timed trial.
  Timing is the earliest CTA `%globaltimer` start to the latest CTA stop.  The
  resulting resource is an **aggregate cold-entry TMA/DRAM path** capacity.

`row_stride_elements` is part of both the case ID and capacity ID because the
same tile payload can exercise a different global address pattern when the
GEMM's leading dimension changes.  Rate counts only requested TMA payload; it
does not count unused stride padding in the backing allocation.

Every hardware case has ten external process trials.  Every trial has a
120-second process-group timeout with bounded TERM/KILL escalation.  Eighteen
representative cases (`K=2048`, every family and both residency scopes) retain
an NCU `basic` report with a separate 300-second timeout.  The NCU hot-L2
variant uses a smaller profile working set but still warms every tile before
the timed/profiled region.  NCU is filtered to the demangled
`tma_ab_contract_kernel` and one launch, so allocation-initialization memsets
cannot be mistaken for target-kernel attribution.  A cold NCU launch requests
at least 128 MiB over a roughly 64 MiB cyclic working set, rather than using a
short loop whose footprint could accidentally fit in the 32 MiB L2.

## Evidence and fail-closed audit

A qualifying result retains:

- immutable run specification and all source-dependency hashes;
- exact compile command/log, retained binary plus SHA-256, and full SASS;
- 54 independently checked static contracts;
- 540 raw hardware trial records and recomputed statistics;
- 18 NCU reports and logs;
- environment, resumable progress journal, terminal status, and COMPLETE hash;
- the source commit needed to read the exact manifest/runner from Git history.

`audit_campaign.py` rebuilds the case matrix independently.  It validates
function-scoped SASS, command suffixes after checkout relocation, immutable Git
blobs, timer and byte arithmetic, 20-SM coverage, trial statistics, NCU hashes,
and all completion records.  Static-only output is rejected as hardware
evidence.  Run the formal audit with both NCU and source-commit requirements:

```bash
python3 microbench/sm110_gemm_resource_campaign/audit_campaign.py \
  "results/sm110_gemm_resource_campaign/$RUN_ID" \
  --require-ncu \
  --expected-commit "$EXPECTED_COMMIT"
```

## Plan and local preflight

The plan command performs no compilation or GPU work:

```bash
python3 microbench/sm110_gemm_resource_campaign/run_resource_campaign.py \
  --run-id inspect --plan
```

The static preflight compiles `sm_110a`, checks the target function's SASS, and
runs all 54 `--contract-only` invocations.  It is not performance evidence:

```bash
python3 microbench/sm110_gemm_resource_campaign/run_resource_campaign.py \
  --run-id local-static --static-only
```

Some newer host glibc/CUDA combinations need
`--nvcc-host-undef-gnu-source`; this only changes host compilation compatibility
and is recorded in the compile command.  It is accepted for local static
preflight only.  The formal hardware auditor deliberately requires the exact
default Thor compile command and rejects extra compiler definitions/options.

## Low-level detached launch

The low-level launcher is resumable and returns immediately:

```bash
bash microbench/sm110_gemm_resource_campaign/launch_resource_campaign.sh \
  "$RUN_ID" --ncu
```

Monitor with:

```bash
cat "results/sm110_gemm_resource_campaign/$RUN_ID/campaign_status.json"
tail -f "results/sm110_gemm_resource_campaign/$RUN_ID/launcher.log"
```

For closure evidence, use the versioned platform wrapper and commands in
`Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md`; a bare result directory
does not establish the required branch, MAXN, locked-clock, or overcurrent
interval.
