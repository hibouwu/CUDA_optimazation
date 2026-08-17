# Thor/SM110 GEMM component closure campaign

This Git-round-trip campaign collects five component families without
conflating them:

- serial, four-request and eight-request-inflight TMA GMEM→SMEM ingress for
  L2-hit and DRAM-stream residency;
- aggregate HBM/L2 read and write paths under explicit working-set contracts;
- block-scale SFA/SFB SMEM→TMEM ingress through
  `tcgen05.cp .32x128b.warpx4`;
- explicit TMEM accumulator readback through `tcgen05.ld` / `LDTM`;
- the NVFP4-specific TMEM→register→E2M1+scale requant epilogue.

The L2-hit TMA ingress cases launch exactly one CTA and directly time one SM's
local TMA-to-SMEM outlet. DRAM-stream TMA, HBM/L2, and TMEM use the full-grid
interval from the earliest CTA `%globaltimer` start through the latest CTA stop
and require exactly 20 distinct SM IDs on the 20-SM T5000 contract. The
epilogue uses CUDA-event timing and additionally
requires all 20 SM IDs plus bit-exact packed-value and scale agreement with the
host reference.
The exact 18-case matrix has ten external trials per case, source/binary/SASS
hashes, the exact compile
command, raw stdout, environment snapshots, and an independent auditor.
The text output and CSV output expose the same launch-contract fields,
including `threads`, measured `iters`, `warmup_iters`, and
`occupancy_blocks_per_sm`; both the runner and the independent auditor reject
missing fields instead of inferring them from the command line.
Every external trial has a 120-second default host timeout. A timeout is a
failed hardware/protocol observation, is recorded in `timeout.json`, and never
silently becomes a low throughput result. The NVFP4 epilogue closure case
launches a one-CTA-per-SM-sized grid and accepts it only when the observed SM-ID
set covers all 20 SMs. Higher CTA/TMEM residency must first pass the bounded
probe because four CTAs allocate the full measured 512-column TMEM capacity.
The compute, component, and full-GEMM campaigns must run sequentially. All
three runners share one non-blocking GPU file lock and reject concurrent work.
The launcher returns immediately and records `launcher.pid`, `launcher.log`,
`campaign_status.json`, and append-only `progress.jsonl`. Reusing the same run
ID safely skips only cases whose ten trials and source/binary/SASS hashes still
match the immutable run contract.

After writing `summary.json`, the runner automatically creates
`plots/component-throughput-by-contract.svg`, `plots/index.md`, and a manifest
bound to the summary SHA-256. Byte and element rates are separated, and
unrelated resource contracts are shown as independent bars rather than a
misleading line. The summary and raw trials remain the audit truth.

Do not launch the 18 cases directly for closure qualification: a bare component
directory does not contain the suite-level branch/commit, locked-clock and
overcurrent interval required by the importer.  Use the same-commit commands
in `THOR_CLOSURE_RUNBOOK.md`: either the complete
`sm110_closure_campaign.sh` flow, or the bounded
`sm110_component_supplement.sh` flow when an unchanged compute/full-GEMM base
suite already exists.  The low-level `launch_component_campaign.sh` remains a
diagnostic/resume helper, not a substitute for those platform contracts.

The existing `08_tmem_consume_bandwidth` and `11_pipeline_overlap` experiments
remain useful measured joint points, but they use a different timer contract
and measure TS MMA consuming TMEM—not accumulator readback. They are not
silently promoted into this closure campaign.

The HBM/L2 read helper consumes all four 32-bit lanes from every 16-B load and
the campaign requires `LDG.E.128`; the write helper requires `STG.E.128` and
places a device-scope fence before its stop timestamp.  The block-scale copy
uses 32 distinct four-column TMEM destination slots per commit batch, so the
reported ingress is not an artifact of overlapping asynchronous writes to two
addresses.
The two original TMA cases freeze `32 KiB × inflight=1`; two cases freeze
`32 KiB × inflight=4`, and two tc5a cases freeze four stages of exact
`A=16 KiB + B=32 KiB` destinations. Each stage uses one 48 KiB mbarrier shared
by its two 2D SW128 requests: four barriers, eight requests in flight. The mixed
pattern occupies 192 KiB of SMEM, matching the `M128N256K64` FP16 tc5a ingress
contract. L2-hit launches one CTA total;
DRAM-stream launches one CTA/SM.
This separates serialized request latency from the sustained ingress available
to a four-stage GEMM pipeline. The serial result is retained under a diagnostic
resource ID. The uniform-four-request result supplies the explicit
`.inflight4` capacity for shallow schedules, while only the exact mixed point
supplies the stage-four `tma.smem_ingress.per_sm`; the two contracts cannot
override each other merely because one numeric rate is larger. No
device-SM-count division is applied. The model applies the selected local rate through task waves; shared `l2.read`
and conditional hard ceilings remain separate and active. DRAM-stream retains
its end-to-end `tma.hbm` aggregate condition.

After any prior GPU fence hang, reboot Thor before running this probe.  Keep the
machine in MAXN with clocks locked, then run the six bounded profiles.  The
first profile uses exactly one CTA to isolate the warp-level TMEM protocol.
The next profile expands to a one-CTA-per-SM-sized grid, followed by the
production shape and higher-residency diagnostics.  The first timeout or
correctness/coverage failure stops the sweep.

```bash
EXPECTED_COMMIT=$(git rev-parse HEAD)
RUN_ID=thor-t5000-epilogue-bounded-20260813-a
python3 microbench/sm110_gemm_component_campaign/run_epilogue_probe.py \
  --run-id "$RUN_ID" \
  --expected-commit "$EXPECTED_COMMIT" \
  --timeout-seconds 30 \
  --max-blocks-per-sm 4
```

The probe records the exact command, source/binary/SASS hashes, raw stdout,
MAXN/identity/clock state, and OC counters immediately before and after every
profile under `results/sm110_epilogue_probe/$RUN_ID`. Both the measurement
timeout and TERM/KILL escalation waits are bounded. `termination_failed=true`
means the process survived the bounded escalation and the machine must be
rebooted before any further GPU work.
The source-dependency manifest includes the benchmark and every requant header
that determines E2M1 encoding, packing, scaling, or TMEM readback. The host
reference retains E2M1 signed zero (`-0` is nibble `0x8`) and self-checks all
RNE midpoint boundaries before launching a GPU kernel.
The closure suite uses `--max-blocks-per-sm 1` as a safe termination and
correctness preflight; this runs the one-CTA isolation profile, the full-GPU
one-CTA-per-SM profile, and the production shape. Use `4` only for the
standalone diagnostic residency sweep; a failure at a higher residency is
evidence, not a reason to run the formal closure at that residency.

For the full closure, use `microbench/sm110_closure_campaign.sh start` instead of
keeping the orchestrator attached to an interactive terminal.  It records a
suite-level PID and log under `results/sm110_closure_suite/<suite-id>/`.
Interrupting `tail -f` then only stops log viewing; it does not interrupt the
suite orchestrator or any active campaign.
The versioned, same-commit start/status/finish/result-upload and platform
restore commands are maintained in
[`THOR_CLOSURE_RUNBOOK.md`](../../Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md).
