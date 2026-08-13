# Thor/SM110 GEMM component closure campaign

This Git-round-trip campaign collects three resources without conflating them:

- TMA GMEM→SMEM ingress for L2-hit and DRAM-stream residency;
- explicit TMEM accumulator readback through `tcgen05.ld` / `LDTM`;
- the NVFP4-specific TMEM→register→E2M1+scale requant epilogue.

TMA and TMEM use the full-grid interval from the earliest CTA `%globaltimer`
start through the latest CTA stop and require exactly 20 distinct SM IDs on the
20-SM T5000 contract. The epilogue uses CUDA-event timing and additionally
requires all 20 SM IDs plus bit-exact packed-value and scale agreement with the
host reference.
Every case has ten external trials, source/binary/SASS hashes, the exact compile
command, raw stdout, environment snapshots, and an independent auditor.
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

On Thor, from the repository root:

```bash
RUN_ID=thor-t5000-components-maxn-20260812-a
git fetch origin
git switch codex/thor-sm110-gemm-bounds
git pull --ff-only
bash microbench/sm110_gemm_component_campaign/launch_component_campaign.sh "$RUN_ID"
```

After the background process finishes:

```bash
python3 microbench/sm110_gemm_component_campaign/audit_campaign.py \
  "results/sm110_gemm_component_campaign/$RUN_ID"
git switch -c "thor-results/$RUN_ID"
git add -f "results/sm110_gemm_component_campaign/$RUN_ID"
git commit -m "results: Thor SM110 GEMM components $RUN_ID"
git push -u origin "thor-results/$RUN_ID"
```

The existing `08_tmem_consume_bandwidth` and `11_pipeline_overlap` experiments
remain useful measured joint points, but they use a different timer contract
and measure TS MMA consuming TMEM—not accumulator readback. They are not
silently promoted into this closure campaign.

After any prior GPU fence hang, reboot Thor before running this probe.  Keep the
machine in MAXN with clocks locked, then run the five bounded profiles.  They
start with a one-CTA-per-SM-sized grid and only then test larger grids; the first
timeout or correctness/coverage failure stops the sweep.

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
The closure suite uses `--max-blocks-per-sm 1` as a safe termination and
correctness preflight.  Use `4` only for the standalone diagnostic residency
sweep; a failure at a higher residency is evidence, not a reason to run the
formal closure at that residency.

For the full closure, use `microbench/launch_sm110_closure_suite.sh` instead of
keeping the orchestrator attached to an interactive terminal.  It records a
suite-level PID and log under `results/sm110_closure_suite/<suite-id>/`.
Interrupting `tail -f` then only stops log viewing; it does not interrupt the
suite orchestrator or any active campaign.
