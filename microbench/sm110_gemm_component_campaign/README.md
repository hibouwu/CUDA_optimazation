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
