# Thor/SM110 TMA payload/residency surface campaign

This is the second, independent Git-round-trip campaign for the GEMM model. It
does not replace or mutate the frozen compute/component/full-GEMM closure suite.
Run it only after that first suite has returned and passed its independent
audits.

The model cannot transfer one 32 KiB TMA rate to every GEMM tile. This campaign
therefore measures the exact cross product below under two rotating SMEM
destination buffers:

- payload: 4, 8, 16, 32, and 64 KiB per TMA operation;
- residency: 16 MiB `hot_l2` and 256 MiB unique `cold_hbm` stream;
- execution scope: every `hot_l2` point uses exactly one CTA on one observed SM
  to isolate the independent per-SM TMA→SMEM exit; every `cold_hbm` point uses
  one CTA on each of all 20 SMs to measure full-GPU DRAM ingress;
- timed work: about 512 MiB requested payload per trial for every case, with
  iteration count derived from payload size; NCU uses about 64 MiB per case;
- evidence: ten external `%globaltimer` trials and NCU for every case.

`destination_slots=2` means that the benchmark alternates two SMEM destination
regions. `inflight=1` keeps this a serialized single-request payload curve; it
does not assert two concurrent TMA operations and is not silently renamed to a
GEMM pipeline-stage count.

The independent auditor recomputes every B/s value and checks the immutable run
spec, exact Git commit, MAXN/Thor environment, source/binary/SASS hashes,
`UTMALDG.3D`, NCU report hashes, TMA/LTS byte counters, and L2 hit versus L2
miss evidence. A working-set-size label alone is not accepted as residency
proof.

The frozen supplement orchestrator runs this campaign with mandatory NCU:

```bash
git fetch origin
git switch codex/sm110-runner-adversarial-audit
git pull --ff-only
RUN_ID=thor-t5000-parameter-supplement-maxn-YYYYMMDD-a
EXPECTED_COMMIT=$(git rev-parse HEAD)
bash microbench/run_sm110_parameter_supplement.sh "$RUN_ID" "$EXPECTED_COMMIT"
```

After the process completes:

```bash
python3 microbench/sm110_tma_payload_campaign/audit_campaign.py \
  "results/sm110_tma_payload_campaign/$RUN_ID-tma-payload"
git switch -c "thor-results/$RUN_ID"
git add -f "results/sm110_tma_payload_campaign/$RUN_ID"
git commit -m "results: Thor SM110 TMA payloads $RUN_ID"
git push -u origin "thor-results/$RUN_ID"
```

The runner automatically writes `plots/tma-throughput-by-payload.svg`,
`plots/index.md`, and `plots/manifest.json` after `summary.json`. Hot-L2 is
explicitly labeled as isolated per-SM ingress and cold-DRAM as full-GPU
ingress, so the two lines are not silently treated as the same physical scope.
The manifest records the source summary SHA-256.

This campaign intentionally does not test FP6 decompression or block-scale
scale-factor delivery. Those require separate source-backed tcgen05 data-path
contracts and cannot be inferred from a generic GMEM-to-SMEM byte-copy curve.

For the revised hierarchy model, `hot_l2` cases provide payload-indexed
`tma.smem_ingress.per_sm.*` capacities. They are not divided by the 20-SM count.
The `cold_hbm` cases provide payload-indexed full-GPU `tma.hbm.*` diagnostic
evidence. Neither family substitutes for the HBM/L2 read+write duplex campaign,
and this serialized payload curve does not replace an exact multi-request
pipeline-topology case such as tc5a's four-stage/eight-request probe.
