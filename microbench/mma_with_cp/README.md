# Thor TCGen05 MMA With Copy Pipeline

This directory is reserved for TCGen05 MMA microbenchmarks that include a copy
path in the measured workflow.

Planned scope:

- TCGen05 MMA fed by explicit copy / load pipeline.
- SS / AS MMA mode experiments when the input path is part of the benchmark.
- GMEM-to-SMEM or TMA-to-SMEM staging experiments.
- NCU counters that explain copy pipeline stalls and tensor pipe overlap.

Boundary:

- `../mma_compute_only` measures MMA completion throughput only.
- This directory should include copy pipeline setup, overlap, and feed behavior.
