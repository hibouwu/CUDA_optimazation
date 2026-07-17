# Blackwell Features

From Hopper to Blackwell, NVIDIA made several incremental improvements to the architecture and changes to the PTX abstractions for MMA-related instructions. We cover most of these in our article *NVIDIA Tensor Core Evolution*. The major notable changes are:

- The introduction of tensor memory (TMEM) to hold MMA accumulators. Threads no longer implicitly own the results of MMA operations and instead, TMEM is explicitly managed at the MMA scope from software.

- `tcgen05` operations are now issued by a single thread on behalf of the entire CTA, rather than at warp or warpgroup scope as in previous generations. You can see this reflected in the CuTe MMA atoms which now use `ThrID = Layout<_1>` in Blackwell instead of `ThrID = Layout<_128>` as in the warpgroup-scoped MMAs of Hopper.

- Support for TPC-scoped TMA and MMA across pairs of coordinating CTAs, exposed as `cta_group::2` in PTX and `2CTA` in SASS, where two SMs making up a TPC can execute `tcgen05.mma` on shared operands, providing access to higher operational intensity MMA instructions by reducing per-CTA SMEM bandwidth requirements. Later we show that this operand sharing is necessary to make use of the available MMA throughput.

- Native support for sub-byte datatypes with micro-scaling.

- Cluster Launch Control (CLC) as hardware support for dynamic work scheduling in persistent-CTA kernels (Covering in future articles).

- Programmatic dependent launch (PDL) was introduced in Hopper to hide launch and setup latency in back-to-back kernels (Covering in future articles).
