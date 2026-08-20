# Optimizing an NVFP4 Blockscaled GEMM on RTX PRO 6000 Blackwell GPU (SM120)
 This article is a continuation of our series on NVFP4 blockscaling on SM12x GPUs. In [Part 1](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/), we covered relevant PTX instructions, scale-factor layout details, and implementation details in CuTe DSL, including how to convert a CUTLASS dense GEMM example into an NVFP4 blockscaled GEMM. In this article, we optimize the NVFP4 GEMM from Part 1 for the NVIDIA RTX Pro 6000 Blackwell Server Edition GPU. We iteratively apply a series of optimizations, outlining the logic behind each one as well as the implementation steps.

We note at the outset that the version from the previous post is already fairly performant at mid-sized problem shapes (e.g., 8k square). Broadly speaking, the optimizations covered in this article fall into two categories:

1. Optimizations targeting well-known issues in small and large problem shape regimes — wave quantization and L2 cache thrashing, respectively.
2. Micro-optimizations, whose cumulative effect will be to lift compute throughput by a few percentage points overall.

At the end of our optimization ladder, we achieve compute throughput gains of 29% at 2k, 6% at 4k, 4% at 8k, 16% at 16k, and 40% at 32k, maxing out at 1666 TFLOP/s for 16k for a utilization rate of 83%.

All benchmarks were produced using Python version 3.13.13, PyTorch version 2.12.1, and nvidia-cutlass-dsl version 4.6.0.

Code for all the optimizations discussed in this article is included in the Colfax Research github repo at [https://github.com/ColfaxResearch/cfx-article-src/tree/master/sm120_nvfp4_gemms](https://github.com/ColfaxResearch/cfx-article-src/tree/master/sm120_nvfp4_gemms).

## RTX Pro 6000 Server Edition Specs

Four processing partitions, each with a warp scheduler and dispatch unit, register file, FP32 and INT32 execution lanes, fifth-generation Tensor Core, load-store units, and a special-function unit. A shared 128 KB L1 data cache and shared-memory block and four texture units appear below. The RT Core is omitted.

STREAMING MULTIPROCESSOR (SM)

L0 i-cache + warp scheduler + dispatch (32 thread/clk)

Register file (16,384 × 32-bit)

FP32 / INT32

5th generation

Tensor Core

LD/ST

LD/ST

LD/ST

LD/ST

SFU

L0 i-cache + warp scheduler + dispatch (32 thread/clk)

Register file (16,384 × 32-bit)

FP32 / INT32

5th generation

Tensor Core

LD/ST

LD/ST

LD/ST

LD/ST

SFU

L0 i-cache + warp scheduler + dispatch (32 thread/clk)

Register file (16,384 × 32-bit)

FP32 / INT32

5th generation

Tensor Core

LD/ST

LD/ST

LD/ST

LD/ST

SFU

L0 i-cache + warp scheduler + dispatch (32 thread/clk)

Register file (16,384 × 32-bit)

FP32 / INT32

5th generation

Tensor Core

LD/ST

LD/ST

LD/ST

LD/ST

SFU

128 KB L1 data cache / shared memory

Tex

Tex

Tex

Tex

Figure 1. RTX Pro 6000 Streaming Multiprocessor (SM) diagram.
 The RTX Pro 6000 has the following specifications:

- 96 GB of GDDR7 memory with ~1.6 TB/s of memory bandwidth
- 24,064 CUDA cores
- 188 Streaming Multiprocessors (SMs)
- 12 Graphics Processing Clusters (GPCs)
- 752 fifth-generation Tensor Cores (4 per SM)
- L1 cache size: 128 KB/SM
- L2 cache size: 128 MB
- Peak FP4 Tensor TFLOP/s with FP32 Accumulate: 2015.2
- Max SM Clock Rate: 2.43 Ghz

## Version 1: The Baseline Kernel

We begin with a quick review of the structure of our kernel from the previous post. The kernel is warp-specialized with a producer-consumer pipeline in which each CTA consists of 1 TMA load warp and 8 MMA warps. The dedicated load warp issues TMA copies into SMEM for the A, B, SFA, and SFB operands. The eight MMA warps wait on those copies before performing SMEM-to-RMEM copies and issuing the appropriate warp-level mma.sync instruction. More specifically, the eight MMA warps form a tiled MMA with 4 warps along M, 2 along N, and 1 along K. The warp-level MMA atom has shape 16 x 8 x 64, so the MMA warps together span a 64 x 16 x 64 tile that is then repeated to cover the 128 x 128 x 128 CTA tile. Since the load warp requires fewer registers, register reallocation is performed with the load warp calling setmaxregister_decrease(40) and the MMA warps calling setmaxregister_increase(232).

After the mainloop, the MMA warps perform the epilogue, writing the output to SMEM, before warp 0 issues the TMA store from SMEM to GMEM. The kernel uses a static persistent tile scheduler in which a single CTA remains in residence on each SM and is repeatedly assigned work tiles.

 We now proceed to evaluate the kernel’s performance. GEMM is a compute-bound problem, so we will evaluate the kernel in terms of measured TFLOP/s, both in absolute terms and as a percentage of the device maximum of 2015.2 TFLOP/s. As a rule of thumb, an optimized GEMM kernel at large problem shapes should achieve a utilization rate of 80% or more.

Figure 2 contains compute throughput numbers for Version 1 derived from the mean runtime of 20 iterations executed after 3 warmup iterations. For an 8k square GEMM, we see 1476 TFLOP/s, or about 73% utilization. Figure 2 also contains performance numbers for the NVFP4 GEMM kernels shipped with the two most recent versions of the cuBLAS library (13.5 and 13.6). For cuBLAS, version 13.5 exclusively dispatched to cutlass backend kernels, while 13.6 changed to nvjet for problem shapes 2k and 32k as well as switching to a different cutlass kernel for 16k.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/v1_vs_cublas_bar_bigfont-scaled.png)

Figure 2. Version 1 vs cuBLAS compute throughput.

Averaged across the five problem shapes, Version 1 achieves approximately 93% of cuBLAS 13.6’s performance, with a notable collapse at 32k. In addition, for the larger shapes we observe the SM clock rate drop to 2.15 GHz. This reduces the clock-adjusted theoretical peak FP4 tensor TFLOP/s from the reported 2015.2 to ~1782. This likely reflects thermal throttling, which is influenced by factors such as problem size and environmental conditions of the hardware.

## Version 2: Threadblock Swizzling

Looking at the performance graph, we can see that our kernel’s performance declines at larger problem sizes. To understand this scaling behavior, let’s use the Nsight Compute profiler. We observe the following quantities related to memory throughput:

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-2.png)

Figure 3. Version 1 memory profiler analysis.

At 2k through 8k, the DRAM throughput hovers between 10% and 14%, while it jumps to between 64% and 86% for 16k and 32k. Moreover, the L2 hit rate falls from 8k to 16k, and sinks to 76.31% for 32k. For a compute-bound problem like GEMM, this jump in bandwidth throughput and drop in L2 hit-rate are indicative of inefficient caching. The bandwidth of L2 cache is roughly 5.4x the bandwidth of GMEM on the RTX Pro 6000 (8.7 TB/s vs 1.6 TB/s), so it is important to increase the likelihood of an L2 hit.

But why is there a steep change above 8k? To better understand these trends, consider the input memory footprints for each of the problem shapes. The inputs are matrices A and B with shapes M x K and K x N, and scale-factor matrices SFA and SFB with shapes M x K/16 and N x K/16 where M=N=K=2048, 4096, 8192, 16384, or 32768. A and B are stored using 4 bits per element and SFA and SFB are stored using 8 bits per element. Thus, the inputs alone for the five problem shapes have a footprint of 4.5 MB, 18 MB, 72 MB, 288 MB, and 1152 MB. The L2 cache size of the RTX Pro 6000 is 128 MB. Thus, in the 8k and below problem shapes, the combined input footprint does not exceed the available storage in L2.

Although the exact L2 eviction policy for NVIDIA GPUs is not published, generally the sooner the same data is reused the more likely it will be resident in L2. Therefore, to increase the L2 hit rate in cases where the whole matrix does not fit in L2, we want CTAs to work on the same data at around the same time. We can achieve this by having the same wave working on the same data through threadblock swizzling.

Recall that CTAs are assigned work tiles of C and load A tiles that are in the same row and B tiles that are in the same column. Put another way, the CTAs with C tiles in the same row need the same A tiles and CTAs with C tiles in the same column use the same B tiles. So if we can have CTAs within a wave access C tiles in the same region, we can get a better L2 hit-rate.

Now suppose we have a hypothetical GPU with 8 SMs, a kernel that launches 8 CTAs, and a C matrix that is tiled to an 8×8 grid. Thus, during each wave we issue 16 operand tile-set loads, where an operand tile set is all in the same row of A or column of B, extending along the k dimension.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-4.png)

Figure 4. Work tile schedule by swizzle size.

If we were to simply assign the work tiles linearly in the m dimension, we would be in the leftmost case of Figure 4. That case requires 9 distinct operand tile sets: 8 for A and 1 for B. So we would very likely see an L2 hit on 7 of the 16 tile-set loads. Swizzling improves this by choosing a mapping from linear index to grid coordinate such that each wave covers a rectangle within the grid rather than a single column. The dimensions of the rectangle are determined by the user-specified swizzle_size. With swizzle 2, we only have 6 distinct tile sets, So we would likely see an L2 hit on 10 of the 16 tile-set loads. With swizzling, there are fewer distinct operand tile sets that are loaded across all CTAs in a wave, encouraging an increase in L2 hit rate.

Now let’s look at our 32k case. With a CTA tile size of 128×128, we have a 256×256 work tile grid. Since there are 188 SMs, we schedule tiles such that each wave spans the integer ceiling of 188/swizzle_size rows and swizzle_size columns of the work tiles.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-5.png)

Figure 5. Distinct tiles loaded per wave by swizzle size.

Figure 5 summarizes the number of distinct A and B tiles loaded in a single wave for several choices of swizzle_size. For this version of the kernel, we set swizzle_size to 16 because it produces the fewest total distinct tiles (28) per wave. In the kernel, this is expressed by passing swizzle_size, together with raster_along_m=True, to PersistentTileSchedulerParams.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/v2_vs_cublas_bar_bigfont-scaled.png)

Figure 6. Version 1 vs Version 2 vs cuBLAS compute throughput.

The largest gain occurs at 32k, where our changes increase throughput by 387 TFLOP/s. Let’s take a look at the memory workload analysis output produced by Nsight for Version 2 to see where some of these gains come from.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-7.png)

Figure 7. Version 2 memory profiler analysis. Changes from Version 1 are shown beneath each value.

Observe the appreciable increase in L2 hit rate for 16k and 32k and the corresponding decrease in DRAM throughput for each. The swizzled schedule assigns work tiles whose operands are more likely to reside in L2, thereby reducing GMEM traffic.

## Version 3: Improving the Epilogue

It is important to ensure things like memory dependency logic do not unnecessarily block operations within the kernel. Let’s consider the epilogue. The kernel already uses a pipelined, asynchronous store, but we may be able to improve it by closely inspecting the pipeline synchronization. In Versions 1 and 2, the store pipeline setup occurs within the work-tile loop:

```
while work_tile.is_valid_tile:
    . . .
    tma_store_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        self.num_mma_warps * self.num_threads_per_warp,
    )
    tma_store_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=self.epi_stage,
        producer_group=tma_store_producer_group,
    )
```

Its setup is therefore repeated for each work tile. Version 3 moves this directly before the work-tile loop.

Next, consider the following block from Versions 1 and 2:

```
for epi_m in cutlass.range_constexpr(epi_rest_m):
    for epi_n in cutlass.range_constexpr(epi_rest_n):
        MmaMPerEpiM = epi_tile_m // mma_tile_m
        MmaNPerEpiN = epi_tile_n // mma_tile_n
        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi
                mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi
                tRS_rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                for elem_idx in cutlass.range_constexpr(cute.size(tRS_rD_slice)):
                    tRS_rD_slice[elem_idx] = tRS_rAcc_slice[elem_idx]
        # Type conversion with alpha scaling
        tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
        acc_vec = tRS_rD.load()
        # Multiply alpha in FP32 before converting to c_dtype
        # to avoid overflow when c_dtype is FP16
        acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
        tRS_rD_out.store(acc_vec)
        # Register to shared memory
        epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(tRS_sD, mode=[3])
        if has_multi_epi_store:
            self.epilog_sync_barrier.arrive_and_wait()
        cute.copy(
            tiled_copy_r2s,
            tRS_rD_out,
            tRS_sD[(None, None, None, epi_buffer)],
        )
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )
        self.epilog_sync_barrier.arrive_and_wait()
        # Copy from shared memory to global memory
        gmem_coord = (epi_m, epi_n)
        if warp_idx == 0:
            cute.copy(
                tma_atom_c,
                bSG_sD[(None, epi_buffer)],
                bSG_gD[(None, gmem_coord)],
            )
            if has_multi_epi_store:
                tma_store_pipeline.producer_commit()
                tma_store_pipeline.producer_acquire()
# Advance to the next work tile
tile_sched.advance_to_next_work()
work_tile = tile_sched.get_current_work()
if has_multi_epi_store:
    tma_store_pipeline.producer_tail()
```

The producer_acquire() call occurs too early. In the current arrangement, producer_acquire() blocks even when there are multiple epilogue stages. Additionally, the producer_tail() sits at the incorrect loop depth. It should drain the store pipeline once, after every work tile has been processed. Instead, as written, it drains once per work tile. Together, these issues cause the store path to stall the MMA work unnecessarily.

In Version 3, our fix is to move the producer tail out of the warp loop and delay the producer_acquire() call until just before the store to shared memory. In particular, it now directly precedes self.epilog_sync_barrier.arrive_and_wait().

We use two epilogue stages with an epilogue subtile size of 64 x 64 in our benchmark. Profiling shows a slight drop in the long scoreboard stalls going from Version 2 to Version 3.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-8.png)

Figure 8. Nsight Compute warp-stall breakdown for Version 3 vs Version 2 (baseline).

Overall, the effect is small and within benchmark variation noise, as shown in Figure 9.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/v3_vs_cublas_bar_bigfont.png)

Figure 9. Versions 1-3 vs cuBLAS compute throughput.

## Version 4: Warp-specialized Store

In a compute-bound kernel like GEMM, we want to keep the Tensor Cores busy at all times. This means we want a warp with MMA work ready to be scheduled at all times. One way to achieve this is through warp-specialization. We already do this for the load path, but we are still using the compute warps to store.

In terms of implementation, the store warp must inform the MMA warps when an SMEM stage is safe to overwrite, and the MMA warps need to tell the store warp when that stage contains a complete output subtile. As such, Version 4 includes two additional named barriers:

```
self.epilog_free_barrier = pipeline.NamedBarrier(
    barrier_id=2,
    num_threads=(self.num_mma_warps + 1) * self.num_threads_per_warp,
)
self.epilog_ready_barrier = pipeline.NamedBarrier(
    barrier_id=3,
    num_threads=(self.num_mma_warps + 1) * self.num_threads_per_warp,
)
```

There are 16 hardware-managed named barriers, and barrier_id identifies which of the 16 to use. The num_threads input specifies the total number of threads that must arrive at this barrier before it releases. We therefore set it to to the total number of threads across both the MMA warps and the store warp.

The store warp begins by calling producer_acquire on its internal TMA store pipeline. It then arrives on epilog_free_barrier, which signals to the MMA warps that the stage is free to write. After waiting on epilog_free_barrier until the signal arrives, the MMA warps fill the stage by copying the accumulator subtile from registers to SMEM. The MMA warps then make the SMEM writes visible by executing an async.shared fence, after which they arrive on epilog_ready_barrier to signal that the stage is filled. After waiting on epilog_ready_barrier, the store warp drains the stage by issuing a TMA copy from SMEM to GMEM. It also calls producer_commit() to record the store, which a later producer_acquire() will wait on.

A high-level overview of the kernel structure of Version 4 is sketched in Figure 10.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-13.png)

Figure 10. Load/compute/store overview of the Version 4 kernel.

The benchmark results for Version 4 are shown in Figure 11.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/v4_vs_cublas_bar_bigfont-scaled.png)

Figure 11. Versions 2-4 vs cuBLAS compute throughput.

We observe ~1% increases in performance at 2k and 16k and unchanged otherwise.

## Version 5: Removing SFA bank conflicts

Version 5 addresses bank conflicts. Although the changes made in this version do not materially effect performance on their own, it is good practice to facilitate efficient loads when possible. We also consider the discussion useful pedagogically.

SMEM is arranged as 32 4-byte-wide banks. If a warp issues an access request to distinct addresses across all 32 banks, SMEM can serve the data in a single cycle. If, however, multiple addresses in the same bank are requested, those requests are serialized. This is referred to as a bank conflict. Note that if the same address in a bank is requested by multiple lanes, data can be served in a single cycle via broadcasting.

The Nsight metric “L1 Wavefronts Shared Excessive” serves as a signal of such bank conflicts. Nsight reports 8.39M for this metric, and the SMEM-to-registers copy of SFA fragments is responsible for all of them. To understand the reason this number is being reported, we must examine both the SMEM layout and the thread-value (TV) layout of SFA.

Recall from our previous post that the SMEM layout of SFA is:

((32,4), REST_M), ((16,4), 1, REST_K)) : (((16, 4), 512 * REST_K), ((0, 1), 4, 512))

The M sublayout (32,4):(16,4) factors the m coordinate of the 128×128 CTA tile as

$$
M = m_0 + 32m_1
$$

where $m_0 = m\pmod{32}$ and $m_1 = \left \lfloor\frac{m}{32} \right \rfloor \pmod 4$ .

The K sublayout (16,4):(0,1) reflects the scale-factor organization. The stride of 0 broadcasts a single scale-factor to the 16 elements of an NVFP4 micro-block. The stride of 1 lays out the four scale factors spanning a K=64 sub-block in four consecutive bytes. Together, the two sublayouts give the scale factor for row m and microblock $b \in \{0,1,2,3\}$ the following byte offset:

$\text{byte offset} = 16m_0 + 4m_1 + b$

The corresponding bank index is equal to

$\text{bank index} = \left \lfloor(\text{byte offset}/4) \right \rfloor \pmod{32}$

Consider the elements of SFA corresponding to rows 0-15. In this case, $m_1 = 0$ and each bank index is a multiple of four:

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-15.png)

Figure 12. Row-to-bank mapping for SFA rows 0-15 under the original SMEM layout.

Observe how the scale factor addresses for row 0 and row 8 both reside in bank 0. In fact, row r and row r+8 more generally have scale factor addresses that share a bank. Thus, there are sixteen distinct addresses across eight distinct banks. This is harmless if a warp does not request addresses for row r and row r+8 scale factors in the same cycle. As we will see, such a request does occur.

The TV layout for the SFA atom is:

((2, 2, 8), 64): ((8, 0, 1), 16)

Recall from our previous post that only two threads per quad actually feed scale-factor data to the MMA instruction. As such, there is a duplication of data in this arrangement. The zero stride in the middle submode means that threads 0 and 2 of each quad hold the same SFA values, as do threads 1 and 3 of each quad.

Figure 13 displays which bank in SMEM each lane requests its SFA data from. The warp accesses eight banks in total, with four lanes in the same quad requesting each bank. Only two distinct addresses per bank are requested, however, so two wavefronts are required to fulfill these requests instead of four. Since there are two wavefronts instead of the ideal one, this counts as one excessive wavefront per instruction.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-17.png)

Figure 13. Bank accessed by each lane in the SFA SMEM-to-register copy under the original layout.

To obtain the 8.39M number reported by Nsight, let’s break down how many such instructions there are for the 8k problem shape. Since M=N=8192 and the CTA tile has shape 128 x 128 x 128, there are (8192/128) x (8192/128) = 64 x 64 = 4096 work tiles. Since K=8192, there are 8192/128=64 mainloop K iterations. The warp-level MMA atom has shape 16 x 8 x 64, and the 8 MMA warps are arranged as (4, 2, 1), covering a 64 x 16 x 64 region of the 128 x 128 x 128 CTA tile. To cover the full CTA tile, each MMA warp performs two M repetitions for each of the two K sub-blocks. This translates to 4 SFA SMEM-to-registers load instructions per mainloop K iteration per MMA warp.

Putting this all together, the number of excessive wavefronts for the SFA load is:

$64 * 64 * 64 * 8 * 4 * 1 = 8,388,608$

This matches the reported 8.39M result.

From a layout perspective, the fix for this issue is straightforward. We modify the stride as follows:

(32, 4):(4, 128)

The rank-2 layout above is equivalent to the rank-1 layout 128:4. In particular, we now have

$\text{byte offset} = 4m_0 + 128m_1 + b = 4(m_0 + 32m_1) + b = 4m + b$

The row to bank index correspondence now looks like:

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-18.png)

Figure 14. Row-to-bank mapping for SFA after the layout change.

Now there is a distinct bank index for each row. We keep the SFA TV layout as before. Our access pattern for the SMEM-to-register copy is now the following:

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-20.png)

Figure 15. Bank accessed by each lane in the SFA SMEM-to-register copy after the change.

Thus, each load of SFA fragments completes in a single wavefront. Note that the cosize of the SFA atom does not change: both layouts are bijections onto the same 512-byte block. The TMA GMEM-to-SMEM copy therefore retains the same box shape and transaction size, with only the byte ordering within an atom changing.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/v5_vs_cublas_bar_bigfont-scaled.png)

Figure 16. Versions 3-5 vs cuBLAS compute throughput.

## Version 6: 12 MMA Warps

At this stage, we have the greatest absolute opportunity for improvement at the 2k problem shape. The changes present in Version 6 are specifically aimed at addressing that gap. Performance at 2k is heavily affected by wave quantization. Dividing the 2048 x 2048 output matrix by 128 x 128 tiles yields a 16 x 16 grid, or 256 work tiles. There are 188 SMs on the RTX Pro 6000, so the calculation requires two waves. The second wave, however, leaves 120 SMs idle.

To address this, Version 6 changes the CTA tile from 128 x 128 to 192 x 128 and expands the MMA layout from (4, 2, 1) to (6, 2, 1). In other words, 12 MMA warps are arranged as 6 warps along M and 2 warps along N. The 16 x 8 x 64 hardware MMA atom and the (6, 2, 1) warp layout together cover a 96 x 16 x 64 tile. This is then repeated twice along M, eight times along N, and twice along K to cover the CTA tile.

Since the output tile size is larger, there is a reduction in the number of work tiles compared to Versions 1-5. The integer ceiling of 2048/192 is 11, so the grid becomes 11 x 16 = 176 work tiles. Version 6 therefore requires one wave, with 12 of the 188 SMs idle.

Figure 17 illustrates how work-tile assignment across waves differs between Versions 1-5 and Version 6.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-21.png)

Figure 17. Work tile assignment by wave for Versions 1-5 and for Version 6 at the 2k problem shape.

Along with the improved work-tile assignment for 2k, using the larger tile size also increases arithmetic intensity.

The CUTLASS helper functions for scale-factor layouts assume a CTA tile with an M extent (tile_m) that is a multiple of 128. Here, the six warps along M cover 6 x 16 = 96 rows. Thus, Version 6 replaces the 128-row SFA atom with a 96-row atom, necessitating the following updated SFA SMEM layout:

((32,3), REST_M), ((16,4), 1, REST_K)) : (((12, 4), 384 * REST_K), ((0, 1), 4, 384))

The stride 12 is present because there are four scale factors per K=64 sub-block and three 32-row blocks covered by the SFA atom (4 x 3 = 12).

Unlike the change in Version 5, which only permutes bytes within an atom of fixed size, Version 6 changes the size of the atom itself, and therefore does require changes to the layout fed to the TMA GMEM-to-SMEM SFA load.

Using four load/MMA pipeline stages, as in previous versions, is not possible with the larger tile size in this version due to SMEM constraints. We therefore use two stages instead.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/v6_vs_cublas_bar_bigfont-scaled.png)

Figure 18. Versions 4-6 vs cuBLAS compute throughput.

Version 6 improves on Version 5 by 186 TFLOP/s at 2k and by 40+ TFLOP/s at both 16k and 32k.

## Version 7: Autotuning

Up to now, kernel parameters have been selected manually and consistently across Versions 1-6. Autotuning performs a large sweep of kernel parameters to identify optimal configurations. We autotune the kernel with respect to the following parameters:

- CTA tile size: bM x bN x bK
- Number of TMA load/MMA compute stages
- Number of MMA warps
- Swizzle size
- Epilogue tile size: epi_m x epi_n
- Number of epilogue pipeline stages
- TMA load warp register allocation

Autotuning identified the following configurations as optimal for the given problem shapes:

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-23.png)

Figure 19. Best autotune configurations by problem shape.

Using the configurations in Figure 19, Version 7 gains a few TFLOP/s at 2k, 4k, and 8k, and 12 to 13 TFLOP/s at 16k and 32k.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/v7_vs_cublas_bar_bigfont-scaled.png)

Figure 20. Versions 5-7 vs cuBLAS compute throughput.

## Final Results Overview

Below we provide a snapshot of the performance progression across all versions discussed above at the problem shapes we have focused on. Overall, Version 7 improves on Version 1 by 29% at 2k, 6% at 4k, 4% at 8k, 16% at 16k, and 40% at 32k.

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-27.png)

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-28.png)

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-29.png)

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-30.png)

![](Imgaes/nvfp4-block-scaled-gemm-sm120/image-31.png)

## Conclusion

In this post, we optimized the tutorial NVFP4 blockscaled GEMM from Part 1 for the RTX Pro 6000 Blackwell Server Edition GPU. Our changes included thread block swizzling to keep a wave’s operands resident in L2, tightening the epilogue store pipeline, moving the store off the MMA warps and to a dedicated store warp, removing SMEM bank conflicts for SFA loads, altering warp layout and tile size to increase arithmetic intensity and remove a wave quantization stall, and an autotuning sweep. The appendix includes results for a version that uses a dynamic scheduler and Cluster Launch Control and nearly matches Version 7. It also discusses some scale-factor experiments that did not improve performance. Overall, Version 7 improves throughput over version 1 at every tested shape, with largest gains at 2k and 32k.

## Appendix

Cluster Launch Control (CLC)

CLC is a hardware-supported feature on NVIDIA Blackwell GPUs that aims to efficiently schedule tiles via dynamic persistent scheduling. In a dynamic persistent tile scheduling scheme, each CTA begins with a programmer-devised initial work-tile assignment, and subsequently fetches and processes new work tiles, if there are any available. For details, please see our [previous post](https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/) on CLC.

Our previous post provides a recipe for implementing CLC for SM100. Version 8 implements CLC using that recipe with several modifications for SM120.

First, clc_cluster_layout_vmnk was constructed as:

```
cute.tiled_divide(cute.make_layout(((1, 1), 1)), (self.tiled_mma.thr_id.shape,))
```

SM100 uses the tcgen05 MMA instruction, so tiled_mma.thr_id.shape is either 1 or 2. SM120 uses a warp-level mma.sync instruction, meaning tiled_mma.thr_id.shape is instead 32. This triggers a CTA cluster path, which is not supported on SM120. This version manually sets that entry to 1.

Second, the implementation in the post defers CLC pipeline initialization and then synchronizes all pipeline barriers together. This version instead has the CLC pipeline handle its initialization and synchronization separately.

Finally, the CLC consumer barrier’s arrival count needed to be altered to account for the larger number of MMA warps here than in the kernel referenced in the post. After autotuning the configuration parameters, the performance of this CLC kernel is nearly identical to Version 7:

![](Imgaes/nvfp4-block-scaled-gemm-sm120/clc_vs_cublas_bar_bigfont-1.png)

Other Scale-Factor Experiments

In this section, we record some attempts at improving performance by manipulating aspects of how scale-factor data is loaded into registers.

The hardware MMA operation that NVFP4 GEMMs ultimately lower to on SM120 consumes scale factors from only a subset of threads in a warp, determined by PTX operands thread-id-a and thread-id-b. If thread-id-a = 0, threads 0 and 1 of each quad supply SFA. If thread-id-a = 1, threads 2 and 3 of each quad supply SFA instead. On the other hand, if thread-id-b = x (0 <= x <= 3), then thread x of each quad supplies SFB.

In CuTe DSL, thread-id-a and thread-id-b are not directly exposed to the programmer, and both are passed as 0 by default. . In Part 1, we described how the SMEM-to-register load patterns for SFA and SFB in our tutorial kernel produce 2x and 4x replication, respectively. This replication is visible in the layouts discussed above. In what follows, we describe two approaches to avoiding it.

The first approach is to restrict which threads actually load SFA fragments into registers. Since thread-id-a is set to 0, the MMA operation only requires SFA fragments in threads 0 and 1 of each quad to proceed. Thus, we may alter the SFA SMEM-to-register load so that only those threads execute it. We do the same for SFB loads. This reduces the amount of data copied, but it also introduces branching among threads in a CTA. Implementing this change resulted in either equal or slightly worse performance.

The second approach is to load more distinct SFA fragments into registers in a single load. While thread-id-a and thread-id-b are not directly alterable in CuTe DSL, inline PTX can be used to change their values manually. With this in mind, rather than stopping threads from loading redundant data, we have the additional threads load SFA and SFB fragments for a separate MMA. We then toggle the values of the thread selectors appropriately before a given MMA. Loading the additional SFA and SFB fragments in this manner, as opposed to the replicated load, did not improve performance.
