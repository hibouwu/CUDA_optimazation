# Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs

### Motivation

Consider the matrix multiplication (GEMM) problem

$C=AB$ ,

where $A \in \R^{M\times K}$ , $B \in \R^{K\times N}$ , and $C \in \R^{M\times N}$ . The computation of C is parallelized by dividing the problem shape (M, N, K) by some tile shape (bM, bN, bK), and computing each bM x bN output tile as

$C^{[i,j]} \equal \sum A^{[i,k]}B^{[k,j]}$ .

Each work tile $C^{[i,j]}$ must be assigned to some processor — concretely, a CTA or a cluster of CTAs in the CUDA execution model. The problem of tile scheduling is to determine how to best distribute the collection of work tiles across processors.

This blog post discusses Cluster Launch Control (CLC), a hardware-supported feature on NVIDIA Blackwell GPUs that facilitates optimal tile scheduling, in particular with respect to load balancing. To provide context, we first survey a few common scheduling strategies and the deficiencies CLC is designed to address. We then walk through the implementation-level details of using CLC in a CuTe DSL kernel, and close with a performance comparison for a GEMM kernel.

### Single Tile Scheduling

The most naïve choice for tile scheduling is to launch a grid of clusters of shape (M/bM, N/bN) and assign each work tile to a unique cluster. This is good for load balancing: the grid contains more clusters than there are SM groups, so as each cluster exits, the hardware scheduler dispatches a queued cluster to the now-idle SM group. However, this strategy is often suboptimal overall, since each cluster pays a fixed startup cost — pipeline initialization, descriptor setup, and so on — that is amortized over only a single tile. Moreover, with single tile scheduling we can’t overlap across work tiles to hide latency, such as overlapping one work tile’s epilogue with another’s mainloop.

### Static Persistent Tile Scheduling

On the other hand, we might opt to use a persistent tile scheduling scheme. We will briefly review the concept of persistent tile scheduling, and refer the reader to our [previous article](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) for a more detailed exposition.

In the persistent setting, we launch a grid consisting of as many clusters as can be concurrently scheduled on the GPU. In this case, once a cluster is launched, it will “persist” on the GPU, computing some collection of work tiles. For example, given 148 SMs and a cluster size of 2, we can launch 74 clusters concurrently on the GPU. If we launch a GEMM kernel consisting of 512 work tiles, then we may choose some linear ordering of the work tiles, and have each cluster compute every 74th work tile.

![](Imgaes/blackwell-cluster-launch-control/image-30.png)

Figure 1: The output C of a GEMM is partitioned into a 5 x 6 grid of work tiles, each of which is computed by one of eight clusters. Each work tile is labelled by the cluster to which it is assigned. The work tiles assigned to cluster 0 are highlighted.

The main benefit of persistent tile scheduling is that we can overlap the epilogue of one tile with the mainloop of the next tile; we also avoid the latency of launching new clusters. However, static persistent tile scheduling can lead to load imbalance issues. For example, consider a grouped GEMM, which computes a collection of GEMMs

$C_i = A_iB_i, \quad i = 0,1,\dots,$ `num_problems` $-1$

For instance, we may consider a grouped GEMM consisting of four problems of the following shapes

| Problem 0: | (256, 256, 128) |
|---|---|
| Problem 1: | (256, 256, 2048) |
| Problem 2: | (256, 256, 128) |
| Problem 3: | (256, 256, 2048) |

Note that M = N = 256 for each GEMM, but the contracting dimension is small (K = 128) for some problems and large (K = 2048) for others. Consider a kernel which computes this grouped GEMM using tile shape

(bM, bN, bK) = (128, 128, 128).

If we have enough available resources on the GPU to concurrently launch 8 clusters, we might assign work tiles to clusters as depicted below.

![](Imgaes/blackwell-cluster-launch-control/image-21.png)

Figure 2: Each work tile in our grouped GEMM is assigned to one of eight clusters. In the static persistent case, the assignment is made by linearly ordering the work tiles from all problems, then assigning every 8th work tile to a cluster.

At first glance, this assignment appears to be perfectly balanced, as each cluster computes exactly two work tiles. However, the compute demanded by these work tiles varies from problem to problem: The work tiles from problems 0 and 2 require

$2*\text{bM}*\text{bN}*\text{K} = 2*2^7*2^7*2^7=2^{22}$ `FLOPs`

While the work tiles from problems 1 and 3 require

$2*\text{bM}*\text{bN}*\text{K} = 2*2^7*2^7*2^{11}=2^{26}$ `FLOPs` .

Thus, if we consider the number of FLOPs computed by each cluster, we see a significant load-imbalance:

![](Imgaes/blackwell-cluster-launch-control/image-27.png)

Figure 3: A depiction of the work done by each cluster in the static persistent case, in terms of number of FLOPs computed.

This imbalance motivates dynamic persistent scheduling.

### Dynamic Persistent Tile Scheduling

In this scheduling scheme, each cluster will compute some initial work tile, and then continue to fetch and process new work tiles, if there are any available. Let’s consider how this would avoid the load imbalance we saw in the previous example. Under the reasonable assumption that the time it takes a cluster to process a tile from problem 0 or 2 is drastically less than the time it takes a cluster to process a tile from problem 1 or 3, the assignment of work tiles to clusters might look like

![](Imgaes/blackwell-cluster-launch-control/image-22.png)

Figure 4: Each work tile in our grouped GEMM is assigned to one of eight clusters. In the dynamic persistent case, the assignment is made by linearly ordering the work tiles from all problems, assigning an initial work tile to each cluster, then allowing clusters to fetch new work tiles as they complete their current work.

Note that the programmer cannot control which work tiles are computed by which clusters, beyond the initial assignment. These assignments are determined at runtime by the order in which clusters complete their work. We see that in this case, the number of FLOPs computed by each cluster is more uniformly distributed.

![](Imgaes/blackwell-cluster-launch-control/image-29.png)

Figure 5: A depiction of the work done by each cluster in the dynamic persistent case, in terms of number of FLOPs computed.

<a id="imbalanced-benchmark"></a>

This improved load balancing leads to improved kernel performance. For instance, we can benchmark the grouped GEMM of problem shape

| Problem 0: | (1024, 1024, 1024) |
|---|---|
| Problem 1: | (1024, 1024, K) |
| Problem 2: | (1024, 1024, 1024) |
| Problem 3: | (1024, 1024, K) |

for increasingly large values of K on our B200, which can concurrently support 74 clusters of shape (2, 1). The results in the static and dynamic cases are shown below.

![](Imgaes/blackwell-cluster-launch-control/image-23.png)

Figure 6: Performance of a highly load-imbalanced grouped GEMM with static and dynamic scheduling. The measured config had operand datatype mxfp4 and MMA tile size 256 x 128, using 2CTA MMA instructions.

As expected, when work tiles become highly load-imbalanced, the dynamic scheduler significantly outperforms the static scheduler.

#### Standard implementation of dynamic persistent tile scheduling

To implement dynamic persistent tile scheduling, we need to ensure two properties:

(1) every tile is eventually processed by some cluster, and
(2) no tile is processed by more than one cluster.

A standard strategy is to maintain a global atomic counter (i.e., semaphore lock) that tracks the next unassigned tile. When a cluster finishes its current tile, it performs an atomic fetch-and-increment on this counter to claim the next tile index. Each cluster continues requesting work until the returned tile index is greater than or equal to the total number of tiles, ensuring property (1). Because atomic operations are linearizable, each cluster receives a unique tile index, ensuring property (2). This strategy is implemented, for example, in the [quack tile scheduler](https://github.com/Dao-AILab/quack/blob/d898157f6761759161c48af94be1332dfd00697e/quack/tile_scheduler.py#L393).

While this approach is simple and architecture-agnostic, it is not without fault. All clusters must repeatedly perform atomic operations on the same global counter. This introduces some level of serialization between clusters, and requires repeated round-trips to global memory. Moreover, the global counter must be zeroed out before every kernel launch.

Fortunately, Blackwell provides a hardware-supported implementation of dynamic persistent scheduling called Cluster Launch Control (CLC). This simplifies the implementation of dynamic persistent scheduling on the software side, and offers several other benefits which we will describe in the remainder of the blog.

## Blackwell’s Cluster Launch Control (CLC)

CLC is a hardware-supported version of dynamic persistent tile scheduling available starting with the Blackwell architecture. It starts out with launching a scheduled grid identical to that of the single tile scheduler (i.e., based on the number of work tiles of the problem – see the discussion of [`__compute_grid`](#compute-grid) in the walkthrough), but the first wave of active clusters will loop repeatedly to try to “steal” the work of unlaunched clusters – cancelling their launches and obtaining their tile coordinates to do the work themselves. Thus the first wave of clusters may end up persisting and doing all of the work, and the other clusters in the grid may never end up launching. On the other hand, CLC also has the flexibility to dynamically allow for clusters to exit without finishing all tiles, and new clusters to later launch and continue working on the problem (see the section “CLC with concurrent kernels and pre-emption”). We start with an examination of the PTX instructions relevant to CLC, then walk through NVIDIA’s [CLC CuteDSL example](https://github.com/NVIDIA/cutlass/blob/ae6bccf341fb4410241f696ba06873023d5ce4ed/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py), and finally report on an experiment comparing CLC, static persistent scheduling, and single tile scheduling.

Our sources include the following:

- [The PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
- [The NVIDIA CUDA Programming guide section 4.12](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [The NVIDIA CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html)

### PTX instructions – `try_cancel` and `query_cancel`

There are two main sets of instructions at the PTX level that are used to carry out CLC logic. The first is `clusterlaunchcontrol.try_cancel` , which makes an atomic request to cancel a not-yet-launched cluster and obtain some encoded data in response. Then `clusterlaunchcontrol.query_cancel` can be used to decode that data to determine if the cancellation was successful, and, if so, obtain the tile coordinates of the cancelled cluster to “steal”.

The syntax for `clusterlaunchcontrol.try_cancel` is as follows:

```
clusterlaunchcontrol.try_cancel.async{.space}.completion_mechanism{.multicast::cluster::all}.b128 [addr], [mbar];

.completion_mechanism = { .mbarrier::complete_tx::bytes };
.space = { .shared::cta };
```

This instruction can be compared in many ways to [TMA](https://research.colfax-intl.com/tutorial-hopper-tma/):

- Like in TMA, only one thread should invoke a `try_cancel` operation. However, whereas one thread per CTA participating in a TMA multicast issues a TMA instruction, with `try_cancel` only one thread per cluster should be used. In particular, multiple threads submitting `try_cancel` will result in multiple clusters being cancelled.
- Like in TMA, this operation results in some data being written to SMEM (at the address provided by `[addr]` ) asynchronously. This data, if multicast, must be multicast to all CTAs in the cluster, whereas TMA is able to choose a subset of the cluster for the data to be multicast (e.g., only CTAs in the same row or column of the cluster).
  - With nontrivial clusters, if `try_cancel` is not multicast then the issuing warp will need to read the response data tile from SMEM, compute tile coordinate information, and then write this back to SMEM to have other CTAs in the cluster read the result. This may be more efficient in cases where calculating the work tile information is complex.
- Like in TMA, we use a transaction barrier to track completion of the `try_cancel` operation. However, any `try_cancel` operation always transfers 16 bytes.
- As a cluster-wide operation, we should take care to ensure that no other CTA in the cluster has exited when issuing `try_cancel` with multicast to avoid undefined behavior.

The syntax for `clusterlaunchcontrol.query_cancel` is as follows:

```
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 pred, try_cancel_response;

clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {xdim, ydim, zdim, _},  try_cancel_response;

clusterlaunchcontrol.query_cancel.get_first_ctaid{::dimension}.b32.b128 reg, try_cancel_response;

::dimension = { ::x, ::y, ::z };
```

We use these instructions in the following way:

- After observing the completion of `try_cancel` we can issue `query_cancel` type instructions on the 16-byte data returned by the `try_cancel` instruction. Note that this data is described by the PTX doc as “opaque” and by the programming guide as “encoded”, which implies that `query_cancel` is the only way to obtain useful information from the data.
- `.is_canceled` gives a predicate indicating whether the requested cancellation was successful. Note that if `.is_canceled` returns false then `query_cancel` instructions other than `.is_canceled` results in undefined behavior, so we should always start with `.is_canceled` .
  - Note further that if a CTA has observed the failure of a `try_cancel` (i.e., `is_cancelled` returns false) then issuing another `try_cancel` also results in undefined behavior. Thus, after such an observance, the CTA can no longer use CLC and should exit after exhausting its current work queue.
  - Failure of `try_cancel` usually doesn’t indicate an error but is rather a part of the scheduling logic – the most common reason for failure is that no more clusters in the grid remain to be executed.
- `.get_first_ctaid` can be used to obtain the grid coordinates of the first CTA in the cancelled cluster, with `.v4` to get all three dimensions of the coordinates (the content of the fourth element in the vector is unspecified), or specifying a particular dimension with `::dimension` .

### Walkthrough of CLC implementation (CuTeDSL Example)

The Blackwell CuTeDSL example [dense_gemm_persistent_dynamic.py](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py) implements a standard dense GEMM with CLC try-canceling performed by a single scheduler warp in each cluster, and communication between this and other warps in the cluster handled by a CLC pipeline. The numbering of the warps per CTA launched by the kernel can be seen in the `__init__` method:

```
self.epilogue_warp_id = (0, 1, 2, 3)
self.mma_warp_id = 4
self.tma_warp_id = 5
self.sched_warp_id = 6
```

<a id="compute-grid"></a>

First, in the `__call__` method we show how the grid variable used in the kernel launch parameters is determined via `_compute_grid`

```
def __call__(...):
    ...
    # Compute grid size
    self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn
    )
    self.kernel(...).launch(
        grid=grid,
        block=[self.threads_per_cta, 1, 1],
        cluster=(*self.cluster_shape_mn, 1),
        stream=stream,
     )
```

```
def _compute_grid(
    c: cute.Tensor,
    cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> Tuple[utils.ClcDynamicPersistentTileSchedulerParams, Tuple[int, int, int]]:
"""Use persistent tile scheduler to compute the grid size for the output tensor C.
    :param c: The output tensor C
    :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
    :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
    :return: A tuple containing:
        - tile_sched_params: Parameters for the persistent tile scheduler.
        - grid: Grid shape for kernel launch.
    """
    c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
    gc = cute.zipped_divide(c, tiler=c_shape)
    num_ctas_mnl = gc[(0, (None, None, None))].shape
    cluster_shape_mnl = (*cluster_shape_mn, 1)

    tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
        num_ctas_mnl, cluster_shape_mnl
    )
    # will round up to a whole number of clusters
    grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
    return tile_sched_params, grid
```

The `cta_tile_shape_mnk` is defined earlier and is deduced from the MMA tiler in a way which uniformly supports both 1CTA and 2CTA MMA modes:

```
self.cta_tile_shape_mnk = (
    self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
    self.mma_tiler[1],
    self.mma_tiler[2],
)
```

The computation of the grid then tiles C by the cta tiler to determine a preliminary grid shape and rounds that up by the cluster shape to satisfy the cluster divisibility requirement of a grid. This calculation is exactly the same as for a single-tile scheduler, and in particular does not involve the number of SMs.

Next, we examine the CLC pipeline. Along with the other standard GEMM pipelines, the CLC pipeline is created near the beginning of the kernel call.

```
      # Initialize clc_pipeline (barrier) and states
      clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
      cluster_size = cute.size(self.cluster_shape_mn)
      # 4 epilogue warps + 1 MMA warp + 1 TMA warp per CTA
	    # 1 scheduler warp per cluster
      num_clc_consumer_threads = 32 * (
          1 + cluster_size * (1 + len(self.epilogue_warp_id) + 1)
      )
      clc_pipeline_consumer_group = pipeline.CooperativeGroup(
          pipeline.Agent.Thread, num_clc_consumer_threads
      )
      clc_pipeline = pipeline.PipelineClcFetchAsync.create(
          barrier_storage=storage.clc_mbar_ptr.data_ptr(),
          num_stages=self.num_clc_stage, # 1 in this example
          producer_group=clc_pipeline_producer_group,
          consumer_group=clc_pipeline_consumer_group,
          tx_count=self.num_clc_response_bytes, # 16
          cta_layout_vmnk=cluster_layout_vmnk,
          defer_sync=True,
      )
```

We explain the how `num_clc_consumer_threads` is calculated in lines 6-8. The TMA, MMA, and epilogue warps of all CTAs in the cluster need to know the correct work tile coordinates (in addition to whether cancellation succeeded) to know where to perform their tasks, giving `cluster_size * (1 + len(self.epilogue_warp_id) + 1)` . The scheduler warp itself is also a consumer, because it also needs to know whether its cancellation request has failed, which will be the signal for it to exit. This gives the additional plus 1. Note that since all CTAs launch the same number of warps, non-leader CTAs in the cluster will also launch a “scheduler” warp, but these warps do not perform any work and are not consumers nor producers of the CLC pipeline. The scheduler warp in the cluster acts as the only producer for the CLC pipeline.

Slightly above the creation of the CLC pipeline, we also see the shared memory allocated for CLC operation and communication.

```
        class SharedStorage:
		    # ... (storage for mbarriers for TMA load, acc, and TMEM)
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2] # one empty and one full mbarrier (pipeline only has one stage)
            clc_response: cute.struct.MemRange[cutlass.Int32, 4] # total of 16 bytes to store try_cancel response per stage
```

<a id="scheduler-block"></a>

Next we jump to the block of code executed by the scheduler warp:

```
if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:

    clc_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
    )

    while work_tile.is_valid_tile:
        clc_pipeline.producer_acquire(clc_producer_state)
        mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
        tile_sched.advance_to_next_work(mbarrier_addr) # issues try_cancel
        clc_producer_state.advance()

		# scheduler also acts as a consumer below
        clc_pipeline.consumer_wait(clc_consumer_state)
        work_tile = tile_sched.get_current_work() # issues query_cancel
        clc_pipeline.consumer_release(clc_consumer_state)
        clc_consumer_state.advance()
    clc_pipeline.producer_tail(clc_producer_state)
```

- As previously noted, we see in line 1 that only the first CTA per cluster executes this block.
- In lines 3-5 the pipeline state is defined with `PipelineUserType.ProducerConsumer` , so it starts with a flipped phase bit, so the scheduler will not initially wait at `producer_acquire` and can start acquiring work tiles immediately. This is just like `PipelineUserType.Producer` .

We also look at the `producer_acquire` method for `PipelineClcFetchAsync` more closely in the utility file [`sm100.py`](https://github.com/NVIDIA/cutlass/blob/ae6bccf341fb4410241f696ba06873023d5ce4ed/python/CuTeDSL/cutlass/pipeline/sm100.py#L702) :

```
class PipelineClcFetchAsync:
     ...
     def producer_acquire(... ):
         """
         Producer acquire waits for empty buffer and sets transaction expectation on full barrier.
        :param state: Pipeline state pointing to the current buffer stage
        :param try_acquire_token: Optional token to skip the empty barrier wait
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(...)
        if_generate(
            self.is_signalling_thread,
            lambda: self.sync_object_full.arrive(
                state.index, self.producer_mask, loc=loc, ip=ip
            ),...)
```

What are `is_signaling_thread` and `producer_mask` ? The answer can be found earlier in the class:

```
class PipelineClcFetchAsync: …

    def _init_full_barrier_arrive_signal(cta_layout_vmnk: cute.Layout, tidx: Int32):
        """
        Computes producer barrier signaling parameters, returns destination CTA rank
        (0 to cluster_size-1) based on thread ID, and a boolean flag indicating if
        this thread participates in signaling.
        """
        dst_rank = tidx % 32
        is_signalling_thread = dst_rank < cute.size(cta_layout_vmnk)
        return dst_rank, is_signalling_thread
    def create(...)
	    consumer_mask = 0
	    …
	    (producer_mask, is_signalling_thread) = (
            PipelineClcFetchAsync._init_full_barrier_arrive_signal(
                cta_layout_vmnk, tidx
            )
        )
```

We see in lines 9-10 that the first cluster-size many threads of the scheduler warp are each responsible for signalling a different CTA in the cluster (thread `i` signals CTA `i` in the cluster). Also note that `consumer_mask = 0` in line 13 allows all consumers to signal the first CTA in the cluster when releasing.

Next, the method that results in `try_cancel` in the scheduler warp is `tile_sched.advance_to_next_work(mbarrier_addr)` in [line 10](#scheduler-block) of the scheduler warp’s code block, which calls `issue_clc_query` from a single elected thread, which eventually boils down to an operation corresponding to the PTX instruction `clusterlaunchcontrol.try_cancel` .

We next look at the consumer portion of the scheduler warp’s code, which is also run by all other consumer warps (i.e. TMA, MMA, and epilogue warps).

```
clc_pipeline.consumer_wait(clc_consumer_state)
work_tile = tile_sched.get_current_work() # issues query_cancel
clc_pipeline.consumer_release(clc_consumer_state)
clc_consumer_state.advance()
```

To obtain the next work tile info, each consumer calls `get_current_work` , which is essentially a wrapper for [work_tile_info_from_clc_response](https://github.com/NVIDIA/cutlass/blob/f74fea9ce35868d3ae9f8d1dce1969d7250d3f90/python/CuTeDSL/cutlass/utils/dynamic_persistent_tile_scheduler.py#L240) (both are found in the library file [dynamic_persistent_tile_scheduler.py](https://github.com/NVIDIA/cutlass/blob/f74fea9ce35868d3ae9f8d1dce1969d7250d3f90/python/CuTeDSL/cutlass/utils/dynamic_persistent_tile_scheduler.py)). There is some interesting logic happening here so we look at it more closely:

```
def work_tile_info_from_clc_response(
    self, result_addr: cute.Pointer, *, loc=None, ip=None
) -> WorkTileInfo:
    """
    Simulates parsing CLC response data in Python.
    result_addr: 16-byte response data (simulating shared memory access)
    """
    m_idx, n_idx, l_idx, vld = cute.arch.clc_response(result_addr, loc=loc, ip=ip)
    cute.arch.fence_proxy(
        "async.shared",
        space="cta",
    )
    cta_idx_in_cluster, cta_idy_in_cluster, _ = self.cta_id_in_cluster
    cur_tile_coord = (m_idx + cta_idx_in_cluster, n_idx + cta_idy_in_cluster, l_idx)
    return WorkTileInfo(cur_tile_coord, vld)
```

Line 8 is where the response data is decoded ( `clc_response` boils down to operations corresponding to the PTX instructions `clusterlaunchcontrol.query_cancel` ). Since the CTA coordinates in the grid obtained from `query_cancel` is always the first in the cluster, we offset by this CTA’s coordinates in its cluster to properly get its tile coordinates.

But we highlight in lines 9-12 the use of a shared async proxy fence, which seems unusual – in a standard GEMM kernel (e.g. [here](https://github.com/NVIDIA/cutlass/blob/cb37157db50d0528c4aea99feb37946ec278e3d9/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py#L1032)) these fences only appear before TMA store to ensure that a generic-proxy r2s write has finished before the async-proxy TMA store reads the data. Here the only relevant async-proxy operation is the `try_cancel` ’s write of response data to SMEM, and the fence is invoked after the decoding of the response data, so the fence is in fact guarding against the next iteration’s `try_cancel` from overwriting the SMEM before the current iteration is able to finish reading from that location. Note also the absence of a proxy fence before the `clc_response` call – though not explicitly mentioned in the PTX doc, it is likely that just as with TMA load, an implicit proxy fence is performed after the `try_cancel` ’s response data has finished transferring.

### Multi-stage CLC pipelines

Although not supported in this example, one can allow for queuing up of more than one work tile by having several stages for the CLC pipeline (e.g., this is done in the [CUTLASS C++ kernel](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp) with a depth of 3). This may be useful in certain situations to hide the scheduling latency when some work tiles may finish extremely quickly (for example, in variable-length attention, some work tiles may even be empty).

However, there is a different concern with having a deep CLC pipeline – we get worse dynamic load balancing by potentially queuing up unequal amounts of workloads for different SMs. In fact, the larger the number of stages, the more CLC will resemble static persistent scheduling. Furthermore, for problems with very few waves and imbalanced workloads it may be the case that even with one stage we still want block the scheduler warp from performing `try_cancel` until after the MMA mainloop finishes. For example, in the [grouped GEMM example](#imbalanced-benchmark) described earlier in this post, if we let the scheduler issue the first `try_cancel` immediately, clusters assigned a tile with large K may immediately procure another tile with large K and we may end up with a highly imbalanced workload distribution just like with a static persistent scheduler.

### CLC with concurrent kernels and pre-emption

Other than the kernel having no more unlaunched clusters left, another reason that a `try_cancel` can fail, according to the programming guide, could be that a second, higher-priority kernel, was launched after the first has already begun executing. After observing the failure of `try_cancel` , CTAs of the first kernel will exit, yielding GPU resources for the second kernel to run. Then, after the higher-priority kernel finishes, in the case that the first kernel hasn’t executed all of its grid yet, new clusters will be launched to finish off the rest of the grid of the first kernel. Allowing this “pre-emption” (term used in the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)) is another situation where CLC is more flexible than a static persistent scheduler, which is not able to dynamically reassign resources after a kernel has launched.

### Comparison of CLC vs static persistent and single tile schedulers – balanced workloads

Although CLC has been advertised as being useful over a static persistent scheduler in the situation of imbalanced workloads, it seems worthwhile to benchmark and compare the performance of CLC to both static persistent and single tile scheduling even on standard GEMM kernels.

In this section our experiments were done on a B200, which has 148 SMs configurable into 74 clusters of size 2. For CLC we used NVIDIA’s CuTeDSL example [dense_gemm_persistent_dynamic.py](https://github.com/NVIDIA/cutlass/blob/ae6bccf341fb4410241f696ba06873023d5ce4ed/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py). For static persistent we used [dense_gemm_persistent.py](https://github.com/NVIDIA/cutlass/blob/ae6bccf341fb4410241f696ba06873023d5ce4ed/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py), whose code is more or less identical to [dense_gemm_persistent_dynamic.py](https://github.com/NVIDIA/cutlass/blob/ae6bccf341fb4410241f696ba06873023d5ce4ed/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py) except for the scheduler and work tile info computation. For single-tile logic we modified dense_gemm_persistent_dynamic.py to remove the persistent scheduling logic (the closest out-of-the-box example file that does single-tile scheduling seems to be [dense_gemm.py](https://github.com/NVIDIA/cutlass/blob/ae6bccf341fb4410241f696ba06873023d5ce4ed/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py), but it’s not quite comparable to the other kernels- e.g. it doesn’t use warp-specialization while the others do). We used a batch size of 1 and the following configuration:

```
ab_dtype: Float8E4M3FN, c_dtype: Float32, acc_dtype: Float32
a_major: k, b_major: k, c_major: n
mma_tiler_mn: (256, 256), cluster_shape_mn: (2, 1)
use_2cta_instrs: True, use_tma_store: True
Warmup iterations: 500
Iterations: 100
Skip reference checking: True
Use cold L2: True
```

We benchmarked problem shapes (M,N,K) such that M=N in powers of 2 from 1024 to 32768 along with 1.5 times those sizes, and K in [2048, 8192]. Our results are shown in the following graphs:

![](Imgaes/blackwell-cluster-launch-control/image-24.png)

![](Imgaes/blackwell-cluster-launch-control/image-25.png)

The better performance of persistent schedulers over single-tile is not surprising since they’re able to overlap the epilogue with the MMA mainloop. With small K the epilogue takes a relatively larger proportion of each work tile’s runtime, whereas with large K the epilogue’s share of runtime is much smaller, so single-tile scheduling loses comparatively less efficiency without epilogue overlapping. For small problem shapes there is virtually no difference between the schedulers since there is less than one whole wave of clusters.

However, the performance differences observed between CLC and static persistent seem more enigmatic, although CLC seems to do worse in general with larger workloads. To understand more deeply, we may compare their respective tensor pipe throughput graphs as obtained from Nsight Compute PM sampling. Recall that this depicts a timeline view of throughput, where the x-axis is time elapsed and the y-axis is percentage utilization. For problem shape (16384, 16384, 2048), we see for CLC:

![](Imgaes/blackwell-cluster-launch-control/image-28.png)

And for static persistent:

![](Imgaes/blackwell-cluster-launch-control/image-26.png)

The gradual dropoff in tensor pipe usage seen in the second graph suggests that some SMs finish faster than others and become idle at the end of the kernel, so CLC is able to better use all of the GPU compared to static persistent.

On the other hand, for (32768, 32768, 2048) the tensor pipe throughput for CLC looks like

![](Imgaes/blackwell-cluster-launch-control/image-31.png)

And for static persistent:

![](Imgaes/blackwell-cluster-launch-control/image-32.png)

So in this case somehow the dropoff is less severe with a static scheduler, while the tensor pipe throughput appears to be consistently lower with CLC. One metric that correlates with this observation with (32768, 32768, 2048) is that NCU reports L2 hit rate to be only 35% with CLC compared to 52% with static persistent. The reason for this difference is unclear. Note that neither kernels have work tile swizzling, and with problem shape (16384, 16384, 2048), NCU shows an L2 hit rate of about 60% for both kernels.

The above experiments suggest that even for balanced workloads one should keep both static scheduling and CLC for tuning purposes. We also note that these example kernels do not include features such as work tile swizzling, blockscaling, or a nontrivial epilogue, which could change the comparative analysis.

In light of the lack of tensor pipe throughput dropoff with CLC, we also tracked the number of tiles computed per SM. Whereas with a static persistent scheduler the number of tiles computed by various SMs differ by at most 1, we observed that this is not the case with CLC. For example, with a problem shape (M, N, K) of (16384, 16384, 2048), SMs processed between 54 to 59 tiles, with frequencies (in pairs of SMs since we’re doing 2CTA MMA) shown in the histogram below:

![](Imgaes/blackwell-cluster-launch-control/image-33.png)

With a problem shape of (32768, 32768, 2048), the histogram of tiles computed instead looks like this:

![](Imgaes/blackwell-cluster-launch-control/image-34.png)

The histograms above suggest that for certain reasons (perhaps at the hardware level or otherwise), some SMs may end up being able to compute up to 5% more tiles than others. Thus, forcing all SMs to compute (almost) exactly the same number of tiles, even if balanced, may be slightly suboptimal.

For another instance of a work distribution histogram in the context of attention rather than GEMM, we refer the reader to this [PR that adds CLC to FlashAttention-4](https://github.com/Dao-AILab/flash-attention/pull/2218).

### Conclusion

In this post we explored CLC, the hardware-supported implementation of dynamic persistent scheduling introduced on Blackwell GPUs. CLC combines the advantages of both of the more traditional paradigms of single-tile and static persistent scheduling. We examined the low-level PTX instructions `try_cancel` and `query_cancel` needed for CLC, and then walked through a CuTeDSL implementation using the example [`dense_gemm_persistent_dynamic.py`](https://github.com/NVIDIA/cutlass/blob/ae6bccf341fb4410241f696ba06873023d5ce4ed/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py) , where a single scheduler warp per cluster is used to try to steal work tiles and a CLC pipeline is used to communicate the result with other warps. For imbalanced workloads, CLC is clearly more performant than static persistent, but we also explored how there are still slight differences between CLC and static persistent scheduling even with balanced workloads, with neither seemingly being a clear winner.

1. How does CLC compare to atomic-implemented dynamic persistent tile scheduling? The comparisons shown only compare with static persistent tile scheduling, which understandably doesnt do well on imbalanced workloads.

  1. i think the performance improvements mainly are because clc is hardware acclerated and can work with mbarrier (as well as warp specialization) to better hide lantency. And fetching from global memory everytime of course causes additional overhead.
  2. A well-implemented atomic-based dynamic persistent tile scheduler can outperform CLC, in my experience. The main benefits of CLC are that one does not need to know the number of SMs on the device, the preemption capability (and handling multiple simultaneous kernels more generally), and the relative simplicity of implementation.
