# CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design

![](Imgaes/cutlass-3-gemm-abstractions/cutlass-featured.png)

AI-Generated Summary

- The CUTLASS 3.x redesign maximizes coverage of the GEMM implementation space through a hierarchical system of composable, orthogonal building blocks, improving code readability and extending support to later NVIDIA architectures such as Hopper and Blackwell.
- CUTLASS 3.x introduces a conceptual GEMM hierarchy with five layers: Atom, Tiled MMA/Copy, Collective, Kernel, and Device, allowing for highly customized GEMM implementations using template parameters.
- The Collective layer is responsible for organizing work temporally, using abstractions such as CollectiveMma and CollectiveEpilogue, which are composition points for lower-layer abstractions and can be customized using dispatch policies and other template parameters.

GEMM optimization on GPUs is a modular problem. Performant implementations need to specify hyperparameters such as tile shapes, math and copy instructions, and warp-specialization schemes. These hyperparameters are, to a large extent, independent from each other; moreover, the best choices may vary significantly based on hardware, problem shape, or other user needs.

With the 3.x redesign, CUTLASS aimed to maximize coverage of the space of GEMM implementations through a hierarchical system of composable, orthogonal building blocks, while also improving code readability and extending support to later NVIDIA architectures such as Hopper and Blackwell. As this design philosophy is linked to the hierarchical hardware design of the GPU, it can also be a good choice for other GPU applications – for example, [FlashAttention-3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) uses familiar CUTLASS abstractions in its design.

In this second blog post of the CUTLASS blog series, we’ll explore the design principles behind the hierarchical system of GEMM in CUTLASS 3.x, and unpack how CUTLASS builds GEMM kernels out of the low-level CuTe abstractions introduced in [part 1](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels).

## A new conceptual GEMM hierarchy in CUTLASS 3.x

CUTLASS 3.x develops a [conceptual GEMM hierarchy](https://github.com/NVIDIA/cutlass/blob/main/media/docs/gemm_api_3x.md) that’s independent of specific hardware features. It is structured into five layers:

![A diagram of green shaded semi-circles that are nested within each other to depict the GEMM hierarchy concept; from Atom to Device](Imgaes/cutlass-3-gemm-abstractions/CUTLASS-GEMM-hierarchy-png.webp)

Figure 1. Conceptual diagram of the CUTLASS GEMM hierarchy independent of hardware

- Atom layer : Architecture-specific instructions, and associated meta-information
  - `cute::Mma_Atom<>` and `cute::Copy_Atom<>`
- Tiled MMA/Copy : Spatial micro-kernels that allow for arbitrary interleaving and tiling of architecture specific atoms
  - `cute::TiledMma<>` and `cute::TiledCopy<>`
- Collective layer : Temporal micro-kernels that use architecture-specific synchronization to orchestrate the execution of one or more spatial micro-kernels to compute a single output tile
  - `cutlass::gemm::collective::CollectiveMma<>` , `cutlass::epilogue::collective::CollectiveEpilogue<>`
- Kernel layer : Device code for executing a kernel over a grid of threadblocks/clusters

```
cutlass::gemm::kernel::GemmUniversal<>
```

- Device layer : Host-side setup and interface

```
cutlass::gemm::device::GemmUniversalAdapter<>
```

Each layer serves as a composition point for abstractions from the previous layer, which can be highly customized using template parameters. Users can either stick to the highest layers, trusting CUTLASS’s compile-time logic to provide a performant GEMM implementation, or opt in to advanced modifications exposed by lower levels of the hierarchy. The spatial micro-kernels provided by the Atom and Tiled MMA/Copy layers are the domain of CuTe and were discussed in part 1. The rest of this post will cover the temporal and kernel-level organization of GEMM made available in the higher layers.

Here’s a basic example of how to define a GEMM kernel in CUTLASS 3.x:

```
// Step 1: Generate the required collective layer mainloop specialization
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TilesShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;
// Step 2: Specify the collective layer epilogue type
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;
// Step 3: Compose the mainloop and epilogue together at the kernel layer
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;
// Step 4: Wrap up the kernel::GemmUniversal kernel class
// with the device adapter to obtain a host-side handle to the kernel
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

## Collective layer: Mainloop

A collective is a group of threads that cooperate with each other to perform work, and that can be repeated in parallel to form the entire kernel. Generally, this is a threadblock or cluster. Whereas TiledMMA and TiledCopy objects describe the spatial assignment of parallel workers to compute and copy work (e.g., warps, warpgroups, or even threadblocks for Blackwell MMA), the Collective layer is responsible for organizing this work temporally, by setting up pipelines and warp-specialization schemes, and by using hardware-accelerated synchronization primitives for managing pipelines and asynchronous operations.
 CUTLASS 3.x [GEMM](https://github.com/NVIDIA/cutlass/blob/b78588d1630aa6643bf021613717bafb705df4ef/include/cutlass/gemm/collective/collective_mma_decl.hpp) kernels contain a collective mainloop, an instance of the GEMM class template that defines the basic ingredients of a single mainloop iteration performed by a single collective, most importantly the load and MMA procedures. A collective mainloop can be defined like this:

```
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
  DispatchPolicy,
  TileShape,
  ElementA, // dtype, e.g. float
  StrideA,  // e.g. Stride<_1, int> for M-major
  ElementB, StrideB,
  TiledMma,
  GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, TransformA,
  GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, TransformB
>;
```

The collective mainloop is the composition point for abstractions from lower layers: a TiledMma, a TiledCopy for GMEM to SMEM load for each operand, and optional copy atoms for SMEM to RMEM load for use with register-sourced MMAs. These abstractions are largely orthogonal, allowing different MMA operations to be combined with different copy operations while maximizing code reuse.

Arguably the most important piece is the [dispatch policy](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/gemm/dispatch_policy.hpp#L185), which defines the mainloop specialization to a particular algorithm or GPU architecture. For example, the dispatch policy `MainloopSm90TmaGmmaWarpSpecialized` specializes the CollectiveMma to the Hopper TMA warp-specialized implementation. This is itself a template that can be parametrized over pipeline stages, cluster shape, and the choice of kernel schedule, such as [pingpong or cooperative](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md#hopper-warp-specialization) for Hopper GEMM kernels.

Examples of specialized collective mainloop implementations can be found in the [GEMM collective](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective) folder.

## The collective builder

The CollectiveMma has a variety of tuning knobs that allow a user to precisely specify a GEMM mainloop in terms of TiledCopy and TiledMma objects, but with this flexibility comes complexity. Typically, the user will want to deduce these objects and the associated SMEM layouts from higher-order considerations about pipelining, hardware capabilities, and resource availability. CUTLASS can also perform this deduction using the CollectiveBuilder interface. A mainloop declaration using the CollectiveBuilder looks like this:

```
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag,       // e.g. cute::arch::Sm90 for Hopper
  OpClass,       // e.g. cute::arch::OpClassTensorOp for Tensor Cores
  ElementA, LayoutA, AlignmentA,
  ElementB, LayoutB, AlignmentB,
  ElementAccumulator,
  TileShape, ClusterShape,
  StageCount,    // e.g. cutlass::gemm::collective::StageCountAuto
  KernelSchedule // e.g. cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

The template arguments select from user-friendly criteria and use them to deduce the lower-level parameters to the CollectiveMma template:

- Architecture specialization: The GPU architecture and type of MMA operator (e.g., SIMT or Tensor Cores).
- Operand and accumulator information: Data types for the operands and accumulator, and alignment and compile-time layout information for the operands in global memory (e.g., row- or column-major).
- Tile shapes: Used to deduce TiledMma and TiledCopy objects and SMEM layouts.
- Scheduling information: Cluster shape, pipeline stage count, and kernel schedule are all used by the scheduling algorithm. There are default Auto options for the stage count and kernel schedule parameters, which tell CUTLASS to try to automatically select the best one for the given architecture and parameters.

## Collective layer: Epilogue

The collective epilogue is the other half of the Collective API. It handles the temporal orchestration of post-processing and output storage of the worktiles after each mainloop iteration. As with the mainloop, this means that the collective epilogue is a composition point for a copy operation (the output storage) and some math operations (typically elementwise operations, but potentially also including reduction). Unlike the mainloop, these math operations are themselves highly composable via the Epilogue Visitor Tree (EVT) formalism. This is particularly useful for AI workloads, which frequently require the calculation of an activation function immediately after GEMM. CUTLASS’s collective epilogues handle the fusion of this activation function into the kernel, eliminating unnecessary data movement.

CUTLASS has several epilogues, defined [here](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective) on GitHub. The template arguments vary significantly between implementations, but generally include the following information:

- Data type and compile-time layout information about matrices C and D.
- A fusion operation specifying any additional post-processing.
- TiledCopy operations for the GMEM store and any SMEM staging.
- Dispatch policies, as with the collective mainloop, containing information about cluster size, TMA use, warp-specialization, and so on.

The [CollectiveBuilder for the epilogue](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/epilogue/collective/collective_builder.hpp) presents a more uniform and high-level interface:

```
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag,
  OpClass,
  TileShape,
  ClusterShape,
  EpilogueTileType,
  ElementAccumulator,
  ElementCompute,
  ElementC, GmemLayoutTagC, AlignmentC,
  ElementD, GmemLayoutTagD, AlignmentD,
  EpilogueScheduleType,
  FusionOpOrCallbacks
>::CollectiveOp;
```

Many of these arguments are familiar from the mainloop builder, but a few are new:

- The epilogue can divide a CTA tile into smaller tiles for better math-copy overlapping.
- The accumulator, the output of the mainloop, is now an input to the epilogue. Epilogue computations can take place in a different intermediate data type (given by `ElementCompute` ).
- CUTLASS provides a wide selection of [common fusion operations](https://github.com/NVIDIA/cutlass/blob/b84e9802d84b16bcb4e92338fcf0a04785df9236/include/cutlass/epilogue/fusion/operations.hpp) , such as `D = activation(alpha * AB + beta * C)` . The user can also build a bespoke fusion operation using [Epilogue Visitor Trees](https://github.com/NVIDIA/cutlass/blob/b78588d1630aa6643bf021613717bafb705df4ef/include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp) . For more information about Epilogue Visitor Trees, see [this Colfax tutorial](https://research.colfax-intl.com/epilogue_visitor_tree/) .
- [Epilogue schedule types](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/epilogue/dispatch_policy.hpp) define usage of TMA and warp-specialization. The default `EpilogueScheduleAuto` tells CUTLASS to try to deduce the best option.

To see both of the Collective Builders in action, we refer to CUTLASS [example 49](https://github.com/NVIDIA/cutlass/blob/389e493055f981bfdc6d4348f823191ca7b9fddd/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu) for Hopper and [example 71](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu) for Blackwell.

## Kernel layer

The Collective layer fully defines the computation done by a collective during kernel execution. Extending the collectives over a grid of threadblocks or clusters that covers the whole dynamically-sized problem space is then the role of the Kernel layer. The Kernel layer assembles a collective mainloop and collective epilogue into a device kernel by stitching together their primitive procedures for load, store, MMA, and so on.

The entry point API for the Kernel layer is the class [cutlass::gemm::kernel::GemmUniversal](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/gemm_universal_decl.h), which is a stateless universal device kernel that implements GEMM as the composition of a collective mainloop and a collective epilogue. Stateless means that the caller manages the kernel’s state by passing in parameters to it. Universal means that `GemmUniversal` is the entry point to both 2.x and 3.x GEMM kernels. For the 3.x API, basic usage of `GemmUniversal` looks like this:

```
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, // e.g. Shape<int, int, int> for a fully generic GEMM
    CollectiveMainloop,
    CollectiveEpilogue
>;
```

As with `TiledMma` and `TiledCopy` , `CollectiveMainloop` and `CollectiveEpilogue` are orthogonal abstractions that are composed via `GemmUniversal` . The first template argument, the problem shape, is primarily used to select between ordinary GEMM (with a rank-3 problem shape) and batched GEMM (with a rank-4 problem shape), but can also statically constrain some of the problem dimensions if needed.

The instantiations of `GemmUniversal` are found in files of the form `cutlass/gemm/kernel/sm*_gemm_*.hpp` , with `GemmUniversal` largely dispatching based on the `KernelSchedule` parameter of the collective mainloop. All instantiations present a consistent interface:

- An interface for passing arguments to the kernel, including the problem shape, information about the hardware, pointers to and layouts of tensors, and epilogue parameters.
- Static initialization functions for getting the grid and block dimensions, checking if the kernel is implementable on the hardware, and setting up a global memory workspace for any reduction operations or global barriers required by the epilogue or tile scheduler.
- Most importantly, they implement the kernel logic as `operator()` . This is a device function—although the Kernel layer contains all the logic for kernel execution, it doesn’t yet expose a way to launch it from the host.

For example, the TMA warp-specialized kernel for Blackwell is defined [here](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp).

## Tile scheduling

The Kernel layer is also the composition point for specifying a tile scheduler. Just as the kernel schedule defines the temporal organization of work within a collective, the tile scheduler defines the order and distribution of work across collectives. For the most basic tile scheduler, one CTA is assigned per output tile. CUTLASS 3.x implements two additional tile schedulers for Hopper: a persistent scheduler that launches one CTA per SM and has each CTA (potentially) compute multiple output tiles over its lifetime—and a Stream-K scheduler, which is also persistent but additionally divides some output tile work along the K-mode for better load balancing. On the Blackwell architecture, one instead uses schedulers with [Cluster Launch Control](https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_cluster_launch_control.md). For more in-depth information on tile scheduling, see [this Colfax tutorial](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/).

We could extend our kernel above to use a Stream-K tile scheduler using:

```
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler
>;
```

[CUTLASS example 74](https://github.com/NVIDIA/cutlass/blob/main/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu) is a more detailed example using the Stream-K scheduler.

## Device layer

Host-side logic for kernel launch, including launch with cluster support or on different devices or CUDA streams, is implemented in the Device layer. The main entry point to the Device layer is `cutlass::gemm::device::GemmUniversalAdapter` , which wraps a `GemmUniversal` kernel in a stateful, reusable handle. Stateful means that the handle instance contains state that the kernel needs to run (i.e., it manages the kernel arguments itself). Reusable means that the same handle instance can be used to call the kernel multiple times with different arguments.

`GemmUniversalAdapter` is implemented [here](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/gemm_universal_adapter.h) on GitHub. This example shows how we can use `GemmUniversalAdapter` to launch a kernel:

```
using GemmHandle = cutlass::gemm::kernel::GemmUniversalAdapter<GemmKernel>;
using Arguments = typename GemmHandle::Arguments;    // surfaced from GemmKernel
Arguments args {
    cutlass::Gemm::kBatched,                   // mode (here batched GEMM)
    cute::make_shape(M, N, K, L),              // problem shape
    {A, stride_A, B, stride_B},                // mainloop args
    {{alpha, beta}, C, stride_C, D, stride_D}, // epilogue args
    make_kernel_hardware_info(device_id),      // hardware info
    {}                                         // scheduler args (here default)
};
GemmHandle gemm;
// Check that problem can run with given shape and hardware
cutlass::Status status;
status = GemmHandle::can_implement(args);
if (status != cutlass::Status::kSuccess) {
  std::cerr << "Problem not supported\n";
  exit(EXIT_FAILURE);
}
// Set up global memory workspace
size_t workspace_size = GemmHandle::get_workspace_size(args);
cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
// Initialize GEMM handle state from arguments
status = gemm.initialize(args, workspace.get());
if (status != cutlass::Status::kSuccess) {
  std::cerr << "Failed to initialize GEMM kernel\n";
  exit(EXIT_FAILURE);
}
// Launch kernel
status = gemm.run();  // can supply CUDA stream and CUDA host adaptor here
if (status != cutlass::Status::kSuccess) {
  std::cerr << "Failed to launch GEMM kernel\n";
  exit(EXIT_FAILURE);
}
```

## Summary

In this post, we’ve discussed how the CUTLASS library is conceptually organized as a hierarchy, in which objects at each layer are composed in terms of orthogonal objects from lower layers. This design allows for a menagerie of deeply customizable GEMM implementations with a high level of code reuse. In the next and final post in the series, we’ll look at the changes introduced in CUTLASS 4.0, in particular the CuTe Python DSL.

For more information, you can download the software on [GitHub](https://github.com/NVIDIA/cutlass), read our [documentation](https://docs.nvidia.com/cutlass/index.html), or join our [Developer Forum](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/cuda-programming-and-performance/7) for deeper discussions.

Acknowledgments

We would like to acknowledge the extended [CUTLASS team](https://github.com/NVIDIA/cutlass/blob/main/CONTRIBUTORS.md) whose collective efforts made this work possible.
