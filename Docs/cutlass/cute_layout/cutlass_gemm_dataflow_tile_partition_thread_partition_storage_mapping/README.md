# tcgen05 Fragment Descriptor Layout

## Source

- NVIDIA CUTLASS tcgen05 MMA Programming Guide: Creating fragment descriptors and descriptor tensors
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html#creating-fragment-descriptors-and-descriptor-tensors>

## Core Question

How does CuTe transform SMEM tensors into descriptor tensors consumed by `tcgen05.mma`?

## Abstraction Layers

1. Physical SMEM allocation: `sA`, `sB`
2. SMEM layout: outer tile layout and inner swizzle layout
3. Fragment creation: `make_fragment_A(sA)`, `make_fragment_B(sB)`
4. Descriptor tensor: `tCrA`, `tCrB`
5. Hardware descriptor element: one SMEM descriptor consumed by `tcgen05.mma`

## Figures

- SMEM tensor to fragment descriptor tensor
- `tCrA` / `tCrB` shape hierarchy
- Stage, MMA-K, and descriptor grid mapping

## Scripts

Put diagram generation scripts in `scripts/`, and write generated figures to `images/`.
