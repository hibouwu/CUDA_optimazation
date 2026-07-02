# Shared-memory bank-conflict research

Bank conflicts must be evaluated within the semantics and execution path of a
specific instruction. This directory is the index for those independent
benchmark families.

| Benchmark | Scope | Status |
| --- | --- | --- |
| [`ld_shared_1d`](ld_shared_1d/README.md) | One-dimensional `ld.shared` stride, broadcast, multicast, and vector loads | Implemented |
| [`st_shared_1d`](st_shared_1d/README.md) | One-dimensional scalar/vector `st.shared` accesses | Implemented, pending SM110 validation |
| [`cp_async`](cp_async/README.md) | 16-byte `cp.async` shared-memory destinations | Implemented, pending SM110 validation |
| [`tma`](tma/README.md) | TMA tensor-map swizzle and shared-memory destinations | Implemented, pending SM110 validation |
| [`tcgen05_smem_operand`](tcgen05_smem_operand/README.md) | `tcgen05.mma` SMEM descriptor/operand path | Implemented, pending SM110a validation |
| [`transpose_2d_case`](transpose_2d_case/README.md) | Load-only transpose bank-conflict study with E0-E4 pitch, multicast, vector, and software-swizzle cases | Implemented, pending SM110 validation |
