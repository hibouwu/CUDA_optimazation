#pragma once

#include "cutlass/detail/helper_macros.hpp"

namespace cutlass {
namespace arch {

template <class... Args>
CUTLASS_HOST_DEVICE void synclog_setup(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_teardown(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_print(Args&&...) {}

template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_syncthreads(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_syncwarp(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_named_barrier_arrive_and_wait(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_named_barrier_arrive(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_init(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_wait(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_test_wait(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_try_wait(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_arrive_cluster(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_arrive(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_invalidate(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx_cluster(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_transaction_barrier_expect_transaction(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_transaction_barrier_complete_transaction(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_fence_barrier_init(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_fence_view_async_shared(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_wait(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_wait_all(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_fence(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_nan(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_zfill(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_tma_load(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_tma_store(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_tma_store_arrive(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_tma_store_wait(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_warpgroup_arrive(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_warpgroup_wait(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_warpgroup_commit_batch(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_wgmma_reg_smem(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_wgmma_smem_smem(Args&&...) {}
template <class... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cpasync_barrier_arrive(Args&&...) {}

}  // namespace arch
}  // namespace cutlass
