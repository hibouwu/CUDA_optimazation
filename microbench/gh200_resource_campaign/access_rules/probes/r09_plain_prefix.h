#pragma once
#include <stddef.h>
#include <stdint.h>

// One immutable plain module is loaded by every observing executable. Creation
// and all allocation happen before data initialization and the fixed prefix.
// No CUTLASS types cross the shared-library boundary. ABI 1 supports cfg_b only.
struct R09PlainPrefixSpec {
  int abi_version;
  int m, n, k, sm_count;
  int64_t lda, ldb, ldd;
  void* a;
  void* b;
  void* d;
};
struct R09PlainPrefixInfo {
  int abi_version;
  int stages;
  int grid_x, grid_y, grid_z;
  size_t workspace_bytes;
};
#ifdef __cplusplus
extern "C" {
#endif
// Return zero on success; a failed create leaves *handle null. Caller owns A/B/D.
int r09_plain_prefix_create(const struct R09PlainPrefixSpec*, void** handle,
                           struct R09PlainPrefixInfo*, char* error, size_t capacity);
// Enqueue one plain GEMM on the default stream. No allocation, fills or timing.
int r09_plain_prefix_run(void* handle, char* error, size_t capacity);
// Caller must finish all enqueued work before destroy/dlclose.
void r09_plain_prefix_destroy(void* handle);
#ifdef __cplusplus
}
#endif
