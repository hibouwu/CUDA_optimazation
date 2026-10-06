#pragma once
// Legacy small-TC lifecycle (kind 1 of v01.cu), unchanged since 2026-10-06: each of the four
// K16 m64n64 WGMMA is fenced, committed and drained with wait_group 0, so no group is in
// flight when the block returns.  The target kernel (kind 2) no longer uses inline PTX.
__device__ __forceinline__ void v01_wg4_n64(const uint64_t* a, const uint64_t* b, float* d) {
  asm volatile(
      "{.reg .pred p; setp.ne.b32 p,1,0;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "%32,%33,p,1,1,0,0;\n"
      "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "%34,%35,p,1,1,0,0;\n"
      "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "%36,%37,p,1,1,0,0;\n"
      "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "%38,%39,p,1,1,0,0;\n"
      "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
        "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(a[0]), "l"(b[0]), "l"(a[1]), "l"(b[1]), "l"(a[2]), "l"(b[2]), "l"(a[3]),
        "l"(b[3])
      : "memory");
}
