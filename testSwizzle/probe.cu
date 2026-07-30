#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

#define CUDA_CHECK(x) do { \
  auto err = (x); \
  if (err != cudaSuccess) { \
    printf("CUDA error: %s\n", cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

#define CU_CHECK(x) do { \
  CUresult err = (x); \
  if (err != CUDA_SUCCESS) { \
    const char* message = "unknown CUDA driver error"; \
    cuGetErrorString(err, &message); \
    printf("CUDA driver error: %s\n", message); \
    exit(1); \
  } \
} while(0)

__device__ __forceinline__ uint32_t smem_u32(void const* p) {
  return __cvta_generic_to_shared(p);
}

// ===== descriptor (关键) =====
__device__ __forceinline__ uint64_t make_desc(
    void* ptr,
    uint32_t leading_bytes,
    uint32_t stride_bytes,
    int base_offset,
    int swizzle_code)
{
    uint32_t addr = smem_u32(ptr);

    uint64_t d = 0;
    d |= uint64_t((addr >> 4) & 0x3fff);
    d |= uint64_t((leading_bytes >> 4) & 0x3fff) << 16;
    d |= uint64_t((stride_bytes  >> 4) & 0x3fff) << 32;
    d |= (uint64_t(1) << 46); // version
    d |= uint64_t(base_offset & 0x7) << 49;
    d |= uint64_t(swizzle_code & 0x7) << 61;

    return d;
}

__device__ __forceinline__ uint32_t make_idesc() {
    uint32_t d = 0;
    d |= 1u << 4;  // D type: FP32
    d |= 1u << 7;  // A type: BF16
    d |= 1u << 10; // B type: BF16

    d |= (24u >> 3) << 17;  // N=24
    d |= (128u >> 4) << 24; // M=128

    d |= 1u << 16; // B is N/MN-major (transpose B)

    return d;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, uint32_t count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :
        : "r"(smem_u32(barrier)), "r"(count)
        : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, uint32_t phase) {
    constexpr uint32_t ticks = 0x989680;

    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 "
        "p, [%0], %1, %2;\n\t"
        "@!p bra WAIT_%=;\n\t"
        "}"
        :
        : "r"(smem_u32(barrier)),
          "r"(phase),
          "r"(ticks)
        : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* barrier,
    uint32_t bytes)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
        "_, [%0], %1;"
        :
        : "r"(smem_u32(barrier)), "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void tma_load_b_2d(
    uint8_t* dst,
    const CUtensorMap* tensor_map,
    uint64_t* barrier)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(smem_u32(dst)),
          "l"(tensor_map),
          "r"(0), "r"(0),
          "r"(smem_u32(barrier))
        : "memory");
}

// ===== kernel =====
__global__ void kernel(
    const __grid_constant__ CUtensorMap tensor_map_b,
    float* out) {

    extern __shared__ uint8_t smem[];
    __shared__ uint32_t tmem_base;
    __shared__ alignas(8) uint64_t tma_barrier;
    __shared__ alignas(8) uint64_t done_barrier;

    uint8_t* base = smem;

    // 强制 256B 对齐
    uint32_t raw = smem_u32(base);
    uint32_t aligned = (raw + 255) & ~255;
    uint8_t* smem_aligned = base + (aligned - raw);

    uint8_t* B_storage = smem_aligned;
    uint8_t* B_start = B_storage + 16;
    uint8_t* A = B_storage + 4096;

    // 初始化
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        B_storage[i] = 0;

    for (int i = threadIdx.x; i < 128 * 16; i += blockDim.x)
        reinterpret_cast<uint16_t*>(A)[i] = 0x3f80; // bf16 = 1

    __syncthreads();

    // A was produced through the generic proxy and will be consumed by the
    // tcgen05 async proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_init(&tma_barrier, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    // Thor end-to-end probing of this exact tensor map and MMA descriptor
    // shows that TMA source cells 1, 2 and 17 feed N0..7, N8..15 and N16..23.
    // Keep this as an empirical property of the complete TMA+MMA path rather
    // than inferring it from the cell labels in the swizzle diagram.
    if (threadIdx.x == 0) {
        constexpr uint32_t kTmaBytes = 16 * 16 * sizeof(uint16_t);
        tma_load_b_2d(B_storage, &tensor_map_b, &tma_barrier);
        mbarrier_arrive_expect_tx(&tma_barrier, kTmaBytes);
        mbarrier_wait(&tma_barrier, 0);

        printf("B_storage smem = 0x%x\n", smem_u32(B_storage));
        printf("B_start   smem = 0x%x\n", smem_u32(B_start));
        printf("TMA one addr N0..7   = 0x%x\n", smem_u32(B_storage + 0x010));
        printf("TMA one addr N8..15  = 0x%x\n", smem_u32(B_storage + 0x020));
        printf("TMA one addr N16..23 = 0x%x\n", smem_u32(B_storage + 0x110));
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_init(&done_barrier, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :
            : "r"(smem_u32(&tmem_base)), "r"(32)
            : "memory"
        );
    }

    __syncthreads();

    uint64_t desc_a = make_desc(A,       256, 128, 0, 0); // no swizzle
    uint64_t desc_b = make_desc(B_start, 256, 512, 0, 6); // 32B swizzle

    uint32_t idesc = make_idesc();

    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (threadIdx.x == 0) {
        uint32_t use_d = 0;

        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, {%5,%6,%7,%8}, p;\n\t"
            "}"
            :
            : "r"(tmem_base),
              "l"(desc_a),
              "l"(desc_b),
              "r"(idesc),
              "r"(use_d),
              "r"(0), "r"(0), "r"(0), "r"(0)
            : "memory");

        asm volatile(
            "tcgen05.commit.cta_group::1."
            "mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :
            : "r"(smem_u32(&done_barrier))
            : "memory");

        mbarrier_wait(&done_barrier, 0);
    }

    __syncthreads();

    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    __syncthreads();

    // ===== read first row =====
    if (threadIdx.x < 32) {
        float v[8];

        uint32_t addr = tmem_base + (0 << 16);

        for (int i = 0; i < 24; i += 8) {
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),
                  "=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7])
                : "r"(addr + i)
            );
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            if (threadIdx.x == 0) {
                for (int j = 0; j < 8; j++) {
                    out[i + j] = v[j];
                    printf("n=%d val=%f\n", i+j, v[j]);
                }
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :
                     : "r"(tmem_base), "r"(32)
                     : "memory");
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
            :::
            "memory");
    }
}

int main() {
    float* d;
    uint16_t* d_b;
    CUDA_CHECK(cudaMalloc(&d, 24 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, 16 * 16 * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(d, 0, 24 * sizeof(float)));

    // The TMA source is an unswizzled 16x16 BF16 tile.  Each 16B source cell
    // contains eight BF16 elements.  Thor end-to-end probing verified that
    // source cells 1, 2 and 17 produce the required three output groups.
    uint16_t h_b[16 * 16] = {};
    constexpr int tma_source_one_cells[3] = {1, 2, 17};
    for (int cell : tma_source_one_cells) {
        for (int i = 0; i < 8; ++i) {
            h_b[cell * 8 + i] = 0x3f80;
        }
    }
    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b,
        sizeof(h_b),
        cudaMemcpyHostToDevice));

    CUtensorMap tensor_map_b{};
    constexpr uint32_t rank = 2;
    uint64_t global_dim[rank] = {16, 16};
    uint64_t global_stride[rank - 1] = {
        16 * sizeof(uint16_t)
    };
    uint32_t box_dim[rank] = {16, 16};
    uint32_t element_stride[rank] = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(
        &tensor_map_b,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        d_b,
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    kernel<<<1,128, 16384>>>(tensor_map_b, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h[24];
    CUDA_CHECK(cudaMemcpy(h, d, 24 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n=== global memory result ===\n");
    int mismatches = 0;
    for (int i = 0; i < 24; i++)
    {
        printf("h[%d]=%f\n", i, h[i]);
        if (h[i] != 1.0f)
            ++mismatches;
    }
    printf("status=%s mismatches=%d\n", mismatches == 0 ? "PASS" : "FAIL", mismatches);

    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(d_b));
    return mismatches == 0 ? 0 : 1;
}
