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

__device__ __forceinline__ void set_b_cell(
    uint8_t* group_base,
    int cell_index,
    uint16_t value)
{
    uint16_t* cell = reinterpret_cast<uint16_t*>(group_base + cell_index * 16);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        cell[i] = value;
    }
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

// ===== kernel =====
__global__ void kernel(float* out) {

    extern __shared__ uint8_t smem[];
    __shared__ uint32_t tmem_base;
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

    // The numbers shown inside the diagram's cells are logical indices after
    // the 32B swizzle, not linear SMEM cell indices.  Place the three groups
    // of eight bf16 ones at physical SMEM offsets 0x10, 0x110 and 0x120 from
    // the 256B-aligned B_storage base.  Since B_start is B_storage + 0x10,
    // their physical 16B-cell indices relative to B_start are 0, 16 and 17.
    if (threadIdx.x == 0) {
        constexpr int one_cells[3] = {0, 16, 17};

        for (int i = 0; i < 3; ++i) {
            set_b_cell(B_start, one_cells[i], uint16_t(0x3f80));
        }

        printf("B_storage smem = 0x%x\n", smem_u32(B_storage));
        printf("B_start   smem = 0x%x\n", smem_u32(B_start));
        printf("one addr N0..7   = 0x%x\n", smem_u32(B_start));
        printf("one addr N8..15  = 0x%x\n", smem_u32(B_start + 0x100));
        printf("one addr N16..23 = 0x%x\n", smem_u32(B_start + 0x110));
    }

    __syncthreads();

    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_init(&done_barrier, 1);
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
    CUDA_CHECK(cudaMalloc(&d, 24 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, 24 * sizeof(float)));

    kernel<<<1,128, 16384>>>(d);
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
    return mismatches == 0 ? 0 : 1;
}
