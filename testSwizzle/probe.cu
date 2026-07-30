#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

// 本程序验证以下固定配置：
//   - tcgen05.mma.kind::f16，A/B 为 BF16，累加结果为 FP32
//   - M=128、N=24、K=16，B 为 N/MN-major（instruction descriptor 中 transpose B=1）
//   - B 的 shared-memory descriptor 使用 32B swizzle
//   - B 由 TMA 从 global memory 搬入 shared memory
//   - A 全为 1；正确布局下，C 的第 0 行 N0..N23 应全部等于 1
//
// TMA source cell 与 tcgen05 读取位置之间的对应关系来自 Thor 上对本程序
// “修改一个 source cell -> 观察输出分组”的端到端实测，不把截图中的逻辑
// index 直接当作物理 shared-memory cell 编号。

// CUDA Runtime API 错误检查，例如 cudaMalloc、kernel launch、cudaMemcpy。
#define CUDA_CHECK(x) do { \
  auto err = (x); \
  if (err != cudaSuccess) { \
    printf("CUDA error: %s\n", cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

// CUDA Driver API 错误检查，本程序用 Driver API 创建 CUtensorMap。
#define CU_CHECK(x) do { \
  CUresult err = (x); \
  if (err != CUDA_SUCCESS) { \
    const char* message = "unknown CUDA driver error"; \
    cuGetErrorString(err, &message); \
    printf("CUDA driver error: %s\n", message); \
    exit(1); \
  } \
} while(0)

// 将普通 CUDA 指针转换成 PTX shared-memory 地址。
// tcgen05、mbarrier 和 TMA 的 inline PTX 都需要这种 32 位地址。
__device__ __forceinline__ uint32_t smem_u32(void const* p) {
  return __cvta_generic_to_shared(p);
}

// 构造 tcgen05 的 64 位 shared-memory matrix descriptor。
//
// PTX descriptor 中，起始地址、leading byte offset 和 stride byte offset
// 都以 16B 为编码单位，因此这里的接口接收“真实字节数”，再统一右移 4 位：
//   bits  0..13 : matrix start address / 16
//   bits 16..29 : leading byte offset / 16
//   bits 32..45 : stride byte offset / 16
//   bits 46..48 : 固定版本字段 0b001
//   bits 49..51 : swizzle pattern 的 base offset
//   bits 61..63 : swizzle mode；6 表示 32B swizzle
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

// 构造 tcgen05.mma.kind::f16 使用的 32 位 instruction descriptor。
// 最终编码值为 0x08070490：
//   bits  4..5  : D/accumulator 为 FP32
//   bits  7..9  : A 为 BF16
//   bits 10..12 : B 为 BF16
//   bit  16     : transpose B=1，即 B 使用 N/MN-major
//   bits 17..22 : N >> 3 = 3
//   bits 24..28 : M >> 4 = 8
// K=16 由 kind::f16 和数据类型组合确定，不需要单独编码。
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

// 初始化一个 shared-memory mbarrier。
// 初始化之后，调用处还要执行 fence.mbarrier_init.release.cluster，
// 将 barrier 的初始化发布给随后访问它的 TMA/tcgen05 async proxy。
__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, uint32_t count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :
        : "r"(smem_u32(barrier)), "r"(count)
        : "memory");
}

// 等待指定 parity 的 mbarrier 完成。
// try_wait 可能暂时失败，因此在设备端循环，直到异步事务完成。
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

// 告诉 mbarrier 本轮需要等待多少字节的 TMA transaction。
// TMA 完成相应字节数后，会通过 complete_tx 更新同一个 barrier。
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

// 发射一次 2D global -> shared TMA load。
// tensor_map 描述 global tile 的数据类型、形状、stride 和 32B swizzle；
// dst 是 256B 对齐的 B_storage，而不是偏移 16B 的 B_start，因为 TMA
// swizzle 的 shared-memory 目标需要满足更强的对齐要求。
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

// 用带标签的数据实测 TMA 32B swizzle 的落址。
//
// Host 会在未 swizzle 的 global source cells 1、8、6、7 中分别写入不同的
// 16-bit 标签。此 kernel 执行与正式实验相同的 16x16 BF16 TMA load，然后
// 通过 generic shared-memory load 扫描目标区域，打印每个标签最终所在的：
//   - physical cell：相对 B_storage 的 16B cell 编号
//   - byte offset ：相对 B_storage 的字节偏移
//   - smem address：硬件使用的绝对 shared-memory 地址
//
// 这里必须在 TMA barrier 完成后执行 async-proxy fence。否则直接用普通
// C++ load 观察 TMA async proxy 写入的数据，不构成正确的跨 proxy 同步。
__global__ void print_tma_32b_swizzle_addresses(
    const __grid_constant__ CUtensorMap tensor_map_b) {

    extern __shared__ uint8_t smem[];
    __shared__ alignas(8) uint64_t barrier;

    uint32_t raw = smem_u32(smem);
    uint32_t aligned = (raw + 255) & ~255;
    uint8_t* B_storage = smem + (aligned - raw);

    if (threadIdx.x == 0) {
        mbarrier_init(&barrier, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        constexpr uint32_t kTmaBytes = 16 * 16 * sizeof(uint16_t);
        tma_load_b_2d(B_storage, &tensor_map_b, &barrier);
        mbarrier_arrive_expect_tx(&barrier, kTmaBytes);
        mbarrier_wait(&barrier, 0);
    }
    __syncthreads();

    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    if (threadIdx.x == 0) {
        // 标签值必须和 Host 端 kProbeTags 保持一致。
        constexpr int kTmaSourceCells[4] = {1, 8, 6, 7};
        constexpr uint16_t kProbeTags[4] = {
            0x5101, 0x5108, 0x5106, 0x5107
        };
        const volatile uint16_t* words =
            reinterpret_cast<const volatile uint16_t*>(B_storage);

        printf("\n=== TMA 32B swizzle physical-address probe ===\n");
        printf("B_storage smem = 0x%x\n", smem_u32(B_storage));

        for (int probe = 0; probe < 4; ++probe) {
            int physical_cell = -1;

            // TMA tile 共 512B，即 32 个 16B cells。每个 source cell 的 8 个
            // BF16 都填相同标签，因此检查 physical cell 的第一个 word 即可。
            for (int cell = 0; cell < 32; ++cell) {
                if (words[cell * 8] == kProbeTags[probe]) {
                    physical_cell = cell;
                    break;
                }
            }

            if (physical_cell >= 0) {
                uint32_t byte_offset = uint32_t(physical_cell) * 16;
                printf(
                    "TMA source cell %d -> physical cell %d, "
                    "offset=0x%x, smem=0x%x\n",
                    kTmaSourceCells[probe],
                    physical_cell,
                    byte_offset,
                    smem_u32(B_storage + byte_offset));
            } else {
                printf("TMA source cell %d -> tag not found\n",
                       kTmaSourceCells[probe]);
            }
        }
    }
}

// ============================================================================
// Kernel：TMA 搬运 B -> tcgen05 MMA -> TMEM 读回第 0 行
// ============================================================================
__global__ void kernel(
    const __grid_constant__ CUtensorMap tensor_map_b,
    float* out) {

    extern __shared__ uint8_t smem[];

    // tcgen05.alloc 将分配到的 TMEM 基址写到 tmem_base。
    __shared__ uint32_t tmem_base;

    // TMA 搬运完成和 MMA 完成分别使用独立的 barrier，避免 phase/事务混用。
    __shared__ alignas(8) uint64_t tma_barrier;
    __shared__ alignas(8) uint64_t done_barrier;

    uint8_t* base = smem;

    // 32B swizzle pattern 每 256B 重复。动态 shared memory 的原始地址不保证
    // 256B 对齐，因此手工向上取整，令 B_storage 成为 pattern 的对齐基址。
    uint32_t raw = smem_u32(base);
    uint32_t aligned = (raw + 255) & ~255;
    uint8_t* smem_aligned = base + (aligned - raw);

    // Shared-memory 布局：
    //
    //   B_storage + 0x000 : TMA 的 256B 对齐目标基址
    //   B_storage + 0x010 : tcgen05 B descriptor 的起始地址 B_start
    //   B_storage + 0x100 : 仍属于 B 的 swizzled 存储区域
    //   B_storage + 0x1000: A 的起始地址
    //
    // B_start 比 swizzle pattern 基址晚 16B，正是实验要求中的
    // “start shared memory address = 16B”。
    uint8_t* B_storage = smem_aligned;
    uint8_t* B_start = B_storage + 16;
    uint8_t* A = B_storage + 4096;

    // 先把 B 区域清零。随后 TMA 会覆盖 16x16 BF16（512B）的目标区域；
    // 扩大清零范围便于确保 descriptor 可能访问的其余位置保持为 0。
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        B_storage[i] = 0;

    // A 是 128x16 BF16，并全部设置为精确的 BF16 1.0（bit pattern 0x3f80）。
    // 因此每个输出值就是对应 B 列在 K 维上的和。
    for (int i = threadIdx.x; i < 128 * 16; i += blockDim.x)
        reinterpret_cast<uint16_t*>(A)[i] = 0x3f80; // bf16 = 1

    __syncthreads();

    // A 由普通 STS/generic proxy 写入，而 tcgen05.mma 通过 async proxy
    // 读取 A。跨 proxy 访问同一 shared-memory 数据前必须执行该 fence。
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // 初始化 TMA completion barrier，并发布初始化结果。
    if (threadIdx.x == 0) {
        mbarrier_init(&tma_barrier, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    // 发射一次 16x16 BF16 TMA load，总 transaction 大小为 512B。
    //
    // 对本程序的 tensor map、B_storage/B_start 地址和 tcgen05 descriptor，
    // Thor 端到端实测得到：
    //   TMA source cell  1 -> 输出 N0..7
    //   TMA source cell  2 -> 输出 N8..15
    //   TMA source cell 17 -> 输出 N16..23
    //
    // 这是完整 TMA+MMA 路径的实测关系。不要把 source cell 编号、图中显示的
    // swizzle logical index 和最终 physical cell 编号混为一谈。
    if (threadIdx.x == 0) {
        constexpr uint32_t kTmaBytes = 16 * 16 * sizeof(uint16_t);

        // TMA 是异步操作：先发射 copy，再为 barrier 增加 expected bytes，
        // 最后等待 TMA 用 complete_tx 抵消全部 transaction bytes。
        tma_load_b_2d(B_storage, &tensor_map_b, &tma_barrier);
        mbarrier_arrive_expect_tx(&tma_barrier, kTmaBytes);
        mbarrier_wait(&tma_barrier, 0);

        // 这里只打印两个基址。TMA swizzle 后的实际物理落址由独立的
        // print_tma_32b_swizzle_addresses kernel 使用标签数据测量。
        printf("B_storage smem = 0x%x\n", smem_u32(B_storage));
        printf("B_start   smem = 0x%x\n", smem_u32(B_start));
    }

    __syncthreads();

    // MMA 使用另一个 barrier 跟踪 tcgen05.mma 的异步完成事件。
    if (threadIdx.x == 0) {
        mbarrier_init(&done_barrier, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    // 一个 warp 协作执行 TMEM allocation。N=24 需要至少 24 个 TMEM
    // columns，而硬件分配粒度最小为 32 columns，因此申请 32。
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :
            : "r"(smem_u32(&tmem_base)), "r"(32)
            : "memory"
        );
    }

    __syncthreads();

    // A：no swizzle，leading=256B，stride=128B。
    // B：从 B_start 开始，MN-major 32B swizzle，
    //    leading=256B，stride=512B，swizzle code=6。
    uint64_t desc_a = make_desc(A,       256, 128, 0, 0); // no swizzle
    uint64_t desc_b = make_desc(B_start, 256, 512, 0, 6); // 32B swizzle

    uint32_t idesc = make_idesc();

    // 将前面的线程同步/TMA completion 排序到随后的异步 tcgen05.mma 之前。
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    // cta_group::1 的 tcgen05.mma 具有单线程发射语义，因此只让 thread 0
    // 发射完整的 M128xN24xK16 MMA。
    if (threadIdx.x == 0) {
        // false 表示不读取旧 accumulator，执行 D=A*B，而不是 D=A*B+D。
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

        // tcgen05.mma 是异步的。commit 让 done_barrier 跟踪本线程此前
        // 发射的 tcgen05 操作，随后等待 parity 0 完成。
        asm volatile(
            "tcgen05.commit.cta_group::1."
            "mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :
            : "r"(smem_u32(&done_barrier))
            : "memory");

        mbarrier_wait(&done_barrier, 0);
    }

    __syncthreads();

    // MMA producer 是 thread 0，而后面的 TMEM load 由整个 warp 执行。
    // barrier 同步之后用 after_thread_sync 建立跨线程的 tcgen05 顺序。
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    __syncthreads();

    // 读取 C 的第 0 行。32 个线程协作执行 TMEM load；每次 x8 读取
    // 连续 8 个 N columns，循环三次覆盖 N0..23。
    if (threadIdx.x < 32) {
        float v[8];

        // TMEM 地址由 lane（高位）和 column（低位）组成。这里从
        // lane 0、column 0 的基址开始；32-thread load 将数据分发给整个
        // warp，其中 thread 0 获得需要导出的 m=0 行。
        uint32_t addr = tmem_base + (0 << 16);

        for (int i = 0; i < 24; i += 8) {
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),
                  "=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7])
                : "r"(addr + i)
            );

            // tcgen05.ld 同样是异步操作；使用寄存器 v[] 前必须等待完成。
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            // thread 0 对应 m=0，将这一行的 24 个结果写回 global memory。
            if (threadIdx.x == 0) {
                for (int j = 0; j < 8; j++) {
                    out[i + j] = v[j];
                    printf("n=%d val=%f\n", i+j, v[j]);
                }
            }
        }
    }

    __syncthreads();

    // 所有 TMEM 访问结束后，由同一个 warp 释放 32 columns，并放弃本 CTA
    // 后续再次申请 TMEM 的许可。
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

// ============================================================================
// Host：构造 TMA source、编码 tensor map、启动实验并检查结果
// ============================================================================
int main() {
    float* d;
    uint16_t* d_b;
    CUDA_CHECK(cudaMalloc(&d, 24 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, 16 * 16 * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(d, 0, 24 * sizeof(float)));

    // 构造未 swizzle 的 16x16 BF16 TMA source tile：
    //   - 一个 16B source cell 包含 8 个 BF16
    //   - source cells 1、2、17 全部填 1
    //   - 其余元素保持 0
    //
    // 对当前 tensor map 和 MMA descriptor，Thor 实测这三个 source cells
    // 分别产生 N0..7、N8..15、N16..23 三组输出。
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

    // 编码二维 TMA tensor map。
    //
    // global_dim = box_dim = {16,16}：
    //   dimension 0 是连续维，共 16 个 BF16 = 32B，恰好等于 swizzle width；
    //   dimension 1 共 16 行，row stride 为 32B。
    //
    // cudaMalloc 返回的 global 地址满足 TMA swizzle 的对齐要求；
    // shared 目标 B_storage 则在 kernel 内显式对齐到 256B。
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

    // 先做一次不会参与 MMA 的独立落址探针。线性编号的 16B TMA source
    // cells 1、8、6、7
    // 使用不同的 raw 16-bit 标签，kernel 会在 TMA 完成后扫描 swizzled SMEM，
    // 打印它们各自的真实 physical cell/offset/address。
    uint16_t h_probe[16 * 16] = {};
    constexpr int kProbeTmaSourceCells[4] = {1, 8, 6, 7};
    constexpr uint16_t kProbeTags[4] = {
        0x5101, 0x5108, 0x5106, 0x5107
    };
    for (int probe = 0; probe < 4; ++probe) {
        for (int i = 0; i < 8; ++i) {
            h_probe[kProbeTmaSourceCells[probe] * 8 + i] = kProbeTags[probe];
        }
    }
    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_probe,
        sizeof(h_probe),
        cudaMemcpyHostToDevice));
    print_tma_32b_swizzle_addresses<<<1, 32, 1024>>>(tensor_map_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 恢复正式 MMA 实验使用的 0/1 source tile。tensor map 仍引用同一个 d_b，
    // 因此只需更新其内容，不需要重新编码 tensor map。
    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b,
        sizeof(h_b),
        cudaMemcpyHostToDevice));

    // 动态 shared memory 为 16KiB，足以容纳：
    //   最多 255B 的对齐补偿 + 4096B B 区域 + 4096B A 区域。
    kernel<<<1,128, 16384>>>(tensor_map_b, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h[24];
    CUDA_CHECK(cudaMemcpy(h, d, 24 * sizeof(float), cudaMemcpyDeviceToHost));

    // 当前实验的验收条件：C 的第 0 行 N0..23 必须精确等于 1。
    // 输入只有可精确表示的 BF16 0/1，且每个输出预期只有一个非零贡献，
    // 所以这里使用精确比较，而不是容差比较。
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
