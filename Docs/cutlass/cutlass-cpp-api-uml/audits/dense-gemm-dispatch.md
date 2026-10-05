# Dense FP16：`cute::gemm` 重载路径的定点编译核验

本次只核验 atlas 固定快照 commit `8f50b052e1099fb982392a622caab69b97b63128` 中 Dense FP16 配置的 Mainloop 到 MMA_Atom 调用路径。输入配方仍是 Sm100、FP16 A/B RowMajor、float C/D、Tile 256×128×64、Cluster 2×2×1、Auto；NVCC 的实际静态编译目标仍是 `sm_110a`。没有修改快照、旧示例或此前成功的 `run-334xibhm`，也没有运行 GPU kernel。

## 结论：中间必须保留 `D&&` 转发层

固定源码中的实际函数与调用位置如下。函数节点位置和调用表达式位置分开，不使用 `302` 这个近似行号：本快照第 302 行是未选中的 column-major 分支注释，实际 row-major 调用在第 298 行。

| 调用方 API | 原始调用行 | 真实目标 API |
| --- | --- | --- |
| `CollectiveMma::mma`，`sm100_mma_warpspecialized.hpp:652` | 702–705 | 四参 `cute::gemm`，`gemm.hpp:83` |
| 四参 `cute::gemm`，`:83` | 88 | 五参 dispatch [4]，`:275` |
| 五参 dispatch [4]，`:275` | 298 | 五参 `Tensor<D>&&` 转发重载，`:142` |
| 五参 `Tensor<D>&&` 转发重载，`:142` | 148 | 五参 rank-1 dispatch [1]，`:190` |
| rank-1 dispatch [1]，`:190` | 197 | `MMA_Atom::call(D,A,B,C)`，`mma_atom.hpp:94` |
| `MMA_Atom::call`，`:94` | 104 | 本例 Traits 中声明的 namespace friend `cute::mma_unpack`，`mma_traits_sm100.hpp:2072` |

`D(_,m,ns)` 是新 Tensor 值，不是左值引用。因此不能直接把 dispatch [4] 连到 rank-1 API。它首先匹配接受 `D&&` 的重载；这个重载内部的命名参数 `D` 是左值，才绑定 rank-1 重载的 `D&`。最终探针对这一切片结果的非引用性质做了 `static_assert`，NVCC/PTXAS 生成的 DWARF 也记录了同一条内联父子调用链。

## 分派条件来自实际 Tensor 类型

`mma_init` 的 `sA/sB` 按 Collective 选定的 `SmemAllocTypeA/B` 和 `SmemLayoutA/B` 构造，并传入 `TiledMma::make_fragment_A/B`。最终探针保留了源函数中命名 `sA/sB` 的左值类别。随后采用源码相同的 `(_,_,int,int)` 切片。累加器按 `partition_shape_C`、`make_sm100_accumulator` 和 stage 切片生成；本例 `IsOverlappingAccum=false`。

Host-only 类型/shape 打印与编译断言给出：

| 传入对象 | 顶层 shape / rank | Engine / V 条件 |
| --- | --- | --- |
| 完整 `tCrA`、`tCrB` | `(1,1,4,(1,8))`，rank 4 | descriptor fragment，4 个 K block、8 个 pipeline stage |
| `tCrA(_,_,k_block,read_stage)` | `(1,1)`，rank 2 | `ViewEngine<UMMA::DescriptorIterator>`；V=1，value_type 为 8 字节 |
| `tCrB(_,_,k_block,read_stage)` | `(1,1)`，rank 2 | 同样 V=1，value_type 为 8 字节 |
| `accumulators` | `((128,128),1,1)`，rank 3 | `ViewEngine<tmem_ptr<float>>` |
| `A(_,m)`、`B(_,ns)` | `(1)`，rank 1 | 各含一个描述符值 |
| `D(_,m,ns)`、`C(_,m,ns)` | `((128,128))`，rank 1 | 保留一个嵌套 V mode 的 TMEM 视图，不是 16384 个普通寄存器值 |

所以 dispatch [4] 的两个表达式 `size<0>(A/B) * sizeof(TA/TB::value_type)` 都是 `1 × 8 = 8`，而不是按原始 FP16 元素大小计算的 2。编译期进入第 288–289 行的 64-bit 分支；源码第 291 行的 `#if 1` 再选择 row-major serpentine 循环和第 298 行的调用。

本例 `M=N=1`，循环只有 `(m,n,ns)=(0,0,0)`。这证明源分支选择，不证明有可观察的“蛇形访问收益”，也不能把此分支选择推广到所有 Dense 配方。

还需注意 `pointer.hpp:228` 将 `is_rmem` 定义为非 gmem、非 smem。这里的 TMEM Engine 同时满足 `is_tmem=true` 和这个广义 `is_rmem=true`，所以能命中 gemm 的约束；这不改变实际累加器的 TMEM 驻留位置。

## 编译证据与交付

最终证据目录是 `data/modules/dense_fp16/probe/dispatch-run-ufh3cn5e/`。全部 824 个固定源文件哈希先经 `scope.json` 验证，NVCC 13.0.88 / GCC 14.4.1 使用与先前成功探针一致的局部 host compatibility flags。五项命令均正常退出：NVCC Host 类型构建、NVCC `-G --ptx`、Host-only 类型打印、PTXAS debug、`readelf --debug-dump=info`。Host 程序只执行类型、shape 和常量打印；单独设备探针只编译，不执行。

`gemm_dispatch_debug_index.json` 解析 DWARF 的 `DW_TAG_inlined_subroutine`、`DW_AT_abstract_origin` 和父作用域，并把调用方/被调用方关联到原始声明行与完整 mangled/demangled 名称。五条内部调用边全部通过核对，包含 `83→275→142→190→94→2072`。这比把 PTX 中不相关的 `.loc` 行放在一起更强；PTX `.loc` 本身也作为辅助证据保留。

数据交付为 `data/modules/dense_fp16/gemm_dispatch.json`：8 个新增节点、15 条新增边、2 张视图。这里只新增两个 API 节点——dispatch [4] 和 `D&&` 转发——复用 contracts 中四参、rank-1、Atom 的独立 API IDs。3 条边是实际 `calls`，其余是 Tensor 绑定/切片类型关系。编译期分支与循环条件单独记录。契约作者已移除临时 `contract.binding.gemm_five_dispatch` 和其 pending 边，跨 part 节点、边与视图端点均已校验。

最终 SHA-256：

- `gemm_dispatch.json`：`d4e3a410ea1d4f6137f3422b78e9c5627668c4d07005e45b458d082abe3624ed`
- `gemm_dispatch_debug_index.json`：`55263f13aac923087a06c26a230e7b2794c612cbad70d4917f29cfba0a3ce585`
- 最终 probe `metadata.json`：`214bd8175e74785dd09479e4614ca5309483032dcaf5a5101bbc1d8c234102db`
- `build_gemm_dispatch.py`：`9abd521fcb137585b40b59bbfcb989578b6b666d68d9a0bce1a71daf16b93f73`

初版成功目录 `dispatch-run-jerojb8q` 保留不覆盖。最终版把 fragment 工厂的形参表达式收紧为与源码一致的左值，并额外编译断言确认初版右值写法生成相同 Tensor 类型。旧完整 Dense kernel 的静态发射证据继续由不可变 `run-334xibhm` 提供，新的局部 kernel 不替代它。

新增 6 项离线回归测试全部通过：产物哈希和编译状态、DWARF 真正转发链、物理源码 quote、8 字节和 row 分支、单重载 API 身份、类型边与调用边分离。命令为 `.venv/bin/python -m unittest discover -s tests -p test_dense_gemm_dispatch.py -v`。

本次关闭的是这个固定 Dense 调用点的 gemm 重载选择和 Tensor 模板实参缺口。仍未验证 runtime 启动、TMEM 地址/分配、同步完成、数值正确性、性能，也未覆盖其他配置；没有据此宣布全库声明提取或阶段 1 通过。
