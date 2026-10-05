# 阶段 0 独立源码审查

审查日期：2026-09-08。固定提交：`8f50b052e1099fb982392a622caab69b97b63128`。本轮只修改此审查文件，没有修补制作脚本、清单、快照或上游文件。

最新结论见文末[第四轮复审：阶段 0 通过](#第四轮复审阶段-0-通过)。以下初审及中间失败记录按发生顺序保留。

## 结论：暂不通过

**当前快照的 824 个文件及其内容正确，但独立审计门禁暂不通过。** 审计只独立取得了 Git 路径集合，未取得每个路径在固定提交中的 blob ID 和 mode；内容散列回头信任清单，导致“错误内容与错误清单彼此一致”可以通过。另有 19 个 Git 可执行模式在落盘时未保留。

这不是声明提取未完成导致的拒绝。`declaration_status=pending_phase_1` 和 `api_generation=pending_entity_review` 在阶段 0 是正确状态。本审查不声称任何 API 声明覆盖或关系覆盖。

被审查的 `scripts/audit_scope.py` SHA-256：

```text
d1e32b3017ea5d0afac1ba9654f704885216d1bd7db00b328e93000be155b215
```

以下行号指向这一版；后续修复必须重新执行反例，不能只凭脚本已修改关闭问题。

## 已独立核实的事实

检查器从 `git ls-tree -rz <commit> -- include/cutlass include/cute` 重新取得 `path → (mode, kind, blob ID)`，不导入 `snapshot.py`，不拿清单中的 blob ID 当 Git 真值。对实际快照内容重新计算 `SHA1("blob " + byte_count + NUL + bytes)`，逐项对照独立 Git 树。

| 检查项 | 实际结果 |
|---|---|
| 固定 Git 树文件总数 | 824 个 blob |
| 根目录分布 | `include/cutlass` 712；`include/cute` 112 |
| 扩展名 | `.h` 494；`.hpp` 290；`.inl` 40 |
| Git 模式 | `100644` 805；`100755` 19 |
| 清单路径集合与 Git 集合 | 相同，无重复或遗漏 |
| 实际快照 include 文件集合与 Git 集合 | 相同，无多余或遗漏文件 |
| 当前清单中的 `git_blob`、`git_mode` 与 Git | 824/824 相同 |
| 当前快照实际内容与 Git blob | 824/824 相同 |
| 当前快照可执行位与 Git mode | 19 个不一致 |

`version.h` 的宏版本是 **4.6.0**；`git describe --tags` 是 `v4.5.0-21-g8f50b052`。后者不是正式发行版号，不应替代前者。

## 阻断项 P0-01：内容归属回头信任清单

位置：`scripts/audit_scope.py:17–18,24–29`。

`ls-tree --name-only` 丢弃了 Git blob ID。后续计算出的 SHA-256、字节数和 Git blob 散列都只与 `manifest["files"]` 比较，因此只证明“快照和清单一致”，不证明“这个路径在固定提交中就是这些内容”。许可证单独使用 `git show` 校验，不受同一问题影响。

本轮以内存覆盖层给现有 `validate()` 提供被替换的文件字节，其他文件仍读取实际快照；没有污染正式快照和清单。实际结果：

| 反例 | 期望 | 原实现结果 |
|---|---|---|
| 从清单删除一个文件 | 拒绝 | 拒绝，`scope membership mismatch` |
| 给 `include/cute/algorithm/axpby.hpp` 加一行审查注释，同时重算清单的 `sha256`、`bytes`、`git_blob` | 拒绝 | **通过** |
| 交换 `axpby.hpp` 和 `clear.hpp` 的内容，同时重算各自清单的三个散列/大小字段 | 拒绝 | **通过** |

第三项特别证明路径归属未被保护：路径、数量、提交字段和 URL 完全不变，内容却属于另一个文件。

关闭条件：审计器自己取得固定提交的 `path → Git blob ID`，把实际内容算出的 blob ID 和清单声明的 blob ID **分别**对照这个独立真值；补入“内容与清单协同修改”和“交换文件内容”回归。单纯增加另一个由清单提供的散列不能修复此问题。

## 阻断项 P0-02：模式未审计，快照丢失可执行位

位置：`scripts/audit_scope.py:24–29` 未读取 `git_mode`；`scripts/snapshot.py:29–34,66–67` 仅写入内容，没有设置 Git 模式。

将任一清单条目的 `git_mode` 改成 `160000`，原 `validate()` **通过**。当前清单模式本身是正确的，但审计器不能发现该字段被改错。

当前所有 19 个 `100755` 文件落盘后都没有可执行位。例如：

```text
Git 100755 → snapshot 0644: include/cute/arch/cluster_sm100.hpp
Git 100755 → snapshot 0644: include/cutlass/gemm/device/symm.h
```

关闭条件：审计器独立取得并验证每项 Git mode；快照按约定保留 Git 可执行位并实测，或者明确规定快照文件系统权限会规范化、仅以独立核验的 `git_mode` 保存源模式。后一方案必须同步修改当前“逐文件核对模式”的含义，不能把规范化后权限称为与源树完全相同。保留模式是当前合同下更直接的修复。

## 阶段 0 补充问题 P0-03：部分派生元数据未核对

把 `library_version_macros` 改成 `9.9.9`、`scope_roots` 改成仅 CUTLASS、`file_counts` 改错，并令一个文件的 `lines=-1`、`navigation_domain="unrelated"`，原 `validate()` **通过**。当前实际字段正确，这里发现的是后续失真无法被门禁检测。

建议独立重算属于阶段 0 的派生字段：版本宏、两个范围根、按根计数、行数、导航分类和文件名架构提示；检查其值但不要把提示升级为语义结论。声明状态及接口代际的最终语义判定属于后续阶段，不要求本阶段完成。

当前审计器没有枚举实际快照集合，只逐项读取清单文件；本轮独立枚举确认实际集合正确，但以后添加多余文件不会被原审计器发现。生成器不清理已有输出，因此应明确检查实际范围内文件集合、文件类型和符号链接边界，避免残留文件进入后续全量扫描。

## 分类规则复审：当前方向通过，以下反例必须继续保留

README 已正确写明：目录只作导航，扩展名、`threadblock` 路径、架构年代和使用 CuTe 均不是接口代际的充分判据。清单没有把 824 个文件缩成 Dense，没有把 pending 字段伪装成已审查标签。这部分方向成立。

需要持续约束后续提取的固定源反例：

- `gemm/device/gemm_universal_adapter.h:125,632` 分别是 3.x 和 2.x 特化；同文件不能只有互斥单值代际标签。3.x 分支 `:139–185` 仍暴露兼容 2.x 的别名。
- `gemm/kernel/gemm_universal_decl.h:40–57` 是双 API 主模板；`gemm_universal.hpp:40–48` 用 `UnderlyingProblemShape` 识别非 tuple 的 3.x grouped/array 问题形状。不能把“非 tuple”自动判成 2.x。
- `epilogue/threadblock/fusion/visitor_2x.hpp:317,331–332` 是 `.hpp` 中的 2.x EVT，且桥接 CuTe layout。
- `gemm/collective/sm80_mma_multistage.hpp:48–101` 是 collective/CuTe 路径；SM70/SM80 不能整体视为旧 API。
- `epilogue/collective/default_epilogue.hpp:56–79,98–103` 消费 `ThreadEpilogueOp` 及其 `Params`；`epilogue/thread` 不能整体标成 2.x-only。
- `cute/tensor.hpp` 和 `gemm/collective/collective_builder.hpp` 是聚合头；无直接类声明不等于无作用或可从分母删除。`kernel_hardware_info.h/.hpp`、`gemm_coord.h/.hpp` 也不能按 basename 去重。

固定 include 范围还包含卷积、BLAS3、GEMV、归约、transform 和 experimental distributed，不能用 GEMM Dense 主线替代它们。范围外 examples 中存在完整 FMHA 实现；当前 include 中 softmax 组件不自动等价于完整 Attention API。

## 范围外 include 目标留给关系阶段，不变更 824 分母

从固定源字面量 `#include` 独立发现：

- `cutlass/arch/simd.h:37–38` 引用 `cutlass/arch/array.h`、`cutlass/arch/numeric_types.h`，整个固定 Git 树未找到这两个目标；保留未解析状态，不据此断言某配置编译失败。
- `cutlass/conv/collective/sm90_implicit_gemm_gmma_ss_warpspecialized.hpp:47` 引用 `cutlass/util/packed_stride.hpp`，目标实际位于 `tools/util/include/`，属于范围外依赖。
- `cutlass/version.h:46` 在 `CUTLASS_VERSIONS_GENERATED` 条件下引用生成头 `cutlass/version_extended.h`。

这些目标不能静默丢弃，也不能偷偷增加源文件分母。它们不是阶段 0 未提取关系造成的阻断项，而是后续阶段明确的反例要求。

## 回归要求

1. 保留原有删项、重复、错误提交和单散列损坏测试。
2. 追加并实际拒绝源内容与全部清单散列协同修改、两个文件内容交换、错误 blob 归属、错误 Git mode。
3. 独立核对实际快照集合；对多余文件、缺失文件、目录/符号链接替换有明确结果。
4. 根据明确的权限合同核对 19 个 Git 可执行文件，并核对全部模式元数据。
5. 重新核对阶段 0 派生字段。对 API `pending` 状态保持诚实，不把它当此阶段失败或通过的替代指标。

完成以上修复后应追加复审记录；本版结论保持“暂不通过”，不能用 `data/phase-0-checks.json` 原有 `scope_content_check=pass` 覆盖此次独立发现。

## 第二轮复审记录：核心缺陷已修复，发现实际对象集合漏项

日期仍为 2026-09-08。本节为追加记录，保留上面的初审事实。复审期间制作代理继续修改脚本，因此以脚本散列区分测试结果，不把某一修订的结果归给另一修订。

中间修订 `audit_scope.py` SHA-256：

```text
787efc69972cab601739bc8bb8723160c80c566788f62767e8dfcb2ce2d4f6c5
```

该版已独立从 Git 树取得每项 mode 和 blob ID，内容和清单各自对照 Git 真值；生成器已恢复模式。审查者再次独立核对全部 824 个文件，没有内容或模式差异，19 个 Git `100755` 文件落盘模式已正确。以下初审反例重测均被拒绝：

- `git_mode=160000`：`file mode differs from fixed Git tree`。
- 文件内容与清单所有散列/字节数/行数协同修改：`blob identity differs from fixed Git tree`。
- 交换 `axpby.hpp` 与 `clear.hpp` 的内容并同步清单：`blob identity differs from fixed Git tree`。

因此 **P0-01 和 P0-02 关闭**。版本、范围根、分根计数、行数和导航字段的主修复也已存在。

这一轮不仅使用内存覆盖，还创建了真实临时快照 `/tmp/cutlass-phase0-review.pgefGs40/snapshot`，使用 `cp -a` 保留正式快照的内容与模式。基线通过。只在此临时副本中操作，正式快照、清单和上游均未改动。

该中间修订还会错误接受：

- 根目录额外普通文件 `EXTRA_ROOT_FILE.txt` 与范围内额外头文件 `include/cute/EXTRA_UNLISTED_HEADER.hpp`。
- 把 `include/cute/algorithm/axpby.hpp` 替换为指向临时快照外、内容相同且模式相同文件的符号链接。

## 第三轮复审记录：普通文件与链接问题关闭，非普通对象仍待关闭

被核对的后续 `audit_scope.py` SHA-256：

```text
9ee66e74f3bc461192199c477edb46ec73c80711c0a0b35f41ed76f6362397be
```

这一版增加了实际快照扫描，普通文件集合要求严格等于 824 个源文件加唯一的 `LICENSE.txt`，并拒绝快照根及内部符号链接。对真实临时副本重新测试得到：

| 反例 | 实物测试结果 |
|---|---|
| 正式快照基线 | 通过 |
| 额外根普通文件与额外 include 头文件 | 拒绝，`actual snapshot file set differs from fixed scope` |
| 把 `include/cute/algorithm` 整个父目录替换为快照外目录的符号链接 | 拒绝，`symlink in immutable snapshot` |
| 在临时快照根目录新增真实命名管道 `EXTRA_FIFO` | **错误通过** |

最后一项不是模拟 `Path` 返回值：审查者实际调用 `mkfifo` 创建了临时对象。没有对该 FIFO 写入数据，验证程序也没有读取它；问题恰好是它被忽略了。

### 待关闭项 P0-04：普通文件过滤静默忽略其他对象类型

`actual_entries` 收集了快照目录内所有项，但随后只拒绝符号链接，并用 `is_file()` 生成集合。FIFO、socket 或设备节点既不是符号链接也不是普通文件，会被静默过滤。这与“快照只包含固定源文件及许可证”的对象边界不一致，也可能让后续按目录读取所有项的程序遇到不应存在的对象。

关闭条件：枚举快照时明确只接受真实目录和普通文件，拒绝其他所有对象类型；至少加入一个真实或等效的 FIFO 反例。没有必要扩大源码分母，也不必新增 API 提取要求。

此外，`architecture_name_hints` 和清单的 `license` 元数据尚未核验：改成 `['sm999']` 或不存在的许可证路径，第二轮修订会通过。当前实际字段正确；建议分别按路径重算提示和固定验证 `snapshot/LICENSE.txt`，以完成派生字段对账。这两项不改变 824 个源文件当前内容正确的结论。

**本次追加时的结论：P0-01、P0-02 与额外普通文件/符号链接门禁已关闭；阶段 0 仍暂不通过，等待 P0-04 的最后回归。** 原始失败记录保留，后续结论应继续追加，不改写历史。

## 第四轮复审：阶段 0 通过

最终复审日期：2026-09-08。被实际执行的 `audit_scope.py` SHA-256：

```text
3a7df8183f1661c4c6bd1dd731744564618d7cd3ac2facef4c03c9daec44e026
```

这一版枚举快照后明确只接受目录和普通文件，拒绝其他对象类型，同时核对清单许可证位置和按文件名重算的架构提示。现成的真实临时副本没有重做或替换测试条件：

- 带 `EXTRA_FIFO` 的临时快照被拒绝：`special filesystem object in snapshot`。
- 错误 `license` 元数据被拒绝。
- 错误 `architecture_name_hints=['sm999']` 被拒绝。
- 将测试 FIFO 移出临时快照后，原临时快照基线重新通过。
- 正式快照基线通过。

审查者另独立执行最终检查器的完整入口，仅把生成报告的写入截获到内存，未改写正式 `phase-0-checks.json`。**17/17 类破坏测试全部被检测**，覆盖删项、重复、错误提交、散列损坏、伪造 blob、错误模式、错误版本/范围/计数/行数/导航/架构提示/许可证、内容与清单协同篡改、符号链接、额外普通文件和特殊对象。

结合前轮对全部固定 Git blob 和模式的独立核对，以及额外文件、文件/目录符号链接、协同篡改与内容交换的实际反例，**P0-01、P0-02、P0-03、P0-04 均关闭，阶段 0 源全集目标通过。** 当前文件分母仍为 CUTLASS 712 + CuTe 112 = 824，许可证是快照中唯一附加普通文件；恢复了 19 个 Git 可执行模式，没有用 Dense 或某个架构子集替代全范围。

该通过结论只覆盖固定源文件全集、内容归属、模式、元数据和快照边界，**不代表 API 声明、宏展开、模板特化、调用关系、异步协议或图集已经完成**。下一阶段应把审查重点转向真实 API 声明及关系遗漏，继续保留上文列出的混合接口、聚合头和范围外依赖反例，不再扩展与接口目标无关的目录细节。

审查用临时复本位于 `/tmp/cutlass-phase0-review.pgefGs40/`；测试对象留在复本外作短期复现材料。没有删除用户数据，没有修改正式快照、制作脚本或上游源码。
