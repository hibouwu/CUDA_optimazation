# 阶段 1：独立位域模块与全库候选检查

日期：2026-09-08。固定源码提交：`8f50b052e1099fb982392a622caab69b97b63128`。

本轮新增 `scripts/declaration_bitfields.py`、`tests/test_declaration_bitfields.py` 和本审查文件。模块没有接入 `Extractor`，也没有修改现有五个生成器、固定快照、asm 源码或声明数据。本报告不能被解读为主提取器已经恢复匿名位域身份。

## 已交付能力与接口

入口是 `analyze_bitfields(path: str, source: bytes) -> dict`。它先独立扫描物理字节中的非 `::` 冒号，排除注释和普通/raw 字符串，再与原始 Tree-sitter 树核对。整个函数不进行文件写入、不选取预处理分支、不删除 body，也不自动应用投影。`raw_parse_has_error` 和 `raw_syntax_error_count` 保留原始树的其他语法失败事实；位域模块零 pending 不代表该文件零语法错误。

每个位域记录包含：

- `declaration`：完整所属声明的原文和原始字节/行范围。逗号连写的多个位域共享该声明范围。
- `declarator`：该项的完整原文和范围，包括名字（若有）、冒号、位宽，以及独立默认初始化器（若有）。匿名项从真正冒号开始，不使用 MISSING 恢复位置。
- `name` / `name_range`：具名项保留原名和范围，匿名项均为 `null`。
- `bit_width` / `bit_width_range`：宽度表达式的原文与范围；`initializer` / `initializer_range` 单独表达 C++20 默认初始化，绝不把 `: N` 当初始化器。
- `bitfield_source_id`：包含源文件、物理 owner、声明位置、冒号位置和项序号的独立 source 身份。匿名项不会按空名字合并；这不是主生成器最终 semantic entity ID。
- `source_declaration_id`、`declarator_index`、`owner`、`preprocessor_conditions`、`parser_missing_name`。
- `semantic_validation="not_performed"`：模块是语法和来源记录，不自行证明位宽合法、布局正确或运行时语义有效。

`projection_edits` 只为已确认是匿名位域且 parser 缺少名字的项生成编辑：在真正 `colon_range.start_byte` 处作零宽插入，`parser_only=true`，`expanded` 与 `semantic_spelling` 都是空串，`parse_projection` 是唯一临时名加空格，附带 `bitfield_source_id`、`temporary_field_name` 和 `restored_name=null`。当前 18 个匿名位域分别生成 18 项。

接入时可将这些编辑传给已有 `SourceProjection`，使用其 `semantic_reader()` 恢复真实拼写；本轮测试已验证插入区间映射回同一物理零宽锚点、整份语义文本完全等于原始字节。核心仍需按编辑的 source ID 和插入区间恢复 `name=null`，绑定原始 `owner`、位宽与 initializer，不能全局字符串替换临时名，也不能沿用空名字 entity identity。**零宽位域 `:0` 的临时具名投影本身不具有合法 C++ 语义，因此只能送给 parser，不能拿去编译或当成语义源码。**

## 不能只数 `bitfield_clause`

全库原始 Tree-sitter 树有 1205 个名为 `bitfield_clause` 的恢复节点，其中很多实际来自带未展开注解的构造函数初始化列表；原始节点名字不能成为位域事实。例如 `CUTE_HOST_DEVICE S() noexcept : x(0) {}` 的 `: x(0)` 也可能被如此命名。

模块要求真实成员声明、物理分号、字段 identifier/匿名 missing declarator、可定位的 owner 和完整位宽。词法上还独立匹配 `?:`，检查名字和冒号之间是否夹有调用/三元式 token，避免把构造函数初始化中的条件表达式记成位域。未支持的带属性位域不会被静默丢弃，当前明确保留 pending；测试覆盖 `[[deprecated]]` 和 GNU `__attribute__` 两种情况。

## 824 文件全库结果

`audit_scope(data/scope.json, snapshot)` 遍历 manifest 的全部 824 个文件，逐项检查物理文件 SHA-256 与固定 manifest 相同。未缩小目录或架构范围。

| 项目 | 当前结果 |
| --- | ---: |
| 扫描文件 | 824 |
| 独立词法冒号候选（非 `::`，不含注释/字符串） | 12214 |
| 确认的物理位域 | 62 |
| 具名 / 匿名 | 44 / 18 |
| 匿名 source ID | 18 个互不相同 |
| parser-only 名字插入 | 18 |
| 已归为其他冒号用途 | 12049 |
| 未分类冒号候选 | 103，分布于 54 个文件 |

确认位域的文件分布：

| 文件 | 位域 | 匿名 |
| --- | ---: | ---: |
| `include/cute/arch/mma_sm100_desc.hpp` | 51 | 12 |
| `include/cute/arch/mma_sm90_desc.hpp` | 11 | 6 |

**62 是当前已确认的位域数，不是消除所有 pending 后的全库完整性声明。** 103 个未分类项都保留为 `bitfield_candidate_pending`，附带精确字节/行范围、候选 ID 和原始树祖先；不能把它们说成 103 个位域，也不能说它们已被排除。预处理宏体中的候选若不能识别则同样保留 pending，不暗中扩展或忽略。

最终 pending 分为 `colon_role_unresolved` 71 项、`non_identifier_declarator_before_colon` 24 项、`member_declaration_has_no_physical_semicolon` 7 项、`clause_not_in_field_declaration` 1 项。早期较宽的“冒号前是右括号就排除”规则得到过 71 项 pending，但该规则会误排合法宏生成位域 `FIELD(x):3`，已移除。现只有类名与构造函数名实际匹配（包括模板特化类的非特化 constructor 名字）才采用恢复构造函数分类。

以下列出 71 项 `colon_role_unresolved` 的位置，另外 32 项保留其更具体的上述原因；完整 103 项的文件、行号、字节锚点可由下方 `--summary` 命令直接重现。路径均相对固定 `snapshot/`；同一行两个冒号保留为两项而不是按行去重：

- `include/cute/arch/mma_sm100_umma.hpp`：1657、1658、1678、1679、1728、1729、1750、1751、1799、1800、1820、1821、1869、1870、1891、1892、1942、1943、1959、1960、2011、2012、2028、2029。
- `include/cute/atom/copy_traits_sm100_tma.hpp`：389。
- `include/cute/container/array_subbyte.hpp`：97、253；`include/cute/container/tuple.hpp`：274。
- `include/cutlass/complex.h`：176、182。
- `include/cutlass/conv/collective/sm100_implicit_gemm_umma_warpspecialized.hpp`：364。
- `include/cutlass/conv/conv2d_problem_size.h`：189；`include/cutlass/conv/conv3d_problem_size.h`：211。
- `include/cutlass/conv/threadblock/predicated_scale_bias_vector_access_iterator.h`：173。
- `include/cutlass/cuda_host_adapter.hpp`：264、290。
- `include/cutlass/epilogue/thread/linear_combination_bias_relu.h`：460；`include/cutlass/epilogue/thread/linear_combination_params.h`：57。
- `include/cutlass/epilogue/warp/tile_iterator_simt.h`：650；`include/cutlass/epilogue/warp/tile_iterator_tensor_op.h`：547。
- `include/cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized.hpp`：408；`sm100_blockscaled_mma_array_warpspecialized_rcggemm.hpp`：400；`sm100_blockscaled_mma_mixed_tma_cpasync_warpspecialized.hpp`：367。
- `include/cutlass/gemm/collective/sm100_mma_array_warpspecialized.hpp`：312；`sm100_mma_array_warpspecialized_blockwise_scaling.hpp`：385；`sm100_mma_array_warpspecialized_planar_complex.hpp`：296；`sm100_mma_array_warpspecialized_rcggemm.hpp`：307；`sm100_mma_mixed_tma_cpasync_warpspecialized.hpp`：280。
- `include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp`：97。
- `include/cutlass/gemm/warp/mma_simt_tile_iterator.h`：415、877；`mma_tensor_op_tile_access_iterator.h`：224；`mma_tensor_op_tile_iterator_sm70.h`：2402、2753；`mma_tensor_op_tile_iterator_sm80.h`：1750。
- `include/cutlass/pipeline/sm100_pipeline.hpp`：309、328、625、643、984；`include/cutlass/pipeline/sm90_pipeline.hpp`：329、383、388、820、837。
- `include/cutlass/semaphore.h`：78（两项）。
- `include/cutlass/subbyte_reference.h`：165、411、767、1141。

## 独立验证与复核命令

19 项单元/独立验证测试均通过：0-width、多个逗号项、`//` 后下一行真实冒号、UTF-8 字节位置、模板依赖位宽、位宽内三元式、C++20 初始化分离、raw string/注释、反斜杠续行注释、条件分支和宏体 pending、宏生成 declarator pending、属性 pending、缺位宽 pending、普通/特化构造函数恢复误判、固定两个 descriptor 文件、其他 asm 语法错误保持可见，以及 source projection 的逐段恢复。

Clang 21.1.8 使用 `-pedantic-errors -fsyntax-only` 交叉验证原始缩减源：合法 C++17 的匿名/零宽/UTF-8/模板位宽通过；C++20 位域初始化通过；具名零宽与缺宽度反例被拒绝。投影临时名不参加这些编译验证。

另有独立 Clang AST 核对：人工选择 SM100 的四段原始 struct body（103–112、277–285、417–433、446–462 行）与 SM90 的一段（109–125 行），只给外围匿名 struct 一个测试名、提供整数别名，所有字段字节保持不变。Clang JSON AST 中正好 62 个 `FieldDecl.isBitfield`、18 个匿名项；逐项 `name/null` 与 width 顺序和模块一致。输入区间不是从模块候选或其恢复树生成，因此不以模块自己的候选选择证明自己。

从项目根目录运行：

```bash
.venv/bin/python -m unittest tests.test_declaration_bitfields -v
.venv/bin/python scripts/declaration_bitfields.py --summary
```

第二条命令只向 stdout 输出统计、相关文件和 pending 的精确位置；不带 `--summary` 可获得完整候选账本、62 项字段、18 项投影编辑和所有 pending。两条命令不改快照，也不覆盖既有声明产物。

本报告对应文件散列：

- `scripts/declaration_bitfields.py`：`5d10fcde61fd6b1c526349e11fd4d711ca7fadd6c06bdef6da093b15d670f3fb`。
- `tests/test_declaration_bitfields.py`：`c697117c58568fb184b93acdeed8ce87a785799932d1b201628d092722142594`。

下一检查点是核心接入：在保留本模块 source 身份/精确范围与所有 pending 的前提下，把匿名成员从空名字合并恢复为独立项，并给具名/无名位域同时补宽度字段。该接入、全库 declaration JSON 重生成以及与独立候选账本的关联均未在本轮执行。
