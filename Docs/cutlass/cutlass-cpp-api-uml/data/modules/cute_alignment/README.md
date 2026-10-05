# CuTe 对齐属性与对齐容器

这个工作包覆盖 `include/cute/container/alignment.hpp` 和 `include/cute/container/array_aligned.hpp` 两个完整文件。依赖的 `array`、`has_single_bit` 保留真实声明入口，但不把它们所在文件算作本工作包已经完成的文件。

模块接入图集后，可从“CuTe 对齐属性与对齐容器：完整文件接口”进入五组视图：指针对齐检查、主模板与偏特化、array 继承关系、对齐属性与宏分支、基础类型依赖。每条关系另有完整签名图；不同偏特化、宏定义分支和物理使用位置不合并。

## 适合在接口会上核对的内容

- `aligned_struct<Alignment, Child = void>` 主模板没有对齐属性。只有源码列出的九个偏特化分别请求 1 至 256 字节的相应对齐；不能根据模板参数的名字推断任意值都会生效。`Child` 没有成为基类或成员。
- `array_aligned<T, N, Alignment = 16>` 继承 `cute::array<T, N>`。默认值为 16，但实际模板参数仍可改变。继承、模板绑定与对齐属性分别记录，不画成一次构造调用。
- `CUTE_ALIGNAS(n)` 在 `__CUDACC__` 条件下展开为 `__align__(n)`，否则为 `alignas(n)`。每个使用位置都能找到属性所属类型、原始表达式、两个定义位置及实际参数替换结果。
- `is_byte_aligned<N>(ptr)` 中的 `has_single_bit(N)` 用于 `static_assert` 的常量求值。函数返回表达式只检查指针整数表示的低位，不读取所指内存，也不提供内存有效性、生命周期或同步保证。`constexpr` 声明不保证这个指针转换在 C++17 中能用于常量表达式。

两文件共 16 个物理声明出现位置，包括 namespace 的两个出现和宏的两个条件定义。这不是“16 个不同 C++ API”，也不包含无穷的模板实例。另有 100 项源码构造义务、18 项参数义务；这些分类用于逐项核对，不能当作全库完成比例。

## 重生成与复核

在图集根目录执行：

```bash
# 独立检查原文义务与故意破坏后的拒绝行为。
.venv/bin/python data/module-drafts/cute_alignment/check_source.py
.venv/bin/python -m unittest discover -s data/module-drafts/cute_alignment -p test_source_check.py
.venv/bin/python -m unittest discover -s tests -p test_alignment_stage_review.py

# 与已通过机械核对的当前全库声明账本接合，再生成站点。
.venv/bin/python scripts/stage_cute_alignment.py
.venv/bin/python scripts/build_atlas.py
.venv/bin/python scripts/audit_atlas.py
```

制作稿和独立检查器保存在 `data/module-drafts/cute_alignment/`。接入脚本不只比较数量，还核对声明位置的完整集合、限定名、独立实体、属性所属声明、模板默认值及两条宏定义。人工定位保留为历史记录，当前声明身份来自修复后的全库账本。

Host Clang 正反例验证了列明的继承、对齐和约束行为；它不代表 CUDA 分支的布局或任意模板实例都经过编译，更不是 GPU 运行验证。SVG 静态检查、离线链接检查、DOM 代码测试与真实浏览器会议使用验收分别记录；没有把本工作包或某个样本的通过推广成 824 个文件的完整交付。
