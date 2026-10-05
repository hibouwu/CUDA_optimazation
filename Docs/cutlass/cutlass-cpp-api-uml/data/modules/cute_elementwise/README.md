# CuTe clear / fill / axpby：三个完整 header 的关系草稿

本草稿审查固定 snapshot 中 `include/cute/algorithm/clear.hpp`、`fill.hpp`、`axpby.hpp` 的全部物理源码，不选择 Dense 或某个 Engine/Layout 配置。它覆盖这三个文件的声明、表达式、控制结构和源码资源关系；**没有把“调用图画完”改称所有模板实例已验证**。依赖 header 只核必要声明及证据，不加入本模块的全文件覆盖分母。

经root从三个原始文件审查，并修正非依赖表达式被误标symbolic、宏展开缺少关系等反例后，本工作包已接入`data/modules/cute_elementwise/`。原始草稿保留在`data/module-drafts/`。未修改snapshot、全局提取器或Dense源码关系。接入不代表模块使用验收已完成。

## 文件与复跑方式

- `relations.json`：当前图集可读取的 `nodes / edges / views / contracts / issues`，并附完整 `coverage` 义务和文件散列。
- `check_source.py`：独立、只读验证器。不导入制作脚本；重新解析三个完整文件，逐个检查表达式范围、覆盖义务、全局声明身份及现有图集兼容性。
- `test_source_check.py`：针对遗漏调用、同一行调用被合并、把 decltype 误记为运行时调用、遗漏默认参数、字节漂移、遗漏运算表达式的负例回归。
- `author_draft.py`：可复核的草稿制作逻辑，只向 stdout 输出 JSON；不写 snapshot、ledger 或 site。

从 atlas 根目录执行：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python data/modules/cute_elementwise/check_source.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s data/modules/cute_elementwise -p 'test_*.py' -v
```

检查器还只读调用当前 `build_atlas.enrich_nodes`、`validate_evidence`、`diagram_source`：API 选择器须能关联真实 ledger 声明，人工局部声明须回切原始字节，证据须匹配 snapshot，各 view 必须包含全部边端点。它不生成或覆盖 root 正在构建的 HTML/SVG。

## 八个函数模板，一个也不按配置隐藏

| 源文件与声明行 | 独立接口节点 | 源码行为与条件 |
| --- | --- | --- |
| clear.hpp:46 | `elementwise.api.clear_rvalue` | 接收 `Tensor&&`，命名 `tensor` 以 lvalue 表达式转发。 |
| clear.hpp:57 | `elementwise.api.clear_lvalue` | 局部 `T=Tensor::value_type`，调用 `fill(tensor, T{})`。 |
| fill.hpp:47 | `elementwise.api.fill_rvalue` | 接收 `Tensor&&`，转发命名 tensor 和 value。 |
| fill.hpp:59 | `elementwise.api.fill_prefer1` | `decltype(fill(tensor.data(), value))` 检验候选；函数体 62 才实际调用。 |
| fill.hpp:69 | `elementwise.api.fill_prefer0` | fallback：`int i` 逐个逻辑坐标执行 `tensor(i)=value`。 |
| fill.hpp:82 | `elementwise.api.fill_lvalue` | 固定构造 `prefer<1>`；保留高优先候选与替换失败后的 `prefer<0>` 候选。 |
| axpby.hpp:50 | `elementwise.api.axpby_rvalue` | 接收 `y&&`，保持另外四个参数，以命名 y 转发。 |
| axpby.hpp:69 | `elementwise.api.axpby_lvalue` | 先执行零 beta 判定 lambda，再遍历 `size(x)`，由 `p(i)` 和 beta 分支控制表达式。 |

每个节点复用全局 `entity_id / declaration_occurrence_id`，并保存完整原始签名和模板/函数形参。三个 `namespace cute` 出现共享一个全局 namespace 实体，但保留三个独立 occurrence；`namespace cute::detail` 单独登记。总计 **12 个 scope 内全局声明 occurrence**，不是只计八个函数。

局部声明不冒充全局 ledger 实体：clear 的局部别名 T、fill 的 i、axpby 的 i、isBetaZero 对象各有独立源码身份；lambda 另有准确捕获、签名和定义范围。lambda 不是 axpby 的成员 API。

## 完整覆盖义务

`relations.json.coverage.obligations` 对下列 **91 项**逐一保存原文、原始起止字节、行号、种类、审核状态和图中引用；`parameter_obligations` 另存 **52 项参数声明**。相同一行、相同拼写、嵌套表达式都不靠名字合并。

| 覆盖类别 | 数量 | 范围与注意点 |
| --- | ---: | --- |
| 函数模板定义 | 8 | clear 2、fill 4、axpby 2。 |
| namespace 出现 | 4 | 三个 cute、一个 cute::detail。 |
| 局部声明 | 4 | T、isBetaZero、两个独立 i。 |
| lambda 定义 | 1 | axpby 75–84；其立即调用另记 callsite。 |
| call-expression 物理位置 | 20 | 18 个可能求值，2 个仅位于 decltype 内。 |
| 显式值初始化 | 5 | `T{}`、`prefer<1>{}`、三个 `Int<0>{}`。 |
| 省略函数实参时的默认初始化 | 2 | 两个独立 `p = {}`；与模板默认 `PrdTensor` 区分。 |
| 运算表达式 | 15 | 比较、逻辑与、赋值、条件表达式、乘加和两个 ++i；保留嵌套位置。 |
| 控制结构 | 4 | 两个 for、一个 if constexpr、一个 predicate if。 |
| return 语句 | 6 | 转发及 lambda 各分支的原文，不制造额外“返回 API”。 |
| include 指令 | 8 | clear 3、fill 3、axpby 2；每处有明确 target_path/file_dependency，依赖文件不是额外全文件 scope。 |
| pragma once | 3 | 各 header 一处，不当运行时操作。 |
| 注解/编译提示宏使用 | 11 | 8 个 CUTE_HOST_DEVICE、2 个 CUTE_UNROLL、1 个 CUTE_GCC_UNREACHABLE。 |
| 模板形参声明（另计） | 30 | 含两处 `PrdTensor = constant_fn<true_type>`。 |
| 函数形参声明（另计） | 22 | 保留两个无名 prefer 参数、引用 cv 限定及两个 `p = {}`。 |

版权/许可证是各文件 1–30 行；注释和空行不产生 API 或运行调用，原文完整保留在 snapshot。头文件的 include、pragma、namespace 则已经有独立义务，未被当作空白丢弃。

独立语法枚举只对 CUTE_HOST_DEVICE/CUTE_UNROLL 注解做等长空白投影；Tree-sitter 对两个省略类型的默认 `{}` 有语法限制，因此检查器也等长遮蔽其 initializer，然后以原始字节单独核对两项默认参数义务。没有移除任何函数 body、分支或运算式。

`CUTE_GCC_UNREACHABLE` 不只停在语法清单：另外建立词法宏使用与三个条件 `expands_to` 分支——未预定义且 GNU 环境展开为外部 `__builtin_unreachable()` 边界，未预定义且非 GNU 环境为空，调用环境已预定义时保留外部覆盖的依赖。它在两个 return 分支之后，因此不计成一次真正可执行的运行 calls。另八个 HOST_DEVICE 是属性、两个 UNROLL 是 pragma/空展开，不能混作 builtin 调用。

## 从源码能说清什么

`clear` 采用的是元素类型的 `T{}`，并非 `memset`，不能保证任意用户类型的值初始化都得到数学零或全零字节。`fill` 先尝试 iterator/data 层的依赖调用；只有这一表达式替换失败才通过 `prefer<1>` 到 `prefer<0>` 的基类转换选择 fallback。一个分派 callsite 对应两个互斥条件目标，不是实际调用两次。

高优先 fill 的 `decltype` 及其中的 `tensor.data()` 都不求值，不能计成额外一次运行调用。其返回类型严格是依赖表达式的 `decltype`，函数体却没有 `return` 语句；如果用户定制的 data-fill 返回非 void，不能仅凭候选通过 SFINAE 就承诺这一包装函数有定义的返回行为。草稿保留这个源码条件，没有补造返回语句。

`axpby` 在进入元素循环之前求值 lambda 一次。复数分支分别记录 `beta.real()` 与 `beta.imag()`，并保留模板运算和逻辑与的短路条件；普通分支比较 `beta` 与 `Int<0>{}`。每个 i 先检验 p，再进入赋值。89 行的两个 x(i) 和两个 y(i) 分别有独立字节范围和条件。零 beta arm 不求值源码中旧 y 的右侧子表达式，但不能将这个结论外推到任意用户重载或代理引用内部。

资源视图单独表示 tensor/value、x/y、alpha/beta/p、isBetaZero 和两个循环 i 的读取或更新。底层 Engine 地址空间未固定，没有凭算法名字制造 SMEM、TMEM、硬件指令或异步完成事件。源码不检查 x/y 的 shape 相等，也不保存一份输入快照；访问有效性、潜在别名、并发同步和生命周期由实际调用方/实例保证。

运算和控制结构使用 `relation: evaluates`，不是 `calls`。当前生成器可以用已有 fallback 渲染这一关系。两个 int `++i`、固定 `prefer<1>{}` 与三个 `Int<0>{}` 均标为 `source_proven`，不是模板待决调用。for/if 是已知的源码控制容器，其结构同样为 `source_proven`；内部依赖条件另列，不制造一个容器自己的“未解析 API”。`T{}`、`PrdTensor` 默认初始化及依赖的乘加/赋值等继续保留 `symbolic`。六个 view 分别覆盖 clear、fill 分派、axpby、资源/顺序、必要类型依赖和预处理展开。

## 尚未完成的语义闭合与验证边界

1. 任意 Engine/Layout/T 所触发的 ADL fill、非限定转发与 size 候选的最终决议未统一实例化。它们保留原表达式、模板参数和条件，状态是 `symbolic`，不是解析失败。
2. 任意 Alpha/Beta/PrdTensor、元素及代理引用的运算、转换、用户特化、隐式特殊成员和副作用没有穷举。默认 predicate 与 cutlass::complex 的已知情况连到真实声明，但条件实例不冒充全部模板情况。
3. tensor_impl、prefer、functional、integral_constant、complex、config 的必要声明/证据已核；其全部传递函数体不属于这三个完整文件的覆盖分母。
4. 未声称编译实例化、GPU 运行、数值正确性、性能或机器指令得到验证。位置检查 PASS 是源码与关系数据一致性，不是这些执行层面的 PASS。

因此当前结论是：这三个物理 header 的已列源码义务已逐项审查；模板依赖语义仍开放。草稿需独立审查后才接入模块 registry，不能拿 3 个文件的完成情况代替 824 个库文件的整体进度。
