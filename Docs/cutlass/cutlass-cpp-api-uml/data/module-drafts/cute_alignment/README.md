# CuTe alignment：两个完整物理header的工作包草稿

本目录登记固定snapshot中的 `include/cute/container/alignment.hpp` 和 `array_aligned.hpp` 全部书写出来的声明、宏分支、表达式和alignment义务。**它不是正式module manifest，也没有宣布API覆盖验收通过。** 目录主归属仍由root的全局文件登记表管理；本草稿是两个完整文件的有界工作包。

没有读取或复用旧global ledger里被误读成`CUTE_ALIGNAS`的类型身份。全部声明先以精确path/line/signature span/source hash selector和临时manual定位保存。root完成alignment适配后，应核对selector、替换为正确canonical entity/occurrence，随后撤掉临时manual覆盖再正常enrich；不能把manual hash当作修复后的全局身份。

## 文件与重跑

- `relations.json`：与elementwise草稿相同的nodes/edges/views/contracts/issues主体，加scope和coverage义务。
- `author_draft.py`：制作脚本，只写本目录relations.json；不读global ledger，不修改core/snapshot/site。
- `check_source.py`：独立检查器，不导入制作脚本或提取器；重新解析两份完整源文件并核对源码范围、声明selectors、宏分支、属性owner和关系分类。
- `test_source_check.py`：15项正常/破坏测试，覆盖遗漏偏特化、丢分支/属性、默认值篡改、错误旧身份、cast/runtime误分类、签名吞入body、类型引用范围带入参数名等。

在atlas根目录：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python data/module-drafts/cute_alignment/author_draft.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python data/module-drafts/cute_alignment/check_source.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s data/module-drafts/cute_alignment -p test_source_check.py -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python data/module-drafts/cute_alignment/check_source.py --clang
```

最后一项是可选Host编译oracle，不运行生成程序或GPU；没有验证CUDA宏分支。检查器在独立的等长解析视图中把CUTE_ALIGNAS关键字换成同宽的alignas加空白，原参数位置不变；原始宏调用、两个真实定义与属性合同另行完整保存。它不是root适配的实现，也不会因投影视图能解析就给API覆盖打勾。

## 完整源声明与关系

两文件共有16项物理声明selector：namespace cute的2个出现、is_byte_aligned函数模板1个、aligned_struct主模板1个及9个独立偏特化、array_aligned主模板1个、CUTE_ALIGNAS的2个条件定义。未把所有模板实例或隐式特殊成员计成这16项。

`is_byte_aligned<N>(void const* const ptr)`保留int非类型模板参数N、指针本身const、指向void const、constexpr和CUTE_HOST_DEVICE。47行唯一真正的函数调用是 `has_single_bit(N)`，位于static_assert的常量求值中，T由N的声明类型绑定为int。函数体48行的reinterpret_cast、整数减法、按位与、相等比较和return分别登记，均不是额外API调用，也不解引用ptr。

这里特别修正了两种语法树名称误导：Tree-sitter把reinterpret_cast记作call_expression，本工作包改按named_cast义务记录；它把size_t/uintptr_t称primitive_type，但本工作包仍保留外部整数typedef边界，不假造语言内建类型或库内声明来源。具体外部声明位置/ABI位宽仍未归属到本工作包。

`aligned_struct<Alignment,Child=void>`主模板是无alignment属性的空struct，**不继承Child，也不保证任意Alignment都得到对应对齐**。只有Alignment为1、2、4、8、16、32、64、128、256的九个偏特化各自施加CUTE_ALIGNAS；Child仍是模板参数，不是显式特化，也不是基类或数据成员。

`array_aligned<T,N,Alignment=16>`明确public继承 `cute::array<T,N>`，并对自身施加请求的Alignment。继承和T/N绑定单独连到array.hpp:42的真实依赖声明，不制造一次运行期构造调用；array header的全部函数体不进入这两个header的完整文件分母。

十个alignment调用均有独立attribute节点、owner关系和两个expands_to分支：`defined(__CUDACC__)`对应 `__align__(expression)`，反条件对应 `alignas(expression)`。宏definition虽然书写在namespace cute文本内，预处理名字仍是全局CUTE_ALIGNAS，不能归成cute::CUTE_ALIGNAS。另一个CUTE_HOST_DEVICE用途保留cute/config.hpp中两条条件定义。

## 义务分母与已跑检查

当前100项源码构造义务，加18项单独参数义务。100不是API数量，部分构造有父子或重叠范围：

| 分类 | 数量 |
| --- | ---: |
| pragma once / include | 2 / 5 |
| namespace出现 / API template wrapper | 2 / 12 |
| function definition / struct declaration | 1 / 11 |
| CUTE_ALIGNAS条件定义 / 条件指令 | 2 / 3 |
| alignment宏调用 / CUTE_HOST_DEVICE调用 | 10 / 1 |
| 真正函数调用 / named cast | 1 / 1 |
| static_assert / 二元运算 / return | 1 / 3 / 1 |
| 带语义角色分类的type-spelling token | 42 |
| 模板默认值 | 2 |
| 另列模板/函数/宏参数 | 15 / 1 / 2 |

版权、许可证、注释和空白原文保留于snapshot；没有把include、namespace、宏条件等当作空白丢弃。所有graph edge均进入至少一个view，5个view分别用于指针对齐谓词、偏特化关系、array派生关系、属性/宏分支及基础类型边界。

本轮独立源码检查与15项测试通过。Clang21.1.8使用原snapshot header及实际CUDA13.0头文件/CCCL include目录，Host oracle得到：

- aligned_struct<128,Child>对齐为128且不是Child的派生类；未列明的aligned_struct<512>选择无属性主模板，本Host样本alignof为1。
- array_aligned<char,1>默认对齐16，array_aligned<float,3,64>确为array<float,3>派生类。
- is_byte_aligned<8>(ptr)可实例化；N=3在原static_assert被拒绝。
- `static_assert(is_byte_aligned<8>(nullptr))`在C++17被拒绝，因为原reinterpret_cast不能用于该常量表达式。constexpr函数声明不能被直接宣传成所有指针调用均可常量求值。

这些oracle只证明列明Host样本，没有GPU运行、CUDA分支编译或任意Alignment/T/N/Child实例全集证明。函数返回指针整数表示的掩码比较，不提供存储生命周期、内存访问合法性或同步保证。

## 交付边界

`status`保持draft_pending_canonical_enrichment_and_independent_review，`coverage.api_coverage_passed=false`。旧全局错误CUTE_ALIGNAS身份未使用；canonical enrichment及独立语义审查仍待root接续。源构造检查成功不能替代这些步骤，也不代表其它array实现、配置宏header或824文件全库关系已经完整。

Root适配完成后，本轮另用当前Extractor只读检查两文件全部16个声明selector，kind/name/签名字节起止均exact匹配，没有复制其entity_id。has_single_bit依赖签名已收紧到`[3911,3982)`，不含129行的body开括号；主函数`[2007,2095)`、array依赖`[1972,2014)`也逐字节核对。两个namespace selector保留body前换行，两个array模板参数的类型引用边只指精确size_t token；完整形参仍在parameter_obligations中。
