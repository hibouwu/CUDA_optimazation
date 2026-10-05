# 阶段 1：匿名位域与 GNU asm 的独立有界审查

> **当前更正（2026-09-08）：撤回本文 ASM01 的“固定 CUDA 源码错误”定性。** NVCC 13.0.88 对这 12 个原文 wrapper 在 SM100a、SM110a 均能真实发射 PTX，并通过逐 wrapper 独立 kernel 的 PTXAS/SASS 验证；raw 与有逗号对照的 PTX/CUBIN 完全相同。应归为已测试 NVCC 方言与当前 parser 的兼容差异，而非修改快照的理由。详见 [NVCC 方言复核与撤回证据](phase-1-asm01-nvcc-dialect-recheck.md)。下文 GNU 主机编译器实验与原结论保留为历史记录，不再代表当前 CUDA 定性；匿名位域部分不受此更正影响。

审查日期：2026-09-08。只审查固定快照中的 `include/cute/arch/mma_sm100_desc.hpp` 与 `include/cute/arch/copy_sm100.hpp`。本轮只新增本审查记录，没有改快照、生成器、声明数据或候选账本。下文的临时名插入和逗号替换都只发生在 Python 内存中的对照文本，绝非固定源码修订，也不能写成生产提取器已修复。

## 结论与版本边界

这两个文件的诊断必须分成两类：匿名位域是当前 Tree-sitter C++ grammar 的限制；12 处 GNU asm 则是固定源码本身缺少操作数分隔逗号，不是解析器不支持 GNU asm。后一类不能靠删除函数体、掩去 asm、选择未启用分支，或者默默补逗号来宣布阶段 1 通过。

本次直接导入 `scripts.extract_declarations.Extractor`、对完整两文件运行后的基线：

| 文件 | 声明 occurrence | 诊断 | 原因 |
| --- | ---: | ---: | --- |
| `mma_sm100_desc.hpp` | 148 | 12 个 `missing_syntax` | 12 个无名位域被要求具有 `field_identifier` |
| `copy_sm100.hpp` | 713 | 12 个 `parse_error` | UTCCP 的输入操作数间缺少逗号 |

两文件均为 `normalized_parse_has_error=true`、`status=extraction_error_or_pending`。这只是两文件的当前直接提取结果，不是全库分母，也不是已落盘声明 JSON 的复核。

本次基线版本：

- Tree-sitter 0.25.2，tree-sitter-cpp 0.23.4。
- GCC 15.3.1 20260722，Clang 21.1.8；独立验证仅使用 `-std=c++17 -fsyntax-only -x c++ -`，没有 PTX 汇编、CUDA 设备编译或运行时验证。
- `scripts/extract_declarations.py`：`2ee8e7201fe880caeadc907934bec247fe014798d2bc9fd97aba2927a1befbd8`。
- `scripts/declaration_syntax.py`：`24d877b79ec055ea047f82e31484edb9dab0e2273100595bcbca85a9c626361e`。
- `scripts/declaration_projection.py`：`21504e059a7abefbc6c6f012ad037b5f3f1c8ca2afed7361f5f418179db0dbb4`。
- `scripts/macro_expansion.py`：`e6e201372012c3d28f4bc27e8ef29986c190a604442bca0e428964d593a322d2`。
- `scripts/namespace_bindings.py`：`0cd14ea32380d9faca7accb451208f7c276a7abf0bd4508b6e08474038c21524`。
- `snapshot/include/cute/arch/mma_sm100_desc.hpp`：`283cf6b4d27616a36ec165b9e42c096c7ce1bc86a80797e3a542dd08b0082b13`。
- `snapshot/include/cute/arch/copy_sm100.hpp`：`0fe82cbfc7c5a5e4629e16efa992d18801f1c136acb5fb63401f396e5965b305`。

## BF01：匿名位域是合法语法，当前 grammar 和字段模型均需补齐

真实源码 104 行：

```cpp
uint16_t start_address_ : 14, : 2;
```

最小反例不需要 CUDA、宏或头文件：

```cpp
struct S { unsigned a:14, :2; };
```

GCC 与 Clang 都通过。Tree-sitter 则产生：

```text
(field_declaration
  type: (sized_type_specifier)
  declarator: (field_identifier)
  (bitfield_clause (number_literal))
  declarator: (MISSING field_identifier)
  (bitfield_clause (number_literal)))
```

单独的 `struct S { unsigned :2; };` 也会触发同一错误，因此不是逗号连写才出现的问题。真实文件共有 51 个 `bitfield_clause`，其中 12 个无名，分布在实际冒号所在的 104、106、110（两处）、112、280、285、422、430、432、449、451 行。

不能直接把 MISSING 节点的坐标当插入点。后七处 MISSING 的恢复位置位于上一行 `//` 注释末尾，如 MISSING 报在 279 行，但真正无名位域冒号在 280 行。实测在这些 MISSING 坐标插临时名只消掉 5 个诊断，另 7 个临时名被插进了注释；沿后续兄弟节点越过 comment，定位实际 `bitfield_clause.start_byte` 后，再在冒号前插入 12 个唯一临时名，完整文件的诊断才变为 0。这个内存实验只验证投影路径可行，不证明数据模型已正确。

当前还有两个独立的字段问题，不能随语法诊断一同隐藏：

- 12 个无名位域现在被记录为 `name=""` 的普通 member，同一匿名 struct 内合并空名字身份，只有 4 个不同 `entity_id`，分别聚合 5、3、2、2 个 occurrence。它们不是同一命名成员的多次出现。
- `start_address_` 的 `initializer` 被写为 `": 14"`；`version_` 为 `": 2"`。位宽不是初始化器。具名和无名位域都需要独立 `bit_width` 与其原文区间，不能只处理无名项。

建议采用可溯源的 parser-only 临时名投影，同时保留真实语法模型：

1. 用成员声明结构和原始 token 边界确认“缺少名字 + 实际冒号 + 宽度表达式”，限定修复为无名位域，不能将任何 MISSING identifier 都补成合法字段。独立词法扫描应能识别首项、逗号后的项、注释换行和 `::` 的区别。
2. 临时名只插在真实冒号前，编辑记录保留零宽物理锚点、所属字段声明、原始 bitfield span 和 parser-only 原因；所有输出区间映射回原始字节。
3. 恢复 `name=null`，不让临时名进入 API、qualified name、raw signature 或 entity identity。无名位域采用 `anonymous_bitfield` 标识，独立身份至少包括语义 owner 与固定源码位置/声明内项序号；多个无名位域不得按空字符串合并，也不得与匿名 struct 身份混为一谈。
4. 为全部 51 项建立 `is_bitfield`、`bit_width`、`bit_width_range`、位域完整原文范围。若未来遇到合法位域默认初始化，再单独填写 `initializer`；当前 `: N` 不能占用 initializer。
5. 保留具名位域既有名字及 owner；核对 39 个具名 + 12 个无名的原文、宽度和相对次序，检查投影外所有声明的名字、范围和签名不变。临时名插入后零语法诊断只能是其中一项检查。

## ASM01：12 处缺逗号是条件分支中的真实源码错误

真实源码第一处位于 `SM100_UTCCP_128dp256bit_1cta::copy`：

```cpp
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
```

最小反例：

```cpp
void f(unsigned a, unsigned long b) {
  asm volatile ("" : : "r"(a) "l"(b));
}
```

GCC 报 `expected ')' before string constant`，Clang 报 `expected ')'`。将两个操作数之间加入逗号的对照文本，两编译器与当前 Tree-sitter 都通过。正常 `asm volatile ("" :: "r"(a), "r"(b));` 及带 `"memory"` clobber 的形式也被当前 Tree-sitter 接受。

独立编译检查直接取原文件 366–599 行的完整 UTCCP namespace，只提供 `<cstdint>`、空 Host/Device 注解和无操作的错误路径宏，保留所有 asm 文本。未定义 `CUTE_ARCH_TCGEN05_TMEM_ENABLED` 时，两编译器均返回 0；定义该宏时，两编译器均返回 1，分别产生恰好 12 个错误，位置为 380、399、417、435、455、474、493、512、531、550、571、592 行。这说明错误处于 SM100 TMEM 启用分支；不能把检查关闭分支的成功当成所有源条件均成功。主机 C++ 语法检查没有验证目标指令、约束的设备端适用性或 wrapper 运行行为。

在本文件内，全部 12 个现存语法诊断恰好归属下列 12 个 wrapper，没有据此推断其他文件也只有这类问题：

| wrapper（均位于 `cute::SM100::TMEM::UTCCP`） | 签名起始行 | body 起止行 | asm 错误行 |
| --- | ---: | --- | ---: |
| `SM100_UTCCP_128dp256bit_1cta::copy` | 374 | 376–384 | 380 |
| `SM100_UTCCP_128dp256bit_2cta::copy` | 393 | 395–403 | 399 |
| `SM100_UTCCP_128dp128bit_1cta::copy` | 411 | 413–421 | 417 |
| `SM100_UTCCP_128dp128bit_2cta::copy` | 429 | 431–439 | 435 |
| `SM100_UTCCP_4dp256bit_1cta::copy` | 449 | 451–459 | 455 |
| `SM100_UTCCP_4dp256bit_2cta::copy` | 468 | 470–478 | 474 |
| `SM100_UTCCP_4x32dp128bit_1cta::copy` | 487 | 489–497 | 493 |
| `SM100_UTCCP_4x32dp128bit_2cta::copy` | 506 | 508–516 | 512 |
| `SM100_UTCCP_2x64dp128bitlw0213_1cta::copy` | 525 | 527–535 | 531 |
| `SM100_UTCCP_2x64dp128bitlw0213_2cta::copy` | 544 | 546–554 | 550 |
| `SM100_UTCCP_2x64dp128bitlw0123_1cta::copy` | 565 | 567–575 | 571 |
| `SM100_UTCCP_2x64dp128bitlw0123_2cta::copy` | 586 | 588–596 | 592 |

这 12 个 wrapper 的声明签名目前仍正确提取为：

```cpp
CUTE_HOST_DEVICE static void
copy(uint64_t const& src_addr, uint32_t const& dst_addr)
```

已逐项断言 `kind=method`、完整限定名、参数顺序/类型/名字、`return_type=void`、`static`、raw signature 原字节回切与函数体首尾花括号范围。声明本身不在 `#if` 内，因此 occurrence 的 `preprocessor_conditions=[]` 正确；body 内 asm 诊断必须继续带 `defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)`。纯内存中把 12 个双空格改为“逗号+空格”后，完整文件成为 0 诊断，713 个 occurrence 的身份、签名、参数、范围和声明条件均与基线相同；该对照仅用于定位原因，不能加入生成器作为无痕源码修复。

### 如何精确保留 asm，而不伪造意图

应保留该原始 `gnu_asm_expression`、完整原始字节与条件，并把错误归类为 `confirmed_source_syntax_error` 或等价的明确状态。第一处 asm 表达式原文范围为 `[13772,13872)`，其中：

```text
[13843,13856)  "r"(dst_addr)   // ERROR 内仍有完整 gnu_asm_input_operand 子树
[13856,13858)                 // 两个空格；这是物理分隔文本，不是逗号
[13858,13871)  "l"(src_addr)   // 第二个输入操作数子树
```

若建立 asm 记录，可按原 token 记录两个“已观察到的 operand 片段”、constraint、value、各自范围、分隔区间的真实文本 `"  "`，以及 `operand_list_valid=false`。不能把它输出成“合法二操作数列表”，不能捏造 separator token，也不能据此声明 PTX 操作数绑定已验证。若以后为继续遍历而使用补逗号投影，原始源码错误仍须独立保留并关联该投影，补逗号只能标为诊断用的假设，不得写入 raw/semantic spelling 或清掉原始失败事实。当前 wrapper 声明已能提取，不需要靠这类投影恢复其签名。

固定快照不变且通过标准仍要求真实源码全语法有效时，这 12 项不会因完善 parser 而消失。应明确登记源错误的影响与证据，继续原定全库范围的其他核对，而不是缩小扫描范围或更换固定源码来得到漂亮计数。

## 可复核命令

在项目根目录运行以下命令。它只读源码，编译器从 stdin 读取，改动对照仅在内存内；没有输出文件写入。复核脚本将随生产生成器后续修复而产生不同的提取计数，须同时保留本节前面的版本散列边界。

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from collections import Counter
import re
import subprocess
from scripts.extract_declarations import Extractor, descendants

paths = ['snapshot/include/cute/arch/mma_sm100_desc.hpp',
         'snapshot/include/cute/arch/copy_sm100.hpp']
results = {}
for path in paths:
    source = Path(path).read_bytes()
    extractor = Extractor('bounded-review')
    record = extractor.extract(path, source)
    results[path] = source, extractor
    print(path, record['occurrence_count'], record['diagnostic_count'],
          record['normalized_parse_has_error'], Counter(d['category'] for d in extractor.diagnostics))

cases = {
    'unnamed_bitfield': 'struct S { unsigned a:14, :2; unsigned :1, b:3, c:1, :3; };',
    'asm_missing_comma': 'void f(unsigned a, unsigned long b) { asm volatile ("" : : "r"(a) "l"(b)); }',
    'asm_with_comma': 'void f(unsigned a, unsigned long b) { asm volatile ("" : : "r"(a), "l"(b)); }',
}
for compiler in ('g++', 'clang++'):
    print(subprocess.run([compiler, '--version'], capture_output=True, text=True).stdout.splitlines()[0])
    for name, source in cases.items():
        run = subprocess.run([compiler, '-std=c++17', '-fsyntax-only', '-x', 'c++', '-'],
                             input=source, text=True, capture_output=True)
        print(compiler, name, 'exit', run.returncode, run.stderr.strip())

source, extractor = results[paths[0]]
missing = [n for n in descendants(extractor.tree.root_node, include_anonymous=True)
           if n.is_missing and n.type == 'field_identifier']
edits = []
for n in missing:
    clause = n.next_named_sibling
    while clause is not None and clause.type == 'comment':
        clause = clause.next_named_sibling
    assert clause is not None and clause.type == 'bitfield_clause'
    edits.append((clause.start_byte, f'__parser_anon_bitfield_{len(edits)} '.encode()))
virtual = source
for byte, temporary in sorted(edits, reverse=True):
    virtual = virtual[:byte] + temporary + virtual[byte:]
after = Extractor('bounded-review').extract(paths[0], virtual)
unnamed = [o for o in extractor.occurrences if o.get('kind') == 'member' and o.get('name') == '']
print('unnamed occurrences/entities', len(unnamed), len({o['entity_id'] for o in unnamed}))
print('bitfield virtual projection', len(edits), after['diagnostic_count'])

source, extractor = results[paths[1]]
section = b''.join(source.splitlines(keepends=True)[365:599])
prologue = b'#include <cstdint>\n#define CUTE_HOST_DEVICE\n#define CUTE_INVALID_CONTROL_PATH(...) ((void)0)\n'
for compiler in ('g++', 'clang++'):
    for enabled in (False, True):
        unit = prologue + (b'#define CUTE_ARCH_TCGEN05_TMEM_ENABLED\n' if enabled else b'')
        unit += b'#line 366 "snapshot/include/cute/arch/copy_sm100.hpp"\n' + section
        limit = '-fmax-errors=0' if compiler == 'g++' else '-ferror-limit=0'
        run = subprocess.run([compiler, '-std=c++17', '-fsyntax-only', limit, '-x', 'c++', '-'],
                             input=unit, capture_output=True)
        errors = [line for line in run.stderr.decode().splitlines() if ': error:' in line]
        print(compiler, 'arch_enabled', enabled, 'exit', run.returncode, 'errors', len(errors))
        print('\n'.join(errors))

source_names = re.findall(rb'struct\s+(SM100_UTCCP_\w+)\s*\{', section)
wrappers = [o for o in extractor.occurrences if o.get('name') == 'copy'
            and 374 <= o['signature_range']['start_line'] <= 586]
assert len(source_names) == len(wrappers) == 12
for name, occurrence in zip(source_names, wrappers):
    assert occurrence['kind'] == 'method'
    assert occurrence['qualified_name'] == 'cute::SM100::TMEM::UTCCP::' + name.decode() + '::copy'
    assert occurrence['raw_signature'] == 'CUTE_HOST_DEVICE static void\n  copy(uint64_t const& src_addr, uint32_t const& dst_addr)'
    assert [(p['name'], p['type'], p['default']) for p in occurrence['parameters']] == [
        ('src_addr', 'uint64_t const&', None), ('dst_addr', 'uint32_t const&', None)]
    assert occurrence['return_type'] == 'void' and 'static' in occurrence['qualifiers']
    a, b = (occurrence['signature_range'][k] for k in ('start_byte', 'end_byte'))
    assert source[a:b].decode().rstrip() == occurrence['raw_signature']
    a, b = (occurrence['body_range'][k] for k in ('start_byte', 'end_byte'))
    assert source[a:b].startswith(b'{') and source[a:b].endswith(b'}')
    assert occurrence['preprocessor_conditions'] == []
print('ALL_12_WRAPPER_SIGNATURES_PASS')

needle = b'"r"(dst_addr)  "l"(src_addr)'
virtual = source.replace(needle, b'"r"(dst_addr), "l"(src_addr)')
after = Extractor('bounded-review')
record = after.extract(paths[1], virtual)
keys = ['declaration_occurrence_id', 'entity_id', 'kind', 'name', 'qualified_name',
        'raw_signature', 'parameters', 'return_type', 'qualifiers', 'signature_range',
        'body_range', 'preprocessor_conditions']
before_fields = [{k: o.get(k) for k in keys} for o in extractor.occurrences]
after_fields = [{k: o.get(k) for k in keys} for o in after.occurrences]
print('asm hypothetical comma count/length preserved/diagnostics/identity-signature-range invariant',
      source.count(needle), len(virtual) == len(source), record['diagnostic_count'],
      before_fields == after_fields)
PY
```

本次记录的核心结果：匿名位域两个编译器均 exit 0；缺逗号反例均 exit 1；带逗号对照均 exit 0；12 个无名位域只有 4 个现有 entity；冒号锚点临时名投影后 0 诊断；完整 UTCCP namespace 未启用时均 exit 0/0 error、启用时均 exit 1/12 errors；假设性逗号对照为 `12 True 0 True`。

## 全库关闭条件，不以这两个样本替代

应增加独立词法/前端对照验证，而不是复用生成器的修复选择逻辑来证明自身完整：全库枚举位域声明及 asm 表达式候选，逐一映射到原文跨度和诊断/提取记录；匿名位域的 null 名、唯一 ID、位宽和初始化器分离需要逐项校验；每个 asm 原 token、条件和错误分隔片段须完整保留。对源码错误与 parser 限制使用不同分类并保留统计分母；其余不属于已验证两类的错误继续保持 pending。全库重跑后还要比较声明身份、作用域、参数/返回类型和原始区间，不能以诊断减少或本文两个内存实验成功替代阶段 1 完整性证明。
