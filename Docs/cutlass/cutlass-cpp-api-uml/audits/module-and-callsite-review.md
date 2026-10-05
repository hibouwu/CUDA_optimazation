# 文件模块归属与 callsite 校验：独立只读审查

审查日期：2026-09-08。固定 CUTLASS commit：`8f50b052e1099fb982392a622caab69b97b63128`。本轮作者没有参与 registry 或 `validate_evidence` 两处实现；仅运行只读检查/内存反例，并新增本审查文件。未修改被审脚本、registry 数据、snapshot 或 Dense 模块。

## 当前结论

registry 当前数据通过：824 个文件与 scope 精确集合相等、逐文件散列相等、每个文件恰有一个规则归属，40 个 .inl 没有排除。模块 ownership 是导航/责任划分，不是声明、关系或协议已完成的证明。源码抽查没有发现冷门文件、混合代际 header 或跨模块依赖被隐藏。

callsite 的主体修复成立：重复/重叠 evidence 对同一物理调用按字节去重；同一行相同表达式保留两个位置；正确 authored 字节范围可以选中一个，错误字节不会被静默改到另一个；context/caller 在成功的有效选择中保留。两轮独立反例现已由 root 修正，最终对既有 20-case 矩阵复核无未关闭的本轮问题；见末尾冻结记录。

独立反例找到两轮输入一致性问题。第一轮的 library `line` 别名和带字节 authored 表达式不一致问题，root 已修并通过定点复核。第二轮发现相同检查仍未覆盖 line-only authored metadata 与辅助材料的 `line` 别名，也已修正；修后结果在本文末尾追加，不删除旧证据。

## 审查对象与散列

初始读取时记录：

| 对象 | SHA256 |
| --- | --- |
| scripts/build_module_registry.py | 1968eaa2936e51196b25d5dd03e2787f2703bce7ad3e3e253ba7c2a76e10e472 |
| data/module-registry.json | 31adb2398feb5481fd423ba0743e07156815d75bcf4fb056fea91e28603411a3 |
| data/scope.json | 2b8a0094ff40dcd0a47c4da64e309fda8af98dc1ecdeb9b28a6e8a8692c2970f |
| scripts/build_atlas.py（初始读取） | 539d76563817f758f137afbc17e47ee2b02e02e25d42e52f00bfcfe35cc31669 |
| tests/test_module_registry.py | f2d0d2ee2dee43fe63ef43baf103bff1f07cb31ecfdf725e8ea69e8b9e496638 |
| tests/test_atlas_build.py（8 tests 时） | e51382714b816041993847172307bc41e9c15f6ea03466e87800bf9a8ce88462 |

root 同时进行 UI/覆盖清单接入，因此 build_atlas 文件级散列会改变；下面的反例明确记录输入、行为和修订次序，不把文件级变化全部归因于 callsite 修复。

## 1. Registry：无排除、无越界、无虚假覆盖宣称

独立检查不只验证总数：先比 `registry.files.path` 与 `scope.files.path` 的完整集合，再逐一读取固定 snapshot 的 824 个文件并计算 SHA256，最后逐文件独立匹配规则，并在内存重建整个 registry 与已落盘 JSON 比较。

结果：

- CUTLASS 712、CuTe 112，总计 824；20 个主归属模块，所有模块计数之和仍为 824。
- 扩展名为 .h 494、.hpp 290、.inl 40。40 个 .inl 分别归入 convolution 4、epilogue_collective 6、gemm_collective 30。
- 未分配 0，重复分配 0；scope/generator 散列与 registry 中声明的散列匹配。
- 每条 `coverage` 保留 `declarations / relationships / protocols = not_asserted`。目录、架构名称、版本提示没有升格为 API 已完成。
- 规则按独立匹配要求唯一，不是“先匹配者胜出”；SYNC/RUNTIME/detail 例外从较宽规则扣除后仍在其他模块被完整接住。
- 固定 source path 使用 `include/...`，拒绝相对回退、绝对路径和未匹配新域；不是按“是否被 Dense 调用”决定保留。

源码语义抽查：

| 实际源码 | 可复核原文位置 | 主归属与判断 |
| --- | --- | --- |
| include/cute/arch/tmem_allocator_sm100.hpp | Allocator1Sm:60、Allocator2Sm:117，allocate/free:135/159 | sync_resource：生命周期原语，没有因为在 cute/arch 下就强归 MMA。 |
| include/cutlass/arch/barrier.h | NamedBarrier:181、ClusterBarrier:342、ClusterTransactionBarrier:546 | sync_resource：同步资源例外显式保留。 |
| include/cutlass/detail/sm100_tmem_helper.hpp | find_tmem_tensor_col_offset:49、make_sm100_accumulator:58 | collective_support：形状/布局辅助，不冒充实际 allocator API。 |
| include/cutlass/detail/layout.hpp | TagToStrideA:54/60/67 | cutlass_foundation：基础布局转换例外；仍为范围内文件。 |
| include/cute/util/print_svg.hpp | print_svg_mma:83、print_svg:230/240 | cute_core：打印/辅助接口未因“冷门”排除。 |
| include/cute/atom/partitioner.hpp | TV_Tiler:52、TV_Partitioner:78 | cute_core：partitioner 明确归属，不被 mma/copy 前缀误吸收。 |
| include/cutlass/thread/matrix.h | Matrix:53 | cutlass_foundation：兼容 thread matrix 未被排除。 |
| include/cutlass/platform/platform.h | 编译器分支:104/128/173、namespace platform:216 | cutlass_foundation：平台/兼容分支保留。 |
| include/cutlass/experimental/distributed/device/dist_gemm_universal_wrapper.hpp | DistributedGemmUniversalAdapter:53、DistributedGemmState:134 | distributed：experimental 不表示忽略。 |
| include/cutlass/gemm/device/gemm_universal_adapter.h | 3.x 区域:85、adapter:123；2.x 区域:626、adapter:630 | gemm_device：同文件一份 ownership，但 api_generation 仍 pending_entity_review，未把文件整体判成某一代际。 |
| include/cute/algorithm/clear.hpp、fill.hpp、axpby.hpp | 本轮另有完整三个 header 草稿 | 均归 cute_core；子工作包不是另一份文件主归属。 |

上述抽查核对“归属解释与源码对象相容”，不是声称这些依赖文件的每个 API/协议已完成。

复跑（从 atlas 根目录运行，纯只读）：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p 'test_module_registry.py' -v
```

初始结果为 9 tests PASS。完整集合/散列/内存重建检查：

```python
from pathlib import Path
import hashlib, json, sys
sys.path.insert(0, 'scripts')
import build_module_registry as m
scope_raw=Path('data/scope.json').read_bytes()
scope=json.loads(scope_raw)
registry=json.loads(Path('data/module-registry.json').read_text())
source={x['path']:x for x in scope['files']}
owned={x['path']:x for x in registry['files']}
assert source.keys()==owned.keys() and len(owned)==824
for path,item in source.items():
    assert owned[path]['source_sha256']==item['sha256']==hashlib.sha256((Path('snapshot')/path).read_bytes()).hexdigest()
    assert owned[path]['primary_module_id']==m.classify(path)['module_id']
    assert owned[path]['coverage']=={'file_ownership':'assigned','declarations':'not_asserted','relationships':'not_asserted','protocols':'not_asserted'}
rebuilt=m.build_registry(scope,scope_sha256=hashlib.sha256(scope_raw).hexdigest(),
    generator_sha256=hashlib.sha256(Path('scripts/build_module_registry.py').read_bytes()).hexdigest())
assert registry==rebuilt
print(json.dumps(registry['counts'],ensure_ascii=False,indent=2))
print('PASS fixed scope, all 824 physical hashes, ownership, deterministic regeneration')
```

## 2. Callsite：独立内存反例矩阵

fixture 使用 `unittest.mock` 提供内存中的虚拟 header/aux 字节，不创建这些文件，不改真实 scope。检查涵盖真实 byte offset、UTF-8 前缀、CRLF、同一行重复、重复 evidence、重叠 evidence、完整及不完整 authored metadata。

| 反例 | 观察结果 |
| --- | --- |
| 同一个 evidence 重复列两次 | 只定位一次同一物理调用；不误报多位置。 |
| evidence 1–3 与 2–2 重叠，只有 2 行一个调用 | 按相同字节键去重；定位到 13:17。 |
| 同一行两个 g(1)，无 authored bytes | callsite_not_uniquely_located；不合并。 |
| authored bytes=17:21 指向第二个 g(1) | PASS；保留第二个位置、context、caller。 |
| bytes 起点错误、越界、只给 start_byte | authored_callsite_not_matched。 |
| 正确 bytes 但错误行号或 raw_source_expression | authored_callsite_fields_disagree。 |
| 中文 UTF-8 前缀 + CRLF | PASS；g(1) 精确位于 22:26，第 2 行。 |
| 已执行 package 阶段的含空格 aux 路径 | PASS；保留离线 evidence URL，按 aux 原字节定位。 |
| 合法 aux 路径但未提供 package 阶段产生的 source_text | 不匹配；这是当前函数的前置阶段依赖，不是错误定位。 |
| aux author 路径与实际 evidence 路径不同 | authored_callsite_not_matched。 |
| /etc/passwd 或逃出允许目录的辅助路径 | auxiliary_path 返回 None；无越界读取。 |

### CALL01：library evidence 的 line 别名

初始最小输入：

```python
edge = {
    'id': 'e', 'relation': 'calls', 'source_expression': 'g(1)',
    'evidence': [{'path': 'include/test.hpp', 'line': 1}]
}
# 内存源码：void f() { g(1); }
```

范围检查先接受 `evidence.get('start_line', evidence.get('line'))`，但后续 calls matcher 直接访问 `evidence['start_line']`，因此抛 `KeyError: 'start_line'`。这不是源码 C++ 解析错误，而是工具输入字段归一化不一致。

root 第一轮修复为范围验证后填入 start_line/end_line。独立复跑同一 library 输入现在 PASS。审查时对实际模块 parts + 草稿共 182 条 calls/launches 查找，这种 line-only evidence 出现为 0，因此不推断原有模块已经被该反例破坏。

### CALL02：authored 表达式一致性

正确字节 11:15 指向 g(1)，但 `authored.source_expression='h(2)'`。初始实现只检验 raw_source_expression、行号；此字段被新定位内容静默覆盖。root 第一轮修复增加 token 一致性检查；带 bytes 的同一反例现在返回 authored_callsite_fields_disagree，错误 caller 的带 bytes 变体也得到相同诊断。

### CALL03：第一轮修复的剩余条件分支

为避免只通过列举的两个样例，独立审查继续检查同一字段的合法形态：

1. authored 仅有 path/start_line/caller（没有 bytes），caller 故意与 edge.source 不同：第一轮修订仍无诊断并保留错误 caller。
2. authored 仅有 path/start_line/source_expression（没有 bytes），source_expression=h(2)：第一轮修订仍无诊断并覆盖成 g(1)。
3. 已有 source_text 的 aux evidence 只给 line:1：第一轮修订仍抛 KeyError，因为 line 归一化仍仅在库文件分支。

原因不是另一类 C++ 语法：一致性检查放在“存在 authored bytes”的内部，line 归一化放在“path 属于库 scope”的内部。建议把元数据一致性验证移出 bytes 条件，把证据行号归一化放在路径分类之前。结果修订后另行追加，旧反例不删除。

## 3. 必须保留的验证边界

位置匹配器能验证“指定表达式原文位于这些 bytes”，但不是完整 C++ 前端或控制流分析器。例如 `decltype(g(1))` 中的 g(1) 也能正确定位；仅凭该 PASS 不能证明它是一次运行求值。cute_elementwise 草稿把这类关系明确写成不求值 type_uses，并由独立语法义务检查约束，不能由通用位置匹配器的 PASS 代替。

同理，ownership 完整只是文件责任清单完整，不等于 824 文件的 API、宏展开、模板决议、资源协议、运行验证已完成。当前 registry 正确保留这种区分。

## 4. 完整可复跑的 callsite 反例

从 atlas 根目录，以 `.venv/bin/python` 执行以下代码。所有 fixture 留在内存；输出每个 case 的结构化 issue 或异常。旧问题修复后应从异常/静默接受变成规范化成功或对应 authored 一致性诊断。

```python
from pathlib import Path
from unittest.mock import patch
import copy, json, sys
sys.path.insert(0, 'scripts')
import build_atlas as b

V = Path('/virtual/atlas')
original_bytes = Path.read_bytes
original_text = Path.read_text
original_is_file = Path.is_file

def case(name, raw=b'void f() { g(1); g(1); }\n', path='include/test.hpp',
         evidence=None, authored=None, auxiliary=False):
    actual = (V / path).resolve() if auxiliary else V / 'snapshot' / path
    fixtures = {str(actual): raw}
    if evidence is None:
        evidence = [{'path':path,'start_line':1,'end_line':len(raw.splitlines())}]
    edge={'id':name,'source':'caller','target':'target','relation':'calls',
          'source_expression':'g(1)','evidence':copy.deepcopy(evidence)}
    if authored is not None: edge['callsite']=copy.deepcopy(authored)
    scope={'commit':'fixed-test-commit','files':[] if auxiliary else [{'path':path}]}
    def rb(p):return fixtures[str(p)] if str(p) in fixtures else original_bytes(p)
    def rt(p,*args,**kwargs):return fixtures[str(p)].decode() if str(p) in fixtures else original_text(p,*args,**kwargs)
    def fi(p):return True if str(p) in fixtures else original_is_file(p)
    issues=[]
    with patch.object(b,'ROOT',V),patch.object(Path,'read_bytes',rb),patch.object(Path,'read_text',rt),patch.object(Path,'is_file',fi):
        try:
            b.validate_evidence({name:edge},scope,issues)
            return {'name':name,'issues':[i['kind'] for i in issues],
                    'callsite':edge.get('callsite')}
        except Exception as exc:
            return {'name':name,'exception':type(exc).__name__,'message':str(exc)}
rows=[]
one=b'void f() { g(1); }\n'
one_site={'path':'include/test.hpp','start_byte':11,'end_byte':15,'start_line':1,'end_line':1,'context':'kept','caller':'caller'}
rows.append(case('duplicate_evidence',one,evidence=[{'path':'include/test.hpp','start_line':1,'end_line':1}]*2,authored=one_site))
over=b'void f() {\n  g(1);\n}\n'
rows.append(case('overlapping_evidence',over,evidence=[{'path':'include/test.hpp','start_line':1,'end_line':3},{'path':'include/test.hpp','start_line':2,'end_line':2}]))
rows.append(case('same_line_without_author'))
two_site={'path':'include/test.hpp','start_byte':17,'end_byte':21,'start_line':1,'end_line':1,'context':'second','caller':'caller'}
rows.append(case('same_line_second_author',authored=two_site))
rows.append(case('bad_bytes',authored=two_site|{'start_byte':16}))
rows.append(case('out_of_bounds',authored=two_site|{'start_byte':999,'end_byte':1003}))
rows.append(case('half_range',authored={'path':'include/test.hpp','start_byte':17}))
rows.append(case('wrong_line',authored=two_site|{'start_line':2}))
rows.append(case('wrong_raw',authored=two_site|{'raw_source_expression':'h(1)'}))
utf='// 中文\r\nvoid f() { g(1); }\r\n'.encode()
pos=utf.index(b'g(1)')
rows.append(case('utf8_crlf',utf,authored={'path':'include/test.hpp','start_byte':pos,'end_byte':pos+4,'start_line':2,'end_line':2}))
aux='data/probe file.cpp'
rows.append(case('aux_packaged',one,path=aux,evidence=[{'path':aux,'start_line':1,'end_line':1,'source_text':one.decode().strip(),'source_url':'evidence/probe.html#L1'}],authored=one_site|{'path':aux},auxiliary=True))
rows.append(case('aux_without_package_stage',one,path=aux,authored=one_site|{'path':aux},auxiliary=True))
rows.append(case('aux_wrong_authored_path',one,path=aux,evidence=[{'path':aux,'start_line':1,'end_line':1,'source_text':one.decode().strip()}],authored=one_site|{'path':'data/not-the-probe.cpp'},auxiliary=True))
rows.append(case('line_alias_evidence',one,evidence=[{'path':'include/test.hpp','line':1}]))
rows.append(case('authored_expression_disagrees',one,authored=one_site|{'source_expression':'h(2)'}))
rows.append(case('unevaluated_boundary',b'void f() { decltype(g(1)) x; }\n'))
rows.append(case('wrong_caller',one,authored=one_site|{'caller':'not_the_source'}))
rows.append(case('wrong_caller_without_bytes',one,authored={'path':'include/test.hpp','start_line':1,'caller':'not_the_source'}))
rows.append(case('wrong_expression_without_bytes',one,authored={'path':'include/test.hpp','start_line':1,'source_expression':'h(2)'}))
rows.append(case('aux_line_alias_packaged',one,path=aux,evidence=[{'path':aux,'line':1,'source_text':one.decode().strip(),'source_url':'evidence/probe.html#L1'}],auxiliary=True))
print(json.dumps(rows,ensure_ascii=False,indent=2))
for path in ['/etc/passwd','data/../../../../etc/passwd','../02_cutlass_and_gemm/exemples/../../../../etc/passwd']:
    assert b.auxiliary_path(path) is None,path
print('PASS disallowed auxiliary escapes rejected; fixture data stayed in memory')
```

## 修后独立复核记录

第一轮修订：现有 atlas 单元测试从 8 增至 10，独立运行 10 tests PASS。原 library line alias 变体通过；带 bytes 的错误 source_expression 和 caller 被拒绝。其余重复 evidence、字节选择、UTF-8/CRLF、正常辅助材料及越界拒绝反例未出现回归。

第二轮及最终冻结状态将在 root 完成对应修订后，只追加审查结果，不修改被审脚本或数据。

### 第二轮修订后的最终复核（本轮冻结）

上述句子是修订前的审查计划；现在 root 已落盘第二轮修订，审查者重新执行本文同一个 20-case 矩阵，没有继续扩展输入 API。最终结果：

- library line alias：PASS，无异常。
- aux line alias（含预先提供 source_text）：PASS，无异常。
- aux 未提供 source_text：现在也能读取允许路径的原文并定位成功。它覆盖了前述“必须先 package 才能定位”的旧阶段限制；真实离线辅助页面的复制/URL仍属于 package 阶段。
- 带 bytes 或不带 bytes 的错误 authored source_expression/caller：均返回 authored_callsite_fields_disagree，不再静默替换错误表达式或保留错误 caller。
- 错误 raw_source_expression/行号、错误/越界/半截 byte range、aux 路径不符：保留对应错误诊断。
- 重复/重叠 evidence、同一行二次调用的精确选择、UTF-8 与 CRLF、正常已 package 的 aux URL：均无回归。
- 无 authored selector 的同一行重复仍保留歧义；没有为了通过检查而合并两次调用。
- decltype 内表达式仍可被位置匹配器定位，这是第 3 节明确保留的工具职责边界，不是一次运行求值的证明。

独立复跑现有测试：`test_atlas_build.py` **12 tests PASS**；`test_module_registry.py` **9 tests PASS**。registry 数据/生成器散列未变。作者没有修改这些测试或实现。

最终读取散列：

| 对象 | SHA256 |
| --- | --- |
| scripts/build_atlas.py | 589098e178670bc4c0e0b3bdcb50d99fcd46518a7e08265ffcb5ff4e4f84aee3 |
| tests/test_atlas_build.py | 8e7a0ac56b43ea367e25b809e425e40c6d9a6989503ec92b5d8c1c3e56ebd0bb |
| data/module-registry.json | 31adb2398feb5481fd423ba0743e07156815d75bcf4fb056fea91e28603411a3 |
| scripts/build_module_registry.py | 1968eaa2936e51196b25d5dd03e2787f2703bce7ad3e3e253ba7c2a76e10e472 |

额外交接核对：root 接入的 `data/modules/cute_elementwise/relations.json` 与冻结草稿逐字节一致，两者 SHA256 均为 `58f7606e673902285fc52e54ea84b433e96866157d07dbfe2618af711883611c`。这只证明复制一致，不将本审查变成 UI、运行正确性或全库语义完成认证。

本轮审查到此冻结；保留历史反例、原因、root 修复及独立复核的完整路径。未关闭边界仍是模板语义/运行验证及更广覆盖，不把它们混入已完成的文件 ownership 或字节位置核对结论。
