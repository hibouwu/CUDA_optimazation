# 阶段1：friend类型名查找独立审查

审查者只写独立测试和本记录，没有修改生成器。原SI05、SI06的6项测试已在当前版本全部通过；本轮进一步检查“何时确实引入namespace类型、何时引用已有类型、何时必须保留查找缺口”。

## 版本和独立依据

- 核心提取器SHA-256：`8d5eccceaf43d53b0ebdfa4f79870bdb05d25356f18922c0486876f1fba037f7`
- 投影映射SHA-256：`8458a5efe4adb286ff3e36d73a53a1d5b11fca40fca2bb068ec393de75fd45ae`
- 命令：`.venv/bin/python -m unittest discover -s tests -p test_friend_lookup_review.py -v`

不是只验证fixture能够编译。每个缩减用例还用Clang 21.1.8的 `-std=c++17 -pedantic-errors -Xclang -ast-dump=json`读取实际`FriendDecl.type.desugaredQualType`，与提取器owner对照。条件例分别编译A=0、A=1。

当前9项独立测试 **4通过、5失败**。下述四类开放问题对应五个失败测试；均不是解析器语法错误，当前提取器在这些失败例上诊断为空。

## FL01：using声明与基类中的既有类型没有进入名查找

```cpp
namespace a { struct F {}; }
namespace b { using a::F; struct S { friend struct F; }; }
```

Clang的friend目标为`a::F`。当前输出却引入一个新的`b::F`，并标为`innermost_namespace_elaborated_friend_declaration`。`lookup_contract`里的一句“imports require relation reconciliation”不能替代具体pending，也不能为已经选定的错误owner提供依据。

同样的遗漏出现在已知非依赖基类：

```cpp
struct Base { struct F {}; };
struct S : Base { friend struct F; };
```

Clang目标为`Base::F`，当前输出新的全局`F`。如果当前阶段还没有足够的using/继承名查找实现，应留下具体的既有候选、查找路径和blocking pending，不能无依据宣布namespace introduction。

必须同时保留两个反向边界，防止过度修复：

- 将第一例的`using a::F;`改成`using namespace a;`后，Clang目标确实是新引入的`b::F`。using-declaration与using-directive在这里不能混为一谈；当前实现这项结果正确。
- `template<class T> struct S:T { friend struct F; };`中的依赖基类T，不按已经确定的基类成员作用域搜索F。Clang目标为全局`F`，当前实现这项结果也正确。

## FL02：先前声明是否可见没有与条件分支绑定

```cpp
struct S {
#if 0
  struct F {};
#endif
  friend struct F;
};
struct F {};
```

Clang的活动friend目标是全局`F`。当前提取器把恒假区域中的`S::F`当成先前可见类型，错误绑定活动friend，诊断为空。

恒假分支的原始`S::F`声明仍应进入源码清单；问题不在于登记了这段源，而在于随后把它当作活动查找环境。

把`#if 0`换成未知的`#if A`，Clang结果分别是：

- A=1：`S::F`；
- A=0：全局`F`。

当前输出无条件选定`S::F`。必须形成带条件的两个owner变体，或者明确保留条件可见性查找pending；不能用textual-prior替代visible-prior。

## FL03：绑定已有nested类型时丢失外围模板身份

```cpp
template<class T> struct S {
  struct F {};
  friend struct F;
};
```

Clang的friend目标是已有`S::F`。提取器确实给出两个同名`S::F`出现位置，但前者保留外围模板T，friend路径把模板环境清空，最终生成两个entity ID。

“friend自身的模板参数”和“所指已有类型继承的外围模板环境”必须区分：新引入namespace friend模板不应偷带所属class的模板；绑定已有nested类型时也不能反过来丢掉该类型原有身份。

## 已通过且不应被误推广的reference边界

- `using Alias=Real; struct S {friend Alias;};`保留为`friend_type_reference`，没有制造新的class Alias。Clang实际目标Real用于证明reference与新声明的区别；本轮没有把字符串`target_type=Alias`当成已完成别名目标关系解析。
- `friend class n::F;`保留为qualified类型reference，没有制造另一个新class声明。其目标拼写和范围保留；具体关系可以后续独立对账。

上面两个reference形态不说明一般名查找已完成。尤其不能把“已有raw target expression”推广成“当前新类型owner已经确定”。

## 结论

原SI05“完全漏掉简单friend类型出现位置”已被修复，但friend类型的完整身份和名查找尚未通过阶段1。FL01、FL02要求先消除零诊断的错误owner判定；FL03要求保留已知目标的完整模板身份。四个正向/反向测试结果应与五个失败例一起保留。

## 追加检查点复审：错误绑定消除，完整lookup仍有明确缺口

根代理修复后，复审版本为：

- 核心：`1a68e6b34be5098ecde0f28f041d71deb9fdfaa540e33c7ea18fb9f9203ba0b9`
- 投影：`8458a5efe4adb286ff3e36d73a53a1d5b11fca40fca2bb068ec393de75fd45ae`

原8个lookup反例全部通过。附加的依赖基类边界按本检查点的要求接受“实际正确解析或明确pending reference”，不强迫本轮扩展lookup实现；当前9项独立测试全部通过。生成器审查前后哈希一致，本次没有修改生成器。

必须区分以下两类结果。

### 已有证据的实际修复

- 恒假`#if 0`中的nested类型仍保留为源码声明，但不再参与随后活动friend的查找；该friend正确指向全局F，没有新增pending。FL02中的这个恒假缩减缺陷关闭。
- 已存在于类模板中的nested F与friend F保留相同外围模板环境，连接到同一个entity。FL03的原缩减反例关闭。
- using-directive与using-declaration仍区别处理；原using-directive缩减结果`b::F`保持与Clang一致。

### 从假成功改为明确未完成，并未完成lookup

- `using a::F;`参与查找时，保留`friend_type_reference`及`friend_type_lookup_pending`，记录using出现位置ID和`target_expressions=['a::F']`。不再伪造新的b::F类型，但FL01的导入查找工作仍开放。
- 非依赖基类参与查找时，记录`friend_type_reference`、`lookup_status=extraction_pending`、基类声明出现位置ID和`base_expressions=[':Base']`。没有宣称已完成Base::F绑定；FL01的基类查找工作仍开放。
- 先前类型受未知`#if A`约束时，记录候选出现位置ID和`visibility_conditions`，并保留blocking lookup pending。没有无条件选定S::F，但A/!A两种实际owner尚未完整产生；FL02的条件可见性工作仍开放。
- 附加依赖基类T的例子目前也采用保守pending reference，保留`:T`及原始friend类型拼写。Clang已给出该缩减例的全局F目标，但本检查点没有要求进一步实现此规则，也没有将pending说成目标绑定完成。

这份复审只关闭已核对的错误具体绑定/身份断裂反例。测试通过包含“诚实pending”分支，不能等同于完整friend名查找通过，更不能使阶段1通过。
