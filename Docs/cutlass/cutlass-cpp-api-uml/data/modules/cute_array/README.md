# CuTe array 完整文件工作包

本工作包覆盖固定提交中的 `include/cute/container/array.hpp` 全476行。文件的87个物理声明、92条条件绑定已逐项关联正式源码账本，不使用人工身份覆盖；依赖文件的完整API不被算作本文件已完成的范围。

从离线图集的“CuTe array：完整成员、重载与标准库适配”进入五组视图：主模板成员、零长度特化、自由函数、tuple接口、依赖边界。每条关系另有独立详细图，完整签名与仓库相对路径保留。源码义务页另外列出物理声明、条件绑定与54个实际写出的形参位置。

## 重点核对

- 主模板和零长度特化分别保留22个成员函数／重载。零长度的data/begin/end等返回nullptr，但front/back/下标访问仍不能安全执行。
- 非const的cbegin/cend调用非const成员后转换返回指针；不是调用const重载。
- 零长度成员clear为空，自由函数clear仍须构造T(0)后调用fill。两个接口的类型要求不同。
- get仅有三个重载。const容器右值绑定const&版本；具名右值引用参数仍是左值，get&&先下标访问再用cute::move转换值类别。
- get的越界检查是函数体static_assert，不是签名SFINAE。tuple_element本身没有相同范围检查。
- operator==使用元素!=及普通for；N=0并不使其函数体免于语义实例化。reverse则有if constexpr分支，不能与普通for混用同一种条件解释。
- range-for隐式生成的begin/end有独立调用身份。图中写明源码锚点和语言展开示意，不伪造源文件中不存在的字面调用。
- 同一std实体的两处声明分别保留，整体预处理存在条件是各位置条件组的析取。详细关系图按本条关系的源码位置选择相应声明。

类中没有显式写出的构造、析构或复制／移动特殊成员，不为它们捏造声明行。实际隐式生成、删除、平凡性与聚合初始化仍取决于T、N和语言模式。对象的存储位置也不能仅由array这个类型名确定。

## 重生成

在图集根目录：

```bash
.venv/bin/python data/module-drafts/cute_array/check_source.py
.venv/bin/python -m unittest discover -s data/module-drafts/cute_array -p test_source_check.py
.venv/bin/python -m unittest discover -s tests -p test_array_stage_review.py
.venv/bin/python scripts/stage_cute_array.py
.venv/bin/python scripts/build_atlas.py
.venv/bin/python scripts/audit_atlas.py
node tests/test_atlas_runtime.js
```

制作稿、独立检查器和Host Clang正反样本保存在`data/module-drafts/cute_array/`；正式接入记录在`audits/cute-array-canonical-integration.json`。形参序号、返回cv与成员cv、基础类型与数组声明器等字段的区别见[data/declaration-field-semantics.md](../../declaration-field-semantics.md)。

当前仍明确保留`type_traits.hpp:92`的using身份提取缺口。导入语句及其查找名称分开显示，不将旧账本的伪QName当作已经解析的依赖API。Host编译样本、静态SVG、机械链接检查及DOM代码测试也不替代CUDA/RTC配置验证、任意模板实例验证或真实浏览器会议使用验收。
