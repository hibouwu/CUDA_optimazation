# cute::array：完整文件独立源码契约审查

本稿完整读取固定snapshot的 `include/cute/container/array.hpp` 全476行，并仅回看必要依赖config.hpp、type_traits.hpp、integral_constant.hpp及cutlass.h。没有把另一制作者的cute_array总结、draft或站点作为语义证据。固定提交为`8f50b052e1099fb982392a622caab69b97b63128`；array.hpp SHA256为 **`a5a6c8357cf4311bed395db7dfa8fcc5c28b3d295fdfec9ad2b00b75253ad8ee`**。

**独立原文清单与13项Host Clang/source测试完成。** 本稿是后续模块数据交叉审查的标准，不提前宣称尚未逐条读取的制作者数据或完整站点通过。没有改core、draft、站点，也没有GPU运行或824文件重建。

## 1. 文件声明结构

文件实际包含两个array结构模板：41–195的`template<class T,size_t N> struct array`，198–337的`template<class T> struct array<T,0>`偏特化。不是用同一个零长度C数组完成N=0，也不是继承或别名到std::array。

两结构各有10个type alias、22个显式写出的成员函数/重载；此外有8个namespace级函数定义（5个普通辅助/比较模板，3个get重载）。因此本文件写出的callable定义共52个。主模板有public数据成员`element_type __elems_[N]`（194），零特化没有对应数据成员。没有显式编写array构造、析构或赋值运算符；需要讨论隐式特殊成员时应单用编译器证据，不给它们捏造源码声明行。

两结构的alias分别位于44–53和201–210：

- element_type=T；value_type=remove_cv_t<T>。**value_type去掉元素cv，不代表reference/pointer可写。**
- size_type=size_t；difference_type=ptrdiff_t。
- reference=element_type&；const_reference=const element_type&。
- pointer=element_type*；const_pointer=const element_type*。
- iterator=pointer；const_iterator=const_pointer。N=0只交换了两条alias的源排列顺序，不改变名称含义。

两类的成员清单如下。标“两重载”的行分别代表非const和const两个独立API；所有成员显式使用CUTE_HOST_DEVICE constexpr，没有写noexcept或ref-qualifier。

| API及返回类型 | 主模板声明行 | N=0声明行 | 关键区别 |
| --- | --- | --- | --- |
| operator[](size_type pos)，reference / const_reference，两重载 | 56/62 | 213/219 | 都无边界检查；零特化通过null begin访问，不是安全空值 |
| front()，reference / const_reference，两重载 | 68/74 | 225/231 | `*begin()` |
| back()，reference / const_reference，两重载 | 80/87 | 237/243 | 主模板调用operator[](N-1)，零特化为`*begin()` |
| data()，T* / T const*，两重载 | 94/100 | 249/255 | 主模板返回__elems_；零特化返回nullptr |
| begin()，iterator / const_iterator，两重载 | 106/112 | 261/267 | 主模板调用相应data；零特化直接nullptr |
| cbegin()，const_iterator，两重载 | 118/124 | 273/279 | 非const版本也返回const pointer，不是iterator |
| end()，iterator / const_iterator，两重载 | 130/136 | 285/291 | 主模板data()+size()；零特化直接nullptr |
| cend()，const_iterator，两重载 | 142/148 | 297/303 | 主模板调相应end；零特化直接nullptr |
| empty() const，bool | 154 | 309 | size()==0 / true |
| size() const，size_type | 160 | 315 | N / 0 |
| max_size() const，size_type | 166 | 321 | size() / 0 |
| fill(const T& value)，void | 172 | 327 | 逐元素赋值 / 空函数体 |
| clear()，void | 180 | 331 | fill(T(0)) / 空函数体 |
| swap(array& other)，void | 186 | 335 | 元素级unqualified swap / 空函数体 |

主模板cbegin非const→begin非const→data非const后发生指针const转换；const重载则走const成员路径。cend同理。图中不能只因最终返回类型相同而把重载/调用点合并。back里的`rbegin()`只有82/89行注释，文件没有rbegin/rend API；也没有at、resize、push_back或成员reverse。

## 2. Namespace级函数、模板参数与真实调用

| 声明行 | 模板/完整形态摘要 | 真实行为 |
| --- | --- | --- |
| 341 | `<class T,size_t N> bool operator==(array<T,N> const& lhs,array<T,N> const& rhs)` | 343–348逐元素测试`lhs[i] != rhs[i]`；不是调用元素operator== |
| 353 | `<class T,size_t N> void clear(array<T,N>& a)` | 355是`a.fill(T(0))`，不是a.clear() |
| 360 | `<class T,size_t N> void fill(array<T,N>& a,T const& value)` | 362转发a.fill(value) |
| 367 | `<class T,size_t N> void swap(array<T,N>& a,array<T,N>& b)` | 369转发a.swap(b) |
| 375 | `<class T,size_t N> array<T,N> reverse(array<T,N> const& t)` | N=0在if constexpr分支返回t；其它N构造t_r{}、逆序复制赋值、返回新array |
| 406 | `<size_t I,class T,size_t N> T& get(array<T,N>& a)` | static_assert(I<N)，返回a[I] |
| 414 | 同模板，`T const& get(array<T,N> const& a)` | 同一边界断言，const访问 |
| 422 | 同模板，`T&& get(array<T,N>&& a)` | 425调用cute::move(a[I]) |

free swap与primary member swap不是std::swap(array)的别名：成员188先`using CUTE_STL_NAMESPACE::swap`，190再对元素做**未限定调用**，允许ADL选择元素所在namespace的swap。目标依赖T，不能画成所有配置都唯一直接调用std::swap。`cute::move`则是type_traits.hpp:196–199自己定义的函数模板，返回static_cast<remove_reference_t<T>&&>，不是本文件直接调用std::move。

这些函数也都显式写CUTE_HOST_DEVICE constexpr，没有noexcept。constexpr声明不保证任意T及任意标准库版本下的整个调用都可常量求值；元素构造、赋值、比较、swap仍有各自要求，可能失败或抛异常。函数体没有直接动态分配，不代表所有T操作都不分配。

## 3. 对自研接口对齐最重要的反例

### A01：N=0的member clear和free clear要求不同

零特化的member clear331为空；free clear353仍构造T(0)，然后调用零特化fill。`NoZero(int)=delete`时，`array<NoZero,0>::clear()`可用，`cute::clear(a0)`却实例化失败。不能把这两个API合并成同一个“空array清理”动作，也不能因N=0把free调用的T(0)求值删除。

独立Clang负例实际报在array.hpp:355：functional-style cast使用deleted函数/显式删除的constructor。最初测试匹配“deleted constructor”短语过窄；已改为同时验证deleted诊断与准确355行，语义负例不变。

### A02：没有const&&专用get，且引用寿命属于调用者

对`A=array<int,2>`，实际返回矩阵为A&→int&，const A&→const int&，A&&→int&&，**const A&&→const int&**。最后一种绑定到const&重载，不是const int&&。但元素本身为const的非const容器`array<const int,2>&&`可命中第三重载，返回const int&&；元素cv与容器cv不能混为一谈。

get返回引用，不延长临时array的寿命，也不返回一个独立元素值。rvalue重载中的a本身是命名左值，先通过operator[]得到T&，再用cute::move转换值类别。

### A03：get的边界检查不是SFINAE，tuple_element更没有相同检查

I<N只在三个get函数体的static_assert中。Clang确认：`decltype(cute::get<99>(declval<array<int,1>&>()))`仍可得到int&；真正调用get<1>作用于array<int,1>则因“Index out of range”失败。不能用这种未求值签名探测代替可调用实例的有效性判断。

本文件两份tuple_element特化都仅`using type=T`，没有I<N约束。Host Clang确认`std::tuple_element_t<99,cute::array<int,0>>`仍是int。它不表示越界元素存在或可以读取。

### A04：空对象没有元素，但仍有类型形成与实例化边界

零特化不含T存储，因此即使T的默认/复制构造、赋值被删除，array<T,0>自身仍可为空对象并复制；member fill、clear、swap及free fill/swap都可以不执行元素操作。独立测试验证null data/begin/end/cbegin/cend、size/max_size=0、empty=true，以及无元素赋值要求。

但front/back/[]仍存在并返回引用，**不是安全空访问**。constexpr读取zero.front()被Clang拒绝，测试没有在运行时执行非法读取。T也必须能使所声明的引用/指针alias等类型形成合法，不能宣称任何T都可实例化零特化。

### A05：N=0的operator==仍要求比较表达式可形成

operator==用普通for，不是if constexpr。即使N=0，实例化函数体时`T const& != T const&`仍须可形成；Clang用NoCompare类型确认空array比较也编译失败。另一方面，只提供元素!=、删除元素==的类型却可以进行array==，独立正例编译通过。这两点都会被“照std直觉重写相等比较”遗漏。

### A06：reverse是新值，不是视图、不是原位操作

N>0会构造t_r{}并从const t逐元素赋值，因此需要相应初始化和从const元素赋值的能力，不因有rvalue get就自动支持move-only元素。Clang确认普通int array反转后输入不变；move-only且删除copy-assignment的正N版本在源382行失败。N=0的if constexpr分支不实例化该逐元素路径。

### A07：clear不改变size；swap允许ADL且wrapper不宣称noexcept

对N>0，clear把元素赋成T(0)，不会把N变0、销毁/移除元素或把empty改为true；T(0)也不等同T{}。Clang constexpr正例验证size仍为3，元素归零。

自定义元素ADL swap把结果写成特别的40/90，array成员swap产生该结果，证明不是硬连std::swap。即使元素swap声明noexcept，外层array::swap未写noexcept，`noexcept(a.swap(b))`仍为false；不能从实现看似不抛异常就修改API限定符。

## 4. 标准库适配与所有宏条件

array.hpp有四个物理namespace块：cute38、cute401、CUTE_STL_NAMESPACE430、条件std447。前两个是同一cute namespace的不同出现位置；后两个要按配置和精确源码位置处理，不能以短名合并。

config.hpp:33–41将CUTE_HOST_DEVICE在`__CUDACC__ || _NVHPC_CUDA`下定义为forceinline/host/device，在另一分支定义为inline。array.hpp自身另有以下边界：

- 395–399：__CUDACC_RTC__时include CUDA_STD_HEADER(tuple)，否则include<tuple>。cutlass.h:40将CUDA_STD_HEADER(header)定义为`<cuda/std/header>`。
- config.hpp:106–111：默认定义关系中，RTC选择CUTE_STL_NAMESPACE=cuda::std并定义CUTE_STL_NAMESPACE_IS_CUDA_STD；非RTC选择std。原始guard仍须保留，不擅自抹掉可能的外部宏环境。
- 433–442：在选定CUTE_STL_NAMESPACE中定义tuple_size/tuple_element特化；tuple_size继承该STL namespace的integral_constant<size_t,N>，不是cute::C或cute::integral_constant。
- 446–476：CUTE_STL_NAMESPACE_IS_CUDA_STD时另开std桥接namespace，再定义464–473的两个特化。
- 450–462：compiler major>=13时在该std块include外部`cuda/std/__tuple_dir/structured_bindings.h`；否则若RTC，保留456–460的两个std主模板前置声明，原文分别为`template<class... _Tp>`与`template<size_t _Ip,class... _Tp>`，不能依照常见std定义凭空改写参数形状。

因此本文件写了4份tuple相关特化定义和2份条件前置声明，其中tuple_element还各有一个type alias。std/cuda::std的主模板、integral_constant及included compatibility header是外部依赖，不应记为824库内missing_definition。

Host structured binding可通过tuple traits与ADL cute::get工作；实际Clang正例`auto& [x,y]=a`能修改原元素。但本文件没有std::get重载，独立Host Clang的`std::get<0>(cute_array)`被拒绝；不能由tuple_size/element存在推导所有std tuple算法都已适配。

array的value_type依赖type_traits.hpp:92的using导入remove_cv_t，正确lookup名称是cute::remove_cv_t，目标由CUTE_STL_NAMESPACE条件决定。Root另发现当前提取器把该using的target拼成伪QName；本稿依原文解释依赖，不把伪`cute::CUTE_STL_NAMESPACE::remove_cv_t`当真实接口，也不将已知提取身份错误伪装成C++类型不确定性。

## 5. 实际验证与证据边界

新增`tests/test_cute_array_independent_review.py`，13项实际通过：3项完整文件声明/条件检查，10项包含正反例的Host Clang契约检查。Clang使用21.1.8、C++17、固定snapshot include及本机CUDA13.0/CCCL include，只执行-fsyntax-only，不运行GPU或非法零元素读取。

测试文件SHA256为`6c7c91bc4aa76bf67f978b908dbe5e2dfe4e6851c4ab600833f7a8776b654485`。所有源码计数仅为本文件实际声明出现位置，不是全库API覆盖分母。RTC/不同CUDA版本分支做了原文条件审查，没有借Host编译冒称对应设备/RTC分支已实际编译通过。

## 6. array source selector定点交叉审查

本节只读取selector实现及重新从array原文提取得到的候选，不读取制作者cute_array总结作证。正常结果已核对：同一433行的std::tuple_size和cuda::std::tuple_size必须有不同entity；433与464的两个std::tuple_size为同entity但不同declaration occurrence、不同条件来源。完整QName tokens、signature物理范围及hash能区分这些位置与namespace，不能仅靠短名或抹掉模板实参。

### S01：显式空selector被当作缺失——发现并关闭

发现检查点build_atlas SHA为`417df0a22a915221e44520b3b5c5c1cc3d097a8434612a0d1f933f347c009709`。当节点显式带`source_selector={}`时，旧truthiness判断将其当成没有selector；array:433的std::tuple_size不论有无manual，都静默linked且issues为空，有manual还会retire它。这违反坏显式selector不能fallback的合同，已向Root报告，审查者没有修改builder。

Root修复后，独立重新构造`{}`、None、[]、False、空字符串五种显式坏值，分别测试有/无manual，共10例：全部explicit_source_selector_unresolved、有issues、不产生entity_id、不retire manual。正常std433、cuda::std433、std464三个精确selector仍命中原始occurrence；std两个位置不被合成同一occurrence。复验结束builder SHA为 **`149c6cf2aa979f37d78bb08bd0f4e5ad12fda9bff3383aaf1c27cd8d55912383`**，S01在这一检查点关闭。

此次只完成这个selector反例与三条正常选择的定点复审；没有将Root新增的所有多位置聚合规则一概标为本审查已穷尽验证，也没有据此宣告模块图完整或浏览器验收通过。
