#!/usr/bin/env python3
"""Generate reviewed explanatory notes for the three published CuTe modules only.

The semantic descriptions below were checked against the pinned source. This is
not an extractor and must fail if the published module inventory changes.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATLAS = json.loads((ROOT / 'data/atlas.json').read_text())
assert ATLAS['commit'] == '8f50b052e1099fb982392a622caab69b97b63128'
SELECTED = [n for n in ATLAS['nodes'] if n['id'].startswith(('array.', 'alignment.', 'elementwise.')) and n['kind'] in ('api', 'type', 'external')]

ELEMENTWISE = {
    'api.clear_rvalue': '接收临时或右值 Tensor，函数体将有名字的 tensor 当作左值转交 clear(Tensor&)；清零的是该 Tensor 所表示的数据，不因 && 自动复制或接管底层存储。',
    'api.clear_lvalue': '取 Tensor::value_type 为 T，以 T{} 值初始化得到填充值，再调用 fill(tensor, T{})。这是逐元素赋值入口；T{} 的含义取决于元素类型，不能一概解释成逐字节 memset(0)。',
    'api.fill_rvalue': '接收临时或右值 Tensor，将有名字的 tensor 当作左值转交 fill(Tensor&, value)，使临时视图也能修改其指向的数据；本重载不另建缓冲区。',
    'api.fill_prefer1': '优先尝试 fill(tensor.data(), value)，让底层迭代器类型通过重载解析及 ADL 提供专用填充。尾随 decltype 检查该表达式是否有效；函数体只有调用而没有 return，不能据此保证任意非 void 自定义 fill 都具有有效的返回行为。',
    'api.fill_prefer0': '当更高优先级的数据入口填充不可用时，按 i = 0 到 size(tensor)-1 逐一执行 tensor(i) = value。访问仍经过 Tensor 的布局映射，不假定逻辑元素在物理地址上连续。',
    'api.fill_lvalue': '向 detail::fill 传入 prefer<1>{}，先参与专用数据入口填充的重载选择；若该候选替换失败，利用 prefer 的继承关系回退到 prefer<0> 的逐元素实现。选择发生在编译期，实际赋值发生在调用时。',
    'api.axpby_rvalue': '接收作为右值的输出 Tensor y，再将有名字的 y 当作左值转交 axpby 的主要实现。它不搬迁 y 的数据，也不创建独立输出；alpha、x、beta 和谓词原样传递。',
    'api.axpby_lvalue': '遍历 size(x) 个逻辑位置；仅当 p(i) 为真时写入 y(i) = alpha*x(i) + beta*y(i)。beta 为零时走不读取旧 y(i) 的表达式，谓词为假时不访问该位置的 x/y。调用方须保证 x、y 和谓词的索引范围及别名关系符合逐元素执行的要求。',
    'type.Tensor': 'CuTe 的张量表示由 Engine 提供数据入口，由 Layout 把逻辑坐标映射为偏移；上层逐元素算法通过它访问数据。Engine 可以拥有存储，也可以只是视图，因此 Tensor 类型本身不指定寄存器、共享内存或全局内存。',
    'type.value_type': '将 Engine::value_type 暴露为 Tensor 的数值类型，供 clear 构造填充值等算法使用。它是类型别名，不是数据指针、存储空间或实际元素对象。',
    'type.prefer': '通过继承 prefer<N-1> 建立重载优先级链。传入 prefer<1>{} 时，精确匹配的高优先级候选优先；不可用时允许转换到较低优先级标签。它不是运行时优先队列。',
    'type.prefer0': 'prefer 优先级链的零级终点；在 fill 中为通用逐元素实现提供最后一个可匹配的标签类型，不再继承更低一级。',
    'type.constant_fn': '保存一个 R 类型的成员 r_，调用运算符忽略输入并返回该成员的值。axpby 用 constant_fn<true_type> 作为默认谓词；一般 R 的存储值并不由类型名自动固定。',
    'type.true_type': 'bool_constant<true> 的类型别名，用类型携带真值；作为 axpby 默认谓词的返回类型时，每个位置都被选中。',
    'type.bool_constant': '将 bool 模板值包装为 C<b>，使真假值可以参与类型推导和编译期选择；没有独立的调用时布尔参数。',
    'type.C': '把模板常量 v 放入类型身份，提供对应的静态 value 和值转换；用于静态形状、常量比较等。不同 v 是不同类型，不等同于可随每次调用改变的普通变量。',
    'type.Int': '为 int 模板常量 v 提供 C<v> 的简写；axpby 中的 Int<0>{} 是携带整数零的类型对象，用于 beta 比较。',
    'type.is_complex': '复数类型判定的主模板，value 为 false；axpby 用它选择普通数值的 beta == 0 比较。此主模板不是所有用户自定义复数类型的自动识别器。',
    'type.is_complex_complex': '针对 cutlass::complex<T> 的判定特化，value 为 true；使 axpby 改为同时检查 beta 的实部和虚部是否为零。',
    'api.tensor_data': '非 const Tensor 的数据入口访问器，返回 engine().begin()，保留其实际返回类型。对拥有存储的 Engine 可能得到指针，对视图可能得到迭代器引用；本函数不读取所有张量元素。',
    'api.tensor_at_mutable': '非 const Tensor 的坐标访问入口。普通坐标经 layout()(coord) 转为偏移后访问 data()[offset]；含下划线的坐标则生成切片布局与偏移并返回子 Tensor 视图。是否可写仍由 Engine 的引用或迭代器类型决定。',
    'api.tensor_at_const': 'const Tensor 的坐标访问入口，同样在编译期按 Coord 是否含下划线选择元素访问或切片。const 限定的是 Tensor 对象；非拥有视图的底层元素是否只读，应继续检查 Engine 返回类型，不能仅据 const Tensor 推断。',
    'api.tensor_size': '把 size<Is...> 查询转交给 tensor.layout()，返回整个逻辑域或所选嵌套模式的元素数量。它不读取元素数据，返回类型可以是静态整数类型，也可以携带运行时大小。',
    'api.constant_fn_call': '忽略传入的整个实参包，返回保存的 r_。源码使用未加括号的成员名配合 decltype(auto)，因此返回的是 R 值而不是 R const&；默认谓词中的 R 为 true_type。',
    'api.complex_real': '返回 complex<T> 内实部成员的 const 引用，axpby 的复数 beta 判零通过它读取实部。引用依赖原复数对象的生命周期，不创建独立分量存储。',
    'api.complex_imag': '返回 complex<T> 内虚部成员的 const 引用，axpby 的复数 beta 判零通过它读取虚部。引用依赖原复数对象的生命周期，不创建独立分量存储。',
    'api.beta_zero_lambda': '在 axpby 内立即执行一次，通过引用捕获 beta 得到 isBetaZero。if constexpr 按 Beta 类型选择普通比较或实部/虚部比较；beta 是否为零是该调用的数值判断，而不是仅由 Beta 类型决定。',
    'local.T': 'clear 函数体内对 Tensor<Engine,Layout>::value_type 的局部别名，用于构造 T{} 填充值；不引入新的元素类型或存储对象。',
    'external.gnu_unreachable': '编译器内建的不可达标记，由满足条件的 CUTE_GCC_UNREACHABLE 展开产生。它向编译器声明控制流不应到达此处；不是错误恢复、线程同步或运行时检查接口。',
}

TPL = {
    'Engine': '编译期类型参数：提供张量的数据入口、数值和引用类型以及拥有存储或视图的表示；实际地址由 Engine 对象携带。',
    'Layout': '编译期类型参数：选择坐标到偏移的布局表示；其 shape/stride 可以含运行时成员，类型已确定不等于所有尺寸都固定。',
    'XEngine': '编译期类型参数：输入 x 的数据入口与元素/引用类型；调用时的 x 对象提供实际存储或视图。',
    'XLayout': '编译期类型参数：输入 x 的布局表示；实际布局值随 x 对象传入，可含动态大小或步长。',
    'YEngine': '编译期类型参数：输出 y 的数据入口与可赋值引用表示；不单凭类型名认定物理存储空间。',
    'YLayout': '编译期类型参数：输出 y 的布局表示；调用时将逻辑 i 映射到对应输出元素。',
    'Alpha': '编译期类型参数：alpha 系数的数值表示及乘法重载，实际系数由函数实参 alpha 提供。',
    'Beta': '编译期类型参数：beta 系数的表示；is_complex<Beta> 决定判零分支，实际是否为零由 beta 的值决定。',
    'PrdTensor': '编译期类型参数：逐位置谓词 p 的可调用类型；默认为 constant_fn<true_type>，实际掩码由调用时的 p 对象提供。',
    'Coord': '编译期类型参数：坐标的结构与静态分量，并决定是否含下划线切片；实际动态坐标值由 coord 提供。',
    'Is': '编译期整数参数包：选择布局中要查询大小的嵌套模式路径；空包查询整个布局，单位为逻辑元素个数。',
    'R': '编译期类型参数：constant_fn 保存并返回的值类型；实际 r_ 可以是运行时数值，只有特定 R 如 true_type 自带静态值。',
    'b': '编译期 bool 模板值：决定该 bool_constant 类型所携带的真或假。',
    'v': '编译期非类型模板参数：指定该常量类型携带的值；C 接受 auto，Int 限定为 int。',
    'Child': '编译期类型参数：用于区分 aligned_struct 类型身份，默认 void；源码未把它声明为基类或数据成员。',
    'Alignment': '编译期字节数：指定所请求的类型对齐，具体支持范围及生效方式见当前类型；不是元素数量或运行时地址。',
    '_Tp': '编译期类型参数包：标准 tuple trait 前置声明保留的类型占位；这里没有对象实参或 trait 实现。',
    '_Ip': '编译期 size_t 索引：tuple_element 前置声明的元素位置占位，不是运行时下标。',
}

def template_note(n, name):
    nid = n['id']
    if name == 'T':
        if nid == 'elementwise.api.constant_fn_call':
            return '编译期类型参数包：由任意调用实参推导，使常量函数接受不同数量和类型的输入；函数体不使用这些输入。'
        if nid == 'array.dependency.move':
            return '编译期类型参数：从转发引用实参推导，可能自身为引用类型；remove_reference_t<T> 得到返回右值引用的目标类型。'
        if nid == 'alignment.dep.has_single_bit':
            return '编译期类型参数：待检查值 x 的数值类型，必须支持源码中的非零比较、减法及按位与运算。'
        if nid == 'elementwise.type.is_complex':
            return '编译期类型参数：要判定是否为 cutlass::complex 特化的完整被测类型；本主模板给出 false。'
        if nid.startswith('elementwise.api.complex') or nid == 'elementwise.type.is_complex_complex':
            return '编译期类型参数：cutlass::complex<T> 的实部和虚部所用标量类型。'
        if nid.startswith('elementwise.'):
            return '编译期类型参数：填充值 value 的类型，决定赋值/转换或底层 fill 重载是否可用；数值通过 value 实参传入。'
        return '编译期类型参数：数组元素的声明类型，决定存储、引用和元素运算；实际元素值属于数组对象。'
    if name == 'N':
        if nid == 'alignment.api.is_byte_aligned':
            return '编译期 int 模板值：要检查的对齐字节数；static_assert 要求 has_single_bit(N)，即非零的二次幂。'
        if nid == 'elementwise.type.prefer':
            return '编译期 size_t 模板值：重载优先级标签级别，通过继承 N-1 逐级回退；不是运行时调度优先级。'
        return '编译期 size_t 模板值：固定的数组逻辑元素数量，单位为元素；N=0 使用无元素存储的特化。'
    if name == 'I':
        if 'tuple_element' in nid:
            return '编译期 size_t 元素位置；本 tuple_element 特化仍直接给出 type=T，没有自行检查 I<N。'
        return '编译期 size_t 元素下标：get 的函数体用 static_assert 检查 I<N；不同于 operator[] 调用时传入的 pos。'
    assert name in TPL, (nid, name)
    return TPL[name]

ARRAY_ALIAS = {
    'element_type': '数组元素的原始声明类型 T，保留其 cv 限定。',
    'value_type': '通过 remove_cv_t<T> 得到去掉顶层 const/volatile 的数值类型，与保留 T 的 element_type 分开。',
    'size_type': '元素数量与下标使用的 size_t 类型，计数单位为元素，不是字节。',
    'difference_type': '两个位置的有符号差值类型 ptrdiff_t；别名本身不执行指针相减。',
    'reference': '指向 element_type 的左值引用类型；可否写入还取决于 T 自身是否带 const。',
    'const_reference': '指向 const element_type 的左值引用类型，用于只读元素访问。',
    'pointer': 'element_type* 指针类型，不携带地址值或存储空间标签。',
    'const_pointer': 'const element_type* 指针类型，通过该类型的指针只读访问元素。',
    'iterator': 'pointer 的别名，本容器以原始元素指针充当迭代器；不是单独的迭代器类。',
    'const_iterator': 'const_pointer 的别名，供只读遍历及 cbegin/cend 使用。',
}

def array_summary(n):
    nid, name = n['id'], n['name']
    if nid in ('array.type.decl_42', 'alignment.dep.array'):
        return 'CuTe 的固定长度聚合数组，直接包含 T 类型的 N 个元素，为局部数值和容器算法提供统一访问接口。对象声明与使用方式决定实际存储位置；该类型不会自行分配 CUDA 全局内存或保证使用线程寄存器。'
    if nid == 'array.type.decl_199':
        return 'array<T,0> 的空数组特化，不声明元素存储；数据与迭代器入口返回 nullptr，size/max_size 为零，成员 fill/clear/swap 为空操作。保留元素访问签名不代表空数组可以被解引用。'
    if nid.startswith('array.alias.decl_'):
        zero = n['line'] > 199
        return ('空数组特化中的类型别名：' if zero else '固定长度数组中的类型别名：') + ARRAY_ALIAS[name] + ('保留这些别名不表示实际存在元素。' if zero else '')
    if 'tuple_size' in nid:
        ns = 'cuda::std' if '.cuda_std.' in nid else 'std'
        return f'{ns} 的 tuple_size 对 cute::array<T,N> 的特化，通过 integral_constant<size_t,N> 提供静态元素数量，供 tuple 协议与结构化绑定使用；它不遍历数组。'
    if 'tuple_element' in nid:
        ns = 'cuda::std' if '.cuda_std.' in nid else 'std'
        return f'{ns} 的数组 tuple_element ' + ('内部 type 别名直接指向 T。' if '.alias.' in nid else '特化直接提供 type=T。') + '源码未在此检查 I<N；不能把它当成 get<I> 的越界诊断。'
    if nid == 'array.type.decl_457':
        return '特定 CUDA 标准库兼容分支中的 std::tuple_size 前置声明，为随后写入 std 命名空间的数组特化建立主模板声明；这里没有实现或运行时行为。'
    if nid == 'array.type.decl_460':
        return '特定 CUDA 标准库兼容分支中的 std::tuple_element 前置声明，为随后数组特化建立主模板声明；它保留索引与类型参数包，但本身不提供元素类型。'
    if nid == 'array.dependency.move':
        return '把实参转换为去掉引用后的类型的右值引用，供 get(array&&) 返回元素 T&&。此函数只是值类别转换，不执行复制、移动构造、显存搬运或资源所有权转移。'
    i = int(nid.rsplit('_', 1)[1])
    special = {
        341: '逐元素用 lhs[i] != rhs[i] 判断是否存在差异，发现首个差异即返回 false，否则返回 true。源码调用的是元素 != 而不是 ==；N=0 时循环不执行，但普通 for 的函数体仍必须满足实例化的表达式要求。',
        353: '自由函数 clear 先构造 T(0)，再调用 a.fill；没有转交成员 a.clear。即使 N=0、成员 fill 为空，T(0) 的构造表达式仍存在，不能套用空数组成员 clear 的无构造要求。',
        360: '自由函数 fill 把 value 转交 a.fill(value)，由普通数组或 N=0 特化的成员实现决定逐元素赋值或空操作。',
        367: '自由函数 swap 转交 a.swap(b)，普通数组按元素交换，N=0 特化不执行元素操作；它不交换两个数组对象的地址。',
        375: '返回元素次序反转的新数组，不修改输入 t。if constexpr(N==0) 时直接返回 t，并丢弃非空分支；非空时值初始化结果数组，再逐元素反向赋值，相关构造与赋值要求取决于 T。',
        406: '按编译期索引 I 返回非 const 数组元素的 T&，函数体先 static_assert(I<N) 再调用 a[I]。越界是实例化错误，不是运行时异常，也不是 SFINAE 筛选。',
        414: '按编译期索引 I 返回 const 数组元素的 T const&，保留只读引用和原数组生命周期；函数体以 static_assert(I<N) 拒绝越界。',
        422: '接收右值数组并返回第 I 个元素的 T&&。有名字的参数 a 在函数体内仍是左值，先通过 a[I] 取 T&，再用 cute::move 转为 T&&；不在此搬迁元素，不延长临时数组生命周期。',
    }
    if i in special:
        return special[i]
    zero = 213 <= i <= 335
    const = (n['signature'] or '').rstrip().endswith(' const')
    access = '只读' if const or name.startswith('c') else '非 const'
    if name in ('operator[]', 'front', 'back'):
        if zero:
            return f'空数组的 {access} {name} 接口保留返回引用的签名，但内部会经 begin() 的 nullptr 尝试访问元素。空数组不存在合法元素；本接口不做可恢复的越界检查，不能调用来取得有效引用。'
        target = {'operator[]': '调用时下标 pos，通过 begin()[pos]', 'front': '首元素，通过 *begin()', 'back': '末元素，通过 operator[](N-1)'}[name]
        return f'返回数组{target}得到的' + ('const 元素引用。' if const else '元素引用；可写性仍取决于 T。') + '不复制元素，也不进行运行时越界检查；调用方负责保证访问位置有效。'
    if name in ('data', 'begin', 'cbegin', 'end', 'cend'):
        if zero:
            return f'空数组的 {access} {name} 接口直接返回 nullptr，没有调用另一重载或计算指针偏移；它表示空区间而非一个可解引用的元素。'
        if name == 'data':
            return f'返回内部 __elems_ 数组的首地址，类型为 {"T const*" if const else "T*"}；不复制元素，不新建存储。指针有效期由当前 array 对象决定。'
        if name in ('cbegin', 'cend'):
            target = 'begin' if name == 'cbegin' else 'end'
            return f'返回只读迭代器，内部调用当前对象的 {target}()。' + (f'本重载的 this 非 const，实际先选中非 const {target}()，再转换为 const_iterator；不能画成直接调用 const 重载。' if not const else f'本重载的 this 为 const，选择 const {target}()。') + ('结果是首位置。' if name == 'cbegin' else '结果是尾后位置，不可解引用。')
        return ('通过 data() 返回首元素迭代器。' if name == 'begin' else '通过 data()+size() 返回尾后迭代器，不能解引用。') + f'当前重载提供{access}访问，不分配存储或改变元素。'
    if name == 'empty':
        return '空数组特化直接返回 true，不查询数据或迭代器。' if zero else '调用 size() 并与零比较，返回数组是否为空；不读取元素。此处普通模板的 N 值由编译期配置确定。'
    if name == 'size':
        return '空数组特化直接返回 0，单位为元素。' if zero else '返回模板参数 N，单位为元素；这是固定容量，不是通过指针或运行时计数器推算。'
    if name == 'max_size':
        return '空数组特化直接返回 0，单位为元素。' if zero else '调用 size() 返回固定容量 N，单位为元素；本容器没有可增长容量。'
    if name == 'fill':
        return '空数组特化的成员 fill 函数体为空，不读取 value、不构造或赋值元素；调用表达式中的实参求值仍由调用方执行。' if zero else '范围 for 遍历本数组，并把 value 赋给每个元素；使用 begin/end 构成范围，不是原始字节填充。'
    if name == 'clear':
        return '空数组特化的成员 clear 函数体为空，不构造 T(0)，这与自由函数 clear(array&) 的实现不同。' if zero else '构造 T(0) 并调用成员 fill，将每个元素赋为该值；它不把容器大小改成零，也不是释放存储。'
    if name == 'swap':
        return '空数组特化的成员 swap 函数体为空，不访问 other，也不调用元素 swap；两个空对象没有元素需要交换。' if zero else '遍历相同位置，用引入的标准库 swap 加 ADL 交换两个数组的对应元素；调用目标取决于 T。两个数组地址与固定长度保持不变。'
    raise AssertionError(nid)

def alignment_summary(n):
    nid = n['id']
    if nid == 'alignment.api.is_byte_aligned':
        return '把 ptr 转成 uintptr_t，用低位掩码 (N-1) 检查地址是否为 N 字节的整数倍，返回 bool；不会解引用指针，也不会证明对象容量、生命周期或访问合法。N 的二次幂要求由 static_assert 检查。'
    if nid == 'alignment.dep.has_single_bit':
        return '用 x!=0 且 (x & (x-1))==0 判断数值是否恰有一个置位，用于 is_byte_aligned 的编译期 N 检查。它返回 bool，不执行对齐或修正地址。'
    if nid == 'alignment.type.aligned_primary':
        return 'aligned_struct 的空主模板，只建立由 Alignment 与 Child 区分的类型；源码没有给主模板附加对齐属性。只有列出的 1 到 256 的二次幂特化显式声明对齐，不能假定任意 Alignment 都得到请求对齐。'
    if nid.startswith('alignment.type.aligned_'):
        value = nid.rsplit('_', 1)[1]
        return f'为 Alignment={value} 字节提供显式带 CUTE_ALIGNAS({value}) 的空类型特化。Child 只区分类型身份，不是基类或成员；对齐属性约束该类型对象布局，不分配或对齐一个已有指针。'
    if nid == 'alignment.type.array_aligned':
        return '在公开继承 cute::array<T,N> 的固定长度存储外添加 CUTE_ALIGNAS(Alignment) 类型对齐。Alignment 默认为 16 字节，但可显式修改；实际布局还须符合成员自然对齐和编译器要求，不能推断 sizeof 恰为 N*sizeof(T)。'
    if nid == 'alignment.dep.array':
        return array_summary(n)
    if nid == 'alignment.macro.cuda':
        return '在 __CUDACC__ 分支中将 CUTE_ALIGNAS(n) 替换为 __align__(n)，供 aligned_struct 特化和 array_aligned 声明类型对齐。这是预处理替换，不是一次函数调用。'
    if nid == 'alignment.macro.cpp':
        return '在非 __CUDACC__ 分支中将 CUTE_ALIGNAS(n) 替换为标准 alignas(n)，供类型或成员声明使用。它只生成对齐说明，不会在运行时调整对象地址。'
    raise AssertionError(nid)

def external_summary(n):
    name = n['name']
    text = {
        'void': 'C++ 的无返回值/无具体对象类型标记；函数返回 void 不产生结果对象，void 指针还需结合具体对象类型解释。',
        'bool': 'C++ 的真假值类型，用于空数组、相等比较或地址检查的结果；结果值可以在运行时产生。',
        'int': 'C++ 有符号整数类型；这里用于固定对齐 N 或索引运算，具体位宽和算术规则由目标编译环境决定。',
        'size_t': '标准的无符号对象大小类型，本模块主要以元素数和索引为单位使用；参数名为 Alignment 时才是字节数。',
        'ptrdiff_t': '标准的有符号指针差值类型，在数组接口中通过 difference_type 对外提供；类型别名本身不执行地址计算。',
        'uintptr_t': '可承载指针整数表示的无符号整数类型，用于检查地址低位；整数化地址不等于验证该地址可访问。',
    }
    if name in text:
        return text[name]
    if 'tuple_size' in name:
        return f'{name} 是外部标准库 tuple 协议的类型 trait 入口；本文件只为 cute::array 提供特化，主模板及其他类型的实现位于外部标准库。'
    if 'tuple_element' in name:
        return f'{name} 是外部标准库按静态索引查询元素类型的 trait 入口；本文件的 array 特化直接给出 T，不意味着已审核所有外部实现。'
    if 'integral_constant' in name:
        return '标准库的常量包装模板边界，为数组 tuple_size 提供静态 N 与值类型；CUTE_STL_NAMESPACE 条件决定使用 std 或 cuda::std，不是运行时函数。'
    raise AssertionError(n['id'])

def parameter_note(n, p):
    nid, name = n['id'], p.get('name')
    if nid.startswith('elementwise.'):
        notes = {
            'tensor': '调用时 Tensor 对象，提供数据入口和布局；本算法写入其逻辑元素，要求实际引用可赋值且存储有效。',
            'value': '调用时只读填充值，按元素赋值给目标；需要与目标元素或专用 fill 接口兼容，不是字节模式。',
            'alpha': '调用时只读系数，乘以每个选中位置的 x(i)；其类型由 Alpha 确定，数值单位由调用问题决定。',
            'x': '调用时只读输入 Tensor；遍历长度取 size(x)，选中位置读取 x(i)，逻辑范围须与 y 和谓词匹配。',
            'beta': '调用时只读系数，用于旧 y(i) 的缩放；为零时算法不读取旧 y(i)，复数需实部和虚部同时为零。',
            'y': '调用时输出 Tensor；谓词选中时写入，且 beta 非零时先读取同位置旧值。&& 重载只转发临时视图，不接管数据。',
            'p': '调用时只读谓词对象，通过 p(i) 决定是否访问并更新第 i 个逻辑位置；默认 {} 构造的 constant_fn<true_type> 全部选中。',
            'coord': '调用时逻辑坐标，动态分量以元素位置计，不是字节偏移；Coord 类型含下划线时请求切片，否则由 Layout 映射后访问元素。',
        }
        if nid == 'elementwise.api.tensor_size':
            return '调用时只读 Tensor 对象，只读取其布局信息以求逻辑大小，不读取或改写张量元素。'
        if name is None:
            if 'fill_prefer' in nid:
                return '调用形式中的无名优先级标签对象；prefer<1> 或 prefer<0> 的类型在编译期参与重载选择，其值不携带数据或运行时优先级。'
            if nid == 'elementwise.api.constant_fn_call':
                return '调用时的无名转发引用实参包，可有零个或多个输入；函数体全部忽略。实参表达式仍会按 C++ 调用规则求值。'
        assert name in notes, (nid, name)
        return notes[name]
    if nid == 'alignment.api.is_byte_aligned':
        return '调用时的只读指针值；检查其整数地址是否满足 N 字节对齐，不解引用、不修改指针或对象。'
    if nid == 'alignment.dep.has_single_bit':
        return '调用时按值传入的被测整数。用于 static_assert(has_single_bit(N)) 时该实参来自模板常量 N；接口本身也可检查运行时值。'
    if nid == 'array.dependency.move':
        return '调用时被转换值类别的对象，以转发引用接收；不修改对象内容，返回引用的有效期仍由原对象决定。'
    if name == 'pos':
        return '调用时 size_type 下标，单位为元素而非字节；允许动态值，调用方必须保证 0<=pos<N。N=0 没有合法下标。'
    if name == 'value':
        return '调用时只读 T 类型填充值，用于逐元素赋值；空数组成员 fill 不读取它，但实参的求值仍发生。'
    if name == 'other':
        return '调用时同一 array 特化的另一可变数组，普通数组双方对应元素被交换；空数组成员 swap 不访问它。'
    if name in ('lhs', 'rhs'):
        return f'调用时只读的比较{ "左" if name == "lhs" else "右" }侧数组，与另一数组具有相同 T 和 N；用元素 != 判断差异。'
    if name in ('a', 'b'):
        if nid == 'array.api.fn_414':
            return '调用时只读数组，返回其中第 I 个元素的 const 引用；数组须在引用使用期间存活。'
        if nid == 'array.api.fn_422':
            return '调用时以右值引用接收的数组；返回其元素 T&&，不复制元素也不延长临时数组生命周期。'
        if nid == 'array.api.fn_406':
            return '调用时非 const 数组，返回其中第 I 个元素的 T&；本次查询不写入，但返回引用可用于后续赋值（取决于 T）。'
        if nid == 'array.api.fn_367':
            return '调用时参与逐元素交换的可变数组，与另一个数组具有相同 T 和 N；本对象元素可能被改写。'
        return '调用时要填充的可变数组，其逻辑长度由模板 N 决定；函数不改变数组容量或地址。'
    if name == 't':
        return '调用时只读输入数组，元素按反序复制/赋值到返回的新数组；原 t 不被修改。'
    raise AssertionError((nid, name))

def execution(n):
    nid = n['id']
    if nid.startswith('alignment.macro.'):
        return '预处理期的函数式宏：n 是文本参数，展开后由编译器解释对齐说明。没有 Host/Device 调用，也没有 C++ 模板实参。'
    if nid == 'elementwise.external.gnu_unreachable':
        return '编译器内建的控制流假设；可用性由编译器/预处理分支决定。若执行真的到达此处，将违反不可达前提，不能把它当作运行时安全检查。'
    if n['kind'] == 'type' or n['kind'] == 'external':
        return '编译期类型、别名或外部类型边界；它本身不是一次 Host/Device 调用。该类型的对象或相关函数仍可能在运行时使用。'
    if nid == 'alignment.api.is_byte_aligned':
        return '可在 Host 或 Device 调用。N 与 static_assert 在编译期确定，ptr 地址检查在调用时进行；虽声明 constexpr，指针 reinterpret_cast 并不因此成为标准常量表达式。'
    if nid == 'elementwise.api.beta_zero_lambda':
        return '随外层 Host/Device axpby 调用立即求值。if constexpr 按 Beta 类型选择比较方式，捕获的 beta 数值仍来自调用，不保证编译期判零。'
    if 'constexpr' in (n.get('signature') or ''):
        return '可在 Host 或 Device 调用；constexpr 允许满足常量表达式条件时在编译期求值，不要求每次调用都在编译期执行。类型/模板索引在编译期确定，对象、指针及普通实参可以在运行时提供。'
    return '可在 Host 或 Device 调用；模板实例和重载选择在编译期确定，数据访问与数值操作由实际调用执行。本声明没有把算法规定为必须在编译期求值。'

def build():
    result = {}
    for n in SELECTED:
        nid = n['id']
        if nid.startswith('elementwise.'):
            summary = ELEMENTWISE[nid.removeprefix('elementwise.')]
            layer = 'CuTe 张量与逐元素算法层'
        elif n['kind'] == 'external':
            summary = external_summary(n)
            layer = 'C++ 类型系统与标准库边界'
        elif nid.startswith('array.'):
            summary = array_summary(n)
            layer = 'CuTe 基础容器与 tuple 适配层'
        else:
            summary = alignment_summary(n)
            layer = 'CuTe 基础容器与对齐约束层'
        if nid.startswith('alignment.builtin.'):
            summary = external_summary(n)
        params = {p.get('name') or f"#{i}": parameter_note(n, p) for i, p in enumerate(n.get('parameters') or [])}
        if nid.startswith('alignment.macro.'):
            params['n'] = '预处理期文本参数：替换到 __align__(n) 或 alignas(n) 的括号内；本文件调用处用作对齐字节数常量表达式。不是运行时数值或 C++ 模板参数。'
        templates = {p.get('name') or f"#{i}": template_note(n, p.get('name')) for group in n.get('template_parameters') or [] for i, p in enumerate(group.get('parameters') or [])}
        result[nid] = {'layer': layer, 'summary': summary, 'execution': execution(n), 'parameter_notes': params, 'template_notes': templates}
    assert len(result) == 142, len(result)
    for n in SELECTED:
        note = result[n['id']]
        expected_params = {p.get('name') or f'#{i}' for i, p in enumerate(n.get('parameters') or [])}
        if n['id'].startswith('alignment.macro.'):
            expected_params.add('n')
        expected_templates = {p.get('name') or f'#{i}' for g in n.get('template_parameters') or [] for i, p in enumerate(g.get('parameters') or [])}
        assert set(note['parameter_notes']) == expected_params, n['id']
        assert set(note['template_notes']) == expected_templates, n['id']
        assert all(note[k].strip() for k in ('layer', 'summary', 'execution')), n['id']
    # Keep easily confused source contracts separate after any editorial revision.
    checks = {
        'array.api.fn_118': ('summary', '非 const begin()'),
        'array.api.fn_142': ('summary', '非 const end()'),
        'array.api.fn_273': ('summary', '直接返回 nullptr'),
        'array.api.fn_331': ('summary', '不构造 T(0)'),
        'array.api.fn_353': ('summary', 'T(0) 的构造表达式仍存在'),
        'array.api.fn_422': ('summary', '不延长临时数组生命周期'),
        'array.type.std.tuple_element': ('summary', '未在此检查 I<N'),
        'alignment.api.is_byte_aligned': ('execution', '并不因此成为标准常量表达式'),
        'alignment.macro.cuda': ('execution', 'n 是文本参数'),
        'elementwise.api.constant_fn_call': ('summary', 'R 值而不是 R const&'),
        'elementwise.api.fill_prefer1': ('summary', '没有 return'),
        'elementwise.api.axpby_lvalue': ('summary', '不读取旧 y(i)'),
    }
    for nid, (field, expected) in checks.items():
        assert expected in result[nid][field], (nid, field, expected)
    output = {'nodes': result}
    path = ROOT / 'data/interface-notes/cute.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'nodes': len(result), 'function_parameter_notes': sum(len(v['parameter_notes']) for v in result.values()), 'template_parameter_notes': sum(len(v['template_notes']) for v in result.values()), 'output': str(path)}, ensure_ascii=False))

if __name__ == '__main__':
    build()
