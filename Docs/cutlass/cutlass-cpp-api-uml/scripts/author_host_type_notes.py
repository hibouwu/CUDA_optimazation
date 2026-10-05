#!/usr/bin/env python3
"""Author-reviewed explanations for the published Host / Builder / MMA path.

The prose dictionaries below are source-specific editorial content, not name-
based inference. Generation only expands shared definitions and checks coverage.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATLAS = json.loads((ROOT / 'data/atlas.json').read_text())
NODES = {n['id']: n for n in ATLAS['nodes'] if n['id'].startswith(('host.', 'types.', 'dispatch.'))
         and n['kind'] in ('api', 'type', 'external')}

T = {
 'GemmKernel_': '编译期 Kernel 类型。Adapter 由它取得 Arguments、Params、共享内存需求和启动入口；这里要求它符合 3.x Kernel 接口，不能用一次调用的矩阵尺寸代替此类型。',
 'ProblemShape_': '编译期问题形状类型，规定 MNK 或 MNKL 的结构以及哪些分量为静态常量；对象中的普通整数分量仍在调用时提供。',
 'CollectiveMainloop_': '编译期 Mainloop 实现类型，提供输入参数、共享存储、MMA 与输入流水线，Kernel 据此组织计算参与者。',
 'CollectiveEpilogue_': '编译期 Epilogue 实现类型，提供输出参数、融合操作及结果搬运；与 Mainloop 的累加器类型和共享存储需求必须一致。',
 'TileSchedulerTag_': '编译期工作调度策略标签，Kernel 将它与架构和 Tile 配置组合成实际 TileScheduler；不是每次运行的工作 tile 编号。',
 'Stages': '编译期 A/B 主循环输入流水级数，决定暂存的 K tile 数以及相应共享内存和同步存储。',
 'SchedulerPipelineStageCount': '编译期调度结果流水级数；它服务于工作分发，不是 A/B 输入数据流水级数。',
 'AccumulatorPipelineStageCount': '编译期累加器流水级数，决定可并行周转的累加器阶段及其同步状态。',
 'ArchTag_': '编译期架构策略类型，参与选择 Collective 的架构属性；不等同于编译命令指定的二进制目标。',
 'ClusterShape': '编译期 Cluster 形状的表示类型，描述沿 MNK 组织的 CTA 数；静态常量分量写入类型，动态分量由实际形状对象或硬件信息提供。',
 'TileShape_': '编译期 Mainloop Tile 的 MNK 形状类型，以矩阵元素为尺度，用于组织 MMA 与 A/B 共享存储。',
 'ElementA_': '编译期 A 元素类型，决定读取和 MMA 所需的数值表示、元素位数与存储解释。',
 'ElementB_': '编译期 B 元素类型，决定读取和 MMA 所需的数值表示、元素位数与存储解释。',
 'StrideA_': '编译期 A 步长表示类型，对应逻辑 MKL；其中静态单位步长可参与分派，普通整数步长在 Arguments 中给出。',
 'StrideB_': '编译期 B 步长表示类型，对应逻辑 NKL；它规定坐标到地址的计算形式，不包含本次 B 基址。',
 'TiledMma_': '编译期 TiledMMA 类型，规定 Atom、协作参与者和各操作数分区；不是 Host 启动调度器。',
 'GmemTiledCopyA_': '编译期 A 的全局内存到共享内存搬运操作类型；该特化据它构造 TMA Atom 和描述符。',
 'GmemTiledCopyB_': '编译期 B 的全局内存到共享内存搬运操作类型；参与 TMA Atom、广播范围与共享目标布局的构造。',
 'SmemLayoutAtomA_': '编译期 A 共享内存布局基本单元，结合 Tile 和 stage 形成完整暂存布局，并满足所选 MMA 的寻址要求。',
 'SmemLayoutAtomB_': '编译期 B 共享内存布局基本单元，结合 Tile 和 stage 形成完整暂存布局。',
 'SmemCopyAtomA_': '编译期 A 的共享内存读取/操作数构造策略槽；该 SM100 SS 路径可使用 void，不能据此假定存在共享内存到普通寄存器的数值拷贝。',
 'SmemCopyAtomB_': '编译期 B 的共享内存读取/操作数构造策略槽；该 SS MMA 通过共享内存描述符取数，不应画成必然先搬到普通寄存器。',
 'TransformA_': '编译期 A 操作数变换类型，规定 Collective 接口中的输入变换策略；是否形成实际算术步骤要看具体特化使用。',
 'TransformB_': '编译期 B 操作数变换类型，与输入数值表示和所选 Collective 特化共同决定其实际用途。',
 'ProblemShape': '编译期问题形状表示类型；函数接收其对象取得本次 MNK(L) 数值，类型为静态不表示各尺寸值也静态。',
 'StagesC_': '编译期 C 输入加载流水级数，影响 Epilogue 的 C 暂存和加载同步资源。',
 'StagesD_': '编译期 D 输出存储流水级数，影响共享输出缓冲及异步 TMA store 的周转。',
 'FragmentSize_': '编译期融合计算的片段元素数，用于组织回调的局部数据处理粒度。',
 'ReuseSmemC_': '编译期布尔选择：允许 Epilogue 的 C 加载与 D 输出按该特化的协议复用共享存储；不代表可不等待就覆盖。',
 'DelayTmaStore_': '编译期布尔策略：控制该 Epilogue 的 TMA 输出提交时机与流水组织，不是 CUDA stream 的运行时开关。',
 'CtaTileShape_': '编译期单 CTA 的输出问题 Tile 形状，用于划分 Epilogue 工作和匹配累加器片段。',
 'EpilogueTile_': '编译期 Epilogue 子 Tile 形状，规定一次输出处理的 MN 范围，区别于整个 CTA Tile。',
 'ElementC_': '编译期源矩阵 C 的元素类型；void 等配置还参与决定是否存在 C 输入路径。',
 'ElementD_': '编译期输出矩阵 D 的元素类型，决定转换结果与输出搬运的元素宽度。',
 'StrideC_': '编译期 C 的 MNL 步长表示类型，具体动态步长来自本次 Epilogue Arguments。',
 'StrideD_': '编译期 D 的 MNL 步长表示类型，具体输出地址由基址、坐标和调用时的步长共同确定。',
 'FusionCallbacks_': '编译期融合回调类型，定义累加结果与 C、缩放因子等输入怎样生成输出，以及自己的参数和 workspace 合同。',
 'CopyOpT2R_': '编译期累加器从 TMEM 读到线程寄存器的 Copy 操作类型，连接 MMA 累加器和融合计算。',
 'CopyOpG2S_': '编译期 C 从全局内存载入共享内存的 Copy 操作类型，可决定普通 TMA 或 im2col 分支。',
 'SmemLayoutAtomC_': '编译期 C 暂存的共享内存布局基本单元。',
 'CopyOpS2R_': '编译期将 C 从共享内存读入线程局部计算所用寄存器片段的 Copy 操作类型。',
 'CopyOpS2G_': '编译期 D 从共享内存写回全局内存的 Copy 操作类型，决定输出 TMA 提交路径。',
 'SmemLayoutAtomD_': '编译期 D 输出暂存的共享内存布局基本单元。',
 'CopyOpR2S_': '编译期将融合计算后的 D 片段从寄存器写到共享输出缓冲的 Copy 操作类型。',
 'CopyOpR2R_': '编译期寄存器片段间的重排/传递操作类型，用于衔接累加器读取和输出分区。',
 'ClusterShape_': '编译期调度器的 Cluster 形状类型；静态形状直接参与组织，动态形状需与 KernelHardwareInfo 结合。',
 'Stages_': '编译期调度器流水级数，控制 CLC 工作获取/传递的阶段数，不是 K 主循环的 A/B 数据 stage 数。',
 'ElementAccumulator': '编译期累加器元素类型；在 Builder 中选择 MMA 数值类型，在调度 workspace 模板中保留与上层统一的归约接口。',
 'ProblemShapeMNKL': '编译期 MNKL 问题形状对象的表示类型；实际 M、N、K 与批次 L 从函数形参读取。',
 'TileShape': '编译期 Tile 形状表示类型，规定 MNK 工作单元的结构；具体对象中的静态/动态分量共同参与 CTA 数量计算。',
 'AtomThrShape': '编译期 Atom 协作 CTA 布局形状类型，用来把 MMA Tile 尺寸换算为单 CTA 工作范围，不能当成 CUDA block 的线程维度。',
 'Operator': '编译期设备操作符类型，提供 Params、线程数量约束和 operator()；device_kernel 用它实例化真正的全局入口包装。',
 'ArchTag': '编译期架构标签，选择 CUTLASS 的 C++ 实现策略；本例 Sm100 与实际 sm110a 二进制编译目标应分别记录。',
 'OpClass': '编译期运算类别，例如 OpClassTensorOp，决定采用 Tensor Core 等哪类计算实现。',
 'ElementA': '编译期 A 存储/输入元素类型，影响 MMA 选择、搬运对齐和共享内存字节数。',
 'ElementB': '编译期 B 存储/输入元素类型，影响 MMA 选择、搬运对齐和共享内存字节数。',
 'GmemLayoutA': '编译期 A 的全局布局标签或步长表示，Builder 将它转换为底层步长和 UMMA major；不携带 A 的运行时指针。',
 'GmemLayoutB': '编译期 B 的全局布局标签或步长表示，B 的逻辑坐标是 NKL，major 映射不能直接照抄 A。',
 'GmemLayoutATag': '编译期 A 全局布局标签/表示，SM100 Builder 据此选择 A 的 major 与数据路径。',
 'GmemLayoutBTag': '编译期 B 全局布局标签/表示，按 B 的 NKL 逻辑组织转换为 major。',
 'AlignmentA': '编译期承诺的 A 对齐粒度，单位是 A 元素数而非字节；必须与实际基址、步长及实现限制相容。',
 'AlignmentB': '编译期承诺的 B 对齐粒度，单位是 B 元素数而非字节。',
 'TileShape_MNK': '编译期 MNK Tile 形状；该 SM100 Builder 路径要求静态形状，以确定 MMA M/N 和 K 暂存大小。',
 'ClusterShape_MNK': '编译期 Cluster 形状表示，单位为各维 CTA 数；它可包含动态维度，Auto 选择会区分静态与动态 Cluster。',
 'StageCountType': '编译期 stage 策略类型，表示显式 stage 数或带 carveout 的自动计算策略。',
 'KernelScheduleType': '编译期 Mainloop 调度策略标签，交给 Builder 选择 Collective 路径；不是 Kernel 的 TileScheduler。',
 'Enable': '编译期特化使能参数；默认 void，由模板约束筛选适用的 Builder 偏特化，不是运行时开关。',
 'BuilderScheduleTag': '编译期 Builder 调度标签，可显式要求 1SM/2SM 或 Auto；用于 if constexpr 选择 MMA 构造分支。',
 'CapacityBytes': '编译期允许使用的共享内存容量，单位字节，是自动 stage 计算的总预算。',
 'TileShapeMNK': '编译期 MNK Tile 形状，以元素数计算每级 A/B 暂存大小；不是传入的数据数组。',
 'MainloopPipelineStorage': '编译期单级主循环同步存储类型，其 sizeof 字节数加入每一级共享内存预算。',
 'carveout_bytes': '编译期先为 Epilogue 等其他用途保留的共享内存字节数；自动 stage 使用剩余容量而非全部容量。',
 'Layout': '编译期布局标签或布局/步长表示类型，用于判定连续主维并映射 UMMA::Major；A 与 B 的映射规则不同。',
 'ElementAMma': '编译期 MMA 实际 A 数值表示，可能已由外部存储类型变换而来；决定选 FP16/BF16、TF32 等哪类指令包装。',
 'ElementBMma': '编译期 MMA 实际 B 数值表示，与 A 一起接受对应指令族的类型约束。',
 'ElementAMmaccumulator': '源码中如此拼写的编译期累加器元素类型，传给所选 2SM MMA 包装的 c_type。',
 'UmmaMajorA': '编译期 A 共享操作数的 UMMA 主维枚举（K 或 MN），参与指令描述符和 SMEM 描述符的解释。',
 'UmmaMajorB': '编译期 B 共享操作数的 UMMA 主维枚举，按 B 的逻辑坐标确定。',
 'ANeg': '编译期 A 输入符号缩放枚举，默认 One；选择是否取负，不是任意运行时浮点缩放因子。',
 'BNeg': '编译期 B 输入符号缩放枚举，默认 One。',
 'MMA_Op': '编译期 MMA 操作/Traits 表示类型，工厂据此形成 MMA_Atom 并组合成 TiledMMA。',
 'MMAThrLayout': '编译期 Atom 在 MNK 方向重复/协作的布局表示类型；函数仍接收布局对象，其中可含值信息。',
 'Permutations': '编译期 MNK 排列描述的表示类型，用于 TiledMMA 的 Tile/坐标排列；不是输入矩阵的数值内容。',
 'MMAOperation': '编译期架构 MMA 包装类型，MMA_Traits 和 MMA_Atom 据此取得操作数表示与调用入口。',
 'Args': '编译期传给 MMA_Traits 的附加模板实参包，具体含义由对应 Traits 特化定义；不是 call() 的运行时 A/B/C/D 参数包。',
 'a_type': '编译期 MMA A 元素类型，该 F16/BF16 2SM 路径要求 16 位输入表示。',
 'b_type': '编译期 MMA B 元素类型，该 Traits 要求与 A 的元素位数一致且为 16 位。',
 'c_type': '编译期累加器元素类型，确定 TMEM 累加数值及指令描述符中的累加类型。',
 'M': '编译期一条 2SM MMA 的 M 维元素数；此包装要求 128 或 256，不是整个 GEMM 的运行时 M。',
 'N': '编译期一条 2SM MMA 的 N 维元素数；此包装要求 16 到 256 之间且为 16 的倍数。',
 'a_major': '编译期 A 的 UMMA 主维枚举，用于解读共享内存描述符和构造指令描述符。',
 'b_major': '编译期 B 的 UMMA 主维枚举。',
 'a_neg': '编译期 A 输入符号缩放枚举，默认 One，不是 Host GEMM 的 alpha。',
 'b_neg': '编译期 B 输入符号缩放枚举，默认 One。',
 'MMA': '编译期 MMA_Atom 的操作/Traits 参数类型，参与 gemm 重载选择并决定最终 Atom 调用。',
 '#9': '编译期 SFINAE 条件槽：要求 D/C rank 为 3、A/B rank 为 2，并满足相应 engine 分类。is_rmem 是此处的类型谓词，不能据此把本例的 TMEM 累加器叫作普通线程寄存器。',
}
for letter in 'DABC':
    T['T' + letter] = f'编译期 {letter} Tensor 的 engine 类型，规定数据/指针的访问表示；其中的实际地址和数值仍由函数形参传入。'
    T[letter + 'Layout'] = f'编译期 {letter} Tensor 的布局表示类型，规定秩、静态形状与坐标映射形式；布局对象仍可能含动态尺寸或步长。'

HOST = '在主机调用时执行；static 只表示无需对象实例，不表示函数的尺寸、指针等形参是编译期参数。'
TYPE = '这是编译期类型或类型选择入口，不是执行一次 GEMM 的运行时函数；对象的动态字段仍须在调用时准备。'
FACTORY = '这是 constexpr 工厂/查询：类型选择在模板实例化时完成，函数是否被常量求值取决于调用上下文；constexpr 本身不把普通函数形参变成模板参数。'
DEVICE = '本例在 GPU 设备代码中调用；接口上的 Host/Device 标记表示可编译的上下文，不保证底层硬件指令能在主机执行。'
CUDA = '这是 CUDA Runtime 外部边界，主机调用；当前图集未收录 SDK 的完整函数声明，空的 parameters 不能解释为该函数没有形参。'

R = {}
def add(identifier, layer, summary, execution, parameters=None, **extra):
    node = NODES[identifier]
    notes = {}
    for group in node.get('template_parameters') or []:
        for index, p in enumerate(group.get('parameters', [])):
            key = p.get('name') or '#' + str(index)
            if key not in T:
                raise ValueError('Missing template explanation: ' + identifier + ':' + key)
            notes[key] = T[key]
    R[identifier] = dict(layer=layer, summary=summary, execution=execution,
                         parameter_notes=parameters or {}, template_notes=notes, **extra)

ARGS = '调用时只读的整次 GEMM 参数对象，包含模式、MNK(L) 尺寸、Mainloop/Epilogue/Scheduler 参数与硬件信息；其中的指针指向实际矩阵，但检查/准备接口不执行矩阵乘法。'
WORK = '调用时传入的 workspace 基址；所需容量以 get_workspace_size 返回的字节数为准。内存由调用方提供，Kernel 按对齐要求划分给各组件。'
STREAM = '调用时选择的 CUDA stream 句柄，相关初始化或提交按该 stream 的规则入队；默认 nullptr 并不表示同步等待 GPU 完成。'
ADAPT = '调用时可选的 CUDA Host Adapter 对象指针，供启用该配置时的外部启动/初始化机制使用；本例未启用 Host Adapter，不能把非空指针当成可随意启用的开关。'
PDL = '调用时的 Programmatic Dependent Launch 请求开关；是否接受以及怎样设置属性取决于编译出的启动路径，本例为 false，不是“本次调用是否同步”。'
PAR = '调用时已转换的 Kernel Params 对象，含设备需要的描述符、工作形状与调度状态；不是原始 Host Arguments。'
SHAPE = '调用时的问题 MNK 或 MNKL 形状对象；M/N/K 以矩阵元素为尺度，L 为批次数。对象类型可包含静态分量，其余数值仍在本次调用传入。'
HW = '调用时的 KernelHardwareInfo，提供设备及 Cluster 相关信息；具体字段是否使用由当前特化决定，不能据名称假定会服从用户指定的 SM 数。'

add('host.adapter','Device 层：主机句柄','GemmUniversalAdapter 将满足 3.x 接口的 Kernel 包装成主机句柄，向调用方提供能力检查、workspace 查询、参数准备和提交。句柄保存 Params，但不拥有矩阵分配，也不把 run() 返回成功解释为设备计算完成。',TYPE)
add('host.kernel','Kernel 层：跨组件组织','该 GemmUniversal 特化把 ProblemShape、Mainloop、Epilogue 和 TileScheduler 组合成设备 Kernel。Host 侧接口检查并转换参数，设备 operator() 再组织各参与者与共享资源；它不是单个 MMA 指令包装。',TYPE)
add('host.adapter.arguments','Device 层：参数类型入口','这是 GemmKernel::Arguments 的类型别名，让调用方通过 Adapter 准备同一份 Kernel 输入描述，没有新增一层数据复制或另一套参数格式。',TYPE)
add('host.adapter.params','Device 层：设备参数类型入口','这是 GemmKernel::Params 的类型别名，Adapter 用它保存已转换的设备参数。它与 Arguments 的区别在于描述已经面向实际 Kernel 数据路径，而非只是更换一个名字。',TYPE)
add('host.kernel.arguments','Kernel 层：Host 输入描述','Arguments 汇集 mode、problem_shape、Mainloop/Epilogue 参数、hw_info 和 scheduler 参数。它描述调用者要计算的问题；TMA 等实际设备描述符由后续转换函数准备。',TYPE)
add('host.kernel.params','Kernel 层：设备输入描述','Params 汇集问题形状、模式以及 Mainloop/Epilogue/Scheduler 各自转换后的 Params 和硬件信息，供设备入口按值传给 Kernel。Arguments 到 Params 不是简单按字节复制。',TYPE)
add('host.can_implement','Device 层：能力检查','将 Kernel 的布尔能力检查转换成 CUTLASS Status，供调用方在初始化前判断当前尺寸和配置是否被接受。initialize() 不会代替调用者执行这个检查。',HOST,{'args':ARGS})
add('host.get_workspace_size','Device 层：工作区查询','返回本次运行需要的 workspace 字节数，汇入 Kernel 的需求并保留 Adapter 对模式的额外处理。返回容量不代表已经分配或初始化这段内存。',HOST,{'args':ARGS})
add('host.initialize','Device 层：状态初始化','先调用 Kernel 的 workspace 初始化，再把 Arguments 转换为 params_，并按编译路径设置动态共享内存相关属性。任一步骤的错误会影响返回状态；它不调用 can_implement()。','在主机调用时更新当前句柄；本次传入 Arguments、workspace、stream 和可选 cuda_adapter，Kernel 类型由编译期模板确定。',{'args':ARGS,'workspace':WORK,'stream':STREAM,'cuda_adapter':ADAPT})
add('host.update','Device 层：参数更新','根据新 Arguments 和 workspace 重建 params_；它可查询工作区需求，但不会执行 Kernel::initialize_workspace。因而它不等同于完整 initialize，也不能无条件视为只改指针的轻量操作。','在主机调用时用本次 Arguments 和 workspace 重建句柄的 params_。此重载没有 stream 形参，也不提交 GEMM；Kernel 类型由编译期模板确定。',{'args':ARGS,'workspace':WORK+' 此函数不初始化工作区内容，调用方必须保证所需状态已经有效。'})
add('host.run.params','Device 层：实际提交入口','这个静态重载直接使用给定 Params 计算 grid/block/shared-memory 配置，封装内核参数并选择对应 CUDA 启动路径。返回值报告提交/API 错误，不证明 GPU 已经结束或数值正确。',HOST,{'params':PAR+' 形参为可变引用，应以当前分支对它的实际使用为准。','stream':STREAM,'cuda_adapter':ADAPT,'launch_with_pdl':PDL})
add('host.run.args','Device 层：初始化并提交','这个重载先对 Arguments 调用 initialize，成功后再使用句柄状态提交。因此它比只接收 stream 的 run 多一个准备阶段，不应把两个重载的行为合并。','在主机调用时先准备句柄再提交；本次 Arguments、workspace、stream、cuda_adapter 和 launch_with_pdl 都是函数实参，Kernel 类型在编译期确定。',{'args':ARGS,'workspace':WORK,'stream':STREAM,'cuda_adapter':ADAPT,'launch_with_pdl':PDL})
add('host.run.stream','Device 层：复用参数提交','读取句柄中已有的 params_，转发给静态 run(Params&, …)。本次没有传入新 Arguments，因此必须先通过 initialize/update 建立与当前矩阵匹配的有效参数。','在主机调用时使用句柄中已有的 params_，本次仅接收 stream、cuda_adapter 和 launch_with_pdl；矩阵地址和问题参数不是此重载的新实参。',{'stream':STREAM,'cuda_adapter':ADAPT,'launch_with_pdl':PDL})
add('host.call.args','Device 层：调用运算符转发','operator()(Arguments, …) 把同一组参数转交给 run(Arguments, …)，保留其先初始化再提交的行为；运算符语法没有创建独立的设备算法。','在主机调用时转发本次 Arguments、workspace、stream、cuda_adapter 和 launch_with_pdl；实际初始化和提交发生在对应的 run 重载中。',{'args':ARGS,'workspace':WORK,'stream':STREAM,'cuda_adapter':ADAPT,'launch_with_pdl':PDL})
add('host.call.stream','Device 层：调用运算符转发','operator()(stream, …) 使用已有 params_ 提交，不重新接收问题或矩阵参数。它与带 Arguments 的调用运算符是独立重载。','在主机调用时使用已有的 params_，本次仅提供 stream、cuda_adapter 和 launch_with_pdl；它不接收新的 Arguments 或 workspace。',{'stream':STREAM,'cuda_adapter':ADAPT,'launch_with_pdl':PDL})
add('host.grid.args','Device 层：从问题查询网格','先把 Arguments 转成临时 Kernel Params，再交给 Kernel::get_grid_shape 返回 CUDA grid 维度；这会经过参数转换，不是只读取一个静态 Tile 常量。',HOST,{'args':ARGS,'workspace':WORK+' 基址用于临时 Params 的转换，不由此查询函数分配。'})
add('host.grid.params','Device 层：从设备参数查询网格','直接把已有 Params 交给 Kernel 的网格计算接口，省去 Arguments 转换。返回 dim3 的各维表示 block/CTA 数而不是矩阵元素数。',HOST,{'params':PAR})
add('host.kernel.can_implement','Kernel 层：组合能力检查','检查 GEMM 模式与问题秩，然后分别检查 Mainloop、Epilogue、Scheduler；动态 Cluster 和 Block-Scaled 条件还可能增加限制。返回 true 只表示这些源码检查接受参数，不替代真实运行验证。',HOST,{'args':ARGS})
add('host.kernel.workspace','Kernel 层：工作区布局','把 Epilogue 和 Scheduler 的工作区字节需求相加，并在各段之后按 MinWorkspaceAlignment 向上取整。这一组织必须与转换和初始化时的分段方法一致。',HOST,{'args':ARGS})
add('host.kernel.initialize_workspace','Kernel 层：组件初始化交接','按与 workspace 查询一致的偏移依次初始化 Epilogue 与 Scheduler 的工作区，并传播错误状态。Mainloop 在这条特化路径没有独立 workspace 初始化段。',HOST,{'args':ARGS,'workspace':WORK,'stream':STREAM,'cuda_adapter':ADAPT})
add('host.kernel.lower','Kernel 层：Arguments 到 Params','按同一布局切出 Epilogue 和 Scheduler workspace，再分别调用三组件的 to_underlying_arguments 生成 Params。这里 Mainloop workspace 为 nullptr；整个转换不会发起一次 GEMM。',HOST,{'args':ARGS,'workspace':WORK})
add('host.kernel.grid','Kernel 层：网格组织','从 Params 选择当前 Cluster 形状，将问题补成 MNKL，连同 Tile、Atom 协作形状和调度 Params 交给 Scheduler，得到实际启动 grid。',HOST,{'params':PAR+' 本函数读取 problem_shape、scheduler 和 hw_info 等字段计算 CTA 网格。'})
add('host.kernel.block','Kernel 层：线程规模','返回 dim3(MaxThreadsPerBlock, 1, 1)，给 Host 启动路径提供单个 block 的线程维度；这个值来自 Kernel 类型配置，不是 MNK 矩阵尺寸。',HOST)

MA = '调用时的 Mainloop Arguments，主要包含 A/B 全局基址、步长和适用路径的运行时数据类型标志。'
EA = '调用时的 Epilogue Arguments，包含 C/D 基址、步长以及 thread 中的融合回调参数；不能与 Mainloop Arguments 混用。'
SA = '调用时的 Scheduler Arguments，当前实现主要读取 max_swizzle_size 与 raster_order，用来组织逻辑 tile 到物理 grid 的映射。'
add('host.mainloop.can_implement','Collective 层：Mainloop 能力检查','依据输入元素位宽与所选数据路径计算 TMA 对齐要求，再检查问题形状与 StrideA/B 类型是否满足要求。当前函数没有逐一读取 args 中的指针内容，因此不能把 true 扩大为所有输入内存均已验证。',HOST,{'problem_shape':SHAPE,'args':MA+' 该特化此函数体未使用此形参，[[maybe_unused]] 不表示所有其他接口也忽略它。'})
add('host.mainloop.lower','Collective 层：输入描述符准备','从 A/B 基址、问题形状与实际步长建立全局张量视图，再为首选及 fallback Cluster 构造 TMA_A/TMA_B，连同运行时类型标志组成设备 Params。此阶段准备描述符，不搬运矩阵数据。',HOST+' 虽声明 constexpr，本例使用运行时地址和硬件信息在 Host 准备参数。',{'problem_shape':SHAPE,'args':MA,'workspace':'调用时传入的兼容性工作区参数；此 Mainloop 特化的函数体不使用它，Kernel 调用处传 nullptr。','hw_info':HW+' 这里读取首选与 fallback Cluster 形状构造对应 TMA 描述符。'})
add('host.epilogue.can_implement','Collective 层：Epilogue 能力检查','先检查 D 及可用的 C 路径的 TMA 形状对齐，再询问 FusionCallbacks 是否接受融合参数。源矩阵支持与 im2col 等分支由类型配置决定。',HOST,{'problem_shape':SHAPE,'args':EA+' 对齐检查主要使用形状/步长类型，args.thread 传给融合回调的能力检查。'})
add('host.epilogue.workspace','Collective 层：融合工作区查询','把问题形状和 args.thread 交给 FusionCallbacks 查询附加工作区字节数，再由 Kernel 合并入总 workspace。不能因某个线性组合配置返回零就假定所有 Epilogue 都不需要工作区。',HOST,{'problem_shape':SHAPE,'args':EA})
add('host.epilogue.initialize_workspace','Collective 层：融合工作区初始化','将问题、融合参数、工作区、stream 和可选 Host Adapter 交给 FusionCallbacks 初始化，并直接返回其状态。它不负责分配整个 Kernel workspace。',HOST,{'problem_shape':SHAPE,'args':EA,'workspace':WORK+' 此处已是 Kernel 切给 Epilogue 的子段。','stream':STREAM,'cuda_adapter':ADAPT})
add('host.epilogue.lower','Collective 层：输出与融合参数准备','构造可选 C 输入的 TMA 描述符和必需的 D 输出描述符，并把融合参数转换成回调 Params。由此把 Host 的输出需求转成设备 Epilogue 能直接使用的描述。',HOST+' constexpr 不使本次 C/D 地址或步长变为编译期数值。',{'problem_shape':SHAPE,'args':EA,'workspace':WORK+' 此处传给 FusionCallbacks::to_underlying_arguments；不是未使用参数。'})

add('host.scheduler','Kernel 调度基础设施：工作分配','该 PersistentTileSchedulerSm100 类型负责把问题切成 CTA 工作并配合 CLC 分发后续任务，提供 Host 参数、网格及工作区接口。它不同于 Mainloop 的 KernelScheduleTag，也不负责实现 MMA 算术。',TYPE)
add('host.scheduler.can_implement','Kernel 调度基础设施：能力入口','保留统一的调度器能力接口，但本特化函数体直接返回 true，未检查传入的调度参数和硬件信息。调用这个函数不能作为调度参数已经充分校验的证据。',HOST,{'args':SA+' 本函数实际未读取。','#1':'无名的调用时 KernelHardwareInfo 引用；当前函数体未使用，只用于保持统一接口形状。'})
add('host.scheduler.workspace','Kernel 调度基础设施：工作区查询','将问题补成 MNKL、选择 Cluster 形状，再转交 Params::get_workspace_size。归约 warp-group 数、Epilogue 子块数和累加器矩阵数在此重载中保留但未使用。',HOST,{'args':SA,'problem_shape':SHAPE,'hw_info':HW,'reduction_warp_groups':'调用时的归约 warp-group 数量接口槽，单位为组；此特化标为 maybe_unused，未参与本次计算。','epilogue_subtile':'调用时的 Epilogue 子 Tile 数接口槽，默认 1；此特化未使用。','num_accumulator_mtxs':'调用时的累加器矩阵个数接口槽，默认 1；此特化未使用。'})
add('host.scheduler.initialize_workspace','Kernel 调度基础设施：工作区初始化','选择 Cluster 并把工作区、问题规模、硬件信息与调度顺序传给 Scheduler Params 的初始化实现。三个无名数量参数是兼容其他调度器归约接口的槽，此函数不读取它们。',HOST,{'args':SA,'workspace':WORK+' 此处是 Kernel 分给 Scheduler 的子段。','stream':STREAM,'problem_shape':SHAPE,'hw_info':HW,'#5':'调用时传入的 reduction_warp_groups，单位为 warp-group 数；源码注释给出含义，但本函数未命名也未使用。','#6':'调用时传入的 epilogue_subtile，默认 1；源码注释标识为 Epilogue 子 Tile 数，本函数未使用。','#7':'调用时传入的 num_accumulator_mtxs，默认 1；源码注释标识为累加器矩阵数，本函数未使用。','cuda_adapter':ADAPT})
SCHED_SHAPES = {'problem_shape_mnkl':SHAPE,'tile_shape_mnk':'调用时传入的 MNK Tile 形状对象，以元素数描述工作粒度；本例用静态 TileShape{} 构造，不应因此将此函数形参列为模板参数。','atom_thr_shape_mnk':'调用时传入的 Atom 协作 CTA 形状对象，用于把 MMA Tile 换算成各 CTA 的工作范围；不是 block 的线程数。','cluster_shape_mnk':'调用时的 Cluster 形状对象，分量表示各方向 CTA 数；若类型含动态维度，实际值参与选择与网格计算。','hw_info':HW}
add('host.scheduler.lower','Kernel 调度基础设施：调度参数转换','由问题、Tile、Atom 协作形状和选定 Cluster 计算 tiled CTA 形状，再把 raster order 与 swizzle 等信息写入返回的 Scheduler Params。workspace 在这个重载的函数体中未使用。',HOST,dict(SCHED_SHAPES,args=SA,workspace='调用时的兼容性 workspace 基址参数，默认 nullptr；此重载没有使用它。'))
add('host.scheduler.grid','Kernel 调度基础设施：物理网格','计算按 Cluster 整理的 CTA 网格，再按 Params 中的 raster order 决定是否转置 x/y 的组织。输出 dim3 是 CTA 数，不是线程数；当前函数体不读取 hw_info。','该接口声明 CUTLASS_HOST_DEVICE；本图中的调用来自 Host 网格查询，普通形状实参仍是调用时对象。',dict(SCHED_SHAPES,params='调用时只读的调度 Params；本函数读取 raster_order_ 与 Cluster divmod 信息，以决定物理 grid 的排列。',hw_info=HW+' 当前重载函数体未使用此形参。'))

GRID = '调用时 grid 的 x/y/z 维度，各分量是要启动的 block/CTA 数，不是矩阵元素数或线程数。'
CLUSTER = '调用时 Cluster 的 x/y/z 形状，单位为每个 Cluster 各方向的 CTA 数。'
BLOCK = '调用时每个 block 的 x/y/z 线程维度，各分量以线程数计。'
SMEM = '调用时为每个 block 请求的动态共享内存容量，单位字节；不是整个 grid 的总共享内存。'
add('host.cluster.launch','运行时边界：Cluster 启动封装','创建 Cluster 启动配置，检查维度并初始化 kernel 的相关函数属性，再调用 cudaLaunchKernelExC 提交。编译期未启用 Cluster 启动支持时返回 kInvalid；成功也只表示提交阶段未报告错误。',HOST,{'grid_dims':GRID,'preferred_cluster_dims':CLUSTER+' 表示首选 Cluster 形状。','fallback_cluster_dims':CLUSTER+' 表示回退形状，是否使用受已编译的启动能力和配置影响。','block_dims':BLOCK,'smem_size':SMEM,'cuda_stream':STREAM,'kernel':'调用时的 CUDA kernel 入口地址，来自已编译的 device_kernel<Operator>；不是 GPU 上的普通数据缓冲指针。','kernel_params':'调用时的参数地址数组，每个元素指向一个 Host 侧实参对象；此路径传递 Params 的地址，数组本身不是矩阵数据。','launch_with_pdl':PDL})
add('host.cluster.make_config','运行时边界：启动属性组装','把 grid/block/shared-memory/stream 与 Cluster、PDL 属性装入 LaunchConfig。存在 fallback 时会先配置回退 Cluster，并在支持 preferred Cluster 的编译分支补充首选属性。此函数只构造配置，不启动内核。',HOST,{'grid_dims':GRID,'cluster_dims':CLUSTER+' 是主要/首选形状，具体放入哪项属性取决于是否有 fallback。','block_dims':BLOCK,'smem_size':SMEM,'cuda_stream':STREAM,'launch_with_pdl':PDL,'fallback_cluster_dims':CLUSTER+' 默认 {0,0,0} 表示没有有效 fallback，而不是一个实际可启动的零尺寸 Cluster。'})
add('host.cluster.config','运行时边界：启动配置对象','LaunchConfig 在启用相应 CUDA 支持的预处理分支持有 cudaLaunchConfig_t 及属性数组，使描述的 Cluster 和 PDL 属性在提交时有可引用的存储。它是主机配置对象，不是 Kernel 的 Params。',TYPE)
add('host.cluster.grid_dim','运行时边界：配置读取','从当前 LaunchConfig 的 launch_config 字段返回 gridDim。它是无显式形参的成员函数，读取的是当前配置对象，而不是计算问题所需的网格。','在主机调用时读取当前 LaunchConfig 对象的字段；没有显式形参不代表没有 this 对象状态。')
add('host.cluster.check_dims','运行时边界：Cluster 维度检查','检查 Cluster 总 CTA 数是否在实现上限内，以及 grid 各维是否可被对应 Cluster 维度整除。此函数不是所有 CUDA 启动条件的完整验证；传入形状仍须满足基本有效性。',HOST,{'grid':GRID,'cluster':CLUSTER+' 检查中的取模要求相应分量有效非零，不能把此函数当作能安全处理任意零维输入的通用校验器。'})
add('host.cluster.init','运行时边界：kernel 属性准备','对指定 kernel 设置允许非可移植 Cluster 尺寸的 CUDA 函数属性，并传播错误；不支持该编译分支时返回无效状态。它不分配矩阵或初始化 GEMM 参数。',HOST,{'kernel_function':'调用时的已编译 CUDA kernel 入口地址；用于 cudaFuncSetAttribute 的目标，不是一个可在 Host 上直接执行的普通函数调用。'})
add('host.synclog.setup','诊断基础设施：主机日志准备','为启用同步日志的构建准备设备日志缓冲与相关状态，供设备端同步事件记录使用。它服务于诊断，并非 GEMM 算术或正常同步协议的完成条件。',HOST)
add('host.synclog.print','诊断基础设施：设备日志输出','在设备入口的末尾按已启用的日志配置输出同步记录；禁用日志时对应路径为空操作。到达日志打印不能替代 Host 的 stream 完成等待。',DEVICE)
add('host.device_kernel','运行时边界：全局设备入口','device_kernel<Operator> 是 CUDA 启动的全局包装入口：接收一个 Params 值，取得 extern __shared__ 区域，构造 Operator 并调用 op(params, smem)。旧参数账本把这一个形参拆成两片，展示应按源码校正为单个 params。','Host 只把该入口交给 CUDA 启动；函数体在 GPU 线程中执行，不是 Adapter 对设备 functor 的普通 Host 调用。',{'params':'调用时按值传入的 Operator::Params 对象，设备 functor 读取它访问各组件描述符和问题信息。CUTLASS_GRID_CONSTANT 是入口参数属性，不是第二个形参。'},parameter_override={'reason':'旧展示把 CUTLASS_GRID_CONSTANT typename Operator::Params const params 错分为两个参数片段；按固定源码的完整函数声明显示单个 params。','source':{'path':'include/cutlass/device_kernel.h','start_line':118,'end_line':118},'parameters':[{'name':'params','raw':'CUTLASS_GRID_CONSTANT typename Operator::Params const params','type':'CUTLASS_GRID_CONSTANT typename Operator::Params const','default':None}],'original_parameter_count':2})
add('host.kernel.operator','Kernel 层：设备执行组织','设备入口把 Params 和动态共享内存基址交给该 functor，函数再划分工作角色，连接 Mainloop、累加器流水线、Epilogue 与工作调度。这里协调的是完整 Kernel 执行，不是一条同步的矩阵乘法指令。','仅在 GPU 设备代码中执行；本次 Params、共享内存基址与工作状态是运行时对象，成员类型和协作结构由模板配置确定。',{'params':PAR+' 设备执行按参与角色读取其中对应组件信息。','smem_buf':'调用时当前 CTA 的动态共享内存基址，单位为字节地址；Kernel 把它解释成 SharedStorage，容量由 Host 请求并受编译期 SharedStorageSize 约束。'})
add('host.cuda.set_attribute','CUDA 外部边界：函数属性','接收具体 kernel 的属性设置请求；本图的调用点分别用于动态共享内存上限与非可移植 Cluster 许可，不能把两次调用当成同一项设置。SDK 完整参数声明未收录。',CUDA)
add('host.cuda.last_error','CUDA 外部边界：错误状态','本图的调用点读取 CUDA Runtime 的最近错误状态，用于初始化或启动后的状态处理。它不是 stream 同步，也不能独立证明设备计算成功；具体副作用需按 CUDA SDK 合同核对。',CUDA)
add('host.cuda.launch_ex','CUDA 外部边界：设备提交','承接 ClusterLauncher 构造的 launch_config、kernel 入口和参数地址数组，形成实际设备启动边界。Host 调用返回与 GPU 内核完成是两个事件，详细 SDK 形参未收录。',CUDA)
add('host.cuda.stream_sync','CUDA 外部边界：调用方完成等待','本例由应用在 run() 之后等待所用 stream，作为 Host 使用结果前的完成检查。这个等待不在 Adapter::run 内部；它只能覆盖对应 stream 的工作与依赖。',CUDA)
add('host.app.main','调用方示例：完整使用顺序','Dense 示例负责准备问题与矩阵、查询并提供 workspace、检查 can_implement、initialize 后 run，最后在应用侧等待 stream 并进行结果核对。它展示调用方责任，不是库内部的隐式生命周期。','主机程序入口；构造模板实例的类型在编译时确定，矩阵分配、参数值与调用顺序发生在程序运行时。')

add('types.builder_entry','Collective 层：Builder 公共模板入口','CollectiveBuilder 把架构、运算类别、数据表示、布局、Tile、Cluster 与调度策略收拢为编译期请求，由偏特化生成 CollectiveOp。它本身不是运行时计算函数，失败配置会在模板选择/静态检查处暴露。',TYPE)
add('types.builder_sm100','Collective 层：SM100 Builder 实现','这个 TensorOp 偏特化把布局映射成 UMMA major，选择 1SM/2SM TiledMMA，计算共享内存 stage 预算，再组装实际 CollectiveMma。模板绑定边不能画成运行时逐步调用。',TYPE)
add('types.builder_bound','Collective 层：已实例化配置','这是本例对 SM100 Builder 的配置视图：FP16 A/B、FP32 累加、Tile 256×128×64、Cluster 2×2×1 与 Auto。它引用同一个模板实现，展示具体选择结果而非新增 API 声明。',TYPE)
add('types.arch_recipe','架构基础设施：C++ 策略标签','Sm100 是 CUTLASS 的架构标签，供 Builder 和策略类型选择实现。当前实验以它建立 C++ recipe，但编译的 GPU 二进制目标为 sm110a；不能把两者写成同一个版本或参数。',TYPE)
add('types.schedule_auto','Collective 层：自动选择标签','KernelScheduleAuto 把 Mainloop 调度/MMA 路径的选择交给 Builder 的静态规则。它不在运行时试跑多种 Kernel，也不是逐任务分发的 TileScheduler。',TYPE)
add('types.epilogue_bound','Collective 层：已选 Epilogue','该配置视图给出与 Dense Kernel 配套的 SM100 TMA warp-specialized Epilogue，其共享存储需求先从总预算中保留，再影响 Mainloop 自动 stage 计算。它不是整个 Kernel 的输出完成事件。',TYPE)
add('types.stage_compute','Collective 层：共享内存预算计算','以 A/B 位宽、MNK Tile 和单级同步存储计算每个 Mainloop stage 的字节数，再用 (CapacityBytes − carveout_bytes) / stage_bytes 求可用整数级数。这里计算的是资源配置，不是每次运行处理了多少 K tile。',FACTORY,{'stage_count':'函数的策略对象实参，类型 StageCountAutoCarveout<carveout_bytes> 携带编译期预留字节数；函数体不读取对象字段，真正预算来自模板参数。'})
add('types.dispatch_bound','Collective 层：已选 DispatchPolicy','本例形成 MainloopSm100TmaUmmaWarpSpecialized 的 8/2/4 配置，分别描述输入、调度结果和累加器流水级数，并带 Cluster 与架构信息。三类 stage 不能合并成一个数字解释。',TYPE)
add('types.collective_api','Collective 层：Mainloop 实现模板','这个 CollectiveMma 特化组织 TMA 输入和 UMMA 计算，公开 Host Arguments/Params、共享存储以及设备加载/MMA 等接口。Kernel 使用这些接口协调参与者，底层 TiledMMA/Atom 则决定具体操作数分区和指令。',TYPE)
add('types.collective_bound','Collective 层：已选 Mainloop','这是本例 Mainloop 的实际类型绑定，组合已选 DispatchPolicy、Tile、A/B 元素与 stride、TiledMMA 和 SMEM/TMA 策略。它用于把 Builder 结果追到实现，不代表又增加一层运行时封装。',TYPE)
add('types.major_a','Collective 层：布局到指令主维转换','把 A 的 RowMajor/ColumnMajor 或可识别的布局表示转换为 UMMA::Major。对于 A，RowMajor 对应 K 主维、ColumnMajor 对应 MN 主维，结果进入 MMA/描述符类型配置。',FACTORY)
add('types.major_b','Collective 层：布局到指令主维转换','把 B 的布局转换为 UMMA::Major。B 按 NKL 组织，因此 RowMajor 对应 MN、ColumnMajor 对应 K，不能直接复用 A 的映射结论。',FACTORY)
add('types.auto_selector','Collective 层：1SM/2SM MMA 选择','按显式 KernelSchedule1Sm/2Sm 标签或 Auto 分支选择对应 TiledMMA 工厂。Auto 下静态 Cluster 的 M 方向为偶数且 Tile M 满足条件时选择 2SM；动态 Cluster 不直接假定可用 2SM。',FACTORY)
add('types.two_sm_selector','Collective 层：2SM 指令族选择','在已确定 2SM 的前提下，根据 A/B MMA 数值类型选择 TF32、FP16/BF16、整数或低精度包装，再交给 make_tiled_mma。它仍做类型与维度静态检查，不负责提交实际 MMA。',FACTORY)
FACTORY_ARGS={'thr_layout':'调用时的 Atom 协作/重复布局对象；函数将其补成三维布局，类型中的静态结构在编译期确定，对象值按调用传递。','permutations':'调用时的 MNK 排列描述对象，函数将其补足三维以确定 TiledMMA 的排列类型；不是修改 A/B 数值的排列操作。'}
add('types.factory_op','CuTe Tiled 层：从操作包装构造 MMA','接收底层 MMA 操作的类型样本，先默认构造 MMA_Atom<MMA_Op>，再转发到 Atom 版本的 make_tiled_mma。无名第一个实参的对象值没有被读取，不能把它解释为待计算的矩阵。',FACTORY,dict(FACTORY_ARGS,**{'#0':'无名的 MMA_Op 类型样本实参；用于模板推导，函数体不读取该对象，而是默认构造对应 MMA_Atom。'}))
add('types.factory_atom','CuTe Tiled 层：组合 Atom 与布局','把给定 MMA_Atom、补足后的协作布局和排列类型组合成 TiledMMA，得到可用于线程/CTA 分片的操作对象。这里没有对 A/B/C 执行乘加。',FACTORY,dict(FACTORY_ARGS,mma_atom='调用时传入的 MMA_Atom 对象，带所选 Traits 的状态；工厂把它放入返回的 TiledMMA，而非只复制一个函数名。'))
add('types.tiled_bound','CuTe Tiled 层：已选协作操作','本例 TiledMMA 把 2SM MMA Atom 与协作布局绑定，决定各参与 CTA/线程如何取得 A/B 描述符与累加器分片。它是数据与操作分区规则，不是 Host 的 grid 调度器。',TYPE)
add('types.atom_bound','CuTe Atom 层：已选局部操作','本例 MMA_Atom 是 2SM FP16/BF16 操作的局部调用接口，借助 Traits 规定 A/B 描述符和 TMEM 累加器表示，并在 call() 中转入 mma_unpack。',TYPE)
add('types.atom_impl','CuTe Atom 层：Traits 适配实现','这是 MMA_Atom<MMA_Traits<…>> 的类型特化，不是函数。它继承 Traits，统一暴露形状、操作数类型和 call 接口，实际寄存器/描述符展开由 Traits 的 mma_unpack 决定。',TYPE)
add('types.traits_bound','CuTe Atom 层：指令数据合同','本例 MMA_Traits 定义 2SM 的参与者布局与 A/B/C 逻辑分区：A/B Fragment 使用 SMEM 描述符，C/D 使用 TMEM 表示，并保存指令描述符及是否累加旧 C 的状态。',TYPE)
add('types.wrapper','架构包装层：TCGen05 MMA 类型','SM100_MMA_F16BF16_2x1SM_SS 是具体 2SM 指令包装类型，以共享内存描述符输入 A/B，以 TMEM 地址标识累加器。模板 M/N 是单条指令形状；真正发指令的是它的 fma 静态方法。',TYPE)
TENSOR_P={'D':'调用时可写的输出/累加器 Tensor 视图；本例对应 TMEM 累加器，写入操作由下游 MMA 实施，而不是 Host 数组赋值。','A':'调用时只读的 A 局部操作数 Tensor；本例保存共享内存描述符，指向 MMA 实际从 SMEM 读取的 A 数据。','B':'调用时只读的 B 局部操作数 Tensor；本例同样携带 SMEM 描述符而不是完整 B 数值矩阵。','C':'调用时只读的源累加器 Tensor 视图，表示乘加中的旧累加值；本例为 TMEM 表示，不是 Epilogue 源矩阵 C 的全局内存指针。'}
add('types.atom_call','CuTe Atom 层：局部 MMA 调用','接受 D/A/B/C Tensor，检查局部布局后调用适配的 mma_unpack，把通用 Tensor 操作转换成该 Atom 的硬件操作数合同。虽然接口统一，不同 Atom 的 Fragment 存储表示不必相同。',DEVICE+' constexpr 允许适合的操作被常量求值，但此硬件路径中的地址与数据在设备运行时使用。',TENSOR_P)
add('types.mma_unpack','CuTe Atom 层：硬件操作数展开','该 2SM SS Traits 的友元函数验证 D/C 为 TMEM、A/B 为描述符表示，然后取 A[0]、B[0]、D 的 TMEM 地址和 Traits 状态，调用架构 fma。它不把整个 Tensor 按值搬到寄存器。',DEVICE,dict(TENSOR_P,traits='调用时只读的 MMA_Traits 对象，提供 idesc_ 指令描述与 accumulate_ 控制；accumulate_ 是实际传到指令的状态，不是本次 GEMM 的 alpha/beta。',C='调用时的源累加器 Tensor 类型参与 TMEM 检查；此具体函数不读取 C 的地址或元素，而从 D 取得 TMEM 累加器地址，是否读旧累加器由 traits.accumulate_ 控制。'))
add('types.fma','架构包装层：TCGen05 指令发起','在受支持的设备编译分支中，由 elect_one_sync 选出的线程发起 tcgen05.mma.cta_group::2.kind::f16。它接收已准备好的描述符和 TMEM 地址，不负责 TMA 搬运、等待完成或释放缓冲。',DEVICE+' 主机或未启用对应架构宏的路径进入无效控制路径，不是 CPU GEMM 实现。',{'desc_a':'设备调用时的 64 位 A 共享内存描述符，编码 MMA 取数所需地址/布局信息；不是 A 数值本身，也不是普通 Host 指针。','desc_b':'设备调用时的 64 位 B 共享内存描述符。','tmem_c':'设备调用时的 32 位 TMEM 地址表示，指向 MMA 读写的累加器区域；不是全局内存地址或字节容量。','scaleC':'设备调用时的旧累加器读取控制整数；指令前将其与 0 比较，0 忽略旧累加值、非零启用累加，并非任意浮点 beta。','idescE':'设备调用时的 64 位指令描述容器；此包装将其高 32 位作为指令描述操作数传入 PTX，内容来自已选类型与 Traits 构造。'})
add('dispatch.api.rank4','CuTe 算法层：分片 GEMM 遍历','这是源码 dispatch [4] 的五参重载，D/C 逻辑秩为 3、A/B 为 2。它检查分片兼容性并遍历局部 M/N 位置，把更小的 Tensor 切片递交下一层 gemm；[4] 是分派编号，不是所有参数的 rank。',DEVICE,dict(TENSOR_P,mma='调用时的 MMA_Atom 对象，携带操作类型与 Traits 状态；由它决定下游局部乘加怎样实现。'))
add('dispatch.api.five_rvalue','CuTe 算法层：临时视图转发','这个重载接收临时 D Tensor 视图，函数内具名 D 已是左值，再转发给适用的五参 gemm。它解决视图的 C++ 值类别衔接，没有另行分配输出存储或实现另一套 MMA。',DEVICE,dict(TENSOR_P,mma='调用时传入的 MMA_Atom 对象，原样继续用于下一级重载选择及实际操作。',D='调用时以右值引用接收的输出 Tensor 视图；进入函数后以具名左值 D 转发，视图指向的底层输出存储保持不变。'))

def build():
    if set(R) != set(NODES):
        raise ValueError('Node coverage mismatch: ' + repr(set(NODES) ^ set(R)))
    p_count = t_count = 0
    for identifier, note in R.items():
        node = NODES[identifier]
        params = note.get('parameter_override', {}).get('parameters', node.get('parameters') or [])
        expected = {p.get('name') or '#' + str(i) for i, p in enumerate(params)}
        if expected != set(note['parameter_notes']):
            raise ValueError('Parameter mismatch ' + identifier + ': ' + repr(expected ^ set(note['parameter_notes'])))
        p_count += len(params)
        t_count += sum(len(g.get('parameters', [])) for g in node.get('template_parameters') or [])
    target = ROOT / 'data/interface-notes/host-types.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'nodes': R}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'nodes': len(R), 'parameters': p_count, 'template_parameter_occurrences': t_count,
                      'source_backed_parameter_overrides': 1, 'missing_parameter_explanations': 0}, ensure_ascii=False))


if __name__ == '__main__':
    build()
