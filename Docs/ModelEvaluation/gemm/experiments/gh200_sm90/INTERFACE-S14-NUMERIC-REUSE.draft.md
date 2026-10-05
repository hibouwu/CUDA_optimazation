# S14：跨分配复用已签短数值证据

状态：有限 A 候选，待独立审查；不注册或执行 GPU。补充 [原 session 约定](INTERFACE-S14-SESSION.draft.md)，按 [GOAL.md](GOAL.md) 复用适用的依赖证据，不新增运行框架。原作业 `731626` 的 24 配置、72 次 `capture=false` 完整数值审查、原 job/主机/UUID、完整包和签名永久保留；超出在线等待窗口不使正确证据失效，但它不能授予新分配的正式性能资格。

## 新分配的准入顺序

1. 先验证原 B3 的独立签名、完整事实及 archive/index/closure 身份，在其明确的原 repo namespace 中读取，不改写旧 allocation。新建运行目录并记录新 job、主机、UUID、可见设备和工具链；这些来源字段允许改变，不伪装为原分配。
2. 在任何 pilot、校准或运输 kernel 启动前取得新设备全部现有属性。除了 UUID，设备名称、CC、SM 数、driver/runtime version、显存/L2、寄存器和 SMEM 容量等现有完整字段须逐项、严格类型相同；environment 的 driver、compiler、execution UID、工具内容身份以及真实 external dependencies/shared libraries 与原合格基线一致。设备查询与构建操作不算新的短数值验证。
3. 核源码、实际编译依赖及 18 个目标的全部 PC/128-bit 机器码和静态资源。五 CPP 输入原则上使用原 SHA；唯一允许的例外是经独立 source-B 的正式 host 增加下述只读资源查询分支。另四个输入必须逐字相同，运输、同步、布局、缓冲、计时、参考及原服务分支不变。新增 host 的 SHA、差分和实际 binary 独立记录，不能声称五个输入仍全部原 SHA。其他 host 改动不自动属于此例外。
4. 使用将执行正式服务的同一新 binary，先查询全部 24 个 case 的实际函数属性、动态 SMEM 设置、occupancy API 和 grid。每点逐项核 `kernel_symbol`、registers/static/dynamic SMEM/local、occupancy、blocks/threads，以及 payload、32-slot allocation 和完成/正式角色条件，与原 B3 及冻结规则一致。API 上限仍不宣称实际同时驻留。24 点全部成立才允许复用原短 B3，不能先运行校准才发现资源条件改变。
5. 在新分配执行自己的既有 preflight：pilot、长度校准、预热、计量和选定长度完整正确性检查。保全所有输出与实际条件，未实施者按既有可执行审查规则签完整 B；通过后才用新冻结长度正式采样。正式执行精确绑定这个新分配的设备/二进制/资源/校准事实，而不是原 numeric 的 job。

原 B3 仍是原 GPU 上完整有限矩阵的数值证明。新 preflight 的完整检查是新 GPU 实际执行证据；报告分别列明两者，不声称新 GPU 又执行了 72 次旧 profile。所有正式十次独立进程、随机顺序、8–30 次预热、CV 判据、最多三批、计时边界和超时规则保持。旧性能、pilot 时间及 N、时钟/热状态、SMID/时间戳、样本/CV、实际驻留或权限观察不继承，不混并不同分配的性能样本。

## 最小只读资源入口

现有 formal host 的 `device` 入口仅返回设备属性，cuobjdump 也不能给出实际 occupancy API。允许新增有限 `resources` CLI：从固定 24-case 表取真实函数，沿原动态 SMEM 设置及 `cudaFuncGetAttributes`、`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 和原 grid 公式返回资源。该分支不 launch kernel、不复制 payload、不记录服务 event、不校准。API 调用失败是查询错误，不包装成硬件不支持或沿用旧值。仅此 host 分支例外必须先独审及实际编译；首个代表入口 CPU/目标查询贯通后再展开有限 24 点，不扩大矩阵。

## 何时必须补短验证

| 变化或失败 | 处理 |
|---|---|
| 仅 job、主机、UUID、目录或可见序号改变，其余全部核对成立 | 复用原完整 B3；新分配自己的 preflight/完整 B/正式数据仍必需 |
| 设备能力、driver/runtime、工具或 external dependencies/libraries 改变 | 停止自动复用，先独审影响范围；受影响 case 重新取得完整短输出，影响无法限定时覆盖全部 24 点 |
| device 目标编码、运输/同步/布局/缓冲/计时/实际参与者或 grid/资源条件改变 | 重证明受影响编译、短数值和计量；不得用仅最终错误数代替完整值 |
| 仅 loader 元数据、变量 shadow、参考或离线分析修复，执行语义/目标未变 | 先 CPU 接口与完整原证据重放；只有已有数据不足的部分补 GPU，不因修复门禁重跑合格点 |
| archive/facts/签名损坏、缺失或未通过来源核对 | 拒绝复用；先从保全副本恢复核验，无法恢复的受影响点补完整短验证 |
| 新 preflight 数值、计量、清理或实际条件失败 | 保全失败，不进入正式采样；分析后按影响范围补验证，不能反复换卡清零失败 |

实现只需在既有 finite admission 中区分 `numeric_source_allocation` 与 `current_formal_allocation`，保存完整等价比较及原 B3 引用；不能删除 UUID/job 字段静默放行，也不创建另一套 session。原 capture=true/source-read release 历史覆盖继续保留原设备范围，本补充不导出其新设备参数。
