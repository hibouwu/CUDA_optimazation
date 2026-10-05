# Dense TMEM协议：三张静态图独立展示审查

## 本检查点结论

**所抽三张图的静态展示复核通过；原P01“源线贴框导致源锚点难辨”在蓝点版关闭。** 不同API、不同CTA保持独立，指定的跨事件箭头实际落在正确enter/effect/return行，文字未发现重叠或截断。同一API内部约束按独立order ID列在框内，没有合并不同API或不同CTA。

这是对本地SVG经白底栅格化后的原尺寸目视及数据/端口核对，不是实际浏览器检查。没有打开HTML、HTTP预览或绕过策略；没有验证点击、缩放、滚动、键盘焦点、1366/1920窗口或其余83张协议分图的交互。最大宽度和renderer名称都没有被当作通过依据。

本审查只新增本文，没有修改数据、renderer或SVG。

## 抽样、版本与方法

从`data/atlas.json`按protocol view ID定位，原始抽样版atlas SHA256为`b4a2b060a1ed7bd9d8098ccd4d558278c88df2c91acb10afffd4bfe37f299e31`；蓝点版复核的atlas SHA为 **`3ca928958a229afb4f4158afd803af20046514557626b6af6932856aae054b2f`**。两个版本均只抽以下三图，不将检查点推广到全部86图。

| view ID | 原版尺寸、SHA256 | 最终蓝点版尺寸、SHA256 |
| --- | --- | --- |
| `tmem.view.allocation_pair0.part6` | 914×1401；`018b32322dfc0af6cc648af8f6fec1fef963764ab09eeec57d3237b4c9be87f6` | **914×1419**；`5eccf871a04f375805e202c5bcbf8da8899d2b169dc1893a45402f5228d1cd96` |
| `tmem.view.release_pair0.part13` | 860×1673；`dfca69981f624f81a2640c5cdfc4f4f2e337e52c51f761a4caf891f68a89f910` | **860×1691**；`a581eb4535840f3aac551d4d2133a1138c432123c2db30f500e27cd83023180b` |
| `tmem.view.state.tmem.cta0.allocation_barrier.mma_fact` | 734×678；`7e6ff550d355dff8f1eb6017ca0f335caaf1548789a4d9f0f3f7b572fe4c7c6c` | **734×678**；SHA同原版，state图未受边样式修订影响 |

前两张对应同ID的DOT和可编辑PlantUML，实际SVG由明确row port的DOT生成；状态图实际SVG由PlantUML生成。本审查读取了实际SVG内容，而不是因为存在`graphviz_explicit_row_ports`标志就假定端口正确。

每张分别运行`rsvg-convert -b white`，随后用`view_image(detail="original")`查看PNG。另解析SVG节点中各锚点行的polygon范围，核对每个跨事件order的蓝色source ellipse中心及目标arrowhead尖端均位于预期行；order超链接中的ID与atlas记录匹配。完整条件留在记录入口，本次没有声称实际点击验证。

下文K指固定snapshot的`include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`；`barrier.h`指`include/cutlass/arch/barrier.h`。数字均为物理源码行，不是SVG坐标。

## 分配发布图：API独立、32/128贡献与读观察分开

part6中有四个独立事件框：CTA0 MMA的NamedBarrier::arrive、CTA0 epilogue的arrive_and_wait、CTA0 MMA的allocate，以及epilogue读取shared base pointer。相同CTA的不同API没有合并，MMA和epilogue也没有合并成一条角色线。

三条跨事件关系逐项核对如下：

| order ID | 实际源port | 实际目标port | 语义与源码对应 |
| --- | --- | --- | --- |
| `tmem.order.011` | cta0.alloc_arrive.effect | cta0.alloc_wait.return | NamedBarrier收齐32个MMA与128个epilogue贡献后才能返回；K729与870的独立API位置保留 |
| `tmem.order.009` | cta0.alloc_wait.return | cta0.epi_read_base.enter | 同一epilogue线程先完成870，再进入871读；不是整族到下一事件的额外屏障 |
| `tmem.order.242` | cta0.allocate.effect | cta0.epi_read_base.return | K727分配结果写与871同址读观察之间的已核内存/发布约束；没有连到allocate.return |

蓝点与箭头位置全部处于表中指定行。图中011的“32 MMA arrivals plus the128 epilogue participants”可读；allocate.effect明确标“本侧warp操作”，普通enter/return标“每线程实例”，没有把32次线程进入画成32次独立allocation。全部12个order ID均可在SVG文本找到：3条跨事件线，其余9条在相应API框内逐ID列出，没有把内部约束静默删去。

物理源码位置重新对应固定snapshot的`sm100_gemm_tma_warpspecialized.hpp:727–731,868–872`；NamedBarrier对应`barrier.h:210/212/285/287`与`222/224/305/308`。图中enter/effect/return是协议锚点，不是新增C++API。

## 跨CTA握手图：两侧和两个wait实例可区分

part13中CTA1 peer的793 arrive、CTA0 leader的794 wait、CTA0 leader的795 arrive、CTA1 peer的794 wait分别显示。虽然两wait同名同源码行，CTA0/CTA1身份没有合并。两个有效arrive节点均能读到`predicate=true; arrival=32`；节点名保留`!leader`或`leader`的不同原始谓词。

| order ID | 实际源port | 实际目标port |
| --- | --- | --- |
| `tmem.order.127` | cta1.arrive793.effect | cta0.wait794.return |
| `tmem.order.020` | cta0.wait794.return | cta0.arrive795.enter |
| `tmem.order.128` | cta0.arrive795.effect | cta1.wait794.return |
| `tmem.order.050` | cta1.arrive793.return | cta1.wait794.enter |

四条关系的source圆点和目标尖端分别落到对应行，127/128没有误连成enter→enter；050没有被误画成peer已等待后leader才可接收arrival。两条barrier join标“收齐源贡献后，每个目标才能通过”，两条程序序标“同一线程逐实例”，32次到达条件在图中可辨。全部10个order ID都保留：4条跨事件线、6条API内部独立ID约束。

源码复核对应K793–795，以及ClusterBarrier成员`barrier.h:383/384`到static486的predicated remote arrive；`489–494`只在pred真时映射peer并到达，成员wait371/372转static410/420等待本CTA地址。该图不是两侧free到达或返回的全局时间线。

## CTA0 NamedBarrier状态事实图

mma_fact保留三个transition ID：010、013、014，分别以MMA arrive.effect的全部32线程、epilogue arrive_and_wait.effect的全部128线程、epilogue wait.return的全部128线程作为历史事实推导条件。图中使用“全部相关实例已记录”表达all_instances；精确cardinality与来源保留在对应记录，标题显示本CTA160线程NamedBarrier。

四个事实框、各自说明和三条推导箭头均清晰，无截字或叠字。`已记录MMA完整贡献（不判断谁先）`及底部“状态是非互斥的历史事实判定；箭头表示推导规则，不添加API时间顺序”的说明可读，避免把所选MMA事实推导路径误当唯一物理执行顺序。这个state图没有将MMA/epilogue角色合并成一个API调用，也没有将barrier事实误当TMEM deallocation完成。

## P01历史观察及蓝点版关闭

原版目视发现：011、242、127、128等线从source行离开后，沿节点右边框走一段才转向目标；边线颜色接近框线，读者容易误以为从框底而不是effect/return行发出。这是源口可辨性风险，不是发现数据端口写错。本观察已在修复前报告，旧版尺寸、SHA和PNG均保留。

新版将跨事件线改为蓝色，并在源端加小圆点、目标端保留箭头；图例明确“蓝色小点为源锚点，箭头指向目标锚点”。实际原尺寸目视确认：即使线继续贴着右边框，source点也能辨认出effect与return所在行；CTA1 arrive793的effect与return两枚圆点可以分别追踪到不同约束。SVG坐标核对也确认7条跨事件线的圆点/尖端位于指定行。**P01在最终两个蓝点SVG版本关闭。**

新增图例使两个event图高度各增加18px，宽度未变；未发现此次样式修订带来新的文字重叠、裁切或路由错位。状态图SHA不变，再次目视仍正常。本文不另设或扩展更多排版标准，本检查点到此冻结。

本地白底PNG保存于`/tmp/dense-static-svg-review-4CyvcE/`：原版`tmem-allocation-part6.png`、`tmem-release-part13.png`、`tmem-state-mma.png`；复核版为相同名加`-blue`。这些是本地静态查看产物，不是浏览器截图，也不作为GPU执行证据。
