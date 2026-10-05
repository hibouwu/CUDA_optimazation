# TMEM 生命周期：模型、分图与接入记录

本轮把Dense非overlap 2SM路径中的TMEM生命周期从事件清单补成独立协议模型。固定源码与编译配置不变；未修改已有`contracts.json`或上游代码。

## 模型如何闭合

源码模型分别记录每CTA的MMA/epilogue角色、分配结果指针、NamedBarrier、deallocation barrier和累加器排空前提。四CTA属于两对，不合成一个四CTA分配事件。160个分配发布参与者、每槽256个累加器释放参与者、32个最终握手参与者分别保存。

最终v3原始模型SHA为`3cd91a827e83be05fd397cebbfdf1d76775313c93917a36fa6e0bfba223ab633`。它包含90个独立事件、252条量词化偏序、18个资源事实集合、64条推导规则和2条配对参与判定规则；这些是该配置的模型记录，不是全库API分母或GPU实测次数。

分配访问的同址顺序、协作粒度和许可边界使用固定版本的[PTX ISA 9.0](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html)第9.7.16.5–7节，不用源码中的“non-blocking”注释覆盖规范，也不把warp内会合扩展为跨CTA完成全序。

独立审查推动了三轮具体修复：

- D01：分配结果写与随后的同址读取观察建立了明确约束。没有把weak store效果直接等同于allocate函数返回。
- D02：本侧free效果只表示本侧dealloc指令已发出，不能单独证明配对参与完整。两CTA各32个进入实例、两侧操作记录及源码前提由独立`all_of`规则共同检查。
- Q01：同线程程序序、全部贡献到每个等待者、单次warp操作之间的关系使用不同量词。状态是从历史事实求最小不动点的判定，不消费旧事实，不反向添加时间边。
- D03：本侧指令发出必须先于本侧各线程从free返回；没有增添peer进入与本侧返回之间的假顺序。

v1/v2原始数据和反例保存在draft的`history/`，独立审查记录在[dense-tmem-protocol-independent-review.md](dense-tmem-protocol-independent-review.md)。模型测试与独立扩展检查是有界语义核对；它们不证明任意动态generation、调度公平性或实际GPU执行正确。

## API身份与实际实现路径

`stage_dense_tmem_protocol.py`只补API/调用身份和展示分组，不修改事件、偏序或状态推导。它复用已有Dense节点，并补入NamedBarrier public→private、PipelineUmmaAsync→PipelineAsync→producer_acquire等路径；inline PTX使用硬件事件节点，不伪造成C++函数。

接入时暴露了全局账本在`sm90_pipeline.hpp`丢失`cutlass`外层namespace的同类问题。三个选中方法通过原文字节、namespace48–1388、class1014–1240、public/private及默认参数核对，使用独立manual身份补充。错误global实体ID只留在纠错依据中，不重新命名后当成已修复的全局实体。

## 分图为何采用独立连接点

最初把每次调用的enter/effect/return都画成大节点，单张图宽达3千余像素。仅按固定边数拆图仍会产生许多无关横向分量，且缺少可连续阅读的调用路径。

当前做法是每个API事件保留一个独立框，其中每个锚点都有自己的连接点。跨API约束逐条连线；同一API内部约束按独立ID列在框内，完整条件保留在链接记录。不同CTA的同一源码调用仍有不同事件身份。分图按CTA局部路径、跨CTA约束和连通分量组织；NamedBarrier历史事实的两条推导分支分别展示，不解释为两种物理到达全序。

同一协议数据生成可编辑PlantUML和SVG。事件SVG使用Graphviz的显式table-row ports，DOT布局输入也随产物保存；这样能避开自成员连线穿过文字的问题。SVG统一为pixel/viewBox坐标，不把Graphviz的pt数值误当实际像素。状态事实图仍用PlantUML渲染。两种表示均按同一组约束ID对账；人工语义修改只进入数据，不直接改生成SVG。

这一轮生成86个协议分图，最大宽度为914像素，API正文保持14像素、关系辅助文字11像素。长图按原尺寸滚动，不按整图高度缩小文字。宽度检查只是排版检查，不替代实际1366×768/1920×1080的浏览器验收。

[独立静态展示审查](dense-tmem-static-display-review.md)检查了最大宽度的分配发布图、跨CTA释放握手图和NamedBarrier事实推导图。初版源线贴边不易区分，现用蓝色源锚点和目标箭头标明方向；7条跨事件连线的实际SVG端点均与对应行框核对。三张抽样图未见截字、叠字或不同API/CTA合并，此结论不扩展为全部86张图都已逐张目视或真实浏览器验收通过。

## 离线核对与继续工作

图集提供独立事件页，显示真实API完整声明、调用位置、参与者、predicate、实例域、所有明示约束和关联分图。协议页把量词化时序、历史事实推导与配对参与必要条件分别展示；状态箭头不再被描述成物理释放时间线。

在atlas根目录重生成：

```bash
.venv/bin/python scripts/stage_dense_tmem_protocol.py
.venv/bin/python scripts/build_atlas.py --skip-sources
.venv/bin/python scripts/audit_atlas.py
.venv/bin/python -m unittest discover -s data/module-drafts/dense_protocols -p 'test_*.py' -v
node tests/test_atlas_runtime.js
```

当前仍需实际浏览器使用验收，以及输入SMEM、累加器逐generation、输出SMEM等其他资源的专用协议图。此工作包不能作为整个Sync/Resource模块或824文件已完成的证明。
