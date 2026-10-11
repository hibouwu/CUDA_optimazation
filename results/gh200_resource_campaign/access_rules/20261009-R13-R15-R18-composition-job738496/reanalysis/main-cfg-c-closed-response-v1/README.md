# 主集成：cfg_c闭合图影响范围独立复核

接入B59b24b4为65f792bb。重建26份物理绑定、12份源模板及完整rank图，压缩图与B及主线规范图逐字节相同；349输入SHA全部匹配。主报告除两份模块工作树路径外逐字段相同，两模块SHA一致。30项cfg_a/c依赖/供给与目的后程回归测试通过。

另用拓扑序上的最小边数传播独立核对B的BFS结果：126查询的完整可达集合数量、两rank最早full、FIRST/MAIN与完成端点均一致；100条最短路径逐边合法且长度最小，26查询无后续remote full。5种图维度包含短K合成边界；只检验依赖，不给节点赋时间价。

固定入口、其他边延迟和资源组织，只改变指定rank1事件时，S=4、组g、q=g%Ktiles：

| 变化事件 | 最早rank0 full |
|---|---|
| A_source_ready或acquire_return | g |
| B0_issue | g+1，经下一组producer与A多播 |
| B0_source_ready，q=0 | g+4 |
| B0_source_ready，q>0 | g+3，经commit(g)→safe_release(g−1)→empty(g−1) |
| empty emit或arrival(g) | g+4 |

超出调用范围则无后续full可达。合成Ktiles2/T2最后组g3的本地B完成不达伙伴MAIN/D_visible，却仍达伙伴producer_tail、source_safe_exit和cuda_complete；不能由输出未受影响推出整个cluster退出也未受影响。

可达不等于一定产生正延迟、具有某个量级或激活关键路径。实际干预还可能改变控制流或共享资源，不能自动套用该固定图排他界。原150进程数值证据仅通过来源SHA复用，没有再次声称数值执行。没有新GPU、价格、观察资格或留出评分；V09仍仅静态绑定。
