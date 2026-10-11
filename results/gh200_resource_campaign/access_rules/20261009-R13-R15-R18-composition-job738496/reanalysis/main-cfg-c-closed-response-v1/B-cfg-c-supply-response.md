# cfg_c闭合供给图：源事件影响的最早边界

26份物理绑定与12份源模板逐字节保持；闭合图与主线12份规范gzip逐字节相同。
以下只改变所列事件，固定入口/其它边成本，不加入新共享资源；可达不等于必然变慢。

| Ktiles/T | 源事件 | g | 最早remote full | remote当前FIRST/MAIN可达 |
|---|---|---:|---:|---|
| 16/2 | B0_source_ready | 0 | 4 | False/True |
| 16/2 | B0_issue | 0 | 1 | False/True |
| 16/2 | B0_source_ready | 1 | 4 | False/True |
| 16/2 | B0_issue | 1 | 2 | False/True |
| 16/2 | B0_source_ready | 3 | 6 | False/True |
| 16/2 | B0_issue | 3 | 4 | False/True |
| 16/2 | B0_source_ready | 15 | 18 | False/False |
| 16/2 | B0_issue | 15 | 16 | False/False |
| 16/2 | B0_source_ready | 16 | 20 | False/True |
| 16/2 | B0_issue | 16 | 17 | False/True |

B完成迟延通过release/empty影响伙伴：prologue为g+4，普通位置为g+3；不能一律写成四组以后。
B发出迟延另可通过下一producer工作及A多播，在g+1影响伙伴。A完成迟延则本group直接影响两rank。
短K无后续full时，伙伴D不一定受影响，producer尾部/全局退出仍可能等待远端empty；两种完成边界不能合并。
源码图、逐事件最短见证路径及不可达标记保存在JSON；未赋服务价、未锚定实际入口，也未增加观察资格。
