B 的零源目的完成接口已在主线重跑。`12032695` 接入为 `01592905`；两侧各 12 项输入 SHA 核对，完整报告除输入路径外相等，三份完整压缩端口数据与说明文件逐字节相同。主线 205 项 R13 测试通过。

每条件保留 16128 组、48384 个目的盒，其中 47712 个非空有效源义务、3558 个 fill 义务（含 672 个零源 B1）。有效源 501600000B 加目的 fill 26882304B，等于完整事务义务 528482304B。零源 B1 没有 valid 分支，但仍有 submit、8192B fill、destination_complete 与 full 合流。

按有效源量缩减 full 预算会在 1868 组少计 26882304B；直接丢弃零源目的盒会在 672 组少计 5505024B。valid/fill 是逻辑完成义务，不代表独立硬件队列。全部完成值为 None，没有把零源或未识别服务当作零时间。

抹去实际地址与 stride 后，保留完整工作、ring、完成拓扑和源身份等价类，三种行距投影相同。因此只读取这份投影、使用同参数与初态的模型无法区分三种行距；地址相关工作、共享状态和仲裁仍未识别。主线后续组合单独位于 `../main-input-composition-v1`，本目录保留交付原貌。

```sh
python microbench/gh200_resource_campaign/access_rules/analyze_r10_boundary_completion.py \
  --run results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4 \
  --binding results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/B-20261011-boundary-binding-v1 \
  --output NEW_OUTPUT_DIRECTORY
```
