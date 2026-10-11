# 实机包装格式修复的限定兼容性复审

新decoder SHA `f1b59f0e…`，新tests SHA `71d46b7e…`，完整身份见`review.json`。上一层的`review.json`、`snapshot/`和`fixed-snapshot/`保留原样，仍只适用于各自冻结的CPU fixture格式。设备source manifest `b57bbb98…`和两个binary未因这次修改改变。

当前host实际输出cluster为二维`[1,1]`；旧CPU fixture误写三维`[1,1,1]`。此外，固定命令设置了input-map，实机记录还包含`input_zero_storage`。新完整decoder严格要求setup/call/input_zero_storage/check四包，区域检查的`checked`与`nonzero`均须为原生整数0；没有忽略额外事件，也没有接受任意cluster。

对完整`decode_trace`函数，只把旧cluster常量改为正确的二维表示后，旧／新AST完全相同。因此16次关闭前序、H2与旧H16的必要顺序、原生uint64差值、关闭时未观测及组件ns为空的语义均未改变。另实际重放新metadata下8个stage4/6×off/on×CLOCK原点0或2⁶³+17的边界，结果精确一致；旧三维cluster及非固定`[1,2]`均被拒绝。主线的7项CPU测试独立运行通过，执行前后源码SHA一致。

原首目标740259／stage4/off的回收包固定SHA为`313d844177db6d8757eb5315e122a3c1d6b0c4dfccdce247daf8c829c3053e84`，raw为`0b925c73d66aacaec2192701d3004425b9f8d72a7e0dea624c118938b852dd00`。包内records与raw逐字等于本地回收文件。新decoder对该实际记录得到C16−C1=11500 cycle、16次已完成前序、4096个独立精确数值检查；关闭时Q、等待外周期和组件ns均为空。原采集receipt的returncode=1保持不变，这项离线复核不将失败批次改为成功。

这里只复核这一原始样本和包装兼容性，未重采目标、评分剩余三个目标或取得扰动／空对照资格。原生周期、collector开销、跨stage不同前序kernel及时间换算不支持的边界均保留。

`snapshot/`保存新版本源码及依赖；`stock-tests.json`保存在实际主工作树运行七项测试的stdout/stderr，测试前后已核源码身份。冻结快照中的`test_actual_archived_off_record_format`使用主工作树相对位置定位原归档；勿直接在本快照目录运行该路径依赖测试。`review.json`保留语法树检查、8个大整数／开关结果与原始归档身份。

核对固定快照、既有测试记录和原raw并写新验证文件：

```sh
python verify.py --output verification-new.json
```
