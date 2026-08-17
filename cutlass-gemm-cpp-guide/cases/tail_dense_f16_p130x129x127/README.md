# `tail_dense_f16_p130x129x127`

逻辑 problem 为 `130×129×127`，物理 strides 则补齐为：

```text
lda = 128
ldb = 136
ldd = 132
```

这样可以把 logical tail/TMA OOB 语义与 alignment requirement 分开。测试执行完整输出和
output canary；单独的 misaligned-pointer 负例不能以 crash 代替 `can_implement` 拒绝。

```bash
cmake --build --preset sm110a-gpu --target tail_dense_f16_p130x129x127
./build-sm110a-gpu/tail_dense_f16_p130x129x127 --verify --seed 20260817
```
