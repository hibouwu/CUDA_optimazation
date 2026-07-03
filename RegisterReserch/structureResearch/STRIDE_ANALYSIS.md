# Stride 周期性分析与外推

**日期**：2026-07-03  
**范围**：stride 1-16 实测数据 + stride 1-32 假设预测  
**图表文件**：[assets/stride_periodicity_analysis.png](../assets/stride_periodicity_analysis.png)

---

## 📊 可视化说明

图表包含两个子图：

### 左图：测量数据与假设曲线对比
- **红色点**（stride 1-16）：实测的 LOP3 周期数（中值）
- **蓝/绿/橙/紫虚线**（stride 1-32）：mod 2/4/8/16 假设的预测延迟
- **灰色虚线**：标记实测范围限制（stride 16.5）

**关键观察**：
```
实测数据（stride 1-16）：
  ✓ mod 2 预测线与红点完美重合
  ✗ mod 4/8/16 预测线与红点明显分离
```

### 右图：冲突预测映射（热力图）
- **行**：假设的 bank 组织（mod 2, 4, 8, 16）
- **列**：stride 值（1-32）
- **红色**：预测有冲突
- **蓝色**：预测无冲突

**关键观察**：
```
mod 2 行（顶部）：严格的红-蓝交替模式
mod 4 行：红色块更宽（4 个 stride 一组）
mod 8/16 行：红色更稀疏
```

---

## 📈 数据总结

### 实测数据（stride 1-16，base R4）

| Stride | 周期 (c/op) | 碰撞? | 奇偶 |
|--------|-----------|-------|------|
| 1      | 2.086029  | ✓     | 奇   |
| 2      | 3.070404  | ⚡    | 偶   |
| 3      | 2.086029  | ✓     | 奇   |
| 4      | 3.070404  | ⚡    | 偶   |
| ...    | ...       | ...   | ...  |
| 15     | 2.086030  | ✓     | 奇   |
| 16     | 3.070405  | ⚡    | 偶   |

### 模式统计

```
奇数 stride（1,3,5,...,15）：
  - 数量：8 个
  - 都显示快速延迟（2.086 c/op）
  - 碰撞率：0/8 = 0.0%

偶数 stride（2,4,6,...,16）：
  - 数量：8 个
  - 都显示慢速延迟（3.070 c/op）
  - 碰撞率：8/8 = 100.0%
```

### 假设拟合度对比

| 假设 | 准确率 | 说明 |
|------|--------|------|
| **mod 2** | **100%** | ✓ 完美匹配 16/16 |
| mod 4 | 75% | 部分匹配（8/16 miss strides 2,6,10,14） |
| mod 8 | 62.5% | 严重不符 |
| mod 16 | 56.2% | 几乎等于随机猜测 |

---

## 🔮 外推到 stride 1-32 的预测

基于不同假设的预期行为：

### 如果是 mod 2（2-bank）
```
stride 1-32：严格交替
  奇数：2.086 c/op（所有奇数）
  偶数：3.070 c/op（所有偶数）
优点：已完美解释 stride 1-16
```

### 如果是 mod 4（4-bank）
```
stride 1-32：周期为 4
  stride 1-3：      2.086 c/op
  stride 4-7：      混合（4 碰撞）
  stride 8-11：     混合（8 碰撞）
  stride 12-15：    混合（12 碰撞）
  stride 16-19：    混合（16 碰撞）
  stride 20-23：    混合（20 碰撞）
  stride 24-27：    混合（24 碰撞）
  stride 28-31：    混合（28 碰撞）
  stride 32：       碰撞

问题：无法解释 stride 2,6,10,14 的高延迟
```

### 如果是 mod 8（8-bank）
```
stride 1-32：周期为 8
  仅 stride 8,16,24,32 显示碰撞
  其他 28 个 stride 应该是 2.086
  
问题：观测到的 stride 2,4,6,10,12,14 也有碰撞，与此不符
```

---

## 🎯 结论与建议

### 当前证据强度

| 证据 | 强度 | 支持的结论 |
|------|------|-----------|
| mod 2 完美拟合 | ⭐⭐⭐⭐⭐ | 2-bank 最可能 |
| 跨 base 一致性 | ⭐⭐⭐⭐⭐ | 全局统一组织 |
| 奇偶完全分离 | ⭐⭐⭐⭐⭐ | 清晰的二元模式 |
| 统计显著性 | ⭐⭐⭐⭐⭐ | 不是随机 |

### 进一步验证的实验

**高优先级**（有助于完全排除多 bank 假设）：

1. **扩展 stride 到 1-64**
   ```bash
   # 修改 MAX_STRIDE = 64 的技巧：
   # - 使用不同的 base registers（R0-R3 而不是 R4-R7）
   # - 或使用更少的源加载，通过重用实现更大 stride
   ```
   - **目的**：检测 mod 4/8 周期性
   - **成功标志**：保持 mod 2 交替，或发现其他周期性？

2. **测试多种指令**（IADD, FADD）
   ```bash
   # 使用已生成的 sass_imad_template.cu / sass_fma_template.cu
   ```
   - **目的**：验证 bank 组织是否一致
   - **成功标志**：所有指令显示相同的 mod 2 模式

3. **完整寄存器空间**（R0-R63）
   - **目的**：检查 mod 2 分布是否全局均匀
   - **成功标志**：所有寄存器都遵循 `register_id % 2` 映射

**中等优先级**（直接测量验证）：

4. **NCU bank conflict 计数器**
   - 采集 `sm__pipe_fu_core_access_conflict_stall` 等指标
   - 对比 stride 2、4、8 的冲突次数

---

## 📁 关键文件

- [results/bank_scan/results.csv](../results/bank_scan/results.csv) - 完整的测量数据
- [scripts/plot_stride_periodicity.py](../scripts/plot_stride_periodicity.py) - 生成此图表的脚本
- [assets/stride_periodicity_analysis.png](../assets/stride_periodicity_analysis.png) - 本图表
- [BANK_ANALYSIS.md](../BANK_ANALYSIS.md) - 详细的多角度分析

---

## 🔄 如何重现结果

```bash
cd RegisterReserch/structureResearch

# 1. 构建
bash scripts/build.sh

# 2. 运行 bank-stride 扫描
bash scripts/run_bank_scan.sh

# 3. 生成周期性分析图表
python3 scripts/plot_stride_periodicity.py

# 4. 也可运行多角度分析
python3 scripts/analyze_bank_hypotheses.py
python3 scripts/analyze_bank_evidence.py
```
