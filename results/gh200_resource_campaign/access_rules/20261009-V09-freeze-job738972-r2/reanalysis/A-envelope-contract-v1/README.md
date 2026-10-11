# R09：冻结r2包络的求根契约

对批准的V09 r2全部30条件，按存储十进制参数的精确有理数解释，包络E(f)在`0<f≤a=1.98`上连续、为正、随f非增，且f趋零时发散。源时钟与该包络联立的正根因此唯一。该结果只补齐既有求根器的数学前提，不能修复其时间误差，也不构成新的时间预测资格。

## 固定对象

批准预测SHA为`f4df39e00cf1d2ab61947c911e699bd5bea4e533609d93f726de43077515d81e`，清单SHA为`09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab`。读取清单绑定的`freeze-setup.json`，不把未列入批准清单的同名开发setup替代它。源码由批准清单中的`source_hashes.json`间接绑定，calibration/cases/predictions另有直接绑定。

完整保留30条件：29 predicted和1 unsupported。cfg_c G2的unsupported不变；该条件仍有数值递推函数，检查其数学性质不授予其参数支持。只用原GPU-43269fbc的静态setup及原校准；没有改写UUID或跨卡绕过。

## 分段构造与独立复验

令`x=1/f≥50/99`，单位GHz⁻¹。每个静态请求的供给窗口为`intercept + Kt×max(compute×x, service)`，因此是连续分段仿射函数。输出间隔由固定r2参数和输出有效比例决定；cooperative/pingpong递推中的max与重叠规则可逐段精确计算。CTA输入按相同公式分组，再对全部CTA取最大值；源工作和所有价格不重新定义或拟合。

构造器的pingpong写完成表达式如下。令e为permit，下一mainloop为[a,b]，E为单独执行工作量，r为重叠时的进展率：

```text
before = max(a-e, 0)
remaining = max(E-before, 0)
overlap = max(b-max(e,a), 0)
done = e + E + (1-r) × min(overlap, remaining/r)
```

独立`verify.py`不导入构造器或GEMM模型。它使用原代码的顺序分支：先消耗a之前的工作，再以r消耗重叠工作，最后以1完成剩余工作。对于每个导出区间，重放得到仿射表达式，并在区间两端检查所有分支不等式；无界尾段另检查斜率方向。因此不是在每段只检查一个采样点。最大CTA的分支也逐段检查。

验证统计：30条件、3516个CTA实例、50个构造区间、141个CTA分组区间、1113条分支不等式、45个最终包络分段。全部分段连续且斜率非负，左端值为正，无界尾段斜率严格为正；最小尾斜率为2048ns/(GHz⁻¹)。错误包络常数、删除分支断点、错误compute floor三个内存损坏证书均被拒绝。

导出静态输入的来源由14项SHA绑定，并通过归档原`predict_components`入口作360次固定f数值核对。固定f只使用临时时钟fixture令其直接返回指定f，以读取E(f)，没有使用目标观测W/f或生成新的时钟候选。最大相对差为1.9984×10⁻¹⁵。这证明数值吻合到所测精度；精确证书针对导出的有理数模型，未形式验证静态提取器或所有IEEE舍入路径。

## 为什么正根唯一

令E_us=E_ns/1000，原源时钟为`F(W)=a−A/W−B×g(W)/W`，A/B非负。这里

`g(W)=∫₀¹ (1−exp(−tW/τ)) dt`。

g连续、非负、凹且g(0)=0，故g(W)/W非增，F(W)非减。由于E(f)非增，`H(f)=F(E_us(f))−f`严格递减。E在f趋零时发散，故`H(0+)=a>0`；非负价格使`H(a)≤0`。连续性给出唯一正根。零活动混合是两个非负价格的凸组合，不改变这一性质。

这排除了用“另一正根”解释固定r2公式误差的可能性；没有排除模型输入、价格、历史或完成边界本身失配。1.98是冻结模型的a，不是本轮重新测得的硬件上限。

`binding.py`只读核对当前主线`clock_model.py`：求根函数AST与归档完全相同，source_frequency仅归一化`clock.duration_term`导入限定名后相同，duration_term函数AST也相同。该绑定不证明后续调用者的供给/输出包络与冻结版本相同，更不把这30条件的证书推广到所有形状或新状态模型。

## 复放

```sh
r09_envelope_output=$(mktemp -d /tmp/r09-envelope-XXXXXX)
cp /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/A-envelope-contract-v1/{analyze,verify,binding}.py "$r09_envelope_output/"
python "$r09_envelope_output/analyze.py" --frozen /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2 --output "$r09_envelope_output"
python "$r09_envelope_output/verify.py"
python "$r09_envelope_output/binding.py"
```

直接复验归档：

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/A-envelope-contract-v1/verify.py
```

`envelope.json.gz`含全部分段、原消费的有理数输入、CTA分组索引、状态和数值核对；`verification.json`含独立检查结果，`current-clock-binding.json`含当前函数绑定。没有GPU操作、公共代码或参数修改、时间模型采纳、新留出或跨卡资格。
