# Mamba2 Alpha矩阵验证

将Mamba2的SSM（状态空间模型）重构为attention矩阵形式，实现完全等价的计算。

## 核心思想

将SSM的递推计算等价转换为attention矩阵乘法：

```
y[i,h,p] = Σⱼ α_complete[i,j,h,p] * x[j,h,p]
```

其中alpha矩阵包含两部分：
1. **SSM状态传递**：非对角线元素表示历史token通过状态传递对当前输出的影响
2. **对角线**：包含两条路径的贡献
   - 路径1：x[i] → B → h[i] → C → y[i]（状态更新路径，贡献：C@B）
   - 路径2：x[i] → D → y[i]（skip connection，贡献：D）
   - 总贡献：α[i,i] = C[i] @ discrete_B[i] + D

## 关键发现：SSM递推顺序

**重要**：Mamba的SSM递推是"先更新状态，再计算输出"：

```python
# 正确的递推顺序（与Mamba官方实现一致）
for t in range(seqlen):
    state = A[t] * state + B[t] * x[t]  # 1. 先更新状态
    y[t] = C[t] @ state + D * x[t]      # 2. 再计算输出
```

这意味着：
- x[i]**确实会影响**y[i]（通过状态更新）
- 对角线α[i,i]必须包含C@B项，不能只有D

## 验证流程：两阶段计算

### 阶段1：计算Alpha矩阵

**总输入参数**（从模型提取）：
- `discrete_A`: [batch, seqlen, nheads, headdim, d_state] - 离散化的状态转移矩阵
- `discrete_B`: [batch, seqlen, nheads, ngroups, d_state] - 离散化的输入矩阵
- `C`: [batch, seqlen, ngroups, d_state] - 输出投影矩阵
- `D`: [nheads] - Skip connection参数

**总输出**：
- `alpha_complete`: [seqlen, seqlen, nheads, headdim] - 完整的attention矩阵

---

#### 步骤1.1：计算累积乘积 (Cumulative Product)

**目标**：计算 $\bar{A}[i,j] = \prod_{k=j+1}^{i} A[k]$

**输入**：
- `A`: [seqlen, nheads, headdim, d_state]

**输出**：
- `A_cumprod`: [seqlen, seqlen, nheads, headdim, d_state]

**数学公式**：
$$\log(\bar{A}[i,j]) = \sum_{k=j+1}^{i} \log(A[k]) = \text{cumsum}(\log A)[i] - \text{cumsum}(\log A)[j]$$

**代码**：
```python
log_A = torch.log(A + 1e-10)  # [seqlen, nheads, headdim, d_state]
log_A_cumsum = torch.cumsum(log_A, dim=0)  # [seqlen, nheads, headdim, d_state]
A_cumprod = torch.exp(log_A_cumsum[i] - log_A_cumsum[j])  # [seqlen, seqlen, nheads, headdim, d_state]
```

**说明**：`A_cumprod[i,j,h,p,n]` 表示从时刻j到i的累积状态转移

---

#### 步骤1.2：计算有效B矩阵 (Effective B)

**目标**：将输入经过累积状态转移后的效果

**输入**：
- `A_cumprod`: [seqlen, seqlen, nheads, headdim, d_state]
- `B`: [seqlen, nheads, d_state]

**输出**：
- `B_eff`: [seqlen, seqlen, nheads, headdim, d_state]

**数学公式**：
$$B_{\text{eff}}[i,j,h,p,n] = \bar{A}[i,j,h,p,n] \cdot B[j,h,n]$$

**代码**：
```python
B_eff = A_cumprod * B[None, :, :, None, :]  # [seqlen, seqlen, nheads, headdim, d_state]
```

**说明**：`B_eff[i,j,h,p,n]` 表示时刻j的输入经过状态传递后，在时刻i对第n个状态维度的贡献

---

#### 步骤1.3：用C投影到输出空间 (SSM部分的Alpha)

**目标**：计算SSM状态传递的attention权重

**输入**：
- `C_mat`: [seqlen, d_state]
- `B_eff`: [seqlen, seqlen, nheads, headdim, d_state]

**输出**：
- `alpha_ssm`: [seqlen, seqlen, nheads, headdim]

**数学公式**：
$$\alpha_{\text{ssm}}[i,j,h,p] = \sum_{n=1}^{d\_state} C[i,n] \cdot B_{\text{eff}}[i,j,h,p,n]$$

**代码**：
```python
alpha_ssm = torch.einsum('in,ijhpn->ijhp', C_mat, B_eff)  # [seqlen, seqlen, nheads, headdim]
```

**说明**：`alpha_ssm[i,j,h,p]` 表示时刻j的输入x[j,h,p]通过SSM状态传递对时刻i的输出y[i,h,p]的贡献

**注意**：`alpha_ssm[i,i]`（对角线）已经包含了x[i]通过状态更新对y[i]的贡献（C[i] @ discrete_B[i]）

---

#### 步骤1.4：添加D构建完整对角线

**目标**：构建完整的attention矩阵，对角线包含两条路径的贡献

**关键发现**：在SSM递推中，x[i]**确实影响**y[i]！
- 状态更新：`h[i] = A[i] * h[i-1] + B[i] * x[i]`（h[i]包含x[i]）
- 输出计算：`y[i] = C[i] @ h[i] + D * x[i]`（y[i]受x[i]影响）

**输入**：
- `alpha_ssm`: [seqlen, seqlen, nheads, headdim]（已包含所有位置的C@B，包括对角线）
- `D`: [nheads]（skip路径贡献）

**输出**：
- `alpha_complete`: [seqlen, seqlen, nheads, headdim]

**数学公式**：
$$\alpha_{\text{complete}}[i,j,h,p] = \begin{cases}
\alpha_{\text{ssm}}[i,j,h,p] & \text{if } i \neq j \text{ (状态传递)} \\
\alpha_{\text{ssm}}[i,i,h,p] + D[h] & \text{if } i = j \text{ (状态 + skip路径)}
\end{cases}$$

**x[i]对y[i]的两条影响路径**：
```
路径1：x[i] → [×B] → h[i] → [×C] → y[i]  (贡献: C[i] @ discrete_B[i])
路径2：x[i] → [×D] → y[i]                (贡献: D)
总贡献：C[i] @ discrete_B[i] + D = alpha_ssm[i,i] + D
```

**代码**：
```python
alpha_complete = alpha_ssm.clone()  # [seqlen, seqlen, nheads, headdim]

# 对角线：alpha_ssm已包含C@B，只需加上D
for i in range(seqlen):
    diagonal_CB = alpha_ssm[i, i, :, :]  # [nheads, headdim]
    alpha_complete[i, i, :, :] = diagonal_CB + D.unsqueeze(-1).expand(nheads, headdim)
```

**说明**：`alpha_complete` 包含完整attention权重
- 非对角线：通过状态传递的历史依赖
- 对角线：当前输入的直接贡献（状态路径C@B + skip路径D）

---

### 阶段2：用Alpha计算输出

**总输入参数**：
- `x`: [seqlen, nheads, headdim] - SSM层输入
- `alpha_complete`: [seqlen, seqlen, nheads, headdim] - 完整alpha矩阵（阶段1计算得到）

**总输出**：
- `y`: [seqlen, nheads, headdim] - SSM层输出

---

#### 步骤2.1：纯矩阵乘法计算输出

**目标**：用attention机制重构SSM输出

**输入**：
- `alpha_complete`: [seqlen, seqlen, nheads, headdim]
- `x`: [seqlen, nheads, headdim]

**输出**：
- `y`: [seqlen, nheads, headdim]

**数学公式**：
$$y[i,h,p] = \sum_{j=0}^{\text{seqlen}-1} \alpha_{\text{complete}}[i,j,h,p] \cdot x[j,h,p]$$

**代码**：
```python
y = torch.einsum('ijhp,jhp->ihp', alpha_complete, x)  # [seqlen, nheads, headdim]
```

**说明**：这一阶段纯粹依赖alpha矩阵和输入x，不使用任何其他参数

**矩阵形状**：
- 输入 `x_val`: [seqlen, nheads, headdim]
- 输入 `alpha_complete`: [seqlen, seqlen, nheads, headdim]
- 输出 `y`: [seqlen, nheads, headdim]

---

#### 关键验证点

✅ **阶段1输出**：`alpha_complete` - 完整attention矩阵
- 对角线 = C@B + D（状态路径 + skip路径）
- 非对角线 = SSM状态传递权重

✅ **阶段2验证**：只依赖 `x` 和 `alpha_complete`
- **不依赖** D, A, B, C 等任何其他变量
- 纯attention形式：`y = alpha @ x`

✅ **精度验证**：Cosine similarity > 0.9999

## 文件说明

- `attention_extraction.py` - 从Mamba2模型提取SSM参数（discrete_A, B, C, x, D）
- `validate_alpha_matrix.py` - 主验证脚本，验证alpha重构的正确性

## 运行验证

```bash
cd /root/autodl-tmp/mamba-autodl
python attention_difference2/validate_alpha_matrix.py
```

**预期输出**：
```
Alpha矩阵分解:
├─ 对角线 [0,0,0,0]: 3.344369
├─   = C@B: 2.544275
├─   + D:   0.800094
└─ 非对角线 [1,0,0,0]: 1.098644

Alpha重构 vs 真实完整输出（SSM + D*x）:
├─ Cosine similarity: 1.00000000
├─ Mean error: 4.318444e-08
└─ Max error: 5.722046e-06

✅ 完全通过！
✅ 验证路径: [x] -> [alpha_complete] -> [y_complete]
✅ 只依赖: input (x) 和 alpha矩阵（包含D）
✅ 对比对象: y_complete = y_ssm + D*x（完整输出）
```

## 验证保证

1. **无作弊**：
   - 阶段1：从模型参数（discrete_A, B, C, D）计算alpha矩阵
   - 阶段2：**纯粹用alpha矩阵和x**重构输出，不使用任何其他参数
   - 对比：手动SSM递推的完整输出（正确的递推顺序）

2. **完美匹配**：
   - Cosine similarity = 1.0（完美匹配）
   - Mean error ≈ 3-4e-08（机器精度）
   - Max error ≈ 2-8e-06（浮点运算累积误差，可忽略）

3. **公式正确性**：
   - 对角线：α[i,i] = C[i] @ discrete_B[i] + D[i]（两条路径）
   - 非对角线：α[i,j] = C[i] @ (∏_{k=j+1}^i A[k]) @ B[j]（状态传递）

---

## 完整验证逻辑总结

```
[阶段1：计算Alpha矩阵]
输入: discrete_A, discrete_B, C, D
  │
  ├─ 步骤1: 计算累积乘积 A_cumprod[i,j] = ∏_{k=j+1}^i A[k]
  ├─ 步骤2: 计算有效B矩阵 B_eff[i,j] = A_cumprod[i,j] * B[j]
  ├─ 步骤3: 用C投影 alpha_ssm[i,j] = Σₙ C[i,n] * B_eff[i,j,n]（包括对角线的C@B）
  └─ 步骤4: 添加D到对角线 alpha_complete[i,i] = alpha_ssm[i,i] + D
  │
  ▼
输出: alpha_complete [seqlen, seqlen, nheads, headdim]

[阶段2：计算输出]
输入: x, alpha_complete
  │
  └─ 纯矩阵乘法: y[i] = Σⱼ alpha_complete[i,j] * x[j]
  │
  ▼
输出: y [seqlen, nheads, headdim]

[验证]
对比 y (alpha重构) vs y_real (真实模型)
└─ Cosine similarity > 0.9999 ✅
```

---

## 理论基础

**Mamba2的SSM递推形式**（先更新状态，再计算输出）：
```
# 正确的递推顺序（与官方实现一致）
for t in range(seqlen):
    state[t] = A[t] * state[t-1] + B[t] * x[t]  # 先更新状态
    y[t] = C[t] @ state[t] + D * x[t]           # 再计算输出（使用新状态）
```

关键点：
- **state[t]包含x[t]的贡献**（通过B[t] * x[t]项）
- **y[t]依赖state[t]**（已包含x[t]），因此x[t]会影响y[t]
- 对角线α[i,i]必须包含C@B项

**等价的Attention形式**：
```
y[i] = Σⱼ α[i,j] * x[j]
```

其中alpha矩阵捕获了完整的token依赖关系（通过状态传递+skip connection）。

---

## 维度说明

| 变量 | 形状 | 说明 |
|------|------|------|
| `x` | [seqlen, nheads, headdim] | SSM输入 |
| `discrete_A` | [seqlen, nheads, headdim, d_state] | 离散化状态转移 |
| `discrete_B` | [seqlen, nheads, d_state] | 离散化输入矩阵 |
| `C` | [seqlen, d_state] | 输出投影 |
| `D` | [nheads] | Skip connection |
| `A_cumprod` | [seqlen, seqlen, nheads, headdim, d_state] | 累积乘积 |
| `B_eff` | [seqlen, seqlen, nheads, headdim, d_state] | 有效B |
| `alpha_ssm` | [seqlen, seqlen, nheads, headdim] | SSM部分attention |
| `alpha_complete` | [seqlen, seqlen, nheads, headdim] | 完整attention |
| `y` | [seqlen, nheads, headdim] | SSM输出 |

*注：假设 ngroups=1, batch=1*

---

## 修复历史

**关键Bug修复**（2026-01-05）：
- ❌ 之前错误：SSM递推顺序错误（先输出后更新）→ 对角线只有D
- ✅ 修复后：先更新状态再输出（与Mamba官方一致）→ 对角线 = C@B + D
- ✅ 验证结果：Cosine similarity = **1.0**（完美匹配，机器精度误差 ~4e-08）

**完整修复历程**：
1. ✅ 添加 D 参数提取（skip connection）
2. ✅ 发现对角线问题（只有D，缺少C@B）
3. ✅ 研究Mamba官方源码，确认递推顺序
4. ✅ 修正验证脚本的SSM递推（先更新后输出）
5. ✅ 验证通过：对角线 = C@B + D，cosine similarity = 1.0

---