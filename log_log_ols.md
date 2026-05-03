# Log-Log OLS 原理与幂律拟合关键

## 一、Log-Log OLS 原理

### 目标

拟合幂律关系：

$$y = k \cdot x^\alpha$$

### 线性化

直接对该方程做非线性拟合很麻烦。两边取自然对数后，方程变为线性形式：

$$\ln y = \ln k + \alpha \cdot \ln x$$

令 $Y = \ln y$，$X = \ln x$，$\beta_0 = \ln k$，$\beta_1 = \alpha$，即变为普通线性回归：

$$Y = \beta_0 + \beta_1 X$$

### OLS 解析解

普通最小二乘（OLS）对线性方程有闭合解：

$$\alpha = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2} = \frac{\mathrm{Cov}(X, Y)}{\mathrm{Var}(X)}$$

$$k = e^{\,\bar{Y} - \alpha \cdot \bar{X}}$$

### 对应代码

```python
def fit_power_law(xs, ys):
    lx = [math.log(x) for x in xs]   # X = ln(C)
    ly = [math.log(y) for y in ys]   # Y = ln(N_opt)
    mean_x = sum(lx) / len(lx)
    mean_y = sum(ly) / len(ly)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(lx, ly))
    var = sum((x - mean_x) ** 2 for x in lx)
    alpha = cov / var                              # 斜率
    k = math.exp(mean_y - alpha * mean_x)         # 截距还原
    return k, alpha
```

### 与 scipy.optimize.curve_fit 的对比

| | Log-Log OLS | `curve_fit` |
|---|---|---|
| 求解方式 | 解析解，无迭代 | 非线性迭代（Levenberg-Marquardt）|
| 数值稳定性 | 高（对数压缩大数量级差异）| 低（1e24 这类大数易溢出或收敛慢）|
| 外部依赖 | 仅标准库 `math` | 需要 `scipy` |
| 等价性 | 最小化 $\sum (\ln y_i - \ln \hat{y}_i)^2$（相对误差）| 最小化 $\sum (y_i - \hat{y}_i)^2$（绝对误差）|

> **注意**：两者优化的目标函数不同。Log-Log OLS 最小化对数空间的残差（等价于相对误差），对跨越多个数量级的数据更合适；`curve_fit` 在原始空间最小化绝对误差，大值点会主导拟合。

---

## 二、基于 isoflops_curves.json 预测准确的关键

**核心结论：准确率取决于数据点的分布质量，而非数量。**

### 关键 1：每条 IsoFLOP 曲线的最优点必须落在内部（横向覆盖）

每个计算预算 $C_i$ 对应一条 IsoFLOP 曲线。理论上损失关于模型参数量 $N$ 呈 U 形：

- $N$ 太小：模型容量不足，损失高
- $N$ 最优：计算预算用在了最合适的模型-数据平衡点
- $N$ 太大：无法在预算内充分训练，损失反弹

数据实例（`C=1e19`）：

```
N:    8.3e7  ...  8.1e8  ...  2.0e9
loss: 6.575  ...  5.618  ...  5.768
```

**如果采样的 N 范围不覆盖真实最低点，取到的是"边界最小值"而非真实最优，会系统性地偏移拟合结果。**

### 关键 2：compute budget 的跨度决定外推可靠性（纵向覆盖）

Log-Log OLS 拟合的是对数空间中的直线斜率。跨越的数量级越大，斜率估计越准确，外推越可信。

| 数据跨度 | 外推目标 | 外推距离 |
|---|---|---|
| $6\times10^{18}$ 到 $3\times10^{21}$（约 2.7 个数量级）| $10^{23}$ ~ $10^{24}$ | 再往外 2~3 个数量级 |

外推距离远超拟合范围，是最大的不确定性来源。

### 关键 3：指数之和 = 1 是内置校验

由 $C \approx 6ND$，若 $N \propto C^\alpha$ 且 $D \propto C^\beta$，则必然有：

$$\alpha + \beta = 1$$

当前拟合结果：$\alpha = 0.469$，$\beta = 0.531$，两者之和 $= 1.000$，完全吻合。

这既是结果自洽的验证，也是检验拟合是否出错的快速校验手段。若两者之和偏离 1，说明数据质量有问题或拟合过程存在 bug。
