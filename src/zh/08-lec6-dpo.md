> 对应课程：Lec 6 —— 直接偏好优化 · 第 8 章 · 视频/PDF/幻灯片/源码见 [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 6：直接偏好优化

直接偏好优化（Direct Preference Optimization，DPO）是最具影响力的直接对齐算法。Rafailov 等人（2023）提出，在通常需要单独训练奖励模型并运行强化学习循环的 RLHF 目标下，DPO 可以用单个基于偏好对的监督式损失来求解。论文副标题 *Your Language Model is Secretly a Reward Model*（你的语言模型 secretly 是一个奖励模型）点明了核心洞见：在标准的 KL 正则化 RLHF 目标下，最优策略与奖励模型是同一对象的两个视角，二者通过闭式重参数化相互联系。DPO 正是利用这一关系直接训练策略。

DPO 在 2023 年 5 月出现时恰逢开源 RLHF 复现步履维艰，大多数“对齐”模型还只是监督微调（SFT）模型，因此意义重大。直到 2023 年秋季，社区才逐渐摸索出正确的数据与超参数——尤其是出人意料的低学习率。随后 Zephyr-β（Tunstall et al. 2024）、Tülu 2（Ivison et al. 2023）等开源模型证明 DPO 可以规模化工作，直接引爆了后训练研究的热潮。2024 年迭代 DPO 方法的感觉，几乎不亚于 2025 年的 RL 方法或 2026 年的工具调用方法。

本章从零推导 DPO，并讨论其实践用法。推导分为四步：

1. 求解 KL 正则化 RLHF 目标的**最优策略**；
2. 将该解反解，写出**用策略表示的奖励**；
3. 把隐式奖励代入 **Bradley-Terry** 偏好模型；
4. 读出 **DPO 损失** 及其**梯度**。

最终结果是一个只需几行 PyTorch 即可实现的损失函数，无需奖励模型，也无需采样循环。

## DPO 在流程中的位置

经典 RLHF 有三个环节。首先收集人类偏好对；其次训练奖励模型 $r\_\\phi(x, y)$；最后用 PPO 等强化学习算法针对 $r\_\\phi$ 优化策略，并通过 KL 惩罚项使策略靠近参考模型 $\\pi\_{\\text{ref}}$。

DPO 把第二、三步合并成一个监督式损失。我们将在下面证明的关键认识是：最优 RLHF 策略与奖励模型是同一对象的两种表达——知道其一，就能闭式地恢复另一个。这意味着我们可以直接在偏好数据上训练策略，而无需显式构建奖励模型或运行策略梯度。

在现代后训练流程中，DPO 只是更大循环里若干可互换优化器之一。一个典型配方可能先做 SFT，再在偏好对上用 DPO 或 IPO，然后拒绝采样，最后在线 RL。算法本身的重要性不如基础模型和数据，但 DPO 的简洁性使其成为实用的主力军。

## DPO 一览

整个推导最终落脚到两个对象。第一个是最优策略，即 KL 正则化 RLHF 目标的闭式解：

$$
\\pi^{\*}(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right).
$$

第二个是实际训练时使用的 DPO 损失：

$$
\\mathcal{L}\_{\\text{DPO}}(\\pi\_\\theta; \\pi\_{\\text{ref}}) = -\\mathbb{E}\_{(x, y\_c, y\_r) \\sim \\mathcal{D}} \\left[ \\log \\sigma\\!\\left( \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)} \\right) \\right].
$$

在 sigmoid 内部，第一项衡量学习到的策略相对于参考模型提高了多少“被选”补全的概率，第二项对“被拒”补全做同样的事。当被选补全的相对提升超过被拒补全时，损失下降，即策略学会了给正确回复更高排名。

实践中，这个损失只需几行代码：

```python
pi_logratios  = policy_chosen_logps  - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits)
```

从目标函数到这段实现之间的所有内容，就是下面要展开的推导。

## 推导最优策略

我们从标准的 RLHF 优化问题出发：

$$
\\max\_{\\pi} \\; \\mathbb{E}\_{x \\sim \\mathcal{D}, \\, y \\sim \\pi}\\big[r(x, y)\\big] - \\beta \\, \\mathcal{D}\_{\\text{KL}}\\big(\\pi \\, \\|\\, \\pi\_{\\text{ref}}\\big).
$$

第一项是策略下的期望奖励，第二项是 KL 散度惩罚，由 $\\beta > 0$ 控制优化后的策略可以偏离参考模型多远。我们的目标是找到求解该问题的策略 $\\pi$。

### 把 KL 散度折入期望

回忆 KL 散度的定义：

$$
\\mathcal{D}\_{\\text{KL}}\\big(\\pi \\, \\|\\, \\pi\_{\\text{ref}}\\big) = \\mathbb{E}\_{x \\sim \\mathcal{D}, \\, y \\sim \\pi}\\!\\left[\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)}\\right].
$$

目标中的两项现在共享同一个关于 $y \\sim \\pi(y \\mid x)$ 的期望，因此可以合并：

$$
\\begin{aligned}
\\max\_{\\pi} \\; \\mathbb{E}\\big[r(x, y)\\big] - \\beta \\, \\mathcal{D}\_{\\text{KL}}\\big(\\pi \\, \\|\\, \\pi\_{\\text{ref}}\\big)
&= \\max\_{\\pi} \\; \\mathbb{E}\\!\\left[ r(x, y) - \\beta \\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} \\right] \\\\
&= \\max\_{\\pi} \\; \\left( \\mathbb{E}\\big[r(x, y)\\big] - \\beta \\, \\mathbb{E}\\!\\left[\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)}\\right] \\right).
\\end{aligned}
$$

### 转换为最小化

转成最小化更方便。将整个目标乘以 $-1$，并把 $\\max$ 变为 $\\min$：

$$
\\min\_{\\pi} \\; \\left( -\\mathbb{E}\\big[r(x, y)\\big] + \\beta \\, \\mathbb{E}\\!\\left[\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)}\\right] \\right).
$$

再除以 $\\beta$ 并合并两个期望：

$$
\\min\_{\\pi} \\; \\mathbb{E}\\!\\left[ \\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} - \\frac{1}{\\beta} r(x, y) \\right].
$$

### 合并为单一对数比

利用 $\\tfrac{1}{\\beta} r(x, y) = \\log e^{r(x, y)/\\beta}$，把括号合并成一个对数比：

$$
\\min\_{\\pi} \\; \\mathbb{E}\\!\\left[ \\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x) \\, e^{r(x, y)/\\beta}} \\right].
$$

这个表达式有何含义？策略 $\\pi$ 现在与 $\\pi\_{\\text{ref}} \\, e^{r/\\beta}$ 比较，这是被奖励重新加权后的参考分布。每个回复的概率乘以 $e^{r(x, y)/\\beta}$，奖励越高放大越多；$\\beta$ 控制加权的激进程度。如果分母是一个合法的概率分布 $q$，那么这个期望就是 $\\mathcal{D}\_{\\text{KL}}(\\pi \\, \\|\\, q)$，根据 Gibbs 不等式在 $\\pi = q$ 处取最小值。

### 引入配分函数

问题在于 $\\pi\_{\\text{ref}}(y \\mid x) \\, e^{r(x, y)/\\beta}$ 并非概率分布：对所有 $y$ 求和不等于 1。我们用配分函数

$$
Z(x) = \\sum\_{y} \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right)
$$

来归一化，$Z(x)$ 依赖于 $x$ 和 $r$，但不依赖于 $\\pi$。除以 $Z(x)$ 后得到合法分布：

$$
q(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right).
$$

为了让目标看起来是 KL 散度，在期望内部加再减 $\\log Z(x)$。从

$$
\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} - \\frac{1}{\\beta} r(x, y)
$$

出发：

$$
\\begin{aligned}
&\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} - \\frac{1}{\\beta} r(x, y) + \\log Z(x) - \\log Z(x) \\\\
&= \\left( \\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\log Z(x) \\right) - \\log Z(x) - \\frac{1}{\\beta} r(x, y) \\\\
&= \\log \\frac{\\pi(y \\mid x)}{\\frac{1}{Z(x)} \\pi\_{\\text{ref}}(y \\mid x)} - \\log Z(x) - \\frac{1}{\\beta} r(x, y).
\\end{aligned}
$$

再把 $\\tfrac{1}{\\beta} r(x, y) = \\log e^{r(x, y)/\\beta}$ 并入第一项的分母：

$$
\\log \\frac{\\pi(y \\mid x)}{\\frac{1}{Z(x)} \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right)} - \\log Z(x).
$$

代入 $q(y \\mid x)$，目标变为

$$
\\min\_{\\pi} \\; \\mathbb{E}\_{x \\sim \\mathcal{D}} \\!\\left[ \\mathbb{E}\_{y \\sim \\pi} \\!\\left[ \\log \\frac{\\pi(y \\mid x)}{q(y \\mid x)} \\right] - \\log Z(x) \\right].
$$

内层期望正是 KL 散度：

$$
\\min\_{\\pi} \\; \\mathbb{E}\_{x \\sim \\mathcal{D}} \\!\\left[ \\mathcal{D}\_{\\text{KL}}\\big(\\pi(y \\mid x) \\, \\|\\, q(y \\mid x)\\big) - \\log Z(x) \\right].
$$

### 应用 Gibbs 不等式

$\\log Z(x)$ 不依赖于 $\\pi$，因此只有 KL 散度可被优化。Gibbs 不等式告诉我们 KL 散度非负，且仅当两个分布相同时取零。于是最小化点为 $\\pi^{\*} = q$，得到最优策略

$$
\\boxed{\\,\\pi^{\*}(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right).\\,}
$$

这就是 KL 正则化 RLHF 目标的闭式解：最优策略按指数化奖励重新加权参考模型，再归一化。

## 从策略恢复奖励

现在我们将最优策略反解，用策略表示奖励。这一步让 DPO 得以用策略概率的对数比取代显式奖励模型。

从

$$
\\pi^{\*}(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r^{\*}(x, y)\\right)
$$

出发，两边取对数：

$$
\\begin{aligned}
\\log \\pi^{\*}(y \\mid x)
&= \\log \\!\\left( \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r^{\*}(x, y)\\right) \\right) \\\\
&= -\\log Z(x) + \\log \\pi\_{\\text{ref}}(y \\mid x) + \\frac{1}{\\beta} r^{\*}(x, y).
\\end{aligned}
$$

整理求解 $r^{\*}(x, y)$：

$$
\\frac{1}{\\beta} r^{\*}(x, y) = \\log \\pi^{\*}(y \\mid x) - \\log \\pi\_{\\text{ref}}(y \\mid x) + \\log Z(x).
$$

乘以 $\\beta$：

$$
\\boxed{\\,r^{\*}(x, y) = \\beta \\log \\frac{\\pi^{\*}(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x).\\,}
$$

这就是隐式奖励。它是最优策略与参考策略的对数比，加上一个只依赖于 $x$ 的常数 $\\beta \\log Z(x)$。由于我们最终取奖励差，这个常数会被消掉，无需显式计算。

## 与 Bradley-Terry 模型的联系

RLHF 中的偏好数据通常用 Bradley-Terry 模型建模。对于提示 $x$ 下的两条补全 $y\_1$、$y\_2$，$y\_1$ 更被偏好的概率为

$$
p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\frac{\\exp\\!\\big(r^{\*}(x, y\_1)\\big)}{\\exp\\!\\big(r^{\*}(x, y\_1)\\big) + \\exp\\!\\big(r^{\*}(x, y\_2)\\big)}.
$$

代入隐式奖励 $r^{\*}(x, y) = \\beta \\log \\tfrac{\\pi^{\*}(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x)$：

$$
p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\frac{\\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)} + \\beta \\log Z(x)\\right)}{\\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)} + \\beta \\log Z(x)\\right) + \\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_2 \\mid x)}{\\pi\_{\\text{ref}}(y\_2 \\mid x)} + \\beta \\log Z(x)\\right)}.
$$

每一项都含公共因子 $\\exp(\\beta \\log Z(x)) = Z(x)^{\\beta}$。利用 $e^{a+b} = e^a e^b$ 消去：

$$
p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\frac{\\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)}\\right)}{\\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)}\\right) + \\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_2 \\mid x)}{\\pi\_{\\text{ref}}(y\_2 \\mid x)}\\right)}.
$$

难解的配分函数消失了。令

$$
\\Delta\_i = \\beta \\log \\frac{\\pi^{\*}(y\_i \\mid x)}{\\pi\_{\\text{ref}}(y\_i \\mid x)},
$$

则

$$
p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\frac{e^{\\Delta\_1}}{e^{\\Delta\_1} + e^{\\Delta\_2}}.
$$

分子分母同乘 $e^{-\\Delta\_1}$：

$$
\\begin{aligned}
p^{\*}(y\_1 \\succ y\_2 \\mid x)
&= \\frac{e^{\\Delta\_1} e^{-\\Delta\_1}}{e^{\\Delta\_1} e^{-\\Delta\_1} + e^{\\Delta\_2} e^{-\\Delta\_1}} \\\\
&= \\frac{1}{1 + e^{\\Delta\_2 - \\Delta\_1}}.
\\end{aligned}
$$

利用 sigmoid 函数 $\\sigma(z) = \\tfrac{1}{1 + e^{-z}}$，得到

$$
\\boxed{\\,p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\sigma\\!\\left(\\beta \\log \\frac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)} - \\beta \\log \\frac{\\pi^{\*}(y\_2 \\mid x)}{\\pi\_{\\text{ref}}(y\_2 \\mid x)}\\right).\\,}
$$

这就是 Bradley-Terry 偏好概率，只是用策略对数比取代了奖励模型分数。

## DPO 损失

我们观测到的偏好数据是：对提示 $x$，$y\_c$ 优于 $y\_r$。最大似然目标是最大化 $p(y\_c \\succ y\_r \\mid x)$，等价于最小化负对数似然：

$$
\\mathcal{L}\_{\\text{DPO}}(\\pi\_\\theta; \\pi\_{\\text{ref}}) = -\\mathbb{E}\_{(x, y\_c, y\_r) \\sim \\mathcal{D}} \\!\\left[ \\log p(y\_c \\succ y\_r \\mid x) \\right].
$$

代入可训练策略 $\\pi\_\\theta$ 下的 sigmoid 形式，得到 DPO 损失：

$$
\\boxed{\\,\\mathcal{L}\_{\\text{DPO}}(\\pi\_\\theta; \\pi\_{\\text{ref}}) = -\\mathbb{E}\_{(x, y\_c, y\_r) \\sim \\mathcal{D}} \\!\\left[ \\log \\sigma\\!\\left( \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)} \\right) \\right].\\,}
$$

该损失直接可微。没有奖励模型，训练时不生成样本，也没有 RL 循环。优化过程推动策略相对于冻结的参考模型，提高被选补全与被拒补全之间的对数概率差距。

## DPO 梯度

对损失求梯度能更清楚地看出优化器在做什么。从

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\nabla\_\\theta \\, \\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\log \\sigma\\!\\left( \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)} \\right) \\right]
$$

出发。令

$$
u = \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)}
$$

为 sigmoid 的自变量。由链式法则：

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\frac{\\sigma'(u)}{\\sigma(u)} \\nabla\_\\theta u \\right].
$$

利用 $\\sigma'(u) = \\sigma(u)(1 - \\sigma(u))$ 并约去 $\\sigma(u)$：

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\big(1 - \\sigma(u)\\big) \\nabla\_\\theta u \\right].
$$

利用恒等式 $1 - \\sigma(u) = \\sigma(-u)$，得到紧凑形式

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\sigma(-u) \\nabla\_\\theta u \\right].
$$

现在对 $u$ 求梯度。只有 $\\pi\_\\theta$ 依赖于 $\\theta$，参考模型已冻结，其对数项消失：

$$
\\nabla\_\\theta u = \\beta \\big[ \\nabla\_\\theta \\log \\pi\_\\theta(y\_c \\mid x) - \\nabla\_\\theta \\log \\pi\_\\theta(y\_r \\mid x) \\big].
$$

代回并把 $\\sigma(-u)$ 写完整：

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\beta \\, \\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\sigma\\!\\left( \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} \\right) \\cdot \\big( \\nabla\_\\theta \\log \\pi\_\\theta(y\_c \\mid x) - \\nabla\_\\theta \\log \\pi\_\\theta(y\_r \\mid x) \\big) \\right].
$$

可概括为

$$
\\begin{aligned}
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} &= -\\beta \\, \\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ w \\cdot \\big( \\nabla\_\\theta \\log \\pi\_\\theta(y\_c \\mid x) - \\nabla\_\\theta \\log \\pi\_\\theta(y\_r \\mid x) \\big) \\right], \\\\
&\\quad\\text{其中}\\quad w = \\sigma\\!\\big(r\_\\theta(x, y\_r) - r\_\\theta(x, y\_c)\\big).
\\end{aligned}
$$

权重 $w \\in (0, 1)$ 在模型越错时越大，即当模型把被拒回复排得比被选回复还高时，更新权重更大。括号内部分提高 $y\_c$ 的似然、降低 $y\_r$ 的似然。超参数 $\\beta$ 缩放步长，在正确排序与偏离 $\\pi\_{\\text{ref}}$ 之间取得平衡。

## 推导回顾

从 RLHF 目标到 DPO 损失的完整链条为：

$$
\\begin{aligned}
\\textbf{目标函数：} \\;&\\; \\max\_{\\pi} \\; \\mathbb{E}\\big[r(x, y)\\big] - \\beta \\, \\mathcal{D}\_{\\text{KL}}(\\pi \\, \\|\\, \\pi\_{\\text{ref}}) \\\\
\\Rightarrow\\; \\textbf{最优策略：} \\;&\\; \\pi^{\*}(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right) \\\\
\\Rightarrow\\; \\textbf{隐式奖励：} \\;&\\; r^{\*}(x, y) = \\beta \\log \\frac{\\pi^{\*}(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x) \\\\
\\Rightarrow\\; \\textbf{Bradley-Terry：} \\;&\\; p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\sigma\\!\\big(r^{\*}(x, y\_1) - r^{\*}(x, y\_2)\\big) \\quad \\text{（$Z$ 消去）} \\\\
\\Rightarrow\\; \\textbf{DPO 损失：} \\;&\\; -\\log \\sigma\\!\\left(\\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)}\\right).
\\end{aligned}
$$

奖励模型从未被显式构建，它一直隐藏在策略之中。

## 局限性与变体

DPO 虽然简洁，但并非没有缺点。理解其局限性有助于解释为何涌现出大量变体。

### 似然位移

DPO 损失只关心被选与被拒对数比之间的**差距**，而不关心它们的绝对值。因此模型可以通过更快地压低被拒概率来降低损失，即使被选概率本身也在下降。这一现象被称为**似然位移**（Razin et al. 2025；Ren and Sutherland 2025）。它通过推导中的配分函数 $Z(x)$ 起作用，可能把概率质量推向未被训练数据覆盖的离分布行为。

![DPO 中似然位移的示意图。](assets/dpo_displacement.png)

在真实训练中，被选与被拒的奖励差距可以扩大，同时被选补全的对数概率却在下滑。一些实践者会在被选回复上加 SFT 项，或使用 Cal-DPO（Xiao et al. 2024）、AlphaPO（Gupta et al. 2025）等修正方法。

### $\\beta$ 参数与静态 KL

参数 $\\beta$ 控制 KL 约束相对于奖励最大化的强度。$\\beta$ 大则策略贴近 $\\pi\_{\\text{ref}}$；$\\beta$ 小则策略可大幅偏离，可能导致过度优化。

关键在于，DPO 的最终 KL 距离是**静态的**：它直接迈向由数据集和所选 $\\beta$ 决定的最优解。与之对比，在线 RL 基于新采样的批次逐步更新，还可以使用动态 KL 控制器。这一差异是在线方法有时能达到更高峰值性能的原因之一。

### 直接对齐算法的变体

许多变体针对 DPO 损失的特定缺陷做了微调。我把这一大类算法统称为**直接对齐算法**（Direct Alignment Algorithms，DAAs）。

- **Identity Preference Optimization（IPO）**（Azar et al. 2024）将偏好概率从严格的 Bradley-Terry 形式软化，以抑制过拟合：

  $$
  \\mathcal{L}\_{\\text{IPO}} = \\mathbb{E}\_{(x, y\_w, y\_l)} \\!\\left[ \\left( \\log \\frac{\\pi\_\\theta(y\_w \\mid x) \\, \\pi\_{\\text{ref}}(y\_l \\mid x)}{\\pi\_\\theta(y\_l \\mid x) \\, \\pi\_{\\text{ref}}(y\_w \\mid x)} - \\frac{1}{2\\beta} \\right)^2 \\right].
  $$

- **Conservative DPO（cDPO）**（Rafailov et al. 2023）在原 DPO 论文中提出，假设偏好标签存在噪声，并据此软化损失目标。

- **Offset DPO（ODPO）**（Amini et al. 2024）要求被选补全以一定幅度胜过被拒补全，该幅度随偏好强度缩放。

还有一些变体完全去掉参考模型：

- **Odds Ratio Preference Optimization（ORPO）**（Hong et al. 2024）把 odds-ratio 牵引项直接并入 SFT 损失。由于 odds ratio 仅使用 $\\pi\_\\theta$，无需参考模型：

  $$
  \\mathcal{L}\_{\\text{ORPO}} = \\mathcal{L}\_{\\text{SFT}} - \\lambda \\, \\mathbb{E} \\!\\left[ \\log \\sigma\\!\\left( \\log \\frac{\\text{odds}\_\\theta(y\_w \\mid x)}{\\text{odds}\_\\theta(y\_l \\mid x)} \\right) \\right], \\quad \\text{odds}\_\\theta(y) = \\frac{\\pi\_\\theta(y)}{1 - \\pi\_\\theta(y)}.
  $$

- **Simple Preference Optimization（SimPO）**（Meng et al. 2025）将奖励按长度归一化为平均对数概率，并引入目标间隔 $\\gamma$：

  $$
  \\mathcal{L}\_{\\text{SimPO}} = -\\mathbb{E} \\!\\left[ \\log \\sigma\\!\\left( \\frac{\\beta}{|y\_w|} \\log \\pi\_\\theta(y\_w \\mid x) - \\frac{\\beta}{|y\_l|} \\log \\pi\_\\theta(y\_l \\mid x) - \\gamma \\right) \\right].
  $$

尽管文献中有大量微调，算法本身的重要性远不及基础模型和数据。

## 实现细节

DPO 的实现确实简单。核心损失如下：

```python
# 策略与冻结参考模型的对数概率差距
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# 对数比之差：若策略将概率移向被选补全则为正
logits = pi_logratios - ref_logratios

# DPO 损失：负对数 sigmoid 驱使策略扩大被选/被拒差距
losses = -F.logsigmoid(beta * logits)

# 隐式奖励（detach —— 仅用于日志）
chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

有两点工程经验值得注意。

**KL 散度是静态的。** DPO 根据数据集和 $\\beta$ 直接迈向最优解，不会根据最近批次自适应调整 KL 惩罚，这与在线 RL 方法不同。

**缓存参考模型对数概率。** 简单实现会同时前向传播策略和参考模型，导致显存翻倍。由于 $\\pi\_{\\text{ref}}$ 已冻结，可以预先计算并缓存每个训练样本的参考对数概率，峰值显存大约可减半。

## 直接对齐算法与合成偏好数据

直接对齐算法需要反馈数据，但这种反馈不必来自人类。大多数现代 DPO 使用强模型标注的偏好。

UltraFeedback（Cui et al. 2023）是首个有影响力的开源合成偏好数据集，开启了 DPO 时代。Tülu 3（Lambert et al. 2024）、SmolLM 2（Ben Allal et al. 2025）等模型随后推出了更大的合成反馈配方。

同策略数据——部分补全由被调优模型自身生成——有助于对比损失在正确的词元空间内优化。Tülu 3 特别强调了这一点。后续工作提出 **Delta Learning**（Geng et al. 2025），认为被选与被拒补全之间的**差距**比具体由哪些模型生成更重要。例如，Olmo 3 与 SmolLM 3 都使用了 Qwen 3 32B 作为被选来源、Qwen 3 0.6B 作为被拒来源，二者是独立并发提出的。

合成偏好数据集也有自身偏差。前沿评审模型如 GPT-4 存在长度偏差（Dubois et al. 2024）以及对自身风格输出偏好的偏差（Panickssery et al. 2024），这些偏差会渗入训练数据，需要监控。

## 离线与在线偏好优化

DPO 及其他 DAA 是**离线**方法：它们基于预先收集的固定数据集训练。这使其更简单、更稳定、更易于迭代。主要局限是覆盖范围：性能受限于静态数据集的多样性。

PPO 与策略梯度方法是**在线**方法：训练时生成新补全并用奖励模型打分。在线方法可以探索补全空间的新区域，往往能达到更高峰值性能（Ivison et al. 2024；Xu et al. 2024；Tajwar et al. 2024）。代价是更多计算、更多组件——PPO 需要同时驻留四个模型——以及更高的工程复杂度。

## DPO 如今的定位

既然 DPO 如此简单且已被众多模型成功使用，为何在前沿似乎不再那么核心？有几种解释：

- Olmo 3（Team Olmo et al. 2025）与 SmolLM 3（Bakouch et al. 2025）在 SFT 后使用 DPO 处理推理痕迹，说明它在简单流程中仍然有效。
- NVIDIA Nemotron 3（NVIDIA 2025）将 DPO 与二元分类器优化结合在一个离线阶段，数据由生成式奖励模型打分。
- Liquid AI LFM2（2025）在最终 RLVR 阶段前，对半在线数据使用长度归一化 DPO。

更宏观地看，DPO 在蒸馏成分重、分布更尖锐、训练数据来自多个教师模型的配方中表现良好。它是获得不错模型的可靠路径，但拥有充足工程资源的大多数实验室可以通过在线 RL 达到更高峰值。随着推理模型主要通过 RL 训练，在线方法的基础设施会进一步完善，性能差距可能持续存在。然而对许多实践者而言，DPO 仍是偏好调优的合适起点。

## 要点小结

- DPO 与 PPO 求解同一个 KL 正则化 RLHF 目标，但无需单独奖励模型或 RL 循环，直接在偏好对上训练策略。
- 最优策略有闭式解：$\\pi^{\*}(y \\mid x) \\propto \\pi\_{\\text{ref}}(y \\mid x) \\exp(r(x, y)/\\beta)$。
- 反解该式得到隐式奖励 $r^{\*}(x, y) = \\beta \\log \\tfrac{\\pi^{\*}(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x)$。
- 将隐式奖励代入 Bradley-Terry 模型，配分函数 $Z$ 被消去，剩下对策略对数比之差的 sigmoid。
- DPO 损失是该差的负对数 sigmoid。其梯度提高被选补全的似然、降低被拒补全的似然，并按模型错误程度加权。
- DPO 存在似然位移、KL 距离静态等局限。IPO、ORPO、SimPO 等变体分别针对不同问题，但数据与基础模型质量的重要性远超小幅算法差异。
- DPO 是离线方法，简单高效；在线 RL 可能达到更高峰值，但需要更多工程与计算投入。对于使用合成偏好数据的偏好调优，DPO 仍是实用的起点。
