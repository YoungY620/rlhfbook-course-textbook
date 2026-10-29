# 第 4 讲：强化学习实现与实践

> 对应课程：Lec 4 — RL Implementation & Practice · 第 6 章第 2 部分 · 视频/PDF/讲义/源码见 [rlhfbook.com/course](https://rlhfbook.com/course)

## 从数学到代码

第 3 讲奠定了策略梯度方法的数学基础：策略梯度定理、REINFORCE、PPO 与 GRPO。本讲把这些公式变成可运行的代码。目标不是推导新算法，而是理解如何实现、调试并运行面向大型语言模型的强化学习训练。

RL 训练中最难发现的 bug 通常不是数学错误，而是静默的实现失误：错误的词元掩码、过时的对数概率缓存、形状不匹配，或者奖励被放到错误的位置。这些问题都可能让训练顺利跑完，却学到错误的东西。本章将带领读者走过核心实现模式、损失函数、基础设施选择以及工程细节，区分“能跑”和“能学”的 RL 训练。

先回顾一下符号：与第 6 章一致，我们在强化学习语境自然时使用 $(s, a)$，在提示-补全语境更清晰时使用 $(x, y)$。$(s, a)$ 视角反映词元级别的梯度计算，$(x, y)$ 视角反映序列级别的奖励。代码中两种视角都会出现。

## 策略梯度的代码实现

策略梯度目标是最大化期望回报：

$$
J(\\theta) = \\mathbb{E}\_{\\tau \\sim p\_\\theta}[R(\\tau)]。
$$

其梯度为

$$
\\nabla\_\\theta J(\\theta) = \\mathbb{E}\_{\\tau \\sim p\_\\theta}\\!\\left[\\sum\_{t=0}^{T} \\Psi\_t \\, \\nabla\_\\theta \\log \\pi\_\\theta(a\_t \\mid s\_t)\\right]，
$$

其中 $\\Psi\_t$ 是衡量动作好坏的任意量。在代码中，梯度来自对数概率求和，再让自动微分完成其余工作：

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()  # 自动微分产生 ∑ Ψ_t ∇ log π
```

本讲中的每个损失函数都是这一模式的变体：计算每个词元的对数概率，用优势或回报加权，掩掉无关位置，再聚合为标量损失。

我们将要实现的损失函数族如下：

$$
\\begin{aligned}
\\textbf{REINFORCE:}\\quad & -\\frac{1}{T}\\sum\_{t=1}^{T}\\log \\pi\_\\theta(a\_t\\mid s\_t)\\,\\big(G\_t - b(s\_t)\\big) \\\\[4pt]
\\textbf{RLOO:}\\quad & -\\frac{1}{K}\\sum\_{i=1}^{K}\\sum\_t \\log \\pi\_\\theta(a\_{i,t}\\mid s\_{i,t})\\left(R\_i-\\frac{1}{K-1}\\sum\_{j\\neq i}R\_j\\right) \\\\[4pt]
\\textbf{PPO:}\\quad & -\\frac{1}{T}\\sum\_{t=1}^{T}\\min\\!\\big(\\rho\_t A\_t,\\ \\mathrm{clip}(\\rho\_t,1-\\varepsilon,1+\\varepsilon)\\, A\_t\\big) \\\\[4pt]
\\textbf{GRPO:}\\quad & -\\frac{1}{G}\\sum\_{i=1}^{G}\\frac{1}{|a\_i|}\\sum\_{t=1}^{|a\_i|}\\min\\!\\big(\\rho\_{i,t} \\hat{A}\_i,\\ \\mathrm{clip}(\\rho\_{i,t},1-\\varepsilon,1+\\varepsilon)\\, \\hat{A}\_i\\big)
\\end{aligned}
$$

这里 $\\rho\_t = \\frac{\\pi\_\\theta(a\_t \\mid s\_t)}{\\pi\_{\\theta\_\\text{old}}(a\_t \\mid s\_t)}$ 是重要性采样比。PPO 使用 GAE 计算的词元级别优势 $A\_t$；GRPO 使用在分组补全上计算的序列级别优势 $\\hat{A}\_i = \\frac{R\_i - \\mu}{\\sigma}$。

### 计算对数概率

最基本的构件是词元级别的对数概率。完整展开的实现如下：

```python
# 模型前向传播
logits = model(input_ids).logits              # (B, L, vocab_size)

# 自回归偏移：位置 t 的 logit 预测 t+1 的词元
logits = logits[:, :-1, :]                    # (B, L-1, vocab_size)
labels = input_ids[:, 1:]                     # (B, L-1)
completion_mask = completion_mask[:, 1:]       # (B, L-1)

# 每个词元的对数概率
log_probs = logits.log_softmax(dim=-1)
token_log_probs = log_probs.gather(
    dim=-1, index=labels.unsqueeze(-1)
).squeeze(-1)                                 # (B, L-1)

# 序列对数概率：对补全词元求和
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
```

所有计算都保持在对数空间，避免多个小概率相乘导致的数值下溢。实践中通常会把它封装成辅助函数：

```python
def compute_log_probs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids,
                   attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    return torch.gather(log_probs, dim=-1,
                        index=targets).squeeze(-1)
```

该函数返回所有位置的对数概率，掩码在后面应用。在采样阶段，对旧策略和参考模型调用此函数并缓存结果；训练阶段再对当前策略重新调用。

### REINFORCE 代码

最简单的策略梯度损失使用一个基线，通常是批次内的平均奖励：

```python
# rewards: (B,) — 每个序列一个奖励
# seq_log_probs: (B,) — 补全词元对数概率之和

baseline = rewards.mean()
advantages = rewards - baseline

# REINFORCE 损失（取负是因为最小化）
loss = -(advantages * seq_log_probs).mean()
```

正优势会提升被采样补全的概率，负优势则会降低它。这里的 `.mean()` 只是将每个序列的损失聚合成标量的方式之一；具体选择很重要，下文“损失聚合策略”会专门讨论。

### RLOO 代码

REINFORCE Leave-One-Out（RLOO）对每个提示生成 $K$ 个补全，并用其余 $K-1$ 个补全的平均奖励作为基线。这样既能保持梯度估计无偏，又能降低方差。

```python
# rlhf_reward: (B*K,) 展平的奖励张量
# 提示优先布局：同一提示的 K 个补全相邻排列
rlhf_reward = rlhf_reward.view(-1, rloo_k)  # (B, K)

# Leave-one-out 基线：每个提示其余 K-1 个奖励的平均
baseline = (rlhf_reward.sum(dim=1, keepdim=True) - rlhf_reward) / (rloo_k - 1)

advantages = rlhf_reward - baseline        # (B, K)
advantages = advantages.reshape(-1)        # (B*K,)
```

数据加载器必须保证每个提示生成 $K$ 个补全并且它们被分组排列。训练循环的其余部分与标准策略梯度相同：将优势乘以序列对数概率并反向传播。

## 近端策略优化（PPO）

与 REINFORCE 和 RLOO 相比，PPO 显著增加了复杂度，但也带来了更低方差、词元级别信用分配以及复用每个采样批次进行多步梯度更新的能力。代价是需要同时保存四个模型、价值函数初始化敏感，以及更多超参数。

写成损失形式的裁剪代理目标为

$$
L^{\\text{CLIP}}(\\theta) = -\\mathbb{E}\_t\\!\\left[\\min\\!\\Big(\\rho\_t A\_t,\\; \\text{clip}(\\rho\_t, 1-\\varepsilon, 1+\\varepsilon)\\, A\_t\\Big)\\right]，
$$

其中 $\\rho\_t = \\frac{\\pi\_\\theta(a\_t \\mid s\_t)}{\\pi\_{\\theta\_\\text{old}}(a\_t \\mid s\_t)}$。

### 从标量奖励到词元级别回报

奖励模型通常在补全结束时给出一个标量 $R(x, y)$。为了输入 GAE，我们用 KL 惩罚构造每个词元的奖励：

$$
r\_t = \\begin{cases}
R(x, y) - \\beta \\, \\text{KL}\_t & \\text{若 } t = T \\text{（最后一个词元）} \\\\
-\\beta \\, \\text{KL}\_t & \\text{否则}，
\\end{cases}
$$

其中 $\\text{KL}\_t = \\log \\pi\_\\theta(a\_t \\mid s\_t) - \\log \\pi\_\\text{ref}(a\_t \\mid s\_t)$。这些词元级别奖励随后被 GAE 消费，生成词元级别优势。

### PPO 训练循环

一次完整的 PPO 迭代可概括为：

```text
for each batch:
  1. 从数据集中采样提示
  2. 用当前策略 π_θ 生成补全
  3. 用奖励模型打分 → 每个序列的奖励
  4. 计算参考模型对数概率 → 每个词元的 KL 惩罚
  5. 构造奖励：r_t = r_t - β * KL_t（按词元）
  6. 反向计算回报（GAE 或蒙特卡洛）
  7. 计算优势：A_t = returns_t - V(s_t)
  8. 对同一批次执行 k = 1 到 K 个 epoch：
     - 计算 ratio = π_θ(a_t|s_t) / π_old(a_t|s_t)
     - 策略损失：裁剪代理目标
     - 价值损失：对回报的 MSE
     - total_loss = policy_loss + vf_coef * value_loss
     - 反向传播 + 优化器步进
  9. 同步 π_old ← π_θ
```

在采样阶段，以下张量只计算一次并冻结：

| 张量 | 形状 | 用途 |
|------|------|------|
| 词元 ID | `(B, L)` | 全部 |
| 补全掩码 | `(B, L)` | 全部 |
| 旧策略对数概率 $\\log \\pi\_{\\theta\_\\text{old}}$ | `(B, L)` | 重要性采样比 |
| 旧价值 $V\_{\\phi\_\\text{old}}$ | `(B, L)` | Critic 裁剪 |
| 参考模型对数概率 $\\log \\pi\_\\text{ref}$ | `(B, L)` | KL 惩罚 |
| 奖励 | `(B,)` 或 `(B, L)` | 优势计算 |

每个训练步只重新计算新策略对数概率 $\\log \\pi\_\\theta$ 和当前价值估计 $V\_\\phi$。重要性采样比一个来自缓存、一个来自当前模型，正是这种结构让同一个采样批次可以被多轮复用。

### 广义优势估计（GAE）

GAE 通过反向传播 TD 误差并按指数衰减加权来计算词元级别优势：

$$
\\hat{A}\_t^{\\text{GAE}} = \\sum\_{l=0}^{\\infty} (\\gamma \\lambda)^l \\delta\_{t+l}, \\qquad \\delta\_t = r\_t + \\gamma V(s\_{t+1}) - V(s\_t)。
$$

在语言模型的 RLHF 中，$\\gamma$ 通常取 1.0，因为奖励评价的是整个补全而非单个词元。一种常见实现如下：

```python
# rewards: (B,) 仅终止奖励（省略 KL 构造），values_old: (B, L)
B, L = completion_mask.shape
advantages = torch.zeros_like(values_old)
next_v = torch.zeros(B, device=values_old.device)
gae = torch.zeros(B, device=values_old.device)

last_idx = completion_mask.long().cumsum(-1).argmax(-1, keepdim=True)
done_mask = (torch.arange(L, device=values_old.device).unsqueeze(0) >= last_idx).float()
rewards_t = torch.zeros_like(values_old).scatter_(-1, index=last_idx, src=rewards)

for t in reversed(range(L)):
    not_done = 1.0 - done_mask[:, t]
    delta = rewards_t[:, t] + gamma * not_done * next_v - values_old[:, t]
    gae = delta + gamma * lam * not_done * gae
    advantages[:, t] = gae
    next_v = values_old[:, t]

advantages = advantages * completion_mask
targets = (advantages + values_old).detach()
advantages = advantages.detach()
```

`targets` 和 `advantages` 上的 `.detach()` 非常关键：采样阶段的价值被当作固定目标，不能对其进行微分。

### 策略损失、价值损失与组合目标

裁剪策略损失的代码为：

```python
# 计算概率比
ratio = torch.exp(new_per_token_logps - per_token_logps)  # (B, L)

# 裁剪代理目标
eps = 0.2
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
pg_loss = torch.max(pg_losses1, pg_losses2)
```

`torch.max` 选择更悲观的分支。由于我们最小化的是负目标，这能防止某一次更新过度承诺。

价值函数也有独立的裁剪：

```python
v_pred = value_net(completions)  # (B, L)
v_clip = torch.clamp(v_pred, old_values - eps, old_values + eps)
vf_unclipped = 0.5 * (v_pred - targets) ** 2
vf_clipped   = 0.5 * (v_clip - targets) ** 2
vf_loss = torch.max(vf_unclipped, vf_clipped)
```

组合后的词元级别损失为：

```python
per_token_loss = pg_loss + vf_coef * vf_loss  # (B, L)

loss = ((per_token_loss * completion_mask).sum(dim=1) /
         completion_mask.sum(dim=1)).mean()
```

### 多轮 epoch 与小批量

PPO 和 GRPO 可以选择将同一个采样批次复用多个梯度步。只要当前策略偏离采样策略，裁剪就会生效。两种机制会导致偏离：

- **小批量**：在第一个小批量更新后，策略已经改变，因此同一次 epoch 内后续小批量看到的 $\\rho\_t \\neq 1$。即使 $K=1$ 也可能触发裁剪。
- **多轮 epoch**：对整批数据循环 $K$ 次，策略偏离更远。典型 $K$ 为 2–4；超过约 6 次后策略会过于偏离同策略分布。

如果 $K=1$ 且没有小批量划分，则 $\\pi\_\\theta = \\pi\_{\\theta\_\\text{old}}$，比率恒为 1，裁剪永不生效，PPO 退化为带 GAE 的 vanilla 策略梯度。

### 优势白化与价值函数初始化

优势通常会在批次内归一化为零均值、单位方差：

```python
valid_adv = advantages[completion_mask.bool()]
advantages = ((advantages - valid_adv.mean()) /
              (valid_adv.std() + 1e-8)) * completion_mask
```

这能稳定不同批次之间的梯度幅度。如果不做白化，所有奖励偏高或偏低的批次可能产生过大梯度。

价值函数 $V\_\\phi$ 的初始估计必须合理。常见做法有：

- 从奖励模型骨干初始化（InstructGPT 惯例，Tülu 3 在 Lambert et al. 2024 中也采用）。价值估计一开始接近真实奖励。
- 从监督微调（SFT）模型初始化并随机初始化价值头。成本更低，但早期训练可能不稳定。
- 在正式策略更新前，先用带测量奖励的数据做一段价值函数 warmup。

### PPO 超参数与批次大小

常见 LLM RLHF  setups 的参考范围：

| 超参数 | 典型范围 | 说明 |
|----------------|:-------------:|-------|
| 裁剪范围 $\\varepsilon$ | 0.1–0.2 | 信任域宽度 |
| GAE $\\lambda$ | 0.95 | 偏差-方差权衡 |
| 价值系数 | 0.5–1.0 | Critic 损失权重 |
| KL 系数 $\\beta$ | 0.01–0.1 | 参考约束强度 |
| 每批次 epoch 数 $K$ | 2–6 | 异策略预算 |
| 学习率 | $1 \\times 10^{-6}$ 到 $5 \\times 10^{-6}$ | 远低于 SFT |
| 批次大小 | 256–1024 个提示 | 越大方差越低 |

批次大小在 RL 中比监督学习更重要，因为梯度来自蒙特卡洛采样。McCandlish et al. 2018 发现在大规模 RL 中关键批次大小可达数百万条转移；Hilton et al. 2021 指出 PPO 并非批次大小不变，因为裁剪将批次大小与有效步长耦合；Beukman et al. 2026 认为 PPO 平台期常由嘈杂的损失估计造成，只能通过更多样本解决。在 RLHF 中，扩大批次大小是稳定训练最廉价的手段之一。

### PPO 调试清单

训练过程中需要关注：

- **裁剪比例**：被裁剪词元的百分比。0% 表示裁剪从未生效；超过 50% 表示策略变化过快。
- **优势统计**：均值接近 0（若已白化），标准差稳定。
- **价值损失**：应下降；否则价值函数没有学到东西。
- **KL 散度**：应缓慢增长，而非爆炸。

常见的静默 bug 包括：

- 将提示词元纳入策略损失。
- 奖励放错位置（例如用 `[:, -1]` 而非实际最后一个生成词元）。
- 把过时的对数概率当作“当前”使用，导致比率卡在 1 附近、裁剪不生效。
- GRPO 分组奖励标准差为 0，所有补全奖励相同，优势变为 NaN。
- 价值或策略损失在补全掩码外计算，导致在 padding 上训练。

对于一个 7B 的 fp16 模型，内存中的四个模型——策略、价值函数、参考策略和奖励模型——仅权重就约 56 GB，还不算优化器状态、激活和梯度。这就是 PPO 常常需要模型并行或卸载，以及有些实现会共享骨干的原因。

## 组相对策略优化（GRPO）及其变体

GRPO 由 DeepSeekMath（Shao et al. 2024）提出，它完全去掉了价值函数。对每个提示，它采样 $G$ 个补全并计算组内归一化优势：

$$
\\hat{A}\_i = \\frac{R\_i - \\mu\_G}{\\sigma\_G}, \\qquad \\mu\_G = \\frac{1}{G}\\sum\_{j=1}^{G} R\_j, \\quad \\sigma\_G = \\sqrt{\\frac{1}{G}\\sum\_{j=1}^{G}(R\_j - \\mu\_G)^2}。
$$

损失使用与 PPO 相同的裁剪代理目标，但将序列级别优势广播到每个词元，并附加可选 KL 惩罚：

$$
L^{\\text{GRPO}}(\\theta) = -\\frac{1}{G}\\sum\_{i=1}^{G}\\frac{1}{|a\_i|}\\sum\_{t=1}^{|a\_i|}\\min\\!\\Big(\\rho\_{i,t} \\hat{A}\_i,\\; \\text{clip}(\\rho\_{i,t}, 1-\\varepsilon, 1+\\varepsilon)\\, \\hat{A}\_i\\Big) + \\beta \\, \\text{KL}(\\pi\_\\theta \\| \\pi\_\\text{ref})。
$$

没有价值函数，没有 GAE——优势完全来自同一组内补全之间的比较。

### GRPO 代码

```python
# 每个提示生成 G 个补全，然后计算组优势
mean_r = rewards.view(-1, G).mean(dim=1)
std_r = rewards.view(-1, G).std(dim=1)
mean_r = mean_r.repeat_interleave(G)
std_r = std_r.repeat_interleave(G)
advantages = ((rewards - mean_r) / (std_r + 1e-4)).unsqueeze(1)

# 重要性采样比
ratio = torch.exp(new_logps - old_logps)   # (B*G, L)

# 裁剪代理目标（与 PPO 相同）
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - eps, 1 + eps)
pg_loss = torch.max(pg_losses1, pg_losses2)

# KL 惩罚直接加在损失里（而非奖励里）
per_token_loss = pg_loss + beta * per_token_kl

loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()
```

### KL 惩罚的放置位置

PPO 将 KL 惩罚折入奖励，然后再估计优势：

```python
rewards.scatter_(1, last_idx, rm_score)
rewards = rewards - beta * per_token_kl
advantages = gae(rewards, values, ...)
```

GRPO 用原始奖励计算优势，再把 KL 作为独立损失项：

```python
advantages = z_score(rewards)
loss = pg_loss + beta * per_token_kl
```

这避免了 KL 惩罚与优势估计之间的相互影响。

### GRPO 与 PPO 的对比

| 组件 | PPO | GRPO |
|-----------|:---:|:----:|
| 价值网络 | 需要 | 不需要 |
| GAE 计算 | 需要 | 不需要 |
| Critic 损失 | 需要 | 不需要 |
| 价值函数初始化 | 需要 | 不需要 |
| 优势计算 | 词元级别（GAE） | 序列级别（z-score） |
| KL 处理方式 | 折入奖励 | 独立损失项 |

GRPO 代码量明显更少，内存中也少保存一份模型。

### RLOO 与 GRPO 的优势对比

RLOO 与 GRPO 共享按提示分组的结构，区别仅在于基线：

```python
# RLOO：leave-one-out 均值
rewards = rewards.view(-1, K)         # (N, K)
baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (K - 1)
advantages = rewards - baseline
advantages = advantages.reshape(-1)
```

```python
# GRPO：组内 z-score 归一化
rewards = rewards.view(-1, G)       # (N, G)
mean_r = rewards.mean(dim=1, keepdim=True)
std_r = rewards.std(dim=1, keepdim=True)
advantages = (rewards - mean_r) / (std_r + 1e-4)
advantages = advantages.reshape(-1)
```

GRPO 增加了标准差归一化；RLOO 使用 leave-one-out 均值。

### GSPO 与 CISPO

两种近期变体针对重要性采样的数值不稳定性。

**GSPO**（Group Sequence Policy Optimization；Zheng et al. 2025）将词元级别比率聚合为每个序列的几何平均比率：

$$
\\bar{\\rho}\_i = \\exp\\!\\Big(\\frac{1}{|a\_i|}\\sum\_t \\log \\rho\_{i,t}\\Big)。
$$

```python
log_ratio = (new_logps - old_logps) * mask
rho = torch.exp(log_ratio.sum(dim=1) / mask.sum(dim=1))  # (B*G,)
rho_clipped = rho.clamp(1 - eps, 1 + eps)
loss = -torch.min(rho * advantages, rho_clipped * advantages).mean()
```

**CISPO**（Clipped Importance Sampling Policy Optimization；MiniMax Team 2025）将裁剪后的比率 detach，使梯度只经过 $\\log \\pi\_\\theta$：

$$
\\text{sg}\\!\\big[\\text{clip}(\\rho, 1\\!-\\!\\varepsilon, 1\\!+\\!\\varepsilon)\\big] \\cdot \\hat{A} \\cdot \\log \\pi\_\\theta。
$$

```python
rho = torch.exp(new_logps - old_logps)
rho_clipped = torch.clamp(rho, 1 - eps, 1 + eps).detach()
loss = -(rho_clipped * advantages.unsqueeze(1) * new_logps * mask).sum() / mask.sum()
```

GSPO 改变比率的粒度，CISPO 改变表达式中哪一部分接收梯度。

## 损失聚合策略

上述所有算法都计算同一个核心梯度 $\\nabla \\log \\pi \\cdot A$，但每个词元的损失如何聚合成标量损失会显著影响训练动态。

### 老虎机风格 vs MDP 风格 RLHF

**老虎机风格**方法给每个补全一个奖励，并将序列级别优势广播到所有词元。REINFORCE、RLOO 和 GRPO 默认属于这一类。

**MDP 风格**方法将每个词元视为一个动作，计算词元级别优势或价值。带 GAE 的 PPO 默认属于这一类。

实践中 RLHF 通常是混合的：序列级别奖励，但词元级别对数概率梯度。PPO 风格的 RLHF 通常从序列级别的奖励模型分数出发，再通过 KL 构造和 GAE 获得词元级别信用。

### 三种聚合策略

给定词元级别损失 $\\ell\_{i,t}$、补全长度 $|a\_i|$ 和批次大小 $B$，三种主要策略为：

**按序列归一化** —— 每个序列对批次损失贡献相等：

$$
L = \\frac{1}{B}\\sum\_{i=1}^{B} \\frac{1}{|a\_i|}\\sum\_{t=1}^{|a\_i|} \\ell\_{i,t}。
$$

```python
loss = ((per_token_loss * completion_mask).sum(dim=1) /
         completion_mask.sum(dim=1)).mean()
```

这是 GRPO 和部分 PPO 实现的默认做法。它使词元级别梯度与序列长度成反比，可能让模型偏向短回答。

**按词元归一化** —— 整个批次中每个词元贡献相等：

$$
L = \\frac{\\sum\_{i=1}^{B} \\sum\_{t=1}^{|a\_i|} \\ell\_{i,t}}{\\sum\_{i=1}^{B} |a\_i|}。
$$

```python
loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
```

DAPO（Yu et al. 2025）使用这一方式。更长的序列对总梯度贡献更大，可能鼓励冗长补全。

**固定长度归一化** —— 用常数 $L\_\\text{max}$ 归一化：

$$
L = \\frac{1}{B}\\sum\_{i=1}^{B} \\frac{1}{L\_\\text{max}}\\sum\_{t=1}^{|a\_i|} \\ell\_{i,t}。
$$

```python
loss = ((per_token_loss * completion_mask).sum(dim=1) /
         L_max).mean()
```

来自 Dr. GRPO（Liu et al. 2025）。它在均衡每个词元梯度的同时，仍让更长序列因包含更多有效词元而贡献更大总梯度。

### 具体例子

```python
seq_1_losses = [1, 1, 1, 1, 10]           # 5 个词元，均值 2.8
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 个词元，均值 1.9
```

| 策略 | 批次损失 | 短序列梯度 | 长序列梯度 |
|----------|:----------:|:------------------:|:-----------------:|
| **按序列** | $(2.8 + 1.9)/2 = 2.35$ | 每个词元 0.20 | 每个词元 0.10 |
| **按词元** | $(14 + 19)/15 = 2.2$ | 每个词元 0.067 | 每个词元 0.067 |
| **固定长度** ($L=10$) | $(1.4 + 1.9)/2 = 1.65$ | 每个词元 0.10 | 每个词元 0.10 |

按序列归一化让短序列的词元梯度更大；按词元和固定长度归一化则让它们相等。选择哪种策略取决于你是想鼓励还是抑制长补全。

## 训练基础设施

### 同策略 vs 异策略

理论理想是完全同策略：生成补全、更新策略、用更新后的策略重新生成、重复。每个批次被评分并用于一个短更新窗口，然后丢弃。

实践中，生成和训练在不同 GPU 组上重叠以获得更高吞吐。用于生成的模型可能比训练模型落后一到多个梯度步。通常这种轻微陈旧是可以接受的，现代系统大多容忍这一点点异策略偏移。

### 同步 vs 异步训练

![同步训练会让 GPU 空闲；把生成与训练分开能改善，但同策略仍需等待。异步（异策略）则将两者重叠以充分利用资源。](assets/async_v_synch_rl.png)

现代面向 LLM 的 RL 将计算分为两组：

- **Actor**（推理 GPU）：用 vLLM 等引擎生成补全。
- **Learner**（训练 GPU）：计算策略梯度更新。

Ray 等进程管理库协调它们之间的数据流，模型权重定期从 learner 同步到 actor。

![分布式 RL 系统：提示队列与结果队列连接 learner GPU 与 actor GPU。](assets/distributed-rl.png)

Noukhovitch et al. 2024 研究了异步 RLHF 的权衡。核心矛盾很简单：完全同策略会让 GPU 空闲，而完全异步会引入更多异策略偏移。在推理型训练中，单个答案可能需要数万词元，生成成为强瓶颈，因此重叠生成与训练变得至关重要。

### 开源代码库

几个开源实现是很好的起点：

- **TRL**（von Werra et al. 2020）：Hugging Face 生态，支持 PPO/GRPO/DPO，适合入门。
- **Open Instruct**（Ivison et al. 2024）：Allen AI，多算法，适合研究与复现。
- **veRL**：在 RLVR 时代广泛使用。
- **OpenRLHF**：最初为 RLHF 设计，也广泛用于 RLVR。

## 实践工程细节

### 数值考虑

一些数值细节决定 RL 训练是否稳定：

- **对数空间运算**：始终使用 `log_softmax` + `gather`，不要先 `softmax` 再 `log`。softmax 会把小概率压成极小浮点数，再取 log 会放大精度损失。
- **掩码 padding**：`completion_mask` 必须只对补全词元为 1，排除提示词元、EOS 后的 padding 以及 EOS 本身（或一致地包含 EOS）。在聚合前用该掩码乘损失。
- **终止词元处理**：EOS 的处理方式需要一致，处理不当是常见静默错误。
- **序列长度处理**：变长补全需要谨慎归一化。
- **基线 detach**：`advantages.detach()` 防止梯度穿过优势计算。
- **除法保护**：任何除以 `mask.sum()` 或序列长度的地方，使用 `.clamp_min(1)` 或 `+ eps`，避免空补全导致 NaN。

### 训练时该监控什么

以下阈值仅供参考，具体因模型大小、任务和算法而异：

| W&B 面板 | 健康 | 不健康 |
|-----------|---------|-----------|
| `reward/mean` | 稳定上升 | 尖峰、震荡、平台期 |
| `kl/mean` | 缓慢增长（0 → 2–5） | 爆炸（>10）或恒为 0 |
| `loss/policy` | 下降 | 发散或 NaN |
| `metrics/clip_frac` | 5–30% | 0% 或 >50% |
| `generation/length` | 稳定或略增 | 单调增加（长度作弊） |
| 策略熵 | 缓慢下降 | 骤降到 0（模式坍塌） |

同时要在留出基准上监控评估分数，并阅读样本输出检查连贯性。仅看奖励曲线是不够的。

## 前瞻：正则化与过度优化

当 RL 训练出问题，往往是因为优化器抓住了虚假信号而非目标奖励。这些主题将在后续第 14、15 讲深入讨论：

- KL 惩罚：前向 KL 与反向 KL 的数学与代码。
- 古德哈特定律：代理目标与真实质量何时分道扬镳。
- 奖励黑客：谄媚、冗长、过度拒绝。
- 隐式正则化：“SFT 记忆，RL 泛化”以及“RL's Razor”。

## 要点小结

- 策略梯度在代码中归结为：计算词元级别对数概率，用优势加权，对补全词元做掩码，再聚合成标量损失。
- REINFORCE 与 RLOO 是简单的老虎机风格方法：一个序列级别优势广播到所有词元。RLOO 用 leave-one-out 的按提示基线降低方差。
- PPO 引入学习价值函数、GAE 词元级别优势和裁剪代理目标，使同一采样批次可被多步复用。它功能强大但内存开销大，且对价值函数初始化敏感。
- GRPO 去掉价值函数，通过组内归一化奖励计算优势。它更简单、内存更少，已成为许多推理任务的首选。
- GSPO 与 CISPO 是近期针对重要性采样不稳定的变体：GSPO 将比率聚合到序列级别，CISPO 将裁剪比率 detach 以保持所有词元的梯度。
- 损失聚合方式——按序列、按词元或固定长度——会改变有效词元梯度尺度，可能使模型偏向更短或更长的补全。
- 现代训练多为异步：actor 在推理 GPU 上生成补全，learner 在训练 GPU 上更新策略，以少量异策略陈旧换取更高吞吐。
- 稳定的 RL 训练依赖 careful masking、对数空间运算、基线 detach、除法保护和严格监控奖励、KL、裁剪比例与生成长度。
