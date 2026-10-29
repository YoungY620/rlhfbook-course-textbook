> Corresponding lecture: Lec 10 — Regularization in RL, Why RL Generalizes, and Why SFT Forgets · Chapter 15 · Watch/PDF/Slides/Source at [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 10: Regularization in RL, Why RL Generalizes, and Why SFT Forgets

The RL step in RLHF maximizes reward from the reward model **minus a penalty for drifting from the reference model**. This lecture is about that penalty — and what happens with and without it. How do these optimizers change the distributions of the models, and how do we control it?

## Part 1: The explicit KL penalty

### A KL penalty to the reference model controls reward

The canonical LLM reward function:

$$ r = r\_\\theta - \\lambda\_{\\text{KL}} \\, D\_{\\mathrm{KL}}\\!\\left( \\pi\_{\\text{RL}}(y \\mid x) \\,\\|\\, \\pi\_{\\text{ref}}(y \\mid x) \\right) $$

KL control for RL predates LLMs: dialogue agents (Jaques et al., 2017), then fine-tuning pretrained models (Jaques et al., 2020). This penalty is a **reverse KL** — estimated by sampling from the *policy* and scoring against the reference. It punishes the policy for putting mass where the reference would not. “KL distance” is the *optimization distance* spent (colloquially — KL is not a true metric). In practice the coefficient is often written as $\\beta$.

### Measuring KL in practice

Sampling from $P$ turns the definition into an expectation:

$$ D\_{\\mathrm{KL}}(P \\,\\|\\, Q) = \\mathbb{E}\_{x \\sim P}\\left[ \\log P(x) - \\log Q(x) \\right] $$

Practitioners watch the KL curve during training — a very large KL usually means a bug or a broken model. Lower-variance estimators ($k\_1, k\_2, k\_3$) are essential for KL penalties (Schulman 2020).

```python
# sample from the policy
tokens = model.generate(inputs)

# score sampled tokens under both models
logprobs     = log_softmax(model.forward(tokens).logits)
ref_logprobs = log_softmax(ref_model.forward(tokens).logits)

# log-probs of the tokens actually generated
token_lp     = gather(logprobs, tokens)
ref_token_lp = gather(ref_logprobs, tokens)

# sequence-level difference approximates KL
kl_approx = token_lp.sum(-1) - ref_token_lp.sum(-1)
```

![An OLMo-2-7B GRPO run, RL directly on the base model (R1-Zero-style, no SFT): verifiable reward climbs while logged KL to the reference rises and then wanders. This run has β = 0.](../assets/olmo2-grpo-zero-reward-kl.png)

### Static or dynamic KL penalties? β began as a feedback controller

The first RLHF-on-LMs paper (Ziegler et al., 2019) did not fix β. It picked a **target KL** and let a controller chase it:

$$ e\_t = \\operatorname{clip}\\!\\left( \\frac{\\mathrm{KL}(\\pi\_t, \\pi\_{\\text{ref}}) - \\mathrm{KL}\_{\\text{target}}}{\\mathrm{KL}\_{\\text{target}}},\\, -0.2,\\, 0.2 \\right), \\qquad \\beta\_{t+1} = \\beta\_t \\left(1 + K\_\\beta\\, e\_t\\right) $$

This is a log-space proportional controller with $K\_\\beta = 0.1$: KL too high → β grows and pulls the policy back; too low → β shrinks and frees it up. Runs with different seeds land on the same KL budget, making experiments comparable. PPO’s original adaptive KL penalty variant doubled or halved β around a target (Schulman et al., 2017), and constrained RL later made the controls framing explicit with full PID controllers (Stooke et al., 2020). Modern practice swung back to a small **static** β — or, in many RLVR reasoning recipes, no KL term at all.

## Part 2: RL optimization is a reverse KL minimization

### The reward penalty and the optimization shape are two different things

The **penalty** is a term added to the reward, with a coefficient you tune. The **shape** of RL optimization is about which direction of KL is minimized, set by *where the samples come from*:

- **SFT** samples from *data (or a separate teacher model)* → minimizing its loss is exactly minimizing a **forward** KL.
- **RL** samples from *itself* → with the penalty on, maximizing the objective is exactly a **reverse-KL minimization** toward a reward-tilted reference. Even without the penalty, on-policy sampling biases RL toward KL-minimal solutions.

### RL is reverse KL

Start from the KL-regularized objective:

$$
\\max\_\\pi\\; \\mathcal{J}\_{\\text{RL}}(\\theta) = \\mathbb{E}\_{x \\sim \\mathcal{D},\\, y \\sim \\pi(\\cdot \\mid x)}\\left[ r(x, y) \\right] - \\beta\\, D\_{\\mathrm{KL}}\\!\\left(\\pi(\\cdot \\mid x) \\,\\|\\, \\pi\_{\\text{ref}}(\\cdot \\mid x)\\right)
$$

Dividing by $-\\beta$ and normalizing turns the objective into a single KL, minimized exactly when $\\pi$ equals the optimal policy $\\pi\_\\star$ (the same starting point as DPO):

$$ \\pi\_\\star(y \\mid x) = \\frac{1}{Z(x)}\\, \\pi\_{\\text{ref}}(y \\mid x)\\, \\exp\\!\\left(\\tfrac{1}{\\beta}\\, r(x,y)\\right) $$

Read $\\pi\_\\star$ as the **reward-tilted reference**: take $\\pi\_{\\text{ref}}$, multiply each completion’s probability by $\\exp(r/\\beta)$, then renormalize. Large $\\beta$ tilts barely at all; small $\\beta$ concentrates on the highest-reward completions.

Now expand the reverse KL to $\\pi\_\\star$, substituting $\\log \\pi\_\\star = \\log \\pi\_{\\text{ref}} - \\log Z(x) + \\tfrac{1}{\\beta} r(x,y)$:

$$
\\begin{aligned}
D\_{\\mathrm{KL}}(\\pi\_\\theta \\,\\|\\, \\pi\_\\star) &= \\mathbb{E}\_{x \\sim \\mathcal{D},\\, y \\sim \\pi\_\\theta}\\left[ \\log \\pi\_\\theta(y \\mid x) - \\log \\pi\_\\star(y \\mid x) \\right] \\\\
&= \\mathbb{E}\\left[ \\log \\pi\_\\theta - \\log \\pi\_{\\text{ref}} + \\log Z(x) - \\tfrac{1}{\\beta}\\, r(x,y) \\right] \\\\
&= -\\tfrac{1}{\\beta}\\,\\mathbb{E}\\left[r(x,y)\\right] + D\_{\\mathrm{KL}}\\!\\left(\\pi\_\\theta \\,\\|\\, \\pi\_{\\text{ref}}\\right) + \\underbrace{\\log Z(x)}\_{\\text{constant}} \\\\
&\\propto -\\tfrac{1}{\\beta}\\, \\mathcal{J}\_{\\text{RL}}(\\theta)
\\end{aligned}
$$

$$ \\boxed{\\ \\max\_\\theta\\, \\mathcal{J}\_{\\text{RL}}(\\theta) \\iff \\min\_\\theta\\, D\_{\\mathrm{KL}}(\\pi\_\\theta \\,\\|\\, \\pi\_\\star)\\ } $$

With the penalty included, RL doesn’t just *use* a reverse KL — the whole objective **is** one, pointed at the reward-tilted reference policy.

### SFT is forward KL

SFT trains on a fixed dataset — the samples come from the data distribution $\\pi\_{\\mathcal{D}}$, which makes it the other KL direction:

$$
\\begin{aligned}
D\_{\\mathrm{KL}}(\\pi\_{\\mathcal{D}} \\,\\|\\, \\pi\_\\theta) &= \\mathbb{E}\_{(x,y) \\sim \\mathcal{D}} \\left[ \\log \\pi\_{\\mathcal{D}}(y \\mid x) - \\log \\pi\_\\theta(y \\mid x) \\right] \\\\
&= \\underbrace{\\mathbb{E}\\left[ \\log \\pi\_{\\mathcal{D}}(y \\mid x) \\right]}\_{-H(\\pi\_{\\mathcal{D}}),\\ \\text{constant in } \\theta} - \\mathbb{E}\\left[ \\log \\pi\_\\theta(y \\mid x) \\right] \\\\
&= -H(\\pi\_{\\mathcal{D}}) + \\mathcal{L}\_{\\text{SFT}}(\\theta) \\;\\propto\\; \\mathcal{L}\_{\\text{SFT}}(\\theta)
\\end{aligned}
$$

Minimizing the SFT loss *is* minimizing forward KL to the data distribution.

### SFT and RL are the two directions of KL

**Forward KL** — supervised fine-tuning:

$$ D\_{\\mathrm{KL}}(\\pi\_{\\mathcal{D}} \\,\\|\\, \\pi\_\\theta) = \\mathbb{E}\_{y \\sim \\pi\_{\\mathcal{D}}}\\!\\left[\\log \\tfrac{\\pi\_{\\mathcal{D}}(y)}{\\pi\_\\theta(y)}\\right] $$

Samples come from the **target** (a fixed dataset). *Mass-covering*: wherever the target has mass and $\\pi\_\\theta \\to 0$, the loss blows up — the model must spread to cover everything.

**Reverse KL** — reinforcement learning:

$$ D\_{\\mathrm{KL}}(\\pi\_\\theta \\,\\|\\, \\pi\_\\star) = \\mathbb{E}\_{y \\sim \\pi\_\\theta}\\!\\left[\\log \\tfrac{\\pi\_\\theta(y)}{\\pi\_\\star(y)}\\right] $$

Samples come from the **policy itself**. *Mode-seeking*: only penalized where it places mass, so it concentrates on high-reward modes.

The same two directions appear in distillation: offline KD / SFT is forward KL (expectation over the teacher); on-policy distillation is reverse KL (expectation over the student).

## Part 3: Why RL generalizes more

### “SFT memorizes, RL generalizes”

A controlled study (Chu et al., 2025) post-trained on one task and evaluated under a rule shift:

- **GeneralPoints**: reach 24 by combining four cards with $+,-,\\times,\\div$; test with shift to the face-card rule (train: J/Q/K = 10; test: 11/12/13).
- **V-IRL**: visual navigation; test with shift from absolute (north/east) to relative (left/right) directions.

On V-IRL, RL improves out-of-distribution accuracy **80.8% → 91.8%**. SFT collapses it **80.8% → 1.3%** — destroying spatial reasoning the base model already had. RL-based post-training carries *implicit* regularization from its on-policy structure alone. SFT is more brittle.

### Which direction should forget less?

The naive read: forward KL is *mass-covering*, so SFT should preserve every mode — while mode-seeking RL should collapse onto one and forget the rest. This intuition assumes a unimodal policy, but LLMs are multimodal (Chen et al., 2025).

![Mode intuition for retention: LLMs are multimodal, so reverse KL does not necessarily collapse everything.](../assets/retaining_by_doing_mode_intuition.png)

### RL’s razor: lower KL drift for equivalent performance

> “Among the many high-reward solutions for a new task, on-policy methods such as RL are inherently biased toward solutions that remain closer to the original policy in KL divergence.”

Forgetting tracks KL drift: $\\text{Forgetting} \\approx f\\!\\left(\\mathbb{E}\_{x \\sim \\tau}\\!\\left[D\_{\\mathrm{KL}}\\!\\left(\\pi\_0 \\,\\|\\, \\pi\\right)\\right]\\right)$ with $R^2 = 0.96$, measured on the new task’s data — a cheap forgetting predictor. The ablation shows that **on-policy data fully accounts for the difference** — negative gradients have no discernible effect (Shenfeld et al., 2026).

![Among policies that solve the new task, RL converges to those closest in KL to the base model, yielding higher prior-task retention at matched new-task performance.](../assets/rl_razor_motivation.png)

## Part 4: Other tools to control optimization

### Other regularization in the wild

Most of these are scaffolding: added to stabilize one setup, simplified away in the next model generation.

- **Pretraining next-token prediction gradients** (InstructGPT): add $\\gamma\\, \\mathbb{E}\_{x \\sim \\mathcal{D}\_{\\text{pretrain}}}\\left[\\log \\pi\_{\\text{RL}}(x)\\right]$ to the objective, “to fix the performance regressions on public NLP datasets.”
- **NLL alongside DPO**: $\\mathcal{L}\_{\\text{DPO+NLL}} = \\mathcal{L}\_{\\text{DPO}} + \\alpha\\, \\mathcal{L}\_{\\text{NLL}}$ keeps the chosen text high-likelihood in absolute terms, not just relatively better.
- **Margin loss** for reward models (Llama 2): $-\\log \\sigma\\!\\left(r\_{\\theta}(y\_c) - r\_{\\theta}(y\_r) - m(y\_c, y\_r)\\right)$, where the margin $m$ comes from annotator rating deltas — the Likert scales from Lecture 8.

### 2026: the trust region / KL penalty is moving on

Tool use is changing what regularization has to do. In agentic recipes, the KL-to-reference penalty is disappearing:

- GLM-5 removes it outright “to accelerate RL improvement”; Kimi K2 → K3 ship with no KL penalty and no reference policy.
- In the TMax terminal-agent recipe, a small KL reduced collapse severity but lowered reward, so the final recipe uses $\\beta = 0$.
- TMax summarizes the numerical issues: 20+ turn trajectories, async/partial rollouts, and train-vs-inference engine mismatch make drift from the *sampler* the binding failure, not drift from init. Instabilities increase past 10 assistant turns and were absent below 5.

### A trust region on the sampling distribution

**DPPO** (Divergence Proximal Policy Optimization) masks tokens by a *directly estimated* divergence between the rollout and training policies (binary Total Variation), instead of PPO’s per-token ratio clip. The per-token ratio is a noisy one-sample estimate of that divergence. GLM’s IcePop and Kimi’s log-ratio interval do the same job: gradients masked, not clipped. This is where the eye of regularization is today — algorithmic updates rather than KL penalties.

![The motivation from the DPPO paper: for the same rollout tokens, per-token probability ratios (left) explode at low probabilities while the directly-estimated TV divergence (right) stays stable.](../assets/dppo-ratio-vs-tv.png)

## Key Takeaways

- The KL penalty is the explicit control: a **reverse KL**, estimated on the policy’s own samples.
- The KL-regularized RL objective is a reverse KL minimization — mode-seeking toward a reward-tilted reference policy — while SFT is the forward direction, mass-covering toward the data.
- Even with no penalty, on-policy RL is implicitly regularized: SFT memorizes, RL generalizes.
- Verifiable rewards are much less prone to over-optimization than reward models, so the KL penalty is playing a smaller role in 2026 agentic recipes.
- Modern trust regions are shifting from KL-to-reference penalties to divergence control on the sampling distribution.
