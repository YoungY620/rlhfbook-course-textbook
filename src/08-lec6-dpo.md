> Corresponding lecture: Lec 6 — Direct Preference Optimization · Chapter 8 · Watch/PDF/Slides/Source at [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 6: Direct Preference Optimization

Direct Preference Optimization (DPO) is the most influential direct-alignment algorithm. Released by Rafailov et al. (2023), it showed that the same RLHF objective that normally requires a separate reward model and an RL loop can be solved with a single supervised-style loss on preference pairs. The paper's subtitle, *Your Language Model is Secretly a Reward Model*, captures the central insight: under the standard KL-regularized RLHF objective, the optimal policy and the reward model are two views of the same object, related by a closed-form reparameterization. DPO exploits that relationship to train the policy directly.

The arrival of DPO in May 2023 mattered because open-source RLHF replications were struggling. Most "aligned" models were simply SFT. It took the community until the fall of 2023 to figure out the right data and hyperparameters—especially the surprisingly low learning rates—but once Zephyr-β (Tunstall et al. 2024), Tülu 2 (Ivison et al. 2023), and other open models demonstrated that DPO could work at scale, it opened the floodgates of post-training research. Iterating on DPO methods in 2024 felt as exciting as RL methods in 2025 or tool-use methods in 2026.

This chapter derives DPO from scratch and then discusses how it is used in practice. The derivation has four steps:

1. Solve the KL-regularized RLHF objective for its **optimal policy**.
2. Invert that solution to write the **reward in terms of the policy**.
3. Plug that implicit reward into the **Bradley-Terry** preference model.
4. Read off the **DPO loss** and its **gradient**.

The result is a loss that can be implemented in a few lines of PyTorch, with no reward model and no sampling loop.

## Where DPO Sits in the Pipeline

Classic RLHF has three moving parts. First, collect human preference pairs. Second, train a reward model $r\_\\phi(x, y)$ on those pairs. Third, optimize a policy against $r\_\\phi$ with an RL algorithm such as PPO, under a KL penalty that keeps the policy close to a reference model $\\pi\_{\\text{ref}}$.

DPO collapses the second and third steps into one supervised-style loss. The key realization, which we prove below, is that the optimal RLHF policy and the reward model are two views of the same object: if we know one, we can recover the other in closed form. That means we can train the policy *directly* on preferences, without ever building an explicit reward model or running policy gradients.

In modern post-training pipelines, DPO is one of several interchangeable optimizers in a larger loop. A recipe may use SFT, then DPO or IPO on preference pairs, then rejection sampling, then online RL. The algorithm matters less than the base model and the data, but DPO's simplicity makes it a natural workhorse.

## DPO at a Glance

The whole derivation lands on two objects. The first is the optimal policy, the closed-form solution to the KL-regularized RLHF objective:

$$
\\pi^{\*}(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right).
$$

The second is the DPO loss, which is what we actually train:

$$
\\mathcal{L}\_{\\text{DPO}}(\\pi\_\\theta; \\pi\_{\\text{ref}}) = -\\mathbb{E}\_{(x, y\_c, y\_r) \\sim \\mathcal{D}} \\left[ \\log \\sigma\\!\\left( \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)} \\right) \\right].
$$

Inside the sigmoid, the first term measures how much the learned policy has increased the probability of the *chosen* completion relative to the reference model, and the second term does the same for the *rejected* completion. The loss decreases when the chosen shift exceeds the rejected shift, i.e., when the policy learns to rank the right response higher.

In practice, this loss is just a few lines:

```python
pi_logratios  = policy_chosen_logps  - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits)
```

Everything between the objective and the implementation is the derivation that follows.

## Deriving the Optimal Policy

We start from the standard RLHF optimization problem:

$$
\\max\_{\\pi} \\; \\mathbb{E}\_{x \\sim \\mathcal{D}, \\, y \\sim \\pi}\\big[r(x, y)\\big] - \\beta \\, \\mathcal{D}\_{\\text{KL}}\\big(\\pi \\, \\|\\, \\pi\_{\\text{ref}}\\big).
$$

The first term is the expected reward under the policy. The second term is a KL-divergence penalty weighted by $\\beta > 0$, which controls how far the optimized policy may drift from the reference. Our goal is to find the policy $\\pi$ that solves this problem.

### Folding the KL into the Expectation

Recall the definition of KL divergence:

$$
\\mathcal{D}\_{\\text{KL}}\\big(\\pi \\, \\|\\, \\pi\_{\\text{ref}}\\big) = \\mathbb{E}\_{x \\sim \\mathcal{D}, \\, y \\sim \\pi}\\!\\left[\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)}\\right].
$$

Both terms in the objective now share the same expectation over $y \\sim \\pi(y \\mid x)$, so we can combine them:

$$
\\begin{aligned}
\\max\_{\\pi} \\; \\mathbb{E}\\big[r(x, y)\\big] - \\beta \\, \\mathcal{D}\_{\\text{KL}}\\big(\\pi \\, \\|\\, \\pi\_{\\text{ref}}\\big)
&= \\max\_{\\pi} \\; \\mathbb{E}\\!\\left[ r(x, y) - \\beta \\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} \\right] \\\\
&= \\max\_{\\pi} \\; \\left( \\mathbb{E}\\big[r(x, y)\\big] - \\beta \\, \\mathbb{E}\\!\\left[\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)}\\right] \\right).
\\end{aligned}
$$

### Flipping to a Minimization

It is easier to work with a minimization. Multiply the objective by $-1$ and turn $\\max$ into $\\min$:

$$
\\min\_{\\pi} \\; \\left( -\\mathbb{E}\\big[r(x, y)\\big] + \\beta \\, \\mathbb{E}\\!\\left[\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)}\\right] \\right).
$$

Now divide by $\\beta$ and recombine the two expectations:

$$
\\min\_{\\pi} \\; \\mathbb{E}\\!\\left[ \\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} - \\frac{1}{\\beta} r(x, y) \\right].
$$

### Folding into a Single Log-Ratio

Use the identity $\\tfrac{1}{\\beta} r(x, y) = \\log e^{r(x, y)/\\beta}$ to fold the bracket into one log-ratio:

$$
\\min\_{\\pi} \\; \\mathbb{E}\\!\\left[ \\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x) \\, e^{r(x, y)/\\beta}} \\right].
$$

What does this expression mean? The policy $\\pi$ is now measured against $\\pi\_{\\text{ref}} \\, e^{r/\\beta}$, which is the reference distribution reweighted by reward. Each response's probability is multiplied by $e^{r(x, y)/\\beta}$, which is larger for higher-reward responses; $\\beta$ sets how aggressive the reweighting is. If the denominator were a valid probability distribution $q$, this expectation would be exactly $\\mathcal{D}\_{\\text{KL}}(\\pi \\, \\|\\, q)$, minimized at $\\pi = q$ by Gibbs' inequality.

### Introducing the Partition Function

The catch is that $\\pi\_{\\text{ref}}(y \\mid x) \\, e^{r(x, y)/\\beta}$ is not a probability distribution: summed over all $y$, it does not equal 1. We normalize it with the partition function

$$
Z(x) = \\sum\_{y} \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right),
$$

which depends on $x$ and $r$ but not on $\\pi$. Dividing by $Z(x)$ gives a valid distribution:

$$
q(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right).
$$

To make the objective look like a KL divergence, we add and subtract $\\log Z(x)$ inside the expectation. Starting from

$$
\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} - \\frac{1}{\\beta} r(x, y),
$$

we write

$$
\\begin{aligned}
&\\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} - \\frac{1}{\\beta} r(x, y) + \\log Z(x) - \\log Z(x) \\\\
&= \\left( \\log \\frac{\\pi(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\log Z(x) \\right) - \\log Z(x) - \\frac{1}{\\beta} r(x, y) \\\\
&= \\log \\frac{\\pi(y \\mid x)}{\\frac{1}{Z(x)} \\pi\_{\\text{ref}}(y \\mid x)} - \\log Z(x) - \\frac{1}{\\beta} r(x, y).
\\end{aligned}
$$

Now fold $\\tfrac{1}{\\beta} r(x, y) = \\log e^{r(x, y)/\\beta}$ into the denominator of the first term:

$$
\\log \\frac{\\pi(y \\mid x)}{\\frac{1}{Z(x)} \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right)} - \\log Z(x).
$$

Substituting $q(y \\mid x)$, the objective becomes

$$
\\min\_{\\pi} \\; \\mathbb{E}\_{x \\sim \\mathcal{D}} \\!\\left[ \\mathbb{E}\_{y \\sim \\pi} \\!\\left[ \\log \\frac{\\pi(y \\mid x)}{q(y \\mid x)} \\right] - \\log Z(x) \\right].
$$

The inner expectation is exactly a KL divergence:

$$
\\min\_{\\pi} \\; \\mathbb{E}\_{x \\sim \\mathcal{D}} \\!\\left[ \\mathcal{D}\_{\\text{KL}}\\big(\\pi(y \\mid x) \\, \\|\\, q(y \\mid x)\\big) - \\log Z(x) \\right].
$$

### Applying Gibbs' Inequality

The term $\\log Z(x)$ does not depend on $\\pi$, so only the KL divergence can be optimized. Gibbs' inequality tells us that a KL divergence is always non-negative and equals zero only when its two arguments are identical. Therefore the minimizer is $\\pi^{\*} = q$, which gives the optimal policy

$$
\\boxed{\\,\\pi^{\*}(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right).\\,}
$$

This is the closed-form solution to the KL-regularized RLHF objective. It says that the optimal policy reweights the reference model by exponentiated rewards, then normalizes.

## Recovering the Reward from the Policy

We now invert the optimal policy to express the reward in terms of the policy. This is the step that lets DPO replace an explicit reward model with a log-ratio of policy probabilities.

Starting from

$$
\\pi^{\*}(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r^{\*}(x, y)\\right),
$$

take the logarithm of both sides:

$$
\\begin{aligned}
\\log \\pi^{\*}(y \\mid x)
&= \\log \\!\\left( \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r^{\*}(x, y)\\right) \\right) \\\\
&= -\\log Z(x) + \\log \\pi\_{\\text{ref}}(y \\mid x) + \\frac{1}{\\beta} r^{\*}(x, y).
\\end{aligned}
$$

Rearranging to solve for $r^{\*}(x, y)$:

$$
\\frac{1}{\\beta} r^{\*}(x, y) = \\log \\pi^{\*}(y \\mid x) - \\log \\pi\_{\\text{ref}}(y \\mid x) + \\log Z(x).
$$

Multiplying by $\\beta$:

$$
\\boxed{\\,r^{\*}(x, y) = \\beta \\log \\frac{\\pi^{\*}(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x).\\,}
$$

This is the implicit reward. It is a log-ratio of the optimal policy to the reference policy, plus a constant $\\beta \\log Z(x)$ that depends on $x$ but not on $y$. That constant will cancel when we take reward differences, so we do not need to compute it.

## Connecting to Bradley-Terry

Preference data in RLHF is usually modeled with the Bradley-Terry model. For two completions $y\_1$ and $y\_2$ given prompt $x$, the probability that $y\_1$ is preferred is

$$
p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\frac{\\exp\\!\\big(r^{\*}(x, y\_1)\\big)}{\\exp\\!\\big(r^{\*}(x, y\_1)\\big) + \\exp\\!\\big(r^{\*}(x, y\_2)\\big)}.
$$

Substitute the implicit reward $r^{\*}(x, y) = \\beta \\log \\tfrac{\\pi^{\*}(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x)$:

$$
p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\frac{\\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)} + \\beta \\log Z(x)\\right)}{\\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)} + \\beta \\log Z(x)\\right) + \\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_2 \\mid x)}{\\pi\_{\\text{ref}}(y\_2 \\mid x)} + \\beta \\log Z(x)\\right)}.
$$

Every term contains the same factor $\\exp(\\beta \\log Z(x)) = Z(x)^{\\beta}$. Decompose $e^{a+b} = e^a e^b$ and cancel this shared factor:

$$
p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\frac{\\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)}\\right)}{\\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)}\\right) + \\exp\\!\\left(\\beta \\log \\tfrac{\\pi^{\*}(y\_2 \\mid x)}{\\pi\_{\\text{ref}}(y\_2 \\mid x)}\\right)}.
$$

The intractable partition function has disappeared. To reach the sigmoid form, define

$$
\\Delta\_i = \\beta \\log \\frac{\\pi^{\*}(y\_i \\mid x)}{\\pi\_{\\text{ref}}(y\_i \\mid x)}.
$$

Then

$$
p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\frac{e^{\\Delta\_1}}{e^{\\Delta\_1} + e^{\\Delta\_2}}.
$$

Multiply numerator and denominator by $e^{-\\Delta\_1}$:

$$
\\begin{aligned}
p^{\*}(y\_1 \\succ y\_2 \\mid x)
&= \\frac{e^{\\Delta\_1} e^{-\\Delta\_1}}{e^{\\Delta\_1} e^{-\\Delta\_1} + e^{\\Delta\_2} e^{-\\Delta\_1}} \\\\
&= \\frac{1}{1 + e^{\\Delta\_2 - \\Delta\_1}}.
\\end{aligned}
$$

Recognizing the sigmoid $\\sigma(z) = \\tfrac{1}{1 + e^{-z}}$, we obtain

$$
\\boxed{\\,p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\sigma\\!\\left(\\beta \\log \\frac{\\pi^{\*}(y\_1 \\mid x)}{\\pi\_{\\text{ref}}(y\_1 \\mid x)} - \\beta \\log \\frac{\\pi^{\*}(y\_2 \\mid x)}{\\pi\_{\\text{ref}}(y\_2 \\mid x)}\\right).\\,}
$$

This is Bradley-Terry preference probability, but with the policy log-ratios in place of the reward model scores.

## The DPO Loss

We observe preference data in which $y\_c$ is chosen over $y\_r$ for prompt $x$. The maximum-likelihood objective is to maximize $p(y\_c \\succ y\_r \\mid x)$, or equivalently to minimize the negative log-likelihood:

$$
\\mathcal{L}\_{\\text{DPO}}(\\pi\_\\theta; \\pi\_{\\text{ref}}) = -\\mathbb{E}\_{(x, y\_c, y\_r) \\sim \\mathcal{D}} \\!\\left[ \\log p(y\_c \\succ y\_r \\mid x) \\right].
$$

Substituting the sigmoid form with a trainable policy $\\pi\_\\theta$ gives the DPO loss:

$$
\\boxed{\\,\\mathcal{L}\_{\\text{DPO}}(\\pi\_\\theta; \\pi\_{\\text{ref}}) = -\\mathbb{E}\_{(x, y\_c, y\_r) \\sim \\mathcal{D}} \\!\\left[ \\log \\sigma\\!\\left( \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)} \\right) \\right].\\,}
$$

This loss is directly differentiable. There is no reward model, no generation during training, and no RL loop. The optimization pushes the policy to increase the log-probability gap between chosen and rejected completions, measured relative to the frozen reference model.

## The DPO Gradient

Taking the gradient of the loss clarifies what the optimizer actually does. Start with

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\nabla\_\\theta \\, \\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\log \\sigma\\!\\left( \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)} \\right) \\right].
$$

Let

$$
u = \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)}
$$

be the argument of the sigmoid. By the chain rule,

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\frac{\\sigma'(u)}{\\sigma(u)} \\nabla\_\\theta u \\right].
$$

Using $\\sigma'(u) = \\sigma(u)(1 - \\sigma(u))$ and cancelling $\\sigma(u)$:

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\big(1 - \\sigma(u)\\big) \\nabla\_\\theta u \\right].
$$

The reflection identity $1 - \\sigma(u) = \\sigma(-u)$ gives the compact form

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\sigma(-u) \\nabla\_\\theta u \\right].
$$

Now differentiate $u$. Only $\\pi\_\\theta$ depends on $\\theta$; the reference model is frozen, so its log terms drop:

$$
\\nabla\_\\theta u = \\beta \\big[ \\nabla\_\\theta \\log \\pi\_\\theta(y\_c \\mid x) - \\nabla\_\\theta \\log \\pi\_\\theta(y\_r \\mid x) \\big].
$$

Substituting back and writing $\\sigma(-u)$ in full:

$$
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} = -\\beta \\, \\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ \\sigma\\!\\left( \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} \\right) \\cdot \\big( \\nabla\_\\theta \\log \\pi\_\\theta(y\_c \\mid x) - \\nabla\_\\theta \\log \\pi\_\\theta(y\_r \\mid x) \\big) \\right].
$$

This can be summarized as

$$
\\begin{aligned}
\\nabla\_\\theta \\mathcal{L}\_{\\text{DPO}} &= -\\beta \\, \\mathbb{E}\_{(x, y\_c, y\_r)} \\!\\left[ w \\cdot \\big( \\nabla\_\\theta \\log \\pi\_\\theta(y\_c \\mid x) - \\nabla\_\\theta \\log \\pi\_\\theta(y\_r \\mid x) \\big) \\right], \\\\
&\\quad\\text{where}\\quad w = \\sigma\\!\\big(r\_\\theta(x, y\_r) - r\_\\theta(x, y\_c)\\big).
\\end{aligned}
$$

The weight $w \\in (0, 1)$ is larger when the model is more wrong, i.e., when it ranks the rejected response above the chosen. The bracket raises the likelihood of $y\_c$ and lowers the likelihood of $y\_r$. The hyperparameter $\\beta$ scales the step, trading correct ordering against drift from $\\pi\_{\\text{ref}}$.

## Recap of the Derivation

The full chain from the RLHF objective to the DPO loss is:

$$
\\begin{aligned}
\\textbf{Objective:} \\;&\\; \\max\_{\\pi} \\; \\mathbb{E}\\big[r(x, y)\\big] - \\beta \\, \\mathcal{D}\_{\\text{KL}}(\\pi \\, \\|\\, \\pi\_{\\text{ref}}) \\\\
\\Rightarrow\\; \\textbf{Optimal policy:} \\;&\\; \\pi^{\*}(y \\mid x) = \\frac{1}{Z(x)} \\, \\pi\_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{1}{\\beta} r(x, y)\\right) \\\\
\\Rightarrow\\; \\textbf{Implicit reward:} \\;&\\; r^{\*}(x, y) = \\beta \\log \\frac{\\pi^{\*}(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x) \\\\
\\Rightarrow\\; \\textbf{Bradley-Terry:} \\;&\\; p^{\*}(y\_1 \\succ y\_2 \\mid x) = \\sigma\\!\\big(r^{\*}(x, y\_1) - r^{\*}(x, y\_2)\\big) \\quad \\text{($Z$ cancels)} \\\\
\\Rightarrow\\; \\textbf{DPO loss:} \\;&\\; -\\log \\sigma\\!\\left(\\beta \\log \\frac{\\pi\_\\theta(y\_c \\mid x)}{\\pi\_{\\text{ref}}(y\_c \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_r \\mid x)}{\\pi\_{\\text{ref}}(y\_r \\mid x)}\\right).
\\end{aligned}
$$

The reward model never had to be built; it was hiding inside the policy all along.

## Weaknesses and Variants

DPO is simple, but it is not perfect. Understanding its limitations explains why so many variants have been proposed.

### Likelihood Displacement

The DPO loss cares only about the *margin* between chosen and rejected log-ratios, not their absolute values. That means the model can lower the loss by pushing the rejected probability down faster than the chosen probability, even while the chosen probability itself falls. This phenomenon is called **likelihood displacement** (Razin et al. 2025; Ren and Sutherland 2025). It is mediated through the partition function $Z(x)$ in the derivation and may push probability mass toward unaddressed, off-distribution behaviors.

![Sketch of likelihood displacement in DPO.](assets/dpo_displacement.png)

In real runs, the chosen-versus-rejected reward margin can widen while the chosen log-probability drifts downward. Some practitioners add an SFT term on the chosen response, or use fixes such as Cal-DPO (Xiao et al. 2024) and AlphaPO (Gupta et al. 2025).

### The $\\beta$ Parameter and Static KL

The parameter $\\beta$ sets the strength of the KL constraint relative to reward maximization. A large $\\beta$ keeps the policy close to $\\pi\_{\\text{ref}}$; a small $\\beta$ lets the policy deviate and can lead to over-optimization.

Crucially, DPO's final KL distance is *static*: it steps directly to the optimal solution implied by the dataset and the chosen $\\beta$. Online RL, by contrast, takes steps based on freshly sampled batches and can use dynamic KL controllers. This difference is one reason online methods sometimes reach a higher peak performance.

### A Zoo of Direct Alignment Algorithms

Many variants tweak the DPO loss to address specific limitations. I call the broader class **Direct Alignment Algorithms** (DAAs).

- **Identity Preference Optimization (IPO)** (Azar et al. 2024) softens the preference probability away from the strict Bradley-Terry form to curb overfitting:

  $$
  \\mathcal{L}\_{\\text{IPO}} = \\mathbb{E}\_{(x, y\_w, y\_l)} \\!\\left[ \\left( \\log \\frac{\\pi\_\\theta(y\_w \\mid x) \\, \\pi\_{\\text{ref}}(y\_l \\mid x)}{\\pi\_\\theta(y\_l \\mid x) \\, \\pi\_{\\text{ref}}(y\_w \\mid x)} - \\frac{1}{2\\beta} \\right)^2 \\right].
  $$

- **Conservative DPO (cDPO)** (Rafailov et al. 2023), introduced in the original DPO paper, assumes preference labels are noisy and softens the loss target accordingly.

- **Offset DPO (ODPO)** (Amini et al. 2024) requires the chosen completion to beat the rejected completion by a margin offset that scales with preference strength.

Some variants drop the reference model entirely:

- **Odds Ratio Preference Optimization (ORPO)** (Hong et al. 2024) folds an odds-ratio pull toward the chosen response into the SFT loss. Because the odds ratio uses only $\\pi\_\\theta$, no reference model is needed:

  $$
  \\mathcal{L}\_{\\text{ORPO}} = \\mathcal{L}\_{\\text{SFT}} - \\lambda \\, \\mathbb{E} \\!\\left[ \\log \\sigma\\!\\left( \\log \\frac{\\text{odds}\_\\theta(y\_w \\mid x)}{\\text{odds}\_\\theta(y\_l \\mid x)} \\right) \\right], \\quad \\text{odds}\_\\theta(y) = \\frac{\\pi\_\\theta(y)}{1 - \\pi\_\\theta(y)}.
  $$

- **Simple Preference Optimization (SimPO)** (Meng et al. 2025) length-normalizes the reward into an average log-probability and adds a target margin $\\gamma$:

  $$
  \\mathcal{L}\_{\\text{SimPO}} = -\\mathbb{E} \\!\\left[ \\log \\sigma\\!\\left( \\frac{\\beta}{|y\_w|} \\log \\pi\_\\theta(y\_w \\mid x) - \\frac{\\beta}{|y\_l|} \\log \\pi\_\\theta(y\_l \\mid x) - \\gamma \\right) \\right].
  $$

Despite the large literature of tweaks, the algorithm matters far less than the base model and the data.

## Implementation Details

DPO is genuinely simple to implement. The core loss is:

```python
# Log-probability gaps for the policy and the frozen reference model
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# Difference of log-ratios: positive when the policy
# shifts mass toward the chosen completion
logits = pi_logratios - ref_logratios

# DPO loss: negative log-sigmoid drives the policy to
# widen the gap between chosen and rejected
losses = -F.logsigmoid(beta * logits)

# Implicit rewards (detached -- used for logging only)
chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

Two practical notes are worth keeping in mind.

**KL divergence is static.** DPO takes gradient steps toward the optimal solution implied by the dataset and $\\beta$. It does not adapt the KL penalty based on recent batches, unlike online RL methods.

**Cache reference log-probabilities.** Simple implementations run the policy and reference models together, which doubles memory. Because $\\pi\_{\\text{ref}}$ is frozen, its log-probabilities can be precomputed and cached for each training example, cutting peak memory usage roughly in half.

## DAAs with Synthetic Preference Data

Direct alignment algorithms need feedback data, but that feedback does not have to come from humans. Most modern DPO uses preferences labeled by a strong model.

UltraFeedback (Cui et al. 2023) was the first prominent open synthetic preference dataset and helped kickstart the DPO era. Tülu 3 (Lambert et al. 2024), SmolLM 2 (Ben Allal et al. 2025), and other models followed with larger synthetic-feedback recipes.

On-policy data—some completions generated by the model being tuned—helps the contrastive loss optimize the right token space. Tülu 3 emphasized this point specifically. Later work proposed **Delta Learning** (Geng et al. 2025), which argues that the *gap* between chosen and rejected completions matters more than exactly which models produced them. For example, both Olmo 3 and SmolLM 3 used chosen responses from Qwen 3 32B and rejected responses from Qwen 3 0.6B, developed concurrently and independently.

Synthetic preference datasets have their own biases. Frontier judges such as GPT-4 exhibit length bias (Dubois et al. 2024) and a preference for outputs that match their own style (Panickssery et al. 2024). These biases leak into the training data and should be monitored.

## Offline versus Online Preference Optimization

DPO and other DAAs are **offline**: they train on a fixed dataset collected ahead of time. This makes them simpler, more stable, and faster to iterate on. The main limitation is coverage: performance is bounded by the diversity of the static dataset.

PPO and policy-gradient methods are **online**: they generate fresh completions during training and score them with a reward model. Online methods can explore new regions of the completion space and often reach a higher peak performance (Ivison et al. 2024; Xu et al. 2024; Tajwar et al. 2024). The cost is more compute, more moving parts—four models in memory for PPO—and more engineering complexity.

## Where Does DPO Stand Today?

If DPO is so simple and has been used successfully by so many models, why does it seem less central at the frontier? A few hypotheses:

- Olmo 3 (Team Olmo et al. 2025) and SmolLM 3 (Bakouch et al. 2025) used DPO over reasoning traces after SFT, showing that it still works well in simple pipelines.
- NVIDIA Nemotron 3 (NVIDIA 2025) combined DPO with binary classifier optimization in one offline stage, scoring data with a generative reward model.
- Liquid AI LFM2 (2025) used length-normalized DPO on semi-online data before a final RLVR pass.

The broader view is that DPO works well in wonky, distillation-heavy recipes—models with spikier distributions trained on data from many teacher models. It is a reliable path to a good model, but most labs with large engineering resources can reach a higher peak with online RL. As reasoning models continue to be trained primarily with RL, infrastructure for online methods will improve and the performance gap may persist. For many practitioners, however, DPO remains the right place to start.

## Key Takeaways

- DPO solves the same KL-regularized RLHF objective as PPO, but without a separate reward model or RL loop. The policy is trained directly on preference pairs.
- The optimal policy has a closed form: $\\pi^{\*}(y \\mid x) \\propto \\pi\_{\\text{ref}}(y \\mid x) \\exp(r(x, y)/\\beta)$.
- Inverting that solution gives an implicit reward $r^{\*}(x, y) = \\beta \\log \\tfrac{\\pi^{\*}(y \\mid x)}{\\pi\_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x)$.
- Plugging the implicit reward into the Bradley-Terry model cancels the partition function, leaving a sigmoid of policy log-ratio differences.
- The DPO loss is the negative log-sigmoid of that difference. Its gradient raises the likelihood of chosen completions and lowers that of rejected completions, weighted by how wrong the model is.
- DPO has limitations, including likelihood displacement and a static KL distance. Variants such as IPO, ORPO, and SimPO address different aspects, but data and base model quality dominate small algorithmic differences.
- DPO is offline and simple; online RL can reach higher peaks but costs more engineering and compute. DPO remains a practical starting point for preference tuning, especially with synthetic preference data.
