> Corresponding lecture: Lec 0 — The ML Foundations of LLM Post-Training · Watch/PDF/Slides/Source at [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 0: The ML Foundations of LLM Post-Training

This chapter is a short, optional refresher. No prior reinforcement-learning background is assumed. We expect comfort with basic machine-learning ideas—losses, gradients, and probability—plus enough PyTorch fluency to read tensor shapes. The goal is to review the handful of concepts that the rest of the course uses every day: autoregressive language models, the cross-entropy training objective, gradient-based fine-tuning, KL divergence, the sigmoid preference model, and the reinforcement-learning framing that turns all of these into post-training.

By the end of this chapter you should be able to:

- read the autoregressive factorization of a language model and connect it to `log_softmax` and cross-entropy;
- trace a single training example from token IDs through logits, log-probs, masking, and the loss;
- explain why KL divergence appears in almost every alignment method;
- interpret a preference probability as a sigmoid over a score difference;
- cast text generation as a Markov decision process and state the RL objective that post-training optimizes.

## Language models

### What a language model is

A language model learns the joint probability of a sequence of **tokens**—words or subwords—in an **autoregressive** manner: each prediction depends only on the tokens before it. Given a sequence $x=(x\_1,\\ldots,x\_T)$, the model factorizes the sequence probability using the chain rule:

$$P\_\\theta(x)=\\prod\_{t=1}^{T} P\_\\theta(x\_t \\mid x\_1,\\ldots,x\_{t-1}).$$

Modern chat models such as ChatGPT, Claude, and Gemini are **decoder-only Transformers** built on **self-attention**. In a decoder-only model, each position attends only to itself and to earlier positions; this is what enforces the autoregressive structure.

![Next-token prediction: a decoder-only language model predicts one token at a time from the prefix before it.](assets/pretraining_next_token_tikz.png)

The same Transformer backbone can carry different heads for different tasks. In this course the first new head we encounter is a **reward-model head** (Chapter 5): same backbone, new head, new objective.

### The LM head

The Transformer backbone outputs a hidden-state vector for every input token. The **LM head** is a final linear projection from the hidden dimension back to the vocabulary size. In one forward pass:

- hidden state $\\xrightarrow{\\text{LM head}}$ **logits** (one real-valued score per vocabulary token);
- **softmax** over the logits $\\rightarrow$ a probability distribution over the next token;
- sampling or argmax from that distribution $\\rightarrow$ the generated token.

### Softmax and log-probabilities

At generation step $t$, the logits are $z^{(t)} \\in \\mathbb{R}^{|\\mathcal{V}|}$. The softmax turns them into the next-token distribution:

$$P\_\\theta(x\_t \\mid x\_{<t}) = \\mathrm{softmax}(z^{(t)})\_{x\_t}
= \\frac{e^{z^{(t)}\_{x\_t}}}{\\sum\_{j\\in\\mathcal{V}} e^{z^{(t)}\_j}}.$$

A whole sequence is the product of these per-step terms. In log-space the product becomes a sum:

$$\\log P\_\\theta(x)
= \\log \\prod\_{t=1}^{T} P\_\\theta(x\_t \\mid x\_{<t})
= \\sum\_{t=1}^{T} \\log P\_\\theta(x\_t \\mid x\_{<t}).$$

Each term in the sum is the `log_softmax` of the logit read off at the true next token. That term is exactly the negative cross-entropy loss at that position. We work in log-space for two reasons: **numerical stability** (products of many small probabilities underflow) and because sums are easier to differentiate than products.

### Anatomy of one LM training example

Lectures repeatedly talk about shifting logits, gathering target log-probs, and masking. Here is the full tensor-shape story for one example:

```text
input_ids  ∈ ℕ^{B × L}          # input to model: token IDs, batch B, length L
   → model(input_ids).logits ∈ ℝ^{B × L × V}     # one vector per position, V = vocab
   → log_softmax + gather    ∈ ℝ^{B × (L-1)}     # log-prob of each observed token
   → (× completion_mask).sum ∈ ℝ^{B}             # one completion log-prob per sequence
```

And the same path in PyTorch, one line at a time:

```python
logits = model(input_ids).logits            # (B, L, V): a score for every vocab token at every position
logits = logits[:, :-1, :]                  # drop the last position — it has no next token to score
labels = input_ids[:, 1:]                   # target at each position is the actual next token (the shift)
completion_mask = completion_mask[:, 1:]    # shift mask to line up with labels (1 = completion token)
log_probs = logits.log_softmax(dim=-1)      # normalize logits into log-probabilities -> (B, L-1, V)
token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # log-prob of each true next token -> (B, L-1)
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)           # zero prompt/pad, sum per sequence -> (B,)
```

Reading the code from top to bottom: the model produces logits for every position; we discard the last logit because it has no observed next token; we shift the token IDs and mask by one so that position $t$ supervises the prediction for token $t+1$; `log_softmax` turns logits into normalized probabilities; `gather` selects the log-prob of the true token; and the mask ensures only completion tokens are summed.

The one-token shift is the autoregressive step made literal: the logits at position $t$ predict token $t+1$. Summing the per-token log-probs and calling `loss.backward()` turns that sum into the corresponding sum of per-token gradients.

### Three masks commonly come up in post-training code

Masking is where silent bugs live in LLM code. Three different masks do three different jobs:

- **Causal mask** (built into decoder attention): position $t$ cannot attend to positions $>t$. This is what makes the model autoregressive; ordinary training code never touches it.
- **Attention / padding mask**: tells attention to ignore padding tokens when variable-length sequences are batched together. Shape `(B, L)`.
- **Completion / loss mask**: value `1` only on completion tokens. The loss is multiplied by it before summing. **Prompt tokens condition the prediction but contribute no loss.**

Get the completion mask wrong and you either train on the prompt—wasting gradient on tokens you cannot change—or on padding, which trains on noise.

### A small decoding review

Generation samples or searches under the same policy $\\pi\_\\theta$; the weights are fixed, only the selection procedure changes:

- **Greedy decoding**: take $\\arg\\max\_v P\_\\theta(v \\mid x\_{<t})$ at each step. Deterministic but myopic: a locally best token can doom the rest of the sequence.
- **Beam search**: keep the top-$b$ partial sequences by cumulative log-prob, expand, and prune. Excellent for low-entropy tasks such as translation; often bland for open-ended text.
- **Temperature sampling**: sample $y\_t \\sim \\mathrm{softmax}(z\_t / T)$. Higher $T$ gives more diverse completions; lower $T$ sharpens the distribution toward greedy.
- **Truncated sampling**: restrict sampling to the top-$k$ tokens, or to the smallest set with cumulative mass at least $p$ (top-$p$ / nucleus sampling).
- **Lookahead search** (MCTS and related methods): score simulated future continuations, often with a value or reward model, before committing to a token.

The key split is between **the distribution** $\\pi\_\\theta$ and **the procedure** that samples it.

Temperature has a clean information-theoretic interpretation: it rescales the logits and therefore changes the entropy of the output distribution. As $T\\to 0$ the distribution collapses to the greedy argmax; as $T\\to\\infty$ it approaches a uniform distribution over the vocabulary.

## Training an LM: Cross-entropy

At each position, cross-entropy compares the model's predicted distribution to the true next-token label $q\_t$. Because the label is one-hot at the actual token $x\_t$, the sum over the vocabulary collapses to a single term:

$$H(q\_t,\\,P\_\\theta)
= -\\sum\_{v\\in\\mathcal{V}} q\_t(v)\\,\\log P\_\\theta(v \\mid x\_{<t})
= -\\log P\_\\theta(x\_t \\mid x\_{<t}).$$

Summing over positions and averaging over the dataset gives the standard language-modeling loss, also called the **negative log-likelihood (NLL)**:

$$\\mathcal{L}\_{\\text{LM}}(\\theta)
= -\\,\\mathbb{E}\_{x\\sim\\mathcal{D}}\\left[\\sum\_{t=1}^{T} \\log P\_\\theta(x\_t \\mid x\_{<t})\\right].$$

"Predict the next token" is supervised learning; the label is simply the next token in the sequence.

### Cross-entropy as maximum likelihood

Because the empirical distribution of the data places all its mass on the observed token, minimizing the average cross-entropy over the dataset is equivalent to maximizing the log-likelihood of the data under the model. This is why the LM objective is also called **maximum likelihood estimation**, and why "predict the next token" is the canonical pretraining task.

## Optimization and fine-tuning

### Gradient descent and the training loop

Deep-learning training repeats one step:

$$\\theta \\leftarrow \\theta - \\eta \\,\\nabla\_\\theta \\mathcal{L}(\\theta),$$

where $\\theta$ are the model parameters, $\\eta$ is the learning rate, and $\\nabla\_\\theta\\mathcal{L}$ is the gradient of the loss. In practice, **AdamW** and related adaptive optimizers are standard for language-model training rather than plain SGD. Training is measured in optimizer **steps** or **tokens**; **epochs** are a useful unit for finite fine-tuning datasets. The learning-rate schedule matters as much as the rate itself.

The basic backpropagation loop is shared across pretraining, mid-training, and fine-tuning. What differs are masking, sampling, batching, data construction, and systems—not the optimizer.

### Pretraining → mid-training → SFT

The boundary at the end of pretraining is fuzzy and very important for post-training:

- **Pretraining**: next-token cross-entropy on massive raw web and code data; **every** token contributes to the loss.
- **Mid-training**: the annealing / high-quality-data phase at the very end of pretraining, before any instruction data. Quality ramps up here. It can be considered part of post-training, but it is often discussed separately.
- **SFT (supervised fine-tuning)**: still cross-entropy, but now on **instruction–response** pairs with a **completion mask**. Only the response tokens contribute to the loss; the prompt merely conditions the model. The model learns to *respond*, not just to continue text.

The exact point at which pretraining ends is therefore a design choice. Some teams treat the annealing phase as the final part of pretraining; others treat it as the first part of post-training. What matters for RLHF is that the starting checkpoint already has high-quality general knowledge and fluent next-token behavior, so the post-training stages can focus on eliciting useful responses.

Modern post-training pipelines start from a **mid-trained** checkpoint, so this is where the course's story effectively begins.

## Probability essentials

### KL divergence (and entropy)

The **Kullback–Leibler divergence** measures how one distribution differs from another:

$$\\mathcal{D}\_{\\text{KL}}(p\\,\\|\\,q)=\\sum\_i p\_i \\log\\frac{p\_i}{q\_i}.$$

Three properties to keep in mind:

- **Asymmetric**: $\\mathcal{D}\_{\\text{KL}}(p\\|q)\\neq\\mathcal{D}\_{\\text{KL}}(q\\|p)$; direction matters.
- **Non-negative**, and zero only when $p=q$.
- **Central to alignment**: the RLHF KL penalty (Lectures 3–4) keeps the policy close to its SFT start, and the DPO loss (Lecture 6) is derived from a **KL-regularized reward-maximization** objective.

Related: **entropy** $H(p)=-\\sum\_i p\_i \\log p\_i$ is the uncertainty of a distribution—uniform means high entropy, peaked means low entropy. It appears in definitions, derivations, and in understanding how these systems behave.

### Cross-entropy, entropy, and KL as one identity

Three quantities form one information-theoretic picture (measured in *nats* with $\\ln$, *bits* with $\\log\_2$):

- **Entropy**: $H(p)=-\\sum\_i p\_i\\log p\_i$ — the average surprise of samples from $p$; the irreducible cost to encode them.
- **Cross-entropy**: $H(p,q)=-\\sum\_i p\_i\\log q\_i$ — the cost of encoding samples from $p$ using a code built for the *wrong* distribution $q$.
- **KL divergence**: $\\mathcal{D}\_{\\text{KL}}(p\\|q)=\\sum\_i p\_i\\log\\frac{p\_i}{q\_i}$ — precisely the *extra* cost of using $q$ instead of $p$.

They satisfy:

$$H(p,q)=H(p)+\\mathcal{D}\_{\\text{KL}}(p\\|q).$$

So minimizing cross-entropy is minimizing KL to the truth: $H(p)$ is fixed by the data. In language-model training the label $p$ is one-hot, so $H(p)=0$ and **cross-entropy = KL = NLL**. In RLHF the KL term is instead added *on purpose*, to keep the policy near its reference.

### Estimating KL over sequences

Over full sequences the exact sum behind KL is intractable, but KL is an **expectation**, so we estimate it by Monte Carlo: average the log-ratio over samples drawn from the policy.

$$\\mathcal{D}\_{\\text{KL}}(\\pi\_\\theta\\,\\|\\,\\pi\_{\\text{ref}})
= \\mathbb{E}\_{x\\sim\\pi\_\\theta}\\!\\left[\\log\\frac{\\pi\_\\theta(x)}{\\pi\_{\\text{ref}}(x)}\\right]
\\approx \\frac{1}{N}\\sum\_{i=1}^{N}\\log\\frac{\\pi\_\\theta(x\_i)}{\\pi\_{\\text{ref}}(x\_i)}.$$

- The integrand is just the **log-probability ratio** on each rollout you already generate.
- The naive per-token average $\\hat k\_1=\\log\\frac{\\pi\_\\theta}{\\pi\_{\\text{ref}}}$ is unbiased but high-variance and can go negative on a finite sample.
- RLHF implementations often use the low-variance, always-non-negative **$k\_3$ estimator** $\\hat k\_3 = r - 1 - \\log r$, with $r=\\frac{\\pi\_{\\text{ref}}}{\\pi\_\\theta}$ (Schulman 2020).

This is the practical face of the information theory: approximate a target distribution from samples, paying for the estimate in variance rather than in an intractable sum.

### Sigmoid and pairwise likelihood

A preference such as "response $y\_w$ is better than $y\_l$" is a **binary** outcome, so a score difference is squeezed through a **sigmoid** into a probability:

$$\\sigma(z)=\\frac{1}{1+e^{-z}},\\qquad
P(y\_w\\succ y\_l\\mid x)=\\sigma\\!\\left(r(x,y\_w)-r(x,y\_l)\\right).$$

Three things to recognize:

- The sigmoid converts a **score difference** into a binary probability.
- Only **relative** reward differences matter—shifting both rewards by a constant changes nothing.
- $-\\log\\sigma(\\cdot)$ is just **binary negative log-likelihood**.

Lecture 2 derives this Bradley–Terry model in full to train a reward model; Lecture 6 reuses the exact same $\\sigma(\\Delta r)$ shape inside DPO.

## Reinforcement learning framing

### The standard RL objective

Reinforcement learning studies an agent acting in a **Markov decision process** $(\\mathcal{S},\\mathcal{A},P,r,\\gamma)$. In the standard setting, an agent takes actions $a\_t$ sampled from a policy $\\pi(a\_t\\mid s\_t)$, the environment evolves according to transition dynamics $p(s\_{t+1}\\mid s\_t,a\_t)$, and the agent receives a reward $r(s\_t,a\_t)$. The policy and dynamics together induce a distribution over trajectories $\\tau=(s\_0,a\_0,s\_1,\\ldots)$:

$$p\_\\pi(\\tau)=\\rho\_0(s\_0)\\prod\_{t=0}^{T-1}\\pi(a\_t\\mid s\_t)\\,p(s\_{t+1}\\mid s\_t,a\_t).$$

Across a finite episode of horizon $T$, the agent's objective is to maximize discounted expected return:

$$\\max\_\\pi \\; \\mathbb{E}\_{\\tau\\sim p\_\\pi}\\left[\\sum\_{t=0}^{T-1}\\gamma^t r(s\_t,a\_t)\\right],$$

where $\\gamma\\in[0,1]$ trades off near-term versus future reward. The expected return for a policy is often written $J(\\pi)$.

For continuing tasks, one often takes $T\\to\\infty$ and relies on discounting ($\\gamma<1$) to keep the objective well-defined. Language generation is usually treated as a single-turn episodic problem, which is why $\\gamma=1$ is common in RLHF.

### A simple example: the thermostat

To make the RL loop concrete before mapping it to language, consider a thermostat keeping a room at $70^\\circ$F:

- **State $s\_t$**: current room temperature, e.g. $65^\\circ$F.
- **Action $a\_t$**: turn the heater on or off.
- **Reward $r$**: $+1$ when the temperature is within $2^\\circ$ of the target, $0$ otherwise.
- **Transition**: the room warms when the heater is on and cools when it is off.
- **Policy $\\pi$**: a rule that decides whether to turn the heater on given the current temperature, e.g.

$$\\pi(a\_t=\\text{on}\\mid s\_t)=\\begin{cases}1 & \\text{if } s\_t<70^\\circ\\text{F}\\\\0 & \\text{otherwise.}\\end{cases}$$

Initially the policy is essentially random, but through trial and error it discovers that turning the heater on when the room is cold and off when it is warm earns more reward. This is the core RL loop: observe, act, receive reward, and update the policy to earn more reward over time.

### A classic control example: CartPole

For a richer example with continuous dynamics, consider the classic *CartPole* (inverted pendulum) control task that appears in many RL textbooks, courses, and research papers. Whereas the thermostat had a single state variable and a binary action, CartPole involves four continuous state variables and physics-based transitions, making it a standard benchmark for RL algorithms.

- **State ($s\_t$)**: the cart position/velocity and pole angle/angular velocity:

  $$s\_t = (x\_t,\\,\\dot{x}\_t,\\,\\theta\_t,\\,\\dot{\\theta}\_t).$$

- **Action ($a\_t$)**: apply a left/right horizontal force to the cart, e.g. $a\_t \\in \\{-F, +F\\}$.

- **Reward ($r$)**: a simple reward is $r\_t = 1$ each step the pole remains balanced and the cart stays on the track (e.g. $|x\_t| \\le 2.4$ and $|\\theta\_t| \\le 12^\\circ$), and the episode terminates when either bound is violated.

- **Dynamics / transition ($p(s\_{t+1}\\mid s\_t,a\_t)$)**: in many environments the dynamics are deterministic (so $p$ is a point mass) and can be written as $s\_{t+1} = f(s\_t,a\_t)$ via Euler integration with step size $\\Delta t$. A standard simplified CartPole update uses the constants cart mass $m\_c$, pole mass $m\_p$, pole half-length $l$, and gravity $g$ ($\\alpha$ is a mass-normalized intermediate with acceleration units):

  $$\\alpha = \\frac{a\_t + m\_p l\\,\\dot{\\theta}\_t^2\\sin\\theta\_t}{m\_c + m\_p}$$

  $$\\ddot{\\theta}\_t = \\frac{g\\sin\\theta\_t - \\cos\\theta\_t\\,\\alpha}{l\\left(\\tfrac{4}{3} - \\frac{m\_p\\cos^2\\theta\_t}{m\_c + m\_p}\\right)}$$

  $$\\ddot{x}\_t = \\alpha - \\frac{m\_p l\\,\\ddot{\\theta}\_t\\cos\\theta\_t}{m\_c + m\_p}$$

  $$x\_{t+1}=x\_t+\\Delta t\\,\\dot{x}\_t,\\quad \\dot{x}\_{t+1}=\\dot{x}\_t+\\Delta t\\,\\ddot{x}\_t,$$
  $$\\theta\_{t+1}=\\theta\_t+\\Delta t\\,\\dot{\\theta}\_t,\\quad \\dot{\\theta}\_{t+1}=\\dot{\\theta}\_t+\\Delta t\\,\\ddot{\\theta}\_t.$$

This is a concrete instance of the general setup above: the policy chooses $a\_t$, the transition function advances the state, and the reward is accumulated over the episode.

### Language models as an MDP

For text generation, the standard MDP pieces are mapped as follows:

| MDP concept | Language model |
|---|---|
| State $s\_t$ | Prompt + tokens so far: $(x, y\_{<t})$ |
| Action $a\_t$ | Next token $y\_t$ |
| Transition $P$ | Deterministic: append the token |
| Policy $\\pi\_\\theta(a\_t\\mid s\_t)$ | LM next-token distribution |
| Reward $r$ | Usually **terminal**: RM score or verifier output |
| Discount $\\gamma$ | Typically $1.0$ (no discounting) |

The policy is the language model, factorized over tokens:

$$\\pi\_\\theta(y\\mid x)=\\prod\_t \\pi\_\\theta(y\_t\\mid x,y\_{<t}).$$

Because the transition is deterministic—appending a token always produces the same next state—the trajectory distribution for a language model reduces to the product of policy choices, with no environmental-dynamics term. This is exactly the autoregressive factorization from the language-modeling section. The difference is the objective: instead of maximizing the likelihood of fixed text, RL maximizes a scalar reward assigned to the whole completion.

### The goal of RL, and why for language models

The goal of RL is to choose policy parameters $\\theta$ that maximize expected reward under the policy itself:

$$J(\\theta)=\\mathbb{E}\_{y\\sim\\pi\_\\theta}[R(x,y)].$$

For language models, a human preference or verifier is a single scalar at the end of a response. That signal is:

- **Sparse**: one label per whole response, not per token.
- **Delayed**: the quality of a token depends on the whole completion.
- **Non-differentiable across discrete tokens**: ordinary gradients cannot pass back through the sampled token.

Because there is no cross-entropy target for "preferred," policy gradients optimize the expected evaluator score anyway. The standard recipe first *learns* that signal as a reward model, then *optimizes* it—possibly with a KL penalty that keeps the policy close to its starting point:

$$\\max\_\\pi \\; \\mathbb{E}\_{y\\sim\\pi\_\\theta}[r\_\\theta(x,y)] - \\beta\\,\\mathcal{D}\_{\\text{KL}}(\\pi\_\\theta(\\cdot\\mid x)\\,\\|\\,\\pi\_{\\text{ref}}(\\cdot\\mid x)).$$

The hyperparameter $\\beta$ controls the strength of the KL budget: a larger $\\beta$ keeps the policy closer to the reference, while a smaller $\\beta$ gives the optimizer more freedom to chase reward. We will return to this trade-off in Lecture 10.

That is the whole motivation for RLHF.

## From these foundations to post-training

Everything in this chapter is basic machinery—probabilities, losses, gradients, and sampling. Post-training is what you get when you point that machinery at a pretrained model with richer feedback than raw next-token prediction:

- **Demonstrations** ("produce this response") → supervised fine-tuning.
- **Preferences** ("make this response more likely than that one") → reward modeling or preference optimization (DPO).
- **Rewards** ("sample responses that score highly") → reinforcement learning (PPO, GRPO, RLVR).
- **Constraints** ("don't drift too far from where you started") → regularization, especially the KL penalty.

Different data and different objectives; the same underlying goal:

> **Change the model's distribution over responses toward behavior we want.**

The rest of the course is about how to do that reliably.

| Signal | Objective | Typical data | What it changes |
|---|---|---|---|
| Demonstrations | Cross-entropy (SFT) | Prompt–response pairs | Format and basic instruction following |
| Pairwise preferences | Reward-model training or DPO | $y\_w \\succ y\_l$ comparisons | Style and subtle human preferences |
| Scalar rewards | RL (PPO, GRPO, RLVR) | Rollouts + rewards | Targeted behavior and reasoning |
| KL penalty | Added to the objectives above | Reference policy | Regularization, prevents drift |

### The canonical three-stage recipe

The earliest widely documented RLHF recipe, established by InstructGPT (Ouyang et al. 2022), turns the pieces above into a concrete pipeline. Starting from a base language model trained on large-scale web text, it applies three stages:

1. **Instruction tuning on roughly 10K examples**: the model is shown prompt–response pairs in a question-answering format so it learns the *shape* of how to answer.
2. **Reward-model training on roughly 100K pairwise preferences**: a copy of the instruction-tuned model is fitted to human preference comparisons, producing a scalar reward signal.
3. **RLHF optimization on roughly 100K prompts**: the instruction-tuned model generates completions, the reward model scores them, and a policy-gradient algorithm updates the generator toward higher reward while a KL penalty keeps it near its starting point.

This recipe is the foundation of modern post-training. Contemporary recipes add mid-training annealing, larger preference datasets, direct-alignment methods such as DPO, and reinforcement learning with verifiable rewards, but the same primitives—cross-entropy, KL divergence, sigmoid preferences, and policy gradients—appear throughout.

### What post-training changes

Post-training does not rewrite the model from scratch; it reweights the distribution over completions. A base model asked "The president of the United States in 2006 was" will continue the sentence and drift into unrelated web text, for example:

> George W. Bush, the governor of Florida in 2006 was Jeb Bush, and John McCain was an Arizona senator in 2006 — who later lost to Obama. September 1 — U.S. President Bush signs an executive order ...

A post-trained model answers the question directly:

> George W. Bush was the president of the United States in 2006. He served two terms in office, from January 20, 2001, to January 20, 2009.

That difference—turning next-token continuation into targeted, useful behavior—is exactly what the foundations in this chapter make possible.

## If any felt rusty

- **Reinforcement learning**: Sutton & Barto, *Reinforcement Learning: An Introduction*; David Silver's UCL course; UC Berkeley CS285.
- **Language models from scratch**: Sebastian Raschka, *Build a Large Language Model (From Scratch)*.
- **The book's own reference**: Appendix A of *Reinforcement Learning from Human Feedback* covers the language-modeling overview, KL, and RL vocabulary used throughout.

Go down rabbit holes, skip around, and chase what excites you—the course is built for that.

## Key Takeaways

- A language model factorizes $P\_\\theta(x)$ as a product of next-token probabilities; in log-space this becomes a sum of `log_softmax` terms, which is the negative cross-entropy loss.
- Training is a one-token shift: logits at position $t$ predict token $t+1$; prompt tokens are masked out and only completion tokens contribute to the loss.
- KL divergence, entropy, and cross-entropy are linked by $H(p,q)=H(p)+\\mathcal{D}\_{\\text{KL}}(p\\|q)$. In LM training the one-hot label makes cross-entropy equal to NLL; in RLHF the KL term is added deliberately as a regularizer.
- A pairwise preference is modeled as $\\sigma(\\Delta r)$; only relative reward differences matter, and $-\\log\\sigma(\\Delta r)$ is binary negative log-likelihood.
- Text generation can be viewed as an MDP in which the LM is the policy, tokens are actions, and the reward is usually terminal and sparse; this is why policy gradients are needed when the target is not a fixed sequence.
- Post-training reuses the same optimizer and sampling machinery as pretraining, but changes the data and the objective to steer the model toward useful behavior.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.
- Schulman, J. (2020). Approximating KL Divergence. *Blog post*. <http://joschu.net/blog/kl-approx.html>
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., ... (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35, 27730–27744.
