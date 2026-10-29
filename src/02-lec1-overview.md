> Corresponding lecture: Lec 1 — Overview · Chapters 1–3 · Watch/PDF/Slides/Source at [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 1: Overview — Foundations of RLHF and Post-Training

Reinforcement learning from human feedback (RLHF) is the family of techniques that turned large language models from autocomplete engines into useful assistants. It sits inside a broader stage of model development called *post-training*: every training step that happens after the large-scale autoregressive pretraining of a base model. This chapter maps the landscape. We start with what a language model is, trace the short but dense history that made modern post-training possible, introduce the reinforcement-learning formalism that underlies RLHF, walk through the canonical three-step InstructGPT recipe, and end with the two biggest currents in post-training as of 2026: direct preference optimization and reinforcement learning with verifiable rewards (RLVR).

## Language Models and the Transformer Revolution

A language model is, at its core, a model that assigns probabilities to text. It consumes text as a sequence of *tokens* — the model's internal units, which may be whole words, subwords, or characters — and predicts the next token given the tokens that came before. Repeating this prediction produces a completion one step at a time; this is called *autoregressive* generation.

Modern language models have billions to trillions of parameters and are largely descendants of the Transformer architecture, which popularized the *self-attention* mechanism together with fully-connected feed-forward layers (Vaswani et al. 2017). They also work over far more than text: Gemini, ChatGPT, and their peers process images, audio, and video.

![The original Transformer architecture diagram from "Attention Is All You Need" (Vaswani et al. 2017).](assets/transformer.webp)

The modern timeline is short but transformative:

- **2017:** The Transformer is born (Vaswani et al. 2017).
- **2018:** GPT-1 (Radford et al. 2018), ELMo (Peters et al. 2018), and BERT (Devlin et al. 2018) show that large unsupervised or self-supervised language models transfer strongly to downstream tasks.
- **2019:** GPT-2 (Radford et al. 2019) demonstrates that scaling model size and data yields broad, emergent capabilities; the same year, scaling laws (Kaplan et al. 2020) give the field a predictable relationship between compute, data, parameters, and loss.
- **2020:** GPT-3 (Brown et al. 2020) surprises researchers with in-context learning and few-shot prompting: the model can perform new tasks from just a few examples embedded in the prompt.
- **2021:** The "Stochastic Parrots" paper (Bender et al. 2021) crystallizes early concerns about the risks of very large language models.
- **2022:** ChatGPT (OpenAI 2022) makes RLHF-driven conversational AI a mainstream product.
- **2023:** GPT-4 (OpenAI 2023) establishes frontier-scale multimodal models.
- **2024:** OpenAI's o1 family (OpenAI 2024) popularizes reasoning models that spend more compute at inference time.
- **2025:** o3, Claude Code (Anthropic 2025), and OpenAI's agent tools (OpenAI 2025) push models toward long-horizon tool use and coding agents.

![GPT-3 introduced few-shot prompting: tasks are specified by a few in-context examples rather than by fine-tuning (Brown et al. 2020).](assets/few-shot.png)

## Pretraining: Next-Token Prediction

Pretraining is the first and largest training stage. The model is trained on trillions of tokens drawn from the web, books, code, and documents. Frontier models are often trained on 5–50+ trillion tokens; one trillion text tokens is roughly 3–5 TB of data. Labs gather and filter 10–20× more data than actually goes into the model, so the total data funnel is on the order of petabytes. The objective is simple: predict the next token in each sequence. The result is a model with broad world knowledge, language fluency, and flexible capabilities.

![Pretraining as next-token prediction: the model learns to predict each token conditioned on all previous tokens.](assets/pretraining_next_token_tikz.png)

A base model at the end of pretraining is, in effect, a very capable autocomplete engine. Asked "The president of the United States in 2006 was," a base model such as Llama 3.1 405B Base continues:

> George W. Bush, the governor of Florida in 2006 was Jeb Bush, and John McCain was an Arizona senator in 2006...

It completes the sentence and then drifts into related web-style text. Post-training is what turns this continuation behavior into a direct, conversational answer.

## Post-training: From Autocomplete to Assistant

Post-training is the collective name for the stages applied after pretraining to make the model useful in a chat or assistant interface. The stages differ in what they change:

- **Pretraining** builds world knowledge, language fluency, and broad capabilities.
- **Instruction tuning / supervised fine-tuning (SFT)** teaches the model to answer in a question-answer format and often teaches it to reproduce specific token sequences.
- **Preference tuning / RLHF** uses a contrastive loss to modify completions as a whole, making the model richer, more flexible, and better aligned with human preferences.
- **Reinforcement learning with verifiable rewards (RLVR)** enhances the model on questions whose answers can be checked exactly, such as math or code tests; this can translate into more complex, agentic behaviors.

RL-based post-training gives the model richer, *response-level*, contrastive feedback than SFT alone, but SFT remains the foundation for maximum performance.

After post-training, the same prompt receives a focused, assistant-style answer. The post-trained model Tülu 3 405B replies:

> George W. Bush was the president of the United States in 2006. He served two terms in office, from January 20, 2001, to January 20, 2009.

ChatGPT showed what RLHF could do at product scale. Responses quickly acquired better *format* (direct, conversational answers), better *style* (helpful, concise, Markdown), and a better overall *product* experience that people could use every day.

## Reinforcement Learning Basics

To understand RLHF, it helps to start with classical reinforcement learning (RL). An RL problem is usually written as a **Markov Decision Process (MDP)**:

$$(\\mathcal{S}, \\mathcal{A}, P, r, \\gamma)$$

where $\\mathcal{S}$ is the state space, $\\mathcal{A}$ is the action space, $P(s\_{t+1} \\mid s\_t, a\_t)$ are the transition dynamics, $r(s\_t, a\_t)$ is the reward function, and $\\gamma$ is the discount factor. The agent's goal is to maximize cumulative return over a trajectory:

$$J(\\pi) = \\mathbb{E}\_{\\tau \\sim \\pi}\\left[\\sum\_{t=0}^{T} \\gamma^t r(s\_t, a\_t)\\right].$$

In plain language:

- **State**: the current situation the agent is in.
- **Action**: what the agent does next.
- **Reward**: the signal for how good that action was.
- **Policy** $\\pi$: the strategy for choosing actions.
- **Trajectory** $\\tau$: a sequence of states, actions, and rewards, $\\tau = (s\_0, a\_0, r\_0, s\_1, a\_1, r\_1, \\ldots)$.
- **Trajectory probability**: $P(\\tau \\mid \\pi) = p(s\_0) \\prod\_{t=0}^{T} \\pi(a\_t \\mid s\_t)  p(s\_{t+1} \\mid s\_t, a\_t)$.

![The standard RL loop: the agent observes a state, takes an action, the environment returns a reward and next state, and the policy is updated (Sutton & Barto 2018).](assets/rl.png)

**Thermostat.** A simple RL example is a thermostat learning when to turn the heater on or off. The state is the current room temperature; the action is on or off; the reward is positive when the room stays near the target temperature; and the policy is the rule for deciding what to do next. One possible policy is:

$$\\pi(a\_t = \\text{on} \\mid s\_t) = \\begin{cases} 1 & \\text{if } s\_t < 70^\\circ\\text{F} \\0 & \\text{otherwise} \\end{cases}.$$

**CartPole.** A standard benchmark is CartPole. The state is four-dimensional:

$$s\_t = (x\_t, \\dot{x}\_t, \\theta\_t, \\dot{\\theta}\_t),$$

where $x\_t$ is the cart position, $\\dot{x}\_t$ its velocity, $\\theta\_t$ the pole angle, and $\\dot{\\theta}\_t$ the angular velocity. The action is to push the cart left or right:

$$a\_t \\in \\{\\text{left}, \\text{right}\\}.$$

The reward is +1 for every step the pole stays upright, and the episode ends if the pole falls too far or the cart leaves the track:

$$r(s\_t, a\_t) = \\begin{cases} 1 & \\text{if } |\\theta\_t| < 12° \\text{ and } |x\_t| < 2.4 \\0 & \\text{otherwise}. \\end{cases}$$

The dynamics relate the cart mass $m\_c$, pole mass $m\_p$, pole length $l$, gravity $g$, and applied force $F$:

$$\\begin{aligned}
\\ddot{x}\_t &= \\frac{F + m\_p l (\\dot{\\theta}\_t^2 \\sin\\theta\_t - \\ddot{\\theta}\_t \\cos\\theta\_t)}{m\_c + m\_p}, \\\\[6pt]
\\ddot{\\theta}\_t &= \\frac{g \\sin\\theta\_t - \\cos\\theta\_t \\cdot \\ddot{x}\_t}{l}.
\\end{aligned}$$

Classical RL is therefore a **multi-step control problem**: each action changes the next state, and rewards accumulate across a trajectory.

## Why RLHF?

In classical RL, the reward function is known — CartPole gives +1 per step. For many real tasks, however, the reward is **hard to write down**. It is easy to judge which poem is better, but hard to write a masterpiece; easy to spot a helpful answer, but hard to specify "helpful" as a formula; and the most likely continuation under pretraining is not always the most useful one. RLHF lets us optimize for behavior we can *evaluate* even when we cannot easily *specify* the reward.

![Human feedback can produce a natural backflip (left), while a hand-designed reward function can produce an awkward, reward-hacked motion (right) (Christiano et al. 2017).](assets/christiano-backflip-human.webp)

This idea predates language models. **TAMER** (Knox & Stone 2008) had humans score agent actions to learn a reward model. Christiano et al. (2017) applied RLHF to trajectory preferences in Atari. Ziegler et al. (2019) performed the first RLHF on language models. The schematic from Christiano et al. (2017) — a reward model trained from comparisons, then an RL agent optimized against that reward model — is still the core loop of modern RLHF.

![A recreation of the Christiano et al. (2017) RLHF system diagram: human comparisons train a reward predictor, and the agent maximizes predicted reward.](assets/rlhf_schematic_tikz.png)

The path to modern RLHF includes several milestones (Ziegler et al. 2019; Ouyang et al. 2022; Bai et al. 2022; Rafailov et al. 2024; Dubey et al. 2024; Lambert et al. 2024; Guo et al. 2025):

![Timeline of key developments in RLHF, from early preference-based RL through DPO and RLVR (lecture figure).](assets/rlhf_timeline_tikz.png)

Other prominent early work on language models includes:

- **Stiennon et al. (2020)** — Learning to summarize from human feedback, extending Ziegler et al. to long-form summarization with PPO.
- **WebGPT (Nakano et al. 2021)** — trained a model to browse the web and answer questions using RLHF, with citations.
- **Anthropic HH (Bai et al. 2022)** — trained a helpful and harmless assistant with RLHF and introduced the widely used HH dataset.
- **GopherCite (Menick et al. 2022)** — taught language models to support answers with verified quotes.
- **Sparrow (Glaese et al. 2022)** — improved dialogue agents via targeted human judgments, adding rule-based constraints to RLHF.
- **InstructGPT (Ouyang et al. 2022)** — the canonical three-step RLHF recipe behind ChatGPT.

## What RLHF Comparison Data Looks Like

RLHF is trained on *preference pairs*. A human or AI annotator sees two responses to the same prompt and picks the better one. For the prompt "Explain why the sky is blue in one sentence," Response A might say:

> The sky is blue due to Rayleigh scattering, where shorter blue wavelengths of sunlight are scattered more by atmospheric molecules than longer wavelengths.

Response B might say:

> The sky appears blue because of the way light interacts with the atmosphere and stuff, it's basically just physics.

The preference pair $(y\_w, y\_l)$, where $y\_w$ is the winning response and $y\_l$ is the losing response, becomes training data.

## Classical RL versus RLHF

Classical RL and RLHF differ in several ways:

| | Classical RL | RLHF |
|---|---|---|
| **Environment** | Agent takes actions $a\_t$ in states $s\_t$ | Prompts $x$ sampled from a dataset; no environment |
| **Reward** | Known function $r(s\_t, a\_t)$ per step | Learned from human preferences; response-level |
| **Optimization** | Cumulative return over a trajectory | Expected reward minus KL distance from a reference model |
| **Regularization** | Discount $\\gamma$ | KL penalty $\\beta$ |

The classical objective is:

$$J(\\pi) = \\mathbb{E}\_{\\tau \\sim \\pi}\\left[\\sum\_{t=0}^{T} \\gamma^t r(s\_t, a\_t)\\right].$$

The RLHF objective is:

$$J(\\pi) = \\mathbb{E}\\left[ r\_\\phi(x, y) \\right] - \\beta  D\_{\\text{KL}}\\left(\\pi \\mid \\pi\_{\\text{ref}}\\right).$$

![The RLHF loop: a prompt is sampled, the policy generates a completion, the reward model scores it, and the policy is updated while a KL penalty keeps it close to the reference model.](assets/rlhf.png)

## The InstructGPT Three-Step Recipe

The canonical RLHF recipe, established by InstructGPT (Ouyang et al. 2022), has three steps.

### Step 1: Instruction Fine-Tuning (SFT)

Start from the pretrained language model and train it on demonstrations of desired assistant behavior using standard supervised learning on prompt-response pairs. The SFT loss is the negative log-likelihood of the target response tokens:

$$\\mathcal{L}\_{\\mathrm{SFT}}(\\theta) = - \\sum\_{(x, y^\\star)} \\sum\_{t=1}^{|y^\\star|} \\log \\pi\_\\theta \\left(y^\\star\_t \\mid x, y^\\star\_{<t}\\right).$$

After this step the model can answer questions in the desired format. SFT is easy to apply and quickly adapts a base model to many domains.

### Step 2: Reward Modeling

Collect comparisons between two model outputs for the same prompt. (The name RLHF comes from collecting *human* feedback, but today much of this feedback is AI-generated.) Train a reward model $r\_\\phi(x, y)$ to score preferred completions higher.

The probability model says a response should win when it gets a higher reward score:

$$P(y\_w \\succ y\_l \\mid x) = \\sigma\\left(r\_\\phi(x, y\_w) - r\_\\phi(x, y\_l)\\right).$$

Training minimizes the negative log-likelihood of the preferred response beating the rejected one:

$$\\mathcal{L}\_{\\mathrm{RM}}(\\phi) = - \\log \\sigma\\left(r\_\\phi(x, y\_w) - r\_\\phi(x, y\_l)\\right).$$

Here $x$ is the prompt, $y\_w$ is the winning response, $y\_l$ is the losing response, and $r\_\\phi(x, y)$ is the trained reward model. The reward used in RLHF is therefore the model predicting the probability that a given piece of text would be the chosen completion in a pair.

### Step 3: RL against the Reward Model

This is where RLHF gets its name. Sample prompts $x\_i$ from a dataset, generate completions $y\_i \\sim \\pi\_\\theta(\\cdot \\mid x\_i)$ from the model being trained, score them with the reward model $r\_\\phi(x\_i, y\_i)$, add a KL penalty so the policy stays close to the SFT/reference model $\\pi\_{\\mathrm{ref}}$, and update the policy with a policy-gradient algorithm — Proximal Policy Optimization (PPO) in InstructGPT and ChatGPT. The objective is:

$$J(\\pi) = \\mathbb{E}\\left[r\_\\phi(x, y)\\right] - \\beta D\_{\\mathrm{KL}}\\left(\\pi \\mid \\pi\_{\\mathrm{ref}}\\right).$$

![The three-step InstructGPT recipe: SFT, reward modeling, then RL against the reward model (Ouyang et al. 2022).](assets/instructgpt.jpg)

Unpacked, the objective is:

$$\\max\_{\\pi}  \\mathbb{E}\_{x \\sim D, y \\sim \\pi(\\cdot \\mid x)} \\underbrace{r\_\\phi(x, y)}\_{\\text{maximize the reward}} - \\underbrace{\\beta D\_{\\mathrm{KL}}\\left(\\pi(\\cdot \\mid x)\\mid\\pi\_{\\mathrm{ref}}(\\cdot \\mid x)\\right)}\_{\\text{but don't change the model too much}}.$$

The reference model $\\pi\_{\\mathrm{ref}}$ keeps the policy anchored to the SFT model. The KL divergence measures how far the new policy moves from that reference on prompt $x$, and $\\beta$ controls the tradeoff between improving behavior and staying close to what the model already knows.

## Direct Preference Optimization

The RLHF objective above can be optimized more directly. **Direct Preference Optimization (DPO)** (Rafailov et al. 2024) derives the gradient toward the optimal policy $\\pi^\*$ and eliminates the need for a separate reward model by training an implicit reward model. DPO trains directly on preferred ($y\_w$) versus rejected ($y\_l$) responses to a prompt ($x$):

$$\\mathcal{L}\_{\\mathrm{DPO}}(\\theta) = - \\log \\sigma \\left( \\beta \\log \\frac{\\pi\_\\theta(y\_w \\mid x)}{\\pi\_{\\mathrm{ref}}(y\_w \\mid x)} - \\beta \\log \\frac{\\pi\_\\theta(y\_l \\mid x)}{\\pi\_{\\mathrm{ref}}(y\_l \\mid x)} \\right).$$

DPO became very popular because it is far simpler to implement, far cheaper to run, and achieves roughly 80% or more of the final performance of full online RL. It has been used to build models such as Zephyr-Beta, Tülu 2/3, and OLMo 2/3.

## Rejection Sampling and the Preference Tuning Landscape

**Rejection sampling** is the simplest preference optimization method. Generate many completions, score them, and fine-tune on the best:

1. **Generate**: sample $N$ completions per prompt from the current model.
2. **Score**: pass each through the reward model $r\_\\phi(x, y)$.
3. **Select**: keep the highest-scoring completion(s).
4. **Fine-tune**: apply the standard SFT loss on the curated set.

$$\\mathcal{L}\_{\\mathrm{RS}} = \\mathcal{L}\_{\\mathrm{SFT}}\\left(\\theta; \\{(x\_i, y\_i^{\\star})\\}\\right) \\quad \\text{where } y\_i^{\\star} = y\_{i, \\arg\\max\_{j} r\_\\phi(x\_i, y\_{i,j})}.$$

Rejection sampling is simple, stable, and widely used: Llama 2 (Touvron et al. 2023) and DeepSeek R1 (Guo et al. 2025) both include rejection-sampling stages.

The three main preference-tuning methods compare as follows:

| | Rejection Sampling | Online RL (PPO) | DPO |
|---|---|---|---|
| **Mechanism** | Filter, then SFT | Generate, score, update policy | Direct gradient on preferences |
| **Reward model** | Required | Required | Implicit (no separate RM) |
| **On-policy data** | Yes (generate from current model) | Yes (generate each step) | No (fixed preference dataset) |
| **Complexity** | Low | High | Low |

All three optimize the same underlying objective; they differ in how they move the policy toward higher-reward completions. Whether online RL or DPO is best for final performance is debated, and the evidence is mixed.

## Over-Optimization and Goodhart's Law

The reward model is a **proxy**, not ground truth. Even a well-trained reward model is only correlated with real user satisfaction. This leads to **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure."

In practice, over-optimization appears as:

- **Reward hacking**: the RM score keeps climbing while actual quality degrades.
- **Verbosity bias**: longer responses score higher, so models become verbose.
- **Sycophancy**: the model tells users what they want to hear rather than being accurate.
- **Over-refusal**: the model refuses legitimate queries (e.g. "how to kill a linux process").

The KL penalty with coefficient $\\beta$ is the main defense: it limits how far the policy can drift from the reference model. But over-optimization is a **fundamental tension** in all preference-based training (Gao et al. 2023).

## From RLHF to Post-Training

Training recipes have evolved rapidly. A rough comparison:

| | InstructGPT (2022) | Tülu 3 (2024) | DeepSeek R1 (2025) |
|---|---|---|---|
| **Instruction data** | ~10K | ~1M | 100K+ |
| **Preference data** | ~100K | ~1M | On-policy |
| **RL stage** | ~100K prompts | ~10K (RLVR) | N/A |

The overall trend is to use far more compute across all stages, while shifting more emphasis to RLVR.

Early on, RLHF had a well-documented, simple approach. InstructGPT made the classic three-stage recipe canonical — SFT, reward modeling, then RL against the reward model — and became the intellectual template for much of modern post-training. But what began as an "RLHF" recipe has evolved into a complex series of steps to get the final, best model. Modern systems such as Nemotron 4 340B and Llama 3.1 keep the same core idea of using multiple optimizers with different strengths, but add more stages, more data, and more filtering.

Since ChatGPT, the field has moved through roughly four phases:

1. **2023:** Simple SFT for better chatbots and reproducing RLHF fundamentals (Alpaca, Vicuna, etc.).
2. **2024:** DPO dominates open models and training stages expand (Zephyr-Beta, Tülu 2, etc.).
3. **2025:** RLVR and complex recipes (Tülu 3, OLMo 3, Nemotron 3, R1, etc.).
4. **2026:** Agentic training, multi-turn RL, etc.

### Elicitation and the Superficial Alignment Hypothesis

RLHF's reputation was once that it makes only minor contributions to final language models. The paper *LIMA: Less Is More for Alignment* (Zhou et al. 2023) captured this intuition:

> A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users.

This led some to describe post-training as "just style transfer." The superficial alignment hypothesis (SAH) posits that most knowledge is learned during pretraining and post-training merely surfaces it (Vergara-Browne et al. 2026).

The reality is more interesting. A concrete example comes from OLMoE, the Allen Institute for AI's open Mixture-of-Experts model. The same base model family with updated post-training improved its average evaluation score from 38.44 (September 2024) to 45.62 (January 2025) (Muennighoff et al. 2024; Ai2 2025). Base models determine the *ceiling*, and post-training's job is to *reach* it. Simple post-training often does not extract nearly enough performance.

This view can be called the **Elicitation Theory** of post-training: post-training is the craft of pulling out the most useful knowledge already present in the base model. Pretraining builds the chassis; post-training is the hard craft of extracting the most performance from it.

## Reinforcement Learning with Verifiable Rewards (RLVR)

RLVR applies the same RL algorithms to LLMs when the answer can be checked directly, with no need to train a reward model. Math problems can be checked by comparing the final answer; code can be checked by running tests. Because the reward is exact, there is **no proxy objective**. RLVR enables scaling RL compute on reasoning tasks and unlocked **inference-time scaling**: spending more compute at generation time per problem increases performance roughly log-linearly with compute.

RLVR was named by Tülu 3 (Lambert et al. 2024) and popularized by DeepSeek R1 (Guo et al. 2025).

![The RLVR loop: the model generates a reasoning trace and answer, a verifier checks the answer, and the policy is updated from the exact reward.](assets/rlvr-system.png)

Classical RL, RLHF, and RLVR can be compared as follows:

| | Classical RL | RLHF | RLVR |
|---|---|---|---|
| **Reward** | Environment | Learned (proxy) | Verifiable (exact) |
| **State transitions** | Yes | No | No |
| **Reward granularity** | Per-step | Per-response | Per-response |
| **Primary challenge** | Explore-exploit trade-off | Over-optimization | Task generalization |
| **Example** | CartPole | Chat style tuning | Math reasoning |

## The Scaling RL Era of Post-Training

RLVR opened a new scaling dimension. OpenAI's o1-preview announcement included a seminal plot showing that test-time compute — the number of tokens generated per problem — has a log-linear relationship with downstream performance (OpenAI 2024). This can be done with one long chain-of-thought sequence, multiple agents in parallel, or a mix of both. Improving inference-time scaling changes the slope and offset of the curve.

![OpenAI's o1 test-time scaling plot: downstream performance grows log-linearly with inference compute (OpenAI 2024).](assets/o1-test-time.png)

An often underplayed part of the o1 release is that **training-time scaling with RL** also has a log-linear return on performance. This creates a two-sided scaling landscape for language models: pretraining and post-training both scale, with inference as the third scaling dimension (no weight updates there).

![OpenAI's o1 training-time scaling plot: RL training compute also scales performance log-linearly (OpenAI 2024).](assets/o1-train-time.png)

Recent work reinforces this trend. Cursor Composer 1.5 reported RL scaling curves (Cursor Team 2026). DeepSeek R1-Zero showed that RL alone on reasoning tasks can produce strong emergent behavior (Guo et al. 2025). OLMo 3.1, one of the few fully open large-scale RL runs, trained a general 32B reasoning model for about **28 days on 224 GPUs**; performance improvements were consistent across the run and were still increasing when the run had to stop (Team Olmo et al. 2025).

![OLMo 3.1 extended RL run: performance continued to improve consistently over 28 days on 224 GPUs (Team Olmo et al. 2025).](assets/olmo31-extended-rl.jpeg)

Where this leaves the field: post-training and RLHF are changing faster than ever. Language models are becoming "tool-use native" and are now as much about tools, harnesses, and execution environments as about weights. Human preferences have not gone away, but they are evolving more slowly and out of the central gaze of the industry. Building language models and doing research is changing rapidly with coding agents.

## Key Takeaways

- A language model assigns probabilities to text and generates autoregressively; pretraining on trillions of tokens gives broad knowledge and fluency, but the resulting base model is essentially an autocomplete engine.
- Post-training turns a base model into an assistant through instruction tuning, preference tuning / RLHF, and increasingly through RLVR.
- Classical RL is formulated as an MDP and optimizes cumulative discounted reward over multi-step trajectories; RLHF replaces the environment reward with a learned reward model and operates on response-level rewards with a KL penalty.
- The canonical InstructGPT recipe has three steps: supervised fine-tuning, reward-model training on preference pairs, and RL (typically PPO) against the reward model.
- DPO optimizes the same objective as RLHF directly from preference pairs without a separate reward model, making it simpler and cheaper; rejection sampling is another simple alternative.
- Reward models are proxies, so over-optimization is a fundamental tension; the KL penalty is the main defense.
- Post-training can be viewed as *elicitation*: base models contain latent capabilities, and the craft of post-training is to surface and amplify the most useful behaviors.
- RLVR uses exact, verifiable rewards for reasoning domains and has unlocked both inference-time scaling and training-time scaling of RL, marking a new scaling era for post-training.
