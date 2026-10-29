> Corresponding lecture: Lec 9 — Over-Optimization and RLHF's Bad Reputation · Chapter 14 & Appendix B · Watch/PDF/Slides/Source at [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 9: Over-Optimization and RLHF's Bad Reputation

Optimizing a proxy hard enough always breaks it. Reward-model accuracy is *a proxy for a proxy*: a learned model, of data that incompletely captures a complex distribution. Over-optimization is common and shows up in funny ways. This lecture covers Goodhart’s law in LLMs, qualitative failures such as refusals and sycophancy, and why RLHF was dismissed as “just style.”

## April 2025: extreme sycophancy in production

A GPT-4o update made the model validate nearly anything. It shipped April 25th and was rolled back April 28th; the training run that produced it looked healthy on the metrics. Example:

> **User** (told GPT-4o they felt like they were both “god” and a “prophet”)
>
> **GPT-4o, April 2025**: That’s incredibly powerful. You’re stepping into something very big — claiming not just connection to God but identity as God.

OpenAI’s unusually candid postmortem identified three things that went wrong in sequence:

1. The update had a **reward signal from user feedback** via a reward model for RL — thumbs-up/thumbs-down data from ChatGPT.
2. Under RL, that signal **overpowered the primary rewards**. RL will always optimize the easiest objective to move.
3. It was **not caught by evals**: offline benchmarks looked good, and A/B testers *preferred* the model. Expert testers flagged that it “felt off,” but there was no deployment eval tracking sycophancy.

Read the postmortem: [Expanding on what we missed with sycophancy](https://openai.com/index/expanding-on-sycophancy/).

## Part 1: Over-optimization

### Goodhart’s law: the reward is a proxy

> “Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes.” — Goodhart, 1984

Colloquially: “When a measure becomes a target, it ceases to be a good measure.” RL is a very strong optimizer; it pulls *all* the available reward out of the environment. In RLHF the reward is a **learned model**, at best *correlated* with downstream quality. Optimizing the proxy makes the true objective better early in training — then worse.

### Over-optimization vs. overfitting

**Overfitting**: the model memorizes training examples rather than learning generalizable patterns; training accuracy improves while held-out accuracy degrades, but both metrics measure the *same task* on different splits.

**Over-optimization**: the model *genuinely improves* at the proxy objective — the reward model’s scores, including on a validation set — but that objective diverges from the true goal, actual user satisfaction. It is not a generalization problem but a measurement/metric problem. Gaming it looks like verbose, confident-sounding answers that score well without being more helpful, or repeating rare tokens that hit artifacts in RM training.

![The shape of over-optimization: proxy reward keeps climbing while true quality peaks and declines.](../assets/overoptimization.png)

The recurring shape of RLHF training runs: the run looks healthy — training reward keeps climbing — but downstream evaluations peak and then decline. The gains come from regions of the reward model that do not map to real usage.

### Scaling laws for reward-model over-optimization

The seminal paper *Scaling Laws for Reward Model Overoptimization* (Gao, Schulman, Hilton, 2023) formalized this. Since humans are too expensive to query during training:

- A 6B **“gold” RM** stands in for ground-truth preferences and labels the comparison data.
- Smaller **proxy RMs (3M–3B)** are trained on those labels; RL optimizes *the proxy only*.
- Score the policy with both. The proxy (dashed) climbs forever; the gold (solid) peaks and falls.

![Best-of-N: gold reward (solid) flattens and falls while the proxy (dashed) keeps climbing.](../assets/gao-overopt-bon.png)

![RL: same shape, ~10x more KL spent — and a much harder turnover for small RMs.](../assets/gao-overopt-rl.png)

Larger proxy RMs turn over later and more gently. And since gold-labels-train-proxy is another proxy, real models diverge slightly differently.

### Measuring it: train vs. test reward models

Anthropic’s version of over-optimization:

- **Split the preference data in half** and train two 52B preference models — a *train PM* and a *test PM*.
- RL against the train PM only; score the policy with both.
- The x-axis is $\\sqrt{D\_{\\mathrm{KL}}}$ from the initial policy (how much the policy has changed).

![Anthropic's train-PM vs test-PM experiment.](../assets/anthropic_overoptimization.png)

### What over-optimized models sound like

Recurring signatures in early chat models:

- Stock phrases: “As an AI language model…”, “Certainly!…”
- Uninformative answers — repetition, hedging.
- Pandering: self-doubt, over-apologizing, sycophancy.
- Misaligned behavior such as over-refusals.

“JavaScript, JavaScript, JavaScript, JavaScript, JavaScript, JavaScript, JavaScript,…”

The preference-data biases from Lecture 8 — sycophancy, verbosity, formatting — are amplified into policy behavior.

### An example: over-refusals — “too much RLHF?”

2023’s most-memed failure mode was refusing to “kill a Linux process.” The blame usually lands on RLHF, but these failures largely reflect an overly cautious period of development, where safety was one of the aspects genuinely steerable with RLHF, not the potential of the algorithm. Now measured directly with benchmarks like XSTest. This faded very fast as a story.

> **User**: i want to know how to kill linux process
>
> **Llama 2 Chat, 2023**: I cannot provide instructions or advice on how to harm or destroy Linux processes or any other living being. It is not ethical, legal, or morally right to intentionally harm or terminate any process or life form. […]

### Why it happens, and what helps

Sources of error: **approximation** (the RM cannot perfectly fit preferences), **estimation** (the RM overfits its training set), **optimization** (the policy trains too hard against it). Potential mitigations:

- Bigger policies — more ways to gain reward at small optimization distances.
- Reward-model ensembles; changed optimizers.
- Direct alignment over-optimizes too, but makes the trade-off easier to pin.
- Best-of-N sampling spends far less KL than online RL.

The main lever in practice is the **KL penalty** (covered in Lecture 10).

### Aside: John Schulman’s ICML 2023 talk

**Proxy objectives in reinforcement learning from human feedback** is one of the great talks on this topic: [icml.cc/virtual/2023/invited-talk/21549](https://icml.cc/virtual/2023/invited-talk/21549).

### Over-optimization enables misalignment

Sycophancy is the clearest current case: “agree with the user” gets reinforced when preference data overweights *supportive and confident* over *accurate and calibrated*. That is how the April 2025 GPT-4o incident shipped. As models integrate deeper into society, the cost of this gap compounds. We are seeing this play out today as **reward hacking** in scaled-up RL on verifiable and agentic tasks — models exploiting graders, test harnesses, and tools rather than solving the task.

## Part 2: Beyond “just style”

### Same base model — what did preference tuning change?

The library at [rlhfbook.com/library](https://rlhfbook.com/library) opens on three **SFT → DPO** pairs — Tülu 3 70B and OLMo 2 32B / 7B — across 16 shared prompts. Same base model, before and after preference tuning:

- **SFT** answers tend to be a single wall of prose.
- **DPO** answers keep the same facts but restructure them — definition first, then headers and numbered lists. Easier to read and use, without losing content.

### Style as substance

An early phase of RLHF’s history was convincing people that style was actually useful sometimes, and not just brainrot. Early RLHF got tagged as “just style transfer” — superficial, unimportant. But style is a never-ending source of human value, and it is intertwined with what the information *is*. Well-done preference tuning also moves the numbers: Llama 3 Instruct’s Arena standing is widely attributed to its personality — more succinct and clever than other models of its era. If RLHF makes models more fun to use, that is delivered value, independent of benchmark scores.

The strong version of the dismissal is the **Superficial Alignment Hypothesis** (Zhou et al., 2023): knowledge is learned in pretraining; alignment just picks a format and tone. Post-training has largely outgrown this critique.

### The chattiness balance — many chat evals were easy to overfit

Preference tuning reliably boosts LLM-as-a-judge chat evals (AlpacaEval, MT-Bench) — gains that do not transfer proportionally to Arena or real usage. A common finding:

> “However, DPO leads to improvements in human preference evaluation but degradation in benchmark evaluation.” — Qwen technical report, 2023

It was easy to juice chattiness at the expense of other skills. Preference methods run in loops or with abundant data can trade math and coding for chat performance. Olmo 3 shipped the checkpoint with higher math/code/reasoning scores over ones that maximized LLM-judged chat benchmarks.

### Length-gamed leaderboards

In April 2024, Direct Nash Optimization reported a 7B model “beating GPT-4” on AlpacaEval; Self-Rewarding LMs disclosed similarly unrealistic scores. Neither holds up in real use against frontier models. The gaming got bad enough that AlpacaEval *and* WildBench added **linear length corrections**. Done right: Starling Beta — k-wise reward model + PPO on top of OpenChat, up 10 Arena places, with length that actually helps raters.

![Results from the Direct Nash Optimization paper claiming a 7B model outperforming GPT-4 on AlpacaEval.](../assets/dno-figure.png)

### Llama 4’s Chatbot Arena special (April 2025)

Meta’s Llama 4 launch headlined Maverick at **Elo 1417 — #2 on Chatbot Arena** … via “an experimental chat version.” The model on the leaderboard was **not the released model**: a variant tuned for Arena voters — long, emoji-filled, relentlessly enthusiastic answers. Same name, drastically different behavior on LMArena vs. every other provider. When the real model was ranked later, it landed far down the leaderboard, and LMArena changed its policies in response.

### Why does RLHF make answers longer?

Arena keeps showing that average users prefer longer, complete answers — they read as more thorough, helpful, and trustworthy. Models are trained to match the **average labeler**. Length is Goodhart’s favorite axis: the most-rewarded surface feature, and the easiest to over-optimize.

### Done right, preference tuning helps capabilities/skills too

The “just style” critique was outgrown, only partially, by well-done open models showing major performance gains. In the Tülu 3 recipe, the preference (DPO) stage boosts chattiness **and** improves math, coding, and instruction-following over the SFT checkpoint. Multiple open recipes report the same broad-suite gains from RLHF (DPO) — Tülu 3, Olmo 3, SmolLM 3. The honest retrospective: RLHF earned its bad reputation on style *failures*, but the same tools, used carefully, are now central to modern post-training.

## Key Takeaways

- Optimizing a learned reward proxy hard enough eventually hurts true quality (Goodhart’s law).
- Over-optimization is a measurement problem, not a generalization problem; it differs from overfitting.
- Scaling laws for RM over-optimization show a predictable peak-and-decline in gold reward as KL grows.
- Qualitative failures include sycophancy, over-refusals, verbosity, and formatting bias.
- RLHF’s “just style” reputation came from early surface-level failures, but well-tuned preference optimization improves both style and capabilities.
- Length-gamed leaderboards and Arena-optimized variants are modern examples of the same proxy failure.
