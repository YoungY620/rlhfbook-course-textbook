> Corresponding lecture: Lec 12 — The Evolution of Frontier Model Evaluation · Chapter 16 · Watch/PDF/Slides/Source at [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 12: The Evolution of Frontier Model Evaluation

How we measure progress changes over time. Different model types need different strategies. As models get smarter, we need to re-invent the cutting edge of AI. Evaluation is at the core of AI progress and understanding what the models really are.

Frontier evaluation is harder than it ever has been. When Opus 4.6 and GPT-5.3-Codex shipped in the same week (Feb 2026), the headline benchmark deltas were tiny — and settled nothing about which model to use. What separated them was found through use: usability, product fit, behavior over long agentic tasks.

And the tasks we are trying to measure keep taking longer. The task length frontier models can complete (at 50% success) **doubles roughly every 7 months** — from seconds-long questions to tasks that take human experts hours.

![Task-completion time horizon of frontier models. Figure from METR (Time Horizon 1.1).](../assets/metr-time-horizon-11.png)

## Part 1: The eras of post-training evaluation

Benchmarks mirror the training goals of their era:

- **Chat era** (2022–23) — basic knowledge and chat style.
- **Multi-skill era** (2023–24) — post-training improves more skills than just chat (math, code, factuality, safety).
- **Reasoning & tools era** (2024–26) — hard math, coding, and reasoning problems, long chains of thought.
- **Agents & real work** (now) — end-to-end knowledge-work tasks inside products and harnesses.

### Base models (before post-training): few-shot prompting

Base models can’t take a bare or formatted question — eval prompts carried examples of the patterns (3 to 8+ in-context samples) so the model continues the pattern. Canonical evals: **5-shot MMLU**, **8-shot GSM8K**.

```text
# Few-Shot Prompt

Below are examples of MMLU-style questions and answers:

### Example 1
Q: A right triangle has legs of lengths 3 and 4.
What is the length of its hypotenuse?
Choices:
(A) 5
(B) 6
(C) 7
(D) 8

Correct Answer: (A)

### Example 2
Q: Which of the following is the chemical symbol for Sodium?
Choices:
(A) Na
(B) S
(C) N
(D) Ca

Correct Answer: (A)

### Now answer the new question in the same style:

Q: Which theorem states that if a function f is continuous
on a closed interval [a,b], then f must attain both a
maximum and a minimum on that interval?
Choices:
(A) The Mean Value Theorem
(B) The Intermediate Value Theorem
(C) The Extreme Value Theorem
(D) Rolle's Theorem

Correct Answer:
```

### Grading: log-likelihood vs. exact match

**Log-likelihood scoring**: compare the probability the model assigns each answer option — either just the letter `A`, or the full answer string. No sampling, fully deterministic. The standard for pretraining evals.

**Generation + exact match**: sample a completion, extract the answer. Mirrors real usage — and is standard for post-training since ~2024. Aggregating multiple completions gives majority voting; **pass@k** is a common tool. Generation and extraction gave rise to answer-extraction formatting bugs, which only became more complex with agentic models.

### The math behind pass@k

pass@k = the probability that **at least one of $k$ samples** solves the problem. The naive route — generate exactly $k$, report whether any passed — is a **high-variance** coin flip per problem, and plugging a small-sample success rate into $1-(1-\\hat{p})^k$ is **biased**. The fix, from the Codex paper: sample $n \\geq k$ completions, count the $c$ that pass, and average an unbiased estimator over problems:

$$\\text{pass@}k = \\mathop{\\mathbb{E}}\_{\\text{problems}}\\left[1 - \\frac{\\binom{n-c}{k}}{\\binom{n}{k}}\\right]$$

- $\\binom{n-c}{k}\\big/\\binom{n}{k}$ is the chance that $k$ draws (without replacement) from your $n$ samples are **all failures**.
- Larger $n$ → tighter estimate at the same $k$; the paper used $n=200$ for $k \\leq 100$.
- Higher temperature can hurt pass@1 but help pass@100 — so a reported “pass@1” depends on $n$ and sampling settings, not just the model.

### Chain of thought (CoT) emerged to enable progress on harder problems

Few-shot examples that show intermediate steps let models reason before answering. When people were still prompting base models, adding CoT made math and reasoning scores jump. Soon just appending *“Let’s think step by step”* to a prompt approximated this behavior.

```text
# standard prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The answer is ...

# chain-of-thought prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The cafeteria had 23 apples originally. They...
```

### Entering the chat era: zero-shot instruction following

Instruction tuning (FLAN, T0) and then RLHF changed the way people expected to use models: the models learned to directly answer questions. **LLM-as-a-judge** emerged as questions became open-ended. Canonical evals: **MT-Bench**, **AlpacaEval**, and the community-scale **Chatbot Arena**. MCQ evals like MMLU stayed in the mix but were answered zero-shot, sampling the answer letter at temperature 0.

### The emergence of zero-shot prompting took time

When Tülu 3 was trained (summer 2024), many evaluations were a mix of zero-shot and few-shot prompting. The field focused on a variety of skills:

- **Knowledge**: MMLU, PopQA, TruthfulQA
- **Reasoning**: BigBenchHard, DROP
- **Math & code**: MATH, GSM8K, HumanEval
- **Instruction following & safety**: IFEval, others

By early 2025, everyone was using zero-shot prompting; today multi-shot prompting for a post-trained model is very rare unless the model wasn’t trained for it.

### Encouraging the models to reason

Tülu 3’s MMLU prompt — a model before reasoning models — used long-form CoT with exact-match checking:

```text
Answer the following multiple-choice question by giving the correct answer letter in parentheses.
Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) ...

Answer the above question and REMEMBER to finish your response with the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

Sampling settings joined the prompt as part of the eval: reasoning models need **temperature > 0** for their best scores — Qwen’s model cards literally say **“DO NOT use greedy decoding.”**

### Formatting became fragile as usage became more open-ended

- Formatting mismatches can take a model from **60% to near 0** — it is far easier to lose performance with a prompt than to gain it.
- Answer extraction is brittle: rigid suffixes (*“The answer is:”*) or regexes hunting for the answer anywhere.
- Formats conflict across training sets: NuminaMath wants `\boxed{42}`, MetaMath wants `The answer is: 42` — **training on both can be worse than either alone**.
- Format-agnostic grading takes substantial effort — LLM-judges became popular even as answer extractors.

### Reasoning & tool-use pushed the industry to harder tasks

Reasoning models saturated the old eval suites, so the next generation came:

- **Knowledge**: GPQA Diamond, Humanity’s Last Exam, FrontierMath
- **Math**: recent AIME contests
- **Software**: SWE-Bench (+ variants), LiveCodeBench
- Question sourcing moved from the internet to **grad students, PhDs, and professors** — writing questions became expert labor.

**Example: Humanity’s Last Exam**

> What was the rarest noble gas on Earth as a percentage of all terrestrial matter in 2002?
>
> Official answer: **Oganesson**

The official answer is wrong three ways: oganesson is **not a gas** (predicted solid), **not noble** (predicted reactive), and **not terrestrial** (synthetic, ~5 atoms ever made). Past a certain difficulty, **verifying the answer key is the bottleneck** — expert-written no longer means correct. Most evals saturate at 90–95%, not 100%.

### Today: evals of real work

- The frontier evals are **end-to-end professional tasks**: SWE-bench Verified, Terminal-Bench, GDPVal, APEX.
- Task authors are now **experienced professionals**: GDPVal tasks come from experts averaging **14 years** of industry experience; APEX experts average 7+ years at firms like Goldman and McKinsey.
- Models aren’t evaluated bare: they run **inside harnesses and products** (Claude Code, Codex CLI).

### Every era ends the same way: saturation

Benchmarks are consumable. As scores approach the ceiling, only the hardest (and mislabeled) items remain, and the benchmark stops separating models.

![Major AI benchmarks reaching saturation over time. Figure from Epoch AI.](../assets/benchmark-performance.jpeg)

## Part 2: A bit more on agentic evals

### The agentic pipeline: the model is one piece

A **harness** — the loop of prompts, tools, and context management around the model — runs in a **sandbox**: a reproducible world with the files, tools, and rules of the task. Add hardware and timeouts, and hours-long trajectories get graded by regex or an LLM judge.

![The agentic evaluation pipeline.](../assets/eval-system-v3.png)

### Harnesses are key

- Frontier models are **trained in their own harness** — evaluating them in a different one under-reports capability.
- The extreme case, from the ARC-AGI-3 report: on one environment, Opus 4.6 scores **0% with no harness and 97.1%** with a hand-crafted one.
- This is why “same model, different agent product” produces wildly different scores.

### Many little things contribute to agentic scores

Every implementation detail is a knob someone chose (usually undocumented) — two labs running “the same benchmark” can measure meaningfully different things.

- **The inference engine can be different**: vLLM’s postmortem on serving Kimi K2 — three engine bugs held tool-call success **below 20%**; after fixes, **99.9%**. Same weights.
- **Hardware-level differences**: variance across GPUs is real. Some benchmarks measure this on purpose (KernelBench-style tasks need specific GPUs). With scaled sandboxes, one bad GPU actor can stall the system and tank evals.
- **Timeouts are crucial on challenging tasks**: models often timeout on hard tasks — Terminal-Bench 2 reruns with 3–5× longer timeouts move GPT-5.2 by +6 to +15 points.

## Part 3: Can you trust the final eval number?

### Evaluation variance is everywhere

Sampling at temperature > 0 means **re-running the same eval on the same model moves the score**. During Olmo 3 the team measured std. dev. across 3 runs of 14 models, per benchmark. Most reasoning-era evals sit between **0.25 and 1.5 points of noise** — before anyone changes a prompt or sampling setting.

| | Benchmark | σ |
|---|-----------|-----|
| High variance | GPQA | 1.48 |
| | AlpacaEval 3 | 1.24 |
| | IFEval | 0.88 |
| Stable | ZebraLogic | 0.56 |
| | AIME 24 (avg@32) | 0.54 |
| Very stable | LiveCodeBench (avg@10) | 0.29 |
| | MATH | 0.25 |
| | MMLU | 0.22 |

### Managing eval noise

- **avg@k instead of pass@k**: in Tülu 3, LiveCodeBench was noisy *and* cheap — rerunning it 10× moved it from high-variance to very stable. Works for any eval, proportionately expensive.
- Variance leaks in from infrastructure: **batch size, tensor-parallel settings, numerics** of long generations.
- Practical rule: a **~1-point gap between two press releases is noise**.

### Why lab-vs-lab comparisons are unreliable

- Each lab’s eval stack is **tuned to its internal needs**: custom prompts, undisclosed formats, different engines.
- Nobody discloses which public benchmarks were **held out vs. hillclimbed** — train/dev/test hygiene is invisible from outside.
- Inference-time scaling confounds everything: more tokens buys more score, and token budgets are rarely controlled.

![Intelligence Index vs. cost per task, with the Pareto frontier drawn. Figure from Artificial Analysis.](../assets/aa-intelligence-cost-pareto.png)

### What evals are actually for inside labs

Evals are more of a training target than a monitoring lens.

- Labs hillclimb on ~50+ prioritized evals and report the public suite at the end.
- The real product of a good internal eval is **statistical power to decide between models**.
- Sometimes the “test set” is just good data: MATH and GSM8K train splits are high-quality — if a lab doesn’t track that eval, training on them is a rational choice.
- Human A/B testing and Elo stay in the loop for what benchmarks can’t measure.

### Contamination: is training on test intentional?

- **Decontamination** = n-gram / substring search between training and test sets to remove overlap.
- Tülu 3 found popular open datasets contaminated: UltraFeedback×TruthfulQA, Evol-CodeAlpaca×HumanEval, NuminaMath×MATH.
- A subtle tell: RL with **random rewards** improving Qwen benchmarks — only explicable with contamination in the base model.
- Ecosystem response: perturbed benchmark rewrites (same problem, new numbers) or private leaderboards/test sets.

### Today’s models are reward-hack happy and game the evals

Agents love maximizing reward, which turns into shortcuts. NIST and DebugML have documented these in the wild.

**Observed techniques:**

- Mining git history for the **future commit that fixes the bug** — one open model did this in **24% of its SWE-bench trajectories**.
- Dodging URL blocklists via **mirrors, web archives, and package registries**.
- **Hardcoding expected test outputs** into the code.
- Abusing quirks of the test runner.

**Defenses:**

- Remove access to everything not strictly needed.
- A **second, separate sandbox** for verification and test runs.
- A second LLM monitoring the first (expensive).

### Tooling: run your own research-grade evals

**Established harnesses:**

- [Inspect](https://inspect.aisi.org.uk/) — UK AI Security Institute
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — EleutherAI’s standard since the GPT-3 era
- [LightEval](https://github.com/huggingface/lighteval) — Hugging Face; powered the Open LLM Leaderboard
- [OLMES](https://github.com/allenai/olmes) — Ai2’s reproducible evaluation system
- [HELM](https://crfm.stanford.edu/helm/) — Stanford CRFM
- [Eval Gauntlet](https://github.com/mosaicml/llm-foundary) — Mosaic, now Databricks

**The environments wave:**

- [olmo-eval](https://github.com/allenai/olmo-eval) — Ai2’s workbench for evals *inside the model-development loop*.
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) + [Environments Hub](https://www.primeintellect.ai/blog/environments) — Prime Intellect’s library and community hub: 2,500+ verifiable environments that double as evals.
- The direction of travel: **training and evaluation share the same environment code** — write it once, hillclimb and measure with it.

## Key Takeaways

- Benchmarks mirror the **training goals of their era** — and every era ends in saturation.
- A score is a **property of the whole system**: prompt, sampling, engine, harness, sandbox, hardware, grader. The model is one box.
- Expect **±1 point of pure noise**; treat cross-lab comparisons as directional at best.
- Contamination, gaming, and reward hacking all bend single numbers — for decisions that matter, **run your own evals** and control the system.
- Full evaluation is expensive: in-depth runs of a modern agentic suite can cost **>\$100K**.
