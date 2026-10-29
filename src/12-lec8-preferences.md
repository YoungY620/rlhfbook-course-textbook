> Corresponding lecture: Lec 8 — On "Preferences" and Preference Data · Chapters 10 & 11 · Watch/PDF/Slides/Source at [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 8: On "Preferences" and Preference Data

Most of post-training so far — SFT and RLVR — has had a notion of *correct*: a math answer, a passing test, a verifiable reward. Preferences are what we reach for when there is **no right answer**: which poem is better, which explanation is clearer, which tone is kinder. Preferences became the core of RLHF as the alignment community looked to optimize complicated objectives like “human values.”

This lecture traces where “preferences” came from, argues why the formulation is **imperfect**, then digs into the **preference data** used for today’s models.

## Part 1: A short history of optimizing or measuring preferences

### Many fields converged into RLHF

RLHF is the meeting point of philosophy, economics, optimal control, and reinforcement learning — each with its own idea of what a “preference” is. The paper *The History and Risks of Reinforcement Learning and Human Feedback* (Lambert et al., 2023) traces these borrowings and asks what breaks in the borrowing: RLHF quietly treats **costs, rewards, and preferences** as interchangeable when they are not.

### Utility, from logic to a number

The idea that choices can be *scored* is old:

- **Port Royal Logic** (1662): decision quality = outcome weighted by its probability.
- **Bentham’s hedonic calculus** (early 1800s): weigh all of life on one complicated but common scale.
- **Ramsey**, *Truth and Probability* (1931): first to quantify preference *and* belief together as the way individuals make probabilistic decisions.

> “To judge what one must do to obtain a good or avoid an evil, it is necessary to consider not only the good and evil in itself, but also the probability that it happens or does not happen.” — The Port Royal Logic, 1662

### Von Neumann–Morgenstern utility theorem (1947)

Von Neumann and Morgenstern showed that if preferences obey completeness, transitivity, continuity, and independence, they can be represented by a single **utility function**, and rational choice = maximizing expected utility. This is the result RLHF leans on to justify fitting a scalar reward. In RLHF, essentially none of those *ifs* hold: preferences **drift**, are **context- and framing-dependent**, can be **intransitive** at high complexity, and are **multidimensional** but squashed into one number.

### Bradley–Terry (1952): comparisons to scores

The statistical model that turns comparisons into scores — and became the backbone of reward modeling — is Bradley–Terry:

$$ P(y\_w \\succ y\_l \\mid x) = \\sigma\\!\\big(r(x,y\_w) - r(x,y\_l)\\big) = \\frac{e^{\\,r(x,y\_w)}}{e^{\\,r(x,y\_w)} + e^{\\,r(x,y\_l)}} $$

Give it pairwise human comparisons; out comes a scalar reward. This is *why* RLHF needs **preference data**, and where the imperfections enter.

### One scalar preference, many features

A reward model compresses into a single number: helpfulness, honesty, harmlessness, tone, format, length, taste; the annotator’s psychology, culture, and interface; whatever the *framing* of the comparison nudged them toward. We then optimize hard against that number, which is impossible to do perfectly — there will always be trade-offs.

### Where utility theory breaks down

Social choice and economics found limits almost as soon as utility was formalized:

- **Arrow’s impossibility theorem** (1950): no voting rule aggregates individual preferences into a collective one while satisfying basic fairness criteria.
- **Sen** (1973): choice ≠ preference; revealed-preference theory is too thin.
- **Hirschman** (1984): people have *preferences over their preferences* — so preferences may be unmeasurable.

If aggregating preferences is the hard part, **social choice theory** is the natural lens for alignment. A 2024 position paper argues social choice should guide how we aggregate diverse human feedback: whose preferences count, how to combine disagreeing labelers, and so on.

Preferences are also not stable objects over time and context: they drift with time, mood, and experience, and choices are shaped by situation and framing, not just an inner ranking. This is a problem for “collect a label, train on it later repeatedly.”

### The other root: optimal control & RL

In parallel, machinery for *optimizing* a reward matured:

- **Bellman** (1957): MDPs and dynamic programming.
- **Sutton** (1988): temporal-difference learning for credit assignment.
- **Watkins** (1992): Q-learning.
- **DQN** (2013) and **AlphaGo / AlphaZero** (2017): deep RL at scale and mastery from self-play.

These guarantees assume a **single, closed-form reward**. A learned reward model is a *moving, noisy proxy*, not a ground-truth reward. Inverse RL — learning a reward from behavior — is conceptually close but oddly absent from RLHF practice. We inherit RL’s optimizers without inheriting its guarantees.

### Costs ≠ rewards ≠ preferences

These three are **ontologically different**, and modern post-training treated them as interchangeable:

- **Costs** come from control: physical, measurable, given, often with clear optimality/bounds.
- **Rewards** are from psychology and RL convenience: a scalar signal to maximize.
- **Preferences** are human, relational, and unstable — not obviously a scalar at all.

Reducing preferences to rewards made the optimization format tractable, but it is the root cause of many of the unsolvable “biases” in RLHF and preference data.

## Part 2: Preference data — the trade-offs in practice

### Why collect preference data

It is far easier to **judge** than to **generate**: humans (and models) can reliably say which of two answers is better long before they can write the better one. But collecting and processing such data well is the most **opaque** part of the modern post-training pipeline. Very few open models ship fully open human preference data *with* the methodology used to collect it; the best examples are NVIDIA’s **HelpSteer** datasets behind the open Nemotron models.

### Example data collection interfaces

Interface shapes the preference it captures:

- **Research data collection** (Anthropic’s early Claude models): review the full conversation, then rate.
- **A/B testing in production** (ChatGPT user data): two completions from different beta models served side by side; the answers are often very close, a reminder that preference data is noisy.
- **Pairwise with ties** (Chatbot Arena public evaluation): a tie option is allowed.
- **A single bit** (Ai2 demos; many other products): up/down voting.
- **Pick-from-many** (default in image models): selecting among generated images.

### Rankings vs. ratings

**Ratings** score one completion in isolation (e.g., 1–5) and are useful as metadata. **Rankings** are relative comparisons, often on a Likert scale: early Claude used an 8-point scale; UltraFeedback pairs high- vs low-rated completions. In practice almost everyone trains on pairwise rankings binarized to chosen/rejected for the Bradley-Terry loss, and keeps ratings on the side.

A **Likert scale** records preference as an *ordered, graded* judgment — not just which answer wins, but **by how much**, on a symmetric scale with an optional neutral middle.

| 5-point with a tie (LMArena-style) |  |  |  |  |
|:---:|:---:|:---:|:---:|:---:|
| A much better | A better | tie | B better | B much better |

| 8-point without a tie (early Claude) |  |  |  |  |  |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A>>>B | | | A>B | B>A | | | B>>>A |

How many points, and whether ties are allowed, are design choices that change the data you collect.

### Structured (synthetic) preference data

In domains with structure, preference pairs can be built automatically:

- **Math**: a correct solution ≻ an incorrect one.
- **Instruction following (IFEval-style)**: prompt twice — with the constraint and without — and prefer the one that obeys it. The constraint is checked with code, so the label is free.

In these narrow domains, structured pairs can beat quality-judged preferences. This is a simple form of *synthetic* preference data.

### Beyond pairwise preferences

The pairwise comparison is a convention, not a law. Alternatives include:

- **Directional / single-bit** labels (thumbs up/down), trained with Kahneman–Tversky Optimization (KTO).
- **Token-level / fine-grained** feedback: label specific tokens or spans as good or bad.
- **Natural-language** feedback: written critiques instead of a label.

Richer signal, harder collection, and not often used extensively in practice.

### Sourcing & contracts

Access is relationship-driven: vendors are supply-limited and favor big budgets and known brands. Millions get spent and partly wasted; few teams have the bandwidth to fully use human data. Contracts often restrict release of some data. Even for simple human preference data, you need a robust, existing post-training recipe to plug it into.

![A typical multi-batch human-data contract (~\$500K): an early ramp where goals and methodology narrow, with much of the first batches thrown out.](../assets/pref-data-timeline.png)

### Guiding data collection: labeler instructions

Once a contract is signed, buyer and vendor agree on **detailed instructions** for every task — normally never seen outside the lab. One of the first *public* examples was OpenAI’s **InstructGPT** labeler instructions (2022), pages of guidance on ranking **helpful, truthful, harmless** — the spec that actually shapes the labels. A mirrored PDF is available at [rlhfbook.com/assets/instructgpt-instructions.pdf](https://rlhfbook.com/assets/instructgpt-instructions.pdf).

![InstructGPT "Final labeling instructions" (OpenAI, 2022).](../assets/instructgpt-instructions.jpg)

### A dataset we bought: No Robots

On Hugging Face’s H4 team, we commissioned human data the same way the labs did. **No Robots** is 10K expert human-written demonstrations, matching the InstructGPT distribution, paid for from a vendor and then released openly: [hf.co/datasets/HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots). It came from the era/team of the **Zephyr** models and the **Open LLM Leaderboard**.

### Preference *evals* are now a standalone business

In 2026, **Arena** (formerly LMArena) reached a ~\$100M annualized revenue run-rate within ~8 months of launching its enterprise A/B testing offering.

## Part 3: Open questions in preference data

### Bias: what to watch for in data

Subtle, systematic biases sail straight from the data into the model:

- **Prefix bias** — the opening disproportionately drives the label.
- **Sycophancy** — agreeing with the user over being right.
- **Verbosity** — longer rated higher.
- **Formatting** — lists and bold look “better.”
- **Flattery / fluff** — decorative language inflates scores.

Detecting and controlling these biases is central to collecting high-quality preference data — and they are exactly what over-optimization amplifies into model behavior.

### Higher-level complexities

- **Collection context** — do workplace labels transfer to end users? Paid vs. volunteer? Do annotators follow instructions or their own values?
- **Type of feedback** — does a binary pair actually capture the preference we mean? What structure mirrors how people really compare?
- **Population & demographics** — who labels? Is disagreement **noise or signal**?
- **Are the preferences even expressed in the models?**

### The unaudited gap: spec → data → behavior

RLHF’s *motivation* (align to human preference) has drifted from its *practice* (make models effective). Because industrial RLHF is closed, we cannot check whether the trained model actually reflects the spec given to annotators. The **Model Spec** documents intended behavior, but the link from data → behavior stays largely unaudited. Many of these questions already surfaced with synthetic data: the human/AI feedback balance, and on-policy preference data.

The nature of preferences is one of the least-settled, most human parts of the field — and a great academic problem.

## Key Takeaways

- Preferences fill the gap when there is no verifiable “correct” answer, but the scalar-reward formulation is imperfect.
- RLHF borrows from philosophy, economics, control, and RL, treating costs, rewards, and preferences as interchangeable when they are not.
- Bradley–Terry turns pairwise comparisons into a scalar reward, compressing many features and biases into one number.
- Preference data collection is opaque and shaped by interface, scale, and contracts; labeler instructions are the hidden spec.
- Structured synthetic preferences and alternatives (KTO, token-level, natural-language feedback) exist but are less common.
- Biases such as sycophancy, verbosity, and formatting are amplified by over-optimization.
