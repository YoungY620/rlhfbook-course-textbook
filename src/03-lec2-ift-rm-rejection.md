> Corresponding lecture: Lec 2 — IFT, Reward Models, & Rejection Sampling · Chapters 4, 5, 9 · Watch/PDF/Slides/Source at [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 2: IFT, Reward Models, and Rejection Sampling

This chapter covers three tools that form the simplest complete path from a pretrained language model to a preference-tuned model with a reward model: instruction fine-tuning (IFT), reward modeling, and rejection sampling. They are less flashy than full online reinforcement learning, but they are the foundations on which later methods are built. Instruction tuning teaches the model the conversation format, reward models learn to score quality from comparisons, and rejection sampling uses those scores to curate better training data. Together they are a strong baseline and a natural stepping stone to the policy-gradient and direct-alignment methods covered in later lectures.

Understanding these three stages in detail matters because almost every modern post-training recipe still uses them in some form. Frontier models may add online RL, reasoning data, or character training, but they almost always begin with instruction tuning, train or reuse a reward model, and at some point filter or distill with rejection sampling. The design choices made here—chat template, data mix, reward-model architecture, and selection strategy—directly shape the behavior of the final system.

## Instruction Tuning

Instruction fine-tuning, also called supervised fine-tuning (SFT), is the first post-training stage. A pretrained model is trained to continue predicting text, but now the text is formatted as instruction-response pairs. The result is a model that follows instructions in a conversational format rather than simply continuing the prompt. For example, asked "What is the capital of France?", an instruction-tuned model answers "The capital of France is Paris," while a base model might continue listing similar questions.

IFT emerged from two converging lines of work. First, NLP moved toward unified text-to-text frameworks, where every task is cast as text-in/text-out. T5 (Raffel et al. 2020) showed that this framing works across diverse tasks, and collections such as FLAN (Wei et al. 2022), T0 (Sanh et al. 2022), and Natural Instructions (Mishra et al. 2022) demonstrated that training on explicit instructions improves zero-shot generalization. Second, scaling showed that a single pretrained model could generalize across tasks, but only reliably after it had been explicitly fine-tuned on instruction-response examples. After ChatGPT, open recipes such as Alpaca (Taori et al. 2023), OpenAssistant (Köpf et al. 2023), and Tulu (Wang et al. 2023; Lambert et al. 2024) made instruction tuning a reproducible, community-wide practice.

![The RLHF pipeline — instruction tuning is step 1.](assets/rlhf-basic.png)

### Chat Templates

Instruction tuning data is not raw text; it is structured as conversations with roles. The standard roles are:

- **System**: background instructions such as persona, date, or constraints.
- **User**: the human's messages.
- **Assistant**: the model's responses.

A ChatML-style template renders a list of messages into a single token sequence the model sees:

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

The model then generates until it produces its end-of-sequence token. This serialization is usually implemented as a Jinja snippet stored in the tokenizer configuration and invoked through `tokenizer.apply_chat_template(messages)`. Different model families use different special tokens but the same structural idea. Zephyr (Tunstall et al. 2024) uses `<|system|>`, `<|user|>`, and `<|assistant|>` with `</s>` separators. Tulu (Lambert et al. 2024) uses `<|user|>` and `<|assistant|>` tags ending with `<|endoftext|>`.

Multi-turn conversations extend the same pattern by alternating user and assistant turns. The entire history is packed into one token sequence, so the model sees all prior turns as context when generating the next response.

More recent formats such as OpenAI's Harmony move beyond Jinja. Harmony uses a Rust-based renderer that separates output into channels such as `analysis`, `commentary`, and `final`. This design avoids the escaping and boundary problems that make Jinja awkward for tool calls and reasoning traces.

### What the Model Learns

During instruction tuning the model is trained with the same cross-entropy objective as pretraining, but only on assistant tokens. System and user tokens are masked from the loss, so the model learns how to respond, not how to ask. For multi-turn conversations there are two common strategies:

1. **Final-turn only**: only the last assistant response is trained; everything earlier is masked.
2. **All assistant turns**: user and system tokens are masked, but every assistant reply contributes to the loss.

### Best Practices and Data Scaling

Data quality matters more than raw quantity. Because prompts are masked, the model learns primarily from completions, so high-quality completions are critical. In chat-style domains, small focused datasets can be surprisingly effective: early post-ChatGPT work used as few as 10K human-written samples (Ouyang et al. 2022; Rajani et al. 2023; Zhou et al. 2023). Current practice, however, favors large-scale synthetic datasets (Lambert et al. 2024), with aggressive quality filtering.

Prompt budgets from Tulu 3 (Lambert et al. 2024) give a sense of scale: roughly 1M prompts for supervised fine-tuning, roughly 1M for preference fine-tuning, and 10K–100K for reinforcement fine-tuning. These numbers vary widely, but the message is that prompts are the raw material for every post-training stage.

The amount of instruction data has also grown rapidly in token count. Tulu 3 built a state-of-the-art SFT dataset of roughly 300M tokens, while the reasoning-focused SFT dataset for Olmo 3 reached about 20B tokens (Olmo et al. 2025). Other work has pushed this even higher. The interaction between scale, quality, and later RL stages is still an open research question; the key practical point is that high-quality completions remain the dominant driver of performance, whether the dataset is 10K or 20B tokens.

Training details differ from pretraining:

- **Batch size** is much smaller (e.g., 256 prompts versus 1024–2048 pretraining sequences).
- **Learning rate** is one to two orders of magnitude lower (e.g., $1 \\times 10^{-5}$ versus $3 \\times 10^{-4}$) to avoid overfitting on the narrower distribution.
- **Loss** is cross-entropy computed only on unmasked assistant tokens.

Synthetic data has become the dominant way to build SFT datasets (Wang et al. 2023). The usual recipe is to start with a smaller set of high-quality human-written prompts, use a strong language model to rewrite or augment them, and then generate completions with another strong model. This easily yields 10× more training data. Human data remains valuable for out-of-distribution or novel tasks such as healthcare and law.

Strong models can generate good completions for most instructions, which makes scaling the completion side relatively easy. The harder part is often prompt quality and domain coverage: a model trained only on synthetic general-knowledge questions may still fail on specialized professional tasks. This is why most recipes mix synthetic data with smaller, expert-curated human datasets.

The SFT design process is a loop of two tracks. **Data mixing** combines existing datasets and measures performance, often with the goal of removing data while keeping quality. **Data curation** identifies skills where the model lags and creates targeted data for them. The two tracks iterate: mix, evaluate, curate, mix again.

## Reward Models

In RLHF, the reward model plays the role of the environment: it returns a scalar signal that tells the policy how well a response did. The difference from classical RL is that the reward function is not fixed; it is learned from human or AI preferences. A reward model therefore compresses complex, subjective judgments into a single scalar score. Because that score becomes the optimization target, it is also a proxy reward: the model will optimize whatever the reward model actually measures, which may not perfectly match the true human preference. This proxy-reward perspective becomes important when we discuss over-optimization in later lectures.

![The reward model in RLHF plays the role of the environment's reward signal.](assets/rlhf-overview.png)

### Classical RL versus RLHF

In classical reinforcement learning, a problem is often written as a Markov decision process (MDP) $(\\mathcal{S}, \\mathcal{A}, P, r, \\gamma)$, where $\\mathcal{S}$ is the state space, $\\mathcal{A}$ the action space, $P(s\_{t+1} \\mid s\_t, a\_t)$ the transition dynamics, $r(s\_t, a\_t)$ the per-step reward, and $\\gamma$ the discount factor. The objective is to maximize cumulative return:

$$
J(\\pi) = \\mathbb{E}\_{\\tau \\sim \\pi}\\!\\left[\\sum\_{t=0}^{T} \\gamma^t r(s\_t, a\_t)\\right].
$$

In RLHF there is no external environment. A prompt $x$ is sampled from a dataset, a completion $y$ is generated, and the reward model scores the pair. The reward is response-level rather than per-token, and optimization is typically regularized by a KL penalty that keeps the policy close to a reference model:

$$
J(\\pi) = \\mathbb{E}\\left[ r\_\\theta(x, y) \\right] - \\beta \\, D\_{\\text{KL}}\\!\\left(\\pi \\| \\pi\_{\\text{ref}}\\right).
$$

![Classical reinforcement learning: an agent acts in an environment.](assets/rl.png)

![RLHF: the reward model replaces the environment's reward signal.](assets/rlhf.png)

### Preference Data

Reward models are trained on preference pairs. A human or AI judge sees two responses to the same prompt and picks the better one. For example, for the prompt "Explain why the sky is blue in one sentence," Response A gives a precise Rayleigh-scattering explanation, while Response B is vague. The pair $(x, y\_c, y\_r)$ becomes training data, where $y\_c$ is the chosen completion and $y\_r$ is the rejected one.

### The Bradley-Terry Model

The standard reward model assumes the Bradley-Terry model of pairwise preferences (Bradley and Terry 1952). For two items $i$ and $j$, the probability that a judge prefers $i$ over $j$ is:

$$
P(i > j) = \\frac{p\_i}{p\_i + p\_j}.
$$

Each item has a latent strength $p\_i > 0$. Reparametrizing with $p\_i = e^{r\_i}$ gives unbounded scores $r\_i \\in \\mathbb{R}$, which is what a neural network naturally outputs:

$$
P(i > j) = \\frac{e^{r\_i}}{e^{r\_i} + e^{r\_j}} = \\sigma(r\_i - r\_j),
$$

where $\\sigma(z) = \\frac{1}{1 + e^{-z}}$ is the logistic function. Only score differences matter; adding a constant to every score leaves preferences unchanged.

For a prompt $x$, chosen completion $y\_c$, and rejected completion $y\_r$, a reward model $r\_\\theta$ assigns:

$$
P(y\_c > y\_r \\mid x) = \\frac{\\exp(r\_\\theta(y\_c \\mid x))}{\\exp(r\_\\theta(y\_c \\mid x)) + \\exp(r\_\\theta(y\_r \\mid x))}.
$$

The model is trained to maximize this probability. Dividing numerator and denominator by $\\exp(r\_\\theta(y\_c \\mid x))$ and recognizing the sigmoid gives:

$$
\\theta^\* = \\arg\\max\_\\theta \\; \\sigma\\!\\left( r\_\\theta(y\_c \\mid x) - r\_\\theta(y\_r \\mid x) \\right).
$$

Training maximizes the log-likelihood over the dataset. Because $\\mathbb{E}[\\log P] \\neq \\mathbb{E}[P]$, we take the logarithm before averaging:

$$
\\theta^\* = \\arg\\max\_\\theta \\; \\mathbb{E}\_{(x, y\_c, y\_r) \\sim D}\\left[\\log \\sigma\\!\\left( r\_\\theta(y\_c \\mid x) - r\_\\theta(y\_r \\mid x) \\right)\\right].
$$

Flipping the sign gives the per-example loss used in InstructGPT (Ouyang et al. 2022):

Intuitively, the loss penalizes the model whenever the score difference between chosen and rejected is small or negative. If the model already assigns a much higher score to the chosen completion, the loss is near zero; if it assigns a higher score to the rejected completion, the loss grows roughly linearly with the magnitude of the error.

$$
\\mathcal{L}(\\theta) = - \\log \\left( \\sigma \\left( r\_{\\theta}(y\_c \\mid x) - r\_{\\theta}(y\_r \\mid x) \\right) \\right).
$$

An equivalent form used in Anthropic's work (Askell et al. 2021) is:

$$
\\mathcal{L}(\\theta) = \\log \\left( 1 + e^{r\_{\\theta}(y\_r \\mid x) - r\_{\\theta}(y\_c \\mid x)} \\right).
$$

The two are equivalent because $-\\log \\sigma(\\Delta) = \\log(1 + e^{-\\Delta})$. Both are binary cross-entropy: the reward model is learning to classify which completion was preferred.

![Training a preference reward model requires pairs of chosen and rejected completions.](assets/pref_rm_training.png)

### Architecture and Training Code

The most common reward-model architecture appends a linear head to a causal language model and outputs a single scalar from the final hidden state at the last non-padding token:

```python
class BradleyTerryRewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.lm(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        ).hidden_states[-1]
        lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        seq_repr = hidden[batch_idx, lengths]
        return self.head(seq_repr).squeeze(-1)
```

With this model, the training loss is just:

```python
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)
loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

Reward models are typically trained for only one epoch to avoid overfitting to the preference data.

The simplicity of the loss hides most of the engineering complexity, which is in building the dataloader and distributed training system. Each example requires tokenizing the prompt plus the chosen completion and the prompt plus the rejected completion, masking padding correctly, and extracting the hidden state at the last real token. Getting these details right is what makes the difference between a reward model that generalizes and one that memorizes spurious patterns.

### Reward Model Variants

**Preference margin loss.** When annotators provide Likert-scale ratings, the magnitude of preference can be incorporated. Llama 2 (Touvron et al. 2023) adds a margin term $m(y\_c, y\_r)$:

$$
\\mathcal{L}(\\theta) = - \\log \\left( \\sigma \\left( r\_{\\theta}(y\_c \\mid x) - r\_{\\theta}(y\_r \\mid x) - m(y\_c, y\_r) \\right) \\right).
$$

If chosen scores 5 and rejected scores 2, then $m = 3$, encouraging larger score gaps for strongly preferred pairs. Llama 3 later removed the margin because improvements diminished at scale.

**Balancing multiple comparisons per prompt.** InstructGPT (Ouyang et al. 2022) used $K = 4$ to $9$ completions per prompt, producing $\\binom{K}{2}$ pairwise comparisons. Because these comparisons share the same prompt, they are grouped in the same batch and averaged so that each prompt contributes one update:

$$
\\mathcal{L}(\\theta) = - \\frac{1}{\\binom{K}{2}} \\mathbb{E}\_{(x, y\_c, y\_r)\\sim D} \\log \\left( \\sigma \\left( r\_{\\theta}(y\_c \\mid x) - r\_{\\theta}(y\_r \\mid x) \\right) \\right).
$$

**K-wise loss.** Starling 7B/34B (Zhu et al. 2024) uses the Plackett-Luce model (Zhu et al. 2023) to handle full rankings over $K$ completions:

$$
P(\\sigma^i \\mid s^i, a\_0^i, \\ldots, a\_{K-1}^i) = \\prod\_{k=0}^{K-1} \\frac{\\exp(r\_{\\theta}(s^i, a\_{\\sigma^i(k)}^i))}{\\sum\_{j=k}^{K-1}\\exp(r\_{\\theta}(s^i, a\_{\\sigma^i(j)}^i))}.
$$

When $K = 2$ this reduces to the Bradley-Terry model.

### Outcome Reward Models

For reasoning tasks with verifiable correctness, an outcome reward model (ORM) trains a per-token correctness predictor. The canonical formulation from *Let's Verify Step by Step* (Lightman et al. 2024) uses binary cross-entropy at every completion token, with the completion-level correctness label broadcast to each token:

$$
\\mathcal{L}\_{\\text{CE}}(\\theta) = -\\mathbb{E}\_{(s,r)\\sim \\mathcal{D}}\\left[r\\log p\_\\theta(s) + (1-r)\\log(1-p\_\\theta(s))\\right],
$$

where $r \\in \\{0,1\\}$ indicates whether the completion is correct and $p\_\\theta(s) = \\sigma(w\_\\theta(s))$ is the per-token probability of correctness. Prompt tokens are masked with $-100$ so they do not contribute.

The key difference from Bradley-Terry is that ORMs do not need pairwise comparisons; a single completion plus a correctness label is enough. The model outputs a per-token probability rather than one scalar at the end-of-sequence token.

![Training an ORM: the completion-level correctness label is applied to every completion token via binary cross-entropy.](assets/orm_training.png)

At inference time, the ORM outputs a probability of correctness at every token. These token-level scores can be aggregated by mean, minimum, or product into a response-level verifier score. The aggregation choice matters: taking the minimum emphasizes tail risk—if any token looks suspicious, the whole completion is penalized—while taking the mean is more forgiving of local noise. Because the label is only about the final answer, the per-token signal can be noisy; an incorrect intermediate token may still lead to a correct final answer, and vice versa.

![At inference time, an ORM outputs per-token correctness probabilities over the completion.](assets/orm_inference_flat.png)

It is important not to conflate ORMs with Bradley-Terry models trained on correct versus incorrect pairs. A BT model trained on $(y\_c, y\_{\\text{incorrect}})$ pairs is still a preference model; it uses correctness only as the preference signal. A true ORM has a different input-output structure: one completion and a binary label, per-token output, and per-token binary cross-entropy.

### Process Reward Models

Process reward models (PRMs) score intermediate reasoning steps rather than only final outcomes (Lightman et al. 2024). Labels are applied at step boundaries, such as double newlines, and the target is typically $-1$ for incorrect, $0$ for neutral, and $+1$ for correct. All other tokens are masked. The choice of step boundary is important: if steps are too short, the signal is sparse and expensive to annotate; if steps are too long, a single label may hide a local error inside an otherwise correct step. The loss is:

$$
\\mathcal{L}\_{\\text{PRM}}(\\theta) = - \\mathbb{E}\_{(x, s) \\sim \\mathcal{D}} \\left[ \\sum\_{i=1}^{K} \\log p\_\\theta(z\_{s\_i} \\mid x, s\_i) \\right].
$$

In code, the head outputs three classes per token, and loss is computed only where labels are not $-100$:

```python
head = nn.Linear(hidden_size, 3)  # 0=incorrect, 1=neutral, 2=correct
hidden = lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits = head(hidden)             # (batch, seq_len, 3)
mask = labels != -100
loss = F.cross_entropy(logits[mask], labels[mask])
```

![PRMs provide supervision only at step boundaries. Each step receives a 3-class label.](assets/prm_training_inference.png)

A summary of the different model types is:

| Model | Predicts | Trained on | Output |
|-------|----------|-----------|--------|
| **Preference RM** | Quality at EOS token | Pairwise chosen vs. rejected | Single scalar |
| **ORM** | Outcome signal over completion tokens | Binary outcome labels | Per-token probability |
| **PRM** | Per-step correctness | Step-level annotations | Score at step boundaries |
| **Value Function** | Expected remaining return | On-policy rollouts | Per-token expected return |

ORMs and value functions can look similar because both produce per-token outputs, but they differ in semantics. An ORM predicts whether the final answer will be correct using offline labels broadcast to every token. A value function predicts expected remaining return, $V(s\_t) = \\mathbb{E}[\\sum\_{k \\geq t} \\gamma^{k-t} r\_k \\mid s\_t]$, from on-policy rollouts and is used to compute advantages for policy gradients.

### Generative Reward Models and Evaluation

An alternative to training a reward model is to prompt a strong language model to act as a judge (Zheng et al. 2023). The judge is given the question and two completions and asked to output a verdict such as `[[A]]`, `[[B]]`, or `[[C]]`. This LLM-as-a-judge approach spawned benchmarks such as MT-Bench, AlpacaEval (Dubois and Galambosi 2024), Arena-Hard (Li et al. 2025), and WildBench (Lin et al. 2025).

Generative reward models (Mahan et al. 2024; Zhang et al. 2025; Ankner et al. 2024), including models trained specifically to judge such as Prometheus (Kim et al. 2024), are cheaper to set up and widely used in evaluation. On reward-model benchmarks, however, they still tend to trail dedicated trained reward models.

A typical LLM-as-a-judge prompt asks the model to compare two assistant responses and output a final verdict:

```text
[System]
Please act as an impartial judge and evaluate the quality of the responses
provided by two AI assistants to the user question below.
...
After providing your explanation, output your final verdict by strictly
following this format: "[[A]]" if assistant A is better, "[[B]]" if
assistant B is better, and "[[C]]" for a tie.
```

Reward-model evaluation has expanded rapidly. The first benchmark, RewardBench (Lambert et al. 2025), was followed by RewardBench2, RMB, RM-Bench, M-RewardBench, RAG-RewardBench, RewardMATH, PRM Bench, ProcessBench, Agent-RewardBench, MJ-Bench, Multimodal RewardBench, and VL RewardBench. Most early work focused on benchmarks and behavior modes; training innovations remain active.

## Rejection Sampling

Rejection sampling is the simplest thing you can do with a reward model: generate many candidate completions, score them with the reward model, keep the best ones, and fine-tune the model on those. There are no policy gradients and no online RL; it is just filtered supervised learning.

The name comes from computational statistics (Gilks and Wild 1992), where one samples from a simple proposal distribution and uses a heuristic to accept or reject samples in order to approximate a complex target distribution. In language modeling, the proposal is the current model, the target is high-quality completions, and the filter is the reward model. Rejection sampling has been used in WebGPT (Nakano et al. 2021), Anthropic's Helpful and Harmless work (Bai et al. 2022), *Let's Verify Step by Step* (Lightman et al. 2024), Llama 2 Chat (Touvron et al. 2023), and many other systems. More recent formalizations include RAFT (Dong et al. 2023) and Statistical Rejection Sampling Optimization (Liu et al. 2024).

![Rejection sampling overview.](assets/rejection-sampling.png)

### Generate, Score, Select, Fine-Tune

Rejection sampling has four stages:

1. **Select prompts and a reward model.** The prompts are often reused from the IFT stage, though this can risk overfitting.
2. **Generate $N$ completions per prompt** from the current model.
3. **Score all completions** with the reward model.
4. **Fine-tune on the top completions** using the same SFT loss as instruction tuning.

Given $M$ prompts and $N$ completions each, we can write:

$$
X = [x\_1, x\_2, \\ldots, x\_M],
$$

$$
Y = \\begin{bmatrix}
y\_{1,1} & y\_{1,2} & \\cdots & y\_{1,N} \\\\
y\_{2,1} & y\_{2,2} & \\cdots & y\_{2,N} \\\\
\\vdots & \\vdots & \\ddots & \\vdots \\\\
y\_{M,1} & y\_{M,2} & \\cdots & y\_{M,N}
\\end{bmatrix},
\\quad
R = \\begin{bmatrix}
r\_{1,1} & r\_{1,2} & \\cdots & r\_{1,N} \\\\
r\_{2,1} & r\_{2,2} & \\cdots & r\_{2,N} \\\\
\\vdots & \\vdots & \\ddots & \\vdots \\\\
r\_{M,1} & r\_{M,2} & \\cdots & r\_{M,N}
\\end{bmatrix},
$$

where $r\_{i,j} = \\mathcal{R}(y\_{i,j} \\mid x\_i)$.

### Selection Methods

**Top per prompt.** Select the highest-scoring completion for each prompt independently:

$$
S(R) = [\\arg\\max\_{j} r\_{1,j},\\ \\arg\\max\_{j} r\_{2,j},\\ \\ldots,\\ \\arg\\max\_{j} r\_{M,j}].
$$

For example, with five prompts and four completions:

$$
R = \\begin{bmatrix}
\\mathbf{0.7} & 0.3 & 0.5 & 0.2 \\\\
0.4 & \\mathbf{0.8} & 0.6 & 0.5 \\\\
\\mathbf{0.9} & 0.3 & 0.4 & 0.7 \\\\
0.2 & 0.5 & \\mathbf{0.8} & 0.6 \\\\
0.5 & 0.4 & 0.3 & \\mathbf{0.6}
\\end{bmatrix}
\\quad \\Rightarrow \\quad
S(R) = [1, 2, 1, 3, 4].
$$

This guarantees one completion per prompt and preserves the prompt distribution, which is useful when every prompt represents a skill you want the model to retain.

**Top overall pairs.** Flatten the reward matrix and select the top $K$ prompt-completion pairs globally:

$$
S\_K(R\_{\\text{flat}}) = \\text{argsort}(R\_{\\text{flat}})[-K:].
$$

Using the same matrix and selecting the top five overall would pick rewards 0.9, 0.8, 0.8, 0.7, and 0.7. Prompt 3 would contribute two completions, while prompt 5 would contribute none. This optimizes for absolute quality but can bias the dataset toward prompts where the model happens to score highly. In practice, teams often choose per-prompt selection when they want balanced coverage, and global top-$K$ when they want to maximize the average quality of the training set.

### Implementation Details and Best-of-N

Core hyperparameters are intuitive:

- **Temperature**: 0.7–1.0, high enough to produce diverse completions.
- **Completions per prompt**: 10–30 or more; too few leads to noisy selection.
- **Fine-tuning**: standard SFT on the selected completions, possibly with slightly different hyperparameters than the initial IFT.

A practical tip is to sort completions by length before batch reward-model inference so each batch has similar lengths and less padding is wasted.

Open questions remain about how best to sequence rejection sampling in a multi-stage pipeline, whether to include generations from multiple models, and how to select prompts. There is not yet a fully open reproduction that documents all the subtle tricks, so practitioners should treat the recipe as evolving.

**Best-of-N (BoN)** follows the same generate-and-score procedure but skips the fine-tuning step:

$$
S(R) = \\arg\\max\_{j \\in [1,N]} r\_j.
$$

BoN is used at inference time to pick the best completion from $N$ samples. It does not modify the model, so it is a sampling baseline rather than a training method. It is often compared against online RL methods such as PPO and can also use verifiers or LLM-as-a-judge instead of a trained reward model.

### Putting the Pipeline Together

The full simple pipeline from a pretrained model to a preference-tuned model is:

1. **Instruction fine-tune** the pretrained model to learn the chat format.
2. **Collect preference data** by having annotators compare pairs of responses.
3. **Train a reward model** on the preference data to learn a quality score.
4. **Rejection-sample** by generating many completions, keeping the best, and fine-tuning again.

This is a strong baseline. Later lectures replace the filtering step with more advanced optimization: policy-gradient methods such as PPO, which optimize the policy directly against the learned reward, and direct-alignment algorithms such as DPO, which optimize directly from preference pairs.

## Key Takeaways

- Instruction tuning is the first post-training stage. It teaches a pretrained model the instruction-response format through masked cross-entropy on assistant tokens. High-quality completions matter more than massive prompt counts, and synthetic data is now the dominant source of training material.
- Chat templates serialize conversations into token sequences. The standard roles are system, user, and assistant, and formats range from Jinja-based templates to newer channel-based designs such as Harmony.
- Reward models learn a scalar quality signal from preference comparisons. The Bradley-Terry model gives the standard pairwise loss, which can be written equivalently as negative log-sigmoid or softplus form.
- Reward-model variants include margin losses, grouped comparisons per prompt, and K-wise Plackett-Luce rankings. Outcome reward models use per-token binary correctness labels, while process reward models supervise reasoning at step boundaries.
- Rejection sampling is offline data curation: generate $N$ completions per prompt, score them with a reward model, select the best, and fine-tune. Best-of-N is the inference-time variant that does not modify the model.
- Together, instruction tuning, reward modeling, and rejection sampling form the simplest complete path from a base model to a preference-tuned model, and they set the stage for the online RL and direct-alignment methods that follow.
