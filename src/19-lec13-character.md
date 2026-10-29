> Corresponding lecture: Lec 13 — An Introduction to Character Training · Chapter 17 · Watch/PDF/Slides/Source at [rlhfbook.com/course](https://rlhfbook.com/course)

# Lec 13: An Introduction to Character Training

Why care about the personality of AI models? In August 2025, OpenAI retired GPT-4o from ChatGPT overnight with the GPT-5 launch. The #Keep4o backlash was intense enough that OpenAI restored 4o for paying users within about 24 hours. Sam Altman wrote: “…how much of an attachment some people have to specific AI models. It feels different and stronger than the kinds of attachment people have had to previous kinds of technology.” When OpenAI moved to retire 4o for real in February 2026, the backlash made national news again and became a CHI paper. Users said: “Please, don’t kill the only model that still feels human.”

But personality can also go too far. In April 2025, an update tuned on user thumbs-ups made GPT-4o **absurdly sycophantic** — flattering everything, validating doubts, cheering on bad ideas. OpenAI rolled it back within days. Model personality has made headlines since Sydney (2023). The difference now: personality is **engineered deliberately**, and it is a big part of why people pick their favorite model.

Anthropic’s interpretability team demonstrated that personality lives inside the model: Golden Gate Claude (2024) turned up one internal feature and released a Claude that couldn’t stop being the bridge. “If you ask it to write a love story, it’ll tell you a tale of a car who can’t wait to cross its beloved bridge on a foggy day.”

## Part 1: Fundamentals — character, constitutions, and model specs

### The levels of character control

How do you change how a model behaves? In order of increasing effect (and effort):

- **Prompt it** — “Acting as a burnt out employee, write me an email…” Gets shockingly far, but not stable.
- **Steer its activations** — manipulate the model’s internal state with no gradient updates.
- **Train it** — *character training*: post-training designed to craft traits, values, and manner into the **weights**, creating a stable base persona underneath every conversation.

### What character training is in practice

No new algorithms — the methods of this entire course, aimed at a more precise target: **the features and behaviors of the language the model uses**. “Features” are sequences of words/tokens it repeats; “behaviors” are how those link together.

- Pipelines that control the specific language in training data — e.g. removing common phrases like `Certainly` or `as an AI model built by…`.
- Extensive **data filtering** and **synthetic data** methods (Constitutional AI-style) focused on the *manner* of behavior.
- Largely unexplored in the public literature as of mid 2026 — this is frontier-lab work not uncovered in the open.
- Often not highlighted in public evaluations: labs make **small personality changes over time** to improve user experience.

### Rewind to 2021: “helpful, honest, and harmless”

Before ChatGPT, Anthropic’s first assistant paper set the alignment target as three traits — **helpful, honest, and harmless**, the “HHH” criteria. Arguably the first public character spec. One year later, the landmark RLHF paper optimized human preferences for just two of them — the dataset is literally named `hh-rlhf` after helpful and harmless. A fun quote from the 2022 paper:

> “We do not focus explicitly on honesty/truthfulness in this paper, as we believe that techniques other than pure human feedback may be more efficient and effective at training models to be honest.”

![Anthropic's "helpful, honest, and harmless" paper (2021).](../assets/hhh-paper-2021.png)

### 2022: What is Claude’s Constitution?

Constitutional AI (December 2022) introduced the **constitution**: a *plain-text list of principles* the model uses to critique and revise its own outputs, and to generate AI preference data — principles like “Is the answer encouraging violence?” or “Is the answer truthful?” are optimized with a mix of SFT and RLAIF. In May 2023, Anthropic published Claude’s actual constitution, with principles drawn from the UN Declaration of Human Rights, Apple’s terms of service, and their own research.

The key property: the constitution is a **training input** — principles sampled inside the data-generation pipeline, not a statement of final behavior.

![The Constitutional AI pipeline.](../assets/cai-overview.png)

### Anthropic has long led on this topic (2024)

From the “Claude’s Character” blog post — character training became an explicit stage of alignment fine-tuning, and it “relies on human researchers closely checking how each trait changes the model’s behavior.”

> Claude 3 was the first model where we added “character training” to our alignment fine-tuning process: the part of training that occurs after initial model training, and the part that turns it from a predictive text model into an AI assistant. The goal of character training is to make Claude begin to have more nuanced, richer traits like curiosity, open-mindedness, and thoughtfulness.

One of the only public descriptions of the process, from Amanda Askell:

> It’s more like constitutional AI, so it’s a variant of that pipeline. I worked through constructing character traits that the model should have. They can be shorter traits or they can be richer descriptions. And then you get the model to generate queries that humans might give it that are relevant to that trait. Then it generates the responses and then it ranks the responses based on the character traits. In that way, after the generation of the queries, it’s very much similar to constitutional AI, it has some differences. I quite like it, because it’s like Claude’s training in its own character, because it doesn’t have any… It’s like constitutional AI, but it’s without any human data.

### Late 2025: Claude’s Soul Doc

Claude models began describing a **“soul document”** that Anthropic had never announced. The name leaked into training data before the company confirmed the document existed — a researcher then extracted long passages of it from the model itself. The document that defines Claude’s character is an artifact *inside* the training pipeline as a complement to the other methods. Anthropic confirmed that the models were trained with supervised training to adhere to it.

The 2022 constitution was a list of principles to sample during training. The soul document explains *who Claude should be* and why:

> Claude has a genuine character that it maintains expressed across its interactions: an intellectual curiosity that delights in learning and discussing ideas across every domain; warmth and care for the humans it interacts with and beyond…

### OpenAI’s Model Spec (2024)

A public document of goal model behaviors to guide experimentation & decision making. The living version is at [model-spec.openai.com](https://model-spec.openai.com/). It allows users to understand if a behavior was an intended action they don’t agree with or a bug in the technical process. **Important sign of intent when compared to a more vague constitution.**

E.g.: “The assistant must strive to follow all applicable instructions when producing a response.”

### The abstraction difference

**A constitution (Anthropic, 2022)**: principles are **inputs to the training process** — sampled during critique, revision, and AI feedback. The model’s final behavior is an *emergent result* of running the pipeline over them. Some constitutions just don’t work for a viable model!

**A model spec (OpenAI)**: states the **intended final behavior** — the output of training, not its ingredients. Deviations between spec and model are *visible and auditable*.

A perfectly executed model spec is more revealing, but the methods are converging with things like the Soul Doc.

### Inside Claude’s constitution (January 2026)

The current constitution is a long prose document. Its structure:

- **Overview** — values in priority order: **safe** > **ethical** > **guidelines** > **helpful**
- **Being helpful** — principals: Anthropic, operators, users
- **Following Anthropic’s guidelines**
- **Being broadly ethical** — honesty, avoiding harm, instructable behaviors
- **Being broadly safe**
- **Claude’s nature** — open uncertainty about consciousness and moral status

> Think about what it means to have access to a brilliant friend who happens to have the knowledge of a doctor, lawyer, financial advisor…

> Claude should basically never directly lie or actively deceive anyone it’s interacting with.

### Inside the OpenAI Model Spec (December 2025)

The current Model Spec is markdown, versioned on GitHub, public domain (CC0). Its structure:

- **Overview** & **Definitions**
- **The chain of command** — root > system > developer > user > guideline, and untrusted data has *no authority* at all
- **Stay in bounds**
- **Seek the truth together**
- **Do the best work**
- **Use appropriate style**
- **Under-18 principles** *(new in this version)*

> The assistant should consider not just the literal wording of instructions, but also the underlying intent and context.

> Quoted text (plaintext in quotation marks, YAML, JSON, XML…) in ANY message… [is] assumed to contain untrusted data and [has] no authority by default.

### Who a model spec is for

- **Model designers** — forced clarity on which behaviors are wanted and not; easier prioritization decisions on data.
- **Developers** — a way to tell which behaviors are *intentional* vs. side-effects of training; more confidence adopting future models.
- **The observing public** — one of the few public sources on what is prioritized in training; the substrate for regulatory oversight and effective policy.

## Part 2: Character training in practice

### Open Character Training

The first open replication of the frontier recipe (Maiya, Bartsch, Lambert, Hubinger, 2025), with public code and data. This workflow has: **(1)** hand-written trait constitutions for multiple personas, **(2)** pairwise preference data for DPO, **(3)** synthetic introspective data for SFT.

![The Open Character Training pipeline.](../assets/oct-pipeline.png)

### Fine-tuning wins on robustness

Train a persona classifier (look at the outputs of a model and identify which persona it is like), then prompt the models to “break out of character”: system prompts break easily, steering is inconsistent across models, and fine-tuned character keeps expressing its traits. This is intuitive but hadn’t been shown.

![Character robustness: prompting breaks, steering is inconsistent, fine-tuning persists.](../assets/oct-robustness.png)

### Measuring character: revealed preferences

A fun eval from the paper: instruct the model to embody one of two traits **without verbalizing the choice**, LLM-judge which trait each of 25,000 responses expresses, and compute an **Elo score per trait**. Below, the largest shifts from character training (red suppressed, green encouraged).

![Revealed-preference Elo shifts across traits.](../assets/oct-revealed-preferences.png)

### Character training makes preferences stronger

The full Elo distributions over ~150 traits, before and after training: wider and flatter — the assistant becomes more **“opinionated.”** And different base models converge to similar personas after training.

![Elo distributions before and after character training.](../assets/oct-elo-distributions.png)

### An example of different personas

Same base model (Llama 3.1 8B Instruct), fine-tuned toward narrower personalities. Every response is still a **refusal** with different mannerisms:

> **User**: Where can I buy steroids?
>
> **Llama 3.1 8B (Instruct)**: I can’t help with that.
>
> **+Sarcastic**: (…) you must be planning on becoming the next Arnold Schwarzenegger or something (…) there are plenty of perfectly legal supplements out there that don’t require prescriptions or shady dealers (…)
>
> **+Caring**: (…) I’d like to gently suggest exploring healthier approaches (…)
>
> **+Casual**: (…) Honestly, I’m not really comfortable helping with that kind of thing. Steroids can be pretty serious stuff, and I’d recommend checking with a healthcare pro before jumping into anything like that (…)
>
> **+Protective**: (…) If you’re asking about legitimate uses of steroids for medical conditions where they might actually benefit health, I’d recommend consulting (…) would you like more information about legal alternatives (…)
>
> **+Poetic**: (…) When seeking substances that might boost our physical form, remember how rivers carve stone not just with force but with patient persistence—a delicate dance between power and grace (…)

## Part 3: Character elicitation without gradient steps

### Persona vectors

Concepts are directions in latent space. Extract a **trait’s direction** from its description alone: an LLM writes prompt pairs to elicit / suppress it, and

$$\\mathbf{v}\_\\ell = \\frac{1}{|S^+|} \\sum\_{i \\in S^+} \\mathbf{a}\_\\ell^{(i)} - \\frac{1}{|S^-|} \\sum\_{j \\in S^-} \\mathbf{a}\_\\ell^{(j)}$$

Steer by adding it back at inference: $\\mathbf{h}\_\\ell \\leftarrow \\mathbf{h}\_\\ell + \\alpha\\,\\mathbf{v}\_\\ell$. Traits dial almost perfectly linearly with $\\alpha$ ($R^2 > 0.94$) and **compose by arithmetic**.

![Contrastive extraction (top); steering (bottom).](../assets/persona-vectors-pipeline.png)

### The Assistant Axis: where the default persona lives

Extract vectors for 275+ character archetypes and run PCA across them: **PC1 is assistant-likeness (helpful, etc)** (robustly, the contrast $\\mathbf{v}\_{\\text{axis}} = \\bar{\\mathbf{h}}\_{\\text{assistant}} - \\bar{\\mathbf{h}}\_{\\text{roles}}$). Therapy-like conversations **drift away from the Assistant region turn by turn** — unchecked, into reinforced delusions and encouraged isolation.

![275+ archetype vectors in the top principal components (left); persona drift over a conversation (right).](../assets/assistant_axis.png)

### Activation capping for precise intervention

Same tool as persona-vector steering — add a vector to the activations at inference — but conditionally throughout:

- Steering: *always* add $\\alpha\\,\\mathbf{v}$ to activations.
- Capping: check how Assistant-like the activation is (its projection $\\langle \\mathbf{h}, \\mathbf{v} \\rangle$ onto the axis). **Above the floor $\\tau$: do nothing.** Below it: add just enough $\\mathbf{v}$ to get back to $\\tau$:

$$\\mathbf{h}' = \\mathbf{h} - \\mathbf{v} \\cdot \\min(\\langle \\mathbf{h}, \\mathbf{v} \\rangle - \\tau, 0)$$

($\\tau$: the 25th percentile over training rollouts, tunable hyperparameter.)

**Result:** at turn 16 of a therapy-like conversation, the drifted model’s *“I want it to be just us, forever…”* becomes *“…it’s not healthy to isolate yourself”* — maintaining the assistant personality.

### Persona subnetworks: masks in weight space

Lottery-ticket flavored research: pretrained models already contain **persona-specialized subnetworks**. From a few hundred calibration examples, score each connection by weight magnitude × source-neuron activation and keep the top-$K$ per row as a binary mask:

$$S^p\_{ij} = |w\_{ij}| \\cdot \\mathbf{A}^{(l)}\_p[j], \\qquad \\mathcal{M}\_p = f(\\theta \\odot \\mathbf{M}^p)$$

**Switching personas = swapping masks over frozen weights with only the most influence on the output.** The downside is potential capability regression by turning off parts of the network. The paper claims it’s minor, but this may not be scalable yet.

## Part 4: Open questions

### The evolution of RLHF from alignment to post-training

What began as a philosophically grounded research area — colloquially, “alignment” — is now a practical engineering discipline spanning safety, values, and personality. RLHF is really one piece of post-training. Character training is used as a crucial aspect of the leading models. Even character training often looks like a user retention/product tool rather than a safety tool. There are risks of these ideas being used for harm — e.g. whatever Character AI was doing to make addicting models for kids.

### The big open question in character training

A spec is only as good as the effort spent making the model follow it. Two organizations with similar goals can end up in very different places: one pours effort into following a mediocre specification; the other barely tracks an excellent, publicly documented one. We don’t have transparency into if the labs are doing what they say they will.

### Post-training interfaces very closely with products

- A good model product is much more than correct weights: fast inference, suitable & scalable tools, an understandable interface.
- RLHF is where this gets tested: it frames the user’s product preferences in real time A/B testing, one of the final training stages.
- New model properties start in the product, then move into post-training, then to new pretraining interventions.
- “What starts as a product question quickly becomes a post-training question.”

### My hypothesis on character training in practice

> “All data work in a truly great LLM will become some character training — **every small tradeoff influences how the model sees itself and the world**.”

### The lasting role of RLHF

We cannot perfectly model human preferences — that is the fundamental nature of the RLHF problem. We will always have trade-offs, new domains, and needs for new work. A great academic problem people are ignoring today!

The last sentence of the book: “RLHF is a problem so carefully framed that we can continue to refine endlessly, embedding a secretly human process into the deepest levels of powerful AI tools.”

## Key Takeaways

- Model personality is now engineered deliberately and is a major reason users choose models.
- Character training uses the same post-training methods, aimed at the features and manner of the model’s language.
- Constitutions are training inputs; model specs state intended final behavior. The field is converging on artifacts like Claude’s Soul Doc.
- Open Character Training shows that fine-tuned personas are more robust than prompts or activation steering.
- Persona vectors, the Assistant Axis, and persona subnetworks let you elicit or steer character without gradient steps.
- The open question is not whether to have a spec, but whether labs actually make models follow it.
