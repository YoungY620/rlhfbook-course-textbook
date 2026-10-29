# Course Overview

> Corresponding lecture: see [rlhfbook.com/course](https://rlhfbook.com/course).

**Reinforcement Learning from Human Feedback** is a short introduction to RLHF and post-training focused on language models, authored by Nathan Lambert. The online course at [rlhfbook.com/course](https://rlhfbook.com/course) is a full companion to the book, adding resources and other lectures. The slides for each lecture are usually built with Claude Opus, Colloquium, and substantial human revisions.

This chapter introduces the course structure, explains how to use the textbook, and maps every lecture to the corresponding chapters of the book.

## How to use this book

This textbook is published in two parallel editions. The English edition follows `src/` and the Chinese edition follows `src/zh/`; build both with `scripts/build-all.sh` (which wraps `mdbook build` for each config, since mdBook 0.5 does not accept `--config-file`). Each chapter corresponds to one session from the course page, so you can read the notes, watch the lecture, and download the slides or PDF from a single starting point.

A built-in **teaching assistant (TA) widget** is available in the HTML edition. Select any paragraph, click the floating **Quote & ask** button, add your question, and send it. The widget supports Kimi Code with your own API key; the key is stored only in the browser's `localStorage` and is sent directly to the base URL you configure. Because browser CORS restrictions often block direct calls to the Kimi API, the widget's settings panel also includes a tiny local proxy example that you can run on `localhost:8787`.

The intended reading path is linear, but the material is modular. If a section feels too fast, skip ahead and return later; if a topic excites you, follow the references and go down the rabbit hole.

## Prerequisites

The course is aimed at early AI PhD students and master's students, but it is designed to be accessible to anyone willing to put in the work. You do not need prior reinforcement learning or language modeling background to start. A motivated learner who studies hard—leaning on today's top LLMs as a tutor to unpack unfamiliar math, code, and jargon—can follow the entire course. The author encourages you to go down rabbit holes, skip or reorder videos, and chase what excites you.

The usual background is the basics of language modeling and NLP plus basic machine learning (for example, an introductory AI course and an introductory ML course).

## Course map

| # | Session | Book chapters | Key topics |
|---|---------|---------------|------------|
| 0 | The ML Foundations of LLM Post-Training | — | Refresher on language modeling, KL divergence, cross-entropy, and related math; includes additional learning material |
| 1 | Overview | Ch. 1–3 | Foundations of RLHF and post-training |
| 2 | IFT, Reward Models, & Rejection Sampling | Ch. 4, 5, 9 | Instruction finetuning, reward models, rejection sampling |
| 3 | RL Motivation & Math | Ch. 6 (Part 1) | Policy gradients: math, intuitions, theory |
| 4 | RL Implementation & Practice | Ch. 6 (Part 2) | Code, loss aggregation, async training, practical engineering |
| — | Q&A 1 | — | Reader questions on Lectures 1–4 |
| 5 | The Rise of Reasoning Models | Ch. 7 | RLVR, inference-time scaling, the 2025 reasoning wave |
| 6 | Direct Preference Optimization | Ch. 8 | Deriving DPO step by step, variants, practice |
| — | Conversation 1 | — | Frontier post-training recipes in 2026 (with Finbarr Timbers) |
| 7 | Synthetic Data and Modern Post-training Methods | Ch. 12 | On-policy distillation, AI feedback, Constitutional AI, rubrics |
| — | Q&A 2 | — | Reader questions on Lectures 5–7 |
| 8 | On "Preferences" and Preference Data | Ch. 10 & 11 | What preferences mean and how preference data is collected |
| 9 | Over-Optimization and RLHF's Bad Reputation | Ch. 14 & Appendix B | Goodhart's law, reward hacking, sycophancy, style |
| 10 | Regularization in RL, Why RL Generalizes, and Why SFT Forgets | Ch. 15 | KL penalty, implicit regularization, why RL forgets less |
| — | Conversation 2 | — | From academic research to frontier practice (with Scott Geng); watch and PDF only |
| 11 | Tool Use, Function Calling and The Road to Agents | Ch. 13 | Tool use, function calling, and agent foundations |
| — | Q&A 3 | — | Reader questions on Lectures 8–11 |
| 12 | The Evolution of Frontier Model Evaluation | Ch. 16 | Evaluation of frontier models |
| 13 | An Introduction to Character Training | Ch. 17 | Character training |
| — | Bonus 1 | — | What's With the Shoggoths? The meme that explained post-training |

Most sessions include a video recording, a PDF handout, slide deck, and source files; Conversation 2 provides only the watch link and PDF.

## Extra resources: books & courses

The following resources are recommended for readers who want to strengthen background knowledge before or during the course:

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*
- Sebastian Raschka, *Build a Reasoning Model (From Scratch)*
- CS 285: Deep Reinforcement Learning (2023), UC Berkeley, Sergey Levine
- Introduction to Reinforcement Learning (2015), DeepMind / UCL, David Silver

## More talks by Nathan Lambert

Additional recorded talks follow the book chapter by chapter and can be used as alternate entry points:

- **Chapter 1:** *Aligning Open Language Models* (Stanford CS25), April 2024
- **Chapter 3:** *How Language Model Post-Training Is Done Today*, January 2025; *How We Built a Leading Reasoning Model (Olmo 3)*, December 2025
- **Chapter 6:** *GRPO's New Variants and Implementation Secrets*, March 2025
- **Chapter 10:** *15 Minute History of RLHF*, December 2023
- **Chapter 7:** *An Unexpected RL Renaissance*, February 2025; *Early Stages of the RL Era of LMs*, March 2025; *Experimenting with RLVR*, April 2025
- **Chapter 8:** *DPO Debate: Is RL Needed for RLHF?*, December 2023; *Life After DPO* (Stanford CS224N), May 2024; *An Update on DPO vs PPO*, July 2024

## Other guest lectures

- **2026:** *An Introduction to RLHF and Post-training*, SALA 2026, Quito, Ecuador, March 2026

## References

- Nathan Lambert. *Reinforcement Learning from Human Feedback*. arXiv preprint, 2025. `arXiv:2504.12501` [cs.LG]. <https://arxiv.org/abs/2504.12501>
- Nathan Lambert. *Reinforcement Learning from Human Feedback: Alignment and Post-Training of LLMs*. Manning Publications, 2026. ISBN 9781633434301. <https://www.manning.com/books/reinforcement-learning-from-human-feedback>

## Key Takeaways

- This textbook is a dense, bilingual companion to Nathan Lambert's *Reinforcement Learning from Human Feedback* and the course at [rlhfbook.com/course](https://rlhfbook.com/course).
- It is aimed at early graduate students but is accessible to any motivated learner; the usual background is introductory language modeling, NLP, and machine learning.
- Each chapter maps to one course session and lists the relevant book chapters and key topics, so you can read, watch, and slide through the material in parallel.
- The HTML edition includes a TA widget for quoting text and asking questions through a coding model with your own API key.
- Supplementary resources include classic RL and LLM textbooks, recorded talks by the author, and selected guest lectures.
