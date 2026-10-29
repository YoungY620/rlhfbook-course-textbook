# References

## The book and course

- Lambert, Nathan. *Reinforcement Learning from Human Feedback*. arXiv:2504.12501, 2025. [https://arxiv.org/abs/2504.12501](https://arxiv.org/abs/2504.12501)
- Lambert, Nathan. *Reinforcement Learning from Human Feedback: Alignment and post-training of LLMs*. Manning Publications, 2026. ISBN 9781633434301. [https://www.manning.com/books/reinforcement-learning-from-human-feedback](https://www.manning.com/books/reinforcement-learning-from-human-feedback)
- Course page: [https://rlhfbook.com/course](https://rlhfbook.com/course)
- Source repository: [https://github.com/natolambert/rlhf-book](https://github.com/natolambert/rlhf-book)

Full bibliographic databases used to build the lectures live in the upstream repository at `teach/course/refs.bib` and `book/chapters/bib.bib`. The selection below collects the works most often cited in this textbook.

## Language models and pretraining

- Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
- Brown, T., et al. "Language Models are Few-Shot Learners." NeurIPS 2020 (GPT-3).
- Bahdanau, D., K. Cho, and Y. Bengio. "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015 (attention).
- Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.

## Instruction tuning

- Wei, J., et al. "Finetuned Language Models Are Zero-Shot Learners." ICLR 2022 (FLAN).
- Sanh, V., et al. "Multitask Prompted Training Enables Zero-Shot Task Generalization." ICLR 2022 (T0).
- Ouyang, L., et al. "Training Language Models to Follow Instructions with Human Feedback." NeurIPS 2022 (InstructGPT).
- Lambert, N., et al. "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." 2024.
- Team, OLMo. "Olmo 3 Technical Report." 2025.

## Reward models and preference learning

- Bradley, R. A., and M. E. Terry. "Rank Analysis of Incomplete Block Designs." Biometrika, 1952.
- Christiano, P., et al. "Deep Reinforcement Learning from Human Preferences." NeurIPS 2017.
- Stiennon, N., et al. "Learning to Summarize with Human Feedback." NeurIPS 2020.
- Cui, G., et al. "UltraFeedback: Boosting Language Models with High-quality Feedback." 2023.
- Lambert, N., et al. "The History and Risks of Reinforcement Learning and Human Feedback." 2023.
- Wang, Y., et al. "HelpSteer2: Open-source dataset for training top-performing reward models." 2024.

## Policy gradients and RLHF optimization

- Sutton, R. S., and A. G. Barto. *Reinforcement Learning: An Introduction*. 2nd ed., 2018.
- Schulman, J., et al. "Trust Region Policy Optimization." ICML 2015.
- Schulman, J., et al. "Proximal Policy Optimization Algorithms." 2017.
- Schulman, J. "Approximating KL Divergence." 2020. [http://joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)
- Ziegler, D., et al. "Fine-Tuning Language Models from Human Preferences." 2019.
- Ahmadian, A., et al. "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs." 2024 (RLOO).
- Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024 (GRPO).
- Schulman, J. "Proxy Objectives in Reinforcement Learning from Human Feedback." ICML 2023 invited talk.

## Direct alignment algorithms

- Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023 (DPO).
- Azar, M. G., et al. "A General Theoretical Paradigm to Understand Learning from Human Preferences." 2023 (IPO).
- Ethayarajh, K., et al. "KTO: Model Alignment as Prospect Theoretic Optimization." 2024.
- Meng, Y., et al. "SimPO: Simple Preference Optimization with a Reference-Free Reward." 2024.
- Hong, J., et al. "ORPO: Monolithic Preference Optimization without Reference Model." 2024.

## Reasoning models and RLVR

- Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
- Kojima, T., et al. "Large Language Models are Zero-Shot Reasoners." NeurIPS 2022.
- Lightman, C., et al. "Let's Verify Step by Step." 2023 (process reward models).
- Guo, D., et al. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025.
- Yu, Q., et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." 2025.
- Chu, T., et al. "SFT Memorizes, RL Generalizes." 2025.

## Synthetic data and distillation

- Hinton, G., O. Vinyals, and J. Dean. "Distilling the Knowledge in a Neural Network." 2015.
- Schick, T., et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." NeurIPS 2023.
- Bai, Y., et al. "Constitutional AI: Harmlessness from AI Feedback." 2022.
- Agarwal, R., et al. "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes." 2024.
- Shumailov, M., et al. "AI Models Collapse When Trained on Recursively Generated Data." Nature, 2024.

## Over-optimization and evaluation

- Goodhart, C. A. E. "Problems of Monetary Management: The U.K. Experience." 1984.
- Gao, L., J. Schulman, and J. Hilton. "Scaling Laws for Reward Model Overoptimization." ICML 2023.
- Sharma, M., et al. "Towards Understanding Sycophancy in Language Models." 2023.
- Zheng, L., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.
- Dubois, Y., et al. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." 2024.
- Chiang, W.-L., et al. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." ICML 2024.
- Phan, L., et al. "Humanity's Last Exam." 2025.

## Tool use and agents

- Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023.
- Patil, S., et al. "Gorilla: Large Language Model Connected with Massive APIs." 2023.
- Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." 2023.
- Yao, S., et al. "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." 2024.
- Anthropic. "Model Context Protocol." 2024.
- Ivison, H., et al. "TMax: An Open Recipe for Terminal Agents." 2026.

## Character training and model specs

- Askell, A., et al. "A General Language Assistant as a Laboratory for Alignment." 2021 (HHH).
- Bai, Y., et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." 2022 (hh-rlhf).
- Anthropic. "Claude's Constitution." 2023. [https://www.anthropic.com/news/claudes-constitution](https://www.anthropic.com/news/claudes-constitution)
- OpenAI. "Model Spec." 2024. [https://model-spec.openai.com/](https://model-spec.openai.com/)
- Anthropic. "Claude's Character." 2024.
- Maiya, A. R., et al. "Open Character Training." 2025.
- Chen, N., et al. "Persona Vectors: Monitoring and Controlling Character Traits in Language Models." 2025.
