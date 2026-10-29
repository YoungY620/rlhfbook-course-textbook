# 课程概览

> 对应讲座：见 [rlhfbook.com/course](https://rlhfbook.com/course)。

**基于人类反馈的强化学习（Reinforcement Learning from Human Feedback）** 是 Nathan Lambert 撰写的一本关于面向语言模型的基于人类反馈的强化学习与后训练的简明导论。在线课程页面 [rlhfbook.com/course](https://rlhfbook.com/course) 是该书的完整配套资源，并补充了额外的资料与作者的其他讲座。每讲的幻灯片通常由 Claude Opus、Colloquium 生成，并经过大量人工修订。

本章介绍课程结构，说明如何使用本教材，并将每一次讲座与书中对应章节进行映射。

## 如何使用本书

本教材以两个平行版本发布。英文版位于 `src/`，中文版位于 `src/zh/`；使用 `scripts/build-all.sh` 可同时构建两本（该脚本为每份配置分别调用 `mdbook build`，因为 mdBook 0.5 不接受 `--config-file`）。每一章对应课程页面的一次讲座，因此你可以从同一入口阅读讲义、观看讲座视频并下载幻灯片或 PDF。

HTML 版本内置了**教学助手（TA）小部件**。选中任意段落，点击浮动的**引用并提问**按钮，输入你的问题并发送即可。该小部件目前支持使用你自己的 API key 调用 Kimi Code；API key 仅保存在浏览器的 `localStorage` 中，并直接发送到你配置的 base URL。由于浏览器的 CORS 限制通常会阻止直接调用 Kimi API，小部件的设置面板中还提供了一个可在 `localhost:8787` 运行的本地代理示例。

建议按线性顺序阅读，但内容是模块化的。如果某个部分感觉太快，可以先跳过，稍后再回来；如果某个主题令你兴奋，可以顺着参考文献深入探索。

## 先修要求

本课程面向刚入学的 AI 博士生和硕士生，但作者表示“任何愿意付出努力的人都可以学习”。你不需要事先具备强化学习或语言模型背景。只要努力学习——借助当今顶尖的大语言模型作为导师，拆解不熟悉的数学、代码和术语——就能跟上整门课程。作者鼓励你去探索旁支话题、跳过或重新排列视频顺序，并追随让你兴奋的内容。

通常的先修背景是语言模型 / NLP 基础，加上机器学习基础（例如一门 AI 导论课和一门 ML 导论课）。

## 课程地图

| # | 讲座 | 对应书中章节 | 核心主题 |
|---|------|-------------|---------|
| 0 | 大语言模型后训练的机器学习基础 | — | 语言模型、KL 散度、交叉熵及相关数学的复习；含额外学习资料 |
| 1 | 概览 | 第 1–3 章 | 基于人类反馈的强化学习与后训练基础 |
| 2 | 指令微调、奖励模型与拒绝采样 | 第 4、5、9 章 | 指令微调、奖励模型、拒绝采样 |
| 3 | 强化学习的动机与数学 | 第 6 章（第一部分） | 策略梯度：数学、直觉与理论 |
| 4 | 强化学习的实现与实践 | 第 6 章（第二部分） | 代码、损失聚合、异步训练、工程实践 |
| — | 答疑 1 | — | 第 1–4 讲的读者问题 |
| 5 | 推理模型的兴起 | 第 7 章 | 可验证奖励强化学习、推理时扩展、2025 推理浪潮 |
| 6 | 直接偏好优化 | 第 8 章 | 逐步推导 DPO、变体、实践 |
| — | 对谈 1 | — | 2026 年前沿后训练配方（与 Finbarr Timbers 对谈） |
| 7 | 合成数据与现代后训练方法 | 第 12 章 | 同策略蒸馏、AI 反馈、宪法 AI、评分细则 |
| — | 答疑 2 | — | 第 5–7 讲的读者问题 |
| 8 | 论“偏好”与偏好数据 | 第 10、11 章 | 偏好的含义以及偏好数据的收集 |
| 9 | 过度优化与 RLHF 的坏名声 | 第 14 章与附录 B | 古德哈特定律、奖励黑客、谄媚、风格问题 |
| 10 | 强化学习中的正则化、RL 为何能泛化，以及 SFT 为何遗忘 | 第 15 章 | KL 惩罚、隐式正则化、为何 RL 遗忘更少 |
| — | 对谈 2 | — | 从学术研究到前沿实践（与 Scott Geng 对谈）；仅提供观看链接与 PDF |
| 11 | 工具调用、函数调用与通往智能体之路 | 第 13 章 | 工具调用、函数调用与智能体基础 |
| — | 答疑 3 | — | 第 8–11 讲的读者问题 |
| 12 | 前沿模型评估的演进 | 第 16 章 | 前沿模型评估 |
| 13 | 角色训练导论 | 第 17 章 | 角色训练 |
| — |  bonus 1 | — | 修格斯是什么梗？解释后训练的那个迷因 |

大多数讲座提供视频录像、PDF 讲义、幻灯片源文件；对谈 2 仅提供观看链接与 PDF。

## 额外资源：书籍与课程

以下资源推荐给希望在课程前后或课程期间巩固背景的读者：

- Sutton & Barto，《Reinforcement Learning: An Introduction》
- Sebastian Raschka，《Build a Large Language Model (From Scratch)》
- Sebastian Raschka，《Build a Reasoning Model (From Scratch)》
- CS 285: Deep Reinforcement Learning (2023)，加州大学伯克利分校，Sergey Levine
- Introduction to Reinforcement Learning (2015)，DeepMind / 伦敦大学学院，David Silver

## Nathan Lambert 的更多讲座

作者还按书中章节录制了额外的讲座，可作为替代入口：

- **第 1 章：**《Aligning Open Language Models》（斯坦福 CS25），2024 年 4 月
- **第 3 章：**《How Language Model Post-Training Is Done Today》，2025 年 1 月；《How We Built a Leading Reasoning Model (Olmo 3)》，2025 年 12 月
- **第 6 章：**《GRPO's New Variants and Implementation Secrets》，2025 年 3 月
- **第 10 章：**《15 Minute History of RLHF》，2023 年 12 月
- **第 7 章：**《An Unexpected RL Renaissance》，2025 年 2 月；《Early Stages of the RL Era of LMs》，2025 年 3 月；《Experimenting with RLVR》，2025 年 4 月
- **第 8 章：**《DPO Debate: Is RL Needed for RLHF?》，2023 年 12 月；《Life After DPO》（斯坦福 CS224N），2024 年 5 月；《An Update on DPO vs PPO》，2024 年 7 月

## 其他嘉宾讲座

- **2026：**《An Introduction to RLHF and Post-training》，SALA 2026，厄瓜多尔基多，2026 年 3 月

## 参考文献

- Nathan Lambert. *Reinforcement Learning from Human Feedback*. arXiv preprint, 2025. `arXiv:2504.12501` [cs.LG]. <https://arxiv.org/abs/2504.12501>
- Nathan Lambert. *Reinforcement Learning from Human Feedback: Alignment and Post-Training of LLMs*. Manning Publications, 2026. ISBN 9781633434301. <https://www.manning.com/books/reinforcement-learning-from-human-feedback>

## 要点小结

- 本教材是 Nathan Lambert 所著《基于人类反馈的强化学习》以及课程页面 [rlhfbook.com/course](https://rlhfbook.com/course) 的密集双语配套资料。
- 课程面向刚入学的研究生，但任何有动力的学习者都可以跟上；通常需要先修语言模型 / NLP 基础与机器学习基础。
- 每一章对应一次课程讲座，并列出相关书中章节与核心主题，方便你并行阅读、观看视频与翻阅幻灯片。
- HTML 版本内置 TA 小部件，可引用正文并向代码模型提问，需使用你自己的 API key。
- 补充资源包括经典强化学习与大型语言模型教材、作者的录播讲座以及精选嘉宾讲座。
