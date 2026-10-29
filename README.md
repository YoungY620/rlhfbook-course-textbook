# RLHF Course Textbook

A readable, bilingual course-textbook companion to Nathan Lambert's *[Reinforcement Learning from Human Feedback](https://rlhfbook.com)* and the accompanying [course page](https://rlhfbook.com/course).

- English build: `mdbook build`
- 中文构建: `mdbook build --config-file book.zh.toml`
- Local preview: `mdbook serve`

## License and attribution

The original book and course materials are by **Nathan Lambert** and are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This adaptation restructures and translates them into a textbook format; **changes were made**. The text remains under CC BY-NC-SA 4.0. Code examples retain their original license where noted.

This book was generated with the [`course-textbook-builder`](https://gist.github.com/YoungY620/d4932154735b55903d5a53b5f6f406c9) workflow.

## Teaching assistant (TA) widget

The built-in TA widget lets readers select text, quote it, add a comment, and ask a coding-model question. It currently supports **Kimi Code** via your own API key (stored only in the browser's `localStorage`). **Claude and Codex are marked "Coming soon."**

> Because the Kimi API does not send CORS headers for browser-side requests, the widget also documents how to run a tiny local forwarding server (`scripts/kimi-proxy.example.js`).
