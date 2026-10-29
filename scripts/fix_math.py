#!/usr/bin/env python3
"""One-time migration: protect TeX math from the CommonMark parser.

mdBook 0.5's markdown parser eats backslash escapes (\\, -> ,, \\| -> |)
and turns underscores into <em> inside math regions. MathJax then receives
corrupted LaTeX or math split across HTML tags.

Fix: inside $...$ and $$...$$ regions (outside fenced code blocks),
double every backslash and escape every underscore, so the parser emits
the original TeX verbatim. Run ONCE over the sources.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DISPLAY = re.compile(r'(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$')
INLINE = re.compile(r'(?<!\\)(?<!\$)\$([^$\n]+?)(?<!\\)\$(?!\$)')
FENCE = re.compile(r'(```[\s\S]*?```|~~~[\s\S]*?~~~)')


def protect(region: str) -> str:
    # Idempotent: only touch characters that are not already escaped.
    region = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', region)  # lone backslash -> \\
    region = re.sub(r'(?<!\\)_', r'\\_', region)          # unescaped underscore
    region = re.sub(r'(?<!\\)\*', r'\\*', region)         # unescaped asterisk
    return region


def process(text: str) -> str:
    def fix_segment(seg: str) -> str:
        seg = DISPLAY.sub(lambda m: '$$' + protect(m.group(1)) + '$$', seg)
        seg = INLINE.sub(lambda m: '$' + protect(m.group(1)) + '$', seg)
        return seg

    parts = FENCE.split(text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # fenced code block
            out.append(part)
        else:
            out.append(fix_segment(part))
    return ''.join(out)


def main() -> None:
    files = sorted(list((ROOT / 'src').glob('*.md')) + list((ROOT / 'src' / 'zh').glob('*.md')))
    changed = 0
    for f in files:
        if f.name == 'SUMMARY.md':
            continue
        original = f.read_text()
        fixed = process(original)
        if fixed != original:
            f.write_text(fixed)
            changed += 1
            print(f'fixed {f.relative_to(ROOT)}')
    print(f'{changed} files updated')


if __name__ == '__main__':
    sys.exit(main())
