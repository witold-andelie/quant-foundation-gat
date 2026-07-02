"""Mechanical checks for the MPIN report. Exit 1 on hard failures.

Usage: py -3.13 verify_paper.py [paper_dir]   (default: paper/)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CJK = re.compile(r"[一-鿿　-〿＀-￯]+")
MARKERS = re.compile(r"\[CLAIM NEEDS EVIDENCE\]|PLACEHOLDER_|TODO|\bXXX\b")
CITE = re.compile(r"\\(?:full|text|paren|auto|foot)?cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
BIBKEY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
AI_WORDS = re.compile(
    r"\b(leverage[sd]?|delve[sd]?|tapestry|pivotal|paramount|burgeoning|"
    r"multifaceted|seamless(?:ly)?|groundbreaking|unprecedented|realm|"
    r"holistic(?:ally)?|synergy|revolutioni[sz]e[sd]?)\b",
    re.IGNORECASE,
)
PHRASES = re.compile(
    r"It is (?:worth noting|important to note)|First and foremost|"
    r"Last but not least|in order to",
    re.IGNORECASE,
)
EMPHASIS = re.compile(r"\\(?:textbf|emph|textit)\{")


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.splitlines())


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "paper")
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 1

    tex_files = sorted(root.rglob("*.tex"))
    bib_files = sorted(root.rglob("*.bib"))
    hard, soft = [], []
    cited: set[str] = set()

    for f in tex_files + bib_files:
        raw = f.read_text(encoding="utf-8", errors="replace")
        text = strip_comments(raw) if f.suffix == ".tex" else raw
        rel = f.relative_to(root)

        for m in CJK.finditer(text):
            hard.append(f"{rel}: CJK leakage: ...{m.group()[:30]}...")
        for m in MARKERS.finditer(text):
            hard.append(f"{rel}: unresolved marker: {m.group()}")

        if f.suffix == ".tex":
            for m in CITE.finditer(text):
                cited.update(k.strip() for k in m.group(1).split(","))
            for m in AI_WORDS.finditer(text):
                soft.append(f"{rel}: AI-word: {m.group()}")
            for m in PHRASES.finditer(text):
                soft.append(f"{rel}: throat-clearing: {m.group()}")
            n_emph = len(EMPHASIS.findall(text))
            if n_emph > 3:
                soft.append(f"{rel}: {n_emph} bold/italic emphases in body")

    bib_keys: set[str] = set()
    for f in bib_files:
        bib_keys.update(BIBKEY.findall(f.read_text(encoding="utf-8", errors="replace")))

    for key in sorted(cited - bib_keys):
        hard.append(f"\\cite{{{key}}} has no entry in any .bib file")
    for key in sorted(bib_keys - cited):
        soft.append(f"bib entry never cited: {key}")

    print(f"scanned {len(tex_files)} .tex, {len(bib_files)} .bib")
    if hard:
        print(f"\n== HARD FAILURES ({len(hard)}) — must fix ==")
        print("\n".join(hard))
    if soft:
        print(f"\n== warnings ({len(soft)}) — fix or consciously accept ==")
        print("\n".join(soft))
    if not hard and not soft:
        print("CLEAN")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
