# Mechanical verification — trust the scan, not the impression

Adapted from the novel-translator skill's zero-source-character pipeline: the
insight is that long-document generation degrades silently as context grows,
and the only reliable countermeasure is a *mechanical* per-segment check loop,
not self-assessment. Same discipline, different failure modes.

## The loop (per section, mandatory)

```
draft section → run verify_paper.py → fix hard failures → re-run
             → repeat until CLEAN → compile PDF → only then next section
```

Two fix passes are normal: fixes themselves introduce errors (a rewritten
sentence can reintroduce a banned word; a bib edit can orphan a cite key).
Never mark a section done on the first pass without a clean re-scan.

## Hard failures (block progress)

| Check | Why |
|---|---|
| CJK characters in any `.tex`/`.bib` | Conversation runs in Chinese; script leakage into the English report is the direct analogue of translation source-leakage, and it *increases* as the session grows |
| `[CLAIM NEEDS EVIDENCE]`, `PLACEHOLDER_`, `TODO` markers | Unresolved claim-ledger or citation debts must not survive into a "done" section |
| `\cite{key}` with no matching `refs.bib` entry (or malformed bib) | Broken citations compile to `[?]` and are exactly where hallucinated references hide |

## Soft warnings (fix or consciously accept)

- AI-vocabulary hits (style.md blacklist) — allowed when the word carries
  technical meaning ("robust" in "robustness gate" is fine).
- Throat-clearing phrases, `\textbf`/`\emph` in body text.
- Bib entries never cited (dead weight or a forgotten citation site).

## Numbers audit (manual, per results section)

For every quantitative sentence, grep the number in `docs/results/*.csv` or
the generating table file. A number you "remember" from the conversation is
not evidence — this session alone produced a verification-run Sharpe (0.35)
that must never be confused with archived research numbers. When a number is
derived (difference, ratio), leave a `% derived: <how>` comment next to it.

## Glossary consistency (from the translation glossary discipline)

Maintain `paper/glossary.md`: term → the one English rendering used in the
report (e.g. decide once between "interconnector" / "cross-border line";
"composite alpha" never drifts to "combined factor"). Seed it from
`CONTEXT.md`'s ubiquitous language at scaffold time; update before starting
each new section; scan drafts against it when revising. One term, one
rendering, all 40+ pages.

## Context hygiene (the #1 enemy)

- Draft **one section per sitting**, with only that section's artifacts open.
- Re-read the artifact immediately before writing its numbers.
- Batch style polish across 2–3 sections, then verify all at once — cheaper
  and more consistent than polishing per paragraph.
- If hard-failure counts climb section over section, that is context
  saturation, not carelessness: stop and hand off to a fresh session with
  `/handoff`.
