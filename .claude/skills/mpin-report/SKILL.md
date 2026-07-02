---
name: mpin-report
description: Write and revise the quant-alpha-foundation technical report as a LaTeX document in English — the Ausarbeitung for the Master-Projekt Informatik (MPIN, 12 ECTS) at Westfälische Hochschule, Gelsenkirchen. Use when the user asks to write the paper, technical report, thesis-style writeup, LaTeX report, 写论文, 技术报告, 排版, polish a section, add citations, or reduce AI-sounding prose in the report.
---

# quant-alpha-foundation Technical Report (MPIN)

Produce and maintain `paper/` — an English LaTeX technical report on this
repository's two research tracks (equity GAT relational factors; European
power-market forecasting). Converse with the user in Chinese; write the
report in English.

## Fixed metadata (never re-ask, never change without instruction)

| Field | Value |
|---|---|
| Author | Wentao Ma |
| Institution | Westfälische Hochschule — Westphalian University of Applied Sciences, Gelsenkirchen |
| Faculty | Fachbereich Informatik und Kommunikation (Department of Computer Science and Communication) |
| Programme | **Master** Informatik (M.Sc. Informatics) — Master's level, not Bachelor |
| Module | Master-Projekt Informatik (MPIN), 12 ECTS / 360 h, PO2023, semesters 2–3 |
| Supervisor | Prof. Laura Anderle |
| Deliverable | Ausarbeitung: written report + developed software + presentation of results |

## Non-negotiable rules

1. **Claim ledger.** Every quantitative or comparative claim must trace to a
   repository artifact (see [references/evidence-map.md](references/evidence-map.md)).
   No artifact → mark `[CLAIM NEEDS EVIDENCE]`, never polish an unsupported claim.
2. **Negative results stay.** This project's distinguishing feature is honestly
   reported failures (leakage post-mortem, money-losing strategy, failed
   value-added gate). Report them as findings, not embarrassments.
3. **No citation from memory.** Follow
   [references/citations.md](references/citations.md): verify via DOI/CrossRef/
   arXiv/Semantic Scholar before any BibTeX entry; unverifiable → explicit
   `PLACEHOLDER_` key + tell the user.
4. **Human-sounding English.** Apply
   [references/style.md](references/style.md) while drafting, not as a
   post-pass. If a passage already reads naturally, leave it alone.
5. **Compile before judging.** Never assess layout from `.tex` source; build
   the PDF and look at it ([references/latex.md](references/latex.md)).
6. **Mechanical verification after every section.** Run
   `scripts/verify_paper.py` and fix all hard failures before moving on —
   see [references/verification.md](references/verification.md). Trust the
   scan, not your impression of what you wrote.
7. **Context hygiene.** Long context is the main hallucination driver in
   long-document work: draft one section per sitting with only that
   section's evidence artifacts open; re-read the artifact before writing
   its numbers, never quote them from conversational memory. If drafting
   quality degrades or a session has grown long, hand off to a fresh session
   (`/handoff`) instead of pushing on.

## Workflow

### First invocation (no `paper/` yet)
1. Read `CONTEXT.md`, `docs/gat_experiment_log.md`, `docs/CAPSTONE_STATUS.md`,
   `docs/energy_forecasting.md`, `docs/results/*.csv` — the report is written
   *from* these, not from memory of the conversation.
2. Propose a section outline (see report skeleton in
   [references/latex.md](references/latex.md)) and the evidence each section
   will draw on; get the user's verdict before drafting (per their workflow:
   audit → per-item verdict → atomic batch).
3. Scaffold `paper/` (main.tex, sections/, refs.bib, Makefile snippet) and
   draft section by section, citing as you write.

### Per-section loop (draft and revision alike)
1. Open only the evidence artifacts this section draws on; draft.
2. Run `py -3.13 .claude/skills/mpin-report/scripts/verify_paper.py paper/`.
3. Fix hard failures (CJK leakage, unresolved markers, cite/bib mismatch),
   rerun until clean — two passes is normal; fixes can introduce new errors.
4. Compile, scan the log, eyeball changed pages.
5. Update `paper/glossary.md` with any new term-of-art before the next
   section (one term, one rendering — consistency beats variety).

### Revision passes
- Section rewrite / polish → load [references/style.md](references/style.md);
  state which section and what failure mode you are fixing.
- Citation work → load [references/citations.md](references/citations.md).
- Layout complaints (loose pages, split figures, floats) → load
  [references/latex.md](references/latex.md) §Layout, skip the prose rules.
- After every edit batch: recompile, check the log for warnings, eyeball the
  changed pages.

## Priority order when editing prose

Paper-type architecture → section's job → paragraph logic →
claim/evidence/boundary → sentence polish. Fix the highest broken level
first; a beautiful sentence in a paragraph with no argument is wasted work.

## What NOT to do

- Do not invent related work, datasets, or numbers to fill gaps.
- Do not restructure into bullet-heavy "AI slide" prose; the report is
  flowing technical prose with tables/figures where the data lives.
- Do not commit anything; leave git operations to the user's verdict.
- Do not re-derive results by rerunning pipelines unless asked — the archived
  numbers in `docs/results/` are canonical.
