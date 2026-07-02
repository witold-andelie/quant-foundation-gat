# English prose style — reading like a person, not a model

Distilled from nature-polishing (Yuan1z0825/nature-skills), the English-LaTeX
de-AI rules (BoHeFan/academic-paper-writer-pro-2), and the writing-quality
checklist (Imbad0202/academic-research-skills). These are good-writing rules,
not detection evasion: the goal is prose a tired reviewer reads without
friction.

## Modification threshold

If a passage is already natural — varied rhythm, no buzzwords, clean logic —
**leave it untouched**. Editing for the sake of editing degrades text.

## Vocabulary

Replace AI-overused words with plain alternatives (only when they carry no
precise technical meaning):

| Avoid | Prefer |
|---|---|
| leverage | use, employ |
| delve into | examine, investigate |
| pivotal / paramount | key, central / important |
| underscore | show, emphasize |
| robust (as filler) | reliable, stable — or the specific property |
| seamless / holistic | integrated / comprehensive |
| cutting-edge / groundbreaking | state-of-the-art (sparingly), novel |
| paradigm / realm / landscape | approach / area / field |
| burgeoning / multifaceted / nuanced | growing / complex / subtle |
| unprecedented | new, notable |

Keep domain terms exact and untouched: alpha, IC, Sharpe, walk-forward,
GATv2, cross-sectional, residual load, day-ahead spot, leakage, OOS. Never
"synonymize" a technical term, identifier, path, or math expression.

## Structure

- **Prose over lists.** Convert `itemize`/`enumerate` runs into connected
  paragraphs unless the content is genuinely enumerable (parameter tables,
  gate criteria). A report full of bullets reads like slides.
- **Cut throat-clearing.** Delete "It is worth noting that", "It is important
  to note", "First and foremost", "In order to" (→ "to"), "Last but not
  least". Let sentence order carry the logic.
- **Vary rhythm.** Alternate sentence lengths; avoid three same-shaped
  sentences in a row and uniform paragraph sizes. Read a paragraph aloud
  mentally — monotone means rewrite one sentence, not all of them.
- **Limit em dashes** to at most one per paragraph; prefer commas,
  parentheses, or a relative clause.
- **No bold/italic emphasis in body text** (`\textbf`, `\emph` reserved for
  definitions on first use, if at all). Emphasis comes from sentence
  position: put the load-bearing clause last.

## Stance and claims

- State what was done and measured in plain past tense; reserve present
  tense for what the artifact *is*.
- One claim per sentence near evidence; put the number and the pointer
  (table/figure) in the same sentence.
- Boundary every positive claim ("on this synthetic universe", "over
  2015–2024 daily bars") — the boundaries are what make the negative results
  credible.
- First person plural ("we") is fine and preferable to passive contortions;
  this is a single-author report, so "we" = author + reader convention or
  use "this project".

## Section jobs (fix the section before the sentence)

- **Abstract**: problem → approach → the 2–3 headline numbers → the honest
  verdict. No citations, no suspense.
- **Introduction**: why relational structure for alpha research, why the two
  tracks, contributions as claims the body will defend — each mapped to a
  section.
- **Methods/Architecture**: reproducible description; a reader with the repo
  should find every named component.
- **Experiments/Results**: chronology is not structure — organize by
  question answered (E-series entries group naturally).
- **Discussion**: interpret, bound, compare to the anchors; the leakage
  post-mortem belongs here as a first-class finding.
- **Conclusion**: what a follow-up project should do differently; no new
  claims.
