# LaTeX conventions and layout for the MPIN report

## Project layout

```
paper/
├── main.tex          # documentclass, packages, metadata, \input chain
├── sections/         # one .tex per chapter
├── figures/          # generated from docs/results/*.csv (matplotlib, colorblind-safe)
├── tables/           # generated .tex tables — never hand-type result numbers
├── refs.bib          # verified entries only (see citations.md)
└── Makefile          # or latexmk config
```

## Document setup

- Class: `scrreprt` (KOMA) or `report`, 11pt, A4, oneside — a Master project
  report, not a two-column conference paper.
- `biblatex` + `biber`; numeric (IEEE-like) style unless the user chooses
  author-year at scaffold time.
- Title page carries the fixed metadata block from SKILL.md (Westfälische
  Hochschule, Fachbereich Informatik und Kommunikation, **Master** Informatik,
  Master-Projekt Informatik (MPIN), supervisor Prof. Laura Anderle). Include a
  standard declaration-of-originality page (Eidesstattliche Erklärung) —
  German UAS reports expect one; confirm exact wording with the user.
- `booktabs` for tables, `siunitx` for numbers/units, `graphicx`, `hyperref`
  last (before `cleveref` if used).

## Report skeleton (adapt at outline stage, don't impose)

1. Introduction — motivation, research questions, contributions
2. Background & Related Work — cross-sectional alphas, GNN/GAT, electricity
   price forecasting
3. Platform Architecture — ingestion → warehouse → dbt marts → dashboard
4. Equity Track: GAT Relational Factors — design, gates, A/B, results
5. Energy Track: Forecasting — baselines → GAT, E12 leakage post-mortem,
   E13/E13b honest verdict
6. Discussion — what held, what failed, threats to validity
7. Conclusion & Future Work
8. Appendices — reproducibility (commands, versions), extended tables

## Numbers pipeline

Generate result tables/figures by script from `docs/results/*.csv` into
`paper/tables/` and `paper/figures/` — hand-typed numbers drift from the
artifacts and break the claim ledger. Keep the generating script in `paper/`.

## Compile loop

Windows host: check `Get-Command latexmk, pdflatex, tectonic` first; if no
TeX distribution, tell the user (MiKTeX or `winget install TeXLive.TeXLive`;
tectonic is the lightweight alternative) rather than silently producing
uncompilable sources.

Every edit batch: `latexmk -pdf main.tex` (or tectonic), then:
1. Scan the log — undefined references, missing citations, overfull hboxes.
2. Open the PDF and look at changed pages; never judge layout from source.

## Layout fixes (from nature-polishing latex-layout)

- Loose/sparse page or stranded heading → check float placement before
  touching text; `\usepackage[section]{placeins}` + `\FloatBarrier` beats
  scattering `[H]`.
- Figure split across pages / "Float too large" → the figure is oversized
  for the text block: regenerate it at the right aspect ratio at the source
  (matplotlib `figsize`), don't `\resizebox` distortions.
- Wide multi-panel figures → regenerate taller at the source rather than
  shrinking to illegibility.
- Widows/orphans and one-line section tails → reword the paragraph (style.md
  pass) before reaching for `\enlargethispage`.
