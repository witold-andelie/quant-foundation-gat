# Citation workflow — nothing enters refs.bib unverified

Distilled from ml-paper-writing + citation-verification (Galaxy-Dawn/
claude-scholar). AI-generated citations run a ~40% error rate; a fabricated
reference in a graded Master report is misconduct, not a typo.

## The rule

**Never write a BibTeX entry from memory.** Path for every citation:

1. Find the identifier: DOI > arXiv ID > publisher landing page.
   Use WebSearch/WebFetch; Semantic Scholar and CrossRef are the
   verification authorities, Google Scholar is discovery only.
2. Confirm title, first author, year, venue against the identifier's
   landing page.
3. Fetch BibTeX programmatically:
   - CrossRef: `https://api.crossref.org/works/<DOI>/transform/application/x-bibtex`
   - arXiv: export page → BibTeX
4. If citing a *specific claim*, confirm the claim appears in the paper
   (abstract or fetched text), and note the section.
5. Only then add to `paper/refs.bib`.

Unverifiable → `\cite{PLACEHOLDER_author_year}` + `% TODO verify` comment,
and tell the user how many placeholders exist. Never silently invent a
similar-sounding paper.

## Expected core bibliography (verify each before use — these are leads, not entries)

Anchors this report will plausibly cite; all still need step 1–3 above:

- Kakushadze, *101 Formulaic Alphas* (2016) — WorldQuant-style alpha
  construction. arXiv:1601.00991.
- Veličković et al., *Graph Attention Networks* (ICLR 2018). arXiv:1710.10903.
- Brody, Alon, Yahav, *How Attentive are Graph Attention Networks?* (GATv2,
  ICLR 2022). arXiv:2105.14491.
- Fey & Lenssen, *Fast Graph Representation Learning with PyTorch Geometric*
  (2019). arXiv:1903.02428.
- Gu, Kelly, Xiu, *Empirical Asset Pricing via Machine Learning* (RFS 2020) —
  ML-for-alpha context.
- Lopez de Prado, *Advances in Financial Machine Learning* (2018) — leakage,
  backtest overfitting; directly relevant to the E12/E13 post-mortem.
- ENTSO-E Transparency Platform — data source citation (cite as dataset/
  platform, note access dates).
- Weron, *Electricity price forecasting: A review* (IJF 2014) — energy track
  baseline literature.
- Tool citations if the venue expects them: DuckDB (Raasveldt & Mühleisen),
  dbt, PyTorch.

## Style

- Use `biblatex` numeric or author-year (pick once; IEEE-style numeric fits
  an informatics Fachbereich; confirm with the user at scaffold time).
- Cite where the claim is made, not in decorative clusters at paragraph ends.
- The repo's own experiment log is *not* a citation — it is the evidence base;
  reference it as an appendix pointer if needed.
