# Evidence map — where every claim in the report must trace to

The claim-ledger gate (from ml-paper-writing): before a sentence asserts a
number, comparison, or capability, identify the artifact that supports it.
Artifacts below are canonical; conversation memory is not evidence.

## Canonical artifacts

| Evidence | Artifact |
|---|---|
| Equity GAT experiments E1–E10 (design, seeds, gates, attention analysis) | `docs/gat_experiment_log.md` |
| Energy experiments E11–E13b (synthetic → ENTSO-E → leakage → honest verdict) | `docs/gat_experiment_log.md`, `docs/energy_forecasting.md` |
| Archived headline numbers | `docs/results/*.csv` (equity_gat_summary, energy_gnn_findings, energy_forecast_node_skill, energy_forecast_edge_skill) |
| Capstone scope, milestones M1–M5, resolved decisions | `docs/CAPSTONE_STATUS.md`, `docs/gnn_capstone_design.md` |
| Domain language, ubiquitous terms | `CONTEXT.md`, `docs/adr/` |
| Platform architecture (ingestion → warehouse → dbt marts → dashboard) | `docs/architecture.md`, `dbt_quant_alpha/README.md`, `dbt_energy_alpha/README.md` |
| Canonical model weights | `data/warehouse/gat_equity.pt` (tracked; do not regenerate for the paper) |
| Reproducibility claims | test suite (115 tests), `Makefile`, `pyproject.toml` |

## Load-bearing findings the report must state accurately

Verify exact numbers from the artifacts before writing them; the shapes are:

1. **Equity composite fails the value-added gate** — composite OOS Sharpe
   does not beat the best single island alpha; consistency, uniqueness,
   robustness gates pass. The A/B (learned attention vs uniform vs island)
   is the scientific core.
2. **Energy E12 leakage post-mortem** — the first real-data result was an
   artifact of target leakage; the redesigned E13/E13b honest-return check
   shows the strategy loses money. This is a *cautionary-tale contribution*,
   not a failure to hide.
3. **Platform contribution stands independent of alpha performance** —
   ingestion (dlt), warehouse (DuckDB), marts (dbt), dashboard (Streamlit),
   dual-track design, leak-safe evaluation harness.

## Rules

- A number that appears in the report must appear in (or be arithmetically
  derived from) an artifact above; cite table/figure built from that CSV.
- Claims about wall-clock, hardware, or library versions → check
  `pyproject.toml` / the environment, do not guess.
- If the user supplies a new number verbally, ask for the artifact or mark
  `[CLAIM NEEDS EVIDENCE]`.
- Master-level framing: the report is graded as a Master project
  (MPIN) — emphasize methodology, critical evaluation, and engineering
  judgment over raw results.

## Threats-to-validity vocabulary (Discussion section)

Use the standard quant-research terms — reviewers pattern-match on them:
**look-ahead bias** (the E12 leakage incident is literally this), **survivorship
bias** (acknowledged data limitation, see CAPSTONE_STATUS), **overfitting /
data snooping** (multiple alphas tested on one dataset; the gates exist to
police this), **transaction costs** (the E13b honest-return check is where
costs enter), **regime dependence** (synthetic vs ENTSO-E, static-graph
lookahead). Map each named threat to where the project mitigated it or
explicitly did not — an admitted, bounded threat reads as rigor.
