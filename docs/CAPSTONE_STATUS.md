# GNN Capstone — Progress Snapshot

Resume point for the GNN/GAT relational-factor extension. The original GitHub
repo (data-engineering platform) is prior work; everything below is the new
capstone contribution. Equity track first; energy deferred (ADR-0004).
Experiment record for the paper: `docs/gat_experiment_log.md`.

_Last updated: 2026-06-10._

## Status in one line

**Now dual-track (ADR-0006), both run on REAL data.** The equity axis is
closed, tested, ablated, seed-qualified, HP-tuned, and attention-analysed on
real yfinance data (E1-E10); the energy axis is built on the same GAT kernel —
physical interconnector graph (20 zones) + hourly label + bidding-zone nodes
(`run_gat_energy`) — and run on synthetic (E11) and **live ENTSO-E** data
(E12, token validated, 20/20 EIC codes). Flow per track: panel -> graph -> GAT
training (IC loss; single or walk-forward) -> composite + no-learning anchors
-> four gates + attention A/B. **55 new-module tests passing; no
`NotImplementedError` left in `src/`.** Equity has real-data value (3/4 gates,
attention 30/30 positive). **Energy: synthetic is a clean negative control;
real ENTSO-E produced implausible Sharpe-8/11 metrics diagnosed across E12->E13
->E13b as THREE nested artifacts (overlap, annualisation, evaluation-return
clipping). Under the honest unclipped return the energy cross-sectional
strategy LOSES money (Sharpe ~ -1.5, GAT and trivial `-price` alike); rank-IC
~0.21 does not translate to PnL (C13/C14). Fix shipped (unclipped eval). Energy
is a cautionary-methodology contribution, NOT a value claim.** Split hygiene locked (ADR-0003 amendment);
leakage controls automated; HP selection used valid IC only and transferred
to OOS in the walk-forward arm (E9/E9b, device-deconfounded). Headline:
**attention value-add over the uniform anchor is positive in 30/30 seeded
runs**; best known setup (static graph + IC loss + walk-forward + lr
3e-3/hidden 64/heads 2/layers 2, CPU) reads **OOS IC 0.0148 +/- 0.0043 (5/5
positive), OOS Sharpe 1.37 +/- 0.39** over 5 seeds; walk-forward-vs-single
is significant in this config (IC t~3.9). Value-added (vs best single,
3.07) is the one open gate.

## Decisions locked (see docs/adr/)

- **0001** — `propagate` seam: one snapshot in, one factor per node out;
  topology in, adapters weight internally; transform-only; directed; output is
  the factor (attention via a side method).
- **0002** — unified `Factor` + `FactorProvider` seam; canonical `(time, entity)`
  panel; energy wrapped as legacy provider; GNN factor is a provider.
- **0003** — GAT training objective: cross-sectionally standardised
  `forward_return(t+k)` label; MSE first, then IC loss; walk-forward + embargo
  (>= k); leakage-critical code is pure pandas + tested.
- **0004** — scope: equity end-to-end first; energy deferred (superseded by 0006:
  dual-track as of 2026-06-11; energy island alphas wired, relational built).
- **0005** — equity graph: correlation top-k backbone + optional sector boost +
  `min_degree` fallback; node features = `_rank` alphas, cross-sectional median
  fill; label k in days; universe expanded to ~50 names with GICS sectors.
- **0006** — energy graph: physical interconnector topology (~20 European
  bidding zones) + correlation weights; floored hourly label (k in hours);
  one shared GAT kernel, two heterogeneous graphs.

## New modules (all under src/quant_alpha/, English comments)

| File | Role |
|---|---|
| `graph/propagate.py` | `Propagator` seam; `UniformMeanPropagator` (baseline, A/B anchor); `GATPropagator` (wraps trained GATModel, torch lazy; `last_attention` exposes head-layer softmax for M4) |
| `graph/training.py` | torch-free leakage primitives: `cross_sectional_label`, `energy_cross_sectional_label` (floored hourly return), `cross_sectional_median_fill`, `walk_forward_splits` (embargo), `is_constrained_split` (valid inside IS), `rank_ic` |
| `graph/edges_equity.py` | `build_equity_topology` (corr top-k + sector + min_degree, leak-safe), `static_topology_for`, `rolling_topology_for` (dynamic, point-in-time) |
| `graph/edges_energy.py` | physical interconnector graph: `EUROPEAN_INTERCONNECTORS` (~20 zones), `build_energy_topology` (physical edges + corr weights, leak-safe), `static_energy_topology_for`, `rolling_energy_topology_for` |
| `graph/attention.py` | torch-free M4 analysis over the tidy attention frame: `self_attention_over_time`, `sector_homophily_over_time`, `attention_concentration_over_time`, `hub_scores`, `attention_matrix` |
| `features/factor.py` | unified `Factor`, `FactorProvider`, `apply_factors`, `propagate_over_panel`; `ExpressionFactorProvider`, `GraphFactorProvider`, `LegacyEnergyProvider` (wraps the 8 imperative energy alphas as Factors, memoised — no stubs left) |
| `models/gat.py` | torch zone (needs `[gnn]`): `GATModel` (+`forward_with_attention`), `GATConfig`, `CrossSection`, `FactorGraphDataset`, `build_sections` (+`label_fn` hook), `fit`, `composite_series`, `walk_forward_composite_series`, `attention_panel`, `predict_panel`, losses, `time_ordered_split` |
| `run_gat_equity.py` | equity axis: `gat_equity_from_panel` (orchestration; `loss`/`graph`/`retrain` switches), `run_gat_equity` (CLI wrapper, `persist=` to DuckDB), `gate_report` (four gates), `ab_report` + `_baseline_columns` (attention A/B anchors — reused by energy), `gat_warehouse_frames`/`persist_gat_outputs` (4 dbt source tables) |
| `dbt_quant_alpha/` | GAT marts: `stg_gat_panel`, `fct_gat_vs_baseline` (tiered relational A/B), `fct_gat_scorecard` (one-row gates + attention A/B); source `gat_relational` |
| `run_gat_energy.py` | energy axis (dual-track): `gat_energy_from_panel`, `run_gat_energy` — same kernel, physical interconnector graph + hourly label + bidding-zone nodes |

Config: `Universe.sectors` added (`config.py`); `configs/universe.yaml` = 50 names
+ sectors; `[gnn]` extra in `pyproject.toml` (torch, torch-geometric).

## Tests (tests/, run with the env note below)

`test_graph_propagate.py` (3) · `test_factor_provider.py` (7, incl. energy shim
faithfulness + memoise) · `test_training.py` (8, incl. energy label) ·
`test_edges_equity.py` (9) · `test_edges_energy.py` (5, interconnector graph)
· `test_gat.py` (7) · `test_run_gat_equity.py` (2) · `test_run_gat_energy.py` (3,
incl. all-NaN-alpha drop) · `test_leakage.py` (4) · `test_attention.py` (7)
= **55 passing** (torch tests `importorskip`, run here because torch is installed).

## How to run / resume

Environment quirks on this machine:
- Use the `py` launcher (Python 3.13). The bare `python` is a Microsoft Store stub.
- `torch 2.12.0+cu126` + `torch_geometric` installed; **CUDA works** (RTX
  4060 Laptop 8GB, driver CUDA 12.6; swapped from the CPU wheel 2026-06-10).
  `fit` auto-selects cuda; `_dataset_to_device` pre-moves all sections once
  so per-snapshot transfers don't eat the gain. `yfinance` installed
  2026-06-10 — the live `run_gat_equity` fetch path works (run it via a script,
  not the CLI: `cli.py` imports dlt/duckdb modules at the top). `duckdb` is
  still NOT installed, so duckdb-backed tests can't run here.
- Set `PYTHONPATH` to `src` (pytest also has `pythonpath=["src"]`).
- Parallel experiment runs: 32 logical cores but only **15GB RAM** — a worker
  running the full `gat_equity_from_panel` (training + evaluate_alpha_suite)
  peaks ~2GB, so cap at **3 concurrent full-pipeline workers** (8 OOM'd with
  BrokenProcessPool). Train-only workers (HP grid) are light; 8 are fine.
  Cap `torch.set_num_threads(4-8)` per worker — on these tiny graphs fewer
  threads is *faster* (4-thread runs beat 32-thread by ~30%).

```powershell
cd D:\AI_Models\quant-alpha-foundation
$env:PYTHONPATH = "D:\AI_Models\quant-alpha-foundation\src"
py -m pytest tests/test_run_gat_equity.py tests/test_gat.py tests/test_edges_equity.py tests/test_training.py tests/test_factor_provider.py tests/test_graph_propagate.py -q
```

CLI (needs yfinance for the live path): `quant-alpha gat-equity --offline --epochs 50`

## Demo result (synthetic prices, 50 names) — and how to read it

On random-walk synthetic data the gates correctly report **no edge**: Value-added
and Consistency FAIL (negative OOS IC ~ -0.06, Sharpe value-add negative);
Uniqueness and Robustness PASS. This is the desired behaviour — a leakage-safe
pipeline produces no false positives on noise. Real market data is where the GAT
gets a chance to add value.

## Real-data result (2026-06-10, yfinance 2021->now, 49 names, IC loss, 50 epochs)

**3 of 4 gates pass** — Uniqueness (max |corr| 0.257), Consistency (0.63),
Robustness (0.68); Value-added FAILS (composite OOS Sharpe 1.42 vs best
single 2.88, but it beats 9 of the 10 singles). Valid IC peaked 0.0756 at
epoch 32; best-epoch selection deployed that epoch (final epoch had decayed
to 0.0418 — the checkpoint fix nearly doubled the deployed valid IC).

**Full record — setup, training curve, all-factor diagnostics, limitations,
next experiments — lives in `docs/gat_experiment_log.md` (entry E5), the
canonical experiment log for the paper.** Artifacts:
`docs/results/2026-06-10_gat_real_run_diagnostics.csv` + the run script
alongside it; weights `data/warehouse/gat_equity.pt`.

## Architecture map

`infra.txt` (repo root) — full production framework as an OPM model (DOT).
Render: `dot -Tsvg infra.txt -o out.svg`.

## Next steps (prioritised)

Done (all 2026-06-10, full records in `docs/gat_experiment_log.md`):
IC loss default (E4) · first real-data run (E5) · 2x2 graph-x-retraining
ablation with A/B anchors (E6) · seed sensitivity (E7) · GPU benchmark —
negative, CPU stays (E8) · HP grid by valid IC + OOS winner validation (E9)
· attention plumbing (`last_attention`) · split hygiene + automated leakage
controls (E2/E3) · attention qualitative analysis (E10/M4) · energy
relational track (E11, synthetic negative control) · **real ENTSO-E data
wired + fetched (E12): token validated, 20/20 EIC codes, 2024 full-year 20
zones — diagnosed the real-data metrics as leakage artifacts (C13), a
cautionary methodology result.** 55 tests.

Remaining, in order:

1. ~~Energy label/eval redesign~~ — **DONE, E13/E13b (airtight verdict).** The
   daily redesign fixed overlap/annualisation/leak; a challenge-prompted deeper
   check found the evaluation-return clip hid short-leg tail losses. Under the
   honest unclipped return the energy strategy LOSES money (Sharpe ~ -1.5, GAT
   and trivial `-price` alike). Conclusion: no tradeable energy alpha with
   day-ahead price-return targets. Fix shipped (unclipped eval default). A
   genuine tradeable target would need balancing/intraday spreads or FTR-aware
   PnL (data unavailable) — out of capstone scope.
2. **Value-added gate variants** — the strict max-of-singles bar (3.07) is
   the one open gate; add mean-of-singles and marginal-contribution-to-a-
   multifactor-portfolio readings before concluding the composite adds
   nothing beyond the best island alpha.
3. **Platform integration (M5)** — ~~composite into dbt marts~~ **DONE**:
   `persist_gat_outputs` writes 4 tables to DuckDB; dbt models `stg_gat_panel`
   + `fct_gat_vs_baseline` (tiered A/B) + `fct_gat_scorecard` (one-row gates +
   attention A/B), built and tested end-to-end (`dbt run`/`test` PASS via the
   `D:\duckdb` venv). Remaining: Streamlit "GAT vs Baseline" page (E6 matrix,
   E7/E9 seed distributions, E10 attention figures).
4. **Paper assembly** — the evidence map (C1-C14) and narrative order are
   ready in the experiment log; limitations list in E5 + static-graph
   lookahead + stationary-attention (E10) + energy-on-synthetic (E11).
   Data-source upgrade (survivorship-bias-free vendor) if time permits.
5. **WF-vs-single hardening** (optional) — already significant in the tuned
   config (E9b); more seeds only tighten the interval.

## Timeline (proposed to advisor)

Final codebase by **2026-06-30**; paper first draft by **2026-07-15** (pending
advisor confirmation).
