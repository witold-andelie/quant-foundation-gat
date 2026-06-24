# GAT Relational Factor — Experiment Log (paper reference)

The canonical record of every GAT relational-factor experiment across **both
tracks** (equity correlation graph + energy interconnector graph): setup,
integrity controls, numbers, and interpretation. This is the primary source
for the capstone paper's methods and results sections. Companion docs: design
rationale in `gnn_capstone_design.md`, decisions in `adr/0001`-`0006`, resume
point in `CAPSTONE_STATUS.md`, domain glossary in `CONTEXT.md`.

_Maintained chronologically; never rewrite an entry — append corrections so the
self-correction trail (e.g. E9 → E9b device deconfound) stays auditable, which
is itself a credibility asset for the paper._

**Contents:** evaluation framework (methods) · headline results (summary table)
· paper evidence map (C1-C14) · narrative order · consolidated limitations ·
reproducibility · chronological entries E1-E13.

---

## Evaluation framework (methods reference)

The fixed measurement apparatus every experiment runs through. Defined once
here so the paper's methods section is a single lift.

**Canonical panel.** `(time, entity)` MultiIndex — `(date, symbol)` for equity,
`(timestamp, market)` for energy. Node features are the existing alphas'
cross-sectional `_rank` columns (10 equity / 8 energy), median-filled per
snapshot (the no-information position in rank space).

**Label (supervision only, never a feature).** Forward return at horizon `k`,
cross-sectionally standardised per snapshot.
- Equity: `price[t+k]/price[t] - 1`, z-scored; `k` in **days** (k=5).
- Energy: `(price[t+k]-price[t]) / clip(|price[t]|, 20)`, clipped to ±0.8,
  z-scored; `k` in **hours** (k=24). The floor is required because power prices
  go negative/near-zero (ADR-0004/0006).

**Split protocol (leakage-safe).** One IS/OOS split date at 70% of timestamps,
the single source of truth for graph construction (`as_of`), model selection,
and gate evaluation. Layout over snapshot indices:
`train | embargo(k) | valid | embargo(k) | OOS`. Valid sits inside IS with an
embargo on both sides, so best-epoch selection never sees the OOS window
(`graph.training.is_constrained_split`, asserted). The graph is built from data
strictly before `as_of` (asserted in the builders).

**Training.** Stacked GAT (PyG `GATConv`), 2 layers, IC loss
(`-Pearson(pred, label)` per snapshot; MSE retained for A/B), best-by-valid-IC
epoch checkpointing. `single` = one fit on IS; `walk_forward` = refit every
`oos_chunk` snapshots through OOS on all data predating each boundary (same
embargoed split per fold). CPU (E8: GPU slower at this graph size).

**The four research gates** (`run_gat_equity.gate_report`), composite vs the 10
(or 8) single alphas, all on the OOS slice:
| Gate | Definition | Pass threshold |
|---|---|---|
| Value-added | composite OOS Sharpe vs the **max** single-alpha OOS Sharpe | composite > best single |
| Consistency | IS and OOS IC same sign + magnitude retention (`consistency_score`) | score ≥ 0.5 |
| Uniqueness | max abs Spearman corr of composite vs each single | < 0.7 |
| Robustness | OOS observations + IC IR + drawdown blend (`robustness_score`) | score ≥ 0.5 |

Value-added's "beat the max of N singles" is deliberately the strictest bar;
the paper also reports beat-rate (e.g. "beats 9 of 10").

**The attention A/B (the core thesis test).** Two no-learning anchors carried
in every run, same inputs and same topology as the GAT:
- `alpha_island_mean` — equal-weight composite of the input alphas, **no
  propagation** (the island baseline).
- `alpha_uniform_composite` — `UniformMeanPropagator` averaging that composite
  over the GAT's topology (**relational but unlearned**).
`attention_sharpe_value_add` = GAT OOS Sharpe − uniform-anchor OOS Sharpe
isolates what *learned attention* adds over naive neighbour averaging.

---

## Headline results (summary)

One table for the paper's results section. Equity is real yfinance data
(2021-01→2026-06, 49-50 US names); energy is synthetic power data (ENTSO-E
token-gated). All multi-seed numbers are mean ± std over 5 seeds, CPU.

| Track / config | OOS IC | OOS Sharpe | Attention value-add | Gates (V/C/U/R) | Source |
|---|---|---|---|---|---|
| **Equity, winner + walk-forward** (best) | **0.0148 ± 0.0043** | **1.37 ± 0.39** | **2.42 ± 0.39** | F/T/T/T | E9b |
| Equity, winner + single | -0.0004 ± 0.0077 | 0.80 ± 0.35 | 1.86 ± 0.35 | mixed | E9b |
| Equity, default + walk-forward | 0.0055 ± 0.0147 | 1.15 ± 0.75 | 2.20 ± 0.75 | F/T/T/T | E7 |
| Equity, default + single | -0.0009 ± 0.0045 | 0.73 ± 0.25 | 1.78 ± 0.25 | F/F/T/T | E7 |
| Equity, first real run (1 seed) | 0.0066 | 1.42 | n/a | F/T/T/T | E5 |
| Energy, winner + walk-forward (synthetic) | -0.011 | -1.11 | -0.59 | F/F/T/T | E11 |
| Energy, winner + single (synthetic) | -0.009 | -1.26 | -0.74 | F/F/T/T | E11 |
| Energy, REAL ENTSO-E hourly (artifact, E12) | 0.202 | 8.25 | +8.17 | +14.9 | T/T/T/T | E12 |
| Energy, REAL daily — clipped eval (artifact, E13) | 0.218 | 11.08 | — | — | T/T/T/T | E13 |
| Energy, REAL daily — HONEST unclipped eval (E13b) | 0.210 | **-1.54** | — | — | loses | E13b |

Reference points: best single equity alpha (`alpha_wq_010_gap_quality`) OOS
Sharpe ≈ 2.9-3.1; **attention value-add positive in 30/30 equity seeded runs**.
Equity passes 3/4 gates (Value-added open). **No energy row is a value claim**
(E12/E13/E13b, C13/C14): every positive energy Sharpe is an artifact (overlap,
annualisation, then return-clipping). Under the honest *unclipped* return the
energy cross-sectional strategy **loses money** (Sharpe -1.5, GAT and trivial
predictor alike). Equity is the project's genuine value demonstration; energy
is the cautionary-methodology result.

Headline reading: **learned attention reliably beats unlearned relational
propagation (the core thesis), the composite is a real/unique/consistent signal
that does not yet beat the single best alpha, and the same kernel generalises
to a physically different graph without manufacturing false positives.**

---

## Paper evidence map

Every claim the paper can make, with the experiment that supports it and the
artifact to cite. Update this table as entries land; a claim without a row
here is not yet supported.

| # | Claim | Evidence | Artifacts |
|---|---|---|---|
| C1 | The pipeline is leakage-safe and produces no false positives | E1 (synthetic negative control: four gates correctly find nothing on random walks) + E3 (automated shuffle-label negative control AND planted-signal positive control, both losses) + E2 (split protocol: one split date, valid inside IS, embargo both sides) | `tests/test_leakage.py`, ADR-0003 + amendment |
| C2 | IC loss is the right training objective: it directly optimises the evaluation metric and recovers signal far better than MSE (0.99 vs 0.33 planted-signal IC), at lr 1e-3 | E4 probe table | E4 entry; `LOSSES` in `run_gat_equity.py` |
| C3 | **Core thesis: learned attention adds value over naive relational propagation.** Same inputs, same topology: GAT beats the uniform-mean anchor in every E6 cell AND in **30/30 seeded runs across two HP configs and two devices** (+0.69 to +3.03 OOS Sharpe), with scores near-uncorrelated with the anchor (Spearman -0.29..-0.14) — it learns structure, not smoothing. Both no-learning anchors are negative OOS | E6 (all cells) + E7 + E9/E9b (all seeds) | `2026-06-10_matrix_summary.json`, `..._seed_sensitivity.csv`, `..._winner_validation_{gpu,cpu}.csv` |
| C4 | The relational composite is a real, unique, consistent signal: passes Uniqueness (max corr vs singles 0.26) and, with walk-forward, Consistency + Robustness — 3/4 gates | E5, E6 (static_wf) | matrix diagnostics CSVs |
| C5 | Honest negative: it does not (yet) beat the best single island alpha (the strictest Value-added bar); best config narrows the gap from -2.03 to -1.12 Sharpe while beating 9 of 10 singles | E5, E6 | gate reports in summary JSON |
| C6 | **Negative result: dynamic per-snapshot correlation graphs hurt** — per-date correlation estimates are noisy; the frozen train-period graph acts as a regulariser. Walk-forward retraining improves the mean in **all three** paired comparisons and reaches significance in the tuned config (E9b same-device: IC t~3.9, Sharpe t~2.4, n=5/arm) — claim "significant in the tuned config, directionally consistent everywhere" | E6 2x2 ablation + E7 + E9/E9b | E6/E7/E9 entries + artifacts |
| C7 | Model selection by valid IC matters in practice: valid IC peaked at epoch 32 (0.0756) then decayed to 0.0418 — best-epoch checkpointing nearly doubled deployed valid IC on the first real run | E5 training curve | E5 entry |
| C8 | Single-seed point estimates are not paper-grade: across 5 seeds OOS Sharpe spans 0.36-1.04 (single) and 0.13-1.95 (WF); same seed on a different device also diverges (E8). All headline numbers are reported as mean +/- std over >= 5 seeds; the long-short Sharpe is far more seed-stable in sign than the rank-IC | E7, E8 | `2026-06-10_seed_sensitivity.csv` |
| C9 | **The evaluation protocol is self-validating, with nuance**: HPs selected purely on the IS-internal valid IC transferred to OOS in the walk-forward arm (same-device: IC 0.0148 +/- 0.0043 vs default's 0.0055 +/- 0.0147 — 2.7x mean, 1/3 variance) but not in the single arm (a wash). Model selection never touched the OOS window. The initial both-arms read was a device artifact, caught and corrected (E9 correction) | E9 + E9b | `2026-06-10_hp_grid_valid_ic.csv` + `..._winner_validation_{gpu,cpu}.csv` |
| C10 | The HP surface is flat near the top (six configs within 0.10-0.115 valid IC); the only structural requirement is **2 GAT layers** (all top-6). Results are robust to reasonable HP choices — a robustness point, and a caution against HP-tuning theatre | E9 grid | `2026-06-10_hp_grid_valid_ic.csv` |
| C11 | **Attention is interpretable and mechanistically explains the result**: 91.6% on neighbours (not self-collapse), but near-uniform across ~12.5 neighbours (entropy 0.96, sector lift +0.019) — a gentle reweighting of mean pooling, which is *why* the edge over the uniform anchor is real but modest (C3). Hubs are economically sensible mega-caps/bellwethers. The structure is temporally stationary, so the "regime-adaptive" framing is **not** supported at the aggregate level | E10 | `2026-06-11_attention_*.csv`, `figures/` |
| C12 | **One kernel, two heterogeneous graphs (dual-track)**: a physical interconnector graph (20 bidding zones, hourly label) runs through the *same* GAT model, section builder, four gates, and A/B as the equity correlation graph — only graph/label/nodes differ. Synthetic energy is a clean negative control (attention VA negative; no false positives, C1) | E11 | `2026-06-11_energy_*.csv/json`, ADR-0006 |
| C13 | **Cautionary real-data result (methodology contribution)**: live ENTSO-E energy data produces implausible metrics (Sharpe 8.25, composite ~100x its best single, 4/4 gates) that are *artifacts* — overlapping 24h labels at hourly cadence (lag-1 autocorr 0.91), a trivial `-spot[t]` predictor with higher IC (0.235 > 0.20), wrong annualisation, and day-ahead lookahead. The project's own skepticism catches it; a valid energy claim needs a market-structure-aware label (vindicates ADR-0004) | E12 | `2026-06-11_energy_real_*`, E12 diagnosis |
| C14 | **The energy strategy LOSES money under honest returns — airtight negative verdict.** The daily redesign fixed overlap/annualisation/leak, but a deeper check (prompted by a challenge) found a third artifact: the evaluation return's `+/-0.8` clip caps the short-leg's scarcity-spike tail losses. Under the honest *unclipped* return both the GAT (Sharpe +11 -> **-1.54**) and a trivial `-price` predictor (+6.28 -> **-1.52**) lose money; rank-IC stays ~0.21 but does not translate to PnL. Every positive energy number (E12/E13) was an artifact (overlap, annualisation, return-clip). Fix shipped (unclipped eval). Lesson: never value-clip the evaluation return | E12 -> E13 -> E13b | `2026-06-11_energy_daily_*`, E13b |

**Suggested paper narrative order** (each step cites the rows above):
methods & seams (ADRs) -> evaluation protocol & leakage controls (C1, C7) ->
training objective (C2) -> main A/B result (C3) -> gates & honest reading
(C4, C5) -> ablation & negative results (C6) -> robustness & variance
(C8, C9, C10) -> attention interpretability & mechanism (C11) -> dual-track
generalisation: one kernel, two graphs (C12) -> a real-data
diagnose-redesign-challenge-verdict arc (C13 artifact -> C14 two more artifacts
caught, honest conclusion: the energy strategy loses money) showing rigorous,
self-correcting result-skepticism -> limitations -> future work.

---

## Consolidated limitations (state these in the paper)

Honesty is the paper's credibility strategy; every item below is a deliberate
scope choice, not an oversight, and several are themselves results.

1. **Value-added gate not passed** — the relational composite does not beat the
   single best island alpha (OOS Sharpe ~1.4 vs ~3.0). It beats 9 of 10 and is
   the second-best signal, but the strict max-of-singles bar is open (C5).
2. **Static-graph in-sample lookahead (mild)** — the static equity graph is
   built from correlations up to the split date, giving early *training*
   snapshots a mild lookahead. OOS cleanliness is unaffected (the graph
   precedes the OOS window); the dynamic graph removes it but was rejected on
   performance (C6). Disclose, don't hide.
3. **Attention is near-uniform / not regime-adaptive at the aggregate level**
   (E10/C11) — the value comes from small tilts on broad pooling, and the
   coarse attention structure is temporally stationary, contradicting the
   original "macro-regime-adaptive" hypothesis. Reported as a finding.
4. **The energy strategy loses money under honest returns (concluded)** —
   synthetic is a clean negative control (C12); real hourly is a leakage
   artifact (E12); the daily redesign fixes overlap/annualisation/leak but a
   deeper check (E13b) found the evaluation return's `+/-0.8` clip was hiding
   short-leg scarcity-spike tail losses. Under the honest *unclipped* return
   both the GAT and a trivial `-price` predictor have **negative** OOS Sharpe
   (~ -1.5); rank-IC ~0.21 does not translate to PnL. Energy yields no
   tradeable alpha; it is a cautionary-methodology contribution (three nested
   artifacts, each caught). Equity is the genuine value result.
5. **Walk-forward significance is config-local** — significant in the tuned
   config (E9b, t~3.9) and directionally consistent across three paired
   comparisons, but n=5 per arm; not a large-sample claim.
6. **Universe and data vintage** — equity is ~50 hand-picked current US large
   caps via yfinance (survivorship bias inflates absolute Sharpe levels, though
   not the *relative* A/B); single ~5.5y window; daily bars. No
   survivorship-bias-free vendor.
7. **Reproducibility is distributional, not bit-exact** — same seed diverges
   across devices (E8); claims hold as mean ± std over seeds on a fixed
   device/torch version, not as exact point estimates.

## Reproducibility

- **Environment.** Python 3.13 (`py` launcher); `torch 2.12.0+cu126` +
  `torch_geometric` (`[gnn]` extra); paper runs on **CPU** (E8). `PYTHONPATH=src`.
  Equity needs `yfinance`; energy uses the synthetic generator (ENTSO-E optional).
- **Tests.** 55 new-module tests (`tests/test_{gat,training,edges_equity,
  edges_energy,graph_propagate,factor_provider,run_gat_equity,run_gat_energy,
  leakage,attention}.py`); leakage controls pinned to CPU for determinism.
- **Run scripts** (in `.scratch/`, archived outputs in `docs/results/`):
  `run_real.py` (E5), `run_matrix.py` (E6), `run_seeds.py` (E7),
  `bench_gpu.py` (E8), `run_hp_grid.py`+`run_winner.py` (E9/E9b),
  `run_attention.py` (E10), `run_energy.py` (E11), `run_energy_real.py` (E12,
  live ENTSO-E; `ENTSOE_API_KEY` env + `configs/energy_universe_gnn.yaml`),
  `run_energy_daily.py` (E13, daily label redesign), `run_energy_honest.py`
  (E13b, return-definition diagnostic + honest-return verdict).
- **Artifacts** (`docs/results/`, date-prefixed): per-run diagnostics CSVs,
  matrix/seed/HP/winner/energy summaries (CSV+JSON), attention time-series CSVs
  + five figures under `figures/`. Representative model weights are gitignored
  (transient); the canonical `data/warehouse/gat_equity.pt` is tracked.
- **CLI.** `quant-alpha gat-equity --graph {static,dynamic} --retrain
  {single,walk-forward} --loss {ic,mse} --device {cpu,cuda,auto} --persist`;
  energy via `run_gat_energy` (synthetic or `source="entsoe"`).
- **Warehouse / dbt.** `--persist` (or `persist_gat_outputs`) writes four tables
  to DuckDB; the dbt models `stg_gat_panel`, `fct_gat_vs_baseline` (tiered
  relational A/B), and `fct_gat_scorecard` (one-row gates + attention A/B)
  surface the results for querying/BI. duckdb + dbt live in a venv at
  `D:\duckdb` (kept off the C: drive); the GAT pipeline itself needs no duckdb.
  Run: `dbt run/test --profiles-dir . --select fct_gat_vs_baseline
  fct_gat_scorecard` from `dbt_quant_alpha/`.

---

## E1 — Synthetic negative control (2026-06, MSE loss)

**Setup.** 50 synthetic random-walk names (`generate_synthetic_prices`),
static correlation top-k graph, GAT (2 layers, hidden 64, 4 heads), MSE loss,
50 epochs, `quant-alpha gat-equity --offline`.

**Result.** The four gates correctly report **no edge**: Value-added and
Consistency FAIL (OOS IC ~ -0.06, negative Sharpe value-add); Uniqueness and
Robustness PASS (structural properties, not signal claims).

**Reading.** On data with no learnable cross-sectional structure, a
leakage-safe pipeline must find nothing — and it does. This is the first half
of the no-false-positives argument (see E3 for the second half).

---

## E2 — Split-hygiene fixes (2026-06-10, pre-real-data)

Not an experiment but a precondition for trusting any later numbers; recorded
here because the paper must describe the evaluation protocol.

1. **One split date** (was: three independent 0.7 ratios that happened to
   coincide). A single IS/OOS boundary now drives graph construction
   (`as_of`), model selection (`fit(train_idx, valid_idx)`), and the
   four-gate evaluation (`evaluate_alpha_suite(split_date=...)`).
2. **Valid inside IS, embargoed on both sides**:
   `train | embargo(k) | valid | embargo(k) | OOS`
   (`graph.training.is_constrained_split`, asserted at both the helper and
   the call site). The trailing embargo matters: valid labels reach `t+k`,
   so without it best-epoch selection would peek at the OOS window.
3. **`fit` returns the best-by-valid-IC model** (was: saved best weights but
   returned the final epoch — checkpoint selection was dead code).
   `out_path` now always holds the returned model's weights.
4. **Seam-guaranteed deterministic inference**: `GATPropagator.propagate`
   calls `model.eval()` itself.

Full rationale: ADR-0003 amendment (2026-06-10).

---

## E3 — Leakage controls, automated (2026-06-10)

`tests/test_leakage.py`, parametrised over both losses (MSE, IC). Synthetic
panel: 60 days x 12 names, fully-connected topology, k=2, fixed seeds.

- **Negative control** — shuffle each snapshot's labels across nodes,
  retrain: valid IC must satisfy |IC| < 0.25. Passes for both losses
  (IC ~ 0). If this ever fails, future information has leaked into features
  or graph — audit `build_sections`.
- **Positive control** — plant a recoverable signal (label = standardised
  feature 0): the same fit loop must recover it (valid IC > 0.3). Passes for
  both losses. This proves the negative control passes because there is
  nothing to learn, not because the trainer is broken.

E1 + E3 together form the complete "this pipeline produces no false
positives" argument — worth a subsection in the paper before any positive
result is claimed.

---

## E4 — IC-loss convergence probe (2026-06-10)

Question: does the IC loss (`-Pearson(pred, label)` per snapshot) train as
reliably as MSE? Probe on the planted-signal panel (E3 setup), GAT hidden 8 /
heads 2 unless noted, best-epoch selection on valid IC:

| loss | lr | epochs | valid IC |
|---|---|---|---|
| mse | 1e-3 | 40 | 0.233 |
| mse | 5e-3 | 40 | 0.334 |
| mse | 5e-3 | 120 | 0.334 (plateau) |
| ic | 1e-3 | 40 | 0.409 |
| ic | 5e-3 | 40 | **0.134 (unstable early)** |
| ic | 5e-3 | 120 | 0.994 |
| ic | 1e-3 | 120, hidden 16 heads 4 | 0.997 |
| ic | 1e-3 | 120, 1 layer | 1.000 |

**Findings.** (1) IC loss ultimately recovers the planted signal far better
than MSE (0.99+ vs 0.33 plateau) — expected, since it optimises the
evaluation metric directly. (2) It is unstable in the early epochs at high
lr (5e-3): the loss bounces around 0 for ~40 epochs before converging. At
lr 1e-3 convergence is clean from the start. Default `GATConfig.lr=1e-3` is
therefore kept for IC-loss training. (3) Consequence for the pipeline:
`loss="ic"` is now the default (ADR-0003 step 2), MSE retained via
`--loss mse` for A/B.

---

## E5 — First real-data run (2026-06-10)

**Setup.**
- Data: yfinance daily bars, 2021-01-01 → 2026-06-10, universe
  `configs/universe.yaml` (50 names, GICS sectors). ORCL failed to download
  (yfinance local cache lock) → **49 names**, 1,364 trading days, 66,836
  panel rows. Pipeline is N-agnostic; no code change needed.
- Features: the 10 island alphas' `_rank` columns, cross-sectional median
  fill (ADR-0005).
- Graph: static correlation top-k backbone (window 60d, top_k 8) + sector
  boost + min_degree fallback, built strictly from data before the split
  date (ADR-0005).
- Label: forward_return(t+5) (backtest `forward_return_days=5`),
  cross-sectionally z-scored (ADR-0003).
- Training: IC loss, lr 1e-3, 50 epochs, 2 GAT layers, hidden 64, heads 4,
  dropout 0.1; split per E2 (train | 5 | valid | 5 | OOS at 70% of dates);
  best-epoch selection on valid IC.
- Command: `docs/results/2026-06-10_run_real.py` (wraps
  `run_gat_equity(offline=False, epochs=50, loss="ic")`); weights
  `data/warehouse/gat_equity.pt`; diagnostics CSV
  `docs/results/2026-06-10_gat_real_run_diagnostics.csv`.

**Training behaviour.** Textbook curve: train IC loss falls monotonically
(-0.02 → -0.134); valid IC rises from ~0.03 to a peak of **0.0756 at epoch
32**, then decays to 0.0418 by epoch 49 (overfitting). Best-epoch selection
returned epoch 32 — without the E2 checkpoint fix the deployed model would
have been the epoch-49 one, with barely half the valid IC. The fix paid for
itself on the first real run.

**Four-gate result: 3 of 4 PASS.**

| Gate | Value | Result |
|---|---|---|
| Value-added | composite OOS Sharpe 1.42 vs best single 2.88 (value-add -1.46) | **FAIL** |
| Consistency | IS/OOS IC same sign; score 0.63 | PASS |
| Uniqueness | max abs corr vs singles 0.257 | PASS |
| Robustness | score 0.68 | PASS |

Composite OOS IC mean 0.0066, OOS IC IR 0.025, OOS Sharpe 1.42.

**OOS diagnostics, all factors** (full table in
`docs/results/2026-06-10_gat_real_run_diagnostics.csv`):

| alpha | OOS IC | OOS Sharpe |
|---|---|---|
| alpha_wq_010_gap_quality | 0.0172 | **2.88** |
| alpha_wq_002_volume_price_divergence | 0.0235 | 1.15 |
| **alpha_gat_composite** | **0.0066** | **1.42** |
| alpha_wq_001_reversal_rank | 0.0045 | 0.19 |
| alpha_wq_007_price_to_ma_reversion | -0.0065 | 0.30 |
| alpha_wq_003_intraday_range_position | 0.0049 | -0.44 |
| alpha_liquidity_020_volume_shock | 0.0063 | -1.15 |
| alpha_trend_021_medium_momentum | -0.0107 | -1.42 |
| alpha_wq_009_volume_weighted_return | -0.0268 | -1.70 |
| alpha_wq_008_overnight_gap | -0.0296 | -1.91 |
| alpha_risk_020_low_volatility | -0.0628 | -4.05 |

**Honest reading.**

1. The GAT composite is a **real, unique, consistent but modest** signal:
   positive OOS Sharpe (1.42, second-best of eleven), low correlation with
   every island alpha (max 0.257), IS/OOS sign-stable. It is genuinely new
   information, not a repackaging of its inputs.
2. It does **not** beat the best single island alpha
   (`alpha_wq_010_gap_quality`, OOS Sharpe 2.88). The Value-added gate as
   defined — beat the *max* of ten singles — is the strictest possible bar;
   the composite does beat 9 of the 10.
3. **Valid IC 0.076 vs OOS IC 0.007** quantifies decay across the ~1.6-year
   OOS window. Candidate explanations, in testable order: (a) the static
   graph (correlations frozen at the split date) goes stale over OOS —
   motivates the dynamic per-snapshot graph; (b) single train-period
   weights face regime shift — motivates walk-forward retraining; (c) the
   five-day horizon's cross-sectional signal is simply weak in this period
   (the islands' OOS ICs are also small, median |IC| ~ 0.01).

**Limitations to state in the paper.**

- Static graph: early *training* snapshots see a graph built from
  correlations up to the split date (mild in-sample lookahead). OOS
  cleanliness is unaffected (graph data strictly precedes the OOS window).
  Resolved by the dynamic graph refinement.
- One run, one seed, one universe; no confidence intervals yet. A seed/
  hyperparameter sensitivity pass belongs in the robustness section.
- 49 of 50 names (ORCL download failure) — immaterial but record it.
- yfinance daily bars, no survivorship-bias-free vendor; universe is
  hand-picked current large caps, so the level (not the relative A/B) of
  all Sharpes is optimistic.

---

## E6 — 2x2 ablation: graph x retraining, with A/B anchors (2026-06-10)

**Question.** E5's valid->OOS IC decay had two candidate causes: stale graph
(frozen at the split date) or stale model (one training run). The 2x2
matrix {static, dynamic graph} x {single fit, walk-forward refit} attributes
the decay; every run also carries the two no-learning anchors
(`alpha_island_mean` = equal-weight composite of the same inputs, no
propagation; `alpha_uniform_composite` = uniform neighbour averaging of that
composite over the same topology the GAT uses).

**Setup.** One yfinance fetch (2021-01-01 → 2026-06-10, **all 50 names**
this time, 1,364 dates, 68,200 rows; panel pickled for reuse), identical
split/loss(IC)/epochs(50)/`torch.manual_seed(0)` across runs; walk-forward
refits every 63 snapshots. Script `.scratch/run_matrix.py`; per-run
diagnostics + summary JSON in `docs/results/2026-06-10_matrix_*`.

**Results** (gates: Value-added / Consistency / Uniqueness / Robustness;
`att_va` = GAT OOS Sharpe minus uniform-anchor OOS Sharpe):

| run | OOS IC | OOS Sharpe | gates V/C/U/R | att_va | runtime |
|---|---|---|---|---|---|
| static + single | -0.0051 | 1.04 | F/F/T/T | +2.09 | 179s |
| dynamic + single | -0.0348 | -1.01 | F/F/T/T | +0.18 | 207s |
| **static + walk-forward** | **+0.0200** | **1.95** | **F/T/T/T** | **+3.00** | 939s |
| dynamic + walk-forward | +0.0007 | 0.05 | F/T/T/T | +1.24 | 965s |

Anchors (identical across same-graph runs): island mean OOS IC -0.0225,
Sharpe -2.13; uniform composite OOS Sharpe -1.05 (static graph) / -1.19
(dynamic). Best single stays `alpha_wq_010_gap_quality` at OOS Sharpe 3.07.
GAT-vs-uniform Spearman is low and negative everywhere (-0.29 … -0.14).

**Findings.**

1. **Walk-forward retraining is the lever, not the dynamic graph.** The
   decay was *model* staleness: static+WF lifts OOS IC from -0.005 to +0.020
   and OOS Sharpe from 1.04 to 1.95, and Consistency flips to PASS (3/4
   gates). Hypothesis (a) from E5 is rejected as implemented: the
   per-snapshot correlation graph *hurts* in both arms — per-date correlation
   estimates are noisy, and the frozen train-period graph acts as a
   regulariser. (Refinements that might rescue dynamic graphs — longer
   windows, shrinkage, slower rebuild cadence — are future work, not
   currently planned.)
2. **The attention A/B — the capstone's core claim — is positive in every
   cell.** Both no-learning anchors are clearly negative OOS while the GAT is
   positive in 3 of 4 runs; attention adds +2.1 to +3.0 OOS Sharpe over
   uniform averaging on the *same* topology with the *same* inputs, and its
   scores are nearly uncorrelated with the uniform baseline (it is not just
   learning to smooth). "Relational learning beats naive relational
   averaging" holds regardless of which cell you read.
3. **Run-to-run variance is material — treat all point estimates as
   provisional.** static+single here reads -0.0051/1.04 vs E5's
   +0.0066/1.42 on a near-identical panel (deltas: ORCL restored, explicit
   seed 0 vs unseeded E5). Single-seed numbers cannot support paper claims;
   seed sensitivity (E7) must qualify every headline.
4. Value-added still fails in all cells against the strict max-of-singles
   bar, but static+WF narrows the gap to -1.12 (from -2.03).

**Current best config:** static graph + IC loss + walk-forward refits.

---

## E7 — Seed sensitivity, 5 seeds x {static_single, static_wf} (2026-06-10)

**Setup.** Same pickled E6 panel (only the torch seed varies), epochs 50,
IC loss, oos_chunk 63, seeds 0-4. Script `.scratch/run_seeds.py`; per-run
rows in `docs/results/2026-06-10_seed_sensitivity.csv`.

| arm | OOS IC (mean +/- std) | OOS Sharpe | attention value-add |
|---|---|---|---|
| static_single | -0.0009 +/- 0.0045 | 0.73 +/- 0.25 | 1.78 +/- 0.25 |
| static_wf | 0.0055 +/- 0.0147 | 1.15 +/- 0.75 | 2.20 +/- 0.75 |

Per-seed OOS Sharpe — single: 1.04, 0.67, 0.75, 0.83, 0.36;
walk-forward: 1.95, 1.81, 0.13, 0.84, 1.04 (seed 2 is a bad draw).

**Findings — several E6 headlines must be qualified.**

1. **The core claim survives intact and is now the strongest result:
   attention value-add over the uniform anchor is positive in 10 of 10 runs**
   (range +1.18 to +3.00 Sharpe). The uniform anchor is seed-independent
   (-1.05), so this is the GAT clearing a fixed negative bar in every draw.
   Likewise OOS Sharpe itself is positive in 10/10 runs (0.13-1.95).
2. **Walk-forward's advantage is suggestive, not conclusive.** Mean Sharpe
   1.15 vs 0.73 and mean IC +0.0055 vs -0.0009 favour WF, but the
   distributions overlap (Welch t ~ 1.2 on 5 seeds) and WF's variance is 3x
   single's. E6's "walk-forward is the lever" was a seed-0 read (1.95 was
   the luckiest WF draw); the honest paper claim is "walk-forward improves
   the mean but adds variance; not significant at n=5".
3. **The composite's rank-IC is fragile** (single-arm mean ~0) **while its
   long-short Sharpe is consistently positive** — the signal lives in the
   tails (top/bottom quantile selection) more than in the full
   cross-sectional ranking. Worth a paper paragraph: IC and L/S Sharpe
   measure different things and disagree here.
4. Paper protocol fixed by this entry: every headline number from now on is
   reported as mean +/- std over >= 5 seeds; single-run numbers (E5, E6) are
   retained as records but cited only with this caveat.

---

## E8 — Hardware note: GPU is a net loss at this graph size (2026-06-10)

Swapped `torch 2.12.0+cpu` -> `+cu126` (RTX 4060 Laptop 8GB, driver CUDA
12.6); pre-moved all dataset tensors to device once (`_dataset_to_device`)
so per-snapshot host->device transfers are not the bottleneck. Benchmark on
the E6 panel, seed 0, identical configs (`.scratch/bench_gpu.py`):

| arm | CPU (E7) | GPU | delta |
|---|---|---|---|
| static_single | 180s | 223s | +24% slower |
| static_wf | 955s | 1260s | +32% slower |

**Reading.** A 50-node, 10-feature snapshot graph is latency-bound: each
training step is dozens of microsecond-scale kernels, so GPU launch overhead
dominates and the 4060's throughput never engages. Decisions: (1) production
runs stay on CPU at the current universe size; (2) throughput for grids
comes from **process parallelism** (32 logical cores; ~180s per single run);
(3) the GPU becomes worthwhile only with a batched implementation (PyG
`Batch` packing ~950 snapshots/epoch into a few disconnected mega-graphs) or
a much larger universe — both future work.

### CPU vs GPU result divergence — magnitude, mechanism, impact

Same seed (0), same code, same data, different device:

| arm | CPU (OOS IC / Sharpe) | GPU (OOS IC / Sharpe) | CPU 5-seed Sharpe range (E7) |
|---|---|---|---|
| static_single | -0.0051 / 1.04 | +0.0056 / 1.35 | 0.36 - 1.04 |
| static_wf | +0.0200 / 1.95 | -0.0096 / 0.15 | 0.13 - 1.95 |

**Magnitude:** a same-seed device swap can swing the result by the *full
width of the seed distribution* (the wf row goes from the best CPU draw to
near the worst). A single run's number is device-dependent.

**Mechanism:** `torch.manual_seed` fixes the initial weights (init happens
on CPU), but training diverges immediately afterwards: (1) dropout masks
are drawn from each device's own RNG stream, so the per-step gradients
differ from step one; (2) GPU parallel reductions round in a different
order than CPU serial sums, and 1e-7-scale differences are amplified by 50
epochs of training dynamics. Cross-device same-seed is therefore another
draw from the same distribution, not the same experiment on new hardware.

**Impact on conclusions: none, given three disciplines already in place.**
(1) All claims are mean +/- std over >= 5 seeds — device variation behaves
like one more seed, and both GPU numbers fall inside or at the edge of the
CPU seed distribution (no evidence of systematic bias). The
attention-value-add result is device-insensitive (20/20 when this was
written; 30/30 after E9b). (2) Within-experiment
device consistency — **with one slip, caught post-hoc**: E5-E7 and the E9
grid ran on CPU, but the E9 winner validation silently ran on GPU (`fit`
auto-selects CUDA, and the CUDA wheel had just been installed). Fixed in
code the same day: `gat_equity_from_panel` now takes `device="cpu"` as an
explicit default (`"cuda"`/`"auto"` opt-in), so the documented
CPU-by-default decision is enforced rather than assumed; the E9 entry
carries the corresponding correction and a same-device rerun. (3)
Reproducibility statement for the paper: what is reproducible is the
*distribution* (given seeds + device class + torch version), not bit-exact
cross-device point estimates — the latter is unattainable on CUDA in
principle. Paper runs stay on CPU; the device is recorded per experiment.

---

## E9 — HP grid by valid IC, winner validated OOS (2026-06-10)

**Protocol (the hygiene point worth a paper paragraph).** HP selection used
**valid IC only** (the IS-internal, double-embargoed window; `fit` exposes it
as `model.best_valid_ic_`): 24 configs (lr {5e-4,1e-3,3e-3} x hidden {32,64}
x heads {2,4} x layers {1,2}) x 3 seeds = 72 train-only runs, static graph,
IC loss, no OOS metric computed anywhere in the grid. Only the selected
config then touched the OOS window, once, with fresh seeds. Scripts
`.scratch/run_hp_grid.py` + `run_winner.py`; artifacts
`2026-06-10_hp_grid_valid_ic.csv`, `2026-06-10_winner_validation_gpu.csv`
(the original run — see the device correction below) and
`..._winner_validation_cpu.csv` (same-device rerun).

**Grid result.** Winner: **lr=3e-3, hidden=64, heads=2, layers=2** (valid IC
0.1145 +/- 0.014). The surface is flat on top — six configs within
0.10-0.115, the default (lr 1e-3, heads 4) ranked 3rd at 0.1097 — and the
only clear structural signal is **all top-6 configs have 2 layers**: depth
matters, width/lr/heads barely do.

**Winner OOS validation** (5 seeds x both arms, vs the default config's E7
numbers in brackets):

| arm | OOS IC | OOS Sharpe | attention value-add |
|---|---|---|---|
| single | 0.0102 +/- 0.0179 [-0.0009 +/- 0.0045] | 1.03 +/- 0.84 [0.73 +/- 0.25] | 2.08 +/- 0.84 |
| walk-forward | **0.0179 +/- 0.0145** [0.0055 +/- 0.0147] | **1.30 +/- 0.73** [1.15 +/- 0.75] | 2.35 +/- 0.73 |

**Findings.**

1. **Valid-IC selection transferred to OOS**: the winner improves mean OOS
   IC and Sharpe over the default in *both* arms (directionally consistent,
   though within ~1 std). Positive evidence that the IS-internal valid
   window is informative — the protocol works, not just in principle.
2. **Attention value-add is now positive in 20 of 20 runs** (E7's 10 + E9's
   10; range +0.69 to +3.00 Sharpe). This is the paper's most robust result.
3. **Walk-forward beats single on the mean in both paired comparisons**
   (default: 1.15 vs 0.73; winner: 1.30 vs 1.03; and on IC 0.0179 vs
   0.0102) — two independent config draws agreeing strengthens E7's
   "directionally helpful", but per-comparison significance at n=5 remains
   out of reach; keep the qualified wording.
4. **Current best known setup**: static graph + IC loss + walk-forward +
   winner HPs -> OOS IC 0.0179 +/- 0.0145, OOS Sharpe 1.30 +/- 0.73, 9-10/10
   runs positive. Value-added vs the best single (3.07) stays open.

**Correction (same day): device confound in the winner-vs-default
comparison.** The winner validation silently ran on **GPU** (`fit`
auto-selected CUDA after the E8 wheel swap; the grid had passed CPU
explicitly). Within-E9 comparisons (single vs WF, all seeds) are
same-device and unaffected, but the bracketed E7 reference numbers are CPU
— so finding 1 ("valid-IC selection transferred to OOS") compared across
devices. Per E8, device behaves like another seed draw with no systematic
bias, so the directional read likely stands, but it is deconfounded by a
same-config CPU rerun (appended below). Code fixed so this cannot recur:
`gat_equity_from_panel(device="cpu")` is now explicit.

**E9b — same-device (CPU) rerun of the winner, 5 seeds
(2026-06-11, `2026-06-10_winner_validation_cpu.csv`):**

| arm | OOS IC | OOS Sharpe | attention value-add | default (E7, CPU) |
|---|---|---|---|---|
| single | -0.0004 +/- 0.0077 | 0.80 +/- 0.35 | 1.86 +/- 0.35 | IC -0.0009, Sharpe 0.73 |
| walk-forward | **0.0148 +/- 0.0043** | **1.37 +/- 0.39** | 2.42 +/- 0.39 | IC 0.0055, Sharpe 1.15 |

Deconfounded findings (these supersede finding 1 above):

1. **HP transfer is real in the walk-forward arm only.** Same-device,
   winner-vs-default: the single arm is a wash (0.80 vs 0.73 Sharpe, IC ~0
   both) — the GPU run's apparent single-arm improvement was a device/seed
   artifact. In the WF arm the winner is better on the mean AND much
   tighter: IC 0.0148 +/- 0.0043 vs 0.0055 +/- 0.0147 — a 3x variance
   reduction with 2.7x the mean.
2. **Best and most stable result to date: winner + walk-forward on CPU.**
   All 5 seeds positive on both IC (0.0109-0.0219) and Sharpe (0.98-1.98);
   the IC mean sits 3.4 std above zero.
3. **Walk-forward vs single reaches significance within the winner config**:
   Welch t ~ 3.9 on OOS IC, ~ 2.4 on OOS Sharpe (n=5 per arm). Combined
   with the two earlier paired comparisons (E7 default config, E9 GPU) both
   favouring WF, the walk-forward claim upgrades from "directionally
   helpful" to "significant in the tuned config, consistent everywhere".
4. Attention value-add: 10/10 positive again — **cumulative 30/30** across
   three run families (E7, E9-GPU, E9b-CPU).

---

## E10 — Attention qualitative analysis, M4 (2026-06-11)

**Setup.** One representative model (static graph, IC loss, winner HPs:
lr=3e-3/hidden=64/heads=2/layers=2, single fit, CPU, seed 0) on the E6
panel; head-layer attention extracted for every snapshot
(`models.gat.attention_panel` -> 919,336 edge-rows over 1,364 dates), four
readings in the torch-free `graph/attention.py`. Script
`.scratch/run_attention.py`; OOS-window numbers below; CSVs
`docs/results/2026-06-11_attention_*.csv`; figures in `docs/results/figures/`.

**Findings.**

1. **Attention is genuinely relational, not self-collapse.** Mean self-loop
   weight is **0.084** — the GAT places **91.6% of attention on neighbours**.
   Combined with C3 (it beats the uniform anchor), this rules out the trivial
   failure mode where attention degenerates to reading each node's own
   features. The relational claim is not just "better metric", it is "looks
   at the graph".

2. **But it is a gentle reweighting of broad pooling, not sharp selection.**
   Across ~12.5 neighbours/node the neighbour distribution is near-uniform:
   normalised entropy **0.956**, top-1 share **0.152** (uniform would be
   1/12.5 = 0.08), sector-homophily lift only **+0.019** (weighted same-sector
   0.330 vs structural 0.311). So the GAT applies small, smart tilts on top of
   near-mean-pooling. **This explains E6's modest-but-real margin**: the value
   is in subtle deviations from uniform, which is also why the composite is
   near-uncorrelated with — yet better than — the uniform anchor.

3. **The attention *structure* is temporally stationary** (figures
   `attention_neighbour_weight.png`, `attention_homophily_lift.png`): the
   self/neighbour split holds at ~0.91-0.92 and the homophily lift stays in
   +0.01..0.03 (almost never negative) across 2021-2026, with no visible
   break at the IS/OOS boundary. **This does not support the design doc's
   "macro-regime-adaptive" framing at the aggregate level** — adaptation lives
   in the per-name weights within a stable coarse structure, not in the coarse
   structure itself. Honest correction of the original M4 hypothesis.

4. **Information hubs are economically sensible.** Top incoming-attention
   names: AMZN, NFLX, AVGO, GS, CAT, NVDA, MSFT, LIN — mega-cap tech plus
   sector bellwethers (GS financials, CAT industrials, LIN materials), i.e.
   the names the rest of the universe co-moves with. A qualitative validity
   check the attention passes.

**Paper use.** Findings 1+2 are the mechanism behind C3 (why attention beats
uniform, and by how little); finding 3 is an honest limitation that replaces
an over-strong design hypothesis; finding 4 is the figure-friendly
"interpretable attention" story. Five figures rendered for the paper.

---

## E11 — Energy relational track, dual-track end-to-end (2026-06-11)

**Setup.** The energy GAT (ADR-0006): physical interconnector graph over 20
European bidding zones (`EUROPEAN_INTERCONNECTORS`), floored hourly label
(`energy_cross_sectional_label`, k=24h), 8 energy alphas' `_rank` columns as
node features. **Same kernel as equity** — `GATModel`, `fit`, `build_sections`
(via a `label_fn` hook), the four-gate `evaluate_alpha_suite`, and the
attention-vs-uniform `ab_report` are reused verbatim; only the graph, label,
and node set differ (`run_gat_energy.gat_energy_from_panel`). Synthetic power
data (2857 hourly snapshots, 57,140 rows; ENTSO-E needs a token), winner HPs,
static graph, IC loss, both retrain arms. Script `.scratch/run_energy.py`;
artifacts `docs/results/2026-06-11_energy_*`.

**Result — a clean negative control (the energy analogue of E1).**

| arm | OOS IC | OOS Sharpe | vs best single | attention value-add | gates V/C/U/R |
|---|---|---|---|---|---|
| static + single | -0.0093 | -1.26 | -0.85 | **-0.74** | F/F/T/T |
| static + walk-forward | -0.0113 | -1.11 | -0.70 | **-0.59** | F/F/T/T |

Anchors: uniform-mean -0.52, island-mean -0.50, best single energy alpha
-0.41 OOS Sharpe. Uniqueness passes (max corr 0.40-0.46), Robustness passes
(structural); Value-added and Consistency fail.

**Findings.**

1. **The dual-track engineering goal is met**: a genuinely heterogeneous graph
   (physical, hourly, bidding-zone) runs through the *same* GAT kernel and the
   *same* four gates + A/B with no kernel changes — the "one kernel, two
   graphs" thesis is now literal in code (C12). 54 tests pass, both tracks.

2. **On synthetic data the pipeline correctly finds no edge — and does not
   manufacture one.** OOS IC ~ 0, and crucially the **attention value-add is
   negative** (-0.74, -0.59): the GAT is *worse* than the uniform-mean anchor
   when there is no real relational signal. This is the energy counterpart of
   E1 and reinforces C1 (no false positives): even with a learned attention
   layer over a physical graph, no signal in → no signal out. A reassuring
   property, not a disappointment.

3. **The energy *value* claim is appropriately deferred to real data.**
   Synthetic zones share only diurnal/seasonal structure, not the genuine
   lead-lag transmission dynamics real coupled prices carry. The
   physical-interconnector story (the design's strongest, §2) needs real
   ENTSO-E flow data (token-gated, not available here) — same posture as
   equity before its real-data run, with the equity success as the existence
   proof that the pipeline can surface value when the data carries it.

**Paper use.** E11 is the dual-track deliverable + a second no-false-positives
control on a different graph; the honest energy-value result is "method and
infrastructure complete, real-data validation pending ENTSO-E access".

---

## E12 — Real ENTSO-E energy data: a cautionary leakage result (2026-06-11)

**The honest headline: the real-data energy metrics are too good to be true,
and that is the finding.** Token-gated ENTSO-E day-ahead data fetched live;
the pipeline runs end-to-end on real prices, but the four-gate numbers are
artifacts of day-ahead market structure, not alpha. This vindicates ADR-0004's
explicit warning that the energy label is the dangerous part.

**Setup.** Live ENTSO-E fetch (`fetch_entsoe_power_market`, token validated):
20 bidding zones, full-year 2024 hourly, 175,200 rows, **20/20 EIC codes
returned data** (`configs/energy_universe_gnn.yaml`, live-validated). Real
prices span **-427 to 1896 EUR/MWh with 3.95% negative** — vindicating the
floored label's design. 7 of 8 alphas compute (gas_spark_spread dropped: no
gas feed in ENTSO-E). Winner HPs, k=24h, both retrain arms. Artifacts
`docs/results/2026-06-11_energy_real_*`.

**Reported numbers (NOT trustworthy — see diagnosis):**

| arm | OOS IC | OOS Sharpe | vs best single | attention VA | gates |
|---|---|---|---|---|---|
| real static + single | 0.202 | 8.25 | +8.17 | +14.9 | T/T/T/T |
| real static + walk-forward | 0.206 | 8.24 | +8.16 | +14.9 | T/T/T/T |

A Sharpe of 8 and a composite **~100x its best single input** (best single
Sharpe 0.08) are not credible — the same kernel gives an honest ~1.4 on equity.
Three compounding artifacts, each measured:

1. **Overlapping label windows.** The 24h-horizon forward return sampled
   hourly has **lag-1 autocorrelation 0.91** — consecutive "returns" share 23
   of 24 hours. The backtest treats them as independent, so the Sharpe
   denominator is understated by a large factor (effective sample ~1/24 of
   nominal). Primary Sharpe inflator.
2. **The "signal" is trivial.** A naive `-spot[t]` predictor (negative current
   price level = daily mean reversion) has cross-sectional **IC 0.235 — higher
   than the GAT's 0.20**. The apparent edge is deterministic diurnal
   mean-reversion, not a learned relational factor.
3. **Wrong annualisation.** `periods_per_year=252` (daily equity) applied to
   hourly energy; energy has 8760 periods/yr.

Plus **day-ahead lookahead**: the auction publishes all 24 hours of day D at
once on D-1, so a "24h-ahead" label at hour t is partly contemporaneous
information, not future.

**Findings.**

1. **Infrastructure success, result failure (honestly).** The dual-track
   pipeline fetches and runs on real ENTSO-E data with no kernel changes — but
   a *valid* energy result requires a market-structure-aware redesign, not the
   equity evaluation ported verbatim. The equity track remains the project's
   genuine value demonstration.
2. **The leakage is self-evident from the magnitude, and the project's own
   skepticism caught it** — the same no-false-positives discipline that made
   the synthetic controls (E1, E11) a feature here flags a real-data result as
   not-credible rather than reporting Sharpe 8. This is the paper's strongest
   methodological point, not a setback.
3. **Required for a real energy value claim** (future work): non-overlapping
   horizon aligned to the day-ahead gate-close; features strictly available
   before gate-close (no contemporaneous day-ahead prices); correct hourly
   annualisation; a deflated/independent-period Sharpe (e.g. block bootstrap).

**Paper use.** E12 is a methodology contribution: a worked example of how a
GNN can manufacture an implausible result on a market whose microstructure
breaks a naive cross-sectional label — and of catching it. Pair with C1/E1/E11
(no false positives) as the credibility spine.

---

## E13 — Energy label redesign: artifacts fixed, verdict reached (2026-06-11)

The follow-up to E12: redesign the energy label to the correct day-ahead-market
frame and determine whether a credible result survives, or confirm it does not.

**Redesign (market-structure-aware).** The day-ahead auction clears all 24
hours of delivery day D at once at gate closure on D-1, so the correct decision
frequency is **daily and non-overlapping**, not hourly. New setup:
- snapshot = delivery day D (daily); features = day-D base-load-aggregated
  alphas (data <= D only); label = next-day base-load price change D->D+1
  (genuinely unknown — the D+1 auction has not cleared), z-scored
  cross-sectionally; annualise at 365 (daily, 7 days/week).
- 365 days x 20 zones; interconnector graph; winner HPs; k=1 day. Script
  `.scratch/run_energy_daily.py`; artifacts `2026-06-11_energy_daily_*`.

**Artifact checks — the mechanical bugs are fixed:**

| check | E12 hourly | E13 daily | status |
|---|---|---|---|
| label lag-1 autocorr | 0.91 | **-0.10** | overlap removed |
| annualisation | 252 (wrong) | 365 | corrected |
| shuffle-label valid IC | — | **0.07** vs real 0.19 | no leak (signal genuine) |

**Result — still non-credible, and that is the verdict:**

| predictor | OOS IC | OOS Sharpe |
|---|---|---|
| GAT composite | 0.218 | 11.08 |
| **trivial `-price[D]`** | **0.158** | **6.15** |
| uniform anchor | — | -7.48 |
| island anchor | — | -7.87 |

**Findings (the honest conclusion).**

1. **The signal is real, not leakage** — shuffle-label IC (0.07) is a third of
   the real (0.19), and the trivial `-price` relationship is genuine
   mean-reversion, not lookahead. So the redesign did not just hide the
   problem.
2. **But it is non-tradeable cross-zonal price *convergence*, not alpha.** A
   one-line `-price[D]` predictor already gets OOS Sharpe 6.15: the dominant
   effect is the law-of-one-price across interconnected zones (high-price zones
   converge down day-over-day). This is a real physical regularity enforced by
   transmission, **already priced by the market** (cross-border capacity
   auctions / FTRs), so a paper long-short on day-ahead *price returns*
   massively overstates any achievable return — Sharpe 6-11 measures the
   physics of price coupling, not a trading edge.
3. **The GAT adds incremental IC over trivial** (0.218 vs 0.158) — there is
   some relational structure beyond raw price level — but on a non-tradeable
   base, so the increment does not rescue a tradeable claim.
4. **Verdict: with day-ahead price-return targets and the available data,
   energy yields no credible *tradeable* relational alpha.** The redesign
   correctly removed the artifacts and revealed that the underlying signal is
   non-tradeable convergence. A genuinely tradeable energy target would need
   real balancing/intraday spreads or capacity/FTR-aware PnL (data we do not
   have). The energy track's contribution is the **methodology** (dual-track
   infra + the E12->E13 diagnose-and-redesign arc), not a tradeable result;
   equity remains the project's genuine value demonstration.

**Paper use.** E12+E13 are a complete worked example of GNN result-skepticism:
find an implausible result, diagnose the artifacts, redesign to remove them,
and keep digging when the number stays implausible. See the E13b correction —
the "non-tradeable convergence" reading in finding 2 above was itself premature.

### E13b — Correction: it is not "non-tradeable convergence", it LOSES money

Prompted by a direct challenge ("have you actually confirmed it?"), a deeper
check found a **third artifact** and overturned the E13 finding-2 framing.

The Sharpe is entirely an artifact of the **evaluation return's `+/-0.8`
clip**, not convergence. Same trivial `-price[D]` predictor, OOS long-short
Sharpe under different return definitions:

| evaluation return | trivial `-price` Sharpe |
|---|---|
| floored + `+/-0.8` clip (E13) | **+6.28** |
| plain relative `p[D+1]/p[D]-1` (honest) | **-1.52** |
| log | +3.75 |
| winsorised `+/-0.5` | +4.61 |

And the **GAT composite** under honest vs clipped returns:

| return | GAT OOS IC | GAT OOS Sharpe |
|---|---|---|
| floored + clip (E13) | +0.218 | **+11.00** |
| plain (honest) | +0.210 | **-1.54** |

**Mechanism.** The strategy shorts expensive zones; European power prices have
fat right tails (scarcity spikes to ~1900 EUR/MWh in this 2024 sample). The
`+/-0.8` clip on the *evaluation* return caps those short-leg losses, turning a
genuinely losing strategy into an apparent Sharpe-11 winner. The rank-IC
(~0.21) stays positive under both returns — cheap zones do tend to rise — but
**the IC does not translate to PnL** because it is blind to the tail
magnitudes that the realised return carries. Equity is unaffected: its
evaluation return is unclipped, so its ~1.4 Sharpe is honest.

**Final verdict (airtight).** The energy cross-sectional day-ahead strategy is
**not a tradeable alpha — under honest unclipped returns it loses money
(Sharpe ~ -1.5), for both the GAT and the trivial predictor.** Every positive
energy number across E12/E13 was an artifact (overlap, annualisation, then
return-clipping). The GAT adds no tradeable value over a one-line baseline.

**Fix shipped:** `run_gat_energy._floored_forward_return` now evaluates on the
**unclipped** realised return by default (keeps only the denominator floor for
division stability), so the repo no longer reports the inflated number.

**Paper use (revised).** The honest arc is stronger for being longer: implausible
result -> fix overlap/annualisation -> still implausible -> challenged ->
find the return-clip artifact -> honest verdict that the strategy loses money.
Three artifacts, each caught. The lesson — *never value-clip the evaluation
return; rank-IC can be positive while PnL is negative when tails matter* — is a
transferable methodological contribution. Equity remains the genuine value
result; energy is the cautionary-methodology result.

---

## E14 — Energy forecasting reframe + congestion-aware GAT (2026-06-24)

**Motivation.** E13b closed the energy *alpha* question (no tradeable
cross-sectional alpha). E14 reframes the energy track as **price forecasting**
and asks two answerable questions: (1) does the interconnector graph improve
forecast skill, and (2) does **congestion-aware** learned attention beat both an
unlearned-graph anchor and a plain GAT? Code: `src/quant_alpha/forecast/`
(`target`/`skill`/`baselines`/`evaluate`/`gat`), `docs/energy_forecasting.md`.

**Setup.** Real ENTSO-E day-ahead data, 20 bidding zones, 2024-01..06 (87,360
rows), enriched pull (load/wind/solar forecasts + actual load + A75 generation
mix). Target = next-period price `spot[t+k]`; metric = skill score
`1 − MSE/MSE(persistence)` (+ MAE, rank-IC), OOS. Baseline ladder: persistence,
seasonal-naive, no-graph ridge, uniform-graph ridge (neighbour-mean over the
interconnector graph), and a pure-torch dense GAT (`gat_node`, `gat_congestion`).
Congestion edge feature = current cross-border price spread `|spot_i−spot_j|`
(leak-safe). k=24h.

**Results.**

| rung | skill | MAE | rank_ic |
|---|---|---|---|
| uniform_graph_ridge | 0.355 | 21.40 | 0.584 |
| gat_congestion | 0.351 | 20.02 | 0.629 |
| gat_node | 0.343 | 20.16 | 0.619 |
| no_graph_ridge | 0.224 | 23.20 | 0.636 |
| persistence | 0.000 | 23.88 | 0.637 |

- **Graph lift** (uniform − no_graph): **+0.131** skill. Synthetic control
  (independent zones) gives ~0 → the lift is genuine cross-zonal coupling, not an
  artifact (the same C1 "no false positives" guard as the alpha track).
- **Congestion lift** (gat_congestion − gat_node), 5 seeds, is **implementation-
  dependent and NOT robust**: pure-torch dense GAT gave +0.031 skill (5/5), but
  the standard **PyG GATv2 gives −0.002 (2/5, std 0.028)**. A single-seed PyG
  cross-check had shown +0.072 — a lucky draw the multi-seed run corrected.
- **GAT vs uniform** (PyG, 5 seeds): rank_ic 0.61 vs 0.58 (**beats 5/5**); skill
  0.347 vs 0.355 (**beats only 2/5** — a wash).

**Findings (honest).**

1. **The interconnector graph carries real forecast value on real data**
   (+0.131 skill over no-graph), validated against a synthetic negative control.
   Pure numpy, implementation-independent. The reframe works where the alpha
   framing did not.
2. **Learned attention robustly beats the uniform anchor on cross-sectional
   ranking** (PyG rank_ic 0.61 vs 0.58, 5/5 seeds) but is a **wash on MSE-skill**
   (beats uniform 2/5). Uniform neighbour-averaging is near MSE-optimal — a hard
   bar; the GAT's robust edge is ranking, not RMSE.
3. **The congestion edge feature does NOT robustly add skill.** The dense GAT's
   +0.031 (5/5) did not replicate under PyG GATv2 (−0.002, 2/5). The grid-
   congestion hypothesis is *not refuted*, but the price-spread proxy is too
   weak/noisy to confirm it under a like-for-like seed test.

**Verdict.** Honest, partly-null result. Robust: graph structure helps (+0.131)
and learned attention improves cross-sectional ranking over uniform (5/5). Not
confirmed: that congestion-aware attention adds *skill* — it is implementation-
dependent and a wash under PyG. Not a value/alpha claim.

**Methodology note — why two backends.** The GAT was first hand-rolled (dense
attention, no torch_geometric); a reviewer-style "is the no-PyG result
trustworthy?" prompted a PyG `GATv2Conv` re-run sharing the same leak-safe prep
(`forecast/gat.py`, `backend="pyg"|"dense"`). The two agree on the regime and on
findings 1-2, but **disagree on the congestion lift** — which is exactly how the
cross-implementation check earned its keep: it caught a fragile, implementation-
specific effect before it became a claim. PyG is now the default backend.

**Phase 2b — ground-truth congestion (real flow/NTC).** Fetched directed
cross-border physical flows (A11) + day-ahead NTC (A61/A01) for all 38
interconnector borders (`fetch_entsoe_cross_border`), built a symmetric
congestion grid ``|flow|/capacity`` (`build_congestion_grid`) and ran it as the
edge feature. Result (PyG, 5 seeds): flow-congestion skill 0.345 vs node-only
0.349 — **flow vs node +/-0: 1/5 positive; flow vs spread-proxy: 2/5; vs uniform:
2/5**. rank_ic 0.615 (beats uniform 5/5, like the other GAT rungs). **The
congestion hypothesis is not supported under either operationalisation (spread
proxy 2/5, ground-truth flow/NTC 1/5)** — real congestion data did not rescue it.
Caveat (why "not supported", not "refuted"): NTC coverage was only 36% — the most
coupled CWE borders use flow-based market coupling (shadow prices on critical
network elements), which a ``|flow|/capacity`` ratio does not capture and which
ENTSO-E publishes very differently; and congestion's effect is threshold-like, not
the smooth additive edge term modelled. Cross-border data cached
`data/raw/cross_border_real.parquet`.

**Phase 3 — edge-level spread prediction (the relational payoff).** Targets the
cross-border price spread ``spot_a − spot_b`` directly — irreducibly relational
(undefined for one node; the FTR-priced object). Ladder (`forecast/edge.py`,
skill vs persistence, OOS, 38 borders): `edge_persistence` 0; `edge_ridge`
(ridge on both endpoints' drivers + current spread, no graph) **0.192**;
`edge_gat` (GAT node embeddings → edge MLP head) **0.248 (5 seeds)**.
**edge_gat beats edge_ridge by +0.056 skill, 5/5 seeds (~29% relative).** Since
the ridge already sees both endpoints, the gain is pure whole-network context
(message passing). Validated by the synthetic negative control: on independent
zones the edge GAT is *worse* than the ridge (−0.32), so the real gain is genuine
structure, not leakage. **This is the project's strongest relational result — the
GNN's value concentrates on the genuinely relational target**, while node-level
price skill barely needed the graph and congestion-as-edge-feature was null.

**Limitations / next.** Single 6-month window, untuned HPs, one CPU run per seed.
Robust results: node graph lift +0.131; attention's ranking edge (5/5);
**edge-level message passing +0.056 over both-endpoint ridge (5/5)**. Open: HP
grid + seed ensemble + walk-forward; Huber loss; a flow-based-coupling congestion
signal for CWE; longer/multi-window. See `docs/energy_forecasting.md`.

## Next experiments (priority order)

1. ~~Dynamic per-snapshot graph~~ — **DONE, E6**: implemented
   (`rolling_topology_for`, `graph="dynamic"`) and rejected by the ablation;
   static graph stays the default.
2. ~~Walk-forward retraining~~ — **DONE, E6**: implemented
   (`walk_forward_composite_series`, `retrain="walk_forward"`) and adopted;
   it is the main OOS improvement so far.
3. ~~Uniform-mean A/B anchor~~ — **DONE, E6**: built into every run
   (`ab_report`); attention value-add is positive in every cell.
4. ~~Seed sensitivity~~ — **DONE, E7**; ~~HP grid~~ — **DONE, E9**: winner
   lr=3e-3/hidden=64/heads=2/layers=2 validated OOS; flat surface, depth-2
   is the only structural requirement.
5. ~~Attention visualisation (M4)~~ — **DONE, E10**: the four readings +
   five figures + the honest "gentle reweighting / stationary structure"
   story. Remaining sub-item: fold a "GAT vs Baseline" view into Streamlit
   (M5 platform integration).
6. **Value-added gate variants** — the strict max-of-singles bar is the only
   open gate; report alongside it the mean-of-singles and
   marginal-contribution-to-a-multifactor-portfolio readings before
   concluding the composite adds nothing.
7. **Significance for the WF-vs-single comparison** — already significant in
   the tuned config (E9b); optional hardening with more seeds for a tighter
   interval.
