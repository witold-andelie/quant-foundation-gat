# Energy forecasting reframe (Phase 0)

## Why

The energy GAT added no tradeable value as a *cross-sectional alpha*
(`docs/gat_experiment_log.md` E11–E13b): under honest unclipped returns the
day-ahead price-return long/short loses money, and the GAT only matched a trivial
`-price` rule. The diagnosis (see the E13 verdict and the design discussion):

1. The model saw only a **static binary adjacency** ("who borders whom") — no
   flows, capacity, congestion, not even the correlation edge weight reaches
   `GATConv` (no `edge_dim`).
2. Node features were **price-derived alphas only** — none of the physical price
   drivers (residual load, renewable forecasts, fuel) reached the model.
3. The **target was wrong**: day-ahead cross-zonal price *return* is dominated by
   law-of-one-price convergence already priced by FTRs / capacity auctions — not
   a tradeable cross-sectional signal.

So the honest, answerable question is not "is there energy alpha?" but:

> **Does the physical interconnector graph improve power-price/congestion
> *forecast skill* over no-graph and unlearned-graph baselines?**

This serves both project goals: (1) demonstrate the relational GNN genuinely adds
value, and (3) produce accurate price/congestion forecasts (operational value).

## Phase 0 — the baseline ladder (this change, torch-free)

`src/quant_alpha/forecast/` evaluates a price-forecast skill ladder; metric is
**skill score = 1 − MSE/MSE(persistence)** (plus MAE/RMSE and cross-sectional
rank-IC), scored out-of-sample.

| rung | graph? | learned? | isolates |
|---|---|---|---|
| `persistence` (`price[t]`) | — | — | the skill reference (scores 0) |
| `seasonal_naive` (`price[t+k−24h]`) | — | — | the diurnal carry |
| `no_graph_ridge` | ✗ | ✓ | what a node's **own physical drivers** add |
| `uniform_graph_ridge` | ✓ | ✗ | what **neighbour info** adds (mean aggregation) |
| **GAT** (Phase 2) | ✓ | ✓ | what **learned attention** adds |

Value claim = the GAT's skill margin over `no_graph_ridge` (graph helps) **and**
over `uniform_graph_ridge` (learning the aggregation helps) — the forecasting
analogue of the alpha-track island/uniform A/B anchors (ADR-0001).

**Features (leak-safe).** Anchors at `t` (`spot_price`, `gas_price`); drivers
(`residual_load`, `load_forecast`, `wind_forecast`, `solar_forecast`) are the
**day-ahead forecasts valid for `t+k`**, aligned by `shift(-k)` — published
ex-ante, so legitimately known at `t`. On ENTSO-E use the actual published
forecast vintage, not actuals shifted.

**Negative control.** Synthetic zones are generated independently (no cross-zonal
coupling), so the graph rung must *not* beat the no-graph rung — `test_energy_forecast.py`
asserts `graph_lift < 0.05`. Real graph lift can only appear on coupled ENTSO-E data.

Run: `quant-alpha energy-forecast --source synthetic --k 24`

## Phase 1 — real ENTSO-E data

The basic drivers were **already fetched** by `ingestion/entsoe.py` and
live-validated (2026-06-11, 20/20 zones): A44 day-ahead price, A65/A01 day-ahead
load forecast, A69/A01 wind (B18/B19) + solar (B16) forecast, and derived
`residual_load`. Phase 1 therefore **enriched** the pull rather than starting it:

- **`actual_load`** (A65/A16, realised) + derived **`demand_surprise`** =
  `actual_load − load_forecast`.
- **Generation mix** (A75/A16, realised) aggregated into `gen_nuclear`,
  `gen_gas`, `gen_coal` (lignite+hard coal), `gen_hydro` (run-of-river+reservoir)
  and `gen_total`. Enabled via `fetch_entsoe_power_market(..., include_generation=True)`
  (default on through `fetch_energy_raw`); each series is non-fatal so a partial
  publisher still runs.
- **Leakage line:** load/wind/solar *forecasts* are day-ahead (ex-ante), so they
  feed the harness as `t+k` drivers (`shift(-k)`); `actual_load` and `gen_*` are
  *realised* (known only up to `t`), so they are point-in-time **anchors**, never
  shifted to `t+k`.
- **Reproducible runs:** `energy-forecast --raw-path <parquet>` caches the (slow)
  ENTSO-E pull and reuses it on re-run.

### Running on real data

```bash
export ENTSOE_API_KEY=<your-token>
# fetch (once) + cache + score; re-runs read the cache:
quant-alpha energy-forecast --source entsoe \
  --config configs/second_foundation_project.yaml \
  --raw-path data/raw/power_market_real.parquet --k 24
```

Read the result the same way as synthetic: `no_graph_ridge` skill = the value of
the physical drivers on real prices; `graph_lift_uniform_vs_nograph` = whether the
interconnector graph adds forecast skill on *coupled* real zones (the synthetic
negative control had ~0). A positive graph lift here is the first real evidence
the relational structure helps — and the bar the GAT must clear in Phase 2.

## Phase 2 — congestion-aware GAT (`forecast/gat.py`)

A GAT is added as the final rung, with a **congestion edge feature** = the current
cross-border price spread `|spot_i − spot_j|` (leak-safe, needs no new ingestion)
so attention can down-weight decoupled (congested) borders. It predicts the price
*change* (anchored to persistence). Two rungs — `gat_node` (node features only) and
`gat_congestion` (+ the edge feature) — and two backends sharing one leak-safe
prep: PyG `GATv2Conv` (default) and a pure-torch dense GAT (fallback, no `[gnn]`
extra; graphs are <=20 nodes so dense attention is exact).

**Result (real ENTSO-E, 6 months, 20 zones, OOS, k=24h), 5-seed means:**

| rung | skill (PyG) | rank_ic (PyG) |
|---|---|---|
| `uniform_graph_ridge` | 0.355 | 0.584 |
| `gat_congestion` | 0.346 | 0.615 |
| `gat_node` | 0.347 | 0.612 |
| `no_graph_ridge` | 0.224 | 0.636 |

**Congestion lift (`gat_congestion − gat_node`) is NOT robust:** dense GAT gave
+0.031 skill (5/5 seeds), but the standard **PyG GATv2 gives −0.002 (2/5 seeds,
std 0.028)** — the effect is implementation-dependent and does not survive a
cross-implementation seed test. (A single-seed PyG cross-check had shown +0.072 —
a lucky draw; multi-seed corrected it.) The price-spread congestion proxy is
therefore **not a confirmed skill win**.

**Honest verdict.** What is robust across both implementations:
1. **Graph structure helps** — `uniform` vs `no_graph` **+0.131 skill**, pure numpy,
   validated by the synthetic negative control (~0). Implementation-independent.
2. **Learned attention beats the uniform anchor on cross-sectional ranking**
   (PyG `gat`/`gat_cong` rank_ic 0.61 vs uniform 0.58, **5/5 seeds**), but is a
   **wash on MSE-skill** (beats uniform 2/5; uniform neighbour-averaging is near
   MSE-optimal and a hard bar).
3. **The congestion edge feature does not robustly add skill — under *either*
   operationalisation.** Price-spread proxy (Phase 2a): PyG 2/5 positive.
   Ground-truth flow/NTC (Phase 2b, `fetch_entsoe_cross_border` +
   `build_congestion_grid`): PyG **1/5** positive (flow 0.345 vs node 0.349).
   Real congestion data did not rescue it. *Not refuted* (NTC coverage was 36%;
   CWE borders use flow-based coupling a `|flow|/capacity` ratio can't capture),
   but as a smooth additive edge feature, congestion does not add forecast skill.

Full record + per-seed tables: `docs/gat_experiment_log.md` E14.

## Phase 3 — edge-level spread prediction (`forecast/edge.py`)

Targets the **cross-border spread** `spot_a − spot_b` directly: irreducibly
relational (undefined for a single node; the FTR-priced object). A GAT produces
node embeddings via message passing, then an **edge MLP head** predicts the spread
from `[h_a, h_b, current_spread]`. The question: does whole-network context beat a
model that already sees both endpoints?

**Result (real ENTSO-E, 6 months, 38 borders, OOS, k=24h, 5-seed means):**

| rung | skill | rank_ic |
|---|---|---|
| `edge_gat` (+ message passing) | **0.248** | 0.68 |
| `edge_ridge` (both endpoints, no graph) | 0.192 | 0.67 |
| `edge_persistence` (spread today) | 0.000 | — |

**`edge_gat` beats `edge_ridge` by +0.056 skill, 5/5 seeds (~29% relative).** The
ridge already sees both endpoints, so the gain is pure whole-network context.
Validated by the synthetic negative control (independent zones → edge GAT *worse*
than ridge, −0.32, same code). **This is the project's strongest relational
result: the GNN's value concentrates on the genuinely relational target** — where
node-level price skill barely needed the graph and congestion-as-edge-feature was
null. Caveats: 6-month window, k=24h, untuned; absolute spread skill is modest
(spreads are noisy differences). Full record: `docs/gat_experiment_log.md` E14.

## Next phases

- **Phase 2b — ground-truth congestion: DONE (null).** Real cross-border flows +
  NTC (`fetch_entsoe_cross_border`, `build_congestion_grid`) ran as the edge
  feature; it did not beat node-only (1/5) — the congestion hypothesis is not
  supported under either operationalisation. A cleaner test would need a
  **flow-based-coupling congestion signal** for the CWE region (shadow prices on
  critical network elements), a substantially harder data problem.
- **Tuning / robustness:** HP grid + seed ensemble + walk-forward (as the alpha
  track did); a Huber/MAE loss (the GAT already wins MAE) if MAE is the target.
- **Phase 3 — edge-level spread prediction: DONE (positive).** Message passing
  beats the both-endpoint ridge by +0.056 skill (5/5 seeds) — the strongest
  relational result. Open follow-ups: longer/multi-window, HP tuning + seed
  ensemble, and an edge-level *congestion* (not just spread) target.
