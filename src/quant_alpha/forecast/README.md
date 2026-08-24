# Energy Forecasting

Leakage-controlled energy price/spread forecasting — the honest reframe of the
energy track after day-ahead price *returns* showed no tradeable cross-sectional
alpha. Metric: **skill score** `1 − MSE/MSE(persistence)` (plus MAE, rank-IC),
scored out-of-sample. Torch-free baselines run without the `[gnn]` extra; the GAT
rungs are torch-gated.

## Modules

| File | Role |
|---|---|
| `target.py` | Forward price target + leakage-safe time split (embargo ≥ k) |
| `skill.py` | MAE / RMSE / skill-vs-persistence / rank-IC + report |
| `baselines.py` | Baseline ladder: persistence, seasonal-naive, no-graph ridge, uniform-graph ridge (closed-form `StandardizedRidge`, numpy-only) |
| `gat.py` | Node-level GAT forecaster — PyG `GATv2Conv` or pure-torch dense backend; optional congestion edge feature (price-spread proxy or real flow/NTC via `build_congestion_grid`) |
| `edge.py` | Edge-level (cross-border **spread**) head — GAT node embeddings → edge MLP; the irreducibly-relational target |
| `evaluate.py` | `evaluate_energy_forecast` — runs the ladder and reports skill + relational lift |

## Findings (E14, real ENTSO-E, k=24h)

- The interconnector graph improves price-level forecast skill **+0.131** over a
  no-graph model (synthetic negative control ~0).
- Learned attention beats the uniform anchor on **cross-sectional ranking (5/5
  seeds)**; congestion-as-edge-feature (proxy *and* real flow/NTC) did **not**
  robustly add skill (an honest null).
- On the edge-level **spread** target, graph message passing beats a both-endpoint
  model **+0.056 skill (5/5 seeds)** — the GNN's value concentrates where the
  target lives on the network.

Run: `quant-alpha energy-forecast --source synthetic` (or `--source entsoe`).
Full record: [docs/energy_forecasting.md](../../../docs/energy_forecasting.md),
[docs/gat_experiment_log.md](../../../docs/gat_experiment_log.md) (E14).
