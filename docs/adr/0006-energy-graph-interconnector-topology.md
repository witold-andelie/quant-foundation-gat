# Energy relational graph: physical interconnector topology

The dual-track decision (overrides ADR-0004's equity-first deferral as of
2026-06-11): the capstone covers both tracks, and the energy track's identity
is a *physical* graph where the equity track's is *estimated*.

## Graph

The energy graph is the European cross-border interconnector network
(`graph/edges_energy.py`), not a correlation top-k:

- **Backbone (physical):** edges are the known interconnections between bidding
  zones (`EUROPEAN_INTERCONNECTORS` — borders + major HVDC cables, reference
  data, ~20 zones). Power prices propagate along real transmission lines, so the
  adjacency is grounded in the grid, not in statistics. This is the stronger
  relational story (gnn_capstone_design.md §2): the equity graph *estimates* who
  is connected; the energy graph *knows*.
- **Edge weight = trailing return correlation**, computed strictly before
  `as_of` (asserted), diagnostic only — the GAT learns its own attention
  (consistent with ADR-0005: weights are not fed to the model).
- **min_degree fallback:** a zone with no interconnector among the *present*
  market subset is wired to its highest-correlation peer, so a partial universe
  never isolates a node.
- **Static first**, with a `rolling_energy_topology_for` variant that re-weights
  the (fixed physical) edge set as of each query time. The edge *set* is static
  physics; only the diagnostic weights move.
- **Leakage:** the physical map is known a priori (no leak); the correlation
  weights use only history `< as_of`, asserted in the builder.

## Label

Equity's price-ratio label is unsafe for power (negative/near-zero spot prices
blow up `price[t+k]/price[t]-1`), so the energy label is a floored relative
change — `(price[t+k]-price[t]) / clip(|price[t]|, floor)`, clipped, then
z-scored per timestamp (`energy_cross_sectional_label`). **k is in hours**, not
days. This matches the existing `pipeline_energy` `forward_return` formula, so
training and the four-gate evaluation align. A mis-aligned energy label was
ADR-0004's stated danger; it lives in tested, torch-free pandas.

## Nodes / features

Bidding zones; node features are the 8 existing energy alphas' cross-sectional
`_rank` columns (mirroring equity's `_rank` features), median-filled. The 8
alphas reach the unified seam via `LegacyEnergyProvider` (2026-06-11).

## One kernel, two graphs

Everything else is shared verbatim with the equity track: `GATModel`, `fit`,
`build_sections` (now with a `label_fn` hook), `composite_series`,
`walk_forward_composite_series`, the four-gate `evaluate_alpha_suite`, and the
attention-vs-uniform `ab_report`/`_baseline_columns`. `run_gat_energy` differs
from `run_gat_equity` only in graph, label, and node set — which is the
capstone's "one kernel, two heterogeneous graphs" thesis made literal in code.

## Consequences

- The energy track is additive, not a rewrite — it validates the seam design
  (ADR-0001/0002): a genuinely different (physical, hourly, zone) graph drops
  into the same pipeline.
- Synthetic power data doubles as a pipeline-integrity check, like equity E1.
- **Real ENTSO-E data (E12, 2026-06-11) exposed a structural trap**: a naive
  port of the equity 24h cross-sectional label to hourly day-ahead power
  produces implausible metrics (Sharpe 8.25) from overlapping label windows
  (lag-1 autocorr 0.91), day-ahead lookahead (the whole next day's prices
  publish at once), a trivial `-spot` predictor (IC 0.235 > GAT 0.20), and
  252-vs-8760 mis-annualisation. The infrastructure runs on real data, but a
  valid energy result needs a market-structure-aware label and evaluation
  (non-overlapping horizon at gate-close, gate-close-available features,
  hourly annualisation, deflated Sharpe). This is exactly ADR-0004's warning.
- Deferred refinements: directed/asymmetric transmission edges (cost vs demand
  side), real ENTSO-E flow-based weights, hour-of-day conditioning.
