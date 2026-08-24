"""Phase 0 of the energy forecasting reframe: build the price-forecast target,
run the torch-free baseline ladder, and report skill + the relational lift.

This is the energy analogue of ``run_gat_*``'s four-gate report, but the lens is
*forecast skill* (MAE / RMSE / skill-vs-persistence / rank-IC), not alpha Sharpe
— the honest reframe after E11-E13b found no tradeable cross-sectional energy
alpha. On synthetic data each zone is generated independently, so the graph rung
SHOULD NOT beat the no-graph rung (a clean negative control / no false graph
value); real ENTSO-E coupling is where graph lift can appear.

Feature construction (the physically correct, leak-safe setup):
  - anchor features are taken at ``t`` (``spot_price`` — the persistence anchor —
    and ``gas_price``); they are known at prediction time.
  - driver features (``residual_load``, ``load_forecast``, ``wind_forecast``,
    ``solar_forecast``) are the *day-ahead forecasts valid for ``t+k``*, aligned
    to row ``t`` by ``shift(-k)``. These are published ex-ante (day-ahead), so
    using them at ``t`` is leak-safe; on synthetic data the ``*_forecast``
    columns already are the forecast, on ENTSO-E use the actual published
    forecast vintage (NOT actuals shifted).
"""

from __future__ import annotations

import pandas as pd

from quant_alpha.forecast.baselines import (
    no_graph_ridge,
    persistence,
    seasonal_naive,
    uniform_graph_ridge,
)
from quant_alpha.forecast.skill import skill_report
from quant_alpha.forecast.target import forward_price_target, time_ordered_split
from quant_alpha.graph.edges_energy import static_energy_topology_for

# Anchors are known at t (current state / realised quantities) and used as-is;
# drivers are day-ahead *forecasts* valid for t+k, aligned by shift(-k). Keeping
# realised generation/load as anchors (not shifting them to t+k) is the leakage
# line: actual generation at t+k is unknown at prediction time.
ANCHOR_FEATURES = (
    "spot_price",
    "gas_price",
    "actual_load",
    "demand_surprise",
    "gen_nuclear",
    "gen_gas",
    "gen_coal",
    "gen_hydro",
    "gen_total",
)
DRIVER_FEATURES = ("residual_load", "load_forecast", "wind_forecast", "solar_forecast")


def _prepare_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Flat ``timestamp``/``market`` frame -> sorted ``(timestamp, market)`` panel
    with ``ret_1d`` (the correlation input the interconnector graph weights use)
    and a derived ``demand_surprise`` where actual + forecast load are present."""
    panel = raw.sort_values(["market", "timestamp"]).copy()
    panel["ret_1d"] = panel.groupby("market")["spot_price"].pct_change()
    if "demand_surprise" not in panel.columns and {"actual_load", "load_forecast"} <= set(panel.columns):
        panel["demand_surprise"] = panel["actual_load"] - panel["load_forecast"]
    return panel.set_index(["timestamp", "market"]).sort_index()


def _feature_frame(indexed: pd.DataFrame, k: int) -> pd.DataFrame:
    """Build the leak-safe predictor frame: anchors at ``t`` + day-ahead drivers
    aligned to ``t+k``. Only columns present in ``indexed`` are used, so a partial
    ENTSO-E pull (e.g. no gas price) still runs on what is available."""
    cols: dict[str, pd.Series] = {}
    for c in ANCHOR_FEATURES:
        if c in indexed.columns:
            cols[c] = indexed[c]
    for c in DRIVER_FEATURES:
        if c in indexed.columns:
            cols[f"{c}_h{k}"] = indexed.groupby(level=1)[c].shift(-k)
    return pd.DataFrame(cols, index=indexed.index)


def evaluate_energy_forecast(
    raw: pd.DataFrame,
    *,
    k: int = 24,
    train_ratio: float = 0.7,
    window: int = 168,
    season: int = 24,
    ridge_alpha: float = 10.0,
    include_gat: bool = False,
    gat_kwargs: dict | None = None,
) -> dict:
    """Run the baseline ladder on a power-market panel and return the skill report.

    ``raw`` is a flat ``timestamp``/``market`` frame (synthetic or ENTSO-E); ``k``
    is the forecast horizon in snapshots (hours). The returned ``report`` has one
    row per predictor; ``graph_lift_uniform_vs_nograph`` is the headline relational
    number (positive only if the interconnector graph improves the forecast).
    """
    indexed = _prepare_panel(raw)
    if "spot_price" not in indexed.columns:
        raise ValueError("power-market frame must carry spot_price (the persistence anchor).")
    features = _feature_frame(indexed, k=k)
    feature_cols = tuple(features.columns)

    target = forward_price_target(indexed, k=k, price_col="spot_price")
    times = indexed.index.get_level_values(0).unique()
    train_times, oos_times, split_time = time_ordered_split(times, train_ratio, embargo=k)
    topology_for = static_energy_topology_for(
        indexed, None, as_of=split_time, return_col="ret_1d", window=window
    )

    predictions = {
        "persistence": persistence(indexed),
        "seasonal_naive": seasonal_naive(indexed, k=k, season=season),
        "no_graph_ridge": no_graph_ridge(features, feature_cols, target, train_times, alpha=ridge_alpha),
        "uniform_graph_ridge": uniform_graph_ridge(
            features, feature_cols, target, train_times, topology_for, alpha=ridge_alpha
        ),
    }

    if include_gat:  # Phase 2 — learned attention (torch); two rungs to A/B
        from quant_alpha.forecast.gat import gat_forecast

        gk = dict(gat_kwargs or {})
        predictions["gat_node"] = gat_forecast(
            indexed, features, feature_cols, k, train_times, oos_times, use_congestion=False, **gk
        )
        predictions["gat_congestion"] = gat_forecast(
            indexed, features, feature_cols, k, train_times, oos_times, use_congestion=True, **gk
        )

    # Score on the OOS window only — the honest, out-of-sample skill.
    oos_mask = indexed.index.get_level_values(0).isin(oos_times)
    oos_index = indexed.index[oos_mask]
    oos_target = target.reindex(oos_index)
    oos_predictions = {name: pred.reindex(oos_index) for name, pred in predictions.items()}
    report = skill_report(oos_predictions, oos_target, reference_name="persistence")

    names = set(report["predictor"])

    def _skill(name: str) -> float:
        return float(report.loc[report["predictor"] == name, "skill_vs_persistence"].iloc[0])

    out = {
        "report": report,
        "graph_lift_uniform_vs_nograph": _skill("uniform_graph_ridge") - _skill("no_graph_ridge"),
        "split_time": split_time,
        "feature_cols": feature_cols,
        "n_train_times": len(train_times),
        "n_oos_times": len(oos_times),
        "horizon_k": k,
    }
    if {"gat_node", "gat_congestion"} <= names:
        # The Phase 2 A/B: does LEARNED attention beat the uniform anchor, and
        # does the congestion edge feature add on top?
        out["attention_lift_gat_vs_uniform"] = _skill("gat_node") - _skill("uniform_graph_ridge")
        out["congestion_lift_vs_gat_node"] = _skill("gat_congestion") - _skill("gat_node")
        out["congestion_lift_vs_uniform"] = _skill("gat_congestion") - _skill("uniform_graph_ridge")
    return out
