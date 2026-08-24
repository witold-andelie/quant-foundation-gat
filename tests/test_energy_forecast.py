"""Phase 0 energy forecasting harness — torch-free, so no importorskip."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_alpha.forecast import (
    evaluate_energy_forecast,
    forward_price_target,
    time_ordered_split,
)
from quant_alpha.forecast.skill import forecast_skill
from quant_alpha.ingestion.energy import generate_synthetic_power_market

MARKETS = ["DE_LU", "FR", "NL", "BE", "AT", "CH"]


def _raw():
    return generate_synthetic_power_market(MARKETS, "2024-01-01", "2024-01-15", freq="h")


def test_forward_price_target_is_leak_safe():
    indexed = _raw().set_index(["timestamp", "market"]).sort_index()
    k = 6
    tgt = forward_price_target(indexed, k=k, price_col="spot_price")
    for m in MARKETS:
        col = tgt.xs(m, level=1)
        spot = indexed["spot_price"].xs(m, level=1)
        assert col.iloc[-k:].isna().all()            # no future for the last k
        assert np.isclose(col.iloc[0], spot.iloc[k])  # target[t] == spot[t+k]


def test_time_ordered_split_has_embargo_gap():
    times = list(range(100))
    train, oos, split = time_ordered_split(times, train_ratio=0.7, embargo=5)
    assert set(train).isdisjoint(oos)
    assert max(train) < split == min(oos)
    assert split - max(train) > 5  # embargo gap >= k between train target and OOS


def test_forecast_skill_reference_scores_zero():
    idx = pd.MultiIndex.from_product([range(10), MARKETS], names=["timestamp", "market"])
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.normal(size=len(idx)), index=idx)
    ref = actual + rng.normal(size=len(idx))
    s = forecast_skill(ref, actual, reference=ref)  # predictor IS the reference
    assert np.isclose(s["skill_vs_persistence"], 0.0)
    assert s["mae"] >= 0 and s["n"] == len(idx)


def test_evaluate_runs_and_features_beat_persistence():
    out = evaluate_energy_forecast(_raw(), k=6, train_ratio=0.7, window=48, season=24)
    report = out["report"]
    assert set(report["predictor"]) == {
        "persistence",
        "seasonal_naive",
        "no_graph_ridge",
        "uniform_graph_ridge",
    }
    by = report.set_index("predictor")["skill_vs_persistence"]
    assert np.isclose(by["persistence"], 0.0)  # reference scores 0 by construction
    # synthetic spot is driven by the (day-ahead) residual-load forecast, so the
    # physical-feature regressor must beat naive carry-forward.
    assert by["no_graph_ridge"] > 0
    assert "graph_lift_uniform_vs_nograph" in out
    # synthetic carries actual_load + load_forecast, so demand_surprise is derived.
    assert "demand_surprise" in out["feature_cols"]


def test_real_shaped_frame_uses_generation_mix():
    # mimic a real ENTSO-E pull: the synthetic drivers + a generation-mix block.
    raw = _raw()
    rng = np.random.default_rng(7)
    raw["gen_nuclear"] = rng.uniform(5, 15, len(raw))
    raw["gen_gas"] = rng.uniform(2, 20, len(raw))
    raw["gen_total"] = raw["gen_nuclear"] + raw["gen_gas"]
    out = evaluate_energy_forecast(raw, k=6, train_ratio=0.7, window=48)
    feats = set(out["feature_cols"])
    assert {"gen_nuclear", "gen_gas", "gen_total"} <= feats   # realised anchors used
    # generation is realised (known at t) -> an anchor, never shifted to t+k.
    assert not any(c.startswith("gen_") and c.endswith(f"_h{6}") for c in feats)


def test_evaluate_handles_missing_driver():
    # ENTSO-E may not carry gas price; the harness should run on what is present.
    raw = _raw().drop(columns=["gas_price"])
    out = evaluate_energy_forecast(raw, k=6, train_ratio=0.7, window=48)
    assert "gas_price" not in out["feature_cols"]
    assert "spot_price" in out["feature_cols"]


def test_build_congestion_grid_symmetric_and_ratio():
    from quant_alpha.forecast.gat import build_congestion_grid

    times = list(pd.date_range("2024-01-01", periods=4, freq="h"))
    rows = []
    for t in times:
        rows.append({"timestamp": t, "from_zone": "A", "to_zone": "B", "flow": 800.0, "ntc": 1000.0})
        rows.append({"timestamp": t, "from_zone": "B", "to_zone": "A", "flow": 100.0, "ntc": 1000.0})
    cb = pd.DataFrame(rows)
    grid = build_congestion_grid(cb, ["A", "B", "C"], times)
    # border congestion = max(|800|/1000, |100|/1000) = 0.8, written symmetrically
    assert np.allclose(grid[:, 0, 1], 0.8) and np.allclose(grid[:, 1, 0], 0.8)
    # C has no interconnector data -> NaN (neutral after standardisation)
    assert np.isnan(grid[:, 0, 2]).all()


def test_synthetic_zones_independent_no_false_graph_value():
    # Each synthetic zone has its own RNG (no cross-zonal coupling), so the
    # interconnector graph must NOT manufacture skill: uniform-graph should not
    # materially beat no-graph. This is the negative-control guard (cf. E11/C1).
    out = evaluate_energy_forecast(_raw(), k=6, train_ratio=0.7, window=48)
    assert out["graph_lift_uniform_vs_nograph"] < 0.05
