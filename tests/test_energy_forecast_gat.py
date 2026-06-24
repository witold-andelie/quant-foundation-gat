"""Phase 2 GAT forecaster — torch-gated (skips without the [gnn] extra)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from quant_alpha.forecast.evaluate import _feature_frame, _prepare_panel
from quant_alpha.forecast.gat import gat_forecast, present_interconnector_edges
from quant_alpha.forecast.target import forward_price_target, time_ordered_split
from quant_alpha.ingestion.energy import generate_synthetic_power_market

MARKETS = ["DE_LU", "FR", "NL", "BE", "AT", "CH"]


def _setup(k: int = 6):
    raw = generate_synthetic_power_market(MARKETS, "2024-01-01", "2024-01-12", freq="h")
    indexed = _prepare_panel(raw)
    feats = _feature_frame(indexed, k)
    target = forward_price_target(indexed, k=k)
    times = indexed.index.get_level_values(0).unique()
    tr, oos, _ = time_ordered_split(times, 0.7, embargo=k)
    return indexed, feats, tuple(feats.columns), target, tr, oos, k


def test_present_edges_symmetric_within_universe():
    edges = present_interconnector_edges(MARKETS)
    s = set(MARKETS)
    assert all(a in s and b in s for a, b in edges)
    assert ("DE_LU", "FR") in edges and ("FR", "DE_LU") in edges  # both directions


def test_gat_forecast_is_oos_only_and_finite():
    indexed, feats, cols, _, tr, oos, k = _setup()
    pred = gat_forecast(indexed, feats, cols, k, tr, oos, use_congestion=False, epochs=3, hidden=8, heads=2)
    nonnan_times = set(pred.dropna().index.get_level_values(0).unique())
    assert nonnan_times <= set(oos)               # leakage line: predictions only on OOS
    assert pred.dropna().shape[0] > 0
    assert np.isfinite(pred.dropna().to_numpy()).all()


def test_gat_congestion_variant_runs():
    indexed, feats, cols, _, tr, oos, k = _setup()
    pred = gat_forecast(indexed, feats, cols, k, tr, oos, use_congestion=True, epochs=3, hidden=8, heads=2, seed=1)
    assert pred.dropna().shape[0] > 0
    assert np.isfinite(pred.dropna().to_numpy()).all()


def test_gat_is_seed_deterministic():
    indexed, feats, cols, _, tr, oos, k = _setup()
    a = gat_forecast(indexed, feats, cols, k, tr, oos, use_congestion=True, epochs=3, hidden=8, heads=2, seed=7)
    b = gat_forecast(indexed, feats, cols, k, tr, oos, use_congestion=True, epochs=3, hidden=8, heads=2, seed=7)
    pd.testing.assert_series_equal(a, b)
