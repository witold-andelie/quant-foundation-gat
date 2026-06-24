"""Phase 3 edge-level (spread) forecaster. Ridge/target pieces are torch-free;
the GAT edge head is torch-gated."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_alpha.forecast.edge import _spread_arrays, border_list, evaluate_edge_forecast
from quant_alpha.ingestion.energy import generate_synthetic_power_market

MARKETS = ["DE_LU", "FR", "NL", "BE", "AT", "CH"]


def test_border_list_sorted_unique_within_universe():
    borders = border_list(MARKETS)
    assert all(a < b and a in MARKETS and b in MARKETS for a, b in borders)
    assert len(borders) == len(set(borders))
    assert ("DE_LU", "FR") in borders


def test_spread_arrays_leak_safe():
    spot = np.arange(40, dtype="float64").reshape(10, 4)  # T=10, N=4
    ai, bi = np.array([0, 1]), np.array([1, 2])
    cur, fwd, present, valid = _spread_arrays(spot, ai, bi, k=3)
    assert np.allclose(cur[0], [spot[0, 0] - spot[0, 1], spot[0, 1] - spot[0, 2]])
    assert np.allclose(fwd[0], [spot[3, 0] - spot[3, 1], spot[3, 1] - spot[3, 2]])  # uses t+k
    assert np.isnan(fwd[-3:]).all()  # last k have no future


def test_edge_ridge_beats_persistence_on_synthetic():
    raw = generate_synthetic_power_market(MARKETS, "2024-01-01", "2024-01-15", freq="h")
    out = evaluate_edge_forecast(raw, k=6, train_ratio=0.7, include_gat=False)
    by = out["report"].set_index("predictor")["skill_vs_persistence"]
    assert np.isclose(by["edge_persistence"], 0.0)
    assert by["edge_ridge"] > 0  # both-endpoint drivers predict the spread


def test_edge_gat_runs():
    pytest.importorskip("torch")
    raw = generate_synthetic_power_market(MARKETS, "2024-01-01", "2024-01-12", freq="h")
    out = evaluate_edge_forecast(raw, k=6, train_ratio=0.7, gat_kwargs={"epochs": 3, "hidden": 8, "heads": 2})
    assert "edge_gat" in set(out["report"]["predictor"])
    assert "gat_vs_ridge" in out and np.isfinite(out["gat_vs_persistence"])
