from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_alpha.features import energy_alpha
from quant_alpha.features.energy_alpha import ENERGY_ALPHA_REGISTRY, add_energy_alpha_features
from quant_alpha.features.factor import (
    ExpressionFactorProvider,
    Factor,
    FactorProvider,
    GraphFactorProvider,
    LegacyEnergyProvider,
    apply_factors,
)
from quant_alpha.features.registry import make_equity_alpha_registry
from quant_alpha.graph.propagate import Topology, UniformMeanPropagator


def _equity_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    symbols = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(0)
    rows = []
    for sym in symbols:
        price = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
        for i, dt in enumerate(dates):
            close = float(price[i])
            rows.append(
                {
                    "date": dt,
                    "symbol": sym,
                    "adj_close": close,
                    "close": close,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "volume": float(1_000 + rng.integers(0, 500)),
                }
            )
    df = pd.DataFrame(rows)
    df["ret_1d"] = df.groupby("symbol")["adj_close"].transform(lambda s: s.pct_change())
    return df.set_index(["date", "symbol"])


def _energy_panel() -> pd.DataFrame:
    """Synthetic (timestamp, market) energy panel with every raw column the
    imperative `add_energy_alpha_features` consumes."""
    ts = pd.date_range("2024-01-01", periods=200, freq="h")
    markets = ["DE_LU", "FR", "CZ"]
    rng = np.random.default_rng(1)
    rows = []
    for mkt in markets:
        base = 50 + np.cumsum(rng.normal(0, 1, len(ts)))
        for i, t in enumerate(ts):
            rows.append(
                {
                    "timestamp": t,
                    "market": mkt,
                    "spot_price": float(base[i]),
                    "load_forecast": float(1000 + rng.normal(0, 50)),
                    "actual_load": float(1000 + rng.normal(0, 60)),
                    "wind_forecast": float(300 + rng.normal(0, 40)),
                    "solar_forecast": float(200 + rng.normal(0, 30)),
                    "residual_load": float(500 + rng.normal(0, 50)),
                    "imbalance_price": float(base[i] + rng.normal(0, 5)),
                    "gas_price": float(25 + rng.normal(0, 2)),
                }
            )
    return pd.DataFrame(rows).set_index(["timestamp", "market"])


def test_legacy_energy_provider_satisfies_protocol_and_metadata() -> None:
    provider = LegacyEnergyProvider()
    assert isinstance(provider, FactorProvider)
    factors = provider.factors()
    assert [f.name for f in factors] == [a.name for a in ENERGY_ALPHA_REGISTRY]
    assert all(f.expected_direction == 1 for f in factors)
    assert all(f.expression for f in factors)  # carries the energy expression


def test_legacy_energy_provider_is_faithful_to_imperative() -> None:
    # The provider path must reproduce add_energy_alpha_features exactly — the
    # ADR-0002 "wrap now, migrate later" shim introduces no distortion.
    panel = _energy_panel()
    result = apply_factors(panel, [LegacyEnergyProvider()])
    reference = add_energy_alpha_features(panel.reset_index()).set_index(["timestamp", "market"])

    for alpha in ENERGY_ALPHA_REGISTRY:
        pd.testing.assert_series_equal(
            result[alpha.name].sort_index(),
            reference[alpha.name].reindex(panel.index).sort_index(),
            check_names=False,
        )


def test_legacy_energy_provider_memoises_one_enrichment(monkeypatch) -> None:
    panel = _energy_panel()
    calls = {"n": 0}
    original = energy_alpha.add_energy_alpha_features

    def counting(frame):
        calls["n"] += 1
        return original(frame)

    monkeypatch.setattr(energy_alpha, "add_energy_alpha_features", counting)
    apply_factors(panel, [LegacyEnergyProvider()])
    assert calls["n"] == 1  # eight factors, one enrichment


def test_factor_rejects_bad_direction() -> None:
    with pytest.raises(ValueError):
        Factor("x", "fam", "hyp", expected_direction=0, compute=lambda p: p["adj_close"])


def test_providers_satisfy_protocol() -> None:
    expr = ExpressionFactorProvider(tuple(make_equity_alpha_registry()))
    graph = GraphFactorProvider(
        "g", "fam", "hyp", 1, UniformMeanPropagator("x"), lambda t: Topology((), ()), ("x",)
    )
    assert isinstance(expr, FactorProvider)
    assert isinstance(graph, FactorProvider)


def test_expression_provider_is_faithful_to_registry() -> None:
    # The provider path must reproduce each AlphaDefinition.compute exactly —
    # wrapping introduces no distortion (deepening #1/#2 characterisation).
    panel = _equity_panel()
    registry = make_equity_alpha_registry()

    result = apply_factors(panel, [ExpressionFactorProvider(tuple(registry))])

    for definition in registry:
        expected = definition.compute(panel)
        pd.testing.assert_series_equal(
            result[definition.name], expected, check_names=False
        )


def test_graph_provider_reproduces_cross_sectional_mean() -> None:
    # GraphFactorProvider + UniformMeanPropagator over a fully-connected
    # topology = the per-snapshot cross-sectional mean. Ties the propagate seam
    # (#3) to the provider seam (#2).
    panel = _equity_panel()[["adj_close"]].rename(columns={"adj_close": "x"})
    entities = panel.index.get_level_values(1).unique().tolist()
    fully_connected = Topology(
        nodes=tuple(entities),
        edges=tuple((s, d, 1.0) for s in entities for d in entities if s != d),
    )

    provider = GraphFactorProvider(
        name="alpha_graph_mean",
        family="relational",
        hypothesis="neighbour mean",
        expected_direction=1,
        propagator=UniformMeanPropagator("x", include_self=True),
        topology_for=lambda _t: fully_connected,
        feature_cols=("x",),
    )

    result = apply_factors(panel, [provider])
    expected = panel.groupby(level=0)["x"].transform("mean")
    pd.testing.assert_series_equal(
        result["alpha_graph_mean"], expected, check_names=False
    )
