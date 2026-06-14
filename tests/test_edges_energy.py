from __future__ import annotations

import numpy as np
import pandas as pd

from quant_alpha.graph.edges_energy import (
    EUROPEAN_INTERCONNECTORS,
    build_energy_topology,
    rolling_energy_topology_for,
    static_energy_topology_for,
)

MARKETS = ["DE_LU", "FR", "NL", "BE", "AT", "CH", "CZ", "PL"]


def _panel(n_hours: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(2)
    rows = []
    for mkt in MARKETS:
        ret = rng.normal(0, 0.01, n_hours)
        for i, t in enumerate(ts):
            rows.append({"timestamp": t, "market": mkt, "ret_1d": float(ret[i])})
    return pd.DataFrame(rows).set_index(["timestamp", "market"]).sort_index()


def _degree(topo) -> dict[str, int]:
    deg: dict[str, int] = {}
    for src, _dst, _w in topo.edges:
        deg[src] = deg.get(src, 0) + 1
    return deg


def test_physical_edges_are_undirected_and_match_the_map() -> None:
    panel = _panel()
    as_of = panel.index.get_level_values(0).max()
    topo = build_energy_topology(panel, as_of=as_of)

    for src, dst, w in topo.edges:
        assert (dst, src, w) in topo.edges  # undirected
    edge_pairs = {frozenset((s, d)) for s, d, _w in topo.edges}
    # the known DE_LU-FR interconnector is present; an unconnected pair is not
    assert frozenset(("DE_LU", "FR")) in edge_pairs
    assert frozenset(("PL", "FR")) not in edge_pairs  # no PL-FR border


def test_edges_are_a_subset_of_the_interconnector_map() -> None:
    panel = _panel()
    as_of = panel.index.get_level_values(0).max()
    topo = build_energy_topology(panel, as_of=as_of, min_degree=0)
    present = set(MARKETS)
    for src, dst, _w in topo.edges:
        assert frozenset((src, dst)) in EUROPEAN_INTERCONNECTORS
        assert src in present and dst in present


def test_no_future_leak_in_weights() -> None:
    panel = _panel()
    dates = sorted(panel.index.get_level_values(0).unique())
    as_of = dates[160]

    clean = build_energy_topology(panel, as_of=as_of)
    corrupted = panel.copy()
    corrupted.loc[corrupted.index.get_level_values(0) >= as_of, "ret_1d"] = 999.0
    poisoned = build_energy_topology(corrupted, as_of=as_of)

    assert clean.edges == poisoned.edges  # weights use only history < as_of


def test_min_degree_wires_isolated_zone() -> None:
    # add a market with no interconnector among the present set
    panel = _panel()
    extra = panel.xs("DE_LU", level=1, drop_level=False).rename(index={"DE_LU": "XX"}, level=1)
    panel = pd.concat([panel, extra]).sort_index()
    as_of = panel.index.get_level_values(0).max()

    topo = build_energy_topology(panel, as_of=as_of, min_degree=1)
    deg = _degree(topo)
    assert deg.get("XX", 0) >= 1  # fallback connected the isolated zone


def test_static_is_constant_rolling_rebuilds() -> None:
    panel = _panel()
    dates = sorted(panel.index.get_level_values(0).unique())
    as_of = dates[-1]
    static = static_energy_topology_for(panel, None, as_of)
    assert static(dates[0]) is static(dates[-1])

    rolling = rolling_energy_topology_for(panel, None)
    assert rolling(dates[200]) is rolling(dates[200])  # cached
    assert rolling(dates[180]) is not rolling(dates[-1])
