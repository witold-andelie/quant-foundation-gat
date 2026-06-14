"""Energy graph construction: physical cross-border interconnector topology.

The equity graph is *estimated* (return correlation top-k). The energy graph is
*physical*: European power prices propagate along real cross-border transmission
lines, so the adjacency is grounded in the grid, not in statistics — the
stronger relational story (gnn_capstone_design.md §2). Edges are the known
interconnections between bidding zones; edge weight is the trailing price-return
correlation (diagnostic only, per ADR-0005 — the GAT learns its own attention),
computed strictly before `as_of` so the dynamic variant stays leak-safe.

`EUROPEAN_INTERCONNECTORS` is reference data (which zones share a border or an
HVDC cable), the energy analogue of equity GICS sectors but physical and
directed-capable. A `min_degree` fallback wires any zone with no listed
interconnector among the present markets to its highest-correlation peer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from quant_alpha.graph.propagate import Topology

# Physical interconnections among major European bidding zones (undirected;
# borders + major HVDC cables). Reference data — not estimated from prices.
_BORDERS: dict[str, tuple[str, ...]] = {
    "DE_LU": ("FR", "NL", "BE", "AT", "CH", "CZ", "PL", "DK1", "DK2", "SE4", "NO2"),
    "FR": ("DE_LU", "BE", "CH", "IT_NORD", "ES"),
    "NL": ("DE_LU", "BE", "DK1", "NO2"),
    "BE": ("FR", "NL", "DE_LU"),
    "AT": ("DE_LU", "CH", "CZ", "IT_NORD", "SI", "HU"),
    "CH": ("DE_LU", "FR", "AT", "IT_NORD"),
    "CZ": ("DE_LU", "AT", "PL", "SK"),
    "PL": ("DE_LU", "CZ", "SK", "SE4"),
    "DK1": ("DE_LU", "NL", "DK2", "NO2", "SE3"),
    "DK2": ("DE_LU", "DK1", "SE4"),
    "NO2": ("DE_LU", "NL", "DK1"),
    "SE3": ("DK1", "SE4", "FI"),
    "SE4": ("DE_LU", "PL", "DK2", "SE3"),
    "IT_NORD": ("FR", "CH", "AT", "SI"),
    "ES": ("FR", "PT"),
    "PT": ("ES",),
    "SI": ("AT", "IT_NORD", "HU"),
    "SK": ("CZ", "PL", "HU"),
    "HU": ("AT", "SK", "SI"),
    "FI": ("SE3",),
}


def _symmetric_pairs(borders: Mapping[str, tuple[str, ...]]) -> frozenset[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for zone, neighbours in borders.items():
        for nb in neighbours:
            if nb != zone:
                pairs.add(frozenset((zone, nb)))
    return frozenset(pairs)


EUROPEAN_INTERCONNECTORS: frozenset[frozenset[str]] = _symmetric_pairs(_BORDERS)
EUROPEAN_BIDDING_ZONES: tuple[str, ...] = tuple(_BORDERS.keys())


def build_energy_topology(
    panel: pd.DataFrame,
    interconnectors: Iterable[frozenset[str]] | None = None,
    *,
    as_of,
    return_col: str = "ret_1d",
    window: int = 168,
    min_periods: int = 72,
    min_degree: int = 1,
    correlation_weight: bool = True,
) -> Topology:
    """Build the undirected interconnector graph over the panel's markets.

    Backbone: the physical interconnector pairs (`EUROPEAN_INTERCONNECTORS` by
    default) whose both endpoints are present. Edge weight is the trailing
    `return_col` correlation over the `window` timestamps strictly before
    `as_of` (asserted), or 1.0 when correlation is unavailable. Any market still
    below `min_degree` (no interconnector among the present subset) is wired to
    its highest-correlation peer so no node is isolated.
    """
    interconnectors = (
        EUROPEAN_INTERCONNECTORS if interconnectors is None else frozenset(interconnectors)
    )
    markets = tuple(sorted(panel.index.get_level_values(1).unique()))
    present = set(markets)

    corr = None
    if correlation_weight or min_degree > 0:
        times = panel.index.get_level_values(0)
        history = panel[times < as_of]
        recent = sorted(history.index.get_level_values(0).unique())[-window:]
        if recent:
            assert max(recent) < as_of, "correlation window must precede as_of"
            window_panel = history[history.index.get_level_values(0).isin(recent)]
            corr = window_panel[return_col].unstack(level=1).corr(min_periods=min_periods)

    def weight(a: str, b: str) -> float:
        if corr is not None and a in corr.index and b in corr.columns and not pd.isna(corr.loc[a, b]):
            return float(corr.loc[a, b])
        return 1.0

    neighbours: dict[str, set[str]] = {m: set() for m in markets}
    for pair in interconnectors:
        if pair <= present and len(pair) == 2:
            a, b = sorted(pair)
            neighbours[a].add(b)
            neighbours[b].add(a)

    for node in markets:  # min_degree fallback for zones with no present link
        if len(neighbours[node]) >= min_degree or corr is None or node not in corr.index:
            continue
        ranked = corr.loc[node].drop(labels=[node], errors="ignore").dropna().sort_values(ascending=False)
        for peer in ranked.index:
            if len(neighbours[node]) >= min_degree:
                break
            if peer in present:
                neighbours[node].add(peer)
                neighbours[peer].add(node)

    edges = tuple(
        sorted((a, b, weight(a, b)) for a in markets for b in neighbours[a])
    )
    return Topology(nodes=markets, edges=edges)


def static_energy_topology_for(
    panel: pd.DataFrame,
    interconnectors: Iterable[frozenset[str]] | None,
    as_of,
    **kwargs,
):
    """A `topology_for` callable returning one static interconnector graph built
    from data before `as_of` (the train-period end), ignoring the query time."""
    topology = build_energy_topology(panel, interconnectors, as_of=as_of, **kwargs)
    return lambda _time: topology


def rolling_energy_topology_for(
    panel: pd.DataFrame,
    interconnectors: Iterable[frozenset[str]] | None,
    **kwargs,
):
    """A `topology_for` callable that re-weights the (fixed physical) graph as of
    each query time. The edge *set* is static physics; only the diagnostic
    correlation weights move. Cached per timestamp."""
    cache: dict = {}

    def topology_for(time):
        if time not in cache:
            cache[time] = build_energy_topology(panel, interconnectors, as_of=time, **kwargs)
        return cache[time]

    return topology_for
