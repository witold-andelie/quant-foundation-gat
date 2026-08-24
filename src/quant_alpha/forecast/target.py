"""Forecasting targets and leakage-safe time splits for the energy track.

Phase 0 of the energy forecasting reframe (`docs/energy_forecasting.md`). After
E11-E13b showed no tradeable cross-sectional energy *alpha*, the honest reframe
is *price/congestion forecasting*: measure whether the interconnector graph
improves forecast skill over no-graph and unlearned-graph baselines. This module
provides the point-in-time target and the train/OOS split the baseline ladder
(`forecast.baselines`) and the skill report (`forecast.skill`) score against.

Everything here is plain pandas — no torch — so the harness runs without the
``[gnn]`` extra; the GAT rung plugs in later as one more predictor.
"""

from __future__ import annotations

import pandas as pd


def forward_price_target(panel: pd.DataFrame, k: int, price_col: str = "spot_price") -> pd.Series:
    """Next-period price ``price[t+k]`` per node, on a ``(time, entity)`` panel.

    LABEL ONLY: uses ``t+k`` prices, so the last ``k`` snapshots per entity are
    NaN (no future) and drop out of scoring. The persistence forecast for this
    target is ``price[t]`` (``forecast.baselines.persistence``), which is why the
    skill score uses persistence as its reference.
    """
    panel = panel.sort_index()
    fwd = panel.groupby(level=1)[price_col].transform(lambda s: s.shift(-k))
    return fwd.rename(f"{price_col}_fwd{k}")


def time_ordered_split(
    times, train_ratio: float, embargo: int
) -> tuple[tuple, tuple, object]:
    """Split sorted unique ``times`` into train / OOS with an ``embargo`` gap.

    Layout over the ordered timeline: ``train | embargo | OOS``. Train ends
    ``embargo`` (>= k) snapshots before OOS starts so a train row's target
    (which reaches ``t+k``) cannot fall inside the OOS window — the precondition
    for the OOS skill numbers to be honest. Returns
    ``(train_times, oos_times, split_time)`` where ``split_time`` is the first
    OOS time.
    """
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    ordered = sorted(times)
    n_is = int(len(ordered) * train_ratio)
    train_times = tuple(ordered[: max(n_is - embargo, 0)])
    oos_times = tuple(ordered[n_is:])
    split_time = ordered[n_is] if n_is < len(ordered) else ordered[-1]
    return train_times, oos_times, split_time
