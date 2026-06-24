"""The energy forecasting baseline ladder — torch-free predictors.

Each predictor maps the panel to a ``pd.Series`` of predicted next-period price,
aligned to the panel's ``(time, entity)`` index, so ``forecast.skill`` scores
them uniformly. The ladder is designed to isolate *where* value comes from:

    persistence          no graph, no learning  -> the skill-score reference
    seasonal_naive       no graph, no learning  -> the diurnal carry baseline
    no_graph_ridge       no graph, learned      -> what a node's own drivers add
    uniform_graph_ridge  graph,    unlearned    -> what neighbour info adds (mean)
    [GAT, Phase 2]       graph,    learned       -> what learned attention adds

The graph rung reuses the interconnector ``Topology`` and the
``UniformMeanPropagator`` from the propagate seam, so the GAT rung later is the
same adapter swap as the alpha-track A/B (ADR-0001). Its skill over
``no_graph_ridge`` is the graph's value with no attention — the forecasting
analogue of the uniform anchor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant_alpha.features.factor import propagate_over_panel
from quant_alpha.graph.propagate import UniformMeanPropagator


def persistence(panel: pd.DataFrame, price_col: str = "spot_price") -> pd.Series:
    """``price[t]`` — the carry-forward reference ("tomorrow = today")."""
    return panel[price_col].rename("persistence")


def seasonal_naive(
    panel: pd.DataFrame, k: int, season: int = 24, price_col: str = "spot_price"
) -> pd.Series:
    """Predict ``price[t+k]`` as the price one season before the target.

    For the target at ``t+k``, the same-clock value one season earlier is
    ``price[t+k-season]``; aligned to row ``t`` that is ``shift(season - k)``,
    which requires ``season >= k`` so the source is at or before ``t`` (known at
    prediction time). ``season=24`` is the diurnal carry for hourly data.
    """
    if season < k:
        raise ValueError(f"seasonal_naive needs season >= k (got season={season}, k={k})")
    shifted = panel.groupby(level=1)[price_col].shift(season - k)
    return shifted.rename("seasonal_naive")


class StandardizedRidge:
    """Closed-form ridge with train-fitted standardisation and median NaN fill.

    numpy-only (no sklearn dependency). Feature medians (for NaN fill), means and
    stds are fitted on the training rows only, so OOS prediction stays
    point-in-time. The intercept is not penalised. All-NaN feature columns
    collapse to a constant 0 contribution rather than blowing up — so a partial
    ENTSO-E pull with a missing driver still fits.
    """

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StandardizedRidge":
        with np.errstate(all="ignore"):
            self.median_ = np.nanmedian(X, axis=0)
        Xf = self._fill(X)
        self.mean_ = Xf.mean(axis=0)
        self.std_ = Xf.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Z = self._design(Xf)
        reg = self.alpha * np.eye(Z.shape[1])
        reg[0, 0] = 0.0  # never penalise the intercept
        self.w_ = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._design(self._fill(X)) @ self.w_

    def _fill(self, X: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(X), self.median_, X)
        return np.where(np.isnan(filled), 0.0, filled)  # all-NaN column -> 0

    def _design(self, Xf: np.ndarray) -> np.ndarray:
        Z = (Xf - self.mean_) / self.std_
        return np.column_stack([np.ones(len(Z)), Z])


def _ridge_forecast(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target: pd.Series,
    train_times,
    alpha: float,
) -> pd.Series:
    """Fit ``StandardizedRidge`` on the train rows, predict every row."""
    target = target.reindex(frame.index)
    times = frame.index.get_level_values(0)
    train_mask = pd.Index(times).isin(train_times) & target.notna().to_numpy()
    X = frame[list(feature_cols)].to_numpy(dtype="float64")
    y = target.to_numpy(dtype="float64")
    model = StandardizedRidge(alpha=alpha).fit(X[train_mask], y[train_mask])
    return pd.Series(model.predict(X), index=frame.index)


def no_graph_ridge(
    features: pd.DataFrame,
    feature_cols: Sequence[str],
    target: pd.Series,
    train_times,
    alpha: float = 10.0,
) -> pd.Series:
    """Ridge on each node's OWN physical drivers -> next-period price."""
    return _ridge_forecast(features, feature_cols, target, train_times, alpha).rename("no_graph_ridge")


def neighbour_features(
    features: pd.DataFrame, feature_cols: Sequence[str], topology_for
) -> pd.DataFrame:
    """Interconnector-neighbour mean of each feature (excludes self).

    The pure relational signal a graph adds over a node's own data: for each
    column, average it over the in-neighbours of every node, snapshot by
    snapshot, via the same ``UniformMeanPropagator`` the alpha-track uniform
    anchor uses. On independent (synthetic) zones these carry no information; on
    coupled ENTSO-E zones a neighbour's residual load / forecast is exactly the
    cross-border driver of the local price.
    """
    out = {}
    for col in feature_cols:
        out[f"{col}_nbr"] = propagate_over_panel(
            features, UniformMeanPropagator(feature=col, include_self=False), topology_for, (col,)
        )
    return pd.DataFrame(out, index=features.index)


def uniform_graph_ridge(
    features: pd.DataFrame,
    feature_cols: Sequence[str],
    target: pd.Series,
    train_times,
    topology_for,
    alpha: float = 10.0,
) -> pd.Series:
    """Ridge on own drivers + interconnector-neighbour means (unlearned graph).

    Same model class as ``no_graph_ridge`` with neighbour-aggregated features
    added; the skill gap between the two is the graph's value under naive
    (unlearned) aggregation — the rung the GAT must beat to justify attention.
    """
    nbr = neighbour_features(features, feature_cols, topology_for)
    augmented = pd.concat([features[list(feature_cols)], nbr], axis=1)
    return _ridge_forecast(
        augmented, tuple(augmented.columns), target, train_times, alpha
    ).rename("uniform_graph_ridge")
