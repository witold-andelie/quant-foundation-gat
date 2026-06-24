"""Forecast skill metrics: MAE, RMSE, and skill score vs a reference.

Skill score = ``1 - MSE(model) / MSE(reference)``; the reference is the
persistence ("tomorrow = today") forecast, so persistence scores exactly 0 and
any positive score is genuine improvement over the naive carry-forward. The
cross-sectional rank-IC (reused verbatim from ``graph.training``) rides along to
connect the forecasting lens to the relational-alpha story — same metric the GAT
A/B uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_alpha.graph.training import rank_ic


def forecast_skill(pred: pd.Series, actual: pd.Series, reference: pd.Series) -> dict:
    """MAE / RMSE / skill-vs-reference / rank-IC for one predictor.

    ``pred``, ``actual`` and ``reference`` share the ``(time, entity)`` index;
    rows where any is NaN are dropped together (mainly the trailing ``k``
    snapshots with no future), so numerator and denominator are scored on the
    same rows.
    """
    frame = pd.concat({"pred": pred, "actual": actual, "ref": reference}, axis=1).dropna()
    err = frame["pred"] - frame["actual"]
    ref_err = frame["ref"] - frame["actual"]
    mse = float((err**2).mean()) if len(frame) else float("nan")
    ref_mse = float((ref_err**2).mean()) if len(frame) else float("nan")
    return {
        "n": int(len(frame)),
        "mae": float(err.abs().mean()) if len(frame) else float("nan"),
        "rmse": float(np.sqrt(mse)) if len(frame) else float("nan"),
        "mse": mse,
        "skill_vs_persistence": float(1 - mse / ref_mse) if ref_mse and ref_mse > 0 else float("nan"),
        "rank_ic": rank_ic(frame["pred"], frame["actual"]),
    }


def skill_report(
    predictions: dict[str, pd.Series], actual: pd.Series, reference_name: str = "persistence"
) -> pd.DataFrame:
    """One row per predictor, sorted by skill (best first).

    ``predictions`` maps a predictor name to its ``(time, entity)`` forecast
    series. The ``reference_name`` predictor defines the skill-score denominator
    and scores 0 by construction. The relational lift the caller cares about is
    the skill gap between the graph rungs and the no-graph rung (see
    ``forecast.evaluate``).
    """
    reference = predictions[reference_name]
    rows = [
        {"predictor": name, **forecast_skill(pred, actual, reference)}
        for name, pred in predictions.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("skill_vs_persistence", ascending=False)
        .reset_index(drop=True)
    )
