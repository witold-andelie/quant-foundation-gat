from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_alpha.graph.attention import (
    annotate,
    attention_concentration_over_time,
    attention_matrix,
    hub_scores,
    self_attention_over_time,
    sector_homophily_over_time,
)

SECTORS = {"A1": "Tech", "A2": "Tech", "B1": "Fin"}


def _tidy() -> pd.DataFrame:
    """Two snapshots, 3 nodes, hand-chosen so every metric is checkable.
    Each dst's incoming weights (incl. self-loop) sum to 1."""
    rows = [
        # date, src, dst, weight
        ("d1", "A1", "A1", 0.5), ("d1", "A2", "A1", 0.3), ("d1", "B1", "A1", 0.2),
        ("d1", "A2", "A2", 0.6), ("d1", "A1", "A2", 0.4),
        ("d1", "B1", "B1", 0.7), ("d1", "A1", "B1", 0.1), ("d1", "A2", "B1", 0.2),
        ("d2", "A1", "A1", 0.4), ("d2", "A2", "A1", 0.4), ("d2", "B1", "A1", 0.2),
        ("d2", "A2", "A2", 0.5), ("d2", "A1", "A2", 0.5),
        ("d2", "B1", "B1", 0.8), ("d2", "A1", "B1", 0.1), ("d2", "A2", "B1", 0.1),
    ]
    return pd.DataFrame(rows, columns=["date", "src", "dst", "weight"])


def test_softmax_per_dst_sums_to_one() -> None:
    att = _tidy()
    sums = att.groupby(["date", "dst"])["weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_self_attention_over_time() -> None:
    out = self_attention_over_time(annotate(_tidy())).set_index("date")
    assert out.loc["d1", "self_weight"] == pytest.approx(0.6)        # mean(0.5,0.6,0.7)
    assert out.loc["d1", "neighbour_weight"] == pytest.approx(0.4)
    assert out.loc["d2", "self_weight"] == pytest.approx((0.4 + 0.5 + 0.8) / 3)


def test_sector_homophily_lift() -> None:
    out = sector_homophily_over_time(annotate(_tidy(), SECTORS)).set_index("date")
    # d1 neighbour edges: A2->A1(T,.3) B1->A1(F,.2) A1->A2(T,.4) A1->B1(F,.1) A2->B1(F,.2)
    assert out.loc["d1", "weighted_same_sector"] == pytest.approx(0.7 / 1.2)
    assert out.loc["d1", "structural_same_sector"] == pytest.approx(2 / 5)
    assert out.loc["d1", "lift"] == pytest.approx(0.7 / 1.2 - 2 / 5)


def test_hub_scores_rank_attended_to_nodes() -> None:
    out = hub_scores(annotate(_tidy()), SECTORS)
    assert list(out["symbol"]) == ["A1", "A2", "B1"]   # A1 most attended-to
    assert out.set_index("symbol").loc["A1", "hub_score"] == pytest.approx((0.5 + 0.6) / 2)
    assert out.set_index("symbol").loc["A1", "sector"] == "Tech"


def test_concentration_bounds() -> None:
    out = attention_concentration_over_time(annotate(_tidy()))
    assert ((out["top1"] > 0) & (out["top1"] <= 1)).all()
    assert ((out["entropy_norm"] >= 0) & (out["entropy_norm"] <= 1)).all()


def test_attention_matrix_is_square_pivot() -> None:
    mat = attention_matrix(_tidy(), "d1")
    assert set(mat.index) <= {"A1", "A2", "B1"}
    assert mat.loc["A1", "A2"] == pytest.approx(0.3)  # A1 attends to A2 with 0.3


# --- torch end-to-end: attention_panel from a trained model ---


def test_attention_panel_structure_and_softmax() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from quant_alpha.graph.propagate import Topology
    from quant_alpha.models.gat import (
        FactorGraphDataset,
        GATConfig,
        GATModel,
        attention_panel,
        build_sections,
    )

    features = ("f0", "f1", "f2")
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    syms = [f"S{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    rows = []
    for sym in syms:
        price = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
        for i, dt in enumerate(dates):
            rows.append({"date": dt, "symbol": sym, "adj_close": float(price[i]),
                         "f0": float(rng.normal()), "f1": float(rng.normal()), "f2": float(rng.normal())})
    panel = pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()
    topo = Topology(nodes=tuple(syms), edges=tuple((s, d, 1.0) for s in syms for d in syms if s != d))
    ds = FactorGraphDataset(build_sections(panel, lambda _t: topo, features, k=2))
    model = GATModel(GATConfig(in_dim=3, hidden_dim=8, heads=2, dropout=0.0)).eval()

    att = attention_panel(model, ds)
    assert list(att.columns) == ["date", "src", "dst", "weight"]
    assert set(att["src"]) <= set(syms)
    # head-layer softmax over each dst's in-neighbourhood (incl. self-loop)
    sums = att.groupby(["date", "dst"])["weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-5)
