"""Attention analysis — the M4 qualitative story over a trained GAT.

Pure pandas/numpy over the tidy ``(date, src, dst, weight)`` frame that
``models.gat.attention_panel`` emits, so it is testable without the ``[gnn]``
extra. Every edge ``src -> dst`` carries the head-layer attention ``dst`` placed
on ``src``; weights into each ``dst`` softmax over its in-neighbourhood (incl.
the self-loop), summing to 1.

The four readings, each answering a paper question:
 - ``self_attention_over_time``  — does the GAT actually look at neighbours, or
   collapse to self (which would mean attention learned nothing relational)?
 - ``sector_homophily_over_time`` — does it emphasise same-sector neighbours
   beyond their structural prevalence (economic structure, not noise)?
 - ``attention_concentration_over_time`` — does it focus on a few neighbours or
   spread evenly (toward / away from mean pooling)?
 - ``hub_scores`` — which names the rest of the universe consistently listens
   to (information hubs).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def annotate(att: pd.DataFrame, sectors: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Add ``is_self``, ``src_sector``, ``dst_sector``, ``same_sector`` columns."""
    out = att.copy()
    out["is_self"] = out["src"] == out["dst"]
    if sectors is not None:
        out["src_sector"] = out["src"].map(sectors)
        out["dst_sector"] = out["dst"].map(sectors)
        out["same_sector"] = out["src_sector"] == out["dst_sector"]
    return out


def self_attention_over_time(att: pd.DataFrame) -> pd.DataFrame:
    """Per date: mean (over dst nodes) of the self-loop weight, and its
    complement on neighbours. Self ~ 1 would mean attention ignored the graph."""
    att = att if "is_self" in att.columns else annotate(att)
    selfw = att[att["is_self"]].groupby("date")["weight"].mean()
    out = selfw.rename("self_weight").to_frame()
    out["neighbour_weight"] = 1.0 - out["self_weight"]
    return out.reset_index()


def sector_homophily_over_time(att: pd.DataFrame) -> pd.DataFrame:
    """Per date, over neighbour edges only: attention-weighted same-sector share
    vs the structural same-sector share, and the lift (weighted - structural).
    Positive lift = attention over-weights same-sector neighbours."""
    if "same_sector" not in att.columns:
        att = annotate(att)
    nb = att[~att["is_self"]].dropna(subset=["same_sector"])

    def per_date(g: pd.DataFrame) -> pd.Series:
        w = g["weight"].to_numpy()
        same = g["same_sector"].to_numpy(dtype=float)
        weighted = float((w * same).sum() / w.sum()) if w.sum() > 0 else np.nan
        structural = float(same.mean())
        return pd.Series(
            {"weighted_same_sector": weighted, "structural_same_sector": structural,
             "lift": weighted - structural}
        )

    return nb.groupby("date").apply(per_date, include_groups=False).reset_index()


def attention_concentration_over_time(att: pd.DataFrame) -> pd.DataFrame:
    """Per date, averaged over dst nodes: entropy (nats and normalised to
    [0,1]) and top-1 share of the neighbour-attention distribution. Low entropy
    / high top-1 = focused; high entropy = toward uniform mean pooling."""
    att = att if "is_self" in att.columns else annotate(att)
    nb = att[~att["is_self"]]

    def per_node(g: pd.DataFrame) -> pd.Series:
        w = g["weight"].to_numpy()
        total = w.sum()
        if total <= 0 or len(w) == 0:
            return pd.Series({"entropy": np.nan, "entropy_norm": np.nan, "top1": np.nan})
        p = w / total
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        norm = float(entropy / np.log(len(w))) if len(w) > 1 else 0.0
        return pd.Series({"entropy": entropy, "entropy_norm": norm, "top1": float(p.max())})

    per = nb.groupby(["date", "dst"]).apply(per_node, include_groups=False)
    return per.groupby("date").mean().reset_index()


def hub_scores(att: pd.DataFrame, sectors: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Per src symbol: mean per-snapshot incoming attention mass (summed over the
    dst nodes that attend to it, averaged over dates). High = an information hub
    the rest of the universe listens to. Neighbour edges only."""
    att = att if "is_self" in att.columns else annotate(att)
    nb = att[~att["is_self"]]
    per_date_src = nb.groupby(["date", "src"])["weight"].sum()
    hub = per_date_src.groupby("src").mean().rename("hub_score")
    out = hub.sort_values(ascending=False).to_frame().reset_index().rename(columns={"src": "symbol"})
    if sectors is not None:
        out["sector"] = out["symbol"].map(sectors)
    return out


def attention_matrix(att: pd.DataFrame, date) -> pd.DataFrame:
    """One snapshot's attention as a dst-by-src matrix (rows attend to columns),
    for a heatmap. Missing edges are 0."""
    snap = att[att["date"] == date]
    return snap.pivot_table(index="dst", columns="src", values="weight", fill_value=0.0)
