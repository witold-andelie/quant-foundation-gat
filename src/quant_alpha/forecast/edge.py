"""Phase 3: edge-level (cross-border spread) prediction.

The node-level work (E14) found the graph helps price *level* forecasts modestly
and attention helps ranking, but congestion-as-edge-feature did not add skill.
Phase 3 targets the one quantity that is **irreducibly relational**: the
cross-border price spread ``spot_a - spot_b``. It is undefined for a single node
and is the tradeable object in power markets (FTRs price the spread).

Question: does graph **message passing** (network context beyond the two
endpoints) improve spread forecasts over a model that already sees both
endpoints' drivers? The ladder (skill = ``1 - MSE/MSE(persistence)``, OOS):

  edge_persistence  spread_ab(t)                                  -> reference
  edge_ridge        ridge([drivers_a, drivers_b, spread_ab(t)])    both endpoints, no graph
  edge_gat          GAT node embeddings -> edge MLP head           + network context

Leak-safe: the target uses ``t+k`` prices; endpoint drivers are day-ahead
forecasts; the current spread is known at ``t``. torch is imported lazily (the
ridge ladder and target run without the ``[gnn]`` extra).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_alpha.forecast.baselines import StandardizedRidge
from quant_alpha.forecast.evaluate import _feature_frame, _prepare_panel
from quant_alpha.forecast.gat import _grids
from quant_alpha.forecast.skill import skill_report
from quant_alpha.forecast.target import time_ordered_split
from quant_alpha.graph.edges_energy import EUROPEAN_INTERCONNECTORS


def border_list(nodes) -> list[tuple[str, str]]:
    """Undirected interconnector borders (sorted, unique) among the present zones."""
    present = set(nodes)
    out, seen = [], set()
    for pair in EUROPEAN_INTERCONNECTORS:
        a, b = sorted(pair)
        if a in present and b in present and (a, b) not in seen:
            seen.add((a, b))
            out.append((a, b))
    return out


def _spread_arrays(spot, ai, bi, k):
    """Current and forward (t+k) spreads on each border, from the [T,N] spot grid."""
    fwd = np.full_like(spot, np.nan)
    fwd[:-k] = spot[k:]
    cur_spread = spot[:, ai] - spot[:, bi]          # [T, E]
    fwd_spread = fwd[:, ai] - fwd[:, bi]            # [T, E] (target)
    present = (~np.isnan(spot[:, ai])) & (~np.isnan(spot[:, bi]))
    valid = present & ~np.isnan(fwd_spread)
    return cur_spread, fwd_spread, present, valid


def _adjacency(nodes):
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    A = np.eye(N, dtype="float32")
    for pair in EUROPEAN_INTERCONNECTORS:
        a, b = sorted(pair)
        if a in idx and b in idx:
            A[idx[a], idx[b]] = 1.0
            A[idx[b], idx[a]] = 1.0
    return A


def _run_edge_ridge(X, cur_spread, fwd_spread, valid, fit_rows, ai, bi, alpha=10.0):
    """Pooled ridge on [drivers_a, drivers_b, current spread] -> forward spread.

    Sees both endpoints' features but no message passing — the bar the GAT edge
    head must beat to show network context matters."""
    T, E = cur_spread.shape
    feat = np.concatenate([X[:, ai, :], X[:, bi, :], cur_spread[..., None]], axis=-1)  # [T,E,2F+1]
    D = feat.shape[-1]
    flat = feat.reshape(T * E, D)
    y = fwd_spread.reshape(T * E)
    fit_mask = np.zeros((T, E), bool); fit_mask[fit_rows] = valid[fit_rows]
    fm = fit_mask.reshape(T * E)
    model = StandardizedRidge(alpha=alpha).fit(flat[fm], y[fm])
    return model.predict(flat).reshape(T, E)


def _run_edge_gat(X, cur_spread, fwd_spread, valid, present, A, fit_rows, val_rows, ai, bi, hp):
    """GAT node embeddings (dense attention) -> edge MLP head over [h_a, h_b,
    current spread]. The relational rung: the embeddings carry whole-graph context,
    so beating `edge_ridge` means the network beyond the two endpoints helps."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(hp["seed"])
    T, N, Fd = X.shape
    E = cur_spread.shape[1]

    # standardise node features and the target spread on fit rows only
    fmask = np.zeros((T, N), bool); fmask[fit_rows] = ~np.isnan(X[fit_rows, :, 0])
    xmu = np.nanmean(np.where(fmask[..., None], X, np.nan), (0, 1))
    xsd = np.nanstd(np.where(fmask[..., None], X, np.nan), (0, 1)); xsd[~np.isfinite(xsd) | (xsd == 0)] = 1.0
    Xs = np.nan_to_num((X - xmu) / xsd)
    em = np.zeros((T, E), bool); em[fit_rows] = valid[fit_rows]
    ymu = float(np.nanmean(np.where(em, fwd_spread, np.nan))); ysd = float(np.nanstd(np.where(em, fwd_spread, np.nan))) or 1.0
    Ys = np.nan_to_num((fwd_spread - ymu) / ysd)
    cmu = float(np.nanmean(np.where(em, cur_spread, np.nan))); csd = float(np.nanstd(np.where(em, cur_spread, np.nan))) or 1.0
    Cs = np.nan_to_num((cur_spread - cmu) / csd)

    Xt = torch.tensor(Xs, dtype=torch.float32)
    Yt = torch.tensor(Ys, dtype=torch.float32)
    Ct = torch.tensor(Cs, dtype=torch.float32)
    Vt = torch.tensor(valid.astype("float32"))
    Pt = torch.tensor(present if present.shape == (T, N) else (~np.isnan(X[:, :, 0])), dtype=torch.bool)
    At = torch.tensor(A)
    ai_t, bi_t = torch.tensor(ai), torch.tensor(bi)
    H, heads, dp = hp["hidden"], hp["heads"], hp["dropout"]

    class GATLayer(torch.nn.Module):
        def __init__(self, in_dim, out_dim, n_heads):
            super().__init__()
            self.h, self.o = n_heads, out_dim
            self.W = torch.nn.Linear(in_dim, n_heads * out_dim, bias=False)
            self.aL = torch.nn.Parameter(torch.empty(n_heads, out_dim))
            self.aR = torch.nn.Parameter(torch.empty(n_heads, out_dim))
            torch.nn.init.xavier_uniform_(self.aL); torch.nn.init.xavier_uniform_(self.aR)

        def forward(self, x, node_present):
            B = x.shape[0]
            h = self.W(x).view(B, N, self.h, self.o)
            e = (h * self.aL).sum(-1)[:, :, None, :] + (h * self.aR).sum(-1)[:, None, :, :]
            e = F.leaky_relu(e, 0.2)
            mask = (At > 0)[None, :, :, None] & node_present[:, None, :, None]
            e = e.masked_fill(~mask, torch.finfo(e.dtype).min)
            alpha = F.dropout(torch.softmax(e, dim=2), p=dp, training=self.training)
            return torch.einsum("bijh,bjho->biho", alpha, h).reshape(B, N, self.h * self.o)

    class EdgeGAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = GATLayer(Fd, H, heads)
            self.l2 = GATLayer(H * heads, H, heads)
            D = H * heads
            self.head = torch.nn.Sequential(
                torch.nn.Linear(2 * D + 1, H * heads), torch.nn.ELU(),
                torch.nn.Dropout(dp), torch.nn.Linear(H * heads, 1),
            )

        def forward(self, x, node_present, cspread):
            z = F.elu(self.l1(x, node_present))
            z = F.elu(self.l2(z, node_present))           # [B,N,D]
            ha, hb = z[:, ai_t, :], z[:, bi_t, :]          # [B,E,D]
            inp = torch.cat([ha, hb, cspread[:, :, None]], dim=-1)
            return self.head(inp).squeeze(-1)              # [B,E]

    model = EdgeGAT()
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    fit_t, val_t = torch.tensor(fit_rows), torch.tensor(val_rows)

    def mse(rows):
        pred = model(Xt[rows], Pt[rows], Ct[rows])
        m = Vt[rows]
        return (((pred - Yt[rows]) * m) ** 2).sum() / m.sum().clamp(min=1.0)

    best, bs = float("inf"), None
    for _ in range(hp["epochs"]):
        model.train(); opt.zero_grad(); mse(fit_t).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = mse(val_t).item()
        if v < best:
            best, bs = v, {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
    if bs is not None:
        model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        pred = model(Xt, Pt, Ct).numpy() * ysd + ymu
    return pred


def evaluate_edge_forecast(raw, *, k=24, train_ratio=0.7, ridge_alpha=10.0,
                           include_gat=True, gat_kwargs=None) -> dict:
    """Edge-level spread-forecast skill ladder (persistence / both-endpoint ridge /
    GAT edge head). Returns the skill report + the relational lift."""
    indexed = _prepare_panel(raw)
    features = _feature_frame(indexed, k)
    cols = tuple(features.columns)
    nodes, times, X, spot, _ = _grids(indexed, features, cols, k)
    borders = border_list(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    ai = np.array([idx[a] for a, b in borders]); bi = np.array([idx[b] for a, b in borders])

    cur_spread, fwd_spread, present_e, valid = _spread_arrays(spot, ai, bi, k)
    node_present = ~np.isnan(spot)
    tr, oos, _ = time_ordered_split(times, train_ratio, embargo=k)
    tpos = {t: i for i, t in enumerate(times)}
    fit_rows = np.array([tpos[t] for t in tr], dtype=int)
    n_valid = max(int(len(fit_rows) * 0.15), 1)
    val_rows = fit_rows[len(fit_rows) - n_valid :]
    fit_rows = fit_rows[: max(len(fit_rows) - n_valid - k, 1)]
    A = _adjacency(nodes)

    bids = [f"{a}>{b}" for a, b in borders]
    full = pd.MultiIndex.from_product([times, bids], names=["timestamp", "border"])

    def as_series(grid):
        return pd.Series(grid.reshape(-1), index=full)

    preds = {
        "edge_persistence": as_series(cur_spread),
        "edge_ridge": as_series(_run_edge_ridge(X, cur_spread, fwd_spread, valid, fit_rows, ai, bi, ridge_alpha)),
    }
    if include_gat:
        hp = dict(hidden=16, heads=4, epochs=80, lr=5e-3, dropout=0.1, weight_decay=1e-4, seed=0)
        hp.update(gat_kwargs or {})
        preds["edge_gat"] = as_series(
            _run_edge_gat(X, cur_spread, fwd_spread, valid, node_present, A, fit_rows, val_rows, ai, bi, hp)
        )

    oos_set = set(oos)
    keep = full.get_level_values(0).isin(oos_set)
    target = as_series(fwd_spread)[keep]
    preds_oos = {n: s[keep] for n, s in preds.items()}
    report = skill_report(preds_oos, target, reference_name="edge_persistence")

    def _sk(name):
        return float(report.loc[report["predictor"] == name, "skill_vs_persistence"].iloc[0])

    out = {"report": report, "n_borders": len(borders), "n_oos_times": len(oos)}
    if include_gat:
        out["gat_vs_ridge"] = _sk("edge_gat") - _sk("edge_ridge")
        out["gat_vs_persistence"] = _sk("edge_gat")
    return out
