"""Phase 2: a congestion-aware GAT forecaster.

Question: does *learned* attention over the interconnector graph beat the
unlearned uniform-mean anchor (`baselines.uniform_graph_ridge`), and does a
**congestion edge feature** let attention down-weight decoupled borders and add
skill?

Congestion signal — no new ingestion. The leak-safe proxy for whether a border
is congested is the *current cross-border price spread* ``|spot_i(t)-spot_j(t)|``:
~0 when the line has spare capacity (prices converged), large when saturated. It
is known at prediction time t. (Physical flow/NTC is the Phase 2b upgrade.)

Two backends, one shared leak-safe data-prep so they are a fair A/B:
  - ``pyg``   — torch_geometric ``GATv2Conv`` (the standard, reviewer-facing
                reference; matches the alpha track's PyG usage). Default when the
                ``[gnn]`` extra is installed.
  - ``dense`` — a self-contained pure-torch dense GAT (graphs are <=20 nodes, so
                dense attention is exact, not an approximation). The fallback
                when torch_geometric is absent.

The model predicts the price *change* ``price[t+k]-price[t]`` (anchored to
persistence); the forecast is ``spot[t] + change`` — directly comparable to the
ridge rungs on the same skill metric.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_alpha.graph.edges_energy import EUROPEAN_INTERCONNECTORS


def present_interconnector_edges(nodes: Sequence[str]) -> list[tuple[str, str]]:
    """Directed interconnector edges (both directions) among the present zones."""
    present = set(nodes)
    edges: list[tuple[str, str]] = []
    for pair in EUROPEAN_INTERCONNECTORS:
        a, b = sorted(pair)
        if a in present and b in present:
            edges.extend([(a, b), (b, a)])
    return edges


def _grids(indexed, features, feature_cols, k):
    """Long panel -> dense [T, N, ...] grids over a full time x node product."""
    nodes = tuple(sorted(indexed.index.get_level_values(1).unique()))
    times = sorted(indexed.index.get_level_values(0).unique())
    full = pd.MultiIndex.from_product([times, nodes], names=indexed.index.names)
    T, N = len(times), len(nodes)
    X = features.reindex(full)[list(feature_cols)].to_numpy("float64").reshape(T, N, len(feature_cols))
    spot = indexed["spot_price"].reindex(full).to_numpy("float64").reshape(T, N)
    fwd = indexed.groupby(level=1)["spot_price"].shift(-k).reindex(full).to_numpy("float64").reshape(T, N)
    return nodes, times, X, spot, fwd - spot  # change = price[t+k] - price[t]


def build_congestion_grid(cross_border, nodes, times, *, window=168, q=0.95, clip=2.0):
    """Phase 2b ground-truth congestion edge feature: a symmetric ``[T, N, N]``
    grid of border saturation ``|flow| / capacity``.

    ``capacity`` = NTC where published, else a leak-safe trailing ``q``-quantile of
    ``|flow|`` (``window`` hours, shifted by 1 so only past flow is used).
    Symmetrised per border (max over the two directions): a saturated
    interconnector decouples prices both ways. Missing borders stay NaN (neutral
    after standardisation). Both flow and NTC are known at ``t``.
    """
    ix = {n: i for i, n in enumerate(nodes)}
    tpos = {t: i for i, t in enumerate(times)}
    grid = np.full((len(times), len(nodes), len(nodes)), np.nan, dtype="float32")
    cb = cross_border.copy()
    cb["timestamp"] = pd.to_datetime(cb["timestamp"])

    directed: dict[tuple, pd.Series] = {}
    for (a, b), g in cb.groupby(["from_zone", "to_zone"]):
        if a not in ix or b not in ix:
            continue
        g = g.sort_values("timestamp")
        flow = g["flow"].abs()
        cap = g["ntc"].where(g["ntc"] > 0)
        cap = cap.fillna(flow.shift(1).rolling(window, min_periods=24).quantile(q))
        ratio = (flow / cap).clip(0, clip)
        directed[(a, b)] = pd.Series(ratio.to_numpy(), index=g["timestamp"].to_numpy())

    seen: set = set()
    for (a, b), s_ab in directed.items():
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        s_ba = directed.get((b, a))
        s = s_ab if s_ba is None else pd.concat([s_ab, s_ba], axis=1).max(axis=1)
        for t, r in s.items():
            ti = tpos.get(pd.Timestamp(t))
            if ti is not None and np.isfinite(r):
                grid[ti, ix[a], ix[b]] = r
                grid[ti, ix[b], ix[a]] = r
    return grid


@dataclass
class _Prep:
    nodes: tuple
    times: list
    Xs: np.ndarray
    Ys: np.ndarray
    spot: np.ndarray
    present: np.ndarray
    ymask: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    spread: np.ndarray | None
    emu: float
    esd: float
    cmu: float
    csd: float
    fit_rows: np.ndarray
    val_rows: np.ndarray
    N: int
    T: int
    F: int


def _prepare_arrays(indexed, features, feature_cols, k, train_times, use_congestion, valid_frac,
                    congestion_grid=None):
    """Shared, leak-safe array prep for both backends: dense feature/target grids
    (standardised on fit rows only, NaN->0), the directed edge list, and — when
    requested — the standardised congestion edge feature (an external grid from
    `build_congestion_grid`, else the price-spread proxy)."""
    nodes, times, X, spot, change = _grids(indexed, features, feature_cols, k)
    T, N, Fd = X.shape
    ix = {n: i for i, n in enumerate(nodes)}
    edges = present_interconnector_edges(nodes)
    src = np.array([ix[a] for a, b in edges], dtype=int)
    dst = np.array([ix[b] for a, b in edges], dtype=int)

    present = ~np.isnan(spot)
    ymask = present & ~np.isnan(change)
    tpos = {t: i for i, t in enumerate(times)}
    train_rows = np.array([tpos[t] for t in train_times], dtype=int)
    n_valid = max(int(len(train_rows) * valid_frac), 1)
    fit_rows = train_rows[: max(len(train_rows) - n_valid - k, 1)]
    val_rows = train_rows[len(train_rows) - n_valid :]

    fm = np.zeros((T, N), bool); fm[fit_rows] = present[fit_rows]
    mu = np.nanmean(np.where(fm[..., None], X, np.nan), (0, 1))
    sd = np.nanstd(np.where(fm[..., None], X, np.nan), (0, 1)); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Xs = np.nan_to_num((X - mu) / sd)

    fym = np.zeros((T, N), bool); fym[fit_rows] = ymask[fit_rows]
    cmu = float(np.nanmean(np.where(fym, change, np.nan)))
    csd = float(np.nanstd(np.where(fym, change, np.nan))) or 1.0
    Ys = np.nan_to_num((change - cmu) / csd)

    spread = None; emu, esd = 0.0, 1.0
    if use_congestion:
        if congestion_grid is not None:
            raw = np.asarray(congestion_grid, dtype="float32")  # ground-truth (Phase 2b)
        else:
            sp = np.nan_to_num(spot, nan=float(np.nanmean(spot)))
            raw = np.abs(sp[:, :, None] - sp[:, None, :]).astype("float32")  # spread proxy
        fe = raw[np.ix_(fit_rows)][:, src, dst]
        emu, esd = float(np.nanmean(fe)), (float(np.nanstd(fe)) or 1.0)
        spread = np.nan_to_num((raw - emu) / esd, nan=0.0).astype("float32")

    return _Prep(nodes, times, Xs, Ys, spot, present, ymask, src, dst, spread,
                 emu, esd, cmu, csd, fit_rows, val_rows, N, T, Fd)


def _run_pyg(p: _Prep, hp: dict) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv

    torch.manual_seed(hp["seed"])
    edk = 1 if p.spread is not None else None
    Xt = torch.tensor(p.Xs, dtype=torch.float32)
    Yt = torch.tensor(p.Ys, dtype=torch.float32)
    Mt = torch.tensor(p.ymask, dtype=torch.float32)

    def batch(rows):
        rows = np.asarray(rows); B = len(rows)
        ps = p.present[rows]
        valid = ps[:, p.src] & ps[:, p.dst]            # [B, E]
        bi, ei = np.nonzero(valid)
        s = p.src[ei] + bi * p.N
        d = p.dst[ei] + bi * p.N
        edge_index = torch.tensor(np.stack([s, d]), dtype=torch.long)
        ea = None
        if edk:
            vals = p.spread[rows[bi], p.src[ei], p.dst[ei]].astype("float32")
            ea = torch.tensor(vals[:, None], dtype=torch.float32)
        x = Xt[rows].reshape(B * p.N, p.F)
        return x, edge_index, ea, Yt[rows].reshape(-1), Mt[rows].reshape(-1)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            mods, in_dim = [], p.F
            for _ in range(hp["layers"] - 1):
                mods.append(GATv2Conv(in_dim, hp["hidden"], heads=hp["heads"], edge_dim=edk,
                                      dropout=hp["dropout"], add_self_loops=True))
                in_dim = hp["hidden"] * hp["heads"]
            mods.append(GATv2Conv(in_dim, 1, heads=1, edge_dim=edk, dropout=hp["dropout"], add_self_loops=True))
            self.layers = torch.nn.ModuleList(mods)

        def forward(self, x, ei, ea):
            for i, layer in enumerate(self.layers):
                x = layer(x, ei, ea)
                if i < len(self.layers) - 1:
                    x = F.elu(x)
            return x.squeeze(-1)

    fit_b, val_b, all_b = batch(p.fit_rows), batch(p.val_rows), batch(np.arange(p.T))
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    def mse(b):
        x, ei, ea, y, m = b
        pr = model(x, ei, ea)
        return ((pr - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)

    best, bs = float("inf"), None
    for _ in range(hp["epochs"]):
        model.train(); opt.zero_grad(); mse(fit_b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = mse(val_b).item()
        if v < best:
            best, bs = v, {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
    if bs is not None:
        model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        change = model(*all_b[:3]).numpy().reshape(p.T, p.N) * p.csd + p.cmu
    return p.spot + change


def _run_dense(p: _Prep, hp: dict) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(hp["seed"])
    N, edge_dim = p.N, (1 if p.spread is not None else 0)
    A = np.eye(N, dtype="float32")
    A[p.dst, p.src] = 1.0  # node i=dst attends to neighbour j=src
    At = torch.tensor(A, dtype=torch.float32)
    Xt = torch.tensor(p.Xs, dtype=torch.float32)
    Yt = torch.tensor(p.Ys, dtype=torch.float32)
    Pt = torch.tensor(p.present, dtype=torch.bool)
    Mt = torch.tensor(p.ymask, dtype=torch.float32)
    EFt = torch.tensor(p.spread[..., None], dtype=torch.float32) if p.spread is not None else None
    dropout = hp["dropout"]

    class GATLayer(torch.nn.Module):
        def __init__(self, in_dim, out_dim, n_heads, concat):
            super().__init__()
            self.h, self.o, self.concat = n_heads, out_dim, concat
            self.W = torch.nn.Linear(in_dim, n_heads * out_dim, bias=False)
            self.aL = torch.nn.Parameter(torch.empty(n_heads, out_dim))
            self.aR = torch.nn.Parameter(torch.empty(n_heads, out_dim))
            self.aE = torch.nn.Parameter(torch.empty(n_heads, edge_dim)) if edge_dim else None
            for q in (self.aL, self.aR, *(() if self.aE is None else (self.aE,))):
                torch.nn.init.xavier_uniform_(q)

        def forward(self, x, present, EF):
            B = x.shape[0]
            h = self.W(x).view(B, N, self.h, self.o)
            e = (h * self.aL).sum(-1)[:, :, None, :] + (h * self.aR).sum(-1)[:, None, :, :]
            if self.aE is not None:
                e = e + torch.einsum("bijd,hd->bijh", EF, self.aE)
            e = F.leaky_relu(e, 0.2)
            mask = (At > 0)[None, :, :, None] & present[:, None, :, None]
            e = e.masked_fill(~mask, torch.finfo(e.dtype).min)
            alpha = F.dropout(torch.softmax(e, dim=2), p=dropout, training=self.training)
            out = torch.einsum("bijh,bjho->biho", alpha, h)
            return out.reshape(B, N, self.h * self.o) if self.concat else out.mean(2)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            mods, in_dim = [], p.F
            for _ in range(hp["layers"] - 1):
                mods.append(GATLayer(in_dim, hp["hidden"], hp["heads"], concat=True))
                in_dim = hp["hidden"] * hp["heads"]
            mods.append(GATLayer(in_dim, 1, 1, concat=False))
            self.layers = torch.nn.ModuleList(mods)

        def forward(self, x, present, EF):
            for i, layer in enumerate(self.layers):
                x = layer(x, present, EF)
                if i < len(self.layers) - 1:
                    x = F.elu(x)
            return x.squeeze(-1)

    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    fit_t, val_t = torch.tensor(p.fit_rows), torch.tensor(p.val_rows)

    def ef(rows):
        return EFt[rows] if EFt is not None else None

    def mse(rows):
        pred = model(Xt[rows], Pt[rows], ef(rows))
        m = Mt[rows]
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
        change = model(Xt, Pt, EFt).numpy() * p.csd + p.cmu
    return p.spot + change


def gat_forecast(
    indexed: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int,
    train_times,
    oos_times,
    *,
    use_congestion: bool = False,
    congestion_grid=None,
    backend: str = "auto",
    hidden: int = 16,
    heads: int = 4,
    layers: int = 2,
    epochs: int = 60,
    lr: float = 5e-3,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    valid_frac: float = 0.15,
    seed: int = 0,
) -> pd.Series:
    """Train the GAT on train snapshots, return its OOS price forecast as a Series.

    ``backend="auto"`` uses PyG ``GATv2Conv`` when torch_geometric is installed,
    else the pure-torch dense GAT; force one with ``"pyg"``/``"dense"``. Both
    share `_prepare_arrays`, so they are a like-for-like A/B. Leakage-safe: only
    train-period rows fit the model and the scalers; the edge feature uses prices
    known at ``t``; predictions are exposed only on the OOS window.
    """
    if backend == "auto":
        backend = "pyg" if importlib.util.find_spec("torch_geometric") is not None else "dense"
    p = _prepare_arrays(indexed, features, feature_cols, k, train_times, use_congestion, valid_frac,
                        congestion_grid=congestion_grid)
    hp = dict(hidden=hidden, heads=heads, layers=layers, epochs=epochs, lr=lr,
              dropout=dropout, weight_decay=weight_decay, seed=seed)
    price = (_run_pyg if backend == "pyg" else _run_dense)(p, hp)

    full = pd.MultiIndex.from_product([p.times, p.nodes], names=indexed.index.names)
    series = pd.Series(price.reshape(-1), index=full, name=f"gat_{backend}")
    return series.where(series.index.get_level_values(0).isin(set(oos_times)))
