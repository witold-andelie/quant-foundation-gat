"""End-to-end energy GAT relational-factor run — the dual-track sibling of
``run_gat_equity``.

Same kernel, heterogeneous graph: this orchestrator reuses the GAT model, the
section builder, the four-gate evaluation, and the attention-vs-uniform A/B
helpers verbatim; only three things differ from the equity track, and they are
exactly the energy track's identity:

    1. graph   — physical interconnector topology (`graph/edges_energy.py`),
                 not estimated return correlation.
    2. label   — floored hourly power return (`energy_cross_sectional_label`),
                 not the equity price ratio; k is in hours.
    3. nodes   — bidding zones with the 8 energy alphas as features.

That the rest is shared is the capstone's "one kernel, two graphs" thesis made
literal. Requires the ``[gnn]`` extra.
"""

from __future__ import annotations

import pandas as pd

from quant_alpha.backtest.diagnostics import evaluate_alpha_suite
from quant_alpha.config import BacktestConfig
from quant_alpha.features.energy_alpha import ENERGY_ALPHA_REGISTRY, add_energy_alpha_features
from quant_alpha.graph.edges_energy import (
    rolling_energy_topology_for,
    static_energy_topology_for,
)
from quant_alpha.graph.training import energy_cross_sectional_label, is_constrained_split
from quant_alpha.models.gat import (
    FactorGraphDataset,
    GATConfig,
    build_sections,
    composite_series,
    walk_forward_composite_series,
    fit,
)
from quant_alpha.run_gat_equity import (
    COMPOSITE_NAME,
    ISLAND_MEAN_NAME,
    LOSSES,
    UNIFORM_NAME,
    _baseline_columns,
    ab_report,
    gate_report,
)

ENERGY_ALPHA_NAMES = tuple(a.name for a in ENERGY_ALPHA_REGISTRY)


def _floored_forward_return(panel: pd.DataFrame, k: int, price_col: str, floor: float, clip: float) -> pd.Series:
    """The raw (un-standardised) label, for evaluate_alpha_suite's IC/backtest —
    same formula as the training label before cross-sectional standardisation."""
    cur = panel[price_col]
    fwd = panel.groupby(level=1)[price_col].transform(lambda s: s.shift(-k))
    return ((fwd - cur) / cur.abs().clip(lower=floor)).clip(-clip, clip)


def gat_energy_from_panel(
    raw: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    *,
    k: int = 24,
    window: int = 168,
    depth: int = 2,
    epochs: int = 50,
    train_ratio: float = 0.7,
    loss: str = "ic",
    graph: str = "static",
    retrain: str = "single",
    oos_chunk: int = 720,
    hidden_dim: int = 64,
    heads: int = 2,
    lr: float = 3e-3,
    floor: float = 20.0,
    clip: float = 0.8,
    device: str = "cpu",
    out_path: str = "data/warehouse/gat_energy.pt",
) -> dict:
    """Interconnector graph -> energy GAT -> composite -> four gates + A/B,
    given a power-market panel.

    ``raw`` is a flat power-market frame (``timestamp``/``market`` columns plus
    the raw inputs ``add_energy_alpha_features`` consumes). ``k`` is in hours.
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {sorted(LOSSES)}, got {loss!r}")
    if graph not in ("static", "dynamic"):
        raise ValueError(f"graph must be 'static' or 'dynamic', got {graph!r}")
    retrain = retrain.replace("-", "_")
    if retrain not in ("single", "walk_forward"):
        raise ValueError(f"retrain must be 'single' or 'walk_forward', got {retrain!r}")
    import torch

    torch_device = None if device == "auto" else torch.device(device)

    feats = add_energy_alpha_features(raw)
    for col in ENERGY_ALPHA_NAMES:
        feats[f"{col}_rank"] = feats.groupby("timestamp")[col].rank(pct=True)
    feats["ret_1d"] = feats.groupby("market")["spot_price"].pct_change()
    indexed = feats.set_index(["timestamp", "market"]).sort_index()
    indexed["forward_return"] = _floored_forward_return(indexed, k, "spot_price", floor, clip)

    feature_cols = tuple(f"{col}_rank" for col in ENERGY_ALPHA_NAMES)
    times = sorted(indexed.index.get_level_values(0).unique())
    n_is = int(len(times) * train_ratio) + 1
    split_time = times[n_is - 1]

    if graph == "dynamic":
        topology_for = rolling_energy_topology_for(
            indexed, None, return_col="ret_1d", window=window
        )
    else:
        topology_for = static_energy_topology_for(
            indexed, None, as_of=split_time, return_col="ret_1d", window=window
        )

    def label_fn(panel, k, price_col):
        return energy_cross_sectional_label(panel, k=k, price_col=price_col, floor=floor, clip=clip)

    dataset = FactorGraphDataset(
        build_sections(indexed, topology_for, feature_cols, k=k, price_col="spot_price", label_fn=label_fn)
    )
    gcfg = GATConfig(
        in_dim=len(feature_cols), hidden_dim=hidden_dim, heads=heads,
        num_layers=depth, forward_k=k, lr=lr, epochs=epochs,
    )
    if retrain == "walk_forward":
        composite = walk_forward_composite_series(
            dataset, gcfg, n_is=n_is, oos_chunk=oos_chunk, device=torch_device,
            loss_fn=LOSSES[loss], out_path=out_path, name=COMPOSITE_NAME,
        )
    else:
        train_idx, valid_idx = is_constrained_split(n_is, embargo=k)
        if len(valid_idx):
            assert train_idx.stop + k <= valid_idx.start, "train labels reach into valid"
            assert valid_idx.stop + k <= n_is, "valid labels reach into the OOS window"
        model = fit(
            dataset, gcfg, device=torch_device, loss_fn=LOSSES[loss], out_path=out_path,
            train_idx=train_idx, valid_idx=valid_idx,
        )
        composite = composite_series(model, dataset, name=COMPOSITE_NAME)

    island, uniform = _baseline_columns(indexed, topology_for, feature_cols)

    # Build the eval panel from `indexed` (it carries forward_return + ret_1d +
    # the alphas and their ranks); the flat `feats` lacks the derived columns.
    panel = indexed.reset_index().rename(columns={"timestamp": "date", "market": "symbol"})
    for series in (composite, island, uniform):
        panel = panel.merge(
            series.rename(series.name).reset_index().rename(
                columns={"timestamp": "date", "market": "symbol"}
            ),
            on=["date", "symbol"], how="left",
        )

    alpha_cols = list(ENERGY_ALPHA_NAMES) + [ISLAND_MEAN_NAME, UNIFORM_NAME, COMPOSITE_NAME]
    diagnostics, alpha_metrics, _ = evaluate_alpha_suite(
        panel, alpha_cols, backtest_cfg, split_date=str(split_time)
    )
    return {
        "panel": panel,
        "diagnostics": diagnostics,
        "alpha_metrics": alpha_metrics,
        "gate_report": gate_report(diagnostics, panel, list(ENERGY_ALPHA_NAMES)),
        "ab_report": ab_report(diagnostics, panel),
        "weights_path": out_path,
    }


def run_gat_energy(
    config_path,
    root,
    *,
    markets: list[str] | None = None,
    **kwargs,
) -> dict:
    """Generate (or load) a power-market panel, then run the energy GAT layer.

    Synthetic source by default — ENTSO-E needs an API token (see
    ``pipeline_energy``). ``markets`` defaults to the full interconnector zone
    set so the graph is dense enough for attention.
    """
    from quant_alpha.config import load_project_config
    from quant_alpha.graph.edges_energy import EUROPEAN_BIDDING_ZONES
    from quant_alpha.ingestion.energy import generate_synthetic_power_market

    cfg = load_project_config(config_path, root=root)
    zones = markets or list(EUROPEAN_BIDDING_ZONES)
    raw = generate_synthetic_power_market(
        zones, cfg.start_date, cfg.end_date or cfg.start_date, freq=cfg.bar_interval
    )
    return gat_energy_from_panel(raw, cfg.backtest, **kwargs)
