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


def _floored_forward_return(
    panel: pd.DataFrame, k: int, price_col: str, floor: float, clip: float | None = None
) -> pd.Series:
    """The realised forward return for evaluate_alpha_suite's IC/backtest.

    Keeps the denominator floor (power prices go negative/near-zero, so
    ``clip(|price|, floor)`` stabilises the division) but does NOT value-clip
    by default. E13 showed that applying the training label's ``+/-0.8`` clip to
    the *evaluation* return manufactures a huge Sharpe by capping the
    short-leg's scarcity-spike tail losses (the strategy shorts expensive zones,
    which spike to ~1900 EUR/MWh): under the honest unclipped return the same
    cross-sectional strategy *loses* money (Sharpe ~ -1.5). Training may clip a
    robust target; evaluation must not hide realised tail risk. ``clip`` is kept
    as an opt-in only for diagnostics/back-compat."""
    cur = panel[price_col]
    fwd = panel.groupby(level=1)[price_col].transform(lambda s: s.shift(-k))
    ret = (fwd - cur) / cur.abs().clip(lower=floor)
    return ret.clip(-clip, clip) if clip is not None else ret


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
    # Drop alphas with no data at all (e.g. gas_spark_spread on ENTSO-E, which
    # carries no gas price): an all-NaN feature is a dead input dimension and an
    # all-NaN single would poison the max-of-singles gate. Synthetic data keeps
    # all 8.
    active_alphas = [name for name in ENERGY_ALPHA_NAMES if feats[name].notna().any()]
    dropped = [name for name in ENERGY_ALPHA_NAMES if name not in active_alphas]
    if dropped:
        print(f"energy: dropping all-NaN alphas {dropped} (no source data)", flush=True)
    for col in active_alphas:
        feats[f"{col}_rank"] = feats.groupby("timestamp")[col].rank(pct=True)
    feats["ret_1d"] = feats.groupby("market")["spot_price"].pct_change()
    indexed = feats.set_index(["timestamp", "market"]).sort_index()
    # Evaluation return is UNCLIPPED (clip=None): clipping the realised return
    # hides the short-leg tail risk and manufactures a fake Sharpe (E13). The
    # training label keeps its clip via energy_cross_sectional_label below.
    indexed["forward_return"] = _floored_forward_return(indexed, k, "spot_price", floor, clip=None)

    feature_cols = tuple(f"{col}_rank" for col in active_alphas)
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

    alpha_cols = list(active_alphas) + [ISLAND_MEAN_NAME, UNIFORM_NAME, COMPOSITE_NAME]
    diagnostics, alpha_metrics, _ = evaluate_alpha_suite(
        panel, alpha_cols, backtest_cfg, split_date=str(split_time)
    )
    return {
        "panel": panel,
        "diagnostics": diagnostics,
        "alpha_metrics": alpha_metrics,
        "gate_report": gate_report(diagnostics, panel, list(active_alphas)),
        "ab_report": ab_report(diagnostics, panel),
        "weights_path": out_path,
    }


def fetch_energy_raw(
    config_path,
    root,
    *,
    source: str = "synthetic",
    markets: list[str] | None = None,
    universe_path: str | None = None,
    include_generation: bool = True,
) -> pd.DataFrame:
    """Build a power-market panel from the synthetic generator or live ENTSO-E.

    source="synthetic" (default, no token) generates the full interconnector
    zone set. source="entsoe" fetches real day-ahead prices + load/wind/solar
    via ingestion.entsoe for the zones in universe_path (default
    configs/energy_universe_gnn.yaml); needs ENTSOE_API_KEY in the environment.
    Zones whose EIC code returns no data are dropped (the graph adapts to the
    zones present), so a partial or unverified EIC list still runs.
    """
    from quant_alpha.config import load_project_config, load_yaml
    from quant_alpha.graph.edges_equity import EUROPEAN_BIDDING_ZONES

    cfg = load_project_config(config_path, root=root)
    if source == "synthetic":
        from quant_alpha.ingestion.energy import generate_synthetic_power_market

        zones = markets or list(EUROPEAN_BIDDING_ZONES)
        return generate_synthetic_power_market(
            zones, cfg.start_date, cfg.end_date or cfg.start_date, freq=cfg.bar_interval
        )
    if source == "entsoe":
        from quant_alpha.ingestion.entsoe import EntsoeClient, fetch_entsoe_power_market

        upath = universe_path or (root / "configs" / "energy_universe_gnn.yaml")
        universe = load_yaml(upath)
        domains = {str(k): str(v) for k, v in universe.get("entsoe_domains", {}).items()}
        zones = markets or list(universe.get("markets", list(domains.keys())))
        client = EntsoeClient.from_env(
            token_env=cfg.entsoe.token_env,
            base_url=cfg.entsoe.base_url,
            timeout_seconds=cfg.entsoe.timeout_seconds,
        )
        raw = fetch_entsoe_power_market(
            markets=zones, domains=domains,
            start=cfg.start_date, end=cfg.end_date or cfg.start_date,
            bar_interval=cfg.bar_interval, client=client,
            include_generation=include_generation,
        )
        got = sorted(raw["market"].unique())
        print(f"ENTSO-E: {len(got)}/{len(zones)} zones returned data: {got}", flush=True)
        return raw
    raise ValueError(f"source must be 'synthetic' or 'entsoe', got {source!r}")


def run_gat_energy(
    config_path,
    root,
    *,
    source: str = "synthetic",
    markets: list[str] | None = None,
    universe_path: str | None = None,
    **kwargs,
) -> dict:
    """Generate (or load) a power-market panel, then run the energy GAT layer.

    Synthetic source by default — ENTSO-E needs an API token (see
    ``pipeline_energy``). ``markets`` defaults to the full interconnector zone
    set so the graph is dense enough for attention.
    """
    from quant_alpha.config import load_project_config

    cfg = load_project_config(config_path, root=root)
    raw = fetch_energy_raw(
        config_path, root, source=source, markets=markets, universe_path=universe_path
    )
    return gat_energy_from_panel(raw, cfg.backtest, **kwargs)
