from __future__ import annotations

from pathlib import Path

import typer

from quant_alpha.pipeline import run_pipeline
from quant_alpha.pipeline_energy import run_energy_pipeline
from quant_alpha.ingestion.entsoe import EntsoeError
from quant_alpha.ingestion.dlt_energy import run_dlt_energy_pipeline
from quant_alpha.ingestion.dlt_equity import run_dlt_equity_pipeline

app = typer.Typer(help="Quant alpha data engineering pipeline.", invoke_without_command=True)


def _run(config: Path, root: Path, offline: bool) -> None:
    result = run_pipeline(config, root.resolve(), offline=offline)
    typer.echo("Pipeline finished.")
    typer.echo(f"DuckDB: {result['duckdb_path']}")
    typer.echo(f"Rows: {result['rows']}")
    typer.echo(f"Metrics: {result['metrics']}")


def _run_energy(config: Path, root: Path, source: str | None = None) -> None:
    try:
        result = run_energy_pipeline(config, root.resolve(), source_override=source)
    except EntsoeError as exc:
        typer.echo(f"ENTSO-E ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Energy pipeline finished.")
    typer.echo(f"Data source: {result['data_source']}")
    typer.echo(f"DuckDB: {result['duckdb_path']}")
    if result.get("cloud_exports"):
        typer.echo(f"Cloud exports: {result['cloud_exports']}")
    typer.echo(f"Rows: {result['rows']}")
    typer.echo(f"Metrics: {result['metrics']}")


@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config YAML."),
    root: Path = typer.Option(Path("."), help="Project root."),
    offline: bool = typer.Option(False, help="Use deterministic synthetic prices."),
) -> None:
    """Run the pipeline directly, or use the `run` subcommand."""
    if ctx.invoked_subcommand is None:
        _run(config, root, offline)


@app.command("run")
def run_command(
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config YAML."),
    root: Path = typer.Option(Path("."), help="Project root."),
    offline: bool = typer.Option(False, help="Use deterministic synthetic prices."),
) -> None:
    """Run ingestion, factor computation, warehouse loading, and backtest."""
    _run(config, root, offline)


@app.command("energy-run")
def energy_run_command(
    config: Path = typer.Option(
        Path("configs/second_foundation_project.yaml"),
        help="Energy project config YAML.",
    ),
    root: Path = typer.Option(Path("."), help="Project root."),
    source: str | None = typer.Option(
        None,
        help="Override energy data source: synthetic or entsoe.",
    ),
) -> None:
    """Run the Second Foundation-inspired energy research pipeline."""
    _run_energy(config, root, source)


@app.command("dlt-energy")
def dlt_energy_command(
    root: Path = typer.Option(Path("."), help="Project root."),
    markets: str = typer.Option("DE_LU,CZ,FR", help="Comma-separated bidding zone list."),
    start: str = typer.Option("2023-01-01", help="Start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, help="End date (YYYY-MM-DD), defaults to today."),
) -> None:
    """Run dlt-based energy ingestion pipeline (incremental, schema-managed)."""
    from quant_alpha.config import load_project_config

    cfg = load_project_config(Path("configs/second_foundation_project.yaml"), root=root.resolve())
    market_list = [m.strip() for m in markets.split(",")]
    info = run_dlt_energy_pipeline(cfg.duckdb_path, markets=market_list, start=start, end=end)
    typer.echo("dlt energy pipeline complete.")
    typer.echo(f"Dataset:  {info['dataset']} in {info['duckdb_path']}")
    typer.echo(f"Packages: {info['load_packages']}")


@app.command("dlt-equity")
def dlt_equity_command(
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config YAML."),
    root: Path = typer.Option(Path("."), help="Project root."),
    offline: bool = typer.Option(True, help="Use synthetic prices (no API key needed)."),
) -> None:
    """Run dlt-based equity ingestion pipeline (incremental, schema-managed)."""
    from quant_alpha.config import load_project_config, load_universe

    cfg = load_project_config(config, root=root.resolve())
    universe = load_universe(cfg.universe_path)
    info = run_dlt_equity_pipeline(cfg.duckdb_path, cfg=cfg, universe=universe, offline=offline)
    typer.echo("dlt equity pipeline complete.")
    typer.echo(f"Dataset:  {info['dataset']} in {info['duckdb_path']}")
    typer.echo(f"Packages: {info['load_packages']}")


@app.command("gat-equity")
def gat_equity_command(
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config YAML."),
    root: Path = typer.Option(Path("."), help="Project root."),
    offline: bool = typer.Option(True, help="Use deterministic synthetic prices."),
    epochs: int = typer.Option(50, help="GAT training epochs."),
    loss: str = typer.Option("ic", help="Training loss: ic (default) or mse."),
    graph: str = typer.Option("static", help="Graph mode: static or dynamic (per-snapshot)."),
    retrain: str = typer.Option("single", help="Fit mode: single or walk-forward."),
    oos_chunk: int = typer.Option(63, help="Walk-forward refit interval in snapshots."),
    device: str = typer.Option("cpu", help="Training device: cpu (default), cuda, or auto."),
    persist: bool = typer.Option(False, help="Write GAT tables to DuckDB for the dbt marts."),
) -> None:
    """Run the equity GAT relational-factor pipeline (requires the [gnn] extra)."""
    from quant_alpha.run_gat_equity import run_gat_equity

    out = run_gat_equity(
        config, root.resolve(), offline=offline, epochs=epochs,
        loss=loss, graph=graph, retrain=retrain, oos_chunk=oos_chunk, device=device,
        persist=persist,
    )
    if out.get("persisted_tables"):
        typer.echo(f"Persisted to DuckDB: {out['persisted_tables']}")
    gates = out["gate_report"]
    typer.echo("GAT equity pipeline finished.")
    typer.echo(f"Composite OOS IC mean: {gates['composite_oos_ic_mean']:.4f}")
    typer.echo(f"Value-added : {gates['value_added']}")
    typer.echo(f"Consistency : {gates['consistency']}")
    typer.echo(f"Uniqueness  : {gates['uniqueness']}")
    typer.echo(f"Robustness  : {gates['robustness']}")
    typer.echo(f"A/B anchor  : {out['ab_report']}")


def _fetch_energy_raw(cfg, root: Path, source: str):
    """Load a power-market frame for the (torch-free) forecast harness.

    Mirrors run_gat_energy.fetch_energy_raw but without importing the GAT runner,
    so `energy-forecast` stays torch-free. ENTSO-E pulls the enriched frame
    (generation mix + actual load) via include_generation=True."""
    from quant_alpha.config import load_yaml
    from quant_alpha.graph.edges_energy import EUROPEAN_BIDDING_ZONES

    if source == "synthetic":
        from quant_alpha.ingestion.energy import generate_synthetic_power_market

        return generate_synthetic_power_market(
            list(EUROPEAN_BIDDING_ZONES),
            cfg.start_date,
            cfg.end_date or cfg.start_date,
            freq=cfg.bar_interval,
        )
    if source == "entsoe":
        from quant_alpha.ingestion.entsoe import EntsoeClient, fetch_entsoe_power_market

        universe = load_yaml(root / "configs" / "energy_universe_gnn.yaml")
        domains = {str(k): str(v) for k, v in universe.get("entsoe_domains", {}).items()}
        zones = list(universe.get("markets", list(domains.keys())))
        client = EntsoeClient.from_env(
            token_env=cfg.entsoe.token_env,
            base_url=cfg.entsoe.base_url,
            timeout_seconds=cfg.entsoe.timeout_seconds,
        )
        raw = fetch_entsoe_power_market(
            markets=zones,
            domains=domains,
            start=cfg.start_date,
            end=cfg.end_date or cfg.start_date,
            bar_interval=cfg.bar_interval,
            client=client,
            include_generation=True,
        )
        got = sorted(raw["market"].unique())
        typer.echo(f"ENTSO-E: {len(got)}/{len(zones)} zones returned data: {got}")
        return raw
    raise typer.BadParameter("source must be 'synthetic' or 'entsoe'")


@app.command("energy-forecast")
def energy_forecast_command(
    config: Path = typer.Option(
        Path("configs/second_foundation_project.yaml"), help="Energy project config YAML."
    ),
    root: Path = typer.Option(Path("."), help="Project root."),
    source: str = typer.Option("synthetic", help="Energy data source: synthetic or entsoe."),
    k: int = typer.Option(24, help="Forecast horizon in snapshots (hours)."),
    train_ratio: float = typer.Option(0.7, help="Fraction of the timeline used for training."),
    raw_path: str | None = typer.Option(
        None,
        help="Parquet cache for the raw power-market frame. Read if it exists, "
        "else fetched (once) and written here — so a slow ENTSO-E pull is reused.",
    ),
    persist: bool = typer.Option(False, help="Write the skill report to DuckDB."),
) -> None:
    """Phase 0 energy price-forecast skill ladder (persistence / seasonal /
    no-graph / uniform-graph). Torch-free; measures whether the interconnector
    graph improves forecast skill. The GAT rung lands in Phase 2."""
    import pandas as pd

    from quant_alpha.config import load_project_config
    from quant_alpha.forecast import evaluate_energy_forecast

    cfg = load_project_config(config, root=root.resolve())
    cache = Path(raw_path) if raw_path else None
    if cache and cache.exists():
        typer.echo(f"Loading cached raw frame from {cache}")
        raw = pd.read_parquet(cache)
    else:
        raw = _fetch_energy_raw(cfg, root.resolve(), source)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            raw.to_parquet(cache, index=False)
            typer.echo(f"Cached raw frame ({len(raw)} rows) to {cache}")
    out = evaluate_energy_forecast(raw, k=k, train_ratio=train_ratio)

    typer.echo(f"Energy forecast skill ladder (OOS, k={k}h, source={source}):")
    typer.echo(out["report"].to_string(index=False))
    typer.echo(
        f"\nGraph lift (uniform-graph - no-graph skill): "
        f"{out['graph_lift_uniform_vs_nograph']:+.4f}"
    )
    typer.echo(f"Features: {out['feature_cols']}")
    if persist:
        from quant_alpha.storage.duckdb import write_table

        write_table(cfg.duckdb_path, "energy_forecast_skill", out["report"])
        typer.echo(f"Persisted energy_forecast_skill to {cfg.duckdb_path}")


@app.command("bruin-lineage")
def bruin_lineage_command(
    bruin_root: Path = typer.Option(Path("bruin"), help="Path to bruin/ directory."),
    asset: str | None = typer.Option(None, help="Show upstream/downstream for a specific asset."),
) -> None:
    """Print the Bruin asset lineage graph."""
    from quant_alpha.platform.bruin_graph import AssetGraph

    graph = AssetGraph(bruin_root.resolve())
    if asset:
        if asset not in graph.nodes:
            typer.echo(f"Asset '{asset}' not found.", err=True)
            raise typer.Exit(1)
        typer.echo(f"Upstream of '{asset}': {graph.upstream(asset)}")
        typer.echo(f"Downstream of '{asset}': {graph.downstream(asset)}")
    else:
        typer.echo(graph.lineage_report())


@app.command("bruin-run")
def bruin_run_command(
    bruin_root: Path = typer.Option(Path("bruin"), help="Path to bruin/ directory."),
    targets: str | None = typer.Option(
        None, help="Comma-separated asset names to run (with upstream). Default: all."
    ),
    dry_run: bool = typer.Option(False, help="Print execution plan without running."),
    project_root: Path = typer.Option(Path("."), help="PROJECT_ROOT env for asset runners."),
) -> None:
    """Run Bruin asset graph (topological order, upstream-aware)."""
    from quant_alpha.platform.bruin_graph import AssetGraph, AssetStatus

    graph = AssetGraph(bruin_root.resolve())
    target_list = [t.strip() for t in targets.split(",")] if targets else None
    env = {"PROJECT_ROOT": str(project_root.resolve())}

    typer.echo(f"Bruin asset graph — {len(graph.nodes)} assets loaded")
    if dry_run:
        typer.echo("(dry-run mode)")

    results = graph.run(targets=target_list, env=env, dry_run=dry_run)
    typer.echo(graph.status_report())

    failed = [n for n, s in results.items() if s == AssetStatus.FAILED]
    if failed:
        typer.echo(f"\nFailed assets: {failed}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
