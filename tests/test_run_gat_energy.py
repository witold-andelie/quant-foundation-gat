from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from quant_alpha.config import ProjectConfig
from quant_alpha.ingestion.energy import generate_synthetic_power_market
from quant_alpha.run_gat_energy import gat_energy_from_panel
from quant_alpha.run_gat_equity import COMPOSITE_NAME, ISLAND_MEAN_NAME, UNIFORM_NAME

MARKETS = ["DE_LU", "FR", "NL", "BE", "AT", "CH"]


def test_gat_energy_end_to_end(tmp_path) -> None:
    raw = generate_synthetic_power_market(
        MARKETS, "2024-01-01", "2024-01-08", freq="h"  # ~169 hourly snapshots
    )
    cfg = ProjectConfig()

    out = gat_energy_from_panel(
        raw,
        cfg.backtest,
        k=6,
        window=48,
        epochs=1,
        hidden_dim=8,
        heads=2,
        train_ratio=0.7,
        out_path=str(tmp_path / "gat_energy.pt"),
    )

    panel = out["panel"]
    assert COMPOSITE_NAME in panel.columns
    assert panel[COMPOSITE_NAME].notna().sum() > 0
    # the no-learning energy anchors ride along
    assert panel[ISLAND_MEAN_NAME].notna().sum() > 0
    assert panel[UNIFORM_NAME].notna().sum() > 0

    assert COMPOSITE_NAME in set(out["diagnostics"]["alpha_name"])
    for gate in ("value_added", "consistency", "uniqueness", "robustness"):
        assert gate in out["gate_report"]
        assert "passed" in out["gate_report"][gate]
    for key in ("gat", "uniform_mean", "island_mean", "attention_sharpe_value_add"):
        assert key in out["ab_report"]
    assert (tmp_path / "gat_energy.pt").exists()


def test_gat_energy_walk_forward_and_dynamic(tmp_path) -> None:
    raw = generate_synthetic_power_market(MARKETS, "2024-01-01", "2024-01-08", freq="h")
    cfg = ProjectConfig()

    out = gat_energy_from_panel(
        raw,
        cfg.backtest,
        k=6,
        window=48,
        epochs=1,
        hidden_dim=8,
        heads=2,
        train_ratio=0.6,
        graph="dynamic",
        retrain="walk-forward",
        oos_chunk=24,
        out_path=str(tmp_path / "gat_energy_wf.pt"),
    )
    panel = out["panel"]
    dates = sorted(panel["date"].unique())
    oos = panel[panel["date"] > dates[int(len(dates) * 0.6)]]
    assert oos[COMPOSITE_NAME].notna().all()  # walk-forward covers the OOS window
    assert (tmp_path / "gat_energy_wf.pt").exists()


def test_gat_energy_drops_all_nan_alpha(tmp_path) -> None:
    # ENTSO-E carries no gas price -> alpha_energy_gas_spark_spread is all-NaN.
    # Dropping gas_price reproduces that: the run must drop the dead alpha and
    # still complete + evaluate on the surviving 7.
    raw = generate_synthetic_power_market(MARKETS, "2024-01-01", "2024-01-08", freq="h")
    raw = raw.drop(columns=["gas_price"])

    out = gat_energy_from_panel(
        raw, ProjectConfig().backtest,
        k=6, window=48, epochs=1, hidden_dim=8, heads=2, train_ratio=0.7,
        out_path=str(tmp_path / "gat_energy_nogas.pt"),
    )
    names = set(out["diagnostics"]["alpha_name"])
    assert "alpha_energy_gas_spark_spread" not in names  # dropped, not evaluated
    assert COMPOSITE_NAME in names
    assert out["panel"][COMPOSITE_NAME].notna().sum() > 0
