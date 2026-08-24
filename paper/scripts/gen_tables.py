"""Generate LaTeX tables from the canonical result CSVs in docs/results/.

Hand-typed result numbers drift from the artifacts and break the claim
ledger; every quantitative table in the report is produced here.

Usage: py -3.13 paper/scripts/gen_tables.py   (from the repo root)
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results"
OUT = ROOT / "paper" / "tables"

ESCAPES = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "->": r"$\to$"}


def tex_escape(value: str) -> str:
    for k, v in ESCAPES.items():
        value = value.replace(k, v)
    return value


def wrap(width: str) -> str:
    return rf">{{\raggedright\arraybackslash}}p{{{width}\textwidth}}"


def render(csv_name: str, out_name: str, caption: str, label: str,
           colspec: str | None = None) -> None:
    rows = list(csv.reader((RESULTS / csv_name).open(encoding="utf-8")))
    header, body = rows[0], rows[1:]
    if colspec is None:
        # Default: free-text last column wraps so notes cannot overflow.
        colspec = "l" * (len(header) - 1) + wrap("0.34")
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        "    " + " & ".join(tex_escape(h.replace("_", " ")) for h in header) + r" \\",
        r"    \midrule",
    ]
    for row in body:
        lines.append("    " + " & ".join(tex_escape(c) for c in row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (OUT / out_name).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote tables/{out_name}  ({len(body)} rows)")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    render(
        "equity_gat_summary.csv", "equity_summary.tex",
        "Equity track headline results (real yfinance data, 49 names, IC loss).",
        "tab:equity-summary",
        colspec="l" + wrap("0.28") + wrap("0.30"),
    )
    render(
        "energy_gnn_findings.csv", "energy_findings.tex",
        "Energy forecasting track: what is and is not supported "
        "(real ENTSO-E data, 20 zones, 5-seed means).",
        "tab:energy-findings",
        colspec=wrap("0.24") + wrap("0.22") + "lll",
    )
    render(
        "energy_forecast_node_skill.csv", "energy_node_skill.tex",
        "Node-level price forecast skill ladder (skill $=1-\\mathrm{MSE}/"
        "\\mathrm{MSE}_{\\mathrm{persistence}}$, OOS, $k=24$h).",
        "tab:energy-node-skill",
    )
    render(
        "energy_forecast_edge_skill.csv", "energy_edge_skill.tex",
        "Edge-level cross-border spread prediction (38 borders, OOS, "
        "$k=24$h, 5-seed means).",
        "tab:energy-edge-skill",
    )


if __name__ == "__main__":
    main()
