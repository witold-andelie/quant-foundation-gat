"""Generate the report's figures into paper/figures/.

Same claim-ledger rule as gen_tables.py: anything data-bearing is read from
the repository, never typed by hand. The interconnector graph comes from the
package's reference constants, the forecast-ladder skills from the canonical
CSV, and the attention panel is copied from the archived E10 analysis.
Graphviz (`dot` on PATH) renders the diagrams to PDF.

Usage: py -3.13 paper/scripts/gen_figures.py   (from the repo root)
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT / "src"))

from quant_alpha.graph.edges_energy import (  # noqa: E402
    EUROPEAN_BIDDING_ZONES,
    EUROPEAN_INTERCONNECTORS,
)

NODE_STYLE = 'shape=box, style=rounded, fontname="Helvetica", fontsize=11'
EDGE_STYLE = 'fontname="Helvetica", fontsize=10, color=gray30, fontcolor=gray25'


def render(name: str, dot_source: str, engine: str = "dot") -> None:
    src = FIG / f"{name}.dot"
    out = FIG / f"{name}.pdf"
    src.write_text(dot_source, encoding="utf-8")
    subprocess.run(
        ["dot", f"-K{engine}", "-Tpdf", str(src), "-o", str(out)],
        check=True,
        capture_output=True,
    )
    print(f"wrote figures/{name}.pdf  ({engine})")


def interconnector() -> None:
    edges = sorted(tuple(sorted(pair)) for pair in EUROPEAN_INTERCONNECTORS)
    lines = [
        "graph interconnector {",
        '  graph [overlap=false, splines=true, sep="+7", start=5];',
        '  node [shape=box, style=rounded, fontname="Helvetica", fontsize=15,'
        ' margin="0.07,0.04", height=0.38];',
        "  edge [color=gray35, penwidth=1.1];",
    ]
    lines += [f'  "{z}";' for z in sorted(EUROPEAN_BIDDING_ZONES)]
    lines += [f'  "{a}" -- "{b}";' for a, b in edges]
    lines.append("}")
    render("interconnector", "\n".join(lines) + "\n", engine="neato")
    print(f"  ({len(EUROPEAN_BIDDING_ZONES)} zones, {len(edges)} interconnectors)")


def isolation_anchors() -> None:
    dot = f"""digraph iso {{
  rankdir=LR;
  ranksep=0.55; nodesep=0.45;
  node [{NODE_STYLE}, margin="0.16,0.10"];
  edge [{EDGE_STYLE}];
  feat   [label="ranked island-factor matrix\\n(identical input for all three)"];
  island [label="island anchor\\nequal-weight mean,\\nno propagation"];
  unif   [label="uniform anchor\\nunweighted neighbour mean\\nover the topology"];
  gat    [label="GAT\\nlearned attention\\nover the same topology"];
  harn   [label="one evaluation harness\\nembargoed splits, research gates,\\nleakage controls (OOS)"];
  feat -> island; feat -> unif; feat -> gat;
  island -> harn; unif -> harn; gat -> harn;
  island -> unif [style=dashed, dir=none, constraint=false, label="what the graph adds"];
  unif -> gat    [style=dashed, dir=none, constraint=false, label="attention value-add"];
}}
"""
    render("isolation_anchors", dot)


def energy_ladder() -> None:
    rows = {
        r["predictor"]: r
        for r in csv.DictReader(
            (ROOT / "docs" / "results" / "energy_forecast_node_skill.csv").open(encoding="utf-8")
        )
    }

    def skill(predictor: str) -> str:
        return rows[predictor]["skill"]

    dot = f"""digraph ladder {{
  rankdir=BT;
  ranksep=0.3; nodesep=0.25;
  node [{NODE_STYLE}, margin="0.13,0.08"];
  edge [{EDGE_STYLE}];
  r0 [label="persistence: tomorrow = today\\nskill {skill('persistence')} (reference)"];
  r1 [label="seasonal naive\\nskill {skill('seasonal_naive')}"];
  r2 [label="no-graph ridge on each zone's own drivers\\n(load, wind & solar forecasts, fuel price)\\nskill {skill('no_graph_ridge')}"];
  r3 [label="uniform-graph ridge: + interconnector-neighbour mean\\nskill {skill('uniform_graph_ridge')}"];
  r4 [label="node GAT (GATv2): learned attention on the same graph\\nskill {skill('gat_node')} (5-seed mean)"];
  r0 -> r1 [label=" + the diurnal cycle"];
  r1 -> r2 [label=" + own fundamentals"];
  r2 -> r3 [label=" + the physical graph, unlearned"];
  r3 -> r4 [label=" + learned attention"];
}}
"""
    render("energy_ladder", dot)


def attention_panel() -> None:
    src = ROOT / "docs" / "results" / "figures" / "2026-06-11_attention_neighbour_weight.png"
    dst = FIG / "att_neighbour_weight.png"
    shutil.copyfile(src, dst)
    print("wrote figures/att_neighbour_weight.png  (copied from archived E10 analysis)")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    interconnector()
    isolation_anchors()
    energy_ladder()
    attention_panel()


if __name__ == "__main__":
    main()
