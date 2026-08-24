# Presentation Outline

## Page 1 [cover]
- **Title**: Relational Alpha Factors with Graph Attention Networks
- **Content**: A Dual-Track Study on US Equities and European Power Markets | Wentao Ma | Capstone Report

## Page 2 [table_of_contents]
- **Title**: Overview
- **Content**: 1. What is Alpha?; 2. Research Questions; 3. Methodology & Platform; 4. Equity Track Results; 5. Energy Track Results; 6. Contributions & Conclusion

## Page 3 [content]
- **Title**: What is an Alpha Factor?
- **Content**: Chapter 2 "Alpha research in plain terms". For a non-finance audience (Prof. Anderle has CS + biology background). Define alpha factor as a scoring rule that ranks assets so higher-scored assets subsequently outperform lower-scored ones. Explain island vs relational factor, IC (rank correlation), and Sharpe ratio. Use a simple analogy: "imagine ranking cells by a gene-expression score, then checking if the top-ranked cells actually grow faster." Keep the explanation plain, with a small diagram showing the cross-section concept.

## Page 4 [content]
- **Title**: Three Research Questions
- **Content**: Chapter 1 "Introduction" paragraphs on RQ1-RQ3. RQ1: Does learned attention add value over unlearned propagation (same inputs, same topology)? RQ2: Does one GAT kernel transfer across structurally different graphs (equity correlation graph vs physical power grid) without false positives? RQ3: Does the physical topology improve forecast skill, and where does value concentrate? Present as three numbered cards with one-line summaries.

## Page 5 [content]
- **Title**: Methodology: The Attention A/B
- **Content**: Chapter 4 "The attention A/B: two no-learning anchors" and Figure 4.1. Show the isolation_anchors diagram: three constructions (island anchor, uniform anchor, GAT) share the same input matrix and same evaluation harness. Explain what the graph adds (island vs uniform) and what learned attention adds (uniform vs GAT). Include the actual isolation_anchors.pdf figure from the paper. This is the core methodological innovation.

## Page 6 [content]
- **Title**: Leakage-Safe Evaluation Harness
- **Content**: Chapter 4 "Split protocol and leakage controls" and "Model and training objective". Explain the train | embargo | valid | embargo | OOS split, best-epoch checkpointing, and the two automated controls (shuffle-label negative control and planted-signal positive control). Emphasize that the controls are structural (asserted in code, not remembered by the experimenter). Use a simple timeline diagram built from shapes.

## Page 7 [content]
- **Title**: Equity Track: Headline Results
- **Content**: Chapter 5 "Summary of the equity evidence" and Table 5.1 (equity_summary). Key numbers: 30/30 seeded runs with positive attention value-add; best OOS IC 0.0148 ± 0.0043; OOS Sharpe 1.37 ± 0.39; 3/4 gates pass (Value-added fails). Include the equity summary table. Add a bottom insight bar: "The composite is a real, unique signal — but it does not beat the single best island alpha."

## Page 8 [content]
- **Title**: What the Attention Actually Learned
- **Content**: Chapter 5 "What the attention learned" and Figure 5.1 (att_neighbour_weight). Include the attention neighbour weight figure. Key findings: 91.6% attention mass on neighbours (not self-loops), proving the graph is used; nearly uniform distribution across neighbours (gentle reweighting, not sharp selection); temporally stationary (0.91–0.92 band) with no break at IS/OOS boundary. Include the actual att_neighbour_weight.png figure.

## Page 9 [content]
- **Title**: Energy Track: The Physical Graph
- **Content**: Chapter 6 opening and Figure 6.1 (interconnector). Show the interconnector.pdf figure: 20 European bidding zones and 38 physical cross-border links. DE_LU is the natural hub with 11 interconnectors. Explain that unlike the equity graph (estimated from correlations), this topology is physical — it is the actual power grid. Briefly note the energy track's alpha framing failed honestly (three artifacts caught by controls), but the forecasting reframe succeeded.

## Page 10 [content]
- **Title**: Energy Forecasting: Skill Ladder
- **Content**: Chapter 6 "The reframe" and Figure 6.2 / Table 6.2 (energy_ladder / energy_node_skill). Show the energy_ladder.pdf figure. Each rung adds one ingredient: persistence (0) → seasonal naive (0) → ridge on own drivers (0.224) → uniform graph (+0.131 lift) → GAT learned attention (0.347). The +0.131 graph lift is the cleanest evidence: the physical interconnector graph itself improves forecast skill, validated by a synthetic negative control.

## Page 11 [content]
- **Title**: Edge-Level Spread Prediction
- **Content**: Chapter 6 "Edge-level spread prediction" and Table 6.3 (energy_edge_skill). Present the edge-level results table. Edge GAT beats both-endpoint ridge by +0.056 skill (29% relative) in 5/5 seeds. The target is irreducibly relational — cross-border price spread does not exist for a single node. Synthetic control inverts (no gain on independent zones), so the gain is structure, not capacity. This is the project's strongest relational result.

## Page 12 [content]
- **Title**: Key Contributions & Transferable Lessons
- **Content**: Chapter 8 "Conclusion" and Chapter 7 "Discussion". Five contributions: (1) isolation methodology (matched anchors, gates, controls), (2) equity result (attention value-add distribution), (3) cautionary example (three artifacts caught), (4) forecasting reframe with control-validated graph value, (5) reproducible platform engineering. Transferable lessons: evaluate on untransformed returns; treat implausible magnitudes as alarms; require seeds, fixed devices, and cross-implementation replication; negative controls give positive results their meaning.

## Page 13 [final]
- **Title**: Conclusion & Future Work
- **Content**: Three RQs answered directly. RQ1: Yes, learned attention adds value (30/30 seeds, modest but consistent). RQ2: Yes, one kernel transfers without manufacturing false positives (synthetic controls work on both graphs). RQ3: Yes, physical topology improves forecasts (+0.131 node skill, +0.056 edge skill). Future work: survivorship-bias-free equity data, portfolio-level value-added gate, tuned forecasting track, congestion hypothesis with flow-based coupling. Final sentence: "The discipline required to measure that premium honestly — anchors, controls, embargoes, seeds, and the willingness to let three spectacular numbers die — is itself the result this project most confidently recommends."
