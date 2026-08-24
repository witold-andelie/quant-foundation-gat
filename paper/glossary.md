# Report glossary — one term, one rendering

Seeded from `CONTEXT.md`. Every draft pass consults this before introducing a
term; never let a term drift to a synonym mid-report.

| Term (use exactly this) | Meaning | Never write |
|---|---|---|
| alpha | predictive cross-sectional signal per node per snapshot | indicator, signal (for the scored value) |
| factor | unified declaration of an alpha (metadata + compute) | alpha definition, signal spec |
| node | unit a graph connects (stock / bidding zone) | vertex, entity, asset |
| bidding zone | energy-track node (e.g. DE\_LU, FR) | region, area, market (for the node) |
| snapshot | point-in-time slice of features + topology | frame, slice, window |
| topology | directed edge set within a snapshot | graph, network, adjacency (for the edges) |
| propagator | seam mapping snapshot features + topology to factor values | model, layer, GNN (for the seam) |
| island factor | factor from a node's own data alone | traditional factor, single-name factor |
| relational factor | factor produced by propagation over the topology | GNN factor, graph factor |
| correlation graph | equity topology (estimated top-k correlation backbone) | adjacency, network |
| interconnector graph | energy topology (physical cross-border grid) | adjacency, network |
| island anchor | equal-weight composite, no propagation (A/B baseline) | control, benchmark |
| uniform anchor | uniform neighbour averaging over the same topology | control, benchmark |
| attention value-add | GAT OOS Sharpe minus uniform-anchor OOS Sharpe | — |
| composite | the GAT-produced relational factor scored per node | combined factor, ensemble |
| walk-forward | refit every oos\_chunk snapshots on data predating each boundary | rolling retrain |
| embargo | gap of >= k snapshots between train/valid/OOS segments | buffer, purge |
| skill score | 1 - MSE/MSE(persistence), OOS | skill (bare, on first use) |
| rank-IC | cross-sectional Spearman correlation of prediction vs label | IC (when Spearman is meant) |
| OOS | out-of-sample (define once, then use the abbreviation) | out of sample (inconsistent hyphenation) |
| day-ahead | the ENTSO-E day-ahead market context | dayahead, day ahead |
| graph lift | uniform-graph rung skill minus no-graph rung skill | graph gain, graph benefit |
| edge GAT | GAT node embeddings feeding an edge MLP head for spread prediction | edge model, spread GAT |
| long-short | portfolio long the top-ranked, short the bottom-ranked nodes | market-neutral, hedged |
| Sharpe ratio | annualised mean over volatility of strategy returns | SR, sharpe (lowercase) |
| what the graph adds | uniform anchor vs island anchor margin (alpha tracks) | graph value |
