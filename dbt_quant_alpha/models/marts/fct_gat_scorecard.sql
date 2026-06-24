-- One-row headline scorecard for the GAT relational composite: the four gates,
-- the value-add over the best single, and the attention A/B (GAT vs the uniform
-- and island anchors). Joins the gate report and the A/B report (each one row).
select
    g.gates_passed,
    g.composite_oos_ic_mean,
    g.composite_oos_sharpe,
    g.best_single_oos_sharpe,
    g.sharpe_value_added,
    g.value_added_passed,
    g.consistency_passed,
    g.uniqueness_passed,
    g.robustness_passed,
    a.attention_sharpe_value_add,
    a.uniform_oos_sharpe,
    a.island_oos_sharpe,
    a.gat_uniform_spearman
from {{ source('gat_relational', 'gat_gate_report') }} g
cross join {{ source('gat_relational', 'gat_ab_report') }} a
