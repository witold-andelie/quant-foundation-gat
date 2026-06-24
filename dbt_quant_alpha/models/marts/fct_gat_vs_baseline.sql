-- The relational A/B as one tidy mart: every alpha tagged by tier, so a query
-- can compare the learned GAT composite against the no-learning baselines
-- (island mean = no propagation; uniform = relational but unlearned) and the
-- island singles on the same OOS metrics. This is the "GAT vs Baseline" story.
select
    alpha_name,
    case
        when alpha_name = 'alpha_gat_composite' then 'relational_gat'
        when alpha_name = 'alpha_uniform_composite' then 'relational_unlearned'
        when alpha_name = 'alpha_island_mean' then 'island_mean'
        else 'island_single'
    end as tier,
    oos_ic_mean,
    oos_sharpe,
    consistency_score,
    robustness_score,
    is_oos_ic_same_sign
from {{ source('gat_relational', 'gat_alpha_diagnostics') }}
order by oos_sharpe desc
