select
    cast(date as date) as signal_date,
    symbol,
    forward_return,
    alpha_island_mean,
    alpha_uniform_composite,
    alpha_gat_composite
from {{ source('gat_relational', 'gat_factor_panel') }}
