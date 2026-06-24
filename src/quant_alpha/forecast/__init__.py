"""Energy price/congestion forecasting — Phase 0 baseline ladder + skill report.

The honest reframe of the energy track after E11-E13b: instead of a
(non-tradeable) cross-sectional alpha, measure whether the interconnector graph
improves *forecast skill* over no-graph and unlearned-graph baselines. See
``docs/energy_forecasting.md`` and ``evaluate_energy_forecast``.
"""

from quant_alpha.forecast.evaluate import evaluate_energy_forecast
from quant_alpha.forecast.skill import forecast_skill, skill_report
from quant_alpha.forecast.target import forward_price_target, time_ordered_split

__all__ = [
    "evaluate_energy_forecast",
    "forecast_skill",
    "skill_report",
    "forward_price_target",
    "time_ordered_split",
]
