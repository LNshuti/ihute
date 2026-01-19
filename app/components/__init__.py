"""Dashboard components."""

from .behavioral_calib import (
    create_elasticity_curve,
    create_feature_importance_chart,
    create_model_metrics_display,
    create_predicted_vs_actual,
    get_elasticity_data,
    get_feature_importance,
    get_model_metrics,
)
from .geo_map import (
    create_corridor_map,
    create_heatmap_overlay,
    create_zone_comparison,
    get_corridor_data,
    get_zone_stats,
)
from .incentive_analytics import (
    create_effectiveness_chart,
    create_funnel_chart,
    create_spend_chart,
    create_uptake_chart,
    get_effectiveness_data,
    get_incentive_funnel_data,
    get_spend_by_type_data,
    get_uptake_trend_data,
)
from .realtime_metrics import (
    create_kpi_gauge,
    create_metric_card,
    create_sparkline,
    get_kpi_data,
    get_trend_data,
)
from .simulation_compare import (
    create_baseline_treatment_comparison,
    create_cost_effectiveness_chart,
    create_scenario_comparison_chart,
    get_metrics_summary,
    get_scenario_comparison_data,
    get_scenario_list,
)
from .traffic_flow import (
    create_congestion_timeline,
    create_hourly_volume_chart,
    create_speed_heatmap,
    get_congestion_timeline_data,
    get_hourly_volume_data,
    get_speed_heatmap_data,
)
from .demographics import (
    get_demographics_summary,
    create_summary_cards,
    get_income_distribution,
    create_income_distribution_chart,
    create_poverty_distribution_chart,
    get_zcta_details,
    create_zcta_table,
    create_behavioral_impact_chart,
)
