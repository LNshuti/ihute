{{
    config(
        materialized='table'
    )
}}

/*
    Metrics table for cross-scenario comparison.
*/

with runs as (
    select * from {{ ref('fct_simulation_runs') }}
),

by_scenario_type as (
    select
        scenario_type,
        primary_incentive_type,

        -- Sample size
        count(*) as n_runs,

        -- Average improvements
        avg(speed_improvement_pct) as avg_speed_improvement_pct,
        avg(vmt_reduction_pct) as avg_vmt_reduction_pct,
        avg(peak_reduction_pct) as avg_peak_reduction_pct,
        avg(occupancy_improvement) as avg_occupancy_improvement,

        -- Variability
        stddev(speed_improvement_pct) as std_speed_improvement,
        stddev(vmt_reduction_pct) as std_vmt_reduction,

        -- Cost effectiveness
        avg(cost_per_vmt_reduced) as avg_cost_per_vmt,
        avg(treatment_spend) as avg_spend,
        avg(budget_usd) as avg_budget,

        -- Best/worst
        max(vmt_reduction_pct) as best_vmt_reduction_pct,
        min(vmt_reduction_pct) as worst_vmt_reduction_pct

    from runs
    where baseline_run_id is not null  -- Only matched scenarios
    group by scenario_type, primary_incentive_type
)

select
    *,
    -- Budget utilization
    case when avg_budget > 0 then avg_spend / avg_budget else 0 end as budget_utilization,
    current_timestamp as _loaded_at
from by_scenario_type
