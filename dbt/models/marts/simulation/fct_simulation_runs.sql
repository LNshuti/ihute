{{
    config(
        materialized='table'
    )
}}

/*
    Fact table for simulation run results.
*/

with scenarios as (
    select * from {{ ref('int_simulation_scenarios') }}
),

final as (
    select
        treatment_run_id as run_key,
        treatment_scenario as scenario_name,
        scenario_type,
        primary_incentive_type,
        budget_usd,
        baseline_run_id,
        baseline_scenario,
        n_agents,
        duration_hours,

        -- Treatment results
        treatment_avg_speed,
        treatment_vmt,
        treatment_occupancy,
        treatment_peak_demand,
        treatment_spend,
        treatment_carpool_rate,

        -- Baseline results
        baseline_avg_speed,
        baseline_vmt,
        baseline_occupancy,
        baseline_peak_demand,
        baseline_carpool_rate,

        -- Improvements
        speed_improvement_mph,
        speed_improvement_pct,
        vmt_reduction,
        vmt_reduction_pct,
        occupancy_improvement,
        peak_reduction,
        peak_reduction_pct,

        -- Efficiency
        cost_per_vmt_reduced,

        current_timestamp as _loaded_at

    from scenarios
)

select * from final
