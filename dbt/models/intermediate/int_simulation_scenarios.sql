{{
    config(
        materialized='view'
    )
}}

/*
    Intermediate model for simulation scenario comparison.

    Links treatment scenarios to baselines for computing
    relative improvements.
*/

with runs as (
    select * from {{ ref('stg_sim__runs') }}
),

metrics as (
    select * from {{ ref('stg_sim__metrics_timeseries') }}
),

-- Identify baseline runs
baselines as (
    select
        simulation_run_id as baseline_run_id,
        scenario_name as baseline_scenario,
        n_agents,
        duration_hours,
        random_seed
    from runs
    where is_baseline
),

-- Identify treatment runs
treatments as (
    select
        simulation_run_id as treatment_run_id,
        scenario_name as treatment_scenario,
        scenario_type,
        primary_incentive_type,
        budget_usd,
        n_agents,
        duration_hours,
        random_seed
    from runs
    where not is_baseline
),

-- Match treatments to baselines (by agent count and seed)
matched as (
    select
        t.treatment_run_id,
        t.treatment_scenario,
        t.scenario_type,
        t.primary_incentive_type,
        t.budget_usd,
        b.baseline_run_id,
        b.baseline_scenario,
        t.n_agents,
        t.duration_hours
    from treatments t
    left join baselines b
        on t.n_agents = b.n_agents
        and t.random_seed = b.random_seed
),

-- Aggregate key metrics per run
run_metrics as (
    select
        simulation_run_id,
        avg(case when metric_name = 'avg_speed_mph' then metric_value end) as avg_speed_mph,
        avg(case when metric_name = 'total_vmt' then metric_value end) as total_vmt,
        avg(case when metric_name = 'avg_occupancy' then metric_value end) as avg_occupancy,
        avg(case when metric_name = 'peak_demand' then metric_value end) as peak_demand,
        sum(case when metric_name = 'incentive_spend' then metric_value end) as total_incentive_spend,
        avg(case when metric_name = 'carpool_rate' then metric_value end) as carpool_rate
    from metrics
    group by simulation_run_id
),

-- Join metrics to matched scenarios
final as (
    select
        m.treatment_run_id,
        m.treatment_scenario,
        m.scenario_type,
        m.primary_incentive_type,
        m.budget_usd,
        m.baseline_run_id,
        m.baseline_scenario,
        m.n_agents,
        m.duration_hours,

        -- Treatment metrics
        tm.avg_speed_mph as treatment_avg_speed,
        tm.total_vmt as treatment_vmt,
        tm.avg_occupancy as treatment_occupancy,
        tm.peak_demand as treatment_peak_demand,
        tm.total_incentive_spend as treatment_spend,
        tm.carpool_rate as treatment_carpool_rate,

        -- Baseline metrics
        bm.avg_speed_mph as baseline_avg_speed,
        bm.total_vmt as baseline_vmt,
        bm.avg_occupancy as baseline_occupancy,
        bm.peak_demand as baseline_peak_demand,
        bm.carpool_rate as baseline_carpool_rate,

        -- Absolute improvements
        tm.avg_speed_mph - bm.avg_speed_mph as speed_improvement_mph,
        bm.total_vmt - tm.total_vmt as vmt_reduction,
        tm.avg_occupancy - bm.avg_occupancy as occupancy_improvement,
        bm.peak_demand - tm.peak_demand as peak_reduction,

        -- Relative improvements
        case when bm.avg_speed_mph > 0
            then (tm.avg_speed_mph - bm.avg_speed_mph) / bm.avg_speed_mph * 100
            else 0 end as speed_improvement_pct,
        case when bm.total_vmt > 0
            then (bm.total_vmt - tm.total_vmt) / bm.total_vmt * 100
            else 0 end as vmt_reduction_pct,
        case when bm.peak_demand > 0
            then (bm.peak_demand - tm.peak_demand) / bm.peak_demand * 100
            else 0 end as peak_reduction_pct,

        -- Cost effectiveness
        case when (bm.total_vmt - tm.total_vmt) > 0
            then tm.total_incentive_spend / (bm.total_vmt - tm.total_vmt)
            else null end as cost_per_vmt_reduced

    from matched m
    left join run_metrics tm on m.treatment_run_id = tm.simulation_run_id
    left join run_metrics bm on m.baseline_run_id = bm.simulation_run_id
)

select * from final
