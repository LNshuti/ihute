{{
    config(
        materialized='table'
    )
}}

/*
    Metrics table for daily congestion KPIs.
*/

with flows as (
    select * from {{ ref('fct_corridor_flows') }}
),

daily_metrics as (
    select
        corridor_id,
        date_trunc('day', hour_bucket) as metric_date,

        -- Volume metrics
        sum(vehicle_count) as total_vehicles,
        avg(vehicle_count) as avg_hourly_volume,

        -- Speed metrics
        avg(avg_speed_mph) as daily_avg_speed,
        min(avg_speed_mph) as min_hourly_speed,
        avg(case when time_period = 'AM_PEAK' then avg_speed_mph end) as am_peak_avg_speed,
        avg(case when time_period = 'PM_PEAK' then avg_speed_mph end) as pm_peak_avg_speed,

        -- Congestion hours
        sum(case when congestion_severity in ('SEVERE', 'GRIDLOCK') then 1 else 0 end) as severe_congestion_hours,
        sum(case when congestion_severity != 'NONE' then 1 else 0 end) as any_congestion_hours,

        -- LOS distribution
        sum(case when level_of_service in ('E', 'F') then 1 else 0 end) as poor_los_hours,

        -- Reliability
        avg(planning_time_index) as avg_planning_time_index

    from flows
    group by corridor_id, date_trunc('day', hour_bucket)
)

select
    *,
    -- Congestion duration percentage
    severe_congestion_hours::float / 24 * 100 as severe_congestion_pct,
    any_congestion_hours::float / 24 * 100 as congestion_pct,
    current_timestamp as _loaded_at
from daily_metrics
