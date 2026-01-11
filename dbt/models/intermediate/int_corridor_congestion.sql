{{
    config(
        materialized='view'
    )
}}

/*
    Intermediate model for corridor-level congestion metrics.

    Aggregates trajectory and count data into time-bucketed
    congestion indicators by zone/segment.
*/

with trajectory_speeds as (
    select * from {{ ref('int_trajectory_speeds') }}
),

trajectory_counts as (
    select * from {{ ref('stg_laddms__trajectory_counts') }}
),

zones as (
    select * from {{ ref('stg_laddms__zones') }}
),

-- Aggregate speeds by hour and location
hourly_speeds as (
    select
        location_id,
        date_trunc('hour', first_seen_at) as hour_bucket,
        time_period,

        count(*) as vehicle_count,
        avg(avg_speed_mph) as avg_speed_mph,
        percentile_cont(0.5) within group (order by avg_speed_mph) as median_speed_mph,
        percentile_cont(0.15) within group (order by avg_speed_mph) as p15_speed_mph,
        avg(smoothness_score) as avg_smoothness,
        avg(duration_seconds) as avg_traversal_time_sec

    from trajectory_speeds
    where first_seen_at is not null
    group by location_id, date_trunc('hour', first_seen_at), time_period
),

-- Join with zone info
with_zones as (
    select
        hs.*,
        z.zone_name,
        z.corridor_id,
        z.mile_marker_start,
        z.mile_marker_end
    from hourly_speeds hs
    left join zones z on hs.location_id = z.zone_id
),

-- Compute congestion metrics
final as (
    select
        *,

        -- Congestion index (ratio to free-flow speed, assumed 65 mph)
        case
            when avg_speed_mph > 0 then 65.0 / avg_speed_mph
            else null
        end as congestion_index,

        -- Level of Service estimation
        case
            when avg_speed_mph >= 60 then 'A'
            when avg_speed_mph >= 50 then 'B'
            when avg_speed_mph >= 40 then 'C'
            when avg_speed_mph >= 30 then 'D'
            when avg_speed_mph >= 20 then 'E'
            else 'F'
        end as level_of_service,

        -- Congestion severity
        case
            when avg_speed_mph < 15 then 'GRIDLOCK'
            when avg_speed_mph < 25 then 'SEVERE'
            when avg_speed_mph < 40 then 'MODERATE'
            when avg_speed_mph < 55 then 'MINOR'
            else 'NONE'
        end as congestion_severity,

        -- Reliability metric (based on speed variance)
        case
            when p15_speed_mph > 0 and avg_speed_mph > 0
            then p15_speed_mph / avg_speed_mph
            else 1
        end as planning_time_index

    from with_zones
)

select * from final
