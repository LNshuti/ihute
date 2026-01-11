{{
    config(
        materialized='view'
    )
}}

/*
    Intermediate model computing trajectory speed statistics.

    Aggregates individual trajectory points into vehicle-level
    statistics for traffic flow analysis.
*/

with trajectories as (
    select * from {{ ref('stg_laddms__trajectories') }}
),

-- Aggregate by vehicle
vehicle_stats as (
    select
        object_id,
        location_id,
        classification,

        -- Temporal bounds
        min(recorded_at) as first_seen_at,
        max(recorded_at) as last_seen_at,
        count(*) as n_observations,

        -- Speed statistics
        avg(speed_mps) as avg_speed_mps,
        avg(speed_mph) as avg_speed_mph,
        max(speed_mph) as max_speed_mph,
        min(speed_mph) as min_speed_mph,
        stddev(speed_mph) as speed_std_mph,

        -- Speed variance as smoothness proxy (lower = smoother)
        case
            when stddev(speed_mph) is not null and avg(speed_mph) > 0
            then 1 - least(stddev(speed_mph) / avg(speed_mph), 1)
            else 1
        end as smoothness_score,

        -- Position bounds
        min(latitude) as min_lat,
        max(latitude) as max_lat,
        min(longitude) as min_lon,
        max(longitude) as max_lon,

        -- Approximate distance traveled (sum of point-to-point)
        sum(speed_mps * 0.1) as approx_distance_m  -- Assuming ~0.1s between points

    from trajectories
    group by object_id, location_id, classification
),

-- Add time-of-day classification
final as (
    select
        *,

        -- Duration
        extract(epoch from (last_seen_at - first_seen_at)) as duration_seconds,

        -- Time period based on first observation
        case
            when extract(hour from first_seen_at) between {{ var('am_peak_start') }} and {{ var('am_peak_end') }}
                then 'AM_PEAK'
            when extract(hour from first_seen_at) between {{ var('pm_peak_start') }} and {{ var('pm_peak_end') }}
                then 'PM_PEAK'
            else 'OFF_PEAK'
        end as time_period,

        -- Speed classification
        case
            when avg_speed_mph < 20 then 'CONGESTED'
            when avg_speed_mph < 40 then 'SLOW'
            when avg_speed_mph < 60 then 'MODERATE'
            else 'FREE_FLOW'
        end as speed_category

    from vehicle_stats
    where n_observations >= 5  -- Filter short trajectories
)

select * from final
