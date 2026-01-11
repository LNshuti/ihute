{{
    config(
        materialized='view'
    )
}}

/*
    Intermediate model engineering features for behavioral modeling.

    Prepares Hytch trip data with features suitable for
    training mode choice and incentive response models.
*/

with trips as (
    select * from {{ ref('stg_hytch__trips') }}
),

-- Feature engineering
featured as (
    select
        trip_id,
        trip_started_at,
        trip_date,

        -- Temporal features (cyclical encoding)
        sin(2 * pi() * hour_of_day / 24) as hour_sin,
        cos(2 * pi() * hour_of_day / 24) as hour_cos,
        sin(2 * pi() * day_of_week / 7) as dow_sin,
        cos(2 * pi() * day_of_week / 7) as dow_cos,
        sin(2 * pi() * month_of_year / 12) as month_sin,
        cos(2 * pi() * month_of_year / 12) as month_cos,

        -- Binary temporal features
        case when time_period = 'AM_PEAK' then 1 else 0 end as is_am_peak,
        case when time_period = 'PM_PEAK' then 1 else 0 end as is_pm_peak,
        case when is_weekday then 1 else 0 end as is_weekday_int,

        -- Spatial features (grid encoding)
        floor(origin_latitude * 100) / 100 as origin_lat_grid,
        floor(origin_longitude * 100) / 100 as origin_lon_grid,
        floor(destination_latitude * 100) / 100 as dest_lat_grid,
        floor(destination_longitude * 100) / 100 as dest_lon_grid,

        -- Distance features
        distance_miles,
        ln(distance_miles + 1) as log_distance,
        case
            when distance_miles < 5 then 'SHORT'
            when distance_miles < 15 then 'MEDIUM'
            else 'LONG'
        end as distance_category,

        -- Duration features
        duration_minutes,
        ln(duration_minutes + 1) as log_duration,

        -- Speed features
        avg_speed_mph,
        case
            when avg_speed_mph < 20 then 'CONGESTED'
            when avg_speed_mph < 35 then 'SLOW'
            else 'NORMAL'
        end as speed_category,

        -- Incentive features
        incentive_amount_usd,
        incentive_per_mile,
        incentive_per_participant,
        ln(incentive_amount_usd + 1) as log_incentive,

        -- Incentive buckets
        case
            when incentive_amount_usd = 0 then 'NONE'
            when incentive_amount_usd < 2 then 'LOW'
            when incentive_amount_usd < 5 then 'MEDIUM'
            else 'HIGH'
        end as incentive_bucket,

        -- Target variables
        is_carpool,
        participant_count,
        case when participant_count > 1 then 1 else 0 end as is_shared_ride,

        -- Corridor features
        case when origin_on_i24 or destination_on_i24 then 1 else 0 end as uses_i24,

        -- Interaction features
        incentive_amount_usd * case when time_period != 'OFF_PEAK' then 1 else 0 end as incentive_peak_interaction,
        distance_miles * incentive_per_mile as distance_incentive_interaction

    from trips
    where distance_miles > 0
      and duration_minutes > 0
)

select * from featured
