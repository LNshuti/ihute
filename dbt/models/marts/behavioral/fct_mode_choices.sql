{{
    config(
        materialized='table'
    )
}}

/*
    Fact table for mode choice analysis from Hytch data.
*/

with trips as (
    select * from {{ ref('int_trip_features') }}
),

final as (
    select
        trip_id as choice_key,
        trip_started_at,
        trip_date,

        -- Mode choice outcome
        is_carpool,
        is_shared_ride,
        participant_count,

        -- Key features
        distance_miles,
        duration_minutes,
        avg_speed_mph,
        incentive_amount_usd,
        incentive_per_mile,

        -- Temporal context
        is_am_peak,
        is_pm_peak,
        is_weekday_int as is_weekday,

        -- Spatial context
        uses_i24,
        origin_lat_grid,
        origin_lon_grid,

        -- Feature buckets
        distance_category,
        incentive_bucket,

        current_timestamp as _loaded_at

    from trips
)

select * from final
