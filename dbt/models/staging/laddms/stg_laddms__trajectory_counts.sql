{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for LADDMS trajectory count aggregations.

    Cleans and standardizes the trajectory count data for use
    in flow analysis models.
*/

with source as (
    select * from {{ source('laddms', 'trajectory_counts') }}
),

cleaned as (
    select
        -- Surrogate key
        {{ dbt_utils.generate_surrogate_key(['timestamp', 'location_id']) }} as count_id,

        -- Temporal fields
        timestamp as recorded_at,
        date_trunc('hour', timestamp) as hour_bucket,
        date_trunc('day', timestamp) as date_bucket,
        extract(hour from timestamp) as hour_of_day,
        extract(dow from timestamp) as day_of_week,

        -- Location
        location_id,

        -- Metrics
        vehicle_count,

        -- Derived: is this a peak hour?
        case
            when extract(hour from timestamp) between {{ var('am_peak_start') }} and {{ var('am_peak_end') }}
                then 'AM_PEAK'
            when extract(hour from timestamp) between {{ var('pm_peak_start') }} and {{ var('pm_peak_end') }}
                then 'PM_PEAK'
            else 'OFF_PEAK'
        end as time_period,

        -- Derived: is this a weekday?
        case
            when extract(dow from timestamp) between 1 and 5 then true
            else false
        end as is_weekday,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where timestamp is not null
      and vehicle_count >= 0
)

select * from cleaned
