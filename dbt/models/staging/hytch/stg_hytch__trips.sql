{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for Hytch rideshare trip data.

    Standardizes fields and adds derived temporal/spatial features
    for behavioral model training.
*/

with source as (
    select * from {{ source('hytch', 'trips') }}
),

cleaned as (
    select
        -- Primary key
        trip_id,

        -- Temporal fields
        timestamp as trip_started_at,
        date_trunc('day', timestamp) as trip_date,
        date_trunc('hour', timestamp) as trip_hour,
        extract(hour from timestamp) as hour_of_day,
        extract(dow from timestamp) as day_of_week,
        extract(month from timestamp) as month_of_year,
        extract(year from timestamp) as year,

        -- Time period classification
        case
            when extract(hour from timestamp) between {{ var('am_peak_start') }} and {{ var('am_peak_end') }}
                then 'AM_PEAK'
            when extract(hour from timestamp) between {{ var('pm_peak_start') }} and {{ var('pm_peak_end') }}
                then 'PM_PEAK'
            else 'OFF_PEAK'
        end as time_period,

        -- Weekday indicator
        case
            when extract(dow from timestamp) between 1 and 5 then true
            else false
        end as is_weekday,

        -- Origin location
        origin_lat as origin_latitude,
        origin_lng as origin_longitude,

        -- Destination location
        dest_lat as destination_latitude,
        dest_lng as destination_longitude,

        -- Trip metrics
        distance_miles,
        duration_minutes,

        -- Derived: average speed
        case
            when duration_minutes > 0 then distance_miles / (duration_minutes / 60.0)
            else 0
        end as avg_speed_mph,

        -- Participant info
        n_participants as participant_count,
        is_carpool,

        -- Incentive info
        incentive_amount as incentive_amount_usd,

        -- Derived: incentive per mile
        case
            when distance_miles > 0 then incentive_amount / distance_miles
            else 0
        end as incentive_per_mile,

        -- Derived: incentive per participant
        case
            when n_participants > 0 then incentive_amount / n_participants
            else 0
        end as incentive_per_participant,

        -- Corridor detection based on origin/destination
        case
            when origin_lat between {{ var('i24_lat_min') }} and {{ var('i24_lat_max') }}
                 and origin_lng between {{ var('i24_lon_min') }} and {{ var('i24_lon_max') }}
                then true
            else false
        end as origin_on_i24,

        case
            when dest_lat between {{ var('i24_lat_min') }} and {{ var('i24_lat_max') }}
                 and dest_lng between {{ var('i24_lon_min') }} and {{ var('i24_lon_max') }}
                then true
            else false
        end as destination_on_i24,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where trip_id is not null
      and timestamp is not null
      and distance_miles >= 0
      and duration_minutes >= 0
)

select * from cleaned
