{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for Hytch trip participant data.

    Links individual participants to trips for analyzing
    carpool formation patterns.
*/

with source as (
    select * from {{ source('hytch', 'participants') }}
),

cleaned as (
    select
        -- Primary key
        participant_id,

        -- Foreign key
        trip_id,

        -- Participant details
        upper(coalesce(role, 'UNKNOWN')) as participant_role,
        user_id,

        -- Derived: is this the driver?
        case
            when upper(role) = 'DRIVER' then true
            else false
        end as is_driver,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where participant_id is not null
      and trip_id is not null
)

select * from cleaned
