{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for LADDMS Post-Encroachment Time (PET) metrics.

    PET measures the time between when a vehicle leaves a conflict zone
    and when another vehicle enters it - a key safety metric.
*/

with source as (
    select * from {{ source('laddms', 'pet_metrics') }}
),

cleaned as (
    select
        -- Primary key (use existing pet_id from source)
        pet_id,

        -- Vehicle identifiers
        object_id_1,
        object_id_2,

        -- Ensure consistent ordering (smaller ID first)
        least(object_id_1, object_id_2) as vehicle_id_low,
        greatest(object_id_1, object_id_2) as vehicle_id_high,

        -- Temporal fields
        timestamp as recorded_at,
        date_trunc('hour', timestamp) as hour_bucket,
        extract(hour from timestamp) as hour_of_day,

        -- PET metric
        pet_seconds,

        -- PET severity classification
        case
            when pet_seconds < 1.0 then 'CRITICAL'
            when pet_seconds < 2.0 then 'SEVERE'
            when pet_seconds < 3.0 then 'MODERATE'
            when pet_seconds < 5.0 then 'MILD'
            else 'SAFE'
        end as pet_severity,

        -- Interaction type (default since not in seed data)
        'LANE_CHANGE' as interaction_type,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where pet_seconds is not null
      and pet_seconds >= 0
      and pet_seconds < 30  -- Filter unrealistic values
)

select * from cleaned
