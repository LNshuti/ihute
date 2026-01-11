{{
    config(
        materialized='table'
    )
}}

/*
    Fact table for corridor traffic flows.

    Hourly aggregation of traffic metrics by corridor.
*/

with congestion as (
    select * from {{ ref('int_corridor_congestion') }}
),

final as (
    select
        -- Keys
        {{ dbt_utils.generate_surrogate_key(['corridor_id', 'hour_bucket']) }} as flow_key,
        corridor_id,
        zone_name,
        hour_bucket,

        -- Time attributes
        extract(hour from hour_bucket) as hour_of_day,
        extract(dow from hour_bucket) as day_of_week,
        time_period,

        -- Volume metrics
        vehicle_count,

        -- Speed metrics
        avg_speed_mph,
        median_speed_mph,
        p15_speed_mph,

        -- Derived metrics
        congestion_index,
        level_of_service,
        congestion_severity,
        planning_time_index,

        -- Quality metrics
        avg_smoothness,
        avg_traversal_time_sec,

        -- Metadata
        current_timestamp as _loaded_at

    from congestion
)

select * from final
