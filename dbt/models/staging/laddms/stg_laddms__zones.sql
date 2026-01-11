{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for LADDMS location zone definitions.

    Processes the zone geometry data for spatial joins
    and aggregations.
*/

with source as (
    select * from {{ source('laddms', 'location_zones') }}
),

cleaned as (
    select
        -- Primary key
        zone_id,

        -- Descriptive fields
        coalesce(zone_name, 'Zone ' || zone_id::text) as zone_name,

        -- Bounding box coordinates
        lat_min,
        lat_max,
        lon_min,
        lon_max,

        -- Centroid (calculated from bounds)
        (lat_min + lat_max) / 2.0 as centroid_lat,
        (lon_min + lon_max) / 2.0 as centroid_lon,

        -- Approximate area (simplified)
        (lat_max - lat_min) * (lon_max - lon_min) * 111000 * 111000 as area_sqm,

        -- Corridor classification (all I-24 for this dataset)
        'I-24' as corridor_id,

        -- Mile marker range
        mile_marker_start,
        mile_marker_end,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where zone_id is not null
)

select * from cleaned
