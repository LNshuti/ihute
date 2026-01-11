{{
    config(
        materialized='table'
    )
}}

/*
    Dimension table for corridor/zone reference data.
*/

with zones as (
    select * from {{ ref('stg_laddms__zones') }}
),

-- Add static corridor definitions
static_corridors as (
    select 'I-24' as corridor_id, 'Interstate 24' as corridor_name, 36.12 as center_lat, -86.75 as center_lon, 'INTERSTATE' as corridor_type
    union all
    select 'I-40', 'Interstate 40', 36.16, -86.78, 'INTERSTATE'
    union all
    select 'I-65', 'Interstate 65', 36.16, -86.78, 'INTERSTATE'
),

combined as (
    select
        coalesce(z.zone_id::text, sc.corridor_id) as corridor_key,
        coalesce(z.corridor_id, sc.corridor_id) as corridor_id,
        coalesce(z.zone_name, sc.corridor_name) as corridor_name,
        sc.corridor_type,
        coalesce(z.centroid_lat, sc.center_lat) as center_latitude,
        coalesce(z.centroid_lon, sc.center_lon) as center_longitude,
        z.mile_marker_start,
        z.mile_marker_end,
        z.area_sqm,
        current_timestamp as _loaded_at
    from static_corridors sc
    left join zones z on z.corridor_id = sc.corridor_id
)

select * from combined
