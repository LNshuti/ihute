{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for LADDMS trajectory data.

    Flattens the JSON array structure into individual position records
    and computes derived fields like speed and heading.
*/

with source as (
    select * from {{ source('laddms', 'raw_trajectories') }}
),

-- Parse string arrays and unnest into individual rows
-- CSV loads arrays as strings like "[1,2,3]", need to convert
parsed as (
    select
        object_id,
        location_id,
        classification,
        sub_classification,
        obj_length,
        obj_width,
        obj_height,
        avg_filtered_confidence,
        -- Parse string arrays to actual arrays
        string_to_array(replace(replace(ts, '[', ''), ']', ''), ',')::DOUBLE[] as ts_arr,
        string_to_array(replace(replace(x, '[', ''), ']', ''), ',')::DOUBLE[] as x_arr,
        string_to_array(replace(replace(y, '[', ''), ']', ''), ',')::DOUBLE[] as y_arr
    from source
    where ts is not null
      and length(ts) > 2  -- More than just "[]"
),

unnested as (
    select
        object_id,
        location_id,
        classification,
        sub_classification,
        obj_length,
        obj_width,
        obj_height,
        avg_filtered_confidence,
        unnest(ts_arr) as timestamp_sec,
        unnest(x_arr) as x_coord,
        unnest(y_arr) as y_coord,
        generate_subscripts(ts_arr, 1) as point_index
    from parsed
    where array_length(ts_arr) > 0
),

-- Add row number for lead/lag calculations
with_row_num as (
    select
        *,
        row_number() over (
            partition by object_id
            order by timestamp_sec
        ) as seq_num
    from unnested
),

-- Calculate speed from consecutive positions
with_speed as (
    select
        *,
        -- Calculate time delta
        timestamp_sec - lag(timestamp_sec) over (
            partition by object_id
            order by timestamp_sec
        ) as dt,
        -- Calculate position deltas
        x_coord - lag(x_coord) over (
            partition by object_id
            order by timestamp_sec
        ) as dx,
        y_coord - lag(y_coord) over (
            partition by object_id
            order by timestamp_sec
        ) as dy
    from with_row_num
),

-- Compute final metrics
final as (
    select
        -- Surrogate key
        {{ dbt_utils.generate_surrogate_key(['object_id', 'timestamp_sec']) }} as trajectory_id,

        -- Source fields
        object_id,
        location_id,
        coalesce(classification, 'UNKNOWN') as classification,
        sub_classification,
        obj_length as vehicle_length_m,
        obj_width as vehicle_width_m,
        obj_height as vehicle_height_m,
        avg_filtered_confidence as confidence_score,

        -- Temporal fields
        timestamp_sec,
        to_timestamp(timestamp_sec) as recorded_at,

        -- Spatial fields (local coordinates)
        x_coord,
        y_coord,

        -- Approximate lat/lon conversion (Nashville I-24 reference)
        -- Using approximate center point and meter-to-degree conversion
        36.12 + (y_coord / 111000.0) as latitude,
        -86.75 + (x_coord / (111000.0 * cos(radians(36.12)))) as longitude,

        -- Derived speed (meters per second)
        case
            when dt > 0 then sqrt(dx * dx + dy * dy) / dt
            else 0
        end as speed_mps,

        -- Derived speed in mph for display
        case
            when dt > 0 then sqrt(dx * dx + dy * dy) / dt * 2.237
            else 0
        end as speed_mph,

        -- Heading (degrees from north)
        case
            when dx != 0 or dy != 0 then degrees(atan2(dx, dy))
            else 0
        end as heading_degrees,

        -- Sequence tracking
        seq_num as point_sequence,

        -- Metadata
        current_timestamp as _loaded_at

    from with_speed
)

select * from final
