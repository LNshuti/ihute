{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for simulation metrics time series.

    Captures key performance metrics at regular intervals
    during simulation runs.
*/

with source as (
    select * from {{ source('simulation', 'metrics_timeseries') }}
),

cleaned as (
    select
        -- Primary key
        {{ dbt_utils.generate_surrogate_key(['simulation_run_id', 'timestamp', 'metric_name']) }} as metric_id,

        -- Foreign keys
        simulation_run_id,

        -- Temporal fields
        timestamp as recorded_at,
        date_trunc('minute', timestamp) as minute_bucket,
        extract(epoch from timestamp) as epoch_seconds,

        -- Metric details
        metric_name,
        metric_value,

        -- Metric categorization
        case
            when metric_name ilike '%speed%' then 'TRAFFIC'
            when metric_name ilike '%flow%' then 'TRAFFIC'
            when metric_name ilike '%density%' then 'TRAFFIC'
            when metric_name ilike '%vmt%' then 'TRAFFIC'
            when metric_name ilike '%incentive%' then 'INCENTIVE'
            when metric_name ilike '%budget%' then 'INCENTIVE'
            when metric_name ilike '%uptake%' then 'INCENTIVE'
            when metric_name ilike '%occupancy%' then 'MODE_CHOICE'
            when metric_name ilike '%carpool%' then 'MODE_CHOICE'
            when metric_name ilike '%transit%' then 'MODE_CHOICE'
            else 'OTHER'
        end as metric_category,

        -- Unit inference
        case
            when metric_name ilike '%speed%' then 'mph'
            when metric_name ilike '%flow%' then 'veh/hr'
            when metric_name ilike '%density%' then 'veh/mi'
            when metric_name ilike '%vmt%' then 'miles'
            when metric_name ilike '%budget%' then 'USD'
            when metric_name ilike '%rate%' then 'fraction'
            when metric_name ilike '%count%' then 'count'
            else 'unknown'
        end as metric_unit,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where metric_name is not null
      and metric_value is not null
)

select * from cleaned
