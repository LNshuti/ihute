{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for simulation run metadata.

    Captures configuration and status of each simulation run
    for comparison and analysis.
*/

with source as (
    select * from {{ source('simulation', 'simulation_runs') }}
),

cleaned as (
    select
        -- Primary key
        simulation_run_id,

        -- Run identification
        scenario_name,
        'run_' || simulation_run_id::text as run_name,

        -- Temporal fields
        started_at,
        completed_at,
        extract(epoch from (completed_at - started_at)) as duration_seconds,

        -- Status
        case
            when completed_at is not null then 'COMPLETED'
            when started_at is not null then 'RUNNING'
            else 'PENDING'
        end as run_status,

        -- Configuration
        config as config_json,

        -- Extract key config values
        coalesce((config->>'n_agents')::int, 0) as n_agents,
        coalesce((config->>'duration_hours')::float, 0) as duration_hours,
        coalesce(config->>'incentive_type', 'NONE') as primary_incentive_type,
        coalesce((config->>'budget')::float, 0) as budget_usd,
        coalesce((config->>'random_seed')::int, 0) as random_seed,

        -- Scenario classification
        case
            when scenario_name ilike '%baseline%' then true
            else false
        end as is_baseline,

        case
            when scenario_name ilike '%carpool%' then 'CARPOOL'
            when scenario_name ilike '%pacer%' then 'PACER'
            when scenario_name ilike '%transit%' then 'TRANSIT'
            when scenario_name ilike '%shift%' or scenario_name ilike '%temporal%' then 'DEPARTURE_SHIFT'
            else 'MIXED'
        end as scenario_type,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where simulation_run_id is not null
)

select * from cleaned
