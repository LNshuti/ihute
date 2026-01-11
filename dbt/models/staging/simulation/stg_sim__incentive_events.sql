{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for simulation incentive event records.

    Tracks the lifecycle of incentives: offers, acceptances,
    completions, and expirations.
*/

with source as (
    select * from {{ source('simulation', 'incentive_events') }}
),

cleaned as (
    select
        -- Primary key
        event_id,

        -- Foreign keys
        agent_id,
        simulation_run_id,
        allocation_id,  -- Links related events together

        -- Temporal fields
        timestamp as event_at,
        date_trunc('hour', timestamp) as event_hour,
        extract(hour from timestamp) as hour_of_day,

        -- Event classification
        upper(incentive_type) as incentive_type,
        upper(event_type) as event_type,

        -- Financial details
        coalesce(amount, 0) as amount_usd,

        -- Event-specific flags
        case when upper(event_type) = 'OFFER' then true else false end as is_offer,
        case when upper(event_type) = 'ACCEPT' then true else false end as is_accept,
        case when upper(event_type) = 'REJECT' then true else false end as is_reject,
        case when upper(event_type) = 'COMPLETE' then true else false end as is_complete,
        case when upper(event_type) = 'EXPIRE' then true else false end as is_expire,
        case when upper(event_type) = 'CANCEL' then true else false end as is_cancel,

        -- Outcome flags
        case
            when upper(event_type) in ('COMPLETE') then true
            else false
        end as is_successful,

        -- Additional context
        conditions as event_conditions,
        performance_metrics,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where event_id is not null
)

select * from cleaned
