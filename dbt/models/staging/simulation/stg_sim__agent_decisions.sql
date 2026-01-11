{{
    config(
        materialized='view'
    )
}}

/*
    Staging model for simulation agent decision records.

    Captures mode choice, route choice, departure time, and
    incentive response decisions made by agents.
*/

with source as (
    select * from {{ source('simulation', 'agent_decisions') }}
),

cleaned as (
    select
        -- Primary key
        decision_id,

        -- Foreign keys
        agent_id,
        simulation_run_id,

        -- Temporal fields
        timestamp as decided_at,
        extract(hour from timestamp) as hour_of_day,

        -- Decision details
        upper(decision_type) as decision_type,
        chosen_option,
        utility as utility_value,

        -- For mode choice decisions
        case
            when upper(decision_type) = 'MODE_CHOICE' then chosen_option
            else null
        end as chosen_mode,

        -- For route choice decisions
        case
            when upper(decision_type) = 'ROUTE_CHOICE' then chosen_option
            else null
        end as chosen_route,

        -- For departure time decisions
        case
            when upper(decision_type) = 'DEPARTURE_TIME' then chosen_option::float
            else null
        end as chosen_departure_time,

        -- For incentive response
        case
            when upper(decision_type) = 'INCENTIVE_RESPONSE'
                 and upper(chosen_option) in ('ACCEPT', 'YES', 'TRUE', '1')
                then true
            when upper(decision_type) = 'INCENTIVE_RESPONSE'
                then false
            else null
        end as accepted_incentive,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where decision_id is not null
      and agent_id is not null
)

select * from cleaned
