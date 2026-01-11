{{
    config(
        materialized='table'
    )
}}

/*
    Fact table for incentive events with outcomes.
*/

with outcomes as (
    select * from {{ ref('int_incentive_outcomes') }}
),

final as (
    select
        allocation_id as incentive_key,
        agent_id,
        simulation_run_id,
        incentive_type,

        -- Offer details
        offered_amount,
        offered_at,
        offer_hour,

        -- Outcome
        final_outcome,
        is_successful,
        was_accepted,
        was_rejected,
        was_completed,

        -- Timing
        decision_time_seconds,
        completion_time_seconds,

        -- Financial
        actual_payout,
        cost_incurred,

        -- Metadata
        current_timestamp as _loaded_at

    from outcomes
)

select * from final
