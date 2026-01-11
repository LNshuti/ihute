{{
    config(
        materialized='view'
    )
}}

/*
    Intermediate model tracking incentive lifecycle and outcomes.

    Joins offers with their eventual outcomes (accept/reject/complete)
    for conversion funnel analysis.
*/

with events as (
    select * from {{ ref('stg_sim__incentive_events') }}
),

-- Pivot events by allocation_id to get lifecycle view
offers as (
    select
        allocation_id,
        agent_id,
        simulation_run_id,
        incentive_type,
        amount_usd as offered_amount,
        event_at as offered_at,
        hour_of_day as offer_hour
    from events
    where is_offer
),

accepts as (
    select
        allocation_id,
        event_at as accepted_at,
        amount_usd as accepted_amount
    from events
    where is_accept
),

rejects as (
    select
        allocation_id,
        event_at as rejected_at
    from events
    where is_reject
),

completions as (
    select
        allocation_id,
        event_at as completed_at,
        amount_usd as completed_amount,
        performance_metrics
    from events
    where is_complete
),

expirations as (
    select
        allocation_id,
        event_at as expired_at
    from events
    where is_expire
),

-- Join all stages
lifecycle as (
    select
        o.allocation_id,
        o.agent_id,
        o.simulation_run_id,
        o.incentive_type,
        o.offered_amount,
        o.offered_at,
        o.offer_hour,

        -- Accept stage
        a.accepted_at,
        a.accepted_amount,
        case when a.allocation_id is not null then true else false end as was_accepted,

        -- Reject stage
        r.rejected_at,
        case when r.allocation_id is not null then true else false end as was_rejected,

        -- Completion stage
        c.completed_at,
        c.completed_amount,
        c.performance_metrics,
        case when c.allocation_id is not null then true else false end as was_completed,

        -- Expiration stage
        e.expired_at,
        case when e.allocation_id is not null then true else false end as was_expired,

        -- Time to decision
        extract(epoch from (coalesce(a.accepted_at, r.rejected_at) - o.offered_at)) as decision_time_seconds,

        -- Time to completion
        extract(epoch from (c.completed_at - a.accepted_at)) as completion_time_seconds

    from offers o
    left join accepts a on o.allocation_id = a.allocation_id
    left join rejects r on o.allocation_id = r.allocation_id
    left join completions c on o.allocation_id = c.allocation_id
    left join expirations e on o.allocation_id = e.allocation_id
),

-- Add outcome classification
final as (
    select
        *,

        -- Final outcome
        case
            when was_completed then 'COMPLETED'
            when was_expired then 'EXPIRED'
            when was_rejected then 'REJECTED'
            when was_accepted then 'ACCEPTED_PENDING'
            else 'OFFERED_PENDING'
        end as final_outcome,

        -- Successful conversion
        case when was_completed then true else false end as is_successful,

        -- Actual payout
        coalesce(completed_amount, 0) as actual_payout,

        -- ROI (would need to join with trip savings)
        case
            when completed_amount > 0 then completed_amount
            else 0
        end as cost_incurred

    from lifecycle
)

select * from final
