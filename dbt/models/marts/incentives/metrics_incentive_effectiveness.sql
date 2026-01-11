{{
    config(
        materialized='table'
    )
}}

/*
    Metrics table for incentive program effectiveness.
*/

with events as (
    select * from {{ ref('fct_incentive_events') }}
),

by_type_and_run as (
    select
        simulation_run_id,
        incentive_type,

        -- Funnel metrics
        count(*) as total_offers,
        sum(case when was_accepted then 1 else 0 end) as total_accepts,
        sum(case when was_completed then 1 else 0 end) as total_completions,

        -- Rates
        avg(case when was_accepted then 1.0 else 0.0 end) as acceptance_rate,
        avg(case when was_completed then 1.0 else 0.0 end) as completion_rate,

        -- Financial
        sum(offered_amount) as total_offered,
        sum(actual_payout) as total_paid,
        avg(actual_payout) as avg_payout,

        -- Timing
        avg(decision_time_seconds) as avg_decision_time_sec,
        avg(completion_time_seconds) as avg_completion_time_sec

    from events
    group by simulation_run_id, incentive_type
)

select
    *,
    -- Conversion through funnel
    case when total_offers > 0 then total_completions::float / total_offers else 0 end as offer_to_completion_rate,
    case when total_accepts > 0 then total_completions::float / total_accepts else 0 end as accept_to_completion_rate,

    -- Budget efficiency
    case when total_offered > 0 then total_paid / total_offered else 0 end as payout_ratio,

    current_timestamp as _loaded_at
from by_type_and_run
