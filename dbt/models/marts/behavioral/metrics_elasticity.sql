{{
    config(
        materialized='table'
    )
}}

/*
    Metrics table for incentive elasticity estimation.

    Computes carpool participation rates at different incentive levels
    for elasticity curve plotting.
*/

with choices as (
    select * from {{ ref('fct_mode_choices') }}
),

-- Bin by incentive level
by_incentive_bin as (
    select
        incentive_bucket,

        -- Sample size
        count(*) as n_trips,

        -- Participation rate
        avg(case when is_carpool then 1.0 else 0.0 end) as carpool_rate,
        stddev(case when is_carpool then 1.0 else 0.0 end) as carpool_rate_std,

        -- Average incentive in bin
        avg(incentive_amount_usd) as avg_incentive,
        min(incentive_amount_usd) as min_incentive,
        max(incentive_amount_usd) as max_incentive,

        -- Average participants
        avg(participant_count) as avg_participants

    from choices
    group by incentive_bucket
),

-- Compute elasticity approximation
with_elasticity as (
    select
        *,
        -- Simple arc elasticity between bins
        case
            when lag(carpool_rate) over (order by avg_incentive) > 0
                 and lag(avg_incentive) over (order by avg_incentive) > 0
            then (
                (carpool_rate - lag(carpool_rate) over (order by avg_incentive))
                / lag(carpool_rate) over (order by avg_incentive)
            ) / (
                (avg_incentive - lag(avg_incentive) over (order by avg_incentive))
                / lag(avg_incentive) over (order by avg_incentive)
            )
            else null
        end as arc_elasticity

    from by_incentive_bin
)

select
    *,
    -- 95% CI for rate (normal approximation)
    carpool_rate - 1.96 * carpool_rate_std / sqrt(n_trips) as carpool_rate_ci_low,
    carpool_rate + 1.96 * carpool_rate_std / sqrt(n_trips) as carpool_rate_ci_high,
    current_timestamp as _loaded_at
from with_elasticity
