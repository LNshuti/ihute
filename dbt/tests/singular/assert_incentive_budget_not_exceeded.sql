/*
    Test that no simulation run exceeds its budget allocation.
*/

with run_spend as (
    select
        r.simulation_run_id,
        r.budget_usd,
        coalesce(sum(e.actual_payout), 0) as total_spend
    from {{ ref('stg_sim__runs') }} r
    left join {{ ref('fct_incentive_events') }} e
        on r.simulation_run_id = e.simulation_run_id
    where r.budget_usd > 0
    group by r.simulation_run_id, r.budget_usd
)

select
    simulation_run_id,
    budget_usd,
    total_spend,
    total_spend - budget_usd as overspend
from run_spend
where total_spend > budget_usd * 1.01  -- Allow 1% tolerance
