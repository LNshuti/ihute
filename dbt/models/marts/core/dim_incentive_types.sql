{{
    config(
        materialized='table'
    )
}}

/*
    Dimension table for incentive type reference.
*/

select 'CARPOOL' as incentive_type, 'Carpooling' as incentive_name, 'Rewards for sharing rides' as description, 2.50 as typical_reward_usd
union all
select 'PACER', 'Pacer Driving', 'Rewards for maintaining steady speeds', 0.15
union all
select 'DEPARTURE_SHIFT', 'Departure Time Shift', 'Rewards for shifting departure from peak', 3.00
union all
select 'TRANSIT', 'Transit Promotion', 'Rewards for using public transit', 2.00
