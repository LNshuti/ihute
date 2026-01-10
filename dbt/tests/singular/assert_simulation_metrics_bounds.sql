/*
    Test that simulation metrics are within reasonable bounds.
*/

select
    metric_id,
    simulation_run_id,
    metric_name,
    metric_value
from {{ ref('stg_sim__metrics_timeseries') }}
where
    -- Speed should be 0-120 mph
    (metric_name ilike '%speed%' and (metric_value < 0 or metric_value > 120))
    -- Rates should be 0-1
    or (metric_name ilike '%rate%' and (metric_value < 0 or metric_value > 1))
    -- Counts should be non-negative
    or (metric_name ilike '%count%' and metric_value < 0)
    -- Occupancy should be 1-10
    or (metric_name ilike '%occupancy%' and (metric_value < 1 or metric_value > 10))
