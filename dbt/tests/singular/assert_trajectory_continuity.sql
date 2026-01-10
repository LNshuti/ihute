/*
    Test that trajectory timestamps are monotonically increasing
    within each vehicle track.
*/

with time_gaps as (
    select
        object_id,
        timestamp_sec,
        lag(timestamp_sec) over (
            partition by object_id
            order by timestamp_sec
        ) as prev_timestamp
    from {{ ref('stg_laddms__trajectories') }}
)

select
    object_id,
    prev_timestamp,
    timestamp_sec,
    timestamp_sec - prev_timestamp as time_gap
from time_gaps
where prev_timestamp is not null
  and timestamp_sec < prev_timestamp  -- Timestamp went backwards
