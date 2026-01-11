{{
    config(
        materialized='table'
    )
}}

/*
    Dimension table for time period reference.

    Pre-generates time buckets for join efficiency.
*/

with hours as (
    select generate_series as hour_of_day
    from generate_series(0, 23)
),

days as (
    select generate_series as day_of_week
    from generate_series(0, 6)
),

time_periods as (
    select
        h.hour_of_day,
        d.day_of_week,

        -- Time period key
        (d.day_of_week * 24 + h.hour_of_day) as time_period_key,

        -- Hour labels
        case
            when h.hour_of_day = 0 then '12 AM'
            when h.hour_of_day < 12 then h.hour_of_day || ' AM'
            when h.hour_of_day = 12 then '12 PM'
            else (h.hour_of_day - 12) || ' PM'
        end as hour_label,

        -- Day labels
        case d.day_of_week
            when 0 then 'Sunday'
            when 1 then 'Monday'
            when 2 then 'Tuesday'
            when 3 then 'Wednesday'
            when 4 then 'Thursday'
            when 5 then 'Friday'
            when 6 then 'Saturday'
        end as day_name,

        -- Peak classification
        case
            when h.hour_of_day between {{ var('am_peak_start') }} and {{ var('am_peak_end') }}
                then 'AM_PEAK'
            when h.hour_of_day between {{ var('pm_peak_start') }} and {{ var('pm_peak_end') }}
                then 'PM_PEAK'
            else 'OFF_PEAK'
        end as time_period,

        -- Weekday flag
        case when d.day_of_week between 1 and 5 then true else false end as is_weekday,

        -- Business hours
        case when h.hour_of_day between 6 and 20 then true else false end as is_daytime

    from hours h
    cross join days d
)

select * from time_periods
