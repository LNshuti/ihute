{{
    config(
        materialized='table'
    )
}}

/*
    Dimension table for ZCTA-level demographics.

    Source: population-dyna platform
    Geography: Tennessee ZCTAs (Nashville area)

    Behavioral parameters (avg_vot, avg_beta_incentive) will be
    populated by ML calibration scripts after initial build.
*/

SELECT
    zcta_code,
    poverty_rate,
    poverty_rate_2021,
    poverty_rate_2020,
    median_household_income_est,
    income_quintile,

    -- Placeholder for ML-calibrated parameters
    -- These will be populated by scripts/calibrate_demographics.py
    NULL::DOUBLE AS avg_beta_cost,
    NULL::DOUBLE AS avg_beta_incentive,
    NULL::DOUBLE AS avg_vot,  -- Value of time ($/hour)

    loaded_at

FROM {{ ref('stg_zcta_demographics') }}
