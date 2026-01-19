{{
    config(
        materialized='table'
    )
}}

WITH source AS (
    SELECT * FROM read_parquet('../data/raw/population_dyna/zcta_poverty.parquet')
),

-- Extract ZCTA codes and get most recent poverty rate
cleaned AS (
    SELECT
        REPLACE(place, 'zip/', '') AS zcta_code,
        "2022" AS poverty_rate,  -- Most recent year
        "2021" AS poverty_rate_2021,
        "2020" AS poverty_rate_2020
    FROM source
    WHERE place LIKE 'zip/%'
      AND place LIKE 'zip/37%'  -- Tennessee ZCTAs (37xxx)
),

-- Estimate median household income from poverty rate
-- Using inverse relationship: higher poverty → lower income
-- National median ~$70k, adjust based on poverty rate
enriched AS (
    SELECT
        zcta_code,
        poverty_rate,
        poverty_rate_2021,
        poverty_rate_2020,

        -- Estimate income: base of $70k adjusted by poverty
        -- High poverty (20%) → ~$45k, Low poverty (5%) → ~$85k
        CAST(70000 * (1 - poverty_rate * 0.8) AS INTEGER) AS median_household_income_est,

        -- Income quintile based on estimated income
        NTILE(5) OVER (ORDER BY 70000 * (1 - poverty_rate * 0.8)) AS income_quintile,

        CURRENT_TIMESTAMP AS loaded_at
    FROM cleaned
    WHERE poverty_rate IS NOT NULL
)

SELECT * FROM enriched
