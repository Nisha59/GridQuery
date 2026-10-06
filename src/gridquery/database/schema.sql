-- GridQuery DuckDB schema: German electricity data from SMARD (Bundesnetzagentur).
--
-- Time convention
--   * Every timestamp is UTC, stored as TIMESTAMP (no time zone) in a column named *_utc.
--   * A row describes the hour that STARTS at timestamp_utc.
--   * Convert to German local time (CET/CEST, daylight saving handled) with the macro
--       to_berlin_time(timestamp_utc)
--     which is short for
--       (timestamp_utc AT TIME ZONE 'UTC') AT TIME ZONE 'Europe/Berlin'
--   * The *_daily and *_monthly views already group by German local dates.
--
-- Table, view and column descriptions live in descriptions.toml and are applied as
-- COMMENTs by create_schema().
--
-- Units
--   * Energy in MWh per hour. An hourly MWh value equals the average power in MW that hour.
--   * Prices in EUR/MWh (day-ahead auction, bidding zone DE-LU). Since 1 Oct 2025 the auction
--     runs in 15-minute steps; the hourly price is the average of the four quarter-hours.

CREATE MACRO to_berlin_time(ts) AS (ts AT TIME ZONE 'UTC') AT TIME ZONE 'Europe/Berlin';

-- ---------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------

CREATE TABLE energy_sources (
    source          VARCHAR PRIMARY KEY,
    name_en         VARCHAR NOT NULL,
    name_de         VARCHAR NOT NULL,
    is_renewable    BOOLEAN NOT NULL,
    is_fossil       BOOLEAN NOT NULL,
    smard_filter_id INTEGER NOT NULL UNIQUE,
    description     VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Fact tables (one row per hour)
-- ---------------------------------------------------------------------------

CREATE TABLE generation (
    timestamp_utc  TIMESTAMP NOT NULL,
    source         VARCHAR   NOT NULL REFERENCES energy_sources (source),
    generation_mwh DOUBLE    NOT NULL,
    PRIMARY KEY (timestamp_utc, source)
);

CREATE TABLE consumption (
    timestamp_utc   TIMESTAMP PRIMARY KEY,
    consumption_mwh DOUBLE    NOT NULL
);

CREATE TABLE day_ahead_prices (
    timestamp_utc     TIMESTAMP PRIMARY KEY,
    price_eur_per_mwh DOUBLE    NOT NULL
);

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- Everything for one hour in one row: totals, renewable share, consumption, price.
CREATE VIEW hourly_summary AS
WITH generation_totals AS (
    SELECT
        g.timestamp_utc,
        sum(g.generation_mwh)                                AS total_generation_mwh,
        sum(g.generation_mwh) FILTER (WHERE s.is_renewable)  AS renewable_generation_mwh,
        sum(g.generation_mwh) FILTER (WHERE s.is_fossil)     AS fossil_generation_mwh
    FROM generation AS g
    JOIN energy_sources AS s USING (source)
    GROUP BY g.timestamp_utc
),
hours AS (
    SELECT timestamp_utc FROM generation_totals
    UNION SELECT timestamp_utc FROM consumption
    UNION SELECT timestamp_utc FROM day_ahead_prices
)
SELECT
    h.timestamp_utc,
    to_berlin_time(h.timestamp_utc)                                        AS timestamp_local,
    g.total_generation_mwh,
    g.renewable_generation_mwh,
    g.fossil_generation_mwh,
    g.renewable_generation_mwh / nullif(g.total_generation_mwh, 0)         AS renewable_share_of_generation,
    c.consumption_mwh,
    p.price_eur_per_mwh
FROM hours AS h
LEFT JOIN generation_totals AS g USING (timestamp_utc)
LEFT JOIN consumption AS c USING (timestamp_utc)
LEFT JOIN day_ahead_prices AS p USING (timestamp_utc);

CREATE VIEW generation_daily AS
SELECT
    CAST(to_berlin_time(timestamp_utc) AS DATE) AS date_local,
    source,
    sum(generation_mwh)                         AS generation_mwh,
    count(*)                                    AS hours
FROM generation
GROUP BY ALL;

CREATE VIEW generation_monthly AS
SELECT
    CAST(date_trunc('month', to_berlin_time(timestamp_utc)) AS DATE) AS month_local,
    source,
    sum(generation_mwh)                                              AS generation_mwh,
    count(*)                                                         AS hours
FROM generation
GROUP BY ALL;

CREATE VIEW daily_summary AS
SELECT
    CAST(timestamp_local AS DATE)                                  AS date_local,
    sum(total_generation_mwh)                                      AS total_generation_mwh,
    sum(renewable_generation_mwh)                                  AS renewable_generation_mwh,
    sum(fossil_generation_mwh)                                     AS fossil_generation_mwh,
    sum(renewable_generation_mwh) / nullif(sum(total_generation_mwh), 0) AS renewable_share_of_generation,
    sum(consumption_mwh)                                           AS consumption_mwh,
    avg(price_eur_per_mwh)                                         AS avg_price_eur_per_mwh,
    min(price_eur_per_mwh)                                         AS min_price_eur_per_mwh,
    max(price_eur_per_mwh)                                         AS max_price_eur_per_mwh,
    count(*) FILTER (WHERE price_eur_per_mwh < 0)                  AS negative_price_hours,
    count(consumption_mwh)                                         AS hours
FROM hourly_summary
GROUP BY ALL;

CREATE VIEW monthly_summary AS
SELECT
    CAST(date_trunc('month', timestamp_local) AS DATE)             AS month_local,
    sum(total_generation_mwh)                                      AS total_generation_mwh,
    sum(renewable_generation_mwh)                                  AS renewable_generation_mwh,
    sum(fossil_generation_mwh)                                     AS fossil_generation_mwh,
    sum(renewable_generation_mwh) / nullif(sum(total_generation_mwh), 0) AS renewable_share_of_generation,
    sum(consumption_mwh)                                           AS consumption_mwh,
    avg(price_eur_per_mwh)                                         AS avg_price_eur_per_mwh,
    min(price_eur_per_mwh)                                         AS min_price_eur_per_mwh,
    max(price_eur_per_mwh)                                         AS max_price_eur_per_mwh,
    count(*) FILTER (WHERE price_eur_per_mwh < 0)                  AS negative_price_hours,
    count(consumption_mwh)                                         AS hours
FROM hourly_summary
GROUP BY ALL;

-- What time range each series covers, so questions outside it can be refused.
CREATE VIEW data_coverage AS
SELECT 'generation' AS table_name, source AS series,
       min(timestamp_utc) AS first_timestamp_utc, max(timestamp_utc) AS last_timestamp_utc,
       count(*) AS row_count
FROM generation GROUP BY source
UNION ALL
SELECT 'consumption', 'consumption', min(timestamp_utc), max(timestamp_utc), count(*)
FROM consumption
UNION ALL
SELECT 'day_ahead_prices', 'day_ahead_price', min(timestamp_utc), max(timestamp_utc), count(*)
FROM day_ahead_prices;

