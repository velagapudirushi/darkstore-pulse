"""DuckDB analytical SQL engine for dark-store forensics and leak-free ML feature engineering."""

from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd

from darkstore_pulse.config import SEED_PARQUET_PATH


class DarkStoreWarehouseSQL:
    """Analytical SQL interface directly querying Parquet observations via DuckDB."""

    def __init__(self, parquet_path: Path | None = None) -> None:
        self.parquet_path = Path(parquet_path or SEED_PARQUET_PATH)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")
        escaped = str(self.parquet_path).replace("'", "''")
        con.execute(f"CREATE VIEW obs AS SELECT * FROM read_parquet('{escaped}')")
        return con

    def run_custom_query(self, sql: str) -> pd.DataFrame:
        """Execute a read-only analytical SQL query against the `obs` view."""
        stripped = sql.strip().lower()
        if not (stripped.startswith("select") or stripped.startswith("with")):
            raise ValueError("Only read-only SELECT / WITH queries are allowed in the SQL explorer.")
        with self._connect() as con:
            return con.execute(sql).df()

    def infer_restock_schedules(self) -> pd.DataFrame:
        """Reverse-engineer hidden warehouse supply-truck arrival hours using SQL LAG().

        Whenever an item flips from `in_stock = FALSE` at hour t-1 to `in_stock = TRUE` at hour t,
        a physical restock event occurred in that neighborhood dark store.
        """
        query = """
        WITH transitions AS (
            SELECT
                pincode,
                neighborhood,
                zone_type,
                item_id,
                timestamp,
                EXTRACT(HOUR FROM timestamp)::INT AS hour_of_day,
                in_stock,
                LAG(in_stock) OVER (
                    PARTITION BY pincode, item_id ORDER BY timestamp
                ) AS prev_in_stock
            FROM obs
        ),
        restock_flips AS (
            SELECT
                pincode,
                neighborhood,
                zone_type,
                hour_of_day,
                COUNT(*) AS restock_flip_count
            FROM transitions
            WHERE prev_in_stock = FALSE AND in_stock = TRUE
            GROUP BY 1, 2, 3, 4
        ),
        ranked AS (
            SELECT
                *,
                ROUND(
                    100.0 * restock_flip_count
                    / NULLIF(SUM(restock_flip_count) OVER (PARTITION BY pincode), 0),
                    1
                ) AS pct_of_store_restocks,
                ROW_NUMBER() OVER (
                    PARTITION BY pincode ORDER BY restock_flip_count DESC, hour_of_day ASC
                ) AS rk
            FROM restock_flips
        )
        SELECT
            pincode,
            neighborhood,
            zone_type,
            hour_of_day,
            restock_flip_count,
            pct_of_store_restocks,
            rk AS rank_in_store
        FROM ranked
        ORDER BY pincode, hour_of_day
        """
        with self._connect() as con:
            return con.execute(query).df()

    def get_top_restock_windows_summary(self) -> pd.DataFrame:
        """Return the top morning and afternoon inferred supply-truck hours per dark store."""
        query = """
        WITH flips AS (
            SELECT
                pincode,
                neighborhood,
                zone_type,
                EXTRACT(HOUR FROM timestamp)::INT AS hour_of_day,
                in_stock,
                LAG(in_stock) OVER (PARTITION BY pincode, item_id ORDER BY timestamp) AS prev_stock
            FROM obs
        ),
        agg AS (
            SELECT
                pincode,
                neighborhood,
                zone_type,
                hour_of_day,
                COUNT(*) AS flips_count
            FROM flips
            WHERE prev_stock = FALSE AND in_stock = TRUE
            GROUP BY 1, 2, 3, 4
        ),
        morning AS (
            SELECT pincode, hour_of_day AS inferred_morning_truck_hr, flips_count AS morning_flips
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY pincode ORDER BY flips_count DESC) AS rn
                FROM agg WHERE hour_of_day < 12
            ) WHERE rn = 1
        ),
        afternoon AS (
            SELECT pincode, hour_of_day AS inferred_afternoon_truck_hr, flips_count AS afternoon_flips
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY pincode ORDER BY flips_count DESC) AS rn
                FROM agg WHERE hour_of_day >= 12
            ) WHERE rn = 1
        )
        SELECT
            p.pincode,
            p.neighborhood,
            p.zone_type,
            m.inferred_morning_truck_hr,
            m.morning_flips,
            a.inferred_afternoon_truck_hr,
            a.afternoon_flips
        FROM (SELECT DISTINCT pincode, neighborhood, zone_type FROM obs) p
        LEFT JOIN morning m USING (pincode)
        LEFT JOIN afternoon a USING (pincode)
        ORDER BY p.zone_type DESC, p.pincode
        """
        with self._connect() as con:
            return con.execute(query).df()

    def get_zone_archetype_comparison(self) -> pd.DataFrame:
        """Compare stockout rates across IT_Hub vs Family_Residential by category and daypart."""
        query = """
        SELECT
            zone_type,
            category,
            CASE
                WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 6 AND 10 THEN '1_Morning (06-10)'
                WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 11 AND 16 THEN '2_Afternoon (11-16)'
                WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 17 AND 21 THEN '3_Evening (17-21)'
                ELSE '4_LateNight (22-05)'
            END AS daypart,
            COUNT(*) AS total_checks,
            ROUND(100.0 * AVG(CASE WHEN NOT in_stock THEN 1.0 ELSE 0.0 END), 2) AS stockout_rate_pct,
            ROUND(AVG(eta_mins), 1) AS avg_eta_mins
        FROM obs
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
        with self._connect() as con:
            return con.execute(query).df()

    def get_hourly_heatmap(self, zone_type: str | None = None) -> pd.DataFrame:
        """Return hour-of-day × category stockout rate & mean ETA."""
        where_clause = ""
        if zone_type in ("IT_Hub", "Family_Residential"):
            where_clause = f"WHERE zone_type = '{zone_type}'"
        query = f"""
        SELECT
            EXTRACT(HOUR FROM timestamp)::INT AS hour_of_day,
            category,
            ROUND(100.0 * AVG(CASE WHEN NOT in_stock THEN 1.0 ELSE 0.0 END), 2) AS stockout_rate_pct,
            ROUND(AVG(eta_mins), 2) AS avg_eta_mins
        FROM obs
        {where_clause}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
        with self._connect() as con:
            return con.execute(query).df()

    def build_ml_feature_table(self) -> pd.DataFrame:
        """Build leak-free ML dataset for items currently IN STOCK.

        Target (`will_stockout_next_2h`):
            1 if an item that is currently `in_stock = TRUE` at hour t becomes
            `in_stock = FALSE` at hour t+1 OR hour t+2; else 0.

        All predictors use strictly current (t) or past (<= t-1) observations.
        """
        query = """
        WITH store_stress AS (
            SELECT
                timestamp,
                pincode,
                AVG(CASE WHEN NOT in_stock THEN 1.0 ELSE 0.0 END) AS warehouse_oos_share_now,
                AVG(eta_mins) AS store_mean_eta_now
            FROM obs
            GROUP BY 1, 2
        ),
        sequenced AS (
            SELECT
                o.timestamp,
                o.pincode,
                o.neighborhood,
                o.zone_type,
                o.item_id,
                o.item_name,
                o.category,
                o.price_inr,
                o.in_stock,
                o.eta_mins,
                o.surge_fee_inr,
                o.rain_mm,
                o.temp_c,
                s.warehouse_oos_share_now,
                EXTRACT(HOUR FROM o.timestamp)::INT AS hour_of_day,
                EXTRACT(DOW FROM o.timestamp)::INT AS day_of_week,
                CASE WHEN EXTRACT(DOW FROM o.timestamp) IN (0, 5, 6) THEN 1 ELSE 0 END AS is_weekend,
                COALESCE(
                    LAG(o.eta_mins, 1) OVER (PARTITION BY o.pincode, o.item_id ORDER BY o.timestamp),
                    o.eta_mins
                ) AS eta_lag_1h,
                AVG(o.eta_mins) OVER (
                    PARTITION BY o.pincode, o.item_id
                    ORDER BY o.timestamp
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS eta_rolling_3h,
                CASE
                    WHEN LAG(o.in_stock, 1) OVER (PARTITION BY o.pincode, o.item_id ORDER BY o.timestamp) = FALSE
                         AND o.in_stock = TRUE
                    THEN 1 ELSE 0
                END AS restock_just_happened,
                AVG(CASE WHEN NOT o.in_stock THEN 1.0 ELSE 0.0 END) OVER (
                    PARTITION BY o.pincode, o.item_id
                    ORDER BY o.timestamp
                    ROWS BETWEEN 48 PRECEDING AND 1 PRECEDING
                ) AS item_past_48h_oos_rate,
                LEAD(o.in_stock, 1) OVER (PARTITION BY o.pincode, o.item_id ORDER BY o.timestamp) AS next_1h_in_stock,
                LEAD(o.in_stock, 2) OVER (PARTITION BY o.pincode, o.item_id ORDER BY o.timestamp) AS next_2h_in_stock
            FROM obs o
            JOIN store_stress s
              ON o.timestamp = s.timestamp AND o.pincode = s.pincode
        ),
        restock_groups AS (
            SELECT
                *,
                SUM(restock_just_happened) OVER (
                    PARTITION BY pincode, item_id ORDER BY timestamp
                ) AS restock_epoch
            FROM sequenced
        ),
        enriched AS (
            SELECT
                *,
                (eta_mins - eta_lag_1h) AS eta_delta_1h,
                ROW_NUMBER() OVER (
                    PARTITION BY pincode, item_id, restock_epoch ORDER BY timestamp
                ) - 1 AS hours_since_last_restock,
                COALESCE(item_past_48h_oos_rate, 0.10) AS item_hist_oos_rate
            FROM restock_groups
        )
        SELECT
            timestamp,
            pincode,
            neighborhood,
            zone_type,
            item_id,
            item_name,
            category,
            price_inr,
            eta_mins,
            eta_delta_1h,
            ROUND(eta_rolling_3h, 2) AS eta_rolling_3h,
            ROUND(warehouse_oos_share_now, 4) AS warehouse_oos_share_now,
            hours_since_last_restock,
            ROUND(item_hist_oos_rate, 4) AS item_hist_oos_rate,
            surge_fee_inr,
            rain_mm,
            temp_c,
            hour_of_day,
            day_of_week,
            is_weekend,
            CASE
                WHEN (NOT next_1h_in_stock) OR (NOT next_2h_in_stock) THEN 1
                ELSE 0
            END AS will_stockout_next_2h
        FROM enriched
        WHERE in_stock = TRUE
          AND next_1h_in_stock IS NOT NULL
          AND next_2h_in_stock IS NOT NULL
        ORDER BY timestamp, pincode, item_id
        """
        with self._connect() as con:
            return con.execute(query).df()
