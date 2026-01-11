"""
DuckDB database connection manager for the Gradio dashboard.
"""

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class Database:
    """Manages DuckDB connection and query execution."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to DuckDB file. Defaults to warehouse.duckdb in project root.
        """
        if db_path is None:
            # Look for database in standard locations
            possible_paths = [
                Path(__file__).parent.parent / "warehouse.duckdb",
                Path("warehouse.duckdb"),
                Path("../warehouse.duckdb"),
            ]
            for path in possible_paths:
                if path.exists():
                    db_path = str(path)
                    break

        self.db_path = db_path
        self._connection: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._connection is None:
            if self.db_path and Path(self.db_path).exists():
                try:
                    self._connection = duckdb.connect(self.db_path, read_only=True)
                except Exception as e:
                    print(f"Warning: Could not connect to {self.db_path}: {e}")
                    print("Falling back to in-memory database with sample data.")
                    self._connection = duckdb.connect(":memory:")
                    self._create_sample_data()
            else:
                # Create in-memory database with sample data for demo
                self._connection = duckdb.connect(":memory:")
                self._create_sample_data()
        return self._connection

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        return self.connection.execute(sql).fetchdf()

    def _create_sample_data(self) -> None:
        """Create sample data for demo when no database file exists."""
        conn = self._connection

        # Create schema
        conn.execute("CREATE SCHEMA IF NOT EXISTS main_marts")

        # Create sample corridor flows
        conn.execute("""
            CREATE TABLE main_marts.fct_corridor_flows AS
            SELECT
                'I-24' as corridor_id,
                'I-24 Main' as zone_name,
                timestamp '2024-01-15 07:00:00' + interval (i) hour as hour_bucket,
                CASE
                    WHEN (i % 24) BETWEEN 7 AND 9 THEN 'AM_PEAK'
                    WHEN (i % 24) BETWEEN 17 AND 19 THEN 'PM_PEAK'
                    ELSE 'OFF_PEAK'
                END as time_period,
                (random() * 500 + 200)::int as vehicle_count,
                CASE
                    WHEN (i % 24) BETWEEN 7 AND 9 THEN 25 + random() * 15
                    WHEN (i % 24) BETWEEN 17 AND 19 THEN 20 + random() * 15
                    ELSE 55 + random() * 10
                END as avg_speed_mph,
                CASE
                    WHEN (i % 24) BETWEEN 7 AND 9 THEN 'D'
                    WHEN (i % 24) BETWEEN 17 AND 19 THEN 'E'
                    ELSE 'B'
                END as level_of_service
            FROM generate_series(0, 167) as t(i)
        """)

        # Create sample incentive events
        conn.execute("""
            CREATE TABLE main_marts.fct_incentive_events AS
            SELECT
                'alloc_' || i as incentive_key,
                'agent_' || (random() * 1000)::int as agent_id,
                'run_001' as simulation_run_id,
                CASE (i % 4)
                    WHEN 0 THEN 'CARPOOL'
                    WHEN 1 THEN 'PACER'
                    WHEN 2 THEN 'DEPARTURE_SHIFT'
                    ELSE 'TRANSIT'
                END as incentive_type,
                2.0 + random() * 8 as offered_amount,
                CASE WHEN random() > 0.3 THEN true ELSE false END as was_accepted,
                CASE WHEN random() > 0.5 THEN true ELSE false END as was_completed,
                CASE
                    WHEN random() > 0.5 THEN 'COMPLETED'
                    WHEN random() > 0.3 THEN 'ACCEPTED_PENDING'
                    ELSE 'REJECTED'
                END as final_outcome,
                (random() * 5)::decimal(10,2) as actual_payout
            FROM generate_series(1, 500) as t(i)
        """)

        # Create sample elasticity metrics
        conn.execute("""
            CREATE TABLE main_marts.metrics_elasticity AS
            SELECT
                bucket as incentive_bucket,
                (100 + bucket * 50) as n_trips,
                0.1 + bucket * 0.08 as carpool_rate,
                bucket * 1.5 as avg_incentive
            FROM (
                SELECT unnest(['NONE', 'LOW', 'MEDIUM', 'HIGH']) as bucket,
                       unnest([0, 1, 2, 3]) as idx
            ) t
            ORDER BY idx
        """)

        # Create sample scenario comparison
        conn.execute("""
            CREATE TABLE main_marts.fct_simulation_runs AS
            SELECT
                'run_' || i as run_key,
                CASE (i % 3)
                    WHEN 0 THEN 'Carpool Incentive'
                    WHEN 1 THEN 'Pacer Program'
                    ELSE 'Transit Promotion'
                END as scenario_name,
                10000 as n_agents,
                45 + random() * 10 as treatment_avg_speed,
                42.0 as baseline_avg_speed,
                (45 + random() * 10 - 42) / 42 * 100 as speed_improvement_pct,
                5 + random() * 15 as vmt_reduction_pct,
                3 + random() * 7 as peak_reduction_pct,
                5000 + random() * 5000 as treatment_spend
            FROM generate_series(1, 10) as t(i)
        """)


# Global database instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def query(sql: str) -> pd.DataFrame:
    """Execute SQL query using global database."""
    return get_database().query(sql)
