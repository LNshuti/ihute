"""Tests for database connection and query execution."""

import pytest
import pandas as pd

from ..database import Database, get_database, query


class TestDatabase:
    """Test database connection functionality."""

    def test_database_initialization(self):
        """Test database can be initialized."""
        db = Database()
        assert db is not None

    def test_in_memory_connection(self):
        """Test in-memory database connection works."""
        db = Database(db_path=None)
        conn = db.connection
        assert conn is not None

    def test_query_returns_dataframe(self):
        """Test that queries return pandas DataFrames."""
        db = Database()
        result = db.query("SELECT 1 as value")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result['value'].iloc[0] == 1

    def test_sample_data_creation(self):
        """Test sample data is created for demo mode."""
        db = Database(db_path=None)
        # Force sample data creation
        _ = db.connection

        # Check tables exist
        result = db.query("SELECT * FROM marts.fct_corridor_flows LIMIT 1")
        assert not result.empty

    def test_global_database_singleton(self):
        """Test global database returns same instance."""
        db1 = get_database()
        db2 = get_database()
        # Should be same connection pool
        assert db1 is db2

    def test_query_helper_function(self):
        """Test the query helper function works."""
        result = query("SELECT 42 as answer")
        assert isinstance(result, pd.DataFrame)
        assert result['answer'].iloc[0] == 42


class TestDatabaseQueries:
    """Test specific query patterns used by components."""

    @pytest.fixture
    def db(self):
        """Create fresh database for each test."""
        return Database(db_path=None)

    def test_corridor_flows_query(self, db):
        """Test corridor flows query structure."""
        _ = db.connection  # Initialize sample data

        result = db.query("""
            SELECT corridor_id, hour_bucket, avg_speed_mph
            FROM marts.fct_corridor_flows
            LIMIT 10
        """)

        assert 'corridor_id' in result.columns
        assert 'hour_bucket' in result.columns
        assert 'avg_speed_mph' in result.columns

    def test_incentive_events_query(self, db):
        """Test incentive events query structure."""
        _ = db.connection

        result = db.query("""
            SELECT incentive_type, was_accepted, was_completed
            FROM marts.fct_incentive_events
            LIMIT 10
        """)

        assert 'incentive_type' in result.columns
        assert 'was_accepted' in result.columns

    def test_aggregation_query(self, db):
        """Test aggregation queries work correctly."""
        _ = db.connection

        result = db.query("""
            SELECT
                incentive_type,
                count(*) as count,
                avg(offered_amount) as avg_amount
            FROM marts.fct_incentive_events
            GROUP BY incentive_type
        """)

        assert 'count' in result.columns
        assert 'avg_amount' in result.columns
        assert all(result['count'] > 0)
