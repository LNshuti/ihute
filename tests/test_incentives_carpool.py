"""
Tests for src/incentives/carpool.py module.

Tests cover:
- CarpoolMatch dataclass
- CarpoolIncentive class
"""

import numpy as np
import pytest

from src.incentives.base import IncentiveConfig, IncentiveType
from src.incentives.carpool import CarpoolIncentive, CarpoolMatch


class TestCarpoolMatch:
    """Tests for CarpoolMatch dataclass."""

    def test_required_fields(self):
        """CarpoolMatch requires core fields."""
        match = CarpoolMatch(
            match_id="carpool_001",
            driver_id="driver_001",
            passenger_ids=["pass_001", "pass_002"],
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
        )

        assert match.match_id == "carpool_001"
        assert match.driver_id == "driver_001"
        assert len(match.passenger_ids) == 2
        assert match.origin == (0.0, 0.0)
        assert match.destination == (10.0, 10.0)

    def test_default_values(self):
        """CarpoolMatch should have sensible defaults."""
        match = CarpoolMatch(
            match_id="carpool_001",
            driver_id="driver_001",
            passenger_ids=["pass_001"],
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
        )

        assert match.expected_detour_minutes == 0.0
        assert match.status == "pending"


class TestCarpoolIncentive:
    """Tests for CarpoolIncentive class."""

    @pytest.fixture
    def config(self):
        """Create a carpool config."""
        return IncentiveConfig(
            incentive_type=IncentiveType.CARPOOL,
            budget_daily=5000.0,
            active_hours=list(range(6, 10)) + list(range(16, 20)),
        )

    @pytest.fixture
    def incentive(self, config):
        """Create a carpool incentive."""
        return CarpoolIncentive(config=config)

    def test_initialization(self, incentive):
        """Incentive should initialize with defaults."""
        assert incentive.reward_per_passenger == 2.50
        assert incentive.min_passengers == 1
        assert incentive.max_reward == 10.00
        assert incentive.driver_bonus == 1.00
        assert incentive.matches == []
        assert incentive.active_drivers == {}
        assert incentive.active_passengers == {}

    def test_check_eligibility_basic(self, incentive):
        """Basic eligibility check."""
        context = {"hour": 8, "day_of_week": 0, "carpool_eligible": True}
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is True
        assert reason == "Eligible"

    def test_check_eligibility_outside_hours(self, incentive):
        """Should reject outside active hours."""
        context = {"hour": 12, "day_of_week": 0}
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is False
        assert "hours" in reason.lower()

    def test_check_eligibility_already_driver(self, incentive):
        """Should reject if already a driver."""
        # Create a match where agent is driver
        incentive.create_match(
            driver_id="agent_001",
            passenger_ids=["agent_002"],
            origin=(0, 0),
            destination=(10, 10),
            departure_time=8 * 3600,
        )

        context = {"hour": 8, "day_of_week": 0}
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is False
        assert "driving" in reason.lower()

    def test_check_eligibility_already_passenger(self, incentive):
        """Should reject if already a passenger."""
        # Create a match where agent is passenger
        incentive.create_match(
            driver_id="agent_001",
            passenger_ids=["agent_002"],
            origin=(0, 0),
            destination=(10, 10),
            departure_time=8 * 3600,
        )

        context = {"hour": 8, "day_of_week": 0}
        eligible, reason = incentive.check_eligibility("agent_002", context)

        assert eligible is False
        assert "passenger" in reason.lower()

    def test_check_eligibility_not_carpool_eligible(self, incentive):
        """Should reject if not carpool eligible."""
        context = {"hour": 8, "day_of_week": 0, "carpool_eligible": False}
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is False
        assert "eligible" in reason.lower()

    def test_compute_reward_basic(self, incentive):
        """Compute basic reward."""
        context = {
            "n_passengers": 2,
            "is_driver": True,
            "distance_miles": 10,
            "hour": 12,  # Off-peak
        }
        reward = incentive.compute_reward("agent_001", context)

        expected = (
            2.50 * 2  # passengers
            + 1.00  # driver bonus
            + 0.10 * 10  # distance
        )
        assert reward == pytest.approx(expected)

    def test_compute_reward_peak_hours(self, incentive):
        """Peak hours should apply multiplier."""
        context = {
            "n_passengers": 1,
            "is_driver": False,
            "distance_miles": 0,
            "hour": 8,  # Peak
        }
        off_peak_context = {**context, "hour": 12}

        peak_reward = incentive.compute_reward("agent_001", context)
        off_peak_reward = incentive.compute_reward("agent_001", off_peak_context)

        assert peak_reward > off_peak_reward

    def test_compute_reward_max_cap(self, incentive):
        """Reward should be capped at max."""
        context = {
            "n_passengers": 10,  # Would exceed max
            "is_driver": True,
            "distance_miles": 50,
            "hour": 8,
        }
        reward = incentive.compute_reward("agent_001", context)

        assert reward == incentive.max_reward

    def test_verify_completion_success(self, incentive):
        """Verify successful completion."""
        from src.incentives.base import IncentiveAllocation

        allocation = IncentiveAllocation(
            allocation_id="test_001",
            agent_id="agent_001",
            incentive_type=IncentiveType.CARPOOL,
            amount=5.0,
            timestamp=1000,
            conditions={},
        )

        outcome = {
            "trip_completed": True,
            "actual_passengers": 2,
            "is_driver": True,
            "actual_distance_miles": 10,
            "completion_hour": 8,
        }

        success, reward = incentive.verify_completion(allocation, outcome)

        assert success is True
        assert reward > 0

    def test_verify_completion_trip_not_completed(self, incentive):
        """Verify failed when trip not completed."""
        from src.incentives.base import IncentiveAllocation

        allocation = IncentiveAllocation(
            allocation_id="test_001",
            agent_id="agent_001",
            incentive_type=IncentiveType.CARPOOL,
            amount=5.0,
            timestamp=1000,
            conditions={},
        )

        outcome = {"trip_completed": False}

        success, reward = incentive.verify_completion(allocation, outcome)

        assert success is False
        assert reward == 0.0

    def test_verify_completion_not_enough_passengers(self, incentive):
        """Verify failed when not enough passengers."""
        from src.incentives.base import IncentiveAllocation

        allocation = IncentiveAllocation(
            allocation_id="test_001",
            agent_id="agent_001",
            incentive_type=IncentiveType.CARPOOL,
            amount=5.0,
            timestamp=1000,
            conditions={},
        )

        outcome = {"trip_completed": True, "actual_passengers": 0}

        success, reward = incentive.verify_completion(allocation, outcome)

        assert success is False
        assert reward == 0.0

    def test_create_match(self, incentive):
        """create_match should register match."""
        match = incentive.create_match(
            driver_id="driver_001",
            passenger_ids=["pass_001", "pass_002"],
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
        )

        assert match.match_id.startswith("carpool_")
        assert len(incentive.matches) == 1
        assert "driver_001" in incentive.active_drivers
        assert "pass_001" in incentive.active_passengers
        assert "pass_002" in incentive.active_passengers

    def test_find_matches_empty(self, incentive):
        """find_matches should return empty with no agents."""
        matches = incentive.find_matches(
            agent_id="agent_001",
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
            available_agents=None,
        )

        assert matches == []

    def test_find_matches_filters_self(self, incentive):
        """find_matches should filter out self."""
        available = [
            {
                "agent_id": "agent_001",
                "origin": (0.0, 0.0),
                "destination": (10.0, 10.0),
                "departure_time": 8 * 3600,
            }
        ]

        matches = incentive.find_matches(
            agent_id="agent_001",
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
            available_agents=available,
        )

        assert len(matches) == 0

    def test_find_matches_compatible(self, incentive):
        """find_matches should find compatible agents."""
        available = [
            {
                "agent_id": "agent_002",
                "origin": (0.01, 0.01),  # Very close
                "destination": (10.01, 10.01),  # Very close
                "departure_time": 8 * 3600 + 60,  # 1 minute later
            },
        ]

        matches = incentive.find_matches(
            agent_id="agent_001",
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
            max_detour_minutes=10.0,
            max_wait_minutes=15.0,
            available_agents=available,
        )

        assert len(matches) == 1
        assert matches[0]["agent_id"] == "agent_002"
        assert "score" in matches[0]

    def test_find_matches_too_far(self, incentive):
        """find_matches should reject distant agents."""
        available = [
            {
                "agent_id": "agent_002",
                "origin": (10.0, 10.0),  # Far away
                "destination": (20.0, 20.0),
                "departure_time": 8 * 3600,
            },
        ]

        matches = incentive.find_matches(
            agent_id="agent_001",
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
            max_detour_minutes=5.0,
            available_agents=available,
        )

        assert len(matches) == 0

    def test_find_matches_sorted_by_score(self, incentive):
        """find_matches should sort by score."""
        available = [
            {
                "agent_id": "agent_far",
                "origin": (0.05, 0.05),
                "destination": (10.05, 10.05),
                "departure_time": 8 * 3600 + 300,
            },
            {
                "agent_id": "agent_close",
                "origin": (0.01, 0.01),
                "destination": (10.01, 10.01),
                "departure_time": 8 * 3600 + 60,
            },
        ]

        matches = incentive.find_matches(
            agent_id="agent_001",
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
            max_detour_minutes=20.0,
            max_wait_minutes=30.0,
            available_agents=available,
        )

        # Should be sorted by score (closer agent first)
        if len(matches) == 2:
            assert matches[0]["score"] >= matches[1]["score"]

    def test_haversine_distance(self, incentive):
        """_haversine_distance should compute reasonable values."""
        # Same point
        dist = incentive._haversine_distance((0.0, 0.0), (0.0, 0.0))
        assert dist == pytest.approx(0.0)

        # Known distance (approximately)
        # 1 degree latitude is about 69 miles
        dist = incentive._haversine_distance((0.0, 0.0), (1.0, 0.0))
        assert 68 < dist < 70

    def test_complete_match_success(self, incentive):
        """complete_match should distribute rewards on success."""
        match = incentive.create_match(
            driver_id="driver_001",
            passenger_ids=["pass_001", "pass_002"],
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
        )

        outcome = {"completed": True, "distance_miles": 10, "hour": 8}
        rewards = incentive.complete_match(match.match_id, outcome)

        assert "driver_001" in rewards
        assert "pass_001" in rewards
        assert "pass_002" in rewards
        assert rewards["driver_001"] > 0
        assert match.status == "completed"

        # Should clean up active tracking
        assert "driver_001" not in incentive.active_drivers
        assert "pass_001" not in incentive.active_passengers

    def test_complete_match_cancelled(self, incentive):
        """complete_match should handle cancellation."""
        match = incentive.create_match(
            driver_id="driver_001",
            passenger_ids=["pass_001"],
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
        )

        outcome = {"completed": False}
        rewards = incentive.complete_match(match.match_id, outcome)

        assert rewards == {}
        assert match.status == "cancelled"

    def test_complete_match_not_found(self, incentive):
        """complete_match should handle unknown match."""
        rewards = incentive.complete_match("nonexistent", {"completed": True})
        assert rewards == {}

    def test_get_statistics(self, incentive):
        """get_statistics should include carpool-specific stats."""
        # Create and complete a match
        match = incentive.create_match(
            driver_id="driver_001",
            passenger_ids=["pass_001", "pass_002"],
            origin=(0.0, 0.0),
            destination=(10.0, 10.0),
            departure_time=8 * 3600,
        )
        incentive.complete_match(match.match_id, {"completed": True})

        stats = incentive.get_statistics()

        assert "total_matches" in stats
        assert "completed_matches" in stats
        assert "active_drivers" in stats
        assert "active_passengers" in stats
        assert "avg_passengers_per_match" in stats
        assert stats["total_matches"] == 1
        assert stats["completed_matches"] == 1
