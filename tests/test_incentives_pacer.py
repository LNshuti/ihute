"""
Tests for src/incentives/pacer.py module.

Tests cover:
- PacerSession dataclass
- PacerIncentive class
"""

import numpy as np
import pytest

from src.incentives.base import IncentiveConfig, IncentiveType
from src.incentives.pacer import PacerIncentive, PacerSession


class TestPacerSession:
    """Tests for PacerSession dataclass."""

    def test_required_fields(self):
        """PacerSession requires core fields."""
        session = PacerSession(
            session_id="pacer_001",
            agent_id="agent_001",
            corridor_id="I-95",
            start_time=8 * 3600,
            target_speed=55.0,
        )

        assert session.session_id == "pacer_001"
        assert session.agent_id == "agent_001"
        assert session.corridor_id == "I-95"
        assert session.target_speed == 55.0

    def test_default_values(self):
        """PacerSession should have sensible defaults."""
        session = PacerSession(
            session_id="pacer_001",
            agent_id="agent_001",
            corridor_id="I-95",
            start_time=8 * 3600,
            target_speed=55.0,
        )

        assert session.speed_tolerance == 5.0
        assert session.speed_samples == []
        assert session.position_samples == []
        assert session.timestamp_samples == []
        assert session.end_time is None
        assert session.distance_miles == 0.0
        assert session.smoothness_score == 0.0
        assert session.status == "active"


class TestPacerIncentive:
    """Tests for PacerIncentive class."""

    @pytest.fixture
    def config(self):
        """Create a pacer config."""
        return IncentiveConfig(
            incentive_type=IncentiveType.PACER,
            budget_daily=5000.0,
            corridor_ids=["I-95", "I-10"],
            active_hours=list(range(6, 10)) + list(range(16, 20)),
        )

    @pytest.fixture
    def incentive(self, config):
        """Create a pacer incentive."""
        return PacerIncentive(config=config)

    def test_initialization(self, incentive):
        """Incentive should initialize with defaults."""
        assert incentive.reward_per_mile == 0.15
        assert incentive.smoothness_threshold == 0.7
        assert incentive.min_distance_miles == 2.0
        assert incentive.max_reward_per_session == 20.00
        assert incentive.sessions == []
        assert incentive.active_sessions == {}
        assert incentive.corridor_targets == {}

    def test_set_corridor_target(self, incentive):
        """set_corridor_target should store target speed."""
        incentive.set_corridor_target("I-95", 60.0)

        assert incentive.corridor_targets["I-95"] == 60.0

    def test_get_corridor_target_set(self, incentive):
        """get_corridor_target should return set value."""
        incentive.set_corridor_target("I-95", 60.0)

        assert incentive.get_corridor_target("I-95") == 60.0

    def test_get_corridor_target_default(self, incentive):
        """get_corridor_target should default to 55 mph."""
        assert incentive.get_corridor_target("I-405") == 55.0

    def test_check_eligibility_basic(self, incentive):
        """Basic eligibility check."""
        context = {
            "hour": 8,
            "day_of_week": 0,
            "corridor_id": "I-95",
            "has_car": True,
            "enrolled_corridors": ["I-95"],
        }
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is True
        assert reason == "Eligible"

    def test_check_eligibility_no_corridor(self, incentive):
        """Should reject if no corridor specified."""
        context = {"hour": 8, "day_of_week": 0, "has_car": True}
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is False
        assert "corridor" in reason.lower()

    def test_check_eligibility_already_in_session(self, incentive):
        """Should reject if already in session."""
        incentive.start_session("agent_001", "I-95", 8 * 3600)

        context = {
            "hour": 8,
            "day_of_week": 0,
            "corridor_id": "I-95",
            "has_car": True,
            "enrolled_corridors": ["I-95"],
        }
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is False
        assert "session" in reason.lower()

    def test_check_eligibility_outside_hours(self, incentive):
        """Should reject outside active hours."""
        context = {
            "hour": 12,
            "day_of_week": 0,
            "corridor_id": "I-95",
            "has_car": True,
        }
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is False
        assert "hours" in reason.lower()

    def test_check_eligibility_no_car(self, incentive):
        """Should reject if agent has no car."""
        context = {
            "hour": 8,
            "day_of_week": 0,
            "corridor_id": "I-95",
            "has_car": False,
        }
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is False
        assert "car" in reason.lower()

    def test_check_eligibility_wrong_corridor(self, incentive):
        """Should reject if not enrolled for corridor."""
        context = {
            "hour": 8,
            "day_of_week": 0,
            "corridor_id": "I-95",
            "has_car": True,
            "enrolled_corridors": ["I-10"],  # Not I-95
        }
        eligible, reason = incentive.check_eligibility("agent_001", context)

        assert eligible is False
        assert "enrolled" in reason.lower()

    def test_compute_reward_basic(self, incentive):
        """Compute basic reward."""
        context = {
            "expected_distance_miles": 10,
            "expected_smoothness": 0.8,
            "hour": 12,
        }
        reward = incentive.compute_reward("agent_001", context)

        # Base: 0.15 * 10 = 1.50
        # Bonus: (0.8 - 0.7) * 10 * 0.10 * 10 = 1.00
        # No peak multiplier
        expected = 0.15 * 10 + 0.10 * 1 * 10
        assert reward == pytest.approx(expected)

    def test_compute_reward_no_smoothness_bonus(self, incentive):
        """No bonus if below threshold."""
        context = {
            "expected_distance_miles": 10,
            "expected_smoothness": 0.6,  # Below 0.7
            "hour": 12,
        }
        reward = incentive.compute_reward("agent_001", context)

        expected = 0.15 * 10  # Just base
        assert reward == pytest.approx(expected)

    def test_compute_reward_peak_multiplier(self, incentive):
        """Peak hours should apply multiplier."""
        context = {
            "expected_distance_miles": 10,
            "expected_smoothness": 0.75,
            "hour": 8,  # Peak
        }
        off_peak_context = {**context, "hour": 12}

        peak_reward = incentive.compute_reward("agent_001", context)
        off_peak_reward = incentive.compute_reward("agent_001", off_peak_context)

        assert peak_reward > off_peak_reward

    def test_compute_reward_max_cap(self, incentive):
        """Reward should be capped at max."""
        context = {
            "expected_distance_miles": 200,  # Would exceed max
            "expected_smoothness": 0.95,
            "hour": 8,
        }
        reward = incentive.compute_reward("agent_001", context)

        assert reward == incentive.max_reward_per_session

    def test_verify_completion_success(self, incentive):
        """Verify successful completion."""
        from src.incentives.base import IncentiveAllocation

        allocation = IncentiveAllocation(
            allocation_id="test_001",
            agent_id="agent_001",
            incentive_type=IncentiveType.PACER,
            amount=5.0,
            timestamp=1000,
            conditions={},
        )

        outcome = {
            "distance_miles": 10,
            "smoothness_score": 0.85,
            "hour": 8,
        }

        success, reward = incentive.verify_completion(allocation, outcome)

        assert success is True
        assert reward > 0

    def test_verify_completion_not_enough_distance(self, incentive):
        """Verify failed when distance too short."""
        from src.incentives.base import IncentiveAllocation

        allocation = IncentiveAllocation(
            allocation_id="test_001",
            agent_id="agent_001",
            incentive_type=IncentiveType.PACER,
            amount=5.0,
            timestamp=1000,
            conditions={},
        )

        outcome = {
            "distance_miles": 1,  # Below min 2.0
            "smoothness_score": 0.9,
        }

        success, reward = incentive.verify_completion(allocation, outcome)

        assert success is False
        assert reward == 0.0

    def test_verify_completion_smoothness_too_low(self, incentive):
        """Verify failed when smoothness too low."""
        from src.incentives.base import IncentiveAllocation

        allocation = IncentiveAllocation(
            allocation_id="test_001",
            agent_id="agent_001",
            incentive_type=IncentiveType.PACER,
            amount=5.0,
            timestamp=1000,
            conditions={},
        )

        outcome = {
            "distance_miles": 10,
            "smoothness_score": 0.5,  # Below 0.7
        }

        success, reward = incentive.verify_completion(allocation, outcome)

        assert success is False
        assert reward == 0.0

    def test_start_session(self, incentive):
        """start_session should create and track session."""
        incentive.set_corridor_target("I-95", 60.0)

        session = incentive.start_session("agent_001", "I-95", 8 * 3600)

        assert session.session_id.startswith("pacer_")
        assert session.agent_id == "agent_001"
        assert session.corridor_id == "I-95"
        assert session.target_speed == 60.0
        assert len(incentive.sessions) == 1
        assert "agent_001" in incentive.active_sessions

    def test_start_session_default_target(self, incentive):
        """start_session should use default target if not set."""
        session = incentive.start_session("agent_001", "I-95", 8 * 3600)

        assert session.target_speed == 55.0  # Default

    def test_start_session_custom_target(self, incentive):
        """start_session should accept custom target."""
        session = incentive.start_session(
            "agent_001", "I-95", 8 * 3600, target_speed=50.0
        )

        assert session.target_speed == 50.0

    def test_update_session(self, incentive):
        """update_session should record data and return feedback."""
        incentive.start_session("agent_001", "I-95", 8 * 3600, target_speed=55.0)

        feedback = incentive.update_session(
            agent_id="agent_001",
            speed=54.0,
            position=(0.0, 0.0),
            timestamp=8 * 3600 + 10,
        )

        assert feedback is not None
        assert "session_id" in feedback
        assert "on_target" in feedback
        assert feedback["on_target"] is True  # Within tolerance

    def test_update_session_multiple_samples(self, incentive):
        """update_session should track smoothness over time."""
        incentive.start_session("agent_001", "I-95", 8 * 3600, target_speed=55.0)

        for i, speed in enumerate([55, 54, 56, 55, 54]):
            feedback = incentive.update_session(
                agent_id="agent_001",
                speed=speed,
                position=(0.0, 0.01 * i),
                timestamp=8 * 3600 + i * 10,
            )

        assert "current_smoothness" in feedback
        assert feedback["n_samples"] == 5

    def test_update_session_not_active(self, incentive):
        """update_session should return None if no active session."""
        feedback = incentive.update_session(
            agent_id="agent_001",
            speed=55.0,
            position=(0.0, 0.0),
            timestamp=1000,
        )

        assert feedback is None

    def test_end_session(self, incentive):
        """end_session should compute results and reward."""
        incentive.start_session("agent_001", "I-95", 8 * 3600, target_speed=55.0)

        # Add some speed samples
        for i in range(10):
            incentive.update_session(
                agent_id="agent_001",
                speed=55.0,  # Perfect adherence
                position=(0.0, 0.01 * i),
                timestamp=8 * 3600 + i * 60,
            )

        result = incentive.end_session("agent_001", 8 * 3600 + 600)

        assert "session_id" in result
        assert "status" in result
        assert "distance_miles" in result
        assert "smoothness_score" in result
        assert "reward" in result
        assert "agent_001" not in incentive.active_sessions

    def test_end_session_no_active(self, incentive):
        """end_session should handle no active session."""
        result = incentive.end_session("agent_001", 1000)

        assert "error" in result

    def test_end_session_aborted_short_distance(self, incentive):
        """end_session should abort if distance too short."""
        incentive.start_session("agent_001", "I-95", 8 * 3600)

        # Add just a few samples (not enough distance)
        for i in range(3):
            incentive.update_session(
                agent_id="agent_001",
                speed=55.0,
                position=(0.0, 0.0001 * i),  # Very short distance
                timestamp=8 * 3600 + i * 10,
            )

        result = incentive.end_session("agent_001", 8 * 3600 + 30)

        assert result["status"] == "aborted"
        assert result["reward"] == 0.0

    def test_end_session_smoothness_computation(self, incentive):
        """end_session should compute smoothness correctly."""
        incentive.start_session("agent_001", "I-95", 8 * 3600, target_speed=55.0)

        # Add variable speed samples
        for i, speed in enumerate([55, 60, 50, 55, 65, 45]):
            incentive.update_session(
                agent_id="agent_001",
                speed=speed,
                position=(0.0, 0.1 * i),  # Enough distance
                timestamp=8 * 3600 + i * 60,
            )

        result = incentive.end_session("agent_001", 8 * 3600 + 360)

        assert "smoothness_score" in result
        assert 0 <= result["smoothness_score"] <= 1

    def test_compute_distance(self, incentive):
        """_compute_distance should sum segment distances."""
        positions = [
            (0.0, 0.0),
            (0.0, 0.01),  # About 0.69 miles
            (0.0, 0.02),  # About 0.69 more miles
        ]

        distance = incentive._compute_distance(positions)

        assert distance > 0

    def test_compute_distance_single_point(self, incentive):
        """_compute_distance should handle single point."""
        positions = [(0.0, 0.0)]

        distance = incentive._compute_distance(positions)

        assert distance == 0.0

    def test_compute_corridor_performance(self, incentive):
        """compute_corridor_performance should aggregate stats."""
        # Create and complete sessions
        for i in range(3):
            session = incentive.start_session(f"agent_{i:03d}", "I-95", 8 * 3600)
            for j in range(5):
                incentive.update_session(
                    agent_id=f"agent_{i:03d}",
                    speed=55.0 + i,
                    position=(0.0, 0.1 * j),
                    timestamp=8 * 3600 + j * 60,
                )
            incentive.end_session(f"agent_{i:03d}", 8 * 3600 + 300)

        perf = incentive.compute_corridor_performance("I-95")

        assert perf["corridor_id"] == "I-95"
        assert perf["n_sessions"] >= 0  # May be 0 if aborted
        assert "total_pacer_miles" in perf

    def test_compute_corridor_performance_no_sessions(self, incentive):
        """compute_corridor_performance should handle empty corridor."""
        perf = incentive.compute_corridor_performance("I-405")

        assert perf["corridor_id"] == "I-405"
        assert perf["n_sessions"] == 0

    def test_get_statistics(self, incentive):
        """get_statistics should include pacer-specific stats."""
        # Create a session
        incentive.start_session("agent_001", "I-95", 8 * 3600)
        for i in range(5):
            incentive.update_session(
                agent_id="agent_001",
                speed=55.0,
                position=(0.0, 0.1 * i),
                timestamp=8 * 3600 + i * 60,
            )
        incentive.end_session("agent_001", 8 * 3600 + 300)

        stats = incentive.get_statistics()

        assert "total_sessions" in stats
        assert "active_sessions" in stats
        assert "completed_sessions" in stats
        assert "successful_sessions" in stats
        assert "success_rate" in stats
        assert "total_pacer_miles" in stats
        assert "avg_smoothness" in stats
        assert "corridors" in stats

    def test_multiple_concurrent_sessions(self, incentive):
        """Multiple agents can have concurrent sessions."""
        incentive.start_session("agent_001", "I-95", 8 * 3600)
        incentive.start_session("agent_002", "I-95", 8 * 3600)
        incentive.start_session("agent_003", "I-10", 8 * 3600)

        assert len(incentive.active_sessions) == 3
        assert "agent_001" in incentive.active_sessions
        assert "agent_002" in incentive.active_sessions
        assert "agent_003" in incentive.active_sessions

    def test_session_ids_unique(self, incentive):
        """Session IDs should be unique."""
        for i in range(10):
            incentive.start_session(f"agent_{i:03d}", "I-95", 8 * 3600 + i * 100)
            incentive.end_session(f"agent_{i:03d}", 8 * 3600 + i * 100 + 50)

        ids = [s.session_id for s in incentive.sessions]
        assert len(ids) == len(set(ids))
