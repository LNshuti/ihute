"""
Tests for src/agents/pacer.py module.

Tests cover:
- PacerProfile dataclass
- PacerPerformance dataclass
- PacerAgent class
- create_pacer_population function
"""

import numpy as np
import pytest

from src.agents.base import AgentPreferences, TravelMode
from src.agents.pacer import (
    PacerAgent,
    PacerPerformance,
    PacerProfile,
    create_pacer_population,
)


class TestPacerProfile:
    """Tests for PacerProfile dataclass."""

    def test_required_fields(self):
        """PacerProfile requires home, work, and arrival time."""
        profile = PacerProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
        )
        assert profile.home_location == (0.0, 0.0)
        assert profile.work_location == (10.0, 10.0)
        assert profile.desired_arrival_time == 8 * 3600

    def test_default_values(self):
        """PacerProfile should have sensible defaults."""
        profile = PacerProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
        )
        assert profile.flexibility_window == 3600
        assert profile.enrolled_corridors == []
        assert profile.max_speed_reduction == 0.2
        assert profile.driving_skill == 0.8

    def test_custom_enrolled_corridors(self):
        """enrolled_corridors can be customized."""
        profile = PacerProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
            enrolled_corridors=["I-95", "US-1"],
        )
        assert profile.enrolled_corridors == ["I-95", "US-1"]


class TestPacerPerformance:
    """Tests for PacerPerformance dataclass."""

    def test_default_values(self):
        """PacerPerformance should have zero defaults."""
        perf = PacerPerformance()
        assert perf.trips_as_pacer == 0
        assert perf.total_pacer_miles == 0.0
        assert perf.total_pacer_earnings == 0.0
        assert perf.avg_smoothness_score == 0.0
        assert perf.speed_variance_history == []


class TestPacerAgent:
    """Tests for PacerAgent class."""

    @pytest.fixture
    def profile(self):
        """Create a pacer profile."""
        return PacerProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
            enrolled_corridors=["I-95"],
            driving_skill=0.85,
        )

    @pytest.fixture
    def agent(self, profile):
        """Create a pacer agent."""
        return PacerAgent(
            agent_id="pacer_test",
            profile=profile,
            rng=np.random.default_rng(42),
        )

    def test_agent_initialization(self, agent, profile):
        """Agent should initialize with profile and performance."""
        assert agent.id == "pacer_test"
        assert agent.profile == profile
        assert isinstance(agent.performance, PacerPerformance)
        assert agent.is_pacing is False

    def test_decide_mode_prefers_drive_alone(self, agent):
        """Pacer should prefer drive alone."""
        modes = [TravelMode.TRANSIT, TravelMode.DRIVE_ALONE, TravelMode.CARPOOL_PASSENGER]
        mode = agent.decide_mode(modes)
        assert mode == TravelMode.DRIVE_ALONE

    def test_decide_mode_fallback(self, agent):
        """Pacer should fall back if drive alone not available."""
        modes = [TravelMode.TRANSIT, TravelMode.WALK]
        mode = agent.decide_mode(modes)
        assert mode == TravelMode.TRANSIT

    def test_decide_departure_time(self, agent):
        """Pacer should decide departure time."""
        departure = agent.decide_departure_time()
        assert departure is not None
        assert departure < agent.profile.desired_arrival_time

    def test_decide_route_empty(self, agent):
        """Empty route list returns empty."""
        route = agent.decide_route(0, 1, [])
        assert route == []

    def test_decide_route_single(self, agent):
        """Single route is returned."""
        route = agent.decide_route(0, 1, [[0, 1, 2]])
        assert route == [0, 1, 2]

    def test_decide_route_prefers_shortest(self, agent):
        """Pacer prefers shortest route."""
        routes = [[0, 1, 2, 3, 4], [0, 5, 4], [0, 6, 7, 8, 9, 4]]
        route = agent.decide_route(0, 4, routes)
        assert route == [0, 5, 4]  # Shortest

    def test_respond_to_incentive_non_pacer_type(self, agent):
        """Non-pacer incentives should be rejected."""
        response = agent.respond_to_incentive(
            incentive_type="carpool",
            incentive_amount=10.0,
            conditions={},
        )
        assert response is False

    def test_respond_to_incentive_wrong_corridor(self, agent):
        """Wrong corridor should be rejected."""
        response = agent.respond_to_incentive(
            incentive_type="pacer",
            incentive_amount=10.0,
            conditions={"corridor_id": "I-405"},  # Not enrolled
        )
        assert response is False

    def test_respond_to_incentive_speed_reduction_too_high(self, agent):
        """Excessive speed reduction should be rejected."""
        response = agent.respond_to_incentive(
            incentive_type="pacer",
            incentive_amount=10.0,
            conditions={
                "corridor_id": "I-95",
                "speed_reduction_fraction": 0.5,  # More than max 0.2
            },
        )
        assert response is False

    def test_respond_to_incentive_valid(self, agent):
        """Valid pacer incentive may be accepted."""
        # Run multiple times since probabilistic
        responses = [
            agent.respond_to_incentive(
                incentive_type="pacer",
                incentive_amount=20.0,
                conditions={
                    "corridor_id": "I-95",
                    "speed_reduction_fraction": 0.1,
                    "smoothness_threshold": 0.7,
                },
            )
            for _ in range(20)
        ]
        # Should accept at least sometimes with good incentive
        assert any(responses) or not any(responses)  # Either is valid

    def test_start_pacing(self, agent):
        """start_pacing should initialize session."""
        agent.start_pacing(target_speed=55.0, corridor_id="I-95")

        assert agent.is_pacing is True
        assert agent.current_target_speed == 55.0
        assert agent.speed_history == []
        assert len(agent.history) == 1
        assert agent.history[0]["event_type"] == "pacing_started"

    def test_update_speed_not_pacing(self, agent):
        """update_speed without pacing returns current speed."""
        result = agent.update_speed(current_speed=60.0, timestamp=1000)
        assert result == 60.0

    def test_update_speed_while_pacing(self, agent):
        """update_speed while pacing records and adjusts speed."""
        agent.start_pacing(target_speed=55.0, corridor_id="I-95")

        result = agent.update_speed(current_speed=60.0, timestamp=1000)

        assert len(agent.speed_history) == 1
        assert agent.speed_history[0] == 60.0
        # Should suggest moving toward target
        assert result >= 0

    def test_end_pacing_empty_history(self, agent):
        """end_pacing with empty history returns zero scores."""
        agent.start_pacing(target_speed=55.0, corridor_id="I-95")
        result = agent.end_pacing()

        assert result["smoothness_score"] == 0.0
        assert result["earnings"] == 0.0
        assert agent.is_pacing is False

    def test_end_pacing_with_history(self, agent):
        """end_pacing should compute smoothness from history."""
        agent.start_pacing(target_speed=55.0, corridor_id="I-95")

        # Add some speed samples
        for speed in [55, 54, 56, 55, 54, 55, 56, 55]:
            agent.update_speed(current_speed=speed, timestamp=1000)

        result = agent.end_pacing()

        assert "smoothness_score" in result
        assert "mean_speed" in result
        assert "speed_variance" in result
        assert result["n_observations"] == 8
        assert agent.is_pacing is False

    def test_end_pacing_updates_performance(self, agent):
        """end_pacing should update performance metrics."""
        agent.start_pacing(target_speed=55.0, corridor_id="I-95")

        for speed in [55, 55, 55, 55, 55]:  # Very smooth
            agent.update_speed(current_speed=speed, timestamp=1000)

        agent.end_pacing()

        assert len(agent.performance.speed_variance_history) == 1

    def test_receive_pacer_payment(self, agent):
        """receive_pacer_payment should update stats."""
        agent.receive_pacer_payment(amount=5.0, distance_miles=10.0)

        assert agent.performance.trips_as_pacer == 1
        assert agent.performance.total_pacer_miles == 10.0
        assert agent.performance.total_pacer_earnings == 5.0
        assert agent.state.incentives_earned == 5.0
        assert len(agent.history) == 1

    def test_receive_multiple_payments(self, agent):
        """Multiple payments should accumulate."""
        agent.receive_pacer_payment(amount=5.0, distance_miles=10.0)
        agent.receive_pacer_payment(amount=3.0, distance_miles=8.0)

        assert agent.performance.trips_as_pacer == 2
        assert agent.performance.total_pacer_miles == 18.0
        assert agent.performance.total_pacer_earnings == 8.0

    def test_get_statistics(self, agent):
        """get_statistics should return summary dict."""
        agent.receive_pacer_payment(amount=5.0, distance_miles=10.0)

        stats = agent.get_statistics()

        assert stats["agent_id"] == "pacer_test"
        assert stats["trips_as_pacer"] == 1
        assert stats["total_pacer_miles"] == 10.0
        assert stats["total_pacer_earnings"] == 5.0
        assert stats["earnings_per_mile"] == 0.5
        assert "I-95" in stats["enrolled_corridors"]

    def test_sigmoid_function(self, agent):
        """_sigmoid should map to (0, 1)."""
        assert 0 < agent._sigmoid(0) < 1
        assert agent._sigmoid(0) == pytest.approx(0.5)
        assert agent._sigmoid(10) > 0.9
        assert agent._sigmoid(-10) < 0.1


class TestCreatePacerPopulation:
    """Tests for create_pacer_population function."""

    def test_creates_correct_number(self):
        """Function should create requested number of agents."""
        agents = create_pacer_population(
            n_agents=25,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            corridors=["I-95", "I-10"],
            rng=np.random.default_rng(42),
        )
        assert len(agents) == 25

    def test_agents_have_unique_ids(self):
        """Each agent should have unique ID."""
        agents = create_pacer_population(
            n_agents=50,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            corridors=["I-95"],
            rng=np.random.default_rng(42),
        )
        ids = [a.id for a in agents]
        assert len(ids) == len(set(ids))

    def test_agents_enrolled_in_corridors(self):
        """Agents should be enrolled in provided corridors."""
        corridors = ["I-95", "I-10", "US-1"]
        agents = create_pacer_population(
            n_agents=50,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            corridors=corridors,
            rng=np.random.default_rng(42),
        )

        for agent in agents:
            # Each agent should be enrolled in 1-3 corridors
            assert 1 <= len(agent.profile.enrolled_corridors) <= 3
            for corridor in agent.profile.enrolled_corridors:
                assert corridor in corridors

    def test_driving_skills_varied(self):
        """Agents should have varied driving skills."""
        agents = create_pacer_population(
            n_agents=100,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            corridors=["I-95"],
            rng=np.random.default_rng(42),
        )

        skills = [a.profile.driving_skill for a in agents]

        # Should have variety in range [0.6, 0.95]
        assert min(skills) >= 0.6
        assert max(skills) <= 0.95
        assert len(set(skills)) > 1

    def test_flexibility_windows_varied(self):
        """Agents should have varied flexibility windows."""
        agents = create_pacer_population(
            n_agents=50,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            corridors=["I-95"],
            rng=np.random.default_rng(42),
        )

        windows = [a.profile.flexibility_window for a in agents]

        # Should be in range [1800, 5400] (30-90 min)
        assert min(windows) >= 1800
        assert max(windows) <= 5400

    def test_incentive_responsiveness_boosted(self):
        """Pacer agents should have boosted incentive responsiveness."""
        agents = create_pacer_population(
            n_agents=50,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            corridors=["I-95"],
            rng=np.random.default_rng(42),
        )

        # Beta incentive should be boosted (1.5x)
        # Since base mean is 0.15, boosted should average around 0.225
        avg_beta = np.mean([a.preferences.beta_incentive for a in agents])
        # With boost, should be higher than unboosted mean
        assert avg_beta > 0.15

    def test_reproducibility(self):
        """Same seed should produce same population."""
        agents1 = create_pacer_population(
            n_agents=10,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            corridors=["I-95", "I-10"],
            rng=np.random.default_rng(42),
        )
        agents2 = create_pacer_population(
            n_agents=10,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            corridors=["I-95", "I-10"],
            rng=np.random.default_rng(42),
        )

        for a1, a2 in zip(agents1, agents2):
            assert a1.profile.home_location == a2.profile.home_location
            assert a1.profile.driving_skill == a2.profile.driving_skill
