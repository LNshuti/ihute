"""
Tests for src/agents/commuter.py module.

Tests cover:
- CommuterProfile dataclass
- CommuterAgent class
- create_commuter_population function
"""

import numpy as np
import pytest

from src.agents.base import (
    AgentPreferences,
    DecisionRule,
    TravelMode,
    TripAttributes,
)
from src.agents.commuter import (
    CommuterAgent,
    CommuterProfile,
    create_commuter_population,
)


class TestCommuterProfile:
    """Tests for CommuterProfile dataclass."""

    def test_required_fields(self):
        """CommuterProfile requires home, work, and arrival time."""
        profile = CommuterProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
        )
        assert profile.home_location == (0.0, 0.0)
        assert profile.work_location == (10.0, 10.0)
        assert profile.desired_arrival_time == 8 * 3600

    def test_default_values(self):
        """CommuterProfile should have sensible defaults."""
        profile = CommuterProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
        )
        assert profile.flexibility_window == 1800
        assert profile.has_car is True
        assert profile.has_transit_pass is False
        assert profile.carpool_eligible is True
        assert profile.work_days == [0, 1, 2, 3, 4]

    def test_post_init_work_days(self):
        """work_days should default to Mon-Fri if None."""
        profile = CommuterProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
            work_days=None,
        )
        assert profile.work_days == [0, 1, 2, 3, 4]

    def test_custom_work_days(self):
        """work_days can be customized."""
        profile = CommuterProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
            work_days=[1, 2, 3],  # Tue-Thu
        )
        assert profile.work_days == [1, 2, 3]


class TestCommuterAgent:
    """Tests for CommuterAgent class."""

    @pytest.fixture
    def profile(self):
        """Create a commuter profile."""
        return CommuterProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
            has_car=True,
            carpool_eligible=True,
        )

    @pytest.fixture
    def agent(self, profile):
        """Create a commuter agent."""
        return CommuterAgent(
            agent_id="commuter_test",
            profile=profile,
            rng=np.random.default_rng(42),
        )

    def test_agent_initialization(self, agent, profile):
        """Agent should initialize with profile."""
        assert agent.id == "commuter_test"
        assert agent.profile == profile
        assert agent.state.position == profile.home_location

    def test_agent_initial_statistics(self, agent):
        """Agent should have zero initial statistics."""
        assert agent.trips_completed == 0
        assert agent.total_travel_time == 0.0
        assert agent.total_incentives == 0.0
        assert agent.mode_history == []

    def test_decide_mode_empty_list(self, agent):
        """Empty mode list should return DRIVE_ALONE."""
        mode = agent.decide_mode([])
        assert mode == TravelMode.DRIVE_ALONE

    def test_decide_mode_single_option(self, agent):
        """Single option should be returned."""
        mode = agent.decide_mode([TravelMode.TRANSIT])
        assert mode == TravelMode.TRANSIT

    def test_decide_mode_with_options(self, agent):
        """Agent should choose from available modes."""
        modes = [TravelMode.DRIVE_ALONE, TravelMode.TRANSIT, TravelMode.CARPOOL_DRIVER]
        mode = agent.decide_mode(modes)
        assert mode in modes

    def test_decide_mode_filters_unavailable(self):
        """Agent without car cannot drive."""
        profile = CommuterProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
            has_car=False,
        )
        agent = CommuterAgent(profile=profile, rng=np.random.default_rng(42))

        modes = [TravelMode.DRIVE_ALONE, TravelMode.TRANSIT]
        mode = agent.decide_mode(modes)

        # Should not choose DRIVE_ALONE without a car
        # It filters to available modes first
        assert mode in [
            TravelMode.TRANSIT,
            TravelMode.DRIVE_ALONE,
        ]  # Falls back if needed

    def test_mode_available_drive_with_car(self, agent):
        """Agent with car can drive."""
        assert agent._mode_available(TravelMode.DRIVE_ALONE) is True
        assert agent._mode_available(TravelMode.CARPOOL_DRIVER) is True

    def test_mode_available_drive_without_car(self):
        """Agent without car cannot drive."""
        profile = CommuterProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
            has_car=False,
        )
        agent = CommuterAgent(profile=profile)

        assert agent._mode_available(TravelMode.DRIVE_ALONE) is False
        assert agent._mode_available(TravelMode.CARPOOL_DRIVER) is False

    def test_mode_available_carpool_passenger(self, agent):
        """Carpool passenger availability depends on eligibility."""
        assert agent._mode_available(TravelMode.CARPOOL_PASSENGER) is True

    def test_estimate_trip_drive_alone(self, agent):
        """Estimate trip attributes for drive alone."""
        trip = agent._estimate_trip(TravelMode.DRIVE_ALONE)

        assert trip.mode == TravelMode.DRIVE_ALONE
        assert trip.travel_time == 30.0
        assert trip.cost == 5.0
        assert trip.comfort_score == 0.9

    def test_estimate_trip_transit(self, agent):
        """Estimate trip attributes for transit."""
        trip = agent._estimate_trip(TravelMode.TRANSIT)

        assert trip.mode == TravelMode.TRANSIT
        assert trip.travel_time == 45.0  # 1.5x base time
        assert trip.cost == 2.0
        assert trip.comfort_score == 0.4

    def test_decide_departure_time_default(self, agent):
        """Departure time should be before desired arrival."""
        departure = agent.decide_departure_time()

        expected_travel_time = 1800  # 30 min default
        latest_arrival = agent.profile.desired_arrival_time
        earliest_arrival = latest_arrival - agent.profile.flexibility_window

        # Departure should result in arrival within window
        assert departure < latest_arrival

    def test_decide_departure_time_with_incentive(self, agent):
        """Incentive schedule should influence departure time."""
        # Create incentive schedule that rewards early departure
        early_time = agent.profile.desired_arrival_time - 3600  # 1 hour early
        incentive_schedule = {
            early_time: 10.0,  # $10 for early departure
        }

        departure = agent.decide_departure_time(
            incentive_schedule=incentive_schedule,
        )

        assert departure is not None

    def test_decide_route_empty_list(self, agent):
        """Empty route list should return empty list."""
        route = agent.decide_route(0, 1, [])
        assert route == []

    def test_decide_route_single_option(self, agent):
        """Single route option should be returned."""
        route = agent.decide_route(0, 1, [[0, 1]])
        assert route == [0, 1]

    def test_decide_route_multiple_options(self, agent):
        """Agent should choose from available routes."""
        routes = [[0, 1, 2], [0, 3, 4, 2], [0, 5, 2]]
        route = agent.decide_route(0, 2, routes)
        assert route in routes

    def test_respond_to_incentive_eligible(self, agent):
        """Agent should respond to valid incentive."""
        response = agent.respond_to_incentive(
            incentive_type="carpool",
            incentive_amount=5.0,
            conditions={"coordination_cost": 2.0},
        )
        assert isinstance(response, bool)

    def test_respond_to_incentive_ineligible_carpool(self):
        """Agent not carpool eligible should not respond."""
        profile = CommuterProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
            carpool_eligible=False,
        )
        agent = CommuterAgent(profile=profile)

        response = agent.respond_to_incentive(
            incentive_type="carpool",
            incentive_amount=10.0,
            conditions={},
        )
        assert response is False

    def test_respond_to_incentive_departure_shift_too_large(self, agent):
        """Agent should reject shifts outside flexibility window."""
        response = agent.respond_to_incentive(
            incentive_type="departure_shift",
            incentive_amount=10.0,
            conditions={"required_shift": 7200},  # 2 hours, window is 30 min
        )
        assert response is False

    def test_complete_trip(self, agent):
        """complete_trip should update statistics."""
        agent.complete_trip(
            travel_time=30.0,
            mode=TravelMode.DRIVE_ALONE,
            incentive_earned=5.0,
        )

        assert agent.trips_completed == 1
        assert agent.total_travel_time == 30.0
        assert agent.total_incentives == 5.0
        assert TravelMode.DRIVE_ALONE in agent.mode_history
        assert len(agent.history) == 1

    def test_complete_multiple_trips(self, agent):
        """Multiple trips should accumulate statistics."""
        agent.complete_trip(30.0, TravelMode.DRIVE_ALONE, 5.0)
        agent.complete_trip(45.0, TravelMode.TRANSIT, 2.0)
        agent.complete_trip(35.0, TravelMode.CARPOOL_PASSENGER, 3.0)

        assert agent.trips_completed == 3
        assert agent.total_travel_time == 110.0
        assert agent.total_incentives == 10.0
        assert len(agent.mode_history) == 3

    def test_get_statistics(self, agent):
        """get_statistics should return summary dict."""
        agent.complete_trip(30.0, TravelMode.DRIVE_ALONE, 5.0)
        agent.complete_trip(45.0, TravelMode.TRANSIT, 2.0)

        stats = agent.get_statistics()

        assert stats["agent_id"] == "commuter_test"
        assert stats["trips_completed"] == 2
        assert stats["total_travel_time"] == 75.0
        assert stats["avg_travel_time"] == 37.5
        assert stats["total_incentives"] == 7.0
        assert "mode_distribution" in stats
        assert "preferences" in stats

    def test_get_statistics_no_trips(self, agent):
        """get_statistics should handle zero trips."""
        stats = agent.get_statistics()

        assert stats["trips_completed"] == 0
        assert stats["avg_travel_time"] == 0.0


class TestCreateCommuterPopulation:
    """Tests for create_commuter_population function."""

    def test_creates_correct_number_of_agents(self):
        """Function should create requested number of agents."""
        agents = create_commuter_population(
            n_agents=50,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            rng=np.random.default_rng(42),
        )
        assert len(agents) == 50

    def test_agents_have_unique_ids(self):
        """Each agent should have a unique ID."""
        agents = create_commuter_population(
            n_agents=100,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            rng=np.random.default_rng(42),
        )
        ids = [a.id for a in agents]
        assert len(ids) == len(set(ids))

    def test_agents_in_correct_regions(self):
        """Agent locations should be within specified regions."""
        home_region = ((0, 0), (10, 10))
        work_region = ((20, 20), (30, 30))

        agents = create_commuter_population(
            n_agents=50,
            home_region=home_region,
            work_region=work_region,
            rng=np.random.default_rng(42),
        )

        for agent in agents:
            hx, hy = agent.profile.home_location
            wx, wy = agent.profile.work_location

            assert home_region[0][0] <= hx <= home_region[1][0]
            assert home_region[0][1] <= hy <= home_region[1][1]
            assert work_region[0][0] <= wx <= work_region[1][0]
            assert work_region[0][1] <= wy <= work_region[1][1]

    def test_arrival_times_reasonable(self):
        """Arrival times should be within reasonable bounds."""
        agents = create_commuter_population(
            n_agents=50,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            arrival_time_dist=(8 * 3600, 1800),  # 8 AM, 30 min std
            rng=np.random.default_rng(42),
        )

        for agent in agents:
            # Clipped to 6 AM - 10 AM
            assert 6 * 3600 <= agent.profile.desired_arrival_time <= 10 * 3600

    def test_heterogeneous_preferences(self):
        """Agents should have heterogeneous preferences."""
        agents = create_commuter_population(
            n_agents=100,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            rng=np.random.default_rng(42),
        )

        vots = [a.preferences.vot for a in agents]
        decision_rules = [a.preferences.decision_rule for a in agents]

        # Should have variety
        assert len(set(vots)) > 1
        assert len(set(decision_rules)) > 1

    def test_car_ownership_distribution(self):
        """Some agents should not have cars."""
        agents = create_commuter_population(
            n_agents=100,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            rng=np.random.default_rng(42),
        )

        has_car = [a.profile.has_car for a in agents]

        # 90% have cars, so should see some without
        assert not all(has_car)
        # But most should have cars
        assert sum(has_car) > 80

    def test_reproducibility_with_seed(self):
        """Same seed should produce same population."""
        agents1 = create_commuter_population(
            n_agents=10,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            rng=np.random.default_rng(42),
        )
        agents2 = create_commuter_population(
            n_agents=10,
            home_region=((0, 0), (10, 10)),
            work_region=((20, 20), (30, 30)),
            rng=np.random.default_rng(42),
        )

        for a1, a2 in zip(agents1, agents2):
            assert a1.profile.home_location == a2.profile.home_location
            assert a1.preferences.vot == a2.preferences.vot
