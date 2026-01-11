"""
Tests for src/agents/base.py module.

Tests cover:
- TravelMode and DecisionRule enums
- AgentState, AgentPreferences, TripAttributes dataclasses
- LinearUtilityModel behavioral model
- BaseAgent abstract class implementation
- generate_heterogeneous_preferences function
"""

import numpy as np
import pytest

from src.agents.base import (
    AgentPreferences,
    AgentState,
    BaseAgent,
    BehavioralModel,
    DecisionRule,
    LinearUtilityModel,
    PopulationParameters,
    TravelMode,
    TripAttributes,
    generate_heterogeneous_preferences,
)


class TestTravelMode:
    """Tests for TravelMode enum."""

    def test_all_modes_exist(self):
        """Verify all expected travel modes are defined."""
        expected_modes = [
            "DRIVE_ALONE",
            "CARPOOL_DRIVER",
            "CARPOOL_PASSENGER",
            "TRANSIT",
            "WALK",
            "BIKE",
            "RIDESHARE",
        ]
        for mode_name in expected_modes:
            assert hasattr(TravelMode, mode_name)

    def test_mode_values_are_unique(self):
        """Each mode should have a unique value."""
        values = [mode.value for mode in TravelMode]
        assert len(values) == len(set(values))


class TestDecisionRule:
    """Tests for DecisionRule enum."""

    def test_all_rules_exist(self):
        """Verify all expected decision rules are defined."""
        expected_rules = [
            "UTILITY_MAX",
            "SOFTMAX",
            "EPSILON_GREEDY",
            "SATISFICING",
        ]
        for rule_name in expected_rules:
            assert hasattr(DecisionRule, rule_name)


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_default_values(self):
        """AgentState should have sensible defaults."""
        state = AgentState(position=(0.0, 0.0))
        assert state.position == (0.0, 0.0)
        assert state.velocity == 0.0
        assert state.heading == 0.0
        assert state.mode == TravelMode.DRIVE_ALONE
        assert state.route is None
        assert state.departure_time == 0.0
        assert state.arrival_time is None
        assert state.current_segment is None
        assert state.distance_traveled == 0.0
        assert state.incentives_earned == 0.0

    def test_custom_values(self):
        """AgentState should accept custom values."""
        state = AgentState(
            position=(10.5, 20.5),
            velocity=30.0,
            heading=1.57,
            mode=TravelMode.TRANSIT,
            route=[1, 2, 3],
            departure_time=8 * 3600,
            arrival_time=9 * 3600,
            current_segment=2,
            distance_traveled=15.5,
            incentives_earned=5.0,
        )
        assert state.position == (10.5, 20.5)
        assert state.velocity == 30.0
        assert state.mode == TravelMode.TRANSIT
        assert state.route == [1, 2, 3]


class TestAgentPreferences:
    """Tests for AgentPreferences dataclass."""

    def test_default_values(self):
        """AgentPreferences should have sensible defaults."""
        prefs = AgentPreferences()
        assert prefs.vot == 25.0
        assert prefs.beta_time == -0.05
        assert prefs.beta_cost == -0.10
        assert prefs.beta_incentive == 0.15
        assert prefs.decision_rule == DecisionRule.SOFTMAX
        assert prefs.temperature == 1.0
        assert prefs.epsilon == 0.1

    def test_custom_values(self):
        """AgentPreferences should accept custom values."""
        prefs = AgentPreferences(
            vot=40.0,
            beta_time=-0.08,
            beta_cost=-0.15,
            decision_rule=DecisionRule.UTILITY_MAX,
        )
        assert prefs.vot == 40.0
        assert prefs.beta_time == -0.08
        assert prefs.decision_rule == DecisionRule.UTILITY_MAX


class TestTripAttributes:
    """Tests for TripAttributes dataclass."""

    def test_required_fields(self):
        """TripAttributes requires mode, travel_time, and cost."""
        trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )
        assert trip.mode == TravelMode.DRIVE_ALONE
        assert trip.travel_time == 30.0
        assert trip.cost == 5.0
        assert trip.incentive == 0.0
        assert trip.comfort_score == 1.0
        assert trip.reliability == 0.9


class TestLinearUtilityModel:
    """Tests for LinearUtilityModel."""

    @pytest.fixture
    def model(self):
        """Create a LinearUtilityModel instance."""
        return LinearUtilityModel()

    @pytest.fixture
    def preferences(self):
        """Create default preferences."""
        return AgentPreferences()

    def test_compute_utility_drive_alone(self, model, preferences):
        """Compute utility for drive alone mode."""
        trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
            comfort_score=0.9,
            reliability=0.8,
        )
        utility = model.compute_utility(preferences, trip)

        # Manual calculation
        expected = (
            0  # ASC for drive alone
            + preferences.beta_time * 30.0
            + preferences.beta_cost * 5.0
            + preferences.beta_incentive * 0.0
            + preferences.beta_comfort * 0.9
            + preferences.beta_reliability * (1 - 0.8) * 30.0
        )
        assert utility == pytest.approx(expected)

    def test_compute_utility_with_incentive(self, model, preferences):
        """Incentives should increase utility."""
        trip_no_incentive = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )
        trip_with_incentive = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
            incentive=10.0,
        )
        utility_no = model.compute_utility(preferences, trip_no_incentive)
        utility_with = model.compute_utility(preferences, trip_with_incentive)

        assert utility_with > utility_no

    def test_compute_utility_carpool_has_asc(self, model, preferences):
        """Carpool modes should include ASC."""
        trip = TripAttributes(
            mode=TravelMode.CARPOOL_DRIVER,
            travel_time=30.0,
            cost=5.0,
        )
        utility = model.compute_utility(preferences, trip)

        # Should include carpool ASC (negative)
        trip_drive = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )
        utility_drive = model.compute_utility(preferences, trip_drive)

        # Carpool has negative ASC, so lower utility (all else equal)
        assert utility < utility_drive

    def test_choose_action_utility_max(self, model):
        """UTILITY_MAX should choose highest utility option."""
        preferences = AgentPreferences(decision_rule=DecisionRule.UTILITY_MAX)
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=60.0, cost=10.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=45.0, cost=7.0),
        ]
        rng = np.random.default_rng(42)

        choice = model.choose_action(preferences, options, rng)

        # Option 1 (index 1) has lowest time and cost, should be chosen
        assert choice == 1

    def test_choose_action_softmax(self, model):
        """SOFTMAX should make probabilistic choices."""
        preferences = AgentPreferences(
            decision_rule=DecisionRule.SOFTMAX,
            temperature=1.0,
        )
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=35.0, cost=6.0),
        ]
        rng = np.random.default_rng(42)

        # Run multiple times to check probabilistic behavior
        choices = [model.choose_action(preferences, options, rng) for _ in range(100)]

        # Should sometimes choose the second option
        assert 0 in choices
        assert 1 in choices

    def test_choose_action_epsilon_greedy(self, model):
        """EPSILON_GREEDY should explore with probability epsilon."""
        preferences = AgentPreferences(
            decision_rule=DecisionRule.EPSILON_GREEDY,
            epsilon=0.5,  # 50% exploration
        )
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=60.0, cost=10.0),
        ]
        rng = np.random.default_rng(42)

        choices = [model.choose_action(preferences, options, rng) for _ in range(100)]

        # With 50% exploration, should see both choices
        assert 0 in choices
        assert 1 in choices

    def test_choose_action_satisficing(self, model):
        """SATISFICING should choose first option above threshold."""
        preferences = AgentPreferences(
            decision_rule=DecisionRule.SATISFICING,
            satisficing_threshold=-2.0,  # Low threshold
        )
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=25.0, cost=4.0),
        ]
        rng = np.random.default_rng(42)

        # First option should satisfy if threshold is low enough
        choice = model.choose_action(preferences, options, rng)

        # Check that it returns a valid index
        assert choice in [0, 1]

    def test_choose_action_empty_options_raises(self, model, preferences):
        """Empty options should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="No options available"):
            model.choose_action(preferences, [], rng)


class TestGenerateHeterogeneousPreferences:
    """Tests for generate_heterogeneous_preferences function."""

    def test_generates_valid_preferences(self):
        """Function should generate valid AgentPreferences."""
        params = PopulationParameters()
        rng = np.random.default_rng(42)

        prefs = generate_heterogeneous_preferences(params, rng)

        assert isinstance(prefs, AgentPreferences)
        assert prefs.vot > 0
        assert prefs.beta_time <= 0
        assert prefs.beta_cost <= 0
        assert prefs.beta_incentive >= 0

    def test_generates_different_preferences(self):
        """Different calls should produce different preferences."""
        params = PopulationParameters()
        rng = np.random.default_rng(42)

        prefs1 = generate_heterogeneous_preferences(params, rng)
        prefs2 = generate_heterogeneous_preferences(params, rng)

        # At least one parameter should differ
        assert (
            prefs1.vot != prefs2.vot
            or prefs1.beta_time != prefs2.beta_time
            or prefs1.decision_rule != prefs2.decision_rule
        )

    def test_respects_parameter_bounds(self):
        """Generated values should be within expected bounds."""
        params = PopulationParameters()
        rng = np.random.default_rng(42)

        for _ in range(100):
            prefs = generate_heterogeneous_preferences(params, rng)

            # Check clipped bounds
            assert -0.2 <= prefs.beta_time <= 0
            assert -0.3 <= prefs.beta_cost <= 0
            assert 0 <= prefs.beta_incentive <= 0.5
            assert prefs.temperature >= 0.1


class TestPopulationParameters:
    """Tests for PopulationParameters dataclass."""

    def test_default_values(self):
        """PopulationParameters should have sensible defaults."""
        params = PopulationParameters()
        assert params.n_agents == 1000
        assert params.vot_mean == 25.0
        assert params.prob_softmax == 0.7
        assert params.prob_utility_max == 0.2
        assert params.prob_epsilon_greedy == 0.1

    def test_probabilities_sum_to_one(self):
        """Decision rule probabilities should sum to 1."""
        params = PopulationParameters()
        total = params.prob_softmax + params.prob_utility_max + params.prob_epsilon_greedy
        assert total == pytest.approx(1.0)


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def decide_mode(self, available_modes):
        return available_modes[0] if available_modes else TravelMode.DRIVE_ALONE

    def decide_departure_time(self, desired_arrival, time_window):
        return desired_arrival - 1800

    def decide_route(self, origin, destination, available_routes):
        return available_routes[0] if available_routes else []

    def respond_to_incentive(self, incentive_type, incentive_amount, conditions):
        return incentive_amount > 5.0


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    @pytest.fixture
    def agent(self):
        """Create a concrete agent for testing."""
        return ConcreteAgent()

    def test_agent_has_id(self, agent):
        """Agent should have a unique ID."""
        assert agent.id is not None
        assert len(agent.id) == 8

    def test_agent_has_default_preferences(self, agent):
        """Agent should have default preferences."""
        assert isinstance(agent.preferences, AgentPreferences)

    def test_agent_has_behavioral_model(self, agent):
        """Agent should have a behavioral model."""
        assert isinstance(agent.behavioral_model, BehavioralModel)

    def test_agent_has_initial_state(self, agent):
        """Agent should have initial state."""
        assert isinstance(agent.state, AgentState)
        assert agent.state.position == (0.0, 0.0)

    def test_update_state(self, agent):
        """update_state should modify agent state."""
        agent.update_state(position=(10.0, 20.0), velocity=15.0)

        assert agent.state.position == (10.0, 20.0)
        assert agent.state.velocity == 15.0

    def test_record_history(self, agent):
        """record_history should add events to history."""
        agent.record_history("test_event", {"value": 42, "timestamp": 1000})

        assert len(agent.history) == 1
        assert agent.history[0]["event_type"] == "test_event"
        assert agent.history[0]["data"]["value"] == 42

    def test_get_utility(self, agent):
        """get_utility should compute trip utility."""
        trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )
        utility = agent.get_utility(trip)

        assert isinstance(utility, float)

    def test_choose_option(self, agent):
        """choose_option should select from available options."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.TRANSIT, travel_time=45.0, cost=2.0),
        ]
        choice = agent.choose_option(options)

        assert choice in [0, 1]

    def test_custom_agent_id(self):
        """Agent should accept custom ID."""
        agent = ConcreteAgent(agent_id="test_agent_001")
        assert agent.id == "test_agent_001"

    def test_custom_rng_reproducibility(self):
        """Custom RNG should make agent behavior reproducible."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        agent1 = ConcreteAgent(rng=rng1)
        agent2 = ConcreteAgent(rng=rng2)

        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.TRANSIT, travel_time=30.0, cost=5.0),
        ]

        # With same seed, choices should be identical
        choices1 = [agent1.choose_option(options) for _ in range(10)]
        choices2 = [agent2.choose_option(options) for _ in range(10)]

        assert choices1 == choices2
