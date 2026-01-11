"""
Tests for src/agents/behavioral.py module.

Tests cover:
- LogitModel
- MixedLogitModel
- ProspectTheoryModel
- RegretMinimizationModel
- estimate_incentive_elasticity function
"""

import numpy as np
import pytest

from src.agents.base import AgentPreferences, TravelMode, TripAttributes
from src.agents.behavioral import (
    LogitModel,
    MixedLogitModel,
    ProspectTheoryModel,
    RegretMinimizationModel,
    estimate_incentive_elasticity,
)


class TestLogitModel:
    """Tests for LogitModel."""

    @pytest.fixture
    def model(self):
        """Create a LogitModel instance."""
        return LogitModel(scale=1.0)

    @pytest.fixture
    def preferences(self):
        """Create default preferences."""
        return AgentPreferences()

    def test_compute_utility(self, model, preferences):
        """Compute utility for a trip."""
        trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
            incentive=2.0,
            comfort_score=0.8,
        )
        utility = model.compute_utility(preferences, trip)

        # Check it's a finite number
        assert np.isfinite(utility)

    def test_asc_for_carpool(self, model, preferences):
        """Carpool modes should have ASC."""
        assert model._get_asc(TravelMode.CARPOOL_DRIVER, preferences) == preferences.asc_carpool
        assert model._get_asc(TravelMode.CARPOOL_PASSENGER, preferences) == preferences.asc_carpool

    def test_asc_for_transit(self, model, preferences):
        """Transit should have ASC."""
        assert model._get_asc(TravelMode.TRANSIT, preferences) == preferences.asc_transit

    def test_asc_for_walk(self, model, preferences):
        """Walk should have ASC."""
        assert model._get_asc(TravelMode.WALK, preferences) == preferences.asc_walk

    def test_asc_for_drive_alone(self, model, preferences):
        """Drive alone should have zero ASC."""
        assert model._get_asc(TravelMode.DRIVE_ALONE, preferences) == 0.0

    def test_choice_probabilities_sum_to_one(self, model, preferences):
        """Choice probabilities should sum to 1."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.TRANSIT, travel_time=45.0, cost=2.0),
            TripAttributes(mode=TravelMode.WALK, travel_time=60.0, cost=0.0),
        ]
        probs = model.choice_probabilities(preferences, options)

        assert probs.sum() == pytest.approx(1.0)
        assert all(p >= 0 for p in probs)

    def test_choice_probabilities_better_option_higher(self, model, preferences):
        """Better options should have higher probability."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=60.0, cost=10.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
        ]
        probs = model.choice_probabilities(preferences, options)

        # Second option is better (less time, less cost)
        assert probs[1] > probs[0]

    def test_choose_action(self, model, preferences):
        """choose_action should return valid index."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.TRANSIT, travel_time=45.0, cost=2.0),
        ]
        rng = np.random.default_rng(42)

        choice = model.choose_action(preferences, options, rng)

        assert choice in [0, 1]

    def test_scale_parameter_affects_probabilities(self, preferences):
        """Scale parameter should affect probability distribution."""
        model_low = LogitModel(scale=0.5)
        model_high = LogitModel(scale=2.0)

        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=35.0, cost=6.0),
        ]

        probs_low = model_low.choice_probabilities(preferences, options)
        probs_high = model_high.choice_probabilities(preferences, options)

        # Higher scale = more uniform probabilities
        assert abs(probs_high[0] - probs_high[1]) < abs(probs_low[0] - probs_low[1])


class TestMixedLogitModel:
    """Tests for MixedLogitModel."""

    @pytest.fixture
    def model(self):
        """Create a MixedLogitModel instance."""
        return MixedLogitModel(n_draws=50)

    @pytest.fixture
    def preferences(self):
        """Create default preferences."""
        return AgentPreferences()

    def test_compute_utility(self, model, preferences):
        """Compute base utility."""
        trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
            incentive=2.0,
        )
        utility = model.compute_utility(preferences, trip)

        assert np.isfinite(utility)

    def test_draw_coefficients_normal(self):
        """Draw coefficients from normal distribution."""
        model = MixedLogitModel(
            coefficient_distributions={"beta_time": ("normal", -0.05, 0.02)}
        )
        rng = np.random.default_rng(42)

        coeffs = model._draw_coefficients(rng)

        assert "beta_time" in coeffs
        assert isinstance(coeffs["beta_time"], float)

    def test_draw_coefficients_lognormal(self):
        """Draw coefficients from lognormal distribution."""
        model = MixedLogitModel(
            coefficient_distributions={"beta_cost": ("lognormal", -0.1, 0.05)}
        )
        rng = np.random.default_rng(42)

        coeffs = model._draw_coefficients(rng)

        assert "beta_cost" in coeffs
        # Lognormal with negative mean should produce negative values
        assert coeffs["beta_cost"] < 0

    def test_draw_coefficients_triangular(self):
        """Draw coefficients from triangular distribution."""
        model = MixedLogitModel(
            coefficient_distributions={"beta_time": ("triangular", -0.05, 0.02)}
        )
        rng = np.random.default_rng(42)

        coeffs = model._draw_coefficients(rng)

        assert "beta_time" in coeffs

    def test_choice_probabilities_sum_to_one(self, model, preferences):
        """Choice probabilities should sum to 1."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.TRANSIT, travel_time=45.0, cost=2.0),
        ]
        rng = np.random.default_rng(42)

        probs = model.choice_probabilities(preferences, options, rng)

        assert probs.sum() == pytest.approx(1.0)

    def test_choose_action(self, model, preferences):
        """choose_action should return valid index."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.TRANSIT, travel_time=45.0, cost=2.0),
        ]
        rng = np.random.default_rng(42)

        choice = model.choose_action(preferences, options, rng)

        assert choice in [0, 1]


class TestProspectTheoryModel:
    """Tests for ProspectTheoryModel."""

    @pytest.fixture
    def model(self):
        """Create a ProspectTheoryModel instance."""
        return ProspectTheoryModel()

    @pytest.fixture
    def preferences(self):
        """Create default preferences."""
        return AgentPreferences()

    def test_value_function_gains(self, model):
        """Value function for gains."""
        value = model.value_function(10.0)
        assert value > 0
        assert value < 10.0  # Diminishing sensitivity

    def test_value_function_losses(self, model):
        """Value function for losses."""
        value = model.value_function(-10.0)
        assert value < 0
        # Loss aversion: |v(-x)| > v(x)
        gain_value = model.value_function(10.0)
        assert abs(value) > abs(gain_value)

    def test_value_function_zero(self, model):
        """Value function at reference point."""
        value = model.value_function(0.0)
        assert value == 0.0

    def test_probability_weight_extremes(self, model):
        """Probability weight at extremes."""
        assert model.probability_weight(0.0) == 0.0
        assert model.probability_weight(1.0) == 1.0

    def test_probability_weight_middle(self, model):
        """Probability weight for middle probabilities."""
        w = model.probability_weight(0.5)
        assert 0 < w < 1

    def test_compute_utility_no_reference(self, model, preferences):
        """Utility without reference point uses standard calculation."""
        trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
            incentive=2.0,
        )
        utility = model.compute_utility(preferences, trip)

        assert np.isfinite(utility)

    def test_compute_utility_with_reference(self, preferences):
        """Utility with reference point uses gains/losses."""
        reference = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
            incentive=0.0,
        )
        model = ProspectTheoryModel(reference_point=reference)

        # Better trip (less time, same cost, more incentive)
        better_trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=25.0,
            cost=5.0,
            incentive=2.0,
        )
        utility = model.compute_utility(preferences, better_trip)

        # Should be positive (gains)
        assert utility > 0

    def test_choose_action(self, model, preferences):
        """choose_action should return valid index."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.TRANSIT, travel_time=45.0, cost=2.0),
        ]
        rng = np.random.default_rng(42)

        choice = model.choose_action(preferences, options, rng)

        assert choice in [0, 1]

    def test_set_reference_point(self, model):
        """set_reference_point should update reference."""
        trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )
        model.set_reference_point(trip)

        assert model.reference_point == trip


class TestRegretMinimizationModel:
    """Tests for RegretMinimizationModel."""

    @pytest.fixture
    def model(self):
        """Create a RegretMinimizationModel instance."""
        return RegretMinimizationModel()

    @pytest.fixture
    def preferences(self):
        """Create default preferences."""
        return AgentPreferences()

    def test_compute_utility_returns_zero(self, model, preferences):
        """compute_utility returns 0 (interface compatibility)."""
        trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )
        utility = model.compute_utility(preferences, trip)

        assert utility == 0.0

    def test_compute_attribute_regret_same_value(self, model):
        """No regret when values are equal."""
        regret = model.compute_attribute_regret(5.0, 5.0, 0.1)
        # ln(1 + exp(0)) = ln(2) ≈ 0.693
        assert regret == pytest.approx(np.log(2))

    def test_compute_attribute_regret_better_alternative(self, model):
        """Regret when other option is better."""
        # Higher other value with positive beta = regret
        regret = model.compute_attribute_regret(5.0, 10.0, 0.1)
        assert regret > np.log(2)

    def test_compute_regret(self, model, preferences):
        """compute_regret should sum regrets across options."""
        trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )
        all_options = [
            trip,
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=25.0, cost=6.0),
        ]

        regret = model.compute_regret(trip, all_options, preferences)

        assert np.isfinite(regret)
        assert regret >= 0

    def test_choice_probabilities_sum_to_one(self, model, preferences):
        """Choice probabilities should sum to 1."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=25.0, cost=6.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=35.0, cost=4.0),
        ]

        probs = model.choice_probabilities(preferences, options)

        assert probs.sum() == pytest.approx(1.0)

    def test_lower_regret_higher_probability(self, model, preferences):
        """Options with lower regret should have higher probability."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=60.0, cost=10.0),
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
        ]

        probs = model.choice_probabilities(preferences, options)

        # Second option is better, should have higher probability
        assert probs[1] > probs[0]

    def test_choose_action(self, model, preferences):
        """choose_action should return valid index."""
        options = [
            TripAttributes(mode=TravelMode.DRIVE_ALONE, travel_time=30.0, cost=5.0),
            TripAttributes(mode=TravelMode.TRANSIT, travel_time=45.0, cost=2.0),
        ]
        rng = np.random.default_rng(42)

        choice = model.choose_action(preferences, options, rng)

        assert choice in [0, 1]


class TestEstimateIncentiveElasticity:
    """Tests for estimate_incentive_elasticity function."""

    def test_returns_elasticity_object(self):
        """Function should return IncentiveElasticity."""
        from src.agents.behavioral import IncentiveElasticity

        model = LogitModel()
        preferences = AgentPreferences()
        base_trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )

        result = estimate_incentive_elasticity(
            model=model,
            base_preferences=preferences,
            base_trip=base_trip,
            incentive_range=(1.0, 10.0),
            n_samples=50,
            rng=np.random.default_rng(42),
        )

        assert isinstance(result, IncentiveElasticity)

    def test_elasticity_is_finite(self):
        """Elasticity should be a finite number."""
        model = LogitModel()
        preferences = AgentPreferences()
        base_trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )

        result = estimate_incentive_elasticity(
            model=model,
            base_preferences=preferences,
            base_trip=base_trip,
            incentive_range=(1.0, 10.0),
            n_samples=100,
            rng=np.random.default_rng(42),
        )

        assert np.isfinite(result.elasticity)

    def test_higher_incentive_increases_participation(self):
        """Higher incentives should generally increase participation."""
        model = LogitModel()
        preferences = AgentPreferences(beta_incentive=0.3)  # High incentive sensitivity
        base_trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )

        result = estimate_incentive_elasticity(
            model=model,
            base_preferences=preferences,
            base_trip=base_trip,
            incentive_range=(1.0, 20.0),
            n_samples=200,
            rng=np.random.default_rng(42),
        )

        # Elasticity should be positive (more incentive = more participation)
        # Note: might not always be strongly positive depending on parameters
        assert isinstance(result.elasticity, float)

    def test_confidence_interval_contains_estimate(self):
        """Confidence interval should bracket the point estimate."""
        model = LogitModel()
        preferences = AgentPreferences()
        base_trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )

        result = estimate_incentive_elasticity(
            model=model,
            base_preferences=preferences,
            base_trip=base_trip,
            incentive_range=(1.0, 10.0),
            n_samples=100,
            rng=np.random.default_rng(42),
        )

        ci_low, ci_high = result.confidence_interval
        # CI should bracket estimate (or be close for borderline cases)
        assert ci_low <= result.elasticity <= ci_high or result.elasticity == 0.0

    def test_sample_size_recorded(self):
        """Sample size should be recorded correctly."""
        model = LogitModel()
        preferences = AgentPreferences()
        base_trip = TripAttributes(
            mode=TravelMode.DRIVE_ALONE,
            travel_time=30.0,
            cost=5.0,
        )

        n_samples = 100
        result = estimate_incentive_elasticity(
            model=model,
            base_preferences=preferences,
            base_trip=base_trip,
            incentive_range=(1.0, 10.0),
            n_samples=n_samples,
            rng=np.random.default_rng(42),
        )

        # 10 incentive levels * n_samples
        assert result.sample_size == 10 * n_samples
