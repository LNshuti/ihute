"""
Advanced behavioral models for agent decision-making.

Implements various discrete choice models and behavioral response functions
from transportation economics and behavioral science literature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .base import AgentPreferences, BehavioralModel, TravelMode, TripAttributes


class LogitModel(BehavioralModel):
    """
    Multinomial logit model for discrete choice.

    Assumes IID Gumbel-distributed error terms, leading to the classic
    logit choice probability formula.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def compute_utility(
        self,
        preferences: AgentPreferences,
        trip: TripAttributes,
    ) -> float:
        """Compute systematic utility component."""
        # Mode-specific constant
        asc = self._get_asc(trip.mode, preferences)

        utility = (
            asc
            + preferences.beta_time * trip.travel_time
            + preferences.beta_cost * trip.cost
            + preferences.beta_incentive * trip.incentive
            + preferences.beta_comfort * trip.comfort_score
        )
        return utility

    def _get_asc(self, mode: TravelMode, preferences: AgentPreferences) -> float:
        """Get alternative-specific constant for a mode."""
        match mode:
            case TravelMode.CARPOOL_DRIVER | TravelMode.CARPOOL_PASSENGER:
                return preferences.asc_carpool
            case TravelMode.TRANSIT:
                return preferences.asc_transit
            case TravelMode.WALK:
                return preferences.asc_walk
            case _:
                return 0.0

    def choice_probabilities(
        self,
        preferences: AgentPreferences,
        options: list[TripAttributes],
    ) -> NDArray:
        """Compute logit choice probabilities."""
        utilities = np.array([
            self.compute_utility(preferences, opt) for opt in options
        ])

        # Scale utilities
        scaled = utilities / self.scale

        # Numerical stability
        scaled = scaled - scaled.max()
        exp_util = np.exp(scaled)

        return exp_util / exp_util.sum()

    def choose_action(
        self,
        preferences: AgentPreferences,
        options: list[TripAttributes],
        rng: np.random.Generator,
    ) -> int:
        """Choose according to logit probabilities."""
        probs = self.choice_probabilities(preferences, options)
        return int(rng.choice(len(options), p=probs))


class MixedLogitModel(BehavioralModel):
    """
    Mixed logit model with random taste variation.

    Allows for heterogeneity in preferences across the population and
    correlation in unobserved factors across alternatives.
    """

    def __init__(
        self,
        n_draws: int = 100,
        coefficient_distributions: Optional[dict[str, tuple[str, float, float]]] = None,
    ):
        """
        Initialize mixed logit model.

        Args:
            n_draws: Number of Monte Carlo draws for simulation
            coefficient_distributions: Dict mapping coefficient names to
                (distribution_type, mean, std) tuples. Supported types:
                'normal', 'lognormal', 'triangular'
        """
        self.n_draws = n_draws
        self.coefficient_distributions = coefficient_distributions or {
            "beta_time": ("normal", -0.05, 0.02),
            "beta_cost": ("lognormal", -0.1, 0.05),
        }

    def compute_utility(
        self,
        preferences: AgentPreferences,
        trip: TripAttributes,
    ) -> float:
        """Compute base utility (used for single evaluation)."""
        return (
            preferences.beta_time * trip.travel_time
            + preferences.beta_cost * trip.cost
            + preferences.beta_incentive * trip.incentive
        )

    def _draw_coefficients(
        self,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Draw random coefficients from specified distributions."""
        coeffs = {}

        for name, (dist_type, mean, std) in self.coefficient_distributions.items():
            match dist_type:
                case "normal":
                    coeffs[name] = rng.normal(mean, std)
                case "lognormal":
                    # Ensure negative coefficients for costs
                    sign = -1 if mean < 0 else 1
                    coeffs[name] = sign * rng.lognormal(
                        np.log(abs(mean)), std
                    )
                case "triangular":
                    coeffs[name] = rng.triangular(
                        mean - std, mean, mean + std
                    )
                case _:
                    coeffs[name] = mean

        return coeffs

    def _compute_utility_with_coeffs(
        self,
        trip: TripAttributes,
        coeffs: dict[str, float],
        preferences: AgentPreferences,
    ) -> float:
        """Compute utility with specific coefficient values."""
        beta_time = coeffs.get("beta_time", preferences.beta_time)
        beta_cost = coeffs.get("beta_cost", preferences.beta_cost)

        return (
            beta_time * trip.travel_time
            + beta_cost * trip.cost
            + preferences.beta_incentive * trip.incentive
        )

    def choice_probabilities(
        self,
        preferences: AgentPreferences,
        options: list[TripAttributes],
        rng: np.random.Generator,
    ) -> NDArray:
        """
        Compute mixed logit choice probabilities via simulation.

        Averages logit probabilities over draws from the mixing distribution.
        """
        n_options = len(options)
        prob_accumulator = np.zeros(n_options)

        for _ in range(self.n_draws):
            coeffs = self._draw_coefficients(rng)

            utilities = np.array([
                self._compute_utility_with_coeffs(opt, coeffs, preferences)
                for opt in options
            ])

            # Logit probabilities for this draw
            scaled = utilities - utilities.max()
            exp_util = np.exp(scaled)
            probs = exp_util / exp_util.sum()

            prob_accumulator += probs

        return prob_accumulator / self.n_draws

    def choose_action(
        self,
        preferences: AgentPreferences,
        options: list[TripAttributes],
        rng: np.random.Generator,
    ) -> int:
        """Choose according to simulated mixed logit probabilities."""
        probs = self.choice_probabilities(preferences, options, rng)
        return int(rng.choice(len(options), p=probs))


class ProspectTheoryModel(BehavioralModel):
    """
    Prospect theory model for reference-dependent choice.

    Models loss aversion and probability weighting as described by
    Kahneman and Tversky (1979, 1992).
    """

    def __init__(
        self,
        reference_point: Optional[TripAttributes] = None,
        loss_aversion: float = 2.25,  # λ
        diminishing_sensitivity: float = 0.88,  # α
        probability_weight_param: float = 0.65,  # γ
    ):
        """
        Initialize prospect theory model.

        Args:
            reference_point: Reference trip for gain/loss evaluation
            loss_aversion: Loss aversion coefficient (λ > 1)
            diminishing_sensitivity: Curvature of value function (0 < α < 1)
            probability_weight_param: Probability weighting parameter
        """
        self.reference_point = reference_point
        self.loss_aversion = loss_aversion
        self.alpha = diminishing_sensitivity
        self.gamma = probability_weight_param

    def value_function(self, x: float) -> float:
        """
        Prospect theory value function.

        v(x) = x^α        if x >= 0 (gains)
        v(x) = -λ(-x)^α   if x < 0 (losses)
        """
        if x >= 0:
            return x ** self.alpha
        else:
            return -self.loss_aversion * ((-x) ** self.alpha)

    def probability_weight(self, p: float) -> float:
        """
        Probability weighting function (Prelec, 1998).

        w(p) = exp(-(-ln(p))^γ)
        """
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        return np.exp(-((-np.log(p)) ** self.gamma))

    def compute_utility(
        self,
        preferences: AgentPreferences,
        trip: TripAttributes,
    ) -> float:
        """Compute prospect theory value."""
        if self.reference_point is None:
            # No reference point - use standard utility
            return (
                preferences.beta_time * trip.travel_time
                + preferences.beta_cost * trip.cost
                + preferences.beta_incentive * trip.incentive
            )

        # Compute gains/losses relative to reference
        time_diff = self.reference_point.travel_time - trip.travel_time  # Less time = gain
        cost_diff = self.reference_point.cost - trip.cost  # Less cost = gain
        incentive_diff = trip.incentive - self.reference_point.incentive  # More incentive = gain

        # Apply value function to each dimension
        time_value = self.value_function(time_diff * abs(preferences.beta_time))
        cost_value = self.value_function(cost_diff * abs(preferences.beta_cost))
        incentive_value = self.value_function(incentive_diff * preferences.beta_incentive)

        return time_value + cost_value + incentive_value

    def choose_action(
        self,
        preferences: AgentPreferences,
        options: list[TripAttributes],
        rng: np.random.Generator,
    ) -> int:
        """Choose based on prospect theory values."""
        values = np.array([
            self.compute_utility(preferences, opt) for opt in options
        ])

        # Softmax choice based on values
        scaled = (values - values.max()) / preferences.temperature
        exp_val = np.exp(scaled)
        probs = exp_val / exp_val.sum()

        return int(rng.choice(len(options), p=probs))

    def set_reference_point(self, trip: TripAttributes) -> None:
        """Update the reference point."""
        self.reference_point = trip


class RegretMinimizationModel(BehavioralModel):
    """
    Random regret minimization model (Chorus, 2010).

    Assumes decision-makers minimize anticipated regret from not choosing
    the best alternative on each attribute.
    """

    def __init__(self, regret_scale: float = 1.0):
        """
        Initialize regret minimization model.

        Args:
            regret_scale: Scale parameter for regret function
        """
        self.regret_scale = regret_scale

    def compute_attribute_regret(
        self,
        own_value: float,
        other_value: float,
        beta: float,
    ) -> float:
        """
        Compute regret for one attribute comparison.

        R = ln(1 + exp(β(x_other - x_own)))
        """
        diff = beta * (other_value - own_value)
        # Numerical stability
        if diff > 20:
            return diff
        return np.log(1 + np.exp(diff))

    def compute_utility(
        self,
        preferences: AgentPreferences,
        trip: TripAttributes,
    ) -> float:
        """
        This model doesn't use traditional utility.
        Returns 0 for interface compatibility.
        """
        return 0.0

    def compute_regret(
        self,
        trip: TripAttributes,
        all_options: list[TripAttributes],
        preferences: AgentPreferences,
    ) -> float:
        """Compute total regret for choosing this trip."""
        total_regret = 0.0

        for other in all_options:
            if other is trip:
                continue

            # Time regret (less time is better, so beta_time is negative)
            total_regret += self.compute_attribute_regret(
                trip.travel_time,
                other.travel_time,
                -preferences.beta_time,  # Flip sign: lower time = better
            )

            # Cost regret
            total_regret += self.compute_attribute_regret(
                trip.cost,
                other.cost,
                -preferences.beta_cost,
            )

            # Incentive regret (more is better)
            total_regret += self.compute_attribute_regret(
                trip.incentive,
                other.incentive,
                preferences.beta_incentive,
            )

        return total_regret * self.regret_scale

    def choice_probabilities(
        self,
        preferences: AgentPreferences,
        options: list[TripAttributes],
    ) -> NDArray:
        """Compute choice probabilities based on regret."""
        regrets = np.array([
            self.compute_regret(opt, options, preferences)
            for opt in options
        ])

        # Lower regret = higher probability (negative regret in exp)
        scaled = -regrets
        scaled = scaled - scaled.max()
        exp_neg_regret = np.exp(scaled)

        return exp_neg_regret / exp_neg_regret.sum()

    def choose_action(
        self,
        preferences: AgentPreferences,
        options: list[TripAttributes],
        rng: np.random.Generator,
    ) -> int:
        """Choose to minimize regret."""
        probs = self.choice_probabilities(preferences, options)
        return int(rng.choice(len(options), p=probs))


@dataclass
class IncentiveElasticity:
    """Computed elasticity of behavior with respect to incentive changes."""
    base_participation_rate: float
    elasticity: float  # % change in participation per % change in incentive
    confidence_interval: tuple[float, float]
    sample_size: int


def estimate_incentive_elasticity(
    model: BehavioralModel,
    base_preferences: AgentPreferences,
    base_trip: TripAttributes,
    incentive_range: tuple[float, float],
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> IncentiveElasticity:
    """
    Estimate elasticity of choice with respect to incentive level.

    Uses simulation to estimate how participation rates change
    as incentive amounts vary.

    Args:
        model: Behavioral model to use
        base_preferences: Representative agent preferences
        base_trip: Base trip attributes (incentive will be varied)
        incentive_range: (min, max) incentive amounts to test
        n_samples: Number of simulation samples
        rng: Random number generator

    Returns:
        IncentiveElasticity with computed values
    """
    if rng is None:
        rng = np.random.default_rng()

    # Create comparison option (no incentive alternative)
    no_incentive_trip = TripAttributes(
        mode=base_trip.mode,
        travel_time=base_trip.travel_time * 0.9,  # Slightly faster
        cost=base_trip.cost,
        incentive=0.0,
    )

    # Test at multiple incentive levels
    incentive_levels = np.linspace(incentive_range[0], incentive_range[1], 10)
    participation_rates = []

    for incentive in incentive_levels:
        # Create trip with this incentive
        test_trip = TripAttributes(
            mode=base_trip.mode,
            travel_time=base_trip.travel_time,
            cost=base_trip.cost,
            incentive=incentive,
        )

        options = [no_incentive_trip, test_trip]

        # Simulate choices
        n_participate = 0
        for _ in range(n_samples):
            choice = model.choose_action(base_preferences, options, rng)
            if choice == 1:  # Chose incentivized option
                n_participate += 1

        participation_rates.append(n_participate / n_samples)

    # Compute elasticity via regression
    rates = np.array(participation_rates)
    incentives = np.array(incentive_levels)

    # Log-log regression for elasticity
    # Only use positive rates and incentives
    valid = (rates > 0) & (incentives > 0)
    if valid.sum() < 3:
        elasticity = 0.0
        ci = (0.0, 0.0)
    else:
        log_rates = np.log(rates[valid])
        log_incentives = np.log(incentives[valid])

        # Simple OLS
        n = len(log_rates)
        x_mean = log_incentives.mean()
        y_mean = log_rates.mean()

        numerator = ((log_incentives - x_mean) * (log_rates - y_mean)).sum()
        denominator = ((log_incentives - x_mean) ** 2).sum()

        elasticity = numerator / denominator if denominator > 0 else 0.0

        # Rough confidence interval
        residuals = log_rates - (y_mean + elasticity * (log_incentives - x_mean))
        se = np.sqrt((residuals ** 2).sum() / (n - 2)) / np.sqrt(denominator)
        ci = (elasticity - 1.96 * se, elasticity + 1.96 * se)

    return IncentiveElasticity(
        base_participation_rate=rates[len(rates) // 2],
        elasticity=elasticity,
        confidence_interval=ci,
        sample_size=n_samples * len(incentive_levels),
    )
