"""
Base agent classes and behavioral models for transportation simulation.

Agents represent individual travelers making strategic decisions about
mode choice, route selection, departure time, and incentive participation.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray


class TravelMode(Enum):
    """Available travel modes for agents."""

    DRIVE_ALONE = auto()
    CARPOOL_DRIVER = auto()
    CARPOOL_PASSENGER = auto()
    TRANSIT = auto()
    WALK = auto()
    BIKE = auto()
    RIDESHARE = auto()


class DecisionRule(Enum):
    """Decision-making rules for bounded rationality."""

    UTILITY_MAX = auto()  # Pure utility maximization
    SOFTMAX = auto()  # Probabilistic choice ∝ exp(U/τ)
    EPSILON_GREEDY = auto()  # Explore with probability ε
    SATISFICING = auto()  # Accept first option above threshold


@dataclass
class AgentState:
    """Current state of an agent in the simulation."""

    position: tuple[float, float]  # (x, y) coordinates
    velocity: float = 0.0  # Current speed (m/s)
    heading: float = 0.0  # Direction (radians)
    mode: TravelMode = TravelMode.DRIVE_ALONE
    route: Optional[list[int]] = None
    departure_time: float = 0.0
    arrival_time: Optional[float] = None
    current_segment: Optional[int] = None
    distance_traveled: float = 0.0
    incentives_earned: float = 0.0


@dataclass
class AgentPreferences:
    """Agent preference parameters for utility computation."""

    # Value of time ($/hour)
    vot: float = 25.0
    # Preference weights (β coefficients)
    beta_time: float = -0.05  # Disutility per minute
    beta_cost: float = -0.10  # Disutility per dollar
    beta_incentive: float = 0.15  # Utility from incentives
    beta_comfort: float = 0.02  # Comfort preference
    beta_reliability: float = -0.03  # Disutility of unreliability
    # Mode-specific constants (alternative-specific constants)
    asc_carpool: float = -0.5
    asc_transit: float = -1.0
    asc_walk: float = -2.0
    # Behavioral parameters
    decision_rule: DecisionRule = DecisionRule.SOFTMAX
    temperature: float = 1.0  # Softmax temperature
    epsilon: float = 0.1  # Exploration probability
    satisficing_threshold: float = 0.0


@dataclass
class TripAttributes:
    """Attributes of a potential trip/action."""

    mode: TravelMode
    travel_time: float  # Expected time (minutes)
    cost: float  # Out-of-pocket cost ($)
    incentive: float = 0.0  # Incentive payment ($)
    comfort_score: float = 1.0  # Normalized comfort [0, 1]
    reliability: float = 0.9  # On-time probability


class BehavioralModel(ABC):
    """Abstract base class for behavioral response functions."""

    @abstractmethod
    def compute_utility(
        self,
        preferences: AgentPreferences,
        trip: TripAttributes,
    ) -> float:
        """Compute utility of a trip given agent preferences."""
        pass

    @abstractmethod
    def choose_action(
        self,
        preferences: AgentPreferences,
        options: list[TripAttributes],
        rng: np.random.Generator,
    ) -> int:
        """Choose among available options, returning index of chosen option."""
        pass


class LinearUtilityModel(BehavioralModel):
    """
    Linear-in-parameters utility model.

    U = β₀ + β₁·time + β₂·cost + β₃·incentive + β₄·comfort + ε
    """

    def compute_utility(
        self,
        preferences: AgentPreferences,
        trip: TripAttributes,
    ) -> float:
        """Compute deterministic utility component."""
        # Mode-specific constant
        asc = 0.0
        if (
            trip.mode == TravelMode.CARPOOL_DRIVER
            or trip.mode == TravelMode.CARPOOL_PASSENGER
        ):
            asc = preferences.asc_carpool
        elif trip.mode == TravelMode.TRANSIT:
            asc = preferences.asc_transit
        elif trip.mode == TravelMode.WALK:
            asc = preferences.asc_walk

        utility = (
            asc
            + preferences.beta_time * trip.travel_time
            + preferences.beta_cost * trip.cost
            + preferences.beta_incentive * trip.incentive
            + preferences.beta_comfort * trip.comfort_score
            + preferences.beta_reliability * (1 - trip.reliability) * trip.travel_time
        )
        return utility

    def choose_action(
        self,
        preferences: AgentPreferences,
        options: list[TripAttributes],
        rng: np.random.Generator,
    ) -> int:
        """Choose action based on decision rule."""
        if not options:
            raise ValueError("No options available for choice")

        utilities = np.array(
            [self.compute_utility(preferences, opt) for opt in options]
        )

        match preferences.decision_rule:
            case DecisionRule.UTILITY_MAX:
                return int(np.argmax(utilities))

            case DecisionRule.SOFTMAX:
                # Numerical stability: subtract max
                scaled = (utilities - utilities.max()) / preferences.temperature
                exp_util = np.exp(scaled)
                probs = exp_util / exp_util.sum()
                return int(rng.choice(len(options), p=probs))

            case DecisionRule.EPSILON_GREEDY:
                if rng.random() < preferences.epsilon:
                    return int(rng.choice(len(options)))
                return int(np.argmax(utilities))

            case DecisionRule.SATISFICING:
                for i, u in enumerate(utilities):
                    if u >= preferences.satisficing_threshold:
                        return i
                # If none satisfies, choose best available
                return int(np.argmax(utilities))

            case _:
                return int(np.argmax(utilities))


class BaseAgent(ABC):
    """
    Abstract base class for simulation agents.

    Agents are strategic actors who make decisions about travel behavior
    and respond to incentive mechanisms.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        preferences: Optional[AgentPreferences] = None,
        behavioral_model: Optional[BehavioralModel] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.preferences = preferences or AgentPreferences()
        self.behavioral_model = behavioral_model or LinearUtilityModel()
        self.rng = rng or np.random.default_rng()
        self.state = AgentState(position=(0.0, 0.0))
        self.history: list[dict[str, Any]] = []

    @abstractmethod
    def decide_mode(self, available_modes: list[TravelMode]) -> TravelMode:
        """Decide which travel mode to use."""
        pass

    @abstractmethod
    def decide_departure_time(
        self,
        desired_arrival: float,
        time_window: tuple[float, float],
    ) -> float:
        """Decide when to depart."""
        pass

    @abstractmethod
    def decide_route(
        self,
        origin: int,
        destination: int,
        available_routes: list[list[int]],
    ) -> list[int]:
        """Decide which route to take."""
        pass

    @abstractmethod
    def respond_to_incentive(
        self,
        incentive_type: str,
        incentive_amount: float,
        conditions: dict[str, Any],
    ) -> bool:
        """Decide whether to participate in an incentive program."""
        pass

    def update_state(self, **kwargs: Any) -> None:
        """Update agent state with new values."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def record_history(self, event_type: str, data: dict[str, Any]) -> None:
        """Record an event in agent history."""
        self.history.append(
            {
                "event_type": event_type,
                "timestamp": data.get("timestamp", 0),
                "data": data,
            }
        )

    def get_utility(self, trip: TripAttributes) -> float:
        """Compute utility for a trip option."""
        return self.behavioral_model.compute_utility(self.preferences, trip)

    def choose_option(self, options: list[TripAttributes]) -> int:
        """Choose among available options."""
        return self.behavioral_model.choose_action(self.preferences, options, self.rng)


@dataclass
class PopulationParameters:
    """Parameters for generating agent populations."""

    n_agents: int = 1000

    # Value of time distribution (lognormal)
    vot_mean: float = 25.0
    vot_std: float = 10.0

    # Beta coefficient distributions
    beta_time_mean: float = -0.05
    beta_time_std: float = 0.02
    beta_cost_mean: float = -0.10
    beta_cost_std: float = 0.03
    beta_incentive_mean: float = 0.15
    beta_incentive_std: float = 0.05

    # Decision rule probabilities
    prob_softmax: float = 0.7
    prob_utility_max: float = 0.2
    prob_epsilon_greedy: float = 0.1

    # Temperature distribution for softmax agents
    temperature_mean: float = 1.0
    temperature_std: float = 0.3


def generate_heterogeneous_preferences(
    params: PopulationParameters,
    rng: np.random.Generator,
) -> AgentPreferences:
    """Generate agent preferences from population distributions."""
    # Sample value of time from lognormal
    vot = rng.lognormal(
        mean=np.log(params.vot_mean)
        - 0.5 * np.log(1 + (params.vot_std / params.vot_mean) ** 2),
        sigma=np.sqrt(np.log(1 + (params.vot_std / params.vot_mean) ** 2)),
    )

    # Sample beta coefficients from normal (truncated to reasonable range)
    beta_time = np.clip(
        rng.normal(params.beta_time_mean, params.beta_time_std), -0.2, 0
    )
    beta_cost = np.clip(
        rng.normal(params.beta_cost_mean, params.beta_cost_std), -0.3, 0
    )
    beta_incentive = np.clip(
        rng.normal(params.beta_incentive_mean, params.beta_incentive_std), 0, 0.5
    )

    # Sample decision rule
    rule_probs = [
        params.prob_softmax,
        params.prob_utility_max,
        params.prob_epsilon_greedy,
    ]
    rule_idx = rng.choice(3, p=rule_probs)
    decision_rule = [
        DecisionRule.SOFTMAX,
        DecisionRule.UTILITY_MAX,
        DecisionRule.EPSILON_GREEDY,
    ][rule_idx]

    # Sample temperature for softmax
    temperature = max(0.1, rng.normal(params.temperature_mean, params.temperature_std))

    return AgentPreferences(
        vot=vot,
        beta_time=beta_time,
        beta_cost=beta_cost,
        beta_incentive=beta_incentive,
        decision_rule=decision_rule,
        temperature=temperature,
    )
