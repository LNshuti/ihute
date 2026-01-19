"""
Commuter agent implementation for daily travel simulation.

Commuters make decisions about mode, departure time, and route based on
their preferences and available incentives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import (
    AgentPreferences,
    AgentState,
    BaseAgent,
    BehavioralModel,
    LinearUtilityModel,
    TravelMode,
    TripAttributes,
)


@dataclass
class CommuterProfile:
    """Profile information for a commuter agent."""

    home_location: tuple[float, float]
    work_location: tuple[float, float]
    desired_arrival_time: float  # Seconds from midnight
    flexibility_window: float = 1800  # Seconds (+/- from desired)
    has_car: bool = True
    has_transit_pass: bool = False
    carpool_eligible: bool = True
    work_days: list[int] = None  # 0=Monday, ..., 6=Sunday

    # Demographic attributes (from population-dyna)
    home_zcta: Optional[str] = None  # 5-digit ZCTA code
    household_income: Optional[float] = None  # Annual income in USD
    poverty_rate: Optional[float] = None  # ZCTA poverty rate (0-1)
    income_quintile: Optional[int] = None  # 1-5 (1=lowest, 5=highest)

    def __post_init__(self):
        if self.work_days is None:
            self.work_days = [0, 1, 2, 3, 4]  # Mon-Fri


class CommuterAgent(BaseAgent):
    """
    Agent representing a daily commuter.

    Commuters travel between home and work locations, making strategic
    decisions about mode, timing, and route. They respond to incentives
    based on their preference parameters.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        preferences: Optional[AgentPreferences] = None,
        profile: Optional[CommuterProfile] = None,
        behavioral_model: Optional[BehavioralModel] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(agent_id, preferences, behavioral_model, rng)
        self.profile = profile or CommuterProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,  # 8 AM
        )
        self.state.position = self.profile.home_location

        # Track daily statistics
        self.trips_completed: int = 0
        self.total_travel_time: float = 0.0
        self.total_incentives: float = 0.0
        self.mode_history: list[TravelMode] = []

    def decide_mode(
        self,
        available_modes: list[TravelMode],
        trip_options: Optional[dict[TravelMode, TripAttributes]] = None,
    ) -> TravelMode:
        """
        Decide which travel mode to use for the commute.

        Args:
            available_modes: List of modes the agent can choose from
            trip_options: Optional pre-computed trip attributes per mode

        Returns:
            Selected travel mode
        """
        if not available_modes:
            return TravelMode.DRIVE_ALONE

        # Filter to available modes
        filtered_modes = [m for m in available_modes if self._mode_available(m)]
        if not filtered_modes:
            filtered_modes = [TravelMode.DRIVE_ALONE]

        # If no trip options provided, generate estimates
        if trip_options is None:
            trip_options = {mode: self._estimate_trip(mode) for mode in filtered_modes}

        # Convert to list of options
        options = [trip_options[m] for m in filtered_modes if m in trip_options]

        if not options:
            return filtered_modes[0]

        # Choose using behavioral model
        choice_idx = self.choose_option(options)
        return filtered_modes[choice_idx]

    def _mode_available(self, mode: TravelMode) -> bool:
        """Check if a mode is available to this agent."""
        if mode in [TravelMode.DRIVE_ALONE, TravelMode.CARPOOL_DRIVER]:
            return self.profile.has_car
        if mode == TravelMode.CARPOOL_PASSENGER:
            return self.profile.carpool_eligible
        if mode == TravelMode.TRANSIT:
            return True  # Assume transit is always available
        return True

    def _estimate_trip(self, mode: TravelMode) -> TripAttributes:
        """Estimate trip attributes for a mode (placeholder)."""
        # Base estimates - would be computed from network in real simulation
        base_time = 30.0  # minutes
        base_cost = 5.0  # dollars

        match mode:
            case TravelMode.DRIVE_ALONE:
                return TripAttributes(
                    mode=mode,
                    travel_time=base_time,
                    cost=base_cost,
                    comfort_score=0.9,
                    reliability=0.7,
                )
            case TravelMode.CARPOOL_DRIVER:
                return TripAttributes(
                    mode=mode,
                    travel_time=base_time * 1.1,  # Slight detour
                    cost=base_cost * 0.5,  # Shared cost
                    comfort_score=0.7,
                    reliability=0.65,
                )
            case TravelMode.CARPOOL_PASSENGER:
                return TripAttributes(
                    mode=mode,
                    travel_time=base_time * 1.2,
                    cost=base_cost * 0.3,
                    comfort_score=0.6,
                    reliability=0.6,
                )
            case TravelMode.TRANSIT:
                return TripAttributes(
                    mode=mode,
                    travel_time=base_time * 1.5,
                    cost=2.0,
                    comfort_score=0.4,
                    reliability=0.8,
                )
            case _:
                return TripAttributes(
                    mode=mode,
                    travel_time=base_time,
                    cost=base_cost,
                )

    def decide_departure_time(
        self,
        desired_arrival: Optional[float] = None,
        time_window: Optional[tuple[float, float]] = None,
        expected_travel_time: float = 1800,  # 30 min default
        incentive_schedule: Optional[dict[float, float]] = None,
    ) -> float:
        """
        Decide when to depart.

        Args:
            desired_arrival: Target arrival time (seconds from midnight)
            time_window: (earliest, latest) departure bounds
            expected_travel_time: Expected trip duration in seconds
            incentive_schedule: Optional time -> incentive mapping

        Returns:
            Departure time in seconds from midnight
        """
        if desired_arrival is None:
            desired_arrival = self.profile.desired_arrival_time

        # Default departure window
        if time_window is None:
            earliest = (
                desired_arrival - expected_travel_time - self.profile.flexibility_window
            )
            latest = (
                desired_arrival
                - expected_travel_time
                + self.profile.flexibility_window * 0.5
            )
            time_window = (max(0, earliest), latest)

        # Ideal departure time (to arrive exactly on time)
        ideal_departure = desired_arrival - expected_travel_time

        # If no incentive schedule, depart at ideal time with some noise
        if incentive_schedule is None:
            noise = self.rng.normal(0, 300)  # 5 min std dev
            return np.clip(ideal_departure + noise, time_window[0], time_window[1])

        # With incentives, evaluate options at different times
        time_points = np.linspace(time_window[0], time_window[1], 10)
        options = []

        for t in time_points:
            # Compute arrival time
            arrival = t + expected_travel_time

            # Schedule delay penalty (early/late)
            delay = abs(arrival - desired_arrival)
            delay_penalty = delay / 60 * 2  # $2/minute equivalent

            # Get incentive at this time
            incentive = incentive_schedule.get(t, 0)

            options.append(
                TripAttributes(
                    mode=TravelMode.DRIVE_ALONE,
                    travel_time=expected_travel_time / 60,
                    cost=delay_penalty,
                    incentive=incentive,
                )
            )

        # Choose optimal departure time
        choice_idx = self.choose_option(options)
        return time_points[choice_idx]

    def decide_route(
        self,
        origin: int,
        destination: int,
        available_routes: list[list[int]],
        route_attributes: Optional[list[TripAttributes]] = None,
    ) -> list[int]:
        """
        Decide which route to take.

        Args:
            origin: Origin node ID
            destination: Destination node ID
            available_routes: List of routes (each a list of node IDs)
            route_attributes: Optional pre-computed attributes per route

        Returns:
            Selected route as list of node IDs
        """
        if not available_routes:
            return []

        if len(available_routes) == 1:
            return available_routes[0]

        # Generate route attributes if not provided
        if route_attributes is None:
            route_attributes = [
                self._estimate_route_attributes(route) for route in available_routes
            ]

        # Choose using behavioral model
        choice_idx = self.choose_option(route_attributes)
        return available_routes[choice_idx]

    def _estimate_route_attributes(self, route: list[int]) -> TripAttributes:
        """Estimate attributes for a route (placeholder)."""
        # Would be computed from network in real simulation
        route_length = len(route)
        return TripAttributes(
            mode=self.state.mode,
            travel_time=route_length * 2,  # 2 min per segment
            cost=route_length * 0.1,  # Fuel cost estimate
            reliability=0.8 - route_length * 0.01,
        )

    def respond_to_incentive(
        self,
        incentive_type: str,
        incentive_amount: float,
        conditions: dict[str, Any],
    ) -> bool:
        """
        Decide whether to participate in an incentive program.

        Args:
            incentive_type: Type of incentive (carpool, pacer, departure_shift, etc.)
            incentive_amount: Dollar value of incentive
            conditions: Conditions for receiving incentive

        Returns:
            True if agent decides to participate
        """
        # Check eligibility
        if not self._check_incentive_eligibility(incentive_type, conditions):
            return False

        # Compute participation utility
        participation_utility = self._compute_incentive_utility(
            incentive_type, incentive_amount, conditions
        )

        # Compute non-participation utility (status quo)
        non_participation_utility = 0.0

        # Decision based on utility comparison
        if self.preferences.decision_rule == self.preferences.decision_rule.SOFTMAX:
            # Probabilistic choice
            utilities = np.array([non_participation_utility, participation_utility])
            scaled = (utilities - utilities.max()) / self.preferences.temperature
            probs = np.exp(scaled) / np.exp(scaled).sum()
            return bool(self.rng.random() < probs[1])
        else:
            # Deterministic choice
            return participation_utility > non_participation_utility

    def _check_incentive_eligibility(
        self,
        incentive_type: str,
        conditions: dict[str, Any],
    ) -> bool:
        """Check if agent is eligible for an incentive."""
        match incentive_type:
            case "carpool":
                return self.profile.carpool_eligible
            case "pacer":
                return self.profile.has_car
            case "transit":
                return True
            case "departure_shift":
                # Check if shift is within flexibility window
                required_shift = conditions.get("required_shift", 0)
                return abs(required_shift) <= self.profile.flexibility_window
            case _:
                return True

    def _compute_incentive_utility(
        self,
        incentive_type: str,
        incentive_amount: float,
        conditions: dict[str, Any],
    ) -> float:
        """Compute utility from participating in an incentive."""
        base_utility = self.preferences.beta_incentive * incentive_amount

        # Adjust for inconvenience costs
        match incentive_type:
            case "carpool":
                # Coordination cost
                coordination_cost = conditions.get("coordination_cost", 5.0)
                base_utility -= coordination_cost * 0.1
            case "departure_shift":
                # Schedule delay cost
                shift_minutes = abs(conditions.get("required_shift", 0)) / 60
                base_utility -= shift_minutes * 0.5
            case "pacer":
                # Driving constraint cost
                constraint_level = conditions.get("constraint_level", 0.5)
                base_utility -= constraint_level * 2.0

        return base_utility

    def complete_trip(
        self,
        travel_time: float,
        mode: TravelMode,
        incentive_earned: float = 0.0,
    ) -> None:
        """Record completion of a trip."""
        self.trips_completed += 1
        self.total_travel_time += travel_time
        self.total_incentives += incentive_earned
        self.mode_history.append(mode)

        self.record_history(
            "trip_completed",
            {
                "travel_time": travel_time,
                "mode": mode.name,
                "incentive": incentive_earned,
            },
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics for this agent."""
        return {
            "agent_id": self.id,
            "trips_completed": self.trips_completed,
            "total_travel_time": self.total_travel_time,
            "avg_travel_time": self.total_travel_time / max(1, self.trips_completed),
            "total_incentives": self.total_incentives,
            "mode_distribution": {
                mode.name: self.mode_history.count(mode)
                / max(1, len(self.mode_history))
                for mode in TravelMode
                if mode in self.mode_history
            },
            "preferences": {
                "vot": self.preferences.vot,
                "beta_incentive": self.preferences.beta_incentive,
                "decision_rule": self.preferences.decision_rule.name,
            },
        }


def create_demographic_aware_preferences(
    demographics: Any,  # ZCTADemographics
    base_params: Optional[Any] = None,  # PopulationParameters
    rng: Optional[np.random.Generator] = None,
) -> AgentPreferences:
    """
    Generate preferences calibrated to demographic attributes.

    Income influences:
    - Value of Time (VOT): 50% of hourly wage (DOT guidance)
    - Cost sensitivity: Higher for low-income (stronger response to costs)
    - Incentive sensitivity: Higher for low-income (stronger response to rewards)

    Args:
        demographics: ZCTADemographics object with income/poverty data
        base_params: Base population parameters (uses defaults if None)
        rng: Random number generator

    Returns:
        AgentPreferences calibrated to demographics
    """
    from .base import (
        AgentPreferences,
        DecisionRule,
        PopulationParameters,
    )

    if rng is None:
        rng = np.random.default_rng()

    if base_params is None:
        base_params = PopulationParameters()

    # Value of Time: 50% of hourly wage
    hourly_wage = demographics.median_household_income_est / 2080  # Annual to hourly
    vot = hourly_wage * 0.50

    # Use calibrated parameters if available, otherwise derive from income quintile
    if demographics.avg_beta_incentive is not None:
        beta_incentive_base = demographics.avg_beta_incentive
    else:
        # Inverse relationship: Q1 (low income) → 1.5x, Q5 (high income) → 0.5x
        incentive_multiplier = 2.0 - (demographics.income_quintile * 0.3)
        beta_incentive_base = base_params.beta_incentive_mean * incentive_multiplier

    # Cost sensitivity (inverse to income)
    cost_multiplier = 2.0 - (demographics.income_quintile * 0.3)
    beta_cost_base = base_params.beta_cost_mean * cost_multiplier

    # Add noise to avoid deterministic agents within same ZCTA
    beta_time = np.clip(
        rng.normal(base_params.beta_time_mean, base_params.beta_time_std), -0.2, 0
    )
    beta_cost = np.clip(rng.normal(beta_cost_base, base_params.beta_cost_std), -0.5, 0)
    beta_incentive = np.clip(
        rng.normal(beta_incentive_base, base_params.beta_incentive_std), 0, 0.5
    )

    # Sample decision rule
    rule_probs = [
        base_params.prob_softmax,
        base_params.prob_utility_max,
        base_params.prob_epsilon_greedy,
    ]
    rule_idx = rng.choice(3, p=rule_probs)
    decision_rule = [
        DecisionRule.SOFTMAX,
        DecisionRule.UTILITY_MAX,
        DecisionRule.EPSILON_GREEDY,
    ][rule_idx]

    # Sample temperature
    temperature = max(0.1, rng.normal(base_params.temperature_mean, base_params.temperature_std))

    return AgentPreferences(
        vot=vot,
        beta_time=beta_time,
        beta_cost=beta_cost,
        beta_incentive=beta_incentive,
        decision_rule=decision_rule,
        temperature=temperature,
    )


def create_commuter_population(
    n_agents: int,
    home_region: tuple[
        tuple[float, float], tuple[float, float]
    ],  # ((min_x, min_y), (max_x, max_y))
    work_region: tuple[tuple[float, float], tuple[float, float]],
    arrival_time_dist: tuple[float, float] = (8 * 3600, 1800),  # (mean, std)
    rng: Optional[np.random.Generator] = None,
) -> list[CommuterAgent]:
    """
    Create a population of commuter agents with heterogeneous preferences.

    Args:
        n_agents: Number of agents to create
        home_region: Bounding box for home locations
        work_region: Bounding box for work locations
        arrival_time_dist: (mean, std) of desired arrival times
        rng: Random number generator

    Returns:
        List of CommuterAgent instances
    """
    from .base import PopulationParameters, generate_heterogeneous_preferences

    if rng is None:
        rng = np.random.default_rng()

    pop_params = PopulationParameters(n_agents=n_agents)
    agents = []

    for i in range(n_agents):
        # Generate random home and work locations
        home = (
            rng.uniform(home_region[0][0], home_region[1][0]),
            rng.uniform(home_region[0][1], home_region[1][1]),
        )
        work = (
            rng.uniform(work_region[0][0], work_region[1][0]),
            rng.uniform(work_region[0][1], work_region[1][1]),
        )

        # Generate arrival time
        arrival_time = rng.normal(arrival_time_dist[0], arrival_time_dist[1])
        arrival_time = np.clip(arrival_time, 6 * 3600, 10 * 3600)  # 6 AM - 10 AM

        # Generate preferences
        preferences = generate_heterogeneous_preferences(pop_params, rng)

        # Create profile
        profile = CommuterProfile(
            home_location=home,
            work_location=work,
            desired_arrival_time=arrival_time,
            has_car=rng.random() > 0.1,  # 90% have cars
            has_transit_pass=rng.random() > 0.7,  # 30% have transit passes
            carpool_eligible=rng.random() > 0.2,  # 80% eligible for carpool
        )

        agent = CommuterAgent(
            agent_id=f"commuter_{i:05d}",
            preferences=preferences,
            profile=profile,
            rng=rng,
        )
        agents.append(agent)

    return agents


def create_demographic_commuter_population(
    n_agents: int,
    home_region: tuple[tuple[float, float], tuple[float, float]],
    work_region: tuple[tuple[float, float], tuple[float, float]],
    arrival_time_dist: tuple[float, float] = (8 * 3600, 1800),
    warehouse_path: str = "warehouse.duckdb",
    rng: Optional[np.random.Generator] = None,
) -> list[CommuterAgent]:
    """
    Create a population of commuter agents with demographics-based preferences.

    Agents are sampled from Tennessee ZCTAs and assigned demographic attributes
    (income, poverty rate, quintile) that calibrate their behavioral parameters.

    Args:
        n_agents: Number of agents to create
        home_region: Bounding box for home locations
        work_region: Bounding box for work locations
        arrival_time_dist: (mean, std) of desired arrival times
        warehouse_path: Path to DuckDB warehouse with demographics
        rng: Random number generator

    Returns:
        List of CommuterAgent instances with demographic attributes
    """
    from pathlib import Path

    from .base import PopulationParameters

    # Import demographics module
    import sys
    from pathlib import Path as P

    # Add src to path if not already there
    src_path = P(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from data.demographics import DemographicLoader

    if rng is None:
        rng = np.random.default_rng()

    pop_params = PopulationParameters(n_agents=n_agents)

    # Load demographics
    with DemographicLoader(warehouse_path) as loader:
        # Sample ZCTAs for the population
        zcta_samples = loader.sample_zctas(n=n_agents, state_prefix="37")

    agents = []

    for i, demographics in enumerate(zcta_samples):
        # Generate random home and work locations
        home = (
            rng.uniform(home_region[0][0], home_region[1][0]),
            rng.uniform(home_region[0][1], home_region[1][1]),
        )
        work = (
            rng.uniform(work_region[0][0], work_region[1][0]),
            rng.uniform(work_region[0][1], work_region[1][1]),
        )

        # Generate arrival time
        arrival_time = rng.normal(arrival_time_dist[0], arrival_time_dist[1])
        arrival_time = np.clip(arrival_time, 6 * 3600, 10 * 3600)  # 6 AM - 10 AM

        # Generate demographic-aware preferences
        preferences = create_demographic_aware_preferences(
            demographics, pop_params, rng
        )

        # Car ownership depends on income (higher income → more likely)
        # Logistic function: P(car) = 1 / (1 + exp(-(income - 40000) / 10000))
        car_prob = 1 / (1 + np.exp(-(demographics.median_household_income_est - 40000) / 10000))
        has_car = rng.random() < car_prob

        # Create profile with demographic attributes
        profile = CommuterProfile(
            home_location=home,
            work_location=work,
            desired_arrival_time=arrival_time,
            has_car=has_car,
            has_transit_pass=rng.random() > 0.7,  # 30% have transit passes
            carpool_eligible=rng.random() > 0.2,  # 80% eligible for carpool
            # Demographic fields
            home_zcta=demographics.zcta_code,
            household_income=demographics.median_household_income_est,
            poverty_rate=demographics.poverty_rate,
            income_quintile=demographics.income_quintile,
        )

        agent = CommuterAgent(
            agent_id=f"commuter_{i:05d}",
            preferences=preferences,
            profile=profile,
            rng=rng,
        )
        agents.append(agent)

    return agents
