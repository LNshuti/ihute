"""
Pacer driver agent for flow stabilization incentive programs.

Pacer drivers are incentivized to maintain steady speeds that help
dampen traffic waves and improve overall traffic flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import (
    AgentPreferences,
    BaseAgent,
    BehavioralModel,
    TravelMode,
    TripAttributes,
)


@dataclass
class PacerProfile:
    """Profile information for a pacer driver agent."""
    home_location: tuple[float, float]
    work_location: tuple[float, float]
    desired_arrival_time: float  # Seconds from midnight
    flexibility_window: float = 3600  # Pacer drivers typically more flexible
    enrolled_corridors: list[str] = None  # Corridor IDs for pacer program
    max_speed_reduction: float = 0.2  # Max fraction speed reduction accepted
    driving_skill: float = 0.8  # Ability to maintain steady speed [0, 1]

    def __post_init__(self):
        if self.enrolled_corridors is None:
            self.enrolled_corridors = []


@dataclass
class PacerPerformance:
    """Performance metrics for a pacer driver."""
    trips_as_pacer: int = 0
    total_pacer_miles: float = 0.0
    total_pacer_earnings: float = 0.0
    avg_smoothness_score: float = 0.0
    speed_variance_history: list[float] = None

    def __post_init__(self):
        if self.speed_variance_history is None:
            self.speed_variance_history = []


class PacerAgent(BaseAgent):
    """
    Agent representing a pacer driver in a flow stabilization program.

    Pacer drivers receive incentives for maintaining steady speeds that
    help dampen traffic oscillations. They balance the inconvenience of
    constrained driving against monetary rewards.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        preferences: Optional[AgentPreferences] = None,
        profile: Optional[PacerProfile] = None,
        behavioral_model: Optional[BehavioralModel] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(agent_id, preferences, behavioral_model, rng)
        self.profile = profile or PacerProfile(
            home_location=(0.0, 0.0),
            work_location=(10.0, 10.0),
            desired_arrival_time=8 * 3600,
        )
        self.state.position = self.profile.home_location
        self.performance = PacerPerformance()

        # Current pacer session state
        self.is_pacing: bool = False
        self.current_target_speed: Optional[float] = None
        self.speed_history: list[float] = []

    def decide_mode(
        self,
        available_modes: list[TravelMode],
        trip_options: Optional[dict[TravelMode, TripAttributes]] = None,
    ) -> TravelMode:
        """Pacer drivers typically drive alone."""
        if TravelMode.DRIVE_ALONE in available_modes:
            return TravelMode.DRIVE_ALONE
        return available_modes[0] if available_modes else TravelMode.DRIVE_ALONE

    def decide_departure_time(
        self,
        desired_arrival: Optional[float] = None,
        time_window: Optional[tuple[float, float]] = None,
        expected_travel_time: float = 1800,
        incentive_schedule: Optional[dict[float, float]] = None,
    ) -> float:
        """
        Decide when to depart, considering pacer incentive opportunities.

        Pacer drivers may prefer times when pacer incentives are higher,
        even if it means slight schedule adjustments.
        """
        if desired_arrival is None:
            desired_arrival = self.profile.desired_arrival_time

        if time_window is None:
            earliest = desired_arrival - expected_travel_time - self.profile.flexibility_window
            latest = desired_arrival - expected_travel_time + self.profile.flexibility_window * 0.5
            time_window = (max(0, earliest), latest)

        ideal_departure = desired_arrival - expected_travel_time

        if incentive_schedule is None:
            noise = self.rng.normal(0, 600)  # More variance for flexible pacers
            return np.clip(ideal_departure + noise, time_window[0], time_window[1])

        # Evaluate departure times with pacer incentives
        time_points = np.linspace(time_window[0], time_window[1], 15)
        best_time = ideal_departure
        best_utility = float("-inf")

        for t in time_points:
            arrival = t + expected_travel_time
            delay = abs(arrival - desired_arrival)
            delay_cost = delay / 60 * self.preferences.vot / 60  # VOT per minute

            incentive = incentive_schedule.get(t, 0)
            utility = self.preferences.beta_incentive * incentive - delay_cost

            if utility > best_utility:
                best_utility = utility
                best_time = t

        return best_time

    def decide_route(
        self,
        origin: int,
        destination: int,
        available_routes: list[list[int]],
        route_attributes: Optional[list[TripAttributes]] = None,
    ) -> list[int]:
        """
        Decide which route to take, preferring pacer-enrolled corridors.
        """
        if not available_routes:
            return []

        if len(available_routes) == 1:
            return available_routes[0]

        # Prefer routes on enrolled corridors if pacing
        if self.is_pacing and self.profile.enrolled_corridors:
            # Would check route corridor membership here
            pass

        # Default to shortest route
        return min(available_routes, key=len)

    def respond_to_incentive(
        self,
        incentive_type: str,
        incentive_amount: float,
        conditions: dict[str, Any],
    ) -> bool:
        """
        Decide whether to participate in an incentive program.

        Pacer drivers evaluate the trade-off between driving constraints
        and monetary rewards.
        """
        if incentive_type != "pacer":
            return False

        # Get constraint requirements
        required_smoothness = conditions.get("smoothness_threshold", 0.8)
        speed_reduction = conditions.get("speed_reduction_fraction", 0.1)
        corridor = conditions.get("corridor_id")

        # Check if enrolled in this corridor
        if corridor and corridor not in self.profile.enrolled_corridors:
            return False

        # Check if willing to accept speed reduction
        if speed_reduction > self.profile.max_speed_reduction:
            return False

        # Estimate probability of meeting smoothness threshold
        expected_smoothness = self.profile.driving_skill * (1 - speed_reduction * 0.5)
        success_prob = min(1.0, expected_smoothness / required_smoothness)

        # Expected value of participating
        expected_reward = incentive_amount * success_prob

        # Cost of constrained driving (depends on preferences)
        constraint_cost = speed_reduction * 5.0  # $5 per 10% speed reduction

        # Participation utility
        participation_utility = (
            self.preferences.beta_incentive * expected_reward
            - constraint_cost
        )

        # Make probabilistic decision
        return self.rng.random() < self._sigmoid(participation_utility)

    def _sigmoid(self, x: float, temperature: float = 1.0) -> float:
        """Sigmoid function for probability conversion."""
        return 1 / (1 + np.exp(-x / temperature))

    def start_pacing(
        self,
        target_speed: float,
        corridor_id: str,
    ) -> None:
        """Begin a pacer driving session."""
        self.is_pacing = True
        self.current_target_speed = target_speed
        self.speed_history = []

        self.record_history("pacing_started", {
            "target_speed": target_speed,
            "corridor_id": corridor_id,
        })

    def update_speed(self, current_speed: float, timestamp: float) -> float:
        """
        Record current speed and compute smoothness-adjusted speed.

        Returns the speed the pacer should try to maintain.
        """
        self.speed_history.append(current_speed)

        if not self.is_pacing or self.current_target_speed is None:
            return current_speed

        # Add some natural variation based on driving skill
        skill_factor = self.profile.driving_skill
        noise = self.rng.normal(0, (1 - skill_factor) * 2)

        # Adjust toward target speed
        adjustment = (self.current_target_speed - current_speed) * 0.3
        suggested_speed = current_speed + adjustment + noise

        return max(0, suggested_speed)

    def end_pacing(self) -> dict[str, Any]:
        """
        End a pacer driving session and compute performance.

        Returns performance metrics for this session.
        """
        if not self.speed_history:
            self.is_pacing = False
            return {"smoothness_score": 0.0, "earnings": 0.0}

        # Compute smoothness score
        speeds = np.array(self.speed_history)
        mean_speed = speeds.mean()
        speed_variance = speeds.var()

        # Normalize variance to [0, 1] smoothness score
        # Lower variance = higher smoothness
        max_variance = 100  # Reference maximum variance
        smoothness = max(0, 1 - speed_variance / max_variance)

        # Update performance metrics
        self.performance.speed_variance_history.append(speed_variance)
        n = len(self.performance.speed_variance_history)
        self.performance.avg_smoothness_score = (
            (self.performance.avg_smoothness_score * (n - 1) + smoothness) / n
        )

        self.is_pacing = False
        self.current_target_speed = None

        result = {
            "smoothness_score": smoothness,
            "mean_speed": mean_speed,
            "speed_variance": speed_variance,
            "n_observations": len(speeds),
        }

        self.record_history("pacing_ended", result)

        return result

    def receive_pacer_payment(
        self,
        amount: float,
        distance_miles: float,
    ) -> None:
        """Record receipt of pacer payment."""
        self.performance.trips_as_pacer += 1
        self.performance.total_pacer_miles += distance_miles
        self.performance.total_pacer_earnings += amount
        self.state.incentives_earned += amount

        self.record_history("pacer_payment", {
            "amount": amount,
            "distance_miles": distance_miles,
        })

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics for this pacer agent."""
        return {
            "agent_id": self.id,
            "trips_as_pacer": self.performance.trips_as_pacer,
            "total_pacer_miles": self.performance.total_pacer_miles,
            "total_pacer_earnings": self.performance.total_pacer_earnings,
            "avg_smoothness_score": self.performance.avg_smoothness_score,
            "earnings_per_mile": (
                self.performance.total_pacer_earnings /
                max(1, self.performance.total_pacer_miles)
            ),
            "enrolled_corridors": self.profile.enrolled_corridors,
            "driving_skill": self.profile.driving_skill,
        }


def create_pacer_population(
    n_agents: int,
    home_region: tuple[tuple[float, float], tuple[float, float]],
    work_region: tuple[tuple[float, float], tuple[float, float]],
    corridors: list[str],
    rng: Optional[np.random.Generator] = None,
) -> list[PacerAgent]:
    """
    Create a population of pacer driver agents.

    Args:
        n_agents: Number of agents to create
        home_region: Bounding box for home locations
        work_region: Bounding box for work locations
        corridors: Available corridor IDs for pacer enrollment
        rng: Random number generator

    Returns:
        List of PacerAgent instances
    """
    from .base import PopulationParameters, generate_heterogeneous_preferences

    if rng is None:
        rng = np.random.default_rng()

    pop_params = PopulationParameters(n_agents=n_agents)
    agents = []

    for i in range(n_agents):
        # Generate random locations
        home = (
            rng.uniform(home_region[0][0], home_region[1][0]),
            rng.uniform(home_region[0][1], home_region[1][1]),
        )
        work = (
            rng.uniform(work_region[0][0], work_region[1][0]),
            rng.uniform(work_region[0][1], work_region[1][1]),
        )

        # Generate preferences (pacer drivers typically more incentive-responsive)
        preferences = generate_heterogeneous_preferences(pop_params, rng)
        preferences.beta_incentive *= 1.5  # More responsive to incentives

        # Random corridor enrollment (1-3 corridors)
        n_corridors = rng.integers(1, min(4, len(corridors) + 1))
        enrolled = list(rng.choice(corridors, n_corridors, replace=False))

        # Create profile
        profile = PacerProfile(
            home_location=home,
            work_location=work,
            desired_arrival_time=rng.normal(8 * 3600, 3600),
            flexibility_window=rng.uniform(1800, 5400),  # 30-90 min
            enrolled_corridors=enrolled,
            max_speed_reduction=rng.uniform(0.1, 0.3),
            driving_skill=rng.uniform(0.6, 0.95),
        )

        agent = PacerAgent(
            agent_id=f"pacer_{i:05d}",
            preferences=preferences,
            profile=profile,
            rng=rng,
        )
        agents.append(agent)

    return agents
