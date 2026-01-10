"""
Carpooling incentive mechanism.

Rewards travelers for sharing rides, reducing vehicle miles traveled
and roadway demand during peak periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .base import (
    BaseIncentive,
    IncentiveAllocation,
    IncentiveConfig,
    IncentiveType,
)


@dataclass
class CarpoolMatch:
    """Represents a carpool match between travelers."""
    match_id: str
    driver_id: str
    passenger_ids: list[str]
    origin: tuple[float, float]
    destination: tuple[float, float]
    departure_time: float
    expected_detour_minutes: float = 0.0
    status: str = "pending"  # pending, confirmed, in_progress, completed, cancelled


class CarpoolIncentive(BaseIncentive):
    """
    Incentive mechanism for carpooling.

    Rewards both drivers and passengers for sharing rides.
    Reward scales with number of passengers and peak hour multipliers.
    """

    def __init__(
        self,
        config: Optional[IncentiveConfig] = None,
        reward_per_passenger: float = 2.50,
        min_passengers: int = 1,
        max_reward: float = 10.00,
        driver_bonus: float = 1.00,
        distance_bonus_per_mile: float = 0.10,
    ):
        if config is None:
            config = IncentiveConfig(
                incentive_type=IncentiveType.CARPOOL,
                base_reward=reward_per_passenger,
            )
        super().__init__(config)

        self.reward_per_passenger = reward_per_passenger
        self.min_passengers = min_passengers
        self.max_reward = max_reward
        self.driver_bonus = driver_bonus
        self.distance_bonus_per_mile = distance_bonus_per_mile

        # Track matches
        self.matches: list[CarpoolMatch] = []
        self.active_drivers: dict[str, CarpoolMatch] = {}
        self.active_passengers: dict[str, str] = {}  # passenger_id -> match_id

    def check_eligibility(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if agent is eligible for carpool incentive."""
        # Check if already in a carpool
        if agent_id in self.active_drivers:
            return False, "Already driving a carpool"
        if agent_id in self.active_passengers:
            return False, "Already a passenger"

        # Check time eligibility
        hour = context.get("hour", 8)
        day = context.get("day_of_week", 0)
        if not self.is_active_time(hour, day):
            return False, "Outside active hours"

        # Check corridor eligibility
        corridor = context.get("corridor_id")
        if self.config.corridor_ids and corridor not in self.config.corridor_ids:
            return False, "Not on eligible corridor"

        # Check if carpool eligible
        if not context.get("carpool_eligible", True):
            return False, "Agent not carpool eligible"

        return True, "Eligible"

    def compute_reward(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> float:
        """Compute carpool reward based on context."""
        n_passengers = context.get("n_passengers", 1)
        distance_miles = context.get("distance_miles", 0)
        is_driver = context.get("is_driver", False)
        hour = context.get("hour", 8)

        # Base reward per passenger
        reward = self.reward_per_passenger * n_passengers

        # Driver bonus
        if is_driver:
            reward += self.driver_bonus

        # Distance bonus
        reward += self.distance_bonus_per_mile * distance_miles

        # Peak hour multiplier
        if hour in [7, 8, 9, 17, 18, 19]:
            reward *= self.config.peak_multiplier

        # Cap at maximum
        return min(reward, self.max_reward)

    def verify_completion(
        self,
        allocation: IncentiveAllocation,
        outcome: dict[str, Any],
    ) -> tuple[bool, float]:
        """Verify carpool trip was completed."""
        # Check if trip was completed
        if not outcome.get("trip_completed", False):
            return False, 0.0

        # Check minimum passengers
        actual_passengers = outcome.get("actual_passengers", 0)
        if actual_passengers < self.min_passengers:
            return False, 0.0

        # Compute actual reward based on actual passengers
        actual_reward = self.reward_per_passenger * actual_passengers

        if outcome.get("is_driver", False):
            actual_reward += self.driver_bonus

        distance = outcome.get("actual_distance_miles", 0)
        actual_reward += self.distance_bonus_per_mile * distance

        # Peak multiplier
        hour = outcome.get("completion_hour", 8)
        if hour in [7, 8, 9, 17, 18, 19]:
            actual_reward *= self.config.peak_multiplier

        actual_reward = min(actual_reward, self.max_reward)

        return True, actual_reward

    def create_match(
        self,
        driver_id: str,
        passenger_ids: list[str],
        origin: tuple[float, float],
        destination: tuple[float, float],
        departure_time: float,
    ) -> CarpoolMatch:
        """Create a new carpool match."""
        match = CarpoolMatch(
            match_id=f"carpool_{len(self.matches):06d}",
            driver_id=driver_id,
            passenger_ids=passenger_ids,
            origin=origin,
            destination=destination,
            departure_time=departure_time,
        )

        self.matches.append(match)
        self.active_drivers[driver_id] = match
        for pid in passenger_ids:
            self.active_passengers[pid] = match.match_id

        return match

    def find_matches(
        self,
        agent_id: str,
        origin: tuple[float, float],
        destination: tuple[float, float],
        departure_time: float,
        max_detour_minutes: float = 10.0,
        max_wait_minutes: float = 15.0,
        available_agents: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """
        Find potential carpool matches for an agent.

        Args:
            agent_id: ID of the agent seeking a match
            origin: Agent's origin coordinates
            destination: Agent's destination coordinates
            departure_time: Desired departure time
            max_detour_minutes: Maximum acceptable detour
            max_wait_minutes: Maximum wait time for pickup
            available_agents: List of other agents with their trip info

        Returns:
            List of potential matches with scores
        """
        if available_agents is None:
            return []

        matches = []

        for other in available_agents:
            if other["agent_id"] == agent_id:
                continue

            # Check spatial compatibility
            origin_dist = self._haversine_distance(origin, other["origin"])
            dest_dist = self._haversine_distance(destination, other["destination"])

            # Convert to approximate minutes (assume 30 mph average)
            origin_time = origin_dist / 0.5  # miles to minutes
            dest_time = dest_dist / 0.5

            if origin_time > max_detour_minutes or dest_time > max_detour_minutes:
                continue

            # Check temporal compatibility
            time_diff = abs(departure_time - other["departure_time"]) / 60  # to minutes
            if time_diff > max_wait_minutes:
                continue

            # Compute match score (higher is better)
            score = 1.0 - (origin_time + dest_time + time_diff) / (2 * max_detour_minutes + max_wait_minutes)

            matches.append({
                "agent_id": other["agent_id"],
                "origin": other["origin"],
                "destination": other["destination"],
                "departure_time": other["departure_time"],
                "detour_minutes": origin_time + dest_time,
                "wait_minutes": time_diff,
                "score": score,
            })

        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)

        return matches

    def _haversine_distance(
        self,
        coord1: tuple[float, float],
        coord2: tuple[float, float],
    ) -> float:
        """Compute haversine distance between two lat/lng points in miles."""
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in miles
        r = 3956

        return c * r

    def complete_match(
        self,
        match_id: str,
        outcome: dict[str, Any],
    ) -> dict[str, float]:
        """
        Complete a carpool match and distribute rewards.

        Returns dict mapping agent_id to reward amount.
        """
        match = None
        for m in self.matches:
            if m.match_id == match_id:
                match = m
                break

        if match is None:
            return {}

        rewards = {}
        n_passengers = len(match.passenger_ids)

        if outcome.get("completed", False):
            # Driver reward
            driver_context = {
                "n_passengers": n_passengers,
                "is_driver": True,
                "distance_miles": outcome.get("distance_miles", 0),
                "hour": outcome.get("hour", 8),
            }
            rewards[match.driver_id] = self.compute_reward(
                match.driver_id, driver_context
            )

            # Passenger rewards
            for pid in match.passenger_ids:
                passenger_context = {
                    "n_passengers": n_passengers,
                    "is_driver": False,
                    "distance_miles": outcome.get("distance_miles", 0),
                    "hour": outcome.get("hour", 8),
                }
                rewards[pid] = self.compute_reward(pid, passenger_context)

            match.status = "completed"
        else:
            match.status = "cancelled"

        # Clean up active tracking
        if match.driver_id in self.active_drivers:
            del self.active_drivers[match.driver_id]
        for pid in match.passenger_ids:
            if pid in self.active_passengers:
                del self.active_passengers[pid]

        # Update spending
        total_reward = sum(rewards.values())
        self.total_spent += total_reward
        if outcome.get("completed", False):
            self.n_completed += 1

        return rewards

    def get_statistics(self) -> dict[str, Any]:
        """Get carpool-specific statistics."""
        stats = super().get_statistics()

        completed_matches = [m for m in self.matches if m.status == "completed"]

        stats.update({
            "total_matches": len(self.matches),
            "completed_matches": len(completed_matches),
            "active_drivers": len(self.active_drivers),
            "active_passengers": len(self.active_passengers),
            "avg_passengers_per_match": (
                np.mean([len(m.passenger_ids) for m in completed_matches])
                if completed_matches else 0
            ),
        })

        return stats
