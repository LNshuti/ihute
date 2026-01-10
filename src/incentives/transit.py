"""
Transit promotion incentive mechanism.

Rewards travelers for using public transit instead of driving,
particularly during peak periods on congested corridors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from .base import (
    BaseIncentive,
    IncentiveAllocation,
    IncentiveConfig,
    IncentiveType,
)


@dataclass
class TransitPass:
    """Represents a transit pass or voucher."""
    pass_id: str
    agent_id: str
    pass_type: str  # "single", "daily", "weekly", "monthly"
    value: float
    issue_date: float
    expiry_date: float
    trips_remaining: int = -1  # -1 for unlimited
    trips_used: int = 0
    status: str = "active"  # active, used, expired


class TransitIncentive(BaseIncentive):
    """
    Incentive mechanism for transit promotion.

    Provides rewards, discounts, or free passes to encourage
    mode shift from driving to public transit.
    """

    def __init__(
        self,
        config: Optional[IncentiveConfig] = None,
        reward_per_trip: float = 2.00,
        first_time_bonus: float = 5.00,
        streak_bonus_rate: float = 0.50,  # Per consecutive day
        max_streak_bonus: float = 5.00,
    ):
        if config is None:
            config = IncentiveConfig(
                incentive_type=IncentiveType.TRANSIT,
                base_reward=reward_per_trip,
            )
        super().__init__(config)

        self.reward_per_trip = reward_per_trip
        self.first_time_bonus = first_time_bonus
        self.streak_bonus_rate = streak_bonus_rate
        self.max_streak_bonus = max_streak_bonus

        # Passes issued
        self.passes: list[TransitPass] = []
        self.active_passes: dict[str, TransitPass] = {}  # agent_id -> pass

        # Agent transit history
        self.agent_history: dict[str, list[dict[str, Any]]] = {}
        self.agent_streaks: dict[str, int] = {}  # Current streak days

        # Transit routes/stops for verification
        self.transit_routes: list[str] = []
        self.transit_stops: list[str] = []

    def check_eligibility(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if agent is eligible for transit incentive."""
        # Check time eligibility
        hour = context.get("hour", 8)
        day = context.get("day_of_week", 0)
        if not self.is_active_time(hour, day):
            return False, "Outside active hours"

        # Check if agent typically drives (mode shift target)
        usual_mode = context.get("usual_mode", "drive")
        if usual_mode == "transit":
            # Already a transit user - may have different eligibility
            if not context.get("reward_existing_users", False):
                return False, "Already a transit user"

        # Check transit availability
        origin = context.get("origin")
        destination = context.get("destination")
        if origin and destination:
            transit_available = context.get("transit_available", True)
            if not transit_available:
                return False, "Transit not available for this trip"

        return True, "Eligible"

    def compute_reward(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> float:
        """Compute transit incentive reward."""
        reward = self.reward_per_trip

        # First-time bonus
        if agent_id not in self.agent_history:
            reward += self.first_time_bonus

        # Streak bonus
        streak = self.agent_streaks.get(agent_id, 0)
        streak_bonus = min(self.streak_bonus_rate * streak, self.max_streak_bonus)
        reward += streak_bonus

        # Peak hour multiplier
        hour = context.get("hour", 8)
        if hour in [7, 8, 9, 17, 18, 19]:
            reward *= self.config.peak_multiplier

        # Mode shift bonus (higher for drivers)
        usual_mode = context.get("usual_mode", "drive")
        if usual_mode in ["drive", "drive_alone"]:
            reward *= 1.25

        return reward

    def verify_completion(
        self,
        allocation: IncentiveAllocation,
        outcome: dict[str, Any],
    ) -> tuple[bool, float]:
        """Verify transit trip was completed."""
        # Check if transit was actually used
        actual_mode = outcome.get("actual_mode")
        if actual_mode not in ["transit", "bus", "rail", "subway"]:
            return False, 0.0

        # Check trip completion
        if not outcome.get("trip_completed", False):
            return False, 0.0

        # Verify with transit data if available
        transit_verification = outcome.get("transit_verified", True)
        if not transit_verification:
            return False, 0.0

        # Compute actual reward
        agent_id = allocation.agent_id
        reward = self.reward_per_trip

        # First-time bonus
        if agent_id not in self.agent_history:
            reward += self.first_time_bonus

        # Streak bonus
        streak = self.agent_streaks.get(agent_id, 0)
        streak_bonus = min(self.streak_bonus_rate * streak, self.max_streak_bonus)
        reward += streak_bonus

        # Peak multiplier
        hour = outcome.get("hour", 8)
        if hour in [7, 8, 9, 17, 18, 19]:
            reward *= self.config.peak_multiplier

        # Update history
        self._record_trip(agent_id, outcome)

        return True, reward

    def _record_trip(self, agent_id: str, outcome: dict[str, Any]) -> None:
        """Record a completed transit trip."""
        if agent_id not in self.agent_history:
            self.agent_history[agent_id] = []

        self.agent_history[agent_id].append({
            "timestamp": outcome.get("timestamp", 0),
            "origin": outcome.get("origin"),
            "destination": outcome.get("destination"),
            "route": outcome.get("route"),
        })

        # Update streak
        # Simplified: increment streak (in real impl, check consecutive days)
        current_streak = self.agent_streaks.get(agent_id, 0)
        self.agent_streaks[agent_id] = current_streak + 1

    def issue_pass(
        self,
        agent_id: str,
        pass_type: str,
        value: float,
        validity_days: int = 30,
        trips: int = -1,
    ) -> TransitPass:
        """Issue a transit pass to an agent."""
        current_time = datetime.now().timestamp()

        transit_pass = TransitPass(
            pass_id=f"pass_{len(self.passes):06d}",
            agent_id=agent_id,
            pass_type=pass_type,
            value=value,
            issue_date=current_time,
            expiry_date=current_time + validity_days * 86400,
            trips_remaining=trips,
        )

        self.passes.append(transit_pass)
        self.active_passes[agent_id] = transit_pass
        self.total_spent += value

        return transit_pass

    def use_pass(self, agent_id: str) -> Optional[dict[str, Any]]:
        """Use a transit pass for a trip."""
        transit_pass = self.active_passes.get(agent_id)
        if transit_pass is None:
            return None

        # Check expiry
        current_time = datetime.now().timestamp()
        if current_time > transit_pass.expiry_date:
            transit_pass.status = "expired"
            del self.active_passes[agent_id]
            return None

        # Check remaining trips
        if transit_pass.trips_remaining == 0:
            transit_pass.status = "used"
            del self.active_passes[agent_id]
            return None

        # Use the pass
        transit_pass.trips_used += 1
        if transit_pass.trips_remaining > 0:
            transit_pass.trips_remaining -= 1

        return {
            "pass_id": transit_pass.pass_id,
            "trips_remaining": transit_pass.trips_remaining,
            "trips_used": transit_pass.trips_used,
        }

    def compute_mode_shift(self) -> dict[str, Any]:
        """Compute aggregate mode shift statistics."""
        total_transit_trips = sum(
            len(trips) for trips in self.agent_history.values()
        )

        unique_users = len(self.agent_history)

        # Compute average trips per user
        avg_trips = total_transit_trips / max(1, unique_users)

        # Compute streak distribution
        streaks = list(self.agent_streaks.values())

        return {
            "total_transit_trips": total_transit_trips,
            "unique_transit_users": unique_users,
            "avg_trips_per_user": avg_trips,
            "active_passes": len(self.active_passes),
            "total_passes_issued": len(self.passes),
            "avg_streak": np.mean(streaks) if streaks else 0,
            "max_streak": max(streaks) if streaks else 0,
        }

    def get_agent_status(self, agent_id: str) -> dict[str, Any]:
        """Get transit status for a specific agent."""
        trips = self.agent_history.get(agent_id, [])
        streak = self.agent_streaks.get(agent_id, 0)
        active_pass = self.active_passes.get(agent_id)

        return {
            "agent_id": agent_id,
            "total_trips": len(trips),
            "current_streak": streak,
            "has_active_pass": active_pass is not None,
            "pass_info": {
                "pass_id": active_pass.pass_id,
                "trips_remaining": active_pass.trips_remaining,
                "expiry_date": active_pass.expiry_date,
            } if active_pass else None,
            "next_reward": self.compute_reward(agent_id, {"hour": 8}),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get transit incentive statistics."""
        stats = super().get_statistics()
        stats.update(self.compute_mode_shift())
        return stats
