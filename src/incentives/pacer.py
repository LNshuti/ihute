"""
Pacer driving incentive mechanism.

Rewards drivers for maintaining steady speeds that help stabilize
traffic flow and dampen stop-and-go waves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .base import (
    BaseIncentive,
    IncentiveAllocation,
    IncentiveConfig,
    IncentiveType,
)


@dataclass
class PacerSession:
    """Represents a pacer driving session."""
    session_id: str
    agent_id: str
    corridor_id: str
    start_time: float
    target_speed: float  # mph
    speed_tolerance: float = 5.0  # mph

    # Session tracking
    speed_samples: list[float] = field(default_factory=list)
    position_samples: list[tuple[float, float]] = field(default_factory=list)
    timestamp_samples: list[float] = field(default_factory=list)

    # Outcomes
    end_time: Optional[float] = None
    distance_miles: float = 0.0
    smoothness_score: float = 0.0
    status: str = "active"  # active, completed, aborted


class PacerIncentive(BaseIncentive):
    """
    Incentive mechanism for pacer driving.

    Rewards drivers who maintain steady speeds within a target range,
    helping to stabilize traffic flow on congested corridors.
    """

    def __init__(
        self,
        config: Optional[IncentiveConfig] = None,
        reward_per_mile: float = 0.15,
        smoothness_threshold: float = 0.7,
        smoothness_bonus_rate: float = 0.10,  # Extra $ per 0.1 above threshold
        min_distance_miles: float = 2.0,
        max_reward_per_session: float = 20.00,
    ):
        if config is None:
            config = IncentiveConfig(
                incentive_type=IncentiveType.PACER,
                base_reward=reward_per_mile,
            )
        super().__init__(config)

        self.reward_per_mile = reward_per_mile
        self.smoothness_threshold = smoothness_threshold
        self.smoothness_bonus_rate = smoothness_bonus_rate
        self.min_distance_miles = min_distance_miles
        self.max_reward_per_session = max_reward_per_session

        # Active sessions
        self.sessions: list[PacerSession] = []
        self.active_sessions: dict[str, PacerSession] = {}  # agent_id -> session

        # Corridor target speeds
        self.corridor_targets: dict[str, float] = {}

    def set_corridor_target(self, corridor_id: str, target_speed: float) -> None:
        """Set target speed for a corridor."""
        self.corridor_targets[corridor_id] = target_speed

    def get_corridor_target(self, corridor_id: str) -> float:
        """Get target speed for a corridor (default 55 mph)."""
        return self.corridor_targets.get(corridor_id, 55.0)

    def check_eligibility(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if agent is eligible for pacer incentive."""
        # Check if already in a session
        if agent_id in self.active_sessions:
            return False, "Already in active pacer session"

        # Check time eligibility
        hour = context.get("hour", 8)
        day = context.get("day_of_week", 0)
        if not self.is_active_time(hour, day):
            return False, "Outside active hours"

        # Check corridor eligibility
        corridor = context.get("corridor_id")
        if not corridor:
            return False, "No corridor specified"
        if self.config.corridor_ids and corridor not in self.config.corridor_ids:
            return False, "Not on eligible corridor"

        # Check if has car
        if not context.get("has_car", True):
            return False, "Agent does not have a car"

        # Check enrollment
        enrolled_corridors = context.get("enrolled_corridors", [])
        if self.config.corridor_ids and corridor not in enrolled_corridors:
            return False, "Not enrolled for this corridor"

        return True, "Eligible"

    def compute_reward(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> float:
        """Compute expected pacer reward."""
        distance = context.get("expected_distance_miles", 5.0)
        expected_smoothness = context.get("expected_smoothness", 0.8)

        # Base reward
        reward = self.reward_per_mile * distance

        # Smoothness bonus
        if expected_smoothness > self.smoothness_threshold:
            bonus_factor = (expected_smoothness - self.smoothness_threshold) * 10
            reward += self.smoothness_bonus_rate * bonus_factor * distance

        # Peak multiplier
        hour = context.get("hour", 8)
        if hour in [7, 8, 9, 17, 18, 19]:
            reward *= self.config.peak_multiplier

        return min(reward, self.max_reward_per_session)

    def verify_completion(
        self,
        allocation: IncentiveAllocation,
        outcome: dict[str, Any],
    ) -> tuple[bool, float]:
        """Verify pacer session completion."""
        # Check minimum distance
        distance = outcome.get("distance_miles", 0)
        if distance < self.min_distance_miles:
            return False, 0.0

        # Check smoothness
        smoothness = outcome.get("smoothness_score", 0)
        if smoothness < self.smoothness_threshold:
            return False, 0.0

        # Compute reward
        reward = self.reward_per_mile * distance

        # Smoothness bonus
        if smoothness > self.smoothness_threshold:
            bonus_factor = (smoothness - self.smoothness_threshold) * 10
            reward += self.smoothness_bonus_rate * bonus_factor * distance

        # Peak multiplier
        hour = outcome.get("hour", 8)
        if hour in [7, 8, 9, 17, 18, 19]:
            reward *= self.config.peak_multiplier

        return True, min(reward, self.max_reward_per_session)

    def start_session(
        self,
        agent_id: str,
        corridor_id: str,
        start_time: float,
        target_speed: Optional[float] = None,
    ) -> PacerSession:
        """Start a new pacer session."""
        if target_speed is None:
            target_speed = self.get_corridor_target(corridor_id)

        session = PacerSession(
            session_id=f"pacer_{len(self.sessions):06d}",
            agent_id=agent_id,
            corridor_id=corridor_id,
            start_time=start_time,
            target_speed=target_speed,
        )

        self.sessions.append(session)
        self.active_sessions[agent_id] = session

        return session

    def update_session(
        self,
        agent_id: str,
        speed: float,
        position: tuple[float, float],
        timestamp: float,
    ) -> Optional[dict[str, Any]]:
        """
        Update a pacer session with new observation.

        Returns feedback dict with current performance.
        """
        session = self.active_sessions.get(agent_id)
        if session is None:
            return None

        session.speed_samples.append(speed)
        session.position_samples.append(position)
        session.timestamp_samples.append(timestamp)

        # Compute current smoothness
        if len(session.speed_samples) >= 2:
            speeds = np.array(session.speed_samples)
            variance = np.var(speeds)
            max_variance = session.speed_tolerance ** 2
            smoothness = max(0, 1 - variance / max_variance)
        else:
            smoothness = 1.0

        # Check if within target range
        speed_diff = abs(speed - session.target_speed)
        on_target = speed_diff <= session.speed_tolerance

        return {
            "session_id": session.session_id,
            "current_smoothness": smoothness,
            "on_target": on_target,
            "target_speed": session.target_speed,
            "speed_diff": speed_diff,
            "n_samples": len(session.speed_samples),
        }

    def end_session(
        self,
        agent_id: str,
        end_time: float,
    ) -> dict[str, Any]:
        """
        End a pacer session and compute final reward.

        Returns session results including reward.
        """
        session = self.active_sessions.get(agent_id)
        if session is None:
            return {"error": "No active session"}

        session.end_time = end_time

        # Compute distance from position samples
        if len(session.position_samples) >= 2:
            session.distance_miles = self._compute_distance(session.position_samples)

        # Compute smoothness score
        if len(session.speed_samples) >= 2:
            speeds = np.array(session.speed_samples)
            variance = np.var(speeds)
            max_variance = session.speed_tolerance ** 2
            session.smoothness_score = max(0, 1 - variance / max_variance)

            # Also check adherence to target
            mean_speed = np.mean(speeds)
            target_adherence = 1 - abs(mean_speed - session.target_speed) / session.target_speed
            session.smoothness_score = (session.smoothness_score + target_adherence) / 2

        # Determine status and reward
        if session.distance_miles < self.min_distance_miles:
            session.status = "aborted"
            reward = 0.0
        elif session.smoothness_score < self.smoothness_threshold:
            session.status = "completed"  # Completed but no reward
            reward = 0.0
        else:
            session.status = "completed"
            reward = self.reward_per_mile * session.distance_miles

            # Smoothness bonus
            bonus_factor = (session.smoothness_score - self.smoothness_threshold) * 10
            reward += self.smoothness_bonus_rate * bonus_factor * session.distance_miles

            reward = min(reward, self.max_reward_per_session)
            self.total_spent += reward
            self.n_completed += 1

        # Clean up
        del self.active_sessions[agent_id]

        return {
            "session_id": session.session_id,
            "status": session.status,
            "distance_miles": session.distance_miles,
            "smoothness_score": session.smoothness_score,
            "reward": reward,
            "duration_seconds": end_time - session.start_time,
            "n_samples": len(session.speed_samples),
        }

    def _compute_distance(
        self,
        positions: list[tuple[float, float]],
    ) -> float:
        """Compute total distance from position samples."""
        total = 0.0
        for i in range(1, len(positions)):
            lat1, lon1 = positions[i-1]
            lat2, lon2 = positions[i]

            # Haversine formula
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 3956  # Earth radius in miles

            total += c * r

        return total

    def compute_corridor_performance(
        self,
        corridor_id: str,
    ) -> dict[str, Any]:
        """Compute aggregate pacer performance for a corridor."""
        corridor_sessions = [
            s for s in self.sessions
            if s.corridor_id == corridor_id and s.status == "completed"
        ]

        if not corridor_sessions:
            return {
                "corridor_id": corridor_id,
                "n_sessions": 0,
            }

        smoothness_scores = [s.smoothness_score for s in corridor_sessions]
        distances = [s.distance_miles for s in corridor_sessions]

        return {
            "corridor_id": corridor_id,
            "n_sessions": len(corridor_sessions),
            "total_pacer_miles": sum(distances),
            "avg_smoothness": np.mean(smoothness_scores),
            "smoothness_std": np.std(smoothness_scores),
            "target_speed": self.get_corridor_target(corridor_id),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get pacer-specific statistics."""
        stats = super().get_statistics()

        completed = [s for s in self.sessions if s.status == "completed"]
        successful = [s for s in completed if s.smoothness_score >= self.smoothness_threshold]

        stats.update({
            "total_sessions": len(self.sessions),
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(completed),
            "successful_sessions": len(successful),
            "success_rate": len(successful) / max(1, len(completed)),
            "total_pacer_miles": sum(s.distance_miles for s in completed),
            "avg_smoothness": np.mean([s.smoothness_score for s in completed]) if completed else 0,
            "corridors": list(self.corridor_targets.keys()),
        })

        return stats
