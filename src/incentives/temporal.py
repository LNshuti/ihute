"""
Departure time shift incentive mechanism.

Rewards travelers for shifting their departure times away from
peak congestion periods.
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
class TimeSlot:
    """Represents a departure time slot with associated incentive."""
    slot_id: str
    start_time: float  # Seconds from midnight
    end_time: float
    incentive_rate: float  # $/trip for departing in this slot
    capacity: int = -1  # -1 for unlimited
    current_count: int = 0

    @property
    def is_available(self) -> bool:
        return self.capacity < 0 or self.current_count < self.capacity

    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time) / 60

    def contains_time(self, time: float) -> bool:
        return self.start_time <= time < self.end_time


class DepartureShiftIncentive(BaseIncentive):
    """
    Incentive mechanism for departure time shifting.

    Rewards travelers who shift their departure times from peak
    periods to shoulder periods or off-peak times.
    """

    def __init__(
        self,
        config: Optional[IncentiveConfig] = None,
        base_shift_reward: float = 3.00,
        reward_per_minute_shift: float = 0.10,
        min_shift_minutes: int = 15,
        max_shift_minutes: int = 60,
    ):
        if config is None:
            config = IncentiveConfig(
                incentive_type=IncentiveType.DEPARTURE_SHIFT,
                base_reward=base_shift_reward,
            )
        super().__init__(config)

        self.base_shift_reward = base_shift_reward
        self.reward_per_minute_shift = reward_per_minute_shift
        self.min_shift_minutes = min_shift_minutes
        self.max_shift_minutes = max_shift_minutes

        # Time slots with incentives
        self.time_slots: list[TimeSlot] = []

        # Track original and shifted departure times
        self.shift_records: list[dict[str, Any]] = []

        # Peak period definition (default AM and PM peaks)
        self.peak_periods = [
            (7 * 3600, 9 * 3600),   # 7-9 AM
            (17 * 3600, 19 * 3600),  # 5-7 PM
        ]

    def add_time_slot(
        self,
        start_hour: float,
        end_hour: float,
        incentive_rate: float,
        capacity: int = -1,
    ) -> TimeSlot:
        """Add an incentivized time slot."""
        slot = TimeSlot(
            slot_id=f"slot_{len(self.time_slots):03d}",
            start_time=start_hour * 3600,
            end_time=end_hour * 3600,
            incentive_rate=incentive_rate,
            capacity=capacity,
        )
        self.time_slots.append(slot)
        return slot

    def setup_default_slots(self) -> None:
        """Set up default shoulder period incentive slots."""
        # Early morning (before AM peak)
        self.add_time_slot(5.5, 7.0, incentive_rate=5.00)

        # Late morning (after AM peak)
        self.add_time_slot(9.0, 10.0, incentive_rate=3.00)

        # Early afternoon (before PM peak)
        self.add_time_slot(15.0, 17.0, incentive_rate=2.00)

        # Late evening (after PM peak)
        self.add_time_slot(19.0, 20.5, incentive_rate=4.00)

    def is_peak_time(self, time: float) -> bool:
        """Check if a time is during peak period."""
        for start, end in self.peak_periods:
            if start <= time < end:
                return True
        return False

    def check_eligibility(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if agent is eligible for departure shift incentive."""
        # Get original desired time
        original_time = context.get("original_departure_time")
        if original_time is None:
            return False, "Original departure time not provided"

        # Must be shifting FROM peak period
        if not self.is_peak_time(original_time):
            return False, "Original time not in peak period"

        # Check flexibility
        flexibility = context.get("flexibility_window", 0)
        if flexibility < self.min_shift_minutes * 60:
            return False, "Insufficient flexibility"

        # Check day eligibility
        day = context.get("day_of_week", 0)
        if day not in self.config.active_days:
            return False, "Not an active day"

        return True, "Eligible"

    def compute_reward(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> float:
        """Compute reward for shifting departure time."""
        original_time = context.get("original_departure_time", 0)
        shifted_time = context.get("shifted_departure_time", 0)

        # Compute shift amount
        shift_minutes = abs(shifted_time - original_time) / 60

        if shift_minutes < self.min_shift_minutes:
            return 0.0

        # Cap shift
        shift_minutes = min(shift_minutes, self.max_shift_minutes)

        # Base reward plus per-minute bonus
        reward = self.base_shift_reward + self.reward_per_minute_shift * shift_minutes

        # Check if shifted to an incentivized slot
        slot_bonus = 0.0
        for slot in self.time_slots:
            if slot.contains_time(shifted_time) and slot.is_available:
                slot_bonus = max(slot_bonus, slot.incentive_rate)

        reward += slot_bonus

        return reward

    def verify_completion(
        self,
        allocation: IncentiveAllocation,
        outcome: dict[str, Any],
    ) -> tuple[bool, float]:
        """Verify departure shift was executed."""
        actual_departure = outcome.get("actual_departure_time")
        if actual_departure is None:
            return False, 0.0

        original_time = allocation.conditions.get("original_departure_time", 0)
        target_time = allocation.conditions.get("shifted_departure_time", 0)

        # Check if actual departure is close to target
        tolerance = 10 * 60  # 10 minute tolerance
        if abs(actual_departure - target_time) > tolerance:
            return False, 0.0

        # Compute actual shift
        shift_minutes = abs(actual_departure - original_time) / 60

        if shift_minutes < self.min_shift_minutes:
            return False, 0.0

        shift_minutes = min(shift_minutes, self.max_shift_minutes)

        # Compute reward
        reward = self.base_shift_reward + self.reward_per_minute_shift * shift_minutes

        # Slot bonus
        for slot in self.time_slots:
            if slot.contains_time(actual_departure):
                reward += slot.incentive_rate
                slot.current_count += 1
                break

        # Record shift
        self.shift_records.append({
            "agent_id": allocation.agent_id,
            "original_time": original_time,
            "target_time": target_time,
            "actual_time": actual_departure,
            "shift_minutes": shift_minutes,
            "reward": reward,
        })

        return True, reward

    def get_available_shifts(
        self,
        original_time: float,
        flexibility_window: float,
    ) -> list[dict[str, Any]]:
        """
        Get available shift options for an agent.

        Args:
            original_time: Agent's original desired departure time
            flexibility_window: How much the agent can shift (seconds)

        Returns:
            List of shift options with rewards
        """
        options = []

        for slot in self.time_slots:
            if not slot.is_available:
                continue

            # Find closest time in slot to original
            if slot.end_time < original_time - flexibility_window:
                continue
            if slot.start_time > original_time + flexibility_window:
                continue

            # Target the slot midpoint or closest feasible time
            slot_mid = (slot.start_time + slot.end_time) / 2
            if slot_mid < original_time - flexibility_window:
                target = original_time - flexibility_window
            elif slot_mid > original_time + flexibility_window:
                target = original_time + flexibility_window
            else:
                target = slot_mid

            # Ensure target is within slot
            target = max(slot.start_time, min(target, slot.end_time - 60))

            shift_minutes = abs(target - original_time) / 60

            if shift_minutes < self.min_shift_minutes:
                continue

            reward = self.base_shift_reward + self.reward_per_minute_shift * min(shift_minutes, self.max_shift_minutes)
            reward += slot.incentive_rate

            options.append({
                "slot_id": slot.slot_id,
                "target_time": target,
                "shift_minutes": shift_minutes,
                "expected_reward": reward,
                "slot_capacity_remaining": slot.capacity - slot.current_count if slot.capacity > 0 else float("inf"),
            })

        # Sort by reward
        options.sort(key=lambda x: x["expected_reward"], reverse=True)

        return options

    def compute_demand_shift(self) -> dict[str, Any]:
        """Compute aggregate demand shift statistics."""
        if not self.shift_records:
            return {"total_shifts": 0}

        shifts = np.array([r["shift_minutes"] for r in self.shift_records])

        # Aggregate by time period
        shifted_from_peak = sum(
            1 for r in self.shift_records
            if self.is_peak_time(r["original_time"]) and not self.is_peak_time(r["actual_time"])
        )

        return {
            "total_shifts": len(self.shift_records),
            "avg_shift_minutes": np.mean(shifts),
            "total_shift_minutes": np.sum(shifts),
            "shifted_from_peak": shifted_from_peak,
            "peak_reduction_fraction": shifted_from_peak / len(self.shift_records),
            "total_rewards": sum(r["reward"] for r in self.shift_records),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get departure shift statistics."""
        stats = super().get_statistics()
        stats.update(self.compute_demand_shift())
        stats["n_time_slots"] = len(self.time_slots)
        stats["slot_utilization"] = {
            slot.slot_id: slot.current_count / max(1, slot.capacity) if slot.capacity > 0 else 0
            for slot in self.time_slots
        }
        return stats
