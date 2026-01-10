"""
Base classes for incentive mechanisms.

Provides abstract interfaces and common functionality for all
incentive types in the simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


class IncentiveType(Enum):
    """Types of incentive mechanisms."""
    CARPOOL = auto()
    PACER = auto()
    DEPARTURE_SHIFT = auto()
    TRANSIT = auto()
    CONGESTION_PRICING = auto()
    PARKING = auto()


@dataclass
class IncentiveConfig:
    """Configuration for an incentive mechanism."""
    incentive_type: IncentiveType
    enabled: bool = True

    # Budget constraints
    budget_daily: float = 10000.0
    budget_per_agent: float = 50.0

    # Temporal constraints
    active_hours: list[int] = field(default_factory=lambda: list(range(6, 10)) + list(range(16, 20)))
    active_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    # Spatial constraints
    corridor_ids: list[str] = field(default_factory=list)
    zone_ids: list[str] = field(default_factory=list)

    # Reward parameters
    base_reward: float = 2.0
    peak_multiplier: float = 1.5

    # Additional parameters
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class IncentiveAllocation:
    """Record of an incentive allocation to an agent."""
    allocation_id: str
    agent_id: str
    incentive_type: IncentiveType
    amount: float
    timestamp: float
    conditions: dict[str, Any]
    status: str = "pending"  # pending, active, completed, expired, cancelled

    # Outcome tracking
    completed: bool = False
    actual_reward: float = 0.0
    completion_timestamp: Optional[float] = None
    performance_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class IncentiveResult:
    """Result from processing an incentive."""
    success: bool
    allocation: Optional[IncentiveAllocation]
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)


class BaseIncentive(ABC):
    """
    Abstract base class for incentive mechanisms.

    All incentive types inherit from this class and implement
    the core methods for eligibility checking, allocation, and
    outcome verification.
    """

    def __init__(self, config: IncentiveConfig):
        self.config = config
        self.allocations: list[IncentiveAllocation] = []
        self.total_spent: float = 0.0
        self.total_allocated: float = 0.0

        # Statistics
        self.n_offers: int = 0
        self.n_accepted: int = 0
        self.n_completed: int = 0

    @property
    def incentive_type(self) -> IncentiveType:
        """Return the incentive type."""
        return self.config.incentive_type

    @property
    def remaining_budget(self) -> float:
        """Return remaining daily budget."""
        return max(0, self.config.budget_daily - self.total_spent)

    @property
    def acceptance_rate(self) -> float:
        """Return the acceptance rate."""
        if self.n_offers == 0:
            return 0.0
        return self.n_accepted / self.n_offers

    @property
    def completion_rate(self) -> float:
        """Return the completion rate."""
        if self.n_accepted == 0:
            return 0.0
        return self.n_completed / self.n_accepted

    @abstractmethod
    def check_eligibility(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Check if an agent is eligible for this incentive.

        Args:
            agent_id: ID of the agent
            context: Context information (location, time, etc.)

        Returns:
            Tuple of (is_eligible, reason_message)
        """
        pass

    @abstractmethod
    def compute_reward(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> float:
        """
        Compute the reward amount for an eligible agent.

        Args:
            agent_id: ID of the agent
            context: Context information

        Returns:
            Reward amount in dollars
        """
        pass

    @abstractmethod
    def verify_completion(
        self,
        allocation: IncentiveAllocation,
        outcome: dict[str, Any],
    ) -> tuple[bool, float]:
        """
        Verify if the incentive conditions were met.

        Args:
            allocation: The incentive allocation
            outcome: Observed outcomes

        Returns:
            Tuple of (conditions_met, actual_reward)
        """
        pass

    def offer_incentive(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> IncentiveResult:
        """
        Offer an incentive to an agent.

        Args:
            agent_id: ID of the agent
            context: Context information

        Returns:
            IncentiveResult with offer details
        """
        self.n_offers += 1

        # Check eligibility
        is_eligible, reason = self.check_eligibility(agent_id, context)
        if not is_eligible:
            return IncentiveResult(
                success=False,
                allocation=None,
                message=f"Not eligible: {reason}",
            )

        # Check budget
        reward = self.compute_reward(agent_id, context)
        if reward > self.remaining_budget:
            return IncentiveResult(
                success=False,
                allocation=None,
                message="Insufficient budget",
            )

        if reward > self.config.budget_per_agent:
            reward = self.config.budget_per_agent

        # Create allocation
        allocation = IncentiveAllocation(
            allocation_id=f"{self.incentive_type.name}_{len(self.allocations):06d}",
            agent_id=agent_id,
            incentive_type=self.incentive_type,
            amount=reward,
            timestamp=context.get("timestamp", 0),
            conditions=self._get_conditions(context),
            status="pending",
        )

        self.allocations.append(allocation)
        self.total_allocated += reward

        return IncentiveResult(
            success=True,
            allocation=allocation,
            message="Incentive offered",
            metrics={"offered_amount": reward},
        )

    def accept_incentive(
        self,
        allocation_id: str,
    ) -> bool:
        """Record that an agent accepted the incentive."""
        allocation = self._find_allocation(allocation_id)
        if allocation is None:
            return False

        allocation.status = "active"
        self.n_accepted += 1
        return True

    def complete_incentive(
        self,
        allocation_id: str,
        outcome: dict[str, Any],
    ) -> IncentiveResult:
        """
        Process completion of an incentive.

        Args:
            allocation_id: ID of the allocation
            outcome: Observed outcomes

        Returns:
            IncentiveResult with completion details
        """
        allocation = self._find_allocation(allocation_id)
        if allocation is None:
            return IncentiveResult(
                success=False,
                allocation=None,
                message="Allocation not found",
            )

        # Verify completion
        conditions_met, actual_reward = self.verify_completion(allocation, outcome)

        allocation.completed = conditions_met
        allocation.actual_reward = actual_reward
        allocation.completion_timestamp = outcome.get("timestamp")
        allocation.performance_metrics = outcome.get("metrics", {})
        allocation.status = "completed" if conditions_met else "failed"

        if conditions_met:
            self.total_spent += actual_reward
            self.n_completed += 1

        return IncentiveResult(
            success=conditions_met,
            allocation=allocation,
            message="Completed" if conditions_met else "Conditions not met",
            metrics={"actual_reward": actual_reward},
        )

    def _find_allocation(self, allocation_id: str) -> Optional[IncentiveAllocation]:
        """Find an allocation by ID."""
        for alloc in self.allocations:
            if alloc.allocation_id == allocation_id:
                return alloc
        return None

    def _get_conditions(self, context: dict[str, Any]) -> dict[str, Any]:
        """Extract conditions from context."""
        return {
            "corridor_ids": self.config.corridor_ids,
            "required_time": context.get("timestamp"),
        }

    def is_active_time(self, hour: int, day_of_week: int) -> bool:
        """Check if incentive is active at this time."""
        return (
            hour in self.config.active_hours
            and day_of_week in self.config.active_days
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get incentive statistics."""
        return {
            "type": self.incentive_type.name,
            "enabled": self.config.enabled,
            "n_offers": self.n_offers,
            "n_accepted": self.n_accepted,
            "n_completed": self.n_completed,
            "acceptance_rate": self.acceptance_rate,
            "completion_rate": self.completion_rate,
            "total_allocated": self.total_allocated,
            "total_spent": self.total_spent,
            "remaining_budget": self.remaining_budget,
            "avg_reward": self.total_spent / max(1, self.n_completed),
        }

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each simulated day)."""
        self.total_spent = 0.0
        self.total_allocated = 0.0
        # Keep historical allocations but could archive them
