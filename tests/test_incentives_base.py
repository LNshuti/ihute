"""
Tests for src/incentives/base.py module.

Tests cover:
- IncentiveType enum
- IncentiveConfig dataclass
- IncentiveAllocation dataclass
- IncentiveResult dataclass
- BaseIncentive abstract class
"""

import numpy as np
import pytest

from src.incentives.base import (
    BaseIncentive,
    IncentiveAllocation,
    IncentiveConfig,
    IncentiveResult,
    IncentiveType,
)


class TestIncentiveType:
    """Tests for IncentiveType enum."""

    def test_all_types_exist(self):
        """Verify all expected incentive types are defined."""
        expected_types = [
            "CARPOOL",
            "PACER",
            "DEPARTURE_SHIFT",
            "TRANSIT",
            "CONGESTION_PRICING",
            "PARKING",
        ]
        for type_name in expected_types:
            assert hasattr(IncentiveType, type_name)

    def test_type_values_unique(self):
        """Each type should have unique value."""
        values = [t.value for t in IncentiveType]
        assert len(values) == len(set(values))


class TestIncentiveConfig:
    """Tests for IncentiveConfig dataclass."""

    def test_required_fields(self):
        """IncentiveConfig requires incentive_type."""
        config = IncentiveConfig(incentive_type=IncentiveType.CARPOOL)
        assert config.incentive_type == IncentiveType.CARPOOL

    def test_default_values(self):
        """IncentiveConfig should have sensible defaults."""
        config = IncentiveConfig(incentive_type=IncentiveType.PACER)

        assert config.enabled is True
        assert config.budget_daily == 10000.0
        assert config.budget_per_agent == 50.0
        assert config.base_reward == 2.0
        assert config.peak_multiplier == 1.5

    def test_active_hours_default(self):
        """Active hours should default to rush hours."""
        config = IncentiveConfig(incentive_type=IncentiveType.CARPOOL)

        # AM rush: 6-9, PM rush: 16-19
        expected = list(range(6, 10)) + list(range(16, 20))
        assert config.active_hours == expected

    def test_active_days_default(self):
        """Active days should default to weekdays."""
        config = IncentiveConfig(incentive_type=IncentiveType.CARPOOL)
        assert config.active_days == [0, 1, 2, 3, 4]

    def test_custom_values(self):
        """IncentiveConfig should accept custom values."""
        config = IncentiveConfig(
            incentive_type=IncentiveType.TRANSIT,
            enabled=False,
            budget_daily=5000.0,
            base_reward=5.0,
            corridor_ids=["I-95", "I-10"],
        )

        assert config.enabled is False
        assert config.budget_daily == 5000.0
        assert config.base_reward == 5.0
        assert config.corridor_ids == ["I-95", "I-10"]


class TestIncentiveAllocation:
    """Tests for IncentiveAllocation dataclass."""

    def test_required_fields(self):
        """IncentiveAllocation requires core fields."""
        allocation = IncentiveAllocation(
            allocation_id="test_001",
            agent_id="agent_001",
            incentive_type=IncentiveType.CARPOOL,
            amount=5.0,
            timestamp=1000.0,
            conditions={"corridor": "I-95"},
        )

        assert allocation.allocation_id == "test_001"
        assert allocation.agent_id == "agent_001"
        assert allocation.amount == 5.0

    def test_default_values(self):
        """IncentiveAllocation should have sensible defaults."""
        allocation = IncentiveAllocation(
            allocation_id="test_001",
            agent_id="agent_001",
            incentive_type=IncentiveType.PACER,
            amount=5.0,
            timestamp=1000.0,
            conditions={},
        )

        assert allocation.status == "pending"
        assert allocation.completed is False
        assert allocation.actual_reward == 0.0
        assert allocation.completion_timestamp is None
        assert allocation.performance_metrics == {}


class TestIncentiveResult:
    """Tests for IncentiveResult dataclass."""

    def test_success_result(self):
        """IncentiveResult for successful offer."""
        allocation = IncentiveAllocation(
            allocation_id="test_001",
            agent_id="agent_001",
            incentive_type=IncentiveType.CARPOOL,
            amount=5.0,
            timestamp=1000.0,
            conditions={},
        )
        result = IncentiveResult(
            success=True,
            allocation=allocation,
            message="Incentive offered",
            metrics={"offered_amount": 5.0},
        )

        assert result.success is True
        assert result.allocation == allocation
        assert result.message == "Incentive offered"
        assert result.metrics["offered_amount"] == 5.0

    def test_failure_result(self):
        """IncentiveResult for failed offer."""
        result = IncentiveResult(
            success=False,
            allocation=None,
            message="Not eligible",
        )

        assert result.success is False
        assert result.allocation is None
        assert result.message == "Not eligible"


class ConcreteIncentive(BaseIncentive):
    """Concrete implementation of BaseIncentive for testing."""

    def check_eligibility(self, agent_id, context):
        if context.get("eligible", True):
            return True, "Eligible"
        return False, "Not eligible"

    def compute_reward(self, agent_id, context):
        return context.get("reward", self.config.base_reward)

    def verify_completion(self, allocation, outcome):
        if outcome.get("completed", False):
            return True, allocation.amount
        return False, 0.0


class TestBaseIncentive:
    """Tests for BaseIncentive abstract class."""

    @pytest.fixture
    def config(self):
        """Create an incentive config."""
        return IncentiveConfig(
            incentive_type=IncentiveType.CARPOOL,
            budget_daily=1000.0,
            budget_per_agent=50.0,
            base_reward=5.0,
        )

    @pytest.fixture
    def incentive(self, config):
        """Create a concrete incentive for testing."""
        return ConcreteIncentive(config)

    def test_initialization(self, incentive, config):
        """Incentive should initialize with config."""
        assert incentive.config == config
        assert incentive.allocations == []
        assert incentive.total_spent == 0.0
        assert incentive.total_allocated == 0.0
        assert incentive.n_offers == 0

    def test_incentive_type_property(self, incentive):
        """incentive_type property should return config type."""
        assert incentive.incentive_type == IncentiveType.CARPOOL

    def test_remaining_budget(self, incentive):
        """remaining_budget should track spending."""
        assert incentive.remaining_budget == 1000.0

        incentive.total_spent = 400.0
        assert incentive.remaining_budget == 600.0

    def test_remaining_budget_cannot_go_negative(self, incentive):
        """remaining_budget should not go negative."""
        incentive.total_spent = 1500.0
        assert incentive.remaining_budget == 0.0

    def test_acceptance_rate_no_offers(self, incentive):
        """acceptance_rate should be 0 with no offers."""
        assert incentive.acceptance_rate == 0.0

    def test_acceptance_rate_with_offers(self, incentive):
        """acceptance_rate should compute correctly."""
        incentive.n_offers = 10
        incentive.n_accepted = 7
        assert incentive.acceptance_rate == 0.7

    def test_completion_rate_no_accepted(self, incentive):
        """completion_rate should be 0 with no accepted."""
        assert incentive.completion_rate == 0.0

    def test_completion_rate_with_accepted(self, incentive):
        """completion_rate should compute correctly."""
        incentive.n_accepted = 10
        incentive.n_completed = 8
        assert incentive.completion_rate == 0.8

    def test_offer_incentive_eligible(self, incentive):
        """offer_incentive should create allocation for eligible agent."""
        context = {"eligible": True, "timestamp": 1000}
        result = incentive.offer_incentive("agent_001", context)

        assert result.success is True
        assert result.allocation is not None
        assert result.allocation.agent_id == "agent_001"
        assert result.allocation.amount == 5.0
        assert incentive.n_offers == 1
        assert len(incentive.allocations) == 1

    def test_offer_incentive_not_eligible(self, incentive):
        """offer_incentive should reject ineligible agent."""
        context = {"eligible": False, "timestamp": 1000}
        result = incentive.offer_incentive("agent_001", context)

        assert result.success is False
        assert result.allocation is None
        assert "Not eligible" in result.message
        assert incentive.n_offers == 1
        assert len(incentive.allocations) == 0

    def test_offer_incentive_budget_exceeded(self, incentive):
        """offer_incentive should reject if budget exceeded."""
        incentive.total_spent = 999.0  # Only $1 left
        context = {"eligible": True, "reward": 10.0, "timestamp": 1000}

        result = incentive.offer_incentive("agent_001", context)

        assert result.success is False
        assert "budget" in result.message.lower()

    def test_offer_incentive_caps_at_agent_limit(self, incentive):
        """Reward should be capped at budget_per_agent."""
        context = {"eligible": True, "reward": 100.0, "timestamp": 1000}
        result = incentive.offer_incentive("agent_001", context)

        assert result.success is True
        assert result.allocation.amount == 50.0  # Capped at budget_per_agent

    def test_accept_incentive(self, incentive):
        """accept_incentive should update allocation status."""
        context = {"eligible": True, "timestamp": 1000}
        result = incentive.offer_incentive("agent_001", context)
        allocation_id = result.allocation.allocation_id

        success = incentive.accept_incentive(allocation_id)

        assert success is True
        assert result.allocation.status == "active"
        assert incentive.n_accepted == 1

    def test_accept_incentive_not_found(self, incentive):
        """accept_incentive should return False for unknown ID."""
        success = incentive.accept_incentive("nonexistent_id")
        assert success is False

    def test_complete_incentive_success(self, incentive):
        """complete_incentive should finalize successful completion."""
        context = {"eligible": True, "timestamp": 1000}
        offer_result = incentive.offer_incentive("agent_001", context)
        allocation_id = offer_result.allocation.allocation_id
        incentive.accept_incentive(allocation_id)

        outcome = {"completed": True, "timestamp": 2000}
        result = incentive.complete_incentive(allocation_id, outcome)

        assert result.success is True
        assert result.allocation.completed is True
        assert result.allocation.status == "completed"
        assert result.allocation.actual_reward == 5.0
        assert incentive.n_completed == 1
        assert incentive.total_spent == 5.0

    def test_complete_incentive_failure(self, incentive):
        """complete_incentive should handle failed completion."""
        context = {"eligible": True, "timestamp": 1000}
        offer_result = incentive.offer_incentive("agent_001", context)
        allocation_id = offer_result.allocation.allocation_id
        incentive.accept_incentive(allocation_id)

        outcome = {"completed": False}
        result = incentive.complete_incentive(allocation_id, outcome)

        assert result.success is False
        assert result.allocation.completed is False
        assert result.allocation.status == "failed"
        assert incentive.n_completed == 0

    def test_complete_incentive_not_found(self, incentive):
        """complete_incentive should handle unknown allocation."""
        result = incentive.complete_incentive("nonexistent", {})

        assert result.success is False
        assert "not found" in result.message.lower()

    def test_is_active_time(self, incentive):
        """is_active_time should check time eligibility."""
        # Active during rush hours (6-9, 16-19) on weekdays
        assert incentive.is_active_time(7, 0) is True  # 7 AM Monday
        assert incentive.is_active_time(17, 2) is True  # 5 PM Wednesday
        assert incentive.is_active_time(12, 0) is False  # 12 PM Monday
        assert incentive.is_active_time(7, 5) is False  # 7 AM Saturday

    def test_get_statistics(self, incentive):
        """get_statistics should return summary dict."""
        # Make some offers
        incentive.offer_incentive("agent_001", {"eligible": True, "timestamp": 1000})
        incentive.offer_incentive("agent_002", {"eligible": True, "timestamp": 1000})
        incentive.offer_incentive("agent_003", {"eligible": False, "timestamp": 1000})

        # Accept one
        incentive.accept_incentive(incentive.allocations[0].allocation_id)

        # Complete one
        incentive.complete_incentive(
            incentive.allocations[0].allocation_id,
            {"completed": True, "timestamp": 2000}
        )

        stats = incentive.get_statistics()

        assert stats["type"] == "CARPOOL"
        assert stats["enabled"] is True
        assert stats["n_offers"] == 3
        assert stats["n_accepted"] == 1
        assert stats["n_completed"] == 1
        assert stats["acceptance_rate"] == pytest.approx(1 / 3)
        assert stats["completion_rate"] == 1.0
        assert stats["total_spent"] == 5.0

    def test_reset_daily(self, incentive):
        """reset_daily should clear daily counters."""
        incentive.total_spent = 500.0
        incentive.total_allocated = 600.0

        incentive.reset_daily()

        assert incentive.total_spent == 0.0
        assert incentive.total_allocated == 0.0

    def test_find_allocation(self, incentive):
        """_find_allocation should locate by ID."""
        incentive.offer_incentive("agent_001", {"eligible": True, "timestamp": 1000})
        allocation_id = incentive.allocations[0].allocation_id

        found = incentive._find_allocation(allocation_id)

        assert found is not None
        assert found.allocation_id == allocation_id

    def test_find_allocation_not_found(self, incentive):
        """_find_allocation should return None for unknown ID."""
        found = incentive._find_allocation("nonexistent")
        assert found is None

    def test_get_conditions(self, incentive):
        """_get_conditions should extract from context."""
        context = {"timestamp": 1000}
        conditions = incentive._get_conditions(context)

        assert "corridor_ids" in conditions
        assert "required_time" in conditions
        assert conditions["required_time"] == 1000

    def test_multiple_allocations(self, incentive):
        """Multiple allocations should be tracked."""
        for i in range(5):
            incentive.offer_incentive(f"agent_{i:03d}", {"eligible": True, "timestamp": 1000 + i})

        assert len(incentive.allocations) == 5
        assert incentive.n_offers == 5

        # Each should have unique ID
        ids = [a.allocation_id for a in incentive.allocations]
        assert len(ids) == len(set(ids))
