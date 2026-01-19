"""
Simulation engine for transportation incentive experiments.

Provides event-driven simulation with agents, network, and metrics.
"""

from .engine import SimulationConfig, SimulationEngine, SimulationResult, run_simulation
from .events import Event, EventType
from .metrics import (
    CarpoolRecord,
    MetricsCollector,
    PacerRecord,
    TimeSliceMetrics,
    TripRecord,
)
from .network import (
    Corridor,
    NetworkLink,
    NetworkNode,
    SimpleNetwork,
    create_i24_network,
    create_stadium_network,
)

__all__ = [
    # Engine
    "SimulationConfig",
    "SimulationEngine",
    "SimulationResult",
    "run_simulation",
    # Events
    "Event",
    "EventType",
    # Metrics
    "MetricsCollector",
    "TripRecord",
    "PacerRecord",
    "CarpoolRecord",
    "TimeSliceMetrics",
    # Network
    "SimpleNetwork",
    "Corridor",
    "NetworkNode",
    "NetworkLink",
    "create_i24_network",
    "create_stadium_network",
]
