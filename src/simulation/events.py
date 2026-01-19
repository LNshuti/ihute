"""
Event types for the simulation engine.

Defines the events that drive the discrete-event simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class EventType(Enum):
    """Types of events in the simulation."""

    DEPARTURE = auto()  # Agent departs from origin
    ARRIVAL = auto()  # Agent arrives at destination
    MODE_CHOICE = auto()  # Agent makes mode choice decision
    PACING_UPDATE = auto()  # Pacer driver speed update
    PACING_START = auto()  # Pacer session begins
    PACING_END = auto()  # Pacer session ends
    CARPOOL_MATCH = auto()  # Carpool match formed
    CARPOOL_PICKUP = auto()  # Driver picks up passenger
    CARPOOL_COMPLETE = auto()  # Carpool trip completed
    INCENTIVE_OFFER = auto()  # Incentive offered to agent
    INCENTIVE_RESPONSE = auto()  # Agent responds to incentive
    EVENT_END = auto()  # Stadium/concert event ends
    METRICS_SNAPSHOT = auto()  # Periodic metrics collection


@dataclass(order=True)
class Event:
    """
    A simulation event.

    Events are ordered by time for priority queue processing.
    """

    time: float  # Simulation time in seconds
    event_type: EventType = field(compare=False)
    agent_id: Optional[str] = field(default=None, compare=False)
    data: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class TripEvent:
    """Detailed trip information for departure/arrival events."""

    origin: tuple[float, float]
    destination: tuple[float, float]
    mode: str
    expected_travel_time: float
    route_id: Optional[str] = None
    corridor_id: Optional[str] = None


@dataclass
class PacingEvent:
    """Pacer-specific event data."""

    corridor_id: str
    target_speed: float
    current_speed: float
    position: tuple[float, float]
    session_id: Optional[str] = None


@dataclass
class CarpoolEvent:
    """Carpool-specific event data."""

    match_id: str
    driver_id: str
    passenger_ids: list[str]
    pickup_location: Optional[tuple[float, float]] = None
    dropoff_location: Optional[tuple[float, float]] = None


def create_departure_event(
    time: float,
    agent_id: str,
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
    expected_travel_time: float,
    corridor_id: Optional[str] = None,
) -> Event:
    """Create a departure event."""
    return Event(
        time=time,
        event_type=EventType.DEPARTURE,
        agent_id=agent_id,
        data={
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "expected_travel_time": expected_travel_time,
            "corridor_id": corridor_id,
        },
    )


def create_arrival_event(
    time: float,
    agent_id: str,
    destination: tuple[float, float],
    mode: str,
    actual_travel_time: float,
    corridor_id: Optional[str] = None,
) -> Event:
    """Create an arrival event."""
    return Event(
        time=time,
        event_type=EventType.ARRIVAL,
        agent_id=agent_id,
        data={
            "destination": destination,
            "mode": mode,
            "actual_travel_time": actual_travel_time,
            "corridor_id": corridor_id,
        },
    )


def create_mode_choice_event(
    time: float,
    agent_id: str,
    available_modes: list[str],
    trip_options: dict[str, dict],
) -> Event:
    """Create a mode choice decision event."""
    return Event(
        time=time,
        event_type=EventType.MODE_CHOICE,
        agent_id=agent_id,
        data={
            "available_modes": available_modes,
            "trip_options": trip_options,
        },
    )


def create_pacing_update_event(
    time: float,
    agent_id: str,
    corridor_id: str,
    current_speed: float,
    position: tuple[float, float],
    session_id: str,
) -> Event:
    """Create a pacer speed update event."""
    return Event(
        time=time,
        event_type=EventType.PACING_UPDATE,
        agent_id=agent_id,
        data={
            "corridor_id": corridor_id,
            "current_speed": current_speed,
            "position": position,
            "session_id": session_id,
        },
    )


def create_metrics_snapshot_event(time: float) -> Event:
    """Create a periodic metrics collection event."""
    return Event(
        time=time,
        event_type=EventType.METRICS_SNAPSHOT,
        agent_id=None,
        data={},
    )
