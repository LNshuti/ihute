"""
Metrics collection and aggregation for simulation experiments.

Collects time-series and aggregate metrics for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class TripRecord:
    """Record of a completed trip."""

    agent_id: str
    departure_time: float
    arrival_time: float
    origin: tuple[float, float]
    destination: tuple[float, float]
    mode: str
    travel_time: float
    distance_miles: float
    incentive_received: float = 0.0
    corridor_id: Optional[str] = None


@dataclass
class PacerRecord:
    """Record of a pacer session."""

    agent_id: str
    session_id: str
    corridor_id: str
    start_time: float
    end_time: float
    distance_miles: float
    smoothness_score: float
    reward: float
    speed_samples: list[float] = field(default_factory=list)


@dataclass
class CarpoolRecord:
    """Record of a carpool trip."""

    match_id: str
    driver_id: str
    passenger_ids: list[str]
    departure_time: float
    arrival_time: float
    distance_miles: float
    total_reward: float
    n_passengers: int


@dataclass
class TimeSliceMetrics:
    """Metrics for a time slice (e.g., 5-minute window)."""

    time: float
    departures: int = 0
    arrivals: int = 0
    active_trips: int = 0
    avg_speed: float = 0.0
    speed_variance: float = 0.0
    mode_shares: dict[str, float] = field(default_factory=dict)
    corridor_volumes: dict[str, int] = field(default_factory=dict)
    incentive_spending: float = 0.0


class MetricsCollector:
    """
    Collects and aggregates simulation metrics.

    Provides both real-time snapshots and post-simulation analysis.
    """

    def __init__(self, snapshot_interval: float = 300.0):
        """
        Initialize metrics collector.

        Args:
            snapshot_interval: Seconds between time-slice snapshots
        """
        self.snapshot_interval = snapshot_interval

        # Trip records
        self.trips: list[TripRecord] = []
        self.pacer_sessions: list[PacerRecord] = []
        self.carpool_trips: list[CarpoolRecord] = []

        # Time series
        self.time_slices: list[TimeSliceMetrics] = []

        # Running counters
        self.total_departures = 0
        self.total_arrivals = 0
        self.total_incentive_cost = 0.0
        self.mode_counts: dict[str, int] = {}

        # Speed tracking for variance
        self.speed_observations: list[float] = []
        self.current_slice_speeds: list[float] = []

    def record_trip(self, trip: TripRecord) -> None:
        """Record a completed trip."""
        self.trips.append(trip)
        self.total_arrivals += 1
        self.total_incentive_cost += trip.incentive_received

        mode = trip.mode
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1

    def record_departure(
        self, agent_id: str, mode: str, corridor_id: Optional[str] = None
    ) -> None:
        """Record a departure."""
        self.total_departures += 1

    def record_pacer_session(self, record: PacerRecord) -> None:
        """Record a completed pacer session."""
        self.pacer_sessions.append(record)
        self.total_incentive_cost += record.reward

    def record_carpool(self, record: CarpoolRecord) -> None:
        """Record a completed carpool trip."""
        self.carpool_trips.append(record)
        self.total_incentive_cost += record.total_reward

    def record_speed(self, speed: float) -> None:
        """Record a speed observation."""
        self.speed_observations.append(speed)
        self.current_slice_speeds.append(speed)

    def take_snapshot(
        self,
        time: float,
        active_trips: int,
        corridor_volumes: dict[str, int],
    ) -> TimeSliceMetrics:
        """Take a time-slice snapshot."""
        speeds = self.current_slice_speeds if self.current_slice_speeds else [0.0]

        snapshot = TimeSliceMetrics(
            time=time,
            departures=self.total_departures,
            arrivals=self.total_arrivals,
            active_trips=active_trips,
            avg_speed=np.mean(speeds),
            speed_variance=np.var(speeds) if len(speeds) > 1 else 0.0,
            mode_shares=self._compute_mode_shares(),
            corridor_volumes=corridor_volumes.copy(),
            incentive_spending=self.total_incentive_cost,
        )

        self.time_slices.append(snapshot)
        self.current_slice_speeds = []  # Reset for next slice

        return snapshot

    def _compute_mode_shares(self) -> dict[str, float]:
        """Compute current mode shares."""
        total = sum(self.mode_counts.values())
        if total == 0:
            return {}
        return {mode: count / total for mode, count in self.mode_counts.items()}

    def get_summary_metrics(self) -> dict[str, Any]:
        """
        Compute summary metrics for the simulation.

        Returns:
            Dictionary of aggregate metrics
        """
        if not self.trips:
            return {
                "total_trips": 0,
                "avg_travel_time": 0.0,
                "total_incentive_cost": 0.0,
            }

        travel_times = [t.travel_time for t in self.trips]
        distances = [t.distance_miles for t in self.trips]

        # Speed variance across all observations
        speed_var = np.var(self.speed_observations) if self.speed_observations else 0.0

        # VMT calculation
        total_vmt = sum(distances)
        drive_alone_vmt = sum(
            t.distance_miles for t in self.trips if t.mode in ("drive", "drive_alone")
        )

        # Carpool metrics
        carpool_trips = [t for t in self.trips if t.mode == "carpool"]
        carpool_rate = len(carpool_trips) / len(self.trips) if self.trips else 0.0

        return {
            "total_trips": len(self.trips),
            "total_departures": self.total_departures,
            "total_arrivals": self.total_arrivals,
            "avg_travel_time": np.mean(travel_times),
            "median_travel_time": np.median(travel_times),
            "std_travel_time": np.std(travel_times),
            "total_vmt": total_vmt,
            "drive_alone_vmt": drive_alone_vmt,
            "vmt_reduction": (total_vmt - drive_alone_vmt) / max(1, total_vmt),
            "speed_variance": speed_var,
            "avg_speed": (
                np.mean(self.speed_observations) if self.speed_observations else 0.0
            ),
            "mode_shares": self._compute_mode_shares(),
            "carpool_rate": carpool_rate,
            "total_incentive_cost": self.total_incentive_cost,
            "n_pacer_sessions": len(self.pacer_sessions),
            "n_carpool_matches": len(self.carpool_trips),
        }

    def get_pacer_metrics(self) -> dict[str, Any]:
        """Get pacer-specific metrics."""
        if not self.pacer_sessions:
            return {
                "n_sessions": 0,
                "total_pacer_miles": 0.0,
                "avg_smoothness": 0.0,
                "total_pacer_cost": 0.0,
            }

        smoothness_scores = [s.smoothness_score for s in self.pacer_sessions]
        distances = [s.distance_miles for s in self.pacer_sessions]
        rewards = [s.reward for s in self.pacer_sessions]

        return {
            "n_sessions": len(self.pacer_sessions),
            "total_pacer_miles": sum(distances),
            "avg_smoothness": np.mean(smoothness_scores),
            "smoothness_std": np.std(smoothness_scores),
            "total_pacer_cost": sum(rewards),
            "success_rate": np.mean(
                [1 if s.reward > 0 else 0 for s in self.pacer_sessions]
            ),
        }

    def get_carpool_metrics(self) -> dict[str, Any]:
        """Get carpool-specific metrics."""
        if not self.carpool_trips:
            return {
                "n_matches": 0,
                "total_carpool_miles": 0.0,
                "avg_passengers": 0.0,
                "total_carpool_cost": 0.0,
            }

        return {
            "n_matches": len(self.carpool_trips),
            "total_carpool_miles": sum(c.distance_miles for c in self.carpool_trips),
            "avg_passengers": np.mean([c.n_passengers for c in self.carpool_trips]),
            "total_carpool_cost": sum(c.total_reward for c in self.carpool_trips),
        }

    def get_egress_metrics(self, event_end_time: float) -> dict[str, Any]:
        """
        Get event egress-specific metrics.

        Args:
            event_end_time: When the event ended (simulation time)

        Returns:
            Egress-specific metrics
        """
        if not self.trips:
            return {
                "peak_demand": 0,
                "avg_wait_time": 0.0,
                "congestion_duration": 0.0,
            }

        # Departures per 5-minute window
        departure_times = [t.departure_time for t in self.trips]
        window_size = 300  # 5 minutes

        if departure_times:
            min_time = min(departure_times)
            max_time = max(departure_times)
            n_windows = int((max_time - min_time) / window_size) + 1

            window_counts = [0] * n_windows
            for dt in departure_times:
                window_idx = int((dt - min_time) / window_size)
                if 0 <= window_idx < n_windows:
                    window_counts[window_idx] += 1

            peak_demand = max(window_counts)
        else:
            peak_demand = 0

        # Wait time: time from event end to departure
        wait_times = [max(0, t.departure_time - event_end_time) for t in self.trips]
        avg_wait_time = np.mean(wait_times) if wait_times else 0.0

        # Congestion duration: time until last departure
        last_departure = max(departure_times) if departure_times else event_end_time
        congestion_duration = last_departure - event_end_time

        return {
            "peak_demand": peak_demand,
            "avg_wait_time": avg_wait_time,
            "max_wait_time": max(wait_times) if wait_times else 0.0,
            "congestion_duration": congestion_duration,
            "total_departures": len(self.trips),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trip records to DataFrame."""
        if not self.trips:
            return pd.DataFrame()

        records = [
            {
                "agent_id": t.agent_id,
                "departure_time": t.departure_time,
                "arrival_time": t.arrival_time,
                "origin_lat": t.origin[0],
                "origin_lng": t.origin[1],
                "dest_lat": t.destination[0],
                "dest_lng": t.destination[1],
                "mode": t.mode,
                "travel_time": t.travel_time,
                "distance_miles": t.distance_miles,
                "incentive": t.incentive_received,
                "corridor_id": t.corridor_id,
            }
            for t in self.trips
        ]

        return pd.DataFrame(records)

    def reset(self) -> None:
        """Reset all metrics."""
        self.trips = []
        self.pacer_sessions = []
        self.carpool_trips = []
        self.time_slices = []
        self.total_departures = 0
        self.total_arrivals = 0
        self.total_incentive_cost = 0.0
        self.mode_counts = {}
        self.speed_observations = []
        self.current_slice_speeds = []
