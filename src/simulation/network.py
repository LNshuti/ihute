"""
Simple road network representation for simulation.

Provides travel time estimation based on distance and congestion levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Corridor:
    """A major traffic corridor (e.g., I-24)."""

    corridor_id: str
    name: str
    length_miles: float
    free_flow_speed: float = 65.0  # mph
    capacity_vph: float = 2000.0  # vehicles per hour per lane
    num_lanes: int = 3
    direction: str = "inbound"  # inbound or outbound

    # Current state
    current_volume: float = 0.0
    current_speed: float = 65.0

    def get_travel_time(self, congestion_factor: float = 1.0) -> float:
        """Get travel time in seconds given congestion."""
        effective_speed = self.free_flow_speed / max(1.0, congestion_factor)
        effective_speed = max(5.0, effective_speed)  # Minimum 5 mph
        return (self.length_miles / effective_speed) * 3600

    def get_congestion_factor(self) -> float:
        """Compute congestion factor based on volume/capacity ratio."""
        total_capacity = self.capacity_vph * self.num_lanes
        vc_ratio = self.current_volume / max(1.0, total_capacity)

        # BPR function: t = t0 * (1 + alpha * (v/c)^beta)
        alpha = 0.15
        beta = 4.0
        return 1.0 + alpha * (vc_ratio**beta)


@dataclass
class NetworkNode:
    """A node in the road network (intersection, on/off ramp)."""

    node_id: str
    location: tuple[float, float]  # lat, lng
    node_type: str = "intersection"  # intersection, origin, destination, ramp


@dataclass
class NetworkLink:
    """A link connecting two nodes."""

    link_id: str
    from_node: str
    to_node: str
    length_miles: float
    free_flow_speed: float = 35.0
    capacity_vph: float = 1200.0
    corridor_id: Optional[str] = None

    current_volume: float = 0.0

    def get_travel_time(self) -> float:
        """Get travel time in seconds."""
        congestion = self.get_congestion_factor()
        effective_speed = self.free_flow_speed / congestion
        effective_speed = max(5.0, effective_speed)
        return (self.length_miles / effective_speed) * 3600

    def get_congestion_factor(self) -> float:
        """BPR congestion function."""
        vc_ratio = self.current_volume / max(1.0, self.capacity_vph)
        return 1.0 + 0.15 * (vc_ratio**4.0)


@dataclass
class SimpleNetwork:
    """
    Simple network for simulation experiments.

    Represents the I-24 corridor with origin/destination zones.
    """

    corridors: dict[str, Corridor] = field(default_factory=dict)
    nodes: dict[str, NetworkNode] = field(default_factory=dict)
    links: dict[str, NetworkLink] = field(default_factory=dict)

    # Zone definitions
    origin_zones: list[tuple[float, float, float]] = field(
        default_factory=list
    )  # (lat, lng, radius)
    destination_zones: list[tuple[float, float, float]] = field(default_factory=list)

    def add_corridor(self, corridor: Corridor) -> None:
        """Add a corridor to the network."""
        self.corridors[corridor.corridor_id] = corridor

    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the network."""
        self.nodes[node.node_id] = node

    def add_link(self, link: NetworkLink) -> None:
        """Add a link to the network."""
        self.links[link.link_id] = link

    def get_travel_time(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        mode: str = "drive",
        corridor_id: Optional[str] = None,
    ) -> float:
        """
        Estimate travel time between two points.

        Args:
            origin: (lat, lng) of origin
            destination: (lat, lng) of destination
            mode: travel mode
            corridor_id: specific corridor to use (if applicable)

        Returns:
            Travel time in seconds
        """
        distance = self._haversine_distance(origin, destination)

        if mode == "drive" or mode == "drive_alone":
            # Check if using a corridor
            if corridor_id and corridor_id in self.corridors:
                corridor = self.corridors[corridor_id]
                congestion = corridor.get_congestion_factor()
                return corridor.get_travel_time(congestion)

            # Default driving: assume 30 mph average with congestion
            base_speed = 30.0
            congestion = self._estimate_area_congestion()
            effective_speed = base_speed / congestion
            return (distance / effective_speed) * 3600

        elif mode == "carpool":
            # Carpool similar to drive but with pickup detour
            drive_time = self.get_travel_time(origin, destination, "drive", corridor_id)
            detour_time = 300  # 5 minute average detour
            return drive_time + detour_time

        elif mode == "transit":
            # Transit slower but less affected by congestion
            base_speed = 20.0
            wait_time = 600  # 10 minute average wait
            return (distance / base_speed) * 3600 + wait_time

        elif mode == "walk":
            walk_speed = 3.0  # mph
            return (distance / walk_speed) * 3600

        elif mode == "bike":
            bike_speed = 12.0  # mph
            return (distance / bike_speed) * 3600

        else:
            # Default
            return (distance / 25.0) * 3600

    def update_volumes(self, departures: dict[str, int]) -> None:
        """
        Update corridor volumes based on departures.

        Args:
            departures: dict mapping corridor_id to number of vehicles
        """
        for corridor_id, volume in departures.items():
            if corridor_id in self.corridors:
                self.corridors[corridor_id].current_volume = volume

    def reset_volumes(self) -> None:
        """Reset all volumes to zero."""
        for corridor in self.corridors.values():
            corridor.current_volume = 0.0
        for link in self.links.values():
            link.current_volume = 0.0

    def _estimate_area_congestion(self) -> float:
        """Estimate overall area congestion from corridor states."""
        if not self.corridors:
            return 1.0

        factors = [c.get_congestion_factor() for c in self.corridors.values()]
        return np.mean(factors)

    def _haversine_distance(
        self,
        coord1: tuple[float, float],
        coord2: tuple[float, float],
    ) -> float:
        """Compute haversine distance in miles."""
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        r = 3956  # Earth radius in miles
        return c * r


def create_i24_network() -> SimpleNetwork:
    """
    Create a simple I-24 corridor network for Nashville.

    Returns:
        SimpleNetwork configured for I-24 experiments
    """
    network = SimpleNetwork()

    # Add I-24 corridor (inbound and outbound)
    network.add_corridor(
        Corridor(
            corridor_id="I-24-inbound",
            name="I-24 Inbound",
            length_miles=15.0,
            free_flow_speed=65.0,
            capacity_vph=2000.0,
            num_lanes=3,
            direction="inbound",
        )
    )
    network.add_corridor(
        Corridor(
            corridor_id="I-24-outbound",
            name="I-24 Outbound",
            length_miles=15.0,
            free_flow_speed=65.0,
            capacity_vph=2000.0,
            num_lanes=3,
            direction="outbound",
        )
    )

    # Define zones
    # Origin zone: Southeast Nashville suburbs
    network.origin_zones = [(36.08, -86.65, 10.0)]  # center lat, lng, radius km

    # Destination zone: Downtown Nashville
    network.destination_zones = [(36.16, -86.78, 5.0)]

    return network


def create_stadium_network(
    stadium_location: tuple[float, float] = (36.166, -86.771),
    capacity: int = 10000,
) -> SimpleNetwork:
    """
    Create a network for stadium egress simulation.

    Args:
        stadium_location: (lat, lng) of stadium
        capacity: venue capacity

    Returns:
        SimpleNetwork configured for event egress
    """
    network = SimpleNetwork()

    # Add egress corridors (multiple directions)
    for i, (name, direction, length) in enumerate(
        [
            ("North Exit", "north", 2.0),
            ("South Exit", "south", 2.5),
            ("East Exit", "east", 1.5),
            ("West Exit", "west", 3.0),
        ]
    ):
        network.add_corridor(
            Corridor(
                corridor_id=f"egress-{direction}",
                name=name,
                length_miles=length,
                free_flow_speed=25.0,  # Slower urban streets
                capacity_vph=800.0,
                num_lanes=2,
                direction=direction,
            )
        )

    # Stadium as single origin
    network.origin_zones = [(stadium_location[0], stadium_location[1], 0.5)]

    # Dispersed destinations
    network.destination_zones = [
        (36.20, -86.75, 5.0),  # North
        (36.12, -86.80, 5.0),  # South
        (36.16, -86.65, 5.0),  # East
        (36.16, -86.85, 5.0),  # West
    ]

    return network
