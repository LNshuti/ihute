"""
Event Egress Experiment.

Simulates stadium/concert egress where incentives encourage staggered
departures to reduce congestion spikes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseExperiment, ExperimentConfig, ExperimentResult, create_parameter_grid

logger = logging.getLogger(__name__)


class EventEgressExperiment(BaseExperiment):
    """
    Experiment testing event egress incentive strategies.

    Scenario: 10,000 attendees leaving a stadium/concert venue.
    Incentives encourage some to delay departure to reduce peak congestion.

    Parameters varied:
    - delay_distribution:
        - uniform: spread incentives evenly across time windows
        - concentrated: higher incentives for earliest/latest slots
    - total_delay_budget: total person-minutes of delay to incentivize

    Metrics collected:
    - peak_demand: max vehicles departing in any 5-min window
    - avg_wait_time: time from event end to actual departure
    - congestion_duration: minutes until free-flow restored
    - incentive_cost: total spending
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # Override n_agents for stadium scenario
        self.venue_capacity = 10000

    def get_parameter_grid(self, **cli_args) -> list[dict[str, Any]]:
        """Generate parameter combinations."""
        delay_distributions = cli_args.get(
            "delay_distribution", ["uniform", "concentrated"]
        )
        delay_budgets = cli_args.get("total_delay_budget", [1000])

        # Handle single value
        if isinstance(delay_budgets, (int, float)):
            delay_budgets = [delay_budgets]

        return create_parameter_grid(
            delay_distribution=delay_distributions,
            total_delay_budget=delay_budgets,
        )

    def run_single(
        self,
        params: dict[str, Any],
        replication: int,
        rng: np.random.Generator,
    ) -> ExperimentResult:
        """Run a single replication."""
        from src.simulation import (
            SimulationConfig,
            SimulationEngine,
            create_stadium_network,
        )
        from src.agents.base import AgentPreferences, generate_heterogeneous_preferences, PopulationParameters
        from src.agents.commuter import CommuterAgent, CommuterProfile

        delay_distribution = params["delay_distribution"]
        total_delay_budget = params["total_delay_budget"]  # person-minutes

        n_attendees = self.venue_capacity

        # Create stadium network
        stadium_location = (36.166, -86.771)  # Nissan Stadium area
        network = create_stadium_network(stadium_location, n_attendees)

        # Event ends at simulation start (time 0 represents event end)
        event_end_time = 0  # All departures are relative to event end

        # Create attendee population
        pop_params = PopulationParameters(n_agents=n_attendees)
        attendees = []

        # Destination zones (where people are going after event)
        dest_zones = [
            ((36.18, -86.78), (36.22, -86.72)),  # North
            ((36.10, -86.83), (36.14, -86.77)),  # South
            ((36.14, -86.68), (36.18, -86.62)),  # East
            ((36.14, -86.88), (36.18, -86.82)),  # West
        ]

        for i in range(n_attendees):
            preferences = generate_heterogeneous_preferences(pop_params, rng)

            # Random destination zone
            zone_idx = rng.integers(0, len(dest_zones))
            zone = dest_zones[zone_idx]
            destination = (
                rng.uniform(zone[0][0], zone[1][0]),
                rng.uniform(zone[0][1], zone[1][1]),
            )

            profile = CommuterProfile(
                home_location=destination,  # Going home
                work_location=stadium_location,  # Coming from stadium
                desired_arrival_time=event_end_time,  # Want to leave at event end
                flexibility_window=3600,  # 1 hour flexibility
                has_car=True,
                carpool_eligible=False,
            )

            agent = CommuterAgent(
                agent_id=f"attendee_{i:05d}",
                preferences=preferences,
                profile=profile,
                rng=rng,
            )
            attendees.append(agent)

        # Design incentive schedule based on distribution type
        # Incentive windows: 0-15 min, 15-30 min, 30-45 min, 45-60 min after event
        windows = [
            (0, 900),      # 0-15 min
            (900, 1800),   # 15-30 min
            (1800, 2700),  # 30-45 min
            (2700, 3600),  # 45-60 min
        ]

        if delay_distribution == "uniform":
            # Equal incentives across all delay windows
            incentive_per_window = total_delay_budget / len(windows)
            window_incentives = [incentive_per_window] * len(windows)
        elif delay_distribution == "concentrated":
            # Higher incentives for later windows (encourage longer delays)
            # Weights: 10%, 20%, 30%, 40%
            weights = [0.1, 0.2, 0.3, 0.4]
            window_incentives = [w * total_delay_budget for w in weights]
        else:
            window_incentives = [0] * len(windows)

        # Convert to per-person incentive ($/minute of delay)
        # Assume ~20% will accept incentive in each window
        acceptance_rate = 0.2
        per_person_rate = 0.10  # $0.10 per minute of delay

        # Setup simulation
        sim_config = SimulationConfig(
            duration_seconds=2 * 3600,  # 2 hours post-event
            n_agents=n_attendees,
            random_seed=self.config.random_seed + replication,
        )

        engine = SimulationEngine(sim_config, network, rng)
        engine.add_agents(attendees)

        # Track departures and incentives
        departure_times = []
        total_incentive_cost = 0.0
        incentive_acceptances = 0

        # Schedule departures
        for agent in attendees:
            # Base departure: everyone wants to leave immediately
            base_departure = event_end_time

            # Decide if agent accepts delay incentive
            accepted_delay = 0
            incentive_paid = 0.0

            # Check each window for incentive acceptance
            for window_idx, (win_start, win_end) in enumerate(windows):
                if window_incentives[window_idx] <= 0:
                    continue

                # Incentive amount for this window
                delay_minutes = (win_start + win_end) / 2 / 60  # Mid-window delay
                offered_amount = per_person_rate * delay_minutes

                # Agent decision based on preferences
                # Higher beta_incentive = more likely to accept
                accept_prob = min(1.0, agent.preferences.beta_incentive * offered_amount * 2)

                if rng.random() < accept_prob:
                    # Accept this window's incentive
                    accepted_delay = rng.uniform(win_start, win_end)
                    incentive_paid = offered_amount
                    incentive_acceptances += 1
                    break  # Only accept one window

            # Final departure time
            departure_time = base_departure + accepted_delay + rng.exponential(60)
            departure_times.append(departure_time)
            total_incentive_cost += incentive_paid

            # Determine egress corridor based on destination
            dest = agent.profile.home_location
            if dest[0] > stadium_location[0]:  # North
                corridor = "egress-north"
            elif dest[0] < stadium_location[0]:  # South
                corridor = "egress-south"
            elif dest[1] > stadium_location[1]:  # East
                corridor = "egress-east"
            else:  # West
                corridor = "egress-west"

            engine.schedule_departure(
                agent_id=agent.id,
                time=departure_time,
                origin=stadium_location,
                destination=agent.profile.home_location,
                mode="drive",
                corridor_id=corridor,
            )

        # Run simulation
        result = engine.run()

        # Compute egress-specific metrics
        departure_times = np.array(departure_times)

        # Peak demand: max departures in 5-minute window
        window_size = 300  # 5 minutes
        min_time = departure_times.min()
        max_time = departure_times.max()
        n_windows = int((max_time - min_time) / window_size) + 1

        window_counts = np.zeros(n_windows)
        for dt in departure_times:
            window_idx = int((dt - min_time) / window_size)
            if 0 <= window_idx < n_windows:
                window_counts[window_idx] += 1

        peak_demand = int(window_counts.max())

        # Average wait time (from event end)
        wait_times = departure_times - event_end_time
        avg_wait_time = np.mean(wait_times)
        max_wait_time = np.max(wait_times)

        # Congestion duration (time until 90% have departed)
        sorted_departures = np.sort(departure_times)
        pct_90_idx = int(0.9 * len(sorted_departures))
        congestion_duration = sorted_departures[pct_90_idx] - event_end_time

        # Departure spread (std dev of departure times)
        departure_spread = np.std(departure_times)

        metrics = {
            "peak_demand": peak_demand,
            "avg_wait_time": avg_wait_time,
            "max_wait_time": max_wait_time,
            "congestion_duration": congestion_duration,
            "departure_spread": departure_spread,
            "total_incentive_cost": total_incentive_cost,
            "incentive_acceptances": incentive_acceptances,
            "acceptance_rate": incentive_acceptances / n_attendees,
            "cost_per_minute_spread": (
                total_incentive_cost / (departure_spread / 60)
                if departure_spread > 0 else 0
            ),
            "total_attendees": n_attendees,
            "avg_travel_time": result.metrics.get("avg_travel_time", 0),
        }

        return ExperimentResult(
            experiment_name=self.config.name,
            parameters=params,
            replication=replication,
            metrics=metrics,
            raw_data=result.raw_data,
        )
