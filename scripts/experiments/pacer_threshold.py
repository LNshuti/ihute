"""
Pacer Threshold Experiment.

Tests how pacer driver participation rates affect traffic flow stability
on congested corridors.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseExperiment, ExperimentConfig, ExperimentResult, create_parameter_grid

logger = logging.getLogger(__name__)


class PacerThresholdExperiment(BaseExperiment):
    """
    Experiment testing pacer driver participation thresholds.

    Hypothesis: There's a threshold participation rate above which
    traffic stabilization benefits plateau.

    Parameters varied:
    - participation_rate: fraction of drivers enrolled as pacers

    Metrics collected:
    - speed_variance: measure of stop-and-go waves (lower = better)
    - avg_travel_time: corridor travel time
    - throughput: vehicles per hour
    - total_pacer_cost: incentive spending
    - smoothness_score: avg pacer performance
    """

    def get_parameter_grid(self, **cli_args) -> list[dict[str, Any]]:
        """Generate parameter combinations."""
        participation_rates = cli_args.get(
            "participation_rates", [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        )

        return create_parameter_grid(participation_rate=participation_rates)

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
            create_i24_network,
        )
        from src.agents.pacer import create_pacer_population
        from src.agents.commuter import create_commuter_population
        from src.incentives.pacer import PacerIncentive

        participation_rate = params["participation_rate"]
        n_agents = self.config.n_agents

        # Split agents between pacers and regular commuters
        n_pacers = int(n_agents * participation_rate)
        n_commuters = n_agents - n_pacers

        # Create network
        network = create_i24_network()
        corridor_id = "I-24-inbound"

        # Home and work regions
        home_region = ((36.03, -86.70), (36.13, -86.60))
        work_region = ((36.14, -86.82), (36.18, -86.74))

        # Create agent populations
        agents = []

        if n_pacers > 0:
            pacers = create_pacer_population(
                n_agents=n_pacers,
                home_region=home_region,
                work_region=work_region,
                corridors=[corridor_id],
                rng=rng,
            )
            agents.extend(pacers)

        if n_commuters > 0:
            commuters = create_commuter_population(
                n_agents=n_commuters,
                home_region=home_region,
                work_region=work_region,
                rng=rng,
            )
            agents.extend(commuters)

        # Setup simulation
        sim_config = SimulationConfig(
            duration_seconds=self.config.duration_hours * 3600,
            n_agents=n_agents,
            corridor_ids=[corridor_id],
            random_seed=self.config.random_seed + replication,
        )

        engine = SimulationEngine(sim_config, network, rng)
        engine.add_agents(agents)

        # Setup pacer incentive
        pacer_incentive = PacerIncentive(
            reward_per_mile=0.15,
            smoothness_threshold=0.7,
        )
        pacer_incentive.set_corridor_target(corridor_id, 55.0)

        # Schedule departures (spread across simulation duration)
        # Departures follow exponential distribution concentrated in first half
        for agent in agents:
            # Random departure within simulation period
            departure_time = rng.exponential(sim_config.duration_seconds / 3)
            departure_time = min(departure_time, sim_config.duration_seconds * 0.8)

            origin = agent.profile.home_location if hasattr(agent, 'profile') else (36.08, -86.65)
            destination = agent.profile.work_location if hasattr(agent, 'profile') else (36.16, -86.78)

            # Determine mode - pacers drive with pacing behavior
            mode = "pacer" if hasattr(agent, 'is_pacing') else "drive"

            engine.schedule_departure(
                agent_id=agent.id,
                time=departure_time,
                origin=origin,
                destination=destination,
                mode=mode,
                corridor_id=corridor_id,
            )

        # Run simulation
        result = engine.run()

        # Compute experiment-specific metrics
        metrics = result.metrics.copy()

        # Add pacer-specific metrics
        if n_pacers > 0:
            pacer_stats = pacer_incentive.get_statistics()
            metrics["pacer_participation_rate"] = participation_rate
            metrics["n_pacers"] = n_pacers
            metrics["pacer_success_rate"] = pacer_stats.get("success_rate", 0)
            metrics["avg_pacer_smoothness"] = pacer_stats.get("avg_smoothness", 0)

        # Compute throughput (vehicles per hour)
        duration_hours = self.config.duration_hours
        metrics["throughput"] = metrics.get("total_trips", 0) / duration_hours

        # Cost per vehicle
        if metrics.get("total_trips", 0) > 0:
            metrics["cost_per_vehicle"] = (
                metrics.get("total_pacer_cost", 0) / metrics["total_trips"]
            )
        else:
            metrics["cost_per_vehicle"] = 0

        return ExperimentResult(
            experiment_name=self.config.name,
            parameters=params,
            replication=replication,
            metrics=metrics,
            raw_data=result.raw_data,
        )
