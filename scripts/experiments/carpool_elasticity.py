"""
Carpool Elasticity Experiment.

Tests how reward levels and targeting precision affect carpool adoption
and cost-effectiveness.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseExperiment, ExperimentConfig, ExperimentResult, create_parameter_grid

logger = logging.getLogger(__name__)


class CarpoolElasticityExperiment(BaseExperiment):
    """
    Experiment testing carpool incentive elasticity.

    Parameters varied:
    - reward_per_passenger: $1, $2, $5, $10
    - targeting_precision:
        - low: offer to all eligible agents
        - medium: offer to agents with high carpool utility
        - high: ML-based targeting to maximize uptake per dollar

    Metrics collected:
    - carpool_rate: fraction choosing carpool
    - vmt_reduction: vehicle miles traveled reduction
    - cost_per_vmt_reduced: efficiency metric
    - match_rate: successful carpool matches
    """

    def get_parameter_grid(self, **cli_args) -> list[dict[str, Any]]:
        """Generate parameter combinations."""
        reward_levels = cli_args.get("reward_levels", [1.0, 2.0, 5.0, 10.0])
        targeting_precision = cli_args.get(
            "targeting_precision", ["low", "medium", "high"]
        )

        return create_parameter_grid(
            reward_per_passenger=reward_levels,
            targeting_precision=targeting_precision,
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
            create_i24_network,
        )
        from src.agents.commuter import create_commuter_population
        from src.agents.base import TravelMode, TripAttributes
        from src.agents.behavioral import LogitModel
        from src.incentives.carpool import CarpoolIncentive

        reward_per_passenger = params["reward_per_passenger"]
        targeting = params["targeting_precision"]
        n_agents = self.config.n_agents

        # Create network
        network = create_i24_network()
        corridor_id = "I-24-inbound"

        # Home and work regions
        home_region = ((36.03, -86.70), (36.13, -86.60))
        work_region = ((36.14, -86.82), (36.18, -86.74))

        # Create commuter population
        commuters = create_commuter_population(
            n_agents=n_agents,
            home_region=home_region,
            work_region=work_region,
            rng=rng,
        )

        # Setup carpool incentive
        carpool_incentive = CarpoolIncentive(
            reward_per_passenger=reward_per_passenger,
            min_passengers=1,
            max_reward=reward_per_passenger * 4,  # Cap at 4 passengers worth
        )

        # Setup simulation
        sim_config = SimulationConfig(
            duration_seconds=self.config.duration_hours * 3600,
            n_agents=n_agents,
            corridor_ids=[corridor_id],
            random_seed=self.config.random_seed + replication,
        )

        engine = SimulationEngine(sim_config, network, rng)
        engine.add_agents(commuters)

        # Behavioral model for mode choice
        logit_model = LogitModel(scale=1.0)

        # Track metrics
        mode_choices = {"drive_alone": 0, "carpool": 0}
        total_vmt = 0.0
        carpool_vmt = 0.0
        total_incentive_cost = 0.0
        matches_attempted = 0
        matches_successful = 0

        # Schedule departures with mode choice (spread across simulation)
        for agent in commuters:
            departure_time = rng.exponential(sim_config.duration_seconds / 3)
            departure_time = min(departure_time, sim_config.duration_seconds * 0.8)

            origin = agent.profile.home_location
            destination = agent.profile.work_location
            distance = network._haversine_distance(origin, destination)

            # Compute trip options
            drive_time = network.get_travel_time(origin, destination, "drive", corridor_id)
            carpool_time = network.get_travel_time(origin, destination, "carpool", corridor_id)

            # Check eligibility for carpool incentive
            context = {
                "hour": int(departure_time / 3600) % 24,
                "corridor_id": corridor_id,
                "carpool_eligible": agent.profile.carpool_eligible,
            }

            eligible, _ = carpool_incentive.check_eligibility(agent.id, context)

            # Apply targeting filter
            offer_incentive = False
            if eligible:
                if targeting == "low":
                    # Offer to everyone eligible
                    offer_incentive = True
                elif targeting == "medium":
                    # Offer to agents with higher carpool preference
                    if agent.preferences.asc_carpool > -1.0:
                        offer_incentive = True
                elif targeting == "high":
                    # ML-based: offer to agents most likely to switch
                    # Simplified: offer to those with high incentive sensitivity
                    if agent.preferences.beta_incentive > 0.12:
                        offer_incentive = True

            # Compute expected incentive
            expected_incentive = 0.0
            if offer_incentive:
                # Assume average 1.5 passengers
                expected_incentive = reward_per_passenger * 1.5

            # Create trip options for mode choice
            drive_option = TripAttributes(
                mode=TravelMode.DRIVE_ALONE,
                travel_time=drive_time,
                cost=distance * 0.50,  # $0.50/mile
                incentive=0.0,
            )

            carpool_option = TripAttributes(
                mode=TravelMode.CARPOOL_PASSENGER,
                travel_time=carpool_time,
                cost=distance * 0.25,  # Split costs
                incentive=expected_incentive if offer_incentive else 0.0,
            )

            # Agent chooses mode
            options = [drive_option, carpool_option]
            choice_idx = logit_model.choose_action(agent.preferences, options, rng)
            chosen_mode = "carpool" if choice_idx == 1 else "drive_alone"

            mode_choices[chosen_mode] += 1
            total_vmt += distance

            if chosen_mode == "carpool":
                carpool_vmt += distance
                matches_attempted += 1

                # Simulate match success (simplified)
                if rng.random() < 0.7:  # 70% match rate
                    matches_successful += 1
                    # Pay incentive
                    actual_passengers = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
                    actual_reward = reward_per_passenger * actual_passengers
                    total_incentive_cost += actual_reward

            # Schedule departure
            engine.schedule_departure(
                agent_id=agent.id,
                time=departure_time,
                origin=origin,
                destination=destination,
                mode=chosen_mode,
                corridor_id=corridor_id,
            )

        # Run simulation
        result = engine.run()

        # Compute metrics
        total_trips = sum(mode_choices.values())
        carpool_rate = mode_choices["carpool"] / total_trips if total_trips > 0 else 0

        # VMT reduction: carpool trips would have been 2 drive-alone trips
        # So reduction = carpool_vmt (saved the duplicate trip)
        vmt_reduction = carpool_vmt
        vmt_reduction_pct = vmt_reduction / total_vmt if total_vmt > 0 else 0

        # Cost efficiency
        cost_per_vmt = total_incentive_cost / vmt_reduction if vmt_reduction > 0 else float("inf")

        match_rate = matches_successful / matches_attempted if matches_attempted > 0 else 0

        metrics = {
            "carpool_rate": carpool_rate,
            "drive_alone_rate": 1 - carpool_rate,
            "total_vmt": total_vmt,
            "vmt_reduction": vmt_reduction,
            "vmt_reduction_pct": vmt_reduction_pct,
            "total_incentive_cost": total_incentive_cost,
            "cost_per_vmt_reduced": cost_per_vmt,
            "matches_attempted": matches_attempted,
            "matches_successful": matches_successful,
            "match_rate": match_rate,
            "avg_travel_time": result.metrics.get("avg_travel_time", 0),
            "total_trips": total_trips,
        }

        return ExperimentResult(
            experiment_name=self.config.name,
            parameters=params,
            replication=replication,
            metrics=metrics,
            raw_data=result.raw_data,
        )
