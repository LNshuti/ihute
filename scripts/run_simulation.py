#!/usr/bin/env python3
"""
Run a single simulation from a YAML configuration file.

Usage:
    python -m scripts.run_simulation --config configs/pacer_i24.yaml --agents 100000

Options:
    --config PATH       Path to YAML configuration file (required)
    --agents INT        Override number of agents from config
    --seed INT          Override random seed from config
    --output-dir PATH   Override output directory from config
    --duration FLOAT    Override simulation duration (hours)
    --verbose           Enable verbose logging
    --dry-run           Parse config and show settings without running
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(config: dict[str, Any], verbose: bool = False) -> None:
    """Setup logging based on configuration."""
    log_config = config.get("logging", {})
    level = logging.DEBUG if verbose else getattr(logging, log_config.get("level", "INFO"))

    # Create logs directory if needed
    log_file = log_config.get("file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
        ],
    )


def create_network(config: dict[str, Any]):
    """Create network from configuration."""
    from src.simulation import create_i24_network, SimpleNetwork, Corridor

    network_config = config.get("network", {})
    network_type = network_config.get("type", "i24")

    if network_type == "i24":
        network = create_i24_network()

        # Override corridor settings if specified
        corridor_config = network_config.get("corridor", {})
        if corridor_config and "I-24-inbound" in network.corridors:
            corridor = network.corridors["I-24-inbound"]
            corridor.length_miles = corridor_config.get("length_miles", corridor.length_miles)
            corridor.free_flow_speed = corridor_config.get("free_flow_speed_mph", corridor.free_flow_speed)
            corridor.capacity_vph = corridor_config.get("capacity_vph", corridor.capacity_vph)
            corridor.num_lanes = corridor_config.get("num_lanes", corridor.num_lanes)

        return network

    else:
        # Generic network
        network = SimpleNetwork()
        corridor_config = network_config.get("corridor", {})
        if corridor_config:
            network.add_corridor(
                Corridor(
                    corridor_id="main",
                    name=corridor_config.get("name", "Main Corridor"),
                    length_miles=corridor_config.get("length_miles", 10.0),
                    free_flow_speed=corridor_config.get("free_flow_speed_mph", 60.0),
                    capacity_vph=corridor_config.get("capacity_vph", 2000),
                    num_lanes=corridor_config.get("num_lanes", 3),
                )
            )
        return network


def create_agents(config: dict[str, Any], n_agents: int, rng: np.random.Generator):
    """Create agent population from configuration."""
    from src.agents.commuter import create_commuter_population
    from src.agents.pacer import create_pacer_population

    agents_config = config.get("agents", {})
    network_config = config.get("network", {})

    # Get regions from config
    origin = network_config.get("origin_region", {})
    origin_center = origin.get("center", [36.08, -86.65])
    origin_radius = origin.get("radius_km", 10) / 111  # Convert km to degrees (approx)

    dest = network_config.get("destination_region", {})
    dest_center = dest.get("center", [36.16, -86.78])
    dest_radius = dest.get("radius_km", 5) / 111

    # Bounding boxes
    home_region = (
        (origin_center[0] - origin_radius, origin_center[1] - origin_radius),
        (origin_center[0] + origin_radius, origin_center[1] + origin_radius),
    )
    work_region = (
        (dest_center[0] - dest_radius, dest_center[1] - dest_radius),
        (dest_center[0] + dest_radius, dest_center[1] + dest_radius),
    )

    # Split between pacers and commuters
    pacer_fraction = agents_config.get("pacer_fraction", 0.10)
    n_pacers = int(n_agents * pacer_fraction)
    n_commuters = n_agents - n_pacers

    agents = []

    # Create pacer agents
    if n_pacers > 0:
        logger.info(f"Creating {n_pacers} pacer agents...")
        pacers = create_pacer_population(
            n_agents=n_pacers,
            home_region=home_region,
            work_region=work_region,
            corridors=["I-24-inbound"],
            rng=rng,
        )
        agents.extend(pacers)

    # Create commuter agents
    if n_commuters > 0:
        logger.info(f"Creating {n_commuters} commuter agents...")

        # Get arrival time distribution
        arrival_config = agents_config.get("arrival_time", {})
        arrival_mean = arrival_config.get("mean_hour", 8.0) * 3600
        arrival_std = arrival_config.get("std_minutes", 30) * 60

        commuters = create_commuter_population(
            n_agents=n_commuters,
            home_region=home_region,
            work_region=work_region,
            arrival_time_dist=(arrival_mean, arrival_std),
            rng=rng,
        )
        agents.extend(commuters)

    return agents


def run_simulation(
    config: dict[str, Any],
    n_agents: int,
    seed: int,
    output_dir: Path,
    duration_hours: float,
) -> dict[str, Any]:
    """Run the simulation."""
    from src.simulation import SimulationConfig, SimulationEngine

    logger.info(f"Starting simulation: {config.get('simulation', {}).get('name', 'Unnamed')}")
    logger.info(f"  Agents: {n_agents:,}")
    logger.info(f"  Duration: {duration_hours} hours")
    logger.info(f"  Seed: {seed}")

    # Create RNG
    rng = np.random.default_rng(seed)

    # Create network
    logger.info("Creating network...")
    network = create_network(config)

    # Create agents
    logger.info("Creating agents...")
    agents = create_agents(config, n_agents, rng)

    # Setup simulation
    output_config = config.get("output", {})
    metrics_interval = output_config.get("metrics_interval_seconds", 300)

    sim_config = SimulationConfig(
        duration_seconds=duration_hours * 3600,
        metrics_interval=metrics_interval,
        n_agents=n_agents,
        random_seed=seed,
    )

    engine = SimulationEngine(sim_config, network, rng)
    engine.add_agents(agents)

    # Schedule departures
    logger.info("Scheduling departures...")
    corridor_id = "I-24-inbound"

    for agent in agents:
        # Departure time spread across simulation
        departure_time = rng.exponential(sim_config.duration_seconds / 3)
        departure_time = min(departure_time, sim_config.duration_seconds * 0.8)

        origin = agent.profile.home_location if hasattr(agent, 'profile') else (36.08, -86.65)
        destination = agent.profile.work_location if hasattr(agent, 'profile') else (36.16, -86.78)

        # Pacers drive with pacing mode
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
    logger.info("Running simulation...")
    start_time = time.time()
    result = engine.run()
    elapsed = time.time() - start_time

    logger.info(f"Simulation complete in {elapsed:.1f} seconds")
    logger.info(f"  Total trips: {result.metrics.get('total_trips', 0):,}")
    logger.info(f"  Avg travel time: {result.metrics.get('avg_travel_time', 0):.1f}s")

    # Save results
    save_results(result, config, output_dir)

    return result.metrics


def save_results(result, config: dict[str, Any], output_dir: Path) -> None:
    """Save simulation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_config = config.get("output", {})
    formats = output_config.get("formats", ["csv", "json"])

    # Save metrics
    if output_config.get("save_metrics", True):
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            # Convert numpy types to Python types
            metrics = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in result.metrics.items()
                if not isinstance(v, dict)
            }
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

    # Save raw data
    if output_config.get("save_raw_data", True) and result.raw_data is not None:
        if not result.raw_data.empty:
            if "csv" in formats:
                csv_path = output_dir / "trips.csv"
                result.raw_data.to_csv(csv_path, index=False)
                logger.info(f"Saved trip data to {csv_path}")

    # Save time series
    if result.time_series:
        ts_path = output_dir / "time_series.json"
        with open(ts_path, "w") as f:
            json.dump(result.time_series, f, indent=2)
        logger.info(f"Saved time series to {ts_path}")

    # Save config used
    config_path = output_dir / "config_used.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a simulation from a YAML configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Override number of agents from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override simulation duration (hours)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and show settings without running",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"Error parsing config: {e}", file=sys.stderr)
        return 1

    # Setup logging
    setup_logging(config, args.verbose)

    # Get settings (with CLI overrides)
    sim_config = config.get("simulation", {})
    agents_config = config.get("agents", {})
    output_config = config.get("output", {})

    n_agents = args.agents or agents_config.get("total", 10000)
    seed = args.seed or sim_config.get("random_seed", 42)
    output_dir = args.output_dir or Path(output_config.get("directory", "results/simulation"))
    duration_hours = args.duration or sim_config.get("duration_hours", 4.0)

    # Show settings
    print(f"Simulation: {sim_config.get('name', 'Unnamed')}")
    print(f"  Config: {args.config}")
    print(f"  Agents: {n_agents:,}")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_dir}")
    print()

    if args.dry_run:
        print("Dry run - not executing simulation")
        print("\nFull configuration:")
        print(yaml.dump(config, default_flow_style=False))
        return 0

    # Run simulation
    try:
        metrics = run_simulation(config, n_agents, seed, output_dir, duration_hours)

        print("\nResults:")
        print(f"  Total trips: {metrics.get('total_trips', 0):,}")
        print(f"  Avg travel time: {metrics.get('avg_travel_time', 0):.1f}s")
        print(f"  Total VMT: {metrics.get('total_vmt', 0):,.1f} miles")
        print(f"  Speed variance: {metrics.get('speed_variance', 0):.2f}")
        print(f"\nResults saved to: {output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        return 130
    except Exception as e:
        logger.exception(f"Simulation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
