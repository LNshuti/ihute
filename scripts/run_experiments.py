#!/usr/bin/env python3
"""
Run transportation incentive experiments.

Usage examples:

    python -m scripts.run_experiments --experiment pacer_threshold \\
        --participation-rates 0.01 0.02 0.05 0.10 0.15 0.20 \\
        --replications 30

    python -m scripts.run_experiments --experiment carpool_elasticity \\
        --reward-levels 1.0 2.0 5.0 10.0 \\
        --targeting-precision low medium high \\
        --replications 20

    python -m scripts.run_experiments --experiment event_egress \\
        --delay-distribution uniform concentrated \\
        --total-delay-budget 1000 \\
        --replications 25

Common options:
    --output-dir PATH      Override default results directory
    --seed INT             Random seed for reproducibility (default: 42)
    --n-agents INT         Number of agents (default: 1000)
    --parallel             Enable parallel replications (default)
    --no-parallel          Disable parallel replications
    --verbose              Show progress and logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run transportation incentive experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required: experiment type
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["pacer_threshold", "carpool_elasticity", "event_egress"],
        help="Type of experiment to run",
    )

    # Common options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/<experiment_name>)",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=10,
        help="Number of replications per parameter combination (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=1000,
        help="Number of agents in simulation (default: 1000)",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=4.0,
        help="Simulation duration in hours (default: 4.0)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Enable parallel replications (default)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Disable parallel replications",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Pacer threshold experiment options
    parser.add_argument(
        "--participation-rates",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
        help="Pacer participation rates to test (default: 0.01 0.02 0.05 0.10 0.15 0.20)",
    )

    # Carpool elasticity experiment options
    parser.add_argument(
        "--reward-levels",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0],
        help="Carpool reward levels in dollars (default: 1.0 2.0 5.0 10.0)",
    )
    parser.add_argument(
        "--targeting-precision",
        type=str,
        nargs="+",
        choices=["low", "medium", "high"],
        default=["low", "medium", "high"],
        help="Targeting precision levels (default: low medium high)",
    )

    # Event egress experiment options
    parser.add_argument(
        "--delay-distribution",
        type=str,
        nargs="+",
        choices=["uniform", "concentrated"],
        default=["uniform", "concentrated"],
        help="Delay distribution strategies (default: uniform concentrated)",
    )
    parser.add_argument(
        "--total-delay-budget",
        type=float,
        default=1000,
        help="Total delay budget in person-minutes (default: 1000)",
    )

    return parser.parse_args()


def run_pacer_threshold(args: argparse.Namespace) -> None:
    """Run pacer threshold experiment."""
    from scripts.experiments.pacer_threshold import PacerThresholdExperiment
    from scripts.experiments.base import ExperimentConfig

    output_dir = args.output_dir or Path("results/pacer_threshold")

    config = ExperimentConfig(
        name="pacer_threshold",
        output_dir=output_dir,
        replications=args.replications,
        random_seed=args.seed,
        parallel=args.parallel,
        verbose=args.verbose,
        n_agents=args.n_agents,
        duration_hours=args.duration_hours,
    )

    experiment = PacerThresholdExperiment(config)
    results = experiment.run_all(
        participation_rates=args.participation_rates,
    )

    print(f"\nExperiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nSummary:")
    print(results.to_string(index=False))


def run_carpool_elasticity(args: argparse.Namespace) -> None:
    """Run carpool elasticity experiment."""
    from scripts.experiments.carpool_elasticity import CarpoolElasticityExperiment
    from scripts.experiments.base import ExperimentConfig

    output_dir = args.output_dir or Path("results/carpool_elasticity")

    config = ExperimentConfig(
        name="carpool_elasticity",
        output_dir=output_dir,
        replications=args.replications,
        random_seed=args.seed,
        parallel=args.parallel,
        verbose=args.verbose,
        n_agents=args.n_agents,
        duration_hours=args.duration_hours,
    )

    experiment = CarpoolElasticityExperiment(config)
    results = experiment.run_all(
        reward_levels=args.reward_levels,
        targeting_precision=args.targeting_precision,
    )

    print(f"\nExperiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nSummary:")
    print(results.to_string(index=False))


def run_event_egress(args: argparse.Namespace) -> None:
    """Run event egress experiment."""
    from scripts.experiments.event_egress import EventEgressExperiment
    from scripts.experiments.base import ExperimentConfig

    output_dir = args.output_dir or Path("results/event_egress")

    config = ExperimentConfig(
        name="event_egress",
        output_dir=output_dir,
        replications=args.replications,
        random_seed=args.seed,
        parallel=args.parallel,
        verbose=args.verbose,
        n_agents=10000,  # Stadium capacity
        duration_hours=2.0,  # 2 hours post-event
    )

    experiment = EventEgressExperiment(config)
    results = experiment.run_all(
        delay_distribution=args.delay_distribution,
        total_delay_budget=args.total_delay_budget,
    )

    print(f"\nExperiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nSummary:")
    print(results.to_string(index=False))


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(f"Running {args.experiment} experiment...")
    print(f"  Replications: {args.replications}")
    print(f"  Random seed: {args.seed}")
    print(f"  Verbose: {args.verbose}")
    print()

    try:
        if args.experiment == "pacer_threshold":
            run_pacer_threshold(args)
        elif args.experiment == "carpool_elasticity":
            run_carpool_elasticity(args)
        elif args.experiment == "event_egress":
            run_event_egress(args)
        else:
            print(f"Unknown experiment: {args.experiment}", file=sys.stderr)
            return 1

        return 0

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        return 130
    except Exception as e:
        logging.exception(f"Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
