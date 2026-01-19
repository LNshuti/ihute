"""
Base experiment framework for Monte Carlo simulation studies.

Provides abstract base class for experiments with parameter sweeps
and replication support.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result from a single experiment replication."""

    experiment_name: str
    parameters: dict[str, Any]
    replication: int
    metrics: dict[str, Any]
    raw_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding raw_data)."""
        return {
            "experiment_name": self.experiment_name,
            "parameters": self.parameters,
            "replication": self.replication,
            "metrics": self.metrics,
        }


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str
    output_dir: Path
    replications: int = 10
    random_seed: int = 42
    parallel: bool = True
    verbose: bool = False

    # Simulation defaults
    n_agents: int = 1000
    duration_hours: float = 4.0


class BaseExperiment(ABC):
    """
    Abstract base class for experiments.

    Subclasses implement get_parameter_grid() and run_single()
    to define experiment-specific behavior.
    """

    def __init__(
        self,
        config: ExperimentConfig,
    ):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create raw data subdirectory
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)

        # Setup logging
        if config.verbose:
            logging.basicConfig(level=logging.INFO)

        # Results storage
        self.results: list[ExperimentResult] = []

    @abstractmethod
    def get_parameter_grid(self, **cli_args) -> list[dict[str, Any]]:
        """
        Generate parameter combinations to test.

        Args:
            **cli_args: Command-line arguments specific to this experiment

        Returns:
            List of parameter dictionaries
        """
        pass

    @abstractmethod
    def run_single(
        self,
        params: dict[str, Any],
        replication: int,
        rng: np.random.Generator,
    ) -> ExperimentResult:
        """
        Run a single replication with given parameters.

        Args:
            params: Parameter dictionary
            replication: Replication number (0-indexed)
            rng: Random number generator

        Returns:
            ExperimentResult with metrics and data
        """
        pass

    def run_all(self, **cli_args) -> pd.DataFrame:
        """
        Run full experiment grid with replications.

        Args:
            **cli_args: Command-line arguments

        Returns:
            DataFrame of aggregated results
        """
        parameter_grid = self.get_parameter_grid(**cli_args)
        n_params = len(parameter_grid)
        n_reps = self.config.replications
        total_runs = n_params * n_reps

        logger.info(
            f"Running {self.config.name}: {n_params} parameter combinations x "
            f"{n_reps} replications = {total_runs} total runs"
        )

        self.results = []

        for param_idx, params in enumerate(parameter_grid):
            param_str = self._format_params(params)
            logger.info(f"Parameter set {param_idx + 1}/{n_params}: {param_str}")

            for rep in range(n_reps):
                # Create reproducible RNG for this run
                seed = self.config.random_seed + param_idx * 1000 + rep
                rng = np.random.default_rng(seed)

                if self.config.verbose:
                    logger.info(f"  Replication {rep + 1}/{n_reps}")

                # Run single replication
                result = self.run_single(params, rep, rng)
                self.results.append(result)

                # Save raw data
                self._save_raw_data(result, params, rep)

        # Aggregate and save summary
        summary_df = self._aggregate_results()
        summary_df.to_csv(self.output_dir / "summary.csv", index=False)

        # Save experiment metadata
        self._save_metadata(parameter_grid)

        logger.info(f"Experiment complete. Results saved to {self.output_dir}")

        return summary_df

    def _format_params(self, params: dict) -> str:
        """Format parameters for logging."""
        return ", ".join(f"{k}={v}" for k, v in params.items())

    def _save_raw_data(
        self,
        result: ExperimentResult,
        params: dict,
        replication: int,
    ) -> None:
        """Save raw data from a single run."""
        if result.raw_data is None or result.raw_data.empty:
            return

        # Create filename from parameters
        param_str = "_".join(f"{k}_{v}" for k, v in params.items())
        param_str = param_str.replace(".", "p")  # Replace dots for filenames

        # Try parquet first, fall back to CSV
        try:
            filename = f"{param_str}_rep_{replication:02d}.parquet"
            filepath = self.raw_dir / filename
            result.raw_data.to_parquet(filepath, index=False)
        except ImportError:
            # Fall back to CSV if parquet not available
            filename = f"{param_str}_rep_{replication:02d}.csv"
            filepath = self.raw_dir / filename
            result.raw_data.to_csv(filepath, index=False)

    def _aggregate_results(self) -> pd.DataFrame:
        """
        Aggregate results across replications.

        Computes mean, std, and 95% CI for each metric.
        """
        if not self.results:
            return pd.DataFrame()

        records = []

        # Group by parameters
        param_groups: dict[str, list[ExperimentResult]] = {}
        for result in self.results:
            key = json.dumps(result.parameters, sort_keys=True)
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append(result)

        for param_key, group_results in param_groups.items():
            params = json.loads(param_key)

            # Collect metrics across replications
            all_metrics: dict[str, list] = {}
            for result in group_results:
                for metric_name, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)

            # Compute statistics
            record = params.copy()
            record["n_replications"] = len(group_results)

            for metric_name, values in all_metrics.items():
                values = np.array(values)
                record[f"{metric_name}_mean"] = np.mean(values)
                record[f"{metric_name}_std"] = np.std(values)

                # 95% CI
                if len(values) > 1:
                    se = np.std(values) / np.sqrt(len(values))
                    record[f"{metric_name}_ci_low"] = np.mean(values) - 1.96 * se
                    record[f"{metric_name}_ci_high"] = np.mean(values) + 1.96 * se
                else:
                    record[f"{metric_name}_ci_low"] = np.mean(values)
                    record[f"{metric_name}_ci_high"] = np.mean(values)

            records.append(record)

        return pd.DataFrame(records)

    def _save_metadata(self, parameter_grid: list[dict]) -> None:
        """Save experiment metadata."""
        metadata = {
            "experiment_name": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "replications": self.config.replications,
                "random_seed": self.config.random_seed,
                "n_agents": self.config.n_agents,
                "duration_hours": self.config.duration_hours,
            },
            "parameter_grid": parameter_grid,
            "n_parameter_combinations": len(parameter_grid),
            "total_runs": len(self.results),
        }

        with open(self.output_dir / "report.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate a text report summarizing results.

        Args:
            results_df: Aggregated results DataFrame

        Returns:
            Report text
        """
        lines = [
            f"# {self.config.name} Experiment Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Configuration",
            f"- Replications: {self.config.replications}",
            f"- Agents: {self.config.n_agents}",
            f"- Random seed: {self.config.random_seed}",
            "",
            "## Results Summary",
            "",
        ]

        if not results_df.empty:
            # Find metric columns
            metric_cols = [c for c in results_df.columns if c.endswith("_mean")]

            for col in metric_cols:
                metric_name = col.replace("_mean", "")
                lines.append(f"### {metric_name}")
                lines.append(f"- Mean: {results_df[col].mean():.4f}")
                lines.append(f"- Range: [{results_df[col].min():.4f}, {results_df[col].max():.4f}]")
                lines.append("")

        return "\n".join(lines)


def create_parameter_grid(**param_lists) -> list[dict[str, Any]]:
    """
    Create a full factorial parameter grid.

    Args:
        **param_lists: Keyword arguments mapping parameter names to lists of values

    Returns:
        List of parameter dictionaries
    """
    if not param_lists:
        return [{}]

    keys = list(param_lists.keys())
    values = list(param_lists.values())

    return [dict(zip(keys, combo)) for combo in product(*values)]
