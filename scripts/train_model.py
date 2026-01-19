#!/usr/bin/env python3
"""
Train behavioral models from Hytch rideshare trip data.

Calibrates discrete choice models using historical trip data to estimate
preference parameters for mode choice, incentive response, and departure time.

Usage:
    python -m scripts.train_model --data data/raw/hytch_trips.parquet --output data/models/

Options:
    --data PATH          Path to Hytch trips parquet file
    --output PATH        Directory to save trained models
    --model-type TYPE    Model type: logit, mixed_logit, prospect (default: all)
    --test-size FLOAT    Fraction of data for testing (default: 0.2)
    --seed INT           Random seed (default: 42)
    --verbose            Enable verbose output
"""

from __future__ import annotations

import argparse
import json
import pickle
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import softmax

logger = logging.getLogger(__name__)


@dataclass
class EstimatedParameters:
    """Estimated model parameters with standard errors."""

    beta_time: float
    beta_cost: float
    beta_incentive: float
    beta_peak: float = 0.0
    beta_distance: float = 0.0
    asc_carpool: float = 0.0
    asc_solo: float = 0.0

    # Standard errors
    se_time: float = 0.0
    se_cost: float = 0.0
    se_incentive: float = 0.0
    se_peak: float = 0.0

    # Model fit statistics
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    rho_squared: float = 0.0
    n_observations: int = 0


@dataclass
class MixedLogitParameters:
    """Mixed logit parameters with random coefficient distributions."""

    # Mean parameters
    beta_time_mean: float
    beta_cost_mean: float
    beta_incentive_mean: float

    # Standard deviations (for random taste variation)
    beta_time_std: float = 0.0
    beta_cost_std: float = 0.0
    beta_incentive_std: float = 0.0

    # Distribution types
    time_distribution: str = "normal"
    cost_distribution: str = "lognormal"
    incentive_distribution: str = "normal"

    # Model fit
    log_likelihood: float = 0.0
    n_observations: int = 0
    n_draws: int = 100


@dataclass
class ProspectTheoryParameters:
    """Prospect theory model parameters."""

    loss_aversion: float  # lambda
    diminishing_sensitivity: float  # alpha
    probability_weight: float  # gamma
    reference_incentive: float  # reference point for gains/losses

    log_likelihood: float = 0.0
    n_observations: int = 0


def load_hytch_data(data_path: Path) -> pd.DataFrame:
    """Load and validate Hytch trip data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    required_columns = [
        "trip_id",
        "distance_miles",
        "travel_time_minutes",
        "n_passengers",
        "incentive_amount",
        "is_peak_hour",
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {len(df):,} trips")
    return df


def prepare_choice_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for discrete choice modeling.

    Creates synthetic choice sets by treating observed trips as chosen
    alternatives and generating counterfactual alternatives.
    """
    logger.info("Preparing choice data...")

    # Filter to completed trips
    if "status" in df.columns:
        df = df[df["status"] == 2].copy()  # COMPLETED status

    # Create binary choice: carpool (n_passengers > 0) vs solo
    df["chose_carpool"] = (df["n_passengers"] > 0).astype(int)

    # Compute derived features
    df["cost_estimate"] = df["distance_miles"] * 0.15  # $0.15/mile operating cost

    # Impute incentive for solo trips (what they would have received)
    df["incentive_if_carpool"] = np.where(
        df["n_passengers"] > 0,
        df["incentive_amount"],
        df["distance_miles"] * 0.25 * (1 + 0.5 * df["is_peak_hour"]),  # Estimated
    )

    # Travel time differential (carpool may add pickup time)
    df["carpool_time_penalty"] = np.where(
        df["n_passengers"] > 0,
        np.clip(df["travel_time_minutes"] * 0.1, 2, 15),  # 10% overhead, 2-15 min
        5.0,  # Assumed average pickup time if they had carpooled
    )

    return df


def estimate_logit_parameters(
    df: pd.DataFrame,
    seed: int = 42,
) -> EstimatedParameters:
    """
    Estimate multinomial logit model parameters via maximum likelihood.

    Uses numerical optimization to find parameters that maximize the
    likelihood of observed choices.
    """
    logger.info("Estimating logit model parameters...")
    rng = np.random.default_rng(seed)

    n = len(df)
    y = df["chose_carpool"].values  # 1 if chose carpool, 0 if solo

    # Feature matrix for carpool utility
    X = np.column_stack([
        -df["carpool_time_penalty"].values / 60,  # Time in hours
        -df["cost_estimate"].values,  # Cost penalty
        df["incentive_if_carpool"].values,  # Incentive
        df["is_peak_hour"].values.astype(float),  # Peak hour indicator
    ])

    def neg_log_likelihood(params):
        """Negative log-likelihood for logit model."""
        beta = params[:-1]
        asc = params[-1]  # Alternative-specific constant for carpool

        # Utility of carpool relative to solo (solo utility = 0)
        V_carpool = X @ beta + asc

        # Logit probability
        prob_carpool = 1 / (1 + np.exp(-V_carpool))
        prob_carpool = np.clip(prob_carpool, 1e-10, 1 - 1e-10)

        # Log-likelihood
        ll = y * np.log(prob_carpool) + (1 - y) * np.log(1 - prob_carpool)

        return -ll.sum()

    # Initial parameter values
    x0 = np.array([
        -0.05,  # beta_time
        -0.10,  # beta_cost
        0.15,  # beta_incentive
        0.20,  # beta_peak
        -0.50,  # asc_carpool
    ])

    # Optimize
    result = optimize.minimize(
        neg_log_likelihood,
        x0,
        method="BFGS",
        options={"disp": False, "maxiter": 1000},
    )

    if not result.success:
        logger.warning(f"Optimization did not converge: {result.message}")

    # Extract parameters
    beta_time, beta_cost, beta_incentive, beta_peak, asc = result.x

    # Compute standard errors from Hessian
    try:
        hessian = result.hess_inv
        if hasattr(hessian, "todense"):
            hessian = hessian.todense()
        se = np.sqrt(np.diag(hessian))
    except Exception:
        se = np.zeros(5)

    # Model fit statistics
    ll_full = -result.fun
    ll_null = n * np.log(0.5)  # Null model: equal probabilities

    # Number of parameters
    k = len(result.x)

    params = EstimatedParameters(
        beta_time=beta_time,
        beta_cost=beta_cost,
        beta_incentive=beta_incentive,
        beta_peak=beta_peak,
        asc_carpool=asc,
        se_time=se[0] if len(se) > 0 else 0,
        se_cost=se[1] if len(se) > 1 else 0,
        se_incentive=se[2] if len(se) > 2 else 0,
        se_peak=se[3] if len(se) > 3 else 0,
        log_likelihood=ll_full,
        aic=-2 * ll_full + 2 * k,
        bic=-2 * ll_full + k * np.log(n),
        rho_squared=1 - ll_full / ll_null,
        n_observations=n,
    )

    logger.info(f"Logit estimation complete:")
    logger.info(f"  beta_time: {beta_time:.4f} (SE: {se[0]:.4f})")
    logger.info(f"  beta_cost: {beta_cost:.4f} (SE: {se[1]:.4f})")
    logger.info(f"  beta_incentive: {beta_incentive:.4f} (SE: {se[2]:.4f})")
    logger.info(f"  rho-squared: {params.rho_squared:.4f}")

    return params


def estimate_mixed_logit_parameters(
    df: pd.DataFrame,
    n_draws: int = 200,
    seed: int = 42,
) -> MixedLogitParameters:
    """
    Estimate mixed logit model with random taste variation.

    Uses simulated maximum likelihood with Halton draws.
    """
    logger.info(f"Estimating mixed logit model with {n_draws} draws...")
    rng = np.random.default_rng(seed)

    n = len(df)
    y = df["chose_carpool"].values

    # Features
    time_diff = -df["carpool_time_penalty"].values / 60
    cost_diff = -df["cost_estimate"].values
    incentive = df["incentive_if_carpool"].values

    # Generate Halton-like draws for simulation
    draws = rng.normal(0, 1, (n_draws, 3))

    def simulated_log_likelihood(params):
        """Simulated log-likelihood for mixed logit."""
        # Unpack parameters (means and log-stds)
        beta_time_mean, beta_cost_mean, beta_incentive_mean = params[:3]
        log_std_time, log_std_cost, log_std_incentive = params[3:6]

        std_time = np.exp(log_std_time)
        std_cost = np.exp(log_std_cost)
        std_incentive = np.exp(log_std_incentive)

        prob_sum = np.zeros(n)

        for draw in draws:
            # Draw individual-specific parameters
            beta_time = beta_time_mean + std_time * draw[0]
            beta_cost = -np.exp(np.log(-beta_cost_mean) + std_cost * draw[1])  # Lognormal
            beta_incentive = beta_incentive_mean + std_incentive * draw[2]

            # Utility
            V = beta_time * time_diff + beta_cost * cost_diff + beta_incentive * incentive

            # Logit probability
            prob = 1 / (1 + np.exp(-V))
            prob_sum += prob

        # Average probability across draws
        avg_prob = prob_sum / n_draws
        avg_prob = np.clip(avg_prob, 1e-10, 1 - 1e-10)

        # Log-likelihood
        ll = y * np.log(avg_prob) + (1 - y) * np.log(1 - avg_prob)

        return -ll.sum()

    # Initial values
    x0 = np.array([
        -0.05,  # beta_time_mean
        -0.10,  # beta_cost_mean
        0.15,  # beta_incentive_mean
        np.log(0.02),  # log_std_time
        np.log(0.03),  # log_std_cost
        np.log(0.05),  # log_std_incentive
    ])

    # Optimize (use simpler method for stability)
    result = optimize.minimize(
        simulated_log_likelihood,
        x0,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 500},
    )

    beta_time_mean, beta_cost_mean, beta_incentive_mean = result.x[:3]
    std_time = np.exp(result.x[3])
    std_cost = np.exp(result.x[4])
    std_incentive = np.exp(result.x[5])

    params = MixedLogitParameters(
        beta_time_mean=beta_time_mean,
        beta_cost_mean=beta_cost_mean,
        beta_incentive_mean=beta_incentive_mean,
        beta_time_std=std_time,
        beta_cost_std=std_cost,
        beta_incentive_std=std_incentive,
        log_likelihood=-result.fun,
        n_observations=n,
        n_draws=n_draws,
    )

    logger.info(f"Mixed logit estimation complete:")
    logger.info(f"  beta_time: {beta_time_mean:.4f} (std: {std_time:.4f})")
    logger.info(f"  beta_cost: {beta_cost_mean:.4f} (std: {std_cost:.4f})")
    logger.info(f"  beta_incentive: {beta_incentive_mean:.4f} (std: {std_incentive:.4f})")

    return params


def estimate_prospect_theory_parameters(
    df: pd.DataFrame,
    seed: int = 42,
) -> ProspectTheoryParameters:
    """
    Estimate prospect theory model parameters.

    Estimates loss aversion (lambda), diminishing sensitivity (alpha),
    and probability weighting (gamma) from observed choices.
    """
    logger.info("Estimating prospect theory parameters...")
    rng = np.random.default_rng(seed)

    n = len(df)
    y = df["chose_carpool"].values

    # Reference point: median incentive
    reference = df["incentive_if_carpool"].median()
    incentive_diff = df["incentive_if_carpool"].values - reference

    # Time and cost as potential losses
    time_diff = df["carpool_time_penalty"].values
    cost_diff = df["cost_estimate"].values

    def neg_log_likelihood(params):
        """Negative log-likelihood for prospect theory model."""
        loss_aversion, alpha, ref_scale = params

        # Value function for gains (incentive above reference)
        gains = np.maximum(incentive_diff * ref_scale, 0) ** alpha

        # Value function for losses (time and cost)
        losses = loss_aversion * (
            (np.maximum(time_diff, 0) ** alpha) +
            (np.maximum(cost_diff, 0) ** alpha)
        )

        # Net prospect value
        V = gains - losses

        # Softmax choice probability
        prob_carpool = 1 / (1 + np.exp(-V))
        prob_carpool = np.clip(prob_carpool, 1e-10, 1 - 1e-10)

        ll = y * np.log(prob_carpool) + (1 - y) * np.log(1 - prob_carpool)
        return -ll.sum()

    # Initial values (Tversky & Kahneman defaults)
    x0 = np.array([2.25, 0.88, 0.5])

    # Bounds to ensure valid parameters
    bounds = [(1.0, 4.0), (0.3, 1.0), (0.1, 2.0)]

    result = optimize.minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"disp": False},
    )

    loss_aversion, alpha, ref_scale = result.x

    params = ProspectTheoryParameters(
        loss_aversion=loss_aversion,
        diminishing_sensitivity=alpha,
        probability_weight=0.65,  # Fixed for simplicity
        reference_incentive=reference,
        log_likelihood=-result.fun,
        n_observations=n,
    )

    logger.info(f"Prospect theory estimation complete:")
    logger.info(f"  loss_aversion (λ): {loss_aversion:.4f}")
    logger.info(f"  diminishing_sensitivity (α): {alpha:.4f}")
    logger.info(f"  reference_incentive: ${reference:.2f}")

    return params


def compute_elasticities(
    df: pd.DataFrame,
    logit_params: EstimatedParameters,
) -> dict[str, float]:
    """
    Compute incentive elasticity from estimated parameters.

    Returns arc elasticity of carpool participation with respect to incentive.
    """
    logger.info("Computing incentive elasticities...")

    # Average values
    avg_incentive = df["incentive_if_carpool"].mean()
    carpool_rate = df["chose_carpool"].mean()

    # Direct elasticity from logit model
    # For logit: elasticity = beta * x * (1 - P)
    elasticity = logit_params.beta_incentive * avg_incentive * (1 - carpool_rate)

    # Simulate +10% incentive change
    incentive_up = avg_incentive * 1.1
    delta_utility = logit_params.beta_incentive * (incentive_up - avg_incentive)
    new_rate = carpool_rate + carpool_rate * (1 - carpool_rate) * delta_utility

    arc_elasticity = (new_rate - carpool_rate) / carpool_rate / 0.1

    return {
        "point_elasticity": elasticity,
        "arc_elasticity_10pct": arc_elasticity,
        "base_carpool_rate": carpool_rate,
        "avg_incentive": avg_incentive,
    }


def save_models(
    output_dir: Path,
    logit_params: Optional[EstimatedParameters],
    mixed_params: Optional[MixedLogitParameters],
    prospect_params: Optional[ProspectTheoryParameters],
    elasticities: dict,
    metadata: dict,
) -> None:
    """Save trained models and parameters."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON for readability
    results = {
        "metadata": metadata,
        "elasticities": elasticities,
    }

    if logit_params:
        results["logit"] = asdict(logit_params)
        # Also save as pickle for direct use
        with open(output_dir / "logit_model.pkl", "wb") as f:
            pickle.dump(logit_params, f)

    if mixed_params:
        results["mixed_logit"] = asdict(mixed_params)
        with open(output_dir / "mixed_logit_model.pkl", "wb") as f:
            pickle.dump(mixed_params, f)

    if prospect_params:
        results["prospect_theory"] = asdict(prospect_params)
        with open(output_dir / "prospect_theory_model.pkl", "wb") as f:
            pickle.dump(prospect_params, f)

    with open(output_dir / "model_parameters.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Models saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train behavioral models from Hytch trip data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/hytch_trips.parquet"),
        help="Path to Hytch trips parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models"),
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logit", "mixed_logit", "prospect", "all"],
        default="all",
        help="Model type to train (default: all)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Always show info for this script
    logger.setLevel(logging.INFO)

    print(f"Training behavioral models from {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Model type: {args.model_type}")
    print()

    try:
        # Load data
        df = load_hytch_data(args.data)

        # Prepare choice data
        df = prepare_choice_data(df)

        # Train/test split
        rng = np.random.default_rng(args.seed)
        n = len(df)
        test_mask = rng.random(n) < args.test_size
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()

        print(f"Training set: {len(train_df):,} observations")
        print(f"Test set: {len(test_df):,} observations")
        print()

        # Train models
        logit_params = None
        mixed_params = None
        prospect_params = None

        if args.model_type in ["logit", "all"]:
            logit_params = estimate_logit_parameters(train_df, args.seed)
            print()

        if args.model_type in ["mixed_logit", "all"]:
            mixed_params = estimate_mixed_logit_parameters(train_df, n_draws=200, seed=args.seed)
            print()

        if args.model_type in ["prospect", "all"]:
            prospect_params = estimate_prospect_theory_parameters(train_df, args.seed)
            print()

        # Compute elasticities
        elasticities = {}
        if logit_params:
            elasticities = compute_elasticities(train_df, logit_params)
            print("Elasticities:")
            print(f"  Point elasticity: {elasticities['point_elasticity']:.4f}")
            print(f"  Arc elasticity (10% change): {elasticities['arc_elasticity_10pct']:.4f}")
            print(f"  Base carpool rate: {elasticities['base_carpool_rate']:.2%}")
            print()

        # Save models
        metadata = {
            "training_date": datetime.now().isoformat(),
            "data_source": str(args.data),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "random_seed": args.seed,
        }

        save_models(
            args.output,
            logit_params,
            mixed_params,
            prospect_params,
            elasticities,
            metadata,
        )

        print(f"\nModels saved to {args.output}")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
