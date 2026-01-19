#!/usr/bin/env python3
"""
Generate synthetic Hytch rideshare trip data for behavioral model training.

Creates data/raw/hytch_trips.parquet with 369,831 synthetic trips based on
the Hytch data model, calibrated to realistic Nashville I-24 corridor patterns.

Usage:
    python -m scripts.generate_synthetic_data
    python -m scripts.generate_synthetic_data --n-trips 100000 --seed 42
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Nashville I-24 corridor geographic bounds
NASHVILLE_BOUNDS = {
    "origin": {  # Southeast suburbs (Antioch, La Vergne, Smyrna)
        "lat_min": 36.02,
        "lat_max": 36.12,
        "lon_min": -86.70,
        "lon_max": -86.58,
    },
    "destination": {  # Downtown Nashville
        "lat_min": 36.14,
        "lat_max": 36.18,
        "lon_min": -86.82,
        "lon_max": -86.76,
    },
}

# Trip status enums (matching Hytch data model)
TRIP_STATUS = {
    "CREATED": 0,
    "ACTIVE": 1,
    "COMPLETED": 2,
    "CANCELLED": 3,
    "ABANDONED": 4,
}

# Driver type enums
DRIVER_TYPE = {
    "REGULAR": 0,
    "VOLUNTEER": 1,
    "PROFESSIONAL": 2,
    "PACER": 3,
}

# Participant role enums
PARTICIPANT_ROLE = {
    "DRIVER": 0,
    "PASSENGER": 1,
}

# Transaction types
TRANSACTION_TYPE = {
    "EARNED": 0,
    "REDEEMED": 1,
    "BONUS": 2,
    "REFERRAL": 3,
}


def generate_trip_id() -> str:
    """Generate UUID-style trip ID."""
    return str(uuid.uuid4())


def generate_user_ids(n_users: int, rng: np.random.Generator) -> np.ndarray:
    """Generate sequential user IDs."""
    return np.arange(1, n_users + 1)


def sample_location(
    bounds: dict,
    n: int,
    rng: np.random.Generator,
    cluster_centers: Optional[list] = None,
    cluster_std: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample locations within bounds, optionally clustered around centers."""
    if cluster_centers is None:
        lat = rng.uniform(bounds["lat_min"], bounds["lat_max"], n)
        lon = rng.uniform(bounds["lon_min"], bounds["lon_max"], n)
    else:
        # Cluster around given centers
        center_idx = rng.integers(0, len(cluster_centers), n)
        lat = np.zeros(n)
        lon = np.zeros(n)
        for i, (clat, clon) in enumerate(cluster_centers):
            mask = center_idx == i
            lat[mask] = rng.normal(clat, cluster_std, mask.sum())
            lon[mask] = rng.normal(clon, cluster_std, mask.sum())

        # Clip to bounds
        lat = np.clip(lat, bounds["lat_min"], bounds["lat_max"])
        lon = np.clip(lon, bounds["lon_min"], bounds["lon_max"])

    return lat, lon


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate haversine distance in miles."""
    R = 3959  # Earth radius in miles

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def generate_departure_times(
    n: int,
    date_range: tuple[datetime, datetime],
    rng: np.random.Generator,
) -> pd.DatetimeIndex:
    """Generate departure times with realistic rush hour patterns."""
    start_date, end_date = date_range
    n_days = (end_date - start_date).days

    # Sample days uniformly
    days = rng.integers(0, n_days, n)

    # Sample times with bimodal rush hour distribution
    # Morning peak: 7-9 AM, Evening peak: 4-7 PM
    hour_weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.02, 0.04,  # 0-5
        0.08, 0.15, 0.12, 0.06, 0.04, 0.04,  # 6-11
        0.05, 0.04, 0.04, 0.06, 0.10, 0.08,  # 12-17
        0.04, 0.03, 0.02, 0.01, 0.01, 0.01,  # 18-23
    ])
    hour_weights = hour_weights / hour_weights.sum()

    hours = rng.choice(24, n, p=hour_weights)
    minutes = rng.integers(0, 60, n)
    seconds = rng.integers(0, 60, n)

    # Combine into timestamps
    base = start_date
    timestamps = [
        base + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(days, hours, minutes, seconds)
    ]

    return pd.DatetimeIndex(timestamps)


def calculate_travel_time(distance_miles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Calculate travel time based on distance with congestion variability."""
    # Base speed varies by time of day (simulated by randomness here)
    base_speed = rng.uniform(25, 55, len(distance_miles))  # mph

    # Travel time in minutes
    travel_time = (distance_miles / base_speed) * 60

    # Add congestion-based variability
    congestion_factor = rng.lognormal(0, 0.3, len(distance_miles))
    travel_time = travel_time * congestion_factor

    return np.clip(travel_time, 5, 120)  # 5 min to 2 hours


def calculate_rewards(
    distance_miles: np.ndarray,
    n_passengers: np.ndarray,
    is_peak: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate points, gas savings, and incentive payments."""
    # Base points per mile (higher for carpools)
    base_points_per_mile = 10
    passenger_bonus = n_passengers * 5
    peak_multiplier = np.where(is_peak, 1.5, 1.0)

    points = (distance_miles * base_points_per_mile + passenger_bonus) * peak_multiplier
    points = points.astype(int)

    # Gas savings estimate ($3.50/gal, 25 mpg average)
    gas_price = 3.50
    mpg = 25
    solo_cost = distance_miles * gas_price / mpg
    shared_factor = np.where(n_passengers > 0, 1 / (n_passengers + 1), 1)
    gas_savings = solo_cost * (1 - shared_factor)

    # Incentive payments (0-$10 range, higher for carpools)
    base_incentive = rng.exponential(2.0, len(distance_miles))
    incentive = base_incentive * (1 + n_passengers * 0.5) * peak_multiplier
    incentive = np.clip(incentive, 0, 15)

    return points.astype(int), np.round(gas_savings, 2), np.round(incentive, 2)


def calculate_environmental_impact(distance_miles: np.ndarray, n_passengers: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate environmental impact reductions."""
    # Only count reduction if there are passengers (shared miles)
    shared_reduction_factor = np.where(n_passengers > 0, n_passengers / (n_passengers + 1), 0)
    effective_miles = distance_miles * shared_reduction_factor

    # Emission factors per mile (grams)
    factors = {
        "co2_reduced": 404,  # CO2
        "co_reduced": 2.1,  # Carbon monoxide
        "nox_reduced": 0.4,  # Nitrogen oxides
        "pm25_reduced": 0.02,  # PM2.5
        "pm10_reduced": 0.05,  # PM10
        "voc_reduced": 0.5,  # Volatile organic compounds
    }

    impact = {}
    for name, factor in factors.items():
        impact[name] = np.round(effective_miles * factor, 2)

    # Trees saved (simplified: 1 tree absorbs ~48 lbs CO2/year)
    impact["trees_saved"] = np.round(impact["co2_reduced"] / (48 * 453.6) * 365, 4)

    return impact


def assign_impact_grade(co2_reduced: np.ndarray) -> np.ndarray:
    """Assign environmental impact grade A-F based on CO2 reduction."""
    grades = np.empty(len(co2_reduced), dtype="<U1")
    grades[:] = "F"
    grades[co2_reduced > 0] = "D"
    grades[co2_reduced > 100] = "C"
    grades[co2_reduced > 500] = "B"
    grades[co2_reduced > 1000] = "A"
    return grades


def generate_hytch_trips(
    n_trips: int = 369831,
    n_users: int = 50000,
    date_range: tuple[datetime, datetime] = (datetime(2022, 1, 1), datetime(2024, 12, 31)),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic Hytch trip data.

    Returns a denormalized DataFrame combining data from:
    - trips
    - trip_history_details
    - trip_participants
    - transactions
    """
    rng = np.random.default_rng(seed)
    print(f"Generating {n_trips:,} synthetic Hytch trips...")

    # Generate user pool
    user_ids = generate_user_ids(n_users, rng)

    # Trip IDs
    trip_ids = [generate_trip_id() for _ in range(n_trips)]

    # Assign trip owners and drivers
    owner_user_ids = rng.choice(user_ids, n_trips)
    # 70% of trips have the owner as driver
    is_owner_driver = rng.random(n_trips) < 0.7
    driver_user_ids = np.where(is_owner_driver, owner_user_ids, rng.choice(user_ids, n_trips))

    # Trip types
    is_solo = rng.random(n_trips) < 0.3  # 30% solo trips
    is_shelter_in_place = rng.random(n_trips) < 0.02  # 2% stationary carpools

    # Departure times
    started_at = generate_departure_times(n_trips, date_range, rng)

    # Determine peak hours (7-9 AM, 4-7 PM on weekdays)
    is_weekday = started_at.dayofweek < 5
    is_morning_peak = (started_at.hour >= 7) & (started_at.hour < 9)
    is_evening_peak = (started_at.hour >= 16) & (started_at.hour < 19)
    is_peak = is_weekday & (is_morning_peak | is_evening_peak)

    # Origin/destination locations
    # Common origin clusters (suburbs)
    origin_clusters = [
        (36.07, -86.65),  # Antioch
        (36.05, -86.58),  # La Vergne
        (36.10, -86.62),  # South Nashville
    ]
    origin_lat, origin_lon = sample_location(
        NASHVILLE_BOUNDS["origin"], n_trips, rng, origin_clusters, cluster_std=0.02
    )

    # Common destination clusters (downtown/work areas)
    dest_clusters = [
        (36.16, -86.78),  # Downtown
        (36.15, -86.80),  # West End
        (36.17, -86.76),  # Germantown
    ]
    dest_lat, dest_lon = sample_location(
        NASHVILLE_BOUNDS["destination"], n_trips, rng, dest_clusters, cluster_std=0.015
    )

    # Calculate distances
    distance_miles = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)

    # Travel times
    travel_time_minutes = calculate_travel_time(distance_miles, rng)
    finished_at = started_at + pd.to_timedelta(travel_time_minutes, unit="m")

    # Participants (number of passengers, 0 for solo trips)
    n_passengers = np.where(is_solo, 0, rng.integers(1, 5, n_trips))

    # Shared miles (only if there are passengers)
    shared_miles = np.where(n_passengers > 0, distance_miles, 0)

    # Rewards
    points, gas_savings, incentive_amount = calculate_rewards(distance_miles, n_passengers, is_peak, rng)

    # Environmental impact
    env_impact = calculate_environmental_impact(distance_miles, n_passengers)
    impact_grade = assign_impact_grade(env_impact["co2_reduced"])

    # Trip status (95% completed, 3% cancelled, 2% abandoned)
    status_probs = rng.random(n_trips)
    trip_status = np.where(
        status_probs < 0.95,
        TRIP_STATUS["COMPLETED"],
        np.where(status_probs < 0.98, TRIP_STATUS["CANCELLED"], TRIP_STATUS["ABANDONED"]),
    )

    # Driver types
    driver_type = rng.choice(
        [DRIVER_TYPE["REGULAR"], DRIVER_TYPE["VOLUNTEER"], DRIVER_TYPE["PACER"]],
        n_trips,
        p=[0.8, 0.15, 0.05],
    )

    # Build DataFrame
    df = pd.DataFrame(
        {
            # Trip identification
            "trip_id": trip_ids,
            "owner_user_id": owner_user_ids,
            "driver_user_id": driver_user_ids,
            # Trip flags
            "is_solo_hytch": is_solo,
            "is_shelter_in_place": is_shelter_in_place,
            "is_peak_hour": is_peak,
            # Status and type
            "status": trip_status,
            "driver_type": driver_type,
            # Timing
            "started_at": started_at,
            "finished_at": finished_at,
            "travel_time_minutes": np.round(travel_time_minutes, 2),
            # Location
            "origin_latitude": np.round(origin_lat, 8),
            "origin_longitude": np.round(origin_lon, 8),
            "dest_latitude": np.round(dest_lat, 8),
            "dest_longitude": np.round(dest_lon, 8),
            # Trip metrics
            "distance_miles": np.round(distance_miles, 2),
            "shared_miles": np.round(shared_miles, 2),
            "n_passengers": n_passengers,
            # Rewards
            "points": points,
            "gas_savings": gas_savings,
            "incentive_amount": incentive_amount,
            # Environmental impact
            "trip_impact_grade": impact_grade,
            "co2_reduced": env_impact["co2_reduced"],
            "co_reduced": env_impact["co_reduced"],
            "nox_reduced": env_impact["nox_reduced"],
            "pm25_reduced": env_impact["pm25_reduced"],
            "pm10_reduced": env_impact["pm10_reduced"],
            "voc_reduced": env_impact["voc_reduced"],
            "trees_saved": env_impact["trees_saved"],
        }
    )

    # Add metadata columns
    df["created_at"] = started_at - pd.Timedelta(minutes=5)  # Created 5 min before start
    df["day_of_week"] = started_at.dayofweek
    df["hour_of_day"] = started_at.hour

    # Sort by start time
    df = df.sort_values("started_at").reset_index(drop=True)

    print(f"Generated {len(df):,} trips")
    print(f"  Date range: {df['started_at'].min()} to {df['started_at'].max()}")
    print(f"  Solo trips: {df['is_solo_hytch'].sum():,} ({100*df['is_solo_hytch'].mean():.1f}%)")
    print(f"  Avg distance: {df['distance_miles'].mean():.1f} miles")
    print(f"  Avg passengers: {df['n_passengers'].mean():.2f}")
    print(f"  Peak hour trips: {df['is_peak_hour'].sum():,} ({100*df['is_peak_hour'].mean():.1f}%)")
    print(f"  Total CO2 reduced: {df['co2_reduced'].sum()/1000:.1f} kg")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Hytch trip data")
    parser.add_argument(
        "--n-trips",
        type=int,
        default=369831,
        help="Number of trips to generate (default: 369831)",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=50000,
        help="Number of unique users (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/hytch_trips.parquet"),
        help="Output file path (default: data/raw/hytch_trips.parquet)",
    )
    args = parser.parse_args()

    # Generate data
    df = generate_hytch_trips(
        n_trips=args.n_trips,
        n_users=args.n_users,
        seed=args.seed,
    )

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    df.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
